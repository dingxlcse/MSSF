import os
import torch
import random
import pickle
import argparse
import numpy as np
import torch.nn as nn
import sys
import time
from math import sqrt
import torch.utils.data
from copy import deepcopy
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from model import Mulmodel
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score, \
precision_score, recall_score, confusion_matrix,cohen_kappa_score,matthews_corrcoef


np.random.seed(42)
random.seed(42)


def read_raw_data(rawdata_dir, data_train, data_test):
    gii = open(rawdata_dir + '/' + 'Text_similarity_one.pkl', 'rb')
    drug_Tfeature_one = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + '/' + 'Text_similarity_two.pkl', 'rb')
    drug_Tfeature_two = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + '/' + 'Text_similarity_three.pkl', 'rb')
    drug_Tfeature_three = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + '/' + 'Text_similarity_four.pkl', 'rb')
    drug_Tfeature_four = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + '/' + 'Text_similarity_five.pkl', 'rb')
    drug_Tfeature_five = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + '/' + 'side_effect_semantic.pkl', 'rb')
    effect_side_semantic = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + '/' + 'drug_mol.pkl', 'rb')
    Drug_word2vec = pickle.load(gii)
    gii.close()
    Drug_word_sim = cosine_similarity(Drug_word2vec)

    gii = open(rawdata_dir + '/' + 'glove_wordEmbedding.pkl', 'rb')
    glove_word = pickle.load(gii)
    gii.close()
    side_glove_sim = cosine_similarity(glove_word)

    gii = open(rawdata_dir + '/' + 'drug_target.pkl', 'rb')
    drug_target = pickle.load(gii)
    gii.close()
    drug_target_sim = cosine_similarity(drug_target)

    gii = open(rawdata_dir + '/' + 'fingerprint_similarity.pkl', 'rb')
    drug_f_sim = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + '/' + 'drug_side.pkl', 'rb')
    drug_side = pickle.load(gii)
    gii.close()

    for i in range(data_test.shape[0]):
        drug_side[data_test[i, 0], data_test[i, 1]] = 0

    drug_side_sim = cosine_similarity(drug_side)

    drug_side_label = np.zeros((drug_side.shape[0], drug_side.shape[1]))
    for i in range(drug_side.shape[0]):
        for j in range(drug_side.shape[1]):
            if drug_side[i, j] > 0:
                drug_side_label[i, j] = 1
    drug_side_label_sim = cosine_similarity(drug_side_label)

    drug_features, side_features = [], []
    drug_features.append(drug_Tfeature_one)
    drug_features.append(drug_Tfeature_two)
    drug_features.append(drug_Tfeature_three)
    drug_features.append(drug_Tfeature_four)
    drug_features.append(drug_Tfeature_five)
    drug_features.append(Drug_word_sim)
    drug_features.append(drug_target_sim)
    drug_features.append(drug_f_sim)
    drug_features.append(drug_side_sim)
    drug_features.append(drug_side_label_sim)

    side_drug_sim = cosine_similarity(drug_side.T)
    side_drug_label_sim = cosine_similarity(drug_side_label.T)

    side_features.append(effect_side_semantic)
    side_features.append(side_glove_sim)
    side_features.append(side_drug_sim)
    side_features.append(side_drug_label_sim)

    return drug_features, side_features


def fold_files(data_train, data_test,args):
    rawdata_dir = args.rawpath
    data_train = np.array(data_train)
    data_test = np.array(data_test)

    drug_features, side_features = read_raw_data(rawdata_dir, data_train, data_test)

    drug_features_matrix = drug_features[0]
    for i in range(1, len(drug_features)):
        drug_features_matrix = np.hstack((drug_features_matrix, drug_features[i]))

    side_features_matrix = side_features[0]
    for i in range(1, len(side_features)):
        side_features_matrix = np.hstack((side_features_matrix, side_features[i]))

    drug_test = drug_features_matrix[data_test[:, 0]]
    side_test = side_features_matrix[data_test[:, 1]]
    f_test = data_test[:, 2]

    drug_train = drug_features_matrix[data_train[:, 0]]
    side_train = side_features_matrix[data_train[:, 1]]
    f_train = data_train[:, 2]

    return drug_test, side_test, f_test, drug_train, side_train, f_train

def train_test(data_train, data_test, fold, args):
    drug_test, side_test, f_test, drug_train, side_train, f_train = fold_files(data_train, data_test,args)
    trainset = torch.utils.data.TensorDataset(torch.FloatTensor(drug_train), torch.FloatTensor(side_train),
                                              torch.FloatTensor(f_train))
    testset = torch.utils.data.TensorDataset(torch.FloatTensor(drug_test), torch.FloatTensor(side_test),
                                             torch.FloatTensor(f_test))
    _train = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=16, pin_memory=False)
    _test = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True,
                                        num_workers=16, pin_memory=False) 
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    model = Mulmodel(args).to(device)
  

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    acc_tested = 0
    wf1_tested =  0
    maf1_tested = 0
    mif1_tested = 0
    ka_tested = 0
    mcc_tested = 0
    

    

    for epoch in range(1, args.epochs + 1):
        # ====================   training    ====================
        train(model, _train, optimizer, device)
        # ====================     test       ====================
        
        acc_tr,weighted_f1_tr,macro_f1_tr,micro_f1_tr,kappa_tr,mcc_tr,rating_tr,pred_tr = test(model,_train,device)
        acc_te,weighted_f1_te,macro_f1_te,micro_f1_te,kappa_te,mcc_te,rating_te,pred_te = test(model,_test,device)
        if  acc_te>acc_tested:
            
            acc_tested = acc_te
            wf1_tested =  weighted_f1_te
            maf1_tested = macro_f1_te
            mif1_tested = micro_f1_te
            ka_tested = kappa_te
            mcc_tested = mcc_te
    
            print("Epoch: %d <Test> acc: %.5f, weighted_f1: %.5f, macro_f1: %.5f,micro_f1: %.5f, kappa: %.5f ,mcc: %.5f" % (
            epoch, acc_te,weighted_f1_te,macro_f1_te,micro_f1_te,kappa_te,mcc_te))

            

        

        print("Epoch: %d <Train> acc: %.5f, weighted_f1: %.5f, macro_f1: %.5f,micro_f1: %.5f, kappa: %.5f ,mcc: %.5f" % (
        epoch, acc_tr,weighted_f1_tr,macro_f1_tr,micro_f1_tr,kappa_tr,mcc_tr))
        
        


    
    
    print(" <Best Test> acc_tr: %.5f, weighted_f1: %.5f, macro_f1: %.5f, micro_f1: %.5f, kappa: %.5f ,mcc: %.5f" % (
        acc_tested,wf1_tested,maf1_tested,mif1_tested,ka_tested,mcc_tested))

    return acc_tested,wf1_tested,maf1_tested,mif1_tested,ka_tested,mcc_tested

def kl_func(mu,logvar):
    return - 0.5 * (1 + logvar - mu**2 - torch.exp(logvar)).sum(dim=1)

def calculate_loss(multi_pred,recCon,recAdd,mu, logvar,batch_ratings,batch_drug,batch_side,device):
    kl_div = kl_func(mu, logvar).mean()

    loss_func = nn.CrossEntropyLoss() 
    multi_labels = (batch_ratings.long()-1).to(device) # 0，1，2，3，4

    batch_vec = torch.cat((batch_drug, batch_side), dim=1)


    drug1, drug2, drug3, drug4, drug5, drug6, drug7, drug8, drug9, drug10 = batch_drug.chunk(10, 1)
    side1, side2, side3, side4 = batch_side.chunk(4, 1)

    drugs = drug1+ drug2+ drug3+ drug4+ drug5+ drug6+ drug7+ drug8+ drug9+ drug10
    sides = side1+side2+side3+side4

    add_features = torch.cat((drugs,sides),dim=1)

 
    multi_loss = loss_func(multi_pred,multi_labels)
    reconst_loss = nn.MSELoss(reduction='none')
    rec_loss1 = reconst_loss(recCon,batch_vec.to(device)).sum(dim=-1).mean()
    rec_loss2 = reconst_loss(recAdd,add_features.to(device)).sum(dim=-1).mean()



    
    Loss = multi_loss+0.001*kl_div+0.0001*rec_loss1+0.0001*rec_loss2
   
    return Loss


def train(model, train_loader, optimizer, device):

    model.train()
    avg_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        batch_drug, batch_side, batch_ratings = data
       
        optimizer.zero_grad()
        
        multi_pred,recCon,recAdd,mu,logvar = model(batch_drug,batch_side, device)
        
        loss = calculate_loss(multi_pred,recCon,recAdd,mu, logvar,batch_ratings,batch_drug,batch_side,device)
        loss.backward(retain_graph = True)
        optimizer.step()
        avg_loss += loss.item()

    return 0

def test(model, test_loader, device):
    model.eval()
    pred_all = []
    multi_label_all = []

    for test_drug, test_side, test_ratings in test_loader:
        
        multi_pred,recCon,recAdd,mu, logvar = model(test_drug,test_side, device)

        pred = torch.argmax(multi_pred.cpu(), dim=1).numpy() # 0,1,2,3,4
        pred_all.append(list(pred))
        
        multi_label_all.append(list((test_ratings.long()-1).cpu().numpy())) # 0,1,2,3,4

       

    pred_all = np.array(sum(pred_all, [])) 
    multi_label_all = np.array(sum(multi_label_all, []))

    acc = accuracy_score(multi_label_all,pred_all)
    weighted_f1 = f1_score(multi_label_all,pred_all,average="weighted")
    macro_f1 = f1_score(multi_label_all,pred_all,average="macro") 
    micro_f1 = f1_score(multi_label_all,pred_all,average="micro") 
    kappa = cohen_kappa_score(multi_label_all,pred_all)
    mcc = matthews_corrcoef(multi_label_all,pred_all)

    return acc,weighted_f1,macro_f1,micro_f1,kappa,mcc,multi_label_all,pred_all

def ten_fold(args):
    rawpath = args.rawpath
    gii = open(rawpath+'/drug_side.pkl', 'rb')
    drug_side = pickle.load(gii)
    gii.close()
    final_positive_sample = Extract_positive_negative_samples(drug_side)
    
    final_sample = final_positive_sample

    X = final_sample[:, 0::]
    final_target = final_sample[:, final_sample.shape[1] - 1]
    y = final_target
    data = []
    data_x = []
    data_y = []
    
    
    for i in range(X.shape[0]):
        data_x.append((X[i, 0], X[i, 1]))
        data_y.append((int(float(X[i, 2]))))
        data.append((X[i, 0], X[i, 1], X[i, 2]))
    fold = 1
    kfold = StratifiedKFold(10,random_state=42,shuffle=True)

    total_acc, total_wf1, total_maf1, total_mif1,total_kappa,total_mcc = [], [], [], [], [], []
    for k, (train, test) in enumerate(kfold.split(data_x, data_y)):
        print("==================================fold {} start".format(fold))
        data = np.array(data)
        acc,weighted_f1,macro_f1,micro_f1,kappa,mcc = train_test(data[train].tolist(), data[test].tolist(), fold, args)

        total_acc.append(acc)
        total_wf1.append(weighted_f1)
        total_maf1.append(macro_f1)
        total_mif1.append(micro_f1)
        total_kappa.append(kappa)
        total_mcc.append(mcc)
        print("==================================fold {} end".format(fold))
        
        print('Total_acc:')
        print(np.mean(total_acc))
        print('Total_weighted_f1:')
        print(np.mean(total_wf1))
        print('Total_macro_f1:')
        print(np.mean(total_maf1))
        print('Total_micro_f1:')
        print(np.mean(total_mif1))
        print('Total_kappa:')
        print(np.mean(total_kappa))
        print('Total_mcc:')
        print(np.mean(total_mcc))

        with open("log.txt",'a') as f:
            print("fold:"+str(fold),file=f)

            print('Total_acc:',file=f)
            print(np.mean(total_acc),file=f)

            print('Total_weighted_f1:',file=f)
            print(np.mean(total_wf1),file=f)

            print('Total_macro_f1:',file=f)
            print(np.mean(total_maf1),file=f)

            print('Total_micro_f1:',file=f)
            print(np.mean(total_mif1),file=f)

            print('Total_kappa:',file=f)
            print(np.mean(total_kappa),file=f)

            print('Total_mcc:',file=f)
            print(np.mean(total_mcc),file=f)

            print("\n",file=f)
        fold += 1

        sys.stdout.flush()


def Extract_positive_negative_samples(DAL):
    k = 0
    interaction_target = np.zeros((DAL.shape[0]*DAL.shape[1], 3)).astype(int)
    for i in range(DAL.shape[0]):
        for j in range(DAL.shape[1]):
            interaction_target[k, 0] = i
            interaction_target[k, 1] = j
            interaction_target[k, 2] = DAL[i, j]
            k = k + 1
    data_shuffle = interaction_target[interaction_target[:, 2].argsort()]
    number_positive = len(np.nonzero(data_shuffle[:, 2])[0])
    final_positive_sample = data_shuffle[interaction_target.shape[0] - number_positive::]

    return final_positive_sample

def main():
    # Training settings
    parser = argparse.ArgumentParser(description = 'Model')
    parser.add_argument('--epochs', type = int, default = 128,
                        metavar = 'N', help = 'number of epochs to train')
    parser.add_argument('--lr', type = float, default = 0.0001,
                        metavar = 'FLOAT', help = 'learning rate')
    parser.add_argument('--embed_dim', type = int, default = 128,
                        metavar = 'N', help = 'embedding dimension')
    parser.add_argument('--weight_decay', type = float, default = 0.00001,
                        metavar = 'FLOAT', help = 'weight decay')
    parser.add_argument('--N', type = int, default = 30000,
                        metavar = 'N', help = 'L0 parameter')
    parser.add_argument('--droprate', type = float, default = 0.4,
                         metavar = 'FLOAT', help = 'dropout rate')
    parser.add_argument('--gp', type = int, default = 64,
                        metavar = 'gp', help = 'hyper_gauss')

    parser.add_argument('--batch_size', type = int, default = 128,
                        metavar = 'N', help = 'input batch size for training')
    parser.add_argument('--test_batch_size', type = int, default = 128,
                        metavar = 'N', help = 'input batch size for testing')
    parser.add_argument('--dataset', type = str, default = 'hh',
                        metavar = 'STRING', help = 'dataset')
    parser.add_argument('--rawpath', type=str, default='/homeb/lidingxi/SDpred/data',
                        metavar='STRING', help='rawpath')
    args = parser.parse_args()

    print('-------------------- Hyperparams --------------------')
    print('N: ' + str(args.N))
    print('weight decay: ' + str(args.weight_decay))
    print('dropout rate: ' + str(args.droprate))
    print('learning rate: ' + str(args.lr))
    print('dimension of embedding: ' + str(args.embed_dim))
    ten_fold(args)

if __name__ == "__main__":
    main()