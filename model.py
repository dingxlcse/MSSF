from torch import nn
import torch
import numpy as np


class GaussianParametrizer(nn.Module):
    def __init__(self,
                 feature_dim,
                 latent_dim, 
                 ):
        super(GaussianParametrizer,self).__init__()

        self.h1 = nn.Linear(feature_dim, latent_dim)
        self.h2 = nn.Linear(feature_dim, latent_dim)
    
    def forward(self, x):
        mu = self.h1(x)
        log_var = self.h2(x) 
        return mu, log_var 


         

class Attention(nn.Module): # transform
    def __init__(self,inputdim,heads):
        super(Attention,self).__init__()

        self.inputdim = inputdim
        self.heads = heads
        self.dq = self.dk = self.dv = inputdim//heads

        self.WQ = torch.nn.Linear(self.inputdim, self.dq * self.heads, bias=False)
        self.WK = torch.nn.Linear(self.inputdim, self.dk * self.heads, bias=False)
        self.WV = torch.nn.Linear(self.inputdim, self.dv * self.heads, bias=False)

        self.LN1 = torch.nn.LayerNorm(inputdim)
        self.l1 = torch.nn.Linear(inputdim, inputdim)
        self.LN2 = torch.nn.LayerNorm(inputdim)

    def forward(self,x):



        Q = self.WQ(x).view(-1, self.heads, self.dq).transpose(0, 1) 
        K = self.WK(x).view(-1, self.heads, self.dk).transpose(0, 1)
        V = self.WV(x).view(-1, self.heads, self.dv).transpose(0, 1)

        QK = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dk)
        QK = torch.nn.Softmax(dim=-1)(QK)
        att = torch.matmul(QK, V)                                 
        
        att = att.transpose(1, 2).reshape(-1, self.heads * self.dv)  

        x = self.LN1(att + x)
        output = self.l1(x)
        x = self.LN2(output + x)
                                                    
        return x




class EncoderConnection(nn.Module):
    def __init__(self,drugs_inputdim,sides_inputdim,latent_dim,feature_dim,heads,dp):
        super(EncoderConnection,self).__init__()

        # parameters
        self.drugs_inputdim = drugs_inputdim
        self.sides_inputdim = sides_inputdim
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.heads = heads

        self.reluDrop = nn.Sequential(nn.LeakyReLU(0.01),nn.Dropout(dp))

        # encoder
        self.l1 = nn.Sequential(
            nn.Linear(self.drugs_inputdim+self.sides_inputdim,self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            self.reluDrop

        )
        self.attention = Attention(inputdim=self.latent_dim,heads=self.heads)
        self.l2 = nn.Linear(self.latent_dim,self.feature_dim)

        # decoder
        self.l3 = nn.Sequential(
            nn.Linear(self.feature_dim,self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            self.reluDrop,

            nn.Linear(self.latent_dim,self.drugs_inputdim+self.sides_inputdim)
        )


    def forward(self,drugs,sides):
        x = torch.cat((drugs,sides),dim=1)
        x = self.l1(x)
        x = self.attention(x)
        x = self.l2(x)

        rec_conn = self.l3(x)

        return x,rec_conn



class EncoderAddition(nn.Module):
    def __init__(self,drugs_inputdim,sides_inputdim,latent_dim,feature_dim,heads,dp):
            super(EncoderAddition,self).__init__()

            # parameters
            self.drugs_inputdim = drugs_inputdim
            self.sides_inputdim = sides_inputdim
            self.latent_dim = latent_dim
            self.feature_dim = feature_dim
            self.heads = heads


            self.reluDrop = nn.Sequential(nn.LeakyReLU(0.01),nn.Dropout(dp))

            # encoder
            self.l1 = nn.Sequential(
                nn.Linear(self.drugs_inputdim+self.sides_inputdim,self.latent_dim),
                nn.BatchNorm1d(self.latent_dim),
                self.reluDrop

            )
            self.attention = Attention(inputdim=self.latent_dim,heads=self.heads)
            self.l2 = nn.Linear(self.latent_dim,self.feature_dim)

            # decoder
            self.l3 = nn.Sequential(
                nn.Linear(self.feature_dim,self.latent_dim),
                nn.BatchNorm1d(self.latent_dim),
                self.reluDrop,

                nn.Linear(self.latent_dim,self.drugs_inputdim+self.sides_inputdim)
            )


    def forward(self,drug_features,side_features):
        drug1, drug2, drug3, drug4, drug5, drug6, drug7, drug8, drug9, drug10 = drug_features.chunk(10, 1)
        side1, side2, side3, side4 = side_features.chunk(4, 1)

        drugs = drug1+ drug2+ drug3+ drug4+ drug5+ drug6+ drug7+ drug8+ drug9+ drug10
        sides = side1+side2+side3+side4

        add_features = torch.cat((drugs,sides),dim=1)
        x = self.l1(add_features)
        x = self.attention(x)
        x = self.l2(x)

        rec_add = self.l3(x)

        return x,rec_add




class Preprocess(nn.Module):
    def __init__(self,drug_inputdim,side_inputdim,embeddim,dp):
        super(Preprocess,self).__init__()
        self.drug_inputdim = drug_inputdim
        self.side_inputdim = side_inputdim
        self.embdeddim = embeddim

        self.reluDrop = nn.Sequential(nn.LeakyReLU(0.01),nn.Dropout(dp))

        # drug preprocessing
        self.drug1_pre = nn.Sequential(
            nn.Linear(self.drug_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.drug2_pre = nn.Sequential(
            nn.Linear(self.drug_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.drug3_pre = nn.Sequential(
            nn.Linear(self.drug_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.drug4_pre = nn.Sequential(
            nn.Linear(self.drug_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.drug5_pre = nn.Sequential(
            nn.Linear(self.drug_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.drug6_pre = nn.Sequential(
            nn.Linear(self.drug_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.drug7_pre = nn.Sequential(
            nn.Linear(self.drug_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.drug8_pre = nn.Sequential(
            nn.Linear(self.drug_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.drug9_pre = nn.Sequential(
            nn.Linear(self.drug_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.drug10_pre = nn.Sequential(
            nn.Linear(self.drug_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )

        # side preprocessing
        self.side1_pre = nn.Sequential(
            nn.Linear(self.side_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.side2_pre = nn.Sequential(
            nn.Linear(self.side_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.side3_pre = nn.Sequential(
            nn.Linear(self.side_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.side4_pre = nn.Sequential(
            nn.Linear(self.side_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )



    def forward(self,drug_features,side_features):
        drug1, drug2, drug3, drug4, drug5, drug6, drug7, drug8, drug9, drug10 = drug_features.chunk(10, 1)
        side1, side2, side3, side4 = side_features.chunk(4, 1)
        
        drug1 = self.drug1_pre(drug1)
        drug2 = self.drug2_pre(drug2)
        drug3 = self.drug3_pre(drug3)
        drug4 = self.drug4_pre(drug4)
        drug5 = self.drug5_pre(drug5)
        drug6 = self.drug6_pre(drug6)
        drug7 = self.drug7_pre(drug7)
        drug8 = self.drug8_pre(drug8)
        drug9 = self.drug9_pre(drug9)
        drug10 = self.drug10_pre(drug10)
        
        side1 = self.side1_pre(side1)
        side2 = self.side2_pre(side2)
        side3 = self.side3_pre(side3)
        side4 = self.side4_pre(side4)

        drugs = [drug1,drug2,drug3,drug4,drug5,drug6,drug7,drug8,drug9,drug10]
        sides = [side1,side2,side3,side4]

        return drugs,sides


class CrossProduction(nn.Module): 
    def __init__(self,cross_dim,feature_dim,input_channel):
        super(CrossProduction,self).__init__()

        self.cross_dim = cross_dim
        self.feature_dim = feature_dim
        self.kernel_size = 4
        self.strides = 4
        self.latent_channel = 32
        self.input_channel = input_channel


        self.cnn = nn.Sequential(
        
            nn.Conv2d(self.input_channel, self.latent_channel, kernel_size=self.kernel_size, stride=self.strides), 
            nn.BatchNorm2d(self.latent_channel),
            nn.ReLU(),
            
            nn.Conv2d(self.latent_channel, self.latent_channel, kernel_size=self.kernel_size, stride=self.strides), 
            nn.BatchNorm2d(self.latent_channel),
            nn.ReLU(),
            
            nn.Conv2d(self.latent_channel, self.latent_channel, kernel_size=self.kernel_size, stride=self.strides), 
            nn.BatchNorm2d(self.latent_channel),
            nn.ReLU(),
        
        )

    def forward(self,drugs,sides):
        crosspro = []
        for i in range(len(drugs)):
            for j in range(len(sides)):
                crosspro.append(torch.bmm(drugs[i].unsqueeze(2), sides[j].unsqueeze(1)))
        # add channels
        crosspro2d = crosspro[0].view((-1, 1, self.cross_dim, self.cross_dim))

        for i in range(1, len(crosspro)):
            crossproEach = crosspro[i].view((-1, 1, self.cross_dim, self.cross_dim))
            crosspro2d = torch.cat([crosspro2d, crossproEach], dim=1)

        x = self.cnn(crosspro2d).view((-1,self.feature_dim))
        return x

        
class Classifier(nn.Module):
    def __init__(self,latent_dim,classes,dp):
        super(Classifier,self).__init__()

        self.latent_dim = latent_dim
        self.classes = classes

        self.reluDrop = nn.Sequential(nn.LeakyReLU(0.01),nn.Dropout(dp))

        self.classifier=nn.Sequential(      
            nn.Linear(self.latent_dim,self.latent_dim//2),
            self.reluDrop,

            nn.Linear(self.latent_dim//2,self.classes), 
        ) 


    def forward(self,x):
        x = self.classifier(x)
        return x



class Mulmodel(nn.Module):
    def __init__(self,args):
        super(Mulmodel,self).__init__()
        self.args=args
        self.feature_nums = 4*10

        gp =args.gp
        droprate = args.droprate

        self.encoderConnection = EncoderConnection(drugs_inputdim=757*10,sides_inputdim=994*4,latent_dim=256,feature_dim=128,heads=4,dp=droprate)
        self.encoderAddition = EncoderAddition(drugs_inputdim=757,sides_inputdim=994,latent_dim=256,feature_dim=128,heads = 4,dp=droprate)
        self.preprocess = Preprocess(drug_inputdim=757,side_inputdim=994,embeddim=128,dp=droprate)
        self.crossProduction = CrossProduction(cross_dim=128,feature_dim=128,input_channel=self.feature_nums)

        self.attention = Attention(inputdim=128*3,heads=4)

        self.gaussian_parametrizer = GaussianParametrizer(feature_dim=128*3,latent_dim=gp)

        self.classifier = Classifier(latent_dim=gp,classes=5,dp=droprate)


    def reparameterize(self, mu, logvar): 
        if self.training:
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std) 
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self,drugs,sides,device):     
        drugs = drugs.to(device)
        sides = sides.to(device)
        
        feature1,recCon = self.encoderConnection(drugs,sides)

        feature2,recAdd = self.encoderAddition(drugs,sides)

        drugs,sides = self.preprocess(drugs,sides)

        feature3 = self.crossProduction(drugs,sides)

        features = torch.cat((feature1,feature2,feature3),dim=1)

        features = self.attention(features)

        mu,logvar = self.gaussian_parametrizer(features)

        latent_features = self.reparameterize(mu,logvar)

        results = self.classifier(latent_features)

        

        return results,recCon,recAdd,mu,logvar