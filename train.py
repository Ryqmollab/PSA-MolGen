import torch
from torchvision import datasets, transforms
from torch.utils.data import dataloader
from torch.utils import data
import matplotlib.pyplot as plt
from network import Encoder,Decoder
import torch.nn as nn
import joblib
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import tqdm
torch.backends.cudnn.enable =True
import pandas as pd
import torch.autograd as autograd
import os
from data_generator import queue_datagen
from keras.utils.data_utils import GeneratorEnqueuer
import argparse
from argparse import RawTextHelpFormatter, RawDescriptionHelpFormatter
from sklearn import preprocessing, model_selection
import random
#####################

device = torch.device('cuda')
d = """Train or predict the features based on protein-ligand complexes.

"""
parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
parser.add_argument("-i", "--input", required=True, help="Path to input .npy file.")
parser.add_argument("-fn_train", type=str, default=["features_1.csv", ], nargs="+",
                        help="Input. The docked cplx feature training set.")
parser.add_argument("-fn_test", type=str, default=["features_2.csv", ],nargs="+",
                    help="Input. The PDBBind feature validating set.")
parser.add_argument("--remove_H", type=int, default=0,
                        help="Input, optional. Default is 0. Whether remove hydrogens. ")
parser.add_argument("-n_features", default=3840, type=int,
                        help="Input. Default is 3840. Number of features in the input dataset.")
parser.add_argument("-scaler", type=str, default="StandardScaler.model",
                        help="Output. The standard scaler file to save. ")
parser.add_argument("-reshape", type=int, default=[64, 60, 1], nargs="+",
                        help="Input. Default is 64 60 1. Reshape the dataset. ")
parser.add_argument('-epochs', type=int, default=10, help='epoch (default=1000)')
args = parser.parse_args()
savedir='./model/'
log_file = open(os.path.join(savedir, "log.txt"), "w")

cap_loss = 0.
batch_size =32

savedir = args.output_dir
os.makedirs(savedir, exist_ok=True)
smiles = np.load(args.input)
smiles = smiles.tolist()
random.shuffle(smiles)
train_smiles = smiles[:80064]
test_smiles =smiles[:32000]

train_smiles = np.array(train_smiles)
test_smiles = np.array(test_smiles)
import multiprocessing
multiproc = multiprocessing.Pool(6)
my_gen = queue_datagen(train_smiles, batch_size=batch_size, mp_pool=multiproc)
test_gen = queue_datagen(test_smiles, batch_size=batch_size, mp_pool=multiproc)

mg = GeneratorEnqueuer(my_gen)
tg = GeneratorEnqueuer(test_gen)
mg.start()
tg.start()
mt_gen = mg.get()
tg_gen = tg.get()

def remove_shell_features(dat, shell_index, features_n=64):
    
    df = dat.copy()

    start = shell_index * features_n
    end = start + features_n

    zeroes = np.zeros((df.shape[0], features_n))

    df[:, start:end] = zeroes

    return df


def remove_atomtype_features(dat, feature_index, shells_n=60):

    df = dat.copy()

    for i in range(shells_n):
        ndx = i * 64 + feature_index

        zeroes = np.zeros(df.shape[0])
        df[:, ndx] = zeroes

    return df


def remove_all_hydrogens(dat, n_features):
    df = dat.copy()

    for f in df.columns.values[:n_features]:
        if "H_" in f or "_H_" in f:
            v = np.zeros(df.shape[0])
            df[f] = v

    return df


class NoamOptRMSprop:
    "Optimizer wrapper that implements rate decay (adapted from\
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)"
    def __init__(self, model_size, factor, warmup, model_parameters, optimizer_params):
        self.optimizer = optim.RMSprop(model_parameters, **optimizer_params)  # 使用RMSprop优化器
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size

        self.state_dict = self.optimizer.state_dict()
        self.state_dict['step'] = 0
        self.state_dict['rate'] = 0

    def step(self):
        "Update parameters and rate"
        self.state_dict['step'] += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.state_dict['rate'] = rate
        self.optimizer.step()
        for k, v in self.optimizer.state_dict().items():
            self.state_dict[k] = v

    def rate(self, step=None):
        "Implement 'lrate' above"
        if step is None:
            step = self.state_dict['step']
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def load_state_dict(self, state_dict):
        self.state_dict = state_dict
        
lr = 1e-4
step_decay_weight = 0.95
lr_decay_step = 20000
def exp_lr_scheduler(optimizer, step, init_lr=lr, lr_decay_step=lr_decay_step, step_decay_weight=step_decay_weight):
    current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return optimizer
###############################################

num_epoch = 100
beta_weight = 0.075
loss_diff = DiffLoss()
loss_latent = LatentLoss()
loss_diff = loss_diff.cuda()
loss_latent = loss_latent.cuda()
loss_similarity = SimLoss().cuda()

def noise(size):
    n = torch.randn(size, 128,8,8,8).cuda()
    return n

encoder = Encoder()
decoder = Decoder()
encoder.to(device)
decoder.to(device)

caption_params = list(decoder.parameters()) + list(encoder.parameters())
caption_optimizer = NoamOptRMSprop(128, 1, 10000,
                                   model_parameters=caption_params,
                                   optimizer_params={'lr': 0, 'eps': 1e-9})
encoder.train()
decoder.train()

dg_criterion = nn.BCELoss() 
lr = 1e-4
loss_fn = nn.BCELoss()
criterion = nn.CrossEntropyLoss()

####################

X, y = None, []
for i, fn in enumerate(args.fn_train):
    if os.path.exists(fn):
        df = pd.read_csv(fn, index_col=0, header=0).dropna()
        if args.remove_H:
            df = remove_all_hydrogens(df, args.n_features)
        if i == 0:
            X = df.values[:, :args.n_features]
        else:
            X = np.concatenate((X, df.values[:, :args.n_features]), axis=0)
            
Xtest, ytest = None, []
for i, fn in enumerate(args.fn_test):
    if os.path.exists(fn):
        df = pd.read_csv(fn, index_col=0, header=0).dropna()
        if args.remove_H:
            df = remove_all_hydrogens(df, args.n_features)

        if i == 0:
            Xtest = df.values[:, :args.n_features]
        else:
            Xtest = np.concatenate((Xtest, df.values[:, :args.n_features]), axis=0)

scaler = preprocessing.StandardScaler()
X_train_val = np.concatenate((X, Xtest), axis=0)
scaler.fit(X_train_val)#得到scaler，scaler里面存的有计算出来的均值和方差
        
joblib.dump(scaler, args.scaler)

Xtrain = scaler.transform(X).reshape((-1, args.reshape[0],#再用scaler中的均值和方差来转换X，使X标准化
                                        args.reshape[1],
                                        args.reshape[2]))
Xtest = scaler.transform(Xtest).reshape((-1, args.reshape[0],
                                            args.reshape[1],
                                            args.reshape[2]))
xtrain = torch.utils.data.DataLoader(dataset=Xtrain,batch_size=32,shuffle=True)
xtest = torch.utils.data.DataLoader(dataset=Xtest,batch_size=32,shuffle=True)

print("DataSet Scaled")


#####################
tq_gen = enumerate(mt_gen)
sq_gen =enumerate(tg_gen)
log_file = open(os.path.join(savedir, "log.txt"), "w")

alpha_weight = 0.01
beta_weight = 0.25
plt.figure()
lists=[]
lists1=[]
lists2=[]

cap_loss = 0.
latloss =0.
criterion_loss=0.
caption_start = 40         

def train(epoch):
    encoder.train()
    decoder.train()
    
    train_loss =0
    for i, (real_data,caption, length) in tq_gen:
        
        cap_loss = 0.
        data_source_iter = iter(xtrain)
        input_complex = data_source_iter.__next__()
        input_complex = input_complex.float().to(device)
            
        real_data=real_data.cuda()
  
              
        caption_= Variable(caption.long()).to(device)

            
        target = pack_padded_sequence(caption_, length, batch_first=True,enforce_sorted=False)[0]#按列取出元素
        decoder.zero_grad()
        encoder.zero_grad()
                   
        single_mu,single_logvar,features = encoder(real_data,input_complex)
  
        
        latloss = loss_latent(single_mu,single_logvar)  

        cap_loss += latloss 
 
        outputs = decoder(features, caption_,length)
               

            
        criterion_loss=criterion(outputs, target)
        cap_loss +=criterion_loss
               
        cap_loss.backward()
        train_loss +=cap_loss.item()      
        caption_optimizer.step()
     

        if (i + 1) % 100 == 0:
           
        if (i+1) % 500 ==0:
            torch.save(decoder.state_dict(),os.path.join(savedir,'decoder-%d.pkl' % (i + 1)))
            torch.save(encoder.state_dict(),
                    os.path.join(savedir,
                                    'encoder-%d.pkl' % (i + 1)))
        result = "Step: {}, loss: {:.5f}, ".format(i + 1,float(cap_loss.data.cpu().numpy()) if type(cap_loss) != float else 0.)
        log_file.write(result + "\n")
        log_file.flush() 
    print('> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss ))
    return train_loss

def test():
    encoder.eval()
    decoder.eval()
    test_loss = 0
    cap_loss = 0.
    with torch.no_grad():
        for i, (real_data,caption, length) in sq_gen:
            data_source_iter = iter(xtest)
            input_complex = data_source_iter.__next__()
            input_complex = input_complex.float().to(device)
               
            real_data=real_data.cuda()
                
            caption_= Variable(caption.long()).to(device)
            
            target = pack_padded_sequence(caption_, length, batch_first=True,enforce_sorted=False)[0]#按列取出元素
                
            single_mu,single_logvar,features = encoder(real_data,input_complex)
            latloss = loss_latent(single_mu,single_logvar)  

            cap_loss += latloss 

            outputs = decoder(features, caption_,length)

            criterion_loss=criterion(outputs, target)

            cap_loss +=criterion_loss
        cap_loss /=len(sq_gen)
        return cap_loss
    
train_loss_list = []
test_loss_list = []
for epoch in range(1, args.epochs+1):
    train_loss = train(epoch)
    train_loss_list.append(train_loss)
    test_loss = test()
    test_loss_list.append(test_loss)

    
    torch.save(decoder.state_dict(),os.path.join(savedir,'decoder-%d.pkl' % (epoch + 1)))
    torch.save(encoder.state_dict(),
                os.path.join(savedir,
                                'encoder-%d.pkl' % (epoch + 1)))
        
    result = "Step: {}, train_loss: {:.5f}, test_loss: {:.5f}".format(i + 1,float(train_loss.data.cpu().numpy()) if type(train_loss) != float else 0.,float(test_loss.data.cpu().numpy()) if type(test_loss) != float else 0.)
    log_file.write(result + "\n")
    log_file.flush()
   

print('End Training')
