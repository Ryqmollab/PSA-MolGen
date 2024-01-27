import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from SE_weight_module import SEWeightModule
from PSA_Module import PSAModule

voc_set=['pad', 'bos', 'eos', '5', 'Y', ')', 'Z', '[', ']', '-', 
    'S', '1', 'O', 'N', "'", ' ', 'C', '(', 'n', 'c', '#', 's', '6', 
    'X', '4', ',', '2', 'o', 'F', '=', '3', '.', 'I', '/', '+', '\\', '@', 'H', 'P']
vocab_i2c_v1 = {i: x for i, x in enumerate(voc_set)}
vocab_c2i_v1 = {vocab_i2c_v1[i]: i for i in vocab_i2c_v1}
def decode_smiles(in_tensor):
    """
    Decodes input tensor to a list of strings.
    :param in_tensor:
    :return:
    """
    gen_smiles = []
    for sample in in_tensor:
        csmile = ""
        for xchar in sample[1:]:
            if xchar == 2:
                break
            csmile += vocab_i2c_v1[xchar]
        gen_smiles.append(csmile)
    return gen_smiles

def decode_smiles1(sample):
    """
    Decodes input tensor to a list of strings.
    :param in_tensor:
    :return:
    """
    csmile = ""
    # print('sample:',sample)
    for xchar in sample[1:]:
        if xchar == 2:
            break
        csmile += vocab_i2c_v1[xchar]
        
    return csmile

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        enc = []
        shr_enc = []
        self.relu = nn.ReLU()
        in_channels=19
        out_channels = 32
        for i in range(4):
            enc.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            enc.append(nn.BatchNorm3d(out_channels))
            enc.append(nn.ReLU())
            in_channels=out_channels

            if (i+1) % 2 ==0: 
                out_channels *= 2
                enc.append(nn.MaxPool3d((2, 2, 2)))
        enc.pop()
        self.fc11 = nn.Linear(512,512)
        self.fc12 = nn.Linear(512,512)
        self.single_encoder=nn.Sequential(*enc)
        enc1 =[]
        in_channels1=64
        out_channels1=128
        for i in range(4):
            enc1.append(nn.Conv3d(in_channels1, out_channels1, kernel_size=3, padding=1))
            enc1.append(nn.BatchNorm3d(out_channels1))
            enc1.append(nn.ReLU())
            in_channels1=out_channels1

            if (i+1) % 2 ==0: 
                out_channels1 *= 2
                enc1.append(nn.MaxPool3d((2, 2, 2)))
                
        enc1.pop()
        self.single_encoder1=nn.Sequential(*enc1)
        self.PSAmodule = PSAModule(inplans=64, planes=64, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16])
        self.PSAmodule1 = PSAModule(inplans=256, planes=256, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16])
        self.maxpool = nn.MaxPool3d((2, 2, 2))
        enc2 =[]
        in_channels2=1
        out_channels2=32
        for i in range(4):
            enc2.append(nn.Conv2d(in_channels2, out_channels2, kernel_size=4, padding=1))
            enc2.append(nn.BatchNorm2d(out_channels2))
            enc2.append(nn.ReLU())
            in_channels2=out_channels2

            if (i+1) % 2 ==0: 
                out_channels2 *= 2
                enc2.append(nn.MaxPool2d(2,stride=2))
                
        enc2.pop()
        self.single_encoder2=nn.Sequential(*enc2)
        enc3 =[]
        in_channels3=64
        out_channels3=128
        for i in range(4):
            enc3.append(nn.Conv2d(in_channels3, out_channels3, kernel_size=4, padding=1))
            enc3.append(nn.BatchNorm2d(out_channels3))
            enc3.append(nn.ReLU())
            in_channels3=out_channels3

            if (i+1) % 2 ==0: 
                out_channels3 *= 2
                enc3.append(nn.MaxPool2d(2,stride=2))
                
        enc3.pop()
        self.single_encoder3=nn.Sequential(*enc3)
        
        self.PSAmodule2 = PSAModule_complex(inplans=64, planes=64, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16])
        self.PSAmodule3 = PSAModule_complex(inplans=256, planes=256, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16])
        self.maxpool1 = nn.MaxPool2d((2,2))
    def reparametrize(self, mu, logvar,factor):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()# 生成随机数组
        eps = Variable(eps).cuda()
        return (eps.mul(std)* factor).add_(mu)
    
    def forward(self, input_data,input_complex,factor=1.):
        result=[]   
        x = self.single_encoder(input_data)
        y = self.PSAmodule(x)
        y =x*y
        
        x1 = self.single_encoder1(y)
        x = self.PSAmodule1(x1)
        x =x*x1
        x = self.maxpool(x)
        x = x.mean(dim=2).mean(dim=2).mean(dim=2) #压缩指定的维度
        
        input_complex=input_complex.permute(0,3,1,2)
        cpl0 = self.single_encoder2(input_complex)
        cpl1 = self.PSAmodule2(cpl0)
        cpl = cpl0*cpl1
        cpl2 = self.single_encoder3(cpl)
        cpl3 = self.PSAmodule3(cpl2)
        cpl = cpl2* cpl3
        cpl = self.maxpool1(cpl) 
        cpl = cpl.mean(dim=2).mean(dim=2)
        x = torch.cat((cpl,x),dim=1)
        
        single_mu,single_logvar = self.fc11(x), self.fc12(x)
        single_latent = self.reparametrize(single_mu,single_logvar,factor=factor)# 重新参数化成正态分布
        
        result.extend([single_mu,single_logvar, single_latent])
        return result
    
class Decoder(nn.Module):
    def __init__(self):
        """Set the hyper-parameters and build the layers."""
        super(Decoder, self).__init__()
        embed_size=512
        hidden_size=1024
        vocab_size=39
        num_layers=1
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
        self.linear1 =  nn.Linear(embed_size,960)
        self.linear2 =  nn.Linear(embed_size,512-8)
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        """Decode shapes feature vectors and generates SMILES."""
 
        embedding = self.embedding(captions)
        embedding = torch.cat((features.unsqueeze(1), embedding), 1)
        packed = pack_padded_sequence(embedding, lengths, batch_first=True,enforce_sorted=False)

        hiddens, _ = self.lstm(packed)

        outputs = self.linear(hiddens[0])
   
        return outputs
    
    def sample(self, features,states=None):
        """Samples SMILES tockens for given shape features (Greedy search)."""
        sampled_ids = []
  
        inputs = features.unsqueeze(1)

        for i in range(80):
            hiddens, states = self.lstm(inputs,states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.max(1)[1]#[1]表示索引值

            sampled_ids.append(predicted)
            inputs = self.embedding(predicted) 
            inputs = inputs.unsqueeze(1)
            
        return sampled_ids
    def sample_prob(self, features, states=None):
        """Samples SMILES tockens for given shape features (probalistic picking)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(80):  # maximum sampling length
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            if i == 0:
                predicted = outputs.max(1)[1]
            else:
                probs = F.softmax(outputs, dim=1)

                # Probabilistic sample tokens
                if probs.is_cuda:
                    probs_np = probs.data.cpu().numpy()
                else:
                    probs_np = probs.data.numpy()

                rand_num = np.random.rand(probs_np.shape[0])
                iter_sum = np.zeros((probs_np.shape[0],))
                tokens = np.zeros(probs_np.shape[0], dtype=np.int)

                for i in range(probs_np.shape[1]):
                    c_element = probs_np[:, i]
                    iter_sum += c_element
                    valid_token = rand_num < iter_sum
                    update_indecies = np.logical_and(valid_token,
                                                     np.logical_not(tokens.astype(np.bool)))#astype:0代表False 非0代表True
                    tokens[update_indecies] = i

                # put back on the GPU.
                if probs.is_cuda:
                    predicted = Variable(torch.LongTensor(tokens.astype(np.int)).cuda())
                else:
                    predicted = Variable(torch.LongTensor(tokens.astype(np.int)))#int:0代表0，非0代表1

            sampled_ids.append(predicted)
        
            inputs = self.embedding(predicted)
            inputs = inputs.unsqueeze(1)
      
        return sampled_ids
    


