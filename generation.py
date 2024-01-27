#!/usr/bin/env python
# coding: utf-8
from mol_gen import MolecularGenerator
import os
import torch
import pandas as pd
import csv
from rdkit import Chem
from rdkit.Chem import AllChem
torch.backends.cudnn.enabled = False
class MolecularGenerator:
    def __init__(self, use_cuda=True):

        self.use_cuda = False

        self.encoder = Encoder()
        self.decoder = Decoder()

     
        self.encoder.eval()
        self.decoder.eval()

        if use_cuda:
            assert torch.cuda.is_available()
            self.encoder.cuda()
            self.decoder.cuda()
      
            self.use_cuda = True
   
    def load_weight(self, encoder_weights, decoder_weights):
        """
        Load the weights of the models.
        :param vae_weights: str - VAE model weights path
        :param encoder_weights: str - captioning model encoder weights path
        :param decoder_weights: str - captioning model decoder model weights path
        :return: None
        """
      
        self.encoder.load_state_dict(torch.load(encoder_weights, map_location='cpu'))
        self.decoder.load_state_dict(torch.load(decoder_weights, map_location='cpu'))
        
    def generate_molecules(self, n_attemps=300, lam_fact=1., probab=False,filter_unique_valid=True):
        """
        Generate novel compounds from a seed compound.
        :param smile_str: string - SMILES representation of a molecule
        :param n_attemps: int - number of decoding attempts
        :param lam_fact: float - latent space pertrubation factor
        :param probab: boolean - use probabilistic decoding
        :return: list of RDKit molecules.
        """
        z = Variable(torch.randn(n_attemps, 512)).to(device)
        if probab:
            captions = self.decoder.sample_prob(z)
        else:
            captions = self.decoder.sample(z)
        return captions

my_gen = MolecularGenerator(use_cuda=True)  # set use_cuda=False if you do not have a GPU.

from network import  Encoder,Decoder
device = torch.device('cuda')

# Load the weights of the models
encoder_weights =  os.path.join("./mode/encoder.pkl")
decoder_weights =os.path.join("./model/decoder.pkl")
my_gen.load_weight(encoder_weights, decoder_weights)

gen_mols = my_gen.generate_molecules(
                                        n_attemps=500,  # How many attemps of generations will be carried out
                                        lam_fact=5.,  # Variability factor
                                        probab=True, # Probabilistic RNN decoding
                                        filter_unique_valid=True)  # Filter out invalids and replicates
with open('./model/gen_smi_test.csv', 'a') as f:
    for i in range(len(gen_mols)):
        smi = gen_mols[i]
        f.write(smi+"\n")
