# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 00:55:33 2023

@author: armel
"""

import sys
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import random


path = 'model_weights.pth'

index=10 #hours in the hear days*24 + hours of when you want to predict

class Lstm(nn.Module):
    
    def __init__(self,nhid):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(1, nhid)
        self.mlp = nn.Linear(nhid, 1)
        self.hidden_size = nhid


    def forward(self, x):
        # x = B, T, d
        xx = x.transpose(0,1)
        y,_=self.lstm(xx)
        T,B,H = y.shape
        
        y = self.mlp(y.view(T*B,H))
        y = y.view(T,B,-1)
        y = y.transpose(0,1)
        
        return y
    
class Rnn(nn.Module):
    def __init__(self,nhid):
        super(Rnn, self).__init__()
        self.rnn = nn.RNN(1,nhid)
        self.mlp = nn.Linear(nhid,1)

    def forward(self,x):
        # x = B, T, d
        xx = x.transpose(0,1)
        y,_=self.rnn(xx)
        T,B,H = y.shape
        
        y = self.mlp(y.view(T*B,H))
        y = y.view(T,B,-1)
        y = y.transpose(0,1)
        return y

mod_rnn = Rnn(10)
#mod_rnn = Lstm(10)


#charge
mod_rnn.load_state_dict(torch.load(path))
    
dn = 50
k=10000

with open("frequentation_queue/2019.csv","r") as f: ls=f.readlines()
data = torch.Tensor([float(l.split(',')[1])/50 for l in ls]).view(1,-1,1)
max_value = torch.max(data)
min_val= torch.min(data)

# Normaliser le tensor entre 0 et 1
data = torch.div(data, max_value)
with torch.no_grad():
    predictions = mod_rnn(data)
    predictions= predictions*(max_value-min_val*k)
    prediction =int(predictions[0, index, 0])
    prediction = "current predicted waiting time: " +str(int(prediction/60)) +"hours and "+str(prediction%60)+" minutes. "
    print(prediction)
#display the prediction

