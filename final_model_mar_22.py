'''
The file allows the construction of models to learn the patterns. The model trained is the lstm model which takes in a sequence of (lat,lon) and gives out a single location as the output. The number of parameters of the model will be around 200. This will be saved in the text files too
'''

#======================================================================
# IMPORTS
#======================================================================

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn
import math
import copy
from einops import rearrange
import time
import csv

print('Importing is complete')

#======================================================================
# DATA IMPORTING
#======================================================================

train_data = torch.load('/mnt/qb/work/goswami/ranganathabr08/common_directory/train_data_6.pt')
test_data = torch.load('/mnt/qb/work/goswami/ranganathabr08/common_directory/test_data_6.pt')
long_data = torch.load('/mnt/qb/work/goswami/ranganathabr08/common_directory/long_data_24.pt')

# ===============================================================================
# ARCHITECTURE OF THE CLOSED LOOP PREDICTION MODEL
# ===============================================================================

class lstm_small(nn.Module):
    
    def __init__(self, input_dim,hidden_dim,n_layers, do):
        
        """Initialize the network architecture
        
        Args:
            input_dim: Number of time lags to look at for the current prediction/seq len
            hidden_dim: dimension of the LSTM output
            n_layers: Number of stacked LSTMs
            do: float, optional: dropout for regularization
        """
        super(lstm_small, self).__init__()
        self.ip_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = do
        
        self.rnn = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim,
                           num_layers = n_layers, dropout = do, batch_first = True)
        self.fc = nn.Linear(in_features = hidden_dim, out_features = 2, bias = False)

    def forward(self,input):
        out,_ = self.rnn(input)
        out = self.fc(out)       
        return out

    def predict(self,input):
        with torch.no_grad():
            predictions = self.forward(input)
        return predictions
        
#=======================================================================
# OTHER USEFUL FUNCTIONS
#=======================================================================
      
# Number of parameters of a given function

def num_params(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params+=param
    #print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

# The distance metric on the globe. It returns the distance between the sets of locations given. The input of the function is the normalized set of latitudes and longitudes

def haversine(x, y):
    
    first_rad = x
    second_rad = y
    
    a = (
        torch.sin((second_rad[:,:,0]-first_rad[:,:,0])*60*torch.pi/2/180)**2 
        + torch.cos((second_rad[:,:,0]*60 - 15)*torch.pi/180)
        *torch.cos((first_rad[:,:,0]*60 - 15)*torch.pi/180)
        *torch.sin((second_rad[:,:,1] -first_rad[:,:,1])*90*torch.pi/2/180)**2 
        ) + 1e-6

    distance = ((torch.atan2(torch.sqrt(a),torch.sqrt(1-a) + 1e-6)))
    mean_distance = distance.mean()

    return mean_distance
    
model1 = lstm_small(2,5,n_layers = 1, do = 0.1)
num_params(model1)

#=======================================================================
# META DATA STORAGE
#=======================================================================

with open('/mnt/qb/work/goswami/ranganathabr08/common_directory/Mar_results/final_lstm_mar_22.txt','w') as f:
	f.write('This file contains the meta data for the run of closed loop small prediction model.')
	f.write('\n')
	f.write('There are two models at hand.')
	f.write('\n')
	f.write('The first model is the lstm model. The details of which are as follows')
	f.write('\n')
	f.write(str(model1))
	f.write('\n')
	f.write('The number of parameters in this model are' +str(num_params(model1)))
	f.write('\n')
	f.write('The training data is from the year 1979 to 2011.')
	f.write('\n')
	f.write('The testing data is from the year 2011 to 2019')
	f.write('\n')
	f.write('The shape of the training dataset is'+str(train_data.shape))
	f.write('\n')
	f.write('The shape of the testing datalist is'+str(test_data.shape))
	f.write('\n')
	f.write('Batchsize is set to 32')
	f.write('\n')
	f.write('The shape of the long dataset is'+str(long_data.shape))
	f.write('\n')
f.close()

num_ensembles = 20

for ensemble_index in range(num_ensembles):

    '''
    Rearranging the data is very important since initial clusters are long and take up a lot of data. This can
    keep the datasets very similar leading to fitting problems.
    '''

    train_data = train_data[torch.randperm(train_data.size()[0])]
    test_data = test_data[torch.randperm(test_data.size()[0])]

    dl = DataLoader(train_data, batch_size = 32, shuffle = True, drop_last = True)
    vdl = DataLoader(test_data, batch_size = 32, shuffle = True, drop_last = True)
    tdl = DataLoader(long_data, batch_size = 32, shuffle = True, drop_last = True)

    #=======================================================================
    # Model details
    #=======================================================================

    model1 = lstm_small(2,5,n_layers = 1, do = 0.1)
    optim1 = torch.optim.AdamW(model1.parameters(), lr = 0.001)

    epochs = 100

    tl1, vl1 = [], []

    for e in range(epochs):

        ls1 = 0
        for sample in dl:
            x = sample[:,:-1]
            y = sample[:,-1:]
            prediction = model1(x)
            loss = haversine(prediction[:,-1:], y)
            ls1 += loss.item()
            loss.backward()
            optim1.step()
            optim1.zero_grad(set_to_none = True)
        tl1.append(ls1/len(dl))

        vls1 = 0
        for sample in vdl:
            x = sample[:,:-1]
            y = sample[:,-1:]
            prediction = model1.predict(x)
            loss = haversine(prediction[:,-1:],y)
            vls1 += loss.item()

        vl1.append(vls1/len(vdl))
        
    print('The training and testing of the data is complete')
    t = time.time()
    #=======================================================================
    # Saving models and the training errors
    #=======================================================================
    vl1_tensor = torch.Tensor(vl1).view(-1,1)
    tl1_tensor = torch.Tensor(tl1).view(-1,1)
    error1_tensor = torch.concat((tl1_tensor,vl1_tensor),axis = 1)
    torch.save(error1_tensor,'/mnt/qb/work/goswami/ranganathabr08/common_directory/Mar_results/lstm_errors_'+str(t)+'.pt')
    
    #=======================================================================
    # Closed loop predictions
    #=======================================================================
    ls_list = []
    for count,sample in enumerate(tdl):
    	x0 = sample[:,:5]
    	for i in range(24-6):
    	    x = sample[:,i:i+5]
    	    y = sample[:,i+5:i+6]
    	    prediction = model1.predict(x0)
    	    loss = haversine(prediction[:,-1:],y)
    	    ls_list.append(loss.item())
    	    x0 = torch.concat((x0[:,1:],prediction[:,-1:]),axis = 1)
    	tensor_long = torch.Tensor(ls_list).view(-1,18)	
    	torch.save(tensor_long,'/mnt/qb/work/goswami/ranganathabr08/common_directory/Mar_results/lstm_error_trend_'+str(t)+'.pt')	    

