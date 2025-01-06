# This file contains the code for training a closed loop prediction model where the input prediction pattern is (6,12)
# The code also has the provision for storing the meta data of the models and the underlying data in a separate file
# The indices of long storm cases will be saved in a separate file

print('The current run contains the results of a big model for closed loop predictions')

# ===============================================================================
# IMPORTS
# ===============================================================================

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

# ===============================================================================
# DATA IMPORTING AND PREPROCESSING
# ===============================================================================


train_datalist = []
test_datalist = []
long_dataset = []
long_indices = []
num_prev = 12
collist = ['time', 'vol_sum', 'lat_mean', 'lon_mean','per_index']
for year_index in range(1979,1979+32):
    
    dataframe = pd.read_csv('/mnt/qb/work/goswami/ranganathabr08/cluster_files/cluster_files/mix_clusters'+str(year_index)+'.csv',index_col = False, usecols = collist)
    temp_index_list = set(dataframe['per_index'].to_list())
    len_index_list = len(temp_index_list)
    
    for index in temp_index_list:
        temp_dataframe = dataframe[dataframe['per_index'] == index]
        len_cluster = len(temp_dataframe)
        
        if len_cluster >= 24:
            long_indices.append('per_index')
            long_dataset.append([temp_dataframe.iloc[:24,2].to_list(),
                                temp_dataframe.iloc[:24,3].to_list()])
            
        if len_cluster >= num_prev:
            for i in range(len_cluster-num_prev):
                train_datalist.append([temp_dataframe.iloc[i:i+num_prev,2].to_list(),
                                       temp_dataframe.iloc[i:i+num_prev,3].to_list()])
                                       
for year_index in range(2011,2020):
    dataframe = pd.read_csv('/mnt/qb/work/goswami/ranganathabr08/cluster_files/cluster_files/mix_clusters'+str(year_index)+'.csv',index_col = False, usecols = collist)
    temp_index_list = set(dataframe['per_index'].to_list())
    len_index_list = len(temp_index_list)
    
    for index in temp_index_list:
        temp_dataframe = dataframe[dataframe['per_index'] == index]
        len_cluster = len(temp_dataframe)
        
        if len_cluster >= 24:
            long_indices.append('per_index')
            long_dataset.append([temp_dataframe.iloc[:24,2].to_list(),
                                temp_dataframe.iloc[:24,3].to_list()])
            
        if len_cluster >= num_prev:
            for i in range(len_cluster-num_prev):
                test_datalist.append([temp_dataframe.iloc[i:i+num_prev,2].to_list(),
                                       temp_dataframe.iloc[i:i+num_prev,3].to_list()])  
                                       
                                       
# ===============================================================================
# DATA NORMATLIZATION
# ===============================================================================

# The data is first detrended by keeping the value higher than 0 and further using the max-min normalization, the final values are kept between 0 and 1


train_data = torch.transpose(torch.from_numpy(np.array(train_datalist)).float(),0,1)
train_data = rearrange(train_data, 'coords samples steps -> samples steps coords')

train_data[:,:,0] = (train_data[:,:,0] + 15)/60
train_data[:,:,1] = (train_data[:,:,1] - 30)/90

test_data = torch.transpose(torch.from_numpy(np.array(test_datalist)).float(),0,1)
test_data = rearrange(test_data, 'coords samples steps -> samples steps coords')

test_data[:,:,0] = (test_data[:,:,0] + 15)/60
test_data[:,:,1] = (test_data[:,:,1] - 30)/90    

long_data = torch.transpose(torch.from_numpy(np.array(long_dataset)).float(),0,1)
long_data = rearrange(long_data, 'coords samples steps -> samples steps coords')

long_data[:,:,0] = (long_data[:,:,0] + 15)/60
long_data[:,:,1] = (long_data[:,:,1] - 30)/90

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
# ARCHITECTURE OF THE SMALL FULLY CONNECTED DEEP NEURAL NETWORK MODEL
#=======================================================================
 
class FC_big_model(nn.Module):
    
    def __init__(self, input_dim):
        
        """Initialize the network architecture
        
        Args:
            input_dim: Number of time lags to look at for the current prediction/seq len
        """
        super(FC_big_model,self).__init__()
        self.fc1 = nn.Linear(in_features = input_dim, out_features = 400, bias = True)
        self.relu1 = nn.GELU()
        self.fc2 = nn.Linear(in_features = 400, out_features = 206, bias = True)
        self.relu2 = nn.GELU()
        self.fc3 = nn.Linear(in_features = 206, out_features = 2, bias = True)

    def forward(self,input):
        out = self.fc1(input)  
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        
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

model1 = lstm_small(2,64,n_layers = 3, do = 0.1)
num_params(model1)
model2 = FC_big_model(2)
num_params(model2)   

#=======================================================================
# META DATA STORAGE
#=======================================================================

with open('/mnt/qb/work/goswami/ranganathabr08/common_directory/Mar_results/big_closed/meta/meta_data_big_lstm_fc.txt','w') as f:
	f.write('This file contains the meta data for the run of closed loop big prediction model.')
	f.write('\n')
	f.write('There are two models at hand.')
	f.write('\n')
	f.write('The first model is the lstm model. The details of which are as follows')
	f.write('\n')
	f.write(str(model1))
	f.write('\n')
	f.write('The number of parameters in this model are' +str(num_params(model1)))
	f.write('\n')
	f.write('The second model is the fc model. The details of which are as follows')
	f.write('\n')
	f.write(str(model2))
	f.write('The number of parameters in this model is'+ str(num_params(model2)))
	f.write('\n')
	f.write('The training data is from the year 1979 to 2011.')
	f.write('\n')
	f.write('The testing data is from the year 2011 to 2019')
	f.write('\n')
	f.write('The number of storms in training data is'+str(len(train_datalist)))
	f.write('\n')
	f.write('The number of storms in testing data is'+str(len(test_datalist)))
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

#=======================================================================
# MAIN LOOP
#=======================================================================

num_ensembles = 10

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

    model1 = lstm_small(2,64,n_layers = 3, do = 0.1)
    model2 = FC_big_model(2)
    optim1 = torch.optim.AdamW(model1.parameters(), lr = 0.001)
    optim2 = torch.optim.AdamW(model2.parameters(), lr = 0.001)

    epochs = 100

    tl1, vl1, tl2, vl2 = [], [], [], []

    for e in range(epochs):
        print(e)
        ls1 = 0
        for sample in dl:
            prediction = torch.zeros(32,11,2)
            y = sample[:,1:]
            for t in range(11):
                if t < 6:
                    x = sample[:,t].view(-1,1,2)
                else:
                    x = prediction[:,t-1].view(-1,1,2)
                prediction[:,t:t+1,:] = model1(x)
            loss = haversine(prediction,y)
            ls1 += loss.item()
            optim1.zero_grad(set_to_none = True)
            loss.backward()
            optim1.step()
        tl1.append(ls1/len(dl))
        
        vls1 = 0
        for sample in vdl:
            prediction = torch.zeros(32,11,2)
            y = sample[:,1:]
            for t in range(11):
                if t < 6:
                    x = sample[:,t].view(-1,1,2)
                else:
                    x = prediction[:,t-1].view(-1,1,2)
                prediction[:,t:t+1,:] = model1(x)
            loss = haversine(prediction,y)
            vls1 += loss.item()
            
        vl1.append(vls1/len(vdl))
        
        ls2 = 0
        for sample in dl:
            prediction = torch.zeros(32,11,2)
            y = sample[:,1:]
            for t in range(11):
                if t < 6:
                    x = sample[:,t].view(-1,1,2)
                else:
                    x = prediction[:,t-1].view(-1,1,2)
                prediction[:,t:t+1,:] = model2(x)
            loss = haversine(prediction,y)
            ls2 += loss.item()
            optim2.zero_grad(set_to_none = True)
            loss.backward()
            optim2.step()
        tl2.append(ls2/len(dl))
        
        vls2 = 0
        for sample in vdl:
            prediction = torch.zeros(32,11,2)
            y = sample[:,1:]
            for t in range(11):
                if t < 6:
                    x = sample[:,t].view(-1,1,2)
                else:
                    x = prediction[:,t-1].view(-1,1,2)
                prediction[:,t:t+1,:] = model2(x)
            loss = haversine(prediction,y)
            vls2 += loss.item()
            
        vl2.append(vls2/len(vdl))
        
    print('The training and testing of the data is complete')
    
    #=======================================================================
    # Closed loop predictions
    #=======================================================================
    ls_tensor1 = torch.zeros(len(tdl),23)
    ls1 = 0
    for count, sample in enumerate(tdl):
    	prediction = torch.zeros(32,23,2)
    	y = sample[:,1:]
    	for t in range(23):
    		if t<6:
    			x = sample[:,t].view(-1,1,2)
    		else:
    			x = prediction[:,t-1].view(-1,1,2)
    		prediction[:,t:t+1,:] = model1.predict(x)
    	loss = haversine(prediction,y)
    	ls1+= loss.item()
    	for i in range(23):
    		ls_tensor1[count,i] = haversine(prediction[:,i].view(-1,1,2), y[:,i].view(-1,1,2))
    		
    ls_tensor2 = torch.zeros(len(tdl),23)
    ls1 = 0
    for count, sample in enumerate(tdl):
    	prediction = torch.zeros(32,23,2)
    	y = sample[:,1:]
    	for t in range(23):
    		if t<6:
    			x = sample[:,t].view(-1,1,2)
    		else:
    			x = prediction[:,t-1].view(-1,1,2)
    		prediction[:,t:t+1,:] = model2.predict(x)
    	loss = haversine(prediction,y)
    	ls1+= loss.item()
    	for i in range(23):
    		ls_tensor2[count,i] = haversine(prediction[:,i].view(-1,1,2), y[:,i].view(-1,1,2))

    		

    #=======================================================================
    # Saving models and the training errors
    #=======================================================================
    t = time.time()
    vl1_tensor = torch.Tensor(vl1).view(-1,1)
    tl1_tensor = torch.Tensor(tl1).view(-1,1)
    error1_tensor = torch.concat((tl1_tensor,vl1_tensor),axis = 1)
    torch.save(error1_tensor,'/mnt/qb/work/goswami/ranganathabr08/common_directory/Mar_results/big_closed/errors/lstm_'+str(ensemble_index)+str(t)+'.pt')
    torch.save(model1, '/mnt/qb/work/goswami/ranganathabr08/common_directory/Mar_results/big_closed/models/lstm'+str(ensemble_index)+str(t)+'.pt')

    vl2_tensor = torch.Tensor(vl2).view(-1,1)
    tl2_tensor = torch.Tensor(tl2).view(-1,1)
    error2_tensor = torch.concat((tl2_tensor,vl2_tensor),axis = 1)
    torch.save(error2_tensor,'/mnt/qb/work/goswami/ranganathabr08/common_directory/Mar_results/big_closed/errors/fc_'+str(ensemble_index)+str(t)+'.pt')
    torch.save(model2, '/mnt/qb/work/goswami/ranganathabr08/common_directory/Mar_results/big_closed/models/fc_'+str(ensemble_index)+str(t)+'.pt')
    trend1 = torch.mean(ls_tensor1, axis = 0)
    torch.save(trend1, '/mnt/qb/work/goswami/ranganathabr08/common_directory/Mar_results/big_closed/error_trend/lstm_'+str(ensemble_index)+str(t)+'.pt')
    trend2 = torch.mean(ls_tensor2, axis = 0)
    torch.save(trend2, '/mnt/qb/work/goswami/ranganathabr08/common_directory/Mar_results/big_closed/error_trend/fc_'+str(ensemble_index)+str(t)+'.pt')

