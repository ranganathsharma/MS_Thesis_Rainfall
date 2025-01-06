# This program trains a model to predict a shifted sequence by one time step

#=======================================================================
# IMPORTS
#=======================================================================

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn
import math
import copy
from einops import rearrange

print('Importing is complete')

train_datalist = []
test_datalist = []
long_dataset = []
num_prev = 19
for year_index in range(1979,1979+32):
    dataframe = pd.read_csv('/mnt/qb/work/goswami/ranganathabr08/cluster_files/cluster_files/mix_clusters'+str(year_index)+'.csv',index_col = False)
    dataframe = dataframe.drop('Unnamed: 0', axis = 1)
    dataframe = dataframe.drop('Unnamed: 0.1', axis = 1)
    
    temp_index_list = set(dataframe['per_index'].to_list())
    len_index_list = len(temp_index_list)
    
    for index in temp_index_list:
        temp_dataframe = dataframe[dataframe['per_index'] == index]
        len_cluster = len(temp_dataframe)
        
        if len_cluster >= 30:
            long_dataset.append([temp_dataframe.iloc[:30,3].to_list(),
                                temp_dataframe.iloc[:30,4].to_list()])
            
        if len_cluster >= num_prev:
            for i in range(len_cluster-num_prev):
                train_datalist.append([temp_dataframe.iloc[i:i+num_prev,3].to_list(),
                                       temp_dataframe.iloc[i:i+num_prev,4].to_list()])

for year_index in range(2011,2020):
    dataframe = pd.read_csv('/mnt/qb/work/goswami/ranganathabr08/cluster_files/cluster_files/mix_clusters'+str(year_index)+'.csv',index_col = False)
    dataframe = dataframe.drop('Unnamed: 0',axis = 1)
    dataframe = dataframe.drop('Unnamed: 0.1',axis = 1)
    
    temp_index_list = set(dataframe['per_index'].to_list())
    len_index_list = len(temp_index_list)
    
    for index in temp_index_list:
        temp_dataframe = dataframe[dataframe['per_index']== index]
        len_cluster = len(temp_dataframe)
        
        if len_cluster >= 30:
            long_dataset.append([temp_dataframe.iloc[:30,3].to_list(),
                                temp_dataframe.iloc[:30,4].to_list()])
        if len_cluster >= num_prev:
            for i in range(len_cluster-num_prev):
                test_datalist.append([temp_dataframe.iloc[i:i+num_prev,3].to_list(), 
                                      temp_dataframe.iloc[i:i+num_prev,4].to_list()])
    
final_data_close = torch.transpose(torch.from_numpy(np.array(long_dataset)).float(),0,2)

final_data_close = copy.deepcopy(rearrange(final_data_close, 'steps coords samples -> samples steps coords'))
final_data_close[:,:,0] = (final_data_close[:,:,0] + 15)/60
final_data_close[:,:,1] = (final_data_close[:,:,1] - 30)/90

#=======================================================================
# ARCHITECTURE OF THE SMALL LSTM MODEL
#=======================================================================

class np_lstm_small(nn.Module):
    
    def __init__(self, input_dim,hidden_dim,n_layers, do):
        
        """Initialize the network architecture
        
        Args:
            input_dim: Number of time lags to look at for the current prediction/seq len
            hidden_dim: dimension of the LSTM output
            n_layers: Number of stacked LSTMs
            do: float, optional: dropout for regularization
        """
        super(np_lstm_small, self).__init__()
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
# ARCHITECTURE OF THE BIG LSTM MODEL
#=======================================================================

    
class np_lstm_big(nn.Module):
    
    def __init__(self, input_dim,hidden_dim,n_layers, do):
        
        """Initialize the network architecture
        
        Args:
            input_dim: Number of time lags to look at for the current prediction/seq len
            hidden_dim: dimension of the LSTM output
            n_layers: Number of stacked LSTMs
            do: float, optional: dropout for regularization
        """
        super(np_lstm_big, self).__init__()
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
# ARCHITECTURE OF THE BIG FULLY CONNECTED DEEP NEURAL NETWORK MODEL
#=======================================================================
       
class FC_big_model(nn.Module):
    
    def __init__(self, input_dim):
        
        """Initialize the network architecture
        
        Args:
            input_dim: Number of time lags to look at for the current prediction/seq len
        """
        super(FC_big_model,self).__init__()
        self.fc1 = nn.Linear(in_features = input_dim, out_features = 200, bias = True)
        self.relu1 = nn.GELU()
        self.fc2 = nn.Linear(in_features = 200, out_features = 100, bias = True)
        self.relu2 = nn.GELU()
        self.fc3 = nn.Linear(in_features = 100, out_features = 36, bias = True)

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
# ARCHITECTURE OF THE SMALL FULLY CONNECTED DEEP NEURAL NETWORK MODEL
#=======================================================================
 
class FC_small_model(nn.Module):
    
    def __init__(self, input_dim):
        
        """Initialize the network architecture
        
        Args:
            input_dim: Number of time lags to look at for the current prediction/seq len
        """
        super(FC_small_model,self).__init__()
        self.fc1 = nn.Linear(in_features = input_dim, out_features = 36, bias = False)
        self.relu1 = nn.ReLU()
        
    def forward(self,input):
        out = self.fc1(input)  
        out = self.relu1(out)
        
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
    print(f"Total Trainable Params: {total_params}")
    return total_params

# The distance metric on the globe. It returns the distance between the sets of locations given. The input of the function is the normalized set of latitudes and longitudes

def haversine(x, y):
    
    first_rad = x
    second_rad = y
    
    a = (
        torch.sin((second_rad[:,0]-first_rad[:,0])*60*torch.pi/2/180)**2 
        + torch.cos((second_rad[:,0]*60 - 15)*torch.pi/180)
        *torch.cos((first_rad[:,0]*60 - 15)*torch.pi/180)
        *torch.sin((second_rad[:,1] -first_rad[:,1])*90*torch.pi/2/180)**2 
        ) + 1e-6

    distance = ((torch.atan2(torch.sqrt(a),torch.sqrt(1-a) + 1e-6)))
    mean_distance = distance.mean()

    return mean_distance

def haversine_seq(pred,x, y):
    
    first_rad = pred
    second_rad = torch.concat((x[:,1:],y), axis = 1)

    a = (
        torch.sin((second_rad[:,0]-first_rad[:,0])*60*torch.pi/2/180)**2 
        + torch.cos((second_rad[:,0]*60 - 15)*torch.pi/180)
        *torch.cos((first_rad[:,0]*60 - 15)*torch.pi/180)
        *torch.sin((second_rad[:,1] -first_rad[:,1])*90*torch.pi/2/180)**2 
        ) + 1e-6

    distance = ((torch.atan2(torch.sqrt(a),torch.sqrt(1-a) + 1e-6)))
    mean_distance = distance.mean()

    return mean_distance

#=======================================================================
# DATA NORMALIZATION
#=======================================================================

# The data is first detrended by keeping the value higher than 0 and further using the max-min normalization, the final values are kept between 0 and 1


train_data = torch.transpose(torch.from_numpy(np.array(train_datalist)).float(),0,1)
train_data = rearrange(train_data, 'coords samples steps -> samples steps coords')

train_data[:,:,0] = (train_data[:,:,0] + 15)/60
train_data[:,:,1] = (train_data[:,:,1] - 30)/90

test_data = torch.transpose(torch.from_numpy(np.array(test_datalist)).float(),0,1)
test_data = rearrange(test_data, 'coords samples steps -> samples steps coords')

test_data[:,:,0] = (test_data[:,:,0] + 15)/60
test_data[:,:,1] = (test_data[:,:,1] - 30)/90


#=======================================================================
# MAIN LOOP
#=======================================================================

num_ensembles = 20

for ensemble_index in range(num_ensembles):

    '''
    Rearranging the data is very important since initial clusters are long and take up a lot of data. This can
    keep the datasets very similar leading to fitting problems.
    '''

    train_data = train_data[torch.randperm(train_data.size()[0])]
    test_data = test_data[torch.randperm(test_data.size()[0])]

    dl = DataLoader(train_data, batch_size = 30, shuffle = True)
    vdl = DataLoader(test_data, batch_size = 30, shuffle = True)

    #=======================================================================
    # Model details
    #=======================================================================

    # model1 = np_lstm_big(2,20,n_layers = 2, do = 0.1)
    model1 = np_lstm_small(2,16,n_layers = 1, do = 0.1)
    # model1 = FC_big_model(36)
    model2 = FC_small_model(36)
    optim1 = torch.optim.AdamW(model1.parameters(), lr = 0.001)
    optim2 = torch.optim.AdamW(model2.parameters(), lr = 0.001)

    epochs = 200

    tl1, vl1, tl2, vl2 = [], [], [], []

    for e in range(epochs):

        ls1 = 0
        for sample in dl:
            x = sample[:,:-1]
            y = sample[:,-1:]

            prediction = model1(x)

            loss = haversine_seq(prediction,x, y)
            ls1 += loss.item()
            loss.backward()
            optim1.step()
            optim1.zero_grad(set_to_none = True)
            #print(ls)
        tl1.append(ls1/len(dl))

        vls1 = 0
        for sample in vdl:
            x = sample[:,:-1]
            y = sample[:,-1:]
            prediction = model1.predict(x)
            loss = haversine_seq(prediction,x,y)
            #loss = torch.sqrt(loss_fn(prediction[:,-1], y))
            vls1 += loss.item()

        vl1.append(vls1/len(vdl))


        ls2 = 0

        for sample in dl:
            x = sample[:,:-1]
            x_new = torch.concat((x[:,:,0],x[:,:,1]), axis =1)
            y = sample[:,-1:]

            prediction = model2(x_new)
            prediction_unflat = torch.concat((prediction[:,:18].view(-1,18,1),prediction[:,18:].view(-1,18,1)),axis = 2)
            loss = haversine_seq(prediction_unflat,x,y)
            ls2 += loss.item()
            loss.backward()
            optim2.step()
            optim2.zero_grad(set_to_none = True)
        tl2.append(ls2/len(dl))

        vls2 = 0
        for sample in vdl:
            x = sample[:,:-1]
            x_new = torch.concat((x[:,:,0],x[:,:,1]), axis =1)
            y = sample[:,-1:]

            prediction = model2.predict(x_new)
            prediction_unflat = torch.concat((prediction[:,:18].view(-1,18,1),prediction[:,18:].view(-1,18,1)),axis = 2)
            loss = haversine_seq(prediction_unflat,x,y)
            vls2 += loss.item()

        vl2.append(vls2/len(vdl))

    print('The training and testing of the data is complete')

    #=======================================================================
    # Saving models and the training errors
    #=======================================================================

    vl1_tensor = torch.Tensor(vl1).view(-1,1)
    tl1_tensor = torch.Tensor(tl1).view(-1,1)
    error1_tensor = torch.concat((tl1_tensor,vl1_tensor),axis = 1)
    torch.save(error1_tensor,'/mnt/qb/work/goswami/ranganathabr08/common_directory/Feb_results/many_points_errors_lstm_small_model_'+str(ensemble_index)+'.pt')
    torch.save(model1, '/mnt/qb/work/goswami/ranganathabr08/common_directory/Feb_results/many_points_lstm_small_model_'+str(ensemble_index)+'.pt')

    vl2_tensor = torch.Tensor(vl2).view(-1,1)
    tl2_tensor = torch.Tensor(tl2).view(-1,1)
    error2_tensor = torch.concat((tl2_tensor,vl2_tensor),axis = 1)
    torch.save(error2_tensor,'/mnt/qb/work/goswami/ranganathabr08/common_directory/Feb_results/many_points_errors_fc_small_model_'+str(ensemble_index)+'.pt')
    torch.save(model2, '/mnt/qb/work/goswami/ranganathabr08/common_directory/Feb_results/many_points_fc_small_model_'+str(ensemble_index)+'.pt')


    #=======================================================================
    # Closed loop predictions
    #=======================================================================

    x0 = final_data_close[:,:18]
    ls = []
    ls_close = []

    for i in range(30-18):
        x = final_data_close[:,i:i+18]
        y = final_data_close[:,i+18:i+19]
        prediction_close = model1.predict(x0)
        prediction = model1.predict(x)
        x0 = torch.concat((x0[:,1:], prediction[:,-1:]), axis = 1)
        loss_close = haversine_seq(prediction_close,x, y)
        loss = haversine_seq(prediction,x,y)
        ls_close.append(loss_close.item())
        ls.append(loss.item())

    ls_tensor = torch.Tensor(ls).view(-1,1)
    ls_close_tensor = torch.Tensor(ls_close).view(-1,1)

    ls_full_tensor = torch.concat((ls_tensor,ls_close_tensor),axis = 1)
    torch.save(ls_full_tensor, '/mnt/qb/work/goswami/ranganathabr08/common_directory/Feb_results/many_points_small_close_open_error_lstm'+str(ensemble_index)+'.pt')

    x0 = final_data_close[:,:18]
    ls = []
    ls_close = []

    for i in range(30-18):
        
        x = final_data_close[:,i:i+18]
        x_flat = torch.concat((x[:,:,0], x[:,:,1]), axis = 1)
        y = final_data_close[:,i+18:i+19]
        x0_flat = torch.concat((x0[:,:,0], x0[:,:,1]), axis = 1)
        prediction_close = model2.predict(x0_flat)
        prediction = model2.predict(x_flat)
        prediction_close_unflat = torch.concat((prediction_close[:,:18].view(-1,18,1), prediction_close[:,18:].view(-1,18,1)), axis = 2)
        prediction_unflat = torch.concat((prediction[:,:18].view(-1,18,1), prediction[:,18:].view(-1,18,1)), axis = 2)
        x0 = torch.concat((x0[:,:1,:], prediction_unflat[:,1:]), axis = 1)
        loss_close = haversine_seq(prediction_close_unflat, x, y)
        loss = haversine_seq(prediction_unflat,x,y)
        ls_close.append(loss_close.item())
        ls.append(loss.item())

    ls_tensor = torch.Tensor(ls).view(-1,1)
    ls_close_tensor = torch.Tensor(ls_close).view(-1,1)
    ls_full_tensor = torch.concat((ls_tensor, ls_close_tensor),axis = 1)
    torch.save(ls_full_tensor, '/mnt/qb/work/goswami/ranganathabr08/common_directory/Feb_results/many_points_small_close_open_error_fc'+str(ensemble_index)+'.pt')

print('The code processing is complete')

