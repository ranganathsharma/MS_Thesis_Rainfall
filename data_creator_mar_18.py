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

train_datalist = []
test_datalist = []
long_dataset = []
long_indices = []
num_prev = 19
collist = ['time', 'vol_sum', 'lat_mean', 'lon_mean','per_index']
for year_index in range(1979,1979+32):
    
    dataframe = pd.read_csv('/mnt/qb/work/goswami/ranganathabr08/cluster_files/cluster_files/mix_clusters'+str(year_index)+'.csv',index_col = False, usecols = collist)
    temp_index_list = set(dataframe['per_index'].to_list())
    len_index_list = len(temp_index_list)
    
    for index in temp_index_list:
        temp_dataframe = dataframe[dataframe['per_index'] == index]
        len_cluster = len(temp_dataframe)
        
        if len_cluster >= 30:
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
        
        if len_cluster >= 30:
            long_indices.append('per_index')
            long_dataset.append([temp_dataframe.iloc[:24,2].to_list(),
                                temp_dataframe.iloc[:24,3].to_list()])
            
        if len_cluster >= num_prev:
            for i in range(len_cluster-num_prev):
                test_datalist.append([temp_dataframe.iloc[i:i+num_prev,2].to_list(),
                                       temp_dataframe.iloc[i:i+num_prev,3].to_list()])  
                                       

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

torch.save(train_data,'/mnt/qb/work/goswami/ranganathabr08/common_directory/train_data_18.pt')
torch.save(test_data,'/mnt/qb/work/goswami/ranganathabr08/common_directory/test_data_18.pt')
torch.save(long_data,'/mnt/qb/work/goswami/ranganathabr08/common_directory/long_data_30.pt')
