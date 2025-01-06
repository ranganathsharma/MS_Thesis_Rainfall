# This program is to form the distributions of duration, size and distance travelled by the storms for a particular year_index

# IMPORTS

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import time
import copy
import os
import torch

def haversine(x, y):
    
    first_rad = x
    second_rad = y
    
    a = (
        torch.sin((second_rad[:,0]-first_rad[:,0])*1*torch.pi/2/180)**2 
        + torch.cos((second_rad[:,0]*1 - 0)*torch.pi/180)
        *torch.cos((first_rad[:,0]*1 - 0)*torch.pi/180)
        *torch.sin((second_rad[:,1] -first_rad[:,1])*1*torch.pi/2/180)**2 
        ) + 1e-6

    distance = ((torch.atan2(torch.sqrt(a),torch.sqrt(1-a) + 1e-6)))
    mean_distance = distance.sum()

    return mean_distance 
    
time_list = []
size_list = []
dist_list = []
time_list_land = []
size_list_land = []
dist_list_land = []
time_list_mix = []
size_list_mix = []
dist_list_mix = []  

year_index = 1979

file_address = '/mnt/qb/work/goswami/ranganathabr08/cluster_files/cluster_files/mix_clusters'+str(year_index)+'.csv'
collist = ['time', 'vol_sum','lat_mean','lon_mean','per_index','land']
data = pd.read_csv(file_address, usecols = collist, index_col = False)
index_list = set(data['per_index'].to_list())
t = time.time()
print(data.head())
for count, index in enumerate(index_list):
    temp = data[data['per_index'] == index]
    start = temp.iloc[0,0]
    end = temp.iloc[-1,0]
    if start != end:
        time_list.append((end-start)/3600)
    
        volume = np.sum(temp.iloc[:,1])
        size_list.append(volume)
        
        temp_tensor = torch.zeros(len(temp),2)
        temp_tensor[:,0] = torch.tensor(temp.iloc[:,2].to_list())
        temp_tensor[:,1] = torch.tensor(temp.iloc[:,3].to_list())
        distance = haversine(temp_tensor[:-1],temp_tensor[1:])
        dist_list.append(distance.item())
        
        if np.sum(temp.iloc[:,5])/len(temp) >= 0.8:
            time_list_land.append((end-start)/3600)
            size_list_land.append(volume)
            dist_list_land.append(distance.item())
        else:
            time_list_mix.append((end-start)/3600)
            size_list_mix.append(volume)
            dist_list_mix.append(distance.item())

temp_tensor_list = torch.Tensor(time_list)
torch.save(temp_tensor_list, '/mnt/qb/work/goswami/ranganathabr08/common_directory/Mar_results/powerlaw_all_time.pt')
temp_tensor_list = torch.Tensor(size_list)
torch.save(temp_tensor_list, '/mnt/qb/work/goswami/ranganathabr08/common_directory/Mar_results/powerlaw_all_size.pt')
temp_tensor_list = torch.Tensor(dist_list)
torch.save(temp_tensor_list, '/mnt/qb/work/goswami/ranganathabr08/common_directory/Mar_results/powerlaw_all_dist.pt')
temp_tensor_list = torch.Tensor(time_list_land)
torch.save(temp_tensor_list, '/mnt/qb/work/goswami/ranganathabr08/common_directory/Mar_results/powerlaw_land_time.pt')
temp_tensor_list = torch.Tensor(size_list_land)
torch.save(temp_tensor_list, '/mnt/qb/work/goswami/ranganathabr08/common_directory/Mar_results/powerlaw_land_size.pt')
temp_tensor_list = torch.Tensor(dist_list_land)
torch.save(temp_tensor_list, '/mnt/qb/work/goswami/ranganathabr08/common_directory/Mar_results/powerlaw_land_dist.pt')
temp_tensor_list = torch.Tensor(time_list_mix)
torch.save(temp_tensor_list, '/mnt/qb/work/goswami/ranganathabr08/common_directory/Mar_results/powerlaw_mix_time.pt')
temp_tensor_list = torch.Tensor(size_list_mix)
torch.save(temp_tensor_list, '/mnt/qb/work/goswami/ranganathabr08/common_directory/Mar_results/powerlaw_mix_size.pt')
temp_tensor_list = torch.Tensor(dist_list_mix)
torch.save(temp_tensor_list, '/mnt/qb/work/goswami/ranganathabr08/common_directory/Mar_results/powerlaw_mix_dist.pt')





   
