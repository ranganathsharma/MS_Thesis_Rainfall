# This program contains the detials of statistical properties of land, mix and all clusters in a complete scale

#===============================================================
# IMPORTS
#===============================================================

import numpy as np
import powerlaw
import pandas as pd
import matplotlib.pyplot as plt
import deepgraph as dg
import copy
import time
import torch

#===============================================================
# DATA IMPORTING
#===============================================================
t = time.time()
vol_list = []
col_list = ['vol_sum', 'per_index']

for year_index in range(1979,2020):

    data = pd.read_csv('/mnt/qb/work/goswami/ranganathabr08/cluster_files/cluster_files/mix_clusters'+str(year_index)+'.csv',index_col = False, usecols = col_list)
    g = dg.DeepGraph(data)
    index_list = set(data['per_index'].to_list())
    feature_funcs = {'vol_sum': [sum]}
    for count, index in enumerate(index_list):
        gt_perm = copy.deepcopy(g)
        gt_perm.filter_by_values_v('per_index',index)
        st,stv = gt_perm.partition_nodes('per_index',feature_funcs,return_gv = True)
        vol_list.append(st.iloc[0,1])
 
#===============================================================================
# DATA PLOTTING
#===============================================================================
        
x,bins,p = plt.hist(vol_list,bins = np.logspace(np.log10(min(vol_list)),np.log10(max(vol_list)),30),density = True,histtype = 'step',label = 'data')
plt.xscale('log')
plt.yscale('log')

data_fit1 = powerlaw.Fit(fit_method = 'Likelihood',data = vol_list)
print(data_fit1.alpha,data_fit1.xmin)
theor_dist = powerlaw.Power_Law(xmin = data_fit1.xmin,parameters = [data_fit1.alpha])
simulated_data = theor_dist.generate_random(10**5)
x1,bins1,p1 = plt.hist(simulated_data,bins = np.logspace(np.log10(min(simulated_data)),np.log10(max(vol_list)),20),density = True,histtype = 'step',label = 'powerlaw')
plt.xscale('log')
plt.yscale('log')
       
plt.legend()
plt.xlabel('cluster size (S)')
plt.ylabel('p(S)')
plt.show()       
plt.savefig('/mnt/qb/work/goswami/ranganathabr08/common_directory/Feb_results/mix_powerlaw_feb_14.jpg')              
        
file1 = open("/mnt/qb/work/goswami/ranganathabr08/common_directory/Feb_results/alphas.txt","a")
file1.write('mix - alpha:'+str(data_fit1.alpha)+'\n')
file1.close() 

vol_tensor = torch.Tensor(vol_list)
torch.save(vol_tensor, "/mnt/qb/work/goswami/ranganathabr08/common_directory/Feb_results/mix_powerlaw_vol_list.pt")
 
