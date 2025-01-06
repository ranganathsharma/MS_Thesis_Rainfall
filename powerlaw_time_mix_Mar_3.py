# Imports

# Imports
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import torch
import csv
import powerlaw

#file_address = '/home/ranganath/Bedartha/common_directory/nintynine_percent/cluster_files/mix_clusters1979.csv'

collist = ['time','per_index']
time_list = []

for year_index in range(1979,2020):
	file_address = '/mnt/qb/work/goswami/ranganathabr08/cluster_files/cluster_files/mix_clusters'+str(year_index)+'.csv'
	data = pd.read_csv(file_address, index_col = False, usecols = collist)
	data.head()

	index_list = set(data['per_index'].to_list())
	len(index_list)

	t = time.time()
	time_list = []
	for count, index in enumerate(index_list):
		temp = data[data['per_index'] == index]
		start = temp.iloc[0,0]
		end = temp.iloc[-1,0]
		if start != end:
		    time_list.append((end-start)/3600)
		if count%5000 == 0:
		    print(time.time()-t)
		    
	time_list.remove(max(time_list))
	
	if year_index == 1979:
		form = 'w'
	else:
		form = 'a'
		
	with open('/mnt/qb/work/goswami/ranganathabr08/common_directory/Mar_results/power_law_all.txt',form) as f:
		f.write('This file contains the meta data for the run of closed loop big prediction model.')
		f.write('\n')	
		f.write('The number of indices in the year '+str(year_index)+' is '+str(len(index_list)))
		f.write('\n')  
	
fit = powerlaw.Fit(time_list,xmin = 1)
R,p = fit.distribution_compare('power_law', 'truncated_power_law', normalized_ratio = True)
with open('/mnt/qb/work/goswami/ranganathabr08/common_directory/Mar_results/powerlaw_time_mix.txt',form) as f:
	f.write('The alpha value for the powerlaw fit is ' + str(fit.alpha))
	f.write('\n')
	f.write('The D value for the powerlaw fit is '+str(fit.D))
	f.write('\n')
	f.write('The alpha value for the truncated powerlaw is '+str(fit.truncated_power_law.alpha))
	f.write('\n')
	f.write('The lambda value for the truncated powerlaw is ' +str(fit.truncated_power_law.Lambda))
	f.write('\n')
	f.write('The D value for the truncated powerlaw fit is '+str(fit.truncated_power_law.D))
	f.write('\n')
	f.write('The R value for the comparison between powerlaw and truncated powerlaw is '+str(R))
	f.write('\n')
	f.write('The p value for the comparison between powerlaw and truncated powerlaw is '+str(p))
	f.write('\n')
	f.write('The minimum value of x in the powerlaw case is '+str(fit.xmin))
	f.write('\n')
	f.write('The minimum value of x in the truncated powerlaw case is '+str(fit.truncated_power_law.xmin))
	f.write('\n')

powerlaw.plot_pdf(time_list)
plt.xlabel('duration')
plt.ylabel('p(d)')
plt.savefig('/mnt/qb/work/goswami/ranganathabr08/common_directory/Mar_results/powerlaw_time_mix.png')



  
