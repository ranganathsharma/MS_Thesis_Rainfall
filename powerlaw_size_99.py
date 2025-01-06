# This file is to convert the processed data into the probability distribution functions for time and size distributions of the clusters using the powerlaw package. The data is saved in the common_directory/sizedist_files and timedist_files directories. The images are saved in sizedist_images and timedist_images directories respectively. 

#======================================================================
# imports
#======================================================================

import powerlaw 
import matplotlib.pyplot as plt
from numpy import *
import copy
import pandas as pd

#======================================================================

print('The imports are done')

# data loading for the non recursive case. Comment it out once all the files are procured and ready for a iterative way of analysis

year_index = 1979

# datafile = open('/home/goswami/ranganathabr08/common_directory/nintyfive_percent/sizedist_files/sizedist_'+str(year_index),'r')

datafile = pd.read_csv('/home/ranganath/Bedartha/common_directory/nintynine_percent/sizedist_files/size_list'+str(year_index)+'.csv')
sizelist1 = datafile.columns.to_list()
sizelist2 = []
for i in range(len(sizelist1)):
	sizelist2.append(float(sizelist1[i]))
	
x,bins,p = plt.hist(sizelist2,bins = logspace(log10(min(sizelist2)),log10(max(sizelist2)),20),density = True,histtype = 'step',label = 'data')
plt.xscale('log')
plt.yscale('log')
#plt.show()
print(x,bins)

data_fit1 = powerlaw.Fit(fit_method = 'Likelihood',data = sizelist2)
print(data_fit1.alpha,data_fit1.xmin)
theor_dist = powerlaw.Power_Law(xmin = data_fit1.xmin,parameters = [data_fit1.alpha])
simulated_data = theor_dist.generate_random(10**4)
x1,bins1,p1 = plt.hist(simulated_data,bins = logspace(log10(min(simulated_data)),log10(max(sizelist2)),20),density = True,histtype = 'step',label = 'powerlaw')
plt.xscale('log')
plt.yscale('log')
print(x1,bins1)


theor_dist2 = powerlaw.Truncated_Power_Law(xmin = data_fit1.xmin,parameters = [data_fit1.truncated_power_law.alpha, data_fit1.truncated_power_law.Lambda])
print('hey')
simulated_data2 = theor_dist2.generate_random(10**4)
print('hey')
x2,bins2,p2 = plt.hist(simulated_data2,bins = logspace(log10(min(simulated_data2)),log10(max(sizelist2)),20),density = True,histtype = 'step',label = 'truncated powerlaw')
plt.xscale('log')
plt.yscale('log')


plt.legend()
plt.xlabel('cluster size (S)')
plt.ylabel('p(S)')
plt.show()
#plt.savefig('/home/ranganath/Bedartha/common_directory/Codes/dist_comp.png')
plt.close()







