# This file is to convert the processed data into the probability distribution functions for time and size distributions of the clusters using the powerlaw package. The data is saved in the common_directory/sizedist_files and timedist_files directories. The images are saved in sizedist_images and timedist_images directories respectively. 

#======================================================================
# imports
#======================================================================

import powerlaw 
import matplotlib.pyplot as plt
from numpy import *
import copy

#======================================================================

print('The imports are done')

# data loading for the non recursive case. Comment it out once all the files are procured and ready for a iterative way of analysis

year_index = 1979

datafile = open('/home/goswami/ranganathabr08/common_directory/sizedist_files/sizedist_'+str(year_index),'r')
sizelist = datafile.read()
datafile.close()

print('The datafile has been read')

sizelist = sizelist.split('\n')
while ('' in sizelist) == 1:
	sizelist.remove('')
	
sizearray = []
for i in range(len(sizelist)):
	temp = sizelist[i].split(',')
	while ('' in temp) ==1:
		temp.remove('')
	sizearray += [temp]
	
for i in range(len(sizearray)):
	for j in range(len(sizearray[i])):
		sizearray[i][j] = float(sizearray[i][j])/10**6
		
# preprocessing of the data is complete. The reduction of a factor of 10**6 is to bring the cluster size in km**3/hour units


#====================================================================
# Using the powerlaw package to fit the data
#====================================================================


#data_fit2 = powerlaw.Fit(fit_method = 'Likelihood',data = sizearray[i])
data_fit2 = powerlaw.Fit(fit_method = 'Likelihood',data = sizearray[i])
plotter = powerlaw.plot_pdf(data = sizearray[i],marker = '.')
x = logspace(log(min(sizearray[0])),log(max(sizearray[0])),100)
y = x**-data_fit2.alpha
plt.plot(x,y,label = str(year_index)+', a ='+str(round(data_fit2.power_law.alpha,4)))


plt.ylim(10**-7,1)
plt.xlim(1,10**4)
plt.xlabel('cluster size - s ($km^3$)')
plt.ylabel('p(s)')
plt.legend()
plt.show()
#====================================================================

