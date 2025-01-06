# This file is to convert the processed data into the probability distribution functions for time and size distributions of the clusters using the powerlaw package. The data is saved in the common_directory/sizedist_files and timedist_files directories. The images are saved in sizedist_images and timedist_images directories respectively. 

#======================================================================
# imports
#======================================================================

import powerlaw 
import matplotlib.pyplot as plt
from numpy import *
import copy

#======================================================================

# data loading for the non recursive case. Comment it out once all the files are procured and ready for a iterative way of analysis

year_index = 1979

datafile = open('/home/goswami/ranganathabr08/common_directory/timedist_files/timedist_'+str(year_index),'r')
sizelist = datafile.read()
datafile.close()

timelist = timelist.split('\n')
while ('' in timelist) == 1:
	timelist.remove('')
	
timearray = []
for i in range(len(timelist)):
	temp = timelist[i].split(',')
	while ('' in temp) ==1:
		temp.remove('')
	timearray += [temp]
	
for i in range(len(timearray)):
	for j in range(len(timearray[i])):
		timearray[i][j] = float(timearray[i][j])
		
# preprocessing of the data is complete.


#====================================================================
# Using the powerlaw package to fit the data
#====================================================================


data_fit2 = powerlaw.Fit(fit_method = 'Likelihood',data = timearray[i])
plotter = powerlaw.plot_pdf(data = timearray[i],marker = '.')
x = logspace(log(min(timearray[0])),log(max(timearray[0])),100)
y = x**-data_fit2.alpha
plt.plot(x,y,label = str(year_index)+', a ='+str(round(data_fit2.power_law.alpha,4)))


plt.ylim(10**-7,1)
plt.xlim(1,10**4)
plt.xlabel('cluster duration - d')
plt.ylabel('p(d)')
plt.legend()
plt.show()
#====================================================================







