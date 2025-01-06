# This program helps to visualize the storms that have been constructed using deepgraph methodology

#=======================================================================
# imports
#=======================================================================

from numpy import *
import matplotlib.pyplot as plt
import mpl_toolkits.basemap
import copy

#=======================================================================

print('The imports are complete')

# importing the data file with information of average lat, lon and vol

datafile = open('/home/goswami/ranganathabr08/common_directory/cluster_files/clusters_path'+str(year_index)+'.csv', 'r')
data = datafile.read()
datafile.close()

print('The data has been loaded')

templist = datafile.split('\n')
heading = templist[0]

del templist

cluster_list = data.split(heading)
while ('' in cluster_list) ==1:
	cluster_list.remove('')

num = 1 # do not use 0 as it gives unrealistic data, assumed to be the distribution of the centers of all the clusters.
cluster_list = cluster_list[num]

cluster_list = cluster_list.split('\n')
while ('' in cluster_list) == 1:
	cluster_list.remove('')
data_array = zeros((len(cluster_list),5))

for i in range(len(cluster_list)):
	templist = cluster_list[i].split('\n')
	
	while ('' in templist) == 1:
		templist.remove('')
		
	cluster_list[i] = templist
	for j in range(5):
		cluster_list[i][j] = float(cluster_list[i][j])

print('The construction of the cluster list is complete')

data_array =  zeros((len(cluster_list),5))



		
