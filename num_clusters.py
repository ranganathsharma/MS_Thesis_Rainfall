# This program is to find the number of storms in each year from 1979 to 2019. The total number of storms, and the avearge along with the standard deviation will be the output.

from numpy import *
import csv
import pandas as pd

num = []
for year_index in range(1979,2020):
	file_address = '/home/ranganath/Bedartha/common_directory/nintynine_percent/cluster_files/clusters_path_filtered'+str(year_index)+'.csv'
	data = pd.read_csv(file_address)
	Index_list = set(data['per_index'].to_list())
	num.append(len(Index_list))

print(average(num),std(num))
print(sum(num))	
