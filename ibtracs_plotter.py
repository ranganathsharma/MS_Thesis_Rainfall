from numpy import *
import matplotlib.pyplot as plt
import pandas as pd

# This file is to plot the ibtracs data to check the level of overlap.

file_address = '/home/ranganath/Bedartha/common_directory/ibtracs/ibtracs_filtered_data.csv'
data = pd.read_csv(file_address)

lat_list = []
lon_list = []

index_list = set(data['SID'].to_list())

data['LAT'] = round(data['LAT']/1.176471)*1.176471
data['LON'] = round(data['LON']/1.176471)*1.176471

for index in index_list:
	data_temp = data[data['SID'] == index]
	lat_list += [data_temp['LAT'].to_list()]
	lon_list += [data_temp['LON'].to_list()]
	
for i in range(len(lat_list)):
	plt.scatter(lon_list[i],lat_list[i],s = 0.4)

plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('storm tracks')
plt.show()
data.to_csv('/home/ranganath/Bedartha/common_directory/ibtracs/ibtracs_filtered_data_rounded_coarse.csv')
