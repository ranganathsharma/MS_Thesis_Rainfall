from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import copy

start_year = 1979
end_year = 1980

lats = []
lons = []

for year_index in range(start_year,end_year):
	datafile = pd.read_csv('/home/ranganath/Bedartha/common_directory/nintynine_percent/cluster_files/clusters_path_filtered'+str(year_index)+'.csv')
	index_list = set(datafile['per_index'].to_list())
	for count, track_index in enumerate(index_list,0):
		tempdata = datafile[datafile['per_index'] == track_index]
		lats.append(tempdata['lat_mean'].to_list())
		lons.append(tempdata['lon_mean'].to_list())
		if count%100 == 0:
			print(count)
			
mainmap = Basemap(projection = 'merc', llcrnrlat = -15, llcrnrlon = 30, urcrnrlat = 45, urcrnrlon = 120,resolution = 'i')
mainmap.drawcoastlines(linewidth = 0.25)
mainmap.drawmeridians(arange(30,120,30))
mainmap.drawparallels(arange(-15,45,30))

len_total_data = len(lats)
print(lats)
for i in range(len_total_data):
	x,y = mainmap(lons[i],lats[i])
	mainmap.plot(x,y, color = "grey", alpha = 0.4,linestyle = 'dashed',linewidth = 0.15)

plt.ylabel('lat')
plt.xlabel('lon')
#plt.savefig('/home/ranganath/Bedartha/common_directory/nintynine_percent/cluster_images/'+str(hours_index)+'-15_long_storm.png')	
plt.show()


plt.close()
