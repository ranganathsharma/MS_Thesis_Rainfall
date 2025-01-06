import time
import deepgraph as dg
from numpy import *
from netCDF4 import Dataset
from netCDF4 import num2date
import os
import pandas as pd
import netCDF4 as nc
import xarray
import matplotlib.pyplot as plt
import mpl_toolkits
import mpl_toolkits.basemap
import datetime
import scipy
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from csv import writer

start_time = time.time()

print('The importing process is complete')
r = 0.05
p = 0.9
year_index = 2018

try: f.close()
except: pass

# decode_times = True gives the final values of time column in the format year, and time until seconds. 

f = xarray.open_dataset('/home/goswami/ranganathabr08/data_folder/total_precip/hourly/imdaa_singlelevel_hourly_totalprecip_'+str(year_index)+'.nc',decode_times = False)

# This step creates two new columns called as x and y

f['x'] = (('longitude'), arange(len(f.longitude)))
f['y'] = (('latitude'), arange(len(f.latitude)))

# Conversion into dataframe
v = f.to_dataframe()


v = v[v.APCP_sfc >= r]
# this step if given true also displays the index values in v
v.reset_index(inplace=True)

f.close()


#print('The time is saved in unix format')
ftime = f.time.units.split()[2:]
year, month, day = ftime[0].split('-')
hour = ftime[1].split(':')[0]

f.close()

v['abstime'] = pd.to_datetime(v['time'],unit = 's')

v['latitude'] = v['latitude'].astype(float16)
v['longitude'] = v['longitude'].astype(float16)
v['x'] = v['x'].astype(uint16)
v['y'] = v['y'].astype(uint16)

# the area has to be redefined corresponding to the grids. 

v['area'] = 111**2 * 0.11984**2 *cos(2*pi*v.latitude / 360.0)
v['area'] = v['area'].astype(float64)

# compute "volume of water precipitated" for each event. The factor of 1 is because the data has a time difference of one hour between measurements.
v['vol'] = v.APCP_sfc * 1 * v.area
v['vol'] = v['vol'].astype(float64)

v['g_id'] = v.groupby(['longitude', 'latitude']).grouper.group_info[0]
v['g_id'] = v['g_id'].astype(uint32)

v = v.groupby('g_id').apply(lambda x: x[x.APCP_sfc >= x.APCP_sfc.quantile(p)])

v['month'] = pd.DatetimeIndex(v['abstime']).month
v = v[(v['month'] >= 6) & (v['month']<=9)]

v.sort_values('time', inplace=True)
v.set_index(arange(len(v)), inplace=True)
v.rename(columns={'APCP_sfc': 'r',
                  'latitude': 'lat',
                  'longitude': 'lon',
                  'time': 'time',
                  'abstime': 'atime'},
         inplace=True)


print('The extraction of data from the .nc file is complete')
# The column names of v are latitude, longitude, time and APCP_sfc
# However if v.columns is called we only get time and APCP_sfc
# v.index prints the values of latitudes, longitudes and time_amax

#====================================================================
# construction of the deepgraph with edges defined by connectors and selectors
#====================================================================

g = dg.DeepGraph(v)

def grid_2d_dx(x_s, x_t):
    dx = x_t - x_s
    return dx

def grid_2d_dy(y_s, y_t):
    dy = y_t - y_s
    return dy

# and **selectors**

# In[7]:

def s_grid_2d_dx(dx, sources, targets):
    dxa = abs(dx)
    sources = sources[dxa <= 1]
    targets = targets[dxa <= 1]
    return sources, targets

def s_grid_2d_dy(dy, sources, targets):
    dya = abs(dy)
    sources = sources[dya <= 1]
    targets = targets[dya <= 1]
    return sources, targets
    
g.create_edges_ft(ft_feature=('time', 3600),
                  connectors=[grid_2d_dx, grid_2d_dy],
                  selectors=[s_grid_2d_dx, s_grid_2d_dy],
                  r_dtype_dic={'ft_r': bool,
                               'dx': int8,
                               'dy': int8},
                  logfile='create_e'+str(year_index),
                  max_pairs=1e7)
                  
#====================================================================
# Construction complete
#====================================================================

# rename fast track relation
g.e.rename(columns={'ft_r': 'dt'}, inplace=True)
g.plot_logfile('create_e')  

g.append_cp(consolidate_singles=True)
# we don't need the edges any more
del g.e

#g.v.cp.max()
print('The construction of the deepgraph is complete')

#print(g.v.cp.value_counts())

feature_funcs = {'time': [min, max],'vol': [sum],'lat': [mean],'lon': [mean]}

cpv, gv = g.partition_nodes('cp', feature_funcs, return_gv=True)
cpv['g_ids'] = gv['g_id'].apply(set)

# append cardinality of g_id sets
cpv['n_unique_g_ids'] = cpv['g_ids'].apply(len)

# append time spans
cpv['dt'] = cpv['time_max'] - cpv['time_min']

# append spatial coverage
def area(group):
    return group.drop_duplicates('g_id').area.sum()
cpv['area'] = gv.apply(area)

plt.close()


# To save up memory, delete the main dataframe called v
del v

print('Partitioning the deepgraph to get the clusters is complete')

#====================================================================
# This section is for plotting the size distribution in log scale #====================================================================

cpv = cpv[cpv.vol_sum >= 10**6]

'''
hister,bins, _ = plt.hist(cpv.vol_sum.values,bins = 20 )
logbins = logspace(log10(bins[0]),log10(bins[-1]),len(bins))
histernew = plt.hist(cpv.vol_sum.values,bins = logbins[:])
plt.close()

#print('histernew is = ',histernew[0],len(histernew[0]))
#print('histernew part2 is = ',histernew[1],len(histernew[1]))

plt.scatter(histernew[1][1:]/10**6,histernew[0],marker = '.')
plt.xlabel('Cluster Size $km^3$')
plt.ylabel('cluster frequency')
plt.xscale('log')
plt.yscale('log')
plt.title('Histogram of size of clusters of '+str(year_index))
plt.savefig('/home/goswami/ranganathabr08/common_directory/sizedist_images/sizedist'+str(year_index)+'.png')
plt.close()

with open('/home/goswami/ranganathabr08/common_directory/sizedist_files/sizehistdetails'+str(year_index)+'.csv','a') as datafile:
	writer_object = writer(datafile)
	writer_object.writerow(histernew[1]/3600)
	writer_object.writerow(histernew[0])
	datafile.close()
'''	
print('The value of volume of precipitation is not in standard units. The division by 10**6 must be done to get back to km**3/hour unit')

with open('/home/goswami/ranganathabr08/common_directory/sizedist_files/size_list'+str(year_index)+'.csv','a') as datafile:
	writer_object = writer(datafile)
	writer_object.writerow(cpv.vol_sum.values)
	datafile.close()


print('The process of writing the data of cluster size into a file is complete')
#////////////////////////////////////////////////////////////////////


#====================================================================
# This section is to check the distribution of the time duration of storms. Hence instead of using the volume as the measure, we use dt
#====================================================================

cpv = cpv[cpv.dt >= 3600*2]
'''
hister,bins, _ = plt.hist(cpv.dt.values,bins = 200 )
logbins = logspace(log10(bins[0]),log10(bins[-1]),len(bins))
histernew = plt.hist(cpv.dt.values,bins = logbins[:])
plt.close()

plt.scatter(histernew[1][1:]/(3600),histernew[0],marker = '.')
plt.xlabel('Cluster duration in hours - dt')
plt.ylabel('p(dt)')
plt.xscale('log')
plt.yscale('log')
plt.title('Histogram of cluster duration of '+str(year_index))
plt.savefig('/home/goswami/ranganathabr08/common_directory/Monsoon/timedist'+str(year_index)+'.png')
plt.close()

with open('/home/goswami/ranganathabr08/common_directory/Monsoon/time_hg_details.csv','a') as datafile:
	writer_object = writer(datafile)
	writer_object.writerow(histernew[1]/3600)
	writer_object.writerow(histernew[0])
	datafile.close()
'''	
with open('/home/goswami/ranganathabr08/common_directory/timedist_files/timedist_list'+str(year_index)+'.csv','a') as datafile:
	writer_object = writer(datafile)
	writer_object.writerow(cpv.dt.values)
	datafile.close()

print('The process of writing the time duration of the clusters into a file is complete')
#print(histernew)

#////////////////////////////////////////////////////////////////////


#====================================================================
# This section is to save the clusters with information of the volume of precipitation and mean latitude and longitude, time along with the cluster index for a given year and a month.
#====================================================================

gt = dg.DeepGraph(g.v)

feature_funcs = {'vol': [sum],'lat': [mean],'lon': [mean]}
datafile = open('/home/goswami/ranganathabr08/common_directory/cluster_files/clusters_path'+str(year_index)+'.csv','w')

print(gt.v.cp.max())

for i in range(int(gt.v.cp.max()+1)):
	gt_perm = copy.deepcopy(gt)
	gt_perm.filter_by_values_v('cp',i)
	st,stv = gt_perm.partition_nodes('time',feature_funcs,return_gv = True)
	#savetxt('/home/ranganath/Bedartha/Codes/csvfile_monsoon'+str(year_index)+'second.csv', st.values,delimiter = ' ')
	st.to_csv(datafile)
print('The process of writing the average positions of all the clusters into a csv file is complete')

#=====================================================================
#/////////////////////////////////////////////////////////////////////

#=====================================================================
# This is the attempt to access the average latitude and longitude
# for any given cluster and plot them for say first cluster
#=====================================================================

from mpl_toolkits.basemap import Basemap

gt = dg.DeepGraph(g.v)
gt_copy = copy.deepcopy(gt)
gt.filter_by_values_v('cp',1)

#plt.subplot(1,2,1)

feature_funcs = {'vol': [sum],'lat': [mean],'lon': [mean]}
st,stv = gt.partition_nodes('time',feature_funcs,return_gv = True)

mainmap = Basemap(projection = 'merc', llcrnrlat = -15, llcrnrlon = 30, urcrnrlat = 45, urcrnrlon = 120,resolution = 'i')
mainmap.drawcoastlines(linewidth = 0.25)
mainmap.drawmeridians(arange(30,120,30))
mainmap.drawparallels(arange(-15,45,30))

x,y = mainmap(st.lon_mean.values, st.lat_mean.values)

pos = mainmap.scatter(x,y,c = (st.index.values-st.index[0])/(3600*24),marker = '.',alpha = 0.5)

char = plt.colorbar(pos)
char.set_label('Duration of storm in days')

plt.ylabel('lat')
plt.xlabel('lon')
plt.savefig('/home/goswami/ranganathabr08/common_directory/cluster_images/largest_cluster'+str(year_index)+'.png')
plt.close()

print('Saving an image of the largest cluster of the year is complete')

#=================================================================================
# plotting the largest cluster of the year_index
#=================================================================================

filename = '/home/goswami/ranganathabr08/common_directory/cluster_files/clusters_path'+str(year_index)+'.csv'

datafile = open(filename,'r')
data = datafile.read()
datafile.close()

datatemp = data.split('\n')
heading = datatemp[0]
data = data.split(heading)

del datatemp
del heading

while ('' in data) == 1:
	data.remove('')
data = data[1].split('\n')
while ('' in data) == 1:
	data.remove('')
	
data_array = zeros((len(data),5))	
for i in range(len(data)):
	tempdata = data[i].split(',')
	while ('' in tempdata) == 1:
		tempdata.remove('')
	data[i] = tempdata
	for j in range(5):
		data_array[i,j] = float(data[i][j])
data = array(data)		


pos = plt.scatter(data_array[:,3],data_array[:,4],c= (data_array[:,0]-data_array[0,0])/(3600*24),marker = '.')
plt.colorbar(pos)
char = plt.colorbar(pos)
char.set_label('Duration of storm in days')

plt.savefig('/home/goswami/ranganathabr08/common_directory/cluster_images/largest_cluster_reconstructed'+str(year_index)+'.png')
plt.close()

print('done')
print("--- %s seconds ---" % (time.time() - start_time))

datafile = open('/home/goswami/ranganathabr08/common_directory/time.txt','w')
datafile.write(time.time()-start_time)
datafile.close()
