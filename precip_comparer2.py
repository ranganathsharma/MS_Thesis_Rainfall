#================================================================
# Description
#================================================================

# This program imports the data from the ibtracs folder and processes it such that the positions of all the storm centers are considered with their timestamps. The maximum amount of precipitation during the span 1979 to 2019 will be considered for every grid point and the percentile of the track data will be calculated. This yields the information about the kind of precipitation that has taken place in the storm tracks which haven't be considered. 

# NOTE : THIS PROGRAM HAS TO BE RUN ON SLURM

#================================================================
# imports
#================================================================

from numpy import *
import csv
import pandas as pd
import xarray
from datetime import datetime
from scipy import stats
#import os

#================================================================
# The ibtracs_filtered_data_rounded is a file with rounded grid points to have a better agreement with that of the reanalysis data.

fileaddress = '/home/goswami/ranganathabr08/common_directory/Codes/ibtracs/ibtracs_filtered_data_rounded.csv'
#fileaddress = '/home/ranganath/Bedartha/common_directory/Codes/ibtracs/ibtracs_filtered_data_rounded_coarse.csv'
#fileaddress = '/home/ranganath/Bedartha/common_directory/Codes/ibtracs/ibtracs_filtered_data_rounded.csv'
ibtracs_data = pd.read_csv(fileaddress)

ibtracs_data['datetime'] = pd.to_datetime(ibtracs_data['Time'],unit = 's')
ibtracs_data['year'] = ibtracs_data['datetime'].dt.year
ibtracs_data = ibtracs_data.drop(['datetime','Unnamed: 0'],axis = 1)
len_file = len(ibtracs_data)

percentile_list = []
SID_list = []
for year_index in range(1979,2020):
	
	print(year_index)
	ibtracs_specific = ibtracs_data[ibtracs_data['year'] == year_index]
	#f = xarray.open_dataset('/home/ranganath/Bedartha/Codes/processed_data/'+str(year_index)+'.nc')
	#f = xarray.open_dataset('/home/goswami/ranganathabr08/common_directory/Codes/processed_data/'+str(year_index)+'.nc')
	f = xarray.open_dataset('/home/goswami/ranganathabr08/data_folder/total_precip/hourly/imdaa_singlelevel_hourly_totalprecip_'+str(year_index)+'.nc', decode_times=False)
	v_perm = f.to_dataframe()
	v_perm.reset_index(inplace = True)
	f.close()
	#for j in range(len(ibtracs_specific)):
	for j in range(len(ibtracs_specific)):
		SID = ibtracs_specific.iloc[j][0]
		time = ibtracs_specific.iloc[j][3]
		time_timestamp = pd.Timestamp(time,unit = 's')
	
		v = v_perm[(v_perm['latitude'] >= ibtracs_specific.iloc[j][1] - 0.01) & (v_perm['latitude'] <= ibtracs_specific.iloc[j][1] + 0.01) & (v_perm['longitude'] >= ibtracs_specific.iloc[j][2] - 0.01) & (v_perm['longitude'] <= ibtracs_specific.iloc[j][2] + 0.01)]
		#v = v_perm[(v_perm['latitude'] >= ibtracs_specific.iloc[j][1] - 0.58) & (v_perm['latitude'] <= ibtracs_specific.iloc[j][1] + 0.58) & (v_perm['longitude'] >= ibtracs_specific.iloc[j][2] - 0.58) & (v_perm['longitude'] <= ibtracs_specific.iloc[j][2] + 0.58)]
		
		final_dataframe = v[v['time'] == time_timestamp]
		
		try:
			percentile_value = stats.percentileofscore(v['APCP_sfc'].to_list(), final_dataframe.iloc[0][3])
			percentile_list.append(percentile_value)
			SID_list.append(SID)
			
		except:
			percentile_list = percentile_list


	try:
		del v_perm
		del v
		del ibtracs_specific
	except:
		print('Done')	
			
data = {'SID':SID_list,'pct_score':percentile_list}
df = pd.DataFrame(data)
#df.to_csv('/home/ranganath/Bedartha/common_directory/percentile_coarse.csv')
df.to_csv
			
'''
new_list = ['2009201N21089', '1997232N20088', '2015191N23085', '1998272N18068', '1997210N21089', '1994206N22107', '2016222N22089', '2006264N23087', '2017232N19130', '2007151N14072', '1991208N20090', '2006182N20088', '2006228N21088', '2016178N22068', '1995269N21088', '2019160N11073', '2019264N19071', '2005262N13127', '2005257N15120', '1990232N22089', '1996167N16072']

a = pd.read_csv('/home/ranganath/Bedartha/common_directory/percentile_coarse.csv')

b = a[~a.SID.isin(new_list)]
c = a[a.SID.isin(new_list)]

print(a.head())

print('non selected storm percentile is on an average',average(b['pct_score'].to_list()),std(b['pct_score'].to_list()))
print('selected storm percentile is on an average', average(c['pct_score'].to_list()),std(c['pct_score'].to_list()))
'''
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

