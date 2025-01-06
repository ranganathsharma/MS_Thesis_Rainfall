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

#================================================================
# The ibtracs_filtered_data_rounded is a file with rounded grid points to have a better agreement with that of the reanalysis data.

fileaddress = '/home/goswami/ranganathabr08/common_directory/Codes/ibtracs/ibtracs_filtered_data_rounded_coarse.csv'
#fileaddress = '/home/ranganath/Bedartha/common_directory/Codes/ibtracs/ibtracs_filtered_data_rounded_coarse.csv'
ibtracs_data = pd.read_csv(fileaddress)
loc_list = []
len_file = len(ibtracs_data['SID'].to_list())


ibtracs_data['datetime'] = pd.to_datetime(ibtracs_data['Time'],unit = 's')
ibtracs_data['year'] = ibtracs_data['datetime'].dt.year
print(ibtracs_data.columns)


for year_index in range(1979,1980):

	ibtracs_specific = ibtracs_data[ibtracs_data['year'] == year_index]
	ibtracs_specific = ibtracs_specific.groupby(['LAT','LON']).size().reset_index().rename(columns={0:'count'})

	ibtracs_specific.drop(['count'],axis = 1)
	
	print(ibtracs_specific[:10])
	print(ibtracs_specific.columns)
	print(len(ibtracs_specific))
	
	for i in range(1):

		f = xarray.open_dataset('/home/goswami/ranganathabr08/data_folder/total_precip/hourly/imdaa_singlelevel_hourly_totalprecip_'+str(year_index)+'.nc', decode_times=False)
		#f = xarray.open_dataset('/home/ranganath/Bedartha/Codes/processed_data/'+str(year_index)+'.nc')
		v = f.to_dataframe()
		v.reset_index(inplace = True)
		v = v[(v['latitude'] >= ibtracs_specific.iloc[i][1] - 10**-3) | (v['latitude'] <= ibtracs_specific.iloc[i][1] + 10**3)]
		v = v[(v['longitude'] >= ibtracs_specific.iloc[i][2] - 10**-3) | (v['longitude'] <= ibtracs_specific.iloc[i][2] + 10**3)]
		f.close()
			
		# Now loading all 41 years' data into the list is complete.

		print(v[:10])
		
		percentile_list = []
		v['pct_score'] = v.APCP_sfc.rank(pct = True)*100
		
		print(v[:10])
		
		v.to_csv('/home/goswami/ranganathabr08/pct_scores.csv')
		


	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

