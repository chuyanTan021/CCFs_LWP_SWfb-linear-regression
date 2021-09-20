### modified by chuyan

import netCDF4
import numpy as np
import matplotlib.pyplot as plt

fpath1 = '/glade/work/chuyan/tas_Amon_CESM2_abrupt-4xCO2_r1i1p1f1_gn_000101-015012.nc'
fpath2 = '/glade/work/chuyan/clwvi_Amon_CESM2_abrupt-4xCO2_r1i1p1f1_gn_000101-015012.nc'
fpath3 = '/glade/work/chuyan/clivi_Amon_CESM2_abrupt-4xCO2_r1i1p1f1_gn_000101-015012.nc'

ftas = netCDF4.Dataset(fpath1, 'r')
fiwp = netCDF4.Dataset(fpath3, 'r')
fawp = netCDF4.Dataset(fpath2, 'r')


time = fiwp.variables['time'][:]
#print(time)
lat  = fiwp.variables['lat'][:]   #..180/192 degree/ [-90,90]
#print(lat)
lon  = fiwp.variables['lon'][:]   #..1.25 degree/ [0,358.75]

# annual-mean surface T and clivi, clwvi for simulation time period

tas  = np.zeros((150,192,288))
iwp  = np.zeros((150,192,288))
awp  = np.zeros((150,192,288))
for i in range(0, 150):
    tas[i,:,:]  = np.nanmean(ftas.variables['tas'][i*12:(i+1)*12,:,:], axis = 0)
    iwp[i,:,:]  = np.nanmean(fiwp.variables['clivi'][i*12:(i+1)*12,:,:], axis=0)
    awp[i,:,:]  = np.nanmean(fawp.variables['clwvi'][i*12:(i+1)*12,:,:], axis=0)

#print(tas[1:3,:,:])
LWP = np.array(awp) - np.array(iwp) 

lat0  = -80. 
lati0 = min(range(len(lat)), key= lambda i:abs(lat[i]-lat0) )
lat1  = -40. 
lati1 = min(range(len(lat)), key= lambda i:abs(lat[i]-lat1) )

# global-mean surface T and Southern-Ocean-region-mean liquid water path

#..Area mean the data by cosine(lat):
#..weighted by Global:
xlon, ylat  = np.meshgrid(lon, lat)

weighted_metrix1 =  np.cos(np.deg2rad(ylat))   #..metrix has the same shape as tas/lwp, its value = cos(lat)
toc1  = np.sum(weighted_metrix1)   #..total of cos(lat metrix) for global region

tas_weighted =  tas * weighted_metrix1 / toc1
GMT  = np.sum(tas_weighted, axis=(1,2))

#..weighted by Southern Ocean region:
weighted_metrix2 =  weighted_metrix1[lati0:lati1,:]
toc2  = np.sum(weighted_metrix2)   #..total of cos(lat metrix) for (40-80) region


LWP_weighted  = LWP[:, lati0:lati1,:] * weighted_metrix2 / toc2
lwp_SO  = np.sum(LWP_weighted, axis=(1,2))

#GMT  = np.nanmean(tas_weighted, axis=(1,2) )
#lwp_SO = np.nanmean(LWP[:, lati0:lati1,:], axis=(1,2))
print(GMT)

fig  = plt.figure(figsize=(10.3,8.4))  
ax1  = plt.axes()
plt.scatter(GMT,lwp_SO)
plt.xlabel('GMT')
plt.ylabel('LWP$_{ex}$')
plt.ylim((0.08, 0.16))
#plt.yticks([0.08,0.09,0.10,0.11,0.12])

parameter = np.polyfit(GMT, lwp_SO, 3)
z1  = np.poly1d(parameter)
print(z1)

y2 = parameter[0]*GMT**3 + parameter[1]*GMT**2 +parameter[2]*GMT + parameter[3]
plt.plot(GMT,y2,color='r')

plt.savefig('lwp_over_GMT_p3.png')

