
# ## modified by CHUYAN at March 26/2021; Moisture convergence

import xarray as xr
import PyNIO as Nio

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker   #..important for gridlinrs
import cartopy.crs as ccrs   #..projection method
import cartopy.feature as cfeat
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter   #..x,y --> lon, lat



f5   = xr.open_dataset('era5_monthly_averaged_2019_uv.grib',  engine= 'pynio')
f6   = xr.open_dataset('era5_monthly_averaged_2019.grib', engine= 'pynio')

# print(f6.dims)
# print(f6.coords)

#print(f6.variables['TP_GDS0_SFC_S130'])
print(f6.coords['g0_lon_2'])


E   =  f6['E_GDS0_SFC_S130'][6,::-1,:]  * 1000.   #..Surface Evaporation,
P   =  f6['TP_GDS0_SFC_S130'][6,::-1,:] * 1000.   #..Precipitation, Convert the units from m of water per day-> mm*day^-1 by multiply 1000
E.attrs['units']  = 'mm/day'
P.attrs['units']  = 'mm/day'
MC  = f6['VIDMOF_GDS0_EATM_S123'][6,::-1,:] * (24.*60.*60.)   #..Moisture Convergence. Convert the units from kg m^-2 s^-1 -> mm*day^-1
MC.attrs['units']  = 'kg m**-2 day**-1'

# print(np.min(MC.values))
# print(E.values)

MC_p_e = P+E                     #....Evaporation flux value originally is negative(upward), positive(downward), so in convention, "P-E" = P+E
print(np.mean(MC_p_e.values))   #..Janu:0.233
print(-np.mean(MC.values))      #..Janu:0.169

# U   = f5['U_GDS0_ISBL_S123']
# print(f5.coords['lv_ISBL1'])   #..29 levels, 700-[17]


#..test for p-e distribution
lat   = f6['g0_lat_1'][::-1]   #..south_to_north
lon   = f6['g0_lon_2'][:]
X, Y  = np.meshgrid(lon, lat)


proj   = ccrs.PlateCarree(central_longitude=180)
fig   = plt.figure( figsize=(6., 8.), dpi =120)

ax1   = plt.subplot(211, projection = proj)
ax2   = plt.subplot(212, projection =proj)
#..fig, (ax1,ax2) = plt.subplots(2,1)

#..map attributes country border, coastline, Rivers, Lakes..
ax1.add_feature(cfeat.COASTLINE.with_scale('110m'), zorder=1, linewidth=0.6)
ax2.add_feature(cfeat.COASTLINE.with_scale('110m'), zorder=1, linewidth=0.6)


#..label the geographic outlines by "gridlines" statement in cartopy:
'''
gl   =  ax1.gridlines(crs=proj,  draw_labels=True, linestyle='--')
gl.xlabels_top  = False #..turn off top label
gl.ylabels_right = False #
gl.xformatter   = LongitudeFormatter #..x set to longitude format
gl.yformatter   = LatitudeFormatter
gl.xlocator     = mticker.FixedLocator(np.arange(-180,180,30))
gl.ylocator     = mticker.FixedLocator(np.arange(-90,90, 30))
'''

#..another way to label Latitude/lonitude line
xticks = np.arange(-180.,180., 45.)
yticks = np.arange(-90.,90., 30.)
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
ax1.xaxis.set_major_formatter(LongitudeFormatter())
ax1.yaxis.set_major_formatter(LatitudeFormatter())
ax1.xaxis.set_minor_locator(mticker.MultipleLocator(5))   #..minor ticks

ax2.set_xticks(xticks)
ax2.set_yticks(yticks)
ax2.xaxis.set_major_formatter(LongitudeFormatter())
ax2.yaxis.set_major_formatter(LatitudeFormatter())
ax2.xaxis.set_minor_locator(mticker.MultipleLocator(5))   #..minor ticks


clevel = np.arange(-8.,8., 0.2)
contourf1 = ax1.contourf(X, Y, -MC, levels= clevel, cmap='BrBG', zorder=2, extend='both')
cb1  = fig.colorbar(contourf1, ax=ax1, orientation='horizontal', label='[mm/day]', extend='both', shrink=0.72, pad=0.10)
cb1.set_ticks(np.array([-8.,-7.,-6.,-5.,-4.,-3.,-2.,-1.,1., 2.,3.,4.,5., 6.,7., 8.]))
cb1.set_ticklabels([-8,-7,-6,-5,-4,-3,-2,-1,1, 2,3,4,5, 6, 7, 8])


contourf2 = ax2.contourf(X, Y, P+E, levels= clevel, cmap='BrBG', zorder=2, extend='both')
cb2  = fig.colorbar(contourf2, ax=ax2, orientation='horizontal', label='[mm/day]', extend='both', shrink=0.72, pad=0.10)

cb2.set_ticks(np.array([-8.,-7.,-6.,-5.,-4.,-3.,-2.,-1.,1., 2.,3.,4.,5., 6.,7., 8.]))
cb2.set_ticklabels([-8,-7,-6,-5,-4,-3,-2,-1,1, 2,3,4,5, 6, 7, 8])

ax1.set_title('moisture conv', fontsize=12, loc='left')
ax2.set_title('P-E', fontsize=12, loc='left')

plt.suptitle('July_2019, monthly mean data from ERA5')
plt.savefig('MC_vs_P-E_July.png')
