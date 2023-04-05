# #...for all functions use 'binned_statistic_2d' func in scipy


import netCDF4
import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats


def binned_cyGlobal5(S, lat, lon):
    '''
    Calculate the binned array for the mean value within 5X5 degree Bin Boxes in global REGION
    '''
    if max(lon) > 180.:
        print("Please convert 0 - 360 longitude scale to -180 - 180 scale.")
    XX, YY = np.meshgrid(lon, lat, indexing='xy')
    #..Global region from 90S ~ 90N
    x_range = np.arange(-180., 180.5, 5.)   #.. number: 72
    y_range = np.arange(-90., 90, 5.)   #.. (36)
    
    xbins, ybins = len(x_range) - 1, len(y_range)
    
    S_binned_array = np.zeros((S.shape[0],ybins,xbins))
    
    for i in np.arange(S.shape[0]):
        S_time_step = S[i,:,:] *1.
        
        #..find and subtract the missing points
        ind = np.isnan(S[i,:,:]) == False
        S_binned_time, xedge, yedge, binnumber = stats.binned_statistic_2d(XX[ind].ravel(),YY[ind].ravel(), values = S_time_step[ind].ravel(), statistic ='mean', bins=[xbins, ybins], expand_binnumbers =True)
        
        S_binned_array[i,:,:] = S_binned_time.T
    
    return S_binned_array



def binned_cySouthOcean5(S, lat, lon):
    '''
    Calculate the binned array for the mean value within 5X5 degree Bin Boxes in Southern Ocean REGION
    '''

    XX, YY = np.meshgrid(lon, lat, indexing='xy')
    #..Southern Ocean region from 85S 40S
    x_range = np.arange(-180., 180.5 , 5.)   #..number:72
    y_range = np.arange(-85., -35., 5.)   #.. (9)
    
    xbins, ybins = len(x_range) - 1, len(y_range) - 1
    
    S_binned_array = np.zeros((S.shape[0], ybins, xbins))
    
    for i in np.arange(S.shape[0]):
        S_time_step = S[i,:,:] *1.
        
        #..find and subtract the missing points
        ind = np.isnan(S[i,:,:]) == False
        S_binned_time, xedge, yedge, binnumber=stats.binned_statistic_2d(XX[ind].ravel(),YY[ind].ravel(), values = S_time_step[ind].ravel(), statistic = 'mean', bins=[xbins, ybins], expand_binnumbers =True)
        
        S_binned_array[i,:,:] = S_binned_time.T
    
    return S_binned_array


def binned_cySO_availabledata(S, lat, lon):
    '''
    Calculate the "sum / count" binned array for 5X5 degree Bin Boxes in Southern Ocean REGION:
    '''

    XX, YY = np.meshgrid(lon, lat, indexing='xy')
    #..Southern Ocean region from 85S 40S
    x_range = np.arange(-180., 180.5, 5.)   #..number:73
    y_range = np.arange(-85., -35., 5.)   #.. (9)
    
    xbins, ybins = len(x_range) - 1, len(y_range) - 1
    S_binned_count = np.zeros((S.shape[0], ybins, xbins))
    S_binned_sum = np.zeros((S.shape[0], ybins, xbins))
    
    for i in np.arange(S.shape[0]):
        S_time_step = S[i,:,:] *1.
        
        #..find and subtract the missing points
        ind = np.isnan(S[i,:,:]) == False
        S_binned_time_count, xedge1, yedge1, binnumber2 = stats.binned_statistic_2d(XX[ind].ravel(),YY[ind].ravel(), values = S_time_step[ind].ravel(), statistic = 'count', bins = [xbins, ybins], expand_binnumbers = True)
        
        S_binned_time_sum, xedge1, yedge1, binnumber2 = stats.binned_statistic_2d(XX[ind].ravel(),YY[ind].ravel(), values = S_time_step[ind].ravel(), statistic = 'sum', bins = [xbins, ybins], expand_binnumbers = True)
        
        S_binned_count[i, :, :] = S_binned_time_count.T
        S_binned_sum[i, :, :] = S_binned_time_sum.T
    S_binned_ratio = S_binned_sum / S_binned_count  # The ratio of missing_value inside the bin boxes.
    return S_binned_ratio



#..for calculating the pcolor map: use two variables to distinguish LWP amount: skin Temperature(SST)& moisture amount(PRW/WVP or p-e/MC)
def binned_skewTamnW(XX,  YY,  S, lat_Y, lon_X):
    '''
    Calculate the binned array for the mean value of LWP over T_skew-Wvp bidimensional axes
    '''

    #XX, YY  = np.meshgrid(lon_X, lat_Y, indexing='xy')
    #..split Tskew as N1 parts, Wvp as N2 parts ..
    #x_range  = np.arange(0., 364, 5.)   #..number:73
    #y_range  = np.arange(-90., 90, 5.)   #..(37)
    
    xbins, ybins = len(lon_X)-1, len(lat_Y)-1
    
    S_binned_array  = np.zeros((S.shape[0], ybins, xbins))   #..Take as output
    
    for i in np.arange(S.shape[0]):
        
        XX_time_step  = XX[i,:,:]
        YY_time_step  = YY[i,:,:]
        S_time_step  = S[i,:,:]

        
        #..find and subtract the missing points
        ind =  np.isnan(S[i,:,:]* XX[i,:,:]* YY[i,:,:])== False  #..Ind_true
        S_binned_time , xedge, yedge, binnumber  = stats.binned_statistic_2d(XX_time_step[ind].ravel(),YY_time_step[ind].ravel(), values = S_time_step[ind].ravel(),
                                                                            statistic ='mean', bins=[lon_X, lat_Y], expand_binnumbers =True)
        
        S_binned_array[i,:,:] = S_binned_time.T
    
    return S_binned_array

