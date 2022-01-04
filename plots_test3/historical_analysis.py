# get the data we needed from read module:'get_LWPCMIP6', and do some data-processing for building the linear regression CCFs_Clouds models:
# transform data to annual-mean/ monthly-mean bin array or flattened array;
##  f;

import netCDF4
from numpy import *
import matplotlib.pyplot as plt
import xarray as xr
import PyNIO as Nio
import pandas as pd
import glob
from scipy.stats import *
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm

from area_mean import *
from binned_cyFunctions5 import *
from read_hs_file import read_var_mod


from get_LWPCMIP6data import *
from useful_func_cy import *

from run_simple_cmip6_pc import *


def historical_analysis(startyr, endyr, **model_data):
    rawdata_dict  = {}
    
    
    # get variable data
    if model_data['cmip'] == 'cmip6':
        
        inputVar_his  = get_historical(startyr, endyr, **model_data)
        
    else:
        print('not historical data')
        
    #..get the shapes of monthly data
    
    shape_lat = len(inputVar_his['lat'])
    shape_lon = len(inputVar_his['lon'])
    shape_time = len(inputVar_his['times'])
    
    print(shape_lat, shape_lon, shape_time)
    
    
    #..dimesnsions info
    lons        = array(inputVar_his['lon'])
    lats        = array(inputVar_his['lat'][:])
    
    levels      = array(inputVar_his['pres'])
    times     = array(inputVar_his['times'])
    
    
    #..choose lat from 40 -85 °S as the Southern-Ocean Regions
    lati0 = -40.
    latsi0 = min(range(len(lats)), key = lambda i: abs(lats[i] - lati0))
    lati1 = -85.
    latsi1 = min(range(len(lats)), key = lambda i: abs(lats[i] - lati1))
    print('lat index for 40.S; 85.S', latsi0, latsi1)
    
    
    shape_latSO =  (latsi0+1) - latsi1
    print(" shape of Southern Ocean latititude point= ", shape_latSO)
    
    
    #.. Variables: LWP, SST, Subsidence @ 500 mb
    LWP_his  = array(inputVar_his['clwvi']) - array(inputVar_his['clivi'])   #..units in kg m^-2
    
    SST_his  = array(inputVar_his['sfc_T'])
    
    Precip_his =  array(inputVar_his['P']) * (24.*60.*60.)   #..Precipitation. Convert the units from kg m^-2 s^-1 -> mm*day^-1
    print('historical period average Pr(mm/ day): ', nanmean(Precip_his))   #.. IPSL/abr2.80..  CNRM ESM2 1/abr 2.69.. CESM2/abr 2.74..
    Eva_his    =  array(inputVar_his['E']) * (24.*60.*60.)   #..evaporation, mm day^-1
    print('historical period average Evapor(mm/ day): ', nanmean(Eva_his))         #.. IPSL/abr2.50..  CNRM ESM2 1/abr 2.43.. CESM2/abr 2.43..
    
    MC_his  = Precip_his - Eva_his   #..Moisture Convergence calculated from abrupt4xCO2's P - E, Units in mm day^-1
    
    Twp_his  = array(inputVar_his['clwvi'])
    Iwp_his  = array(inputVar_his['clivi'])
    prw_his  = array(inputVar_his['prw'])
    
    print('historical period data shape in: ', Eva_his.shape )
    
    
    #..Meteological Subsidence  at 500 hPa, units in Pa s^-1:
    Subsidence_his =  array(inputVar_his['sub'])
    
    
    # put monthly data into Dictionary, stored: SST, SUB at 500mb, LWP and other variables
    dict0_var = { 'LWP': LWP_his, 'TWP': Twp_his, 'IWP': Iwp_his, 'PRW': prw_his, 'SST': SST_his, 'p_e': MC_his, 'SUB': Subsidence_his, 'lat': lats, 'lon':lons, 'times':times, 'pres':levels}
    
    #.. calced raw CCF(SST& SUB at 500) metrics with raw LWP, wvp in 'historical' period
    rawdata_dict['dict0_var'] = dict0_var
    
    
    
    # get the Annual-mean, Southern-Ocean region arrays
    
    datavar_nas = ['LWP', 'TWP', 'IWP', 'PRW', 'SST', 'SUB']   #..6 varisables (in the same shape)
    
    dict1_var_yr  = {}
    shape_yr  = shape_time//12

    
    dict1_var_yr = get_annually_dict_so(dict0_var, datavar_nas, shape_time, latsi1, latsi0, shape_lon)
    print("dict1_var_yr data shape in: ", dict1_var_yr['LWP_yr'].shape)
    
    #.. Annually-mean metrics
    rawdata_dict['dict1_var_yr'] = dict1_var_yr
    
    
    
    #..set are-mean range/ Grids' longitude, Latitude and SO's Latitude
    x_range  = arange(-180., 183, 5.)   #..logitude sequences edge: number:73
    s_range  = arange(-90., 90, 5.)  + 2.5   #..global-region latitude edge:(36)

    y_range  = arange(-85, -35., 5.) + 2.5   #..southern-ocaen latitude edge:10
    
    
    # Calc binned array('r': any resolution) for Annually variable: 
    
    lat_array  = lats[latsi1:latsi0+1]
    lon_array  =  lons
    lat_array1 =  lats
    
    dict1_yr_bin  = {}
    for b in range(len(datavar_nas)):

        dict1_yr_bin[datavar_nas[b]+'_yr_bin']   =   binned_cySouthOcean_anr(dict1_var_yr[datavar_nas[b]+'_yr'], lat_array , lon_array, 5)
    
    rawdata_dict['dict1_yr_bin']  = dict1_yr_bin
    
    # Calc binned array('r': any resolution) for Monthly variable:
    
    dict1_mon_bin  = {}
    for c in range(len(datavar_nas)):

        dict1_mon_bin[datavar_nas[c]+'_mon_bin'] =    binned_cySouthOcean_anr(dict0_var[datavar_nas[c]][:, latsi1:latsi0+1, :], lat_array, lon_array, 5)
    
    rawdata_dict['dict1_mon_bin']  = dict1_mon_bin
    
    
    return rawdata_dict