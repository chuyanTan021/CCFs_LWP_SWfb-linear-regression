### get the data we need from read func: 'get_LWPCMIP6', and do some data-processing for building the linear regression CCFs-Clouds model; ###
### transform data to annual-mean/ monthly-mean bin array or flattened array; ###
### fitting the linear regression with 2&4 regimes models from pi-Control CCFs' sensitivities to the cloud properties, then do the regressions and save the data. ###

import netCDF4
from numpy import *
import matplotlib.pyplot as plt
import xarray as xr

# import PyNIO as Nio
import pandas as pd
import glob
from copy import deepcopy
from scipy.stats import *
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm

from area_mean import *
from binned_cyFunctions5 import *
from read_hs_file import read_var_mod


from get_LWPCMIP6data import *
from fitLRM_cy2 import *
from useful_func_cy import *


def calc_LRM_metrics(THRESHOLD_sst, THRESHOLD_sub, **model_data):
    # get variable data
    if model_data['cmip'] == 'cmip6':
        
        inputVar_pi, inputVar_abr = get_LWPCMIP6(**model_data)
        
    else:
        print('not cmip6')
    
    #..get the shapes of monthly data
    shape_lat = len(inputVar_pi['lat'])
    shape_lon = len(inputVar_pi['lon'])
    shape_time_pi = len(inputVar_pi['times'])
    shape_time_abr = len(inputVar_abr['times'])
    #print(shape_lat, shape_lon, shape_time_pi, shape_time_abr)
    
    
    #..choose lat 40 -85 °S as the Southern-Ocean Regions
    lons        = inputVar_pi['lon']
    lats        = inputVar_pi['lat'][:]
    
    levels      = array(inputVar_abr['pres'])
    times_abr   = inputVar_abr['times']
    times_pi    = inputVar_pi['times']
    
    
    lati0 = -40.
    latsi0= min(range(len(lats)), key = lambda i: abs(lats[i] - lati0))
    lati1 = -85.
    latsi1= min(range(len(lats)), key = lambda i: abs(lats[i] - lati1))
    print('lat index for 40.s; 85.s', latsi0, latsi1)
    
    
    shape_latSO =  (latsi0+1) - latsi1
    #print(shape_latSO)
    
    
    #..abrupt-4xCO2 Variables: LWP, tas(gmt), SST, (MC), p-e; SW radiation metrics
    LWP_abr  = asarray(inputVar_abr['clwvi']) - asarray(inputVar_abr['clivi'])   #..units in kg m^-2
    
    gmt_abr  = asarray(inputVar_abr['tas'])
    
    SST_abr  = asarray(inputVar_abr['sfc_T'])
    
    
    Precip_abr =  asarray(inputVar_abr['P']) * (24.*60.*60.)   #.. Precipitation. Convert the units from kg m^-2 s^-1 -> mm*day^-1
    print('abr4x average Pr(mm/ day): ', nanmean(Precip_abr))   #.. IPSL/abr2.80..  CNRM ESM2 1/abr 2.69.. CESM2/abr 2.74..
    Eva_abr    =  asarray(inputVar_abr['E']) * (24.*60.*60.)   #.. Evaporation, mm day^-1
    print('abr4x average Evapor(mm/ day): ', nanmean(Eva_abr))         #.. IPSL/abr2.50..  CNRM ESM2 1/abr 2.43.. CESM2/abr 2.43..
    
    MC_abr  = Precip_abr - Eva_abr   #..Moisture Convergence calculated from abrupt4xCO2's P - E, Units in mm day^-1
    
    Twp_abr  = asarray(inputVar_abr['clwvi'])
    Iwp_abr  = asarray(inputVar_abr['clivi'])
    prw_abr  = asarray(inputVar_abr['prw'])
    
    # SW radiation metrics
    Rsdt_abr = asarray(inputVar_abr['rsdt'])
    Rsut_abr = asarray(inputVar_abr['rsut'])
    Rsutcs_abr = asarray(inputVar_abr['rsutcs'])
    print("shape of data in 'abrupt-4xCO2':  ",  Rsut_abr.shape, " mean abr-4x upwelling SW radiation flux in the SO (Assume with cloud): ",  nanmean(Rsut_abr[:, latsi1:latsi0 +1,:]))
    # print("shape of data in 'abrupt-4XCO2' exp:", Eva_abr.shape, 'abr4x mean-gmt(K): ', nanmean(gmt_abr))
    
    # albedo, albedo_clear sky 
    Albedo_abr = Rsut_abr / Rsdt_abr
    Albedo_cs_abr = Rsutcs_abr / Rsdt_abr
    
    #..pi-Control Variables: LWP, tas(gmt), SST, (MC), p-e ; SW radiation metrics (rsdt, rsut, rsutcs)
    LWP  = array(inputVar_pi['clwvi']) - array(inputVar_pi['clivi'])   #..units in kg m^-2
    
    gmt  = asarray(inputVar_pi['tas'])
    
    SST  = asarray(inputVar_pi['sfc_T'])
    
    
    Precip =  asarray(inputVar_pi['P'])* (24.*60.*60.)    #..Precipitation. Convert the units from kg m^-2 s^-1 -> mm*day^-1
    print('pi-C average Pr(mm/ day): ', nanmean(Precip))   #.. IPSL/piC 2.43..CNRM/piC 2.40.. CESM2/PIc 2.39
    Eva    =  asarray(inputVar_pi['E']) * (24.*60.*60.)   #..evaporation, mm day^-1
    print('pi-C average Evapor(mm/day): ', nanmean(Eva))   #.. IPSL/piC  2.21..CNRM/piC 2.20.. CESM2/PIc 2.17..
    
    MC  = Precip - Eva   #..Moisture Convergence calculated from pi-Control's P - E, Units in mm day^-1
    
    Twp  = asarray(inputVar_pi['clwvi'])
    Iwp  = asarray(inputVar_pi['clivi'])
    prw_pi  = asarray(inputVar_pi['prw'])
    
    # SW radiation metrics
    Rsdt_pi = asarray(inputVar_pi['rsdt'])
    Rsut_pi = asarray(inputVar_pi['rsut'])
    Rsutcs_pi = asarray(inputVar_pi['rsutcs'])
    print("shape of data in 'piControl':  ", Rsut_pi.shape, " mean pi-C upwelling SW radiation flux in the SO (Assume with cloud): "
, nanmean(Rsut_pi[:, latsi1:latsi0 +1,:]))
    # print("shape of data in 'piControl' data: ", Eva.shape, 'pi-C mean-gmt(K): ', nanmean(gmt))
    
    
    # albedo, albedo_clear sky
    Albedo_pi = Rsut_pi / Rsdt_pi
    Albedo_cs_pi = Rsutcs_pi / Rsdt_pi
    
    #..abrupt-4xCO2
    # Lower Tropospheric Stability (LTS):
    k  = 0.286
    
    theta_700_abr  = array(inputVar_abr['T_700']) * (100000./70000.)**k
    theta_skin_abr = array(inputVar_abr['sfc_T']) * (100000./asarray(inputVar_abr['sfc_P']))**k 
    LTS_m_abr  = theta_700_abr - theta_skin_abr
    
    
    #..Subtract the outliers in T_700 and LTS_m, 'nan' comes from missing T_700 data
    LTS_e_abr  = ma.masked_where(theta_700_abr >= 500, LTS_m_abr)
    
    # Meteorology Subsidence at 500 hPa, units in Pa s^-1:
    Subsidence_abr =  array(inputVar_abr['sub'])
    
    
    #..pi-Control
    # Lower Tropospheric Stability (LTS):
    theta_700  = array(inputVar_pi['T_700']) * (100000./70000.)**k
    theta_skin = array(inputVar_pi['sfc_T']) * (100000./asarray(inputVar_pi['sfc_P']))**k
    LTS_m  = theta_700 - theta_skin
    
    #..Subtract the outliers in T_700 and LTS_m 
    LTS_e  = ma.masked_where(theta_700 >= 500, LTS_m)
    
    #..Meteological Subsidence  at 500 hPa, units in Pa s^-1:
    Subsidence =  array(inputVar_pi['sub'])
    
    
    # define Dictionary to store: CCFs(4), gmt, other variables :
    dict0_PI_var = {'gmt': gmt, 'LWP': LWP, 'TWP': Twp, 'IWP': Iwp,  'PRW': prw_pi, 'SST': SST, 'p_e': MC, 'LTS': LTS_e, 'SUB': Subsidence, 'rsdt': Rsdt_pi, 'rsut': Rsut_pi, 'rsutcs': Rsutcs_pi, 
                     'albedo' : Albedo_pi, 'albedo_cs': Albedo_cs_pi, 'lat':lats, 'lon':lons, 'times': times_pi, 'pres':levels}

    dict0_abr_var = {'gmt': gmt_abr, 'LWP': LWP_abr, 'TWP': Twp_abr, 'IWP': Iwp_abr,  'PRW': prw_abr, 'SST': SST_abr, 'p_e': MC_abr, 'LTS': LTS_e_abr ,'SUB': Subsidence_abr, 'rsdt': Rsdt_abr, 'rsut': Rsut_abr, 'rsutcs': Rsutcs_abr, 
                      'albedo': Albedo_abr, 'albedo_cs': Albedo_cs_abr, 'lat':lats, 'lon':lons, 'times': times_abr, 'pres':levels}

    
    
    # get the Annual-mean, Southern-Ocean region arrays

    datavar_nas = ['LWP', 'TWP', 'IWP', 'PRW', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'SST', 'p_e', 'LTS', 'SUB']   #..13 varisables except gmt (lon dimension diff)

    dict1_PI_yr  = {}
    dict1_abr_yr = {}
    shape_yr_pi  = 99  # shape_time_pi//12
    shape_yr_abr =  shape_time_abr//12
    
    layover_yr_abr = zeros((len(datavar_nas), shape_yr_abr, shape_latSO, shape_lon))
    layover_yr_pi  = zeros((len(datavar_nas), shape_yr_pi, shape_latSO, shape_lon))

    layover_yr_abr_gmt = zeros((shape_yr_abr, shape_lat, shape_lon))
    layover_yr_pi_gmt = zeros((shape_yr_pi, shape_lat, shape_lon))


    for a in range(len(datavar_nas)):

        # a_array = dict0_abr_var[datavar_nas[a]]

        for i in range(shape_time_abr//12):
            layover_yr_abr[a, i,:,:] = nanmean(dict0_abr_var[datavar_nas[a]][i*12:(i+1)*12, latsi1:latsi0 +1,:], axis=0)

        dict1_abr_yr[datavar_nas[a]+'_yr'] =  layover_yr_abr[a,:]


        #b_array = dict0_PI_var[datavar_nas[a]]
        for j in range(shape_time_pi//12):
            layover_yr_pi[a, j,:,:] = nanmean(dict0_PI_var[datavar_nas[a]][j*12:(j+1)*12, latsi1:latsi0 +1,:], axis=0)

        dict1_PI_yr[datavar_nas[a]+'_yr'] = layover_yr_pi[a,:]
        print(datavar_nas[a])

    #print(dict1_PI_yr['LWP_yr'])
    
    # gmt
    for i in range(shape_time_abr//12):

        layover_yr_abr_gmt[i,:,:]  =  nanmean(dict0_abr_var['gmt'][i*12:(i+1)*12, :,:], axis=0)
    dict1_abr_yr['gmt_yr']  =   layover_yr_abr_gmt
    
    
    for j in range(shape_time_pi//12):
        layover_yr_pi_gmt[j,:,:]  =   nanmean(dict0_PI_var['gmt'][j*12:(j+1)*12, :,:], axis=0)
    dict1_PI_yr['gmt_yr']  =   layover_yr_pi_gmt

    #print(dict1_PI_yr['gmt_yr'])
    dict0_PI_var['dict1_yr'] = dict1_PI_yr
    dict0_abr_var['dict1_yr'] = dict1_abr_yr



    # Calculate 5*5 bin array for variables (LWP, CCFs) in Sounthern Ocean Region:
    
    #..set are-mean range and define functio
    x_range  = arange(-180., 180., 5.)   #..logitude sequences edge: number: 72
    s_range  = arange(-90., 90, 5.) + 2.5   #..global-region latitude edge:(36)

    y_range  = arange(-85, -40., 5.) +2.5   #..southern-ocaen latitude edge: 9

    
    # Annually variables in bin box:

    lat_array = lats[latsi1:latsi0+1]
    lon_array = lons
    lat_array1 = lats
    dict1_PI_var = {}             #..add at Dec.30th, at 2021. Purpose: shrink the output savez data dictionary: rawdata
    dict1_abr_var = {}            #..add at Dec.30th, at 2021. Purpose: shrink the output savez data dictionary: rawdata
    dict1_yr_bin_PI = {}
    dict1_yr_bin_abr = {}
    
    for b in range(len(datavar_nas)):

        dict1_yr_bin_abr[datavar_nas[b]+'_yr_bin'] = binned_cySouthOcean5(dict1_abr_yr[datavar_nas[b]+'_yr'], lat_array, lon_array)
        dict1_yr_bin_PI[datavar_nas[b]+'_yr_bin'] = binned_cySouthOcean5(dict1_PI_yr[datavar_nas[b]+'_yr'], lat_array, lon_array)


    #print(dict1_yr_bin_abr['PRW_yr_bin'].shape)
    #print(dict1_yr_bin_abr['gmt_yr_bin'])   #..(150, 36, 73)
    #print(dict1_yr_bin_PI['SUB_yr_bin'].shape)   #..(100, 10, 73)
    dict1_yr_bin_abr['gmt_yr_bin'] = binned_cyGlobal5(dict1_abr_yr['gmt_yr'], lat_array1, lon_array)
    dict1_yr_bin_PI['gmt_yr_bin'] = binned_cyGlobal5(dict1_PI_yr['gmt_yr'], lat_array1, lon_array)

    print('gmt_yr_bin')
    
    dict1_abr_var['dict1_yr_bin_abr']  =  dict1_yr_bin_abr
    dict1_PI_var['dict1_yr_bin_PI']  = dict1_yr_bin_PI


    # Monthly variables (same as above):
    dict1_mon_bin_PI  = {}
    dict1_mon_bin_abr = {}
    
    for c in range(len(datavar_nas)):

        dict1_mon_bin_abr[datavar_nas[c]+'_mon_bin'] = binned_cySouthOcean5(dict0_abr_var[datavar_nas[c]][0::12, latsi1:latsi0 +1,:], lat_array, lon_array)
        dict1_mon_bin_PI[datavar_nas[c]+'_mon_bin'] = binned_cySouthOcean5(dict0_PI_var[datavar_nas[c]][0::12, latsi1:latsi0 +1,:], lat_array, lon_array)

    dict1_mon_bin_abr['gmt_mon_bin'] =  binned_cyGlobal5(dict0_abr_var['gmt'][:,:,:], lat_array1, lon_array)
    dict1_mon_bin_PI['gmt_mon_bin']  =  binned_cyGlobal5(dict0_PI_var['gmt'][:,:,:], lat_array1, lon_array)

    print('gmt_mon_bin', " +", "Monthly other data: Jan")
    
    dict1_abr_var['dict1_mon_bin_abr'] = dict1_mon_bin_abr
    dict1_PI_var['dict1_mon_bin_PI'] = dict1_mon_bin_PI


    # input the shapes of year and month of pi&abr exper into the raw data dictionaries:
    dict1_abr_var['shape_yr'] = shape_yr_abr
    dict1_PI_var['shape_yr'] = shape_yr_pi

    dict1_abr_var['shape_mon'] = shape_time_abr
    dict1_PI_var['shape_mon'] = shape_time_pi

    # Output a dict for processing function in 'calc_LRM_metrics', stored the data dicts for PI and abr, with the model name_dict
    C_dict =  {'dict1_PI_var': dict1_PI_var, 'dict1_abr_var': dict1_abr_var, 'model_data': model_data}    #..revised in Dec.30th, at 2021,, note the name.
    D_dict  = deepcopy(C_dict)   # 'notice for the difference between shallow copy (object.copy()) and deep copy (copy.deepcopy(object))'
    B_dict  = deepcopy(C_dict)


    ###..Put data into 'fitLRM' FUNCTION to get predicted LWP splitted by 'Tr_sst'/'Tr_sub' infos_models:
    TR_sst   = THRESHOLD_sst    ###.. Important line
    TR_sub   = THRESHOLD_sub   ###.threshold of 500 mb Subsidences
    WD = '/glade/work/chuyan/Research/Cloud_CCFs_RMs/Course_objective_ana/data_file/'


    rawdata_dict1 = fitLRM3(TR_sst=TR_sst, s_range=s_range, y_range=y_range, x_range=x_range, C_dict = B_dict)
    rawdata_dict3 = p4plot1(s_range=s_range, y_range=y_range, x_range=x_range, shape_yr_pi=shape_yr_pi, shape_yr_abr=shape_yr_abr, rawdata_dict=rawdata_dict1)

    rawdata_dict3['TR_sst'] =  THRESHOLD_sst

    savez(WD+C_dict['model_data']['modn']+'_r2_hotcold(Jan)_(largestpiR2)_'+str(round(TR_sst, 2))+'_dats', model_data = C_dict['model_data'],rawdata_dict = rawdata_dict3)
    #.. best fit save_2lrm command:
    # savez(WD+C_dict['model_data']['modn']+'_best(test5)fit_'+str(round(TR_sst, 2))+'_dats', model_data = C_dict['model_data'],rawdata_dict = rawdata_dict3)

    rawdata_dict2 = fitLRM4(TR_sst=TR_sst, TR_sub=TR_sub, s_range=s_range, y_range=y_range, x_range=x_range, C_dict = D_dict)
    rawdata_dict4 = p4plot1(s_range=s_range, y_range=y_range, x_range=x_range, shape_yr_pi=shape_yr_pi, shape_yr_abr=shape_yr_abr, rawdata_dict=rawdata_dict2)

    rawdata_dict4['TR_sst'] =  THRESHOLD_sst
    rawdata_dict4['TR_sub'] =  THRESHOLD_sub

    savez(WD+C_dict['model_data']['modn']+'_r4(Jan)_(largestpiR2)_'+str(round(TR_sst, 2))+'K_'+'ud'+str(round(TR_sub*100, 2))+'_dats', model_data =  C_dict['model_data'],rawdata_dict = rawdata_dict4)
    
    #.. best fit save_4lrm command:
    # savez(WD+C_dict['model_data']['modn']+'_best(test5)fit_'+str(round(TR_sst, 2))+'K_'+ 'ud'+str(round(TR_sub*100, 2))+'_dats', model_data = C_dict['model_data'],rawdata_dict = rawdata_dict4)


    return None



