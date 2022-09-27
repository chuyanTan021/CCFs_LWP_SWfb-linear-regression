### This module is to get the model data we need from read func: 'get_LWPCMIP6', and calculate for CCFs and the required Cloud properties; 
## Crop regions, Transform the data to be annually mean, binned array form;
## Create the linear regression 2 & 4 regimes models from piControl sensitivity of cloud properties to the CCFs, then do the regressions on 'abrupt4xCO2' and save the data.

import netCDF4
from numpy import *
import matplotlib.pyplot as plt
import xarray as xr

# import PyNIO as Nio   #  deprecated
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

from get_LWPCMIP5data import *
from get_LWPCMIP6data import *
from fitLRM_cy1 import *
from fitLRM_cy2 import *
# from fitLRM_cy4 import *

from useful_func_cy import *
from calc_Radiation_LRM_1 import *
from calc_Radiation_LRM_2 import *



def calc_LRM_metrics(THRESHOLD_sst, THRESHOLD_sub, **model_data):
    # get variable data
    if model_data['cmip'] == 'cmip6':

        inputVar_pi, inputVar_abr = get_LWPCMIP6(**model_data)

    elif model_data['cmip'] == 'cmip5':
        
        inputVar_pi, inputVar_abr = get_LWPCMIP5(**model_data)
    else:
        print('not cmip6 & cmip5 data.')
    
    # ******************************* #
    # Radiation Change
    # coef_array_alpha_cre_pi, coef_array_albedo_pi, coef_array_alpha_cre_abr, coef_array_albedo_abr = calc_Radiation_LRM_1(inputVar_pi, inputVar_abr, TR_albedo = 0.25)
    coef_array_alpha_cre_pi, coef_array_albedo_pi, coef_array_alpha_cre_abr, coef_array_albedo_abr = calc_Radiation_LRM_2(inputVar_pi, inputVar_abr)
    
    # ******************************* #
    #..get the shapes of monthly data
    shape_lat = len(inputVar_pi['lat'])
    shape_lon = len(inputVar_pi['lon'])
    shape_time_pi = len(inputVar_pi['times'])
    shape_time_abr = len(inputVar_abr['times'])
    #print(shape_lat, shape_lon, shape_time_pi, shape_time_abr)

    
    #..choose lat 40 -85 °S as the Southern-Ocean Regions
    lons = inputVar_pi['lon'] *1.
    lats = inputVar_pi['lat'][:] *1.

    levels = array(inputVar_abr['pres'])
    times_abr = inputVar_abr['times'] *1.
    times_pi = inputVar_pi['times'] *1.
    
    lati0 = -40.
    latsi0= min(range(len(lats)), key = lambda i: abs(lats[i] - lati0))
    lati1 = -85.
    latsi1= min(range(len(lats)), key = lambda i: abs(lats[i] - lati1))
    print('lat index for 40.s; 85.s', latsi0, latsi1)

    shape_latSO =  (latsi0+1) - latsi1
    #print(shape_latSO)

    
    #..abrupt-4xCO2 Variables: LWP, tas(gmt), SST, (MC), p-e; SW radiation metrics
    LWP_abr = array(inputVar_abr['clwvi']) - array(inputVar_abr['clivi'])   #..units in kg m^-2

    gmt_abr = array(inputVar_abr['tas'])

    SST_abr = array(inputVar_abr['sfc_T'])
    
    Precip_abr = array(inputVar_abr['P']) * (24.*60.*60.)   #.. Precipitation. Convert the units from kg m^-2 s^-1 -> mm*day^-1
    print('abr4x average Pr(mm/ day): ', nanmean(Precip_abr))   #.. IPSL/abr2.80..  CNRM ESM2 1/abr 2.69.. CESM2/abr 2.74..
    lh_vaporization_abr = (2.501 - (2.361 * 10**-3) * (SST_abr - 273.15)) * 1e6  # the latent heat of vaporization at the surface Temperature
    # Eva_abr2 = array(inputVar_abr['E']) * (24. * 60 * 60)
    Eva_abr1 = array(inputVar_abr['E']) / lh_vaporization_abr * (24. * 60 * 60)  #.. Evaporation, mm day^-1
    print('abr4x average Evapor(mm/ day): ', nanmean(Eva_abr1))         #.. IPSL/abr2.50..  CNRM ESM2 1/abr 2.43.. CESM2/abr 2.43..
    MC_abr = Precip_abr - Eva_abr1   #..Moisture Convergence calculated from abrupt4xCO2's P - E, Units in mm day^-1
    
    Twp_abr = array(inputVar_abr['clwvi'])
    Iwp_abr = array(inputVar_abr['clivi'])

    # SW radiation metrics
    Rsdt_abr = array(inputVar_abr['rsdt'])
    Rsut_abr = array(inputVar_abr['rsut'])
    Rsutcs_abr = array(inputVar_abr['rsutcs'])
    print("shape of data in 'abrupt-4xCO2':  ",  Rsut_abr.shape, " mean 'abrupt-4xCO2' upwelling SW radiation flux in the SO (Assume with cloud): ",  nanmean(Rsut_abr[:, latsi1:latsi0 +1,:]))
    print("shape of data in 'abrupt-4XCO2' exp:", Eva_abr1.shape, 'abr4x mean-gmt(K): ', nanmean(gmt_abr))

    # albedo, albedo_clear sky, albedo_cre: all-sky - clear-sky
    Albedo_abr = Rsut_abr / Rsdt_abr
    Albedo_cs_abr = Rsutcs_abr / Rsdt_abr
    Alpha_cre_abr = Albedo_abr - Albedo_cs_abr

    if np.min(LWP_abr)<0:
        LWP_abr = Twp_abr
        print('clwvi mislabeled')
    
    #..piControl Variables: LWP, tas(gmt), SST, (MC), p-e ; SW radiation metrics (rsdt, rsut, rsutcs)
    LWP = array(inputVar_pi['clwvi']) - array(inputVar_pi['clivi'])   #..units in kg m^-2
    
    gmt = array(inputVar_pi['tas'])
    SST = array(inputVar_pi['sfc_T'])
    
    Precip = array(inputVar_pi['P'])* (24.*60.*60.)    #..Precipitation. Convert the units from kg m^-2 s^-1 -> mm*day^-1
    print('pi-C average Pr(mm/ day): ', nanmean(Precip))   #.. IPSL/piC 2.43..CNRM/piC 2.40.. CESM2/PIc 2.39
    lh_vaporization = (2.501 - (2.361 * 10**-3) * (SST - 273.15)) * 1e6  # the latent heat of vaporization at the surface Temperature
    Eva1 = array(inputVar_pi['E']) / lh_vaporization * (24. * 60 * 60)
    # Eva2 = array(inputVar_pi['E']) * (24.*60.*60.)   #..evaporation, mm day^-1
    
    print('pi-C average Evapor(mm/day): ', nanmean(Eva1))   #.. IPSL/piC  2.21..CNRM/piC 2.20.. CESM2/PIc 2.17..
    MC = Precip - Eva1   #..Moisture Convergence calculated from pi-Control's P - E, Units in mm day^-1
    
    Twp = array(inputVar_pi['clwvi'])
    Iwp = array(inputVar_pi['clivi'])

    
    # SW radiation metrics
    Rsdt_pi = array(inputVar_pi['rsdt'])
    Rsut_pi = array(inputVar_pi['rsut'])
    Rsutcs_pi = array(inputVar_pi['rsutcs'])
    print("shape of data in 'piControl':  ", Rsut_pi.shape, " mean 'piControl' upwelling SW radiation flux in the SO (Assume with cloud): "
, nanmean(Rsut_pi[:, latsi1:latsi0 +1,:]))
    print("shape of data in 'piControl' data: ", Eva1.shape, 'pi-C mean-gmt(K): ', nanmean(gmt))

    # albedo, albedo_clear sky; albedo(alpha)_cre: all-sky - clear-sky
    Albedo_pi = Rsut_pi / Rsdt_pi
    Albedo_cs_pi = Rsutcs_pi / Rsdt_pi
    Alpha_cre_pi = Albedo_pi - Albedo_cs_pi

    if np.min(LWP)<0:
        LWP = Twp
        print('clwvi mislabeled')

    #..abrupt-4xCO2
    # Lower Tropospheric Stability (LTS):
    k = 0.286

    theta_700_abr = array(inputVar_abr['T_700']) * (100000./70000.)**k
    theta_skin_abr = array(inputVar_abr['sfc_T']) * (100000./array(inputVar_abr['sfc_P']))**k 
    LTS_m_abr = theta_700_abr - theta_skin_abr

    #..Subtract the outliers in T_700 and LTS_m, 'nan' comes from missing T_700 data
    LTS_e_abr = ma.masked_where(theta_700_abr >= 500, LTS_m_abr)
    
    # Meteorology Subsidence at 500 hPa, units in Pa s^-1:
    Subsidence_abr = array(inputVar_abr['sub'])
    
    #.. piControl
    # Lower Tropospheric Stability (LTS):
    theta_700 = array(inputVar_pi['T_700']) * (100000./70000.)**k
    theta_skin = array(inputVar_pi['sfc_T']) * (100000./array(inputVar_pi['sfc_P']))**k
    LTS_m = theta_700 - theta_skin

    #..Subtract the outliers in T_700 and LTS_m 
    LTS_e = ma.masked_where(theta_700 >= 500, LTS_m)
    
    #..Meteological Subsidence  at 500 hPa, units in Pa s^-1:
    Subsidence = array(inputVar_pi['sub'])
    
    # define Dictionary to store: CCFs(4), gmt, other variables :
    dict0_PI_var = {'gmt': gmt, 'LWP': LWP, 'TWP': Twp, 'IWP': Iwp, 'SST': SST, 'p_e': MC, 'LTS': LTS_e, 'SUB': Subsidence, 'rsdt': Rsdt_pi, 'rsut': Rsut_pi, 'rsutcs': Rsutcs_pi, 'albedo' : Albedo_pi, 'albedo_cs': Albedo_cs_pi, 'alpha_cre': Alpha_cre_pi, 'lat': lats, 'lon': lons, 'times': times_pi, 'pres': levels}

    dict0_abr_var = {'gmt': gmt_abr, 'LWP': LWP_abr, 'TWP': Twp_abr, 'IWP': Iwp_abr, 'SST': SST_abr, 'p_e': MC_abr, 'LTS': LTS_e_abr ,'SUB': Subsidence_abr, 'rsdt': Rsdt_abr, 'rsut': Rsut_abr, 'rsutcs': Rsutcs_abr, 'albedo': Albedo_abr, 'albedo_cs': Albedo_cs_abr, 'alpha_cre': Alpha_cre_abr, 'lat': lats, 'lon': lons, 'times': times_abr, 'pres': levels}


    
    # get the Annual-mean, Southern-Ocean region arrays

    datavar_nas = ['LWP', 'TWP', 'IWP', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'alpha_cre', 'SST', 'p_e', 'LTS', 'SUB']   #..13 varisables except gmt (lon dimension diff)

    dict1_PI_yr = {}
    dict1_abr_yr = {}
    shape_yr_pi = shape_time_pi//12
    shape_yr_abr = shape_time_abr//12

    layover_yr_abr = zeros((len(datavar_nas), shape_yr_abr, shape_latSO, shape_lon))
    layover_yr_pi = zeros((len(datavar_nas), shape_yr_pi, shape_latSO, shape_lon))

    layover_yr_abr_gmt = zeros((shape_yr_abr, shape_lat, shape_lon))
    layover_yr_pi_gmt = zeros((shape_yr_pi, shape_lat, shape_lon))


    for a in range(len(datavar_nas)):

        # a_array = dict0_abr_var[datavar_nas[a]]

        for i in range(shape_time_abr//12):
            layover_yr_abr[a, i,:,:] = nanmean(dict0_abr_var[datavar_nas[a]][i*12:(i+1)*12, latsi1:latsi0 +1,:], axis=0)

        dict1_abr_yr[datavar_nas[a]+'_yr'] = layover_yr_abr[a,:]


        # b_array = dict0_PI_var[datavar_nas[a]]
        for j in range(shape_time_pi//12):
            layover_yr_pi[a, j,:,:] = nanmean(dict0_PI_var[datavar_nas[a]][j*12:(j+1)*12, latsi1:latsi0 +1,:], axis=0)

        dict1_PI_yr[datavar_nas[a]+'_yr'] = layover_yr_pi[a,:]
        print(datavar_nas[a])

    #print(dict1_PI_yr['LWP_yr'])
    
    # gmt
    for i in range(shape_time_abr//12):

        layover_yr_abr_gmt[i,:,:] = nanmean(dict0_abr_var['gmt'][i*12:(i+1)*12, :,:], axis=0)
    dict1_abr_yr['gmt_yr'] = layover_yr_abr_gmt
    
    for j in range(shape_time_pi//12):
        layover_yr_pi_gmt[j,:,:] = nanmean(dict0_PI_var['gmt'][j*12:(j+1)*12, :,:], axis=0)
    dict1_PI_yr['gmt_yr'] = layover_yr_pi_gmt

    # print(dict1_PI_yr['gmt_yr'])
    dict0_PI_var['dict1_yr'] = dict1_PI_yr
    dict0_abr_var['dict1_yr'] = dict1_abr_yr

    
    # Calculate 5*5 bin array for variables (LWP, CCFs) in Sounthern Ocean Region:
    #..set are-mean range and define function
    s_range = arange(-90., 90., 5.) + 2.5  #..global-region latitude edge: (36)
    x_range = arange(-180., 180., 5.)  #..logitude sequences edge: number: 72
    y_range = arange(-85, -40., 5.) +2.5  #..southern-ocaen latitude edge: 9
    
    # Annually variables in bin box:

    lat_array = lats[latsi1:latsi0+1] *1.
    lon_array = lons *1.
    lat_array1 = lats *1.

    dict1_PI_var = {}   #..add at Dec.30th, at 2021. Purpose: shrink the output savez data dictionary: rawdata
    dict1_abr_var = {}   #..add at Dec.30th, at 2021. Purpose: shrink the output savez data dictionary: rawdata
    dict1_yr_bin_PI = {}
    dict1_yr_bin_abr = {}

    for b in range(len(datavar_nas)):

        dict1_yr_bin_abr[datavar_nas[b]+'_yr_bin'] = binned_cySouthOcean5(dict1_abr_yr[datavar_nas[b]+'_yr'], lat_array, lon_array)
        dict1_yr_bin_PI[datavar_nas[b]+'_yr_bin'] = binned_cySouthOcean5(dict1_PI_yr[datavar_nas[b]+'_yr'], lat_array, lon_array)

    # print(dict1_yr_bin_abr['PRW_yr_bin'].shape)
    # print(dict1_yr_bin_abr['gmt_yr_bin'])  #..(150, 36, 73)
    # print(dict1_yr_bin_PI['SUB_yr_bin'].shape)  #..(100, 10, 73)
    dict1_yr_bin_abr['gmt_yr_bin'] = binned_cyGlobal5(dict1_abr_yr['gmt_yr'], lat_array1, lon_array)
    dict1_yr_bin_PI['gmt_yr_bin'] = binned_cyGlobal5(dict1_PI_yr['gmt_yr'], lat_array1, lon_array)
    print('gmt_yr_bin')

    dict1_abr_var['dict1_yr_bin_abr'] = dict1_yr_bin_abr
    dict1_PI_var['dict1_yr_bin_PI'] = dict1_yr_bin_PI
    
    # Monthly variables (same as above):
    dict1_mon_bin_PI = {}
    dict1_mon_bin_abr = {}

    for c in range(len(datavar_nas)):
        dict1_mon_bin_abr[datavar_nas[c]+'_mon_bin'] = binned_cySouthOcean5(dict0_abr_var[datavar_nas[c]][0:, latsi1:latsi0+1,:], lat_array, lon_array)
        dict1_mon_bin_PI[datavar_nas[c]+'_mon_bin'] = binned_cySouthOcean5(dict0_PI_var[datavar_nas[c]][0:, latsi1:latsi0+1,:], lat_array, lon_array)

    dict1_mon_bin_abr['gmt_mon_bin'] = binned_cyGlobal5(dict0_abr_var['gmt'][0:,:,:], lat_array1, lon_array)
    dict1_mon_bin_PI['gmt_mon_bin'] = binned_cyGlobal5(dict0_PI_var['gmt'][0:,:,:], lat_array1, lon_array)
    print("Every month monthly data")

    dict1_abr_var['dict1_mon_bin_abr'] = dict1_mon_bin_abr
    dict1_PI_var['dict1_mon_bin_PI'] = dict1_mon_bin_PI

    
    # input the shapes of year and month of pi&abr exper into the raw data dictionaries:
    dict1_abr_var['shape_yr'] = shape_yr_abr
    dict1_PI_var['shape_yr'] = shape_yr_pi

    dict1_abr_var['shape_mon'] = shape_time_abr
    dict1_PI_var['shape_mon'] = shape_time_pi
    
    # Output a dict for processing function in 'calc_LRM_metrics', stored the data dicts for PI and abr, with the model name_dict
    C_dict = {'dict1_PI_var': dict1_PI_var, 'dict1_abr_var': dict1_abr_var, 'model_data': model_data, 'coef_array_alpha_cre_pi': coef_array_alpha_cre_pi, 'coef_array_albedo_pi': coef_array_albedo_pi, 'coef_array_alpha_cre_abr': coef_array_alpha_cre_abr, 'coef_array_albedo_abr': coef_array_albedo_abr}  #..revised on June 23th, 2022.
    D_dict = deepcopy(C_dict)   # 'notice for the difference between shallow copy (object.copy()) and deep copy(copy.deepcopy(object))'
    B_dict = deepcopy(C_dict)


    ###..Put data into 'fitLRM' FUNCTION to get predicted LWP splitted by 'Tr_sst'/'Tr_sub' infos_models:
    TR_sst = THRESHOLD_sst   ###.. threshold skin T
    TR_sub = THRESHOLD_sub   ###.. threshold of 500 mb Subsidence
    WD = '/glade/scratch/chuyan/CMIP_output/CMIP_lrm_RESULT/'
    
    
    rawdata_dict1 = fitLRM3(C_dict = B_dict, TR_sst=TR_sst, s_range=s_range, y_range=y_range, x_range=x_range, lats=lats, lons=lons)
    rawdata_dict3 = p4plot1(s_range=s_range, y_range=y_range, x_range=x_range, Mean_training = rawdata_dict1['Mean_training'], Stdev_training = rawdata_dict1['Stdev_training'], shape_yr_pi=shape_yr_pi, shape_yr_abr=shape_yr_abr, rawdata_dict=rawdata_dict1)

    rawdata_dict3['TR_sst'] = THRESHOLD_sst

    savez(WD+C_dict['model_data']['modn']+'_r2r1_hotcold(Jan)_(largestpiR2)_Sep9th_Normalized_'+str(round(TR_sst, 2))+'_dats', model_data = C_dict['model_data'],rawdata_dict = rawdata_dict3)
    
    #.. best fit save_2lrm command:
    # savez(WD+C_dict['model_data']['modn']+'_r1r1_(Jan)_(largestpiR2)_Sep9th_Normalized_'+'0.0K'+'_dats', model_data = C_dict['model_data'], rawdata_dict = rawdata_dict3, Mean_LWP_training = rawdata_dict1['Mean_training'], Stdev_LWP_training= rawdata_dict1['Stdev_training'])
    
    
    rawdata_dict2 = fitLRM4(C_dict = D_dict, TR_sst=TR_sst, TR_sub=TR_sub, s_range=s_range, y_range=y_range, x_range=x_range, lats=lats, lons=lons)
    rawdata_dict4 = p4plot1(s_range=s_range, y_range=y_range, x_range=x_range,  Mean_training = rawdata_dict1['Mean_training'], Stdev_training = rawdata_dict1['Stdev_training'], shape_yr_pi=shape_yr_pi, shape_yr_abr=shape_yr_abr, rawdata_dict=rawdata_dict2)

    rawdata_dict4['TR_sst'] = THRESHOLD_sst
    rawdata_dict4['TR_sub'] = THRESHOLD_sub

    savez(WD+C_dict['model_data']['modn']+'_r4r1(Jan)_(largestpiR2)_Sep9th_Normalized_'+str(round(TR_sst, 2))+'K_'+'ud'+str(round(TR_sub*100, 2))+'_dats', model_data =  C_dict['model_data'], rawdata_dict = rawdata_dict4)
    
    #.. best fit save_4lrm command:
    # savez(WD+C_dict['model_data']['modn']+'_r2r1_updown(Jan)_(largestpiR2)_Sep9th_Normalized_'+ '0.0K_ud'+str(round(TR_sub*100, 2))+'_dats', model_data = C_dict['model_data'], rawdata_dict = rawdata_dict4, Mean_LWP_training = rawdata_dict2['Mean_training'], Stdev_LWP_training= rawdata_dict2['Stdev_training'])
    

    return None
    
