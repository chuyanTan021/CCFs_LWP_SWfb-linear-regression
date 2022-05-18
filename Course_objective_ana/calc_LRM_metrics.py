# get the data we need from read func: 'get_LWPCMIP6', and do some data-processing for building the linear regression CCFs_Clouds models;ss
# transform data to annual-mean/ monthly-mean bin array or flattened array;
# fit the regression model 1&2 from pi-Control CCFs' sensitivities to the LWP, then do the regressions and save the data:

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
# from get_annual_so import *
# from fitLRM_cy import *
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
    LWP_abr  = array(inputVar_abr['clwvi']) - array(inputVar_abr['clivi'])   #..units in kg m^-2
    
    gmt_abr  = asarray(inputVar_abr['tas'])
    
    SST_abr  = asarray(inputVar_abr['sfc_T'])
    
    
    Precip_abr =  asarray(inputVar_abr['P']) * (24.*60.*60.)   #..Precipitation. Convert the units from kg m^-2 s^-1 -> mm*day^-1
    print('abr4x average Pr(mm/ day): ', nanmean(Precip_abr))   #.. IPSL/abr2.80..  CNRM ESM2 1/abr 2.69.. CESM2/abr 2.74..
    Eva_abr    =  asarray(inputVar_abr['E']) * (24.*60.*60.)   #..evaporation, mm day^-1
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
    shape_yr_pi  = shape_time_pi//12
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
    x_range  = arange(-180., 180., 5.)   #..logitude sequences edge: number:73
    s_range  = arange(-90., 90, 5.) + 2.5   #..global-region latitude edge:(36)

    y_range  = arange(-85, -40., 5.) +2.5   #..southern-ocaen latitude edge:10

    
    # Annually variables in bin box:

    lat_array  = lats[latsi1:latsi0+1]
    lon_array  = lons
    lat_array1 =  lats
    dict1_PI_var   =  {}             #..add at Dec.30th, at 2021. Purpose: shrink the output savez data dictionary: rawdata
    dict1_abr_var  =   {}            #..add at Dec.30th, at 2021. Purpose: shrink the output savez data dictionary: rawdata
    dict1_yr_bin_PI  = {}
    dict1_yr_bin_abr = {}
    
    for b in range(len(datavar_nas)):

        dict1_yr_bin_abr[datavar_nas[b]+'_yr_bin']  =   binned_cySouthOcean5(dict1_abr_yr[datavar_nas[b]+'_yr'], lat_array, lon_array)
        dict1_yr_bin_PI[datavar_nas[b]+'_yr_bin']   =  binned_cySouthOcean5(dict1_PI_yr[datavar_nas[b]+'_yr'], lat_array, lon_array)


    #print(dict1_yr_bin_abr['PRW_yr_bin'].shape)
    #print(dict1_yr_bin_abr['gmt_yr_bin'])   #..(150, 36, 73)
    #print(dict1_yr_bin_PI['SUB_yr_bin'].shape)   #..(100, 10, 73)
    dict1_yr_bin_abr['gmt_yr_bin']   =  binned_cyGlobal5(dict1_abr_yr['gmt_yr'], lat_array1, lon_array)
    dict1_yr_bin_PI['gmt_yr_bin']   =  binned_cyGlobal5(dict1_PI_yr['gmt_yr'], lat_array1, lon_array)

    print('gmt_yr_bin')
    
    dict1_abr_var['dict1_yr_bin_abr']  =  dict1_yr_bin_abr
    dict1_PI_var['dict1_yr_bin_PI']  = dict1_yr_bin_PI


    # Monthly variables (same as above):
    dict1_mon_bin_PI  = {}
    dict1_mon_bin_abr = {}
    
    for c in range(len(datavar_nas)):

        dict1_mon_bin_abr[datavar_nas[c]+'_mon_bin']  =   binned_cySouthOcean5(dict0_abr_var[datavar_nas[c]][:, latsi1:latsi0 +1,:], lat_array, lon_array)
        dict1_mon_bin_PI[datavar_nas[c]+'_mon_bin']   =  binned_cySouthOcean5(dict0_PI_var[datavar_nas[c]][:, latsi1:latsi0 +1,:], lat_array, lon_array)

    dict1_mon_bin_abr['gmt_mon_bin']   =  binned_cyGlobal5(dict0_abr_var['gmt'], lat_array1, lon_array)
    dict1_mon_bin_PI['gmt_mon_bin']  =  binned_cyGlobal5(dict0_PI_var['gmt'], lat_array1, lon_array)

    print('gmt_mon_bin')
    
    dict1_abr_var['dict1_mon_bin_abr']  = dict1_mon_bin_abr
    dict1_PI_var['dict1_mon_bin_PI']  = dict1_mon_bin_PI


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


    rawdata_dict1 = fitLRM( TR_sst=TR_sst, s_range=s_range, y_range=y_range, x_range=x_range, C_dict = B_dict)
    rawdata_dict3 = p4plot1(s_range=s_range, y_range=y_range, x_range=x_range, shape_yr_pi=shape_yr_pi, shape_yr_abr=shape_yr_abr, rawdata_dict=rawdata_dict1)

    rawdata_dict3['TR_sst'] =  THRESHOLD_sst

    savez(WD+C_dict['model_data']['modn']+'_swrpredi(largestpiR2)_'+str(round(TR_sst, 2))+'_dats', model_data = C_dict['model_data'],rawdata_dict = rawdata_dict3)
    #.. best fit save_2lrm command:
    # savez(WD+C_dict['model_data']['modn']+'_best(test5)fit_'+str(round(TR_sst, 2))+'_dats', model_data = C_dict['model_data'],rawdata_dict = rawdata_dict3)

    rawdata_dict2 = fitLRM2(TR_sst=TR_sst, TR_sub=TR_sub, s_range=s_range, y_range=y_range, x_range=x_range, C_dict = D_dict)
    rawdata_dict4 = p4plot1(s_range=s_range, y_range=y_range, x_range=x_range, shape_yr_pi=shape_yr_pi, shape_yr_abr=shape_yr_abr, rawdata_dict=rawdata_dict2)

    rawdata_dict4['TR_sst'] =  THRESHOLD_sst
    rawdata_dict4['TR_sub'] =  THRESHOLD_sub

    savez(WD+C_dict['model_data']['modn']+'_swrpredi(largestpiR2)_'+str(round(TR_sst, 2))+'K_'+'ud'+str(round(TR_sub*100, 2))+'_dats', model_data =  C_dict['model_data'],rawdata_dict = rawdata_dict4)
    #.. best fit save_4lrm command:
    # savez(WD+C_dict['model_data']['modn']+'_best(test5)fit_'+str(round(TR_sst, 2))+'K_'+ 'ud'+str(round(TR_sub*100, 2))+'_dats', model_data = C_dict['model_data'],rawdata_dict = rawdata_dict4)


    return None




def fitLRM(C_dict, TR_sst, s_range, y_range, x_range):
    # 'C_dict' is the raw data dict, 'TR_sst' is the pre-defined skin_Temperature Threshold to distinguish two Multi-Linear Regression Models

    # 's_range , 'y_range', 'x_range' used to do area mean for repeat gmt ARRAY

    dict0_abr_var = C_dict['dict1_abr_var']
    dict0_PI_var  = C_dict['dict1_PI_var']
    #print(dict0_PI_var['times'])

    model = C_dict['model_data']   #.. type in dict

    datavar_nas = ['LWP', 'TWP', 'IWP', 'PRW', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'SST', 'p_e', 'LTS', 'SUB']   #..13 varisables except gmt (lon dimension  diff)

    # load annually-mean bin data.
    dict1_yr_bin_PI  = dict0_PI_var['dict1_yr_bin_PI']
    dict1_yr_bin_abr  = dict0_abr_var['dict1_yr_bin_abr']
    #print(dict1_yr_bin_PI['LWP_yr_bin'].shape)
    
    # load monthly bin data.
    dict1_mon_bin_PI  = dict0_PI_var['dict1_mon_bin_PI']
    dict1_mon_bin_abr  = dict0_abr_var['dict1_mon_bin_abr']

    # data array in which shapes?
    shape_yr_PI_3 = dict1_yr_bin_PI['LWP_yr_bin'].shape
    shape_yr_abr_3 = dict1_yr_bin_abr['LWP_yr_bin'].shape

    shape_yr_PI_gmt = dict1_yr_bin_PI['gmt_yr_bin'].shape
    shape_yr_abr_gmt = dict1_yr_bin_abr['gmt_yr_bin'].shape

    shape_mon_PI = dict1_mon_bin_PI['LWP_mon_bin'].shape
    shape_mon_abr = dict1_mon_bin_abr['LWP_mon_bin'].shape

    shape_mon_PI_gmt = dict1_mon_bin_PI['gmt_mon_bin'].shape
    shape_mon_abr_gmt = dict1_mon_bin_abr['gmt_mon_bin'].shape

    #.. archieve the 'shape' infos
    C_dict['shape_yr_PI_3']  = shape_yr_PI_3
    C_dict['shape_yr_abr_3']  = shape_yr_abr_3
    C_dict['shape_yr_PI_gmt_3']  = shape_yr_PI_gmt
    C_dict['shape_yr_abr_gmt_3']  = shape_yr_abr_gmt

    C_dict['shape_mon_PI_3']  = shape_mon_PI
    C_dict['shape_mon_abr_3']  = shape_mon_abr
    C_dict['shape_mon_PI_gmt_3']  = shape_mon_PI_gmt
    C_dict['shape_mon_abr_gmt_3']  = shape_mon_abr_gmt


    dict2_predi_fla_PI = {}
    dict2_predi_fla_abr = {}

    dict2_predi_nor_PI = {}
    dict2_predi_nor_abr = {}

    #.. Ravel binned array /Standardized data ARRAY :
    for d in range(len(datavar_nas)):
        dict2_predi_fla_PI[datavar_nas[d]] = dict1_mon_bin_PI[datavar_nas[d]+'_mon_bin'].flatten()
        dict2_predi_fla_abr[datavar_nas[d]] = dict1_mon_bin_abr[datavar_nas[d]+'_mon_bin'].flatten()

        # normalized the predict array
        dict2_predi_nor_PI[datavar_nas[d]] =  (dict2_predi_fla_PI[datavar_nas[d]] - nanmean(dict2_predi_fla_PI[datavar_nas[d]]) )/ nanstd(dict2_predi_fla_PI[datavar_nas[d]])
        dict2_predi_nor_abr[datavar_nas[d]] =  (dict2_predi_fla_abr[datavar_nas[d]] - nanmean(dict2_predi_fla_abr[datavar_nas[d]]) )/ nanstd(dict2_predi_fla_abr[datavar_nas[d]])

    #..Use area_mean method, 'np.repeat' and 'np.tile' to reproduce gmt area-mean Array as the same shape as other flattened variables
    GMT_pi_mon  = area_mean(dict1_mon_bin_PI['gmt_mon_bin'],  s_range,  x_range)   #..ALL in shape : shape_yr_abr(single dimension)
    ## dict2_predi_fla_PI['gmt']  = GMT_pi.repeat(730)   # something wrong when calc dX_dTg(dCCFS_dgmt)
    GMT_abr_mon  = area_mean(dict1_mon_bin_abr['gmt_mon_bin'], s_range, x_range)   #..ALL in shape : shape_yr_abr(single dimension)
    ## dict2_predi_fla_abr['gmt'] = GMT_abr.repeat(730)
    
    # Use the southernOCEAN value as the gmt variable
    dict2_predi_fla_PI['gmt'] = dict1_mon_bin_PI['gmt_mon_bin'][:,1:11,:].flatten()
    dict2_predi_fla_abr['gmt'] = dict1_mon_bin_abr['gmt_mon_bin'][:,1:11,:].flatten()

    dict2_predi_nor_PI['gmt'] = (dict2_predi_fla_PI['gmt'] - nanmean(dict2_predi_fla_PI['gmt']) )/ nanstd(dict2_predi_fla_PI['gmt'])
    dict2_predi_nor_abr['gmt'] = (dict2_predi_fla_abr['gmt'] - nanmean(dict2_predi_fla_abr['gmt']) )/ nanstd(dict2_predi_fla_abr['gmt'])
    
    
    # save into rawdata_dict:
    C_dict['dict2_predi_fla_PI'] =  dict2_predi_fla_PI
    C_dict['dict2_predi_fla_abr'] = dict2_predi_fla_abr
    C_dict['dict2_predi_nor_PI'] =  dict2_predi_nor_PI
    C_dict['dict2_predi_nor_abr']  = dict2_predi_nor_abr
    C_dict['GMT_pi_mon']  = GMT_pi_mon
    C_dict['GMT_abr_mon'] = GMT_abr_mon

    #.. Training Module (2lrm)
    

    #.. PI
    
    predict_dict_PI, ind6_PI, ind7_PI, coef_array, shape_fla_training = rdlrm_2_training(dict2_predi_fla_PI, TR_sst, predictant='LWP')
    predict_dict_PI_iwp, ind6_PI_iwp, ind7_PI_iwp, coef_array_iwp, shape_fla_training_iwp = rdlrm_2_training(dict2_predi_fla_PI, TR_sst, predictant='IWP')
    
    predict_dict_PI_albedo, _, _, coef_array_albedo = rdlrm_2_training(dict2_predi_fla_PI, TR_sst, predictant='albedo', predictor=['LWP'], r = 2)[0:4]
    predict_dict_PI_rsut, _, _, coef_array_rsut = rdlrm_2_training(dict2_predi_fla_PI, TR_sst, predictant='rsut', predictor=['LWP'], r = 2)[0:4]
    
    # Added on May 13th, 2022: for second step using LWP to predict the albedo
    dict2_predi_fla_PI['LWP_lrm'] = deepcopy(predict_dict_PI['value'])
    dict2_predi_nor_PI['LWP_lrm'] = (dict2_predi_fla_PI['LWP_lrm'] - nanmean(dict2_predi_fla_PI['LWP_lrm']) )/ nanstd(dict2_predi_fla_PI['LWP_lrm'])
    predict_dict_PI_albedo_lL, _, _, coef_array_albedo_lL = rdlrm_2_training(dict2_predi_fla_PI, TR_sst, predictant='albedo', predictor=['LWP_lrm'], r = 2)[0:4]
    predict_dict_PI_rsut_lL, _, _, coef_array_rsut_lL = rdlrm_2_training(dict2_predi_fla_PI, TR_sst, predictant='rsut', predictor=['LWP_lrm'], r = 2)[0:4]


    # Save into the rawdata dict
    C_dict['Coef_dict'] = coef_array
    C_dict['Predict_dict_PI']  = predict_dict_PI
    C_dict['ind_Hot_PI'] = ind6_PI
    C_dict['ind_Cold_PI'] = ind7_PI
    
    C_dict['Coef_dict_IWP']= coef_array_iwp
    C_dict['Predict_dict_PI_IWP']  = predict_dict_PI_iwp
    # C_dict['ind_Hot_PI_IWP'] = ind6_PI_iwp
    # C_dict['ind_Cold_PI_IWP'] = ind7_PI_iwp
    
    
    # Albedo and radiation 
    C_dict['Coef_dict_albedo'] = coef_array_albedo
    C_dict['Predict_dict_PI_albedo'] = predict_dict_PI_albedo
    
    C_dict['Coef_dict_rsut'] = coef_array_rsut
    C_dict['Predict_dict_PI_rsut'] = predict_dict_PI_rsut
    
    C_dict['Coef_dict_albedo_lL'] = coef_array_albedo_lL
    C_dict['Predict_dict_PI_albedo_lL'] = predict_dict_PI_albedo_lL
    
    C_dict['Coef_dict_rsut_lL'] = coef_array_rsut_lL
    C_dict['Predict_dict_PI_rsut_lL'] = predict_dict_PI_rsut_lL


    # 'YB' is the predicted value of LWP in 'piControl' experiment
    YB = predict_dict_PI['value']
    # print("2lrm predicted mean LWP: ", nanmean(YB), " in 'piControl' ")

    YB_iwp = predict_dict_PI_iwp['value']
    # print("2lrm predicted mean IWP: ", nanmean(YB_iwp), " in 'piControl' ")
    
    YB_albedo = predict_dict_PI_albedo['value']
    # print("2lrm predicted mean Albedo (with cloud): ", nanmean(YB_albedo), " in 'piControl' ")
    YB_rsut = predict_dict_PI_rsut['value']

    YB_albedo_lL = predict_dict_PI_albedo_lL['value']
    print("2lrm predicted mean Albedo (with cloud) using lrm LWP:", nanmean(YB_albedo_lL), " in 'piControl' ")
    YB_rsut_lL = predict_dict_PI_rsut_lL['value']


    # Save 'YB', and resampled into the shape of 'LWP_yr_bin':
    
    C_dict['LWP_predi_bin_PI'] = asarray(YB).reshape(shape_mon_PI)
    C_dict['IWP_predi_bin_PI'] = asarray(YB_iwp).reshape(shape_mon_PI) 
    C_dict['albedo_predi_bin_PI'] = asarray(YB_albedo).reshape(shape_mon_PI)
    C_dict['rsut_predi_bin_PI'] = asarray(YB_rsut).reshape(shape_mon_PI)

    C_dict['albedo_lL_predi_bin_PI'] = asarray(YB_albedo_lL).reshape(shape_mon_PI)
    C_dict['rsut_lL_predi_bin_PI'] = asarray(YB_rsut_lL).reshape(shape_mon_PI)


    #.. Test performance
    
    stats_dict_PI = Test_performance_2(dict2_predi_fla_PI['LWP'], YB, ind6_PI, ind7_PI)
    stats_dict_PI_iwp = Test_performance_2(dict2_predi_fla_PI['IWP'], YB_iwp, ind6_PI_iwp, ind7_PI_iwp)
    stats_dict_PI_albedo = Test_performance_2(dict2_predi_fla_PI['albedo'], YB_albedo, ind6_PI, ind7_PI)
    stats_dict_PI_rsut = Test_performance_2(dict2_predi_fla_PI['rsut'], YB_rsut, ind6_PI, ind7_PI)
    
    stats_dict_PI_albedo_lL = Test_performance_2(dict2_predi_fla_PI['albedo'], YB_albedo_lL, ind6_PI, ind7_PI)
    stats_dict_PI_rsut_lL = Test_performance_2(dict2_predi_fla_PI['rsut'], YB_rsut_lL, ind6_PI, ind7_PI)
    print("Mean of report & predicted albedo in 'piControl' for SST>=TR_sst (ind6):", nanmean(dict2_predi_fla_PI['albedo'][ind6_PI]), '& ', nanmean(YB_albedo_lL[ind6_PI]))
    print("Mean of report & predicted albedo in 'piControl' for SST< TR_sst (ind7):" , nanmean(dict2_predi_fla_PI['albedo'][ind7_PI]), '& ',  nanmean(YB_albedo_lL[ind7_PI]))

    # #########################################################################
    #.. ABR
    
    #.. Predicting module (2lrm)

    predict_dict_abr, ind6_abr, ind7_abr, shape_fla_testing = rdlrm_2_predict(dict2_predi_fla_abr, coef_array, TR_sst, predictant = 'LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 2)
    predict_dict_abr_iwp, ind6_abr_iwp, ind7_abr_iwp, shape_fla_testing_iwp = rdlrm_2_predict(dict2_predi_fla_abr, coef_array, TR_sst, predictant = 'IWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 2)
    
    predict_dict_abr_albedo = rdlrm_2_predict(dict2_predi_fla_abr, coef_array_albedo, TR_sst, predictant = 'albedo', predictor = ['LWP'], r = 2)[0]
    predict_dict_abr_rsut = rdlrm_2_predict(dict2_predi_fla_abr, coef_array_rsut, TR_sst, predictant = 'rsut', predictor= ['LWP'], r = 2)[0]
    
    # Added on May 13th, 2022: for second step using LWP to predict the albedo
    dict2_predi_fla_abr['LWP_lrm'] = deepcopy(predict_dict_abr['value'])
    dict2_predi_nor_abr['LWP_lrm'] = (dict2_predi_fla_abr['LWP_lrm'] - nanmean(dict2_predi_fla_abr['LWP_lrm']) )/ nanstd(dict2_predi_fla_abr['LWP_lrm'])
    predict_dict_abr_albedo_lL = rdlrm_2_predict(dict2_predi_fla_abr, coef_array_albedo, TR_sst, predictant='albedo', predictor=['LWP_lrm'], r = 2)[0]
    predict_dict_abr_rsut_lL = rdlrm_2_predict(dict2_predi_fla_abr, coef_array_rsut, TR_sst, predictant='rsut', predictor=['LWP_lrm'], r = 2)[0]


    # Save into the rawdata dict

    C_dict['Predict_dict_abr']  = predict_dict_abr
    C_dict['ind_Hot_abr'] = ind6_abr
    C_dict['ind_Cold_abr'] = ind7_abr
    
    C_dict['Predict_dict_abr_IWP']  = predict_dict_abr_iwp
    C_dict['ind_Hot_abr_IWP'] = ind6_abr_iwp
    C_dict['ind_Cold_abr_IWP'] = ind7_abr_iwp
    
    C_dict['Predict_dict_abr_albedo'] = predict_dict_abr_albedo
    C_dict['Predict_dict_abr_rsut'] = predict_dict_abr_rsut
    
    C_dict['Predict_dict_abr_albedo_lL'] = predict_dict_abr_albedo_lL
    C_dict['Predict_dict_abr_rsut_lL'] = predict_dict_abr_rsut_lL


    # 'YB_abr' is the predicted value of LWP in 'abrupt-4xCO2' experiment
    YB_abr = predict_dict_abr['value']
    # print("2lrm predicted mean LWP ", nanmean(YB_abr), " in 'abrupt-4xCO2' ")

    YB_abr_iwp = predict_dict_abr_iwp['value']
    # print("2lrm predicted mean IWP ", nanmean(YB_abr_iwp), " in 'abrupt-4xCO2' ")
    
    YB_abr_albedo = predict_dict_abr_albedo['value']
    # print("2lrm predicted mean Albedo (with cloud)", nanmean(YB_abr_albedo), " in 'abrupt-4xCO2' ")
    YB_abr_rsut = predict_dict_abr_rsut['value']
    
    YB_abr_albedo_lL = predict_dict_abr_albedo_lL['value']
    print("2lrm predicted mean Albedo (with cloud) using lrm's LWP", nanmean(YB_abr_albedo_lL), " in 'abrupt-4xCO2' ")
    YB_abr_rsut_lL = predict_dict_abr_rsut_lL['value']
    
    print(" Mean report & predicted albedo_lL for 'abrupt-4xCO2' (all):", nanmean(dict2_predi_fla_abr['albedo']), '& ', nanmean(YB_abr_albedo_lL))
    print(" Mean report & predicted albedo for 'abrupt-4xCO2' (all):", nanmean(dict2_predi_fla_abr['albedo']), '& ', nanmean(YB_abr_albedo))



    # Save 'YB_abr', reshapled into the shape of 'LWP_yr_bin_abr':
    C_dict['LWP_predi_bin_abr'] =  asarray(YB_abr).reshape(shape_mon_abr)
    C_dict['IWP_predi_bin_abr'] =  asarray(YB_abr_iwp).reshape(shape_mon_abr)
    C_dict['albedo_predi_bin_abr'] = asarray(YB_abr_albedo).reshape(shape_mon_abr)
    C_dict['rsut_predi_bin_abr'] = asarray(YB_abr_rsut).reshape(shape_mon_abr)
    
    C_dict['albedo_lL_predi_bin_abr'] = asarray(YB_abr_albedo_lL).reshape(shape_mon_abr)
    C_dict['rsut_lL_predi_bin_abr'] = asarray(YB_abr_rsut_lL).reshape(shape_mon_abr)

    
    # Test performance for abrupt-4xCO2 (testing) data set
    
    stats_dict_abr = Test_performance_2(dict2_predi_fla_abr['LWP'], YB_abr, ind6_abr, ind7_abr)
    stats_dict_abr_iwp = Test_performance_2(dict2_predi_fla_abr['IWP'], YB_abr_iwp, ind6_abr_iwp, ind7_abr_iwp)
    stats_dict_abr_albedo = Test_performance_2(dict2_predi_fla_abr['albedo'], YB_abr_albedo, ind6_abr, ind7_abr)
    stats_dict_abr_rsut = Test_performance_2(dict2_predi_fla_abr['rsut'], YB_abr_rsut, ind6_abr, ind7_abr)

    stats_dict_abr_albedo_lL = Test_performance_2(dict2_predi_fla_abr['albedo'], YB_abr_albedo_lL, ind6_abr, ind7_abr)
    stats_dict_abr_rsut_lL = Test_performance_2(dict2_predi_fla_abr['rsut'], YB_abr_rsut_lL, ind6_abr, ind7_abr)
    
    
    '''
    # calc D(CCFs) to DGMT and save into 'Dx/DtG' ARRAY
    regr3 = linear_model.LinearRegression()
    re_LWP= regr3.fit(dict2_predi_fla_abr['gmt'][logical_or(ind7, ind6)].reshape(-1,1), dict2_predi_fla_abr['LWP'][logical_or(ind7, ind6)])
    print('d(LWP)/d(gmt)| (has LTS VALUES) = ', re_LWP.coef_)
    print('b of D(LWP) /D(gmt) : ', re_LWP.intercept_)
    regr4 = linear_model.LinearRegression()
    re_IWP= regr4.fit(dict2_predi_fla_abr['gmt'][logical_or(ind7, ind6)].reshape(-1,1), dict2_predi_fla_abr['IWP'][logical_or(ind7, ind6)])
    regr5 = linear_model.LinearRegression()
    regr6 = linear_model.LinearRegression()
    regr7 = linear_model.LinearRegression()
    regr8 = linear_model.LinearRegression()
    re_SST = regr5.fit(dict2_predi_fla_abr['gmt'][logical_or(ind7, ind6)].reshape(-1,1), dict2_predi_fla_abr['SST'][logical_or(ind7, ind6)])
    re_p_e = regr6.fit(dict2_predi_fla_abr['gmt'][logical_or(ind7, ind6)].reshape(-1,1), dict2_predi_fla_abr['p_e'][logical_or(ind7, ind6)])
    re_LTS = regr7.fit(dict2_predi_fla_abr['gmt'][logical_or(ind7, ind6)].reshape(-1,1), dict2_predi_fla_abr['LTS'][logical_or(ind7, ind6)])
    re_SUB = regr8.fit(dict2_predi_fla_abr['gmt'][logical_or(ind7, ind6)].reshape(-1,1), dict2_predi_fla_abr['SUB'][logical_or(ind7, ind6)])
    print('d(CCFs)/d(gmt)| (has LTS VALUES)= ', re_SST.coef_, re_p_e.coef_, re_LTS.coef_,  re_SUB.coef_)
    #..save into rawdata_dict:
    Dx_DtG =[re_LWP.coef_, re_IWP.coef_, re_SST.coef_,  re_p_e.coef_,  re_LTS.coef_,  re_SUB.coef_]
    C_dict['dX_dTg'] =  Dx_DtG
    '''

    #.. save test performance metrics into rawdata_dict

    C_dict['stats_dict_PI'] = stats_dict_PI
    C_dict['stats_dict_PI_iwp'] = stats_dict_PI_iwp
    
    C_dict['stats_dict_abr'] = stats_dict_abr
    C_dict['stats_dict_abr_iwp'] = stats_dict_abr_iwp

    C_dict['stats_dict_PI_albedo'] = stats_dict_PI_albedo
    C_dict['stats_dict_abr_albedo'] = stats_dict_abr_albedo

    C_dict['stats_dict_PI_rsut'] = stats_dict_PI_rsut
    C_dict['stats_dict_abr_rsut'] = stats_dict_abr_rsut

    C_dict['stats_dict_PI_albedo_lL'] = stats_dict_PI_albedo_lL
    C_dict['stats_dict_abr_albedo_lL'] = stats_dict_abr_albedo_lL


    C_dict['stats_dict_PI_rsut_lL'] = stats_dict_PI_rsut_lL
    C_dict['stats_dict_abr_rsut_lL'] = stats_dict_abr_rsut_lL
    
    
    return C_dict



def p4plot1(s_range, y_range, x_range, shape_yr_pi, shape_yr_abr, rawdata_dict):

    ### 's_range , 'y_range', 'x_range' used to do area mean for repeat gmt ARRAY

    # retriving datas from big dict...
    dict0_abr_var = rawdata_dict['dict1_abr_var']
    dict0_PI_var  = rawdata_dict['dict1_PI_var']
    
    shape_yr_PI_3  = rawdata_dict['shape_yr_PI_3']
    shape_yr_abr_3  = rawdata_dict['shape_yr_abr_3']
    shape_mon_PI_3  = rawdata_dict['shape_mon_PI_3']
    shape_mon_abr_3  = rawdata_dict['shape_mon_abr_3']
    
    model = rawdata_dict['model_data']   #.. type in dict


    datavar_nas = ['LWP', 'TWP', 'IWP', 'PRW', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'SST', 'p_e', 'LTS', 'SUB']   #..13 varisables except gmt (lon dimension diff)

    # load annually-mean bin data:
    dict1_yr_bin_PI  = dict0_PI_var['dict1_yr_bin_PI']
    dict1_yr_bin_abr  = dict0_abr_var['dict1_yr_bin_abr']

    # load monthly bin data:
    dict1_mon_bin_PI = dict0_PI_var['dict1_mon_bin_PI']
    dict1_mon_bin_abr= dict0_abr_var['dict1_mon_bin_abr']

    # Calc area-mean ARRAY for annually variables on 'abr' /'pi' exp:
    areamean_dict_PI = {}
    areamean_dict_abr  = {}
    
    for e in range(len(datavar_nas)):
    
        #  "monthly" convert to "annually":
        areamean_dict_PI[datavar_nas[e]+ '_yr_bin'] =  get_annually_metric(dict1_mon_bin_PI[datavar_nas[e]+ '_mon_bin'], shape_mon_PI_3[0],  shape_mon_PI_3[1], shape_mon_PI_3[2])                   
        areamean_dict_abr[datavar_nas[e]+ '_yr_bin'] =  get_annually_metric(dict1_mon_bin_abr[datavar_nas[e]+ '_mon_bin'], shape_mon_abr_3[0],  shape_mon_abr_3[1], shape_mon_abr_3[2])
        
        #  "yr_bin"  area_meaned to 'shape_yr_':
        areamean_dict_PI[datavar_nas[e]+ '_area_yr'] =  area_mean(areamean_dict_PI[datavar_nas[e]+ '_yr_bin'], y_range, x_range)
        areamean_dict_abr[datavar_nas[e]+ '_area_yr'] =  area_mean(areamean_dict_abr[datavar_nas[e]+ '_yr_bin'], y_range, x_range)

    areamean_dict_PI['gmt_area_yr']  =  area_mean(dict1_yr_bin_PI['gmt_yr_bin'], s_range, x_range)
    areamean_dict_abr['gmt_area_yr']  =  area_mean(dict1_yr_bin_abr['gmt_yr_bin'], s_range, x_range)
    
    # Calc annually mean predicted LWP, IWP, and SW radiation metrics
    
    ########### Annually predicted data:
    # areamean_dict_predi['LWP_area_yr_pi']  =   area_mean(rawdata_dict['LWP_predi_bin_PI'], y_range, x_range)
    # areamean_dict_predi['LWP_area_yr_abr']  =   area_mean(rawdata_dict['LWP_predi_bin_abr'], y_range, x_range)
    ############ end yr

    ########### Monthly predicted data:
    
    areamean_dict_predi =  {}
    datapredi_nas = ['LWP', 'IWP', 'albedo', 'rsut', 'albedo_lL', 'rsut_lL']
    datarepo_nas = ['LWP', 'IWP', 'albedo', 'rsut']
    
    for f in range(len(datapredi_nas)):
        areamean_dict_predi[datapredi_nas[f]+'_predi_yr_bin_pi'] =  get_annually_metric(rawdata_dict[datapredi_nas[f]+'_predi_bin_PI'], shape_mon_PI_3[0], shape_mon_PI_3[1], shape_mon_PI_3[2] )
        
        areamean_dict_predi[datapredi_nas[f]+'_predi_yr_bin_abr'] =  get_annually_metric(rawdata_dict[datapredi_nas[f]+'_predi_bin_abr'], shape_mon_abr_3[0], shape_mon_abr_3[1], shape_mon_abr_3[2] )
    

    ###  Calc area_mean of predicted LWP, IWP and SW radiation metrics
    
    for g in range(len(datapredi_nas)):

        areamean_dict_predi[datapredi_nas[g]+'_area_yr_pi'] = area_mean(areamean_dict_predi[datapredi_nas[g]+'_predi_yr_bin_pi'],  y_range, x_range)
        areamean_dict_predi[datapredi_nas[g]+'_area_yr_abr'] = area_mean(areamean_dict_predi[datapredi_nas[g]+'_predi_yr_bin_abr'],  y_range, x_range)

    
    # print("Annually area_mean predicted  albedo (with cloud) in 'piControl' run: ",areamean_dict_predi['albedo_lL_area_yr_pi'], r'$w m^{-2}$')  # r'$ kg m^{-2}$'
    # print("Annually area_mean predicted  albedo (with cloud) in 'abrupt-4xCO2' run: ",areamean_dict_predi['albedo_lL_area_yr_abr'], r'$ w m^{-2}$')
    
    ############# end mon
    

    # Store the annually report & predicted metrics
    
    rawdata_dict['areamean_dict_predi'] =  areamean_dict_predi
    rawdata_dict['areamean_dict_abr'] = areamean_dict_abr
    rawdata_dict['areamean_dict_PI'] = areamean_dict_PI


    # Generate continous annually-mean array are convenient for plotting LWP changes:
    #..Years from 'piControl' to 'abrupt-4xCO2' experiment, which are choosed years
    Yrs =  arange(shape_yr_pi+shape_yr_abr)

    # global-mean surface air temperature, from 'piControl' to 'abrupt-4xCO2' experiment:
    
    GMT =  full((shape_yr_pi + shape_yr_abr),  0.0)
    GMT[0:shape_yr_pi] = areamean_dict_PI['gmt_area_yr']
    GMT[shape_yr_pi:] = areamean_dict_abr['gmt_area_yr']

    predict_metrics_annually = {}
    report_metrics_annually = {}
    
    # predicted values, from 'piControl' to 'abrupt-4xCO2' experiment
    
    for h in range(len(datapredi_nas)):
        predict_metrics_annually[datapredi_nas[h]] = full((shape_yr_pi + shape_yr_abr),  0.0)
        predict_metrics_annually[datapredi_nas[h]][0:shape_yr_pi] = areamean_dict_predi[datapredi_nas[h]+'_area_yr_pi']
        predict_metrics_annually[datapredi_nas[h]][shape_yr_pi:] = areamean_dict_predi[datapredi_nas[h]+'_area_yr_abr']
        
    # report values, from 'piControl' to 'abrupt-4xCO2' experiment

    for i in range(len(datarepo_nas)):
        report_metrics_annually[datarepo_nas[i]] = full((shape_yr_pi + shape_yr_abr), 0.0)  
        report_metrics_annually[datarepo_nas[i]][0:shape_yr_pi] = areamean_dict_PI[datarepo_nas[i]+'_area_yr']
        report_metrics_annually[datarepo_nas[i]][shape_yr_pi:] = areamean_dict_abr[datarepo_nas[i]+'_area_yr']
    
    print("report albedo (with cloud) using lrm's LWP: ", report_metrics_annually['albedo'])
    print("predicted albedo (with cloud) using lrm's LWP: ", predict_metrics_annually['albedo_lL'])

    
    # put them into the rawdata_dict:
    rawdata_dict['Yrs'] = Yrs
    rawdata_dict['GMT'] = GMT

    rawdata_dict['predicted_metrics'] = predict_metrics_annually
    rawdata_dict['report_metrics'] = report_metrics_annually
    return rawdata_dict




def fitLRM2(C_dict, TR_sst, TR_sub, s_range, y_range, x_range):
    
    # 'C_dict' is the raw data dict, 'TR_sst' accompany with 'TR_sub' are the pre-defined skin_Temperature/ 500 mb Subsidence thresholds to distinguish 4 rdlrms:

    # 's_range , 'y_range', 'x_range' used to do area mean for repeat gmt ARRAY

    dict0_abr_var = C_dict['dict1_abr_var']
    dict0_PI_var  = C_dict['dict1_PI_var']
    #print(dict0_PI_var['times'])
    
    model = C_dict['model_data']   #.. type in dict
    
    datavar_nas = ['LWP', 'TWP', 'IWP', 'PRW', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'SST', 'p_e', 'LTS', 'SUB']   #..13 varisables except gmt (lon dimension diff)
    
    # load annually-mean bin data
    dict1_yr_bin_PI  = dict0_PI_var['dict1_yr_bin_PI']
    dict1_yr_bin_abr  = dict0_abr_var['dict1_yr_bin_abr']
    #print(dict1_yr_bin_PI['LWP_yr_bin'].shape)
    
    # load monthly bin data
    dict1_mon_bin_PI  = dict0_PI_var['dict1_mon_bin_PI']
    dict1_mon_bin_abr  = dict0_abr_var['dict1_mon_bin_abr']

    # data array in which shapes?
    shape_yr_PI_3 = dict1_yr_bin_PI['LWP_yr_bin'].shape
    shape_yr_abr_3 = dict1_yr_bin_abr['LWP_yr_bin'].shape

    shape_yr_PI_gmt = dict1_yr_bin_PI['gmt_yr_bin'].shape
    shape_yr_abr_gmt = dict1_yr_bin_abr['gmt_yr_bin'].shape

    shape_mon_PI = dict1_mon_bin_PI['LWP_mon_bin'].shape
    shape_mon_abr = dict1_mon_bin_abr['LWP_mon_bin'].shape

    shape_mon_PI_gmt = dict1_mon_bin_PI['gmt_mon_bin'].shape
    shape_mon_abr_gmt = dict1_mon_bin_abr['gmt_mon_bin'].shape

    #.. archieve the 'shape' infos:
    C_dict['shape_yr_PI_3']  = shape_yr_PI_3
    C_dict['shape_yr_abr_3']  = shape_yr_abr_3
    C_dict['shape_yr_PI_gmt_3']  = shape_yr_PI_gmt
    C_dict['shape_yr_abr_gmt_3']  = shape_yr_abr_gmt

    C_dict['shape_mon_PI_3']  = shape_mon_PI
    C_dict['shape_mon_abr_3']  = shape_mon_abr
    C_dict['shape_mon_PI_gmt_3']  = shape_mon_PI_gmt
    C_dict['shape_mon_abr_gmt_3']  = shape_mon_abr_gmt

    dict2_predi_fla_PI = {}
    dict2_predi_fla_abr = {}

    dict2_predi_nor_PI = {}
    dict2_predi_nor_abr = {}

    #.. Ravel binned array /Standardized data ARRAY :
    for d in range(len(datavar_nas)):
        dict2_predi_fla_PI[datavar_nas[d]] = dict1_mon_bin_PI[datavar_nas[d]+'_mon_bin'].flatten()
        dict2_predi_fla_abr[datavar_nas[d]] = dict1_mon_bin_abr[datavar_nas[d]+'_mon_bin'].flatten()

        # normalized the predict array
        dict2_predi_nor_PI[datavar_nas[d]] =  (dict2_predi_fla_PI[datavar_nas[d]] - nanmean(dict2_predi_fla_PI[datavar_nas[d]]) )/ nanstd(dict2_predi_fla_PI[datavar_nas[d]])
        dict2_predi_nor_abr[datavar_nas[d]] =  (dict2_predi_fla_abr[datavar_nas[d]] - nanmean(dict2_predi_fla_abr[datavar_nas[d]]) )/ nanstd(dict2_predi_fla_abr[datavar_nas[d]])

    #..Use area_mean method, 'np.repeat' and 'np.tile' to reproduce gmt area-mean Array as the same shape as other flattened variables:
    GMT_pi_mon  = area_mean(dict1_mon_bin_PI['gmt_mon_bin'],  s_range,  x_range)   #..ALL in shape : shape_yr(mon)_abr(single dimension)
    ##  dict2_predi_fla_PI['gmt']  = GMT_pi.repeat(730)
    GMT_abr_mon  = area_mean(dict1_mon_bin_abr['gmt_mon_bin'], s_range, x_range)
    ##  dict2_predi_fla_abr['gmt'] = GMT_abr.repeat(730)

    # Use the southernOCEAN value as the gmt variable
    dict2_predi_fla_PI['gmt'] = dict1_mon_bin_PI['gmt_mon_bin'][:,1:11,:].flatten()
    dict2_predi_fla_abr['gmt'] = dict1_mon_bin_abr['gmt_mon_bin'][:,1:11,:].flatten()

    dict2_predi_nor_PI['gmt'] = (dict2_predi_fla_PI['gmt'] - nanmean(dict2_predi_fla_PI['gmt']) )/ nanstd(dict2_predi_fla_PI['gmt'])
    dict2_predi_nor_abr['gmt'] = (dict2_predi_fla_abr['gmt'] - nanmean(dict2_predi_fla_abr['gmt']) )/ nanstd(dict2_predi_fla_abr['gmt'])
    
    # save into rawdata_dict:
    C_dict['dict2_predi_fla_PI'] =  dict2_predi_fla_PI
    C_dict['dict2_predi_fla_abr'] = dict2_predi_fla_abr
    C_dict['GMT_pi_mon']  = GMT_pi_mon
    C_dict['GMT_abr_mon'] =  GMT_abr_mon
    C_dict['dict2_predi_nor_PI'] =  dict2_predi_nor_PI
    C_dict['dict2_predi_nor_abr']  = dict2_predi_nor_abr

    
    #.. Training Module (4lrm)


    #.. PI
    predict_dict_PI, ind7_PI, ind8_PI, ind9_PI, ind10_PI, coef_array, shape_fla_training = rdlrm_4_training(dict2_predi_fla_PI, TR_sst, TR_sub, predictant='LWP')
    predict_dict_PI_iwp, ind7_PI_iwp, ind8_PI_iwp, ind9_PI_iwp, ind10_PI_iwp, coef_array_iwp, shape_fla_training_iwp = rdlrm_4_training(dict2_predi_fla_PI, TR_sst, TR_sub, predictant='IWP')
    
    predict_dict_PI_albedo, _, _, _, _, coef_array_albedo = rdlrm_4_training(dict2_predi_fla_PI, TR_sst, TR_sub, predictant='albedo', predictor=['LWP'], r = 4)[0:6]
    predict_dict_PI_rsut, _, _, _, _, coef_array_rsut = rdlrm_4_training(dict2_predi_fla_PI, TR_sst, TR_sub, predictant='rsut', predictor=['LWP'], r = 4)[0:6]
    
    # Added on May 13th, 2022: for second step using LWP to predict the albedo
    
    dict2_predi_fla_PI['LWP_lrm'] = deepcopy(predict_dict_PI['value'])
    dict2_predi_nor_PI['LWP_lrm'] = (dict2_predi_fla_PI['LWP_lrm'] - nanmean(dict2_predi_fla_PI['LWP_lrm']) )/ nanstd(dict2_predi_fla_PI['LWP_lrm'])
    predict_dict_PI_albedo_lL, _, _, _, _, coef_array_albedo_lL = rdlrm_4_training(dict2_predi_fla_PI, TR_sst, TR_sub, predictant='albedo', predictor=['LWP_lrm'], r = 4)[0:6]
    predict_dict_PI_rsut_lL, _, _, _, _, coef_array_rsut_lL = rdlrm_4_training(dict2_predi_fla_PI, TR_sst, TR_sub, predictant='rsut', predictor=['LWP_lrm'], r = 4)[0:6]
    

    # Save into the rawdata dict
    
    C_dict['Coef_dict'] = coef_array
    C_dict['Predict_dict_PI']  = predict_dict_PI
    C_dict['ind_Cold_Up_PI'] = ind7_PI
    C_dict['ind_Hot_Up_PI'] = ind8_PI
    C_dict['ind_Cold_Down_PI'] = ind9_PI
    C_dict['ind_Hot_Down_PI'] = ind10_PI
    
    C_dict['Coef_dict_IWP']= coef_array_iwp
    C_dict['Predict_dict_PI_IWP']  = predict_dict_PI_iwp
    # C_dict['ind_Cold_Up_PI_IWP'] = ind7_PI_iwp
    # C_dict['ind_Hot_Up_PI_IWP'] = ind8_PI_iwp
    # C_dict['ind_Cold_Down_PI_IWP'] = ind9_PI_iwp
    # C_dict['ind_Hot_Down_PI_IWP'] = ind10_PI_iwp
    
    # Albedo and radiation 
    C_dict['Coef_dict_albedo'] = coef_array_albedo
    C_dict['Predict_dict_PI_albedo'] = predict_dict_PI_albedo
    
    C_dict['Coef_dict_rsut'] = coef_array_rsut
    C_dict['Predict_dict_PI_rsut'] = predict_dict_PI_rsut
    
    C_dict['Coef_dict_albedo_lL'] = coef_array_albedo_lL
    C_dict['Predict_dict_PI_albedo_lL'] = predict_dict_PI_albedo_lL
    
    C_dict['Coef_dict_rsut_lL'] = coef_array_rsut_lL
    C_dict['Predict_dict_PI_rsut_lL'] = predict_dict_PI_rsut_lL
    
    
    # 'YB' is the predicted value of LWP in 'piControl' experiment
    YB = predict_dict_PI['value']
    # print("4lrm predicted mean LWP ", nanmean(YB), " in 'piControl' ")

    YB_iwp = predict_dict_PI_iwp['value']
    # print("4lrm predicted mean IWP ", nanmean(YB_iwp), " in 'piControl' ")
    
    YB_albedo = predict_dict_PI_albedo['value']
    # print("4lrm predicted mean Albedo (with cloud): ", nanmean(YB_albedo), " in 'piControl' ")
    YB_rsut = predict_dict_PI_rsut['value']

    YB_albedo_lL = predict_dict_PI_albedo_lL['value']
    print("4lrm predicted mean Albedo (with cloud) using lrm LWP:", nanmean(YB_albedo_lL), " in 'piControl' ")
    YB_rsut_lL = predict_dict_PI_rsut_lL['value']
    
    
    # Save 'YB', resampled into the shape of 'LWP_yr_bin':
    C_dict['LWP_predi_bin_PI'] = asarray(YB).reshape(shape_mon_PI)

    C_dict['IWP_predi_bin_PI'] = asarray(YB_iwp).reshape(shape_mon_PI)
    C_dict['albedo_predi_bin_PI'] = asarray(YB_albedo).reshape(shape_mon_PI)
    C_dict['rsut_predi_bin_PI'] = asarray(YB_rsut).reshape(shape_mon_PI)

    C_dict['albedo_lL_predi_bin_PI'] = asarray(YB_albedo_lL).reshape(shape_mon_PI)
    C_dict['rsut_lL_predi_bin_PI'] = asarray(YB_rsut_lL).reshape(shape_mon_PI)


    #.. Test performance

    stats_dict_PI = Test_performance_4(dict2_predi_fla_PI['LWP'], YB, ind7_PI, ind8_PI, ind9_PI, ind10_PI)
    stats_dict_PI_iwp = Test_performance_4(dict2_predi_fla_PI['IWP'], YB_iwp, ind7_PI_iwp, ind8_PI_iwp, ind9_PI_iwp, ind10_PI_iwp)
    stats_dict_PI_albedo = Test_performance_4(dict2_predi_fla_PI['albedo'], YB_albedo, ind7_PI, ind8_PI, ind9_PI, ind10_PI)
    stats_dict_PI_rsut = Test_performance_4(dict2_predi_fla_PI['rsut'], YB_rsut, ind7_PI, ind8_PI, ind9_PI, ind10_PI)
    
    stats_dict_PI_albedo_lL = Test_performance_4(dict2_predi_fla_PI['albedo'], YB_albedo_lL, ind7_PI, ind8_PI, ind9_PI, ind10_PI)
    stats_dict_PI_rsut_lL = Test_performance_4(dict2_predi_fla_PI['rsut'], YB_rsut_lL, ind7_PI, ind8_PI, ind9_PI, ind10_PI)
    
    print(" Mean of report & predicted albedo_lL for 'piControl' (all): ", nanmean(dict2_predi_fla_PI['albedo']), '& ', nanmean(YB_albedo_lL))
    print(" Mean of report & predicted albedo_lL for 'piControl' of Up, Down regime: " , nanmean(dict2_predi_fla_PI['albedo'][ind10_PI]), '& ', nanmean(YB_albedo_lL[ind10_PI]))


    # ####)################################
    #.. ABR
    #.. Predicting module (4lrm)

    predict_dict_abr, ind7_abr, ind8_abr, ind9_abr, ind10_abr, shape_fla_testing = rdlrm_4_predict(dict2_predi_fla_abr, coef_array, TR_sst, TR_sub, predictant = 'LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 4)
    predict_dict_abr_iwp, ind7_abr_iwp, ind8_abr_iwp, ind9_abr_iwp, ind10_abr_iwp, shape_fla_testing_iwp = rdlrm_4_predict(dict2_predi_fla_abr, coef_array, TR_sst, TR_sub, predictant = 'IWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 4)
    
    predict_dict_abr_albedo = rdlrm_4_predict(dict2_predi_fla_abr, coef_array_albedo, TR_sst, TR_sub, predictant = 'albedo', predictor = ['LWP'], r = 4)[0]
    predict_dict_abr_rsut = rdlrm_4_predict(dict2_predi_fla_abr, coef_array_rsut, TR_sst, TR_sub, predictant = 'rsut', predictor= ['LWP'], r = 4)[0]
    
    # Added on May 14th, 2022: for second step using LWP to predict the albedo
    dict2_predi_fla_abr['LWP_lrm'] = deepcopy(predict_dict_abr['value'])
    dict2_predi_nor_abr['LWP_lrm'] = (dict2_predi_fla_abr['LWP_lrm'] - nanmean(dict2_predi_fla_abr['LWP_lrm']) )/ nanstd(dict2_predi_fla_abr['LWP_lrm'])
    predict_dict_abr_albedo_lL = rdlrm_4_predict(dict2_predi_fla_abr, coef_array_albedo, TR_sst, TR_sub, predictant='albedo', predictor=['LWP_lrm'], r = 4)[0]
    predict_dict_abr_rsut_lL = rdlrm_4_predict(dict2_predi_fla_abr, coef_array_rsut, TR_sst, TR_sub, predictant='rsut', predictor=['LWP_lrm'], r = 4)[0]

    # Save into the rawdata dict
    C_dict['Predict_dict_abr']  = predict_dict_abr
    C_dict['ind_Cold_Up_abr'] = ind7_abr
    C_dict['ind_Hot_Up_abr'] = ind8_abr
    C_dict['ind_Cold_Down_abr'] = ind9_abr
    C_dict['ind_Hot_Down_abr'] = ind10_abr
    
    C_dict['Predict_dict_abr_IWP']  = predict_dict_abr_iwp
    # C_dict['ind_Cold_Up_abr_IWP'] = ind7_abr_iwp
    # C_dict['ind_Hot_Up_abr_IWP'] = ind8_abr_iwp
    # C_dict['ind_Cold_Down_abr_IWP'] = ind9_abr_iwp
    # C_dict['ind_Hot_Down_abr_IWP'] = ind10_abr_iwp
    
    C_dict['Predict_dict_abr_albedo'] = predict_dict_abr_albedo
    C_dict['Predict_dict_abr_rsut'] = predict_dict_abr_rsut
    
    C_dict['Predict_dict_abr_albedo_lL'] = predict_dict_abr_albedo_lL
    C_dict['Predict_dict_abr_rsut_lL'] = predict_dict_abr_rsut_lL
    
    
    # 'YB_abr' is the predicted value of LWP in 'abrupt-4xCO2' experiment
    YB_abr = predict_dict_abr['value']
    # print("4lrm predicted mean LWP ", nanmean(YB_abr), " in 'abrupt-4xCO2' ")

    YB_abr_iwp = predict_dict_abr_iwp['value']
    # print("4lrm predicted mean IWP ", nanmean(YB_abr_iwp), " in 'abrupt-4xCO2' ")
    
    YB_abr_albedo = predict_dict_abr_albedo['value']
    # print("4lrm predicted mean Albedo (with cloud)", nanmean(YB_abr_albedo), " in 'abrupt-4xCO2' ")
    YB_abr_rsut = predict_dict_abr_rsut['value']
    
    YB_abr_albedo_lL = predict_dict_abr_albedo_lL['value']
    print("4lrm predicted mean Albedo (with cloud) using lrm's LWP", nanmean(YB_abr_albedo_lL), " in 'abrupt-4xCO2' ")
    YB_abr_rsut_lL = predict_dict_abr_rsut_lL['value']
    
    # print(" 4lrm: predicted LWP of 'abrupt-4xCO2':", YB_abr)
    # print(" 4lrm: report LWP of 'abrupt-4xCO2':", dict2_predi_fla_abr['LWP'])
    # print(" 4lrm: predicted albedo of 'abrupt-4xCO2':", YB_abr_albedo)
    # print(" 4lrm: report albedo of  'abrupt-4xCO2':", dict2_predi_fla_abr['albedo'])
    
    
    # Save 'YB_abr', reshapled into the shape of 'LWP_yr_bin_abr':
    C_dict['LWP_predi_bin_abr'] =  asarray(YB_abr).reshape(shape_mon_abr)
    C_dict['IWP_predi_bin_abr'] =  asarray(YB_abr_iwp).reshape(shape_mon_abr)
    C_dict['albedo_predi_bin_abr'] = asarray(YB_abr_albedo).reshape(shape_mon_abr)
    C_dict['rsut_predi_bin_abr'] = asarray(YB_abr_rsut).reshape(shape_mon_abr)
    
    C_dict['albedo_lL_predi_bin_abr'] = asarray(YB_abr_albedo_lL).reshape(shape_mon_abr)
    C_dict['rsut_lL_predi_bin_abr'] = asarray(YB_abr_rsut_lL).reshape(shape_mon_abr)


    # Test performance for abrupt-4xCO2 (testing) data set
    
    stats_dict_abr = Test_performance_4(dict2_predi_fla_abr['LWP'], YB_abr, ind7_abr, ind8_abr, ind9_abr, ind10_abr)
    stats_dict_abr_iwp = Test_performance_4(dict2_predi_fla_abr['IWP'], YB_abr_iwp, ind7_abr_iwp, ind8_abr_iwp, ind9_abr_iwp, ind10_abr_iwp)
    stats_dict_abr_albedo = Test_performance_4(dict2_predi_fla_abr['albedo'], YB_abr_albedo, ind7_abr, ind8_abr, ind9_abr, ind10_abr)
    stats_dict_abr_rsut = Test_performance_4(dict2_predi_fla_abr['rsut'], YB_abr_rsut, ind7_abr, ind8_abr, ind9_abr, ind10_abr)
    
    stats_dict_abr_albedo_lL = Test_performance_4(dict2_predi_fla_abr['albedo'], YB_abr_albedo_lL, ind7_abr, ind8_abr, ind9_abr, ind10_abr)
    stats_dict_abr_rsut_lL = Test_performance_4(dict2_predi_fla_abr['rsut'], YB_abr_rsut_lL, ind7_abr, ind8_abr, ind9_abr, ind10_abr)

    print(" Mean of report & predicted albedo_lL for 'abrupt-4xCO2' (all): ", nanmean(dict2_predi_fla_abr['albedo']), '& ', nanmean(YB_abr_albedo_lL))
    print(" Mean of report & predicted albedo_lL for 'abrupt-4xCO2' of Up, Down regime: " , nanmean(dict2_predi_fla_abr['albedo'][ind10_abr]), '& ', nanmean(YB_abr_albedo_lL[ind10_abr]))
    '''
    # calc d(CCFs) to d(gmt) for 4 Regime and save them into 'Dx/DtG' dict
    
    
    regr12 = linear_model.LinearRegression()
    re_SST = regr12.fit(dict2_predi_fla_abr['gmt'][ind_true_abr].reshape(-1,1), dict2_predi_fla_abr['SST'][ind_true_abr])
    regr13 = linear_model.LinearRegression()
    re_p_e = regr13.fit(dict2_predi_fla_abr['gmt'][ind_true_abr].reshape(-1,1), dict2_predi_fla_abr['p_e'][ind_true_abr])
    regr14 = linear_model.LinearRegression()
    re_LTS = regr14.fit(dict2_predi_fla_abr['gmt'][ind_true_abr].reshape(-1,1), dict2_predi_fla_abr['LTS'][ind_true_abr])
    regr15 = linear_model.LinearRegression()
    re_SUB = regr15.fit(dict2_predi_fla_abr['gmt'][ind_true_abr].reshape(-1,1), dict2_predi_fla_abr['SUB'][ind_true_abr])
    
    
    regr10 = linear_model.LinearRegression()
    regr20 = linear_model.LinearRegression()
    regr30 = linear_model.LinearRegression()
    regr40 = linear_model.LinearRegression()
    re_LWPr1= regr10.fit(dict2_predi_fla_abr['gmt'][ind7].reshape(-1,1), dict2_predi_fla_abr['LWP'][ind7])
    re_LWPr2= regr20.fit(dict2_predi_fla_abr['gmt'][ind8].reshape(-1,1), dict2_predi_fla_abr['LWP'][ind8])
    re_LWPr3= regr30.fit(dict2_predi_fla_abr['gmt'][ind9].reshape(-1,1), dict2_predi_fla_abr['LWP'][ind9])
    re_LWPr4= regr40.fit(dict2_predi_fla_abr['gmt'][ind10].reshape(-1,1), dict2_predi_fla_abr['LWP'][ind10])
    
    re_LWP = array([re_LWPr1.coef_, re_LWPr2.coef_, re_LWPr3.coef_, re_LWPr4.coef_]).ravel()
    
    
    regr11 = linear_model.LinearRegression()
    regr21 = linear_model.LinearRegression()
    regr31 = linear_model.LinearRegression()
    regr41 = linear_model.LinearRegression()
    re_SSTr1= regr11.fit(dict2_predi_fla_abr['gmt'][ind7].reshape(-1,1), dict2_predi_fla_abr['SST'][ind7])
    re_SSTr2= regr21.fit(dict2_predi_fla_abr['gmt'][ind8].reshape(-1,1), dict2_predi_fla_abr['SST'][ind8])
    re_SSTr3= regr31.fit(dict2_predi_fla_abr['gmt'][ind9].reshape(-1,1), dict2_predi_fla_abr['SST'][ind9])
    re_SSTr4= regr41.fit(dict2_predi_fla_abr['gmt'][ind10].reshape(-1,1), dict2_predi_fla_abr['SST'][ind10])
    
    re_SST = array([re_SSTr1.coef_, re_SSTr2.coef_, re_SSTr3.coef_, re_SSTr4.coef_]).ravel()
    
    
    regr12 = linear_model.LinearRegression()
    regr22 = linear_model.LinearRegression()
    regr32 = linear_model.LinearRegression()
    regr42 = linear_model.LinearRegression()
    re_p_er1= regr12.fit(dict2_predi_fla_abr['gmt'][ind7].reshape(-1,1), dict2_predi_fla_abr['p_e'][ind7])
    re_p_er2= regr22.fit(dict2_predi_fla_abr['gmt'][ind8].reshape(-1,1), dict2_predi_fla_abr['p_e'][ind8])
    re_p_er3= regr32.fit(dict2_predi_fla_abr['gmt'][ind9].reshape(-1,1), dict2_predi_fla_abr['p_e'][ind9])
    re_p_er4= regr42.fit(dict2_predi_fla_abr['gmt'][ind10].reshape(-1,1), dict2_predi_fla_abr['p_e'][ind10])
    
    re_p_e = array([re_p_er1.coef_, re_p_er2.coef_, re_p_er3.coef_, re_p_er4.coef_]).ravel()
    
    
    regr13 = linear_model.LinearRegression()
    regr23 = linear_model.LinearRegression()
    regr33 = linear_model.LinearRegression()
    regr43 = linear_model.LinearRegression()
    re_LTSr1= regr13.fit(dict2_predi_fla_abr['gmt'][ind7].reshape(-1,1), dict2_predi_fla_abr['LTS'][ind7])
    re_LTSr2= regr23.fit(dict2_predi_fla_abr['gmt'][ind8].reshape(-1,1), dict2_predi_fla_abr['LTS'][ind8])
    re_LTSr3= regr33.fit(dict2_predi_fla_abr['gmt'][ind9].reshape(-1,1), dict2_predi_fla_abr['LTS'][ind9])
    re_LTSr4= regr43.fit(dict2_predi_fla_abr['gmt'][ind10].reshape(-1,1), dict2_predi_fla_abr['LTS'][ind10])
    
    re_LTS = array([re_LTSr1.coef_, re_LTSr2.coef_, re_LTSr3.coef_, re_LTSr4.coef_]).ravel()
    
    
    regr14 = linear_model.LinearRegression()
    regr24 = linear_model.LinearRegression()
    regr34 = linear_model.LinearRegression()
    regr44 = linear_model.LinearRegression()
    re_SUBr1= regr14.fit(dict2_predi_fla_abr['gmt'][ind7].reshape(-1,1), output_4lrm_flavra_abr['SUB'][ind7])
    re_SUBr2= regr24.fit(dict2_predi_fla_abr['gmt'][ind8].reshape(-1,1), output_4lrm_flavra_abr['SUB'][ind8])
    re_SUBr3= regr34.fit(dict2_predi_fla_abr['gmt'][ind9].reshape(-1,1), output_4lrm_flavra_abr['SUB'][ind9])
    re_SUBr4= regr44.fit(dict2_predi_fla_abr['gmt'][ind10].reshape(-1,1), output_4lrm_flavra_abr['SUB'][ind10])
    
    re_SUB = array([re_SUBr1.coef_, re_SUBr2.coef_, re_SUBr3.coef_, re_SUBr4.coef_]).ravel()
    
    
    # regr11_iwp = linear_model.LinearRegression()
    # re_IWP= regr11_iwp.fit(dict2_predi_fla_abr['gmt'][ind_true_abr].reshape(-1,1), dict2_predi_fla_abr['IWP'][ind_true_abr])
    regr15 = linear_model.LinearRegression()
    regr25 = linear_model.LinearRegression()
    regr35 = linear_model.LinearRegression()
    regr45 = linear_model.LinearRegression()
    re_IWPr1= regr15.fit(dict2_predi_fla_abr['gmt'][ind7].reshape(-1,1), dict2_predi_fla_abr['IWP'][ind7])
    re_IWPr2= regr25.fit(dict2_predi_fla_abr['gmt'][ind8].reshape(-1,1), dict2_predi_fla_abr['IWP'][ind8])
    re_IWPr3= regr35.fit(dict2_predi_fla_abr['gmt'][ind9].reshape(-1,1), dict2_predi_fla_abr['IWP'][ind9])
    re_IWPr4= regr45.fit(dict2_predi_fla_abr['gmt'][ind10].reshape(-1,1), dict2_predi_fla_abr['IWP'][ind10])
    
    re_IWP = array([re_IWPr1.coef_, re_IWPr2.coef_, re_IWPr3.coef_, re_IWPr4.coef_]).ravel()
    
    
    print('d(CCFs)/d(gmt)| (has LTS VALUES) and Warm&Up regime= ', re_SST[1], re_p_e[1], re_LTS[1],  re_SUB[1])
    
    #..save into rawdata_dict
    Dx_DtG = {'re_LWP': re_LWP, 're_SST': re_SST , 're_p_e': re_p_e, 're_LTS': re_LTS, 're_SUB': re_SUB, 're_IWP': re_IWP}
    C_dict['dX_dTg'] =  Dx_DtG
    '''

    #.. save test performance metrics into rawdata_dict

    C_dict['stats_dict_PI'] = stats_dict_PI
    C_dict['stats_dict_PI_iwp'] = stats_dict_PI_iwp

    C_dict['stats_dict_abr'] = stats_dict_abr
    C_dict['stats_dict_abr_iwp'] = stats_dict_abr_iwp

    C_dict['stats_dict_PI_albedo'] = stats_dict_PI_albedo
    C_dict['stats_dict_abr_albedo'] = stats_dict_abr_albedo

    C_dict['stats_dict_PI_rsut'] = stats_dict_PI_rsut
    C_dict['stats_dict_abr_rsut'] = stats_dict_abr_rsut

    C_dict['stats_dict_PI_albedo_lL'] = stats_dict_PI_albedo_lL
    C_dict['stats_dict_abr_albedo_lL'] = stats_dict_abr_albedo_lL

    C_dict['stats_dict_PI_rsut_lL'] = stats_dict_PI_rsut_lL
    C_dict['stats_dict_abr_rsut_lL'] = stats_dict_abr_rsut_lL

    return C_dict