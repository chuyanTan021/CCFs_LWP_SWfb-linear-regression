# get the data we needed from read module:'get_LWPCMIP6', and do some data-processing for building the linear regression CCFs_Clouds models:
# transform data to annual-mean/ monthly-mean bin array or flattened array;
# fit the regression model 1&2 from pi-Control CCFs' sensitivities to the LWP, then do the regressions and save the data;

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
#from get_annual_so import *
from fitLRM_cy import *



def calc_LRM_metrics(**model_data):
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
    
    levels      = np.array(inputVar_abr['pres'])
    times_abr   = inputVar_abr['times']
    times_pi    = inputVar_pi['times']
    
    
    lati0 = -40.
    latsi0= min(range(len(lats)), key = lambda i: abs(lats[i] - lati0))
    lati1 = -85.
    latsi1= min(range(len(lats)), key = lambda i: abs(lats[i] - lati1))
    print('lat index for 40.s; 85.s', latsi0, latsi1)
    
    
    shape_latSO =  latsi0 - latsi1
    #print(shape_latSO)
    
    
    #..abrupt4xCO2 Variables: LWP, tas(gmt), SST, p-e, LTS, subsidence
    LWP_abr  = np.array(inputVar_abr['clwvi']) - np.array(inputVar_abr['clivi'])   #..units in kg m^-2
    
    gmt_abr  = np.array(inputVar_abr['tas'])
    
    SST_abr  = np.array(inputVar_abr['sfc_T'])
    
    
    Precip_abr =  np.array(inputVar_abr['P']) * (24.*60.*60.)   #..Precipitation. Convert the units from kg m^-2 s^-1 -> mm*day^-1
    print('abr4x average Pr(mm/ day): ', np.nanmean(Precip_abr))   #.. IPSL/abr2.80..  CNRM ESM2 1/abr 2.69.. CESM2/abr 2.74..
    Eva_abr    =  np.array(inputVar_abr['E']) * (24.*60.*60.)   #..evaporation, mm day^-1
    print('abr4x average Evapor(mm/ day): ', np.nanmean(Eva_abr))         #.. IPSL/abr2.50..  CNRM ESM2 1/abr 2.43.. CESM2/abr 2.43..
    
    MC_abr  = Precip_abr - Eva_abr   #..Moisture Convergence calculated from abrupt4xCO2's P - E, Units in mm day^-1
    
    Twp_abr  = np.array(inputVar_abr['clwvi'])
    Iwp_abr  = np.array(inputVar_abr['clivi'])
    prw_abr  = np.array(inputVar_abr['prw'])
    
    print('abr4x Eva:', Eva_abr.shape)
    print('abr4x mean-gmt(K): ', np.nanmean(gmt_abr))
    
    
    #..pi-Control Variables: LWP, tas(gmt), SST, p-e, LTS, subsidence
    LWP  = np.array(inputVar_pi['clwvi']) - np.array(inputVar_pi['clivi'])   #..units in kg m^-2
    
    gmt  = np.array(inputVar_pi['tas'])
    
    SST  = np.array(inputVar_pi['sfc_T'])
    
    
    Precip =  np.array(inputVar_pi['P'])* (24.*60.*60.)    #..Precipitation. Convert the units from kg m^-2 s^-1 -> mm*day^-1
    print('pi-C average Pr(mm/ day): ', np.nanmean(Precip))   #.. IPSL/piC 2.43..CNRM/piC 2.40.. CESM2/PIc 2.39
    Eva    =  np.array(inputVar_pi['E']) * (24.*60.*60.)   #..evaporation, mm day^-1
    print('pi-C average Evapor(mm/day): ', np.nanmean(Eva))   #.. IPSL/piC  2.21..CNRM/piC 2.20.. CESM2/PIc 2.17..
    
    MC  = Precip - Eva   #..Moisture Convergence calculated from pi-Control's P - E, Units in mm day^-1
    
    Twp  = np.array(inputVar_pi['clwvi'])
    Iwp  = np.array(inputVar_pi['clivi'])
    prw_pi  = np.array(inputVar_pi['prw'])
    print('pi-C Eva:', Eva.shape, 'and pi-C mean-gmt(K): ',np.nanmean(gmt))
    
    
    #..abrupt4xCO2
    # Lower Tropospheric Stability:
    
    k  = 0.286
    
    theta_700_abr  = np.array(inputVar_abr['T_700']) * (100000./70000.)**k
    theta_skin_abr = np.array(inputVar_abr['sfc_T']) * (100000./np.array(inputVar_abr['sfc_P']))**k 
    LTS_m_abr  = theta_700_abr - theta_skin_abr
    
    
    #..Subtract the outliers in T_700 and LTS_m, 'nan' comes from missing T_700 data
    LTS_e_abr  = np.ma.masked_where(theta_700_abr >= 500, LTS_m_abr)
    
    
    # Meteorology Subsidence at 500 hPa, units in Pa s^-1:
    Subsidence_abr =  np.array(inputVar_abr['sub'])
    
    
    #..pi-Control 
    # Lower Tropospheric Stability:
    theta_700  = np.array(inputVar_pi['T_700']) * (100000./70000.)**k
    theta_skin = np.array(inputVar_pi['sfc_T']) * (100000./np.array(inputVar_pi['sfc_P']))**k 
    LTS_m  = theta_700 - theta_skin
    
    #..Subtract the outliers in T_700 and LTS_m 
    LTS_e  = np.ma.masked_where(theta_700 >= 500, LTS_m)
    
    
    #..Meteological Subsidence  at 500 hPa, units in Pa s^-1:
    Subsidence =  np.array(inputVar_pi['sub'])
    
    
    # define Dictionary to store: CCFs(4), gmt, other variables 
    dict0_PI_var = {'gmt': gmt, 'LWP': LWP, 'TWP': Twp, 'IWP': Iwp,  'PRW': prw_pi, 'SST': SST, 'p_e': MC, 'LTS': LTS_e, 'SUB': Subsidence
                     ,'lat':lats, 'lon':lons, 'times': times_pi, 'pres':levels}

    dict0_abr_var = {'gmt': gmt_abr, 'LWP': LWP_abr, 'TWP': Twp_abr, 'IWP': Iwp_abr,  'PRW': prw_abr, 'SST': SST_abr, 'p_e': MC_abr, 'LTS': LTS_e_abr 
                     ,'SUB': Subsidence_abr, 'lat':lats, 'lon':lons, 'times': times_abr, 'pres':levels}
    
    
    
    
    # get the Annual-mean, Southern-Ocean region arrays

    datavar_nas = ['LWP', 'TWP', 'IWP', 'PRW', 'SST', 'p_e', 'LTS', 'SUB']   #..8 varisables except gmt (lon dimension diff)

    dict1_PI_yr  = {}
    dict1_abr_yr = {}
    shape_yr_pi  = shape_time_pi//12
    shape_yr_abr =  shape_time_abr//12
    
    layover_yr_abr = np.zeros((len(datavar_nas), shape_yr_abr, shape_latSO, shape_lon))
    layover_yr_pi  = np.zeros((len(datavar_nas), shape_yr_pi, shape_latSO, shape_lon))

    layover_yr_abr_gmt = np.zeros((shape_yr_abr, shape_lat, shape_lon))
    layover_yr_pi_gmt = np.zeros((shape_yr_pi, shape_lat, shape_lon))


    for a in range(len(datavar_nas)):

        #a_array = dict0_abr_var[datavar_nas[a]]

        for i in range(shape_time_abr//12):
            layover_yr_abr[a, i,:,:] =  nanmean(dict0_abr_var[datavar_nas[a]][i*12:(i+1)*12, latsi1:latsi0,:], axis=0)

        dict1_abr_yr[datavar_nas[a]+'_yr'] =  layover_yr_abr[a,:]


        #b_array = dict0_PI_var[datavar_nas[a]]
        for j in range(shape_time_pi//12):
            layover_yr_pi[a, j,:,:]  = nanmean(dict0_PI_var[datavar_nas[a]][j*12:(j+1)*12,  latsi1:latsi0,:], axis=0)

        dict1_PI_yr[datavar_nas[a]+'_yr'] =  layover_yr_pi[a,:]
        print(datavar_nas[a])    

    #print(dict1_PI_yr['LWP_yr'])
    
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
    x_range  = arange(-180., 183, 5.)   #..logitude sequences edge: number:73
    s_range  = arange(-90., 90, 5.) + 2.5   #..global-region latitude edge:(36)

    y_range  = arange(-85, -35., 5.) +2.5   #..southern-ocaen latitude edge:10

    
    # Annually variables in bin box: 
    
    lat_array  = lats[latsi1:latsi0]
    lon_array  = lons
    lat_array1 =  lats

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
    dict0_abr_var['dict1_yr_bin_abr']  = dict1_yr_bin_abr
    dict0_PI_var['dict1_yr_bin_PI']  = dict1_yr_bin_PI


    # Monthly variables (same as above):
    dict1_mon_bin_PI  = {}
    dict1_mon_bin_abr = {}
    
    for c in range(len(datavar_nas)):

        dict1_mon_bin_abr[datavar_nas[c]+'_mon_bin']  =   binned_cySouthOcean5(dict0_abr_var[datavar_nas[c]][:, latsi1:latsi0,:], lat_array, lon_array)
        dict1_mon_bin_PI[datavar_nas[c]+'_mon_bin']   =  binned_cySouthOcean5(dict0_PI_var[datavar_nas[c]][:, latsi1:latsi0,:], lat_array, lon_array)

    dict1_mon_bin_abr['gmt_mon_bin']   =  binned_cyGlobal5(dict0_abr_var['gmt'], lat_array1, lon_array)
    dict1_mon_bin_PI['gmt_mon_bin']   =  binned_cyGlobal5(dict0_PI_var['gmt'], lat_array1, lon_array)

    print('gmt_mon_bin')
    dict0_abr_var['dict1_mon_bin_abr']  = dict1_mon_bin_abr
    dict0_PI_var['dict1_mon_bin_PI']  = dict1_mon_bin_PI
    
    
    # input the shapes of year and month of pi&abr exper into the raw data dictionaries:
    dict0_abr_var['shape_yr'] = shape_yr_abr
    dict0_PI_var['shape_yr'] = shape_yr_pi
    
    dict0_abr_var['shape_mon'] = shape_time_abr
    dict0_PI_var['shape_mon'] = shape_time_pi
    
    # Output a dict for processing function in 'calc_LRM_metrics', stored the data dicts for PI and abr, with the model name_dict
    C_dict =  {'dict0_PI_var':dict0_PI_var, 'dict0_abr_var':dict0_abr_var, 'model_data':model_data}


    ###..Put data into 'fitLRM' FUNCTION to get predicted LWP splitted by 'Tr_sst'/'Tr_sub' infos_models:
    
    TR_sst   = 270.0   ###.. important line
    TR_sub   = 0.0  ###.threshold of 500 mb Subsidences
    rawdata_dict =  fitLRM(C_dict, TR_sst, s_range, y_range, x_range)
    rawdata_dict =  fitLRM2(C_dict, TR_sst, TR_sub, s_range, y_range, x_range)
    rawdata_dict2 = p4plot1(rawdata_dict, s_range, y_range, x_range, shape_yr_pi, shape_yr_abr)
    
    WD = '/glade/work/chuyan/Research/linear_regression_CCFs_Clouds_metrics/plots_test1/'
    savez(WD+C_dict['model_data']['modn']+'_'+str(TR_sst)+'K_ '+ 'ud'+str(TR_sub)+'_dats', model_data=C_dict['model_data'], rawdata_dict=rawdata_dict2)#.. +str(TR_sub) 
    
    
    return rawdata_dict2







def fitLRM(C_dict, TR_sst, s_range, y_range, x_range):
    # 'C_dict' is the raw data dict, 'TR_sst' is the predefined skin_Temperature Threshold to distinguish two Linear-Regression Models
    # 's_range , 'y_range', 'x_range' used to do area mean for repeat gmt ARRAY
    dict0_abr_var = C_dict['dict0_abr_var']
    dict0_PI_var  = C_dict['dict0_PI_var']
    #print(dict0_PI_var['times'])

    model = C_dict['model_data']   #.. type in dict

    datavar_nas = ['LWP', 'TWP', 'IWP', 'PRW', 'SST', 'p_e', 'LTS', 'SUB']   #..8 varisables except gmt (lon dimension diff)

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

    
    dict2_predi_fla_PI = {}
    dict2_predi_fla_abr = {}
    
    dict2_predi_nor_PI = {}
    dict2_predi_nor_abr = {}
    
    #..Ravel binned array /Standardized data ARRAY :
    for d in range(len(datavar_nas)):
        dict2_predi_fla_PI[datavar_nas[d]] = dict1_yr_bin_PI[datavar_nas[d]+'_yr_bin'].flatten()
        dict2_predi_fla_abr[datavar_nas[d]] = dict1_yr_bin_abr[datavar_nas[d]+'_yr_bin'].flatten()

        # normalized the predict array
        dict2_predi_nor_PI[datavar_nas[d]] =  (dict2_predi_fla_PI[datavar_nas[d]] - nanmean(dict2_predi_fla_PI[datavar_nas[d]]) )/ nanstd(dict2_predi_fla_PI[datavar_nas[d]])
        dict2_predi_nor_abr[datavar_nas[d]] =  (dict2_predi_fla_abr[datavar_nas[d]] - nanmean(dict2_predi_fla_abr[datavar_nas[d]]) )/ nanstd(dict2_predi_fla_abr[datavar_nas[d]])

    #..Use area_mean method, 'np.repeat' and 'np.tile' to reproduce gmt area-mean Array as the same shape as other flattened variables:
    GMT_pi  = area_mean(dict1_yr_bin_PI['gmt_yr_bin'],  s_range,  x_range)   #..ALL in shape : shape_yr_abr(single dimension)
    dict2_predi_fla_PI['gmt']  = GMT_pi.repeat(730)
    GMT_abr  = area_mean(dict1_yr_bin_abr['gmt_yr_bin'], s_range, x_range)   #..ALL in shape : shape_yr_abr(single dimension)
    dict2_predi_fla_abr['gmt'] = GMT_abr.repeat(730)
    
    #dict2_predi_nor_PI['gmt']  =  (dict2_predi_fla_PI['gmt'] - nanmean(dict2_predi_fla_PI['gmt']) )/ nanstd(dict2_predi_fla_PI['gmt'])
    #dict2_predi_nor_abr['gmt'] =   (dict2_predi_fla_abr['gmt'] - nanmean(dict2_predi_fla_abr['gmt']) )/ nanstd(dict2_predi_fla_abr['gmt'])
    
    # save into rawdata_dict:
    C_dict['dict2_predi_fla_PI'] =  dict2_predi_fla_PI
    C_dict['dict2_predi_fla_abr'] = dict2_predi_fla_abr
    C_dict['dict2_predi_nor_PI'] =  dict2_predi_nor_PI
    C_dict['dict2_predi_nor_abr']  = dict2_predi_nor_abr
    
    print('shape1: ', dict2_predi_fla_PI['LWP'].shape)   # shape1
    shape_fla_PI   =   dict2_predi_fla_PI['LWP'].shape
    #print(min(dict2_predi_fla_PI['LTS']),  max(dict2_predi_fla_PI['LTS']) )

    # PI
    #..Subtract 'nan' in data, shape1 -> shape2(without 'nan' number) points and shape5('nan' number)

    ind1 = isnan(dict2_predi_fla_PI['LTS'])==False 

    ind_true = nonzero(ind1==True)
    ind_false = nonzero(ind1==False)
    #..Sign the the indexing into YB, or YB value will have a big changes
    print('shape2: ', array(ind_true).shape)        # shape2
    #  print(argwhere(isnan(dict2_predi_fla_PI['LTS'][ind_true])==True))


    #..Split data points with skin Temperature largerorEqual /Smaller than TR_sst: 

    # shape1 split into shape3(larger.equal.Tr_sst) & shape4(smaller,Tr_sst)

    ind_sst_le  = nonzero(dict2_predi_fla_PI['SST'] >= TR_sst)
    ind_sst_st  = nonzero(dict2_predi_fla_PI['SST'] <  TR_sst)
    # shape6:the intersection of shape2 and shape3, places where has LTS value and skin_T >= Threshold
    ind6  = intersect1d(ind_true, ind_sst_le)
    print('shape6: ', ind6.shape)   #.. points, shape6
    # shape7:the intersection of shape5 and shape4, places have LTS value  but skin_T < Threshold
    ind7  = intersect1d(ind_true, ind_sst_st)
    print('shape7: ', ind7.shape)   #.. points, shape7

    #..designate LWP single-array's value, PI
    YB =  full((shape_fla_PI), 0.0)
    YB[ind_false] =  dict2_predi_fla_PI['LWP'][ind_false]   #..LWP single-column array with no LTS points as original values, with has LTS value points as 0.0.
    
    #print('YB(raw PI LWP array) ', YB)
    #print(YB.shape)
    
     #..designate IWP single-array's value, PI
    YB_iwp =  full((shape_fla_PI), 0.0)
    YB_iwp[ind_false] =  dict2_predi_fla_PI['IWP'][ind_false]


    #.. Multiple linear regreesion of Liquid Water Path to CCFs :

    #..Remove abnormal and missing_values, train model with TR sst>= TR_sst, unit in K
    X  = np.array( [dict2_predi_fla_PI['SST'][ind6], dict2_predi_fla_PI['p_e'][ind6], dict2_predi_fla_PI['LTS'][ind6], dict2_predi_fla_PI['SUB'][ind6]] )

    regr1 = linear_model.LinearRegression()
    result1 = regr1.fit(X.T, dict2_predi_fla_PI['LWP'][ind6] )   #..regression for LWP WITH LTS and skin-T >= TR_sst
    
    #..ADD dec7TH:
    regr1_iwp = linear_model.LinearRegression()
    result1_IWP = regr1_iwp.fit(X.T, dict2_predi_fla_PI['IWP'][ind6] )   #..regression for Ice Water Path with having 'LTS' values
    #print('result1 coef: ', result1_iwp.coef_)
    #print('result1 intercept: ', result1_IWP.intercept_)

    #..Save them into rawdata_dict
    aeffi  = result1.coef_
    aint   = result1.intercept_
    
    aeffi_iwp  = result1_IWP.coef_
    aint_iwp  = result1_IWP.intercept_

    '''
    #..for test
    a = load('sensitivity_4ccfs_ipsl270.npy')
    b = load('intercept1_ipsl270.npy')
    print('270K coef for IPSL: ', a)
    print('270K intercept for IPSL: ', b)
    '''

    #..Remove abnormal and missing values , train model with TR sst < Tr_SST.K

    XX = np.array([dict2_predi_fla_PI['SST'][ind7], dict2_predi_fla_PI['p_e'][ind7], dict2_predi_fla_PI['LTS'][ind7], dict2_predi_fla_PI['SUB'][ind7]])
    if len(ind7)!=0:
        regr2=linear_model.LinearRegression()
        result2 = regr2.fit(XX.T, dict2_predi_fla_PI['LWP'][ind7])   #..regression for LWP WITH LTS and skin-T < TR_sst

        beffi  = result2.coef_
        bint   = result2.intercept_
        
        regr2_iwp = linear_model.LinearRegression()
        result2_IWP = regr2_iwp.fit(XX.T, dict2_predi_fla_PI['IWP'][ind7])

        beffi_iwp =  result2_IWP.coef_
        bint_iwp  =   result2_IWP.intercept_
        
    else:
        beffi  = full(4, 0.0)
        bint   = 0.0
        
        beffi_iwp =  full(4, 0.0)
        bint_iwp = 0.0


    #print('result2 coef: ', result2.coef_)
    #print('result2 intercept: ', result2.intercept_)

    #..save them into rawdata_dict
    C_dict['LRM_le'] = (aeffi, aint)
    C_dict['LRM_st'] = (beffi, bint)
    
    C_dict['LRM_le_forIWP']= (aeffi_iwp, aint_iwp)
    C_dict['LRM_st_forIWP'] =(beffi_iwp, bint_iwp)
    
    # Regression for pi VALUES:
    sstlelwp_predi = dot(aeffi.reshape(1, -1),  X)  + aint   #..larger or equal than Tr_SST
    sstltlwp_predi = dot(beffi.reshape(1, -1), XX)  + bint   #..less than Tr_SST
    
    
    sstleiwp_predi  = dot(aeffi_iwp.reshape(1,-1), X) + aint_iwp
    sstltiwp_predi  = dot(beffi_iwp.reshape(1,-1), XX) + bint_iwp
    
    # emsemble into 'YB' predicted data array for Pi:
    YB[ind6] = sstlelwp_predi
    YB[ind7] = sstltlwp_predi
    
    YB_iwp[ind6]  = sstleiwp_predi
    YB_iwp[ind7] =  sstltiwp_predi
    
    # 'YB' resample into the shape of 'LWP_yr_bin':
    C_dict['LWP_predi_bin_PI']   =  array(YB).reshape(shape_yr_PI_3)
    print('  predicted LWP array for PI, shape in ',  C_dict['LWP_predi_bin_PI'].shape)
    
    C_dict['IWP_predi_bin_PI']  = array(YB_iwp).reshape(shape_yr_PI_3)
    print('predicted IWP array for PI, shape in ',  C_dict['IWP_predi_bin_PI'].shape)
    
    #.. Test performance
    MSE_shape6 =  mean_squared_error(dict2_predi_fla_PI['LWP'][ind6].reshape(-1,1), sstlelwp_predi.reshape(-1,1))
    print('RMSE_shape6(PI): ', sqrt(MSE_shape6))
    MSE_shape6_IWP = mean_squared_error(dict2_predi_fla_PI['IWP'][ind6].reshape(-1,1), sstleiwp_predi.reshape(-1,1))
    R_2_shape6_IWP = r2_score(dict2_predi_fla_PI['IWP'][ind6].reshape(-1,1), sstleiwp_predi.reshape(-1, 1))
    if len(ind7)!=0:
        R_2_shape7  = r2_score(dict2_predi_fla_PI['LWP'][ind7].reshape(-1, 1), sstltlwp_predi.reshape(-1, 1))
        R_2_shape7_IWP = r2_score(dict2_predi_fla_PI['IWP'][ind7].reshape(-1,1), sstltiwp_predi.reshape(-1, 1))
        
        print('R_2_shape7 for LWP: ', R_2_shape7)
        
    else:
        R_2_shape7  = 0.0
        R_2_shape7_IWP  = 0.0
        
        print('R_2_shape7 = \'0\' because Tr_sst <= all available T_skin data')

    MSE_shape1 =  mean_squared_error(dict2_predi_fla_PI['LWP'].reshape(-1,1), YB.reshape(-1,1))
    print('RMSE_shape1 for LWP: ', sqrt(MSE_shape1))

    R_2_shape1  = r2_score(dict2_predi_fla_PI['LWP'].reshape(-1, 1), YB.reshape(-1, 1))
    print('R_2_shape1 for LWP: ', R_2_shape1)
    
    R_2_shape1_IWP  = r2_score(dict2_predi_fla_PI['IWP'].reshape(-1, 1), YB_iwp.reshape(-1, 1))

    print('examine regres-mean Lwp for pi-C shape6:', nanmean(dict2_predi_fla_PI['LWP'][ind6]), nanmean(sstlelwp_predi))
    print('examine regres-mean Lwp for pi-C shape7:', nanmean(dict2_predi_fla_PI['LWP'][ind7]), nanmean(sstltlwp_predi))
    print('examine regres-mean IWP for pi-C shape1:', mean(dict2_predi_fla_PI['IWP']) , mean(YB_iwp))
    
    
    
    # ABR
    
    shape_fla_abr   =  dict2_predi_fla_abr['LWP'].shape
    print(dict2_predi_fla_abr['p_e'].shape)  #..compare with the following line

    #..Subtract 'nan' in data, shape1_abr -> shape2_abr points
    ind1_abr =  isnan(dict2_predi_fla_abr['LTS'])==False
    print('shape1_abr :', ind1_abr.shape)
    
    ind_true_abr =  nonzero(ind1_abr ==True)   #..Sign the the indexing of 'Non-NaN' in LTS_yr_bin
    print('shape2_abr :', array(ind_true_abr).shape, dict2_predi_fla_abr['LTS'][ind_true_abr].shape)

    ind_false_abr =  nonzero(ind1_abr==False)   #..Sign the the indexing of 'NaN'
    #dict1_yr_bin_abr[ind_false_abr] = 0.0
    
    
    #..Split the abrupt4x data points with TR_sst 
    

    ind_sst_le_abr  = nonzero(dict2_predi_fla_abr['SST'] >= TR_sst)
    ind6_abr  = intersect1d(ind_true_abr, ind_sst_le_abr)
    print('shape6_abr: ', ind6_abr.shape)   #..shape6_abr
    
    
    ind_sst_st_abr  = nonzero(dict2_predi_fla_abr['SST'] < TR_sst)
    ind7_abr =  intersect1d(ind_true_abr, ind_sst_st_abr)
    print('shape7_abr: ', ind7_abr.shape)  #..shape7_abr points

    
    #..designate LWP single-array's value, abr
    YB_abr   =  full((shape_fla_abr),  0.0)   # predicted LWP value array for future uses
    YB_abr[ind_false_abr] = dict2_predi_fla_abr['LWP'][ind_false_abr]   #..LWP single-column array with no LTS points as original values, with has LTS value points as 0.0. 
    print('YB_abr(raw abrupt4x LWP array: ', YB_abr)
    
    
    YB_abr_iwp   =  full((shape_fla_abr),  0.0)   # predicted LWP value array for future uses
    YB_abr_iwp[ind_false_abr] = dict2_predi_fla_abr['IWP'][ind_false_abr]   #..IWP 
    
    
    
    # Regression for abr LWP VALUES:
    # DIFFERENT LRM: LRM1: model with points that skin_T largerorequal to TR_sst
    
    X_abr   =  np.array([dict2_predi_fla_abr['SST'][ind6_abr], dict2_predi_fla_abr['p_e'][ind6_abr], dict2_predi_fla_abr['LTS'][ind6_abr], dict2_predi_fla_abr['SUB'][ind6_abr]])
    
    # LRM2: model with points that skin_T smaller than TR_sst
    
    XX_abr  =  np.array([dict2_predi_fla_abr['SST'][ind7_abr], dict2_predi_fla_abr['p_e'][ind7_abr], dict2_predi_fla_abr['LTS'][ind7_abr], dict2_predi_fla_abr['SUB'][ind7_abr]])
    
    
    sstlelwp_predi_abr = dot(aeffi.reshape(1, -1),  X_abr)  +  aint   #.. skin_T  larger or equal than Tr_SST
    sstltlwp_predi_abr = dot(beffi.reshape(1, -1),  XX_abr)  + bint   #.. skin_T  less than Tr_SST
    
    sstleiwp_predi_abr =  dot(aeffi_iwp.reshape(1,-1), X_abr) + aint_iwp   #.. SST larger or equal to than TR_sst
    sstltiwp_predi_abr =   dot(beffi_iwp.reshape(1,-1), XX_abr) + bint_iwp   #..SST smaller than
    
    
    # emsemble into 'YB_abr' predicted data array for Abrupt4xCO2:
    YB_abr[ind6_abr]  =   sstlelwp_predi_abr
    YB_abr[ind7_abr]  =   sstltlwp_predi_abr
    
    YB_abr_iwp[ind6_abr]  =  sstleiwp_predi_abr
    YB_abr_iwp[ind7_abr]  =  sstltiwp_predi_abr
    
    # 'YB' reshaple into the shape of 'LWP_yr_bin_abr':
    C_dict['LWP_predi_bin_abr']   =  array(YB_abr).reshape(shape_yr_abr_3)
    C_dict['IWP_predi_bin_abr']  =   array(YB_abr_iwp).reshape(shape_yr_abr_3)
    
    print(' predicted LWP array for abrupt4xCO2, shape in ',  C_dict['LWP_predi_bin_abr'].shape)   
    
    
    # Test performance for abr(predict) set:
    MSE_shape1_abr = mean_squared_error(YB_abr.reshape(-1,1),  dict2_predi_fla_abr['LWP'].reshape(-1, 1))
    R_2_shape1_abr = r2_score(dict2_predi_fla_abr['LWP'].reshape(-1,1), YB_abr.reshape(-1, 1 ))
    MSE_shape1_abr_IWP  = mean_squared_error(YB_abr_iwp.reshape(-1,1), dict2_predi_fla_abr['IWP'].reshape(-1, 1))
    R_2_shape1_abr_IWP = r2_score(dict2_predi_fla_abr['IWP'].reshape(-1,1), YB_abr_iwp.reshape(-1, 1 ))
    
    print('RMSE_shape1_abr: ', sqrt(MSE_shape1_abr))
    print('R_2_shape1_abr: ', R_2_shape1_abr)
    
    
    # calc D(CCFs) to DGMT and save into 'Dx/DtG' ARRAY
    regr3 = linear_model.LinearRegression()
    re_LWP= regr3.fit(dict2_predi_fla_abr['gmt'][ind_true_abr].reshape(-1,1), dict2_predi_fla_abr['LWP'][ind_true_abr])
    print('D(LWP) /D(gmt) (with LTS POINTS) : ', re_LWP.coef_)
    print('b of D(LWP) /D(gmt) : ', re_LWP.intercept_)

    regr4 = linear_model.LinearRegression()
    re_IWP= regr4.fit(dict2_predi_fla_abr['gmt'][ind_true_abr].reshape(-1,1), dict2_predi_fla_abr['IWP'][ind_true_abr])
    regr5 = linear_model.LinearRegression()
    regr6 = linear_model.LinearRegression()
    regr7 = linear_model.LinearRegression()

    regr8 = linear_model.LinearRegression()

    re_SST = regr5.fit(dict2_predi_fla_abr['gmt'][ind_true_abr].reshape(-1,1), dict2_predi_fla_abr['SST'][ind_true_abr])
    re_p_e = regr6.fit(dict2_predi_fla_abr['gmt'][ind_true_abr].reshape(-1,1), dict2_predi_fla_abr['p_e'][ind_true_abr])
    re_LTS = regr7.fit(dict2_predi_fla_abr['gmt'][ind_true_abr].reshape(-1,1), dict2_predi_fla_abr['LTS'][ind_true_abr])

    re_SUB = regr8.fit(dict2_predi_fla_abr['gmt'][ind_true_abr].reshape(-1,1), dict2_predi_fla_abr['SUB'][ind_true_abr])
    print('d(CCFs)/d(gmt)| (has LTS VALUES)= ', re_SST.coef_, re_p_e.coef_, re_LTS.coef_,  re_SUB.coef_)

    #..save into rawdata_dict
    Dx_DtG =[re_LWP.coef_, re_IWP.coef_, re_SST.coef_,  re_p_e.coef_,  re_LTS.coef_,  re_SUB.coef_]
    C_dict['dX_dTg'] =  Dx_DtG
    
    
    
    #..save test performance metrics into rawdata_dict
    EXAMINE_metrics =  {'RMSE_shape1_pi': sqrt(MSE_shape1), 'R_2_shape1_pi': R_2_shape1, 'RMSE_shape6_pi': sqrt(MSE_shape6), 'R_2_shape7': R_2_shape7, \
                        'R_2_shape1_pi_IWP': R_2_shape1_IWP, 'RMSE_shape6_pi_IWP': sqrt(MSE_shape6_IWP), 'R_2_shape6_pi_IWP': R_2_shape6_IWP, 'R_2_shape7_pi_IWP': R_2_shape7_IWP, \
                        'RMSE_shape1_abr': sqrt(MSE_shape1_abr), 'R_2_shape1_abr': R_2_shape1_abr, \
                        'RMSE_shape1_abr_IWP': sqrt(MSE_shape1_abr_IWP), 'R_2_shape1_abr_IWP': R_2_shape1_abr_IWP}
    
    C_dict['EXAMINE_metrics'] = EXAMINE_metrics
    
    return C_dict





def p4plot1(rawdata_dict, s_range, y_range, x_range, shape_yr_pi, shape_yr_abr):
    # 's_range , 'y_range', 'x_range' used to do area mean for repeat gmt ARRAY
    
    # retriving datas from big dict...
    dict0_abr_var = rawdata_dict['dict0_abr_var']
    dict0_PI_var  = rawdata_dict['dict0_PI_var']
    #print(dict0_PI_var['times'])

    model = rawdata_dict['model_data']   #.. type in dict

    datavar_nas = ['LWP', 'TWP', 'IWP', 'PRW', 'SST', 'p_e', 'LTS', 'SUB']   #..8 varisables except gmt (lon dimension diff)

    # load annually-mean bin data
    dict1_yr_bin_PI  = dict0_PI_var['dict1_yr_bin_PI']
    dict1_yr_bin_abr  = dict0_abr_var['dict1_yr_bin_abr']
    
    # load monthly bin data
    ###
    # calc area-mean ARRAY FOR variables:
    
    areamean_dict_PI = {}
    areamean_dict_abr  = {}
    areamean_dict_predi =  {}
    
    for e in range(len(datavar_nas)):
        
        areamean_dict_PI[datavar_nas[e]+ '_area_yr'] =  area_mean(dict1_yr_bin_PI[datavar_nas[e]+ '_yr_bin'], y_range, x_range)
        areamean_dict_abr[datavar_nas[e]+ '_area_yr'] =  area_mean(dict1_yr_bin_abr[datavar_nas[e]+ '_yr_bin'], y_range, x_range)
      
    
    areamean_dict_predi['LWP_area_yr_pi']  =   area_mean(rawdata_dict['LWP_predi_bin_PI'], y_range, x_range)
    areamean_dict_predi['LWP_area_yr_abr']  =   area_mean(rawdata_dict['LWP_predi_bin_abr'], y_range, x_range)
    areamean_dict_predi['IWP_area_yr_pi']   =    area_mean(rawdata_dict['IWP_predi_bin_PI'], y_range, x_range)
    areamean_dict_predi['IWP_area_yr_abr']   =    area_mean(rawdata_dict['IWP_predi_bin_abr'], y_range, x_range)
    
    areamean_dict_PI['gmt_area_yr']  =  area_mean(dict1_yr_bin_PI['gmt_yr_bin'], s_range, x_range)
    areamean_dict_abr['gmt_area_yr']  =  area_mean(dict1_yr_bin_abr['gmt_yr_bin'], s_range, x_range)
    
    
    rawdata_dict['areamean_dict_predi'] =  areamean_dict_predi
    rawdata_dict['areamean_dict_abr']   =  areamean_dict_abr
    rawdata_dict['areamean_dict_PI']    =  areamean_dict_PI
    
    
    # genarate some array convenient for plotting
    #..Years from pi-control to abrupt4xCO2 experiment, which are choosed years
    Yrs =  arange(shape_yr_pi+shape_yr_abr)
    
    # Global-mean surface air temperature, from pi-control to abrupt4xCO2 experiment
    
    GMT =  full((shape_yr_pi+shape_yr_abr),  0.0)
    GMT[0:shape_yr_pi]  =   areamean_dict_PI['gmt_area_yr']
    GMT[shape_yr_pi:]  =   areamean_dict_abr['gmt_area_yr']
    
    # predicted values, from pi-Control to abrupt4xCO2 experiment
    
    predict_lwp  = full((shape_yr_pi + shape_yr_abr), 0.0)
    predict_lwp[0:shape_yr_pi]  =   areamean_dict_predi['LWP_area_yr_pi']
    predict_lwp[shape_yr_pi:]  =   areamean_dict_predi['LWP_area_yr_abr']
    
    predict_iwp  = full((shape_yr_pi + shape_yr_abr), 0.0)
    predict_iwp[0:shape_yr_pi]  =   areamean_dict_predi['IWP_area_yr_pi']
    predict_iwp[shape_yr_pi:]  =   areamean_dict_predi['IWP_area_yr_abr']
    
    # reported values, from pi-Conrol to abrupt4xCO2 experiment
    
    report_lwp  =   full((shape_yr_pi + shape_yr_abr), 0.0)
    report_lwp[0:shape_yr_pi]  =   areamean_dict_PI['LWP_area_yr']
    report_lwp[shape_yr_pi:]   =  areamean_dict_abr['LWP_area_yr']
    
    report_iwp  =   full((shape_yr_pi + shape_yr_abr), 0.0)
    report_iwp[0:shape_yr_pi]  =   areamean_dict_PI['IWP_area_yr']
    report_iwp[shape_yr_pi:]   =  areamean_dict_abr['IWP_area_yr']
    
    
    # put them into the rawdata_dict:
    rawdata_dict['Yrs']  = Yrs
    rawdata_dict['GMT']  =   GMT
    
    rawdata_dict['predict_lwp']  =  predict_lwp
    rawdata_dict['report_lwp']  =   report_lwp
    
    
    rawdata_dict['predict_iwp']  = predict_iwp
    rawdata_dict['report_iwp']   = report_iwp
    

    return rawdata_dict



def fitLRM2(C_dict, TR_sst, TR_sub, s_range, y_range, x_range):
    # 'C_dict' is the raw data dict, 'TR_sst' accompany with 'TR_sub' are the predefined skin_Temperature  / 500 mb Subsidence Thresholds to distinguish 4 diff LRMs:
    # 's_range , 'y_range', 'x_range' used to do area mean for repeat gmt ARRAY
    dict0_abr_var = C_dict['dict0_abr_var']
    dict0_PI_var  = C_dict['dict0_PI_var']
    #print(dict0_PI_var['times'])

    model = C_dict['model_data']   #.. type in dict

    datavar_nas = ['LWP', 'TWP', 'IWP', 'PRW', 'SST', 'p_e', 'LTS', 'SUB']   #..8 varisables except gmt (lon dimension diff) 

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

    
    dict2_predi_fla_PI = {}
    dict2_predi_fla_abr = {}
    
    dict2_predi_nor_PI = {}
    dict2_predi_nor_abr = {}
    
    #..Ravel binned array /Standardized data ARRAY :
    for d in range(len(datavar_nas)):
        dict2_predi_fla_PI[datavar_nas[d]] = dict1_yr_bin_PI[datavar_nas[d]+'_yr_bin'].flatten()
        dict2_predi_fla_abr[datavar_nas[d]] = dict1_yr_bin_abr[datavar_nas[d]+'_yr_bin'].flatten()

        # normalized the predict array
        dict2_predi_nor_PI[datavar_nas[d]] =  (dict2_predi_fla_PI[datavar_nas[d]] - nanmean(dict2_predi_fla_PI[datavar_nas[d]]) )/ nanstd(dict2_predi_fla_PI[datavar_nas[d]])
        dict2_predi_nor_abr[datavar_nas[d]] =  (dict2_predi_fla_abr[datavar_nas[d]] - nanmean(dict2_predi_fla_abr[datavar_nas[d]]) )/ nanstd(dict2_predi_fla_abr[datavar_nas[d]])

    #..Use area_mean method, 'np.repeat' and 'np.tile' to reproduce gmt area-mean Array as the same shape as other flattened variables:
    GMT_pi  = area_mean(dict1_yr_bin_PI['gmt_yr_bin'],  s_range,  x_range)   #..ALL in shape : shape_yr_abr(single dimension)
    dict2_predi_fla_PI['gmt']  = GMT_pi.repeat(730)
    GMT_abr  = area_mean(dict1_yr_bin_abr['gmt_yr_bin'], s_range, x_range)   #..ALL in shape : shape_yr_abr(single dimension)
    dict2_predi_fla_abr['gmt'] = GMT_abr.repeat(730)
    
    #dict2_predi_nor_PI['gmt']  =  (dict2_predi_fla_PI['gmt'] - nanmean(dict2_predi_fla_PI['gmt']) )/ nanstd(dict2_predi_fla_PI['gmt'])
    #dict2_predi_nor_abr['gmt'] =   (dict2_predi_fla_abr['gmt'] - nanmean(dict2_predi_fla_abr['gmt']) )/ nanstd(dict2_predi_fla_abr['gmt'])
    
    # save into rawdata_dict:
    C_dict['dict2_predi_fla_PI'] =  dict2_predi_fla_PI
    C_dict['dict2_predi_fla_abr'] = dict2_predi_fla_abr
    C_dict['dict2_predi_nor_PI'] =  dict2_predi_nor_PI
    C_dict['dict2_predi_nor_abr']  = dict2_predi_nor_abr
    
    print('shape1: ', dict2_predi_fla_PI['LWP'].shape)   # shape1
    shape_fla_PI   =   dict2_predi_fla_PI['LWP'].shape
    #print(min(dict2_predi_fla_PI['LTS']),  max(dict2_predi_fla_PI['LTS']) )
    
    
    # PI
    #..Subtract 'nan' in data, shape1 -> shape2(without 'nan' number) points and shape5('nan' number)

    ind1 = isnan(dict2_predi_fla_PI['LTS'])==False 

    ind_true = nonzero(ind1==True)
    ind_false = nonzero(ind1==False)
    #..Sign the the indexing into YB, or YB value will have a big changes
    print('shape2: ', array(ind_true).shape)        # shape2
    #  print(argwhere(isnan(dict2_predi_fla_PI['LTS'][ind_true])==True))


    #..Split data points with skin Temperature < / >=TR_sst and Subsidence500 <= / > TR_sub(upward motion / downward motion): 

    # shape1 split into shape3(smaller.TR_sst & up)\shape4(larger.equal.TR_sst & up)\shape5(smaller.TR_sst & down)\shape7(larger.equal.TR_sst & down)
    ind_sstlt_up =  nonzero( (dict2_predi_fla_PI['SST'] < TR_sst) & (dict2_predi_fla_PI['SUB'] <= TR_sub))
    ind_sstle_up =  nonzero( (dict2_predi_fla_PI['SST'] >= TR_sst) & (dict2_predi_fla_PI['SUB'] <= TR_sub))
    ind_sstlt_dw  = nonzero((dict2_predi_fla_PI['SST'] < TR_sst) &  (dict2_predi_fla_PI['SUB'] >  TR_sub))
    ind_sstle_dw  = nonzero((dict2_predi_fla_PI['SST'] >= TR_sst) & (dict2_predi_fla_PI['SUB'] >  TR_sub))
    
    # shape7:the intersection of places where has LTS value and skin_T < TR_sst & SUB500 <= TR_sub
    ind7  = intersect1d(ind_true, ind_sstlt_up)
    print('shape7: ', ind7.shape)   #.. points, shape7
    # shape8:the intersection of places where LTS value and skin_T >= TR_sst & SUB500 <= TR_sub
    ind8 = intersect1d(ind_true, ind_sstle_up)
    print('shape8: ', ind8.shape)   #.. points, shape8

    # shape9:the intersection of places where has LTS value and skin_T < TR_sst & SUB500 > TR_sub
    ind9  = intersect1d(ind_true, ind_sstlt_dw)
    print('shape9: ', ind9.shape)   #.. points, shape9
    # shape10:the intersection of places where LTS value and skin_T >= TR_sst & SUB500 > TR_sub                                                                         
    ind10 =  intersect1d(ind_true, ind_sstle_dw)
    print('shape10: ', ind10.shape)              
    
    #..designate LWP single-array's value, PI
    YB =  full((shape_fla_PI), 0.0)
    YB[ind_false] =  dict2_predi_fla_PI['LWP'][ind_false]   #..LWP single-column array with no LTS points as original values, with has LTS value points as 0.0.
    
    #..designate IWP single-array's value, PI
    YB_iwp =  full((shape_fla_PI), 0.0)
    YB_iwp[ind_false] =  dict2_predi_fla_PI['IWP'][ind_false]


    #.. Multiple linear regreesion of Liquid Water Path to CCFs :

    #..Remove abnormal and missing_values, train model with different TR_sst and TR_sub regimes data
    XX_7  = array( [dict2_predi_fla_PI['SST'][ind7], dict2_predi_fla_PI['p_e'][ind7], dict2_predi_fla_PI['LTS'][ind7], dict2_predi_fla_PI['SUB'][ind7]] )
    XX_8  = array( [dict2_predi_fla_PI['SST'][ind8], dict2_predi_fla_PI['p_e'][ind8], dict2_predi_fla_PI['LTS'][ind8], dict2_predi_fla_PI['SUB'][ind8]] )
    XX_9  = array( [dict2_predi_fla_PI['SST'][ind9], dict2_predi_fla_PI['p_e'][ind9], dict2_predi_fla_PI['LTS'][ind9], dict2_predi_fla_PI['SUB'][ind9]] )
    XX_10 = array( [dict2_predi_fla_PI['SST'][ind10], dict2_predi_fla_PI['p_e'][ind10], dict2_predi_fla_PI['LTS'][ind10], dict2_predi_fla_PI['SUB'][ind10]] )
    
    
    if (len(ind7)!=0) & (len(ind8)!=0) & (len(ind9)!=0) & (len(ind10)!=0):
        regr7=linear_model.LinearRegression()
        result7 = regr7.fit(XX_7.T, dict2_predi_fla_PI['LWP'][ind7])   #..regression for LWP WITH LTS and skin-T < TR_sst & 'up'
        aeffi  = result7.coef_
        aint   = result7.intercept_
        
        regr7_iwp = linear_model.LinearRegression()
        result7_IWP =regr7_iwp.fit(XX_7.T, dict2_predi_fla_PI['IWP'][ind7])
        aeffi_iwp =  result7_IWP.coef_
        aint_iwp  =  result7_IWP.intercept_
        
        regr8=linear_model.LinearRegression()
        result8 = regr8.fit(XX_8.T, dict2_predi_fla_PI['LWP'][ind8])   #..regression for LWP WITH LTS and skin-T >= TR_sst &'up'
        beffi  = result8.coef_
        bint   = result8.intercept_
        
        regr8_iwp = linear_model.LinearRegression()
        result8_IWP =regr8_iwp.fit(XX_8.T, dict2_predi_fla_PI['IWP'][ind8])
        beffi_iwp =  result8_IWP.coef_
        bint_iwp  =  result8_IWP.intercept_

        regr9=linear_model.LinearRegression()
        result9 = regr9.fit(XX_9.T, dict2_predi_fla_PI['LWP'][ind9])   #..regression for LWP WITH LTS and skin-T < TR_sst & 'down'
        ceffi  = result9.coef_
        cint   = result9.intercept_
        
        regr9_iwp = linear_model.LinearRegression()
        result9_IWP =regr9_iwp.fit(XX_9.T, dict2_predi_fla_PI['IWP'][ind9])
        ceffi_iwp =  result9_IWP.coef_
        cint_iwp  =  result9_IWP.intercept_
        
        regr10=linear_model.LinearRegression()
        result10 = regr10.fit(XX_10.T, dict2_predi_fla_PI['LWP'][ind10])   #..regression for LWP WITH LTS and skin-T >= TR_sst & 'down'
        deffi  = result10.coef_
        dint   = result10.intercept_
        
        regr10_iwp = linear_model.LinearRegression()
        result10_IWP =regr10_iwp.fit(XX_10.T, dict2_predi_fla_PI['IWP'][ind10])
        deffi_iwp =  result10_IWP.coef_
        dint_iwp  =  result10_IWP.intercept_
    
    elif (len(ind7)==0) & (len(ind9)==0): 
        aeffi  = full(4, 0.0)
        aint   = 0.0
        aeffi_iwp =  full(4, 0.0)
        aint_iwp = 0.0
        
        regr8=linear_model.LinearRegression()
        result8 = regr8.fit(XX_8.T, dict2_predi_fla_PI['LWP'][ind8])   #..regression for LWP WITH LTS and skin-T >= TR_sst &'up'
        beffi  = result8.coef_
        bint   = result8.intercept_
        regr8_iwp = linear_model.LinearRegression()
        result8_IWP =regr8_iwp.fit(XX_8.T, dict2_predi_fla_PI['IWP'][ind8])
        beffi_iwp =  result8_IWP.coef_
        bint_iwp  =  result8_IWP.intercept_
        
        ceffi  = full(4, 0.0)
        cint   = 0.0
        ceffi_iwp =  full(4, 0.0)
        cint_iwp = 0.0
        
        regr10=linear_model.LinearRegression()
        result10 = regr10.fit(XX_10.T, dict2_predi_fla_PI['LWP'][ind10])   #..regression for LWP WITH LTS and skin-T >= TR_sst& 'down'
        deffi  = result10.coef_
        dint   = result10.intercept_
        regr10_iwp = linear_model.LinearRegression()
        result10_IWP =regr10_iwp.fit(XX_10.T, dict2_predi_fla_PI['IWP'][ind10])
        deffi_iwp =  result10_IWP.coef_
        dint_iwp  =  result10_IWP.intercept_
    
    else:
        print('you input a non-wise value for TR_sub at 500 mb')
        print('please try another TR_sub input...')


    #..save them into rawdata_dict
    C_dict['LRM_sstlt_up'] =  (aeffi, aint)
    C_dict['LRM_sstle_up']  = (beffi, bint)
    C_dict['LRM_sstlt_down'] = (ceffi, cint)
    C_dict['LRM_sstle_down'] = (deffi, dint)
    
    C_dict['LRM_sstlt_up_forIWP']= (aeffi_iwp, aint_iwp)
    C_dict['LRM_sstle_up_forIWP'] =(beffi_iwp, bint_iwp)
    C_dict['LRM_sstlt_down_forIWP']= (ceffi_iwp, cint_iwp)
    C_dict['LRM_sstle_down_forIWP'] =(deffi_iwp, dint_iwp)
    
    #..print the coeffieients:
    print('aeffi(up, cold): ', aeffi, 'aint: ', aint)
    print('beffi(up, warm): ', beffi, 'bint: ', bint)
    print('ceffi(down, cold): ', ceffi, 'cint: ', cint)
    print('deffi(Down,Warm): ', deffi, 'dint: ', dint)
    
    ## Regression for pi VALUES:
    sstle_uplwp_predi =  dot(beffi.reshape(1, -1), XX_8)  + bint   #..larger or equal than Tr_SST & SUB at 500 <= TR_sub
    sstlt_uplwp_predi =  dot(aeffi.reshape(1, -1), XX_7)  + aint   #..less than Tr_SST & SUB at 500 <= TR_sub
    sstlt_dwlwp_predi =  dot(ceffi.reshape(1, -1), XX_9)  + cint   #..less than Tr_SST & SUB at 500 > TR_sub
    sstle_dwlwp_predi =  dot(deffi.reshape(1, -1), XX_10)  + dint   #..larger or equal than Tr_SST & SUB at 500 > TR_sub
    
    sstle_upiwp_predi  = dot(beffi_iwp.reshape(1,-1), XX_8) + bint_iwp
    sstlt_upiwp_predi  = dot(aeffi_iwp.reshape(1,-1), XX_7) + aint_iwp 
    sstlt_dwiwp_predi  = dot(ceffi_iwp.reshape(1,-1), XX_9) + cint_iwp
    sstle_dwiwp_predi  = dot(deffi_iwp.reshape(1,-1), XX_10) + dint_iwp
    
    # emsembling into 'YB' predicted data array for Pi:
    YB[ind7] = sstlt_uplwp_predi
    YB[ind8] = sstle_uplwp_predi
    YB[ind9] = sstlt_dwlwp_predi
    YB[ind10] =  sstle_dwlwp_predi

    
    YB_iwp[ind7]  = sstlt_upiwp_predi
    YB_iwp[ind8] =  sstle_upiwp_predi
    YB_iwp[ind9]  = sstlt_dwiwp_predi
    YB_iwp[ind10]  =  sstle_dwiwp_predi
    
    
    # 'YB' resample into the shape of 'LWP_yr_bin':
    C_dict['LWP_predi_bin_PI']   =  array(YB).reshape(shape_yr_PI_3)
    #print('predicted LWP array for PI, shape in ',  C_dict['LWP_predi_bin_PI'].shape)
    
    C_dict['IWP_predi_bin_PI']  = array(YB_iwp).reshape(shape_yr_PI_3)
    #print('predicted IWP array for PI, shape in ',  C_dict['IWP_predi_bin_PI'].shape)
    
    '''
    #.. Test performance
    MSE_shape6 =  mean_squared_error(dict2_predi_fla_PI['LWP'][ind6].reshape(-1,1), sstlelwp_predi.reshape(-1,1))
    print('RMSE_shape6(PI): ', sqrt(MSE_shape6))
    MSE_shape6_IWP = mean_squared_error(dict2_predi_fla_PI['IWP'][ind6].reshape(-1,1), sstleiwp_predi.reshape(-1,1))
    R_2_shape6_IWP = r2_score(dict2_predi_fla_PI['IWP'][ind6].reshape(-1,1), sstleiwp_predi.reshape(-1, 1))
    if len(ind7)!=0:
        R_2_shape7  = r2_score(dict2_predi_fla_PI['LWP'][ind7].reshape(-1, 1), sstltlwp_predi.reshape(-1, 1))
        R_2_shape7_IWP = r2_score(dict2_predi_fla_PI['IWP'][ind7].reshape(-1,1), sstltiwp_predi.reshape(-1, 1))
        
        print('R_2_shape7 for LWP: ', R_2_shape7)
        
    else:
        R_2_shape7  = 0.0
        R_2_shape7_IWP  = 0.0
        
        print('R_2_shape7 = \'0\' because Tr_sst <= all available T_skin data')
    '''
    MSE_shape1 =  mean_squared_error(dict2_predi_fla_PI['LWP'].reshape(-1,1), YB.reshape(-1,1))
    print('RMSE_shape1 for PI exp lwp: ', sqrt(MSE_shape1))

    R_2_shape1  = r2_score(dict2_predi_fla_PI['LWP'].reshape(-1, 1), YB.reshape(-1,1))
    print('R_2_shape1 for PI exp lwp: ', R_2_shape1)
    
    R_2_shape1_IWP  = r2_score(dict2_predi_fla_PI['IWP'].reshape(-1, 1), YB_iwp.reshape(-1,1))

    
    # Examine the effectiveness of regression model:
    print('examine regres-mean LwP for pi-C shape1:', nanmean(dict2_predi_fla_PI['LWP']), nanmean(YB))
    print('examine regres-mean LwP for pi-C shape10:', nanmean(dict2_predi_fla_PI['LWP'][ind10]), nanmean(sstle_dwlwp_predi))
    
    #..print('examine regres-mean IWP for pi-C shape1:', mean(dict2_predi_fla_PI['IWP']) , mean(YB_iwp))
    
                 
    # processing ABR data
    shape_fla_abr   =  dict2_predi_fla_abr['LWP'].shape
    print(dict2_predi_fla_abr['p_e'].shape)  #..compare with the following line

    #..Subtract 'nan' in data, shape1_abr -> shape2_abr points
    ind1_abr =  isnan(dict2_predi_fla_abr['LTS'])==False
    print('shape1_abr :', ind1_abr.shape)
    
    ind_true_abr =  nonzero(ind1_abr ==True)   #..Sign the the indexing of 'Non-NaN' in LTS_yr_bin
    print('shape2_abr :', array(ind_true_abr).shape,  dict2_predi_fla_abr['LTS'][ind_true_abr].shape)
    ind_false_abr = nonzero(ind1_abr==False)
    
    
    
    #..Split the abrupt4x data points with TR_sst and TR_sub
    ind_sstlt_up_abr =  nonzero( (dict2_predi_fla_abr['SST']< TR_sst) &  (dict2_predi_fla_abr['SUB'] <= TR_sub))
    ind_sstle_up_abr =  nonzero( (dict2_predi_fla_abr['SST']>= TR_sst) &  (dict2_predi_fla_abr['SUB'] <= TR_sub))
    ind_sstlt_dw_abr  = nonzero( (dict2_predi_fla_abr['SST']< TR_sst) & (dict2_predi_fla_abr['SUB'] >  TR_sub))
    ind_sstle_dw_abr  = nonzero( (dict2_predi_fla_abr['SST'] >= TR_sst) & (dict2_predi_fla_abr['SUB'] >  TR_sub))
    
    ind7_abr  = intersect1d(ind_true_abr, ind_sstlt_up_abr)
    print('shape7_abr: ', ind7_abr.shape)   #..shape7_abr
    
    ind8_abr  = intersect1d(ind_true_abr, ind_sstle_up_abr)
    print('shape8_abr: ', ind8_abr.shape)   #..shape8_abr
    
    ind9_abr  = intersect1d(ind_true_abr, ind_sstlt_dw_abr)
    print('shape9_abr: ', ind9_abr.shape)   #..shape9_abr
    
    ind10_abr  = intersect1d(ind_true_abr, ind_sstle_dw_abr)
    print('shape10_abr: ', ind10_abr.shape)   #..shape10_abr
    #..designate LWP single-array's value, abr
    YB_abr   =  full((shape_fla_abr),  0.0)   # predicted LWP value array for future uses
    YB_abr[ind_false_abr] = dict2_predi_fla_abr['LWP'][ind_false_abr]   #..LWP single-column array with no LTS points as original values, with has LTS value points as 0.0. 
    print('YB_abr(raw abrupt4x LWP array: ', YB_abr)
    print('actual LWP array: ', dict2_predi_fla_abr['LWP'])
    
    YB_abr_iwp   =  full((shape_fla_abr),  0.0)   # predicted LWP value array for future uses
    YB_abr_iwp[ind_false_abr] = dict2_predi_fla_abr['IWP'][ind_false_abr]   #..IWP 
    
    
    
    # Regression for abr LWP VALUES:    
    XX_7_abr  =  np.array([dict2_predi_fla_abr['SST'][ind7_abr], dict2_predi_fla_abr['p_e'][ind7_abr], dict2_predi_fla_abr['LTS'][ind7_abr], dict2_predi_fla_abr['SUB'][ind7_abr]])
    XX_8_abr  =  np.array([dict2_predi_fla_abr['SST'][ind8_abr], dict2_predi_fla_abr['p_e'][ind8_abr], dict2_predi_fla_abr['LTS'][ind8_abr], dict2_predi_fla_abr['SUB'][ind8_abr]])
    XX_9_abr  =  np.array([dict2_predi_fla_abr['SST'][ind9_abr], dict2_predi_fla_abr['p_e'][ind9_abr], dict2_predi_fla_abr['LTS'][ind9_abr], dict2_predi_fla_abr['SUB'][ind9_abr]])
    XX_10_abr  =  np.array([dict2_predi_fla_abr['SST'][ind10_abr], dict2_predi_fla_abr['p_e'][ind10_abr], dict2_predi_fla_abr['LTS'][ind10_abr], dict2_predi_fla_abr['SUB'][ind10_abr]])
    
    sstlt_uplwp_predi_abr = dot(aeffi.reshape(1, -1),  XX_7_abr)  +  aint
    sstle_uplwp_predi_abr = dot(beffi.reshape(1, -1),  XX_8_abr)  +  bint
    sstlt_dwlwp_predi_abr = dot(ceffi.reshape(1, -1),  XX_9_abr)  +  cint
    sstle_dwlwp_predi_abr = dot(deffi.reshape(1, -1), XX_10_abr)  +  dint
    
    
    sstlt_upiwp_predi_abr =  dot(aeffi_iwp.reshape(1,-1) , XX_7_abr) + aint_iwp   #.. SST smaller than TR_sst & 'up'
    sstle_upiwp_predi_abr =  dot(beffi_iwp.reshape(1,-1) , XX_8_abr) + bint_iwp   #..SST larger or equal to TR_sst & 'up'
    sstlt_dwiwp_predi_abr =  dot(ceffi_iwp.reshape(1,-1) , XX_9_abr) + cint_iwp   #.. SST smaller than TR_sst & 'down'
    sstle_dwiwp_predi_abr =  dot(deffi_iwp.reshape(1,-1) , XX_10_abr) + dint_iwp   #..SST larger or equal to TR_sst & 'down'
    
    


    #    emsembling into 'YB_abr' predicted data array for 'abrupt4xCO2'
    YB_abr[ind7_abr]  =   sstlt_uplwp_predi_abr
    YB_abr[ind8_abr]  =   sstle_uplwp_predi_abr
    YB_abr[ind9_abr]  =   sstlt_dwlwp_predi_abr
    YB_abr[ind10_abr]  =   sstle_dwlwp_predi_abr

    YB_abr_iwp[ind7_abr]  =  sstlt_upiwp_predi_abr
    YB_abr_iwp[ind8_abr]  =  sstle_upiwp_predi_abr
    YB_abr_iwp[ind9_abr]  =  sstlt_dwiwp_predi_abr
    YB_abr_iwp[ind10_abr]  =  sstle_dwiwp_predi_abr
    
    
    #    reshaple YB_abr into the shape of 'LWP_yr_bin_abr':

    C_dict['LWP_predi_bin_abr']   =  array(YB_abr).reshape(shape_yr_abr_3)
    C_dict['IWP_predi_bin_abr']  =   array(YB_abr_iwp).reshape(shape_yr_abr_3)
    
    #..print(' predicted LWP array for abrupt4xCO2, shape in ',  C_dict['LWP_predi_bin_abr'].shape)   
    
    
    # Test performance for abr(predict) set:
    MSE_shape1_abr = mean_squared_error(YB_abr.reshape(-1,1),  dict2_predi_fla_abr['LWP'].reshape(-1, 1))
    R_2_shape1_abr = r2_score(dict2_predi_fla_abr['LWP'].reshape(-1,1), YB_abr.reshape(-1, 1 ))
    MSE_shape1_abr_IWP  = mean_squared_error(YB_abr_iwp.reshape(-1,1), dict2_predi_fla_abr['IWP'].reshape(-1, 1))   #..IWP
    R_2_shape1_abr_IWP = r2_score(dict2_predi_fla_abr['IWP'].reshape(-1,1), YB_abr_iwp.reshape(-1, 1 ))     #..IWP
    print('RMSE_shape1 for ABR LWP: ', sqrt(MSE_shape1_abr))
    print('R_2_shape1 for abr LWP: ', R_2_shape1_abr)
    
    
    # calc D(CCFs) to DGMT and save into 'Dx/DtG' ARRAY
    regr11 = linear_model.LinearRegression()
    re_LWP= regr11.fit(dict2_predi_fla_abr['gmt'][ind_true_abr].reshape(-1,1), dict2_predi_fla_abr['LWP'][ind_true_abr])
    print(' D(LWP) /D(gmt) (has LTS POINTS) : ', re_LWP.coef_)
    print(' b of D(LWP) /D(gmt) : ', re_LWP.intercept_)

    regr11_iwp = linear_model.LinearRegression()
    re_IWP= regr11_iwp.fit(dict2_predi_fla_abr['gmt'][ind_true_abr].reshape(-1,1), dict2_predi_fla_abr['IWP'][ind_true_abr])

    regr12 = linear_model.LinearRegression()
    re_SST = regr12.fit(dict2_predi_fla_abr['gmt'][ind_true_abr].reshape(-1,1), dict2_predi_fla_abr['SST'][ind_true_abr])

    regr13 = linear_model.LinearRegression()
    re_p_e = regr13.fit(dict2_predi_fla_abr['gmt'][ind_true_abr].reshape(-1,1), dict2_predi_fla_abr['p_e'][ind_true_abr])

    regr14 = linear_model.LinearRegression()
    re_LTS = regr14.fit(dict2_predi_fla_abr['gmt'][ind_true_abr].reshape(-1,1), dict2_predi_fla_abr['LTS'][ind_true_abr])

    regr15 = linear_model.LinearRegression()
    re_SUB = regr15.fit(dict2_predi_fla_abr['gmt'][ind_true_abr].reshape(-1,1), dict2_predi_fla_abr['SUB'][ind_true_abr])
    
    
    
    print('d(CCFs)/d(gmt)| (has LTS VALUES)= ', re_SST.coef_, re_p_e.coef_, re_LTS.coef_,  re_SUB.coef_)
    
    
    #..save into rawdata_dict
    Dx_DtG =[re_LWP.coef_, re_IWP.coef_,  re_SST.coef_,  re_p_e.coef_,  re_LTS.coef_,  re_SUB.coef_]
    C_dict['dX_dTg'] =  Dx_DtG



    #..save test performance metrics into rawdata_dict
    EXAMINE_metrics =  {'RMSE_shape1_pi': sqrt(MSE_shape1), 'R_2_shape1_pi': R_2_shape1, \
                        'R_2_shape1_pi_IWP': R_2_shape1_IWP, \
                        'RMSE_shape1_abr': sqrt(MSE_shape1_abr), 'R_2_shape1_abr': R_2_shape1_abr, \
                        'RMSE_shape1_abr_IWP': sqrt(MSE_shape1_abr_IWP), 'R_2_shape1_abr_IWP': R_2_shape1_abr_IWP }
    
    C_dict['EXAMINE_metrics'] = EXAMINE_metrics
    
    return C_dict
