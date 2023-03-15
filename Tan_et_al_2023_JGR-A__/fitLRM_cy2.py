# ## training AND predicting the data by Linear Regession Model(LRM) of 2 and 4 regimes; ###
# ## estimate their statistic performance (RMSE/ R^2); ###


import netCDF4
from numpy import *
import matplotlib.pyplot as plt
import xarray as xr
# import PyNIO as Nio   # deprecated
import pandas as pd
import glob
from scipy.stats import *
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
# For statistics. Requires statsmodels 5.0 or more
from statsmodels.formula.api import ols
# Analysis of Variance (ANOVA) on linear models
from statsmodels.stats.anova import anova_lm
from area_mean import *
from binned_cyFunctions5 import *
from read_hs_file import read_var_mod
from useful_func_cy import *



def fitLRM3(C_dict, TR_sst, s_range, y_range, x_range, lats, lons):
    # 'C_dict' is the raw data dict, 'TR_sst' is the pre-defined skin_Temperature Threshold to distinguish two Multi-Linear Regression Models

    # 's_range , 'y_range', 'x_range' used to do area mean for repeat gmt ARRAY

    dict1_abr_var = C_dict['dict1_abr_var']
    dict1_PI_var = C_dict['dict1_PI_var']
    #print(dict0_PI_var['times'])

    model = C_dict['model_data']   #.. type in dict

    datavar_nas = ['LWP', 'TWP', 'IWP', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'SST', 'p_e', 'LTS', 'SUB']   #..12 varisables except gmt (lon dimension  diff)

    # load annually-mean bin data.
    dict1_yr_bin_PI = dict1_PI_var['dict1_yr_bin_PI']
    dict1_yr_bin_abr = dict1_abr_var['dict1_yr_bin_abr']
    #print(dict1_yr_bin_PI['LWP_yr_bin'].shape)
    
    # load monthly bin data.
    dict1_mon_bin_PI = dict1_PI_var['dict1_mon_bin_PI']
    dict1_mon_bin_abr = dict1_abr_var['dict1_mon_bin_abr']

    # data array in which shapes?
    shape_yr_PI = dict1_yr_bin_PI['LWP_yr_bin'].shape
    shape_yr_abr = dict1_yr_bin_abr['LWP_yr_bin'].shape

    shape_yr_PI_gmt = dict1_yr_bin_PI['gmt_yr_bin'].shape
    shape_yr_abr_gmt = dict1_yr_bin_abr['gmt_yr_bin'].shape

    shape_mon_PI = dict1_mon_bin_PI['LWP_mon_bin'].shape
    shape_mon_abr = dict1_mon_bin_abr['LWP_mon_bin'].shape

    shape_mon_PI_gmt = dict1_mon_bin_PI['gmt_mon_bin'].shape
    shape_mon_abr_gmt = dict1_mon_bin_abr['gmt_mon_bin'].shape

    #.. archieve the 'shape' infos: 3-D
    C_dict['shape_yr_PI_3'] = shape_yr_PI
    C_dict['shape_yr_abr_3'] = shape_yr_abr
    C_dict['shape_yr_PI_gmt_3'] = shape_yr_PI_gmt
    C_dict['shape_yr_abr_gmt_3'] = shape_yr_abr_gmt

    C_dict['shape_mon_PI_3'] = shape_mon_PI
    C_dict['shape_mon_abr_3'] = shape_mon_abr
    C_dict['shape_mon_PI_gmt_3'] = shape_mon_PI_gmt
    C_dict['shape_mon_abr_gmt_3'] = shape_mon_abr_gmt

    # flatten the data array for 'training' lrm coefficient
    
    dict2_predi_fla_PI = {}
    dict2_predi_fla_abr = {}
    
    dict2_predi_ano_PI = {}  # need a climatological arrays of variables
    dict2_predi_ano_abr = {}  # need a climatological arrays of variables
    
    dict2_predi_nor_PI = {}  # standardized anomalies of variables
    dict2_predi_nor_abr = {}
    
    dict2_predi = {}
    
    #.. Ravel binned array /Standardized data ARRAY :
    for d in range(len(datavar_nas)):
    
        dict2_predi_fla_PI[datavar_nas[d]] = dict1_mon_bin_PI[datavar_nas[d]+'_mon_bin'].flatten()
        dict2_predi_fla_abr[datavar_nas[d]] = dict1_mon_bin_abr[datavar_nas[d]+'_mon_bin'].flatten()

        # anomalies in the raw units:
        # 'dict2_predi' as a dict for reference-period (mean state) data
        dict2_predi[datavar_nas[d]] = deepcopy(dict1_mon_bin_PI[datavar_nas[d]+'_mon_bin'])
        
        dict2_predi_ano_PI[datavar_nas[d]] = dict2_predi_fla_PI[datavar_nas[d]] - nanmean(area_mean(dict2_predi[datavar_nas[d]], y_range, x_range))
        dict2_predi_ano_abr[datavar_nas[d]] = dict2_predi_fla_abr[datavar_nas[d]] - nanmean(area_mean(dict2_predi[datavar_nas[d]], y_range,x_range))
        
        # normalized stardard deviation in unit of './std':
        dict2_predi_nor_PI[datavar_nas[d]] = dict2_predi_ano_PI[datavar_nas[d]] / nanstd(dict2_predi_fla_PI[datavar_nas[d]])  # divided by monthly standard variance
        dict2_predi_nor_abr[datavar_nas[d]] = dict2_predi_ano_abr[datavar_nas[d]] / nanstd(dict2_predi_fla_PI[datavar_nas[d]])
    
    #..Use area_mean method, 'np.repeat' and 'np.tile' to reproduce gmt area-mean Array as the same shape as other flattened variables
    GMT_pi_mon = area_mean(dict1_mon_bin_PI['gmt_mon_bin'], s_range,  x_range)   #.. MONTHLY time series of global area_mean surface air temperature
    
    GMT_abr_mon = area_mean(dict1_mon_bin_abr['gmt_mon_bin'], s_range, x_range)   #.. MONTHLY time series of global area_mean surface air temperature
    
    # Use the southernOCEAN value as the gmt variable
    dict2_predi_fla_PI['gmt'] = GMT_pi_mon
    dict2_predi_fla_abr['gmt'] = GMT_abr_mon
    dict2_predi['gmt'] = deepcopy(dict2_predi_fla_PI['gmt'])
    shape_whole_period = asarray(dict2_predi['gmt'].shape[0])
    
    dict2_predi_ano_abr['gmt'] = dict2_predi_fla_abr['gmt'] - np.nanmean(dict2_predi['gmt'])  # shape in (t, lat, lon).flatten()
    dict2_predi_ano_PI['gmt'] = dict2_predi_fla_PI['gmt'] - np.nanmean(dict2_predi['gmt'])  # shape in (t, lat, lon).flatten()
    
    dict2_predi_nor_abr['gmt'] = dict2_predi_ano_abr['gmt'] / np.nanstd(dict2_predi_fla_PI['gmt'])
    dict2_predi_nor_PI['gmt'] = dict2_predi_ano_PI['gmt'] / np.nanstd(dict2_predi_fla_PI['gmt'])
    
    # shape of flattened array:
    metric_training = deepcopy(dict2_predi_ano_PI)
    metric_predict = deepcopy(dict2_predi_ano_abr)
    
    shape_fla_PI = dict2_predi_fla_PI['LWP'].shape
    shape_fla_abr = dict2_predi_fla_abr['LWP'].shape
    
    # save into rawdata_dict:
    C_dict['metric_training'] = dict2_predi_ano_PI
    C_dict['metric_predict'] = dict2_predi_ano_abr
    C_dict['GMT_training'] = GMT_pi_mon
    C_dict['GMT_predict'] = GMT_abr_mon
    
    C_dict['Mean_training'] = nanmean(area_mean(dict2_predi['LWP'], y_range, x_range))
    C_dict['Stdev_training'] = nanstd(nanstd(dict2_predi_fla_PI['LWP']))
    
    
    # The thresholds: TR_SST, TR_SUB:
    '''
    TR_sst_ano = TR_sst - np.nanmean(area_mean(dict2_predi['SST'], y_range, x_range))
    TR_sub_ano = TR_sub - np.nanmean(area_mean(dict2_predi['SUB'], y_range, x_range))
    
    TR_sst_nor = TR_sst_ano / np.nanstd(dict2_predi['SST'].flatten())
    TR_sub_nor = TR_sub_ano / np.nanstd(dict2_predi['SUB'].flatten())
    print('Threhold_anomalies: ', TR_sst_ano, TR_sub_ano)
    print('Threhold_normalized: ', TR_sst_nor, TR_sub_nor)
    '''
    
    #.. Training Module (2LRM Hot & Cold)
    #.. piControl
    predict_dict_PI, ind6_PI, ind7_PI, coef_array, shape_fla_training = rdlrm_2_training(metric_training, TR_sst, predictant='LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 2)
    # predict_dict_PI_iwp, ind6_PI_iwp, ind7_PI_iwp, coef_array_iwp, shape_fla_training_iwp = rdlrm_2_training(metric_training, TR_sst, predictant='IWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 2)
    
    # predict_dict_PI_albedo, _, _, coef_array_albedo = rdlrm_2_training(metric_training, TR_sst, predictant='albedo', predictor=['LWP', 'albedo_cs'], r = 2)[0:4]
    # predict_dict_PI_rsut, _, _, coef_array_rsut = rdlrm_2_training(metric_training, TR_sst, predictant='rsut', predictor=['LWP', 'rsutcs'], r = 2)[0:4]
    
    # Added on May 13th, 2022: for second step using LWP to predict the albedo
    # dict2_predi_fla_PI['LWP_lrm'] = deepcopy(predict_dict_PI['value'])
    # dict2_predi_ano_PI['LWP_lrm'] = dict2_predi_fla_PI['LWP_lrm'] - nanmean(area_mean( dict2_predi_fla_PI['LWP_lrm'].reshape(shape_mon_PI), y_range, x_range))
    # dict2_predi_nor_PI['LWP_lrm'] = dict2_predi_ano_PI['LWP_lrm'] / nanstd(dict2_predi_fla_PI['LWP_lrm']
    # predict_dict_PI_albedo_lL, _, _, coef_array_albedo_lL = rdlrm_2_training(dict2_predi_fla_PI, TR_sst, predictant='albedo', predictor=['LWP_lrm', 'albedo_cs'], r = 2)[0:4]
    # predict_dict_PI_rsut_lL, _, _, coef_array_rsut_lL = rdlrm_2_training(dict2_predi_fla_PI, TR_sst, predictant='rsut', predictor=['LWP_lrm', 'rsutcs'], r = 2)[0:4]

    # Save into the rawdata dict
    C_dict['Coef_dict'] = coef_array
    C_dict['Predict_dict_PI']  = predict_dict_PI
    C_dict['ind_Hot_PI'] = ind6_PI
    C_dict['ind_Cold_PI'] = ind7_PI
    # C_dict['Coef_dict_IWP']= coef_array_iwp
    # C_dict['Predict_dict_PI_IWP']  = predict_dict_PI_iwp
    
    # 'YB' is the predicted value of LWP in 'piControl' experiment
    YB = predict_dict_PI['value']
    # YB_iwp = predict_dict_PI_iwp['value']
    
    # Save 'YB', and resampled into the shape of 'LWP_yr_bin':
    C_dict['LWP_predi_bin_PI'] = asarray(YB).reshape(shape_mon_PI)
    # C_dict['IWP_predi_bin_PI'] = asarray(YB_iwp).reshape(shape_mon_PI)

    #.. Test performance
    stats_dict_PI = Test_performance_2(metric_training['LWP'], YB, ind6_PI, ind7_PI)
    # stats_dict_PI_iwp = Test_performance_2(metric_training['IWP'], YB_iwp, ind6_PI_iwp, ind7_PI_iwp)
 
    
    #.. predict module (2-LRM, Hot & Cold)
    #.. abrupt 4xCO2
    
    predict_dict_abr, ind6_abr, ind7_abr, shape_fla_testing = rdlrm_2_predict(metric_predict, coef_array, TR_sst, predictant = 'LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 2)
    # predict_dict_abr_iwp, ind6_abr_iwp, ind7_abr_iwp, shape_fla_testing_iwp = rdlrm_2_predict(metric_predict, coef_array_iwp, TR_sst, predictant = 'IWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 2)
    
    # Added on May 14th, 2022: for second step using LWP to predict the albedo
    # dict2_predi_fla_abr['LWP_lrm'] = deepcopy(predict_dict_abr['value'])
    # dict2_predi_ano_abr['LWP_lrm'] = dict2_predi_fla_abr['LWP_lrm'] - nanmean(area_mean( dict2_predi_fla_PI['LWP_lrm'].reshape(shape_mon_abr), y_range, x_range))
    # dict2_predi_nor_abr['LWP_lrm'] = (dict2_predi_fla_abr['LWP_lrm'] / nanstd(dict2_predi_fla_abr['LWP_lrm'])
    # predict_dict_abr_albedo_lL = rdlrm_2_predict(dict2_predi_fla_abr, coef_array_albedo, TR_sst, predictant='albedo', predictor=['LWP_lrm', 'albedo_cs'], r = 2)[0]
    # predict_dict_abr_rsut_lL = rdlrm_2_predict(dict2_predi_fla_abr, coef_array_rsut, TR_sst, predictant='rsut', predictor=['LWP_lrm', 'rsutcs'], r = 2)[0]

    # Save into the rawdata dict
    C_dict['Predict_dict_abr'] = predict_dict_abr
    C_dict['ind_Hot_abr'] = ind6_abr
    C_dict['ind_Cold_abr'] = ind7_abr
    
    # C_dict['Predict_dict_abr_IWP'] = predict_dict_abr_iwp

    # 'YB_abr' is the predicted value of LWP in 'abrupt-4xCO2' experiment
    YB_abr = predict_dict_abr['value']
    # YB_abr_iwp = predict_dict_abr_iwp['value']
    
    # Save 'YB_abr', reshapled into the shape of 'LWP_yr_bin_abr':
    C_dict['LWP_predi_bin_abr'] = asarray(YB_abr).reshape(shape_mon_abr)
    # C_dict['IWP_predi_bin_abr'] = asarray(YB_abr_iwp).reshape(shape_mon_abr)
    
    # Test performance for abrupt4xCO2
    stats_dict_abr = Test_performance_2(metric_predict['LWP'], YB_abr, ind6_abr, ind7_abr)
    # stats_dict_abr_iwp = Test_performance_2(metric_predict['IWP'], YB_abr_iwp, ind6_abr_iwp, ind7_abr_iwp)
    
    #.. save test performance metrics into rawdata_dict
    C_dict['stats_dict_PI'] = stats_dict_PI
    # C_dict['stats_dict_PI_iwp'] = stats_dict_PI_iwp
    
    C_dict['stats_dict_abr'] = stats_dict_abr
    # C_dict['stats_dict_abr_iwp'] = stats_dict_abr_iwp
    
    
    return C_dict



def fitLRM4(C_dict, TR_sst, TR_sub, s_range, y_range, x_range, lats, lons):
    
    # 'C_dict' is the raw data dict, 'TR_sst' accompany with 'TR_sub' are the pre-defined skin_Temperature/ 500 mb Subsidence thresholds to distinguish 4 rdlrms:

    # 's_range , 'y_range', 'x_range' used to do area mean for repeat gmt ARRAY

    dict1_abr_var = C_dict['dict1_abr_var']
    dict1_PI_var  = C_dict['dict1_PI_var']
    #print(dict0_PI_var['times'])
    
    model = C_dict['model_data']  #.. type in dict
    
    datavar_nas = ['LWP', 'TWP', 'IWP', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'SST', 'p_e', 'LTS', 'SUB']  #..12 varisables except gmt (lon dimension diff)
    
    # load annually-mean bin data
    dict1_yr_bin_PI = dict1_PI_var['dict1_yr_bin_PI']
    dict1_yr_bin_abr = dict1_abr_var['dict1_yr_bin_abr']
    #print(dict1_yr_bin_PI['LWP_yr_bin'].shape)
    
    # load monthly bin data
    dict1_mon_bin_PI = dict1_PI_var['dict1_mon_bin_PI']
    dict1_mon_bin_abr = dict1_abr_var['dict1_mon_bin_abr']

    # data array in which shapes?
    shape_yr_PI = dict1_yr_bin_PI['LWP_yr_bin'].shape
    shape_yr_abr = dict1_yr_bin_abr['LWP_yr_bin'].shape

    shape_yr_PI_gmt = dict1_yr_bin_PI['gmt_yr_bin'].shape
    shape_yr_abr_gmt = dict1_yr_bin_abr['gmt_yr_bin'].shape

    shape_mon_PI = dict1_mon_bin_PI['LWP_mon_bin'].shape
    shape_mon_abr = dict1_mon_bin_abr['LWP_mon_bin'].shape

    shape_mon_PI_gmt = dict1_mon_bin_PI['gmt_mon_bin'].shape
    shape_mon_abr_gmt = dict1_mon_bin_abr['gmt_mon_bin'].shape

    #.. archieve the 'shape' infos: 3-D
    C_dict['shape_yr_PI_3'] = shape_yr_PI
    C_dict['shape_yr_abr_3'] = shape_yr_abr
    C_dict['shape_yr_PI_gmt_3'] = shape_yr_PI_gmt
    C_dict['shape_yr_abr_gmt_3'] = shape_yr_abr_gmt

    C_dict['shape_mon_PI_3'] = shape_mon_PI
    C_dict['shape_mon_abr_3'] = shape_mon_abr
    C_dict['shape_mon_PI_gmt_3'] = shape_mon_PI_gmt
    C_dict['shape_mon_abr_gmt_3'] = shape_mon_abr_gmt

    # flatten the data array for 'training' lrm coefficient
    
    dict2_predi_fla_PI = {}
    dict2_predi_fla_abr = {}
    
    dict2_predi_ano_PI = {}  # need a climatological arrays of variables
    dict2_predi_ano_abr = {}  # need a climatological arrays of variables
    
    dict2_predi_nor_PI = {}  # standardized anomalies of variables
    dict2_predi_nor_abr = {}
    
    dict2_predi = {}
    
    #.. Ravel binned array /Standardized data ARRAY :
    for d in range(len(datavar_nas)):
    
        dict2_predi_fla_PI[datavar_nas[d]] = dict1_mon_bin_PI[datavar_nas[d]+'_mon_bin'].flatten()
        dict2_predi_fla_abr[datavar_nas[d]] = dict1_mon_bin_abr[datavar_nas[d]+'_mon_bin'].flatten()

        # anomalies in the raw units:
        # 'dict2_predi' as a dict for reference-period (mean state) data
        dict2_predi[datavar_nas[d]] = deepcopy(dict1_mon_bin_PI[datavar_nas[d]+'_mon_bin'])
        
        dict2_predi_ano_PI[datavar_nas[d]] = dict2_predi_fla_PI[datavar_nas[d]] - nanmean(area_mean(dict2_predi[datavar_nas[d]], y_range, x_range))
        dict2_predi_ano_abr[datavar_nas[d]] = dict2_predi_fla_abr[datavar_nas[d]] - nanmean(area_mean(dict2_predi[datavar_nas[d]], y_range,x_range))
        
        # normalized stardard deviation in unit of './std':
        dict2_predi_nor_PI[datavar_nas[d]] = dict2_predi_ano_PI[datavar_nas[d]] / nanstd(dict2_predi_fla_PI[datavar_nas[d]])  # divided by monthly standard variance
        dict2_predi_nor_abr[datavar_nas[d]] = dict2_predi_ano_abr[datavar_nas[d]] / nanstd(dict2_predi_fla_PI[datavar_nas[d]])
    
    #..Use area_mean method, 'np.repeat' and 'np.tile' to reproduce gmt area-mean Array as the same shape as other flattened variables
    GMT_pi_mon = area_mean(dict1_mon_bin_PI['gmt_mon_bin'], s_range,  x_range)   #.. MONTHLY time series of global area_mean surface air temperature
    
    GMT_abr_mon = area_mean(dict1_mon_bin_abr['gmt_mon_bin'], s_range, x_range)   #.. MONTHLY time series of global area_mean surface air temperature
    
    # Use the southernOCEAN value as the gmt variable
    dict2_predi_fla_PI['gmt'] = GMT_pi_mon
    dict2_predi_fla_abr['gmt'] = GMT_abr_mon
    dict2_predi['gmt'] = deepcopy(dict2_predi_fla_PI['gmt'])
    shape_whole_period = asarray(dict2_predi['gmt'].shape[0])
    
    dict2_predi_ano_abr['gmt'] = dict2_predi_fla_abr['gmt'] - np.nanmean(dict2_predi['gmt'])  # shape in (t, lat, lon).flatten()
    dict2_predi_ano_PI['gmt'] = dict2_predi_fla_PI['gmt'] - np.nanmean(dict2_predi['gmt'])  # shape in (t, lat, lon).flatten()
    
    dict2_predi_nor_abr['gmt'] = dict2_predi_ano_abr['gmt'] / np.nanstd(dict2_predi_fla_PI['gmt'])
    dict2_predi_nor_PI['gmt'] = dict2_predi_ano_PI['gmt'] / np.nanstd(dict2_predi_fla_PI['gmt'])
    
    # shape of flattened array:
    metric_training = deepcopy(dict2_predi_ano_PI)
    metric_predict = deepcopy(dict2_predi_ano_abr)
    
    shape_fla_PI = dict2_predi_fla_PI['LWP'].shape
    shape_fla_abr = dict2_predi_fla_abr['LWP'].shape
    
    # save into rawdata_dict:
    C_dict['metric_training'] = dict2_predi_ano_PI
    C_dict['metric_predict'] = dict2_predi_ano_abr
    C_dict['GMT_training'] = GMT_pi_mon
    C_dict['GMT_predict'] = GMT_abr_mon
    
    C_dict['Mean_training'] = nanmean(area_mean(dict2_predi['LWP'], y_range, x_range))
    C_dict['Stdev_training'] = nanstd(dict2_predi_fla_PI['LWP'])
    
    
    # The thresholds: TR_SST, TR_SUB:
    '''
    TR_sst_ano = TR_sst - np.nanmean(area_mean(dict2_predi['SST'], y_range, x_range))
    TR_sub_ano = TR_sub - np.nanmean(area_mean(dict2_predi['SUB'], y_range, x_range))
    
    TR_sst_nor = TR_sst_ano / np.nanstd(dict2_predi['SST'].flatten())
    TR_sub_nor = TR_sub_ano / np.nanstd(dict2_predi['SUB'].flatten())
    print('Threhold_anomalies: ', TR_sst_ano, TR_sub_ano)
    print('Threhold_normalized: ', TR_sst_nor, TR_sub_nor)
    '''
    
    #.. Training Module (4lrm)
    #.. piControl
    predict_dict_PI, ind7_PI, ind8_PI, ind9_PI, ind10_PI, coef_array, shape_fla_training = rdlrm_4_training(metric_training, TR_sst, TR_sub, predictant='LWP', predictor=['SST', 'p_e', 'LTS', 'SUB'], r = 4)
    # predict_dict_PI_iwp, ind7_PI_iwp, ind8_PI_iwp, ind9_PI_iwp, ind10_PI_iwp, coef_array_iwp, shape_fla_training_iwp = rdlrm_4_training(metric_training, TR_sst, TR_sub, predictant='IWP', predictor=['SST', 'p_e', 'LTS', 'SUB'], r = 4)
    
    # predict_dict_PI_albedo, _, _, _, _, coef_array_albedo = rdlrm_4_training(metric_training, TR_sst, TR_sub, predictant='albedo', predictor=['LWP', 'albedo_cs'], r = 4)[0:6]
    # predict_dict_PI_rsut, _, _, _, _, coef_array_rsut = rdlrm_4_training(metric_training, TR_sst, TR_sub, predictant='rsut', predictor=['LWP', 'rsutcs'], r = 4)[0:6]
    
    # Added on May 13th, 2022: for second step using LWP to predict the albedo
    # dict2_predi_fla_PI['LWP_lrm'] = deepcopy(predict_dict_PI['value'])
    # dict2_predi_ano_PI['LWP_lrm'] = dict2_predi_fla_PI['LWP_lrm'] - nanmean(area_mean( dict2_predi_fla_PI['LWP_lrm'].reshape(shape_mon_PI), y_range, x_range))
    # dict2_predi_nor_PI['LWP_lrm'] = dict2_predi_ano_PI['LWP_lrm'] / nanstd(dict2_predi_fla_PI['LWP_lrm'])
    # predict_dict_PI_albedo_lL, _, _, _, _, coef_array_albedo_lL = rdlrm_4_training(dict2_predi_fla_PI, TR_sst, TR_sub, predictant='albedo', predictor=['LWP_lrm', 'albedo_cs'], r = 4)[0:6]
    # predict_dict_PI_rsut_lL, _, _, _, _, coef_array_rsut_lL = rdlrm_4_training(dict2_predi_fla_PI, TR_sst, TR_sub, predictant='rsut', predictor=['LWP_lrm', 'rsutcs'], r = 4)[0:6]
    

    # Save into the rawdata dict
    C_dict['Coef_dict'] = coef_array
    C_dict['Predict_dict_PI'] = predict_dict_PI
    C_dict['ind_Cold_Up_PI'] = ind7_PI
    C_dict['ind_Hot_Up_PI'] = ind8_PI
    C_dict['ind_Cold_Down_PI'] = ind9_PI
    C_dict['ind_Hot_Down_PI'] = ind10_PI
    # C_dict['Coef_dict_IWP']= coef_array_iwp
    # C_dict['Predict_dict_PI_IWP']  = predict_dict_PI_iwp
    
    # 'YB' is the predicted value of LWP in 'piControl' experiment
    YB = predict_dict_PI['value']
    # YB_iwp = predict_dict_PI_iwp['value']
    
    # Save 'YB', resampled into the shape of 'LWP_yr_bin':
    C_dict['LWP_predi_bin_PI'] = asarray(YB).reshape(shape_mon_PI)
    # C_dict['IWP_predi_bin_PI'] = asarray(YB_iwp).reshape(shape_mon_PI)

    #.. test performance
    stats_dict_PI = Test_performance_4(metric_training['LWP'], YB, ind7_PI, ind8_PI, ind9_PI, ind10_PI)
    # stats_dict_PI_iwp = Test_performance_4(metric_training['IWP'], YB_iwp, ind7_PI_iwp, ind8_PI_iwp, ind9_PI_iwp, ind10_PI_iwp)
    
    
    #.. predict module (4-LRM)
    #.. abrupt 4xCO2 
    
    predict_dict_abr, ind7_abr, ind8_abr, ind9_abr, ind10_abr, shape_fla_testing = rdlrm_4_predict(metric_predict, coef_array, TR_sst, TR_sub, predictant = 'LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 4)
    # predict_dict_abr_iwp, ind7_abr_iwp, ind8_abr_iwp, ind9_abr_iwp, ind10_abr_iwp, shape_fla_testing_iwp = rdlrm_4_predict(metric_predict, coef_array_iwp, TR_sst, TR_sub, predictant = 'IWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 4)
    
    # predict_dict_abr_albedo = rdlrm_4_predict(metric_predict, coef_array_albedo, TR_sst, TR_sub, predictant = 'albedo', predictor = ['LWP', 'albedo_cs'], r = 4)[0]
    # predict_dict_abr_rsut = rdlrm_4_predict(metric_predict, coef_array_rsut, TR_sst, TR_sub, predictant = 'rsut', predictor= ['LWP', 'rsutcs'], r = 4)[0]
    
    # Added on May 14th, 2022: for second step using LWP to predict the albedo
    # dict2_predi_fla_abr['LWP_lrm'] = deepcopy(predict_dict_abr['value'])
    # dict2_predi_ano_abr['LWP_lrm'] = dict2_predi_fla_abr['LWP_lrm'] - nanmean(area_mean( dict2_predi_fla_PI['LWP_lrm'].reshape(shape_mon_abr), y_range, x_range))
    # dict2_predi_nor_abr['LWP_lrm'] = (dict2_predi_fla_abr['LWP_lrm'] / nanstd(dict2_predi_fla_abr['LWP_lrm'])
    # predict_dict_abr_albedo_lL = rdlrm_4_predict(dict2_predi_fla_abr, coef_array_albedo, TR_sst, TR_sub, predictant='albedo', predictor=['LWP_lrm', 'albedo_cs'], r = 4)[0]
    # predict_dict_abr_rsut_lL = rdlrm_4_predict(dict2_predi_fla_abr, coef_array_rsut, TR_sst, TR_sub, predictant='rsut', predictor=['LWP_lrm', 'rsutcs'], r = 4)[0]

    # Save into the rawdata dict
    C_dict['Predict_dict_abr'] = predict_dict_abr
    C_dict['ind_Cold_Up_abr'] = ind7_abr
    C_dict['ind_Hot_Up_abr'] = ind8_abr
    C_dict['ind_Cold_Down_abr'] = ind9_abr
    C_dict['ind_Hot_Down_abr'] = ind10_abr
    
    # C_dict['Predict_dict_abr_IWP'] = predict_dict_abr_iwp
    

    # 'YB_abr' is the predicted value of LWP in 'abrupt 4xCO2' experiment
    YB_abr = predict_dict_abr['value']
    # YB_abr_iwp = predict_dict_abr_iwp['value']
    
    # Save 'YB_abr', reshapled into the shape of 'LWP_yr_bin_abr':
    C_dict['LWP_predi_bin_abr'] = asarray(YB_abr).reshape(shape_mon_abr)
    # C_dict['IWP_predi_bin_abr'] = asarray(YB_abr_iwp).reshape(shape_mon_abr)
    
    # Test performance for abrupt 4xCO2
    stats_dict_abr = Test_performance_4(dict2_predi_fla_abr['LWP'], YB_abr, ind7_abr, ind8_abr, ind9_abr, ind10_abr)
    # stats_dict_abr_iwp = Test_performance_4(dict2_predi_fla_abr['IWP'], YB_abr_iwp, ind7_abr_iwp, ind8_abr_iwp, ind9_abr_iwp, ind10_abr_iwp)
    
    #.. save test performance metrics into rawdata_dict
    C_dict['stats_dict_PI'] = stats_dict_PI
    # C_dict['stats_dict_PI_iwp'] = stats_dict_PI_iwp
    
    C_dict['stats_dict_abr'] = stats_dict_abr
    # C_dict['stats_dict_abr_iwp'] = stats_dict_abr_iwp
    
    
    return C_dict
