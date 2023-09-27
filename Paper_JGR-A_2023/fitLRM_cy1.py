# ## training AND predicting the data by Linear Regession Model(LRM) of 1 and 2_UpDown regimes; ###
# ## estimate their statistic performance (RMSE/ R^2); ###


import netCDF4
from numpy import *
import matplotlib.pyplot as plt
import xarray as xr
# import PyNIO as Nio # deprecated
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

def fitLRM1(C_dict, TR_sst, s_range, y_range, x_range, lats, lons):
    # 'C_dict' is the raw data dict, 'TR_sst' is the pre-defined skin_Temperature Threshold to distinguish two Multi-Linear Regression Models

    # 's_range , 'y_range', 'x_range' used to do area mean for repeat gmt ARRAY

    dict0_abr_var = C_dict['dict1_abr_var']
    dict0_PI_var  = C_dict['dict1_PI_var']
    #print(dict0_PI_var['times'])

    model = C_dict['model_data']   #.. type in dict

    datavar_nas = ['LWP', 'TWP', 'IWP', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'SST', 'p_e', 'LTS', 'SUB']   #..12 varisables except gmt (lon dimension  diff)

    # load annually-mean bin data.
    dict1_yr_bin_PI = dict0_PI_var['dict1_yr_bin_PI']
    dict1_yr_bin_abr = dict0_abr_var['dict1_yr_bin_abr']
    #print(dict1_yr_bin_PI['LWP_yr_bin'].shape)
    
    # load monthly bin data.
    dict1_mon_bin_PI = dict0_PI_var['dict1_mon_bin_PI']
    dict1_mon_bin_abr = dict0_abr_var['dict1_mon_bin_abr']

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
    
    # gmt
    
    dict2_predi_fla_PI['gmt'] = deepcopy(dict1_mon_bin_PI['gmt_mon_bin'])
    dict2_predi_fla_abr['gmt'] = deepcopy(dict1_mon_bin_abr['gmt_mon_bin'])
    dict2_predi['gmt'] = deepcopy(GMT_pi_mon)
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

    #.. Training Module (1-LRM)
    #.. piControl
    
    predict_dict_PI, ind6_PI, ind7_PI, coef_array, shape_fla_training = rdlrm_1_training(metric_training, predictant='LWP')
    # predict_dict_PI_iwp, ind6_PI_iwp, ind7_PI_iwp, coef_array_iwp, shape_fla_training_iwp = rdlrm_1_training(dict2_predi_fla_PI, predictant='IWP')
    
    # predict_dict_PI_albedo, _, _, coef_array_albedo = rdlrm_1_training(metric_training, predictant='albedo', predictor=['LWP', 'albedo_cs'], r = 1)[0:4]
    # predict_dict_PI_rsut, _, _, coef_array_rsut = rdlrm_1_training(metric_training, predictant='rsut', predictor=['LWP', 'rsutcs'], r = 1)[0:4]
    
    # Added on May 13th, 2022: for second step using LWP to predict the albedo
    # dict2_predi_fla_PI['LWP_lrm'] = deepcopy(predict_dict_PI['value'])
    # dict2_predi_ano_PI['LWP_lrm'] = dict2_predi_fla_PI['LWP_lrm'] - nanmean(area_mean( dict2_predi_fla_PI['LWP_lrm'].reshape(shape_mon_PI), y_range, x_range))
    # dict2_predi_nor_PI['LWP_lrm'] = dict2_predi_ano_PI['LWP_lrm'] / nanstd(dict2_predi_fla_PI['LWP_lrm'])
    # predict_dict_PI_albedo_lL, _, _, coef_array_albedo_lL = rdlrm_1_training(dict2_predi_fla_PI, predictant='albedo', predictor=['LWP_lrm', 'albedo_cs'], r = 1)[0:4]
    # predict_dict_PI_rsut_lL, _, _, coef_array_rsut_lL = rdlrm_1_training(dict2_predi_fla_PI, predictant='rsut', predictor=['LWP_lrm', 'rsutcs'], r = 1)[0:4]


    # Save into the rawdata dict
    C_dict['Coef_dict'] = coef_array
    C_dict['Predict_dict_PI'] = predict_dict_PI
    C_dict['ind_True_PI'] = ind6_PI  # C_dict['ind_Hot_PI'] = ind6_PI
    C_dict['ind_False_PI'] = ind7_PI  # C_dict['ind_Cold_PI'] = ind7_PI
    # C_dict['Coef_dict_IWP']= coef_array_iwp
    # C_dict['Predict_dict_PI_IWP']  = predict_dict_PI_iwp
    
    # 'YB' is the predicted value of LWP in 'piControl' experiment
    YB = predict_dict_PI['value']
    # print("2lrm predicted mean LWP: ", nanmean(YB), " in 'piControl' ")
    # YB_iwp = predict_dict_PI_iwp['value']
    
    # Save 'YB', and resampled into the shape of 'LWP_yr_bin':
    C_dict['LWP_predi_bin_PI'] = asarray(YB).reshape(shape_mon_PI)
    # C_dict['IWP_predi_bin_PI'] = asarray(YB_iwp).reshape(shape_mon_PI) 
    
    # Test performance
    stats_dict_PI = Test_performance_1(metric_training['LWP'], YB, ind6_PI, ind7_PI)
    # stats_dict_PI_iwp = Test_performance_1(metric_training['IWP'], YB_iwp, ind6_PI_iwp, ind7_PI_iwp)
    

    #.. predict Module (1-LRM)
    #.. abrupt 4xCO2
    
    predict_dict_abr, ind6_abr, ind7_abr, shape_fla_testing = rdlrm_1_predict(metric_predict, coef_array, predictant = 'LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 1)
    # predict_dict_abr_iwp, ind6_abr_iwp, ind7_abr_iwp, shape_fla_testing_iwp = rdlrm_1_predict(dict2_predi_fla_abr, coef_array_iwp, predictant = 'IWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 1)
    
    # Added on May 14th, 2022: for second step using LWP to predict the albedo
    # dict2_predi_fla_abr['LWP_lrm'] = deepcopy(predict_dict_abr['value'])
    # dict2_predi_ano_abr['LWP_lrm'] = dict2_predi_fla_abr['LWP_lrm'] - nanmean(area_mean( dict2_predi_fla_PI['LWP_lrm'].reshape(shape_mon_abr), y_range, x_range))
    # dict2_predi_nor_abr['LWP_lrm'] = (dict2_predi_fla_abr['LWP_lrm'] / nanstd(dict2_predi_fla_abr['LWP_lrm'])
    # predict_dict_abr_albedo_lL = rdlrm_1_predict(dict2_predi_fla_abr, coef_array_albedo, predictant='albedo', predictor=['LWP_lrm', 'albedo_cs'], r = 1)[0]
    # predict_dict_abr_rsut_lL = rdlrm_1_predict(dict2_predi_fla_abr, coef_array_rsut, predictant='rsut', predictor=['LWP_lrm', 'rsutcs'], r = 1)[0]
    
    
    # Save into the rawdata dict
    C_dict['Predict_dict_abr'] = predict_dict_abr
    C_dict['ind_True_abr'] = ind6_abr  # C_dict['ind_Hot_abr'] = ind6_abr
    C_dict['ind_False_abr'] = ind7_abr  # C_dict['ind_Cold_abr'] = ind7_abr
    # C_dict['Predict_dict_abr_IWP'] = predict_dict_abr_iwp
    
    # 'YB_abr' is the predicted value of LWP in 'abrupt-4xCO2' experiment
    YB_abr = predict_dict_abr['value'] 
    # YB_abr_iwp = predict_dict_abr_iwp['value']
    
    # Save 'YB_abr', reshapled into the shape of 'LWP_yr_bin_abr':
    C_dict['LWP_predi_bin_abr'] = asarray(YB_abr).reshape(shape_mon_abr)
    # C_dict['IWP_predi_bin_abr'] = asarray(YB_abr_iwp).reshape(shape_mon_abr)
    
    # Test performance for abrupt-4xCO2 (testing) data set
    stats_dict_abr = Test_performance_1(metric_predict['LWP'], YB_abr, ind6_abr, ind7_abr)
    # stats_dict_abr_iwp = Test_performance_1(metric_predict['IWP'], YB_abr_iwp, ind6_abr_iwp, ind7_abr_iwp)
    
    #.. save test performance metrics into rawdata_dict
    C_dict['stats_dict_PI'] = stats_dict_PI
    # C_dict['stats_dict_PI_iwp'] = stats_dict_PI_iwp
    C_dict['stats_dict_abr'] = stats_dict_abr
    # C_dict['stats_dict_abr_iwp'] = stats_dict_abr_iwp

    return C_dict



def fitLRM2(C_dict, TR_sst, TR_sub, s_range, y_range, x_range, lats, lons):
    
    # 'C_dict' is the raw data dict, 'TR_sst' accompany with 'TR_sub' are the pre-defined skin_Temperature/ 500 mb Subsidence thresholds to distinguish 4 rdlrms:
    
    # 's_range , 'y_range', 'x_range' used to do area mean for repeat gmt ARRAY

    dict0_abr_var = C_dict['dict1_abr_var']
    dict0_PI_var  = C_dict['dict1_PI_var']
    #print(dict0_PI_var['times'])
    
    model= C_dict['model_data']   #.. type in dict
    
    datavar_nas = ['LWP', 'TWP', 'IWP', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'SST', 'p_e', 'LTS', 'SUB']   #..12 varisables except gmt (lon dimension diff)
    
    # load annually-mean bin data
    dict1_yr_bin_PI = dict0_PI_var['dict1_yr_bin_PI']
    dict1_yr_bin_abr = dict0_abr_var['dict1_yr_bin_abr']
    #print(dict1_yr_bin_PI['LWP_yr_bin'].shape)
    
    # load monthly bin data
    dict1_mon_bin_PI = dict0_PI_var['dict1_mon_bin_PI']
    dict1_mon_bin_abr = dict0_abr_var['dict1_mon_bin_abr']

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
    
    #.. Ravel binned array /Standardized data ARRAY:
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
    
    # gmt
    dict2_predi_fla_PI['gmt'] = deepcopy(dict1_mon_bin_PI['gmt_mon_bin'])
    dict2_predi_fla_abr['gmt'] = deepcopy(dict1_mon_bin_abr['gmt_mon_bin'])
    dict2_predi['gmt'] = deepcopy(GMT_pi_mon)
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
    
    #.. Training Module (2 lrm, with Up & Down)
    #.. piControl
    
    predict_dict_PI, ind7_PI, ind8_PI, ind9_PI, ind10_PI, coef_array, shape_fla_training = rdlrm_4_training(metric_training, TR_sst, TR_sub, predictant='LWP', r = 2)
    # predict_dict_PI_iwp, ind7_PI_iwp, ind8_PI_iwp, ind9_PI_iwp, ind10_PI_iwp, coef_array_iwp, shape_fla_training_iwp = rdlrm_4_training(metric_training, TR_sst_nor, TR_sub_nor, predictant='IWP', r = 2)
    
    # predict_dict_PI_albedo, _, _, _, _, coef_array_albedo = rdlrm_4_training(metric_training, TR_sst, TR_sub, predictant='albedo', predictor=['LWP', 'albedo_cs'], r = 2)[0:6]
    # predict_dict_PI_rsut, _, _, _, _, coef_array_rsut = rdlrm_4_training(metric_training, TR_sst, TR_sub, predictant='rsut', predictor=['LWP', 'rsutcs'], r = 2)[0:6]
    
    # Added on May 13th, 2022: for second step using LWP to predict the albedo
    # dict2_predi_fla_PI['LWP_lrm'] = deepcopy(predict_dict_PI['value'])
    # dict2_predi_ano_PI['LWP_lrm'] = dict2_predi_fla_PI['LWP_lrm'] - nanmean(area_mean( dict2_predi_fla_PI['LWP_lrm'].reshape(shape_mon_PI), y_range, x_range))
    # dict2_predi_nor_PI['LWP_lrm'] = dict2_predi_ano_PI['LWP_lrm'] / nanstd(dict2_predi_fla_PI['LWP_lrm'])
    # predict_dict_PI_albedo_lL, _, _, _, _, coef_array_albedo_lL = rdlrm_4_training(dict2_predi_fla_PI, TR_sst, TR_sub, predictant='albedo', predictor=['LWP_lrm', 'albedo_cs'], r  = 2)[0:6]
    # predict_dict_PI_rsut_lL, _, _, _, _, coef_array_rsut_lL = rdlrm_4_training(dict2_predi_fla_PI, TR_sst, TR_sub, predictant='rsut', predictor=['LWP_lrm', 'rsutcs'], r = 2)[0:6]
    
    # Save into the rawdata dict
    C_dict['Coef_dict'] = coef_array
    C_dict['Predict_dict_PI']  = predict_dict_PI
    C_dict['ind_Up_PI'] = ind7_PI  # C_dict['ind_Cold_Up_PI'] = ind7_PI
    C_dict['ind_Down_PI'] = ind8_PI  # C_dict['ind_Hot_Up_PI'] = ind8_PI
    # C_dict['Coef_dict_IWP']= coef_array_iwp
    # C_dict['Predict_dict_PI_IWP']  = predict_dict_PI_iwp

    # 'YB' is the predicted value of LWP in 'piControl' experiment
    YB = predict_dict_PI['value']
    # YB_iwp = predict_dict_PI_iwp['value']
    
    # Save 'YB', resampled into the shape of 'LWP_yr_bin':
    C_dict['LWP_predi_bin_PI'] = asarray(YB).reshape(shape_mon_PI)
    # C_dict['IWP_predi_bin_PI'] = asarray(YB_iwp).reshape(shape_mon_PI)
    
    # Test performance
    stats_dict_PI = Test_performance_2(metric_training['LWP'], YB, ind7_PI, ind8_PI)   #  Test_performance_4(dict2_predi_fla_PI['LWP'], YB, ind7_PI, ind8_PI, ind9_PI, ind10_PI)
    # stats_dict_PI_iwp = Test_performance_2(metric_training['IWP'], YB_iwp, ind7_PI_iwp, ind8_PI_iwp)   # Test_performance_4(dict2_predi_fla_PI['IWP'], YB_iwp, ind7_PI_iwp, ind8_PI_iwp, ind9_PI_iwp, ind10_PI_iwp)
    
    
    #.. predict Module (2LRM-Up & Down)
    #.. abrupt 4xCO2
    
    predict_dict_abr, ind7_abr, ind8_abr, ind9_abr, ind10_abr, shape_fla_testing = rdlrm_4_predict(metric_predict, coef_array, TR_sst, TR_sub, predictant = 'LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 2)
    # predict_dict_abr_iwp, ind7_abr_iwp, ind8_abr_iwp, ind9_abr_iwp, ind10_abr_iwp, shape_fla_testing_iwp = rdlrm_4_predict(metric_predict coef_array_iwp, TR_sst_nor, TR_sub_nor, predictant = 'IWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 2)
    
    # Added on May 14th, 2022: for second step using LWP to predict the albedo
    # dict2_predi_fla_abr['LWP_lrm'] = deepcopy(predict_dict_abr['value'])
    # dict2_predi_ano_abr['LWP_lrm'] = dict2_predi_fla_abr['LWP_lrm'] - nanmean(area_mean( dict2_predi_fla_PI['LWP_lrm'].reshape(shape_mon_abr), y_range, x_range))
    # dict2_predi_nor_abr['LWP_lrm'] = (dict2_predi_fla_abr['LWP_lrm'] / nanstd(dict2_predi_fla_abr['LWP_lrm'])
    # predict_dict_abr_albedo_lL = rdlrm_4_predict(dict2_predi_fla_abr, coef_array_albedo, TR_sst, TR_sub, predictant='albedo', predictor=['LWP_lrm', 'albedo_cs'], r = 2)[0]
    # predict_dict_abr_rsut_lL = rdlrm_4_predict(dict2_predi_fla_abr, coef_array_rsut, TR_sst, TR_sub, predictant='rsut', predictor=['LWP_lrm', 'rsutcs'], r = 2)[0]
    
    # Save into the rawdata dict
    C_dict['Predict_dict_abr'] = predict_dict_abr
    C_dict['ind_Up_abr'] = ind7_abr  # C_dict['ind_Cold_Up_abr'] = ind7_abr
    C_dict['ind_Down_abr'] = ind8_abr  # C_dict['ind_Hot_Up_abr'] = ind8_abr
    # C_dict['Predict_dict_abr_IWP'] = predict_dict_abr_iwp
    
    # 'YB_abr' is the predicted value of LWP in 'abrupt 4xCO2' experiment
    YB_abr = predict_dict_abr['value']
    # YB_abr_iwp = predict_dict_abr_iwp['value']
    
    # Save 'YB_abr', reshapled into the shape of 'LWP_yr_bin_abr':
    C_dict['LWP_predi_bin_abr'] = asarray(YB_abr).reshape(shape_mon_abr)
    # C_dict['IWP_predi_bin_abr'] = asarray(YB_abr_iwp).reshape(shape_mon_abr)

    # Test performance for abrupt 4xCO2
    stats_dict_abr = Test_performance_2(metric_predict['LWP'], YB_abr, ind7_abr, ind8_abr)  # Test_performance_4(dict2_predi_fla_abr['LWP'], YB_abr, ind7_abr, ind8_abr, ind9_abr, ind10_abr)
    # stats_dict_abr_iwp = Test_performance_2(metric_predict['IWP'], YB_abr_iwp, ind7_abr_iwp, ind8_abr_iwp)  # Test_performance_4(dict2_predi_fla_abr['IWP'], YB_abr_iwp, ind7_abr_iwp, ind8_abr_iwp, ind9_abr_iwp, ind10_abr_iwp)
    
    #.. save preditc metrics into rawdata_dict
    C_dict['stats_dict_PI'] = stats_dict_PI
    # C_dict['stats_dict_PI_iwp'] = stats_dict_PI_iwp

    C_dict['stats_dict_abr'] = stats_dict_abr
    # C_dict['stats_dict_abr_iwp'] = stats_dict_abr_iwp

    return C_dict



def p4plot1(s_range, y_range, x_range, Mean_training, Stdev_training, shape_yr_pi, shape_yr_abr, rawdata_dict):
    
    ### 's_range , 'y_range', 'x_range' used to do area mean for repeat gmt ARRAY

    # retriving datas from big dict...
    dict0_abr_var = rawdata_dict['dict1_abr_var']
    dict0_PI_var = rawdata_dict['dict1_PI_var']
    shape_yr_PI_3 = rawdata_dict['shape_yr_PI_3']
    shape_yr_abr_3 = rawdata_dict['shape_yr_abr_3']
    shape_mon_PI_3 = rawdata_dict['shape_mon_PI_3']
    shape_mon_abr_3 = rawdata_dict['shape_mon_abr_3']
    
    model = rawdata_dict['model_data']   #.. type in dict

    datarepo_nas = ['LWP']  # 'IWP', albedo', 'albedo_cs', 'rsut', 'rsutcs'

    # load annual mean binned data:
    dict1_yr_bin_PI = deepcopy(dict0_PI_var['dict1_yr_bin_PI'])
    dict1_yr_bin_abr = deepcopy(dict0_abr_var['dict1_yr_bin_abr'])

    # load monthly binned data:
    dict1_mon_bin_PI = deepcopy(dict0_PI_var['dict1_mon_bin_PI'])
    dict1_mon_bin_abr = deepcopy(dict0_abr_var['dict1_mon_bin_abr'])
    
    # load anomalies (or normalized) monthly binned data:
    dict_metric_actual_PI = deepcopy(rawdata_dict['metric_training'])
    dict_metric_actual_abr = deepcopy(rawdata_dict['metric_predict'])
    
    # load anomalies predicted binned data:
    LWP_metric_predi_PI = deepcopy(rawdata_dict['LWP_predi_bin_PI'])
    LWP_metric_predi_abr = deepcopy(rawdata_dict['LWP_predi_bin_abr'])
    
    # calculate (convert) the predicted data back to raw unit:
    LWP_raw_predi_PI = LWP_metric_predi_PI + Mean_training  # (LWP_metric_predi_PI * Stdev_training) + Mean_training 
    LWP_raw_predi_abr = LWP_metric_predi_abr + Mean_training # (LWP_metric_predi_abr * Stdev_training) + Mean_training
    
    ## Calc annually-mean, area-mean variables on 'abrupt4xCO2' and 'piControl' exps:
    # GCM actual variable
    areamean_dict_PI = {}
    areamean_dict_abr = {}
    
    for e in range(len(datarepo_nas)):
    
        #  "monthly" convert to "annual" data:
        areamean_dict_PI[datarepo_nas[e]+ '_yr_bin'] = annual_mean(dict_metric_actual_PI['LWP'].reshape(shape_mon_PI_3), shape_mon_PI_3[0], shape_mon_PI_3[1], shape_mon_PI_3[2])
        areamean_dict_abr[datarepo_nas[e]+ '_yr_bin'] = annual_mean(dict_metric_actual_abr['LWP'].reshape(shape_mon_abr_3), shape_mon_abr_3[0], shape_mon_abr_3[1], shape_mon_abr_3[2])

        # "yr_bin" area_mean to 1-D array in 'shape_yr_':
        areamean_dict_PI[datarepo_nas[e]+ '_area_yr'] = area_mean(areamean_dict_PI[datarepo_nas[e]+ '_yr_bin'], y_range, x_range)
        areamean_dict_abr[datarepo_nas[e]+ '_area_yr'] = area_mean(areamean_dict_abr[datarepo_nas[e]+ '_yr_bin'], y_range, x_range)
    
    areamean_dict_PI['gmt_area_yr'] = area_mean(dict1_yr_bin_PI['gmt_yr_bin'], s_range, x_range)
    areamean_dict_abr['gmt_area_yr'] = area_mean(dict1_yr_bin_abr['gmt_yr_bin'], s_range, x_range)
    
    
    # LRM predict variable
    areamean_dict_predi =  {}
    datapredi_nas = ['LWP']  # 'IWP', 'albedo', 'rsut', 'albedo_lL', 'rsut_lL'
    
    for f in range(len(datapredi_nas)):
        areamean_dict_predi[datapredi_nas[f]+'_predi_yr_bin_pi'] = annual_mean(rawdata_dict[datapredi_nas[f]+'_predi_bin_PI'], shape_mon_PI_3[0], shape_mon_PI_3[1], shape_mon_PI_3[2])
        areamean_dict_predi[datapredi_nas[f]+'_predi_yr_bin_abr'] = annual_mean(rawdata_dict[datapredi_nas[f]+'_predi_bin_abr'], shape_mon_abr_3[0], shape_mon_abr_3[1], shape_mon_abr_3[2])
    
    # "yr_bin" area_mean to 1-D array in 'shape_yr_':
    for g in range(len(datapredi_nas)):

        areamean_dict_predi[datapredi_nas[g]+'_area_yr_pi'] = area_mean(areamean_dict_predi[datapredi_nas[g]+'_predi_yr_bin_pi'], y_range, x_range)
        areamean_dict_predi[datapredi_nas[g]+'_area_yr_abr'] = area_mean(areamean_dict_predi[datapredi_nas[g]+'_predi_yr_bin_abr'], y_range, x_range)
    
    # Store the annually report & predicted metrics
    rawdata_dict['areamean_dict_predi'] = areamean_dict_predi
    rawdata_dict['areamean_dict_abr'] = areamean_dict_abr
    rawdata_dict['areamean_dict_PI'] = areamean_dict_PI

    # calc d_DeltaLWP /d_DeltaGMT |(abrupt-4xCO2 - avg(piControl)), added on June 27th.
    output_2report_pi = area_mean(annual_mean(dict1_mon_bin_PI['LWP_mon_bin'], shape_mon_PI_3[0], shape_mon_PI_3[1], shape_mon_PI_3[2]), y_range, x_range)[:]
    output_2report_abr = area_mean(annual_mean(dict1_mon_bin_abr['LWP_mon_bin'], shape_mon_abr_3[0], shape_mon_abr_3[1], shape_mon_abr_3[2]), y_range, x_range)[0:150]
    
    output_2predict_pi = area_mean(annual_mean(LWP_raw_predi_PI, shape_mon_PI_3[0],shape_mon_PI_3[1],shape_mon_PI_3[2]), y_range, x_range)[:]
    output_2predict_abr = area_mean(annual_mean(LWP_raw_predi_abr, shape_mon_abr_3[0],shape_mon_abr_3[1],shape_mon_abr_3[2]), y_range, x_range)[0:150]

    output_yrs = arange(99 + 150)
    
    output_dabrmeanpi_report2 = output_2report_abr[0:150] - nanmean(output_2report_pi[0:99])
    output_dabrmeanpi_predict2 = areamean_dict_predi['LWP_area_yr_abr'][0:150] - nanmean(areamean_dict_predi['LWP_area_yr_pi'][0:99])
    output_dabrmeanpi_GMT2 = areamean_dict_abr['gmt_area_yr'][0:150] - mean(areamean_dict_PI['gmt_area_yr'])
    
    # Regressed delta_LWP over delta_GMT, using 'statsmodels' ols functions
    data = pd.DataFrame({'x': output_dabrmeanpi_GMT2, 'y1':output_dabrmeanpi_report2, 'y2':output_dabrmeanpi_predict2})
    
    model_report = ols("y1 ~ x", data).fit()
    model_predicted = ols("y2 ~ x", data).fit()
    
    print(" d_LWP/d_GMT model report summary: ", model_report._results.params[1], model_report._results.params[0])
    print(" d_LWP/d_GMT model predict summary: ", model_predicted._results.params[1], model_predicted._results.params[0])
    
    #..save into rawdata_dict
    Dx_DtG = asarray([[model_report._results.params[1], model_report._results.params[0]], [model_predicted._results.params[1], model_predicted._results.params[0]]])
    rawdata_dict['dX_dTg'] = Dx_DtG
    
    # Generate continous annual mean array are convenient for plotting LWP changes:
    #..Years from 'piControl' to 'abrupt4xCO2' experiment, which are choosed years
    Yrs = arange(shape_yr_pi+shape_yr_abr)
    rawdata_dict['Yrs'] = Yrs

    # global-mean surface air temperature, from 'piControl' to 'abrupt4xCO2' experiment:
    
    GMT = full((shape_yr_pi + areamean_dict_abr['gmt_area_yr'].shape[0]),  0.0)
    GMT[0:shape_yr_pi] = areamean_dict_PI['gmt_area_yr']
    GMT[shape_yr_pi:] = areamean_dict_abr['gmt_area_yr']
    rawdata_dict['GMT'] = GMT
    
    # LRM predict annual mean, area-mean values, from 'piControl' to 'abrupt4xCO2' experiment
    predict_metrics_annually = {}
    report_metrics_annually = {}
    
    for h in range(len(datapredi_nas)):
        predict_metrics_annually[datapredi_nas[h]] = full((shape_yr_pi + areamean_dict_predi[datapredi_nas[h] + '_area_yr_abr'].shape[0]), 0.0)
        predict_metrics_annually[datapredi_nas[h]][0:shape_yr_pi] = areamean_dict_predi[datapredi_nas[h] + '_area_yr_pi'][0:shape_yr_pi]
        predict_metrics_annually[datapredi_nas[h]][shape_yr_pi:(shape_yr_pi + areamean_dict_predi[datapredi_nas[h] + '_area_yr_abr'].shape[0])] = areamean_dict_predi[datapredi_nas[h]+'_area_yr_abr']
        
    # GCM actual annual mean, area-mean values, from 'piControl' to 'abrupt4xCO2' experiments:

    for i in range(len(datarepo_nas)):
        report_metrics_annually[datarepo_nas[i]] = full((shape_yr_pi + areamean_dict_abr[datarepo_nas[i] + '_area_yr'].shape[0]), 0.0)  
        report_metrics_annually[datarepo_nas[i]][0:shape_yr_pi] = areamean_dict_PI[datarepo_nas[i] + '_area_yr'][0:shape_yr_pi]
        report_metrics_annually[datarepo_nas[i]][shape_yr_pi:(shape_yr_pi+areamean_dict_abr[datarepo_nas[i] + '_area_yr'].shape[0])] = areamean_dict_abr[datarepo_nas[i]+'_area_yr']
    
    print("report LWP: ", report_metrics_annually['LWP'])
    print("predicted LWP: ", predict_metrics_annually['LWP'])
    
    # put them into the rawdata_dict:
    rawdata_dict['predicted_metrics'] = predict_metrics_annually
    rawdata_dict['report_metrics'] = report_metrics_annually
    
    return rawdata_dict



def fitLRM1_splitperiod(C_dict, TR_sst, s_range, y_range, x_range, lats, lons):
    # 'C_dict' is the raw data dict, 'TR_sst' is the pre-defined skin_Temperature Threshold to distinguish two multi-linear regression models
    # 's_range , 'y_range', 'x_range' used to do area mean
    
    # 'Result_dict' is the dictionary for storing the piControl coef and the 15-yr splitted coefs in abrupt4xCO2 exp:
    Result_dict = {}
    
    # read data:
    dict0_abr_var = C_dict['dict1_abr_var']
    dict0_PI_var = C_dict['dict1_PI_var']
    #print(dict0_PI_var['times'])

    model = C_dict['model_data']   #.. type in dict
    datavar_nas = ['LWP', 'TWP', 'IWP', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'SST', 'p_e', 'LTS', 'SUB'] #..12 varisables except for gmt
    
    # load annual mean bin data:
    dict1_yr_bin_PI = dict0_PI_var['dict1_yr_bin_PI']
    dict1_yr_bin_abr = dict0_abr_var['dict1_yr_bin_abr']
    #print(dict1_yr_bin_PI['LWP_yr_bin'].shape)
    
    # load monthly bin data:
    dict1_mon_bin_PI = dict0_PI_var['dict1_mon_bin_PI']
    dict1_mon_bin_abr = dict0_abr_var['dict1_mon_bin_abr']
    
    # load data shape:
    shape_yr_PI = dict0_PI_var['shape_yr']
    shape_yr_abr = dict0_abr_var['shape_yr']
    shape_mon_PI = dict0_PI_var['shape_mon']
    shape_mon_abr = dict0_abr_var['shape_mon']
    

    # for the entire 99 years period of piControl exp, calc the LRM coefficients and save:
    dict2_predi_fla_PI = {}
    dict2_predi_ano_PI = {}  # need climatological values of variables
    dict2_predi_nor_PI = {}  # standardized anomalies of variables
    
    dict2_predi = {}
    #.. Flatten binned array /Standardized data ARRAY :
    for a in range(len(datavar_nas)):
        
        dict2_predi_fla_PI[datavar_nas[a]] = dict1_mon_bin_PI[datavar_nas[a]+'_mon_bin'].flatten()
        # anomalies in the raw units:
        # 'dict2_predi' saves the reference-period (piControl) mean for data variable
        dict2_predi[datavar_nas[a]] = deepcopy(dict1_mon_bin_PI[datavar_nas[a]+'_mon_bin'])
        dict2_predi_ano_PI[datavar_nas[a]] = dict2_predi_fla_PI[datavar_nas[a]] - nanmean(area_mean(dict2_predi[datavar_nas[a]], y_range, x_range))
        # normalized stardard deviation in unit of './std':
        dict2_predi_nor_PI[datavar_nas[a]] = dict2_predi_ano_PI[datavar_nas[a]] / nanstd(dict2_predi_fla_PI[datavar_nas[a]])  # divided by 1 standard deviation of monthly data
    
    # gmt
    GMT_pi_mon = area_mean(dict1_mon_bin_PI['gmt_mon_bin'], s_range, x_range)  #.. Monthly time series of global mean surface air temperature
    dict2_predi_fla_PI['gmt'] = GMT_pi_mon
    dict2_predi['gmt'] = deepcopy(dict2_predi_fla_PI['gmt'])
    dict2_predi_ano_PI['gmt'] = dict2_predi_fla_PI['gmt'] - nanmean(dict2_predi['gmt'])
    dict2_predi_nor_PI['gmt'] = dict2_predi_ano_PI['gmt'] / nanstd(dict1_mon_bin_PI['gmt_mon_bin'])
    
    metric_training = deepcopy(dict2_predi_ano_PI)
    #.. Training 1Regime LRM
    #.. piControl
    predict_dict_PI, _, _, coef_array_PI, shape_fla_training = rdlrm_1_training(metric_training, predictant='LWP')
    Result_dict["piControl"] = coef_array_PI
    
    
    # for every 15 years period in abrupt4xCO2 exp, calc the LRM coefficients and saved for data array:
    print(shape_yr_abr)
    for t in range(shape_yr_abr//15):
        print(t)
        # Splitwise according to each 15 years:
        dict2_predi_fla_abr = {}
        dict2_predi_ano_abr = {}
        dict2_predi_nor_abr = {}

        #.. Flatten binned array /Standardized data ARRAY :
        for b in range(len(datavar_nas)):

            dict2_predi_fla_abr[datavar_nas[b]] = dict1_mon_bin_abr[datavar_nas[b]+'_mon_bin'][12*15*t:12*15*(t+1),:,:].flatten()

            # anomalies in the raw units:
            # 'dict2_predi' saves the reference-period (piControl) mean for data variable
            dict2_predi_ano_abr[datavar_nas[b]] = dict2_predi_fla_abr[datavar_nas[b]] - nanmean(area_mean(dict2_predi[datavar_nas[b]], y_range,x_range))
            # normalized stardard deviation in unit of './std':
            dict2_predi_nor_abr[datavar_nas[b]] = dict2_predi_ano_abr[datavar_nas[b]] / nanstd(dict2_predi_fla_PI[datavar_nas[b]]) # divided by 1 standard deviation of monthly data
        
        GMT_abr_mon = area_mean(dict1_mon_bin_abr['gmt_mon_bin'][12*15*t:12*15*(t+1),:,:], s_range, x_range)  #.. Monthly time series of global mean surface air temperature
        dict2_predi_fla_abr['gmt'] = GMT_abr_mon
        dict2_predi_ano_abr['gmt'] = dict2_predi_fla_abr['gmt'] - nanmean(dict2_predi['gmt'])
        dict2_predi_nor_abr['gmt'] = dict2_predi_ano_abr['gmt'] / nanstd(dict1_mon_bin_PI['gmt_mon_bin'])
    
        metric_predict = deepcopy(dict2_predi_ano_abr)
        #.. Training 1Regime LRM
        #.. for every 15 years of abrupt4xCO2
        predict_dict_abr_split, _, _, coef_array_abr_split, shape_fla_abr_split = rdlrm_1_training(metric_predict, predictant='LWP')
        Result_dict["abrupt4xCO2"+"_"+str(t)] = coef_array_abr_split
    
    return Result_dict 
