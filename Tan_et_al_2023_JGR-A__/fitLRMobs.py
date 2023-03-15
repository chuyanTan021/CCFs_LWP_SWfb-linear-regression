# ## training and predict the LWP variation, in 1, 2, or 4 regimes;
# ## estimate their statistic performance (RMSE/ R^2);


import netCDF4
from numpy import *
import matplotlib.pyplot as plt
import xarray as xr
# import PyNIO as Nio  # deprecated
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
from useful_func_cy import *



def fitLRMobs_1(dict_training, dict_predict, s_range, y_range, x_range, lats, lons):
    # training and predicting the historical Observational variation of LWP.
    # dict_training is the dictionary for storing the training data (CCFs, Cloud) in an pre-processed form;
    # dict_predict is the dictionaray for storing the predicting data (CCFs, Cloud) as the same pre-processed form of 'dict_training'.
    # s_range, x_range, y_range are used for doing area_mean.
    
    datavar_obs = ['SST', 'p_e', 'LTS', 'SUB', 'LWP', 'LWP_statistic_error', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'alpha_cre']
    
    # shape of given variables
    shape_training = dict_training['LWP'].shape
    shape_predict = dict_predict['LWP'].shape

    shape_training_gmt = dict_predict['gmt'].shape
    shape_predict_gmt = dict_predict['gmt'].shape
    
    dict2_predi_fla_training = {}
    dict2_predi_fla_predict = {}
    
    dict2_predi_ano_training = {}  # need a climatological arrays of variables
    dict2_predi_ano_predict = {}  # need a climatological arrays of variables
    
    dict2_predi_nor_training = {}
    dict2_predi_nor_predict = {}
    
    dict2_predi = {}
    
    # flatten the variable array for regressing:
    for d in range(len(datavar_obs)):
    
        dict2_predi_fla_training[datavar_obs[d]] = dict_training[datavar_obs[d]].flatten()
        dict2_predi_fla_predict[datavar_obs[d]] = dict_predict[datavar_obs[d]].flatten()
        
        # anomalies in the raw units:
        dict2_predi[datavar_obs[d]] = deepcopy(dict_training[datavar_obs[d]])
        print(dict2_predi[datavar_obs[d]].shape)
        
        dict2_predi_ano_training[datavar_obs[d]] = dict2_predi_fla_training[datavar_obs[d]] - np.nanmean(area_mean(dict2_predi[datavar_obs[d]], y_range, x_range))
        dict2_predi_ano_predict[datavar_obs[d]] = dict2_predi_fla_predict[datavar_obs[d]] - np.nanmean(area_mean(dict2_predi[datavar_obs[d]], y_range, x_range))
        
        # normalized stardard deviation in unit of './std':
        dict2_predi_nor_training[datavar_obs[d]] = dict2_predi_ano_training[datavar_obs[d]] / np.nanstd(dict2_predi_fla_training[datavar_obs[d]])  # divided by std
        dict2_predi_nor_predict[datavar_obs[d]] = dict2_predi_ano_predict[datavar_obs[d]] / np.nanstd(dict2_predi_fla_training[datavar_obs[d]])
    
    
    # Global-Mean surface air Temperature(tas):
    # shape of 'GMT' is the length of time (t)
    dict2_predi_fla_predict['gmt'] = area_mean(dict_predict['gmt'], s_range, x_range)
    ## dict2_predi_fla_PI['gmt'] = GMT_pi.repeat(730)   # something wrong when calc dX_dTg(dCCFS_dgmt)
    dict2_predi_fla_training['gmt'] = area_mean(dict_training['gmt'], s_range, x_range)
    
    dict2_predi['gmt'] = dict2_predi_fla_training['gmt']
    shape_whole_period = np.asarray(dict2_predi['gmt'].shape[0])
    dict2_predi_ano_predict['gmt'] = dict2_predi_fla_predict['gmt'] - np.nanmean(dict2_predi['gmt'])  # shape in (t, lat, lon).flatten()
    dict2_predi_ano_training['gmt'] = dict2_predi_fla_training['gmt'] - np.nanmean(dict2_predi['gmt'])  # shape in (t, lat, lon).flatten()
    
    dict2_predi_nor_predict['gmt'] = dict2_predi_ano_predict['gmt'] / np.nanstd(dict_training['gmt'].flatten())
    dict2_predi_nor_training['gmt'] = dict2_predi_ano_training['gmt'] / np.nanstd(dict_training['gmt'].flatten())
    metric_training = deepcopy(dict2_predi_fla_training)
    metric_predict = deepcopy(dict2_predi_fla_predict)
    
    #.. Training Model (1-LRM, single regime)
    #.. not need the Cloud controlling factor threshold (TR_SST, TR_SUB..)
    
    training_LRM_result, ind_True, ind_False, coef_array_1r, shape_fla_training = rdlrm_1_training(metric_training, predictant='LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 1)
    # 'YB' is the predicted value of LWP in the 'training period':
    YB = training_LRM_result['value']
    
    # Test Performance of LRM
    stats_dict_training = Test_performance_1(metric_training['LWP'], YB, ind_True, ind_False)
    
    # Save 'YB', and resample the shape into 3-D:
    C_dict = {}   # as output
    
    ## currently using raw value as fit/predict values
    C_dict['LWP_actual_training'] = metric_training['LWP'].reshape(shape_training)
    C_dict['LWP_predi_training'] = np.asarray(YB).reshape(shape_training)
    C_dict['training_LRM_result'] = training_LRM_result
    C_dict['coef_dict'] = coef_array_1r
    C_dict['stats_dict_training'] = stats_dict_training
    C_dict['std_LWP_predict'] = np.nanstd(dict2_predi_fla_predict['LWP'])
    C_dict['std_LWP_training'] = np.nanstd(dict2_predi_fla_training['LWP'])
    
    #.. Predict Model (1-LRM, single regime)
    predict_LRM_result, ind_True_predi, ind_False_predi, shape_fla_predicting = rdlrm_1_predict(metric_predict, coef_array_1r, predictant = 'LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 1)
    
    # 'YB_predi' is the predicted value of LWP in the 'predict period':
    YB_predi = predict_LRM_result['value']
    
    # Test Performance of LRM
    stats_dict_predict = Test_performance_1(metric_predict['LWP'], YB_predi, ind_True_predi, ind_False_predi)
    
    # Save 'YB', and resample the shape into 3-D:
    ## currently using raw value as fit/predict values
    C_dict['LWP_actual_predict'] = metric_predict['LWP'].reshape(shape_predict)
    C_dict['LWP_predi_predict'] = np.asarray(YB_predi).reshape(shape_predict)
    C_dict['predict_LRM_result'] = predict_LRM_result
    C_dict['stats_dict_predict'] = stats_dict_predict
    
    C_dict['predict_Array'] = metric_predict
    C_dict['training_Array'] = metric_training
    
    return C_dict


def fitLRMobs_2_updown(dict_training, dict_predict, TR_sst, TR_sub, s_range, y_range, x_range, lats_Array, lons_Array):
    # training and predicting the historical Observational variation of LWP.
    # dict_training is the dictionary for storing the training data (CCFs, Cloud) in an pre-processed form;
    # dict_predict is the dictionaray for storing the predicting data (CCFs, Cloud) as the same pre-processed form of 'dict_training';
    # TR_sst, TR_sub are the thresholds of sea surface temperature & 500 mb Subsidence, for later partition the LRM;
    # s_range, x_range, y_range are used for doing area_mean.
    
    datavar_obs = ['SST', 'p_e', 'LTS', 'SUB', 'LWP', 'LWP_statistic_error', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'alpha_cre']
    
    # shape of given variables
    shape_training = dict_training['LWP'].shape
    shape_predict = dict_predict['LWP'].shape

    shape_training_gmt = dict_predict['gmt'].shape
    shape_predict_gmt = dict_predict['gmt'].shape
    
    dict2_predi_fla_training = {}
    dict2_predi_fla_predict = {}
    
    dict2_predi_ano_training = {}  # need a climatological arrays of variables
    dict2_predi_ano_predict = {}  # need a climatological arrays of variables
    
    dict2_predi_nor_training = {}
    dict2_predi_nor_predict = {}
    
    dict2_predi = {}
    
    # flatten the variable array for regressing:
    for d in range(len(datavar_obs)):
    
        dict2_predi_fla_training[datavar_obs[d]] = dict_training[datavar_obs[d]].flatten()
        dict2_predi_fla_predict[datavar_obs[d]] = dict_predict[datavar_obs[d]].flatten()
        
        # anomalies in the raw units:
        dict2_predi[datavar_obs[d]] = deepcopy(dict_training[datavar_obs[d]])
        print(dict2_predi[datavar_obs[d]].shape)
        
        dict2_predi_ano_training[datavar_obs[d]] = dict2_predi_fla_training[datavar_obs[d]] - np.nanmean(area_mean(dict2_predi[datavar_obs[d]], y_range, x_range))
        dict2_predi_ano_predict[datavar_obs[d]] = dict2_predi_fla_predict[datavar_obs[d]] - np.nanmean(area_mean(dict2_predi[datavar_obs[d]], y_range, x_range))
        
        # normalized stardard deviation in unit of './std':
        dict2_predi_nor_training[datavar_obs[d]] = dict2_predi_ano_training[datavar_obs[d]] / np.nanstd(dict2_predi_fla_training[datavar_obs[d]])  # divided by std
        dict2_predi_nor_predict[datavar_obs[d]] =  dict2_predi_ano_predict[datavar_obs[d]] / np.nanstd(dict2_predi_fla_training[datavar_obs[d]])
    
    # Global-Mean surface air Temperature(tas):
    # shape of 'GMT' is the length of time (t)
    dict2_predi_fla_predict['gmt'] = area_mean(dict_predict['gmt'], s_range, x_range)
    ## dict2_predi_fla_PI['gmt'] = GMT_pi.repeat(730)   # something wrong when calc dX_dTg(dCCFS_dgmt)
    dict2_predi_fla_training['gmt'] = area_mean(dict_training['gmt'], s_range, x_range)
    
    dict2_predi['gmt'] = dict2_predi_fla_training['gmt']
    shape_whole_period = np.asarray(dict2_predi['gmt'].shape[0])
    dict2_predi_ano_predict['gmt'] = dict2_predi_fla_predict['gmt'] - np.nanmean(dict2_predi['gmt'])  # shape in (t, lat, lon).flatten()
    dict2_predi_ano_training['gmt'] = dict2_predi_fla_training['gmt'] - np.nanmean(dict2_predi['gmt'])  # shape in (t, lat, lon).flatten()
    
    dict2_predi_nor_predict['gmt'] = dict2_predi_ano_predict['gmt'] / np.nanstd(dict_training['gmt'].flatten())
    dict2_predi_nor_training['gmt'] = dict2_predi_ano_training['gmt'] / np.nanstd(dict_training['gmt'].flatten())
    metric_training = deepcopy(dict2_predi_nor_training)
    metric_predict = deepcopy(dict2_predi_nor_predict)
    
    # The thresholds: TR_SST, TR_SUB:
    TR_sst_ano = TR_sst - np.nanmean(area_mean(dict2_predi['SST'], y_range, x_range))
    TR_sub_ano = TR_sub - np.nanmean(area_mean(dict2_predi['SUB'], y_range, x_range))
    
    TR_sst_nor = TR_sst_ano / np.nanstd(dict2_predi['SST'].flatten())
    TR_sub_nor = TR_sub_ano / np.nanstd(dict2_predi['SUB'].flatten())
    print(TR_sst_ano, TR_sub_ano)
    print(TR_sst_nor, TR_sub_nor)
    
    #.. Training Module (2 LRM, with Up & Down)
    
    training_LRM_result, ind7_training, ind8_training, ind9_training, ind10_training, coef_array_2r_updown, shape_fla_training = rdlrm_4_training(metric_training, TR_sst_nor, TR_sub_nor, predictant='LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 2)
    # 'ind7_training' / 'ind8_training' are the the non-nan indices corresponding to up / Down regimes, '9' / '10' are simply the up and the down indices.
    # 'YB' is the predicted value of LWP in the 'training period'.
    YB = training_LRM_result['value']
    
    # Test Performance of LRM
    stats_dict_training = Test_performance_2(metric_training['LWP'], YB, ind7_training, ind8_training)
    
    # Save 'YB', and resample the shape into 3-D:
    C_dict = {}   # as output
    ## currently using raw value as fit/predict values
    C_dict['LWP_actual_training'] = metric_training['LWP'].reshape(shape_training)
    C_dict['LWP_predi_training'] = np.asarray(YB).reshape(shape_training)
    C_dict['training_LRM_result'] = training_LRM_result
    C_dict['coef_dict'] = coef_array_2r_updown
    C_dict['stats_dict_training'] = stats_dict_training
    C_dict['std_LWP_predict'] = np.nanstd(dict2_predi_fla_predict['LWP'])
    C_dict['std_LWP_training'] = np.nanstd(dict2_predi_fla_training['LWP'])
    
    #.. Predict Model (2 -LRM, 'Up' and 'Down' regimes)
    predict_LRM_result, ind7_predi, ind8_predi, ind9_predi, ind10_predi, shape_fla_predicting = rdlrm_4_predict(metric_predict, coef_array_2r_updown, TR_sst_nor, TR_sub_nor, predictant = 'LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 2)

    # 'YB_predi' is the predicted value of LWP in the 'predict period':
    YB_predi = predict_LRM_result['value']
    
    # Test Performance of LRM
    stats_dict_predict = Test_performance_2(metric_predict['LWP'], YB_predi, ind7_predi, ind8_predi)
    
    # Save 'YB', and resample the shape into 3-D:
    ## currently using raw value as fit/predict values
    C_dict['LWP_actual_predict'] = metric_predict['LWP'].reshape(shape_predict)
    C_dict['LWP_predi_predict'] = np.asarray(YB_predi).reshape(shape_predict)
    C_dict['predict_LRM_result'] = predict_LRM_result
    
    C_dict['stats_dict_predict'] = stats_dict_predict
    
    return C_dict



def fitLRMobs_2_hotcold(dict_training, dict_predict, TR_sst, TR_sub, s_range, y_range, x_range, lats_Array, lons_Array):
    # training and predicting the historical Observational variation of LWP.
    # dict_training is the dictionary for storing the training data (CCFs, Cloud) in an pre-processed form;
    # dict_predict is the dictionaray for storing the predicting data (CCFs, Cloud) as the same pre-processed form of 'dict_training';
    # TR_sst, TR_sub are the thresholds of sea surface temperature & 500 mb Subsidence, for later partition the LRM;
    # s_range, x_range, y_range are used for doing area_mean.
    
    datavar_obs = ['SST', 'p_e', 'LTS', 'SUB', 'LWP', 'LWP_statistic_error', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'alpha_cre']
    
    # shape of given variables
    shape_training = dict_training['LWP'].shape
    shape_predict = dict_predict['LWP'].shape

    shape_training_gmt = dict_predict['gmt'].shape
    shape_predict_gmt = dict_predict['gmt'].shape
    
    dict2_predi_fla_training = {}
    dict2_predi_fla_predict = {}
    
    dict2_predi_ano_training = {}  # need a climatological arrays of variables
    dict2_predi_ano_predict = {}  # need a climatological arrays of variables
    
    dict2_predi_nor_training = {}
    dict2_predi_nor_predict = {}
    
    dict2_predi = {}

    # flatten the variable array for regressing:
    for d in range(len(datavar_obs)):
    
        dict2_predi_fla_training[datavar_obs[d]] = dict_training[datavar_obs[d]].flatten()
        dict2_predi_fla_predict[datavar_obs[d]] = dict_predict[datavar_obs[d]].flatten()
        
        # anomalies in the raw units:
        dict2_predi[datavar_obs[d]] = deepcopy(dict_training[datavar_obs[d]])
        print(dict2_predi[datavar_obs[d]].shape)
        
        dict2_predi_ano_training[datavar_obs[d]] = dict2_predi_fla_training[datavar_obs[d]] - np.nanmean(area_mean(dict2_predi[datavar_obs[d]], y_range, x_range))
        dict2_predi_ano_predict[datavar_obs[d]] = dict2_predi_fla_predict[datavar_obs[d]] - np.nanmean(area_mean(dict2_predi[datavar_obs[d]], y_range, x_range))
        
        # normalized stardard deviation in unit of './std':
        dict2_predi_nor_training[datavar_obs[d]] = dict2_predi_ano_training[datavar_obs[d]] / np.nanstd(dict2_predi_fla_training[datavar_obs[d]])  # divided by std
        dict2_predi_nor_predict[datavar_obs[d]] =  dict2_predi_ano_predict[datavar_obs[d]] / np.nanstd(dict2_predi_fla_training[datavar_obs[d]])
    
    # Global-Mean surface air Temperature(tas):
    # shape of 'GMT' is the length of time (t)
    dict2_predi_fla_predict['gmt'] = area_mean(dict_predict['gmt'], s_range, x_range)
    ## dict2_predi_fla_PI['gmt'] = GMT_pi.repeat(730)   # something wrong when calc dX_dTg(dCCFS_dgmt)
    dict2_predi_fla_training['gmt'] = area_mean(dict_training['gmt'], s_range, x_range)
    
    dict2_predi['gmt'] = dict2_predi_fla_training['gmt']
    shape_whole_period = np.asarray(dict2_predi['gmt'].shape[0])
    dict2_predi_ano_predict['gmt'] = dict2_predi_fla_predict['gmt'] - np.nanmean(dict2_predi['gmt'])  # shape in (t, lat, lon).flatten()
    dict2_predi_ano_training['gmt'] = dict2_predi_fla_training['gmt'] - np.nanmean(dict2_predi['gmt'])  # shape in (t, lat, lon).flatten()
    
    dict2_predi_nor_predict['gmt'] = dict2_predi_ano_predict['gmt'] / np.nanstd(dict_training['gmt'].flatten())
    dict2_predi_nor_training['gmt'] = dict2_predi_ano_training['gmt'] / np.nanstd(dict_training['gmt'].flatten())
    metric_training = deepcopy(dict2_predi_nor_training)
    metric_predict = deepcopy(dict2_predi_nor_predict)
    
    # The thresholds: TR_SST, TR_SUB:
    TR_sst_ano = TR_sst - np.nanmean(area_mean(dict2_predi['SST'], y_range, x_range))
    TR_sub_ano = TR_sub - np.nanmean(area_mean(dict2_predi['SUB'], y_range, x_range))
    
    TR_sst_nor = TR_sst_ano / np.nanstd(dict2_predi['SST'].flatten())
    TR_sub_nor = TR_sub_ano / np.nanstd(dict2_predi['SUB'].flatten())
    print(TR_sst_ano, TR_sub_ano)
    print(TR_sst_nor, TR_sub_nor)

    
    #.. Training Module (2-LRM, 'Hot' and 'Cold' regimes)
    # 'ind7_training'/ 'ind8_training' are the non-nan indices of 'Hot' & 'Cold' regimes.
    training_LRM_result, ind7_training, ind8_training, coef_array_2r_hotcold, shape_fla_training = rdlrm_2_training(metric_training, TR_sst_nor, predictant='LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'])
    
    # 'YB' is the predicted value of LWP in the 'training period'.
    YB = training_LRM_result['value']
    # Test Performance of LRM
    stats_dict_training = Test_performance_2(metric_training['LWP'], YB, ind7_training, ind8_training)
    
    # Save 'YB', and resample the shape into 3-D:
    C_dict = {}   # as output
    ## currently using raw value as fit/predict values
    C_dict['LWP_actual_training'] = metric_training['LWP'].reshape(shape_training)
    C_dict['LWP_predi_training'] = np.asarray(YB).reshape(shape_training)
    C_dict['training_LRM_result'] = training_LRM_result
    C_dict['coef_dict'] = coef_array_2r_hotcold
    C_dict['stats_dict_training'] = stats_dict_training
    C_dict['std_LWP_predict'] = np.nanstd(dict2_predi_fla_predict['LWP'])
    C_dict['std_LWP_training'] = np.nanstd(dict2_predi_fla_training['LWP'])
    #.. Predict Model (2-LRM, 'Hot' and 'Cold' regimes)
    
    predict_LRM_result, ind7_predi, ind8_predi, shape_fla_testing = rdlrm_2_predict(metric_predict, coef_array_2r_hotcold, TR_sst_nor, predictant = 'LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 2)
    
    # 'YB_predi' is the predicted value of LWP in the 'predict period':
    YB_predi = predict_LRM_result['value']
    
    # Test Performance of LRM
    stats_dict_predict = Test_performance_2(metric_predict['LWP'], YB_predi, ind7_predi, ind8_predi)
    
    # Save 'YB', and resample the shape into 3-D:
    ## currently using raw value as fit/predict values
    C_dict['LWP_actual_predict'] = metric_predict['LWP'].reshape(shape_predict)
    C_dict['LWP_predi_predict'] = np.asarray(YB_predi).reshape(shape_predict)
    C_dict['predict_LRM_result'] = predict_LRM_result
    
    C_dict['stats_dict_predict'] = stats_dict_predict
    
    return C_dict



def fitLRMobs_4(dict_training, dict_predict, TR_sst, TR_sub, s_range, y_range, x_range, lats_Array, lons_Array):
    # training and predicting the historical Observational variation of LWP.
    # dict_training is the dictionary for storing the training data (CCFs, Cloud) in an pre-processed form;
    # dict_predict is the dictionaray for storing the predicting data (CCFs, Cloud) as the same pre-processed form of 'dict_training';
    # TR_sst, TR_sub are the thresholds of sea surface temperature & 500 mb Subsidence, for later partition the LRM;
    # s_range, x_range, y_range are used for doing area_mean.
    
    datavar_obs = ['SST', 'p_e', 'LTS', 'SUB', 'LWP', 'LWP_statistic_error', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'alpha_cre']
    
    # shape of given variables
    shape_training = dict_training['LWP'].shape
    shape_predict = dict_predict['LWP'].shape

    shape_training_gmt = dict_predict['gmt'].shape
    shape_predict_gmt = dict_predict['gmt'].shape
    
    dict2_predi_fla_training = {}
    dict2_predi_fla_predict = {}
    
    dict2_predi_ano_training = {}  # need a climatological arrays of variables
    dict2_predi_ano_predict = {}  # need a climatological arrays of variables
    
    dict2_predi_nor_training = {}
    dict2_predi_nor_predict = {}
    
    dict2_predi = {}
    
    # flatten the variable array for regressing:
    for d in range(len(datavar_obs)):
    
        dict2_predi_fla_training[datavar_obs[d]] = dict_training[datavar_obs[d]].flatten()
        dict2_predi_fla_predict[datavar_obs[d]] = dict_predict[datavar_obs[d]].flatten()
        
        # anomalies in the raw units:
        dict2_predi[datavar_obs[d]] = deepcopy(dict_training[datavar_obs[d]])
        print(dict2_predi[datavar_obs[d]].shape)
        
        dict2_predi_ano_training[datavar_obs[d]] = dict2_predi_fla_training[datavar_obs[d]] - np.nanmean(area_mean(dict2_predi[datavar_obs[d]], y_range, x_range))
        dict2_predi_ano_predict[datavar_obs[d]] = dict2_predi_fla_predict[datavar_obs[d]] - np.nanmean(area_mean(dict2_predi[datavar_obs[d]], y_range, x_range))
        
        # normalized stardard deviation in unit of './std':
        dict2_predi_nor_training[datavar_obs[d]] = dict2_predi_ano_training[datavar_obs[d]] / np.nanstd(dict2_predi_fla_training[datavar_obs[d]])  # divided by std
        dict2_predi_nor_predict[datavar_obs[d]] =  dict2_predi_ano_predict[datavar_obs[d]] / np.nanstd(dict2_predi_fla_training[datavar_obs[d]])
    
    
    # Global-Mean surface air Temperature(tas):
    # shape of 'GMT' is the length of time (t)
    dict2_predi_fla_predict['gmt'] = area_mean(dict_predict['gmt'], s_range, x_range)
    ## dict2_predi_fla_PI['gmt'] = GMT_pi.repeat(730)   # something wrong when calc dX_dTg(dCCFS_dgmt)
    dict2_predi_fla_training['gmt'] = area_mean(dict_training['gmt'], s_range, x_range)
    
    dict2_predi['gmt'] = dict2_predi_fla_training['gmt']
    shape_whole_period = np.asarray(dict2_predi['gmt'].shape[0])
    dict2_predi_ano_predict['gmt'] = dict2_predi_fla_predict['gmt'] - np.nanmean(dict2_predi['gmt'])  # shape in (t, lat, lon).flatten()
    dict2_predi_ano_training['gmt'] = dict2_predi_fla_training['gmt'] - np.nanmean(dict2_predi['gmt'])  # shape in (t, lat, lon).flatten()
    
    dict2_predi_nor_predict['gmt'] = dict2_predi_ano_predict['gmt'] / np.nanstd(dict_training['gmt'].flatten())
    dict2_predi_nor_training['gmt'] = dict2_predi_ano_training['gmt'] / np.nanstd(dict_training['gmt'].flatten())
    metric_training = deepcopy(dict2_predi_nor_training)
    metric_predict = deepcopy(dict2_predi_nor_predict)
    
    # The thresholds: TR_SST, TR_SUB:
    TR_sst_ano = TR_sst - np.nanmean(area_mean(dict2_predi['SST'], y_range, x_range))
    TR_sub_ano = TR_sub - np.nanmean(area_mean(dict2_predi['SUB'], y_range, x_range))
    
    TR_sst_nor = TR_sst_ano / np.nanstd(dict2_predi['SST'].flatten())
    TR_sub_nor = TR_sub_ano / np.nanstd(dict2_predi['SUB'].flatten())
    print(TR_sst_ano, TR_sub_ano)
    print(TR_sst_nor, TR_sub_nor)
    
    #.. Training Module (2 LRM, with Up & Down)
    
    training_LRM_result, ind7_training, ind8_training, ind9_training, ind10_training, coef_array_4r, shape_fla_training = rdlrm_4_training(metric_training, TR_sst_nor, TR_sub_nor, predictant='LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 4)
    
    # 'YB' is the predicted value of LWP in the 'training period':
    YB = training_LRM_result['value']
    
    # Test Performance of LRM
    stats_dict_training = Test_performance_4(metric_training['LWP'], YB, ind7_training, ind8_training, ind9_training, ind10_training)
    
    # Save 'YB', and resample the shape into 3-D:
    C_dict = {}   # as output
    ## currently using raw value as fit/predict values
    C_dict['LWP_actual_training'] = metric_training['LWP'].reshape(shape_training)
    C_dict['LWP_predi_training'] = np.asarray(YB).reshape(shape_training)
    C_dict['training_LRM_result'] = training_LRM_result
    C_dict['coef_dict'] = coef_array_4r
    C_dict['stats_dict_training'] = stats_dict_training
    C_dict['std_LWP_predict'] = np.nanstd(dict2_predi_fla_predict['LWP'])
    C_dict['std_LWP_training'] = np.nanstd(dict2_predi_fla_training['LWP'])
    
    predict_LRM_result, ind7_predi, ind8_predi, ind9_predi, ind10_predi, shape_fla_predicting = rdlrm_4_predict(metric_predict, coef_array_4r, TR_sst_nor, TR_sub_nor, predictant = 'LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 4)
    
    # 'YB_predi' is the predicted value of LWP in the 'predict period':
    YB_predi = predict_LRM_result['value']
    
    # Test Performance of LRM
    stats_dict_predict = Test_performance_4(metric_predict['LWP'], YB_predi, ind7_predi, ind8_predi, ind9_predi, ind10_predi)
    
    # Save 'YB', and resample the shape into 3-D:
    ## currently using raw value as fit/predict values
    
    C_dict['LWP_actual_predict'] = metric_predict['LWP'].reshape(shape_predict)
    C_dict['LWP_predi_predict'] = np.asarray(YB_predi).reshape(shape_predict)
    C_dict['predict_LRM_result'] = predict_LRM_result
    
    C_dict['stats_dict_predict'] = stats_dict_predict
    
    return C_dict

