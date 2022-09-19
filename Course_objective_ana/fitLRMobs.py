### training and predict the LWP variation, in 1, 2, or 4 regimes;
### estimate their statistic performance (RMSE/ R^2);


import netCDF4
from numpy import *
import matplotlib.pyplot as plt
import xarray as xr
# import PyNIO as Nio  # deprecated
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
from useful_func_cy import *
from get_annual_so import *



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
    
    dict2_predi_fla = {}
    
    # flatten the variable array for regressing:
    for d in range(len(datavar_obs)):
    
        dict2_predi_fla_training[datavar_obs[d]] = dict_training[datavar_obs[d]].flatten()
        dict2_predi_fla_predict[datavar_obs[d]] = dict_predict[datavar_obs[d]].flatten()
        
        # anomalies in the raw units:
        # dict2_predi_ano_training[datavar_obs[d]] = dict2_predi_fla_training[datavar_obs[d]] - climatological_Area_mean(t)   # Unfinished
        # dict2_predi_ano_predict[datavar_obs[d]] = dict2_predi_fla_predict[datavar_obs[d]] - climatological_Area_mean(t)   # Unfinished
        
        # normalized stardard deviation in unit of './std':
        # dict2_predi_nor_training[datavar_obs[d]] = dict2_predi_ano_training[datavar_obs[d]] / nanstd(Area_mean(climatological_period_data(t, y, x)))
        # dict2_predi_nor_predict[datavar_obs[d]] =  dict2_predi_ano_predict[datavar_obs[d]] / nanstd(Area_mean(climatological_period_data(t, y, x)))
    
    
    # Global-Mean surface air Temperature(tas):
    # shape of 'GMT' is the length of time (t)
    dict2_predi_fla_predict['gmt'] = area_mean(dict_predict['gmt'], s_range, x_range)
    ## dict2_predi_fla_PI['gmt'] = GMT_pi.repeat(730)   # something wrong when calc dX_dTg(dCCFS_dgmt)
    dict2_predi_fla_training['gmt'] = area_mean(dict_training['gmt'], s_range, x_range)
    
    dict2_predi_fla['gmt'] = np.append(dict2_predi_fla_predict['gmt'], dict2_predi_fla_training['gmt'])
    shape_whole_period = np.asarray(dict2_predi_fla['gmt'].shape[0])
    dict2_predi_ano_predict['gmt'] = dict_predict['gmt'] - np.nanmean(dict2_predi_fla['gmt'])  # shape in (t, lat, lon)
    dict2_predi_ano_training['gmt'] = dict_training['gmt'] - np.nanmean(dict2_predi_fla['gmt'])  # shape in (t, lat, lon)
    
    dict2_predi_nor_predict['gmt'] = dict2_predi_ano_predict['gmt'] / np.nanstd(dict2_predi_fla['gmt'])
    dict2_predi_nor_training['gmt'] = dict2_predi_ano_training['gmt'] / np.nanstd(dict2_predi_fla['gmt'])
    
    #.. Training Model (1-LRM, single regime)
    #..
    
    training_LRM_result, ind_True, ind_False, coef_array_1r, shape_fla_training = rdlrm_1_training(dict2_predi_fla_training, predictant='LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 1)
    # 'YB' is the predicted value of LWP in the 'training period':
    YB = training_LRM_result['value']
    
    # Test Performance of LRM
    stats_dict_training = Test_performance_1(dict2_predi_fla_training['LWP'], YB, ind_True, ind_False)
    
    # Save 'YB', and resample the shape into 3-D:
    C_dict = {}   # as output
    ## currently using raw value as fit/predict values
    C_dict['LWP_actual_training'] = dict2_predi_fla_training['LWP'].reshape(shape_training)
    C_dict['LWP_predi_training'] = np.asarray(YB).reshape(shape_training)
    C_dict['training_LRM_result'] = training_LRM_result
    C_dict['coef_dict'] = coef_array_1r
    C_dict['stats_dict_training'] = stats_dict_training
    
    #.. Predict Model (1-LRM, single regime)
    predict_LRM_result, ind_True_predi, ind_False_predi, shape_fla_predicting = rdlrm_1_predict(dict2_predi_fla_predict, coef_array_1r, predictant = 'LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 1)
    
    # 'YB_predi' is the predicted value of LWP in the 'predict period':
    YB_predi = predict_LRM_result['value']
    
    # Test Performance of LRM
    stats_dict_predict = Test_performance_1(dict2_predi_fla_predict['LWP'], YB_predi, ind_True_predi, ind_False_predi)
    
    # Save 'YB', and resample the shape into 3-D:
    ## currently using raw value as fit/predict values
    C_dict['LWP_actual_predict'] = dict2_predi_fla_predict['LWP'].reshape(shape_predict)
    C_dict['LWP_predi_predict'] = np.asarray(YB_predi).reshape(shape_predict)
    C_dict['predict_LRM_result'] = predict_LRM_result
    
    C_dict['stats_dict_predict'] = stats_dict_predict
    
    
    return C_dict