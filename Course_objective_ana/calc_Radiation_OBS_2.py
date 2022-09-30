### Try to replicate Daniel's methods:

import netCDF4
import numpy as np
import pandas
import glob
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm

from scipy.stats import *
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from get_LWPCMIP5data import *
from get_LWPCMIP6data import *
from get_OBSLRMdata import *
from useful_func_cy import *
from fitLRM_cy1 import *
from fitLRM_cy2 import *

from fitLRMobs import *
from useful_func_cy import *
from calc_Radiation_LRM_1 import *
from calc_Radiation_LRM_2 import *

from area_mean import *
from binned_cyFunctions5 import *
from useful_func_cy import *


def calc_Radiation_OBS_2(s_range, x_range, y_range, valid_range1 = [2002, 7, 15], valid_range2 = [2016, 12, 31], valid_range3 = [1994, 1, 15], valid_range4 = [2001, 12, 31]):
    
    # -----------------
    # 'valid_range1' and 'valid_range2' give the time stamps of starting and ending times of data for training,
    # 'valid_range3' and 'valid_range4' give the time stamps of starting and ending times of data for predicting.
    # 's_range', 'x_range', 'y_range' is the latitude (Global), latitude (Southern Ocean), and longitude for 5 * 5 binned data;
    # ------------------
    
    # get the variables for training:
    inputVar_training_obs = get_OBSLRM(valid_range1=valid_range1, valid_range2=valid_range2)
    
    # get the variables for predicting:
    inputVar_predict_obs = get_OBSLRM(valid_range1=valid_range3, valid_range2=valid_range4)
    
    # As data dictionary:
    datavar_nas = ['LWP', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'alpha_cre']   #..7 varisables except gmt (lon dimension diff)
    variable_MAC = ['LWP', 'LWP_error', 'Maskarray_mac']
    variable_CERES = ['rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'alpha_cre']
    
    # Training Data processing:
    # Liquid water path, Unit in kg m^-2
    LWP_training = inputVar_training_obs['lwp'] / 1000.
    # 1-Sigma Liquid water path statistic error, Unit in kg m^-2
    LWP_error_training = inputVar_training_obs['lwp_error'] / 1000.
    # the MaskedArray of 'MAC-LWP' dataset
    Maskarray_mac_training = inputVar_training_obs['maskarray_mac']
    # ---

    # SW radiative flux:
    Rsdt_training = inputVar_training_obs['rsdt']
    Rsut_training = inputVar_training_obs['rsut']
    Rsutcs_training = inputVar_training_obs['rsutcs']

    albedo_training = Rsut_training / Rsdt_training
    albedo_cs_training = Rsutcs_training / Rsdt_training
    Alpha_cre_training = albedo_training - albedo_cs_training
    
    # abnormal values:
    albedo_cs_training[(albedo_cs_training <= 0.08) & (albedo_cs_training >= 1.00)] = np.nan
    Alpha_cre_training[(albedo_cs_training <= 0.08) & (albedo_cs_training >= 1.00)] = np.nan
    
    dict0_training_var = {'LWP': LWP_training, 'LWP_error': LWP_error_training, 'Maskarray_mac': Maskarray_mac_training, 'rsdt': Rsdt_training, 'rsut': Rsut_training, 'rsutcs': Rsutcs_training, 'albedo' : albedo_training, 'albedo_cs': albedo_cs_training, 'alpha_cre': Alpha_cre_training, 'times': inputVar_training_obs['times_ceres']}
    
    # Crop the regions
    # crop the variables to the Southern Ocean latitude range: (40 ~ 85^o S)
    dict1_SO_training, lat_so, lon_so = region_cropping(dict0_training_var, ['LWP', 'LWP_error', 'Maskarray_mac'], inputVar_training_obs['lat_mac'], inputVar_training_obs['lon_mac'], lat_range =[-85., -40.], lon_range = [-180., 180.])
    
    dict1_SO_training['lat'] = lat_so
    dict1_SO_training['lon'] = lon_so
    
    # As data dictionary:
    datavar_nas = ['LWP', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'alpha_cre']   #..7 varisables except gmt (lon dimension diff)
    variable_MAC = ['LWP', 'LWP_error', 'Maskarray_mac']
    variable_CERES = ['rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'alpha_cre']
    
    # Predict Data processing:
    # Liquid water path, Unit in kg m^-2
    LWP_predict = inputVar_predict_obs['lwp'] / 1000.
    # 1-Sigma Liquid water path statistic error, Unit in kg m^-2
    LWP_error_predict = inputVar_predict_obs['lwp_error'] / 1000.
    # the MaskedArray of 'MAC-LWP' dataset
    Maskarray_mac_predict = inputVar_predict_obs['maskarray_mac']
    # ---

    # SW radiative flux:
    Rsdt_predict = inputVar_predict_obs['rsdt']
    Rsut_predict = inputVar_predict_obs['rsut']
    Rsutcs_predict = inputVar_predict_obs['rsutcs']

    albedo_predict = Rsut_predict / Rsdt_predict
    albedo_cs_predict = Rsutcs_predict / Rsdt_predict
    Alpha_cre_predict = albedo_predict - albedo_cs_predict
    
    # abnormal values
    albedo_cs_predict[(albedo_cs_predict <= 0.08) & (albedo_cs_predict >= 1.00)] = np.nan
    Alpha_cre_predict[(albedo_cs_predict <= 0.08) & (albedo_cs_predict >= 1.00)] = np.nan
    
    dict0_predict_var = {'LWP': LWP_predict, 'LWP_error': LWP_error_predict, 'Maskarray_mac': Maskarray_mac_predict, 'rsdt': Rsdt_predict, 'rsut': Rsut_predict, 'rsutcs': Rsutcs_predict, 'albedo' : albedo_predict, 'albedo_cs': albedo_cs_predict, 'alpha_cre': Alpha_cre_predict, 'times': inputVar_predict_obs['times_ceres']}
    
    # Crop the regions
    # crop the variables to the Southern Ocean latitude range: (40 ~ 85^o S)
    dict1_SO_predict, lat_so, lon_so = region_cropping(dict0_predict_var, ['LWP', 'LWP_error', 'Maskarray_mac'], inputVar_predict_obs['lat_mac'], inputVar_predict_obs['lon_mac'], lat_range =[-85., -40.], lon_range = [-180., 180.])
    
    dict1_SO_predict['lat'] = lat_so
    dict1_SO_predict['lon'] = lon_so
    
    dict2_training_var = deepcopy(dict1_SO_training)
    dict2_predict_var = deepcopy(dict1_SO_predict)
    
    print('the first month in training and predict data: ', dict1_SO_training['times'][0,:][1], dict1_SO_predict['times'][0,:][1])
    
    
    # Choose time frame: January:
    if dict1_SO_training['times'][0,:][1] == 1.0:   # Jan
        shape_mon_training_raw = dict1_SO_training['LWP'][0::12, :,:].shape   # January data shape
        for i in range(len(datavar_nas)):
            dict2_training_var[datavar_nas[i]] = dict1_SO_training[datavar_nas[i]][0::12, :, :]   # January data
    else:
        shape_mon_training_raw = dict1_SO_training['LWP'][int(13 - dict1_SO_training[0,:][1])::12, :,:].shape 
        for i in range(len(datavar_nas)):
            dict2_training_var[datavar_nas[i]] = dict1_SO_training[datavar_nas[i]][int(13 - dict1_SO_training['times'][0,:][1])::12, :, :]

    if dict1_SO_predict['times'][0,:][1] == 1.0:   # Jan
        shape_mon_abr_raw = dict1_SO_predict['LWP'][0::12,:,:].shape   # January data shape
        for j in range(len(datavar_nas)):
            dict2_predict_var[datavar_nas[j]] = dict1_SO_predict[datavar_nas[j]][0::12, :, :]   # January data

    else:
        shape_mon_abr_raw = dict1_SO_predict['LWP'][int(13 - dict1_SO_predict['times'][0,:][1])::12, latsi0:latsi1 +1,:].shape 
        for j in range(len(datavar_nas)):
            dict2_predict_var[datavar_nas[j]] = dict1_SO_predict[datavar_nas[j]][int(13 - dict1_SO_predict['times'][0,:][1])::12, :, :]
    
    # radiative transfer model: single regime LRM:

    # training :

    x_training = 1.* dict2_training_var['LWP']
    
    y2_training = 1.* dict2_training_var['alpha_cre']
    
    y1_training = 1.* dict2_training_var['albedo']
    
    cs_training = 1.* dict2_training_var['albedo_cs']
    
    rsdt_training = 1.* dict2_training_var['rsdt']
    
    # Filter threshold:
    rsdt_training[rsdt_training < 10.0] = np.nan
    
    cs_training[cs_training < 0.0] = np.nan
    cs_training[cs_training > 1.0] = np.nan

    Z_training = (rsdt_training * cs_training * x_training * y2_training * y1_training) *1.
    ind_false_training = np.isnan(Z_training)
    ind_true_training = np.logical_not(ind_false_training)
    
    print(" ratio of not NaN value in Training data :" + str(np.asarray(np.nonzero(ind_true_training == True)).shape[1]/len(ind_true_training.flatten())))
    
    data_training = pandas.DataFrame({'x': x_training[ind_true_training].flatten(), 'y2': y2_training[ind_true_training].flatten(), 'y1': y1_training[ind_true_training].flatten(), 'cs': cs_training[ind_true_training].flatten()})

    # Fit the model
    model1_training = ols("y2 ~ x", data_training).fit()
    model2_training = ols("y1 ~ x + cs", data_training).fit()
    # print the summary
    print("model1_training, alpha_cre = a1 * lwp + A2: ", ' ', model1_training.summary())
    print("model2_training, albedo = a1* lwp + a2 * albedo_cs + A3: ", ' ', model2_training.summary())

    coef_array_alpha_cre_training = np.asarray([model1_training._results.params[1], model1_training._results.params[0]])
    coef_array_albedo_training = np.asarray([model2_training._results.params[1], model2_training._results.params[2], model2_training._results.params[0]])
    
    # Compare to the training:
    # predicting :

    x_predict = 1.* dict2_predict_var['LWP']
    
    y2_predict = 1.* dict2_predict_var['alpha_cre']
    
    y1_predict = 1.* dict2_predict_var['albedo']
    
    cs_predict = 1.* dict2_predict_var['albedo_cs']
    
    rsdt_predict = 1.* dict2_predict_var['rsdt']
    
    # Filter threshold:
    rsdt_predict[rsdt_predict < 10.0] = np.nan
    
    cs_predict[cs_predict < 0] = np.nan
    cs_predict[cs_predict > 1] = np.nan

    Z_predict = (rsdt_predict * cs_predict * x_predict * y2_predict * y1_predict) *1.
    ind_false_predict = np.isnan(Z_predict)
    ind_true_predict = np.logical_not(ind_false_predict)
    
    print(" ratio of not NaN value in Predict data :" + str(np.asarray(np.nonzero(ind_true_predict == True)).shape[1]/len(ind_true_predict.flatten())))
    
    data_predict = pandas.DataFrame({'x': x_predict[ind_true_predict].flatten(), 'y2': y2_predict[ind_true_predict].flatten(), 'y1': y1_predict[ind_true_predict].flatten(), 'cs': cs_predict[ind_true_predict].flatten()})

    # Fit the model
    model1_predict = ols("y2 ~ x", data_predict).fit()
    model2_predict = ols("y1 ~ x + cs", data_predict).fit()
    # print the summary
    print("model1_predict, alpha_cre = a1 * lwp + A2: ", ' ', model1_predict.summary())
    print("model2_predict, albedo = a1* lwp + a2 * albedo_cs + A3: ", ' ', model2_predict.summary())

    coef_array_alpha_cre_predict = np.asarray([model1_predict._results.params[1], model1_predict._results.params[0]])
    coef_array_albedo_predict = np.asarray([model2_predict._results.params[1], model2_predict._results.params[2], model2_predict._results.params[0]])
    
    return coef_array_alpha_cre_training, coef_array_albedo_training, coef_array_alpha_cre_predict, coef_array_albedo_predict