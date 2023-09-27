# ..flatten all the data array, from that built a SST_thershold based Linear Regession Model(LRM);
# assess its behavior (RMSE/ R^2) and do the regression for PI and abr4x experiments, then build an array for whole-period-LWP


import netCDF4
from numpy import *

import matplotlib.pyplot as plt
import matplotlib as mpl 
import xarray as xr
# import PyNIO as Nio # deprecated
import pandas as pd
import glob
from scipy.stats import *
from copy import deepcopy
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm

from scipy.optimize import curve_fit
import seaborn as sns
import cartopy.crs as ccrs   #..projection method
import cartopy.feature as cfeat

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter   #..x,y --> lon, lat



def region_cropping(dict_raw, variable_nas, lats, lons, lat_range = [-85., -40.], lon_range = [-180., 180.]):
    # This function is for cropping the raw data maps (save in dictipnary) to be a defined rectangle region map.
    # 'dict_raw' is the dictionary which saved all variable(s).
    # 'variable_nas' is the list of variable(s) we wants to crop, e.g., variable_nas = ['SST', 'p_e', 'LTS', 'SUB'].
    # 'lat_range[lati1, lati2]' defines the south and north bounds of the cropping map; 'lon_range[lonj1, lonj2]' defines the west and east bounds.
    
    dict_crop = deepcopy(dict_raw)
    
    ind_i = (lats >= min(lat_range)) & (lats <= max(lat_range))
    lats_crop = lats[ind_i]
    ind_j = (lons >= min(lon_range)) & (lons <= max(lon_range))
    lons_crop = lons[ind_j]
    # print(lats_crop, lons_crop)
    
    for v in range(len(variable_nas)):
        
        if dict_crop[variable_nas[v]].ndim == 3:
            dict_crop[variable_nas[v]] = dict_crop[variable_nas[v]][:, ind_i, :]
            dict_crop[variable_nas[v]] = dict_crop[variable_nas[v]][:,:, ind_j]  # variable in dictionary is in 3-D shape
        elif dict_crop[variable_nas[v]].ndim == 2:
            dict_crop[variable_nas[v]] = dict_crop[variable_nas[v]][ind_i, :]
            dict_crop[variable_nas[v]] = dict_crop[variable_nas[v]][:, ind_j]  # variable in dictionary is in 2-D shape
        
        else:
            print(' A not valid dimension value for this variable: ', variable_nas[v])
            continue
    
    print(' ended cropping ')
    
    return dict_crop, lats_crop, lons_crop


def region_cropping_var(data, lats, lons, lat_range = [-85., -40.], lon_range = [-180., 180.]):
    # This function is for cropping the raw data map to be a defined rectangle region map.
    # 'data' is the data variable (only one) which need to crop.
    # 'lat'/ 'lon' is the 1-D latitude/ longitude array.
    # 'lat_range[lati1, lati2]' defines the south and north bounds of the cropping map; 'lon_range[lonj1, lonj2]' defines the west and east bounds.
    S = data *1.
    
    ind_i = (lats >= min(lat_range)) & (lats <= max(lat_range))
    lats_crop = lats[ind_i]
    ind_j = (lons >= min(lon_range)) & (lons <= max(lon_range))
    lons_crop = lons[ind_j]
    # print(lats_crop, lons_crop)
    
    if S.ndim == 3:
        S = data[:, ind_i, :]
        S = S[:,:, ind_j]  # variable in dictionary is in 3-D shape
    elif S.ndim == 2:
        S = data[ind_i, :]
        S = S[:, ind_j]  # variable in dictionary is in 2-D shape

    # print(' ended cropping ')
    return S, lats_crop, lons_crop



def annual_mean(data, shape_m0, shape_1, shape_2):
    ###..'data' is the origin data array for 3D variable(i.e., (times, lat, lon)), 
    ###. 'shape_m0' was the shape of dimension_times, should be in 'mon' before converting to 'yr'
    
    shape_yr = shape_m0//12   #.. times dimension shapes in annual
    ##. 'layover_yr' is the data array for storing the 2-d data array for annual mean
    layover_yr = zeros((shape_yr, shape_1, shape_2))
        
    for i in range(shape_yr):

        layover_yr[i, :, :] = nanmean(data[i*12:(i+1)*12,:,:], axis=0)
    
    return layover_yr



def get_annual_metric(data, times, label = 'mon'):
    # This function is for converting finer time scale data to annual mean data;
    # ..currently can only process: monthly data;
    # The default data shape is: (time, lat, lon);
    # 'times' is the time array of data, which in shape of (length_of_time, 3) for storing (year, month, day) information.
    
    if label == 'mon':
        shape_yr = asarray(times).shape[0]// 12
        annual_array = zeros((shape_yr, asarray(data).shape[1], asarray(data).shape[2]))
        
        # Is the first month of data be 'January'? 
        if times[0, 1]== 1.0:  # start at January
            for i in range(shape_yr):
                annual_array[i,:,:] = nanmean(data[i*12:(i+1)*12, :,:], axis = 0)
        elif any(times[0,1]== x for x in [2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]):  # not start at January
            for i in range(shape_yr):
                annual_array[i,:,:] = nanmean(data[i*12+(13-int(times[0,1])):(i+1)*12+(13-int(times[0,1])),:,:], axis = 0)
        else:
            print('wrong month value.')
    
    return annual_array



def get_annual_dict_so(dict_rawdata, dict_names, shape_time, lat_si0, lat_si1, shape_lon):
    #.. 'dict_rawdat' : originally in monthly data, all variables are 3D(times, lat, lon) data in the same shape;
    #.. 'shape_time' : # of Months, which as the 1st dimension in each variables INSIDE 'dict_rawdata';
    #.. 'dict_names' : the name string list (or a dict) of each variables Inside 'dict_rawdata' and you wanted to calc the annual mean;
    #.. 'lat_si0': the smallest index of latitude of bound; 'lat_si1': the largest index of latitude of bound

    dict_yr  = {}
    shape_lat_so = int(lat_si1)+1 - int(lat_si0)
    
    layover_yr  = zeros((len(dict_names), shape_time//12, shape_lat_so, shape_lon))   #tips: dictionary didn't really copy each changeable value of 'layover_yr', but more like an ‘indicator' who points to the address of 'layover_yr'.



    for a in range(len(dict_names)):
        a_array = dict_rawdata[dict_names[a]]
    
        for i in range(shape_time//12):
            #.. '//' representing 'int' division operation
            
            layover_yr[a, i, :, :] = nanmean(a_array[i*12:(i+1)*12, lat_si0:lat_si1+1, :], axis=0)
        
        #tips: dictionary didn't really copy the value into the 'dict_yr', but works like an ‘indicator'
        dict_yr[dict_names[a]+'_yr'] =  layover_yr[a,:,:,:]
        print(dict_names[a], "convert annual data done")

    return dict_yr



def get_annual_dict(dict_rawdata, variable_nas, times, label = 'mon'):
    #.. This function is for converting finer time scale data in the dict to be annual data, and save into another dict.
    #.. currently can only handle 'monthly' data;
    #.. 'dict_rawdata' : the data dictionary save all the variables, the default shape is: (times, lat, lon);
    #.. 'variable_nas' : the string list of variable(s) that we wanted to calc the annual mean;
    #.. 'times' : is the time array of data, which in shape of (length_of_time, 3) for storing (year, month, day) information.
    
    dict_annual_mean = {}
    
    if label == 'mon':
        shape_yr = asarray(times).shape[0]// 12
        
        for v in range(len(variable_nas)):
            annual_array = zeros((shape_yr, asarray(dict_rawdata[variable_nas[v]]).shape[1], asarray(dict_rawdata[variable_nas[v]]).shape[2]))
            
            # Is the first month of data be 'January'? 
            if times[0, 1]== 1.0:  # start at January
                for i in range(shape_yr):
                    annual_array[i,:,:] = nanmean(dict_rawdata[variable_nas[v]][i*12:(i+1)*12, :,:], axis = 0)
            elif any(times[0,1]== x for x in [2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]):  # not start at January
                for i in range(shape_yr):
                    annual_array[i,:,:] = nanmean(dict_rawdata[variable_nas[v]][i*12+(13-int(times[0,1])):(i+1)*12+(13-int(times[0,1])),:,:], axis = 0)
            else:
                print('wrong month value.')

            dict_annual_mean[variable_nas[v]] = annual_array
    
    return dict_annual_mean



def area_mean(data, lats, lons):
    # This function is for processing the latitudinal weighting of 3-D array.
    '''..3D (time, lat, lon) would reduce to 1D (time)..
    '''
    Time_series = zeros(data.shape[0])
    for i in arange(data.shape[0]):
        
        S_time_step = data[i,:,:]
        # remove the NaN value within the 2-D array and squeeze to 1-D (indexing):
        ind1 = isnan(S_time_step)==False
        # weighted by cosine(lat):
        xlon, ylat = meshgrid(lons, lats)

        weighted_metrix1 = cos(deg2rad(ylat))  # lat matrix
        toc1 = sum(weighted_metrix1[ind1])   # total of cos(lat matrix) for the defined region of lat X lon

        S_weighted = S_time_step[ind1] * weighted_metrix1[ind1] / toc1
        
        Time_series[i] = sum(S_weighted)

    return Time_series



def latitude_mean(X, lats, lons, lat_range=[-85., -40.]):
    """
    Calculate the area(latitudinal)-mean of a 3-d metrics.
    """
    #### X  in shape: time (models), lat, lon
    
    S = zeros(X.shape[0])
    for i in arange(X.shape[0]):
        
        X_time_step = X[i,:,:]
        xx = nanmean(X_time_step, axis=(1)) ## now a vector in latitude- no weights in time or longitude.
        ind1 = (lats<=max(lat_range)) & (lats>=min(lat_range))
        # Remove the NaN value in 1-D vector
        ind_truevector = isnan(xx)==False
        # Combined indexes which both satisfy the Latitude range & are not NaN
        ind2 = logical_and(ind1, ind_truevector)
        # area weighting:
        latrad = cos(lats * pi / 180.)
        xx_weight_mean = sum(xx[ind2]*latrad[ind2]) / sum(latrad[ind2])
    
        
        S[i] = xx_weight_mean
        
        
    return S



def rdlrm_1_training(X_dict, predictant = 'LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 1):
    # single regime model: training the model from 'piControl' variable to get a single set of Coef
    # 'predict_dict' is a dictionary to store the 'predict_label_LWP' and 'predict_value_LWP'
    predict_dict  = {}
    
    # 'predict_label_LWP' is an array to store the regimes_label
    predict_label_LWP = zeros((X_dict['SST'].shape[0]))
    
    # 'predict_value_LWP' is an array to store the predicted LWP
    predict_value_LWP = zeros((X_dict['SST'].shape[0]))
    
    # 'predictors' is an array that has the need predictors in flatten format;
    Predictors = []

    for i in range(len(predictor)):
        Predictors.append(X_dict[predictor[i]] *1.)
    Predictors = asarray(Predictors)
    print("predictors metrix shape: ", Predictors.shape)  # (4,  ..)
    
    shape_fla_training = X_dict[predictant].shape
    print("shape1: ", shape_fla_training)   # shape1
    
    
    # Detecting nan values in the CCFs metrics
    Z = X_dict['LTS'] * 1.

    for j in range(len(predictor)):
        Z = Z * (Predictors[j, :]* 1.)
    Z = Z * (X_dict[predictant]* 1.)

    ind_false = isnan(Z)
    ind_true = logical_not(ind_false)
    
    print("shape2: ", asarray(nonzero(ind_true ==True)).shape)
    
    # Replace '0'/'nan' value in right place
    predict_label_LWP[ind_false] = 0
    predict_value_LWP[ind_false] = nan

    # print Totol # of regimes
    Regimes = [ind_false]
    print(' Total # of regime', len(Regimes))

    # Multiple linear regression of the predictant to the predictor(s) :
    regr0 = linear_model.LinearRegression()
    result0 = regr0.fit(Predictors[:][0:len(predictor), ind_true].T, X_dict[predictant][ind_true])
    #..Save the coef and intp
    aeffi = result0.coef_
    aintp = result0.intercept_

    
    # '1' for valid_data indeing; '0' for invalid_data ('nan') points' indexing
    predict_label_LWP[ind_true] = 1
    
    
    # Save coefs and intps:
    coef_array_list = deepcopy([aeffi, aintp])
    coef_array = asarray(coef_array_list, dtype = object)
    # print(coef_array.shape)
    
    # Save predicted Value, and save values and labels into predict_dict
    predict_value_LWP[ind_true] = dot(aeffi.reshape(1, -1), Predictors[:][0:len(predictor), ind_true]).flatten() + aintp  #.. valid data points

    predict_dict['label'] = predict_label_LWP
    predict_dict['value'] = predict_value_LWP
    
    return predict_dict, ind_true, ind_false, coef_array, shape_fla_training



def rdlrm_1_predict(X_dict, coef_array, predictant = 'LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 1):
    
    # 'predict_dict' is a dictionary to store the 'predict_label_LWP' and 'predict_value_LWP' (for CCF1, 2, 3, 4,.. and the intercept);
    predict_dict = {}

    # 'predict_label_LWP' is an array to store the regimes_lebel of each grid points in 3-D structure of data array
    predict_label_LWP = zeros((X_dict['SST'].shape[0]))

    # 'predict_value_LWP' is an array to store the predicted LWP
    predict_value_LWP = zeros((X_dict['SST'].shape[0]))
    
    # 'predictors' is an array that has the need predictors in flatten format;
    Predictors = []
    
    for i in range(len(predictor)):
        Predictors.append(X_dict[predictor[i]] *1.)
    Predictors = asarray(Predictors)
    print(Predictors.shape)  # (4, ..)

    shape_fla_testing = X_dict[predictant].shape
    print("shape1: ", shape_fla_testing)   # shape1

    # Detecting nan values in the CCFs metrics
    Z  = X_dict['LTS'] * 1. 

    for j in range(len(predictor)):
        Z  =  Z * (Predictors[j, :]* 1.)

    Z = Z * (X_dict[predictant]* 1.)
    
    
    ind_false = isnan(Z)
    ind_true = logical_not(ind_false)
    
    print("shape2: ", asarray(nonzero(ind_true==True)).shape)

    # Replace '0'/ 'nan' value in right place
    predict_label_LWP[ind_false] = 0
    predict_value_LWP[ind_false] = nan

    # print Total # of regimes
    Regimes = [ind_true]
    print(' Total # of regime', len(Regimes))

    for k in range(len(Regimes)):
        print('current # of regimes', k)
        ind = Regimes[k]
        # labels of regimes
        predict_label_LWP[ind] = k + 1

        # predict values
        predict_value_LWP[ind] = dot(coef_array[0].reshape(1, -1), Predictors[:][0:len(predictor), ind]).flatten() + coef_array[1]  #.. valid data pointt
    # print("predict_value_LWP ", predict_value_LWP)
    # print("label", predict_label_LWP)  # '1' for valid_data, '2' for invalid_data ('nan') points

    predict_dict['label'] = predict_label_LWP
    predict_dict['value'] = predict_value_LWP
    

    return predict_dict, ind_true, ind_false, shape_fla_testing



def Test_performance_1(A, B, ind_True, ind_False = None):
    from sklearn.metrics import mean_squared_error, r2_score
    
    stats_dict = {}

    MSE_shape1 = mean_squared_error(A[ind_True].reshape(-1,1), B[ind_True].reshape(-1,1))
    R_2_shape1 = r2_score(A[ind_True].reshape(-1, 1), B[ind_True].reshape(-1, 1))
    stats_shape1 = [sqrt(MSE_shape1), R_2_shape1]

    stats_dict = {'shape1': stats_shape1}

    return stats_dict



def rdlrm_2_training(X_dict, cut_off1, predictant = 'LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 2):

    # 'predict_dict' is a dictionary to store the 'predict_label_LWP' and 'predict_value_LWP'
    predict_dict  = {}

    # 'predict_label_LWP' is an array to store the regimes_label
    predict_label_LWP = zeros((X_dict['SST'].shape[0]))
    
    # 'predict_value_LWP' is an array to store the predicted LWP
    predict_value_LWP = zeros((X_dict['SST'].shape[0]))

    # 'predictors' is an array that has the need predictors in flatten format;
    Predictors = []

    for i in range(len(predictor)):
        Predictors.append(X_dict[predictor[i]] *1.)
    Predictors = asarray(Predictors)
    print("predictors metrix shape: ", Predictors.shape)  # (4, ..)

    shape_fla_training = X_dict[predictant].shape
    print('shape1: ', shape_fla_training)   # shape1

    print('2LRM: HERE TR_sst = ', cut_off1, 'K')  #.. # of total flatten points
    
    # Detecting nan values in the CCFs metrics
    Z  = X_dict['LTS'] * 1.

    for j in range(len(predictor)):
        Z  =  Z * (Predictors[j, :]* 1.)

    Z = Z * (X_dict[predictant]* 1.)
    ind_false = isnan(Z)

    ind_true = logical_not(ind_false)
    print('shape2: ', asarray(nonzero(ind_true==True)).shape )  #.. # of 'non-nan'
    
    # Replace 'nan' value in right place
    predict_label_LWP[ind_false] = 0
    predict_value_LWP[ind_false] = nan

    
    # Split data with skin Temperature (SST) Larger\Equal and Less than Cut_off1
    ind_hot  = X_dict['SST'] >= cut_off1
    ind_cold = X_dict['SST'] < cut_off1

    ind6 = ind_true & ind_hot
    ind7 = ind_true & ind_cold

    # print('shape6: ', ind6.shape)   #.. points with sst >= Cutoff1 and non-nan
    # print('shape7: ', ind7.shape)   #.. points with sst < Cutoff1 and non-nan
    
    Regimes  = [ind6, ind7]
    print(' Total # of regime', len(Regimes))
    
    #.. Multiple linear regreesion of Liquid Water Path to CCFs :

    # train model with sst >= TR_sst, unit in K

    regr1 = linear_model.LinearRegression()
    result1 = regr1.fit(Predictors[:][0:len(predictor), ind6].T,  X_dict[predictant][ind6])
    #..Save the coef and intp:
    aeffi = result1.coef_
    aintp = result1.intercept_

    # train model with SST < TR_sst, unit in K
    if len(ind7)!=0:
        regr2 = linear_model.LinearRegression()
        result2 = regr2.fit(Predictors[:][0:len(predictor), ind7].T, X_dict[predictant][ind7])

        beffi = result2.coef_
        bintp = result2.intercept_
    else:
        beffi = full(len(predictor), 0.0)
        bintp = 0.0

    # '1' for 'Cold' regime; '2' for 'Hot' regime
    predict_label_LWP[ind7] = 1
    predict_label_LWP[ind6] = 2
    
    # Save coefs and intps
    coef_array_list = deepcopy([[beffi, bintp], [aeffi, aintp]])
    coef_array = asarray(coef_array_list, dtype = object)
    # print("coef_array's shape: ", coef_array.shape)
    
    # Save predicted Values
    predict_value_LWP[ind6] = dot(aeffi.reshape(1, -1), Predictors[:][0:len(predictor), ind6]).flatten() +aintp  #..larger or equal than Tr_SST
    predict_value_LWP[ind7] = dot(beffi.reshape(1, -1), Predictors[:][0:len(predictor), ind7]).flatten() +bintp  #..less than Tr_SST

    predict_dict['label'] = predict_label_LWP
    predict_dict['value'] = predict_value_LWP
    
    
    return predict_dict, ind6, ind7, coef_array, shape_fla_training



def rdlrm_2_predict(X_dict, coef_array, cut_off1, predictant = 'LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 2):
    # 'predict_dict' is a dictionary to store the 'predict_label_LWP' and 'predict_value_LWP' (for CCF1, 2, 3, 4,.. and the intercept);
    predict_dict = {}

    # 'predict_label_LWP' is an array to store the regimes_lebel of each grid points in 3-D structure of data array
    predict_label_LWP = zeros((X_dict['SST'].shape[0]))
    
    # 'predict_value_LWP' is an array to store the predicted LWP
    predict_value_LWP = zeros((X_dict['SST'].shape[0]))
    print(predict_value_LWP.shape)
    # 'predictors' is an array that has the need predictors in flatten format;
    Predictors = []
    
    for i in range(len(predictor)):
        Predictors.append(X_dict[predictor[i]] *1.)
    Predictors = asarray(Predictors)
    # print(Predictors.shape)  # (4, ..)
    
    shape_fla_testing = X_dict[predictant].shape
    print('shape1: ', shape_fla_testing)   # shape1
    
    # Detecting nan values in the CCFs metrics
    Z = X_dict['LTS'] * 1. 
    
    for j in range(len(predictor)):
        Z = Z * Predictors[j, :]
    
    Z = Z * (X_dict[predictant]* 1.)
    ind_false = isnan(Z)
    
    ind_true = logical_not(ind_false)
    print('shape2: ', asarray(nonzero(ind_true==True)).shape)  #.. # of 'non-nan'
    
    # Replace 'nan' value in right place
    predict_label_LWP[ind_false] = 0 
    predict_value_LWP[ind_false] = nan
    
    
    # LOOP THROUGH REGIMES ('2'):
    # split data with skin Temperature (SST) Larger\Equal and Less than Cut_off1
    
    ind_hot = X_dict['SST'] >= cut_off1
    ind_cold = X_dict['SST'] < cut_off1 
    # 
    # ind_up   = X_dict['SUB'] <= cut_off2
    # ind_down = X_dict['SUB'] > cut_off2
    ind6 = ind_true & ind_hot
    ind7 = ind_true & ind_cold
    
    Regimes = [ind7, ind6]
    print(' Total # of regime', len(Regimes))
    
    for k in range(len(Regimes)):
        print('current # of regimes', k)
        ind = Regimes[k]
        # labels of regimes
        predict_label_LWP[ind] = k + 1
        
        # predict values
        predict_value_LWP[ind] = dot(coef_array[k,0].reshape(1, -1), Predictors[:][0:len(predictor), ind]).flatten() + coef_array[k,1]  #..larger or equal than Tr_SST
    
    # print("predict_value_LWP ", predict_value_LWP)
    # print("label", predict_label_LWP)  # '1' for 'Cold' regime, '2' for 'Warm' regime
    
    
    predict_dict['label'] = predict_label_LWP
    predict_dict['value'] = predict_value_LWP
    
    return predict_dict, ind6, ind7, shape_fla_testing



def Test_performance_2(A, B, ind6, ind7):
    from sklearn.metrics import mean_squared_error, r2_score
    
    
    stats_dict = {}

    MSE_shape1 =  mean_squared_error(A[logical_or(ind6, ind7)].reshape(-1,1), B[logical_or(ind6, ind7)].reshape(-1,1))
    R_2_shape1  = r2_score(A[logical_or(ind6, ind7)].reshape(-1, 1), B[logical_or(ind6, ind7)].reshape(-1, 1))
    stats_shape1 = [sqrt(MSE_shape1), R_2_shape1]

    if (len(ind7)!=0) & (len(ind6)!=0):
        MSE_shape6 = mean_squared_error(A[ind6].reshape(-1,1), B[ind6].reshape(-1,1))
        R_2_shape6 = r2_score(A[ind6].reshape(-1,1), B[ind6].reshape(-1,1))

        R_2_shape7 = r2_score(A[ind7].reshape(-1, 1), B[ind7].reshape(-1, 1))
        MSE_shape7 = mean_squared_error(A[ind7].reshape(-1, 1), B[ind7].reshape(-1,1))
    
    else:
        print(" R_2_shape7 is zero shape because TR_sst <= all available SST data, ")
        print(" or Non-appropraited TR_sub value if choose 2-lrm with 'Up & Dn'.")
        R_2_shape7  = nan
        MSE_shape7  = nan
    
        R_2_shape6  = nan
        MSE_shape6  = nan
    
    stats_shape6 = [sqrt(MSE_shape6), R_2_shape6]
    stats_shape7 = [sqrt(MSE_shape7), R_2_shape7]

    stats_dict = {'shape1': stats_shape1, 'shape2': stats_shape6, 'shape3': stats_shape7}

    return stats_dict



def rdlrm_4_training(X_dict, cut_off1, cut_off2, predictant = 'LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 4):
    # 'If r = 4: divided by Warm/ Cold (by SST) & Up/ Dn (By SUB500) regimes'
    # 'If r = 2: divided by only Up/ Down (By SUB500) regimes'
    
    # 'predict_dict' is a dictionary to store the 'predict_label_LWP' and 'predict_value_LWP'
    predict_dict  = {}

    # 'predict_label_LWP' is an array to store the regimes_label
    predict_label_LWP = zeros((X_dict['SST'].shape[0]))
    
    # 'predict_value_LWP' is an array to store the predicted LWP
    predict_value_LWP = zeros((X_dict['SST'].shape[0]))

    # 'predictors' is an array that has the need predictors in flatten format;
    Predictors = []

    for i in range(len(predictor)):
        Predictors.append(X_dict[predictor[i]]* 1.)
    Predictors = asarray(Predictors)
    # print(Predictors.shape)  # (4, ..)

    shape_fla_training = X_dict[predictant].shape
    print('shape1: ', shape_fla_training)   # shape1

    # Detecting nan values in the CCFs metrics
    Z  = X_dict['LTS'] * 1.

    for j in range(len(predictor)):
        Z  =  Z * (Predictors[j, ]* 1.)

    Z = Z * (X_dict[predictant]* 1.)
    ind_false = isnan(Z)

    ind_true = logical_not(ind_false)
    print('shape2: ', asarray(nonzero(ind_true==True)).shape)  #.. # of 'non-nan' 
    
    # Replace 'nan' value in right place
    predict_label_LWP[ind_false] = 0
    predict_value_LWP[ind_false] = nan

    if r == 4:
    
        print('4LRM: HERE TR_sst = ', cut_off1, 'K')  #.. # of total flatten points
        print('4LRM:  ... TR_sub = ', cut_off2, 'Pa s-1')

        # Split data with skin Temperature (SST) Larger\Equal and Less than Cut_off1
        ind_hot  = X_dict['SST'] >= cut_off1
        ind_cold = X_dict['SST'] < cut_off1
        # Split data with 500mb Subsidence (SUB) Less\Equal and Larger than Cut_off2
        ind_up   = X_dict['SUB'] <= cut_off2
        ind_down = X_dict['SUB'] > cut_off2

        ind7 = ind_true & ind_cold & ind_up
        ind8 = ind_true & ind_hot & ind_up
        print('shape7 and 8: ', asarray(nonzero(ind7 ==True)).shape, ' and ', asarray(nonzero(ind8 ==True)).shape)
        ind9 = ind_true & ind_cold & ind_down
        ind10 = ind_true & ind_hot & ind_down
        print('shape9 and 10: ', asarray(nonzero(ind9 ==True)).shape, ' and ', asarray(nonzero(ind10 ==True)).shape)

        Regimes  = [ind7, ind8, ind9, ind10]
        print(' Total # of regime', len(Regimes))

        #.. Multiple linear regreesion of Liquid Water Path to CCFs:


        # train model with SST < TR_sst, unit in K
        if (len(ind7)!=0) & (len(ind8)!=0) & (len(ind9)!=0) & (len(ind10)!=0):
            regr7 = linear_model.LinearRegression()
            result7 = regr7.fit(Predictors[:][0:len(predictor), ind7].T, X_dict[predictant][ind7])   #..regression for LWP WITH LTS and skin-T < TR_sst & 'up'
            aeffi = result7.coef_
            aintp = result7.intercept_

            regr8 = linear_model.LinearRegression()
            result8 = regr8.fit(Predictors[:][0:len(predictor), ind8].T, X_dict[predictant][ind8])   #..regression for LWP WITH LTS and skin-T >= TR_sst &'up'
            beffi = result8.coef_
            bintp = result8.intercept_

            regr9 = linear_model.LinearRegression()
            result9 = regr9.fit(Predictors[:][0:len(predictor), ind9].T, X_dict[predictant][ind9])   #..regression for LWP WITH LTS and skin-T < TR_sst & 'down'
            ceffi = result9.coef_
            cintp = result9.intercept_

            regr10 = linear_model.LinearRegression()
            result10 = regr10.fit(Predictors[:][0:len(predictor), ind10].T, X_dict[predictant][ind10])   #..regression for LWP WITH LTS and skin-T >= TR_sst & 'down'
            deffi = result10.coef_
            dintp = result10.intercept_

        elif (len(ind7)==0) & (len(ind9)==0):
            aeffi = full(len(predictors), 0.0)
            aintp = 0.0

            regr8 = linear_model.LinearRegression()
            result8 = regr8.fit(Predictors[:][0:len(predictor), ind8].T, X_dict[predictant][ind8])   #..regression for LWP WITH LTS and skin-T >= TR_sst &'up'
            beffi = result8.coef_
            bintp = result8.intercept_

            ceffi = full(len(predictors), 0.0)
            cintp = 0.0

            regr10 = linear_model.LinearRegression()
            result10 = regr10.fit(Predictors[:][0:len(predictor), ind10].T, X_dict[predictant][ind10])   #..regression for LWP WITH LTS and skin-T >= TR_sst & 'down'
            deffi = result10.coef_
            dintp = result10.intercept_

        else:
            print('you input a non-wise value for TR_sub at 500 mb')
            print('please try another TR_sub input...')

        # '1' for 'Cold' & 'Up' regime; '2' for 'Warm' & 'Up' regime; '3' for 'Cold' and 'Down' regime; and '4' for 'Warm' and 'Down' regime
        predict_label_LWP[ind7] = 1
        predict_label_LWP[ind8] = 2
        predict_label_LWP[ind9] = 3
        predict_label_LWP[ind10] = 4
        
        # Save coefs and intps:
        coef_array_list = deepcopy([[aeffi, aintp], [beffi, bintp], [ceffi, cintp], [deffi, dintp]])
        coef_array = asarray(coef_array_list, dtype = object)
        # print(asarray(coef_array).shape)

        # Save predict Value 
        predict_value_LWP[ind7] = dot(aeffi.reshape(1, -1), Predictors[:][0:len(predictor), ind7]).flatten() +aintp  #..less than Tr_SST and less/euqal to Tr_SUB
        predict_value_LWP[ind8] = dot(beffi.reshape(1, -1), Predictors[:][0:len(predictor), ind8]).flatten() +bintp  #..larger or equal than Tr_SST and less/euqal to Tr_SUB
        predict_value_LWP[ind9] = dot(ceffi.reshape(1, -1), Predictors[:][0:len(predictor), ind9]).flatten() +cintp  #..less than Tr_SST and larger than Tr_SUB
        predict_value_LWP[ind10] = dot(deffi.reshape(1, -1), Predictors[:][0:len(predictor), ind10]).flatten() +dintp  #..larger or equal than Tr_SST and larger than Tr_SUB

        predict_dict['label'] = predict_label_LWP
        predict_dict['value'] = predict_value_LWP

    if r == 2:
    
        print('2LRM: Up& Dn regimes by TR_sub = ', cut_off2, 'Pa s-1')
        
        # Split data with 500mb Subsidence (SUB) Less\Equal and Larger than Cut_off2
        ind9 = X_dict['SUB'] <= cut_off2  # 'ind_up'
        ind10 = X_dict['SUB'] > cut_off2  # 'ind_down'
        
        ind7 = ind_true & ind9
        ind8 = ind_true & ind10
        
        print('shape7 and 8: ', asarray(nonzero(ind7 ==True)), ' and ', asarray(nonzero(ind8 ==True)))
        
        Regimes  = [ind7, ind8]
        print(' Total # of regime', len(Regimes))
        
        #.. Multiple linear regression of predictant to predictor(s) 
        if (len(ind7)!=0) & (len(ind8)!=0):
            regr7 = linear_model.LinearRegression()
            result7 = regr7.fit(Predictors[:][0:len(predictor), ind7].T, X_dict[predictant][ind7])   #..regression for LWP WITH LTS and skin-T < TR_sst & 'up'
            aeffi = result7.coef_
            aintp = result7.intercept_

            regr8 = linear_model.LinearRegression()
            result8 = regr8.fit(Predictors[:][0:len(predictor), ind8].T, X_dict[predictant][ind8])   #..regression for LWP WITH LTS and skin-T >= TR_sst &'up'
            beffi = result8.coef_
            bintp = result8.intercept_
        
        if (len(ind7)==0) or (len(ind8)==0):
            print("Non-appropriated TR_SUB value: has some regime be zero shape.")
            
            aeffi, beffi = full(len(predictor), 0.0), full(len(predictor), 0.0)
            aintp, bintp = 0.0, 0.0
            
        # '1' for 'Up' regime; '2' for 'Down' regime;
        predict_label_LWP[ind7] = 1
        predict_label_LWP[ind8] = 2
        
        
        # Save coefs and intps:
        coef_array_list = deepcopy([[aeffi, aintp], [beffi, bintp]]) # 'Up' and 'Down'
        coef_array = asarray(coef_array_list, dtype = object)
        
        # Save predicted values and labels into predict_dict
        predict_value_LWP[ind7] = dot(aeffi.reshape(1, -1), Predictors[:][0:len(predictor), ind7]).flatten() +aintp  #..less/euqal to Tr_SUB
        predict_value_LWP[ind8] = dot(beffi.reshape(1, -1), Predictors[:][0:len(predictor), ind8]).flatten() +bintp  #..larger than Tr_SUB

        predict_dict['label'] = predict_label_LWP
        predict_dict['value'] = predict_value_LWP

    return predict_dict, ind7, ind8, ind9, ind10, coef_array, shape_fla_training



def rdlrm_4_predict(X_dict, coef_array, cut_off1, cut_off2, predictant = 'LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 4):
    # 'If r = 4: divided by Hot/ Cold (by SST) & Up/ Dn (By SUB500) regimes'
    # 'If r = 2: divided by only Up/ Down (By SUB500) regimes'

    # 'predict_dict' is a dictionary to store the 'predict_label_LWP' and 'predict_value_LWP' (for CCF1, 2, 3, 4,.. and the intercept);
    predict_dict = {}

    # 'predict_label_LWP' is an array to store the regimes_label of each grid points in 3-D structure of data array
    predict_label_LWP = zeros((X_dict['SST'].shape[0]))

    # 'predict_value_LWP' is an array to store the predicted LWP
    predict_value_LWP = zeros((X_dict['SST'].shape[0]))
    
    # 'predictors' is an array that has the needed predictors values in flattened format:
    Predictors = []

    for i in range(len(predictor)):
        Predictors.append(X_dict[predictor[i]]* 1.)
    Predictors = asarray(Predictors)
    # print(Predictors.shape)  # (4, ..)

    shape_fla_testing = X_dict[predictant].shape
    print('shape1: ', shape_fla_testing)   # shape1

    # Detecting nan values in the CCFs metrics
    Z  = X_dict['LTS'] * 1.

    for j in range(len(predictor)):
        Z  =  Z * Predictors[j, :]

    Z = Z * (X_dict[predictant]* 1.)
    ind_false = isnan(Z)

    ind_true = logical_not(ind_false)
    print('shape2: ', asarray(nonzero(ind_true==True)).shape)  #.. # of 'non-nan'

    # Replace '0'/ 'nan' value in the right place:
    predict_label_LWP[ind_false] = 0
    predict_value_LWP[ind_false] = nan

    if r == 4:
        # LOOP THROUGH REGIMES ('4'):
    
        # split data with skin Temperature (SST) Larger\Equal & Less than Cut_off1
        ind_hot = X_dict['SST'] >= cut_off1
        ind_cold = X_dict['SST'] < cut_off1
        # split data with 500mb Subsidence (SUB) Less\Equal & Larger than Cut_off2
        ind_up  = X_dict['SUB'] <= cut_off2
        ind_down = X_dict['SUB'] > cut_off2

        ind7 = ind_true & ind_cold & ind_up
        ind8 = ind_true & ind_hot & ind_up

        ind9 = ind_true & ind_cold & ind_down
        ind10 = ind_true & ind_hot & ind_down

        Regimes = [ind7, ind8, ind9, ind10]
        print(' Total # of regime', len(Regimes))

        for k in range(len(Regimes)):
            print('current # of regimes', k)
            ind  = Regimes[k]
            # labels of regimes
            predict_label_LWP[ind] = k + 1

            # predict values
            predict_value_LWP[ind] = dot(coef_array[k,0].reshape(1, -1), Predictors[:][0:len(predictor), ind]).flatten() + coef_array[k,1]  #..larger or equal than Tr_SST

        # print("predict_value_LWP ", predict_value_LWP)
        # print("label", predict_label_LWP)  # '1' for 'Cold'& 'Up' regime, '2' for 'Hot'& 'Up' regime; '3' for 'Cold'& 'Down' regime; and '4' for 'Hot'& 'Down' regime.

        predict_dict['label'] = predict_label_LWP
        predict_dict['value'] = predict_value_LWP

    if r == 2:
        # LOOP THROUGH REGIMES ('2'):
    
        ind9 = X_dict['SUB'] <= cut_off2   # 'ind_up'
        ind10 = X_dict['SUB'] > cut_off2   # 'ind_down'
    
        ind7 = ind_true & ind9
        ind8 = ind_true & ind10

        Regimes = [ind7, ind8]
        print(' Total # of regime', len(Regimes))

        for k in range(len(Regimes)):
            print('current # of regimes', k)
            ind  = Regimes[k]
            # labels of regimes
            predict_label_LWP[ind] = k + 1
        
            # predict values
            predict_value_LWP[ind] = dot(coef_array[k,0].reshape(1, -1), Predictors[:][0:len(predictor), ind]).flatten() + coef_array[k,1]  #..larger or equal than Tr_SST
        
        # print("predict_value_LWP ", predict_value_LWP)
        # print("label", predict_label_LWP)  # '1' for the 'Up' regime, '2' for the 'Down' regime;

        predict_dict['label'] = predict_label_LWP
        predict_dict['value'] = predict_value_LWP
    return predict_dict, ind7, ind8, ind9, ind10, shape_fla_testing



def Test_performance_4(A, B, ind7, ind8, ind9, ind10):
    
    from sklearn.metrics import mean_squared_error, r2_score

    ind_true1 = logical_or(ind8, ind7)
    ind_true2 = logical_or(ind10, ind9)
    ind_true = logical_or(ind_true1, ind_true2)
    stats_dict = {}

    MSE_shape1 =  mean_squared_error(A[ind_true].reshape(-1,1), B[ind_true].reshape(-1,1))
    R_2_shape1  = r2_score(A[ind_true].reshape(-1, 1), B[ind_true].reshape(-1, 1))
    stats_shape1 = [sqrt(MSE_shape1), R_2_shape1]

    MSE_shape8 = mean_squared_error(A[ind8].reshape(-1,1), B[ind8].reshape(-1,1))
    R_2_shape8 = r2_score(A[ind8].reshape(-1,1), B[ind8].reshape(-1,1))
    stats_shape8 = [sqrt(MSE_shape8), R_2_shape8]
    
    MSE_shape10 = mean_squared_error(A[ind10].reshape(-1,1), B[ind10].reshape(-1,1))
    R_2_shape10 = r2_score(A[ind10].reshape(-1,1), B[ind10].reshape(-1,1))
    stats_shape10 = [sqrt(MSE_shape10), R_2_shape10]

    if (len(ind7)!=0) & (len(ind9)!=0):
        R_2_shape7 = r2_score(A[ind7].reshape(-1, 1), B[ind7].reshape(-1, 1))
        MSE_shape7 = mean_squared_error(A[ind7].reshape(-1, 1), B[ind7].reshape(-1,1))
        
        R_2_shape9 = r2_score(A[ind9].reshape(-1, 1), B[ind9].reshape(-1, 1))
        MSE_shape9 = mean_squared_error(A[ind9].reshape(-1, 1), B[ind9].reshape(-1,1))
    else:
        print(" R_2_shape7 and R_2_shape9 is nan because TR_sst <= all available SST data. ")
        print(" Or input non-wise Tr_SUB value. ")

        R_2_shape7  = nan
        MSE_shape7  = nan
        
        R_2_shape9  = nan
        MSE_shape9  = nan

    stats_shape7 = [sqrt(MSE_shape7), R_2_shape7]
    stats_shape9 = [sqrt(MSE_shape9), R_2_shape9]
    stats_dict = {'shape1': stats_shape1, 'shape7': stats_shape7, 'shape8': stats_shape8, 'shape9': stats_shape9, 'shape10': stats_shape10}

    return stats_dict



# Building functions for individual contributions from each predictors:
def rdlrm_4_predict_individual(X_dict, coef_array, cut_off1, cut_off2 , CCFs = 4 , r = 4):
    
    # 'predict_dict' is a dictionary to store the 'predict_label_LWP' and 'predict_value_LWP' (for CCF1, 2, 3, 4, and the intercept);
    predict_dict  = {}

    # 'predict_label_LWP' is an array to store the regimes_lebel of each grid points in 3-D structure of data array;
    predict_label_LWP = zeros((X_dict['p_e'].shape[0], X_dict['p_e'].shape[1], X_dict['p_e'].shape[2]))
    
    # 'predict_value_LWP' is a list (5) to store the individual CCFs-driven LWP component & the intercepsts' contribution.
    ## Should in shape (CCFs +1); 
    predict_value_LWP = [X_dict['SST'] *1., X_dict['p_e'] *1., X_dict['LTS'] *1., X_dict['SUB'] *1.,
                         ones((X_dict['SST'].shape[0], X_dict['SST'].shape[1], X_dict['SST'].shape[2]))]
    # print(predict_value_LWP)
    
    # individual factor names list: "if_NAS" ;
    if_NAS = ['SST', 'p_e', 'LTS', 'SUB', 'intp']

    
    # LOOP THROUGH REGIMES ('4'):
    
    # indexes for input data that satisfied the 'TR_sst' (cut_off1) and 'TR_sub' (cut_off2')
    
    ind_hot  = X_dict['SST'] >=  cut_off1
    ind_cold = X_dict['SST'] < cut_off1

    ind_up   = X_dict['SUB'] <= cut_off2
    ind_down = X_dict['SUB'] > cut_off2
    
    Regimes  = [ind_cold & ind_up, ind_hot & ind_up, ind_cold & ind_down, ind_hot & ind_down]
    print(' Total # of regime', len(Regimes))
    
    
    for i in range(len(Regimes)):
        print('current # of regimes', i)
        ind  = Regimes[i]
        
        predict_label_LWP[ind] = i + 1
        
        # LOOP THROUGH Cloud Controlling Factors ('4') and intercepts ('+1'):
        
        for j in range(len(predict_value_LWP)):
            print('current # of ccfs', j)
            if j <  4: 
                predict_value_LWP[j][ind]  = (coef_array[i,0][j] * predict_value_LWP[j][ind])
            elif j == len(predict_value_LWP)-1: 
                predict_value_LWP[j][ind]  = (coef_array[i,1] * predict_value_LWP[j][ind])
    
    # Detect 'NaN' values:
    Z  = 1. * X_dict['LTS']
    
    for k in range(len(predict_value_LWP)): 
        Z  =  Z * predict_value_LWP[k]
        ind_f = isnan(Z)
    
    # match all the points with the same 'NaN' POSITIONS:
    
    for l in range(len(predict_value_LWP)): 
        
        predict_value_LWP[l][ind_f] = nan
        
    # print("predict_value_LWP ", predict_value_LWP)
    
    # print("label", predict_label_LWP)  # '1' for 'Cold' & 'Up' regime, '2' for 'Hot' & 'Up' regime; '3' for 'Cold' & 'Down'; '4' for 'Hot' & 'Down'

    predict_dict['label'] =  predict_label_LWP
    predict_dict['value'] =  predict_value_LWP
    

    return predict_dict



def rdlrm_1_training_raw(X_dict, lats, lons, predictant = 'LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 1, Filter_LandIce = False):
    # single regime model: training the model from 'piControl' variable to get a single set of Coef
    # 'predict_dict' is a dictionary to store the 'predict_label_LWP' and 'predict_value_LWP'
    # 'X_dict' in this case is in raw resolution, and global, 12 month data with weird 'nan'/'inf' values in some models, handle them carefully.

    # restrict to January data, and 40~85 O^S region
    
    # define predict_dict:
    predict_dict  = {}
    # 'predict_label_LWP' is an array to store the regimes_label
    predict_label_LWP = zeros((X_dict[predictant].shape[0]))
    # 'predict_value_LWP' is an array to store the predicted LWP
    predict_value_LWP = zeros((X_dict[predictant].shape[0]))
    
    # 'predictors' is an array that has the need predictors in flatten format;
    Predictors = []

    for i in range(len(predictor)):
        Predictors.append(X_dict[predictor[i]] *1.)
    Predictors = asarray(Predictors)
    # print("predictors metrix shape: ", Predictors.shape)  # (4, ..)
    
    # shape of 'Predictant' or ' Predictor' variable:
    shape_fla_training = X_dict[predictant].shape
    print("shape1: ", shape_fla_training)   # shape1
    
    # Detecting nan values in the CCFs metrics:
    Z = X_dict['LTS'] * 1.

    for j in range(len(predictor)):
        Z = Z * (Predictors[j, :]* 1.)
    Z = Z * (X_dict[predictant]* 1.)

    ind_false = isnan(Z)
    ind_true = logical_not(ind_false)

    print("shape2: ", asarray(nonzero(ind_true ==True)).shape)

    # Replace '0'/'nan' value in right place:
    predict_label_LWP[ind_false] = 0
    predict_value_LWP[ind_false] = nan
    
    # print Totol # of regimes
    Regimes = [ind_false]
    print(' Total # of regime', len(Regimes))

    # Multiple linear regression of the predictant to the predictor(s) :
    regr0 = linear_model.LinearRegression()
    result0 = regr0.fit(Predictors[:][0:len(predictor), ind_true].T,  X_dict[predictant][ind_true])
    #..Save the coef and intp
    aeffi = result0.coef_
    aintp = result0.intercept_


    # '1' for valid_data indeing; '0' for invalid_data ('nan') points' indexing
    predict_label_LWP[ind_true] = 1

    
    # Save coefs and intps
    coef_array = deepcopy(asarray([aeffi, aintp]))
    # print(asarray(coef_array).shape)
    
    # Save predicted Value, and save values and labels into predict_dict
    predict_value_LWP[ind_true] = dot(aeffi.reshape(1, -1), Predictors[:][0:len(predictor), ind_true]).flatten() + aintp  #.. valid data points

    predict_dict['label'] =  predict_label_LWP
    predict_dict['value'] =  predict_value_LWP
    
    return predict_dict, ind_true, ind_false, coef_array, shape_fla_training


