### training AND predicting the data by Linear Regession Model(LRM) of 1 and 2_UpDown regimes; ###
### estimate their statistic performance (RMSE/ R^2); ###


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

from area_mean import *
from binned_cyFunctions5 import *
from read_hs_file import read_var_mod
from useful_func_cy import *
from get_annual_so import *



def fitLRM(C_dict, TR_sst, s_range, y_range, x_range, lats, lons):
    # 'C_dict' is the raw data dict, 'TR_sst' is the pre-defined skin_Temperature Threshold to distinguish two Multi-Linear Regression Models

    # 's_range , 'y_range', 'x_range' used to do area mean for repeat gmt ARRAY

    dict0_abr_var = C_dict['dict1_abr_var']
    dict0_PI_var  = C_dict['dict1_PI_var']
    #print(dict0_PI_var['times'])

    model = C_dict['model_data']   #.. type in dict

    datavar_nas = ['LWP', 'TWP', 'IWP', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'SST', 'p_e', 'LTS', 'SUB']   #..12 varisables except gmt (lon dimension  diff)

    # load annually-mean bin data.
    dict1_yr_bin_PI  = dict0_PI_var['dict1_yr_bin_PI']
    dict1_yr_bin_abr  = dict0_abr_var['dict1_yr_bin_abr']
    #print(dict1_yr_bin_PI['LWP_yr_bin'].shape)
    
    # load monthly bin data.
    dict1_mon_bin_PI  = dict0_PI_var['dict1_mon_bin_PI']
    dict1_mon_bin_abr  = dict0_abr_var['dict1_mon_bin_abr']

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
    C_dict['shape_yr_PI_3']  = shape_yr_PI
    C_dict['shape_yr_abr_3']  = shape_yr_abr
    C_dict['shape_yr_PI_gmt_3'] = shape_yr_PI_gmt
    C_dict['shape_yr_abr_gmt_3'] = shape_yr_abr_gmt

    C_dict['shape_mon_PI_3'] = shape_mon_PI
    C_dict['shape_mon_abr_3'] = shape_mon_abr
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
    C_dict['GMT_pi_mon'] = GMT_pi_mon
    C_dict['GMT_abr_mon'] = GMT_abr_mon

    #.. Training Module (2lrm)
    

    #.. PI
    
    predict_dict_PI, ind6_PI, ind7_PI, coef_array, shape_fla_training = rdlrm_1_training(dict2_predi_fla_PI, predictant='LWP')
    predict_dict_PI_iwp, ind6_PI_iwp, ind7_PI_iwp, coef_array_iwp, shape_fla_training_iwp = rdlrm_1_training(dict2_predi_fla_PI, predictant='IWP')
    '''
    predict_dict_PI_albedo, _, _, coef_array_albedo = rdlrm_1_training(dict2_predi_fla_PI, predictant='albedo', predictor=['LWP', 'albedo_cs'], r = 1)[0:4]
    predict_dict_PI_rsut, _, _, coef_array_rsut = rdlrm_1_training(dict2_predi_fla_PI, predictant='rsut', predictor=['LWP', 'rsutcs'], r = 1)[0:4]
    '''
    
    
    # Added on May 13th, 2022: for second step using LWP to predict the albedo
    dict2_predi_fla_PI['LWP_lrm'] = deepcopy(predict_dict_PI['value'])
    dict2_predi_nor_PI['LWP_lrm'] = (dict2_predi_fla_PI['LWP_lrm'] - nanmean(dict2_predi_fla_PI['LWP_lrm']) )/ nanstd(dict2_predi_fla_PI['LWP_lrm'])
    # predict_dict_PI_albedo_lL, _, _, coef_array_albedo_lL = rdlrm_1_training(dict2_predi_fla_PI, predictant='albedo', predictor=['LWP_lrm', 'albedo_cs'], r = 1)[0:4]
    # predict_dict_PI_rsut_lL, _, _, coef_array_rsut_lL = rdlrm_1_training(dict2_predi_fla_PI, predictant='rsut', predictor=['LWP_lrm', 'rsutcs'], r = 1)[0:4]


    # Save into the rawdata dict
    C_dict['Coef_dict'] = coef_array
    C_dict['Predict_dict_PI']  = predict_dict_PI
    C_dict['ind_True_PI'] = ind6_PI  # C_dict['ind_Hot_PI'] = ind6_PI
    C_dict['ind_False_PI'] = ind7_PI  # C_dict['ind_Cold_PI'] = ind7_PI
    
    C_dict['Coef_dict_IWP']= coef_array_iwp
    C_dict['Predict_dict_PI_IWP']  = predict_dict_PI_iwp
    # C_dict['ind_Hot_PI_IWP'] = ind6_PI_iwp
    # C_dict['ind_Cold_PI_IWP'] = ind7_PI_iwp
    
    '''
    # Albedo and radiation 
    C_dict['Coef_dict_albedo'] = coef_array_albedo
    C_dict['Predict_dict_PI_albedo'] = predict_dict_PI_albedo
    
    C_dict['Coef_dict_rsut'] = coef_array_rsut
    C_dict['Predict_dict_PI_rsut'] = predict_dict_PI_rsut
    
    C_dict['Coef_dict_albedo_lL'] = coef_array_albedo_lL
    C_dict['Predict_dict_PI_albedo_lL'] = predict_dict_PI_albedo_lL
    
    C_dict['Coef_dict_rsut_lL'] = coef_array_rsut_lL
    C_dict['Predict_dict_PI_rsut_lL'] = predict_dict_PI_rsut_lL
    '''

    # 'YB' is the predicted value of LWP in 'piControl' experiment
    YB = predict_dict_PI['value']
    # print("2lrm predicted mean LWP: ", nanmean(YB), " in 'piControl' ")

    YB_iwp = predict_dict_PI_iwp['value']
    # print("2lrm predicted mean IWP: ", nanmean(YB_iwp), " in 'piControl' ")
    '''
    YB_albedo = predict_dict_PI_albedo['value']
    # print("2lrm predicted mean Albedo (with cloud): ", nanmean(YB_albedo), " in 'piControl' ")
    YB_rsut = predict_dict_PI_rsut['value']

    YB_albedo_lL = predict_dict_PI_albedo_lL['value']
    print("2lrm predicted mean Albedo (with cloud) using reported LWP:", nanmean(YB_albedo), " in 'piControl' ")
    YB_rsut_lL = predict_dict_PI_rsut_lL['value']
    '''

    # Save 'YB', and resampled into the shape of 'LWP_yr_bin':

    C_dict['LWP_predi_bin_PI'] = asarray(YB).reshape(shape_mon_PI)
    C_dict['IWP_predi_bin_PI'] = asarray(YB_iwp).reshape(shape_mon_PI) 
    # C_dict['albedo_predi_bin_PI'] = asarray(YB_albedo).reshape(shape_mon_PI)
    # C_dict['rsut_predi_bin_PI'] = asarray(YB_rsut).reshape(shape_mon_PI)

    # C_dict['albedo_lL_predi_bin_PI'] = asarray(YB_albedo_lL).reshape(shape_mon_PI)
    # C_dict['rsut_lL_predi_bin_PI'] = asarray(YB_rsut_lL).reshape(shape_mon_PI)


    #.. Test performance
    
    stats_dict_PI = Test_performance_1(dict2_predi_fla_PI['LWP'], YB, ind6_PI, ind7_PI)
    stats_dict_PI_iwp = Test_performance_1(dict2_predi_fla_PI['IWP'], YB_iwp, ind6_PI_iwp, ind7_PI_iwp)
    # stats_dict_PI_albedo = Test_performance_1(dict2_predi_fla_PI['albedo'], YB_albedo, ind6_PI, ind7_PI)
    # stats_dict_PI_rsut = Test_performance_1(dict2_predi_fla_PI['rsut'], YB_rsut, ind6_PI, ind7_PI)
    
    # stats_dict_PI_albedo_lL = Test_performance_1(dict2_predi_fla_PI['albedo'], YB_albedo_lL, ind6_PI, ind7_PI)
    # stats_dict_PI_rsut_lL = Test_performance_1(dict2_predi_fla_PI['rsut'], YB_rsut_lL, ind6_PI, ind7_PI)
    # print("Mean of report & predicted albedo_lL in 'piControl' (All): ", nanmean(dict2_predi_fla_PI['albedo']), '& ', nanmean(YB_albedo_lL))
    # print("Mean of report & predicted albedo in 'piControl' for SST>=TR_sst (ind6):", nanmean(dict2_predi_fla_PI['albedo'][ind6_PI]), '& ', nanmean(YB_albedo_lL[ind6_PI]))
    # print("Mean of report & predicted albedo in 'piControl' for SST< TR_sst (ind7):" , nanmean(dict2_predi_fla_PI['albedo'][ind7_PI]), '& ',  nanmean(YB_albedo_lL[ind7_PI]))

    # #########################################################################
    #.. ABR
    
    #.. Predicting module (2lrm)

    predict_dict_abr, ind6_abr, ind7_abr, shape_fla_testing = rdlrm_1_predict(dict2_predi_fla_abr, coef_array, predictant = 'LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 1)
    predict_dict_abr_iwp, ind6_abr_iwp, ind7_abr_iwp, shape_fla_testing_iwp = rdlrm_1_predict(dict2_predi_fla_abr, coef_array_iwp, predictant = 'IWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 1)
    '''
    predict_dict_abr_albedo = rdlrm_1_predict(dict2_predi_fla_abr, coef_array_albedo, predictant = 'albedo', predictor = ['LWP', 'albedo_cs'], r = 1)[0]
    predict_dict_abr_rsut = rdlrm_1_predict(dict2_predi_fla_abr, coef_array_rsut, predictant = 'rsut', predictor= ['LWP', 'rsutcs'], r = 1)[0]
    
    # Added on May 13th, 2022: for second step using LWP to predict the albedo
    dict2_predi_fla_abr['LWP_lrm'] = deepcopy(predict_dict_abr['value'])
    dict2_predi_nor_abr['LWP_lrm'] = (dict2_predi_fla_abr['LWP_lrm'] - nanmean(dict2_predi_fla_abr['LWP_lrm']) )/ nanstd(dict2_predi_fla_abr['LWP_lrm'])
    predict_dict_abr_albedo_lL = rdlrm_1_predict(dict2_predi_fla_abr, coef_array_albedo, predictant='albedo', predictor=['LWP_lrm', 'albedo_cs'], r = 1)[0]
    predict_dict_abr_rsut_lL = rdlrm_1_predict(dict2_predi_fla_abr, coef_array_rsut, predictant='rsut', predictor=['LWP_lrm', 'rsutcs'], r = 1)[0]
    '''

    # Save into the rawdata dict

    C_dict['Predict_dict_abr']  = predict_dict_abr
    C_dict['ind_True_abr'] = ind6_abr  # C_dict['ind_Hot_abr'] = ind6_abr
    C_dict['ind_False_abr'] = ind7_abr  # C_dict['ind_Cold_abr'] = ind7_abr
    
    C_dict['Predict_dict_abr_IWP'] = predict_dict_abr_iwp
    '''
    C_dict['Predict_dict_abr_albedo'] = predict_dict_abr_albedo
    C_dict['Predict_dict_abr_rsut'] = predict_dict_abr_rsut
    
    C_dict['Predict_dict_abr_albedo_lL'] = predict_dict_abr_albedo_lL
    C_dict['Predict_dict_abr_rsut_lL'] = predict_dict_abr_rsut_lL
    '''

    # 'YB_abr' is the predicted value of LWP in 'abrupt-4xCO2' experiment
    YB_abr = predict_dict_abr['value']
    # print("2lrm predicted mean LWP ", nanmean(YB_abr), " in 'abrupt-4xCO2' ")
    
    YB_abr_iwp = predict_dict_abr_iwp['value']
    # print("2lrm predicted mean IWP ", nanmean(YB_abr_iwp), " in 'abrupt-4xCO2' ")
    '''
    YB_abr_albedo = predict_dict_abr_albedo['value']
    # print("2lrm predicted mean Albedo (with cloud)", nanmean(YB_abr_albedo), " in 'abrupt-4xCO2' ")
    YB_abr_rsut = predict_dict_abr_rsut['value']
    
    YB_abr_albedo_lL = predict_dict_abr_albedo_lL['value']
    print("2lrm predicted mean albedo: ", nanmean(YB_abr_albedo), " in 'abrupt-4xCO2' ")

    YB_abr_rsut_lL = predict_dict_abr_rsut_lL['value']
    '''
    # print(" Mean report & predicted albedo for 'abrupt-4xCO2' (All): ", nanmean(dict2_predi_fla_abr['albedo']), '& ', nanmean(YB_abr_albedo_lL))
    # print(" Mean report & predicted albedo_lL for 'abrupt-4xCO2' ('Hot'):", nanmean(dict2_predi_fla_abr['albedo'][ind7_abr]), '& ', nanmean(YB_abr_albedo_lL[ind7_abr]))
    # print(" Mean report & predicted albedo_lL for 'abrupt-4xCO2' ('Cold'):", nanmean(dict2_predi_fla_abr['albedo'][ind6_abr]), '& ', nanmean(YB_abr_albedo_lL[ind6_abr]))



    # Save 'YB_abr', reshapled into the shape of 'LWP_yr_bin_abr':
    C_dict['LWP_predi_bin_abr'] =  asarray(YB_abr).reshape(shape_mon_abr)
    C_dict['IWP_predi_bin_abr'] =  asarray(YB_abr_iwp).reshape(shape_mon_abr)
    # C_dict['albedo_predi_bin_abr'] = asarray(YB_abr_albedo).reshape(shape_mon_abr)
    # C_dict['rsut_predi_bin_abr'] = asarray(YB_abr_rsut).reshape(shape_mon_abr)
    
    # C_dict['albedo_lL_predi_bin_abr'] = asarray(YB_abr_albedo_lL).reshape(shape_mon_abr)
    # C_dict['rsut_lL_predi_bin_abr'] = asarray(YB_abr_rsut_lL).reshape(shape_mon_abr)

    
    # Test performance for abrupt-4xCO2 (testing) data set

    stats_dict_abr = Test_performance_1(dict2_predi_fla_abr['LWP'], YB_abr, ind6_abr, ind7_abr)
    stats_dict_abr_iwp = Test_performance_1(dict2_predi_fla_abr['IWP'], YB_abr_iwp, ind6_abr_iwp, ind7_abr_iwp)
    # stats_dict_abr_albedo = Test_performance_1(dict2_predi_fla_abr['albedo'], YB_abr_albedo, ind6_abr, ind7_abr)
    # stats_dict_abr_rsut = Test_performance_1(dict2_predi_fla_abr['rsut'], YB_abr_rsut, ind6_abr, ind7_abr)

    # stats_dict_abr_albedo_lL = Test_performance_1(dict2_predi_fla_abr['albedo'], YB_abr_albedo_lL, ind6_abr, ind7_abr)
    # stats_dict_abr_rsut_lL = Test_performance_1(dict2_predi_fla_abr['rsut'], YB_abr_rsut_lL, ind6_abr, ind7_abr)

    
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
    '''
    C_dict['stats_dict_PI_albedo'] = stats_dict_PI_albedo
    C_dict['stats_dict_abr_albedo'] = stats_dict_abr_albedo

    C_dict['stats_dict_PI_rsut'] = stats_dict_PI_rsut
    C_dict['stats_dict_abr_rsut'] = stats_dict_abr_rsut

    C_dict['stats_dict_PI_albedo_lL'] = stats_dict_PI_albedo_lL
    C_dict['stats_dict_abr_albedo_lL'] = stats_dict_abr_albedo_lL
    
    C_dict['stats_dict_PI_rsut_lL'] = stats_dict_PI_rsut_lL
    C_dict['stats_dict_abr_rsut_lL'] = stats_dict_abr_rsut_lL
    '''
    
    return C_dict




def fitLRM2(C_dict, TR_sst, TR_sub, s_range, y_range, x_range, lats, lons):
    
    # 'C_dict' is the raw data dict, 'TR_sst' accompany with 'TR_sub' are the pre-defined skin_Temperature/ 500 mb Subsidence thresholds to distinguish 4 rdlrms:

    # 's_range , 'y_range', 'x_range' used to do area mean for repeat gmt ARRAY

    dict0_abr_var = C_dict['dict1_abr_var']
    dict0_PI_var  = C_dict['dict1_PI_var']
    #print(dict0_PI_var['times'])
    
    model = C_dict['model_data']   #.. type in dict
    
    datavar_nas = ['LWP', 'TWP', 'IWP', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'SST', 'p_e', 'LTS', 'SUB']   #..12 varisables except gmt (lon dimension diff)
    
    # load annually-mean bin data
    dict1_yr_bin_PI  = dict0_PI_var['dict1_yr_bin_PI']
    dict1_yr_bin_abr  = dict0_abr_var['dict1_yr_bin_abr']
    #print(dict1_yr_bin_PI['LWP_yr_bin'].shape)
    
    # load monthly bin data
    dict1_mon_bin_PI  = dict0_PI_var['dict1_mon_bin_PI']
    dict1_mon_bin_abr  = dict0_abr_var['dict1_mon_bin_abr']

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
    C_dict['shape_yr_PI_3']  = shape_yr_PI
    C_dict['shape_yr_abr_3']  = shape_yr_abr
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
    predict_dict_PI, ind7_PI, ind8_PI, ind9_PI, ind10_PI, coef_array, shape_fla_training = rdlrm_4_training(dict2_predi_fla_PI, TR_sst, TR_sub, predictant='LWP', r = 2)
    predict_dict_PI_iwp, ind7_PI_iwp, ind8_PI_iwp, ind9_PI_iwp, ind10_PI_iwp, coef_array_iwp, shape_fla_training_iwp = rdlrm_4_training(dict2_predi_fla_PI, TR_sst, TR_sub, predictant='IWP', r = 2)
    '''
    predict_dict_PI_albedo, _, _, _, _, coef_array_albedo = rdlrm_4_training(dict2_predi_fla_PI, TR_sst, TR_sub, predictant='albedo', predictor=['LWP', 'albedo_cs'], r = 2)[0:6]
    predict_dict_PI_rsut, _, _, _, _, coef_array_rsut = rdlrm_4_training(dict2_predi_fla_PI, TR_sst, TR_sub, predictant='rsut', predictor=['LWP', 'rsutcs'], r = 2)[0:6]
    '''
    # Added on May 13th, 2022: for second step using LWP to predict the albedo
    
    dict2_predi_fla_PI['LWP_lrm'] = deepcopy(predict_dict_PI['value'])
    dict2_predi_nor_PI['LWP_lrm'] = (dict2_predi_fla_PI['LWP_lrm'] - nanmean(dict2_predi_fla_PI['LWP_lrm']) )/ nanstd(dict2_predi_fla_PI['LWP_lrm'])
    '''
    predict_dict_PI_albedo_lL, _, _, _, _, coef_array_albedo_lL = rdlrm_4_training(dict2_predi_fla_PI, TR_sst, TR_sub, predictant='albedo', predictor=['LWP_lrm', 'albedo_cs'], r  = 2)[0:6]
    predict_dict_PI_rsut_lL, _, _, _, _, coef_array_rsut_lL = rdlrm_4_training(dict2_predi_fla_PI, TR_sst, TR_sub, predictant='rsut', predictor=['LWP_lrm', 'rsutcs'], r = 2)[0:6]
    '''

    # Save into the rawdata dict
    C_dict['Coef_dict'] = coef_array
    C_dict['Predict_dict_PI']  = predict_dict_PI
    C_dict['ind_Up_PI'] = ind7_PI  # C_dict['ind_Cold_Up_PI'] = ind7_PI
    C_dict['ind_Down_PI'] = ind8_PI  # C_dict['ind_Hot_Up_PI'] = ind8_PI
    # C_dict['ind_Cold_Down_PI'] = ind9_PI
    # C_dict['ind_Hot_Down_PI'] = ind10_PI
    
    C_dict['Coef_dict_IWP']= coef_array_iwp
    C_dict['Predict_dict_PI_IWP']  = predict_dict_PI_iwp

    '''
    # Albedo and radiation 
    C_dict['Coef_dict_albedo'] = coef_array_albedo
    C_dict['Predict_dict_PI_albedo'] = predict_dict_PI_albedo

    C_dict['Coef_dict_rsut'] = coef_array_rsut
    C_dict['Predict_dict_PI_rsut'] = predict_dict_PI_rsut
    
    C_dict['Coef_dict_albedo_lL'] = coef_array_albedo_lL
    C_dict['Predict_dict_PI_albedo_lL'] = predict_dict_PI_albedo_lL

    C_dict['Coef_dict_rsut_lL'] = coef_array_rsut_lL
    C_dict['Predict_dict_PI_rsut_lL'] = predict_dict_PI_rsut_lL
    '''

    # 'YB' is the predicted value of LWP in 'piControl' experiment
    YB = predict_dict_PI['value']
    # print("4lrm predicted mean LWP ", nanmean(YB), " in 'piControl' ")

    YB_iwp = predict_dict_PI_iwp['value']
    # print("4lrm predicted mean IWP ", nanmean(YB_iwp), " in 'piControl' ")
    '''
    YB_albedo = predict_dict_PI_albedo['value']
    # print("4lrm predicted mean Albedo (with cloud): ", nanmean(YB_albedo), " in 'piControl' ")
    YB_rsut = predict_dict_PI_rsut['value']

    YB_albedo_lL = predict_dict_PI_albedo_lL['value']
    print("4lrm predicted mean Albedo (with cloud) using report LWP:", nanmean(YB_albedo), " in 'piControl' ")
    YB_rsut_lL = predict_dict_PI_rsut_lL['value']
    '''
    
    # Save 'YB', resampled into the shape of 'LWP_yr_bin':
    C_dict['LWP_predi_bin_PI'] = asarray(YB).reshape(shape_mon_PI)

    C_dict['IWP_predi_bin_PI'] = asarray(YB_iwp).reshape(shape_mon_PI)
    # C_dict['albedo_predi_bin_PI'] = asarray(YB_albedo).reshape(shape_mon_PI)
    # C_dict['rsut_predi_bin_PI'] =asarray(YB_rsut).reshape(shape_mon_PI)
    
    # C_dict['albedo_lL_predi_bin_PI'] = asarray(YB_albedo_lL).reshape(shape_mon_PI)
    # C_dict['rsut_lL_predi_bin_PI'] = asarray(YB_rsut_lL).reshape(shape_mon_PI)


    #.. Test performance
    
    stats_dict_PI = Test_performance_2(dict2_predi_fla_PI['LWP'], YB, ind7_PI, ind8_PI)   #  Test_performance_4(dict2_predi_fla_PI['LWP'], YB, ind7_PI, ind8_PI, ind9_PI, ind10_PI)
    stats_dict_PI_iwp = Test_performance_2(dict2_predi_fla_PI['IWP'], YB_iwp, ind7_PI_iwp, ind8_PI_iwp)   # Test_performance_4(dict2_predi_fla_PI['IWP'], YB_iwp, ind7_PI_iwp, ind8_PI_iwp, ind9_PI_iwp, ind10_PI_iwp)
    
    # stats_dict_PI_albedo = Test_performance_2(dict2_predi_fla_PI['albedo'], YB_albedo, ind7_PI, ind8_PI)   # Test_performance_4(dict2_predi_fla_PI['albedo'], YB_albedo, ind7_PI, ind8_PI, ind9_PI, ind10_PI)
    # stats_dict_PI_rsut = Test_performance_2(dict2_predi_fla_PI['rsut'], YB_rsut, ind7_PI, ind8_PI)   # Test_performance_4(dict2_predi_fla_PI['rsut'], YB_rsut, ind7_PI, ind8_PI, ind9_PI, ind10_PI)
    
    # stats_dict_PI_albedo_lL = Test_performance_2(dict2_predi_fla_PI['albedo'], YB_albedo_lL, ind7_PI, ind8_PI)   # Test_performance_4(dict2_predi_fla_PI['albedo'], YB_albedo_lL, ind7_PI, ind8_PI, ind9_PI, ind10_PI)
    # stats_dict_PI_rsut_lL = Test_performance_2(dict2_predi_fla_PI['rsut'], YB_rsut_lL, ind7_PI, ind8_PI)   # Test_performance_4(dict2_predi_fla_PI['rsut'], YB_rsut_lL, ind7_PI, ind8_PI, ind9_PI, ind10_PI)

    # print(" Mean of report & predicted albedo_lL for 'piControl' (all) exp: ", nanmean(dict2_predi_fla_PI['albedo']), '& ', nanmean(YB_albedo_lL))
    # print(" Mean of report & predicted albedo_lL for 'piControl' of Hot, Up regime: " , nanmean(dict2_predi_fla_PI['albedo'][ind8_PI]), '& ', nanmean(YB_albedo_lL[ind8_PI]))


    # ####)################################
    #.. ABR
    #.. Predicting module (4lrm)

    predict_dict_abr, ind7_abr, ind8_abr, ind9_abr, ind10_abr, shape_fla_testing = rdlrm_4_predict(dict2_predi_fla_abr, coef_array, TR_sst, TR_sub, predictant = 'LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 2)
    predict_dict_abr_iwp, ind7_abr_iwp, ind8_abr_iwp, ind9_abr_iwp, ind10_abr_iwp, shape_fla_testing_iwp = rdlrm_4_predict(dict2_predi_fla_abr, coef_array_iwp, TR_sst, TR_sub, predictant = 'IWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 2)
    '''
    predict_dict_abr_albedo = rdlrm_4_predict(dict2_predi_fla_abr, coef_array_albedo, TR_sst, TR_sub, predictant = 'albedo', predictor = ['LWP', 'albedo_cs'], r = 2)[0]
    predict_dict_abr_rsut = rdlrm_4_predict(dict2_predi_fla_abr, coef_array_rsut, TR_sst, TR_sub, predictant = 'rsut', predictor= ['LWP', 'rsutcs'], r = 2)[0]
    '''
    # Added on May 14th, 2022: for second step using LWP to predict the albedo
    dict2_predi_fla_abr['LWP_lrm'] = deepcopy(predict_dict_abr['value'])
    dict2_predi_nor_abr['LWP_lrm'] = (dict2_predi_fla_abr['LWP_lrm'] - nanmean(dict2_predi_fla_abr['LWP_lrm']) )/ nanstd(dict2_predi_fla_abr['LWP_lrm'])
    # predict_dict_abr_albedo_lL = rdlrm_4_predict(dict2_predi_fla_abr, coef_array_albedo, TR_sst, TR_sub, predictant='albedo', predictor=['LWP_lrm', 'albedo_cs'], r = 2)[0]
    # predict_dict_abr_rsut_lL = rdlrm_4_predict(dict2_predi_fla_abr, coef_array_rsut, TR_sst, TR_sub, predictant='rsut', predictor=['LWP_lrm', 'rsutcs'], r = 2)[0]

    # Save into the rawdata dict
    C_dict['Predict_dict_abr']  = predict_dict_abr
    C_dict['ind_Up_abr'] = ind7_abr  # C_dict['ind_Cold_Up_abr'] = ind7_abr
    C_dict['ind_Down_abr'] = ind8_abr  # C_dict['ind_Hot_Up_abr'] = ind8_abr
    # C_dict['ind_Cold_Down_abr'] = ind9_abr
    # C_dict['ind_Hot_Down_abr'] = ind10_abr

    C_dict['Predict_dict_abr_IWP']  = predict_dict_abr_iwp
    
    '''
    C_dict['Predict_dict_abr_albedo'] = predict_dict_abr_albedo
    C_dict['Predict_dict_abr_rsut'] = predict_dict_abr_rsut
    C_dict['Predict_dict_abr_albedo_lL'] = predict_dict_abr_albedo_lL
    C_dict['Predict_dict_abr_rsut_lL'] = predict_dict_abr_rsut_lL
    '''

    # 'YB_abr' is the predicted value of LWP in 'abrupt-4xCO2' experiment
    YB_abr = predict_dict_abr['value']
    # print("4lrm predicted mean LWP ", nanmean(YB_abr), " in 'abrupt-4xCO2'")

    YB_abr_iwp = predict_dict_abr_iwp['value']
    # print("4lrm predicted mean IWP ", nanmean(YB_abr_iwp), " in 'abrupt-4xCO2'")
    '''
    YB_abr_albedo = predict_dict_abr_albedo['value']
    # print("4lrm predicted mean Albedo (with cloud) ", nanmean(YB_abr_albedo)," in 'abrupt-4xCO2' ")
    YB_abr_rsut = predict_dict_abr_rsut['value']

    YB_abr_albedo_lL = predict_dict_abr_albedo_lL['value']
    print("4lrm predicted mean Albedo (with cloud) using report LWP", nanmean(YB_abr_albedo), " in 'abrupt-4xCO2' ")

    YB_abr_rsut_lL = predict_dict_abr_rsut_lL['value']
    '''
    # print(" 4lrm: predicted LWP of 'abrupt-4xCO2':", YB_abr)
    # print(" 4lrm: report LWP of 'abrupt-4xCO2':", dict2_predi_fla_abr['LWP'])
    # print(" 4lrm: predicted albedo of 'abrupt-4xCO2':", YB_abr_albedo)
    # print(" 4lrm: report albedo of  'abrupt-4xCO2':", dict2_predi_fla_abr['albedo'])

    
    # Save 'YB_abr', reshapled into the shape of 'LWP_yr_bin_abr':
    C_dict['LWP_predi_bin_abr'] =  asarray(YB_abr).reshape(shape_mon_abr)
    C_dict['IWP_predi_bin_abr'] =  asarray(YB_abr_iwp).reshape(shape_mon_abr)
    # C_dict['albedo_predi_bin_abr'] = asarray(YB_abr_albedo).reshape(shape_mon_abr)
    # C_dict['rsut_predi_bin_abr'] = asarray(YB_abr_rsut).reshape(shape_mon_abr)

    # C_dict['albedo_lL_predi_bin_abr'] = asarray(YB_abr_albedo_lL).reshape(shape_mon_abr)
    # C_dict['rsut_lL_predi_bin_abr'] = asarray(YB_abr_rsut_lL).reshape(shape_mon_abr)

    
    # Test performance for abrupt-4xCO2 (testing) data set
    
    stats_dict_abr = Test_performance_2(dict2_predi_fla_abr['LWP'], YB_abr, ind7_abr, ind8_abr)  # Test_performance_4(dict2_predi_fla_abr['LWP'], YB_abr, ind7_abr, ind8_abr, ind9_abr, ind10_abr)
    stats_dict_abr_iwp = Test_performance_2(dict2_predi_fla_abr['IWP'], YB_abr_iwp, ind7_abr_iwp, ind8_abr_iwp)  # Test_performance_4(dict2_predi_fla_abr['IWP'], YB_abr_iwp, ind7_abr_iwp, ind8_abr_iwp, ind9_abr_iwp, ind10_abr_iwp)
    
    # stats_dict_abr_albedo = Test_performance_2(dict2_predi_fla_abr['albedo'], YB_abr_albedo, ind7_abr, ind8_abr)  # Test_performance_4(dict2_predi_fla_abr['albedo'], YB_abr_albedo, ind7_abr, ind8_abr, ind9_abr, ind10_abr)
    # stats_dict_abr_rsut = Test_performance_2(dict2_predi_fla_abr['rsut'], YB_abr_rsut, ind7_abr, ind8_abr)  # Test_performance_4(dict2_predi_fla_abr['rsut'], YB_abr_rsut, ind7_abr, ind8_abr, ind9_abr, ind10_abr)

    # stats_dict_abr_albedo_lL = Test_performance_2(dict2_predi_fla_abr['albedo'], YB_abr_albedo_lL, ind7_abr, ind8_abr)  # Test_performance_4(dict2_predi_fla_abr['albedo'], YB_abr_albedo_lL, ind7_abr, ind8_abr, ind9_abr, ind10_abr)
    # stats_dict_abr_rsut_lL = Test_performance_2(dict2_predi_fla_abr['rsut'], YB_abr_rsut_lL, ind7_abr, ind8_abr)  # Test_performance_4(dict2_predi_fla_abr['rsut'], YB_abr_rsut_lL, ind7_abr, ind8_abr, ind9_abr, ind10_abr)

    # print(" Mean of report & predicted albedo_lL for 'abrupt-4xCO2' (all): ", nanmean(dict2_predi_fla_abr['albedo']), '& ', nanmean(YB_abr_albedo_lL))
    # print(" Mean of report & predicted albedo_lL for 'abrupt-4xCO2' of Cold, Down regime: ", nanmean(dict2_predi_fla_abr['albedo'][ind9_abr]), '& ', nanmean(YB_abr_albedo_lL[ind9_abr]))
    # print(" Mean of report & predicted albedo_lL for 'abrupt-4xCO2' of Cold, Up regime: ", nanmean(dict2_predi_fla_abr['albedo'][ind7_abr]), '& ', nanmean(YB_abr_albedo_lL[ind7_abr]))
    # print(" Mean of report & predicted albedo_lL for 'abrupt-4xCO2' of Hot, Up regime: ", nanmean(dict2_predi_fla_abr['albedo'][ind8_abr]), '& ', nanmean(YB_abr_albedo_lL[ind8_abr]))
    # print(" Mean of report & predicted albedo_lL for 'abrupt-4xCO2' of Hot, Down regime: " , nanmean(dict2_predi_fla_abr['albedo'][ind10_abr]), '& ', nanmean(YB_abr_albedo_lL[ind10_abr]))

    
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
    '''
    C_dict['stats_dict_PI_albedo'] = stats_dict_PI_albedo
    C_dict['stats_dict_abr_albedo'] = stats_dict_abr_albedo

    C_dict['stats_dict_PI_rsut'] = stats_dict_PI_rsut
    C_dict['stats_dict_abr_rsut'] = stats_dict_abr_rsut

    C_dict['stats_dict_PI_albedo_lL'] = stats_dict_PI_albedo_lL
    C_dict['stats_dict_abr_albedo_lL'] = stats_dict_abr_albedo_lL

    C_dict['stats_dict_PI_rsut_lL'] = stats_dict_PI_rsut_lL
    C_dict['stats_dict_abr_rsut_lL'] = stats_dict_abr_rsut_lL
    '''
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


    datavar_nas = ['LWP', 'TWP', 'IWP', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'SST', 'p_e', 'LTS', 'SUB']   #..12 varisables except gmt (lon dimension diff)

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
        areamean_dict_PI[datavar_nas[e]+ '_yr_bin'] = dict1_mon_bin_PI[datavar_nas[e]+ '_mon_bin']
        # get_annually_metric(dict1_mon_bin_PI[datavar_nas[e]+ '_mon_bin'], shape_mon_PI_3[0],  shape_mon_PI_3[1], shape_mon_PI_3[2]) 
        areamean_dict_abr[datavar_nas[e]+ '_yr_bin'] =  dict1_mon_bin_abr[datavar_nas[e]+'_mon_bin']
        # get_annually_metric(dict1_mon_bin_abr[datavar_nas[e]+ '_mon_bin'], shape_mon_abr_3[0],  shape_mon_abr_3[1], shape_mon_abr_3[2])

        #  "yr_bin"  area_meaned to 'shape_yr_':
        areamean_dict_PI[datavar_nas[e]+ '_area_yr'] = area_mean(areamean_dict_PI[datavar_nas[e]+ '_yr_bin'], y_range, x_range)
        areamean_dict_abr[datavar_nas[e]+ '_area_yr'] = area_mean(areamean_dict_abr[datavar_nas[e]+ '_yr_bin'], y_range, x_range)

    areamean_dict_PI['gmt_area_yr'] = area_mean(dict1_yr_bin_PI['gmt_yr_bin'], s_range, x_range)
    areamean_dict_abr['gmt_area_yr'] = area_mean(dict1_yr_bin_abr['gmt_yr_bin'], s_range, x_range)
    
    # Calc annually mean predicted LWP, IWP, and SW radiation metrics
    
    ########### Annually predicted data:
    # areamean_dict_predi['LWP_area_yr_pi']  =   area_mean(rawdata_dict['LWP_predi_bin_PI'], y_range, x_range)
    # areamean_dict_predi['LWP_area_yr_abr']  =   area_mean(rawdata_dict['LWP_predi_bin_abr'], y_range, x_range)
    ############ end yr

    ########### Monthly predicted data:
    
    areamean_dict_predi =  {}
    datapredi_nas = ['LWP', 'IWP']  # 'albedo', 'rsut', 'albedo_lL', 'rsut_lL'
    datarepo_nas = ['LWP', 'IWP']  # 'albedo', 'albedo_cs', 'rsut', 'rsutcs'
    
    for f in range(len(datapredi_nas)):
        areamean_dict_predi[datapredi_nas[f]+'_predi_yr_bin_pi'] = rawdata_dict[datapredi_nas[f]+'_predi_bin_PI']
        # get_annually_metric(rawdata_dict[datapredi_nas[f]+'_predi_bin_PI'], shape_mon_PI_3[0], shape_mon_PI_3[1], shape_mon_PI_3[2] )
        
        areamean_dict_predi[datapredi_nas[f]+'_predi_yr_bin_abr'] = rawdata_dict[datapredi_nas[f]+'_predi_bin_abr']
        # get_annually_metric(rawdata_dict[datapredi_nas[f]+'_predi_bin_abr'], shape_mon_abr_3[0], shape_mon_abr_3[1], shape_mon_abr_3[2] )


    ###  Calc area_mean of predicted LWP, IWP and SW radiation metrics
    
    for g in range(len(datapredi_nas)):

        areamean_dict_predi[datapredi_nas[g]+'_area_yr_pi'] = area_mean(areamean_dict_predi[datapredi_nas[g]+'_predi_yr_bin_pi'],  y_range, x_range)
        areamean_dict_predi[datapredi_nas[g]+'_area_yr_abr'] = area_mean(areamean_dict_predi[datapredi_nas[g]+'_predi_yr_bin_abr'],  y_range, x_range)

    
    # print("Annually area_mean predicted  albedo (with cloud) in 'piControl' run: ",areamean_dict_predi['albedo_lL_area_yr_pi'], r'$w m^{-2}$')  # r'$ kg m^{-2}$'
    # print("Annually area_mean predicted  albedo (with cloud) in 'abrupt-4xCO2' run: ",areamean_dict_predi['albedo_lL_area_yr_abr'], r'$ w m^{-2}$')
    
    ############# end mon


    # Store the annually report & predicted metrics
    
    rawdata_dict['areamean_dict_predi'] = areamean_dict_predi
    rawdata_dict['areamean_dict_abr'] = areamean_dict_abr
    rawdata_dict['areamean_dict_PI'] = areamean_dict_PI


    # Generate continous annually-mean array are convenient for plotting LWP changes:
    #..Years from 'piControl' to 'abrupt-4xCO2' experiment, which are choosed years
    Yrs = arange(shape_yr_pi+shape_yr_abr)

    # global-mean surface air temperature, from 'piControl' to 'abrupt-4xCO2' experiment:
    
    GMT = full((shape_yr_pi + shape_yr_abr),  0.0)
    GMT[0:shape_yr_pi] = areamean_dict_PI['gmt_area_yr']
    GMT[shape_yr_pi:] = areamean_dict_abr['gmt_area_yr']

    predict_metrics_annually = {}
    report_metrics_annually = {}
    
    # predicted values, from 'piControl' to 'abrupt-4xCO2' experiment
    
    for h in range(len(datapredi_nas)):
        predict_metrics_annually[datapredi_nas[h]] = full((shape_yr_pi + areamean_dict_predi[datapredi_nas[h]+'_area_yr_abr'].shape[0]), 0.0)
        predict_metrics_annually[datapredi_nas[h]][0:shape_yr_pi] = areamean_dict_predi[datapredi_nas[h]+'_area_yr_pi'][0:shape_yr_pi]
        predict_metrics_annually[datapredi_nas[h]][shape_yr_pi:(shape_yr_pi+areamean_dict_predi[datapredi_nas[h]+'_area_yr_abr'].shape[0])] = areamean_dict_predi[datapredi_nas[h]+'_area_yr_abr']
        
    # report values, from 'piControl' to 'abrupt-4xCO2' experiment

    for i in range(len(datarepo_nas)):
        report_metrics_annually[datarepo_nas[i]] = full((shape_yr_pi + areamean_dict_abr[datarepo_nas[i]+'_area_yr'].shape[0]), 0.0)  
        report_metrics_annually[datarepo_nas[i]][0:shape_yr_pi] = areamean_dict_PI[datarepo_nas[i]+'_area_yr'][0:shape_yr_pi]
        report_metrics_annually[datarepo_nas[i]][shape_yr_pi:(shape_yr_pi+areamean_dict_abr[datarepo_nas[i]+'_area_yr'].shape[0])] = areamean_dict_abr[datarepo_nas[i]+'_area_yr']
    
    print("report LWP: ", report_metrics_annually['LWP'])
    print("predicted LWP: ", predict_metrics_annually['LWP'])

    
    # put them into the rawdata_dict:
    rawdata_dict['Yrs'] = Yrs
    rawdata_dict['GMT'] = GMT

    rawdata_dict['predicted_metrics'] = predict_metrics_annually
    rawdata_dict['report_metrics'] = report_metrics_annually
    
    return rawdata_dict

