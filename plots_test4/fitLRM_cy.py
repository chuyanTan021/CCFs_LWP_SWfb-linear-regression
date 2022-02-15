#..flatten all the data array, from that built a SST_thershold based Linear Regession Model(LRM);
# assess its behavior (RMSE/ R^2) and do the regression for PI and abr4x experiments, then build an array for whole-period-LWP


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

from get_annual_so import *




def fitLRM_cy(C_dict, TR_sst, s_range, y_range, x_range):
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
    shape_yr_PI = dict1_yr_bin_PI['LWP_yr_bin'].shape
    shape_yr_abr = dict1_yr_bin_abr['LWP_yr_bin'].shape

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

    # save into rawdata_dict
    C_dict['dict2_predi_fla_PI'] =  dict2_predi_fla_PI
    C_dict['dict2_predi_fla_abr'] = dict2_predi_fla_abr
    C_dict['dict2_predi_nor_PI'] =  dict2_predi_nor_PI
    C_dict['dict2_predi_nor_abr']  = dict2_predi_nor_abr

    print('shape1: ', dict2_predi_fla_PI['LWP'].shape)     # shape1
    shape_fla_PI   =  dict2_predi_fla_PI['LWP'].shape
    #print(min(dict2_predi_fla_PI['LTS']),  max(dict2_predi_fla_PI['LTS']) )

    # PI
    #..Subtract 'nan' in data, shape1 -> shape2(without 'nan' number) points and shape5('nan' number)

    ind1 = isnan(dict2_predi_fla_PI['LTS'])==False 

    ind_true = nonzero(ind1==True)
    ind_false = nonzero(ind1==False)   
    #..Sign the the indexing into YB, or YB value will have a big changes
    print('shape2: ', array(ind_true).shape)        # shape2
    #print(argwhere(isnan(dict2_predi_fla_PI['LTS'][ind_true])==True))


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

    print('YB(raw PI LWP array) ', YB)
    #print(YB.shape)



    #.. Multiple linear regreesion of Liquid Water Path to CCFs :

    #..Remove abnormal and missing_values, train model with TR sst>= TR_sst, unit in K
    X  = np.array( [dict2_predi_fla_PI['SST'][ind6], dict2_predi_fla_PI['p_e'][ind6], dict2_predi_fla_PI['LTS'][ind6], dict2_predi_fla_PI['SUB'][ind6]] )

    regr1 = linear_model.LinearRegression()
    result1 = regr1.fit(X.T, dict2_predi_fla_PI['LWP'][ind6] )   #..regression for LWP WITH LTS and skin-T >= TR_sst

    #print('result1 coef: ', result1.coef_)
    #print('result1 intercept: ', result1.intercept_)

    '''
    #..for test
    a = load('sensitivity_4ccfs_ipsl270.npy')
    b = load('intercept1_ipsl270.npy')
    print('270K coef for IPSL: ', a)
    print('270K intercept for IPSL: ', b)
    '''

    #..Remove abnormal and missing values , train model with TR sst < Tr_SST.K
    XX = np.array([dict2_predi_fla_PI['SST'][ind7], dict2_predi_fla_PI['p_e'][ind7], dict2_predi_fla_PI['LTS'][ind7], dict2_predi_fla_PI['SUB'][ind7]])

    regr2=linear_model.LinearRegression()
    result2 = regr2.fit(XX.T, dict2_predi_fla_PI['LWP'][ind7])   #..regression for LWP WITH LTS and skin-T < TR_sst

    #print('result2 coef: ', result2.coef_)
    #print('result2 intercept: ', result2.intercept_)


    #..Save them into rawdata_dict
    aeffi  = result1.coef_
    aint   = result1.intercept_

    beffi  = result2.coef_
    bint   = result2.intercept_


    C_dict['LRM_le'] = (aeffi, aint)
    C_dict['LRM_st'] = (beffi, bint)

    # Regression for pi VALUES:
    sstlelwp_predi = dot(aeffi.reshape(1, -1),  X)  + aint   #..larger or equal than Tr_SST
    sstltlwp_predi = dot(beffi.reshape(1, -1), XX)  + bint   #..less than Tr_SST

    # emsemble into 'YB' predicted data array for Pi:
    YB[ind6] = sstlelwp_predi
    YB[ind7] = sstltlwp_predi

    # 'YB' resample into the shape of 'LWP_yr_bin':
    C_dict['LWP_predi_bin_PI']   =  array(YB).reshape(shape_yr_PI)
    print('  predicted LWP array for PI, shape in ',  C_dict['LWP_predi_bin_PI'].shape)


    #.. Test performance
    MSE_shape6 =  mean_squared_error(dict2_predi_fla_PI['LWP'][ind6].reshape(-1,1), sstlelwp_predi.reshape(-1,1))
    print('RMSE_shape6(PI): ', sqrt(MSE_shape6))

    R_2_shape7  = r2_score(dict2_predi_fla_PI['LWP'][ind7].reshape(-1, 1), sstltlwp_predi.reshape(-1, 1))
    print('R_2_shape7: ', R_2_shape7)

    MSE_shape1 =  mean_squared_error(dict2_predi_fla_PI['LWP'].reshape(-1,1), YB.reshape(-1,1))
    print('RMSE_shape1: ', sqrt(MSE_shape1))

    R_2_shape1  = r2_score(dict2_predi_fla_PI['LWP'].reshape(-1, 1), YB.reshape(-1, 1))
    print('R_2_shape1: ', R_2_shape1)

    print('examine regres-mean Lwp for pi-C shape6:', mean(dict2_predi_fla_PI['LWP'][ind6]), mean(sstlelwp_predi))
    print('examine regres-mean Lwp for pi-C shape7:', mean(dict2_predi_fla_PI['LWP'][ind7]), mean(sstltlwp_predi))


    # ABR
    shape_fla_abr   =  dict2_predi_fla_abr['LWP'].shape
    print(dict2_predi_fla_abr['p_e'].shape)  #..compare with the following line

    #..Subtract 'nan' in data, shape1_abr -> shape2_abr points
    ind1_abr =  isnan(dict2_predi_fla_abr['LTS'])==False
    print('shape1_abr :', ind1_abr.shape)



    ind_true_abr =  nonzero(ind1_abr ==True )   #..Sign the the indexing of 'Non-NaN' in LTS_yr_bin
    print('shape2_abr :', array(ind_true_abr).shape, dict2_predi_fla_abr['LTS'][ind_true_abr].shape)

    ind_false_abr =  nonzero(ind1_abr==False)   #..Sign the the indexing of 'NaN'
    #dict1_yr_bin_abr[ind_false_abr] = 0.0


    #..Split the abrupt4x data points with TR_sst 
    #..

    ind_sst_le_abr  = nonzero(dict2_predi_fla_abr['SST'] >= TR_sst)
    ind6_abr  = intersect1d(ind_true_abr, ind_sst_le_abr)
    print('shape6_abr: ', ind6_abr.shape)   #..shape6_


    ind_sst_st_abr  = nonzero(dict2_predi_fla_abr['SST'] < TR_sst)
    ind7_abr =  intersect1d(ind_true_abr, ind_sst_st_abr)
    print('shape7_abr: ', ind7_abr.shape)  #..shape7_abr points


    #..designate LWP single-array's value, abr
    YB_abr   =  full((shape_fla_abr),  0.0)   # predicted LWP value array for future uses

    YB_abr[ind_false_abr] = dict2_predi_fla_abr['LWP'][ind_false_abr]   #..LWP single-column array with no LTS points as original values, with has LTS value points as 0.0. 
    print('YB_abr(raw abrupt4x LWP array: ', YB_abr)


    # Regression for abr LWP VALUES:
    # DIFFERENT LRM: LRM1: model with points that skin_T largerorequal to TR_sst

    X_abr   =  np.array([dict2_predi_fla_abr['SST'][ind6_abr], dict2_predi_fla_abr['p_e'][ind6_abr], dict2_predi_fla_abr['LTS'][ind6_abr], dict2_predi_fla_abr['SUB'][ind6_abr]])

    # LRM2: model with points that skin_T smaller than TR_sst

    XX_abr  =  np.array([dict2_predi_fla_abr['SST'][ind7_abr], dict2_predi_fla_abr['p_e'][ind7_abr], dict2_predi_fla_abr['LTS'][ind7_abr], dict2_predi_fla_abr['SUB'][ind7_abr]])


    sstlelwp_predi_abr = dot(aeffi.reshape(1, -1),  X_abr)  +  aint   #.. skin_T  larger or equal than Tr_SST
    sstltlwp_predi_abr = dot(beffi.reshape(1, -1),  XX_abr)  + bint   #.. skin_T  less than Tr_SST


    # emsemble into 'YB_abr' predicted data array for Abrupt4xCO2:
    YB_abr[ind6_abr]  =   sstlelwp_predi_abr
    YB_abr[ind7_abr]  =   sstltlwp_predi_abr
    # 'YB' reshaple into the shape of 'LWP_yr_bin_abr':
    C_dict['LWP_predi_bin_abr']   =  array(YB_abr).reshape(shape_yr_abr)

    print(' predicted LWP array for abrupt4xCO2, shape in ',  C_dict['LWP_predi_bin_abr'].shape)  


    # Test performance..abr
    MSE_shape1_abr = mean_squared_error(YB_abr.reshape(-1,1),  dict2_predi_fla_abr['LWP'].reshape(-1, 1))
    R_2_shape1_abr = r2_score(dict2_predi_fla_abr['LWP'].reshape(-1,1), YB_abr.reshape(-1, 1))
    print('RMSE_shape1_abr: ', sqrt(MSE_shape1_abr))
    print('R_2_shape1_abr: ', R_2_shape1_abr)

    # calc D(CCFs) to DGMT and save into 'Dx/DtG' ARRAY
    regr3 = linear_model.LinearRegression()
    re_LWP= regr3.fit(dict2_predi_fla_abr['gmt'][ind_true_abr].reshape(-1,1), dict2_predi_fla_abr['LWP'][ind_true_abr])
    print('D(LWP) /D(gmt) (with LTS POINTS) : ', re_LWP.coef_)
    print('b of D(LWP) /D(gmt) : ', re_LWP.intercept_)

    regr4 = linear_model.LinearRegression()
    regr5 = linear_model.LinearRegression()
    regr6 = linear_model.LinearRegression()

    regr7 = linear_model.LinearRegression()

    re_SST = regr4.fit(dict2_predi_fla_abr['gmt'][ind_true_abr].reshape(-1,1), dict2_predi_fla_abr['SST'][ind_true_abr])
    re_p_e = regr5.fit(dict2_predi_fla_abr['gmt'][ind_true_abr].reshape(-1,1), dict2_predi_fla_abr['p_e'][ind_true_abr])
    re_LTS = regr6.fit(dict2_predi_fla_abr['gmt'][ind_true_abr].reshape(-1,1), dict2_predi_fla_abr['LTS'][ind_true_abr])

    re_SUB = regr7.fit(dict2_predi_fla_abr['gmt'][ind_true_abr].reshape(-1,1), dict2_predi_fla_abr['SUB'][ind_true_abr])
    print('dCCF /dGMT (with LTS POINTS): ', re_SST.coef_, re_p_e.coef_, re_LTS.coef_, re_SUB.coef_ )

    #..save into rawdata_dict
    Dx_DtG =[re_LWP.coef_, re_SST.coef_,  re_p_e.coef_,  re_LTS.coef_, re_SUB.coef_]
    C_dict['dX_dTg']  =   Dx_DtG

    #..save test performance metrics into rawdata_dict
    EXAMINE_metrics =  {'RMSE_shape1_pi': sqrt(MSE_shape1), 'R_2_shape1_pi': R_2_shape1, 'RMSE_shape6_pi': sqrt(MSE_shape6), 'R_2_shape7': R_2_shape7, \
                       'RMSE_shape1_abr': sqrt(MSE_shape1_abr), 'R_2_shape1_abr': R_2_shape1_abr}

    C_dict['EXAMINE_metrics'] = EXAMINE_metrics

    return C_dict

