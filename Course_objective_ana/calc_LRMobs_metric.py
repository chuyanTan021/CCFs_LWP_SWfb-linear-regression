## This module is to get the obs data we need from read func: 'get_OBSLRMdata', and calculate for CCFs and the required Cloud properties; 
# Crop regions, Transform the data to be annually mean & binned array form;
# Create the linear regression 2 & 4 regimes models from present day values of sensitivity of cloud properties to the CCFs, save the data

import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


import pandas as pd
import glob
from copy import deepcopy
from scipy.stats import *
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
# self_defined modules
from area_mean import *
from binned_cyFunctions5 import *
from read_hs_file import read_var_mod
from read_var_obs import *
from get_LWPCMIP5data import *
from get_LWPCMIP6data import *
from get_OBSLRMdata import *
from fitLRM_cy1 import *
from fitLRM_cy2 import *
from fitLRM_cy4 import *
from fitLRMobs import *
from useful_func_cy import *
from calc_Radiation_LRM_1 import *
from calc_Radiation_LRM_2 import *


def calc_LRMobs_metrics(valid_range1=[2002, 7, 15], valid_range2=[2016, 12, 31], valid_range3=[1994, 1, 15], valid_range4=[2001, 12, 31], THRESHOLD_sst = 0.0, THRESHOLD_sub = 0.0):
    # -----------------
    # 'valid_range1' and 'valid_range2' give the time stamps of starting and ending times of data for training,
    # 'valid_range3' and 'valid_range4' give the time stamps of starting and ending times of data for predicting.
    # 'THRESHOLD_sst' is the cut-off of 'Sea surface temperature' for partitioning the 'Hot'/'Cold' LRM regimes;
    # 'THRESHOLD_sub' is the cut-off of '500 mb Vertical Velocity (Pressure)' for partitioning 'Up'/'Down' regimes.
    # ..
    # ------------------
    # Southern Ocean 5 * 5 degree bin box
    # Using to do area_mean
    s_range = arange(-90., 90., 5.) + 2.5  #..global-region latitude edge: (36)
    x_range = arange(-180., 180., 5.)  #..logitude sequences edge: number: 72
    y_range = arange(-85, -40., 5.) + 2.5  #..southern-ocaen latitude edge: 9
    
    
    # Function #1 loopping through variables space to find the cut-offs of LRM (Multi-Linear Regression Model).
    dict_training, lats_Array, lons_Array, times_Array_training = Pre_processing(s_range, x_range, y_range, valid_range1 = valid_range1, valid_range2 = valid_range2)
    
    dict_predict, lats_Array, lons_Array, times_Array_predict = Pre_processing(s_range, x_range, y_range, valid_range1 = valid_range3, valid_range2 = valid_range4)
    
    # Loop_OBS_LRM(dict_training, dict_predict, s_range, x_range, y_range)
    
    
    # Function #2 training LRM with using no cut-off, then use it to predict another historical period.
    predict_result_1r = fitLRMobs_1(dict_training, dict_predict, s_range, y_range, x_range, lats_Array, lons_Array)
    
    # Function #3,4,5 training LRM with cut_off (TR_sst &/or TR_sub)
    
    WD = '/glade/scratch/chuyan/obs_output/'
    folder = glob.glob(WD + 'OBS__' + 'STAT_pi+abr_'+'22x_31y_Sep11th'+ '.npz')
    print('cut-off folder', folder)

    output_ARRAY = np.load(folder[0], allow_pickle=True)  # str(TR_sst)
    
    TR_sst1 = output_ARRAY['TR_minabias_SST']
    TR_sub1 = output_ARRAY['TR_minabias_SUB']
    TR_sst2 = output_ARRAY['TR_maxR2_SST']
    TR_sub2 = output_ARRAY['TR_maxR2_SUB']

    print("TR_min_abs(bias): " , TR_sst1, '  K ', TR_sub1 , ' Pa/s ')
    print("TR_large_pi_R_2: ", TR_sst2, '  K ', TR_sub2 , ' Pa/s ')
    
    
    predict_result_2r_updown = fitLRMobs_2_updown(dict_training, dict_predict, TR_sst2, TR_sub2, s_range, y_range, x_range, lats_Array, lons_Array)
    predict_result_2r_hotcold = fitLRMobs_2_hotcold(dict_training, dict_predict, TR_sst2, TR_sub2, s_range, y_range, x_range, lats_Array, lons_Array)
    
    
    
    predict_result_4r = fitLRMobs_4(dict_training, dict_predict, TR_sst2, TR_sub2, s_range, y_range, x_range, lats_Array, lons_Array)
    
    return None



def Pre_processing(s_range, x_range, y_range, valid_range1=[2002, 7, 15], valid_range2=[2016, 12, 31]):
    # get the variables for training:
    inputVar_obs = get_OBSLRM(valid_range1=valid_range1, valid_range2=valid_range2)
    # ------------------------
    # radiation code

    # ------------------------

    # Data processing
    # --Liquid water path, Unit in kg m^-2
    LWP = inputVar_obs['lwp'] / 1000.
    # 1-Sigma Liquid water path statistic error, Unit in kg m^-2
    LWP_error = inputVar_obs['lwp_error'] / 1000.
    # the MaskedArray of 'MAC-LWP' dataset
    Maskarray_mac = inputVar_obs['maskarray_mac']
    # ---

    # GMT: Global mean surface air Temperature (2-meter), Unit in K
    gmt = inputVar_obs['tas'] * 1.
    # SST: Sea Surface Temperature or skin- Temperature, Unit in K
    SST = inputVar_obs['sfc_T'] * 1.
    # Precip: Precipitation, Unit in mm day^-1 (convert from kg m^-2 s^-1)
    Precip = inputVar_obs['P'] * (24. * 60 * 60)
    # Eva: Evaporation, Unit in mm day^-1 (here use the latent heat flux from the sfc, unit convert from W m^-2 --> kg m^-2 s^-1 --> mm day^-1)
    lh_vaporization = (2.501 - (2.361 * 10**-3) * (SST - 273.15)) * 1e6  # the latent heat of vaporization at the surface Temperature
    Eva = inputVar_obs['E'] / lh_vaporization *(24. * 60 * 60)

    # MC: Moisture Convergence, represent the water vapor abundance, Unit in mm day^-1
    MC = Precip - Eva
    # print(MC)

    # LTS: Lower Tropospheric Stability, Unit in K (the same as Potential Temperature):
    k = 0.286

    theta_700 = inputVar_obs['T_700'] * (100000. / 70000.)**k
    theta_skin = inputVar_obs['sfc_T'] * (100000. / inputVar_obs['sfc_P'])**k
    LTS_m = theta_700 - theta_skin  # LTS with np.nan

    #.. mask the place with np.nan value
    LTS_e = np.ma.masked_where(theta_700==np.nan, LTS_m)
    # print(LTS_e)

    Subsidence = inputVar_obs['sub']

    # SW radiative flux:
    Rsdt = inputVar_obs['rsdt']
    Rsut = inputVar_obs['rsut']
    Rsutcs = inputVar_obs['rsutcs']

    albedo = Rsut / Rsdt
    albedo_cs = Rsutcs / Rsdt
    Alpha_cre = albedo - albedo_cs
    # abnormal values:
    albedo_cs[(albedo_cs <= 0.08) & (albedo_cs >= 1.00)] == np.nan
    Alpha_cre[(albedo_cs <= 0.08) & (albedo_cs >= 1.00)] == np.nan

    # define Dictionary to store: CCFs(4), gmt, other variables :
    dict0_var = {'gmt': gmt, 'SST': SST, 'p_e': MC, 'LTS': LTS_m, 'SUB': Subsidence, 'LWP': LWP, 'rsdt': Rsdt, 'rsut': Rsut, 'rsutcs': Rsutcs, 'albedo' : albedo, 'albedo_cs': albedo_cs, 'alpha_cre': Alpha_cre, 'LWP_statistic_error': LWP_error, 'Maskarray_mac': Maskarray_mac}

    # Crop the regions
    # crop the variables to the Southern Ocean latitude range: (40 ~ 85^o S)

    variable_nas = ['gmt', 'SST', 'p_e', 'LTS', 'SUB', 'LWP', 'LWP_statistic_error', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'alpha_cre', 'Maskarray_mac']   # all variables names.
    variable_MERRA2 = ['gmt', 'SST', 'p_e', 'LTS', 'SUB']
    variable_CCF = ['SST', 'p_e', 'LTS', 'SUB']
    variable_MAC = ['LWP', 'LWP_statistic_error', 'Maskarray_mac']
    variable_CERES = ['rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'alpha_cre']

    dict1_SO, lat_merra2_so, lon_merra2_so = region_cropping(dict0_var, ['SST', 'p_e', 'LTS', 'SUB'], inputVar_obs['lat_merra2'], inputVar_obs['lon_merra2'], lat_range = [-85., -40.], lon_range = [-180., 180.])

    dict1_SO, lat_mac_so, lon_mac_so = region_cropping(dict1_SO, ['LWP', 'LWP_statistic_error', 'Maskarray_mac'], inputVar_obs['lat_mac'], inputVar_obs['lon_mac'], lat_range =[-85., -40.], lon_range = [-180., 180.])

    # Time-scale average
    # monthly mean (not changed)
    dict2_SO_mon = deepcopy(dict1_SO)

    # annually mean variable
    dict2_SO_yr = get_annually_dict(dict1_SO, ['gmt', 'SST', 'p_e', 'LTS', 'SUB', 'LWP', 'LWP_statistic_error', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'alpha_cre'], inputVar_obs['times_merra2'], label = 'mon')


    # Propagate the np.nan values in 3 different datasets
    # monthly data
    test_array_mon = np.ones((dict2_SO_mon['LWP'].shape))
    for i in ['SST', 'p_e', 'LTS', 'SUB', 'LWP', 'LWP_statistic_error', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'alpha_cre']:
        if dict2_SO_mon[i].shape == dict2_SO_mon['LWP'].shape:
            test_array_mon = test_array_mon * (1. * dict2_SO_mon[i])

    shape_ratio_mon = np.asarray(np.nonzero(np.isnan(test_array_mon) == True)).shape[1] / len(test_array_mon.flatten())

    Maskarray_all_mon = np.isnan(test_array_mon)  # store the mask positions for monthly MERRA-2, MAC-LWP, CERES data in the SO;

    x_array_mon = np.zeros((dict2_SO_mon['SST'].shape))  # used for count the missing points in monthly binned boxes
    x_array_mon[np.isnan(test_array_mon)] = 1.0
    # print(shape_ratio_mon, x_array_mon)

    # Propagating the .nan into monthly mean data:
    for j in ['SST', 'p_e', 'LTS', 'SUB', 'LWP', 'LWP_statistic_error', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'alpha_cre']:
        if dict2_SO_mon[j].shape == dict2_SO_mon['LWP'].shape:
            dict2_SO_mon[j][Maskarray_all_mon] = np.nan

    # annually data
    test_array_yr = np.ones((dict2_SO_yr['LWP'].shape))
    for i in ['SST', 'p_e', 'LTS', 'SUB', 'LWP', 'LWP_statistic_error', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'alpha_cre']:
            if dict2_SO_yr[i].shape == dict2_SO_yr['LWP'].shape:
                test_array_yr = test_array_yr * (1. * dict2_SO_yr[i])

    shape_ratio_yr = np.asarray(np.nonzero(np.isnan(test_array_yr) == True)).shape[1] / len(test_array_yr.flatten())
    # print(shape_ratio_yr)
    Maskarray_all_yr = np.isnan(test_array_yr)  # store the mask positions for annually mean MERRA-2, MAC-LWP, CERES data in the SO;

    x_array_yr = np.zeros((dict2_SO_yr['SST'].shape))  # used for count the missing points in annually mean binned boxes
    x_array_yr[np.isnan(test_array_yr)] = 1.0

    # Propagating the .nan into annually mean data:
    for j in ['SST', 'p_e', 'LTS', 'SUB', 'LWP', 'LWP_statistic_error', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'alpha_cre']:
        if dict2_SO_yr[j].shape == dict2_SO_yr['LWP'].shape:
            dict2_SO_yr[j][Maskarray_all_yr] = np.nan

    # binned (spatial) avergae.
    # Southern Ocean 5 * 5 degree bin box

    #..set are-mean range and define function
    s_range = arange(-90., 90., 5.) + 2.5  #..global-region latitude edge: (36)
    x_range = arange(-180., 180.5, 5.)  #..logitude sequences edge: number: 72
    y_range = arange(-85, -40., 5.) + 2.5  #..southern-ocaen latitude edge: 9


    # binned Monthly variables:
    dict3_SO_mon_bin = {}

    for e in ['SST', 'p_e', 'LTS', 'SUB', 'LWP', 'LWP_statistic_error', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'alpha_cre']:

        dict3_SO_mon_bin[e] = binned_cySouthOcean5(dict2_SO_mon[e], inputVar_obs['lat_ceres'], inputVar_obs['lon_ceres'])
        # since the latitide/ longitude grid for MERRA-2 (data_type = '2') and MAC-LWP/ CERES-EBAF-TOA_Ed4.1 are the same, it does not matter for the choice of lat/lon.

    dict3_SO_mon_bin['gmt'] = binned_cyGlobal5(dict2_SO_mon['gmt'], inputVar_obs['lat_merra2'], inputVar_obs['lon_merra2'])
    print("End monthly data binned.")

    # binned Annually data (it's different than do the binned operation on the 'dict2_SO_yr'):
    dict3_SO_yr_bin = get_annually_dict(dict3_SO_mon_bin, ['gmt', 'SST', 'p_e', 'LTS', 'SUB', 'LWP', 'LWP_statistic_error', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'alpha_cre'], inputVar_obs['times_merra2'])

    print("End annually data binned.")

    # count the ratio of values that are missing in each bin boxes:
    ratio_array = binned_cySO_count(x_array_mon, inputVar_obs['lat_ceres'], inputVar_obs['lon_ceres'])

    ind_binned_omit = np.where(ratio_array >0.499, True, False)  # ignoring bin boxes which has the ratio of np.nan points over 0.5.

    shape_ratio_bin = np.asarray(np.nonzero(ind_binned_omit == True)).shape[1] / len(ind_binned_omit.flatten())
    # print(shape_ratio_bin)   # ratio of bin boxes that should be omited

    for k in ['SST', 'p_e', 'LTS', 'SUB', 'LWP', 'LWP_statistic_error', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'alpha_cre']:
        if dict3_SO_mon_bin[k].shape == dict3_SO_mon_bin['LWP'].shape:
            dict3_SO_mon_bin[k][ind_binned_omit] = np.nan


    # print(dict3_SO_mon_bin)
    
    return dict3_SO_mon_bin, np.asarray(lat_merra2_so), np.asarray(lon_merra2_so), np.asarray(inputVar_obs['times_merra2'])



def Loop_OBS_LRM(data_array_training, data_array_predict, s_range, x_range, y_range):
    # This function is for looping through the entire(most) variable range of SST & SUB, to determine the statistic metrics for partition a LRM using diffferent cut-off;
    # 'data_array' is the dictionary store all the variables after pre-processing;
    
    # flatten the data array for 'training' lrm's coefficiences
    
    dict2_predi_fla_training = {}
    dict2_predi_fla_predict = {}
    
    dict2_predi_ano_training = {}  # need a climatological arrays of variables
    dict2_predi_ano_predict = {}  # need a climatological arrays of variables
    
    dict2_predi_nor_training = {} # standardized anomalies of variables
    dict2_predi_nor_predict = {}
    
    datavar_nas = ['SST', 'p_e', 'LTS', 'SUB', 'LWP', 'LWP_statistic_error', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'alpha_cre']
    
    #..Ravel binned array /Standardized data ARRAY :
    for d in range(len(datavar_nas)):

        dict2_predi_fla_training[datavar_nas[d]] = data_array_training[datavar_nas[d]].flatten()
        dict2_predi_fla_predict[datavar_nas[d]] = data_array_predict[datavar_nas[d]].flatten()
        
        # anomalies in the raw units:
        # dict2_predi_ano_training[datavar_obs[d]] = dict2_predi_fla_training[datavar_obs[d]] - climatological_Area_mean(t)   # Unfinished
        # dict2_predi_ano_predict[datavar_obs[d]] = dict2_predi_fla_predict[datavar_obs[d]] - climatological_Area_mean(t)   # Unfinished
        
        # normalized stardard deviation in unit of './std':
        # dict2_predi_nor_training[datavar_obs[d]] = dict2_predi_ano_training[datavar_obs[d]] / nanstd(Area_mean(climatological_period_data(t, y, x)))
        # dict2_predi_nor_predict[datavar_obs[d]] =  dict2_predi_ano_predict[datavar_obs[d]] / nanstd(Area_mean(climatological_period_data(t, y, x)))
    
    
    #..Use area_mean method, 'np.repeat' and 'np.tile' to reproduce gmt area-mean Array as the same shape as other flattened variables
    GMT_mon_training = area_mean(data_array_training['gmt'], s_range, x_range)
    ## dict2_predi_fla['gmt'] = GMT.repeat(730)  # something wrong when calc dX_dTg(dCCFS_dgmt)
    GMT_mon_predict = area_mean(data_array_predict['gmt'], s_range, x_range)
    
    shape_fla_training = dict2_predi_fla_training['LWP'].shape
    shape_fla_nonnan_training = np.asarray(np.nonzero(np.isnan(dict2_predi_fla_training['LWP']) == False)).shape[1]
    print("shape_fla_training:", shape_fla_training, "shape_fla_nonnan_training:", shape_fla_nonnan_training)
    shape_fla_predict = dict2_predi_fla_predict['LWP'].shape
    shape_fla_nonnan_predict = np.asarray(np.nonzero(np.isnan(dict2_predi_fla_predict['LWP']) == False)).shape[1]
    
    # For pluging in different sets of cut-off (TR_sst & TR_sub) into a bunch of LRMs:
    
    
    ##  split cut-off: TR_sst and TR_sub for N1 and N2 slices in sort of self-defined (Mon)variable ranges

    YY_ay_gcm = data_array_training['SST']
    XX_ay_gcm = data_array_training['SUB']

    y_gcm = np.linspace(np.nanpercentile(YY_ay_gcm, 0.5), np.nanpercentile(YY_ay_gcm, 99.5), 31)   #..supposed to be changed, 31
    x_gcm = np.linspace(np.nanpercentile(XX_ay_gcm, 1.0), np.nanpercentile(XX_ay_gcm, 99.5), 22)   #.., 22

    print("slice SUB bound:  ", x_gcm)
    print("slice SST bound:  ", y_gcm)
    

    # define cut-off:
    TR_sst = full(len(y_gcm)-1, np.nan)
    TR_sub = full(len(x_gcm) -1, np.nan)
    
    for c in arange(len(y_gcm)-1):
        TR_sst[c]  = (y_gcm[c] + y_gcm[c+1]) /2. 
    print("TR_sst : ", TR_sst)
    
    for f in arange(len(x_gcm) -1):
        TR_sub[f]  = (x_gcm[f] + x_gcm[f+1]) /2.
    print("TR_sub : ", TR_sub)
    
    # storage N1*N2 shape output results:
    
    s1 = np.zeros((len(TR_sst), len(TR_sub)))
    s2 = np.zeros((len(TR_sst), len(TR_sub)))
    s3 = np.zeros((len(TR_sst), len(TR_sub)))  
    s4 = np.zeros((len(TR_sst), len(TR_sub)))   #.. for store training data R^2: coefficient of determination
    
    cut_off1 = np.zeros((len(TR_sst), len(TR_sub)))   #..2d, len(y_gcm)-1 * len(x_gcm)-1
    cut_off2 = np.zeros((len(TR_sst), len(TR_sub)))
    
    coefa = []
    coefb = []
    coefc = []
    coefd = []

    # plug the cut-off into LRM tring function:
    for i in range(len(y_gcm)-1):
        for j in range(len(x_gcm)-1):
            s1[i,j], s2[i,j], s3[i,j], s4[i,j], cut_off1[i,j], cut_off2[i,j], coef_a, coef_b, coef_c, coef_d = train_LRM_4(TR_sst[i], TR_sub[j], dict2_predi_fla_training, dict2_predi_fla_predict, shape_fla_training, shape_fla_nonnan_training, shape_fla_predict, shape_fla_nonnan_predict)
    
            print('number: ',i + j + 1)
        
            coefa.append(coef_a)
            coefb.append(coef_b)
            coefc.append(coef_c)
            coefd.append(coef_d)
    
    # find the least bias and its position:
    min_pedict_absbias_id = unravel_index(nanargmin(s1, axis=None), s1.shape)
    max_training_R2_id = unravel_index(nanargmax(s4, axis=None),  s4.shape)
    
    TR_minabias_SST = y_gcm[min_pedict_absbias_id[0]]
    TR_minabias_SUB = x_gcm[min_pedict_absbias_id[1]]
    
    TR_maxR2_SST = y_gcm[max_training_R2_id[0]]
    TR_maxR2_SUB = x_gcm[max_training_R2_id[1]]
    
    
    # Storage data into .npz file for each GCMs
    WD = '/glade/scratch/chuyan/obs_output/'
    
    savez(WD + 'OBS' + '__' + 'STAT_pi+abr_'+'22x_31y_Sep11th', bound_y = y_gcm,bound_x = x_gcm, stats_1 = s1, stats_2 = s2, stats_3 = s3, stats_4 = s4, cut_off1 = cut_off1, cut_off2 = cut_off2, TR_minabias_SST=TR_minabias_SST, TR_minabias_SUB=TR_minabias_SUB, TR_maxR2_SST=TR_maxR2_SST, TR_maxR2_SUB=TR_maxR2_SUB,  coef_a = coefa, coef_b = coefb, coef_c = coefc, coefd = coefd)

    return None



def train_LRM_4(cut_off1, cut_off2, training_data, predict_data, shape_fla_training, shape_fla_nonnan_training, shape_fla_predict, shape_fla_nonnan_predict):
    
    print('4LRM: HERE TR_sst = ', cut_off1, 'K')
    print('4LRM: HERE TR_sub = ', cut_off2, 'Pa s-1')
    
    # print('shape1: ', training_data['LWP'].shape)   # shape1
    
    # Process Training data
    #.. Subtract 'nan' in data, shape1 -> shape2(without 'nan' number) points and shape5('nan' number)
    ind1 = np.isnan(training_data['LTS']) == False 
    ind_true = np.nonzero(ind1 == True)
    #..Sign the the indexing into YB, or YB value will have a big changes
    ind_false = np.nonzero(ind1 == False)
    
    # print('shape of non-nan points: ', array(ind_true).shape)        # shape2
    
    
    # Split data points with skin Temperature < / >=TR_sst & Subsidence@500mb <= / > TR_sub (upward motion / downward motion): 
    # shape1 split into shape3(smaller.TR_sst & up)\shape4(larger.equal.TR_sst & up)\shape5(smaller.TR_sst & down)\shape6(larger.equal.TR_sst & down)
    ind_sstlt_up = np.nonzero((training_data['SST'] < cut_off1) & (training_data['SUB'] <= cut_off2))
    ind_sstle_up = np.nonzero((training_data['SST'] >= cut_off1) & (training_data['SUB'] <= cut_off2))
    ind_sstlt_dw = np.nonzero((training_data['SST'] < cut_off1) & (training_data['SUB'] > cut_off2))
    ind_sstle_dw = np.nonzero((training_data['SST'] >= cut_off1) & (training_data['SUB'] > cut_off2))

    # shape7:the intersection of places where has LTS value and skin_T < TR_sst & SUB500 <= TR_sub
    ind7 = np.intersect1d(ind_true, ind_sstlt_up)
    # print('shape7: ', ind7.shape)   #.. points, shape 7
    
    # shape8:the intersection of places where LTS value and skin_T >= TR_sst & SUB500 <= TR_sub
    ind8 = np.intersect1d(ind_true, ind_sstle_up)
    # print('shape8: ', ind8.shape)   #.. points, shape 8
    
    # shape9:the intersection of places where has LTS value and skin_T < TR_sst & SUB500 > TR_sub
    ind9 = np.intersect1d(ind_true, ind_sstlt_dw)
    # print('shape9: ', ind9.shape)   #.. points, shape 9
    
    # shape10:the intersection of places where LTS value and skin_T >= TR_sst & SUB500 > TR_sub 
    ind10 = np.intersect1d(ind_true, ind_sstle_dw)
    # print('shape10: ', ind10.shape)
    

    #..designate LWP single-array's value, training_data
    YB = np.full((shape_fla_training), 0.0)
    YB[ind_false] = np.nan   
    
    
    #.. Multiple linear regreesion of Liquid Water Path to CCFs :
    
    #..Remove abnormal and missing_values, train model with different TR_sst and TR_sub regimes data
    XX_7 = np.array([training_data['SST'][ind7], training_data['p_e'][ind7], training_data['LTS'][ind7], training_data['SUB'][ind7]])
    XX_8 = np.array([training_data['SST'][ind8], training_data['p_e'][ind8], training_data['LTS'][ind8], training_data['SUB'][ind8]])
    XX_9 = np.array([training_data['SST'][ind9], training_data['p_e'][ind9], training_data['LTS'][ind9], training_data['SUB'][ind9]])
    XX_10 = np.array([training_data['SST'][ind10], training_data['p_e'][ind10], training_data['LTS'][ind10], training_data['SUB'][ind10]])


    if (len(ind7)!=0) & (len(ind8)!=0) & (len(ind9)!=0) & (len(ind10)!=0):
        regr7 = linear_model.LinearRegression()
        result7 = regr7.fit(XX_7.T, training_data['LWP'][ind7])   #..regression for LWP WITH LTS and skin-T < TR_sst & 'up'
        aeffi  = result7.coef_
        aint   = result7.intercept_

        regr8 = linear_model.LinearRegression()
        result8 = regr8.fit(XX_8.T, training_data['LWP'][ind8])   #..regression for LWP WITH LTS and skin-T >= TR_sst &'up'
        beffi  = result8.coef_
        bint   = result8.intercept_
        
        regr9 = linear_model.LinearRegression()
        result9 = regr9.fit(XX_9.T, training_data['LWP'][ind9])   #..regression for LWP WITH LTS and skin-T < TR_sst & 'down'
        ceffi = result9.coef_
        cint = result9.intercept_
        
        regr10 = linear_model.LinearRegression()
        result10 = regr10.fit(XX_10.T, training_data['LWP'][ind10])   #..regression for LWP WITH LTS and skin-T >= TR_sst & 'down'
        deffi = result10.coef_
        dint = result10.intercept_
    
    elif (len(ind7) == 0) & (len(ind9) == 0):
        aeffi = np.full(4, 0.0)
        aint  = 0.0

        regr8 = linear_model.LinearRegression()
        result8 = regr8.fit(XX_8.T, training_data['LWP'][ind8])   #..regression for LWP WITH LTS and skin-T >= TR_sst &'up'
        beffi = result8.coef_
        bint = result8.intercept_
        
        ceffi = np.full(4, 0.0)
        cint = 0.0
        
        regr10 = linear_model.LinearRegression()
        result10 = regr10.fit(XX_10.T, training_data['LWP'][ind10])   #..regression for LWP WITH LTS and skin-T >= TR_sst& 'down'
        deffi  = result10.coef_
        dint   = result10.intercept_
    
    elif len(ind7) == 0:
        aeffi = np.full(4, 0.0)
        aint = 0.0
        
        regr8 = linear_model.LinearRegression()
        result8 = regr8.fit(XX_8.T, training_data['LWP'][ind8])   #..regression for LWP WITH LTS and skin-T >= TR_sst &'up'
        beffi = result8.coef_
        bint = result8.intercept_

        regr9 = linear_model.LinearRegression()
        result9 = regr9.fit(XX_9.T, training_data['LWP'][ind9])   #..regression for LWP WITH LTS and skin-T < TR_sst & 'down'
        ceffi = result9.coef_
        cint = result9.intercept_

        regr10 = linear_model.LinearRegression()
        result10 = regr10.fit(XX_10.T, training_data['LWP'][ind10])   #..regression for LWP WITH LTS and skin-T >= TR_sst& 'down'
        deffi = result10.coef_
        dint = result10.intercept_
        
    elif len(ind9) == 0:
        regr7 = linear_model.LinearRegression()
        result7 = regr7.fit(XX_7.T, training_data['LWP'][ind7])   #..regression for LWP WITH LTS and skin-T < TR_sst & 'up'
        aeffi = result7.coef_
        aint = result7.intercept_
                
        regr8 = linear_model.LinearRegression()
        result8 = regr8.fit(XX_8.T, training_data['LWP'][ind8])   #..regression for LWP WITH LTS and skin-T >= TR_sst &'up'
        beffi = result8.coef_
        bint = result8.intercept_
        
        ceffi = np.full(4, 0.0)
        cint = 0.0

        regr10 = linear_model.LinearRegression()
        result10 = regr10.fit(XX_10.T, training_data['LWP'][ind10])   #..regression for LWP WITH LTS and skin-T >= TR_sst& 'down'
        deffi = result10.coef_
        dint = result10.intercept_
        
    else:
        aeffi = np.full(4, 0.0)
        beffi = np.full(4, 0.0)
        ceffi = np.full(4, 0.0)
        deffi = np.full(4, 0.0)
        aint = 0.0
        bint = 0.0
        cint = 0.0
        dint = 0.0
        
        print('you input a non-wise value for cut-off(TR_sst, TR_sub at 500 mb')
        print('please try another cut-offs input...')
    
    #..save the coefficients:
    coef_a = [np.array(aeffi), aint]
    coef_b = [np.array(beffi), bint]
    coef_c = [np.array(ceffi), cint]
    coef_d = [np.array(deffi), dint]
    

    # Regression for training DATA:
    
    sstle_uplwp_predi_training = np.dot(beffi.reshape(1, -1), XX_8) + bint   #..larger or equal than Tr_SST & SUB at 500 <= TR_sub
    sstlt_uplwp_predi_training = np.dot(aeffi.reshape(1, -1), XX_7) + aint   #..less than Tr_SST & SUB at 500 <= TR_sub
    sstlt_dwlwp_predi_training = np.dot(ceffi.reshape(1, -1), XX_9) + cint   #..less than Tr_SST & SUB at 500 > TR_sub
    sstle_dwlwp_predi_training = np.dot(deffi.reshape(1, -1), XX_10) + dint   #..larger or equal than Tr_SST & SUB at 500 > TR_sub
    
    
    # emsembling into 'YB' predicted data array for Pi:
    YB[ind7] = sstlt_uplwp_predi_training
    YB[ind8] = sstle_uplwp_predi_training
    YB[ind9] = sstlt_dwlwp_predi_training
    YB[ind10] = sstle_dwlwp_predi_training
    
    
    # Test performance for training data:
    
    abs_BIAS = np.nansum(np.abs(training_data['LWP'][ind_true] - YB[ind_true])) / shape_fla_nonnan_training
    
    MSE_shape1 = mean_squared_error(training_data['LWP'][ind_true].reshape(-1,1), YB[ind_true].reshape(-1,1))
    # print("RMSE_shape1 for training data lwp: ", sqrt(MSE_shape1))
    
    R_2_shape1 = r2_score(training_data['LWP'][ind_true].reshape(-1, 1), YB[ind_true].reshape(-1,1))
    print("R_2_shape1 for trainging data lwp: ", R_2_shape1)
    
    ## print(training_data['LWP'].reshape(-1,1), YB.reshape(-1,1))
    r_shape1, p_shape1 = pearsonr(np.asarray(training_data['LWP'][ind_true]), array(YB[ind_true]))
    print("Pearson correlation coefficient: ", r_shape1,  "p_value = ", p_shape1)
    

    # Examine the effectiveness of regression model:
    print('examine regres-mean LWP for training shape1:', np.nanmean(training_data['LWP']), np.nanmean(YB))
    print('examine regres-mean LWP for training shape10:', np.nanmean(training_data['LWP'][ind10]), np.nanmean(sstle_dwlwp_predi_training))

    
    # Process predict data
    
    #.. Subtract 'nan' in data, shape1 -> shape2(without 'nan' number) points and shape5('nan' number)
    ind2 = np.isnan(predict_data['LTS']) == False 
    ind_true_predict = np.nonzero(ind2 == True)
    #..Sign the the indexing into YB, or YB value will have a big changes
    ind_false_predict = np.nonzero(ind2 == False)
    
    # print('shape of non-nan points: ', array(ind_true_predict).shape)        # shape2
    
    
    # Split data points with skin Temperature < / >=TR_sst & Subsidence@500mb <= / > TR_sub (upward motion / downward motion): 
    # shape1 split into shape3(smaller.TR_sst & up)\shape4(larger.equal.TR_sst & up)\shape5(smaller.TR_sst & down)\shape6(larger.equal.TR_sst & down)
    ind_sstlt_up_predict = np.nonzero((predict_data['SST'] < cut_off1) & (predict_data['SUB'] <= cut_off2))
    ind_sstle_up_predict = np.nonzero((predict_data['SST'] >= cut_off1) & (predict_data['SUB'] <= cut_off2))
    ind_sstlt_dw_predict = np.nonzero((predict_data['SST'] < cut_off1) & (predict_data['SUB'] > cut_off2))
    ind_sstle_dw_predict = np.nonzero((predict_data['SST'] >= cut_off1) & (predict_data['SUB'] > cut_off2))

    # shape7:the intersection of places where has LTS value and skin_T < TR_sst & SUB500 <= TR_sub
    ind7_predict = np.intersect1d(ind_true_predict, ind_sstlt_up_predict)
    # print('shape7 of predict data: ', ind7_predict.shape)   #.. points, shape 7
    
    # shape8:the intersection of places where LTS value and skin_T >= TR_sst & SUB500 <= TR_sub
    ind8_predict = np.intersect1d(ind_true_predict, ind_sstle_up_predict)
    # print('shape8 of predict data: ', ind8_predict.shape)   #.. points, shape 8
    
    # shape9:the intersection of places where has LTS value and skin_T < TR_sst & SUB500 > TR_sub
    ind9_predict = np.intersect1d(ind_true_predict, ind_sstlt_dw_predict)
    # print('shape9 of predict data: ', ind9_predict.shape)   #.. points, shape 9
    
    # shape10:the intersection of places where LTS value and skin_T >= TR_sst & SUB500 > TR_sub 
    ind10_predict = np.intersect1d(ind_true_predict, ind_sstle_dw_predict)
    # print('shape10 of predict data: ', ind10_predict.shape)
    
    
    #..designate LWP single-array's value, training_data
    YB2 = np.full((shape_fla_predict), 0.0)
    YB2[ind_false_predict] = np.nan   
    
    
    #.. Multiple linear regreesion of Liquid Water Path to CCFs :
    
    #..Remove abnormal and missing_values, train model with different TR_sst and TR_sub regimes data
    XX_7_predict = np.array([predict_data['SST'][ind7_predict], predict_data['p_e'][ind7_predict], predict_data['LTS'][ind7_predict], predict_data['SUB'][ind7_predict]])
    XX_8_predict = np.array([predict_data['SST'][ind8_predict], predict_data['p_e'][ind8_predict], predict_data['LTS'][ind8_predict], predict_data['SUB'][ind8_predict]])
    XX_9_predict = np.array([predict_data['SST'][ind9_predict], predict_data['p_e'][ind9_predict], predict_data['LTS'][ind9_predict], predict_data['SUB'][ind9_predict]])
    XX_10_predict = np.array([predict_data['SST'][ind10_predict], predict_data['p_e'][ind10_predict], predict_data['LTS'][ind10_predict], predict_data['SUB'][ind10_predict]])


    if (len(ind7_predict)!=0) & (len(ind8_predict)!=0) & (len(ind9_predict)!=0) & (len(ind10_predict)!=0):
        regr7_predi = linear_model.LinearRegression()
        result7_predi = regr7_predi.fit(XX_7_predict.T, predict_data['LWP'][ind7_predict])   #..regression for LWP WITH LTS and skin-T < TR_sst & 'up'
        aeffi_predi = result7_predi.coef_
        aint_predi = result7_predi.intercept_

        regr8_predi = linear_model.LinearRegression()
        result8_predi = regr8_predi.fit(XX_8_predict.T, predict_data['LWP'][ind8_predict])   #..regression for LWP WITH LTS and skin-T >= TR_sst &'up'
        beffi_predi = result8_predi.coef_
        bint_predi = result8_predi.intercept_
        
        regr9_predi = linear_model.LinearRegression()
        result9_predi = regr9_predi.fit(XX_9_predict.T, predict_data['LWP'][ind9_predict])   #..regression for LWP WITH LTS and skin-T < TR_sst & 'down'
        ceffi_predi = result9_predi.coef_
        cint_predi = result9_predi.intercept_
        
        regr10_predi = linear_model.LinearRegression()
        result10_predi = regr10_predi.fit(XX_10_predict.T, predict_data['LWP'][ind10_predict])   #..regression for LWP WITH LTS and skin-T >= TR_sst & 'down'
        deffi_predi = result10_predi.coef_
        dint_predi = result10_predi.intercept_
    
    elif (len(ind7_predict) == 0) & (len(ind9_predict) == 0):
        aeffi_predi = np.full(4, 0.0)
        aint_predi = 0.0

        regr8_predi = linear_model.LinearRegression()
        result8_predi = regr8_predi.fit(XX_8_predict.T, predict_data['LWP'][ind8_predict])   #..regression for LWP WITH LTS and skin-T >= TR_sst &'up'
        beffi_predi = result8_predi.coef_
        bint_predi = result8_predi.intercept_
        
        ceffi_predi = np.full(4, 0.0)
        cint_predi = 0.0
        
        regr10_predi = linear_model.LinearRegression()
        result10_predi = regr10_predi.fit(XX_10_predict.T, predict_data['LWP'][ind10_predict])   #..regression for LWP WITH LTS and skin-T >= TR_sst& 'down'
        deffi_predi = result10_predi.coef_
        dint_predi = result10_predi.intercept_
    
    elif len(ind7_predict) == 0:
        aeffi_predi = np.full(4, 0.0)
        aint_predi = 0.0
        
        regr8_predi = linear_model.LinearRegression()
        result8 = regr8_predi.fit(XX_8_predict.T, predict_data['LWP'][ind8_predict])   #..regression for LWP WITH LTS and skin-T >= TR_sst &'up'
        beffi_predi = result8_predi.coef_
        bint_predi = result8_predi.intercept_

        regr9_predi = linear_model.LinearRegression()
        result9_predi = regr9_predi.fit(XX_9_predict.T, predict_data['LWP'][ind9_predict])   #..regression for LWP WITH LTS and skin-T < TR_sst & 'down'
        ceffi_predi = result9_predi.coef_
        cint_predi = result9_predi.intercept_

        regr10_predi = linear_model.LinearRegression()
        result10_predi = regr10_predi.fit(XX_10_predict.T, predict_data['LWP'][ind10_predict])   #..regression for LWP WITH LTS and skin-T >= TR_sst& 'down'
        deffi_predi = result10_predi.coef_
        dint_predi = result10_predi.intercept_
        
    elif len(ind9_predict) == 0:
        regr7_predi = linear_model.LinearRegression()
        result7_predi = regr7_predi.fit(XX_7_predict.T, predict_data['LWP'][ind7_predict])   #..regression for LWP WITH LTS and skin-T < TR_sst & 'up'
        aeffi_predi = result7_predi.coef_
        aint_predi = result7_predi.intercept_
                
        regr8_predi = linear_model.LinearRegression()
        result8_predi = regr8_predi.fit(XX_8_predict.T, predict_data['LWP'][ind8_predict])   #..regression for LWP WITH LTS and skin-T >= TR_sst &'up'
        beffi_predi = result8_predi.coef_
        bint_predi = result8_predi.intercept_
        
        ceffi_predi = np.full(4, 0.0)
        cint_predi = 0.0

        regr10_predi = linear_model.LinearRegression()
        result10_predi = regr10_predi.fit(XX_10_predict.T, predict_data['LWP'][ind10_predict])   #..regression for LWP WITH LTS and skin-T >= TR_sst& 'down'
        deffi_predi = result10_predi.coef_
        dint_predi = result10_predi.intercept_
        
    else:
        aeffi_predi = np.full(4, 0.0)
        beffi_predi = np.full(4, 0.0)
        ceffi_predi = np.full(4, 0.0)
        deffi_predi = np.full(4, 0.0)
        aint_predi = 0.0
        bint_predi = 0.0
        cint_predi = 0.0
        dint_predi = 0.0
        
        print('you input a non-wise value for cut-off(TR_sst, TR_sub at 500 mb')
        print('please try another cut-offs input...')
    
    #..save the coefficients:
    coef_a_predi = [np.array(aeffi_predi), aint_predi]
    coef_b_predi = [np.array(beffi_predi), bint_predi]
    coef_c_predi = [np.array(ceffi_predi), cint_predi]
    coef_d_predi = [np.array(deffi_predi), dint_predi]
    

    # Regression for predict DATA:
    
    sstle_uplwp_predi_predict = np.dot(beffi_predi.reshape(1, -1), XX_8_predict) + bint_predi   #..larger or equal than Tr_SST & SUB at 500 <= TR_sub
    sstlt_uplwp_predi_predict = np.dot(aeffi_predi.reshape(1, -1), XX_7_predict) + aint_predi   #..less than Tr_SST & SUB at 500 <= TR_sub
    sstlt_dwlwp_predi_predict = np.dot(ceffi_predi.reshape(1, -1), XX_9_predict) + cint_predi   #..less than Tr_SST & SUB at 500 > TR_sub
    sstle_dwlwp_predi_predict = np.dot(deffi_predi.reshape(1, -1), XX_10_predict) + dint_predi   #..larger or equal than Tr_SST & SUB at 500 > TR_sub
    
    
    # emsembling into 'YB' predicted data array for Pi:
    YB2[ind7_predict] = sstlt_uplwp_predi_predict
    YB2[ind8_predict] = sstle_uplwp_predi_predict
    YB2[ind9_predict] = sstlt_dwlwp_predi_predict
    YB2[ind10_predict] = sstle_dwlwp_predi_predict
    
    
    # Test performance for training data:
    
    abs_BIAS_predict = np.nansum(np.abs(predict_data['LWP'][ind_true_predict] - YB2[ind_true_predict])) / shape_fla_nonnan_predict
    
    MSE_shape1_predict = mean_squared_error(predict_data['LWP'][ind_true_predict].reshape(-1,1), YB2[ind_true_predict].reshape(-1,1))
    # print("RMSE_shape1 for training data lwp: ", sqrt(MSE_shape1_predict))
    
    R_2_shape1_predict = r2_score(predict_data['LWP'][ind_true_predict].reshape(-1, 1), YB2[ind_true_predict].reshape(-1,1))
    print("R_2_shape1 for predict data lwp: ", R_2_shape1_predict)
    
    ## print(training_data['LWP'].reshape(-1,1), YB.reshape(-1,1))
    r_shape1_predict, p_shape1_predict = pearsonr(np.asarray(predict_data['LWP'][ind_true_predict]), array(YB2[ind_true_predict]))
    print("Pearson correlation coefficient for predict data: ", r_shape1_predict,  "p_value = ", p_shape1_predict)
    

    # Examine the effectiveness of regression model:
    # print('examine regres-mean LWP for predict shape1:', np.nanmean(predict_data['LWP']), np.nanmean(YB2))
    # print('examine regres-mean LWP for predict shape10:', np.nanmean(predict_data['LWP'][ind10_predict]), np.nanmean(sstle_dwlwp_predi_predict))
    
    return abs_BIAS_predict, sqrt(MSE_shape1_predict), r_shape1, R_2_shape1, cut_off1, cut_off2, coef_a, coef_b, coef_c, coef_d

