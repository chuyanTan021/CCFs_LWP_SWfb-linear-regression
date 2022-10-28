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
    
    LWP_training[LWP_training <= 0.0] = np.nan
    LWP_training[LWP_training >= np.nanpercentile(LWP_training, 95)] = np.nan
    
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
    
    LWP_predict[LWP_predict <= 0.0] = np.nan
    LWP_predict[LWP_predict >= np.nanpercentile(LWP_predict, 95)] = np.nan
    
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
        shape_mon_training_raw = dict1_SO_training['LWP'][int(13 - dict1_SO_training['times'][0,:][1])::12, :,:].shape 
        for i in range(len(datavar_nas)):
            dict2_training_var[datavar_nas[i]] = dict1_SO_training[datavar_nas[i]][int(13 - dict1_SO_training['times'][0,:][1])::12, :, :]

    if dict1_SO_predict['times'][0,:][1] == 1.0:   # Jan
        shape_mon_abr_raw = dict1_SO_predict['LWP'][0::12,:,:].shape   # January data shape
        for j in range(len(datavar_nas)):
            dict2_predict_var[datavar_nas[j]] = dict1_SO_predict[datavar_nas[j]][0::12, :, :]   # January data

    else:
        shape_mon_abr_raw = dict1_SO_predict['LWP'][int(13 - dict1_SO_predict['times'][0,:][1])::12, :,:].shape 
        for j in range(len(datavar_nas)):
            dict2_predict_var[datavar_nas[j]] = dict1_SO_predict[datavar_nas[j]][int(13 - dict1_SO_predict['times'][0,:][1])::12, :, :]

    
    # radiative transfer model: single regime LRM:
    
    threshold_list = [0.12, 0.15, 0.20, 0.50]   # [0.12, 0.15, 0.20, 0.30, 0.35, 0.50, 1.00]
    # training :
    
    coef_dict_Albedo_training, coef_dict_Alpha_cre_training, coef_dict_Albedo_bin_training = radiative_transfer_model_obs(dict2_training_var, threshold_list, label = 'training')

    # Compare to the training:
    # predicting :

    coef_dict_Albedo_predict, coef_dict_Alpha_cre_predict, coef_dict_Albedo_bin_predict = radiative_transfer_model_obs(dict2_predict_var, threshold_list, label = 'predict')
    
    # PLotting:
    pLot_sca_sensitivity_to_albedo_cs_obs2(dict2_training_var, coef_dict_Albedo_training, coef_dict_Albedo_bin_training, threshold_list, c_albedo_cs= 0.114)
    
    
    return coef_dict_Alpha_cre_training, coef_dict_Albedo_training, coef_dict_Albedo_bin_training, coef_dict_Alpha_cre_predict, coef_dict_Albedo_predict, coef_dict_Albedo_bin_predict



def radiative_transfer_model_obs(data_dict, threshold_list, label = 'training'):
    # ---------------
    # 'data_dict' is the dictionary store the variables for calc radiative tranfer model (lwp, albedo, albedo_cs, ..)
    # 'threshold_list' is a list of the threshold values of 'albedo_cs': for filtering out the points with albedo_cs >= Threshold ;
    ## now calc two different radiative transfer models: 
    # M1. albedo = a1 * lwp + a2 * albedo_cs + a3;
    # M2. alpha_cre = albedo - albedo_cs = a1 * lwp + a2
    # ---------------    
    coef_dict_Albedo = {}
    coef_dict_Alpha_cre = {}
    coef_dict_Albedo_bin = {}
    # Loop through filter threshold:
    for a in range(len(threshold_list)):
        
        TR_albedo_cs = threshold_list[a]
        
        # copy data from dictionary:
        
        x = deepcopy(data_dict['LWP'])
        
        y2 = deepcopy(data_dict['alpha_cre'])
        
        y1 = deepcopy(data_dict['albedo'])
        
        ck_a = deepcopy(data_dict['albedo_cs'])
        
        rsdt = deepcopy(data_dict['rsdt'])
        # conditions:
        rsdt[rsdt < 10.0] = np.nan
        ck_a[ck_a < 0] = np.nan
        ck_a[ck_a >= TR_albedo_cs] = np.nan
        
        # rsdt[rsdt < 10.0] = np.nan
        # ck_a[ck_a < 0] = np.nan
        
        # x[x >= np.nanpercentile(x, 95)] = np.nan
        print("threshold = ", TR_albedo_cs)
        
        # Processing 'nan' in aggregated data:
        Z_training = (rsdt * ck_a * x * y2 * y1) * 1.
        ind_false = np.isnan(Z_training)
        ind_true = np.logical_not(ind_false)
        
        print(" fration of not NaN points to All points" + " in OBS "+label+ " data: " + 
             str(np.asarray(np.nonzero(ind_true == True)).shape[1]/ len(ind_true.flatten())))
        
        # binned data by LWP:
        BINS_lwp = np.linspace(0.00, 0.24, 24)
        
        # data_frame used for binned ditribution regression:
        mean_albedo, bin_edge1, binnumber_albedo = binned_statistic(x[ind_true], y1[ind_true], statistic='mean', bins = BINS_lwp)
        mean_albedo_cs, bin_edge2, binnumber_albedo_cs = binned_statistic(x[ind_true], ck_a[ind_true], statistic='mean', bins = BINS_lwp)
        x_lwp = (BINS_lwp[0:-1] + (BINS_lwp[1] - BINS_lwp[0]) / 2.)
        data_bin = pandas.DataFrame({'y1': mean_albedo, 'ck_a': mean_albedo_cs, 'x': x_lwp})

        # data_frame used for raw distribution regression:
        data = pandas.DataFrame({'x': x[ind_true].flatten(), 'y2': y2[ind_true].flatten(), 'y1': y1[ind_true].flatten(), 'ck_a': ck_a[ind_true].flatten()})

        # Fit the model
        
        # binned distribution regression:
        
        model_binLWP = ols("y1 ~ x + ck_a", data_bin).fit()
        print(" model_binLWP, albedo_bin = a1 * lwp_bin + a2 * albedo_cs_bin + a3: ", model_binLWP.summary())

        coef_array_albedo_bin = np.asarray([model_binLWP._results.params[1], model_binLWP._results.params[2], model_binLWP._results.params[0]])
        
        # raw distribution regression:
        
        model1 = ols("y2 ~ x", data).fit()
        model2 = ols("y1 ~ x + ck_a", data).fit()
        # print the summary
        print(" ")
        print("model1, alpha_cre = a1 * lwp + a2: ", ' ', model1.summary())
        print(" ")
        print("model2, albedo = a1 * lwp + a2 * albedo_cs + a3: ", ' ', model2.summary())
        coef_array_alpha_cre = np.asarray([model1._results.params[1], model1._results.params[0]])
        coef_array_albedo = np.asarray([model2._results.params[1], model2._results.params[2], model2._results.params[0]])
        
        # store the coef dictionary for different albedo_cs thresholds.
        coef_dict_Albedo_bin[str(threshold_list[a] *100.)] = coef_array_albedo_bin
        coef_dict_Albedo[str(threshold_list[a] *100.)] = coef_array_albedo
        coef_dict_Alpha_cre[str(threshold_list[a] *100.)] = coef_array_alpha_cre
    
    
    return coef_dict_Albedo, coef_dict_Alpha_cre, coef_dict_Albedo_bin



def pLot_sca_sensitivity_to_albedo_cs_obs1(data_dict, coef_dict, threshold_list, c_albedo_cs= 0.115):
    # ---------------
    # 'data_dict' is the dictionary store the variables for visualize relation between albedo over lwp, color by albedo_cs;
    # 'threshold_list' is a list of the threshold values of 'albedo_cs': for filtering out the points with albedo_cs >= Threshold;
    # 'coef_dict' is the dictionary store the fitting line coefficients for M1, M2.
    # ---------------
    
    # s_range = arange(-90., 90., 5.) + 2.5  #..global-region latitude edge: (36)
    # x_range = arange(-180., 180., 5.)  #..logitude sequences edge: number: 72
    # y_range = arange(-85, -40., 5.) +2.5  #..southern-ocaen latitude edge: 9
    
    # path1 = '/glade/scratch/chuyan/CMIP_output/CMIP_lrm_RESULT/'
    path6 = '/glade/scratch/chuyan/Plots/CMIP_R_lwp_2/'
    
    albedo = np.array(data_dict['albedo'])
    # print(albedo)
    ck_albedo = np.array(data_dict['albedo_cs'])
    # print(ck_albedo)
    lwp = np.array(data_dict['LWP'])
    # print(lwp)
    
    # PLot:
    from matplotlib import cm
    fig2 = plt.figure(figsize = (12, 9))
    ax2 = fig2.add_subplot(111)
    
    color_list = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:olive", "tab:cyan"]
    
    x = np.linspace(0.005, np.nanmax(lwp), 54)
    y = x
    
    # scatter plot of specific gcm:
    scac2 = ax2.scatter(lwp[(ck_albedo <= 1.0)], albedo[(ck_albedo<= 1.0)]
                            , c = ck_albedo[(ck_albedo<= 1.0)], s = 15, cmap = cm.rainbow)

    # scac2=ax2.scatter(lwp[(ck_albedo>= 0.18)&(ck_albedo<= 0.25)],albedo[(ck_albedo>= 0.18)&(ck_albedo<= 0.25)]
                             # , c = ck_albedo[(ck_albedo>= 0.18)& (ck_albedo<= 0.25)], s = 15, cmap = cm.rainbow)

    ax2.set_xlabel("$LWP,\ [kg\ m^{2}]$", fontsize = 15)
    ax2.set_ylabel(r"$\alpha \ $", fontsize = 15)
    cb2 = fig2.colorbar(scac2, shrink = 0.9, aspect = 6)
    cb2.set_label(r"$clear-sky\ \alpha$", fontsize = 15)
    
    for i in range(len(threshold_list)):
        
        ax2.plot(x, coef_dict[str(threshold_list[i] *100.)][0]* np.log(x) + coef_dict[str(threshold_list[i] *100.)][1] * c_albedo_cs + coef_dict[str(threshold_list[i] *100.)][2], linewidth = 1.56, color = color_list[i], label = r'$ \alpha_{cs} < $' + str(threshold_list[i]))

    plt.title("OBS: " + r"$\ with\ fitting\ line\ of\ appling\ TR_{\alpha_{cs}} $", fontsize = 17)  # \ 0.18 \leq \alpha_{cs} \leq 0.25\
    plt.legend(loc = 'lower right', fontsize = 13)
    plt.savefig(path6 + "LOG OBS:_"+"albedo_LWP(995per)_coloredby_albedo_cslog-scale.jpg", bbox_inches ='tight', dpi = 300)
    
    
    plt.close()
    
    return None


def pLot_sca_sensitivity_to_albedo_cs_obs2(data_dict, coef_dict, coef_dict_bin, threshold_list, c_albedo_cs= 0.115):
    # ---------------
    # 'data_dict' is the dictionary store the variables for visualize relation between albedo over lwp, color by albedo_cs;
    # 'threshold_list' is a list of the threshold values of 'albedo_cs': for filtering out the points with albedo_cs >= Threshold;
    # 'coef_dict' is the dictionary store the fitting line coefficients for raw distributed M1(current) (or M2);
    # 'coef_dict_bin' is the dict store the coefficients for binned distributed M1(current) (or M2);
    # ---------------
    
    # s_range = arange(-90., 90., 5.) + 2.5  #..global-region latitude edge: (36)
    # x_range = arange(-180., 180., 5.)  #..logitude sequences edge: number: 72
    # y_range = arange(-85, -40., 5.) +2.5  #..southern-ocaen latitude edge: 9
    
    # path1 = '/glade/scratch/chuyan/CMIP_output/CMIP_lrm_RESULT/'
    path6 = '/glade/scratch/chuyan/Plots/CMIP_R_lwp_3/'
    
    albedo = np.array(data_dict['albedo'])
    # print(albedo)
    ck_albedo = np.array(data_dict['albedo_cs'])
    # print(ck_albedo)
    lwp = np.array(data_dict['LWP'])
    # print(lwp)
    
    # PLot:
    from matplotlib import cm
    fig = plt.figure(figsize = (16, 6))
    ax1 = fig.add_subplot(121)
    # PLot 'albedo' vs. 'lwp', colored by 'albedo_cs':
    color_list = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:olive", "tab:cyan"]
    
    x = np.linspace(0.005, np.nanmax(lwp), 54)
    y = x
    
    # scatter plot of specific gcm:
    denc1 = ax1.scatter(lwp[(ck_albedo <= 0.30)], albedo[(ck_albedo<= 0.30)]
                            , c = ck_albedo[(ck_albedo<= 0.30)], s = 15, cmap = cm.rainbow, vmin = 0, vmax = 0.30)
    # denc1 = ax1.scatter(lwp[(ck_albedo>= 0.18)&(ck_albedo<= 0.25)],albedo[(ck_albedo>= 0.18)&(ck_albedo<= 0.25)]
                             # , c = ck_albedo[(ck_albedo>= 0.18)& (ck_albedo<= 0.25)], s = 15, cmap = cm.rainbow)

    ax1.set_xlabel("$LWP,\ [kg\ m^{2}]$", fontsize = 15)
    ax1.set_ylabel(r"$\alpha \ $", fontsize = 15)
    cb1 = fig.colorbar(denc1, ax = ax1, shrink = 0.9, aspect = 6.5)
    cb1.set_label(r"$clear-sky\ \alpha$", fontsize = 15)
    
    for i in range(len(threshold_list)):
        # raw distributed M1:
        ax1.plot(x, (coef_dict[str(threshold_list[i] *100.)][0]* x + coef_dict[str(threshold_list[i] *100.)][1] * c_albedo_cs + coef_dict[str(threshold_list[i] *100.)][2]), linewidth = 1.56, color = color_list[i], label = r'$ \alpha_{cs} < $' + str(threshold_list[i]))
        # binned distributed M1:
        ax1.plot(x, (coef_dict_bin[str(threshold_list[i] *100.)][0]* x + coef_dict_bin[str(threshold_list[i] *100.)][1] * c_albedo_cs + coef_dict_bin[str(threshold_list[i] *100.)][2]), linewidth = 2.0, linestyle = '--', color = color_list[i], label = r'$binned\ LWP\ fit,\ \alpha_{cs} <$' + str(threshold_list[i]))
        
    ax1.set_title(r"$\ fitting\ line\ of\ appling\ TR_{\alpha_{cs}} $", fontsize = 15)  # \ 0.18 \leq \alpha_{cs} \leq 0.25\
    ax1.legend(loc = 'lower right', fontsize = 9)

    ax2 = fig.add_subplot(122)
    # PLot the density distribution(#) of points:
    denc2 = ax2.hexbin(lwp[(ck_albedo <= 0.30)], albedo[(ck_albedo<= 0.30)], gridsize =(25, 25), cmap = plt.cm.Greens)  # , cmap = plt.cm.Greens)
    
    ax2.set_xlabel("$LWP,\ [kg\ m^{2}]$", fontsize = 12)
    ax2.set_ylabel(r"$\alpha $", fontsize = 12)
    cb2 = fig.colorbar(denc2, ax = ax2, shrink = 0.9, aspect = 6.5)
    ax2.set_title(" distribution", fontsize = 15)
    
    plt.suptitle("OBS ", fontsize = 17)
    plt.savefig(path6 + "pD+bin OBS:_"+"albedo_LWP(995per)_coloredby_albedo_csle0.30.jpg", bbox_inches ='tight', dpi = 250)
    
    plt.close()
    
    return None
