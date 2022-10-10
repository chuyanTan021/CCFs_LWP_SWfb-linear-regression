### Try to replicate Daniel's methods:

import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm

# import PyNIO as Nio # deprecated
import xarray as xr
import pandas
import glob
from copy import deepcopy
from scipy.stats import *
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from area_mean import *
from binned_cyFunctions5 import *
from read_hs_file import read_var_mod

from get_LWPCMIP5data import *
from get_LWPCMIP6data import *
from useful_func_cy import *


def calc_Radiation_LRM_2(inputVar_pi, inputVar_abr, **model_data):

    # inputVar_pi, inputVar_abr are the data from read module: get_CMIP6data.py

    #..get the shapes of monthly data
    shape_lat = len(inputVar_pi['lat'])
    shape_lon = len(inputVar_pi['lon'])
    shape_time_pi = len(inputVar_pi['times'])
    shape_time_abr = len(inputVar_abr['times'])
    #print(shape_lat, shape_lon, shape_time_pi, shape_time_abr)


    #..choose lat 40 -85 °S as the Southern-Ocean Regions
    lons = inputVar_pi['lon'] *1.
    lats = inputVar_pi['lat'][:] *1.

    levels = np.array(inputVar_abr['pres'])
    times_pi = inputVar_pi['times'] *1.
    times_abr = inputVar_abr['times'] *1.

    lati1 = -40.
    latsi1 = min(range(len(lats)), key = lambda i: abs(lats[i] - lati1))
    lati0 = -85.
    latsi0 = min(range(len(lats)), key = lambda i: abs(lats[i] - lati0))
    print('lat index for -85S; -40S', latsi0, latsi1)

    shape_latSO = (latsi1+1) - latsi0
    print('shape of latitudinal index in raw data: ', shape_latSO)


    # Read the Radiation data and LWP
    # piControl and abrupt-4xCO2
    # LWP
    LWP_pi = np.array(inputVar_pi['clwvi']) - np.array(inputVar_pi['clivi'])   #..units in kg m^-2
    LWP_abr = np.array(inputVar_abr['clwvi']) - np.array(inputVar_abr['clivi'])   #..units in kg m^-2
    
    # abnormal 'Liquid Water Path' value:
    if np.nanmin(LWP_abr)<1e-5:
        LWP_abr = np.array(inputVar_abr['clwvi'])
        print('abr4x clwvi mislabeled')
        
    if np.nanmin(LWP_pi)<1e-5:
        LWP_pi = np.array(inputVar_pi['clwvi'])
        print('piControl clwvi mislabeled')
    
    # SW radiation metrics
    Rsdt_pi = np.array(inputVar_pi['rsdt'])
    Rsut_pi = np.array(inputVar_pi['rsut'])
    Rsutcs_pi = np.array(inputVar_pi['rsutcs'])
    print("shape of data in 'piControl':  ", Rsut_pi.shape, " mean 'piControl' upwelling SW radiation flux in the SO (Assume with cloud): "
    , nanmean(Rsut_pi[:,latsi0:latsi1+1,:]))

    Rsdt_abr = np.array(inputVar_abr['rsdt'])
    Rsut_abr = np.array(inputVar_abr['rsut'])
    Rsutcs_abr = np.array(inputVar_abr['rsutcs'])
    print("shape of data in 'abrupt-4xCO2':  ",  Rsut_abr.shape, " mean 'abrupt-4xCO2' upwelling SW radiation flux in the SO (Assume with cloud): ",  nanmean(Rsut_abr[:,latsi0:latsi1+1,:]))

    print(" mean |'abrupt-4xCO2' - 'piControl'| upwelling SW radiation flux (ALL-sky - Clear-sky) in the SO: ", 
          (nanmean(Rsut_abr[:,latsi0:latsi1+1,:] - Rsutcs_abr[:,latsi0:latsi1+1,:]) - nanmean(Rsut_pi[:,latsi0:latsi1+1,:] - Rsutcs_pi[:,latsi0:latsi1+1,:])))

    # albedo, albedo_clear sky; albedo(alpha)_cre: all-sky - clear-sky
    Albedo_pi = Rsut_pi / Rsdt_pi
    Albedo_cs_pi = Rsutcs_pi / Rsdt_pi
    Alpha_cre_pi = Albedo_pi - Albedo_cs_pi

    Albedo_abr = Rsut_abr / Rsdt_abr
    Albedo_cs_abr = Rsutcs_abr / Rsdt_abr
    Alpha_cre_abr = Albedo_abr - Albedo_cs_abr
    print(" mean |'abrupt-4xCO2' - 'piControl'| albedo (ALL-sky - Clear-sky) in the SO: ", 
          (nanmean(Alpha_cre_abr[:,latsi0:latsi1+1,:]) - nanmean(Alpha_cre_pi[:,latsi0:latsi1+1,:])))
    
    # Pre-processing the data with abnormal values
    Albedo_abr[(Albedo_cs_abr <= 0.08) & (Albedo_cs_abr >= 1.00)] = np.nan
    Albedo_cs_abr[(Albedo_cs_abr <= 0.08) & (Albedo_cs_abr >= 1.00)] = np.nan
    Alpha_cre_abr[(Albedo_cs_abr <= 0.08) & (Albedo_cs_abr >= 1.00)] = np.nan
    LWP_abr[(Albedo_cs_abr <= 0.08) & (Albedo_cs_abr >= 1.00)] = np.nan
    Rsdt_abr[(Albedo_cs_abr <= 0.08) & (Albedo_cs_abr >= 1.00)] = np.nan
    
    Albedo_pi[(Albedo_cs_pi <= 0.08) & (Albedo_cs_pi >= 1.00)] = np.nan
    Albedo_cs_pi[(Albedo_cs_pi <= 0.08) & (Albedo_cs_pi >= 1.00)] = np.nan
    Alpha_cre_pi[(Albedo_cs_pi <= 0.08) & (Albedo_cs_pi >= 1.00)] = np.nan
    LWP_pi[(Albedo_cs_pi <= 0.08) & (Albedo_cs_pi >= 1.00)] = np.nan
    Rsdt_pi[(Albedo_cs_pi <= 0.08) & (Albedo_cs_pi >= 1.00)] = np.nan
    
    # As data dictionary:
    datavar_nas = ['LWP', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'alpha_cre']   #..7 varisables except gmt (lon dimension diff)

    dict0_PI_var = {'LWP': LWP_pi, 'rsdt': Rsdt_pi, 'rsut': Rsut_pi, 'rsutcs': Rsutcs_pi, 'albedo' : Albedo_pi, 'albedo_cs': Albedo_cs_pi, 'alpha_cre': Alpha_cre_pi, 'lat': lats, 'lon': lons, 'times': times_pi, 'pres': levels}

    dict0_abr_var = {'LWP': LWP_abr, 'rsdt': Rsdt_abr, 'rsut': Rsut_abr, 'rsutcs': Rsutcs_abr, 'albedo': Albedo_abr, 'albedo_cs': Albedo_cs_abr, 'alpha_cre': Alpha_cre_abr, 'lat': lats, 'lon': lons, 'times': times_abr, 'pres': levels}

    dict1_PI_var = deepcopy(dict0_PI_var)
    dict1_abr_var = deepcopy(dict0_abr_var)

    print('month in piControl and abrupt-4xCO2: ', times_pi[0,:][1], times_abr[0,:][1])
    
    # Choose time frame: January
    if times_pi[0,:][1] == 1.0:   # Jan
        shape_mon_PI_raw = dict0_PI_var['LWP'][0::12, latsi0:latsi1 +1,:].shape   # January data shape
        for i in range(len(datavar_nas)):
            dict1_PI_var[datavar_nas[i]] = dict1_PI_var[datavar_nas[i]][0::12, :, :]   # January data

    else:
        shape_mon_PI_raw = dict0_PI_var['LWP'][int(13 - times_pi[0,:][1])::12, latsi0:latsi1 +1,:].shape 
        for i in range(len(datavar_nas)):
            dict1_PI_var[datavar_nas[i]] = dict1_PI_var[datavar_nas[i]][int(13 - times_pi[0,:][1])::12, :, :]

    if times_abr[0,:][1] == 1.0:   # Jan
        shape_mon_abr_raw = dict0_abr_var['LWP'][0::12, latsi0:latsi1 +1,:].shape   # January data shape
        for j in range(len(datavar_nas)):
            dict1_abr_var[datavar_nas[j]] = dict1_abr_var[datavar_nas[j]][0::12, :, :]   # January data

    else:
        shape_mon_abr_raw = dict0_abr_var['LWP'][int(13 - times_abr[0,:][1])::12, latsi0:latsi1 +1,:].shape 
        for j in range(len(datavar_nas)):
            dict1_abr_var[datavar_nas[j]] = dict1_abr_var[datavar_nas[j]][int(13 - times_abr[0,:][1])::12, :, :]
    
    
    # Choose regional frame: SO (40 ~ 85 .S)
    for c in range(len(datavar_nas)):
        dict1_PI_var[datavar_nas[c]] = dict1_PI_var[datavar_nas[c]][:,latsi0:latsi1+1, :]   # Southern Ocean data
        dict1_abr_var[datavar_nas[c]] = dict1_abr_var[datavar_nas[c]][:,latsi0:latsi1+1, :]  # Southern Ocean data
    
    # radiative transfer model: single-regime LRM:
    threshold_list = [0.12, 0.15, 0.20, 0.30, 0.35, 0.50, 1.00]
    
    # piControl:
    coef_dict_Albedo_PI, coef_dict_Alpha_cre_PI = radiative_transfer_model(dict1_PI_var, threshold_list, label = 'piControl')
    
    # compare: (abrupt4xCO2)
    coef_dict_Albedo_abr, coef_dict_Alpha_cre_abr = radiative_transfer_model(dict1_abr_var, threshold_list, label = 'abrupt4xCO2')
    
    # Plotting:
    
    print(model_data)
    pLot_sca_sensitivity_to_albedo_cs(dict1_PI_var, coef_dict_Albedo_PI, threshold_list, c_albedo_cs=0.08, **model_data)
    
    
    return coef_dict_Alpha_cre_PI, coef_dict_Albedo_PI, coef_dict_Alpha_cre_abr, coef_dict_Albedo_abr



def radiative_transfer_model(data_dict, threshold_list, label = 'piControl'):
    # ---------------
    # 'data_dict' is the dictionary store the variables for calc radiative tranfer model (lwp, albedo, albedo_cs, ..)
    # 'threshold_list' is a list of the threshold values of 'albedo_cs': for filtering out the points with albedo_cs >= Threshold ;
    ## now calc two different radiative transfer models: 
    # M1. albedo = a1 * lwp + a2 * albedo_cs + a3;
    # M2. alpha_cre = albedo - albedo_cs = a1 * lwp + a2
    # ---------------
    
    coef_dict_Albedo = {}
    coef_dict_Alpha_cre = {}
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
        x[x >= np.nanpercentile(x, 95)] = np.nan
        print("threshold = ", TR_albedo_cs)
        
        # Processing 'nan' in aggregated data:
        Z_PI = (rsdt * ck_a * x * y2 * y1) * 1.
        ind_false = np.isnan(Z_PI)
        ind_true = np.logical_not(ind_false)
        
        print(" fration of not NaN points to All points" + " in "+label+ ": " + 
             str(np.asarray(np.nonzero(ind_true == True)).shape[1]/ len(ind_true.flatten())))
    
        # data_frame used for statsmodel:
        data = pandas.DataFrame({'x': x[ind_true].flatten(), 'y2': y2[ind_true].flatten(), 'y1': y1[ind_true].flatten(), 'ck_a': ck_a[ind_true].flatten()})

        # Fit the model
        model1 = ols("y2 ~ x", data).fit()
        model2 = ols("y1 ~ x + ck_a", data).fit()
        # print the summary
        print(" ")
        print("model1, alpha_cre = a1 * lwp + a2: ", ' ', model1.summary())
        print(" ")
        print("model2, albedo = a1* lwp + a2 * albedo_cs + a3: ", ' ', model2.summary())

        coef_array_alpha_cre = np.asarray([model1._results.params[1], model1._results.params[0]])
        coef_array_albedo = np.asarray([model2._results.params[1], model2._results.params[2], model2._results.params[0]])
        
        
        coef_dict_Albedo[str(threshold_list[a] *100.)] = coef_array_albedo
        coef_dict_Alpha_cre[str(threshold_list[a] *100.)] = coef_array_alpha_cre
    
    return coef_dict_Albedo, coef_dict_Alpha_cre



def pLot_sca_sensitivity_to_albedo_cs(data_dict, coef_dict, threshold_list, c_albedo_cs= 0.1, **model_data):
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
    
    x = np.linspace(-0.005, np.nanmax(lwp), 54)
    y = x
    
    # scatter plot of specific gcm:
    scac2 = ax2.scatter(lwp[(ck_albedo <= 1.0)],albedo[(ck_albedo<= 1.0)]
                              , c = ck_albedo[(ck_albedo<= 1.0)], s = 15, cmap = cm.rainbow)

    # scac2=ax2.scatter(lwp[(ck_albedo>= 0.18)&(ck_albedo<= 0.25)],albedo[(ck_albedo>= 0.18)&(ck_albedo<= 0.25)]
                             # , c = ck_albedo[(ck_albedo>= 0.18)& (ck_albedo<= 0.25)], s = 15, cmap = cm.rainbow)

    ax2.set_xlabel("$LWP,\ [kg\ m^{2}]$", fontsize = 15)
    ax2.set_ylabel(r"$\alpha \ $", fontsize = 15)
    cb2 = fig2.colorbar(scac2, shrink = 0.9, aspect = 6)
    cb2.set_label(r"$clear-sky\ \alpha$", fontsize = 15)
    
    for i in range(len(threshold_list)):
        
        ax2.plot(x, coef_dict[str(threshold_list[i] *100.)][0]*x + coef_dict[str(threshold_list[i] *100.)][1] * c_albedo_cs + coef_dict[str(threshold_list[i] *100.)][2], linewidth = 1.56, color = color_list[i], label = r'$ \alpha_{cs} < $' + str(threshold_list[i]))

    plt.title("GCM: " + model_data['modn'] + r"$\ with\ fitting\ line\ of\ appling\ TR_{\alpha_{cs}} $", fontsize = 17)  # \ 0.18 \leq \alpha_{cs} \leq 0.25\
    plt.legend(loc = 'lower right', fontsize = 13)
    plt.savefig(path6 + "GCM: " + model_data['modn']+"_albedo_LWP(95per)_coloredby_albedo_cs.jpg", bbox_inches ='tight', dpi = 300)
    
    
    plt.close()
    
    return None
    
