### Get the data we need from read func: 'get_LWPCMIP6', and do some data-processing for building the linear regression Clouds- Radiation model;
### transform data to annual-mean/ monthly-mean bin array or flattened array; ###
### Fitting the linear regression in single regime model from piControl LWP and Radiation metrics, then do the regressions and save the data.

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


from get_LWPCMIP6data import *
from useful_func_cy import *


def calc_Radiation_LRM_1(inputVar_pi, inputVar_abr, TR_albedo = 0.15):


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
    LWP_pi  = np.array(inputVar_pi['clwvi']) - np.array(inputVar_pi['clivi'])   #..units in kg m^-2
    LWP_abr = np.array(inputVar_abr['clwvi']) - np.array(inputVar_abr['clivi'])   #..units in kg m^-2

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
    for k in range(len(datavar_nas)):
        dict1_PI_var[datavar_nas[k]] = dict1_PI_var[datavar_nas[k]][:, latsi0:latsi1+1, :]   # Southern Ocean data
        dict1_abr_var[datavar_nas[k]] = dict1_abr_var[datavar_nas[k]][:, latsi0:latsi1+1, :]  # Southern Ocean data


    # radiative transfer model: single regime LRM:

    # training (PI):

    x_pi = dict1_PI_var['LWP'].flatten()
    y_pi = dict1_PI_var['alpha_cre'].flatten()
    y2_pi = dict1_PI_var['albedo'].flatten()
    cs_pi = dict1_PI_var['albedo_cs'].flatten()
    print("'TR_albedo_cs : '", TR_albedo, ".")

    # rule out Land and Sea Ice: albedo_cs< = threshold_alpha:
    ind_nolsi_pi = cs_pi <= TR_albedo
    print("albedo < "+ str(TR_albedo), " ratio in 'piCl': "+str(asarray(nonzero(ind_nolsi_pi == True)).shape[1] / ind_nolsi_pi.shape[0]))
    data_pi = pandas.DataFrame({'x': x_pi[ind_nolsi_pi], 'y': y_pi[ind_nolsi_pi], 'y2': y2_pi[ind_nolsi_pi], 'cs': cs_pi[ind_nolsi_pi]})

    # Fit the model
    model1 = ols("y ~ x", data_pi).fit()
    model2 = ols("y2 ~ x + cs", data_pi).fit()
    # print the summary
    print(model1.summary())
    print(model2.summary())

    coef_array_alpha_cre_pi = np.asarray([model1._results.params[1], model1._results.params[0]])
    coef_array_albedo_pi = np.asarray([model2._results.params[1], model2._results.params[2], model2._results.params[0]])

    # compare: (abrupt-4xCO2)

    x_abr = dict1_abr_var['LWP'].flatten()
    y_abr = dict1_abr_var['alpha_cre'].flatten()
    y2_abr = dict1_abr_var['albedo'].flatten()
    cs_abr = dict1_abr_var['albedo_cs'].flatten()


    # rule out Land and Sea Ice: albedo_cs< = threshold_alpha:
    ind_nolsi_abr = cs_abr <= TR_albedo
    print("albedo < "+ str(TR_albedo), " ratio in 'abr4x': "+str(asarray(nonzero(ind_nolsi_abr == True)).shape[1] / ind_nolsi_abr.shape[0]))
    data_abr = pandas.DataFrame({'x': x_abr[ind_nolsi_abr], 'y': y_abr[ind_nolsi_abr], 'y2': y2_abr[ind_nolsi_abr], 'cs': cs_abr[ind_nolsi_abr]})

    # Fit the model
    model1_abr = ols("y ~ x", data_abr).fit()
    model2_abr = ols("y2 ~ x + cs", data_abr).fit()
    # print the summary
    print(model1_abr.summary())
    print(model2_abr.summary())
    coef_array_alpha_cre_abr = np.asarray([model1_abr._results.params[1], model1_abr._results.params[0]])
    coef_array_albedo_abr = np.asarray([model2_abr._results.params[1], model2_abr._results.params[2], model2_abr._results.params[0]])



    
    return coef_array_alpha_cre_pi, coef_array_albedo_pi, coef_array_alpha_cre_abr, coef_array_albedo_abr
