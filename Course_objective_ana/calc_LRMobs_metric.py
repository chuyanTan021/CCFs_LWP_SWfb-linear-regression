### This module is to get the obs data we need from read func: 'get_OBSLRMdata', and calculate for CCFs and the required Cloud properties; 
## Crop regions, Transform the data to be annually mean, binned array form;
## Create the linear regression 2 & 4 regimes models from current climate sensitivity of cloud properties to the CCFs and save the data.

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
from useful_func_cy import *
from calc_Radiation_LRM_1 import *
from calc_Radiation_LRM_2 import *




def calc_LRMobs_metrics(THRESHOLD_sst, THRESHOLD_sub, test_flag = 'test1'):
    # get the variable:
    inputVar_obs = get_OBSLRM(test = test_flag)
    # ------------------------ 
    # radiation code
    
    # ------------------------
    
    # Data processing
    # GMT: Global mean surface air Temperature (2-meter), Unit in K
    gmt = inputVar_obs['tas'] * 1.
    # SST: Sea Surface Temperature or skin- Temperature, Unit in K
    SST = inputVar_obs['sfc_T'] * 1.
    # Precip: Precipitation, Unit in mm day^-1 (convert from kg m^-2 s^-1)
    Precip = inputVar_obs['P'] * (24. * 60 * 60)
    # Eva: Evaporation, Unit in mm day^-1 (here use the latent heat flux from the sfc, unit convert from W m^-2 --> kg m^-2 s^-1 --> mm day^-1)
    lh_vaporization = (2.501 - (2.361 * 10**-3) * (SST - 273.15)) * 1e6  # the latent heat of vaporization at the surface Temperature
    Eva = inputVar_obs['E'] / lh_vaporization * (24. * 60 * 60)

    # MC: Moisture Convergence, represent the water vapor abundance, Unit in mm day^-1
    MC = Precip - Eva
    print(MC)

    # LTS: Lower Tropospheric Stability, Unit in K (the same as Potential Temperature):
    k = 0.286

    theta_700 = inputVar_obs['T_700'] * (100000. / 70000.)**k
    theta_skin = inputVar_obs['sfc_T'] * (100000. / inputVar_obs['sfc_P'])**k
    LTS_m = theta_700 - theta_skin  # LTS with np.nan

    #..Subtract the outliers in T_700 and LTS_m, 'nan' comes from missing T_700 data
    LTS_e = np.ma.masked_where(theta_700 >= 500, LTS_m)
    print(LTS_e)

    Subsidence = inputVar_obs['sub']

    # define Dictionary to store: CCFs(4), gmt, other variables :
    dict0_var = {'gmt': gmt, 'SST': SST, 'p_e': MC, 'LTS': LTS_e, 'SUB': Subsidence}  #  ,'LWP': LWP, 'rsdt': Rsdt_pi, 'rsut': Rsut_pi, 'rsutcs': Rsutcs_pi, 'albedo' : Albedo_pi, 'albedo_cs': Albedo_cs_pi, 'alpha_cre': Alpha_cre_pi, 

    # Crop the regions
    # crop the variables to the Southern Ocean latitude range: (40 ~ 85^o S)

    variable_nas = ['SST', 'p_e', 'LTS', 'SUB']

    dict1_SO, lat_merra2_so, lon_merra2_so = region_cropping(dict0_var, variable_nas, inputVar_obs['lat_merra2'], inputVar_obs['lon_merra2'], lat_range = [-85., -40.], lon_range = [-180., 180.])
    
    # Time-scale average
    # monthly mean (not changed)
    dict2_SO_mon = deepcopy(dict1_SO)
    
    # annually mean variable
    dict2_SO_yr = get_annually_dict(dict1_SO, ['gmt', 'SST', 'p_e', 'LTS', 'SUB'], inputVar_obs['times_merra2'], label = 'mon')
    
    # binned (spatial) avergae
    # Southern Ocean 5 * 5 degree bin box
    #..set are-mean range and define function
    s_range = arange(-90., 90., 5.) + 2.5  #..global-region latitude edge: (36)
    x_range = arange(-180., 180., 5.)  #..logitude sequences edge: number: 72
    y_range = arange(-85, -40., 5.) +2.5  #..southern-ocaen latitude edge: 9
    # binned Monthly variables:
    dict3_SO_mon_bin = {}

    for c in range(len(variable_nas)):

        dict3_SO_mon_bin[variable_nas[c]] = binned_cySouthOcean5(dict2_SO_mon[variable_nas[c]], lat_merra2_so, lon_merra2_so)

    dict3_SO_mon_bin['gmt'] = binned_cyGlobal5(dict2_SO_mon['gmt'], inputVar_obs['lat_merra2'], lon_merra2_so)
    print("Every monthly data")
    