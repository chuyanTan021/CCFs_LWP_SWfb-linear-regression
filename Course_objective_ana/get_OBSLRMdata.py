## read OBS data for training LRM in CCFs analysis

import netCDF4
from numpy import *
import matplotlib.pyplot as plt
import xarray as xr

import pandas as pd
import glob
from scipy.stats import *

from read_hs_file import read_var_mod
from read_var_obs import *


def get_OBSLRM(test = 'test1'):
    # This function is for reading the observation data from the MERRA-2 Re-analysis, .. for LRM training and testing.
    
    T_alevs, lat_merra2, lon_merra2, Pres, times_merra2 = read_var_obs_MERRA2(varnm = 'T', read_p = True, valid_range1=[2002, 7, 15], valid_range2=[2016, 12, 31])
    T_700 = T_alevs[:, 12, :,:]  # 700 hPa level
    
    sub = read_var_obs_MERRA2(varnm = 'OMEGA500', read_p = False, valid_range1=[2002, 7, 15], valid_range2=[2016, 12, 31])[0]
    sfc_T = read_var_obs_MERRA2(varnm = 'TS', read_p = False, valid_range1=[2002, 7, 15], valid_range2=[2016, 12, 31])[0]
    sfc_P = read_var_obs_MERRA2(varnm = 'PS', read_p = False, valid_range1=[2002, 7, 15], valid_range2=[2016, 12, 31])[0]
    tas = read_var_obs_MERRA2(varnm = 'T2M', read_p = False, valid_range1=[2002, 7, 15], valid_range2=[2016, 12, 31])[0]
    P = read_var_obs_MERRA2(varnm = 'PRECTOT', read_p = False, valid_range1=[2002, 7, 15], valid_range2=[2016, 12, 31])[0]
    E = read_var_obs_MERRA2(varnm = 'EFLUX', read_p = False, valid_range1=[2002, 7, 15], valid_range2=[2016, 12, 31])[0]
    
    
    inputVar_obs = {'sfc_T': sfc_T, 'T_700': T_700, 'sfc_P': sfc_P, 'sub': sub, 'tas': tas, 'P': P, 'E': E, 'pres': Pres, 'lat_merra2':lat_merra2, 'lon_merra2':lon_merra2, 'times_merra2': times_merra2}   # 'clivi': clivi_abr, 'clwvi':clwvi_abr, 'rsdt': rsdt_abr, 'rsut': rsut_abr, 'rsutcs': rsutcs_abr..
    
    return inputVar_obs