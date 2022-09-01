# This module is for loading/ processing the required OBS data for LRM training, for MERRA-2 Reanalysis; MAC-LWP; and CERES_EBAF-TOA_Ed4.1;

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import xarray
import pandas
import glob
from datetime import datetime
from scipy.stats import *
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm

from area_mean import *
from binned_cyFunctions5 import *
from read_hs_file import read_var_mod


pp_path_OBS='/glade/scratch/chuyan/obs_data/'

def read_var_obs_MERRA2(varnm, read_p = False, valid_range1 = [2002, 7, 15], valid_range2 = [2016, 12, 31]):
    ### ---------------
    # This function is for reading MERRA-2 Re-analysis (Meteorology) dataset.
    # read_p = False: 2-D variable; True: 3-D variable (e.g., air Temerature, vetical Pressure velocity)
    # valid_range1 & 2: the starting and end time_stamps of time sequential metric. Only first two numbers are valid for monthly data.
    ### ---------------

    # 'read_hs' functionality:
    # read the data file, according to the variable name: varnm and the data loading path: pp_path_OBS

    folder = pp_path_OBS
    if varnm in ['EFLUX', 'PRECTOT']:
        fn = glob.glob(folder + '*'+ 'tavgM_2d_flx_Nx.' + '*' + 'nc4.nc4?' + 'EFLUX,PRECTOT' + '*')
        # print(fn)
    elif varnm in ['OMEGA500', 'PS', 'QV10M', 'T2M', 'TS']:
        fn = glob.glob(folder + '*'+ 'tavgM_2d_slv_Nx.' + '*' + 'nc4.nc4?' + 'OMEGA500,PS,QV10M,T2M,TS' + '*')
        # print(fn)
    elif varnm in ['OMEGA', 'T']:
        fn = glob.glob(folder + '*'+ 'instM_3d_asm_Np.' + '*' + 'nc4.nc4?' + 'OMEGA,T' + '*')
        # print(fn)

    # 'read_hs_file' functionality:
    # loading the data files one by one through 'netCDF4' module, but in a random order of times
    data = []
    P = []
    timeo = []

    for i in range(len(fn)):

        file = nc.Dataset(fn[i], 'r')

        lat = file.variables['lat']  # Latitude 
        lon = file.variables['lon']  # Longitude
        if read_p == True:
            P = file.variables['lev'][:]  # Pressure levels

        tt = (file.variables['time'])  # numeric value

        # create a shape = (n, 3) array to store the (year, mon, day) cf.datetime object:
        time_i = np.zeros((len(tt), 3))

        for i in range(len(tt)):

            tt1 = nc.num2date(tt[i], file.variables['time'].units,calendar = u'standard')  # cf.Datetime object: including yr, mon, day, hour, minute, second info

            time_i[i,:] = [tt1.year, tt1.month, tt1.day]
        # print(np.asarray(time_i).shape)

        data_pieces = []

        # determine whether the variable_time within the time_range we want:
        if valid_range1[0] != valid_range2[0]:   # case 1, starting time and ending time are the different year.
            if ((time_i[0, 0] > valid_range1[0]) & (time_i[0, 0] < valid_range2[0])) | ((time_i[0, 0] == valid_range1[0]) & (time_i[0, 1] >= valid_range1[1])) | ((time_i[0, 0] == valid_range2[0]) & (time_i[0, 1] <= valid_range2[1])):

                data_pieces = file.variables[varnm]

        elif (valid_range1[0] == valid_range2[0]) & (time_i[0, 0] == valid_range2[0]):   # case 2, starting and ending time are the same year.
            if  (time_i[0, 1] >= valid_range1[1]) & (time_i[0, 1] <= valid_range2[1]):

                data_pieces = file.variables[varnm]


        # end 'read_hs_file' functionality.

        if len(data_pieces) > 0:
            data.append(data_pieces)  # Variable
            timeo.append(time_i)  # Times
    # ending loop, and end 'read_hs' functionality.
    
    print(np.asarray(P).shape)
    
    
    # 'read_var_mod' functionality
    # processing lat, lon, P, data, and time array, output in an ordered arrangement
    
    # use 'np.concatenate' to get rid of one extra axis (the second axes)
    dataOUT = np.concatenate(data, axis = 0)
    timeOUT = np.concatenate(timeo, axis = 0)
    
    # replacing fill value to be 'np.nan'
    dataOUT = np.asarray(dataOUT)
    dataOUT[dataOUT == file.variables[varnm]._FillValue] = np.nan
    
    # use 'np.unique' to get ordered time and data array
    tf = timeOUT[:, 0] + timeOUT[:, 1]/100.
    TF, ind = np.unique(tf, return_index = True)  # TF is the sorted (time from smaller value to bigger value), unique 'tf', and ind is the indices
    
    dataOUT = dataOUT[ind]
    timeOUT = timeOUT[ind]
    
    # return np.ma.array(dataOUT, fill_value=np.nan), np.ma.array(lat[:], fill_value=np.nan), np.ma.array(lon[:], fill_value=np.nan), np.asarray(P), timeOUT
    return np.array(dataOUT), np.array(lat[:]), np.array(lon[:]), np.asarray(P), timeOUT

def read_var_obs_MACLWP():
    ### ---------------
    # This function is for reading MAC-LWP (LWP) dataset.
    
    ### --------------
    return None


def read_var_obs_CERES():
    ### ---------------
    # This function is for reading CERES_EBAF-TOA_Ed4.1 (Radiative Flux) dataset.
    
    ### --------------
    
    return None