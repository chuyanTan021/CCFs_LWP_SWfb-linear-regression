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

def read_var_obs_MERRA2(varnm, read_p = False, valid_range1 = [2002, 7, 15], valid_range2 = [2016, 12, 31], data_type = '2'):
    ### ---------------
    # This function is for reading MERRA-2 Re-analysis (Meteorology) dataset (netCDF-4 classic).
    # read_p = False: 2-D variable; True: 3-D variable (e.g., air Temerature, vetical Pressure velocity)
    # valid_range1 & 2: the starting and end time_stamps of time sequential metric. Only first two numbers are valid for monthly data.
    # data_type: there are two types of MERRA-2 data, '1': raw resolution (0.5 X 0.625, -180 ~ 180, -90 ~ 90)
    # '2': regrided resolution (1 X 1, 0.5 ~ 359.5, 89.5 ~ -89.5)
    ### ---------------
    
    # 'read_hs' functionality:
    # read the data file, according to the variable name: varnm and the data loading path: pp_path_OBS

    folder = pp_path_OBS
    if data_type == '1':
        if varnm in ['EFLUX', 'PRECTOT']:
            fn = glob.glob(folder + '*'+ 'tavgM_2d_flx_Nx.' + '*' + 'nc4.nc4?' + 'EFLUX,PRECTOT' + '*')
            # print(fn)
        elif varnm in ['OMEGA500', 'PS', 'QV10M', 'T2M', 'TS']:
            fn = glob.glob(folder + '*'+ 'tavgM_2d_slv_Nx.' + '*' + 'nc4.nc4?' + 'OMEGA500,PS,QV10M,T2M,TS' + '*')
            # print(fn)
        elif varnm in ['OMEGA', 'T']:
            fn = glob.glob(folder + '*'+ 'instM_3d_asm_Np.' + '*' + 'nc4.nc4?' + 'OMEGA,T' + '*')
            # print(fn)

    if data_type == '2':
        if varnm in ['EFLUX', 'PRECTOT']:
            fn = glob.glob(folder + '*'+ 'tavgM_2d_flx_Nx.' + '*' + '.nc')
            # print(fn)
        elif varnm in ['OMEGA500', 'PS', 'QV10M', 'T2M', 'TS']:
            fn = glob.glob(folder + '*'+ 'tavgM_2d_slv_Nx.' + '*' + '.nc')
            # print(fn)
        elif varnm in ['OMEGA', 'T']:
            fn = glob.glob(folder + '*'+ 'instM_3d_asm_Np.' + '*' + '.nc')
            print(fn)

    # 'read_hs_file' functionality:
    # loading the data files one by one through 'netCDF4' module, but in a random order of times
    data = []
    P = []
    timeo = []

    for i in range(len(fn)):

        file = nc.Dataset(fn[i], 'r')  # random order of times (due to the functionality of "glob")

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

    if data_type == '2':

        lat = lat[::-1]

        lon2 = lon[:] *1.
        lon2[lon2 > 180] = lon2[lon2 > 180] - 360.  # convert to range from -180 to 180.
        ind_sort = np.argsort(lon2)
        lon2 = lon2[ind_sort]
        lon = lon2
        dataOUT1 = dataOUT.copy()
        if read_p == True:
            dataOUT1 = dataOUT1[:, :, ::-1, :]
            dataOUT1 = dataOUT1[:, :, :, ind_sort]

        else:
            dataOUT1 = dataOUT1[:, ::-1, :]
            dataOUT1 = dataOUT1[:, :, ind_sort]

            dataOUT = dataOUT1


    return np.array(dataOUT), np.array(lat[:]), np.array(lon[:]), np.asarray(P), timeOUT



def read_var_obs_MAClwp(varnm = 'cloudlwp', valid_range1 = [2002, 7, 15], valid_range2 = [2016, 12, 31]):
    ### ----------------
    # This function is for reading MAC-LWP (LWP) dataset (netCDF-4 format).
    # noticed there are missing_value portion on the Sea-Ice and Land surface
    # 'varnm' could be 'lwp': Liquid Water Path and 'twp': Total Water Path;
    # valid_range1 & 2: the starting and end time_stamps of time sequential metric. Only first two numbers are valid.
    ### ---------------

    folder = pp_path_OBS

    fn = glob.glob(folder+"maclwp_cloudlwpave_*_v1.nc4")
    # print(fn)
    tt_str = np.arange(valid_range1[0], valid_range2[0]+1, 1)
    print(tt_str)

    data = []  # Variable
    timeo = []  # Times
    data_error = []  # Statistic 1-Sigma Error

    for F in range(0, (valid_range2[0] - valid_range1[0] + 1)):

        f = nc.Dataset(folder + "maclwp_cloudlwpave_" + str(tt_str[F]) + "_v1.nc4")


        # read data_slice, time_slice, lat, lon.
        lat = f.variables['lat']   # Latitude
        lat_bnds = f.variables['lat_bnds']  # latitude bounds, in shape (180, 2)
        lon = f.variables['lon']   # Longitude
        lon_bnds = f.variables['lon_bnds']  # longitude bounds, in shape (360, 2)
        time_slice = f.variables['time']  # numeric value time for 12 months in a year
        
        lon2 = lon[:]*1.
        lon2[lon2 > 180] = lon2[lon2 > 180] - 360.  # convert to range from -180 to 180.
        ind_sort = np.argsort(lon2)
        lon2 = lon2[ind_sort]
        # print(ind_sort)

        # create a shape = (n, 3) array to store the (year, month, day) cf.datetime object
        time_i = np.zeros((len(time_slice), 3))
        for i in range(len(time_slice)):

            tt1 = nc.num2date(time_slice[i], f.variables['time'].units,calendar = u'360_day')  # cf.Datetime object: including yr, mon, day, hour, minute, second info
            # print(tt1)
            time_i[i, :] = [tt1.year, tt1.month, tt1.day]
        # print(time_i)

        # determine whether the variable_time within the time_range we want:
        if valid_range1[0] != valid_range2[0]:   # case 1, starting time and ending time are the different year.
            if (time_i[0, 0] <= valid_range2[0]) & (time_i[0, 0] >= valid_range1[0]):

                if ((time_i[0,0] == valid_range1[0]) & (valid_range1[1] > 1)):
                    data_slice = f.variables[varnm][(valid_range1[1]-1):]  # data in a year, shape in (12, 180, 360)
                    data_errorslice = f.variables[varnm+'_error'][(valid_range1[1]-1):]  # statistic data error, shape also in (12, 180, 360)

                    time_i = time_i[(valid_range1[1]-1):] *1.
                elif ((time_i[0,0] == valid_range2[0]) & (valid_range2[1] < 12)):
                    data_slice = f.variables[varnm][:(valid_range2[1])]
                    data_errorslice = f.variables[varnm+'_error'][:(valid_range2[1])]

                    time_i = time_i[:(valid_range2[1])] *1.
                else:
                    data_slice = np.ma.array(f.variables[varnm])  # ..
                    data_errorslice = f.variables[varnm+'_error']

                    time_i = time_i *1.

        elif (valid_range1[0] == valid_range2[0]) & (time_i[0, 0] == valid_range2[0]):   # case 2, starting and ending time are the same year.

            data_slice = f.variables[varnm][(valid_range1[1]-1):(valid_range2[1])]
            data_errorslice = f.variables[varnm+'_error'][(valid_range1[1]-1):(valid_range2[1])]

            time_i = time_i[(valid_range1[1]-1):(valid_range2[1])] *1.

        # concatenate the data_slice and time_slice:
        if len(data_slice) > 0:

            data.append(data_slice)
            timeo.append(time_i)
            data_error.append(data_errorslice)

    # use 'np.concatenate' to get rid of one extra axis 
    dataOUT = np.ma.concatenate(data, axis = 0)
    timeOUT = np.concatenate(timeo, axis = 0)
    data_errorOUT = np.ma.concatenate(data_error, axis = 0)

    # retrieve the "filled_value" and the mask of "MaskedArray" 
    # and replace the MaskedArray as an ndarray, with mask position as 'np.nan':
    filled_value_mac = f.variables[varnm]._FillValue
    Mask_arrayOUT = np.ma.getmaskarray(dataOUT)

    dataOUT2 = dataOUT.data  # the data array (ndarray), with "filled_value_mac" in the masked position instead of '-'
    data_errorOUT2 = data_errorOUT.data

    dataOUT2[dataOUT2 == filled_value_mac] = np.nan
    data_errorOUT2[data_errorOUT2 == filled_value_mac] = np.nan


    return dataOUT2[:, :, ind_sort], data_errorOUT2[:, :, ind_sort], Mask_arrayOUT[:, :, ind_sort], np.array(lat), np.array(lon2), timeOUT



def read_var_obs_CERES(varnm = 'toa_sw_all_mon', valid_range1 = [2002, 7, 15], valid_range2 = [2016, 12, 31]):
    ### ----------------
    # This function is for reading CERES_EBAF-TOA_Ed4.1 (Radiative Flux) dataset (netCDF-4 classic).
    # 'varnm' could be 1. 'toa_sw_all_mon': top of the Atmosphere Shortwave flux in all-sky (Up), monthly mean value; 
    # 2. 'toa_lw_all_mon': top of the Atmos Longwave flux in all-sky (Up), monthly mean value;
    # 3. 'toa_net_all_mon': top of the Atmos Net flux in all-sky (Down): = 1 + 2 + 4;
    # 4. 'solar_mon': top of the Atmos incoming Shortwave flux (Down), monthly mean value;
    # 5. 'toa_sw_clr_c_mon': top of the Atmos Shortwave flux in clear-sky conditions (Up), monthly mean value;
    # 6. 'toa_lw_clr_c_mon': top of the Atmos Longwave flux in clear-sky conditions (Up), monthly mean value;
    # 7. 'toa_net_clr_c_mon': top of the Atmos Net flux inn clear-sky (Down): = 5 + 6 + 4;
    # 8. 'cldarea_total_daynight_mon': cloud Area Fraction, in daytime and nighttime;
    # 9. 'cldpress_total_daynight_mon': cloud Effective Pressure, unit in hPa, in daytime and nighttime;
    # 10. 'cldtau_total_day_mon': cloud visible Optical Depth, unit in tau (undimensional), Only in daytime;
    # valid_range1 & 2: the starting and end time_stamps of time sequential metric. Only first two numbers are valid.
    ### ---------------
    
    folder = pp_path_OBS
    
    fn = glob.glob(folder+"CERES_EBAF-TOA_Ed4.1_Subset_200207-202203.nc")
    
    data = []
    P = []
    timeo = []
    
    # print(fn)
    f = nc.Dataset(fn[0], 'r')
    
    # Read variable, lat, lon, time:
    
    lat = f.variables['lat']  # Latitude, already shrinked into 40 ~ 85 S 
    lon = f.variables['lon']  # Longitude
    time_slice = f.variables['time']  # numeric value for time from 2002.07 through 2022.03.
    
    data_slice = f.variables[varnm]

    lon2 = lon[:]*1.
    lon2[lon2 > 180] = lon2[lon2 > 180] - 360.  # convert to range from -180 to 180.
    ind_sort = np.argsort(lon2)
    lon2 = lon2[ind_sort]
    # print(ind_sort)
    
    # create a shape = (n, 3) array to store the (year, month, day) cf.datetime object
    time_i = np.zeros((len(time_slice), 3))

    for i in range(len(time_slice)):

        tt1 = nc.num2date(time_slice[i], f.variables['time'].units, calendar = u'standard')  # cf.Datetime object: including yr, mon, day, hour, minute, second info
        # print(tt1)
        time_i[i, :] = [tt1.year, tt1.month, tt1.day]


    # Choose the variable time for the time range we want:
    ind_start = 0
    ind_end = 0
    s = 0  # count for the corresponding times
    for i in range(len(time_i)):
        
        if (time_i[i, 0] == valid_range1[0]) & (time_i[i, 1] == valid_range1[1]):
            ind_start = i
            s += 1
        
        if (time_i[i, 0] == valid_range2[0]) & (time_i[i, 1] == valid_range2[1]):
            ind_end = i
            s += 1
    
    if s == 1 or 2:
        data_slice = np.asarray(data_slice)[ind_start:ind_end+1,:,:]
        time_i = time_i[ind_start:ind_end+1,:]
    else:
        print('existing more than 2 indices for starting & end time stamp, please check.')
    
    # print(ind_start, ind_end, s)
    # print(data_slice.shape)
    
    Fill_value = f.variables[varnm]._FillValue
    print("Fill Value: ", Fill_value)
    # OUTPUT data/ timeo:
    data = data_slice.copy()
    data[data == Fill_value] = np.nan
    timeo = time_i.copy()

    print(np.asarray(data).shape)
    
    if varnm == 'solar_mon':
        
        data[data == 0.0] = np.nan  # polar night, incoming shortwave flux equal zeros, causes inf in 'albedo'
        
    if varnm == 'toa_sw_clr_c_mon':
        
        data[data == 0.0] = np.nan  # polar night, clear-sky shortwave flux equals  zeros, but incoming shortwave flux is not zeros, causes '0' in 'albedo' (while the incoming shorwave flux is also very small.)
    
    return np.array(data[:, :, ind_sort]), np.array(lat), np.array(lon2), timeo
    