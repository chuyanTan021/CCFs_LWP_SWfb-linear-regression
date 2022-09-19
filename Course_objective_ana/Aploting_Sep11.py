# This module is for storing some plotting functions.

import netCDF4 as nc

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import pandas as pd
import glob
from scipy.stats import *
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm

from scipy.optimize import curve_fit
import seaborn as sns
from useful_func_cy import *



def LWP_obs_trends(data_Array_actual, data_Array_predict, time_Array, lats, lons, data_type = '2', running_mean_window = 2):
    # -----------------
    # 'data_Array' is a ndarray for LWP data (monthly or annually, unbinned or binned);
    # 'data_type' gives the type info;
    # 'lats' & 'lons' are the latitude and longitude array of the data;
    # 'time_Array' is an array in shape (N, 3)(N is length of the first dimension of data_Array, which are the (year, mon, day) infomation of the data
    # -----------------
    fig1, ax1  = plt.subplots(1, 1, figsize =(8.1, 4.65))  # (18.2, 13.2)
    # ax1 = plt.axes()
    
    parameters = {'axes.labelsize': 16, 'legend.fontsize': 14,
              'axes.titlesize': 15, 'xtick.labelsize': 14, 'ytick.labelsize': 16}
    
    plt.rcParams.update(parameters)
    path_Plotting = '/glade/work/chuyan/Research/Cloud_CCFs_RMs/Course_objective_ana/plot_file/plots_Sep8_Observation_data/'
    
    if data_type == '1':  # input monthly --> output annually
        
        output_time = np.arange(0, time_Array.shape[0]//12, 1)
        output_actual = area_mean(annually_mean(data_Array_actual, time_Array, label = 'mon'), lats, lons) * 1000.
        output_predict = area_mean(annually_mean(data_Array_predict, time_Array, label = 'mon'), lats, lons) * 1000.
        plt.plot(output_time, output_actual, label = 'OBS LWP: ' + '5*5 annually variation', alpha = 1.0, linewidth= 2.40, linestyle = '--', c = 'green', zorder =1)
        plt.plot(output_time, output_predict, label = 'Predi OBS LWP: ' + '5*5 annually variation', alpha = 1.0, linewidth= 2.40, linestyle = '--', c = 'b', zorder =1)
        print(time_Array[0, 0])
        
        plt.xticks(output_time, (np.arange(time_Array[0, 0], time_Array[0, 0] + time_Array.shape[0]//12, 1)).astype(int), rotation = 45)
        plt.xlabel(' Time ')
        plt.ylabel('LWP '+r'$ [kg*m^{-2}]$')
        plt.title(" LWP over Times ")
    
        # plt.show()
        plt.savefig(path_Plotting + 'Trends_obs_predi_LiquidWaterPathyrvariation.jpg', bbox_inches = 'tight', dpi = 150)
    
    
    elif data_type == '2':  # input monthly --> output monthly
        
        output_time = np.arange(0, time_Array.shape[0], 1)
        output_actual = area_mean(data_Array_actual, lats, lons) *1000.
        output_predict = area_mean(data_Array_predict, lats, lons) *1000.
        plt.plot(output_time, output_actual, label = 'OBS LWP: ' + '5*5 monthly variation', alpha = 1.0, linewidth= 2.40, linestyle = '--', c = 'green', zorder =1)
        plt.plot(output_time, output_predict, label = 'Predi OBS LWP: ' + '5*5 monthly variation', alpha = 1.0, linewidth= 2.40, linestyle = '--', c = 'b', zorder =1)
        print(time_Array[0, 0])
        plt.xticks(output_time[0::12], (np.arange(time_Array[0, 0], time_Array[0, 0] + time_Array.shape[0]//12, 1)).astype(int), rotation = 45)
        plt.xlabel(' Time ')
        plt.ylabel('LWP '+r'$ [kg*m^{-2}]$')
        plt.title(" LWP over Times ")
    
        # plt.show()
        plt.savefig(path_Plotting + 'Trends_obs_predi_LiquidWaterPathmonthvariation.jpg', bbox_inches = 'tight', dpi = 150)
        
    
    elif data_type == '3':  # input monthly --> output running annually mean: e.g. 2yrs or 3yrs
        
        output_time = np.arange(0, time_Array.shape[0]//12, 1)
        df_actual = pd.DataFrame({'A': area_mean(data_Array_actual, lats, lons) * 1000.})
        output_actual = df_actual.rolling((12* running_mean_window), )
        output_predict = area_mean(annually_mean(data_Array_predict, time_Array, label = 'mon'), lats, lons) * 1000.
        plt.plot(output_time, output_actual, label = 'OBS LWP: ' + '5*5 annually variation', alpha = 1.0, linewidth= 2.40, linestyle = '--', c = 'green', zorder =1)
        plt.plot(output_time, output_predict, label = 'Predi OBS LWP: ' + '5*5 annually variation', alpha = 1.0, linewidth= 2.40, linestyle = '--', c = 'b', zorder =1)
        print(time_Array[0, 0])
        
        plt.xticks(output_time, (np.arange(time_Array[0, 0], time_Array[0, 0] + time_Array.shape[0]//12, 1)).astype(int), rotation = 45)
        plt.xlabel(' Time ')
        plt.ylabel('LWP '+r'$ [kg*m^{-2}]$')
        plt.title(" LWP over Times ")
    
        # plt.show()
        plt.savefig(path_Plotting + 'Trends_obs_predi_LiquidWaterPathyrvariation.jpg', bbox_inches = 'tight', dpi = 150)
    
    return None
