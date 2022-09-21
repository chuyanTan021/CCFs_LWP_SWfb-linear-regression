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



def stats_metrics_Visualization(modn = 'OBS'):
    
    WD = '/glade/scratch/chuyan/obs_output/'
    path6 = '/glade/work/chuyan/Research/Cloud_CCFs_RMs/Course_objective_ana/plot_file/plots_Sep8_Observation_data/'
    folder =  glob.glob(WD + modn + '__' + 'STAT_pi+abr_'+'22x_31y_Sep11th'+ '.npz')
    print(folder)
    
    output_ARRAY = np.load(folder[0], allow_pickle=True)  # str(TR_sst)
    x_gcm = np.asarray(output_ARRAY['bound_x'])
    y_gcm = np.asarray(output_ARRAY['bound_y'])
    output_stat1 = output_ARRAY['stats_2']
    output_stat2 = output_ARRAY['stats_4']

    fig3, ax3  = plt.subplots(1, 2, figsize = (15.2, 10.4))  #(16.2, 9.3))

    #..defined a proper LWP ticks within its range
    p10_valuespace1 = np.nanpercentile(output_stat1, 25.) - np.nanpercentile(output_stat1, 15.)
    levels_value1 = np.linspace(np.nanpercentile(output_stat1, 1.5)-p10_valuespace1, np.nanpercentile(output_stat1, 97) + p10_valuespace1, 164)  # arange(0.368, 0.534, 0.002) 
    # print(levels_value1)
    p10_valuespace2 = np.nanpercentile(output_stat2, 25.) - np.nanpercentile(output_stat2, 15.)
    levels_value2 = np.linspace(np.nanpercentile(output_stat2, 1.5)-p10_valuespace2, np.nanpercentile(output_stat2, 99.5) + p10_valuespace2, 164)
    # print(levels_value2)
    
    #..print(linspace(nanpercentile(output_stat, 1.5), nanpercentile(output_stat, 99.5), 164))
    #..pick the desired colormap
    cmap  = plt.get_cmap('YlOrRd') 
    cmap_2 = plt.get_cmap('viridis_r')   # 'YlOrRd'
    norm1 = BoundaryNorm(levels_value1, ncolors= cmap.N, extend='both')
    norm2 = BoundaryNorm(levels_value2, ncolors= cmap_2.N, extend='both')

    im1  = ax3[0].pcolormesh(x_gcm, y_gcm, np.array(output_stat1), cmap=cmap, norm= norm1)   #..anmean_LWP_bin_Tskew_wvp..LWP_bin_Tskin_sub
    ax3[0].set_xlabel('SUB at 500mb, '+ r'$Pa s^{-1}$', fontsize= 15)
    ax3[0].set_ylabel('SST, ' + 'K', fontsize= 15)
    ax3[0].set_title(r"$(a)\ RMSE:(LWP_{predi}|{period1} - LWP_{OBS}|{period1})$", loc='left', fontsize = 11)
    
    im2  = ax3[1].pcolormesh(x_gcm, y_gcm, np.asarray(output_stat2), cmap=cmap_2, norm= norm2)
    ax3[1].set_xlabel('SUB at 500mb, '+ r'$Pa s^{-1}$', fontsize= 15)
    ax3[1].set_ylabel('SST, ' + 'K', fontsize= 15)
    ax3[1].set_title(r"$(b)\ R^{2}(LWP_{predi}|{period2}\ with\ LWP{OBS}|{period2})$", loc='left', fontsize = 11)
    # ax3.set_title("exp 'abrupt-4xCO2' GCM: BCCESM1 predict R_2", loc='left', fontsize = 11)
    
    fig3.colorbar(im1, ax = ax3[0], label= r"$(kg\ m^{-2})$")
    fig3.colorbar(im2, ax = ax3[1], label= r"$ Coefficient of Determination$")

    
    plt.xlabel('SUB at 500mb, '+ r'$Pa s^{-1}$', fontsize= 15)
    plt.ylabel('SST, ' + 'K', fontsize= 15)
    plt.suptitle(modn+ " Bias metrics of predicting LWP", fontsize =18 )

    # plt.legend(loc='upper right',  fontsize= 12)

    plt.savefig(path6 + "Observational_bias.jpg" )
    return None



def LWP_obs_trends_2(data_Array_actual, data_Array_predict, time_Array, lats, lons, data_type = '2', running_mean_window = 2):
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
        output_actual = area_mean(annually_mean(data_Array_actual, time_Array, label = 'mon'), lats, lons)  # * 1000.
        output_predict = area_mean(annually_mean(data_Array_predict, time_Array, label = 'mon'), lats, lons)  # * 1000.
        plt.plot(output_time, output_actual, label = 'OBS LWP: ' + '5*5 annually variation', alpha = 1.0, linewidth= 2.40, linestyle = '--', c = 'green', zorder =1)
        # plt.plot(output_time, output_predict, label = 'Predi OBS LWP: ' + '5*5 annually variation', alpha = 1.0, linewidth= 2.40, linestyle = '--', c = 'b', zorder =1)
        print(time_Array[0, 0])
        
        plt.xticks(output_time, (np.arange(time_Array[0, 0], time_Array[0, 0] + time_Array.shape[0]//12, 1)).astype(int), rotation = 45)
        plt.xlabel(' Time ')
        plt.ylabel('p - e '+r'$ [kg*m^{-2}]$')
        plt.title(" p - e over Times ")
    
        # plt.show()
        plt.savefig(path_Plotting + 'Trends_obs_predi_P-E(1992_2000).jpg', bbox_inches = 'tight', dpi = 150)
    
    
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
        
        output_time = np.arange(0, time_Array.shape[0], 1)
        df_actual = pd.DataFrame({'A': area_mean(data_Array_actual, lats, lons)})  # * 1000.
        output_actual = df_actual.rolling((12* running_mean_window + 1), min_periods = 1, center = True).mean()
        df_predict = pd.DataFrame({'B': area_mean(data_Array_predict, lats, lons)})  # * 1000.
        output_predict = df_predict.rolling((12* running_mean_window + 1), min_periods = 1, center = True).mean()
        
        plt.plot(output_time, output_actual, label = 'OBS LWP: ' + '5*5' + str(running_mean_window) + ' yrs running-mean variation', alpha = 1.0, linewidth= 2.40, linestyle = '--', c = 'green', zorder =1)
        # plt.plot(output_time, output_predict, label = 'Predi OBS LWP: ' + '5*5' + str(running_mean_window) + ' yrs running-mean variation', alpha = 1.0, linewidth= 2.40, linestyle = '--', c = 'b', zorder =1)
        print(time_Array_predict[0, 0])
        
        plt.xticks(output_time[0::12], (np.arange(time_Array[0, 0], time_Array[0, 0] + output_time.shape[0]//12, 1)).astype(int), rotation = 45)
        plt.xlabel(' Time ')
        plt.ylabel('SUB '+r'$ [Pa s^-1]$')  # kg*m^{-2}
        plt.title(" SUB over Times ")
    
        # plt.show()
        plt.savefig(path_Plotting + 'Trends_obs_predict_SUB'+str(running_mean_window)+'yrsrunningvariation.jpg', bbox_inches = 'tight', dpi = 150)
        
    return None



def LWP_obs_trends(data_Array_actual_predict, data_Array_predict_predict, data_Array_actual_training, data_Array_predict_training, time_Array_predict, time_Array_training, lats, lons, data_type = '2', running_mean_window = 2):
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
        
        output_time = np.arange(0, time_Array_training.shape[0] + time_Array_predict.shape[0], 1)
        df_actual = pd.DataFrame({'A': area_mean(np.append(data_Array_actual_predict,data_Array_actual_training, axis = 0), lats, lons)}) * 1000.
        output_actual = df_actual.rolling((12* running_mean_window + 1), min_periods = 1, center = True).mean()
        df_predict = pd.DataFrame({'B': area_mean(np.append(data_Array_predict_predict, data_Array_predict_training, axis = 0), lats, lons)}) * 1000.
        output_predict = df_predict.rolling((12* running_mean_window + 1), min_periods = 1, center = True).mean()
        
        plt.plot(output_time, output_actual, label = 'OBS LWP: ' + '5*5' + str(running_mean_window) + ' yrs running-mean variation', alpha = 1.0, linewidth= 2.40, linestyle = '--', c = 'green', zorder =1)
        plt.plot(output_time, output_predict, label = 'Predi OBS LWP: ' + '5*5' + str(running_mean_window) + ' yrs running-mean variation', alpha = 1.0, linewidth= 2.40, linestyle = '--', c = 'b', zorder =1)
        ax1.axvline(time_Array_predict.shape[0], linestyle = '--', linewidth = 2.0, c = 'k')
        print(time_Array_predict[0, 0])
        
        print(np.append(np.arange(time_Array_predict[0, 0], time_Array_predict[0, 0] + time_Array_predict.shape[0]//12, 1) , np.arange(time_Array_training[0, 0], time_Array_training[0, 0] + time_Array_training.shape[0]//12, 1)))
        plt.xticks(output_time[0::12], (np.append(np.arange(time_Array_predict[0, 0], time_Array_predict[0, 0] + time_Array_predict.shape[0]//12, 1) , np.arange(time_Array_training[0, 0], time_Array_training[0, 0] + time_Array_training.shape[0]//12, 1))).astype(int), rotation = 45)
        plt.xlabel(' Time ')
        plt.ylabel('LWP '+r'$ [kg m^{-2}]$')  # kg*m^{-2}
        plt.title(" LWP over Times ")
    
        # plt.show()
        plt.savefig(path_Plotting + 'Trends_obs_predict+train_2_LWP'+str(running_mean_window)+'yrsrunningvariation.jpg', bbox_inches = 'tight', dpi = 150)
    
    return None