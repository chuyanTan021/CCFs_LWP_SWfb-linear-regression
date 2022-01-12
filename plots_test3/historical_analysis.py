# get the data we needed from read module:'get_LWPCMIP6', and do some data-processing for building the linear regression CCFs_Clouds models:
# transform data to annual-mean/ monthly-mean bin array or flattened array;
##  f;

import netCDF4
from numpy import *
import matplotlib.pyplot as plt
import xarray as xr
import PyNIO as Nio
import pandas as pd
import glob
from scipy.stats import *
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm

from area_mean import *
from binned_cyFunctions5 import *
from read_hs_file import read_var_mod


from get_LWPCMIP6data import *
from useful_func_cy import *

from run_simple_cmip6_pc import *


def historical_analysis(startyr, endyr, **model_data):
    rawdata_dict  = {}
    
    
    # get variable data
    if model_data['cmip'] == 'cmip6':
        
        inputVar_his  = get_historical(startyr, endyr, **model_data)
        
    else:
        print('not historical data')
        
    #..get the shapes of monthly data
    
    shape_lat = len(inputVar_his['lat'])
    shape_lon = len(inputVar_his['lon'])
    shape_time = len(inputVar_his['times'])
    
    print(shape_lat, shape_lon, shape_time)
    
    
    #..dimesnsions info
    lons        = array(inputVar_his['lon'])
    lats        = array(inputVar_his['lat'][:])
    
    levels      = array(inputVar_his['pres'])
    times     = array(inputVar_his['times'])
    
    
    #..choose lat from 40 -85 °S as the Southern-Ocean Regions
    lati0 = -40.
    latsi0 = min(range(len(lats)), key = lambda i: abs(lats[i] - lati0))
    lati1 = -85.
    latsi1 = min(range(len(lats)), key = lambda i: abs(lats[i] - lati1))
    print('lat index for 40.S; 85.S', latsi0, latsi1)
    
    
    shape_latSO =  (latsi0+1) - latsi1
    print(" shape of Southern Ocean latititude point= ", shape_latSO)
    
    
    #.. Variables: LWP, SST, Subsidence @ 500 mb
    LWP_his  = array(inputVar_his['clwvi']) - array(inputVar_his['clivi'])   #..units in kg m^-2
    
    SST_his  = array(inputVar_his['sfc_T'])
    
    Precip_his =  array(inputVar_his['P']) * (24.*60.*60.)   #..Precipitation. Convert the units from kg m^-2 s^-1 -> mm*day^-1
    print('historical period average Pr(mm/ day): ', nanmean(Precip_his))   #.. IPSL/abr2.80..  CNRM ESM2 1/abr 2.69.. CESM2/abr 2.74..
    Eva_his    =  array(inputVar_his['E']) * (24.*60.*60.)   #..evaporation, mm day^-1
    print('historical period average Evapor(mm/ day): ', nanmean(Eva_his))         #.. IPSL/abr2.50..  CNRM ESM2 1/abr 2.43.. CESM2/abr 2.43..
    
    MC_his  = Precip_his - Eva_his   #..Moisture Convergence calculated from abrupt4xCO2's P - E, Units in mm day^-1
    
    Twp_his  = array(inputVar_his['clwvi'])
    Iwp_his  = array(inputVar_his['clivi'])
    prw_his  = array(inputVar_his['prw'])
    
    print('historical period data shape in: ', Eva_his.shape )
    
    
    #..Meteological Subsidence  at 500 hPa, units in Pa s^-1:
    Subsidence_his =  array(inputVar_his['sub'])
    
    
    # put monthly data into Dictionary, stored: SST, SUB at 500mb, LWP and other variables
    dict0_var = { 'LWP': LWP_his, 'TWP': Twp_his, 'IWP': Iwp_his, 'PRW': prw_his, 'SST': SST_his, 'p_e': MC_his, 'SUB': Subsidence_his, 'lat': lats, 'lon':lons, 'times':times, 'pres':levels}
    
    #.. calced raw CCF(SST& SUB at 500) metrics with raw LWP, wvp in 'historical' period
    rawdata_dict['dict0_var'] = dict0_var
    
    
    
    # get the Annual-mean, Southern-Ocean region arrays
    
    datavar_nas = ['LWP', 'TWP', 'IWP', 'PRW', 'SST', 'SUB']   #..6 varisables (in the same shape)
    
    dict1_var_yr  = {}
    shape_yr  = shape_time//12

    
    dict1_var_yr = get_annually_dict_so(dict0_var, datavar_nas, shape_time, latsi1, latsi0, shape_lon)
    print("dict1_var_yr data shape in: ", dict1_var_yr['LWP_yr'].shape)
    
    #.. Annually-mean metrics
    rawdata_dict['dict1_var_yr'] = dict1_var_yr
    
    
    
    #..set are-mean range/ Grids' longitude, Latitude and SO's Latitude
    x_range  = arange(-180., 183, 5.)   #..logitude sequences edge: number:73
    s_range  = arange(-90., 90, 5.)  + 2.5   #..global-region latitude edge:(36)

    y_range  = arange(-85, -35., 5.) + 2.5   #..southern-ocaen latitude edge:10
    
    
    # Calc binned array('r': any resolution) for Annually variable: 
    
    lat_array  = lats[latsi1:latsi0+1]
    lon_array  =  lons
    lat_array1 =  lats
    
    dict1_yr_bin  = {}
    for b in range(len(datavar_nas)):

        dict1_yr_bin[datavar_nas[b]+'_yr_bin']   =   binned_cySouthOcean_anr(dict1_var_yr[datavar_nas[b]+'_yr'], lat_array , lon_array, 5)
    
    rawdata_dict['dict1_yr_bin']  = dict1_yr_bin
    
    # Calc binned array('r': any resolution) for Monthly variable:
    
    dict1_mon_bin  = {}
    for c in range(len(datavar_nas)):

        dict1_mon_bin[datavar_nas[c]+'_mon_bin'] =    binned_cySouthOcean_anr(dict0_var[datavar_nas[c]][:, latsi1:latsi0+1, :], lat_array, lon_array, 5)
    
    rawdata_dict['dict1_mon_bin']  = dict1_mon_bin
    
    
    return rawdata_dict




def pcoloranalysis(startyr, endyr, **model_data):
    ### For comparing and analysizing the LWP over SUB@500 v.s. SST for GCMs(each) AND obs DATA
    rawdata_dict = historical_analysis(startyr, endyr, **model_data)
    rawdata_dict2 = get_obs(startyr, endyr)
    
    
    
    ##. for Pcplor mesh plot: data array:'
    # GCM monthly unbinned data:
    # XX_ay_gcm  = rawdata_dict['dict0_var']['SUB'][ :, 5:54, :]
    # YY_ay_gcm  = rawdata_dict['dict0_var']['SST'][ :, 5:54, :]
    # PC_ay_gcm  =  rawdata_dict['dict0_var']['LWP'][:, 5:54,:] 

    # GCM annually unbinned data:
    # XX_ay_gcm  =  rawdata_dict['dict1_var_yr']['SUB_yr']
    # YY_ay_gcm  =  rawdata_dict['dict1_var_yr']['SST_yr']
    # PC_ay_gcm  =  rawdata_dict['dict1_var_yr']['LWP_yr']

    # GCM annually binned data:
    XX_ay_gcm  =  rawdata_dict['dict1_yr_bin']['SUB_yr_bin']
    YY_ay_gcm  =  rawdata_dict['dict1_yr_bin']['SST_yr_bin']
    PC_ay_gcm  =  rawdata_dict['dict1_yr_bin']['LWP_yr_bin']



    # OBS(MAC:LWP, ERA5: CCFS) monthly unbinned data:
    # XX_ay_obs  =  rawdata_dict2['dict0_var']['SUB']
    # YY_ay_obs  =  rawdata_dict2['dict0_var']['SST']
    # PC_ay_obs  =  rawdata_dict2['dict0_var']['LWP_mac']

    # OBS(MAC:LWP, ERA5: CCFS) annually unbinned data:
    # XX_ay_obs  =  rawdata_dict2['dict1_era_yr']['SUB_yr']
    # YY_ay_obs  =  rawdata_dict2['dict1_era_yr']['SST_yr']
    # PC_ay_obs  =  rawdata_dict2['dict1_mac_yr']['LWP_mac_yr']

    # OBS(MAC:LWP, ERA5: CCFS) annually binned data:
    # XX_ay_obs  =  rawdata_dict2['dict1_era_yr_bin']['SUB_yr_bin']
    # YY_ay_obs  =  rawdata_dict2['dict1_era_yr_bin']['SST_yr_bin']
    PC_ay_obs  =  rawdata_dict2['dict1_mac_yr_bin']['LWP_mac_yr_bin']

    XX_ay_obs  =  rawdata_dict2['dict1_era_yr_bin']['SUB_yr_bin_unmasked']
    YY_ay_obs  =  rawdata_dict2['dict1_era_yr_bin']['SST_yr_bin_unmasked']

    
    #..  Plotting   ..
    y_gcm = linspace(nanpercentile(YY_ay_gcm, 1), nanpercentile(YY_ay_gcm, 99), 22)
    x_gcm = linspace(nanpercentile(XX_ay_gcm, 5), nanpercentile(XX_ay_gcm, 95), 15)
    print(x_gcm, y_gcm)


    y_obs = linspace(nanpercentile(YY_ay_obs, 1), nanpercentile(YY_ay_obs, 99), 22)
    x_obs = linspace(nanpercentile(XX_ay_obs, 5), nanpercentile(XX_ay_obs, 95), 15)
    #print(x_obs, y_obs)


    LWP_bin_Tskin_sub_gcm , count_number_Aa_gcm =  binned_skinTSUB500(XX_ay_gcm, YY_ay_gcm, PC_ay_gcm , y_gcm, x_gcm)

    LWP_bin_Tskin_sub_obs , count_number_Aa_obs =  binned_skinTSUB500(XX_ay_obs, YY_ay_obs, PC_ay_obs , y_obs, x_obs)
    ##. fro plotting Pcolormesh 
    X_gcm, Y_gcm  = meshgrid(x_gcm, y_gcm)
    X_obs, Y_obs  = meshgrid(x_obs, y_obs)



    #..defined a proper LWP ticks within its range
    levels_value  = linspace(nanpercentile(PC_ay_gcm, 22), nanpercentile(PC_ay_gcm, 99.925), 40, dtype=float)
    
    #levels_value = arange(0.02, 0.18, 0.004)
    levels_count = linspace(5, 300, 60, dtype=int)


    #..pick the desired colormap
    cmap = plt.get_cmap('YlOrRd')

    norm1 = BoundaryNorm(levels_value, ncolors= cmap.N, extend='both')
    norm2 = BoundaryNorm(levels_count, ncolors= cmap.N, extend='both')



    #.. what will the pcolormesh plot looks like?
    fig, ax  = plt.subplots(2, 2, figsize =(18.5, 13.5))

    print(ax)

    im1  = ax[0, 0].pcolormesh(x_gcm, y_gcm, LWP_bin_Tskin_sub_gcm, cmap=cmap, norm= norm1) #..anmean_LWP_bin_Tskew_wvp..LWP_bin_Tskin_sub

    ax[0,0].set_title("(a) exp 'historical'("+str(startyr)+'-'+str(endyr)+")  GCM data: "+model_data['modn'], loc='left', fontsize = 11)
    ax[0,0].set_xlabel('omega 500 mb'+ r'$(Pa\ s^{-1})$',   fontsize =12)
    ax[0,0].set_ylabel('SST ' + r'$(K)$',  fontsize=12)
    fig.colorbar(im1, ax = ax[0,0], label="Liquid Water Path " + r"$(kg\ m^{-2}}$)")

    ax[1,0].set_title("(b) exp 'historical'("+str(startyr)+'-'+str(endyr)+")  GCM data: "+model_data['modn'], loc='left', fontsize = 11)
    ax[1,0].set_xlabel('omega 500 mb'+ r'$(Pa\ s^{-1})$',   fontsize =12)
    ax[1,0].set_ylabel('SST ' + r'$(K)$',  fontsize=12)
    im2  = ax[1, 0].pcolormesh(x_gcm, y_gcm, count_number_Aa_gcm, cmap=cmap, norm= norm2)
    fig.colorbar(im2, ax = ax[1,0], label="# of points")


    ax[0,1].set_title("(c) OBS("+str(startyr)+'-'+str(endyr)+") data: MAC_LWP, ERA5_CCFS", loc='left', fontsize = 11)
    ax[0,1].set_xlabel('omega 500 mb'+ r'$(Pa\ s^{-1})$',   fontsize =12)
    ax[0,1].set_ylabel('SST ' + r'$(K)$',  fontsize=10)
    im3  = ax[0,1].pcolormesh(x_obs, y_obs, LWP_bin_Tskin_sub_obs, cmap=cmap, norm= norm1)
    fig.colorbar(im3, ax = ax[0, 1], label="Liquid Water Path " + r"$(kg\ m^{-2}$)" )


    ax[1,1].set_title("(d) OBS("+str(startyr)+'-'+str(endyr)+") data: MAC_LWP, ERA5_CCFS", loc='left', fontsize = 11)
    ax[1,1].set_xlabel('omega 500 mb'+ r'$(Pa\ s^{-1})$',   fontsize =12)
    ax[1,1].set_ylabel('SST ' + r'$(K)$',  fontsize=12)
    im4  = ax[1, 1].pcolormesh(x_obs, y_obs, count_number_Aa_obs, cmap=cmap, norm= norm2)
    fig.colorbar(im4, ax = ax[1,1], label="# of points")
    
    
    '''
    WD = '/glade/work/chuyan/Research/Cloud_CCFs_RMs/plots_test3/'
    
    plt.savefig(WD + 'Pcolor_'+ model_data['modn']+'-_SSTSUB_LWP|annuallybinned')
    '''
    ##. the 2-d indice of maxinum count number of count array
    max_indice_count = unravel_index(nanargmax(count_number_Aa_gcm, axis=None), count_number_Aa_gcm.shape)

    #print(LWP_bin_Tskin_sub_gcm[:, max_indice_count[1]], nanargmax(LWP_bin_Tskin_sub_gcm[:, max_indice_count[1]]))
    ##. the y axis indice of largest mean-binned LWP value of LWP array
    max_index_meanLWP   =nanargmax(LWP_bin_Tskin_sub_gcm[:, max_indice_count[1]])

    #.  calc the # of points above/below the 'max_index_meanLWP' and their ratio:
    C_ab = np.nansum(count_number_Aa_gcm[max_index_meanLWP:, :])
    C_be = np.nansum(count_number_Aa_gcm[0:max_index_meanLWP, :])

    ratios_gcm  = float(C_ab/C_be)

    print('the ratio of # of points above the transfer SST to # of pointa lower than the transfer_sst : ', ratios_gcm)

    #.. print the 'transfermation' point(mean):
    if max_index_meanLWP <= len(y_gcm)-3:
        Tr_sst = (y_gcm[max_index_meanLWP+1] + y_gcm[max_index_meanLWP+2])/ 2.
    else:
        Tr_sst  = 260.0
    
    
    print("determined Tr_sst: ", Tr_sst)
    
    
    
    #.. for loop to see which point more close to 60.%
    ratios_gcm = full((len(y_gcm)-2), 0.00)
    for i_in in arange(len(y_gcm)-2):

        C_ab = np.nansum(count_number_Aa_gcm[i_in:, :])
        C_be = np.nansum(count_number_Aa_gcm[0:i_in,:])

        ratios_gcm[i_in]  = float(C_ab/C_be)

    # find the point index who has (above #)/(below #) most closer to '..':
    ratio_tr1 = 0.75

    ind1 =  min(range(len(ratios_gcm)),key=lambda i: abs(ratios_gcm[i] - ratio_tr1))


    TR_sst=((y_gcm[ind1] + y_gcm[ind1+1])/2.)

    print("determined TR_sst: ", TR_sst)
    
    
    
    calc_LRM_metrics(round(Tr_sst, 2), **model_data)
    
    return None