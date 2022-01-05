## get data :read CMIP6 monthly data for lwp AND cloud controlling factors, and add a function on Jan,1st to read Observational data for lwp and CCFs:

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
from useful_func_cy import *

from read_hs_file import read_var_mod



def get_LWPCMIP6(modn='IPSL-CM6A-LR', consort='IPSL', cmip='cmip6', exper='', ensmem='r1i1p1f1', gg='gr', typevar='Amon'):
    #if exper == 'historical':
    #    amip = False
    #else:
    #    amip = True
    #print(modn)
    #. modn='IPSL-CM5A-LR'; consort='IPSL'; cmip='cmip5'; exper='amip'; ensmem='r1i1p1'; gg=''; typevar='Amon'
    if exper == 'historical':
        time1 = [1950, 1, 1]
        time2 = [2014, 12, 30]
    if exper == 'amip':
        time1 = [1950, 1, 1]
        time2 = [2007, 12, 30]
    
    
    #..abrupt4xCO2
    exper = 'abrupt-4xCO2'
    if exper == 'abrupt-4xCO2':
        
        TEST1_time= read_var_mod(modn=modn,consort=consort,varnm='ps',cmip=cmip,exper=exper,ensmem=ensmem,gg=gg,typevar=typevar,time1=[1,1,1],time2=[ 3349, 12, 31])[-1]
        time1=[int(min(TEST1_time[:,0])),1,1]
        time2=[int(min(TEST1_time[:,0]))+299, 12,31]
        print("retrieve time: ", time1, time2)
        
        
    sfc_T_abr       = read_var_mod(modn=modn, consort=consort, varnm='ts', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= time1, time2= time2)[0]

    T_700_alevs_abr,Pres_abr,lat_abr,lon_abr,times_abr = read_var_mod(modn=modn, consort=consort, varnm='ta', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=True, time1= time1, time2= time2)
    T_700_abr = T_700_alevs_abr[:, 3,:,:]   #..700 hPa levels

    sfc_P_abr       = read_var_mod(modn=modn, consort=consort, varnm='ps', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= time1, time2= time2)[0]

    sub_abr_alevs =  read_var_mod(modn=modn, consort=consort, varnm='wap', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=True, time1= time1, time2= time2)[0]
    sub_abr          =  sub_abr_alevs[:, 5,:,:]
    #..500mb downward motions

    clivi_abr       = read_var_mod(modn=modn, consort=consort, varnm='clivi', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= time1, time2= time2)[0]
    clwvi_abr       = read_var_mod(modn=modn, consort=consort, varnm='clwvi', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= time1, time2= time2)[0]
    
    tas_abr         = read_var_mod(modn=modn, consort=consort, varnm='tas', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= time1, time2= time2)[0]

    P_abr           = read_var_mod(modn=modn, consort=consort, varnm='pr', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= time1, time2= time2)[0]

    E_abr           = read_var_mod(modn=modn, consort=consort, varnm='evspsbl', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= time1, time2= time2)[0]
    
    prw_abr         = read_var_mod(modn=modn, consort=consort, varnm='prw', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= time1, time2= time2)[0]

    #print(sfc_T_abr.shape)
    #..1800 months =150 yrs for CESM2:abrupt experiemnt    
    inputVar_abr = {'sfc_T': sfc_T_abr, 'T_700': T_700_abr, 'sfc_P': sfc_P_abr, 'sub': sub_abr, 'clivi': clivi_abr, 
                    'clwvi':clwvi_abr, 'tas': tas_abr, 'P': P_abr, 'E': E_abr, 'prw': prw_abr, 'pres': Pres_abr, 'lat':lat_abr, 'lon':lon_abr, 'times':times_abr}
    
    
    
    #..pi-Control
    exper = 'piControl'
    if exper == 'piControl':
        
        TEST2_time= read_var_mod(modn=modn,consort=consort,varnm='ps',cmip=cmip,exper=exper,ensmem=ensmem,gg=gg,typevar=typevar,time1=[1,1,1], time2=[8000, 12,31])[-1]
        timep1=[int(min(TEST2_time[:,0])),1,1]   #..max-799
        timep2=[int(min(TEST2_time[:,0]))+98, 12,31]  #..max-750
        
        print ("retrieve time: ", timep1, timep2)
        
    
    sfc_T       = read_var_mod(modn= modn, consort= consort, varnm='ts', cmip=cmip, exper= exper, ensmem=ensmem, typevar='Amon', gg=gg, read_p=False, time1= timep1, time2= timep2)[0]

    T_700_alevs,Pres_pi,lat_pi,lon_pi,times_pi =  read_var_mod(modn= modn, consort= consort, varnm='ta', cmip=cmip, exper= exper, ensmem=ensmem, typevar='Amon', gg=gg, read_p=True, time1= timep1, time2= timep2)
    T_700       = T_700_alevs[:, 3,:,:]

    sfc_P       = read_var_mod(modn= modn, consort= consort, varnm='ps', cmip=cmip, exper= exper, ensmem=ensmem, typevar='Amon', gg=gg, read_p=False, time1= timep1, time2= timep2)[0]   
    #..sea surface Pressure, Units in Pa

    sub_alevs    = read_var_mod(modn= modn, consort= consort, varnm='wap', cmip=cmip, exper= exper, ensmem=ensmem, typevar='Amon', gg=gg, read_p=True, time1= timep1, time2= timep2)[0]
    sub          =  sub_alevs[:, 5,:,:]
    #..500mb downward motion

    clivi       = read_var_mod(modn= modn, consort= consort, varnm='clivi', cmip=cmip, exper= exper, ensmem=ensmem, typevar='Amon', gg=gg, read_p=False, time1= timep1, time2= timep2)[0]
    #..ICE WATER PATH, Units in kg m^-2
    clwvi       = read_var_mod(modn= modn, consort= consort, varnm='clwvi', cmip=cmip, exper= exper, ensmem=ensmem, typevar='Amon', gg=gg, read_p=False, time1= timep1, time2= timep2)[0]
    
    tas         = read_var_mod(modn= modn, consort= consort, varnm='tas', cmip=cmip, exper= exper, ensmem=ensmem, typevar='Amon', gg=gg, read_p=False, time1= timep1, time2= timep2)[0]
    #..2-m air Temperature, for 'gmt'

    P           = read_var_mod(modn= modn, consort= consort, varnm='pr', cmip=cmip, exper= exper, ensmem=ensmem, typevar='Amon', gg=gg, read_p=False, time1= timep1, time2= timep2)[0]
    #..Precipitation, Units in kg m^-2 s^-1 = mm *s^-1
    E           = read_var_mod(modn= modn, consort= consort, varnm='evspsbl', cmip=cmip, exper= exper, ensmem=ensmem, typevar='Amon', gg=gg, read_p=False, time1= timep1, time2= timep2)[0]
    #..Evaporations, Units also in kg m^-2 s^-1 = mm *s^-1
    
    prw_pi      =read_var_mod(modn= modn, consort= consort, varnm='prw', cmip=cmip, exper= exper, ensmem=ensmem, typevar='Amon', gg=gg, read_p=False, time1= timep1, time2= timep2)[0]

    #print(sfc_T.shape)
    #..6000 months =99 yrs for CESM2 piControl experiment
    
    inputVar_pi = {'sfc_T': sfc_T, 'T_700': T_700, 'sfc_P': sfc_P, 'sub': sub, 'clivi': clivi, 
                 'clwvi': clwvi, 'tas': tas, 'P': P, 'E': E, 'prw': prw_pi, 'pres':Pres_pi, 'lat':lat_pi, 'lon':lon_pi, 'times': times_pi}
    '''
    TS = read_var_mod(modn=modn,consort=consort,varnm='ts',cmip=cmip,exper=exper,ensmem=ensmem,gg=gg,typevar=typevar,time1=time1,time2=time2)
    '''
    #return {'SST0': TS[0], 'SST1': TSf[0], 'LWP0': LWP0, 'LWP1': LWP1, 'lat': IWP[2][:], 'lon': IWP[3][:], 'time0': IWP[-1], 'time1': IWPf[-1], 'IWP0': IWP[0], 'IWP1': IWPf[0], 'P0': P[0], 'E0': E[0], 'P1': Pf[0], 'E1': Ef[0], "WVP0": wvp[0], 'WVP1': wvpf[0], 'SI': SI, 'U100': U10_0, 'U101': U10_f, 'TAS0': TAS[0], 'TAS1': TASf[0], 'TA0': TA_0, 'TA1': TA_1, 'SW0': SW[0], 'SW0c': SWc[0], 'SW1': SWf[0], 'SW1c': SWfc[0],'TS0':TS[0],'TS1':TSf[0]}   #,'WAP0':WAP_0,'WAP1':WAP_1,'LTS0':LTS,'LTS1':LTSf}
    
    return inputVar_pi, inputVar_abr





def get_historical(startyr, endyr, modn='IPSL-CM6A-LR', consort='IPSL', cmip='cmip6', exper='historical', ensmem='r1i1p1f1', gg='gr', typevar='Amon'):
    #if exper == 'historical':
    #    amip = False
    #else:
    #    amip = True
    #print(modn)
    #. modn='IPSL-CM5A-LR'; consort='IPSL'; cmip='cmip5'; exper='amip'; ensmem='r1i1p1'; gg=''; typevar='Amon'
    
    if exper == 'historical':
        time1 = [int(startyr), 1, 1]
        time2 = [int(endyr), 12, 30]
    
    if exper == 'historical':
        TEST1_time= read_var_mod(modn=modn,consort=consort,varnm='ts',cmip=cmip, exper=exper, ensmem=ensmem, gg=gg, typevar= typevar, time1=[1850,1,1], time2=[2022, 12, 30])[-1]
        timemin=[int(min(TEST1_time[:,0])),1,1 ]
        timemax=[int(max(TEST1_time[:,0])),12,30]
    
    print("time range of data: ", timemin, timemax)
        
    sfc_T       = read_var_mod(modn=modn, consort=consort, varnm='ts', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= time1, time2= time2)[0]

    #T_700_alevs,Pres_abr,lat_abr,lon_abr,times_abr = read_var_mod(modn=modn, consort=consort, varnm='ta', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=True, time1= time1, time2= time2)
    #T_700 = T_700_alevs_abr[:, 3,:,:]   #..700 hPa levels

    #sfc_P    =  read_var_mod(modn=modn, consort=consort, varnm='ps', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= time1, time2= time2)[0]

    sub_alevs, Pres, lat, lon, times =  read_var_mod(modn=modn, consort=consort, varnm='wap', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=True, time1= time1, time2= time2)
    sub       =  sub_alevs[:, 5, :, :]
    #..500mb downward motions
    
    clivi       = read_var_mod(modn=modn, consort=consort, varnm='clivi', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= time1, time2= time2)[0]
    clwvi       = read_var_mod(modn=modn, consort=consort, varnm='clwvi', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= time1, time2= time2)[0]
    
    
    #..temporarily didn't use them
    P           = read_var_mod(modn= modn, consort= consort, varnm='pr', cmip=cmip, exper= exper, ensmem=ensmem, typevar='Amon', gg=gg, read_p=False, time1= time1, time2= time2)[0]
    #..Precipitation, Units in kg m^-2 s^-1 = mm *s^-1
    E           = read_var_mod(modn= modn, consort= consort, varnm='evspsbl', cmip=cmip, exper= exper, ensmem=ensmem, typevar='Amon', gg=gg, read_p=False, time1= time1, time2= time2)[0]
    #..Evaporations, Units also in kg m^-2 s^-1 = mm *s^-1
    
    prw   = read_var_mod(modn= modn, consort= consort, varnm='prw', cmip=cmip, exper= exper, ensmem=ensmem, typevar='Amon', gg=gg, read_p=False, time1= time1, time2= time2)[0]
    
    
    inputVar_historicalGCM = { 'sfc_T': sfc_T, 'sub': sub , 'clivi': clivi, 'clwvi': clwvi, 'P': P , 'E': E , 'prw': prw, 'pres': Pres, 'lat': lat, 'lon': lon, 'times': times}
                 
    return inputVar_historicalGCM





def get_obs(time_start, time_end):
    rawdata_dict2 = {}
    #.. retrieve MAC_nasa data for "LWP"

    fn1_mac = '/glade/work/chuyan/Course_CliScience/maclwp_cloudlwpave_'
    fn2_mac = '/glade/work/chuyan/Course_CliScience/maclwp_totallwpave_'

    tt = arange(int(time_start), int(time_end)+1)
    print(tt)

    data_LWP_mac = []
    data_TWP_mac = []

    for t in arange(len(tt)):
        #..append 'LWP' & 'total water path = LWP + rain water' for MAC data from 1992 to 2014, monthly    
        data_LWP_mac.append(xr.open_dataset(fn1_mac+ str(tt[t])+ '_v1.nc4', decode_times=False)['cloudlwp'].values)

        data_TWP_mac.append(xr.open_dataset(fn2_mac+ str(tt[t])+ '_v1.nc4', decode_times=False)['totallwp'].values)

    LWP_mac =  array(data_LWP_mac)
    TWP_mac =  array(data_TWP_mac)

    dataout_LWP_MAC  =   concatenate(LWP_mac, axis=0)
    dataout_TWP_MAC  =   concatenate(TWP_mac, axis=0)

    ##. Convert units from m (/m^3)--> kg m-2
    dataout_LWP_MAC = dataout_LWP_MAC / 1000.
    dataout_TWP_MAC = dataout_TWP_MAC / 1000.

    #..Test data
    mac_lwp_datayr14  =  xr.open_dataset('/glade/work/chuyan/Course_CliScience/maclwp_cloudlwpave_2014_v1.nc4', decode_times=False)
    #..print(array(mac_lwp_datayr14['cloudlwp'].values).shape)
    
    

    #..define dimension/coordinates for MAC 23-yrs-data

    lat_mac =  array(mac_lwp_datayr14.coords['lat'])
    lon_mac =  array(mac_lwp_datayr14.coords['lon'])
    times_mac = arange(str(time_start)+ '-01-01', str(int(time_end)+1)+ '-01-01', dtype='datetime64[M]')

    #.print(times_mac)
    
    #.. from 40 S ~ 85 S
    lat0  = -85.
    latmac0 = min(range(len(lat_mac)), key=lambda i:abs(lat_mac[i]-lat0))
    lat1  = -40.
    latmac1 = min(range(len(lat_mac)), key=lambda i:abs(lat_mac[i]-lat1))
    #..print(latmac0, latmac1)

    #.. lat dimension and data_LWP & data_TWP from 40 S ~ 85 S
    lat_mac   =  lat_mac[latmac0:latmac1+1]
    print('SO lat_mac shape in: ', lat_mac.shape)

    LWP_MAC  =  dataout_LWP_MAC[:, latmac0:latmac1+1, :]   #.. Units in kg m-2
    TWP_MAC  =  dataout_TWP_MAC[:, latmac0:latmac1+1, :]   # 25 yrs Monthly 40~85 1*1 degree data, shapes in (300, 46, 360)

    print("mean MAC LWP value: ", nanmean(LWP_MAC), r'in kg*m_{-2}')
    print("MAC_nasa data shape in: ", LWP_MAC.shape )
    
    
    #.. retrieve ERA-5 data for "Cloud Controlling factors"
    f_era5_singlele    =  xr.open_dataset('/glade/work/chuyan/Course_CliScience/era5_monthly_single_19922016.grib', engine='pynio')
    f_era5_pressurele  =   xr.open_dataset('/glade/work/chuyan/Course_CliScience/era5_monthly_pressure_19922016.grib', engine='pynio')
    
    timea = (int(time_start) - 1992) * 12
    timeb = ((int(time_end)+1)  - 1992) * 12
    lwp   =  f_era5_singlele['TCLW_GDS0_SFC_S123'][timea:timeb, ::-1,:]   #..Units in kg m^-2
    iwp   =  f_era5_singlele['TCIW_GDS0_SFC_S123'][timea:timeb, ::-1,:]   #..Units in kg m^-2
    ps   =  f_era5_singlele['SP_GDS0_SFC_S123'][timea:timeb, ::-1,:]      #..Units in 
    prw  =   f_era5_singlele['TCWV_GDS0_SFC_S123'][timea:timeb, ::-1,:]   #..Units in kg m^-2 , shape in (721, 1440, 300)
    tas  =   f_era5_singlele['2T_GDS0_SFC_S123'][timea:timeb, ::-1,:]    #.. 2-m surface -air Temp, K
    E     =  f_era5_singlele['E_GDS0_SFC_S130'][timea:timeb, ::-1,:]   #..Evaporation, in m /d
    MC    =  f_era5_singlele['VIMD_GDS0_SFC_S130'][timea:timeb, ::-1,:]  #..Moisture convergence, not in flux, units in kg m^-2
    P     =  f_era5_singlele['TP_GDS0_SFC_S130'][timea:timeb, ::-1,:]     #。。Precip
    ts    =  f_era5_singlele['SKT_GDS0_SFC_S123'][timea:timeb, ::-1,:]   #..surface temperatur, or skin Temperture, in K

    ta   =   f_era5_pressurele['T_GDS0_ISBL_S123'][timea:timeb, :,::-1,:]
    T_700 = ta[:, 4, :, :]                           #..700mb Temperature, units in K
    wap  =   f_era5_pressurele['W_GDS0_ISBL_S123'][timea:timeb, 2,::-1,:]   #..500mb Subsidence, Pa s**-1
    
    
    lat_era1  = array(f_era5_singlele.coords['g0_lat_1'][::-1])
    lon_era2  = array(f_era5_singlele.coords['g0_lon_2'])

    lat_era2  = array(f_era5_pressurele.coords['g0_lat_2'][::-1])
    lon_era3  = array(f_era5_pressurele.coords['g0_lon_3'])
    #.. print(lat_era1, lat_era2)
    
    
    #.. from 40 S ~ 85 S
    lat0  = -85.
    late0 = min(range(len(lat_era1)), key=lambda i:abs(lat_era1[i]-lat0))
    lat1  = -40.
    late1 = min(range(len(lat_era1)), key=lambda i:abs(lat_era1[i]-lat1))
    lat_era   =   lat_era1[late0:late1+1]
    lon_era   =   lon_era2
    print('SO lat_era shape in', lat_era.shape)

    
    #.. dataout_ERA5 from 40S ~ 85S: 
    lwp  = lwp[:, late0:late1+1, :]   #..Units in kg m^-2
    iwp  = iwp[:, late0:late1+1, :]   #..Units in kg m^-2
    ps  = ps[:, late0:late1+1,: ]   #..Units in 
    prw   =  prw[:, late0:late1+1,: ]   #..Units in kg m^-2 , shape in (721, 1440, 300)
    tas   =  tas[:,:,:]  #.. 2-m surface -air Temp, K, for global region data
    E   =   E[:, late0:late1+1,: ]  #..Evaporation, in m / day
    MC   =  MC[:, late0:late1+1,: ]  #..Moisture convergence, not in flux, units in kg m^-2
    P   =   P[:, late0:late1+1,: ] #。。Precip, in m / day
    ts  =  ts[:, late0:late1+1,: ]  #..surface temperatur, or skin Temperture, in K

    T_700  =  T_700[:, late0:late1+1,: ]   #..700mb Temperature, units in K
    wap   =   wap[:, late0:late1+1,: ]    #..500mb Subsidence, Units in Pa s**-1
    #..print(T_700, wap)

    
    print("ERA reanlysis data shape in: ", lwp.shape)
    print("mean ERA-5 LWP value: ", nanmean(lwp),r' in Kg*m{-2}')
    
    
    #.. Output 4(6)CCFs in ERA5 data: SST, (p-e)/MC, LTS, SUB500, WVP
    #..moisture conv
    p_e  =   array(P+E)* 1000. 
    MoistureConv  =   array(MC) *(-1.)
    #print(nanmean(p_e), nanmean(MoistureConv))
    
    SST  =  array(ts)
    SUB  =   array(wap)
    gmt  =   array(tas)

    LWP_era   =  array(lwp)
    IWP_era   =  array(iwp)

    #..calc Lower Troposphere Stability, K ?
    k  = 0.286
    theta_700  = array(T_700)* (100000./70000.)**k
    
    #print(theta_700_abr)

    theta_skin = array(ts)* (100000./array(ps))**k
    LTS  = theta_700 - theta_skin

    #..Total column water vapour, Units in kg m**-2
    prw  = array(prw)
    
    ORIGIN_obsmetrics = { 'p_e': p_e, 'MC': MoistureConv, 'LTS': LTS, 'SST': SST, 'SUB': SUB, 'prw': prw, 'gmt':gmt, 'LWP_mac': LWP_MAC, 'LWP_era5': LWP_era, 'IWP_era5': IWP_era, 
                           'TWP_mac': TWP_MAC, 'lat_mac': lat_mac, 'lon_mac': lon_mac, 'times_mac':times_mac, 'lat_era': lat_era, 'lon_era': lon_era}
    
    rawdata_dict2['dict0_var'] =  ORIGIN_obsmetrics
    
    
    ##. DECK_NAS..
    deck_all    =  ['SST', 'p_e', 'MC', 'LTS', 'SUB', 'LWP_mac', 'LWP_era5', 'IWP_era5', 'prw', 'TWP_mac']

    deck_era     = ['LWP_era5', 'IWP_era5',  'prw', 'SST', 'p_e', 'MC', 'LTS', 'SUB']

    deck_mac     =  ['LWP_mac',  'TWP_mac']
    
    
    #.  get the Annual-mean data, arrays
    
    dict1_mac_yr  = {}
    dict1_era_yr  = {}
    dict1_gmt_yr = {}
    shape_yr_mac  = LWP_MAC.shape[0]//12
    shape_yr_era   =  LWP_era.shape[0]//12

    #print(shape_yr_era, shape_yr_mac)

    layover_yr_mac = zeros((len(deck_mac), shape_yr_mac, LWP_MAC.shape[1], LWP_MAC.shape[2]))
    layover_yr_era  = zeros((len(deck_era), shape_yr_era, LWP_era.shape[1], LWP_era.shape[2]))
    
    layover_yr_gmt  = zeros((shape_yr_era, gmt.shape[1], gmt.shape[2]))
    
    for a in arange(len(deck_mac)):
        for i in range(shape_yr_mac):
            layover_yr_mac[a, i,:,:]  = nanmean(ORIGIN_obsmetrics[deck_mac[a]][i*12:(i+1)*12,:,:], axis=0)

        dict1_mac_yr[deck_mac[a] +'_yr'] =  layover_yr_mac[a, :, :, :]
        print(deck_mac[a])
    
    
    for b in arange(len(deck_era)):
        for j in range(shape_yr_era):
            layover_yr_era[b, j,:,:]  = nanmean(ORIGIN_obsmetrics[deck_era[b]][j*12:(j+1)*12, :,:], axis=0)
            #..calc for b times
            layover_yr_gmt[j, :,:]        =   nanmean(gmt[j*12:(j+1)*12, :,:], axis=0)

        dict1_era_yr[deck_era[b] +'_yr'] =  layover_yr_era[b, :, :, :]
        dict1_gmt_yr['gmt_yr']  =  layover_yr_gmt
        print(deck_era[b])
        
    print("dict_mac_yr shape in: ", dict1_mac_yr['LWP_mac_yr'].shape)   # dict1_era_yr['LWP_era5_yr'],
    
    
    rawdata_dict2['dict1_mac_yr']  =   dict1_mac_yr
    rawdata_dict2['dict1_era_yr']  =   dict1_era_yr
    rawdata_dict2['dict1_gmt_yr']  =   dict1_gmt_yr
    
    
    
    #..set area-mean range:
    x_range  = arange(-180., 183, 5.)   #..logitude sequences edge: number:73
    s_range  = arange(-90., 90, 5.) + 2.5   #..global-region latitude edge:(36)

    y_range  = arange(-85, -35., 5.) +2.5   #..southern-ocaen latitude edge:10
    
    # Calc binned array('r': any resolution) for Annually variable:
    
    #..define dictionary to store '_yr_bin' data
    dict1_mac_yr_bin  = {}
    dict1_era_yr_bin  = {}
    dict1_gmt_yr_bin  = {}

    #lat_mac  = lat_mac
    #lon_mac  = lon_mac
    #lat_era =  lat_era
    #lon_era =  lon_era
    lat_era_origin  = lat_era1

    #..First transfer finer_resolution data (era5) to coarser data (MAC)resolution: 1_degree X 1_degree :
    for a in arange(len(deck_era)):
        dict1_era_yr_bin[deck_era[a] + '_mediated_yr_bin'], lat_1dt1d, lon_1dt1d = binned_cySouthOCEAN_anr(dict1_era_yr[deck_era[a] + '_yr'], lat_era, lon_era, 1)
        dict1_era_yr_bin[deck_era[a] + '_yr_bin_unmasked']  = binned_cySouthOcean_anr(dict1_era_yr[deck_era[a] + '_yr'], lat_era, lon_era, 5.)
        
        #..Second find the 'nan' point indexes in each time-specific 2 dimensions arrays for LWP_mac_yr:
        mask_lwp = isnan(dict1_mac_yr['LWP_mac_yr'])
        dict1_era_yr_bin[deck_era[a]+'_masked_yr_bin'] = ma.masked_array(dict1_era_yr_bin[deck_era[a]+ '_mediated_yr_bin' ], mask = mask_lwp)

    #print(lon_era, lon_1dt1d)
    #ind_flase_yr  = ma.count_masked(dict1_era_yr_bin['LWP_era5_masked_yr_bin'])
    #print(ind_flase_yr)

    #..Third calc the 5 *5 binned array for 'LWP_mac_yr':
    for b in arange(len(deck_mac)):
        dict1_mac_yr_bin[deck_mac[b] +'_yr_bin']  =   binned_cySouthOcean_anr(dict1_mac_yr[deck_mac[b] +'_yr'], lat_mac, lon_mac, 5)

    #..Fourth calc the 5* 5  binned array for ERA5 data arrays:
    for c in arange(len(deck_era)):
        dict1_era_yr_bin[deck_era[c] +'_yr_bin']  =  binned_cySouthOcean_anr(dict1_era_yr_bin[deck_era[c]+'_masked_yr_bin'], lat_1dt1d, lon_1dt1d, 5)

    dict1_gmt_yr_bin['gmt_yr_bin']  =   binned_cyGlobal_anr(dict1_gmt_yr['gmt_yr'], lat_era_origin,  lon_era, 5.)

    print(dict1_era_yr_bin['LWP_era5_yr_bin'].shape)   #..(25, 10, 73)
    print(dict1_mac_yr_bin['LWP_mac_yr_bin'].shape)   #..(25, 10, 73)
    print(dict1_gmt_yr_bin['gmt_yr_bin'].shape)   #..(25, 36,73)


    rawdata_dict2['dict1_mac_yr_bin']  =   dict1_mac_yr_bin
    rawdata_dict2['dict1_era_yr_bin']  =   dict1_era_yr_bin
    rawdata_dict2['dict1_gmt_yr_bin']  =   dict1_gmt_yr_bin
    
    return rawdata_dict2
    