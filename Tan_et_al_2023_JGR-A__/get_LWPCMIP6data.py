# # read CMIP6 monthly data for lwp AND cloud controlling factors  

import netCDF4
from numpy import *
import matplotlib.pyplot as plt
import xarray as xr
# import PyNIO as Nio # deprecated
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


def get_LWPCMIP6(modn='', consort='', cmip='', exper='', ensmem='', gg='', typevar='Amon'):
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
    
    if modn == 'HadGEM3-GC31-LL':
        ensmem = 'r1i1p1f3'

        TEST1_time= read_var_mod(modn=modn,consort=consort,varnm='pr',cmip=cmip,exper=exper,ensmem=ensmem,gg=gg,typevar=typevar,time1=[1,1,15], time2=[3349, 12, 15])[-1]
        time1=[int(min(TEST1_time[:,0])),1,15]
        time2=[int(min(TEST1_time[:,0]))+149, 12, 15]
    
    elif modn == 'EC-Earth3':
        ensmem = 'r3i1p1f1'

        TEST1_time= read_var_mod(modn=modn,consort=consort,varnm='pr',cmip=cmip,exper=exper,ensmem=ensmem,gg=gg,typevar=typevar,time1=[1,1,1], time2=[3349, 12, 31])[-1]
        time1=[int(min(TEST1_time[:,0])),1,1]
        time2=[int(min(TEST1_time[:,0]))+149, 12, 31]
    
    else:

        TEST1_time= read_var_mod(modn=modn,consort=consort,varnm='pr',cmip=cmip,exper=exper,ensmem=ensmem,gg=gg,typevar=typevar,time1=[1,1,1],time2=[3349, 12, 31])[-1]
        time1=[int(min(TEST1_time[:,0])),1,1]
        time2=[int(min(TEST1_time[:,0]))+149, 12, 31]
        
    print("retrieve time: ", time1, time2)
    
    
    sfc_T_abr        = read_var_mod(modn=modn, consort=consort, varnm='ts', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= time1, time2= time2)[0]

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

    E_abr           = read_var_mod(modn=modn, consort=consort, varnm='hfls', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= time1, time2= time2)[0]
    
    
    rsdt_abr        = read_var_mod(modn=modn, consort=consort, varnm='rsdt', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= time1, time2= time2)[0]
    rsut_abr        = read_var_mod(modn=modn, consort=consort, varnm='rsut', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= time1, time2= time2)[0]
    rsutcs_abr      = read_var_mod(modn=modn, consort=consort, varnm='rsutcs', cmip= cmip, exper=exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1 = time1, time2= time2)[0]
    # print(sfc_T_abr.shape)
    #.. 1800 months =150 yrs for CESM2: abrupt-4xCO2 experiemnt
    inputVar_abr = {'sfc_T': sfc_T_abr, 'T_700': T_700_abr, 'sfc_P': sfc_P_abr, 'sub': sub_abr, 'clivi': clivi_abr, 'clwvi':clwvi_abr, 'tas': tas_abr, 
                    'P': P_abr, 'E': E_abr, 'rsdt': rsdt_abr, 'rsut': rsut_abr, 'rsutcs': rsutcs_abr, 'pres': Pres_abr, 'lat':lat_abr, 'lon':lon_abr, 'times':times_abr}
    
    
    
    #..pi-Control
    exper = 'piControl'
    
    if modn == 'HadGEM3-GC31-LL':
        ensmem = 'r1i1p1f1'
        TEST2_time= read_var_mod(modn=modn,consort=consort,varnm='ps',cmip=cmip, exper=exper,ensmem=ensmem,gg=gg,typevar=typevar,time1=[1,1,15], time2=[8000,12,15])[-1]
        timep1=[int(min(TEST2_time[:,0])), 1,15]   #.. max-799
        timep2=[int(min(TEST2_time[:,0]))+98, 12, 15]  #.. max-750

    elif modn == 'EC-Earth3':
        ensmem = 'r1i1p1f1'
        TEST2_time= read_var_mod(modn=modn,consort=consort,varnm='ps',cmip=cmip, exper=exper,ensmem=ensmem,gg=gg,typevar=typevar,time1=[1,1,1], time2=[8000,12,31])[-1]
        timep1=[int(min(TEST2_time[:,0])), 1, 1]   #.. max-799
        timep2=[int(min(TEST2_time[:,0]))+98, 12, 31]  #.. max-750

    elif modn == 'NESM3':
        ensmem = 'r1i1p1f1'
        TEST2_time= read_var_mod(modn=modn,consort=consort,varnm='ta',cmip=cmip, exper=exper,ensmem=ensmem,gg=gg,typevar=typevar, read_p= True, time1=[1,1,1], time2=[8000,12,31])[-1]
        timep1=[int(min(TEST2_time[:,0])), 1, 1]   #.. max-799
        timep2=[int(min(TEST2_time[:,0]))+98, 12, 31]  #.. max-750
    
    elif modn == 'CNRM-CM6-1':
        ensmem = 'r1i1p1f2'
        TEST2_time= read_var_mod(modn=modn,consort=consort,varnm='hfls',cmip=cmip, exper=exper,ensmem=ensmem,gg=gg,typevar=typevar,time1=[1,1,1], time2=[8000,12,31])[-1]
        timep1=[int(min(TEST2_time[:,0])), 1, 1]   #.. max-799
        timep2=[int(min(TEST2_time[:,0]))+98, 12, 31]  #.. max-750

    else:

        TEST2_time= read_var_mod(modn=modn,consort=consort,varnm='ps',cmip=cmip, exper=exper,ensmem=ensmem,gg=gg,typevar=typevar,time1=[1,1,1], time2=[8000,12,31])[-1]
        timep1=[int(min(TEST2_time[:,0])),1,1]   #.. max-799
        timep2=[int(min(TEST2_time[:,0]))+98, 12, 31]  #.. max-750
    
    print ("retrieve time: ", timep1, timep2)
    
    sfc_T_pi       = read_var_mod(modn= modn, consort= consort, varnm='ts', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= timep1, time2= timep2)[0]

    T_700_alevs,Pres_pi,lat_pi,lon_pi,times_pi =  read_var_mod(modn= modn, consort= consort, varnm='ta', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=True, time1= timep1, time2= timep2)
    T_700_pi       = T_700_alevs[:, 3,:,:]

    sfc_P_pi       = read_var_mod(modn= modn, consort= consort, varnm='ps', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= timep1, time2= timep2)[0]   
    #..sea surface Pressure, Units in Pa

    sub_alevs      = read_var_mod(modn= modn, consort= consort, varnm='wap', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=True, time1= timep1, time2= timep2)[0]
    sub_pi         =  sub_alevs[:, 5,:,:]
    #..500mb downward motion

    clivi_pi       = read_var_mod(modn= modn, consort= consort, varnm='clivi', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= timep1, time2= timep2)[0]
    #..ICE WATER PATH, Units in kg m^-2
    clwvi_pi       = read_var_mod(modn= modn, consort= consort, varnm='clwvi', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= timep1, time2= timep2)[0]
    
    tas_pi         = read_var_mod(modn= modn, consort= consort, varnm='tas', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= timep1, time2= timep2)[0]
    #..2-m air Temperature, for 'gmt'

    P_pi           = read_var_mod(modn= modn, consort= consort, varnm='pr', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= timep1, time2= timep2)[0]
    #..Precipitation, Units in kg m^-2 s^-1 = mm * s^-1
    E_pi           = read_var_mod(modn= modn, consort= consort, varnm='hfls', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= timep1, time2= timep2)[0]
    #..Evaporations, Units in W m^-2 = J * m^-2 * s^-1

    
    rsdt_pi        = read_var_mod(modn=modn, consort=consort, varnm='rsdt', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= timep1, time2= timep2)[0]
    rsut_pi        = read_var_mod(modn=modn, consort=consort, varnm='rsut', cmip=cmip, exper= exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1= timep1, time2= timep2)[0]
    rsutcs_pi      = read_var_mod(modn=modn, consort=consort, varnm='rsutcs', cmip= cmip, exper=exper, ensmem=ensmem, typevar=typevar, gg=gg, read_p=False, time1 = timep1, time2= timep2)[0]
    # print(sfc_T_pi.shape)
    #..6000 months =99 yrs for CESM2 piControl experiment
    
    inputVar_pi = {'sfc_T': sfc_T_pi, 'T_700': T_700_pi, 'sfc_P': sfc_P_pi, 'sub': sub_pi, 'clivi': clivi_pi, 'clwvi': clwvi_pi, 'tas': tas_pi, 
                   'P': P_pi, 'E': E_pi, 'rsdt': rsdt_pi, 'rsut': rsut_pi, 'rsutcs': rsutcs_pi, 'pres':Pres_pi, 'lat':lat_pi, 'lon':lon_pi, 'times': times_pi}
    '''
    TS = read_var_mod(modn=modn,consort=consort,varnm='ts',cmip=cmip,exper=exper,ensmem=ensmem,gg=gg,typevar=typevar,time1=time1,time2=time2)
    '''
    #return {'SST0': TS[0], 'SST1': TSf[0], 'LWP0': LWP0, 'LWP1': LWP1, 'lat': IWP[2][:], 'lon': IWP[3][:], 'time0': IWP[-1], 'time1': IWPf[-1], 'IWP0': IWP[0], 'IWP1': IWPf[0], 'P0': P[0], 'E0': E[0], 'P1': Pf[0], 'E1': Ef[0], "WVP0": wvp[0], 'WVP1': wvpf[0], 'SI': SI, 'U100': U10_0, 'U101': U10_f, 'TAS0': TAS[0], 'TAS1': TASf[0], 'TA0': TA_0, 'TA1': TA_1, 'SW0': SW[0], 'SW0c': SWc[0], 'SW1': SWf[0], 'SW1c': SWfc[0],'TS0':TS[0],'TS1':TSf[0]}   #,'WAP0':WAP_0,'WAP1':WAP_1,'LTS0':LTS,'LTS1':LTSf}
    return inputVar_pi, inputVar_abr

