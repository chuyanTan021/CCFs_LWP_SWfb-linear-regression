# This module is to read the 'Precipitation and Evaporation ' from GCMs and MERRA-2 Observation datasets,
# investigate the 'P - E' patterns of GCM and the real_world, and comparing with the 'delta_P-E' changes from 'piControl' to 'abrupt4xCO2'
# in gcms/

import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import pandas as pd
import glob
from copy import deepcopy
from scipy.stats import *
from sklearn import linear_model

from area_mean import *
from binned_cyFunctions5 import *
from read_hs_file import read_var_mod
from read_var_obs import *
from get_LWPCMIP5data import *
from get_LWPCMIP6data import *
from get_OBSLRMdata import *
from useful_func_cy import *


def P_E_analysis():
    exp = 'piControl'
    # CMIP6 (30)
    AWICM11MR = {'modn': 'AWI-CM-1-1-MR', 'consort': 'AWI', 'cmip': 'cmip6',
                'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    BCCCSMCM2MR = {'modn': 'BCC-CSM2-MR', 'consort': 'BCC', 'cmip': 'cmip6',   # T700 (ta) and sfc_T (ts) have different time shape
                   'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    BCCESM1 = {'modn': 'BCC-ESM1', 'consort': 'BCC', 'cmip': 'cmip6',
                   'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    CAMSCSM1 = {'modn': 'CAMS-CSM1-0', 'consort': 'CAMS', 'cmip': 'cmip6',
                'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    CMCCCM2SR5 = {'modn': 'CMCC-CM2-SR5', 'consort': 'CMCC', 'cmip': 'cmip6', 
                 'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    CESM2 = {'modn': 'CESM2', 'consort': 'NCAR', 'cmip': 'cmip6',
                 'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    CESM2FV2 = {'modn': 'CESM2-FV2', 'consort': 'NCAR', 'cmip': 'cmip6',
                 'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    CESM2WACCM = {'modn': 'CESM2-WACCM', 'consort': 'NCAR', 'cmip': 'cmip6',
                 'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    CESM2WACCMFV2 = {'modn': 'CESM2-WACCM-FV2', 'consort': 'NCAR', 'cmip': 'cmip6',
                 'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}

    CNRMCM61 = {'modn': 'CNRM-CM6-1', 'consort': 'CNRM-CERFACS', 'cmip': 'cmip6', 
                   'exper': exp, 'ensmem': 'r1i1p1f2', 'gg': 'gr', "typevar": 'Amon'}
    CNRMCM61HR = {'modn': 'CNRM-CM6-1-HR', 'consort': 'CNRM-CERFACS', 'cmip': 'cmip6',
                   'exper': exp, 'ensmem': 'r1i1p1f2', 'gg': 'gr', "typevar": 'Amon'}
    CNRMESM21 = {'modn': 'CNRM-ESM2-1', 'consort': 'CNRM-CERFACS', 'cmip': 'cmip6', 
                     'exper': exp, 'ensmem': 'r1i1p1f2', 'gg': 'gr', "typevar": 'Amon'}
    CanESM5 = {'modn': 'CanESM5', 'consort': 'CCCma', 'cmip': 'cmip6',
                   'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    E3SM10 = {'modn': 'E3SM-1-0', 'consort': 'E3SM-Project', 'cmip': 'cmip6',
                  'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gr', "typevar": 'Amon'}
    
    ECEarth3 = {'modn': 'EC-Earth3', 'consort': 'EC-Earth-Consortium', 'cmip': 'cmip6',
           'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gr', "typevar": 'Amon'}
    ECEarth3Veg = {'modn': 'EC-Earth3-Veg', 'consort': 'EC-Earth-Consortium', 'cmip': 'cmip6',
           'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gr', "typevar": 'Amon'}
    
    FGOALSg3 = {'modn': 'FGOALS-g3', 'consort': 'CAS', 'cmip': 'cmip6',
                    'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    GISSE21G = {'modn': 'GISS-E2-1-G', 'consort': 'NASA-GISS', 'cmip': 'cmip6',
                    'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    GISSE21H = {'modn': 'GISS-E2-1-H', 'consort': 'NASA-GISS', 'cmip': 'cmip6',
                    'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    GISSE22G = {'modn': 'GISS-E2-2-G', 'consort': 'NASA-GISS', 'cmip': 'cmip6',
                   'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    GFDLCM4 = {'modn': 'GFDL-CM4', 'consort': 'NOAA-GFDL', 'cmip': 'cmip6',
               'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gr1', "typevar": 'Amon'}
    # HADGEM3 = {'modn': 'HadGEM3-GC31-LL', 'consort': 'MOHC', 'cmip': 'cmip6',
    #             'exper': 'piControl', 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}   #..missing 'wap' in 'piControl' exp(Daniel says that HadGEM3-GC31 not using p-level, so doesn't have variables on p-level
    INM_CM48 = {'modn': 'INM-CM4-8', 'consort': 'INM', 'cmip': 'cmip6', 
                    'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gr1', "typevar": 'Amon'}  #..data not available again 
    IPSLCM6ALR = {'modn': 'IPSL-CM6A-LR', 'consort': 'IPSL', 'cmip': 'cmip6',
                      'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gr', "typevar": 'Amon'}
    MIROCES2L = {'modn': 'MIROC-ES2L', 'consort': 'MIROC', 'cmip': 'cmip6',
                  'exper': exp, 'ensmem': 'r1i1p1f2', 'gg': 'gn', "typevar": 'Amon'}
    MIROC6 = {'modn': 'MIROC6', 'consort': 'MIROC', 'cmip': 'cmip6',
                  'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    MPIESM12LR = {'modn': 'MPI-ESM1-2-LR', 'consort': 'MPI-M', 'cmip': 'cmip6',
                      'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    MRIESM20 = {'modn': 'MRI-ESM2-0', 'consort': 'MRI', 'cmip': 'cmip6',
                    'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    NESM3 = {'modn': 'NESM3', 'consort': 'NUIST', 'cmip': 'cmip6', 
                     'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    NorESM2MM = {'modn': 'NorESM2-MM', 'consort': 'NCC', 'cmip': 'cmip6',
                     'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    SAM0 = {'modn': 'SAM0-UNICON', 'consort': 'SNU', 'cmip': 'cmip6', 
                'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    TaiESM1 = {'modn': 'TaiESM1', 'consort': 'AS-RCEC', 'cmip': 'cmip6', 
                     'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    
    # CMIP5: (14)
    ACCESS10 = {'modn': 'ACCESS1-0', 'consort': 'CSIRO-BOM', 'cmip': 'cmip5',   # 2-d (145) and 3-d (146) variables have different lat shape
                'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    ACCESS13 = {'modn': 'ACCESS1-3', 'consort': 'CSIRO-BOM', 'cmip': 'cmip5',   # 2-d (145) and 3-d (146) variables have different lat shape
                'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    BNUESM = {'modn': 'BNU-ESM', 'consort': 'BNU', 'cmip': 'cmip5',
              'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    CCSM4 = {'modn': 'CCSM4', 'consort': 'NCAR', 'cmip': 'cmip5',
                 'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    CNRMCM5 = {'modn': 'CNRM-CM5', 'consort': 'CNRM-CERFACS', 'cmip': 'cmip5',
                'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    CSIRO_Mk360 = {'modn': 'CSIRO-Mk3-6-0', 'consort': 'CSIRO-QCCCE', 'cmip': 'cmip5',
                'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    CanESM2 = {'modn': 'CanESM2', 'consort': 'CCCma', 'cmip': 'cmip5',
                'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    FGOALSg2 = {'modn': 'FGOALS-g2', 'consort': 'LASG-CESS', 'cmip': 'cmip5',   # missing 'prw' in piControl
                'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    FGOALSs2 = {'modn': 'FGOALS-s2', 'consort': 'LASG-IAP', 'cmip': 'cmip5',
                'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    GFDLCM3 = {'modn': 'GFDL-CM3', 'consort': 'NOAA-GFDL', 'cmip': 'cmip5',
                'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    GISSE2H = {'modn': 'GISS-E2-H', 'consort': 'NASA-GISS', 'cmip': 'cmip5',
               'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    GISSE2R = {'modn': 'GISS-E2-R', 'consort': 'NASA-GISS', 'cmip': 'cmip5',
               'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    IPSLCM5ALR = {'modn': 'IPSL-CM5A-LR', 'consort': 'IPSL', 'cmip': 'cmip5',
                   'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    MIROC5 = {'modn': 'MIROC5', 'consort': 'MIROC', 'cmip': 'cmip5',
                 'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    MPIESMMR = {'modn': 'MPI-ESM-MR', 'consort': 'MPI-M', 'cmip': 'cmip5',
                 'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    NorESM1M = {'modn': 'NorESM1-M', 'consort': 'NCC', 'cmip': 'cmip5',
                 'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    
    deck = [BCCESM1, CanESM5, CESM2, CESM2FV2, CESM2WACCM, CNRMESM21, GISSE21G, GISSE21H, IPSLCM6ALR, MRIESM20, MIROC6, SAM0, E3SM10, FGOALSg3, GFDLCM4, CAMSCSM1, INM_CM48, MPIESM12LR, AWICM11MR, CMCCCM2SR5, CESM2WACCMFV2, CNRMCM61, CNRMCM61HR, ECEarth3, ECEarth3Veg, GISSE22G, MIROCES2L, NESM3, NorESM2MM, TaiESM1, BNUESM, CCSM4, CNRMCM5, CSIRO_Mk360, CanESM2, FGOALSg2, FGOALSs2, GFDLCM3, GISSE2H, GISSE2R, IPSLCM5ALR, MIROC5, MPIESMMR, NorESM1M]  #..current # 30 + 14 (44)


    inputVar_p_e_abr, inputVar_p_e_pi = get_P_E_CMIP(deck)
    inputVar_p_e_obs = get_P_E_OBS()


    # calculate 'P - E':
    # GCM(s) data
    for m in range(len(deck)):
        # 'abrupt4xCO2'
        Precip_abr = np.asarray(inputVar_p_e_abr[deck[m]['modn']]['P']) * (24.*60.*60.)   #.. Precipitation. Convert the units from kg m^-2 s^-1 -> mm*day^-1
        
        lh_vaporization_abr = (2.501 - (2.361 * 10**-3) * (inputVar_p_e_abr[deck[m]['modn']]['sfc_T'] - 273.15)) * 1e6  # the latent heat of vaporization at the surface Temperature
        # Eva_abr2 = array(inputVar_p_e_abr[deck[m]['modn']]['E']) * (24. * 60 * 60)
        Eva_abr1 = array(inputVar_p_e_abr[deck[m]['modn']]['E']) / lh_vaporization_abr * (24. * 60 * 60)  #.. Evaporation, mm day^-1
        
        MC_abr = Precip_abr - Eva_abr1   #..Moisture Convergence calculated from abrupt4xCO2's P - E, Units in mm day^-1
        
        inputVar_p_e_abr[deck[m]['modn']]['MC'] =  MC_abr
        
        # 'piControl'
        Precip_pi = array(inputVar_p_e_pi[deck[m]['modn']]['P']) * (24.*60.*60.)    #..Precipitation. Convert the units from kg m^-2 s^-1 -> mm*day^-1
        
        lh_vaporization_pi = (2.501 - (2.361 * 10**-3) * (inputVar_p_e_pi[deck[m]['modn']]['sfc_T'] - 273.15)) * 1e6  # the latent heat of vaporization at the surface Temperature
        Eva_pi1 = array(inputVar_p_e_pi[deck[m]['modn']]['E']) / lh_vaporization_pi * (24. * 60 * 60)
        # Eva_pi2 = array(inputVar_pi['E']) * (24.*60.*60.)   #..evaporation, mm day^-1
        
        MC_pi = Precip_pi - Eva_pi1   #..Moisture Convergence calculated from pi-Control's P - E, Units in mm day^-1
        
        inputVar_p_e_pi[deck[m]['modn']]['MC'] = MC_pi
    
    # OBS: MERRA-2 Re-analysis data
    SST_obs = inputVar_p_e_obs['sfc_T'] * 1.
    # Precip: Precipitation, Unit in mm day^-1 (convert from kg m^-2 s^-1)
    Precip_obs = inputVar_p_e_obs['P'] * (24. * 60 * 60)
    # Eva: Evaporation, Unit in mm day^-1 (here use the latent heat flux from the sfc, unit convert from W m^-2 --> kg m^-2 s^-1 --> mm day^-1)
    lh_vaporization_obs = (2.501 - (2.361 * 10**-3) * (SST_obs - 273.15)) * 1e6  # the latent heat of vaporization at the surface Temperature
    Eva_obs = inputVar_p_e_obs['E'] / lh_vaporization_obs * (24. * 60 * 60)

    # MC: Moisture Convergence, represent the water vapor abundance, Unit in mm day^-1
    MC_obs = Precip_obs - Eva_obs
    
    inputVar_p_e_obs['MC'] = MC_obs
    
    # calc the 'abr - mean(pi)' value of 'P - E':
    inputVar_p_e_abrminuspi = {}
    for h in range(len(deck)):
        inputVar_p_e_abrminuspi[deck[h]['modn']+'MC'] = inputVar_p_e_abr[deck[h]['modn']]['MC'] - np.nanmean(inputVar_p_e_pi[deck[h]['modn']]['MC'], axis = 0)  # return be 3-D shape array: (length_time_abrupt4xCO2, lat, lon)
        inputVar_p_e_abrminuspi[deck[h]['modn']+'lat'] = inputVar_p_e_abr[deck[h]['modn']]['lat']
        inputVar_p_e_abrminuspi[deck[h]['modn']+'lon'] = inputVar_p_e_abr[deck[h]['modn']]['lon']
    
    
    # do annually mean on the OBS data:
    # annually mean variable
    inputVar2_OBS_yr = get_annually_dict(inputVar_p_e_obs, ['MC'], inputVar_p_e_obs['times'], label = 'mon')
    
    # Plotting function:
    pth_plotting = '/glade/work/chuyan/Research/Cloud_CCFs_RMs/Course_objective_ana/plot_file/P_E/'
    PL_P_E_lat(inputVar_p_e_pi, inputVar_p_e_abrminuspi, inputVar2_OBS_yr, deck, pth_plotting)
    
    # Save the calculated P - E metrics:
    pth_data = '/glade/work/chuyan/Research/Cloud_CCFs_RMs/Course_objective_ana/data_file/P_E/'
    np.savez(pth_data+'_P_E_data(global)', GCM_p_e_abr = inputVar_p_e_abr, GCM_p_e_pi = inputVar_p_e_pi, GCM_p_e_abrminuspi = inputVar_p_e_abrminuspi, OBS_p_e = inputVar_p_e_obs, OBS_p_e_annually_mean = inputVar2_OBS_yr)
    return None



def get_P_E_CMIP(deck):
    
    
    
    # loop through the GCM deck:
    
    inputVar_gcm_abr = {}
    inputVar_gcm_pi = {}
    
    for i in range(len(deck)):
        
        if deck[i]['cmip'] == 'cmip5':
            
            #..abrupt4xCO2
            deck[i]['exper'] = 'abrupt4xCO2'

            TEST1_time= read_var_mod(varnm='pr', read_p =False, time1=[1,1,1], time2=[3349, 12, 31], **deck[i])[-1]
            time1=[int(min(TEST1_time[:,0])),1,1]
            time2=[int(min(TEST1_time[:,0]))+149, 12, 31]

            print("retrieve time: ", time1, time2)
            
            #.. read 'abrupt4xCO2' P and E:
            P_abr  = read_var_mod(varnm='pr', read_p=False, time1= time1, time2= time2, **deck[i])[0]
            E_abr,[],lat_abr_cmip5,lon_abr_cmip5,times_abr_cmip5 = read_var_mod(varnm='hfls', read_p=False, time1= time1, time2= time2, **deck[i])
            sfc_T_abr  = read_var_mod(varnm='ts', read_p=False, time1= time1, time2= time2, **deck[i])[0]
            
            inputVar_abr_cmip5 = {'P': P_abr, 'E': E_abr, 'sfc_T': sfc_T_abr, 'lat': lat_abr_cmip5, 'lon': lon_abr_cmip5, 'times': times_abr_cmip5}
            
            inputVar_gcm_abr[deck[i]['modn']] = inputVar_abr_cmip5
            
            #..pi-Control
            deck[i]['exper'] = 'piControl'

            if deck[i]['modn'] == 'IPSL-CM5A-LR':
                deck[i]['ensmem'] = 'r1i1p1'
                TEST2_time= read_var_mod(varnm='hfls', read_p= False, time1=[1,1,1], time2=[8000,12,31], **deck[i])[-1]
                timep1=[int(min(TEST2_time[:,0])), 1, 1]   #.. max-799
                timep2=[int(min(TEST2_time[:,0]))+98, 12, 31]  #.. max-750

            else:

                TEST2_time= read_var_mod(varnm='ps', read_p= False, time1=[1,1,1], time2=[8000,12,31], **deck[i])[-1]
                timep1=[int(min(TEST2_time[:,0])),1,1]   #.. max-799
                timep2=[int(min(TEST2_time[:,0]))+98, 12, 31]  #.. max-750

            print ("retrieve time: ", timep1, timep2)
            
            #.. read 'piControl' P and E data:
            
            P_pi  = read_var_mod(varnm='pr', read_p=False, time1= timep1, time2= timep2, **deck[i])[0]
            #..Precipitation, Units in kg m^-2 s^-1 = mm * s^-1
            E_pi, [],lat_pi_cmip5,lon_pi_cmip5,times_pi_cmip5 = read_var_mod(varnm='hfls', read_p=False, time1= timep1, time2= timep2, **deck[i])
            #..Evaporations, Units in W m^-2 = J * m^-2 * s^-1
            sfc_T_pi = read_var_mod(varnm='ts', read_p=False, time1= timep1, time2= timep2, **deck[i])[0]
            
            inputVar_pi_cmip5 = {'P': P_pi, 'E': E_pi, 'sfc_T': sfc_T_pi, 'lat':lat_pi_cmip5, 'lon':lon_pi_cmip5, 'times': times_pi_cmip5}
            inputVar_gcm_pi[deck[i]['modn']] = inputVar_pi_cmip5
        
        elif deck[i]['cmip'] == 'cmip6':
            
            #..abrupt4xCO2
            deck[i]['exper'] = 'abrupt-4xCO2'

            if deck[i]['modn'] == 'HadGEM3-GC31-LL':
                deck[i]['ensmem'] = 'r1i1p1f3'

                TEST1_time= read_var_mod(varnm='pr', read_p =False, time1=[1,1,15], time2=[3349, 12, 15], **deck[i])[-1]
                time1=[int(min(TEST1_time[:,0])),1,15]
                time2=[int(min(TEST1_time[:,0]))+149, 12, 15]

            elif deck[i]['modn'] == 'EC-Earth3':
                deck[i]['ensmem'] = 'r3i1p1f1'

                TEST1_time= read_var_mod(varnm='pr', read_p =False, time1=[1,1,1], time2=[3349, 12, 31], **deck[i])[-1]
                time1=[int(min(TEST1_time[:,0])),1,1]
                time2=[int(min(TEST1_time[:,0]))+149, 12, 31]

            else:

                TEST1_time= read_var_mod(varnm='pr', read_p =False, time1=[1,1,1],time2=[3349, 12, 31], **deck[i])[-1]
                time1=[int(min(TEST1_time[:,0])),1,1]
                time2=[int(min(TEST1_time[:,0]))+149, 12, 31]

            print("retrieve time: ", time1, time2)
            
            #.. read 'abrupt-4xCO2' P and E: 
            P_abr  = read_var_mod(varnm='pr', read_p=False, time1= time1, time2= time2, **deck[i])[0]
            E_abr, [],lat_abr_cmip6,lon_abr_cmip6,times_abr_cmip6 = read_var_mod(varnm='hfls', read_p=False, time1= time1, time2= time2, **deck[i])
            sfc_T_abr  = read_var_mod(varnm='ts', read_p=False, time1= time1, time2= time2, **deck[i])[0]
            inputVar_abr_cmip6 = {'P': P_abr, 'E': E_abr, 'sfc_T': sfc_T_abr, 'lat': lat_abr_cmip6, 'lon': lon_abr_cmip6, 'times': times_abr_cmip6}
            
            inputVar_gcm_abr[deck[i]['modn']] = inputVar_abr_cmip6
            
            #..pi-Control
            deck[i]['exper'] = 'piControl'

            if deck[i]['modn'] == 'HadGEM3-GC31-LL':
                deck[i]['ensmem'] = 'r1i1p1f1'
                TEST2_time= read_var_mod(varnm='ps', read_p =False, time1=[1,1,15], time2=[8000,12,15], **deck[i])[-1]
                timep1=[int(min(TEST2_time[:,0])), 1,15]   #.. max-799
                timep2=[int(min(TEST2_time[:,0]))+98, 12, 15]  #.. max-750

            elif deck[i]['modn'] == 'EC-Earth3':
                deck[i]['ensmem'] = 'r1i1p1f1'
                TEST2_time= read_var_mod(varnm='ps', read_p =False, time1=[1,1,1], time2=[8000,12,31], **deck[i])[-1]
                timep1=[int(min(TEST2_time[:,0])), 1, 1]   #.. max-799
                timep2=[int(min(TEST2_time[:,0]))+98, 12, 31]  #.. max-750

            elif deck[i]['modn'] == 'NESM3':
                deck[i]['ensmem'] = 'r1i1p1f1'
                TEST2_time= read_var_mod(varnm ='ta', read_p= True, time1=[1,1,1], time2=[8000,12,31], **deck[i])[-1]
                timep1=[int(min(TEST2_time[:,0])), 1, 1]   #.. max-799
                timep2=[int(min(TEST2_time[:,0]))+98, 12, 31]  #.. max-750

            elif deck[i]['modn'] == 'CNRM-CM6-1':
                deck[i]['ensmem'] = 'r1i1p1f2'
                TEST2_time= read_var_mod(varnm='hfls', read_p =False, time1=[1,1,1], time2=[8000,12,31], **deck[i])[-1]
                timep1=[int(min(TEST2_time[:,0])), 1, 1]   #.. max-799
                timep2=[int(min(TEST2_time[:,0]))+98, 12, 31]  #.. max-750

            else:
                TEST2_time= read_var_mod(varnm='ps', read_p =False, time1=[1,1,1], time2=[8000,12,31], **deck[i])[-1]
                timep1=[int(min(TEST2_time[:,0])),1,1]   #.. max-799
                timep2=[int(min(TEST2_time[:,0]))+98, 12, 31]  #.. max-750

            print ("retrieve time: ", timep1, timep2)
            
            #.. read 'piControl' P and E data: 
            
            P_pi  = read_var_mod(varnm='pr', read_p=False, time1= timep1, time2= timep2, **deck[i])[0]
            #..Precipitation, Units in kg m^-2 s^-1 = mm * s^-1
            E_pi, [],lat_pi_cmip6,lon_pi_cmip6,times_pi_cmip6 = read_var_mod(varnm='hfls', read_p=False, time1= timep1, time2= timep2, **deck[i])
            #..Evaporations, Units in W m^-2 = J * m^-2 * s^-1
            sfc_T_pi  = read_var_mod(varnm='ts', read_p=False, time1= timep1, time2= timep2, **deck[i])[0]
            inputVar_pi_cmip6 = {'P': P_pi, 'E': E_pi, 'sfc_T': sfc_T_pi, 'lat': lat_pi_cmip6, 'lon': lon_pi_cmip6, 'times': times_pi_cmip6}
            
            inputVar_gcm_pi[deck[i]['modn']] = inputVar_pi_cmip6
        
        
    return inputVar_gcm_abr, inputVar_gcm_pi



def get_P_E_OBS(test_flag = 'test1'):
    # This function is for reading the P - E Observation data from the MERRA-2 Re-analysis.
    
    P = read_var_obs_MERRA2(varnm = 'PRECTOT', read_p = False, valid_range1=[2002, 7, 15], valid_range2=[2016, 12, 31])[0]
    E, lat_merra2, lon_merra2, [], times_merra2  = read_var_obs_MERRA2(varnm = 'EFLUX', read_p = False, valid_range1=[2002, 7, 15], valid_range2=[2016, 12, 31])
    sfc_T = read_var_obs_MERRA2(varnm = 'TS', read_p = False, valid_range1=[2002, 7, 15], valid_range2=[2016, 12, 31])[0]
    
    inputVar_obs = {'P': P, 'E': E, 'sfc_T': sfc_T, 'lat': lat_merra2, 'lon': lon_merra2, 'times': times_merra2}   # 'clivi': clivi_abr, 'clwvi':clwvi_abr, 'rsdt': rsdt_abr, 'rsut': rsut_abr, 'rsutcs': rsutcs_abr..
    
    return inputVar_obs



def PL_P_E_lat(data_gcm_pi, data_gcm_DELTA, data_OBS, deck, pth_plotting):
    # This function is for plotting (1)the 'P - E' vs latitude plot, (2)the 'delta_P-E' vs latitude plot.
    # 'data_gcm_pi' is a dict who save the 'piControl' Moisture Convergence data,
    # 'data_gcm_DELTA' save the DELTA_Moisture Convergence | (abrupt4xCO2 - mean(piControl)) data,
    # 'data_obs' save the MERRA-2 Moisture Convergence data;
    # deck is a list for GCM info,
    # pth_plotting provides the plots saving path.
    
    fig6 = plt.figure(figsize=(9.5, 6.5))  #(16.2, 9.3))
    ax6 = fig6.add_axes([0,0,0.7,1])
    parameters = {'axes.labelsize': 16, 'legend.fontsize': 17, 'axes.titlesize': 14, 'xtick.labelsize': 16, 'ytick.labelsize': 16}
    plt.rcParams.update(parameters)
    
    # plotting:
    for i in range(len(deck)):
        # 'piControl' P - E:
        ax6.plot(data_gcm_pi[deck[i]['modn']]['lat'], np.nanmean(data_gcm_pi[deck[i]['modn']]['MC'], axis = (0,2)), color = 'gray', alpha = 0.50, linewidth = 2., label = "mean state P - E ", zorder = 10)
    
    ax52 = ax6.twinx()
    color = 'tab:red'
    # Create a color palette
    palette = plt.get_cmap('bwr')
    ax52.set_ylabel(r"$ \Delta (P\ -\ E)\ [mm\ day^{-1}]$", color=color)  # we already handled the x-label with ax1
    ax52.tick_params(axis='y', labelcolor=color)
    
    # calc the SO mean delta_Moiusture Convergence, use this value for palette:
    
    relative_P_E = []
    for j in range(len(deck)):
        data_gcm_DELTA_SO, lat_gcm_so, lon_gcm_so = region_cropping_var(data_gcm_DELTA[deck[j]['modn']+'MC'], data_gcm_DELTA[deck[j]['modn']+'lat'], data_gcm_DELTA[deck[j]['modn']+'lon'], lat_range = [-85., -40.], lon_range = [-180., 180.])
        
        relative_P_E.append(np.nanmean(area_mean(data_gcm_DELTA_SO[12*140:12*150, :,:], lat_gcm_so, lon_gcm_so)))
    
    # Normalized array between 0, 1: (# = 44)
    nor_SO_P_E = (np.asarray(relative_P_E) - np.min(relative_P_E)) / (np.max(relative_P_E) - np.min(relative_P_E))
    
    for j in range(len(deck)):
        ax52.plot(data_gcm_DELTA[deck[j]['modn']+'lat'], np.nanmean(data_gcm_DELTA[deck[j]['modn']+'MC'][12*140:12*150, :, :], axis = (0, 2)), color = palette(nor_SO_P_E[j]), linewidth = 2.0, alpha = 0.40, linestyle = '-', label = r"$\ Delta P\ -\ E $", zorder = 50)
    
    ax6.plot(data_OBS['lat'], np.nanmean(data_OBS['MC'], axis = (0, 2)), color = 'purple', linewidth = 2.8, alpha = 1.0, linestyle = '-', label = ' MERRA-2 Observation ', zorder = 60)
    
    plt.axhline(0., color = 'k', linewidth = 3., linestyle = '--', zorder = 100)
    plt.axvline(-85., color = 'k', linewidth = 3., linestyle = '--', zorder = 100)
    plt.axvline(-40., color = 'k', linewidth = 3., linestyle = '--', zorder = 100)
    # plotting setting:
    ax6.set_xlim([90., -90.])
    
    ax6.set_ylim([-4.5, 4.5])
    ax52.set_ylim([-3.2, 3.2])
    ax6.set_ylabel(r"$ P\ -\ E\ [mm\ day^{-1}] $")
    ax6.set_xlabel("Lat")
    # plt.legend()
    plt.title(" Precipitation - Evaporation ")
    plt.savefig(pth_plotting+"_P_E_calculation(GLOBAL).jpg", bbox_inches = 'tight', dpi = 150)
    
    return None
    