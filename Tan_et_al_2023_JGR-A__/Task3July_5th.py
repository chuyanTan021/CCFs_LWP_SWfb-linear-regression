import os
import subprocess

import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import glob

from calc_LRM_metric import *
from save_meanstateLWP import *
from calc_LRM_split import *

import sys

def main():
    Number_of_models = int(sys.argv[1])
    h = Examine_Radiative_Sensitivities(N_of_model = Number_of_models)


def Examine_Radiative_Sensitivities(N_of_model):
    exp = 'piControl'
    
    # CMIP6: 31 (30: BCCCSMCM2MR)
    AWICM11MR = {'modn': 'AWI-CM-1-1-MR', 'consort': 'AWI', 'cmip': 'cmip6',
                'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    BCCCSMCM2MR = {'modn': 'BCC-CSM2-MR', 'consort': 'BCC', 'cmip': 'cmip6',
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
                    'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gr1', "typevar": 'Amon'}
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

    # CMIP5: 20 (18, ACCESS10, ACCESS13)

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

    MIROCESM = {'modn': 'MIROC-ESM', 'consort': 'MIROC', 'cmip': 'cmip5', 
                'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    MRICGCM3 = {'modn': 'MRI-CGCM3', 'consort': 'MRI', 'cmip': 'cmip5', 
                'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    MPIESMLR = {'modn': 'MPI-ESM-LR', 'consort': 'MPI-M', 'cmip': 'cmip5',
                'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    bcccsm11 = {'modn': 'bcc-csm1-1', 'consort': 'BCC', 'cmip': 'cmip5', 
                'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    GFDLESM2G = {'modn': 'GFDL-ESM2G', 'consort': 'NOAA-GFDL', 'cmip': 'cmip5', 
                'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    GFDLESM2M = {'modn': 'GFDL-ESM2M', 'consort': 'NOAA-GFDL', 'cmip': 'cmip5', 
               'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    inmcm4 = {'modn': 'inmcm4', 'consort': 'INM', 'cmip': 'cmip5', 
                 'exper': exp, 'ensmem': 'r1i1p1', "typevar": 'Amon'}
    deck = [BCCESM1, CanESM5, CESM2, CESM2FV2, CESM2WACCM, CNRMESM21, GISSE21G, GISSE21H, IPSLCM6ALR, MRIESM20, MIROC6, SAM0, E3SM10, FGOALSg3, GFDLCM4, CAMSCSM1, INM_CM48, MPIESM12LR, AWICM11MR, CMCCCM2SR5, CESM2WACCMFV2, CNRMCM61, CNRMCM61HR, ECEarth3, ECEarth3Veg, GISSE22G, MIROCES2L, NESM3, NorESM2MM, TaiESM1, BNUESM, CCSM4, CNRMCM5, CSIRO_Mk360, CanESM2, FGOALSg2, FGOALSs2, GFDLCM3, GISSE2H, GISSE2R, IPSLCM5ALR, MIROC5, MPIESMMR, NorESM1M, MIROCESM, MRICGCM3, MPIESMLR, bcccsm11, GFDLESM2G, GFDLESM2M]   #..current # 30 + 20
    
    deck_nas = ['BCCESM1', 'CanESM5', 'CESM2', 'CESM2FV2', 'CESM2WACCM', 'CNRMESM21', 'GISSE21G', 'GISSE21H', 'IPSLCM6ALR', 'MRIESM20', 'MIROC6', 'SAM0', 'E3SM10', 'FGOALSg3', 'GFDLCM4', 'CAMSCSM1', 'INM_CM48', 'MPIESM12LR', 'AWICM11MR', 'CMCCCM2SR5', 'CESM2WACCMFV2', 'CNRMCM61', 'CNRMCM61HR', 'ECEarth3', 'ECEarth3Veg', 'GISSE22G', 'MIROCES2L', 'NESM3', 'NorESM2MM', 'TaiESM1', 'BNUESM', 'CCSM4', 'CNRMCM5', 'CSIRO_Mk360', 'CanESM2', 'FGOALSg2', 'FGOALSs2', 'GFDLCM3', 'GISSE2H', 'GISSE2R', 'IPSLCM5ALR', 'MIROC5', 'MPIESMMR', 'NorESM1M', 'MIROCESM', 'MRICGCM3', 'MPIESMLR', 'bcccsm11', 'GFDLESM2G', 'GFDLESM2M', 'inmcm4']  #..current # 30 + 21 ('19': 'BCCCSMCM2MR';)

    calc_bin_Sensitivities(**deck[N_of_model])
    # calc_LRM_metrics(float(0.0), float(0.0), **deck[N_of_model])
    
    return 0



def calc_bin_Sensitivities(**model_data):
    ###### Pre- knowledge:
    # Calculate 5*5 bin array for variables (LWP, CCFs) in Sounthern Ocean Region:
    #..set area-mean range and define function
    s_range = arange(-90., 90., 5.) + 2.5  #..global-region latitude edge: (36)
    x_range = arange(-180., 180., 5.)  #..logitude sequences edge: number: 72
    y_range = arange(-85, -40., 5.) +2.5  #..southern-ocaen latitude edge: 9

    path1 = '/glade/scratch/chuyan/CMIP_output/CMIP_lrm_RESULT/'
    path6 = '/glade/scratch/chuyan/Plots/CMIP_R_lwp_3/'
    path_plot = '/glade/work/chuyan/Research/Cloud_CCFs_RMs/Tan_et_al_2023_JGR-A__/plot_file/Explore_1/'
    ###### 
    
    if model_data['cmip'] == 'cmip6':

        inputVar_pi, inputVar_abr = get_LWPCMIP6(**model_data)

    elif model_data['cmip'] == 'cmip5':

        inputVar_pi, inputVar_abr = get_LWPCMIP5(**model_data)
    else:

        print('not cmip6 & cmip5 data.')



    #..get the shapes of monthly data
    shape_lat = len(inputVar_pi['lat'])
    shape_lon = len(inputVar_pi['lon'])
    shape_time_pi = len(inputVar_pi['times'])
    shape_time_abr = len(inputVar_abr['times'])
    #print(shape_lat, shape_lon, shape_time_pi, shape_time_abr)


    #..choose lat 40 -85 °S as the Southern-Ocean Regions
    lons = inputVar_pi['lon'] *1.
    lats = inputVar_pi['lat'][:] *1.

    levels = np.array(inputVar_abr['pres'] )
    times_pi = inputVar_pi['times'] *1.
    times_abr = inputVar_abr['times'] *1.

    lati1 = -40.
    latsi1 = min(range(len(lats)), key = lambda i: abs(lats[i] - lati1))
    lati0 = -85.
    latsi0 = min(range(len(lats)), key = lambda i: abs(lats[i] - lati0))
    print('lat index for -85S; -40S', latsi0, latsi1)

    shape_latSO = (latsi1+1) - latsi0
    print('shape of latitudinal index in raw data: ', shape_latSO)


    # Read the Radiation data and LWP
    # piControl and abrupt-4xCO2
    # LWP
    LWP_pi = np.array(inputVar_pi['clwvi']) - np.array(inputVar_pi['clivi'])   #..units in kg m^-2
    LWP_abr = np.array(inputVar_abr['clwvi']) - np.array(inputVar_abr['clivi'])   #..units in kg m^-2

    # abnormal 'Liquid Water Path' value:
    if np.min(LWP_abr) < -1e-3:
        LWP_abr = np.array(inputVar_abr['clwvi'])
        print('abr4x clwvi mislabeled')

    if np.min(LWP_pi) < -1e-3:
        LWP_pi = np.array(inputVar_pi['clwvi'])
        print('piControl clwvi mislabeled')

    # IWP
    IWP_pi = np.array(inputVar_pi['clivi'])   #..units in kg m^-2
    IWP_abr = np.array(inputVar_abr['clivi'])   #..units in kg m^-2

    # Global mean surface air temperature
    gmt_abr = array(inputVar_abr['tas'])
    gmt_pi = array(inputVar_pi['tas'])

    # SW radiation metrics
    Rsdt_pi = np.array(inputVar_pi['rsdt'])
    Rsut_pi = np.array(inputVar_pi['rsut'])
    Rsutcs_pi = np.array(inputVar_pi['rsutcs'])

    Rsdt_abr = np.array(inputVar_abr['rsdt'])
    Rsut_abr = np.array(inputVar_abr['rsut'])
    Rsutcs_abr = np.array(inputVar_abr['rsutcs'])

    # albedo, albedo_clear sky; albedo(alpha)_cre: all-sky - clear-sky
    Albedo_pi = Rsut_pi / Rsdt_pi
    Albedo_cs_pi = Rsutcs_pi / Rsdt_pi
    Alpha_cre_pi = Albedo_pi - Albedo_cs_pi

    Albedo_abr = Rsut_abr / Rsdt_abr
    Albedo_cs_abr = Rsutcs_abr / Rsdt_abr
    Alpha_cre_abr = Albedo_abr - Albedo_cs_abr

    # Pre-processing the data with abnormal values
    Albedo_abr[(Albedo_cs_abr <= 0.08) & (Albedo_cs_abr >= 1.00)] = np.nan
    Albedo_cs_abr[(Albedo_cs_abr <= 0.08) & (Albedo_cs_abr >= 1.00)] = np.nan
    Alpha_cre_abr[(Albedo_cs_abr <= 0.08) & (Albedo_cs_abr >= 1.00)] = np.nan
    LWP_abr[(Albedo_cs_abr <= 0.08) & (Albedo_cs_abr >= 1.00)] = np.nan
    LWP_abr[LWP_abr >= np.nanpercentile(LWP_abr, 99.5)] = np.nan
    IWP_abr[(Albedo_cs_abr <= 0.08) & (Albedo_cs_abr >= 1.00)] = np.nan
    IWP_abr[IWP_abr >= np.nanpercentile(IWP_abr, 99.5)] = np.nan
    Rsdt_abr[(Albedo_cs_abr <= 0.08) & (Albedo_cs_abr >= 1.00)] = np.nan

    Albedo_pi[(Albedo_cs_pi <= 0.08) & (Albedo_cs_pi >= 1.00)] = np.nan
    Albedo_cs_pi[(Albedo_cs_pi <= 0.08) & (Albedo_cs_pi >= 1.00)] = np.nan
    Alpha_cre_pi[(Albedo_cs_pi <= 0.08) & (Albedo_cs_pi >= 1.00)] = np.nan
    LWP_pi[(Albedo_cs_pi <= 0.08) & (Albedo_cs_pi >= 1.00)] = np.nan
    LWP_pi[LWP_pi >= np.nanpercentile(LWP_pi, 99.5)] = np.nan
    IWP_pi[(Albedo_cs_pi <= 0.08) & (Albedo_cs_pi >= 1.00)] = np.nan
    IWP_pi[IWP_pi >= np.nanpercentile(IWP_pi, 99.5)] = np.nan
    Rsdt_pi[(Albedo_cs_pi <= 0.08) & (Albedo_cs_pi >= 1.00)] = np.nan

    # Making a data dictionary:
    datavar_nas = ['LWP', 'IWP', 'rsdt', 'rsut', 'rsutcs', 'albedo', 'albedo_cs', 'alpha_cre']   #..7 varisables except gmt (lon dimension diff)

    dict0_PI_var = {'LWP': LWP_pi, 'IWP': IWP_pi, 'gmt': gmt_pi, 'rsdt': Rsdt_pi, 'rsut': Rsut_pi, 'rsutcs': Rsutcs_pi, 'albedo' : Albedo_pi, 'albedo_cs': Albedo_cs_pi, 'alpha_cre': Alpha_cre_pi, 'lat': lats, 'lon': lons, 'times': times_pi, 'pres': levels}

    dict0_abr_var = {'LWP': LWP_abr, 'IWP': IWP_abr, 'gmt': gmt_abr, 'rsdt': Rsdt_abr, 'rsut': Rsut_abr, 'rsutcs': Rsutcs_abr, 'albedo': Albedo_abr, 'albedo_cs': Albedo_cs_abr, 'alpha_cre': Alpha_cre_abr, 'lat': lats, 'lon': lons, 'times': times_abr, 'pres': levels}

    dict1_PI_var = deepcopy(dict0_PI_var)
    dict1_abr_var = deepcopy(dict0_abr_var)

    print('month in piControl and abrupt-4xCO2: ', times_pi[0,:][1], times_abr[0,:][1])


    # Choose regional frame: SO (40 ~ 85 .S)
    for c in range(len(datavar_nas)):

        dict1_PI_var[datavar_nas[c]] = dict1_PI_var[datavar_nas[c]][:, latsi0:latsi1+1, :]   # Southern Ocean data
        dict1_abr_var[datavar_nas[c]] = dict1_abr_var[datavar_nas[c]][:, latsi0:latsi1+1, :]  # Southern Ocean data

    dict1_PI_var['gmt'] = dict1_PI_var['gmt']  # Global
    dict1_abr_var['gmt'] = dict1_abr_var['gmt']  # Global

    # Copy data from dictionary:

    rsdt = deepcopy(dict1_abr_var['rsdt'])
    albedo = deepcopy(dict1_abr_var['albedo'])
    ck_albedo = deepcopy(dict1_abr_var['albedo_cs'])
    lwp = deepcopy(dict1_abr_var['LWP'])
    iwp = deepcopy(dict1_abr_var['IWP'])

    # conditions 1:
    rsdt[rsdt < 10.0] = np.nan
    ck_albedo[ck_albedo < 0.] = np.nan
    ck_albedo[ck_albedo >= 0.25] = np.nan

    # Processing 'nan' in aggregated data:
    Z_training = (rsdt * albedo * ck_albedo * lwp * iwp) * 1.
    ind_false = np.isnan(Z_training)
    ind_true = np.logical_not(ind_false)

    albedo_gcm = albedo[ind_true].flatten()
    # print(albedo_gcm)
    ck_albedo_gcm = ck_albedo[ind_true].flatten()
    # print(ck_albedo_gcm)
    lwp_gcm = lwp[ind_true].flatten()
    # print(lwp_gcm)
    iwp_gcm = iwp[ind_true].flatten()
    # print(iwp_gcm)
    
    

    # Calculate the LWP and IWP for the mean state and 121-140yrs-average of abrupt4xCO2 simulations:

    f20yr_index = 121*12
    l20yr_index = 140*12

    #  
    LWP_all_abr = deepcopy(dict1_abr_var['LWP'])
    IWP_all_abr = deepcopy(dict1_abr_var['IWP'])
    gmt_all_abr = deepcopy(dict1_abr_var['gmt'])
    LWP_all_PI = deepcopy(dict1_PI_var['LWP'])
    IWP_all_PI = deepcopy(dict1_PI_var['IWP'])
    gmt_all_PI = deepcopy(dict1_PI_var['gmt'])

    LWP_average_abr = np.nanmean(area_mean(LWP_all_abr[f20yr_index:l20yr_index,:,:], lats[latsi0:latsi1+1], lons))
    LWP_average_pi = np.nanmean(area_mean(LWP_all_PI, lats[latsi0:latsi1+1], lons))

    IWP_average_abr = np.nanmean(area_mean(IWP_all_abr[f20yr_index:l20yr_index,:,:], lats[latsi0:latsi1+1], lons))
    IWP_average_pi = np.nanmean(area_mean(IWP_all_PI, lats[latsi0:latsi1+1], lons))

    gmt_average_abr = np.nanmean(area_mean(gmt_all_abr[f20yr_index:l20yr_index,:,:], lats, lons))
    gmt_average_pi = np.nanmean(area_mean(gmt_all_PI, lats, lons))

    delta_LWP_delta_gmt_GCM = ((LWP_average_abr - LWP_average_pi) / (gmt_average_abr - gmt_average_pi))  # 40 - 85^{o}S

    delta_IWP_delta_gmt_GCM = ((IWP_average_abr - IWP_average_pi) / (gmt_average_abr - gmt_average_pi))  # 40 - 85^{o}S
    

    print('Average piControl and abrupt4xC02 LWP: ', 1000.* LWP_average_pi, 1000.* LWP_average_abr)
    print('Average piControl and abrupt4xC02 IWP: ', 1000.* IWP_average_pi, 1000.* IWP_average_abr)
    print(r'$\Delta LWP/Delta GMT\ and\ \Delta IWP/Delta GMT:\ $', 1000.* delta_LWP_delta_gmt_GCM, 1000.* delta_IWP_delta_gmt_GCM)
    
    
    
    # Ploting GCM: density plot of albedo versus LWP/IWP + box plot for sensitivities of albedo to LWP and IWP:

    from matplotlib import cm
    import statsmodels.api as sm
    import seaborn as sns
    sns.reset_defaults()
    
    # plot settings:
    parameters = {'axes.labelsize': 23, 'legend.fontsize': 16,  
           'axes.titlesize': 15, 'xtick.labelsize': 16, 'ytick.labelsize': 16}
    plt.rcParams.update(parameters)

    fig2, ax21 = plt.subplots(1, 2, figsize = (11.0, 6), gridspec_kw={'width_ratios': [3, 1]}, tight_layout = True)

    # print(np.nanmax(lwp_gcm), np.nanmin(lwp_gcm))
    # print(np.max(iwp_gcm), np.min(iwp_gcm))
    # print(np.max(albedo_gcm), np.min(albedo_gcm))

    # Subplot (1):
    ax21[0].set_aspect(1)

    # binned data by LWP:
    BIN_lwp = np.linspace(0., 220., 51)
    BIN_iwp = np.linspace(0., 220., 51)

    statistic_count, xedge, yedge, binnumber = binned_statistic_2d(1000.* lwp_gcm, 1000.* iwp_gcm, albedo_gcm, 'count', bins=[BIN_lwp, BIN_iwp])
    albedo_mean, xedge, yedge, binnumber = binned_statistic_2d(1000.* lwp_gcm, 1000.* iwp_gcm, albedo_gcm, 'mean', bins=[BIN_lwp, BIN_iwp])
    ck_albedo_mean, xedge, yedge, binnumber = binned_statistic_2d(1000.* lwp_gcm, 1000.* iwp_gcm, ck_albedo_gcm, 'mean', bins=[BIN_lwp, BIN_iwp])

    X_lwp = (BIN_lwp[0:-1] + (BIN_lwp[1] - BIN_lwp[0]) / 2.)
    Y_iwp = (BIN_iwp[0:-1] + (BIN_iwp[1] - BIN_iwp[0]) / 2.)

    # print(statistic_count, binnumber)
    denc2 = ax21[0].imshow(albedo_mean.T, origin = 'lower', cmap = 'rainbow', vmin = 0.14, vmax = 0.46+0.12)
    ax21[0].annotate('*', xycoords = 'axes fraction', xy = (LWP_average_pi*1000./220., IWP_average_pi*1000./220.), color = 'black', fontsize = 15, zorder = 99)
    ax21[0].annotate('*', xycoords = 'axes fraction', xy = (LWP_average_abr*1000./220., IWP_average_abr*1000./220.), color = 'black', fontsize = 15, zorder = 99)

    ax21[0].set_xticks(np.arange(0, len(BIN_lwp), 10))
    ax21[0].set_xticklabels(labels= np.round(BIN_lwp[0::10]))
    ax21[0].set_yticks(np.arange(0, len(BIN_iwp), 10))
    # ax21[0].set_yticklabels(labels = np.round(BIN_alphacs[0::20], 2))
    ax21[0].set_yticklabels(labels = np.round(BIN_iwp[0::10]))

    ax21[0].set_xlabel("$LWP,\ [g/m^{2}]$", fontsize = 22)
    ax21[0].set_ylabel(r"$IWP,\ [g/m^{2}]$", fontsize = 22)
    ax21[0].set_xlim(0, 51-1)
    ax21[0].set_ylim(0, 51-1)
    cb2 = fig2.colorbar(denc2, ax = ax21[0], shrink = 0.90, aspect = 12)
    cb2.set_label(r"$\ \alpha$", rotation = 360, fontsize = 23)
    plt.minorticks_off()

    ax21[0].set_title(r"$\ \Delta LWP/\Delta GMT \ = \ {0} \ g/m^{1}/K\ $" "\n" r"$\ \Delta IWP/\Delta GMT \ = \ {2} \ g/m^{3}/K\ $".format(
        np.round(1000.* delta_LWP_delta_gmt_GCM, 2), 2, np.round(1000.* delta_IWP_delta_gmt_GCM, 2), 2), loc ='left')


    # Subplot (2):

    n_MAX = 50
    coef_LWP = np.zeros((n_MAX))
    coef_IWP = np.zeros((n_MAX))
    for i in range(n_MAX):
        # Hold IWP be constant bin
        ind_true_x = np.isnan(albedo_mean.T[i, :]) == False

        # X = X_lwp[ind_true_x]
        X = np.column_stack((X_lwp[ind_true_x], ck_albedo_mean.T[i, ind_true_x]))  # add clear-sky albedo as a predictor
        Y = albedo_mean.T[i, ind_true_x]

        # print(X)
        # print(Y)
        # print(len(X))
        if len(X) > 0:
            X1 = sm.add_constant(X)
            model1 = sm.OLS(Y, X1)
            results_lwp = model1.fit()
            coef_LWP[i] = results_lwp.params[1]

        # Hold LWP be constant bin
        ind_true_y = np.isnan(albedo_mean.T[:, i]) == False

        # X = Y_iwp[ind_true_y]
        X = np.column_stack((Y_iwp[ind_true_y], ck_albedo_mean.T[ind_true_y, i]))  # add clear-sky albedo as a predictor
        Y = albedo_mean.T[ind_true_y, i]

        # print(X)
        # print(Y)

        if len(X) > 0:
            X2 = sm.add_constant(X)
            model2 = sm.OLS(Y, X2)
            results_iwp = model2.fit()
            coef_IWP[i] = results_iwp.params[1]

    coef_LWP[coef_LWP == 0.0] = np.nan
    coef_IWP[coef_IWP == 0.0] = np.nan
    print(coef_LWP)
    print(coef_IWP)

    # Data Frame:
    a = np.column_stack((coef_LWP, coef_IWP))
    # print(a.shape)
    d1 = {'col1': np.arange(0, 50*2), 'value': a.ravel(), 'variable': np.array([r"$ \frac{\partial \alpha}{\partial LWP} $", r"$ \frac{\partial \alpha}{\partial IWP} $"]* 50)}
    data1  = pd.DataFrame(data=d1, index=arange(0, 50 * 2))  # radiative sensitivities to LWP and IWP:

    # Making Box Plot:
    bplot01 = sns.boxplot(ax=ax21[1], x = "variable", y = "value", data = d1, width = 0.45, linewidth = 2.6, whis = 5.0)

    ax21[1].yaxis.tick_right()
    ax21[1].yaxis.set_label_position("right")
    ax21[1].set_title(r"$\partial \alpha/\partial (LWP/IWP) $", fontsize = 15)
    ax21[1].set_ylabel(r"$[{(g/m^{2})}^{-1}]$", fontsize = 21)
    ax21[1].set_xlabel(bplot01.get_xlabel(), size = 23)
    # bplot01.set_yticklabels(bplot01.get_yticks(), size = 17)

    plt.suptitle(model_data['modn'], fontsize = 22, x = 0.6)
    # plt.show()

    plt.savefig(path_plot + model_data['modn']+"_Radiation_sensitivities.jpg", bbox_inches = 'tight', dpi = 300)
    
    
    savedata_path = '/glade/scratch/chuyan/CMIP_output/CMIP_radiative_sensitivities_RESULT/'
    
    Radiative_sen = {'SWalbedo_LWP':np.nanmedian(coef_LWP), 'SWalbedo_IWP': np.nanmedian(coef_IWP)}
    Dclouds_Dgmt = {'DLWP_DGMT': 1000.* delta_LWP_delta_gmt_GCM, 'DIWP_DGMT': 1000.* delta_IWP_delta_gmt_GCM}
    np.savez(savedata_path + model_data['modn'] + '_RSENSITIVITIES' + '_dats', model_data = model_data['modn'], Radiative_sen = Radiative_sen, Dclouds_Dgmt = Dclouds_Dgmt)

    
    return None

if __name__== "__main__":
    main()