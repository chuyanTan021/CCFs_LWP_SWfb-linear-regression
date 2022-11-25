#..Create by Chuyan at Nov18th, this file was intended to plot the Figures reuqired by thefirst paper in a better way..

import netCDF4 as nc
# from numpy import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
# import PyNIO as Nio #deprecated
import pandas as pd
import glob
from scipy.stats import *
from copy import deepcopy
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm

from area_mean import *
from scipy.optimize import curve_fit
import seaborn as sns
from copy import deepcopy
from useful_func_cy import *
from fitLRM_cy1 import *
from fitLRM_cy2 import *
from calc_Radiation_LRM_1 import *
from calc_Radiation_LRM_2 import *
from calc_LRM_metric import *
from calc_LRMobs_metric import *
from fitLRMobs import *

from get_LWPCMIP5data import *
from get_LWPCMIP6data import *
from get_OBSLRMdata import *

from Aploting_Sep11 import *

# 12 cmip6 model: deck_nas = ['BCCESM1', 'CanESM5', 'CESM2', 'CESM2FV2', 'CESM2WACCM', 'CNRMESM2', 'GISSE21G', 'GISSE21H', 'IPSLCM6ALR', 'MRIESM20', 'MIROC6', 'SAM0']

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

# basic setting:

# cmip5 + cmip6
deck2 = [BCCESM1, CanESM5, CESM2, CESM2FV2, CESM2WACCM, CNRMESM21, GISSE21G, GISSE21H, IPSLCM6ALR, MRIESM20, MIROC6, SAM0, E3SM10, FGOALSg3, GFDLCM4, CAMSCSM1, INM_CM48, MPIESM12LR, AWICM11MR, CMCCCM2SR5, CESM2WACCMFV2, CNRMCM61, CNRMCM61HR, ECEarth3, ECEarth3Veg, GISSE22G, MIROCES2L, NESM3, NorESM2MM, TaiESM1, BNUESM, CCSM4, CNRMCM5, CSIRO_Mk360, CanESM2, FGOALSg2, FGOALSs2, GFDLCM3, GISSE2H, GISSE2R, IPSLCM5ALR, MIROC5, MPIESMMR, NorESM1M, MIROCESM, MRICGCM3, MPIESMLR, bcccsm11, GFDLESM2G, GFDLESM2M]  # current # 30 + 20 = 50
deck_nas2 = ['BCCESM1', 'CanESM5', 'CESM2', 'CESM2FV2', 'CESM2WACCM', 'CNRMESM21', 'GISSE21G', 'GISSE21H', 'IPSLCM6ALR', 'MRIESM20', 'MIROC6', 'SAM0', 'E3SM10', 'FGOALSg3', 'GFDLCM4', 'CAMSCSM1', 'INM_CM48', 'MPIESM12LR', 'AWICM11MR', 'CMCCCM2SR5', 'CESM2WACCMFV2', 'CNRMCM61', 'CNRMCM61HR', 'ECEarth3', 'ECEarth3Veg', 'GISSE22G', 'MIROCES2L', 'NESM3', 'NorESM2MM', 'TaiESM1', 'BNUESM', 'CCSM4', 'CNRMCM5', 'CSIRO_Mk360', 'CanESM2', 'FGOALSg2', 'FGOALSs2', 'GFDLCM3', 'GISSE2H', 'GISSE2R', 'IPSLCM5ALR', 'MIROC5', 'MPIESMMR', 'NorESM1M', 'MIROCESM', 'MRICGCM3', 'MPIESMLR', 'bcccsm11', 'GFDLESM2G', 'GFDLESM2M']  # current # 30 + 20 = 50

deck_cmip6 = [AWICM11MR, BCCESM1, CanESM5, CESM2, CESM2FV2, CESM2WACCM, CESM2WACCMFV2, CMCCCM2SR5, CNRMESM21, CNRMCM61, CNRMCM61HR, E3SM10, ECEarth3, ECEarth3Veg, FGOALSg3, GFDLCM4, CAMSCSM1, IPSLCM6ALR, INM_CM48, MPIESM12LR, MRIESM20, GISSE21G, GISSE22G, GISSE21H, MIROC6, MIROCES2L, NESM3, NorESM2MM, SAM0, TaiESM1]   #..current # 18 + 12

deck_nas_cmip6 = ['AWICM11MR', 'BCCESM1', 'CanESM5', 'CESM2', 'CESM2FV2', 'CESM2WACCM', 'CESM2WACCMFV2', 'CMCCCM2SR5', 'CNRMESM2', 'CNRMCM61', 'CNRMCM61HR', 'E3SM10', 'ECEarth3', 'ECEarth3Veg', 'FGOALSg3', 'GFDLCM4', 'CAMSCSM1', 'IPSLCM6ALR', 'INM_CM48', 'MPIESM12LR', 'MRIESM20', 'GISSE21G', 'GISSE21H', 'GISSE22G', 'MIROC6', 'MIROCES2L', 'NESM3', 'NorESM2MM', 'SAM0', 'TaiESM1']

# Calculate 5*5 bin array for variables (LWP, CCFs) in Sounthern Ocean Region:
#..set are-mean range and define function
s_range = arange(-90., 90., 5.) + 2.5  #..global-region latitude edge: (36)
x_range = arange(-180., 180., 5.)  #..logitude sequences edge: number: 72
y_range = arange(-85, -40., 5.) + 2.5  #..southern-ocaen latitude edge: 9

#.. current model #: 18 + 12 (except: '19')
path_data = '/glade/scratch/chuyan/CMIP_output/CMIP_lrm_RESULT/'
path_plot = '/glade/work/chuyan/Research/Cloud_CCFs_RMs/Course_objective_ana/plot_file/plots_Oct24_revise_add_YSSAR/'



def Fig1a_base(s_range, x_range, y_range, deck2 = deck2, deck_nas2 = deck_nas2, path1 = path_data, path6 = path_plot):
    #.. Extra-tropical Latitudinal distribution of SWfb (a)..
    # Read CMIP5/ 6 Cloud feedbacks map data:

    fn_cmip5 = '/glade/work/chuyan/Research/Cloud_CCFs_RMs/Course_objective_ana/CMIP5_cld_fbks.nc'
    fn_cmip6 = '/glade/work/chuyan/Research/Cloud_CCFs_RMs/Course_objective_ana/CMIP6_cld_fbkd_July12.nc'
    
    # glob.glob(fn_cmip5)
    f_cmip5 = nc.Dataset(fn_cmip5, 'r')
    f_cmip6 = nc.Dataset(fn_cmip6, 'r')
    # print(f_cmip5.variables['model'])
    # print(f_cmip6.variables)


    # variables from Zelinka's SWfb files:
    lat_mz = np.asarray(f_cmip6.variables['latitude'])
    bound_lat_mz = np.asarray(f_cmip6.variables['bounds_latitude'])

    lon_mz = np.asarray(f_cmip6.variables['longitude'])
    bound_lon_mz = np.asarray(f_cmip6.variables['bounds_longitude'])

    # convert longitude matrix from (0, 360) to (-180., 180.):
    lon_mz2 = lon_mz[:]*1.
    bound_lon_mz2 = bound_lon_mz[:] * 1.
    lon_mz2[lon_mz2 > 180] = lon_mz2[lon_mz2 > 180]-360.
    bound_lon_mz2[bound_lon_mz2 > 180] = bound_lon_mz2[bound_lon_mz2 > 180] - 360
    ind_lon = argsort(lon_mz2)
    lon_mz2 = lon_mz2[ind_lon]

    bound_lon_mz2 = bound_lon_mz2[ind_lon, :]
    # print(lon_mz2)
    # print(bound_lon_mz2)
    
    # model_names:
    cmip6model_nas = f_cmip6.variables['model'].long_name
    cmip5model_nas = f_cmip5.variables['model'].long_name

    cmip6_nas = cmip6model_nas[1:-1].split()
    for i in range(len(cmip6_nas)):
        cmip6_nas[i] = cmip6_nas[i][1:-1]

    cmip5_nas = cmip5model_nas[1:-1].split()
    for i in range(len(cmip5_nas)):
        cmip5_nas[i] = cmip5_nas[i][1:-1]
    # print(cmip6_nas)
    # print(cmip5_nas)

    model_nas = []
    model_nas = np.append(cmip6_nas, cmip5_nas)
    print(model_nas)

    # SW_cloud_Feedback:
    sw_cld_fb_mz = []

    cmip6_sw_cld_fb_mz = np.asarray(f_cmip6.variables['SWCLD_fbk6_map'])
    cmip5_sw_cld_fb_mz = np.asarray(f_cmip5.variables['SWCLD_fbk5_map'])

    sw_cld_fb_mz = np.append(cmip6_sw_cld_fb_mz, cmip5_sw_cld_fb_mz, axis = 2)
    print('sw_cld_fb_mz shape:', sw_cld_fb_mz.shape)

    # Handle nan value
    sw_cld_fb_mz = np.where(sw_cld_fb_mz!=1.e+20, sw_cld_fb_mz, np.nan)
    ind_nan = np.isnan(sw_cld_fb_mz)
    # print(np.nonzero(ind_nan==True))  # #0

    reshape_sw_cld_fb = np.transpose(sw_cld_fb_mz,(2, 0, 1))

    print('reshape_sw_cld_fb shape:', reshape_sw_cld_fb.shape)
    
    # Southern Ocean regional map of SWfb (50 GCMs):
    # SO_sw_cldfb = area_mean(reshape_sw_cld_fb[:,(latsi0):(latsi1+1+1),:], y_range_swcld, x_range_swcld) # -85.S ~ -40.S
    SO_sw_cldfb = latitude_mean(reshape_sw_cld_fb[:,:,:], lat_mz, lon_mz, lat_range=[-85., -40.])
    # SO_sw_cldfb_5085 = area_mean(reshape_sw_cld_fb[:,(latsi0):(latsi2+1+1),:], y_range_swcld5085, x_range_swcld) # -85.S ~ -50.S
    SO_sw_cldfb_5085 = latitude_mean(reshape_sw_cld_fb[:,:,:], lat_mz, lon_mz, lat_range=[-85., -50.])
    # SO_sw_cldfb_4050 = area_mean(reshape_sw_cld_fb[:,(latsi2+1):(latsi1+1+1),:], y_range_swcld4050, x_range_swcld) # -50.S ~ -40.S
    SO_sw_cldfb_4050 = latitude_mean(reshape_sw_cld_fb[:,:,:], lat_mz, lon_mz, lat_range=[-50., -40.])
    # print(SO_sw_cldfb)
    
    SW_FB_4085 = []
    SW_FB_5085 = []
    SW_FB_4050 = []

    for i in range(len(deck2)):
        for j in range(len(SO_sw_cldfb)):
            # 50
            if (deck2[i]['modn']== model_nas[j]):
                # if (i in modelconstraintbystep1_nas):
                SW_FB_4085 = np.append(SW_FB_4085, SO_sw_cldfb[j])
                SW_FB_4050 = np.append(SW_FB_4050, SO_sw_cldfb_4050[j])
                SW_FB_5085 = np.append(SW_FB_5085, SO_sw_cldfb_5085[j])
                
    SWCLD_specific_models = []
    
    # get the SW cloud feedback map data for our models list:
    for i in range(len(deck2)):
        for j in range(reshape_sw_cld_fb.shape[0]):

            if (deck2[i]['modn'] == model_nas[j]):

                SWCLD_specific_models.append(reshape_sw_cld_fb[j])

    SWCLD_specific_models = np.asarray(SWCLD_specific_models)
    # print(reshape_sw_cld_fb[61])
    # print(SWCLD_specific_models[49])

    # processing NaN values:
    ind_false_SWCLD = np.isnan(SWCLD_specific_models)
    SWCLD_withoutNaN = deepcopy(SWCLD_specific_models)
    SWCLD_withoutNaN[ind_false_SWCLD] = 999

    # convert SWCLD metric longitude from (0, 360) to (-180., 180.):
    SWCLD_withoutNaN = SWCLD_withoutNaN[:, :, ind_lon]
    
    print(np.asarray(np.nonzero(ind_false_SWCLD == True)).shape)
    print(SWCLD_withoutNaN[:,2, 3].shape)

      
    # import Mark's data for CMIP5, CMIP6 models' EffCS, SWCLD..:
    import json
    
    f = open('cmip56_forcing_feedback_ecs.json','r')
    data = json.load(f)
    
    # read through CMIP5 + CMIP6 EffCS and SWfb values:
    EffCS = {}
    SWCLD = {}

    for i in range(len(deck_nas2)):

        if deck2[i]['cmip'] == 'cmip5':
            EffCS[deck_nas2[i]] = data['CMIP5'][deck2[i]['modn']][deck2[i]['ensmem']]['ECS']
            SWCLD[deck_nas2[i]] = data['CMIP5'][deck2[i]['modn']][deck2[i]['ensmem']]['SWCLD']

        if deck2[i]['cmip'] == 'cmip6':

            if deck2[i]['modn'] == 'EC-Earth3':
                EffCS[deck_nas2[i]] = data['CMIP6'][deck2[i]['modn']]['r8i1p1f1']['ECS']
                SWCLD[deck_nas2[i]] = data['CMIP6'][deck2[i]['modn']]['r8i1p1f1']['SWCLD']
            else:
                EffCS[deck_nas2[i]] = data['CMIP6'][deck2[i]['modn']][deck2[i]['ensmem']]['ECS']
                SWCLD[deck_nas2[i]] = data['CMIP6'][deck2[i]['modn']][deck2[i]['ensmem']]['SWCLD']
    
    # Plotting: 
    
    # Re-plot of Fig 1: SW_FB in raw resolutions vs. Lat of extra-tropical:
    
    import matplotlib
    parameters = {'axes.labelsize': 16, 'legend.fontsize': 14,
             'axes.titlesize': 22,  'xtick.labelsize': 16,  'ytick.labelsize': 16}
    plt.rcParams.update(parameters)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.minorticks_on()
    ax1.tick_params(axis='x', which='major', direction='out', top=False, length= 5., width = 1., reset = True)
    ax1.tick_params(axis='x', which='minor', direction='out', top=False, length= 3., width = 1., reset = True)

    ax2.minorticks_on()
    ax2.tick_params(axis='x', which='major', direction='out', top=False, length= 5., width = 1., reset = True)
    ax2.tick_params(axis='x', which='minor', direction='out', top=False, length= 3., width = 1., reset = True)


    print('SWCLD_withoutNaN shape:', SWCLD_withoutNaN.shape)

    p50_MM = np.full((SWCLD_withoutNaN.shape[1], 2), 0.00)
    p95_MM = np.full((SWCLD_withoutNaN.shape[1], 2), 0.00)

    # print(lat_mz)
    lat_range1 = [-85., -30.] 
    lat_range2 = [ 30.,  85.]
    ind1 = (lat_mz <=max(lat_range1)) & (lat_mz >= min(lat_range1))
    ind2 = (lat_mz <=max(lat_range2)) & (lat_mz >= min(lat_range2))


    lonmean_MLAT_swcld = np.nanmean(SWCLD_withoutNaN, axis = (2))
    for i in range(SWCLD_withoutNaN.shape[1]):
        p50_MM[i, 0] = np.nanpercentile(lonmean_MLAT_swcld[:, i], 25.)
        p95_MM[i, 0] = np.nanpercentile(lonmean_MLAT_swcld[:, i], 0.)
        p50_MM[i, 1] = np.nanpercentile(lonmean_MLAT_swcld[:, i], 75.)
        p95_MM[i, 1] = np.nanpercentile(lonmean_MLAT_swcld[:, i], 100.)
    mean_LAT_swcld = np.nanmean(SWCLD_withoutNaN, axis = (0, 2))

    ECS = []
    for i in range(len(deck_nas2)):
        ECS.append(EffCS[deck_nas2[i]])
    ECS = np.asarray(ECS)

    sorted_EffCS_index = sorted(range(50), key = lambda index:ECS[index])
    ordinal_EffCS_index = [sorted_EffCS_index.index(i) for i in range(50)]
    # print(sorted_EffCS_index)
    # print(ordinal_EffCS_index)


    # Colors:
    coolwarm_colormap = matplotlib.cm.get_cmap("coolwarm")
    COLORS = [coolwarm_colormap(x) for x in np.linspace(0.05, 0.95, num=50)]
    COLORS = [matplotlib.colors.to_hex(color) for color in COLORS]
    print(COLORS)

    for i in range(len(deck_nas2)):

        ss = SWCLD_withoutNaN[i, :,:]
        xx = np.nanmean(ss, axis = 1)
        # Remove the NaN value in 1-D vector:
        ind_truevector = isnan(xx)==False

        ind1_true = np.logical_and(ind1, ind_truevector)
        ind2_true = np.logical_and(ind2, ind_truevector)
        # print(arange(min(lat_range1), max(lat_range1), 2.))
        ax1.plot(arange(min(lat_range1), max(lat_range1), 2.), xx[ind1_true], color = COLORS[ordinal_EffCS_index[i]], linestyle = '-')
        ax1.plot(arange(min(lat_range1), max(lat_range1), 2.), mean_LAT_swcld[ind1_true], linewidth = 2.5, c = 'k', linestyle = '-')

        ax2.plot(arange(min(lat_range2), max(lat_range2), 2.), xx[ind2_true], color = COLORS[ordinal_EffCS_index[i]], linestyle = '-')
        ax2.plot(arange(min(lat_range2), max(lat_range2), 2.), mean_LAT_swcld[ind2_true], linewidth = 2.5, c = 'k', linestyle = '-')

    # Shading:

    ax1.plot(np.arange(min(lat_range1), max(lat_range1), 2.), p50_MM[:, 0][ind1_true], linestyle = '-', c= 'white', linewidth = 4.0, zorder = 97)
    ax1.plot(np.arange(min(lat_range1), max(lat_range1), 2.), p50_MM[:, 1][ind1_true], linestyle = '-', c= 'white', linewidth = 2.8, zorder = 97)
    # ax1.plot(np.arange(min(lat_range1), max(lat_range1), 2.), p95_MM[:, 0][ind1_true], linestyle = '-', c= 'white', linewidth = 1.4, zorder = 97)
    # ax1.plot(np.arange(min(lat_range1), max(lat_range1), 2.), p95_MM[:, 1][ind1_true], linestyle = '-', c= 'white', linewidth = 1.4, zorder = 97)
    ax1.fill_between(np.arange(min(lat_range1), max(lat_range1), 2.), p50_MM[:, 0][ind1_true], p50_MM[:, 1][ind1_true], color = 'gray', alpha = 0.45, zorder = 98)
    # ax1.fill_between(np.arange(min(lat_range1), max(lat_range1), 2.), p95_MM[:, 0][ind1_true], p95_MM[:, 1][ind1_true], color = 'gray', alpha = 0.30, zorder = 99)

    ax2.plot(np.arange(min(lat_range2), max(lat_range2), 2.), p50_MM[:, 0][ind2_true], linestyle = '-', c= 'white', linewidth = 4.0, zorder = 97)
    ax2.plot(np.arange(min(lat_range2), max(lat_range2), 2.), p50_MM[:, 1][ind2_true], linestyle = '-', c= 'white', linewidth = 2.8, zorder = 97)
    ax2.fill_between(np.arange(min(lat_range2), max(lat_range2), 2.), p50_MM[:, 0][ind2_true], p50_MM[:, 1][ind2_true], color = 'gray', alpha = 0.45, zorder = 98)
    # ax2.fill_between(np.arange(min(lat_range2), max(lat_range2), 2.), p95_MM[:, 0][ind2_true], p95_MM[:, 1][ind2_true], color = 'gray', alpha = 0.30, zorder = 99)

    # Plot setting:
    ax1.set_xlim(-85., -30.)
    ax1.set_xticks(np.arange(-80., -30., 20.))
    ax1.set_ylim(-4.25, 6.25)

    ax2.set_xlim(30., 85.)
    ax2.set_xticks(np.arange(40., 85., 20.))
    ax2.set_ylim(-4.25, 6.25)

    ax1.set_xlabel(r"$lat$")
    ax1.set_ylabel(r"$SW\ Cloud\ Feedback,\ [W/m^{2}/K] $")

    ax2.set_xlabel(r"$lat$")
    # ax2.set_ylabel(r"$SW\ Cloud\ Feedback,\ [W/m^{2}/K] $")

    plt.savefig(path6 +"Fig1:(a).jpg", bbox_inches = 'tight', dpi = 425)
    
    
    return 0


def Fig1b_base(s_range, x_range, y_range, deck2 = deck2, deck_nas2 = deck_nas2, path1 = path_data, path6 = path_plot):
    
    #.. Extra-tropical Latitudinal distribution of \Delta LWP/\Delta gmt (b)..
    
    # import numpy as np
    from scipy.stats import *

    # define the curve function which you intend to fit:
    def target_func_poly1(x, a, b):
        '''
        linear fit
        '''
        y = a* x + b
        return y

    def target_func_poly2(x, a, b, c): 
        '''
        2d polynomial fit
        '''
        y = a* x**2 + b* x**1 + c
        return y

    def target_func_expo(x, a, b, c):

        '''
        exponential fitting
        '''
        y = a * b**x + c
        return y
    
    
    ## Read 50 GCMs raw resolution and 5X5 bin data:

    path2 = '/glade/scratch/chuyan/CMIP_output/CMIP_5X5bin_DATA/'
    output_ARRAY_5x5 = {}   # storage output file

    # Raw data
    output_dict0_PI = {}
    output_dict0_abr = {}

    # Metric raw data in specific units:
    output_PI_bin_var = {}
    output_abr_bin_var = {}
    shape_mon_pi = {}
    shape_mon_abr = {}

    Tr_sst =  0.0

    for i in range(len(deck2)):
        # print("i", i)
        folder_5x5 = glob.glob(path2 + deck2[i]['modn'] + '_raw_5X5bin_Nov11th_'+'_dats.npz')
        print(len(folder_5x5))

        if len(folder_5x5) == 4:
            if (len(folder_5x5[0]) < len(folder_5x5[1])) & (len(folder_5x5[0]) < len(folder_5x5[2])) & (len(folder_5x5[0]) < len(folder_5x5[3])):
                folder_bes_5x5 = folder_5x5[0]
            elif (len(folder_5x5[1]) < len(folder_5x5[0])) & (len(folder_5x5[1]) < len(folder_5x5[2])) & (len(folder_5x5[1]) < len(folder_5x5[3])):
                folder_bes_5x5 = folder_5x5[1]
            elif (len(folder_5x5[2]) < len(folder_5x5[0])) & (len(folder_5x5[2]) < len(folder_5x5[1])) & (len(folder_5x5[2]) < len(folder_5x5[3])):
                folder_bes_5x5 = folder_5x5[2]
            else:
                folder_bes_5x5 = folder_5x5[3]
            print(folder_bes_5x5)

        elif len(folder_5x5) == 3:
            if (len(folder_5x5[1]) <  len(folder_5x5[0])) & (len(folder_5x5[1]) <  len(folder_5x5[2])):
                folder_bes_5x5 = folder_5x5[1]
            elif (len(folder_5x5[0]) <  len(folder_5x5[1])) & (len(folder_5x5[0]) <  len(folder_5x5[2])):
                folder_bes_5x5 = folder_5x5[0]
            else:
                folder_bes_5x5 = folder_5x5[2]
            print(folder_bes_5x5)

        elif len(folder_5x5) == 2:
            if len(folder_5x5[1]) <  len(folder_5x5[0]):
                folder_bes_5x5 = folder_5x5[1]
            else:
                folder_bes_5x5 = folder_5x5[0]
            print(folder_bes_5x5)

        else:
            output_ARRAY_5x5[deck_nas2[i]] = load(folder_5x5[0], allow_pickle = True)  #+'_'+str(Tr_sst)
            print(folder_5x5[0])

        # output_ARRAY[deck_nas2[i]] =  load(folder_bes_5x5, allow_pickle=True)  #+'_'+str(Tr_sst)

        # output_ARRAY[deck_nas2[i]] = load(folder_5x5[0], allow_pickle = True)  #+'_'+str(Tr_sst)
        # output_intermedia[deck_nas2[i]] = output_ARRAY_5x5[deck_nas2[i]]['rawdata_dict']

        output_dict0_PI[deck_nas2[i]] = output_ARRAY_5x5[deck_nas2[i]]['dict0_PI_var']
        output_dict0_abr[deck_nas2[i]] = output_ARRAY_5x5[deck_nas2[i]]['dict0_abr_var']

        # Monthly data
        output_PI_bin_var[deck_nas2[i]] = output_ARRAY_5x5[deck_nas2[i]]['dict1_PI_bin_var']
        output_abr_bin_var[deck_nas2[i]] = output_ARRAY_5x5[deck_nas2[i]]['dict1_abr_bin_var']
        # Annually data

        # Flattened Metric monthly mean bin data
        # shape_mon_pi[deck_nas2[i]] = output_dict0_PI[deck_nas2[i]][()]['shape_mon_PI_3']
        # shape_mon_abr[deck_nas2[i]] = output_dict0_abr[deck_nas2[i]][()]['shape_mon_abr_3']
        
    print('Down read 5 x 5.')
    
    # calc the mean state LWP and delta LWP and delta P-E in global 5 x 5 bin resolutions;

    delta_LWP_dTg_ALL_5X5 = full((len(deck2), 36, 72), 0.0)  # GLOBAL lwp changes
    mean_LWP_ALL_5X5 = full((len(deck2), 36, 72), 0.0)  # GLOBAL Mean State lwp
    delta_P_E_dTg_ALL_5X5 = full((len(deck2), 36, 72), 0.0)  # Global moisture convergence changes
    
    # global mean surface air Temperature (gmt):
    delta_gmt = full(len(deck2), 0.000)
    
    
    f20yr_index = 121*12
    l20yr_index = 141*12
    
    # gmt.
    for i in range(len(deck_nas2)):
        
        delta_gmt[i] = np.nanmean(area_mean(output_abr_bin_var[deck_nas2[i]][()]['gmt_mon_bin'][f20yr_index:l20yr_index, :,:], s_range, x_range)) - np.nanmean(area_mean(output_PI_bin_var[deck_nas2[i]][()]['gmt_mon_bin'], s_range, x_range)) 
    
    # \Delta LWP scaled by gmt; mean State LWP; and \Delta moisture convergence scaled by gmt:
    for i in range(len(deck_nas2)):
        delta_LWP_dTg_ALL_5X5[i] = (np.nanmean(output_abr_bin_var[deck_nas2[i]][()]['LWP_mon_bin'][f20yr_index:l20yr_index,:,:], axis = 0) - np.nanmean(output_PI_bin_var[deck_nas2[i]][()]['LWP_mon_bin'], axis = 0)) / delta_gmt[i]
        
        mean_LWP_ALL_5X5[i] = deepcopy(np.nanmean(output_PI_bin_var[deck_nas2[i]][()]['LWP_mon_bin'], axis = 0))
        
        delta_P_E_dTg_ALL_5X5[i] = (np.nanmean(output_abr_bin_var[deck_nas2[i]][()]['p_e_mon_bin'][f20yr_index:l20yr_index,:,:], axis = 0) - np.nanmean(output_PI_bin_var[deck_nas2[i]][()]['p_e_mon_bin'], axis = 0)) / delta_gmt[i]
    
    
    # import Mark's data for CMIP5, CMIP6 models' EffCS, SWCLD..:
    import json
    
    f = open('cmip56_forcing_feedback_ecs.json','r')
    data = json.load(f)
    
    # read through CMIP5 + CMIP6 EffCS and SWfb values:
    EffCS = {}
    SWCLD = {}

    for i in range(len(deck_nas2)):

        if deck2[i]['cmip'] == 'cmip5':
            EffCS[deck_nas2[i]] = data['CMIP5'][deck2[i]['modn']][deck2[i]['ensmem']]['ECS']
            SWCLD[deck_nas2[i]] = data['CMIP5'][deck2[i]['modn']][deck2[i]['ensmem']]['SWCLD']

        if deck2[i]['cmip'] == 'cmip6':

            if deck2[i]['modn'] == 'EC-Earth3':
                EffCS[deck_nas2[i]] = data['CMIP6'][deck2[i]['modn']]['r8i1p1f1']['ECS']
                SWCLD[deck_nas2[i]] = data['CMIP6'][deck2[i]['modn']]['r8i1p1f1']['SWCLD']
            else:
                EffCS[deck_nas2[i]] = data['CMIP6'][deck2[i]['modn']][deck2[i]['ensmem']]['ECS']
                SWCLD[deck_nas2[i]] = data['CMIP6'][deck2[i]['modn']][deck2[i]['ensmem']]['SWCLD']
    
    # Plotting: 
    
    # Re-plot Fig 1: \Delta LWP/ \Delta GMT in 5X5 degree vs. Extratropical Lat:

    import matplotlib
    parameters = {'axes.labelsize': 16, 'legend.fontsize': 14,
             'axes.titlesize': 22,  'xtick.labelsize': 16,  'ytick.labelsize': 16}
    plt.rcParams.update(parameters)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.minorticks_on()
    ax1.tick_params(axis='x', which='major', direction='out', top=False, length= 5., width = 1., reset = True)
    ax1.tick_params(axis='x', which='minor', direction='out', top=False, length= 3., width = 1., reset = True)

    ax2.minorticks_on()
    ax2.tick_params(axis='x', which='major', direction='out', top=False, length= 5., width = 1., reset = True)
    ax2.tick_params(axis='x', which='minor', direction='out', top=False, length= 3., width = 1., reset = True)

    p50_MM = np.full((len(s_range), 2), 0.00)
    p95_MM = np.full((len(s_range), 2), 0.00)

    # print(lat_mz)
    lat_range1 = [-85., -30.] 
    lat_range2 = [ 30.,  85.]
    ind1 = (s_range <=max(lat_range1)) & (s_range >= min(lat_range1))
    ind2 = (s_range <=max(lat_range2)) & (s_range >= min(lat_range2))
    print(ind1, ind2)

    lonmean_MLAT_DlwpDgmt = np.nanmean(delta_LWP_dTg_ALL_5X5, axis = (2))

    for i in range(len(s_range)):

        p50_MM[i, 0] = np.nanpercentile(lonmean_MLAT_DlwpDgmt[:, i], 25.)
        p95_MM[i, 0] = np.nanpercentile(lonmean_MLAT_DlwpDgmt[:, i], 2.5)
        p50_MM[i, 1] = np.nanpercentile(lonmean_MLAT_DlwpDgmt[:, i], 75.)
        p95_MM[i, 1] = np.nanpercentile(lonmean_MLAT_DlwpDgmt[:, i], 97.5)
    mean_LAT_DlwpDgmt = np.nanmean(delta_LWP_dTg_ALL_5X5, axis = (0, 2))

    ECS = []
    for i in range(len(deck_nas2)):
        ECS.append(EffCS[deck_nas2[i]])
    ECS = np.asarray(ECS)
    sorted_EffCS_index = sorted(range(50), key = lambda index:ECS[index])
    ordinal_EffCS_index = [sorted_EffCS_index.index(i) for i in range(50)]
    # print(sorted_EffCS_index)
    # print(ordinal_EffCS_index)

    # Colors:
    coolwarm_colormap = matplotlib.cm.get_cmap("coolwarm")
    COLORS = [coolwarm_colormap(x) for x in np.linspace(0.05, 0.95, num=50)]
    COLORS = [matplotlib.colors.to_hex(color) for color in COLORS]
    print(COLORS)

    for i in range(len(deck_nas2)):

        ss = delta_LWP_dTg_ALL_5X5[i, :,:]
        xx = nanmean(ss, axis = 1)
        # Remove the NaN value in 1-D vector:
        ind_truevector = isnan(xx)==False

        ind1_true = np.logical_and(ind1, ind_truevector)
        ind2_true = np.logical_and(ind2, ind_truevector)
        # print(arange(min(lat_range1), max(lat_range1), 2.))
        ax1.plot(np.arange(min(lat_range1), max(lat_range1), 5.), xx[ind1_true], color = COLORS[ordinal_EffCS_index[i]], linestyle = '-')
        ax1.plot(np.arange(min(lat_range1), max(lat_range1), 5.), mean_LAT_DlwpDgmt[ind1_true], linewidth = 2.5, c = 'k', linestyle = '-')

        ax2.plot(np.arange(min(lat_range2), max(lat_range2), 5.), xx[ind2_true], color = COLORS[ordinal_EffCS_index[i]], linestyle = '-')
        ax2.plot(np.arange(min(lat_range2), max(lat_range2), 5.), mean_LAT_DlwpDgmt[ind2_true], linewidth = 2.5, c = 'k', linestyle = '-')

    # Shading: 

    ax1.plot(np.arange(min(lat_range1), max(lat_range1), 5.), p50_MM[:, 0][ind1_true], linestyle = '-', c= 'white', linewidth = 4.0, zorder = 97)
    ax1.plot(np.arange(min(lat_range1), max(lat_range1), 5.), p50_MM[:, 1][ind1_true], linestyle = '-', c= 'white', linewidth = 2.8, zorder = 97)
    # ax1.plot(np.arange(min(lat_range1), max(lat_range1), 5.), p95_MM[:, 0][ind1_true], linestyle = '-', c= 'white', linewidth = 1.4, zorder = 97)
    # ax1.plot(np.arange(min(lat_range1), max(lat_range1), 5.), p95_MM[:, 1][ind1_true], linestyle = '-', c= 'white', linewidth = 1.4, zorder = 97)
    ax1.fill_between(np.arange(min(lat_range1), max(lat_range1), 5.), p50_MM[:, 0][ind1_true], p50_MM[:, 1][ind1_true], color = 'gray', alpha = 0.45, zorder = 98)
    # ax1.fill_between(np.arange(min(lat_range1), max(lat_range1), 5.), p95_MM[:, 0][ind1_true], p95_MM[:, 1][ind1_true], color = 'gray', alpha = 0.35, zorder = 99)

    ax2.plot(np.arange(min(lat_range2), max(lat_range2), 5.), p50_MM[:, 0][ind2_true], linestyle = '-', c= 'white', linewidth = 4.0, zorder = 97)
    ax2.plot(np.arange(min(lat_range2), max(lat_range2), 5.), p50_MM[:, 1][ind2_true], linestyle = '-', c= 'white', linewidth = 2.8, zorder = 97)
    ax2.fill_between(np.arange(min(lat_range2), max(lat_range2), 5.), p50_MM[:, 0][ind2_true], p50_MM[:, 1][ind2_true], color = 'gray', alpha = 0.45, zorder = 98)
    # ax2.fill_between(np.arange(min(lat_range2), max(lat_range2), 5.), p95_MM[:, 0][ind2_true], p95_MM[:, 1][ind2_true], color = 'gray', alpha = 0.35, zorder = 99)

    # Plot setting:
    ax1.set_xlim(-85., -30.)
    ax1.set_xticks(arange(-80., -30., 20.))
    ax1.set_ylim(-0.010, 0.017)

    ax2.set_xlim(30., 85.)
    ax2.set_xticks(arange(40., 85., 20.))
    ax2.set_ylim(-0.010, 0.017)

    ax1.set_xlabel(r"$lat$")
    ax1.set_ylabel(r"$\Delta LWP / \Delta GMT,\ [kg/m^{2}/K] $")

    ax2.set_xlabel(r"$lat$")
    # ax2.set_ylabel(r"$\Delta LWP / \Delta GMT,\ [kg/m^{2}/K] $")


    plt.savefig(path6 +"Fig1:(b)_5X5.jpg", bbox_inches = 'tight', dpi = 425)
    
    
    
    '''
    # Plotting FigS1: Annually time series of GCMs’ LWP averaged over 40 — 85S vs. GMT averaged over globally;
    
    # Calc the 40--85 averaged annually changes in LWP; globally averaged annually changes in GMT; colored by \Delta LWP/ \Delta GMT;

    an_delta_LWP_raw = full((len(deck_nas2), 150), 0.0)  # annually time series of 40 -- 85 averaged changes in LWP for GCMs;
    an_delta_GMT = full((len(deck_nas2), 150), 0.0)  # annually time series of globally changes in GMT for GCMs;
    
    for i in range(len(deck_nas2)):
        lat_range1 = [-85., -40.] 
        ind1 = (output_dict0_abr[deck_nas2[i]][()]['lat'] <=max(lat_range1)) & (output_dict0_abr[deck_nas2[i]][()]['lat'] >= min(lat_range1))

        an_delta_LWP_raw[i,:] = (area_mean(annually_mean(output_dict0_abr[deck_nas2[i]][()]['LWP'][:, ind1,:], output_dict0_abr[deck_nas2[i]][()]['times'], label = 'mon'), output_dict0_abr[deck_nas2[i]][()]['lat'][ind1], output_dict0_abr[deck_nas2[i]][()]['lon'])[0:150] - np.nanmean(area_mean(annually_mean(output_dict0_PI[deck_nas2[i]][()]['LWP'][:, ind1,:], output_dict0_PI[deck_nas2[i]][()]['times']), output_dict0_PI[deck_nas2[i]][()]['lat'][ind1], output_dict0_PI[deck_nas2[i]][()]['lon'])))
        
        an_delta_GMT[i,:] = (area_mean(annually_mean(output_dict0_abr[deck_nas2[i]][()]['gmt'], output_dict0_abr[deck_nas2[i]][()]['times'], label = 'mon'), output_dict0_abr[deck_nas2[i]][()]['lat'], output_dict0_abr[deck_nas2[i]][()]['lon'])[0:150] - np.nanmean(area_mean(annually_mean(output_dict0_PI[deck_nas2[i]][()]['gmt'], output_dict0_PI[deck_nas2[i]][()]['times']), output_dict0_PI[deck_nas2[i]][()]['lat'], output_dict0_PI[deck_nas2[i]][()]['lon'])))
    
    
    # Plotting: 
    # print(an_delta_GMT)
    # print(an_delta_LWP_raw)

    import matplotlib
    parameters = {'axes.labelsize': 15, 'legend.fontsize': 14,
             'axes.titlesize': 21,  'xtick.labelsize': 16,  'ytick.labelsize': 16}
    plt.rcParams.update(parameters)

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.5))

    # print(s_range[1:10])
    DLWP_Dgmt = area_mean(delta_LWP_dTg_ALL_5X5[:, 1:10, :], s_range[1:10], x_range)
    # print(DLWP_Dgmt)
    DLWP_Dgmt = np.asarray(DLWP_Dgmt)

    sorted_DLWP_Dgmt_index = sorted(range(50), key = lambda index:DLWP_Dgmt[index])
    ordinal_DLWP_Dgmt_index = [sorted_DLWP_Dgmt_index.index(i) for i in range(50)]
    # print(sorted_DLWP_Dgmt_index)
    # print(ordinal_DLWP_Dgmt_index)

    # Colors:
    inferno_colormap = matplotlib.cm.get_cmap("coolwarm_r")
    COLORS = [inferno_colormap(x) for x in np.linspace(0.01, 0.97, num=50)]
    COLORS = [matplotlib.colors.to_hex(color) for color in COLORS]
    # print(COLORS)

    for i in range(len(deck2)):
        x = an_delta_GMT[i]
        ax.scatter(x, an_delta_LWP_raw[i], c = COLORS[ordinal_DLWP_Dgmt_index[i]], marker = 'x', s= 1)
        POPT_gmtLWP, POCV_gmtLWP = curve_fit(target_func_poly2, an_delta_GMT[i], an_delta_LWP_raw[i])
        # print(POPT_gmtLWP)
        ax.plot(x, POPT_gmtLWP[0] * x**2 + POPT_gmtLWP[1] * x**1 + POPT_gmtLWP[2], color = COLORS[ordinal_DLWP_Dgmt_index[i]], linewidth = 1.6, linestyle = '-', alpha = 0.8, zorder = 99)

    # Plot setting:

    ax.set_xlim(0., 9.4)
    ax.set_xticks(arange(0, 10, 1))
    ax.set_ylim(-0.012, 0.07)
    ax.set_yticks(arange(-0.01, 0.08, 0.01))
    ax.set_xlabel(r"$GMT\ -\ GMT|_{mean\ state},\ [K]$")
    ax.set_ylabel(r"$LWP\ -\ LWP|_{mean\ state},\ [kg/m^{2}]$")

    ax.set_title(r"$LWP\ _{40^{o} -85^{o} S}$", )

    # plt.savefig(path6 +"Figs1:annuallytimeseries_DLWP(SO)_Dgmt(globally).jpg", bbox_inches = 'tight', dpi = 425)
    
    '''
    return 0




def Fig3_base(s_range, x_range, y_range, deck2 = deck2, deck_nas2 = deck_nas2, path1 = path_data, path6 = path_plot):
    ## OBS out-of-sample Test on Delta LWP_obs, and the decomposition (individual ccf components.).
    
    
    # training
    valid_range1=[2009, 1, 15]
    valid_range2=[2016, 12, 31]   # 8 years
    # Predicting
    valid_range3=[1997, 1, 15]
    valid_range4=[2008, 12, 31]   # 12 years

    # 'valid_range1' and 'valid_range2' give the time stamps of starting and ending times of data for training,
    # 'valid_range3' and 'valid_range4' give the time stamps of starting and ending times of data for predicting.
    # 'THRESHOLD_sst' is the cut-off of 'Sea surface temperature' for partitioning the 'Hot'/'Cold' LRM regimes;
    # 'THRESHOLD_sub' is the cut-off of '500 mb Vertical Velocity (Pressure)' for partitioning 'Up'/'Down' regimes.
    # ..
    # ------------------
    # Southern Ocean 5 * 5 degree bin box
    # Using to do area_mean
    s_range = arange(-90., 90., 5.) + 2.5  #..global-region latitude edge: (36)
    x_range = arange(-180., 180., 5.)  #..logitude sequences edge: number: 72
    y_range = arange(-85, -40., 5.) + 2.5  #..southern-ocaen latitude edge: 9


    # Function #1 loopping through variables space to find the cut-offs of LRM (Multi-Linear Regression Model).
    dict_training, lats_Array, lons_Array, times_Array_training = Pre_processing(s_range, x_range, y_range, valid_range1 = valid_range1, valid_range2 = valid_range2)

    dict_predict, lats_Array, lons_Array, times_Array_predict = Pre_processing(s_range, x_range, y_range, valid_range1 = valid_range3, valid_range2 = valid_range4)

    predict_result_1r = fitLRMobs_1(dict_training, dict_predict, s_range, y_range, x_range, lats_Array, lons_Array)
    
    time_Array_training = times_Array_training
    time_Array_predict = times_Array_predict

    lats = y_range
    lons = x_range
    data_Array_actual_predict = predict_result_1r['LWP_actual_predict']
    data_Array_actual_training = predict_result_1r['LWP_actual_training']
    data_Array_predict_predict = predict_result_1r['LWP_predi_predict']
    data_Array_predict_training = predict_result_1r['LWP_predi_training']
    running_mean_window = 2
    
    # Plotting:

    parameters = {'axes.labelsize': 15, 'legend.fontsize': 14,
             'axes.titlesize': 18,  'xtick.labelsize': 16,  'ytick.labelsize': 16}
    plt.rcParams.update(parameters)

    coef_dict = predict_result_1r['coef_dict']
    # print(coef_dict.shape)
    df_CCF1 = pd.DataFrame({'SST': area_mean(np.append(predict_result_1r['predict_Array']['SST'].reshape(dict_predict['LWP'].shape), predict_result_1r['training_Array']['SST'].reshape(dict_training['LWP'].shape), axis = 0), lats, lons)})  # *1000.
    df_CCF2 = pd.DataFrame({'p_e': area_mean(np.append(predict_result_1r['predict_Array']['p_e'].reshape(dict_predict['LWP'].shape), predict_result_1r['training_Array']['p_e'].reshape(dict_training['LWP'].shape), axis = 0), lats, lons)})  # *1000.
    df_CCF3 = pd.DataFrame({'LTS': area_mean(np.append(predict_result_1r['predict_Array']['LTS'].reshape(dict_predict['LWP'].shape), predict_result_1r['training_Array']['LTS'].reshape(dict_training['LWP'].shape), axis = 0), lats, lons)})  # *1000.
    df_CCF4 = pd.DataFrame({'SUB': area_mean(np.append(predict_result_1r['predict_Array']['SUB'].reshape(dict_predict['LWP'].shape), predict_result_1r['training_Array']['SUB'].reshape(dict_training['LWP'].shape), axis = 0), lats, lons)})  # *1000.
    output_CCF1 = df_CCF1.rolling((12* running_mean_window + 1), min_periods = 1, center = True).mean()
    output_CCF2 = df_CCF2.rolling((12* running_mean_window + 1), min_periods = 1, center = True).mean()
    output_CCF3 = df_CCF3.rolling((12* running_mean_window + 1), min_periods = 1, center = True).mean()
    output_CCF4 = df_CCF4.rolling((12* running_mean_window + 1), min_periods = 1, center = True).mean()

    output_time = np.arange(0, time_Array_training.shape[0] + time_Array_predict.shape[0], 1)
    df_actual = pd.DataFrame({'A': area_mean(np.append(data_Array_actual_predict, data_Array_actual_training, axis = 0), lats, lons)})  # *1000.
    output_actual = df_actual.rolling((12* running_mean_window + 1), min_periods = 1, center = True).mean()

    df_predict = pd.DataFrame({'B': area_mean(np.append(data_Array_predict_predict, data_Array_predict_training, axis = 0), lats, lons) })  #*1000.
    output_predict = df_predict.rolling((12* running_mean_window + 1), min_periods = 1, center = True).mean()

    fig = plt.figure( figsize = (9.3, 7.65))
    gs = fig.add_gridspec(2, hspace = 0.137)
    axs = gs.subplots(sharex=True, sharey=True)

    # Hide the right and top spines:
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)

    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)

    # Only show ticks on the left and bottom spines
    axs[1].yaxis.set_ticks_position('left')
    axs[1].xaxis.set_ticks_position('bottom')
    axs[0].xaxis.set_ticks_position('none')

    axs[0].plot(output_time, output_actual, label = r'$OBS\ LWP$', alpha = 1.0, linewidth= 3.40, linestyle = '-', c = 'green', zorder = 98)
    axs[0].plot(output_time, output_predict, label = r'$CCF\ Predicted\ LWP$', alpha = 1.0, linewidth= 3.40, linestyle = '-', c = 'b', zorder = 98)
    axs[0].axvline(time_Array_predict.shape[0], linestyle = '--', linewidth = 2.0, c = 'k')

    axs[1].plot(output_time, coef_dict[0][0] * output_CCF1, label = r'$SST$', alpha = 1.0, linewidth= 1.60, c = 'k', linestyle = '--')
    axs[1].plot(output_time, coef_dict[0][1] * output_CCF2, label = r'$P-E$', alpha = 1.0, linewidth= 1.60, c = 'cyan', linestyle = '--')
    axs[1].plot(output_time, coef_dict[0][2] * output_CCF3, label = r'$LTS$', alpha = 1.0, linewidth= 1.60, c = 'purple', linestyle = '--')
    axs[1].plot(output_time, coef_dict[0][3] * output_CCF4, label = r'$SUB_{500}$', alpha = 1.0, linewidth= 1.60, c = 'red', linestyle = '--')
    # Hide x labels and tick labels for all but bottom plot.
    for ax in axs:
        ax.label_outer()

    axs[0].legend()
    axs[1].legend()
    axs[1].set_xticks(output_time[0::12])
    axs[1].set_xticklabels((np.append(np.arange(time_Array_predict[0, 0], time_Array_predict[0, 0] + time_Array_predict.shape[0]//12, 1), np.arange(time_Array_training[0, 0], time_Array_training[0, 0] + time_Array_training.shape[0]//12, 1))).astype(int), rotation = 45)
    # axs[0].set_xlabel(' Time ')
    axs[0].set_ylabel(r"$\Delta LWP,\ $" + r"$ [std \cdot dev^{-1}]$")  # kg*m^{-2} 
    axs[1].set_ylabel(r"$\Delta X_{i} \cdot \partial LWP/ \partial X_{i},\ $" + r"$ [std \cdot dev^{-1}]$")  # kg*m^{-2}
    axs[0].set_title( "(a) Southern Ocean LWP", fontsize = 17)
    axs[1].set_title( "(b) Southern Ocean CCFs contribution", fontsize = 18)
    # fig.suptitle(" Observational trend of LWP from MAC-LWP ")

    # plt.legend()
    # plt.show()
    plt.savefig(path6 + 'Fig3.jpg', bbox_inches = 'tight', dpi = 425)
    
    return 0


def Fig4_base(s_range, x_range, y_range, deck2 = deck2, deck_nas2 = deck_nas2, path1 = path_data, path6 = path_plot):
    ## Constraints on the \Delta LWP/ \Delta GMT at individual latitudinal band:
    
    ## Read two Regimes (Hot,Cold) data

    output_ARRAY = {}   # storage output file
    output_intermedia = {}   # storage the 'rawdata_dict'

    output_dict0_PI = {}
    output_dict0_abr = {}

    output_GMT = {}
    output_2lrm_predict = {}  # dict, store annualy, area_meaned prediction of LWP
    output_2lrm_report = {}  # dict, store annually, area_meaned actual values of GCMs LWP
    output_2lrm_coef_LWP = {}
    output_2lrm_dict_Albedo = {}  # Coefficients of 2 regimes's albedo trained by report 'LWP' data
    # output_2lrm_coef_albedo_lL = {}

    # Raw data
    output_2lrm_yr_bin_abr = {}
    output_2lrm_yr_bin_PI = {}
    output_2lrm_mon_bin_abr = {}
    output_2lrm_mon_bin_PI = {}

    # Metric raw data in specific units:
    shape_mon_pi = {}
    shape_mon_abr = {}
    output_2lrm_metric_actual_PI = {}
    output_2lrm_metric_actual_abr = {}

    # Statistic metrics of PI:
    output_Mean_training = {}
    output_Stdev_training = {}

    # Predict metric data in specific units:
    output_2lrm_mon_bin_LWPpredi_PI = {}
    output_2lrm_mon_bin_LWPpredi_abr = {}

    # Index for regime(s): Only for 2lrm
    output_ind_Cold_PI = {}
    output_ind_Warm_PI = {}
    output_ind_Cold_abr = {}
    output_ind_Warm_abr = {}

    Tr_sst =  0.0

    for i in range(len(deck2)):
        # print("i", i)
        folder_2lrm = glob.glob(path1+deck2[i]['modn'] + '_r2r1_hotcold(Jan)_(largestpiR2)_Sep9th_Anomalies_Rtest' + '*' + '_dats.npz')
        print(len(folder_2lrm))

        if len(folder_2lrm) == 4:
            if (len(folder_2lrm[0]) < len(folder_2lrm[1])) & (len(folder_2lrm[0]) < len(folder_2lrm[2])) & (len(folder_2lrm[0]) < len(folder_2lrm[3])):
                folder_best2lrm = folder_2lrm[0]
            elif (len(folder_2lrm[1]) < len(folder_2lrm[0])) & (len(folder_2lrm[1]) < len(folder_2lrm[2])) & (len(folder_2lrm[1]) < len(folder_2lrm[3])):
                folder_best2lrm = folder_2lrm[1]
            elif (len(folder_2lrm[2]) < len(folder_2lrm[0])) & (len(folder_2lrm[2]) < len(folder_2lrm[1])) & (len(folder_2lrm[2]) < len(folder_2lrm[3])):
                folder_best2lrm = folder_2lrm[2]
            else:
                folder_best2lrm = folder_2lrm[3]
            print(folder_best2lrm)

        elif len(folder_2lrm) == 3:
            if (len(folder_2lrm[1]) <  len(folder_2lrm[0])) & (len(folder_2lrm[1]) <  len(folder_2lrm[2])):
                folder_best2lrm = folder_2lrm[1]
            elif (len(folder_2lrm[0]) <  len(folder_2lrm[1])) & (len(folder_2lrm[0]) <  len(folder_2lrm[2])):
                folder_best2lrm = folder_2lrm[0]
            else:
                folder_best2lrm = folder_2lrm[2]
            print(folder_best2lrm)

        elif len(folder_2lrm) == 2:
            if len(folder_2lrm[1]) <  len(folder_2lrm[0]):
                folder_best2lrm = folder_2lrm[1]
            else:
                folder_best2lrm = folder_2lrm[0]
            print(folder_best2lrm)

        else:
            output_ARRAY[deck_nas2[i]] = load(folder_2lrm[0], allow_pickle = True)  #+'_'+str(Tr_sst)
            print(folder_2lrm[0])

        output_ARRAY[deck_nas2[i]] = load(folder_best2lrm, allow_pickle = True)  #+'_'+str(Tr_sst)

        # output_ARRAY[deck_nas2[i]] = load(folder_2lrm[0], allow_pickle = True)  #+'_'+str(Tr_sst)
        output_intermedia[deck_nas2[i]] = output_ARRAY[deck_nas2[i]]['rawdata_dict']

        output_GMT[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['GMT']
        output_2lrm_predict[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['predicted_metrics']
        output_2lrm_report[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['report_metrics']

        output_dict0_PI[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['dict1_PI_var']
        output_dict0_abr[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['dict1_abr_var']

        output_2lrm_coef_LWP[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['Coef_dict']
        # print(output_2lrm_dict_Albedo, "i", i, output_intermedia[deck_nas2[i]][()].keys())
        output_2lrm_dict_Albedo[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['coef_dict_Albedo_pi']

        # Monthly data
        output_2lrm_mon_bin_PI[deck_nas2[i]] = output_dict0_PI[deck_nas2[i]]['dict1_mon_bin_PI']
        output_2lrm_mon_bin_abr[deck_nas2[i]] = output_dict0_abr[deck_nas2[i]]['dict1_mon_bin_abr']
        # Annually data
        output_2lrm_yr_bin_PI[deck_nas2[i]] = output_dict0_PI[deck_nas2[i]]['dict1_yr_bin_PI']
        output_2lrm_yr_bin_abr[deck_nas2[i]] = output_dict0_abr[deck_nas2[i]]['dict1_yr_bin_abr']

        # Flattened Metric monthly mean bin data
        shape_mon_pi[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['shape_mon_PI_3']
        shape_mon_abr[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['shape_mon_abr_3']
        output_2lrm_metric_actual_PI[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['metric_training']
        output_2lrm_metric_actual_abr[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['metric_predict']

        # Flattened Predicted monthly bin data
        output_2lrm_mon_bin_LWPpredi_PI[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['LWP_predi_bin_PI']
        output_2lrm_mon_bin_LWPpredi_abr[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['LWP_predi_bin_abr']

        # Statistic metrics of PI:
        output_Mean_training[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['Mean_training']
        output_Stdev_training[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['Stdev_training']

        # Indice for Regimes
        output_ind_Warm_PI[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['ind_Hot_PI']
        output_ind_Cold_PI[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['ind_Cold_PI']

        output_ind_Warm_abr[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['ind_Hot_abr']
        output_ind_Cold_abr[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['ind_Cold_abr']
    
    print('Down read 2-LRM.')
    
    
    
    # Values for calculate the GCMs' responses in each bands and partitioned into "Cold/ Warm":
    # Standard deviation of Cloud Controlling factor (Xi) and Liquid Water Path (LWP):

    sigmaXi_r1 = full((len(deck2), 4), 0.0)  # Cold
    sigmaXi_r2 = full((len(deck2), 4), 0.0)  # Warm

    sigmaLWP_r1 = full((len(deck2)), 0.0)  # Cold
    sigmaLWP_r2 = full((len(deck2)), 0.0)  # Warm
    sigmaLWP_ALL = full((len(deck2)), 0.0)  # Southern Ocean as a whole

    # Changes of variable between 'piControl' (mean-state) and 'abrupt4xCO2' (warming period, take 121 - 140 yrs' mean)
    # Cloud Controlling factor (Xi), Liquid Water Path (LWP), and global mean surface air Temperature (gmt):

    delta_gmt = full(len(deck2), 0.000)
    delta_SST = full((len(deck2), 2), 0.0)  # two Regimes, Cold & Warm
    delta_SST_4050 = full((len(deck2), 2), 0.0)
    delta_SST_5085 = full((len(deck2), 2), 0.0)
    delta_p_e = full((len(deck2), 2), 0.0)
    delta_p_e_4050 = full((len(deck2), 2), 0.0)
    delta_p_e_5085 = full((len(deck2), 2), 0.0)
    delta_LTS = full((len(deck2), 2), 0.0)
    delta_LTS_4050 = full((len(deck2), 2), 0.0)
    delta_LTS_5085 = full((len(deck2), 2), 0.0)
    delta_SUB = full((len(deck2), 2), 0.0)
    delta_SUB_4050 = full((len(deck2), 2), 0.0)
    delta_SUB_5085 = full((len(deck2), 2), 0.0)

    delta_LWP = full((len(deck2), 2), 0.0)
    delta_LWP_ALL = full((len(deck2)), 0.0)  # Southern Ocean lwp changes
    delta_LWP_4050 = full((len(deck2)), 0.0)  # 40 ~ 50^oS lwp changes
    delta_LWP_5085 = full((len(deck2)), 0.0)  # 50 ~ 85^oS lwp changes

    # Standardized changes of Variables
    # Cloud Controlling factor (Xi) scaled by 'gmt', Liquid Water Path (LWP):

    dX_dTg_r1 = full((len(deck2), 4), 0.0)  # Cold
    dX_dTg_r1_4050 = full((len(deck2), 4), 0.0)  # Cold regime at 40--50
    dX_dTg_r1_5085 = full((len(deck2), 4), 0.0)  # Cold regime at 50--85
    dX_dTg_r2 = full((len(deck2), 4), 0.0)  # Warm
    dX_dTg_r2_4050 = full((len(deck2), 4), 0.0)  # Warm regime at 40--50
    dX_dTg_r2_5085 = full((len(deck2), 4), 0.0)  # Warm regime at 50--85

    delta_LWP_dTg = full((len(deck2)), 0.0)  # Southern Ocean lwp changes scaled by gmt 
    delta_LWP_dTg_4050 = full((len(deck2)), 0.0)  # 40 ~ 50^oS lwp changes scaled by gmt
    delta_LWP_dTg_5085 = full((len(deck2)), 0.0)  # 50 ~ 85^oS lwp changes scaled by gmt
    delta_LWP_dTgr1 = full((len(deck2)), 0.0)  # Cold
    delta_LWP_dTgr2 = full((len(deck2)), 0.0)  # Warm

    # Coef of LWP to Cloud controlling factors, Xis, for two regimes
    # GCM values and the OBS values

    stcoef_r1 = full((len(deck2), 4), 0.0)  # Cold
    stcoef_r2 = full((len(deck2), 4), 0.0)  # Warm
    stcoef_obs = full((4), 0.0)  # Warm Regime Only
    
    from copy import deepcopy

    f20yr_index = 121*12
    l20yr_index = 140*12

    #..set are-mean range and define function
    s_range = arange(-90., 90., 5.) + 2.5  #..global-region latitude edge: (36)
    x_range = arange(-180., 180., 5.)  #..logitude sequences edge: number: 72
    y_range = arange(-85, -40., 5.) +2.5  #..southern-ocaen latitude edge: 9

    for i in range(len(deck_nas2)):

        # indice of Regimes;
        ind_Cold_PI = output_ind_Cold_PI[deck_nas2[i]]
        ind_Warm_PI = output_ind_Warm_PI[deck_nas2[i]]
        ind_Cold_abr = output_ind_Cold_abr[deck_nas2[i]]
        ind_Warm_abr = output_ind_Warm_abr[deck_nas2[i]]
        # print(ind_Cold_PI.shape)
        # print(ind_Warm_abr)

        # print(output_2lrm_metric_actual_PI[deck_nas2[i]]['SST'][ind_Hot_PI].shape)
        # calc standard_deviation for CCFs at the training period:

        sigmaXi_r1[i,:] = asarray( [nanstd(output_2lrm_metric_actual_PI[deck_nas2[i]]['SST'][ind_Cold_PI]), nanstd(output_2lrm_metric_actual_PI[deck_nas2[i]]['p_e'][ind_Cold_PI]), 
                             nanstd(output_2lrm_metric_actual_PI[deck_nas2[i]]['LTS'][ind_Cold_PI]), nanstd(output_2lrm_metric_actual_PI[deck_nas2[i]]['SUB'][ind_Cold_PI])])

        sigmaXi_r2[i,:] = asarray( [nanstd(output_2lrm_metric_actual_PI[deck_nas2[i]]['SST'][ind_Warm_PI]), nanstd(output_2lrm_metric_actual_PI[deck_nas2[i]]['p_e'][ind_Warm_PI]), 
                             nanstd(output_2lrm_metric_actual_PI[deck_nas2[i]]['LTS'][ind_Warm_PI]), nanstd(output_2lrm_metric_actual_PI[deck_nas2[i]]['SUB'][ind_Warm_PI])])

        sigmaLWP_r1[i] = nanstd(output_2lrm_metric_actual_PI[deck_nas2[i]]['LWP'][ind_Cold_PI])
        sigmaLWP_r2[i] = nanstd(output_2lrm_metric_actual_PI[deck_nas2[i]]['LWP'][ind_Warm_PI])
        sigmaLWP_ALL[i] = nanstd(output_2lrm_metric_actual_PI[deck_nas2[i]]['LWP'][logical_or(ind_Cold_PI, ind_Warm_PI)])

        # calc changes of variables in two different regimes:

        # indice for 'Hot' and 'Cold' regimes corresponding to the last period
        ind_last20_Cold_abr = deepcopy(output_ind_Cold_abr[deck_nas2[i]]).reshape(shape_mon_abr[deck_nas2[i]])
        ind_last20_Cold_abr[0:f20yr_index, :, :] = False
        ind_last20_Cold_abr[l20yr_index:, :, :] = False
        ind_last20_Warm_abr = deepcopy(output_ind_Warm_abr[deck_nas2[i]]).reshape(shape_mon_abr[deck_nas2[i]])
        ind_last20_Warm_abr[0:f20yr_index, :, :] = False
        ind_last20_Warm_abr[l20yr_index:, :, :] = False
        ind_last20_All_abr = logical_or(ind_last20_Cold_abr, ind_last20_Warm_abr)

        ind_last20_Cold_PI = deepcopy(output_ind_Cold_PI[deck_nas2[i]]).reshape(shape_mon_pi[deck_nas2[i]])
        ind_last20_Warm_PI = deepcopy(output_ind_Warm_PI[deck_nas2[i]]).reshape(shape_mon_pi[deck_nas2[i]])
        ind_last20_All_PI = logical_or(ind_last20_Cold_PI, ind_last20_Warm_PI)

        LWP_all_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['LWP']).reshape(shape_mon_abr[deck_nas2[i]])
        LWP_all_abr[logical_not(ind_last20_All_abr)] = np.nan

        LWP_all_PI = deepcopy(output_2lrm_metric_actual_PI[deck_nas2[i]]['LWP']).reshape(shape_mon_pi[deck_nas2[i]])
        LWP_all_PI[logical_not(ind_last20_All_PI)] = np.nan

        LWP_cold_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['LWP']).reshape(shape_mon_abr[deck_nas2[i]])
        LWP_cold_abr[logical_not(ind_last20_Cold_abr)] = np.nan
        LWP_warm_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['LWP']).reshape(shape_mon_abr[deck_nas2[i]])
        LWP_warm_abr[logical_not(ind_last20_Warm_abr)] = np.nan
        LWP_cold_PI = deepcopy(output_2lrm_metric_actual_PI[deck_nas2[i]]['LWP']).reshape(shape_mon_pi[deck_nas2[i]])
        LWP_cold_PI[logical_not(ind_last20_Cold_PI)] = np.nan
        LWP_warm_PI = deepcopy(output_2lrm_metric_actual_PI[deck_nas2[i]]['LWP']).reshape(shape_mon_pi[deck_nas2[i]])
        LWP_warm_PI[logical_not(ind_last20_Warm_PI)] = np.nan

        delta_LWP_ALL[i] = nanmean(area_mean(LWP_all_abr[f20yr_index:l20yr_index,:,:], y_range, x_range)) - nanmean(area_mean(LWP_all_PI, y_range, x_range))
        # print("Delta LWP 40 -- 85: ", delta_LWP_ALL[i])

        delta_LWP_4050[i] = nanmean(latitude_mean(LWP_all_abr[f20yr_index:l20yr_index,:, :], y_range, x_range, lat_range=[-50., -40.])) - nanmean(latitude_mean(LWP_all_PI, y_range, x_range, lat_range=[-50., -40.]))
        delta_LWP_5085[i] = nanmean(latitude_mean(LWP_all_abr[f20yr_index:l20yr_index,:, :], y_range, x_range, lat_range=[-85., -50.])) - nanmean(latitude_mean(LWP_all_PI, y_range, x_range, lat_range=[-85., -50.]))
        delta_LWP[i, 0] = nanmean(area_mean(LWP_cold_abr[f20yr_index:l20yr_index,:,:], y_range, x_range)) - nanmean(area_mean(LWP_cold_PI, y_range, x_range))
        delta_LWP[i, 1] = nanmean(area_mean(LWP_warm_abr[f20yr_index:l20yr_index,:,:], y_range, x_range)) - nanmean(area_mean(LWP_warm_PI, y_range, x_range))
        # print('1:', area_mean(LWP_cold_abr[f20yr_index:l20yr_index,:,:], y_range, x_range))
        # print('2:', area_mean(LWP_cold_PI[:,:,:], y_range, x_range))


        # SST.
        SST_cold_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['SST']).reshape(shape_mon_abr[deck_nas2[i]])
        SST_cold_abr[logical_not(ind_last20_Cold_abr)] = np.nan
        SST_warm_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['SST']).reshape(shape_mon_abr[deck_nas2[i]])
        SST_warm_abr[logical_not(ind_last20_Warm_abr)] = np.nan
        SST_cold_PI = deepcopy(output_2lrm_metric_actual_PI[deck_nas2[i]]['SST']).reshape(shape_mon_pi[deck_nas2[i]])
        SST_cold_PI[logical_not(ind_last20_Cold_PI)] = np.nan
        SST_warm_PI = deepcopy(output_2lrm_metric_actual_PI[deck_nas2[i]]['SST']).reshape(shape_mon_pi[deck_nas2[i]])
        SST_warm_PI[logical_not(ind_last20_Warm_PI)] = np.nan

        delta_SST[i, 0] = nanmean(area_mean(SST_cold_abr[f20yr_index:l20yr_index,:,:], y_range, x_range)) - nanmean(area_mean(SST_cold_PI, y_range, x_range))
        delta_SST[i, 1] = nanmean(area_mean(SST_warm_abr[f20yr_index:l20yr_index,:,:], y_range, x_range)) - nanmean(area_mean(SST_warm_PI, y_range, x_range))
        delta_SST_4050[i, 0] = nanmean(latitude_mean(SST_cold_abr[f20yr_index:l20yr_index,:,:], y_range, x_range, lat_range=[-50., -40.])) - nanmean(latitude_mean(SST_cold_PI, y_range, x_range, lat_range=[-50., -40.]))
        delta_SST_4050[i, 1] = nanmean(latitude_mean(SST_warm_abr[f20yr_index:l20yr_index,:,:], y_range, x_range, lat_range=[-50., -40.])) - nanmean(latitude_mean(SST_warm_PI, y_range, x_range, lat_range=[-50., -40.]))
        delta_SST_5085[i, 0] = nanmean(latitude_mean(SST_cold_abr[f20yr_index:l20yr_index,:,:], y_range, x_range, lat_range=[-85., -50.])) - nanmean(latitude_mean(SST_cold_PI, y_range, x_range, lat_range=[-85., -50.]))
        delta_SST_5085[i, 0] = nanmean(latitude_mean(SST_warm_abr[f20yr_index:l20yr_index,:,:], y_range, x_range, lat_range=[-85., -50.])) - nanmean(latitude_mean(SST_warm_PI, y_range, x_range, lat_range=[-85., -50.]))

        # P - E.
        p_e_cold_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['p_e']).reshape(shape_mon_abr[deck_nas2[i]])
        p_e_cold_abr[logical_not(ind_last20_Cold_abr)] = np.nan
        p_e_warm_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['p_e']).reshape(shape_mon_abr[deck_nas2[i]])
        p_e_warm_abr[logical_not(ind_last20_Warm_abr)] = np.nan
        p_e_cold_PI = deepcopy(output_2lrm_metric_actual_PI[deck_nas2[i]]['p_e']).reshape(shape_mon_pi[deck_nas2[i]])
        p_e_cold_PI[logical_not(ind_last20_Cold_PI)] = np.nan
        p_e_warm_PI = deepcopy(output_2lrm_metric_actual_PI[deck_nas2[i]]['p_e']).reshape(shape_mon_pi[deck_nas2[i]])
        p_e_warm_PI[logical_not(ind_last20_Warm_PI)] = np.nan

        delta_p_e[i, 0] = nanmean(area_mean(p_e_cold_abr[f20yr_index:l20yr_index,:,:], y_range, x_range)) - nanmean(area_mean(p_e_cold_PI, y_range, x_range))
        delta_p_e[i, 1] = nanmean(area_mean(p_e_warm_abr[f20yr_index:l20yr_index,:,:], y_range, x_range)) - nanmean(area_mean(p_e_warm_PI, y_range, x_range))
        delta_p_e_4050[i, 0] = nanmean(latitude_mean(p_e_cold_abr[f20yr_index:l20yr_index,:,:], y_range, x_range, lat_range=[-50., -40.])) - nanmean(latitude_mean(p_e_cold_PI, y_range, x_range, lat_range=[-50., -40.]))
        delta_p_e_4050[i, 1] = nanmean(latitude_mean(p_e_warm_abr[f20yr_index:l20yr_index,:,:], y_range, x_range, lat_range=[-50., -40.])) - nanmean(latitude_mean(p_e_warm_PI, y_range, x_range, lat_range=[-50., -40.]))
        delta_p_e_5085[i, 0] = nanmean(latitude_mean(p_e_cold_abr[f20yr_index:l20yr_index,:,:], y_range, x_range, lat_range=[-85., -50.])) - nanmean(latitude_mean(p_e_cold_PI, y_range, x_range, lat_range=[-85., -50.]))
        delta_p_e_5085[i, 0] = nanmean(latitude_mean(p_e_warm_abr[f20yr_index:l20yr_index,:,:], y_range, x_range, lat_range=[-85., -50.])) - nanmean(latitude_mean(p_e_warm_PI, y_range, x_range, lat_range=[-85., -50.]))

        # LTS.
        LTS_cold_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['LTS']).reshape(shape_mon_abr[deck_nas2[i]])
        LTS_cold_abr[logical_not(ind_last20_Cold_abr)] = np.nan
        LTS_warm_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['LTS']).reshape(shape_mon_abr[deck_nas2[i]])
        LTS_warm_abr[logical_not(ind_last20_Warm_abr)] = np.nan
        LTS_cold_PI = deepcopy(output_2lrm_metric_actual_PI[deck_nas2[i]]['LTS']).reshape(shape_mon_pi[deck_nas2[i]])
        LTS_cold_PI[logical_not(ind_last20_Cold_PI)] = np.nan
        LTS_warm_PI = deepcopy(output_2lrm_metric_actual_PI[deck_nas2[i]]['LTS']).reshape(shape_mon_pi[deck_nas2[i]])
        LTS_warm_PI[logical_not(ind_last20_Warm_PI)] = np.nan

        delta_LTS[i, 0] = nanmean(area_mean(LTS_cold_abr[f20yr_index:l20yr_index,:,:], y_range, x_range)) - nanmean(area_mean(LTS_cold_PI, y_range, x_range))
        delta_LTS[i, 1] = nanmean(area_mean(LTS_warm_abr[f20yr_index:l20yr_index,:,:], y_range, x_range)) - nanmean(area_mean(LTS_warm_PI, y_range, x_range))
        delta_LTS_4050[i, 0] = nanmean(latitude_mean(LTS_cold_abr[f20yr_index:l20yr_index,:,:], y_range, x_range, lat_range=[-50., -40.])) - nanmean(latitude_mean(LTS_cold_PI, y_range, x_range, lat_range=[-50., -40.]))
        delta_LTS_4050[i, 1] = nanmean(latitude_mean(LTS_warm_abr[f20yr_index:l20yr_index,:,:], y_range, x_range, lat_range=[-50., -40.])) - nanmean(latitude_mean(LTS_warm_PI, y_range, x_range, lat_range=[-50., -40.]))
        delta_LTS_5085[i, 0] = nanmean(latitude_mean(LTS_cold_abr[f20yr_index:l20yr_index,:,:], y_range, x_range, lat_range=[-85., -50.])) - nanmean(latitude_mean(LTS_cold_PI, y_range, x_range, lat_range=[-85., -50.]))
        delta_LTS_5085[i, 0] = nanmean(latitude_mean(LTS_warm_abr[f20yr_index:l20yr_index,:,:], y_range, x_range, lat_range=[-85., -50.])) - nanmean(latitude_mean(LTS_warm_PI, y_range, x_range, lat_range=[-85., -50.]))

        # SUB_500.
        SUB_cold_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['SUB']).reshape(shape_mon_abr[deck_nas2[i]])
        SUB_cold_abr[logical_not(ind_last20_Cold_abr)] = np.nan
        SUB_warm_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['SUB']).reshape(shape_mon_abr[deck_nas2[i]])
        SUB_warm_abr[logical_not(ind_last20_Warm_abr)] = np.nan
        SUB_cold_PI = deepcopy(output_2lrm_metric_actual_PI[deck_nas2[i]]['SUB']).reshape(shape_mon_pi[deck_nas2[i]])
        SUB_cold_PI[logical_not(ind_last20_Cold_PI)] = np.nan
        SUB_warm_PI = deepcopy(output_2lrm_metric_actual_PI[deck_nas2[i]]['SUB']).reshape(shape_mon_pi[deck_nas2[i]])
        SUB_warm_PI[logical_not(ind_last20_Warm_PI)] = np.nan

        delta_SUB[i, 0] = nanmean(area_mean(SUB_cold_abr[f20yr_index:l20yr_index,:,:], y_range, x_range)) - nanmean(area_mean(SUB_cold_PI, y_range, x_range))
        delta_SUB[i, 1] = nanmean(area_mean(SUB_warm_abr[f20yr_index:l20yr_index,:,:], y_range, x_range)) - nanmean(area_mean(SUB_warm_PI, y_range, x_range))
        delta_SUB_4050[i, 0] = nanmean(latitude_mean(SUB_cold_abr[f20yr_index:l20yr_index,:,:], y_range, x_range, lat_range=[-50., -40.])) - nanmean(latitude_mean(SUB_cold_PI, y_range, x_range, lat_range=[-50., -40.]))
        delta_SUB_4050[i, 1] = nanmean(latitude_mean(SUB_warm_abr[f20yr_index:l20yr_index,:,:], y_range, x_range, lat_range=[-50., -40.])) - nanmean(latitude_mean(SUB_warm_PI, y_range, x_range, lat_range=[-50., -40.]))
        delta_SUB_5085[i, 0] = nanmean(latitude_mean(SUB_cold_abr[f20yr_index:l20yr_index,:,:], y_range, x_range, lat_range=[-85., -50.])) - nanmean(latitude_mean(SUB_cold_PI, y_range, x_range, lat_range=[-85., -50.]))
        delta_SUB_5085[i, 0] = nanmean(latitude_mean(SUB_warm_abr[f20yr_index:l20yr_index,:,:], y_range, x_range, lat_range=[-85., -50.])) - nanmean(latitude_mean(SUB_warm_PI, y_range, x_range, lat_range=[-85., -50.]))

        # gmt.
        gmt_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['gmt'])
        delta_gmt[i] = nanmean(gmt_abr[f20yr_index:l20yr_index])

    # print(sigmaXi_r2)
    
    # changes of variables;
    # standardized change of Xi scaled by gmt, lwp change scaled by 'gmt':
    
    for i in range(len(deck_nas2)):

        dX_dTg_r1[i, :] = (np.asarray([delta_SST[i, 0], delta_p_e[i, 0], delta_LTS[i, 0], delta_SUB[i,0]] / delta_gmt[i]).flatten()) / sigmaXi_r1[i, :]  # Cold
        dX_dTg_r2[i, :] = (np.asarray([delta_SST[i, 1], delta_p_e[i, 1], delta_LTS[i, 1], delta_SUB[i,1]] / delta_gmt[i]).flatten()) / sigmaXi_r2[i, :]  # Warm

        delta_LWP_dTg[i] = (delta_LWP_ALL[i] / delta_gmt[i])
        delta_LWP_dTg_4050[i] = (delta_LWP_4050[i] / delta_gmt[i])
        delta_LWP_dTg_5085[i] = (delta_LWP_5085[i] / delta_gmt[i])
        delta_LWP_dTgr1[i] = (delta_LWP[i, 0]) / delta_gmt[i]
        delta_LWP_dTgr2[i] = (delta_LWP[i, 1]) / delta_gmt[i]

    
    # standardized coefficient of GCM:
    coef_cold = []
    intp_cold = []
    coef_warm = []
    intp_warm = []

    for i in range(len(deck_nas2)):
        # print(output_2lrm_coef_LWP[deck_nas2[i]].shape)

        a_lt = output_2lrm_coef_LWP[deck_nas2[i]][0][0].copy()
        a_le = output_2lrm_coef_LWP[deck_nas2[i]][1][0].copy()
        a0_lt = output_2lrm_coef_LWP[deck_nas2[i]][0][1].copy()
        a0_le = output_2lrm_coef_LWP[deck_nas2[i]][1][1].copy()

        coef_cold.append(array(a_lt))
        coef_warm.append(array(a_le))
        intp_cold.append(array(a0_lt))
        intp_warm.append(array(a0_le))

    for j in range(len(deck_nas2)):
        # print(coef_cold[j].shape)
        # print(sigmaXi_r1[j,:].shape)
        stcoef_r1[j, :] = coef_cold[j] * sigmaXi_r1[j, :]  # Cold
        stcoef_r2[j, :] = coef_warm[j] * sigmaXi_r2[j, :]  # Hot

    
    # delta_LWP_dTg_LRM_cold = full(len(deck_nas2), 0.0)
    # delta_LWP_dTg_LRM_warm = full(len(deck_nas2), 0.0)

    delta_LWP_dTg_GCM = delta_LWP_dTg
    delta_LWP_dTg_GCM4050 = delta_LWP_dTg_4050
    delta_LWP_dTg_GCM5085 = delta_LWP_dTg_5085

    # for i in range(len(deck_nas2)):

    #     delta_LWP_dTg_LRM_cold[i] = sum(stcoef_r1[i,:] * dX_dTg_r1[i,:])
    #     delta_LWP_dTg_LRM_warm[i] = sum(stcoef_r2[i,:] * dX_dTg_r2[i,:])

    #     # delta_LWP_dTg_OBS_warm[i] = np.sum(stcoef_obs_LWP_CCFs * dX_dTg_r2[i,:])
    #     # delta_LWP_dTg_LRM_all[i] = (delta_LWP_dTg_LRM_cold[i] + delta_LWP_dTg_LRM_hot[i])   # concept error..

    # print(delta_LWP_dTg_GCM)

    # print(delta_LWP_dTg_LRM_warm)
    # print(delta_LWP_dTg_OBS_warm)
    
    
    ## Values for calculate the CCF models' responses in each bands and partitioned into "Cold/ Warm":
    
    # from copy import deepcopy
    delta_gmt = full(len(deck2), 0.000)

    delta_LWP_dTg_LRM_all = full(len(deck_nas2), 0.000)
    delta_LWP_dTg_LRM_all4050 = full(len(deck_nas2), 0.000)
    delta_LWP_dTg_LRM_all5085 = full(len(deck_nas2), 0.000)
    delta_LWP_dTg_LRM_cold = full(len(deck_nas2), 0.000)
    delta_LWP_dTg_LRM_cold4050 = full(len(deck_nas2), 0.000)
    delta_LWP_dTg_LRM_cold5085 = full(len(deck_nas2), 0.000)
    delta_LWP_dTg_LRM_warm = full(len(deck_nas2), 0.000)
    delta_LWP_dTg_LRM_warm4050 = full(len(deck_nas2), 0.000)
    delta_LWP_dTg_LRM_warm5085 = full(len(deck_nas2), 0.000)

    for i in range(len(deck_nas2)):

        # indice of Regimes;
        ind_Cold_PI = output_ind_Cold_PI[deck_nas2[i]]
        ind_Warm_PI = output_ind_Warm_PI[deck_nas2[i]]
        ind_Cold_abr = output_ind_Cold_abr[deck_nas2[i]]
        ind_Warm_abr = output_ind_Warm_abr[deck_nas2[i]]
        # print(ind_Cold_PI.shape)
        # print(ind_Warm_abr)

        # Indices for choose 'Warm'/ 'Cold' (Added to become non-nan indices) and 
        # the 121--140 yrs period of 'abrupt4xCO2' & the whole period of 'piControl'
        ind_valid_Cold_abr = deepcopy(output_ind_Cold_abr[deck_nas2[i]]).reshape(shape_mon_abr[deck_nas2[i]])
        ind_valid_Cold_abr[0:f20yr_index, :, :] = False
        ind_valid_Cold_abr[l20yr_index:, :, :] = False
        ind_valid_Warm_abr = deepcopy(output_ind_Warm_abr[deck_nas2[i]]).reshape(shape_mon_abr[deck_nas2[i]])
        ind_valid_Warm_abr[0:f20yr_index, :, :] = False
        ind_valid_Warm_abr[l20yr_index:, :, :] = False
        ind_valid_All_abr = logical_or(ind_valid_Cold_abr, ind_valid_Warm_abr)

        ind_valid_Cold_PI = deepcopy(output_ind_Cold_PI[deck_nas2[i]]).reshape(shape_mon_pi[deck_nas2[i]])
        ind_valid_Warm_PI = deepcopy(output_ind_Warm_PI[deck_nas2[i]]).reshape(shape_mon_pi[deck_nas2[i]])
        ind_valid_All_PI = logical_or(ind_valid_Cold_PI, ind_valid_Warm_PI)
        print(deck_nas2[i])
        # GMT.
        gmt_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['gmt'])
        delta_gmt[i] = nanmean(gmt_abr[f20yr_index:l20yr_index])
        # print("Delta gmt: ", delta_gmt[i])
        # LWP.
        LWP_abr_all = deepcopy(output_2lrm_mon_bin_LWPpredi_abr[deck_nas2[i]]).reshape(shape_mon_abr[deck_nas2[i]])
        LWP_abr_all[logical_not(ind_valid_All_abr)] = np.nan
        # print("Mean LWP abr all: ", nanmean(area_mean(LWP_abr_all, y_range, x_range)))
        LWP_PI_all = deepcopy(output_2lrm_mon_bin_LWPpredi_PI[deck_nas2[i]]).reshape(shape_mon_pi[deck_nas2[i]])
        LWP_PI_all[logical_not(ind_valid_All_PI)] = np.nan
        # print("Mean LWP PI all: ", nanmean(area_mean(LWP_PI_all, y_range, x_range)))
        LWP_abr_cold = deepcopy(output_2lrm_mon_bin_LWPpredi_abr[deck_nas2[i]]).reshape(shape_mon_abr[deck_nas2[i]])
        LWP_abr_cold[logical_not(ind_valid_Cold_abr)] = np.nan
        LWP_abr_warm = deepcopy(output_2lrm_mon_bin_LWPpredi_abr[deck_nas2[i]]).reshape(shape_mon_abr[deck_nas2[i]])
        LWP_abr_warm[logical_not(ind_valid_Warm_abr)] = np.nan
        LWP_PI_cold = deepcopy(output_2lrm_mon_bin_LWPpredi_PI[deck_nas2[i]]).reshape(shape_mon_pi[deck_nas2[i]])
        LWP_PI_cold[logical_not(ind_valid_Cold_PI)] = np.nan
        LWP_PI_warm = deepcopy(output_2lrm_mon_bin_LWPpredi_PI[deck_nas2[i]]).reshape(shape_mon_pi[deck_nas2[i]])
        LWP_PI_warm[logical_not(ind_valid_Warm_PI)] = np.nan

        # Changes in LWP scaled by GMT.

        delta_LWP_dTg_LRM_all[i] = (nanmean(area_mean(LWP_abr_all[f20yr_index:l20yr_index,:,:], y_range, x_range)) - nanmean(area_mean(LWP_PI_all, y_range, x_range))) / delta_gmt[i]
        delta_LWP_dTg_LRM_all4050[i] = (nanmean(latitude_mean(LWP_abr_all[f20yr_index:l20yr_index,:,:], y_range, x_range, lat_range= [-50., -40.])) - nanmean(latitude_mean(LWP_PI_all, y_range, x_range, lat_range= [-50., -40.]))) / delta_gmt[i]
        delta_LWP_dTg_LRM_all5085[i] = (nanmean(latitude_mean(LWP_abr_all[f20yr_index:l20yr_index,:, :], y_range, x_range, lat_range= [-85., -50.])) - nanmean(latitude_mean(LWP_PI_all, y_range, x_range, lat_range= [-85., -50.]))) / delta_gmt[i]
        # print("Delta LWP 40 -- 85: ", delta_LWP_dTg_LRM_all[i])
        delta_LWP_dTg_LRM_cold[i] = (nanmean(area_mean(LWP_abr_cold[f20yr_index:l20yr_index,:,:], y_range, x_range)) - nanmean(area_mean(LWP_PI_cold, y_range, x_range))) / delta_gmt[i]
        delta_LWP_dTg_LRM_cold4050[i] = (nanmean(latitude_mean(LWP_abr_cold[f20yr_index:l20yr_index,:,:], y_range, x_range, lat_range= [-50., -40.])) - nanmean(latitude_mean(LWP_PI_cold, y_range, x_range, lat_range= [-50., -40.]))) / delta_gmt[i]
        delta_LWP_dTg_LRM_cold5085[i] = (nanmean(latitude_mean(LWP_abr_cold[f20yr_index:l20yr_index,:, :], y_range, x_range, lat_range= [-85., -50.])) - nanmean(latitude_mean(LWP_PI_cold, y_range, x_range, lat_range= [-85., -50.]))) / delta_gmt[i]
        # print("Delta LWP at Cold of 40 -- 85: ", delta_LWP_dTg_LRM_cold[i]) 
        delta_LWP_dTg_LRM_warm[i] = (nanmean(area_mean(LWP_abr_warm[f20yr_index:l20yr_index,:,:], y_range, x_range)) - nanmean(area_mean(LWP_PI_warm, y_range, x_range))) / delta_gmt[i]
        delta_LWP_dTg_LRM_warm4050[i] = (nanmean(latitude_mean(LWP_abr_warm[f20yr_index:l20yr_index,:,:], y_range, x_range, lat_range= [-50., -40.])) - nanmean(latitude_mean(LWP_PI_warm, y_range, x_range, lat_range= [-50., -40.]))) / delta_gmt[i]
        delta_LWP_dTg_LRM_warm5085[i] = (nanmean(latitude_mean(LWP_abr_warm[f20yr_index:l20yr_index,:, :], y_range, x_range, lat_range= [-85., -50.])) - nanmean(latitude_mean(LWP_PI_warm, y_range, x_range, lat_range= [-85., -50.]))) / delta_gmt[i]
        # print("Delta LWP at Warm of 40 -- 85: ", delta_LWP_dTg_LRM_warm[i]) 
        
    
    # Observational standardized coefficient of LWP to CCFs:
    # training
    valid_range1=[2013, 1, 15]
    valid_range2=[2016, 12, 31]   # 8 years
    # Predicting
    valid_range3=[1997, 1, 15]
    valid_range4=[2008, 12, 31]   # 12 years
    
    # OBS coef of LWP to CCFs:
    
    # Function #1 loopping through variables space to find the cut-offs of LRM (Multi-Linear Regression Model).
    dict_training, lats_Array, lons_Array, times_Array_training = Pre_processing(s_range, x_range, y_range, valid_range1 = valid_range1, valid_range2 = valid_range2)
    dict_predict, lats_Array, lons_Array, times_Array_predict = Pre_processing(s_range, x_range, y_range, valid_range1 = valid_range3, valid_range2 = valid_range4)


    # Function #2 training LRM with using no cut-off, then use it to predict another historical period.
    predict_result_1r = fitLRMobs_1(dict_training, dict_predict, s_range, y_range, x_range, lats_Array, lons_Array)
    coef_obs = predict_result_1r['coef_dict']
    
    
    # Convert Observational dLWP/dXi from kg*m^-2/stddev to g*m^-2/ unit 
    std_dev_LWP = predict_result_1r['std_LWP_training']

    print("coef_obs: ", coef_obs)

    sigmaXi_r2_obs = np.full((4), 0.0)
    sigmaXi_r2_obs = np.asarray([np.nanstd(dict_training['SST']), np.nanstd(dict_training['p_e']), np.nanstd(dict_training['LTS']), np.nanstd(dict_training['SUB'])])
    print("sigmaXi_r2_obs: ", sigmaXi_r2_obs)

    a = 1000. * coef_obs[0] * sigmaXi_r2_obs
    print("standard coef of obs in g/m2/Sigma: ", a)
    
    
    ## Values in changes in LWP predicting at different Lat bands by OBS sensitivity at "Warm" + GCM sensitivities at "Cold":
    # Assemble a coef_dict with: Warm: Observational coef+intercept; Cold: GCMs coef_intercept;

    Coef_dict_assemble = {}
    Ano_metric_training = {}
    Ano_metric_predict = {}

    delta_gmt = full(len(deck2), 0.000)

    delta_LWP_dTg_OBS_cold = full(len(deck_nas2), 0.000)
    delta_LWP_dTg_OBS_cold4050 = full(len(deck_nas2), 0.000)
    delta_LWP_dTg_OBS_cold5085 = full(len(deck_nas2), 0.000)

    delta_LWP_dTg_OBS_warm = full(len(deck_nas2), 0.000)
    delta_LWP_dTg_OBS_warm4050 = full(len(deck_nas2), 0.000)
    delta_LWP_dTg_OBS_warm5085 = full(len(deck_nas2), 0.000)

    delta_LWP_dTg_OBS_all = full(len(deck_nas2), 0.000)
    delta_LWP_dTg_OBS_all4050 = full(len(deck_nas2), 0.000)
    delta_LWP_dTg_OBS_all5085 = full(len(deck_nas2), 0.000)

    for i in range(len(deck_nas2)):

        a_lt = output_2lrm_coef_LWP[deck_nas2[i]][0][0].copy()
        a0_lt = output_2lrm_coef_LWP[deck_nas2[i]][0][1].copy()

        Coef_dict_assemble[deck_nas2[i]] = asarray([[a_lt, a0_lt], coef_obs])

        # Re-predict the training dataset and predict dataset with assembled coef dict:
        Ano_metric_training[deck_nas2[i]] = deepcopy(output_2lrm_metric_actual_PI[deck_nas2[i]])
        Ano_metric_predict[deck_nas2[i]] = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]])

        # Retrive TR_Ts:
        WD = '/glade/scratch/chuyan/CMIP_output/'
        folder = glob.glob(WD+ deck_nas2[i]+'__'+ 'STAT_pi+abr_'+'22x_31y_Sep9th_anomalies'+ '.npz')
        # print(folder)
        output_ARRAY = np.load(folder[0], allow_pickle=True)  # str(TR_sst)
        TR_sst2 = output_ARRAY['TR_maxR2_SST']
        TR_sub2 = output_ARRAY['TR_maxR2_SUB']
        # print("TR_large_pi_R_2: ", TR_sst2, '  K ', TR_sub2 , ' Pa/s ')

        # Predicting LWP: 
        #.. piControl
        predict_dict_PI, ind6_PI, ind7_PI, shape_fla_training = rdlrm_2_predict(Ano_metric_training[deck_nas2[i]], Coef_dict_assemble[deck_nas2[i]], TR_sst2, predictant='LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 2)
        # 'YB' is the predicted value of LWP in 'piControl' experiment
        YB = predict_dict_PI['value']
        # Save 'YB', and resampled into the shape of 'LWP_yr_bin':
        LWP_predi_bin_PI = asarray(YB).reshape(shape_mon_pi[deck_nas2[i]])

        #.. abrupt4xCO2

        predict_dict_abr, ind6_abr, ind7_abr, shape_fla_testing = rdlrm_2_predict(Ano_metric_predict[deck_nas2[i]], Coef_dict_assemble[deck_nas2[i]], TR_sst2, predictant = 'LWP', predictor = ['SST', 'p_e', 'LTS', 'SUB'], r = 2)
        # 'YB_abr' is the predicted value of LWP in 'abrupt 4xCO2' experiment
        YB_abr = predict_dict_abr['value']
        # Save 'YB_abr', reshapled into the shape of 'LWP_yr_bin_abr':
        LWP_predi_bin_abr = asarray(YB_abr).reshape(shape_mon_abr[deck_nas2[i]])

        # Processing:

        # indice of Regimes;
        ind_Cold_PI = output_ind_Cold_PI[deck_nas2[i]]
        ind_Warm_PI = output_ind_Warm_PI[deck_nas2[i]]
        ind_Cold_abr = output_ind_Cold_abr[deck_nas2[i]]
        ind_Warm_abr = output_ind_Warm_abr[deck_nas2[i]]

        # Indices for choose 'Warm'/ 'Cold' (Added to become non-nan indices) and 
        # the 121--140 yrs period of 'abrupt4xCO2' & the whole period of 'piControl'
        ind_valid_Cold_abr = deepcopy(output_ind_Cold_abr[deck_nas2[i]]).reshape(shape_mon_abr[deck_nas2[i]])
        ind_valid_Cold_abr[0:f20yr_index, :, :] = False
        ind_valid_Cold_abr[l20yr_index:, :, :] = False
        ind_valid_Warm_abr = deepcopy(output_ind_Warm_abr[deck_nas2[i]]).reshape(shape_mon_abr[deck_nas2[i]])
        ind_valid_Warm_abr[0:f20yr_index, :, :] = False
        ind_valid_Warm_abr[l20yr_index:, :, :] = False
        ind_valid_All_abr = logical_or(ind_valid_Cold_abr, ind_valid_Warm_abr)

        ind_valid_Cold_PI = deepcopy(output_ind_Cold_PI[deck_nas2[i]]).reshape(shape_mon_pi[deck_nas2[i]])
        ind_valid_Warm_PI = deepcopy(output_ind_Warm_PI[deck_nas2[i]]).reshape(shape_mon_pi[deck_nas2[i]])
        ind_valid_All_PI = logical_or(ind_valid_Cold_PI, ind_valid_Warm_PI)

        # GMT.
        gmt_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['gmt'])
        delta_gmt[i] = nanmean(gmt_abr[f20yr_index:l20yr_index])

        # LWP predicting by assembled coef dict.
        LWP_abr_all = deepcopy(LWP_predi_bin_abr).reshape(shape_mon_abr[deck_nas2[i]])
        LWP_abr_all[logical_not(ind_valid_All_abr)] = np.nan

        LWP_PI_all = deepcopy(LWP_predi_bin_PI).reshape(shape_mon_pi[deck_nas2[i]])
        LWP_PI_all[logical_not(ind_valid_All_PI)] = np.nan

        LWP_abr_cold = deepcopy(LWP_predi_bin_abr).reshape(shape_mon_abr[deck_nas2[i]])
        LWP_abr_cold[logical_not(ind_valid_Cold_abr)] = np.nan
        LWP_abr_warm = deepcopy(LWP_predi_bin_abr).reshape(shape_mon_abr[deck_nas2[i]])
        LWP_abr_warm[logical_not(ind_valid_Warm_abr)] = np.nan
        LWP_PI_cold = deepcopy(LWP_predi_bin_PI).reshape(shape_mon_pi[deck_nas2[i]])
        LWP_PI_cold[logical_not(ind_valid_Cold_PI)] = np.nan
        LWP_PI_warm = deepcopy(LWP_predi_bin_PI).reshape(shape_mon_pi[deck_nas2[i]])
        LWP_PI_warm[logical_not(ind_valid_Warm_PI)] = np.nan

        # Changes in LWP scaled by GMT.

        delta_LWP_dTg_OBS_all[i] = (nanmean(area_mean(LWP_abr_all[f20yr_index:l20yr_index,:,:], y_range, x_range)) - nanmean(area_mean(LWP_PI_all, y_range, x_range))) / delta_gmt[i]
        delta_LWP_dTg_OBS_all4050[i] = (nanmean(latitude_mean(LWP_abr_all[f20yr_index:l20yr_index,:,:], y_range, x_range, lat_range= [-50., -40.])) - nanmean(latitude_mean(LWP_PI_all, y_range, x_range, lat_range= [-50., -40.]))) / delta_gmt[i]
        delta_LWP_dTg_OBS_all5085[i] = (nanmean(latitude_mean(LWP_abr_all[f20yr_index:l20yr_index,:, :], y_range, x_range, lat_range= [-85., -50.])) - nanmean(latitude_mean(LWP_PI_all, y_range, x_range, lat_range= [-85., -50.]))) / delta_gmt[i]
        # print("Delta LWP OBS at SO Averaged:", delta_LWP_dTg_OBS_all[i])
        delta_LWP_dTg_OBS_cold[i] = (nanmean(area_mean(LWP_abr_cold[f20yr_index:l20yr_index,:,:], y_range, x_range)) - nanmean(area_mean(LWP_PI_cold, y_range, x_range))) / delta_gmt[i]
        delta_LWP_dTg_OBS_cold4050[i] = (nanmean(latitude_mean(LWP_abr_cold[f20yr_index:l20yr_index,:,:], y_range, x_range, lat_range= [-50., -40.])) - nanmean(latitude_mean(LWP_PI_cold, y_range, x_range, lat_range= [-50., -40.]))) / delta_gmt[i]
        delta_LWP_dTg_OBS_cold5085[i] = (nanmean(latitude_mean(LWP_abr_cold[f20yr_index:l20yr_index,:, :], y_range, x_range, lat_range= [-85., -50.])) - nanmean(latitude_mean(LWP_PI_cold, y_range, x_range, lat_range= [-85., -50.]))) / delta_gmt[i]

        delta_LWP_dTg_OBS_warm[i] = (nanmean(area_mean(LWP_abr_warm[f20yr_index:l20yr_index,:,:], y_range, x_range)) - nanmean(area_mean(LWP_PI_warm, y_range, x_range))) / delta_gmt[i]
        # print("Delta LWP OBS at warm:", delta_LWP_dTg_OBS_warm[i])
        delta_LWP_dTg_OBS_warm4050[i] = (nanmean(latitude_mean(LWP_abr_warm[f20yr_index:l20yr_index,:,:], y_range, x_range, lat_range= [-50., -40.])) - nanmean(latitude_mean(LWP_PI_warm, y_range, x_range, lat_range= [-50., -40.]))) / delta_gmt[i]
        delta_LWP_dTg_OBS_warm5085[i] = (nanmean(latitude_mean(LWP_abr_warm[f20yr_index:l20yr_index,:, :], y_range, x_range, lat_range= [-85., -50.])) - nanmean(latitude_mean(LWP_PI_warm, y_range, x_range, lat_range= [-85., -50.]))) / delta_gmt[i]

    # print(Ano_metric_predict)

    # print("Assembled Coef dict: ", Coef_dict_assemble[deck_nas2[2]]) 
    
    
    ## PLotting:
    # Observational Constraints:
    X_metric = delta_LWP_dTg_LRM_warm
    Y_metric = delta_LWP_dTg_LRM_all
    Z_metric = delta_LWP_dTg_GCM
    Constraint_metric = delta_LWP_dTg_OBS_warm
    
    
    
    from scipy.optimize import curve_fit
    
    def target_func(x, m, k):

        '''
        1-d line linear fit
        '''
        y = m * x + k
        return y


    def calc_r2(Y_pre, Y):

        residual_ydata = array(Y).reshape(-1,1) - array(Y_pre).reshape(-1,1)

        ss_res_bar  = (residual_ydata**2).sum()
        ss_tot_bar  = ((Y - Y.mean())**2).sum()
        R_square = 1. - (ss_res_bar/ss_tot_bar)

        return R_square
    
    
    # Constraint on "Warm" regime Only from OBS

    import matplotlib.pyplot as plt

    # plot settings:
    parameters = {'axes.labelsize': 17, 'legend.fontsize': 16,  
           'axes.titlesize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16}
    plt.rcParams.update(parameters)

    fig2, axes2 = plt.subplots(1, 1, figsize = (7.0, 6.25))

    n_name = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
              31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]

    x = linspace(-2.0, 9.0, 50)
    y = x

    plot_scat1 = []
    plot_scat2 = []

    x1 = np.min(1000.* Constraint_metric)
    x2 = np.max(1000.* Constraint_metric)
    print("x1:", x1, "x2:", x2)

    for a in range(len(deck_nas2)):
        scp1 = plt.scatter(1000.* X_metric[a], 1000.* Y_metric[a], s = 134, marker = 's', color="tab:red", alpha = 0.50, zorder = 12)

        plot_scat1.append(scp1)
        # scp2 = plt.scatter(1000.* Constraint_metric[a], 1000.* Y_metric[a], s = 132, marker = 's', color="tab:green", alpha = 0.50, zorder = 10)
        # plot_scat2.append(scp2)

        # Add annotate to the first point of each GCM
        axes2.annotate(n_name[a], xy=(1000.* X_metric[a], 1000.* Y_metric[a]),
                xytext=(-4.60, -3.42), textcoords = "offset points", color = 'white', fontsize = 9.8, zorder = 99)
                # horizontalalignment= "left" if output_meandelta_dLWPdGMT[deck_nas2[a]+'_predict_150yrs_0K'] > output_meandelta_dLWPdGMT[deck_nas2[a]+'_actual_150yrs_0K'] else "right", verticalalignment = "bottom")

    #.. linear curve fit for regressed d(LWP) and reported d(LWP) for largest_pi_R_2 model
    POPT_2, POCV_2 = curve_fit(target_func, 1000.* X_metric, 1000.* Y_metric)

    # Calc the R square, plot the fit line:
    pearsonr_2 = pearsonr(1000. *X_metric, 1000. *Y_metric)[0]
    R_square_2 = calc_r2(1000.* X_metric, 1000.* Y_metric)
    fitp3 = axes2.plot(y, POPT_2[0] * x + POPT_2[1], linestyle = '-', color = "tab:red", linewidth = 3.6)
    print('pearsonr r value between y and x: ', pearsonr_2)

    # Add Reference line:
    Refp = axes2.plot(x, y, label = "1-1 reference line", c = 'gray', linestyle= '--', alpha = 0.8, linewidth = 3.6, zorder = 5)  # Blue

    # # Add Reference line 2, 3
    Refp2 = axes2.plot(x, [POPT_2[0] * x1 + POPT_2[1]] * 50, c = "tab:red", linestyle = '--', linewidth = 3.9, zorder = 8)
    y1_all_OBS_range = POPT_2[0] * x1 + POPT_2[1] 

    Refp3 = axes2.plot(x, [POPT_2[0] * x2 + POPT_2[1]] * 50, c = "tab:red", linestyle = '--', linewidth = 3.9, zorder = 7)
    y2_all_OBS_range = POPT_2[0] * x2 + POPT_2[1]
    print("OBS constraint Warm range minimum: ", y1_all_OBS_range)
    print("OBS constraint Warm range maximum: ", y2_all_OBS_range)

    axes2.set_xlim(-2.0, 9.0)
    axes2.set_ylim(-2.0, 9.0)
    # axes2.set_xticks(np.arange(0., 9., 9), np.arange(0, 9, 9))
    # axes2.set_yticks(np.arange(0., 9., 9), np.arange(0, 9, 9))
    axes2.set_xlabel(r"$ Predicted\ by\ CCFs,\ Warm\ Only\ $") # r"$ Mean\ \Delta LWP\ Predicted\ by\ CCFs,\ $" + r"$g\ m^{-2}$"
    axes2.set_ylabel(r"$ Predicted\ by\ CCFs,\ SO\ Averaged\ $") # Mean\ \Delta LWP\ Predicted\ by\ GCMs,\ $" + r"$g\ m^{-2}$"

    axes2.set_title( r"$ (a)\ {40-85^{o}S}\ \ \Delta LWP/\ \Delta GMT\ [g m^{-2}/ K] $", loc ='left')
    axes2.text(4.705, 8.200, "r = %.2f" % pearsonr_2, fontsize = 16)
    # legend61 = axes2.legend([scp1], ['CMIP model'], 
    #                     loc='lower right', bbox_to_anchor=(0.975, 0.194), fontsize = 14)  # scp2 'OBS Constraint for Hot regime '

    plt.axvspan(x1, x2, np.min(y), np.max(y), facecolor = "tab:red", alpha = 0.6, zorder = 9)
    plt.axvline(x1, c = "tab:red", alpha = 0.6, linewidth = 1.4)
    plt.axvline(x2, c = "tab:red", alpha = 0.6, linewidth = 2)

    # axes2.legend()
    # axes2.add_artist(legend61)

    sns.set_style("whitegrid", {"grid.linestyle": "--"})
    plt.savefig(path6 + 'Fig4(a)_4085.jpg', bbox_inches = 'tight', dpi = 425)

    # determine the OBS range of 'Warm' Regime constraint,
    OBS_range_model1 = []
    for a in range(len(deck_nas2)):

        if (1000.* X_metric[a] > x1) & (1000.* X_metric[a] < x2):

            OBS_range_model1.append(int(a))

    print("Model No.(strat at 0) in the  Warm Regime Constraint range of CCF: ", OBS_range_model1)
    
    
    
    # Constraint on "ALL" regime: SO Averaged

    # import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 1, figsize = (7.0, 6.25))

    
    
    x = linspace(-2.0, 9.0, 50)
    y = x

    plot_scat1 = []
    plot_scat2 = []

    x1 = y1_all_OBS_range
    x2 = y2_all_OBS_range

    for b in range(len(deck_nas2)):

        scp1 = plt.scatter(1000.* Y_metric[b], 1000.* Z_metric[b], s = 132, marker = 's', color="tab:brown", alpha = 0.50, zorder = 12)
        plot_scat1.append(scp1)

        # Add annotate to the first point of each GCM
        axes.annotate(n_name[b], xy = (1000.* Y_metric[b], 1000.* Z_metric[b]), 
            xytext=(-4.60, -3.42), textcoords = "offset points", color = 'white', fontsize = 9.8, zorder = 99)
            #  horizontalalignment= "left" if output_meandelta_dLWPdGMT[deck_nas2[a]+'_predict_150yrs_0K'] > output_meandelta_dLWPdGMT[deck_nas2[a]+'_actual_150yrs_0K'] else "right", verticalalignment = "bottom")

    #.. linear curve fit for regressed d(LWP) and reported d(LWP) for largest_pi_R_2 model
    POPT_2, POCV_2 = curve_fit(target_func, 1000. * Y_metric, 1000. * )

    # Calc the R square, plot the fit line:
    pearsonr_2 = pearsonr(1000. * Y_metric, 1000. * )[0]
    R_square_2 = calc_r2(1000. * Y_metric, 1000. * )
    fitp3 = axes.plot(y, POPT_2[0] * x + POPT_2[1], linestyle = '-', color = "tab:brown", linewidth = 2.8)
    print(pearsonr_2)


    # Add Reference line:
    Refp = axes.plot(x, y, label = "1-1 reference line", c = 'gray', linestyle= '--', alpha = 0.8, linewidth = 3.6, zorder = 5)  # Blue

    # Add Reference line 2, 3
    Refp2 = axes.plot(x, [POPT_2[0] * x1 + POPT_2[1]] * 50, c = "tab:brown", linestyle = '--', linewidth = 3.9, zorder = 8)
    y1_all_OBS_range2 = POPT_2[0] * x1 + POPT_2[1] 

    Refp3 = axes.plot(x, [POPT_2[0] * x2 + POPT_2[1]] * 50, c = "tab:brown", linestyle = '--', linewidth = 3.9, zorder = 7)
    y2_all_OBS_range2 = POPT_2[0] * x2 + POPT_2[1]
    print("OBS constraint SO averaged range minimum: ", y1_all_OBS_range2)
    print("OBS constraint SO averaged range maximum: ", y2_all_OBS_range2)

    axes.set_xlim([-2.0, 9.0])
    axes.set_ylim([-2.0, 9.0])
    # axes2.set_xticks(np.arange(0., 9., 9), np.arange(0, 9, 9))
    # axes2.set_yticks(np.arange(0., 9., 9), np.arange(0, 9, 9))
    axes.set_xlabel(r"$ Predicted\ by\ CCFs,\ SO\ Averaged\ $") # r"$ Mean\ \Delta LWP\ Predicted\ by\ CCFs,\ $" + r"$g\ m^{-2}$"
    axes.set_ylabel(r"$ Predicted\ by\ GCM,\ SO\ Averaged\ $") # Mean\ \Delta LWP\ Predicted\ by\ GCMs,\ $" + r"$g\ m^{-2}$"

    axes.set_title( r"$ (b)\ {40- 85^{o}S}\ \ \Delta LWP/\ \Delta GMT\ [g m^{-2}/ K] $", loc ='left')
    axes.text(4.150, 8.100, r"$R^{2} = %.2f $" % R_square_2, fontsize = 16)
    # legend61 = axes.legend([scp1], ['CMIP model'], 
    #                     loc='lower right', bbox_to_anchor=(0.975, 0.194), fontsize = 14)  # scp2 'OBS Constraint for Hot regime '

    plt.axvspan(x1, x2, np.min(y), np.max(y), facecolor = "tab:brown", alpha = 0.6, zorder = 9)
    plt.axvline(x1, c = "tab:brown", alpha = 0.6, linewidth = 1.4)
    plt.axvline(x2, c = "tab:brown", alpha = 0.6, linewidth = 2)
    # axes2.add_artist(legend61)

    sns.set_style("whitegrid", {"grid.linestyle": "--"})
    plt.savefig(path6 + 'Fig4(b)_4085.jpg', bbox_inches = 'tight', dpi = 425)

    # Determine the OBS range of 'ALL' Regime Constraint; 

    OBS_range_model2 = []
    for b in range(len(deck_nas2)):

        if (1000.* Y_metric[b] >= x1) & (1000.* Y_metric[b] <= x2):

            OBS_range_model2.append(int(b))

    print(OBS_range_model2)


    OBS_range_MODEL_common = []

    for J in range(len(OBS_range_model2)):

        if OBS_range_model2[J] in OBS_range_model1:

            OBS_range_MODEL_common.append(OBS_range_model2[J])

    print("Model No.(start at 0) in the SO Averaged Constraint range from CCF: ", OBS_range_MODEL_common)

    
    return 0


def Fig5_base(s_range, x_range, y_range, deck2 = deck2, deck_nas2 = deck_nas2, path1 = path_data, path6 = path_plot):
    ## Box Plot summarizing the Eq (3):
    
    ## Read two Regimes (Hot,Cold) data

    output_ARRAY = {}   # storage output file
    output_intermedia = {}   # storage the 'rawdata_dict'

    output_dict0_PI = {}
    output_dict0_abr = {}

    output_GMT = {}
    output_2lrm_predict = {}  # dict, store annualy, area_meaned prediction of LWP
    output_2lrm_report = {}  # dict, store annually, area_meaned actual values of GCMs LWP
    output_2lrm_coef_LWP = {}
    output_2lrm_dict_Albedo = {}  # Coefficients of 2 regimes's albedo trained by report 'LWP' data
    # output_2lrm_coef_albedo_lL = {}

    # Raw data
    output_2lrm_yr_bin_abr = {}
    output_2lrm_yr_bin_PI = {}
    output_2lrm_mon_bin_abr = {}
    output_2lrm_mon_bin_PI = {}
    
    # Metric raw data in specific units:
    shape_mon_pi = {}
    shape_mon_abr = {}
    output_2lrm_metric_actual_PI = {}
    output_2lrm_metric_actual_abr = {}
    
    # Statistic metrics of PI:
    output_Mean_training = {}
    output_Stdev_training = {}
    
    # Predict metric data in specific units:
    output_2lrm_mon_bin_LWPpredi_PI = {}
    output_2lrm_mon_bin_LWPpredi_abr = {}

    # Index for regime(s): Only for 2lrm
    output_ind_Cold_PI = {}
    output_ind_Hot_PI = {}
    output_ind_Cold_abr = {}
    output_ind_Hot_abr = {}

    Tr_sst =  0.0

    for i in range(len(deck2)):
        # print("i", i)
        folder_2lrm = glob.glob(path1+deck2[i]['modn'] + '_r2r1_hotcold(Jan)_(largestpiR2)_Sep9th_Anomalies_Rtest' + '*' + '_dats.npz')
        print(len(folder_2lrm))

        if len(folder_2lrm) == 4:
            if (len(folder_2lrm[0]) < len(folder_2lrm[1])) & (len(folder_2lrm[0]) < len(folder_2lrm[2])) & (len(folder_2lrm[0]) < len(folder_2lrm[3])):
                folder_best2lrm = folder_2lrm[0]
            elif (len(folder_2lrm[1]) < len(folder_2lrm[0])) & (len(folder_2lrm[1]) < len(folder_2lrm[2])) & (len(folder_2lrm[1]) < len(folder_2lrm[3])):
                folder_best2lrm = folder_2lrm[1]
            elif (len(folder_2lrm[2]) < len(folder_2lrm[0])) & (len(folder_2lrm[2]) < len(folder_2lrm[1])) & (len(folder_2lrm[2]) < len(folder_2lrm[3])):
                folder_best2lrm = folder_2lrm[2]
            else:
                folder_best2lrm = folder_2lrm[3]
            print(folder_best2lrm)

        elif len(folder_2lrm) == 3:
            if (len(folder_2lrm[1]) <  len(folder_2lrm[0])) & (len(folder_2lrm[1]) <  len(folder_2lrm[2])):
                folder_best2lrm = folder_2lrm[1]
            elif (len(folder_2lrm[0]) <  len(folder_2lrm[1])) & (len(folder_2lrm[0]) <  len(folder_2lrm[2])):
                folder_best2lrm = folder_2lrm[0]
            else:
                folder_best2lrm = folder_2lrm[2]
            print(folder_best2lrm)

        elif len(folder_2lrm) == 2:
            if len(folder_2lrm[1]) <  len(folder_2lrm[0]):
                folder_best2lrm = folder_2lrm[1]
            else:
                folder_best2lrm = folder_2lrm[0]
            print(folder_best2lrm)

        else:
            output_ARRAY[deck_nas2[i]] = load(folder_2lrm[0], allow_pickle = True)  #+'_'+str(Tr_sst)
            print(folder_2lrm[0])

        output_ARRAY[deck_nas2[i]] = load(folder_best2lrm, allow_pickle = True)  #+'_'+str(Tr_sst)

        # output_ARRAY[deck_nas2[i]] = load(folder_2lrm[0], allow_pickle = True)  #+'_'+str(Tr_sst)
        output_intermedia[deck_nas2[i]] = output_ARRAY[deck_nas2[i]]['rawdata_dict']

        output_GMT[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['GMT']
        output_2lrm_predict[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['predicted_metrics']
        output_2lrm_report[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['report_metrics']

        output_dict0_PI[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['dict1_PI_var']
        output_dict0_abr[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['dict1_abr_var']

        output_2lrm_coef_LWP[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['Coef_dict']
        # print(output_2lrm_dict_Albedo, "i", i, output_intermedia[deck_nas2[i]][()].keys())
        output_2lrm_dict_Albedo[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['coef_dict_Albedo_pi']

        # Monthly data
        output_2lrm_mon_bin_PI[deck_nas2[i]] = output_dict0_PI[deck_nas2[i]]['dict1_mon_bin_PI']
        output_2lrm_mon_bin_abr[deck_nas2[i]] = output_dict0_abr[deck_nas2[i]]['dict1_mon_bin_abr']
        # Annually data
        output_2lrm_yr_bin_PI[deck_nas2[i]] = output_dict0_PI[deck_nas2[i]]['dict1_yr_bin_PI']
        output_2lrm_yr_bin_abr[deck_nas2[i]] = output_dict0_abr[deck_nas2[i]]['dict1_yr_bin_abr']

        # Flattened Metric monthly mean bin data
        shape_mon_pi[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['shape_mon_PI_3']
        shape_mon_abr[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['shape_mon_abr_3']
        output_2lrm_metric_actual_PI[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['metric_training']
        output_2lrm_metric_actual_abr[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['metric_predict']

        # Flattened Predicted monthly bin data
        output_2lrm_mon_bin_LWPpredi_PI[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['LWP_predi_bin_PI']
        output_2lrm_mon_bin_LWPpredi_abr[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['LWP_predi_bin_abr']

        # Statistic metrics of PI:
        output_Mean_training[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['Mean_training']
        output_Stdev_training[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['Stdev_training']

        # Indice for Regimes
        output_ind_Hot_PI[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['ind_Hot_PI']
        output_ind_Cold_PI[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['ind_Cold_PI']

        output_ind_Hot_abr[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['ind_Hot_abr']
        output_ind_Cold_abr[deck_nas2[i]] = output_intermedia[deck_nas2[i]][()]['ind_Cold_abr']

    print('Down read 2-LRM.')
    
    ## Values for calculate the GCMs' responses in each bands and partitioned into "Cold/ Warm":
    # Standard deviation of Cloud Controlling factor (Xi) and Liquid Water Path (LWP):

    sigmaXi_r1 = full((len(deck2), 4), 0.0)  # Cold
    sigmaXi_r2 = full((len(deck2), 4), 0.0)  # Hot

    sigmaLWP_r1 = full((len(deck2)), 0.0)  # Cold
    sigmaLWP_r2 = full((len(deck2)), 0.0)  # Hot
    sigmaLWP_ALL = full((len(deck2)), 0.0)  # Southern Ocean as a whole

    # Changes of variable between 'piControl' (mean-state) and 'abrupt4xCO2' (warming period, take 121 - 140 yrs' mean of abr4x experiment)
    # Cloud Controlling factor (Xi), Liquid Water Path (LWP), and global mean surface air Temperature (gmt):
    delta_gmt = full(len(deck2), 0.000)

    delta_SST = full((len(deck2), 2), 0.0)  # two Regimes, Cold & Hot
    delta_p_e = full((len(deck2), 2), 0.0)
    delta_LTS = full((len(deck2), 2), 0.0)
    delta_SUB = full((len(deck2), 2), 0.0)
    delta_LWP = full((len(deck2), 2), 0.0)
    delta_LWP_ALL = full((len(deck2)), 0.0)  # Southern Ocean lwp changes

    # Standardized changes of variables
    # Cloud Controlling factor (Xi) scaled by 'gmt', Liquid Water Path (LWP):
    dX_dTg_r1 = full((len(deck2), 4), 0.0)  # Cold
    dX_dTg_r2 = full((len(deck2), 4), 0.0)  # Hot
    delta_LWP_dTg = full((len(deck2)), 0.0)  # Southern Ocean lwp changes scaled by gmt
    delta_LWP_dTgr1 = full((len(deck2)), 0.0)  # Cold
    delta_LWP_dTgr2 = full((len(deck2)), 0.0)  # Hot

    # Coef of LWP to Cloud controlling factors, Xis, for two regimes
    # GCM values and the OBS values

    stcoef_r1 = full((len(deck2), 4), 0.0)  # Cold
    stcoef_r2 = full((len(deck2), 4), 0.0)  # Hot
    stcoef_obs = full((4), 0.0)  # Hot Regime Only 
    
    
    f5yr_index = 121*12
    l5yr_index = 140*12


    for i in range(len(deck_nas2)):

        # indice of Regimes;
        ind_Cold_PI = output_ind_Cold_PI[deck_nas2[i]]
        ind_Hot_PI = output_ind_Hot_PI[deck_nas2[i]]
        ind_Cold_abr = output_ind_Cold_abr[deck_nas2[i]]
        ind_Hot_abr = output_ind_Hot_abr[deck_nas2[i]]
        # print(ind_Cold_PI.shape)
        # print(ind_Hot_abr)

        # print(output_2lrm_metric_actual_PI[deck_nas2[i]]['SST'][ind_Hot_PI].shape)
        ## calc standard_deviation for CCFs at training period:

        sigmaXi_r1[i,:] = np.asarray([np.nanstd(output_2lrm_metric_actual_PI[deck_nas2[i]]['SST'][ind_Cold_PI]), np.nanstd(output_2lrm_metric_actual_PI[deck_nas2[i]]['p_e'][ind_Cold_PI]), 
                              np.nanstd(output_2lrm_metric_actual_PI[deck_nas2[i]]['LTS'][ind_Cold_PI]), np.nanstd(output_2lrm_metric_actual_PI[deck_nas2[i]]['SUB'][ind_Cold_PI])])

        sigmaXi_r2[i,:] = np.asarray([np.nanstd(output_2lrm_metric_actual_PI[deck_nas2[i]]['SST'][ind_Hot_PI]), np.nanstd(output_2lrm_metric_actual_PI[deck_nas2[i]]['p_e'][ind_Hot_PI]), 
                              np.nanstd(output_2lrm_metric_actual_PI[deck_nas2[i]]['LTS'][ind_Hot_PI]), np.nanstd(output_2lrm_metric_actual_PI[deck_nas2[i]]['SUB'][ind_Hot_PI])])

        sigmaLWP_r1[i] = np.nanstd(output_2lrm_metric_actual_PI[deck_nas2[i]]['LWP'][ind_Cold_PI])
        sigmaLWP_r2[i] = np.nanstd(output_2lrm_metric_actual_PI[deck_nas2[i]]['LWP'][ind_Hot_PI])
        sigmaLWP_ALL[i] = np.nanstd(output_2lrm_metric_actual_PI[deck_nas2[i]]['LWP'][logical_or(ind_Cold_PI, ind_Hot_PI)])

        # calc changes of variables in two different regimes:

        # indice for 'Hot' and 'Cold' regimes corresponding to the last period
        ind_last20_Cold_abr = deepcopy(output_ind_Cold_abr[deck_nas2[i]]).reshape(shape_mon_abr[deck_nas2[i]])
        ind_last20_Cold_abr[0:f5yr_index, :, :] = False
        ind_last20_Cold_abr[l5yr_index:, :, :] = False
        ind_last20_Hot_abr = deepcopy(output_ind_Hot_abr[deck_nas2[i]]).reshape(shape_mon_abr[deck_nas2[i]])
        ind_last20_Hot_abr[0:f5yr_index, :, :] = False
        ind_last20_Hot_abr[l5yr_index:, :, :] = False
        ind_last20_All_abr = np.logical_or(ind_last20_Cold_abr, ind_last20_Hot_abr)

        ind_last20_Cold_PI = deepcopy(output_ind_Cold_PI[deck_nas2[i]]).reshape(shape_mon_pi[deck_nas2[i]])
        ind_last20_Hot_PI = deepcopy(output_ind_Hot_PI[deck_nas2[i]]).reshape(shape_mon_pi[deck_nas2[i]])
        ind_last20_All_PI = np.logical_or(ind_last20_Cold_PI, ind_last20_Hot_PI)

        LWP_all_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['LWP']).reshape(shape_mon_abr[deck_nas2[i]])
        LWP_all_abr[np.logical_not(ind_last20_All_abr)] = np.nan

        LWP_all_PI = deepcopy(output_2lrm_metric_actual_PI[deck_nas2[i]]['LWP']).reshape(shape_mon_pi[deck_nas2[i]])
        LWP_all_PI[np.logical_not(ind_last20_All_PI)] = np.nan

        LWP_cold_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['LWP']).reshape(shape_mon_abr[deck_nas2[i]])
        LWP_cold_abr[np.logical_not(ind_last20_Cold_abr)] = np.nan
        LWP_hot_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['LWP']).reshape(shape_mon_abr[deck_nas2[i]])
        LWP_hot_abr[np.logical_not(ind_last20_Hot_abr)] = np.nan
        LWP_cold_PI = deepcopy(output_2lrm_metric_actual_PI[deck_nas2[i]]['LWP']).reshape(shape_mon_pi[deck_nas2[i]])
        LWP_cold_PI[np.logical_not(ind_last20_Cold_PI)] = np.nan
        LWP_hot_PI = deepcopy(output_2lrm_metric_actual_PI[deck_nas2[i]]['LWP']).reshape(shape_mon_pi[deck_nas2[i]])
        LWP_hot_PI[np.logical_not(ind_last20_Hot_PI)] = np.nan

        delta_LWP_ALL[i] = np.nanmean(area_mean(LWP_all_abr[f5yr_index:l5yr_index,:,:], y_range, x_range)) - np.nanmean(area_mean(LWP_all_PI, y_range, x_range))
        delta_LWP[i, 0] = np.nanmean(area_mean(LWP_cold_abr[f5yr_index:l5yr_index,:,:], y_range, x_range)) - np.nanmean(area_mean(LWP_cold_PI, y_range, x_range))
        delta_LWP[i, 1] = np.nanmean(area_mean(LWP_hot_abr[f5yr_index:l5yr_index,:,:], y_range, x_range)) - np.nanmean(area_mean(LWP_hot_PI, y_range, x_range))
        # print('1:', area_mean(LWP_cold_abr[f5yr_index:l5yr_index,:,:], y_range, x_range))
        # print('2:', area_mean(LWP_cold_PI[:,:,:], y_range, x_range))

        # SST.
        SST_cold_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['SST']).reshape(shape_mon_abr[deck_nas2[i]])
        SST_cold_abr[np.logical_not(ind_last20_Cold_abr)] = np.nan
        SST_hot_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['SST']).reshape(shape_mon_abr[deck_nas2[i]])
        SST_hot_abr[np.logical_not(ind_last20_Hot_abr)] = np.nan
        SST_cold_PI = deepcopy(output_2lrm_metric_actual_PI[deck_nas2[i]]['SST']).reshape(shape_mon_pi[deck_nas2[i]])
        SST_cold_PI[np.logical_not(ind_last20_Cold_PI)] = np.nan
        SST_hot_PI = deepcopy(output_2lrm_metric_actual_PI[deck_nas2[i]]['SST']).reshape(shape_mon_pi[deck_nas2[i]])
        SST_hot_PI[np.logical_not(ind_last20_Hot_PI)] = np.nan

        delta_SST[i, 0] = np.nanmean(area_mean(SST_cold_abr[f5yr_index:l5yr_index,:,:], y_range, x_range)) - np.nanmean(area_mean(SST_cold_PI, y_range, x_range))
        delta_SST[i, 1] = np.nanmean(area_mean(SST_hot_abr[f5yr_index:l5yr_index,:,:], y_range, x_range)) - np.nanmean(area_mean(SST_hot_PI, y_range, x_range))

        # p - e.
        p_e_cold_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['p_e']).reshape(shape_mon_abr[deck_nas2[i]])
        p_e_cold_abr[np.logical_not(ind_last20_Cold_abr)] = np.nan
        p_e_hot_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['p_e']).reshape(shape_mon_abr[deck_nas2[i]])
        p_e_hot_abr[np.logical_not(ind_last20_Hot_abr)] = np.nan
        p_e_cold_PI = deepcopy(output_2lrm_metric_actual_PI[deck_nas2[i]]['p_e']).reshape(shape_mon_pi[deck_nas2[i]])
        p_e_cold_PI[np.logical_not(ind_last20_Cold_PI)] = np.nan
        p_e_hot_PI = deepcopy(output_2lrm_metric_actual_PI[deck_nas2[i]]['p_e']).reshape(shape_mon_pi[deck_nas2[i]])
        p_e_hot_PI[np.logical_not(ind_last20_Hot_PI)] = np.nan

        delta_p_e[i, 0] = np.nanmean(area_mean(p_e_cold_abr[f5yr_index:l5yr_index,:,:], y_range, x_range)) - np.nanmean(area_mean(p_e_cold_PI, y_range, x_range))
        delta_p_e[i, 1] = np.nanmean(area_mean(p_e_hot_abr[f5yr_index:l5yr_index,:,:], y_range, x_range)) - np.nanmean(area_mean(p_e_hot_PI, y_range, x_range))

        # LTS.
        LTS_cold_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['LTS']).reshape(shape_mon_abr[deck_nas2[i]])
        LTS_cold_abr[np.logical_not(ind_last20_Cold_abr)] = np.nan
        LTS_hot_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['LTS']).reshape(shape_mon_abr[deck_nas2[i]])
        LTS_hot_abr[np.logical_not(ind_last20_Hot_abr)] = np.nan
        LTS_cold_PI = deepcopy(output_2lrm_metric_actual_PI[deck_nas2[i]]['LTS']).reshape(shape_mon_pi[deck_nas2[i]])
        LTS_cold_PI[np.logical_not(ind_last20_Cold_PI)] = np.nan
        LTS_hot_PI = deepcopy(output_2lrm_metric_actual_PI[deck_nas2[i]]['LTS']).reshape(shape_mon_pi[deck_nas2[i]])
        LTS_hot_PI[np.logical_not(ind_last20_Hot_PI)] = np.nan

        delta_LTS[i, 0] = np.nanmean(area_mean(LTS_cold_abr[f5yr_index:l5yr_index,:,:], y_range, x_range)) - np.nanmean(area_mean(LTS_cold_PI, y_range, x_range))
        delta_LTS[i, 1] = np.nanmean(area_mean(LTS_hot_abr[f5yr_index:l5yr_index,:,:], y_range, x_range)) - np.nanmean(area_mean(LTS_hot_PI, y_range, x_range))

        # SUB_500.
        SUB_cold_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['SUB']).reshape(shape_mon_abr[deck_nas2[i]])
        SUB_cold_abr[np.logical_not(ind_last20_Cold_abr)] = np.nan
        SUB_hot_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['SUB']).reshape(shape_mon_abr[deck_nas2[i]])
        SUB_hot_abr[np.logical_not(ind_last20_Hot_abr)] = np.nan
        SUB_cold_PI = deepcopy(output_2lrm_metric_actual_PI[deck_nas2[i]]['SUB']).reshape(shape_mon_pi[deck_nas2[i]])
        SUB_cold_PI[np.logical_not(ind_last20_Cold_PI)] = np.nan
        SUB_hot_PI = deepcopy(output_2lrm_metric_actual_PI[deck_nas2[i]]['SUB']).reshape(shape_mon_pi[deck_nas2[i]])
        SUB_hot_PI[np.logical_not(ind_last20_Hot_PI)] = np.nan

        delta_SUB[i, 0] = np.nanmean(area_mean(SUB_cold_abr[f5yr_index:l5yr_index,:,:], y_range, x_range)) - np.nanmean(area_mean(SUB_cold_PI, y_range, x_range))
        delta_SUB[i, 1] = np.nanmean(area_mean(SUB_hot_abr[f5yr_index:l5yr_index,:,:], y_range, x_range)) - np.nanmean(area_mean(SUB_hot_PI, y_range, x_range))

        # gmt.
        gmt_abr = deepcopy(output_2lrm_metric_actual_abr[deck_nas2[i]]['gmt'])
        delta_gmt[i] = np.nanmean(gmt_abr[f5yr_index:l5yr_index])

    # print(delta_gmt)
    # print(sigmaLWP_r2)
    # print(sigmaLWP_ALL)
    # print(delta_SUB)
    # print(delta_LWP_ALL)
    
    
    # changes of variables;
    # standardized change of Xi scaled by gmt, lwp change scaled by 'gmt':
    
    for i in range(len(deck_nas2)):
        
        dX_dTg_r1[i, :] = (np.asarray([delta_SST[i, 0], delta_p_e[i, 0], delta_LTS[i, 0], delta_SUB[i,0]] / delta_gmt[i]).flatten()) / sigmaXi_r1[i, :]  # Cold
        dX_dTg_r2[i, :] = (np.asarray([delta_SST[i, 1], delta_p_e[i, 1], delta_LTS[i, 1], delta_SUB[i,1]] / delta_gmt[i]).flatten()) / sigmaXi_r2[i, :]  # Hot
        delta_LWP_dTg[i] = (delta_LWP_ALL[i] / delta_gmt[i])
        delta_LWP_dTgr1[i] = (1000. * delta_LWP[i, 0]) / delta_gmt[i]
        delta_LWP_dTgr2[i] = (1000. * delta_LWP[i, 1]) / delta_gmt[i]
        
    
    # standardized coefficient of GCM:

    coef_cold = []
    intp_cold = []
    coef_hot = []
    intp_hot = []

    for i in range(len(deck_nas2)):
        # print(output_2lrm_coef_LWP[deck_nas2[i]].shape)

        a_lt = output_2lrm_coef_LWP[deck_nas2[i]][0][0].copy()
        a_le = output_2lrm_coef_LWP[deck_nas2[i]][1][0].copy()
        a0_lt = output_2lrm_coef_LWP[deck_nas2[i]][0][1].copy()
        a0_le = output_2lrm_coef_LWP[deck_nas2[i]][1][1].copy()

        coef_cold.append(array(a_lt))
        coef_hot.append(array(a_le))
        intp_cold.append(array(a0_lt))
        intp_hot.append(array(a0_le))

    for j in range(len(deck_nas2)):
        # print(coef_cold[j].shape)
        # print(sigmaXi_r1[j,:].shape)
        stcoef_r1[j, :] = 1000. * coef_cold[j] * sigmaXi_r1[j, :]  # Cold
        stcoef_r2[j, :] = 1000. * coef_hot[j] * sigmaXi_r2[j, :]  # Hot 


    # print(coef_hot)
    # print(stcoef_r2[:,:])
    
    
    # Observational standardized coefficient of LWP to CCFs:
    # training
    valid_range1=[2013, 1, 15]
    valid_range2=[2016, 12, 31]   # 8 years
    # Predicting
    valid_range3=[1997, 1, 15]
    valid_range4=[2008, 12, 31]   # 12 years
    
    # OBS coef of LWP to CCFs:
    
    # Function #1 loopping through variables space to find the cut-offs of LRM (Multi-Linear Regression Model).
    dict_training, lats_Array, lons_Array, times_Array_training = Pre_processing(s_range, x_range, y_range, valid_range1 = valid_range1, valid_range2 = valid_range2)
    dict_predict, lats_Array, lons_Array, times_Array_predict = Pre_processing(s_range, x_range, y_range, valid_range1 = valid_range3, valid_range2 = valid_range4)

    # Loop_OBS_LRM(dict_training, dict_predict, s_range, x_range, y_range)

    # Function #2 training LRM with using no cut-off, then use it to predict another historical period.

    predict_result_1r = fitLRMobs_1(dict_training, dict_predict, s_range, y_range, x_range, lats_Array, lons_Array)
    coef_obs = predict_result_1r['coef_dict']
    
    std_dev_LWP = predict_result_1r['std_LWP_training']

    print("coef_obs: ", coef_obs[0])

    sigmaXi_r2_obs = np.full((4), 0.0)
    sigmaXi_r2_obs = np.asarray([np.nanstd(dict_predict['SST']), np.nanstd(dict_predict['p_e']), np.nanstd(dict_predict['LTS']), np.nanstd(dict_predict['SUB'])])
    print("sigmaXi_r2_obs: ", sigmaXi_r2_obs)

    a = 1000. * coef_obs[0] * sigmaXi_r2_obs
    print("standard coef_obs in g/m2/Sigma: ", a)
    
    
    ## Plotting: 
    delta_LWP_dTg_LRM_cold = np.full(len(deck_nas2), 0.0)
    delta_LWP_dTg_LRM_hot = np.full(len(deck_nas2), 0.0)
    delta_LWP_dTg_OBS_warm = np.full(len(deck_nas2), 0.00)
    delta_LWP_dTg_GCM = delta_LWP_dTg * 1000.
    delta_LWP_dTg_LRM_all = np.full(len(deck_nas2), 0.00)
    print(delta_LWP_dTg_GCM)
    for i in range(len(deck_nas2)):

        delta_LWP_dTg_LRM_cold[i] = np.sum(stcoef_r1[i,:] * dX_dTg_r1[i,:])
        delta_LWP_dTg_LRM_hot[i] = np.sum(stcoef_r2[i,:] * dX_dTg_r2[i,:])
        # print(stcoef_obs)
        # print(dX_dTg_r2[i,:])
        delta_LWP_dTg_OBS_warm[i] = np.sum(a * dX_dTg_r2[i,:])
        delta_LWP_dTg_LRM_all[i] = delta_LWP_dTg_LRM_cold[i] + delta_LWP_dTg_LRM_hot[i]

    print(delta_LWP_dTg_LRM_all)
    # print(delta_LWP_dTg_LRM_cold)
    # print(delta_LWP_dTg_OBS_warm)
    
    
    # standardized contribution of LWP from each individual CCFs:
    
    dC_dTg_Cs1 = stcoef_r1 * dX_dTg_r1  # Cold
    dC_dTg_Cs2 = stcoef_r2 * dX_dTg_r2  # Hot

    CC_ccfdrivenr1 = append(delta_LWP_dTg_LRM_cold.reshape(-1,1), dC_dTg_Cs1, axis =1)
    CC_ccfdrivenr2 = append(delta_LWP_dTg_LRM_hot.reshape(-1,1), dC_dTg_Cs2, axis =1)

    CC_ccfdriven_withtruemodelr1 = append(delta_LWP_dTgr1.reshape(-1,1), CC_ccfdrivenr1, axis = 1)
    CC_ccfdriven_withtruemodelr2 = append(delta_LWP_dTgr2.reshape(-1,1), CC_ccfdrivenr2, axis = 1)
    
    # Box plots:

    # subplot (a):

    fig7, ax7 = plt.subplots(2, 1, figsize = (10, 9.5))

    parameters = {'axes.labelsize': 22, 'legend.fontsize': 12,
              'axes.titlesize': 14, 'xtick.labelsize': 17, 'ytick.labelsize': 17}
    plt.rcParams.update(parameters)
    # specific model No.
    # CESM2
    model_i = 2

    # Data Frame:
    d1 = {'col1': arange(0, 50*4), 'value': stcoef_r2.ravel(), 'CCFs': array([r'$Ts$', r'$P - E$', r'$LTS$', r'$\omega_{500}$'] * 50)}
    data1  = pd.DataFrame(data=d1, index=arange(0, 50 * 4))  # Hot
    d_specGCM1 = {'col1': arange(0, 4), 'value': stcoef_r2[model_i,:].ravel(), 'CCFs': array([r'$Ts$', r'$P - E$', r'$LTS$', r'$\omega_{500}$'])}
    d_specOBS1  = {'col1': arange(0, 4), 'value': a.ravel(), 'CCFs': array([r'$Ts$', r'$P - E$', r'$LTS$', r'$\omega_{500}$'])}

    d2 = {'col1': arange(0, 50*4), 'value': stcoef_r1.ravel(), 'CCFs': array([r'$Ts$', r'$P - E$', r'$LTS$', r'$\omega_{500}$'] * 50)}
    data2  = pd.DataFrame(data=d2, index=arange(0, 50 * 4))
    d_specGCM2 = {'col1': arange(0, 4), 'value': stcoef_r1[model_i,:].ravel(), 'CCFs': array([r'$Ts$', r'$P - E$', r'$LTS$', r'$\omega_{500}$'])}

    # Coefficient plot

    bplot1 = sns.boxplot(ax=ax7[0], x = "CCFs", y = "value", data = d1, width = 0.45, linewidth = 2.6, whis = 1.3)
    stplot1 = sns.stripplot(ax=ax7[0], x = "CCFs", y = "value", data = d1, color="lightblue", jitter=0.2, size = 5)
    stplot_specGCM1 = sns.stripplot(ax=ax7[0], x = "CCFs", y = "value", data = d_specGCM1, color="orange", marker = 'D', jitter = 0.2, size = 11)
    stplot_specOBS1 = sns.stripplot(ax=ax7[0], x = "CCFs", y = "value", data = d_specOBS1, color="blue", marker = '>', jitter = 0.023, size = 14)
    # ax7[0].set_title(" Hot ", loc = 'center', fontsize = 18, pad = 12)
    ax7[0].set_ylim([-20, 60])
    bplot2 = sns.boxplot(ax=ax7[1], x = "CCFs", y = "value", data = d2, width = 0.45, linewidth = 2.6, whis = 1.3)
    stplot2 = sns.stripplot(ax=ax7[1], x = "CCFs", y = "value", data = d2, color="lightblue", jitter=0.2, size = 5)
    stplot_specGCM2 = sns.stripplot(ax=ax7[1], x = "CCFs", y = "value", data = d_specGCM2, color="orange", marker = 'D', jitter=0.2, size = 11)
    # ax7[1].set_title(" Cold ", loc = 'center', fontsize = 18, pad = 12)
    ax7[1].set_ylim([-20, 60])

    ax7[0].text(-0.38, 70., r"$\ (a)\ \partial LWP/ \partial X_{i}$", fontsize = 21, horizontalalignment = 'left')

    # Plot setting

    ax7[0].axhline(0., c = 'k', linestyle = '-', linewidth = 2.4, zorder=0)
    ax7[1].axhline(0., c = 'k', linestyle = '-', linewidth = 2.4, zorder=0)
    ax7[0].set_ylabel(" Warm \n" + r"$\ g/\ m^{2}/\ \sigma $", fontsize = 17)
    ax7[1].set_ylabel(" Cold \n" + r"$\ g/\ m^{2}/\ \sigma $", fontsize = 17)

    # seaborn setting:
    CCFs2 = [r'$Ts$', r'$P - E$', r'$LTS$', r'$\omega_{500}$']
    CCFs_colors2 = ["#1b9e77", "#1f78b4", "#7570b3", "#d95f02"]

    color_dict2 = dict(zip(CCFs2, CCFs_colors2))

    for i in range(0, 4):
        mybox1 = bplot1.artists[i]
        mybox1.set_facecolor(color_dict2[CCFs2[i]])

        mybox2 = bplot2.artists[i]
        mybox2.set_facecolor(color_dict2[CCFs2[i]])

    sns.set_style("whitegrid", {"grid.color": "gray", "grid.linestyle": "--"})

    # plt.subplots_adjust(left=0.125, bottom = 0.105, right=0.9, top = 0.78, wspace = 0.14, hspace = 0.470)

    # plt.savefig(path6+"Box_panel(a)_sensitivity.jpg", bbox_inches = 'tight', dpi = 250)
    plt.savefig(path6 + "Fig5(a).jpg", bbox_inches = 'tight', dpi = 425)
    
    # Box plots:
    
    # subplot (b):

    fig72, ax72 = plt.subplots(2, 1, figsize = (10, 9.5))

    parameters = {'axes.labelsize': 22, 'legend.fontsize': 12,
              'axes.titlesize': 14, 'xtick.labelsize': 17, 'ytick.labelsize': 17}
    plt.rcParams.update(parameters)
    # specific model No.
    # CESM2 (2); TaiESM1 (29)
    model_i = 2

    # Data Frame:
    d3 = {'col1': arange(0, 50*4), 'value': dX_dTg_r2.ravel(), 'CCFs': array([r'$Ts$', r'$P - E$', r'$LTS$', r'$\omega_{500}$'] * 50)}
    data3 = pd.DataFrame(data=d3, index=arange(0, 50 * 4))  # Hot
    d_specGCM3 = {'col1': arange(0, 4), 'value': dX_dTg_r2[model_i,:].ravel(), 'CCFs': array([r'$Ts$', r'$P - E$', r'$LTS$', r'$\omega_{500}$'])}

    d4 = {'col1': arange(0, 50*4), 'value': dX_dTg_r1.ravel(), 'CCFs': array([r'$Ts$', r'$P - E$', r'$LTS$', r'$\omega_{500}$'] * 50)}
    data4 = pd.DataFrame(data=d4, index=arange(0, 50 * 4))
    d_specGCM4 = {'col1': arange(0, 4), 'value': dX_dTg_r1[model_i,:].ravel(), 'CCFs': array([r'$Ts$', r'$P - E$', r'$LTS$', r'$\omega_{500}$'])}

    # Temperature-mediated CCF Change:

    bplot3 = sns.boxplot(ax=ax72[0], x = "CCFs", y = "value", data = d3, width = 0.45, linewidth = 2.6, whis = 1.3)
    stplot3 = sns.stripplot(ax=ax72[0], x = "CCFs", y = "value", data = d3, color="lightblue", jitter=0.2, size = 5)
    stplot_specGCM3 = sns.stripplot(ax=ax72[0], x = "CCFs", y = "value", data = d_specGCM3, color="orange", marker = 'D', jitter = 0.2, size = 11)

    # ax7[0].set_title(" Hot ", loc = 'center', fontsize = 18, pad = 12)
    ax72[0].set_ylim([-0.10, 0.27])
    bplot4 = sns.boxplot(ax=ax72[1], x = "CCFs", y = "value", data = d4, width = 0.45, linewidth = 2.6, whis = 1.3)
    stplot4 = sns.stripplot(ax=ax72[1], x = "CCFs", y = "value", data = d4, color="lightblue", jitter=0.2, size = 5)
    stplot_specGCM4 = sns.stripplot(ax=ax72[1], x = "CCFs", y = "value", data = d_specGCM4, color="orange", marker = 'D', jitter=0.2, size = 11)
    # ax72[1].set_title(" Cold ", loc = 'center', fontsize = 18, pad = 12)
    ax72[1].set_ylim([-0.10, 0.27])

    ax72[0].text(-0.38, 0.319, r"$\ (b)\ \Delta X_{i}/ \Delta GMT\ $", fontsize = 21, horizontalalignment = 'left')

    # Plot setting
    ax72[0].axhline(0., c = 'k', linestyle = '-', linewidth = 2.4, zorder=0)
    ax72[1].axhline(0., c = 'k', linestyle = '-', linewidth = 2.4, zorder=0)
    ax72[0].set_ylabel(" Warm \n" + r"$\ \sigma/\ K $", fontsize = 17)
    ax72[1].set_ylabel(" Cold \n" + r"$\ \sigma/\ K $", fontsize = 17)

    # seaborn setting:
    CCFs2 = [r'$Ts$', r'$P - E$', r'$LTS$', r'$\omega_{500}$']
    CCFs_colors2 = ["#1b9e77", "#1f78b4", "#7570b3", "#d95f02"]

    color_dict2 = dict(zip(CCFs2, CCFs_colors2))

    for j in range(0, 4):
        mybox3 = bplot3.artists[j]
        mybox3.set_facecolor(color_dict2[CCFs2[j]])

        mybox4 = bplot4.artists[j]
        mybox4.set_facecolor(color_dict2[CCFs2[j]])

    sns.set_style("whitegrid", {"grid.color": "gray", "grid.linestyle": "--"})

    # plt.subplots_adjust(left=0.125, bottom = 0.105, right=0.9, top = 0.78, wspace = 0.14, hspace = 0.470)
    # plt.savefig(path6+"Box_panel(b)_changes_of_CCF.jpg", bbox_inches = 'tight', dpi = 250)
    plt.savefig(path6 + "Fig5(b).jpg", bbox_inches = 'tight', dpi = 425)
    
    
    # Box plots:

    # subplot (c):

    fig73, ax73 = plt.subplots(2, 1, figsize = (10, 9.5))

    parameters = {'axes.labelsize': 22, 'legend.fontsize': 12,
              'axes.titlesize': 14, 'xtick.labelsize': 17, 'ytick.labelsize': 17}
    plt.rcParams.update(parameters)
    # specific model No.
    # CESM2(2); TaiESM1(29)
    model_i = 2

    # Data Frame:
    d5 = {'col1': arange(0, 50*6), 'value': CC_ccfdriven_withtruemodelr2.ravel(), 'CCFs': array([' ' * 8, ' ' * 4, r'$Ts$', r'$P - E$', r'$LTS$', r'$\omega_{500}$'] * 50)}
    data5 = pd.DataFrame(data=d5, index=arange(0, 50 * 6))  # Hot
    d_specGCM5 = {'col1': arange(0, 6), 'value': CC_ccfdriven_withtruemodelr2[model_i,:].ravel(), 'CCFs': array(['GCM', '  SUM ', r'$Ts$', r'$P - E$', r'$LTS$', r'$\omega_{500}$'])}

    d6 = {'col1': arange(0, 50*6), 'value': CC_ccfdriven_withtruemodelr1.ravel(), 'CCFs': array([' ' * 8, ' ' * 4, r'$Ts$', r'$P - E$', r'$LTS$', r'$\omega_{500}$'] * 50)}
    data6 = pd.DataFrame(data=d6, index=arange(0, 50 * 6))
    d_specGCM6 = {'col1': arange(0, 6), 'value': CC_ccfdriven_withtruemodelr1[model_i,:].ravel(), 'CCFs': array(['GCM', '  SUM ', r'$Ts$', r'$P - E$', r'$LTS$', r'$\omega_{500}$'])}

    # Cloud-Controlling-factor's individual components and the SUM and the GCM true:

    bplot5 = sns.boxplot(ax=ax73[0], x = "CCFs", y = "value", data = d5, width = 0.45, linewidth = 2.6, whis = 1.3)
    stplot3 = sns.stripplot(ax=ax73[0], x = "CCFs", y = "value", data = d5, color="lightblue", jitter=0.2, size = 5)
    stplot_specGCM3 = sns.stripplot(ax=ax73[0], x = "CCFs", y = "value", data = d_specGCM5, color="orange", marker = 'D', jitter = 0.2, size = 11)
    # ax7[0].set_title(" Hot ", loc = 'center', fontsize = 18, pad = 12)
    R2_Totalccfsdriven_LWP = r2_score(CC_ccfdriven_withtruemodelr2[:, 0], CC_ccfdriven_withtruemodelr2[:, 1])
    r_Totalccfsdriven_LWP, p_value = pearsonr(CC_ccfdriven_withtruemodelr2[:, 1], CC_ccfdriven_withtruemodelr2[:, 0]) 
    ax73[0].annotate(r"$R^{2}\ =\ %.2f$" % R2_Totalccfsdriven_LWP, xy=(0.45, 2.35), textcoords = 'axes fraction', xytext=(0.12, 0.88), fontsize = 16)

    ax73[0].set_ylim([-4.5, 8.7])

    bplot6 = sns.boxplot(ax=ax73[1], x = "CCFs", y = "value", data = d6, width = 0.45, linewidth = 2.6, whis = 1.3)
    stplot6 = sns.stripplot(ax=ax73[1], x = "CCFs", y = "value", data = d6, color="lightblue", jitter=0.2, size = 5)
    stplot_specGCM6 = sns.stripplot(ax=ax73[1], x = "CCFs", y = "value", data = d_specGCM6, color="orange", marker = 'D', jitter=0.2, size = 11)
    # ax72[1].set_title(" Cold ", loc = 'center', fontsize = 18, pad = 12)
    R2_Totalccfsdriven_LWP = r2_score(CC_ccfdriven_withtruemodelr1[:, 0], CC_ccfdriven_withtruemodelr1[:, 1])
    r_Totalccfsdriven_LWP, p_value = pearsonr(CC_ccfdriven_withtruemodelr1[:, 1], CC_ccfdriven_withtruemodelr1[:, 0]) 
    ax73[1].annotate(r"$R^{2}\ =\ %.2f$" % R2_Totalccfsdriven_LWP, xy=(0.45, 3.44), textcoords = 'axes fraction', xytext=(0.12, 0.82), fontsize = 16)
    ax73[1].set_ylim([-4.5, 8.7])

    ax73[1].text(-0.36, 25.642, r"$ (c)\ \Delta LWP/ \Delta GMT\ $", fontsize = 21, horizontalalignment = 'left')

    # Plot setting
    ax73[0].axhline(0., c = 'k', linestyle = '-', linewidth = 2.4, zorder=0)
    ax73[1].axhline(0., c = 'k', linestyle = '-', linewidth = 2.4, zorder=0)

    ax73[0].axvline(1.46, c = 'k', linestyle = '--', linewidth = 2.4)
    ax73[1].axvline(1.46, c = 'k', linestyle = '--', linewidth = 2.4)
    ax73[0].set_ylabel(" Warm \n" + r"$\ g/\ m^{2}/\ K $", fontsize = 17)
    ax73[1].set_ylabel(" Cold \n" + r"$\ g/\ m^{2}/\ K $", fontsize = 17)

    # seaborn setting:
    CCFs = ['GCM', 'SUM', r'$Ts$', r'$P - E$', r'$LTS$', r'$omega_{500}$']
    CCFs_colors = ["black", "#67a9cf", "#1b9e77", "#1f78b4", "#7570b3", "#d95f02"]

    CCFs2 = [r'$Ts$', r'$P - E$', r'$LTS$', r'$\omega_{500}$']
    CCFs_colors2 = ["#1b9e77", "#1f78b4", "#7570b3", "#d95f02"]

    color_dict =  dict(zip(CCFs, CCFs_colors))
    color_dict2 = dict(zip(CCFs2, CCFs_colors2))

    for j in range(0, 6):
        mybox5 = bplot5.artists[j]
        mybox5.set_facecolor(color_dict[CCFs[j]])

        mybox6 = bplot6.artists[j]
        mybox6.set_facecolor(color_dict[CCFs[j]])

    sns.set_style("whitegrid", {"grid.color": "gray", "grid.linestyle": "--"})

    # plt.subplots_adjust(left=0.125, bottom = 0.105, right=0.9, top = 0.78, wspace = 0.14, hspace = 0.470)
    # plt.savefig(path6 + "Box_panel(c)_CCFs_driven_LWPchanges.jpg", bbox_inches = 'tight', dpi = 250)
    plt.savefig(path6 + "Fig5(c).jpg", bbox_inches = 'tight', dpi = 425)
    
    return 0