# main Funcion

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
from get_annual_so import *
from calc_LRM_metric import *



def main():
    #exp = 'historical'
    exp = 'piControl'
    #exp = 'amip'
    # exp = 'abrupt-4xCO2'
    
    ACCESSCM2 = {'modn': 'ACCESS-CM2', 'consort': 'CSIRO-ARCCSS', 'cmip': 'cmip6',
                 'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}#..dont have 'clwvi' variable, even in esgf-node website
    BCCESM1 = {'modn': 'BCC-ESM1', 'consort': 'BCC', 'cmip': 'cmip6',
               'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    CAMSCSM1 = {'modn': 'CAMS-CSM1-0', 'consort': 'CAMS', 'cmip': 'cmip6',
            'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    
    CanESM5 = {'modn': 'CanESM5', 'consort': 'CCCma', 'cmip': 'cmip6',
               'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    CESM2 = {'modn': 'CESM2', 'consort': 'NCAR', 'cmip': 'cmip6',
             'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    CESM2FV2 = {'modn': 'CESM2-FV2', 'consort': 'NCAR', 'cmip': 'cmip6',
             'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    
    CESM2WACCM = {'modn': 'CESM2-WACCM', 'consort': 'NCAR', 'cmip': 'cmip6',
             'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    CNRMCM6 = {'modn': 'CNRM-CM6-1', 'consort': 'CNRM-CERFACS', 'cmip': 'cmip6',
               'exper': exp, 'ensmem': 'r1i1p1f2', 'gg': 'gr', "typevar": 'Amon'}# time doesn't corresponding for 'evspsbl' in  'pi-Control' exper
    CNRMESM2 = {'modn': 'CNRM-ESM2-1', 'consort': 'CNRM-CERFACS', 'cmip': 'cmip6', 
               'exper': exp, 'ensmem': 'r1i1p1f2', 'gg': 'gr', "typevar": 'Amon'}
    E3SM10 = {'modn': 'E3SM-1-0', 'consort': 'E3SM-Project', 'cmip': 'cmip6',
              'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gr', "typevar": 'Amon'}
    FGOALSg3 = {'modn': 'FGOALS-g3', 'consort': 'CAS', 'cmip': 'cmip6',
                'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    GFDLCM4 = {'modn': 'GFDL-CM4', 'consort': 'NOAA-GFDL', 'cmip': 'cmip6',
               'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gr1', "typevar": 'Amon'}
    # repaired Dec.30th
    
    GISSE21G = {'modn': 'GISS-E2-1-G', 'consort': 'NASA-GISS', 'cmip': 'cmip6',
                'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    GISSE21H = {'modn': 'GISS-E2-1-H', 'consort': 'NASA-GISS', 'cmip': 'cmip6',
                'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    
    HADGEM3 = {'modn': 'HadGEM3-GC31-LL', 'consort': 'MOHC', 'cmip': 'cmip6',
               'exper': 'abrupt-4xCO2', 'ensmem': 'r1i1p1f3', 'gg': 'gn', "typevar": 'Amon'}   #  Be careful, failure due to 'day time representation'
    HADGEM3 = {'modn': 'HadGEM3-GC31-LL', 'consort': 'MOHC', 'cmip': 'cmip6',
                'exper': 'piControl', 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'} #..missing 'wap' in 'piControl' exp(Daniel says that HadGEM3-GC31 not using p-level, so don't have variables on p-level
    
    INM_CM48 = {'modn': 'INM-CM4-8', 'consort': 'INM', 'cmip': 'cmip6',
                'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gr1', "typevar": 'Amon'}
    INM_CM50 = {'modn': 'INM-CM5-0', 'consort': 'INM', 'cmip': 'cmip6',
                'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gr1', "typevar": 'Amon'}#..'/glade/' dont have 'Amon' typevar

    IPSLCM6ALR = {'modn': 'IPSL-CM6A-LR', 'consort': 'IPSL', 'cmip': 'cmip6',
                  'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gr', "typevar": 'Amon'}
    
    MPIESM12LR = {'modn': 'MPI-ESM1-2-LR', 'consort': 'MPI-M', 'cmip': 'cmip6',
                  'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    MIROC6 = {'modn': 'MIROC6', 'consort': 'MIROC', 'cmip': 'cmip6',
              'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    MIROCES2L= {'modn': 'MIROC-ES2L', 'consort': 'MIROC', 'cmip': 'cmip6',
              'exper': exp, 'ensmem': 'r1i1p1f2', 'gg': 'gn', "typevar": 'Amon'}   # don't have
    MRIESM20 = {'modn': 'MRI-ESM2-0', 'consort': 'MRI', 'cmip': 'cmip6',
                'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}
    NORESM2LM = {'modn': 'NorESM2-LM', 'consort': 'NCC', 'cmip': 'cmip6',
                 'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}# 'pr', 'tas' are not complete in 'abrupt-4xCO2', while some variables in 'piControl' still not complete
    
    SAM0={'modn': 'SAM0-UNICON', 'consort': 'SNU', 'cmip': 'cmip6',
                'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}


    
    UKESM10 = {'modn': 'UKESM1-0-LL', 'consort': 'MOHC', 'cmip': 'cmip6',
               'exper': exp, 'ensmem': 'r1i1p1f2', 'gg': 'gn', "typevar": 'Amon'}   # the same day time representation issue as 'HadGEM3'
    AWICM11MR = {'modn': 'AWI-CM-1-1-MR', 'consort': 'AWI', 'cmip': 'cmip6',
                 'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'}#..'abrupt-4xCO2' missing variables
    CMCC = {'modn': 'CMCC-CM2-SR5', 'consort': 'CMCC', 'cmip': 'cmip6',
            'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'} #..'/glade/' dont have 'abrupt-4xCO2'/'piControl' exper
    ECE = {'modn': 'EC-Earth3', 'consort': 'EC-Earth-Consortium', 'cmip': 'cmip6',
           'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gn', "typevar": 'Amon'} #..dont have 'Amon' or variable 'tas'
    #ECE has different variants for exper
    ECEV = {'modn': 'EC-Earth3-Veg', 'consort': 'EC-Earth-Consortium', 'cmip': 'cmip6',
           'exper': exp, 'ensmem': 'r1i1p1f1', 'gg': 'gr', "typevar": 'Amon'}#..too hard to operate
    #ECEV has very discrete year in 'abrupt-4xCO2', seems continued but each in one yr file in 'piControl'
    
    
    #deck = [ACCESSCM2, AWICM11MR, BCCESM1, CESM2, CESM2FV2, CESM2WACCM, CNRMESM2, CAMSCSM1, CNRMCM6, ECE, ECEV, E3SM10, GFDLCM4, GISSE21H, GISSE21G, HADGEM3, INM_CM48, IPSLCM6ALR, MRIESM20, MIROC6, MPIESM12LR, NORESM2LM, SAM0]   #..current #23, Total in #30
    # deck  = [ MPIESM12LR]   #..current #TEST 6
    # deck_nas = [ 'MPIESM12LR']
    #.deck =  [BCCESM1, CanESM5, CESM2, CESM2FV2, CESM2WACCM, CNRMESM2 , GISSE21G, GISSE21H, IPSLCM6ALR, MRIESM20, MIROC6, SAM0]   #..Dec29th, #12
    #.. add..HADGEM3, UKESM10(both them don't have 'wap' in 'pi-C' exp, and need to solve 'day number' issue in cftime.Datetime360Day object of time dimension)
    
    # deck = [BCCESM1, CanESM5, CESM2, CESM2FV2, CESM2WACCM, CNRMESM2, GISSE21G, GISSE21H,IPSLCM6ALR, MRIESM20, MIROC6, SAM0]   #..current #12
    # deck_nas  = ['BCCESM1', 'CanESM5', 'CESM2', 'CESM2FV2', 'CESM2WACCM', 'CNRMESM2', 'GISSE21G', 'GISSE21H', 'IPSLCM6ALR', 'MRIESM20', 'MIROC6', 'SAM0']   #..current #12
    
    
    deck = [BCCESM1, CanESM5, CESM2, CESM2FV2, CESM2WACCM, CNRMESM2, GISSE21G, GISSE21H, IPSLCM6ALR, MRIESM20, MIROC6, SAM0, E3SM10, FGOALSg3, GFDLCM4, CAMSCSM1,INM_CM48, MPIESM12LR]   #..current # 18
    deck_nas  = ['BCCESM1', 'CanESM5', 'CESM2', 'CESM2FV2', 'CESM2WACCM', 'CNRMESM2', 'GISSE21G', 'GISSE21H', 'IPSLCM6ALR', 'MRIESM20', 'MIROC6', 'SAM0', 'E3SM10', 'FGOALSg3', 'GFDLCM4', 'CAMSCSM1','INM_CM48', 'MPIESM12LR']   #..current #18
    
    #..HadGEM3-GC31-LL, FGOALS-g3, NorESM2-LM still not being archeved
    
    
    '''
    MP = '/glade/collections/cmip/CMIP6/'
    
    for i in range(len(deck)):
        DP =  MP + 'CMIP' + '/'+deck[i]['consort']+'/'+ deck[i]['modn'] + '/'+ deck[i]['exper'] \
            + '/'+ deck[i]['ensmem'] +'/'+ deck[i]['typevar'] + '/' + 'tas' + '/'+ deck[i]['gg'] + '/'
        if len(glob.glob(DP+ '*/*/'))!=0:
            #print('number of run stored in glade', len(glob.glob(DP+ '*/*/')))
            print('oh!', 'we have data in model: ', deck[i]['modn'])
            try:
                calc_LRM_metrics(273, 0.0, **deck[i])
                print('done ', i)
            except:
                print('fail to call function : calc_LRM_metrics', deck[i])
    '''

    
    # PLUG IN 2 CUT OFF 
    MP = '/glade/collections/cmip/CMIP6/'
    
    for i in range(len(deck)):
        DP =  MP + 'CMIP' + '/'+deck[i]['consort']+'/'+ deck[i]['modn'] + '/'+ deck[i]['exper'] \
            + '/'+ deck[i]['ensmem'] +'/'+ deck[i]['typevar'] + '/' + 'tas' + '/'+ deck[i]['gg'] + '/'
        if len(glob.glob(DP+ '*/*/'))!=0:
            #print('number of run stored in glade', len(glob.glob(DP+ '*/*/')))
            print('oh!', 'we have data in model: ', deck[i]['modn'])
            try:
                WD = '/glade/work/chuyan/Research/Cloud_CCFs_RMs/plots_test5/'

                folder =  glob.glob(WD+ deck_nas[i]+'__'+ 'STAT_pi+abr_'+'22x_31y'+'.npz')
                # print(folder)
                output_ARRAY  =  load(folder[0], allow_pickle=True)  # str(TR_sst)
                TR_sst1 = output_ARRAY['TR_minabias_SST']
                TR_sub1 = output_ARRAY['TR_minabias_SUB']
                TR_sst2  = output_ARRAY['TR_maxR2_SST']
                TR_sub2  = output_ARRAY['TR_maxR2_SUB']

                print("TR_min_abs(bias): " , TR_sst1, '  K ', TR_sub1 , ' Pa/s ')
                print("TR_large_pi_R_2: ", TR_sst2, '  K ', TR_sub2 , ' Pa/s ')
                calc_LRM_metrics(float(TR_sst1), (TR_sub1), **deck[i])
                
                print('done ', i)
            except:
                print('fail to call function : calc_LRM_metrics', deck[i])



if __name__== "__main__":
    main()
