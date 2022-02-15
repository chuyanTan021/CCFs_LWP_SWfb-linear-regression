#..S array has 3 dimensions: (times, lat, lon)

import numpy as np
import matplotlib.pyplot as plt
#import xarray as xr
#import PyNIO as Nio
import pandas as pd

from scipy import stats
from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score
#from read_hs_file import read_var_mod


def area_mean(S, lats, lons):
    
    '''..Only for 1 final value..
    '''
    GMT  = np.zeros(S.shape[0])
    for i in np.arange(S.shape[0]):
        
        S_time_step  = S[i,:,:]
        #..remove the NaN value within the 2-D array and squeeze it to 1-D:
        ind1 =  np.isnan(S_time_step)==False
        #..weighted by cos(lat):
        xlon, ylat  = np.meshgrid(lons, lats)

        weighted_metrix1 =  np.cos(np.deg2rad(ylat))   #..metrix has the same shape as tas/lwp, its value = cos(lat)
        toc1  = np.sum(weighted_metrix1[ind1])   #..total of cos(lat metrix) for global region

        S_weighted =  S_time_step[ind1] * weighted_metrix1[ind1] /  toc1
        
        GMT[i]  = np.sum(S_weighted)
    
    
    
    return GMT