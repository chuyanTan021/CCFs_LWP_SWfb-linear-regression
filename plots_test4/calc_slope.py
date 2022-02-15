## calculate the derivative of a sequence(list) of values: 

import netCDF4
import numpy as np
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
from calc_LRM_metrics import *
from fitLRM_cy import *
from run_simple_cmip6 import *

def calc_slope(x, y):
    
    x1 =  np.array(x[:-1], dtype=float)
    x2 =  np.array(x[1:] , dtype=float)
    y1  = np.array(y[:-1], dtype=float)
    y2  = np.array(y[1:] , dtype=float)
    
    
    m   = (y2-y1) /(x2-x1)
    
    return m



def derivative_dskinT(SST_array, LWP_array, bin_Number):
    
    #.. \SST_array as X axis; LWP_array as Y axis; they are in the shape of (times, lat_bin, lon_bin);
    #cut_Number = 80
    cut_Number  = bin_Number
    
    X_sequence  = SST_array.flatten()
    #print(X_sequence.shape)
    
    Y_sequence  = LWP_array.flatten()
    
    bin_edges = (np.nanpercentile(X_sequence, 5), np.nanpercentile(X_sequence, 95))#
    
    bin_means, bin_edges, bin_Number  =   binned_statistic(X_sequence , Y_sequence, statistic= 'mean', bins= cut_Number, range = bin_edges)
    
    
    LWP_foreach_sst =  bin_means
    #print(cut_Number, LWP_foreach_sst.shape)
    each_sst =  np.linspace(np.nanpercentile(X_sequence, 5), np.nanpercentile(X_sequence, 95) , cut_Number)
    
    
    slope_for_dY_dX   =  calc_slope(each_sst, LWP_foreach_sst)
    #print(slope_for_dY_dX.shape)
    
    
    return each_sst, np.array(LWP_foreach_sst), slope_for_dY_dX
    
    
