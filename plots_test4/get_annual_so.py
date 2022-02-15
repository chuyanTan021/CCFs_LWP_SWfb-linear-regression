# calculate Annual-mean, or Southern-Ocean region only data


import netCDF4
from numpy import *
import matplotlib.pyplot as plt
import xarray as xr
import PyNIO as Nio
import pandas as pd

from scipy.stats import *
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm


def get_annual_so(dict_rawdata, dict_names, shape_time, lat_si0, lat_si1, shape_lon):
    #..'dict_rawdat' : originally in monthly data, all variables are 3D(time, lat, lon) data in the same shape;
    #..'shape_time' : # of Months, which as the 1st dimension in each variables INSIDE 'dict_rawdata';
    #..'dict_names' : the name string list (or a dict) of each variables Inside 'dict_rawdata';

    dict_yr  = {}
    shape_lat_so = int(lat_si1) - int(lat_si0)
    
    layover_yr  = zeros((shape_time//12, shape_lat_so, shape_lon))



    for a in range(len(dict_names)):
        a_array = dict_rawdata[dict_names[a]]
    
        for i in range(shape_time//12):
            #.. '//' representing 'int' division operation
            
            layover_yr[i,:,:]  = nanmean(a_array[i*12:(i+1)*12, lat_si0:lat_si1, :], axis=0)
        
        
        dict_yr[dict_names[a]+'_yr'] =  layover_yr


    return dict_yr



def get_annual(dict_rawdata, dict_names, shape_time, shape_lat, shape_lon):
    #..'dict_rawdat' : originally in monthly data, all variables are 3D(time, lat, lon) data in the same shape;
    #..'shape_time' : # of Months, which as the 1st dimension in each variables INSIDE 'dict_rawdata';
    #..'dict_names' : the name string list (or a dict) of each variables Inside 'dict_rawdata';

    dict_yr  = {}

    layover_yr  = zeros((shape_time//12, shape_lat, shape_lon))



    for a in range(len(dict_names)):
        a_array = dict_rawdata[dict_names[a]]
    
        for i in range(shape_time//12):
            #.. '//' representing 'int' division operation
            
            layover_yr[i,:,:]  = nanmean(a_array[i*12:(i+1)*12,:,:], axis=0)
        
        
        dict_yr[dict_names[a]+'_yr'] =  layover_yr


    return dict_yr