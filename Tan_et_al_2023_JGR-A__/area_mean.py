# ..data array has 3 dimensions: (times, lat, lon)

import numpy as np

import matplotlib.pyplot as plt
from xarray import DataArray
# import PyNIO as Nio # deprecated
import pandas as pd

from scipy import stats
from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score
#from read_hs_file import read_var_mod



def area_mean(data, lats, lons):
    # This function is for processing the latitudinal weighting of 3-D array.
    '''..3D (time, lat, lon) would reduce to 1D (time)..
    '''
    Time_series = np.zeros(data.shape[0])
    for i in np.arange(data.shape[0]):
        
        S_time_step = data[i,:,:]
        # remove the NaN value within the 2-D array and squeeze to 1-D (indexing):
        ind1 = np.isnan(S_time_step)==False
        # weighted by cosine(lat):
        xlon, ylat = np.meshgrid(lons, lats)

        weighted_metrix1 = np.cos(np.deg2rad(ylat))  # lat matrix
        toc1 = np.sum(weighted_metrix1[ind1])   # total of cos(lat matrix) for the defined region of lat X lon

        S_weighted = S_time_step[ind1] * weighted_metrix1[ind1] / toc1
        
        Time_series[i] = np.sum(S_weighted)

    return Time_series


def latitude_mean(X, lats, lons, lat_range=[-85., -40.]):
    """
    Calculate the area(latitudinal)-mean of a 3-d metrics.
    """
    #### X  in shape: time (models), lat, lon
    
    S = np.zeros(X.shape[0])
    for i in np.arange(X.shape[0]):
        
        X_time_step = X[i,:,:]
        xx = np.nanmean(X_time_step, axis=(1)) ## now a vector in latitude- no weights in time or longitude.
        ind1 = (lats<max(lat_range)) & (lats>min(lat_range))
        # Remove the NaN value in 1-D vector
        ind_truevector = np.isnan(xx)==False
        # Combined indexes which both satisfy the Latitude range & are not NaN
        ind2 = np.logical_and(ind1, ind_truevector)
        # area weighting:
        latrad = np.cos(lats*np.pi/ 180.)
        xx_weight_mean = np.sum(xx[ind2]*latrad[ind2])/np.sum(latrad[ind2])
        
        S[i] = xx_weight_mean
    return S



def area_grid(lat, lon):
    """
    Calculate the area of each grid cell
    Area is in square meters
    
    Input
    -----------
    lat: vector of latitude in degrees
    lon: vector of longitude in degrees
    
    Output
    -----------
    area: grid-cell area in square-meters with dimensions, [lat,lon]
    
    Notes
    -----------
    Based on the function in
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """
    
    

    xlon, ylat = np.meshgrid(lon, lat)
    R = earth_radius(ylat)

    dlat = np.deg2rad(np.gradient(ylat, axis=0))
    dlon = np.deg2rad(np.gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * np.cos(np.deg2rad(ylat))

    area = dy * dx

    xda = DataArray(
        area,
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon},
        attrs={
            "long_name": "area_per_pixel",
            "description": "area per pixel",
            "units": "m^2",
        },
    )
    return xda



def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid
    defined by WGS84
    
    Input
    ---------
    lat: vector or latitudes in degrees  
    
    Output
    ----------
    r: vector of radius in meters
    
    Notes
    -----------
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    '''
    

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)
    
    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = np.deg2rad(lat)
    lat_gc = np.arctan( (1-e2)*np.tan(lat) )

    # radius equation
    # see equation 3-107 in WGS84
    r = (
        (a * (1 - e2)**0.5) 
         / (1 - (e2 * np.cos(lat_gc)**2))**0.5 
        )

    return r

    '''
    How to use the above code(Feb.26th, 2022 Added): 
    ###  "ds" is the data he used in this example, read by 'xr.open_dataset()' method
    
    
    # area dataArray 
    da_area = area_grid(ds['latitude'], ds['longitude'])
    # 总面积
    total_area = da_area.sum(['latitude','longitude'])
    # 由网格单元面积加权的温度
    temp_weighted = (ds['temperature']*da_area) / total_area
    
    # 面积加权平均温度
    temp_weighted_mean = temp_weighted.sum(['latitude','longitude'])
    '''
