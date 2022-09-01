import numpy as np
def latwght(x,lat):
    #### x= time, lat, lon
    xx=np.nanmean(x,axis=(0,2)) ## now a vector in latitude- no weights in time or longitude.
    latrad=np.cos(lat*np.pi/180.)
    ### deal with NaNs
    ind=np.isnan(xx)==False
    weight_mean=np.sum(xx[ind]*latrad[ind])/np.sum(latrad[ind])
    return weight_mean
