from __future__ import absolute_import
from __future__ import print_function
from numpy import *
from six.moves import range
import numpy.ma as ma

pp_path_cisl='/glade/collections/cmip/'
# ON my Scratch: 
# ON JASMIN: /badc/cmip6/data/CMIP6/CMIP/CNRM-CERFACS/CNRM-CM6-1/historical/r1i1p1f2/Amon/clw/gr/latest/
# ON CISL: /glade/collections/cmip/CMIP6/CMIP/CNRM-CERFACS/CNRM-CM6-1/historical/r1i1p1f2/Amon/clw/gr/v20180917/clw/
# ON CISL:NCAR MODEL: /glade/collections/cdg/data/CMIP6/CMIP/NCAR/CESM2/piControl/r1i1p1f1/Amon/evspsbl/gn/v20190320/


def read_var_mod(modn='CNRM-CM6-1', consort='CNRM-CERFACS', varnm='clwvi', cmip='cmip6', exper='historical', ensmem='r1i1p1f2', typevar='Amon', gg='gr', read_p=False, time1=[1, 1, 15], time2=[8000, 12, 31]):
    ### ------------------
    # Reads in data from named GCM for specified time range
    # For 3D data read_p=True.
    # Will need mods for different sub experiments.
    ### ------------------
    

    if cmip == 'cmip6':
        MIP = 'CMIP'
        # if 'ssp' in exper:
        #     MIP = 'ScenarioMIP'
        # if exper=='amip-p4K':
        #     MIP = 'CFMIP'
        pth = pp_path_cisl+'CMIP6/'+MIP+'/'+consort+'/'+modn + \
            '/'+ exper+'/'+ensmem +'/'+ typevar+'/'+varnm+'/'+gg+ '/'
        #if consort == 'NACR':
        #    pth = '/glade/collections/cdg/data/'+'/CMIP6/'+'CMIP'+'/'+consort+'/'+modn + \
        #        '/'+exper+'/'+ensmem+'/'+typevar+'/'+varnm+'/'+gg+'/latest/'
    if cmip == 'cmip5':
        pth = '/glade/collections/cdg/data/'+cmip+'/output1/'+consort+'/'+modn + \
            '/'+exper+'/mon/atmos/'+ typevar +'/'+ensmem+'/'
        if typevar == 'OImon':
            pth = pp_path_cisl+cmip+'/data/cmip5/'+output+'/'+consort+'/'+modn + \
            '/'+exper+'/mon/seaIce/'+typevar+'/'+ensmem+'/'
    # data_source flag
    ds_flag = 2
    if cmip == 'cmip6':
        try:
            data, P, lat, lon, time, filled_value = read_hs('/glade/scratch/chuyan/CMIP6data/', varnm, 
                    read_p=read_p, modnm=modn, exper=exper, ensmem=ensmem, typevar=typevar, time1=time1, time2=time2, ds_flag = ds_flag)
        except UnboundLocalError:
            ds_flag = 1
            print('TRYING CISL FILES')
            data, P, lat, lon, time, filled_value = read_hs(pth, varnm, read_p=read_p, modnm=modn, time1=time1, time2=time2, ds_flag = ds_flag)

        except IndexError:
            ds_flag = 1
            print('LOCAL FILES Indexing Error')
            data, P, lat, lon, time, filled_value = read_hs(pth, varnm, read_p=read_p, modnm=modn, time1=time1, time2=time2, ds_flag = ds_flag)
            
    ds_flag = 2
    if cmip == 'cmip5':
        try:
            data, P, lat, lon, time, filled_value = read_hs('/glade/scratch/chuyan/CMIP5data/', varnm, 
                    read_p=read_p, modnm=modn, exper=exper, ensmem=ensmem, typevar=typevar, time1=time1, time2=time2, ds_flag = ds_flag)
        except UnboundLocalError:
            ds_flag = 1
            print('TRYING CISL FILES')
            data, P, lat, lon, time, filled_value = read_hs(pth, varnm, read_p=read_p, modnm=modn, time1=time1, time2=time2, ds_flag = ds_flag)

        except IndexError:
            ds_flag = 1
            print('LOCAL FILES Indexing Error')
            data, P, lat, lon, time, filled_value = read_hs(pth, varnm, read_p=read_p, modnm=modn, time1=time1, time2=time2, ds_flag = ds_flag)
    if read_p:
        if asarray(P).ndim != 1:
            P = concatenate(P, axis=0)  # P is duplicated here. Modified on Aug 28th

    dataOUT = ma.concatenate(data, axis=0)
    lon2 = lon[:]*1.
    lon2[lon2 > 180] = lon2[lon2 > 180]-360.
    ind = argsort(lon2)
    if read_p == False:
        dataOUT = dataOUT[:, :, ind]
    else:
        dataOUT = dataOUT[:, :, :, ind]
    lon2 = lon2[ind]
    timeo = concatenate(time, axis=0)
    # Processing 'filled_value':
    print('type of dataOUT in read_hs_file', type(dataOUT))
    dataOUT2 = dataOUT * 1.
    dataOUT2[dataOUT2 == filled_value] = nan
    dataOUT2, time = get_unique_time(dataOUT2, timeo)
    print(dataOUT2.shape)                                                       # 5
    return dataOUT2.filled(fill_value=nan), P, lat[:].filled(fill_value=nan), lon2.filled(fill_value=nan), time  # concatenate(time,axis=0)
# concatenate(P,axis=0),lat,lon


def get_unique_time(data, time):
    tf = time[:, 0]+time[:, 1]/100
    TF, ind = unique(tf, return_index=True)
    return data[ind], time[ind]



# ex:time1,time2, atual value above in Parameters of 'read_var_mod':
def read_hs(wd, varnm, read_p=False, modnm='', exper='', ensmem='', typevar='', time1=[2000, 1, 15], time2=[2005, 12, 31], ds_flag=0):
    import glob

    if ds_flag == 2:
        folder = wd
        print(modnm)
        fn = glob.glob(folder+ '*' +varnm+'_*'+typevar+'*' +modnm+'_'+exper+'*'+ensmem+'*nc*')
        print('1')
    
    elif ds_flag == 1:
        folder=glob.glob(wd+'*/*/')
        print(modnm)                                                           # 1
        if (modnm== 'MIROC6') or modnm == ('SAM0-UNICON'):
            fn = glob.glob(folder[0]+'/*'+varnm+'_*'+typevar+'*' +modnm+'_'+exper+'*'+ensmem+'*nc*')
            print('3')
        elif (modnm =='CESM2'):
            folder=glob.glob(wd+ '*/')
            # print(folder)
            fn = glob.glob(folder[-1]+'/*'+varnm+'_*'+typevar+'*' +modnm+'_'+exper+'*'+ensmem+'*nc*')
            print('3')
        elif (modnm =='FGOALS-g3'):
            fn = glob.glob(folder[-1]+'/*'+varnm+'_*'+typevar+'*' +modnm+'_'+exper+'*'+ensmem+'*nc*')
            for i in range(len(fn)):
                if fn[i] == '/glade/collections/cmip/CMIP6/CMIP/CAS/FGOALS-g3/piControl/r1i1p1f1/Amon/ta/gn/v20190818/ta/ta_Amon_FGOALS-g3_piControl_r1i1p1f1_gn_055001-055912.nc':
                    fn[i] = '/glade/work/chuyan/Research/Cloud_CCFs_RMs/Plots_proposal/ta_Amon_FGOALS-g3_piControl_r1i1p1f1_gn_055001-055912.nc'        
            print('3')
        elif (modnm =='CCSM4'):
            folder=glob.glob(wd+ '*/')
            #..print(folder[-1])
            fn = glob.glob(folder[-1]+ varnm+'/*'+varnm+'*'+typevar+'*'+modnm+'_'+exper+'*'+ensmem+'*nc*')
        else:
            fn = glob.glob(folder[-1]+'/*'+varnm+'_*'+typevar+'*' +modnm+'_'+exper+'*'+ensmem+'*nc*')
            print('2')                                                         # 2

    print(folder)                                                              # 3
    print(fn)                 #..folder[0]+'/*'+varnm+'*'+typevar+'*'+modnm+'_'+exper+'*'+ensmem+'*nc*'   # 3'
    
    data = []
    P = []
    timeo = []
    # filter out the data file with times too larger
    print(" Variable", varnm, ' ', exper)                                      # 4
    for i in range(len(fn)):
        # print(fn[i])                                                         # 4'
        tt = read_hs_file(fn[i], varnm, read_p=read_p, time1=time1, time2=time2)
        if len(tt['data']) > 0:
            data.append(tt['data'])
            lat = tt['lat']
            lon = tt['lon']
            timeo.append(tt['time'])
            filled_value = tt['filled_value']
            
            if read_p == True:
                P = tt['P']
    return data, P, lat, lon, timeo, filled_value



def read_hs_file(fn, varnm, time1, time2, read_p=False):
    # fn='clw_Amon_CNRM-CM6-1_historical_r1i1p1f2_gr_195001-201412.nc'
    # varnm='clw'
    import netCDF4 as nc
    from datetime import datetime
    
    f = nc.Dataset(fn, 'r')
    latvar = 'lat'
    lonvar = 'lon'
    lat = f.variables[latvar]
    lon = f.variables[lonvar]
    tvar = 'time'
    tt = f.variables[tvar]
    filled_value_gcm = f.variables[varnm]._FillValue
    # print('filled_value_gcm', filled_value_gcm)
    timeout = zeros((len(tt[:]), 3))
    
    for i in range(timeout.shape[0]):
        tt1 = nc.num2date(tt[i], f.variables[tvar].units,
                          calendar=f.variables[tvar].calendar)
        timeout[i, :] = [tt1.year, tt1.month, tt1.day]
    # select the indices of numeric variable 'tt' which most close to the starting & ending time_stamp:
    ind1 = nc.date2index(
        datetime(time1[0], time1[1], time1[2]), tt, select='nearest')
    ind2 = nc.date2index(
        datetime(time2[0], time2[1], time2[2]), tt, select='nearest')
    ind = arange(ind1, ind2+1)
    data = []
    P = []
    
    if ind1 != ind2:
        data = f.variables[varnm][ind]
        P = None
        if read_p:
            if 'plev' in list(f.variables.keys()):
                P = f.variables['plev'][:]
            else:
                P = get_pressure_nc(f, ind)

    return {'data': data, 'P': P, 'lat': lat, 'lon': lon, 'time': timeout[ind, :], 'filled_value': filled_value_gcm}


def get_pressure_nc(f, ind):
    vv = list(f.variables.keys())
    print(vv)                                                                     # possible 4
    for i in range(len(vv)):
        if 'formula' in f.variables[vv[i]].ncattrs():
            formul_p = f.variables[vv[i]].formula
            print(formul_p)
            break
    # formul_p=f.variables['lev'].formula
    if (formul_p == 'p = ap + b*ps') | (formul_p == 'p(n,k,j,i) = ap(k) + b(k)*ps(n,j,i)'):
        ap = f.variables['ap']
        b = f.variables['b']
        ps = f.variables['ps'][ind]
        lev = f.variables['lev']
        P = zeros((len(ind), len(lev), ps.shape[1], ps.shape[2]))*NaN
        for i in range(len(lev)):
            P[:, i, :, :] = ap[i]+b[i]*ps
    if (formul_p == 'p = a*p0 + b*ps') | (formul_p == 'p(n,k,j,i) = a(k)*p0 + b(k)*ps(n,j,i)'):
        a = f.variables['a']
        b = f.variables['b']
        p0 = f.variables['p0']
        ps = f.variables['ps'][ind]
        lev = f.variables['lev']
        P = zeros((len(ind), len(lev), ps.shape[1], ps.shape[2]))*NaN
        for i in range(len(lev)):
            P[:, i, :, :] = p0*a[i]+b[i]*ps
    if formul_p == 'p = ptop + sigma*(ps - ptop)':
        ptop = f.variables['ptop']
        ps = f.variables['ps'][ind]
        lev = f.variables['lev'][:]
        P = zeros((len(ind), len(lev), ps.shape[1], ps.shape[2]))
        for i in range(len(lev)):
            P[:, i, :, :] = ptop+lev[i]*(ps-ptop)

    return P
