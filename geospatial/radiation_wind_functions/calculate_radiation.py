#!/usr/bin/env python
# coding: utf-8

import argparse, os
import numpy as np
from pyproj import Proj
import rasterio as rio

def process_command_line():
    '''Parse the commandline'''
    parser = argparse.ArgumentParser(description='Script to calcualte 24-hr radiation.')
    parser.add_argument('dem_raster', help='path of DEM raster.')
    parser.add_argument('slope_raster',help='path of slope raster.')
    parser.add_argument('aspect_raster',help='path of aspect raster.')
    parser.add_argument('DOY',help='number of days of a year.')
    parser.add_argument('proj4',help='string that PROJ uses to describe coordinate systems and transformations.')
    parser.add_argument('opath',help='output path.')
    args = parser.parse_args()
    return(args)

def read_raster(file):   
    with rio.open(file) as ff:
        data  = ff.read(1)
        mask = ff.read_masks(1)
    data_ma = np.ma.masked_array(data, mask==0)
    return data_ma

def calcualte_Sw(latitude,elev,slope,aspect,DOY):
    # Reference: Allen, R.G., Trezza, R. and Tasumi, M., 2006. 
    # Analytical integrated functions for daily solar radiation on slopes. Agricultural and Forest Meteorology, 139(1-2), pp.55-73.

    RH = 0.3
    AirTemp = 0
    Kt = 1           # empirical turbidity coefficient. (0,1]. 1 is for clear area.
    albedo = 0.15

    # solar geometry variables
    # Sw_rad = np.empty_like(elev)
    declin = (23.45*np.pi/180.0)*np.sin(2*np.pi*((284+DOY)/365.0)) #Earth's declination [rad]
    d_squared = 1/(1+0.033*np.cos(DOY*2*np.pi/365))     #Eqn 2, earth sun distance
    a = np.sin(declin)*np.cos(latitude)*np.sin(slope)*np.cos(aspect)-np.sin(declin)*np.sin(latitude)*np.cos(slope) #Eqn 11a, constant
    b = np.cos(declin)*np.cos(latitude)*np.cos(slope)+np.cos(declin)*np.sin(latitude)*np.sin(slope)*np.cos(aspect) #Eqn 11b, constant
    c = np.cos(declin)*np.sin(slope)*np.sin(aspect)   #Eqn 11c, constant
    S = 1367                                            #Soloar constant[W/m2]

    #INTEGRATION LIMITS (SUNRISE/SUNSET)
    print('INTEGRATION LIMITS')
    #Calculate Integration Limits for extraterrestrial radiation (KET)
    #Step A.1. - Calculate Sunrise/Sunset for Horizontal Slope - not considering double sunrise situations
    ws2 = np.arccos(-1*np.tan(declin)*np.tan(latitude))                     #Eqn 8, Sunset [rad]
    ws1 = ws2*(-1)                                                          #Sunrise [rad]
    #Check for sun never setting (N. hemisphere)                  
    ws2[(latitude>0)&((declin+latitude) > np.pi/2)] = np.pi
    ws1[(latitude>0)&((declin+latitude) > np.pi/2)] = -1*np.pi
    #Check for sun never rising (N.hemisphere)
    ws2[(latitude>0)&(np.absolute(declin-latitude) > np.pi/2)] = 0
    ws1[(latitude>0)&(np.absolute(declin-latitude) > np.pi/2)] = 0
    #Check fo sun never setting (S. hemisphere)
    ws2[(latitude<0)&((declin+latitude) < -1*np.pi/2)] = np.pi
    ws1[(latitude<0)&((declin+latitude) < -1*np.pi/2)] = -1*np.pi
    #Check for sun never rising (S. hemisphere)
    ws2[(latitude<0)&(np.absolute(declin-latitude) < -1*np.pi/2)] = np.pi
    ws1[(latitude<0)&(np.absolute(declin-latitude) < -1*np.pi/2)] = -1*np.pi

    cos_ws1 = -a+b*np.cos(ws1)+c*np.sin(ws1)                                #Eqn 14
    cos_ws2 = -a+b*np.cos(ws2)+c*np.sin(ws2)                                #Eqn 14

    #Step A.2. Find beginning integration limit
    quad = b**2+c**2-a**2                                                   #Quadratic function for eqn 13
    quad[quad<0] = 0.0001                                                   #Limit quadratic function to greater than 0 as per step A.4.i.
    sin_w1 = (a*c-b*np.sqrt(quad))/(b**2+c**2)                              #Eqn 13a
    sin_w1[sin_w1>1] = 1                                                    #Limit sin_w1 to between 1 and -1 as per step A.4.iii.
    sin_w1[sin_w1<1*-1] = -1                                                #Limit sin_w1 to between 1 and -1 as per step A.4.iii.
    w1 = np.arcsin(sin_w1)
    cos_w1 = -a+b*np.cos(w1)+c*np.sin(w1)                                   #Eqn 14
    w1x = -1*np.pi-w1
    cos_w1x = -a+b*np.cos(w1x)+c*np.sin(w1x)                                #Eqn 14

    w1_24 = w1
    w1_24[(cos_ws1 <=cos_w1)&(cos_w1<0.001)] = w1[(cos_ws1 <=cos_w1)&(cos_w1<0.001)]      # Apply conditions in A.2.iv and A.2.v
    w1_24[cos_w1x > 0.001] = ws1[cos_w1x > 0.001]
    w1_24[(cos_w1x <= 0.001) & (w1x <= ws1)] = ws1[(cos_w1x <= 0.001) & (w1x <= ws1)]
    w1_24[(cos_w1x<=0.001) & (w1x >ws1)] = w1x[(cos_w1x<=0.001) & (w1x >ws1)]

    w1_24 = np.maximum(w1_24, ws1)                                          #Sunrise [rad], Step A.2.vi #Element-wise maximum of array elements.

    #Step A.3. Find end integration limit
    sin_w2 = (a*c+b*np.sqrt(quad))/(b**2+c**2)                              #Eqn 13b
    sin_w2[sin_w2>1] = 1                                                    #Limit sin_w2 to between 1 and -1 as per step A.4.iii.
    sin_w2[sin_w2<1*-1] = -1                                                #Limit sin_w2 to between 1 and -1 as per step A.4.iii.
    w2 = np.arcsin(sin_w2)
    cos_w2 = -a+b*np.cos(w2)+c*np.sin(w2)                                   #Eqn 14
    w2x = np.pi-w2
    cos_w2x = -a+b*np.cos(w2x)+c*np.sin(w2x)                                #Eqn 14

    w2_24 = w2
    w2_24[(cos_ws2 <= cos_w2) & (cos_w2 <0.001)] = w2[(cos_ws2 <= cos_w2) & (cos_w2 <0.001)]                                                              #Apply conditions in A.3.iv and A.3.v
    w2_24[cos_w2x > 0.001] = ws2[cos_w2x > 0.001]
    w2_24[(cos_w2x <= 0.001) & (w2x >= ws2)] = ws2[(cos_w2x <= 0.001) & (w2x >= ws2)]
    w2_24[(cos_w2x<=0.001) & (w2x<ws2)]= w2x[(cos_w2x<=0.001) & (w2x<ws2)]

    w2_24 = np.minimum(w2_24, ws2)                                         #Sunset [rad], Step A.3.vi      

    #If sunrise is before sunset, then set equal (slope is always shaded) as per A.4.ii.
    w1_24[w2_24<w1_24] = w2_24[w2_24<w1_24]     

    #----------------------------------------------------------------------------------------------------------------------------------
    #CALCULATE EXTRATERRESTRIAL RADIATION
    #----------------------------------------------------------------------------------------------------------------------------------
    print('CALCULATE EXTRATERRESTRIAL RADIATION')
    #Find cos(theta)
    #Only one sunrise/sunset, integrated between w1_24 and w2_24
    cos_theta = np.sin(declin) * np.sin(latitude) * np.cos(slope) * (w2_24 - w1_24) -\
    np.sin(declin) * np.cos(latitude) * np.sin(slope) * np.cos(aspect) * (w2_24 - w1_24) +\
    np.cos(declin) * np.cos(latitude) * np.cos(slope) * (np.sin(w2_24) - np.sin(w1_24)) +\
    np.cos(declin) * np.sin(latitude) * np.sin(slope) * np.cos(aspect) * (np.sin(w2_24) - np.sin(w1_24)) -\
    np.cos(declin) * np.sin(slope) * np.sin(aspect) * (np.cos(w2_24) - np.cos(w1_24))                      #Eqn 5

    #Calculate 24H Extraterrestrial Radiation
    Ra_24 = S * cos_theta / d_squared                                #24 Hour extraterrestrial radiation [W/m2]. Eqn 6.
    Ra_24 = np.maximum(0, Ra_24)                                     #Limit to positive values

    #-------------------------------------------------------------------------------------------------------------------------------------
    #CLEAR SKY SOLAR RADIATION
    #-------------------------------------------------------------------------------------------------------------------------------------
    print('CLEAR SKY SOLAR RADIATION')
    #Calculate various parameters for transmissivity
    e = 0.6108 * np.exp(17.27 * AirTemp / (AirTemp + 237.3)) * RH     #Vapor Pressure as per ASCE_EWRI (2005) [kPa]. Eqn 7.
    P = 101.3 * ((293 - 0.0065 * elev) / 293)**5.36                   #Atmospheric Pressure as per ASCE-EWRI (2005) [kPa]. Eqn 34.
    W = 0.14 * e * P + 2.1                                            #Eqn 18
    g = np.sin(declin) * np.sin(latitude)                             #Eqn 27
    h = np.cos(declin) * np.cos(latitude)                             #Eqn 28

    #Clear Sky Solar Radiation for Horizontal Surface
    sin_Beta_24H = elev.copy()
    K_Do_24H = elev.copy()
    K_Bo_24H = elev.copy()
    
    #Eqn 26, Beta is angle of sun above horizon [rad]
    cond_flat = (slope==0)
    sin_Beta_24H[cond_flat] = (2 * (g[cond_flat]**2) * ws2[cond_flat] + \
        4 * g[cond_flat] * h[cond_flat] * np.sin(ws2[cond_flat]) + \
        (h[cond_flat]**2) * (ws2[cond_flat] + \
        0.5 * np.sin(2 * ws2[cond_flat]))) / \
        (2 * (g[cond_flat] * ws2[cond_flat] + h[cond_flat] * np.sin(ws2[cond_flat])))
   
    #Eqn 17, Clearness index for direct beam radiation [-]
    K_Bo_24H[cond_flat] = 0.98 * np.exp((-0.00146 * P[cond_flat] / (Kt * sin_Beta_24H[cond_flat])) \
                                         - 0.075 * (W[cond_flat] / sin_Beta_24H[cond_flat])**0.4)  

    #Eqn 19, Index for diffuse beam radiation [-]
    cond1 = (K_Bo_24H >= 0.15) & (~np.isnan(K_Bo_24H))
    K_Do_24H[cond1] = 0.35 - K_Bo_24H[cond1]
    
    cond2 = (K_Bo_24H > 0.065) & (K_Bo_24H < 0.15) & (~np.isnan(K_Bo_24H))
    K_Do_24H[cond2] = 0.18 + 0.82*K_Bo_24H[cond2]
    
    cond3 = (K_Bo_24H <= 0.065) & (~np.isnan(K_Bo_24H))
    K_Do_24H[cond3] = 0.10 + 2.08*K_Bo_24H[cond3]
    del cond1, cond2, cond3

    R_So_24H = (K_Bo_24H + K_Do_24H) * Ra_24  #Eqn 23, 24-H clear sky radiation [W/m2]      

    #Clear Sky Solar Radiation for Inclined Surface
    #DIRECT RADIATION (R_Bo)
    f1 = np.sin(w2_24) - np.sin(w1_24)
    f2 = np.cos(w2_24) - np.cos(w1_24)
    f3 = w2_24 - w1_24
    f4 = np.sin(2 * w2_24) - np.sin(2 * w1_24)
    f5 = np.sin(w2_24)**2 - np.sin(w1_24)**2

    #Eqn 22, Beta is angle of sun above horizon [rad]
    sin_Beta = ((b * g - a * h) * f1 - c * g * f2 + (0.5 * b * h - a * g) * f3 + 0.25 * b * h * f4 + 0.5 * c * h * f5)/ (b * f1 - c * f2 - a * f3) 
    sin_Beta[(f1 <= 0) & (f2 <= 0) & (f3 <= 0) & (f4 <= 0) & (f5 <= 0)] = 0.001
    sin_Beta = np.maximum(0.0001, sin_Beta)       

    K_Bo = 0.98 * np.exp((-0.00146 * P / (Kt * sin_Beta)) - 0.075 * (W / sin_Beta)**0.4)                                                 #Eqn 17, Direct beam clearness index [-]
    R_Bo_24 = Ra_24 * K_Bo                                                                                                             #Eqn 30, Direct component of radiation [W/m2]
    R_Bo_24 = np.maximum(R_Bo_24, 0)                                                                                                         #Limit to only positive values
    R_Bo_24[cond_flat] = np.nan      

    #DIFFUSE RADIATION (R_Do)
    #Horizontal Components
    Ra_H_24 = (S / d_squared) * (np.sin(declin) * np.sin(latitude) * (ws2 - ws1) + np.cos(declin) * np.cos(latitude) * (np.sin(ws2) - np.sin(ws1)))      #Eqn 5,ET radiation for horizontal surface [W/m2]
    sin_Beta_H = (2 * (g**2) * ws2 + 4 * g * h * np.sin(ws2) + (h**2) * (ws2 + 0.5 * np.sin(2 * ws2))) / (2 * (g * ws2 + h * np.sin(ws2)))    #Eqn 26, Angle of sun above horizon for horizontal surface [W/m2]
    K_Bo_H = 0.98 * np.exp((-0.00146 * P / (Kt * sin_Beta_H)) - 0.075 * (W / sin_Beta_H)**0.4)                                           #Eqn 17, Direct radiation index [-]

    K_Do_H = 0.35 - 0.36 * K_Bo_H   
    cond1 = (K_Bo_H > 0.065) & (K_Bo_H < 0.15)                                                                                                   #Eqn 19, Diffuse radiation index [-]
    K_Do_H[cond1] = 0.18 + 0.82 * K_Bo_H[cond1] 
    cond2 = (K_Bo_H <= 0.065)
    K_Do_H[cond2] = 0.1 * 2.08 * K_Bo_H[cond2]
    del cond1,cond2

    R_Do_H = K_Do_H * Ra_H_24                                                                                                          #Eqn 25, Diffuse radiation for horizontal surface [W/m2]

    #Fractions
    fi = 0.75 + 0.25 * np.cos(slope) - 0.5 * slope /np.pi                                                                          #Eqn 32, Sky view factor for isotropic conditions [-]
    fb = (K_Bo / K_Bo_H) * (Ra_24 / Ra_H_24)                                                                                           #Eqn 34, Ratio of expected direct beam radiation on slope to radiation on horizontal
    fia = (1 - K_Bo_H) * (1 + ((K_Bo_H / (K_Bo_H + K_Do_H))**0.5) * np.sin(slope / 2)**3) * fi + fb * K_Bo_H                       #Eqn 33, Sky view factor for anisotropic conditions [-]

    #Diffuse Radiation
    R_Do_24 = fia * R_Do_H                                                                                                             #Eqn 31, Diffuse radiation [W/m2]
    R_Do_24 = np.maximum(R_Do_24, 0)                                                                                                         #Limit to positive values

    #REFLECTED RADATION (R_Ro)
    cond_nan = (np.isnan(R_So_24H))
    R_So_24H[cond_nan] = (K_Bo_H[cond_nan] + K_Do_H[cond_nan]) * Ra_H_24[cond_nan]     #Eqn 23, Clear sky solar radiation for horizontal surface [W/m2]
    del cond_nan

    R_Ro_24 = R_So_24H * albedo * (1 - fi)                                                                                             #Eqn 36, Reflected radiation [W/m2]
    R_Ro_24 = np.maximum(R_Ro_24, 0) 
    R_Ro_24[cond_flat] = np.nan

    #TOTAL RADIATION
    R_So_24 = R_Bo_24 + R_Do_24 + R_Ro_24
    R_So_24[cond_flat] = R_So_24H[cond_flat] 
    Sw_DOY = R_So_24/(2 * np.pi)

    return Sw_DOY

# main
if __name__ == '__main__':

    # process command line
    args = process_command_line()
    dem_raster=args.dem_raster    #'dem.tif'
    slp_raster=args.slope_raster  #'slope.tif'
    asp_raster=args.aspect_raster #'aspect.tif'
    DOY=int(args.DOY)
    proj4 = args.proj4
    
    opath=args.opath 
    if not os.path.exists(opath):
        os.makedirs(opath)
    ofile_raster='sw_DOY'+str(DOY)+'.tif'
    
    Sw_nodata=-9999
    
    #====================================================
    # read input rasters
    print('read inputs')
    slp_ma = read_raster(slp_raster)
    asp_ma = read_raster(asp_raster)
    elev_ma = read_raster(dem_raster)
    
    #====================================================
    # derive latitude 
    print('derive latitude')
    with rio.open(dem_raster) as ff:
        mask = ff.read_masks(1)
        gt = ff.transform
        
        nodatavals = ff.nodatavals
        dst_crs = ff.crs
        out_meta = ff.meta.copy()
        
    x_size = gt[0]
    y_size = -gt[4]
    upper_left_x = gt[2]
    upper_left_y = gt[5]
    
    # add half the cell size to centre the point
    (ny,nx) = np.shape(elev_ma)    
    x_index=np.arange(nx)
    y_index=np.arange(ny)
    x_coords = x_index * x_size + upper_left_x + (x_size / 2.0) 
    y_coords = y_index * y_size + upper_left_y + (y_size / 2.0) 
    
    # turn 1d coordinates to 2d
    x_coords_2d = np.repeat(np.reshape(x_coords,(1,nx)), ny, axis=0)
    y_coords_2d = np.repeat(np.reshape(y_coords,(ny,1)), nx, axis=1)
    
    # converts from longitude, latitude to native map projection x,y coordinates
    # p = Proj(proj='utm',zone=13,ellps='GRS80',datum='NAD83', preserve_units=False)
    # p = Proj("+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m no_defs")
    p = Proj(proj4)
    lon, lat = p(x_coords_2d, y_coords_2d, inverse=True) # inverse transform
    lat_ma = np.ma.masked_array(lat, mask==0)
    
    #====================================================
    # calculate Sw
    print('calculate Sw')
    
    lat_ma = np.deg2rad(lat_ma)
    slp_ma = np.deg2rad(slp_ma)
    asp_ma = np.deg2rad(asp_ma)-np.pi # Surface aspect redefined so that 0 degrees is south and pi/2 is west [rad]
    
    Sw_DOY = calcualte_Sw(lat_ma,elev_ma,slp_ma,asp_ma,DOY)
    # Sw_DOY is a masked array because all inputs are masked arrays.
    
    #====================================================
    # save Sw as raster
    print('save Sw raster')
    Sw_DOY_value = Sw_DOY.filled(fill_value=nodatavals)  # return a copy of self, with masked values filled with a given value.
    Sw_DOY_ma = np.ma.masked_array(Sw_DOY_value,mask==0) # assign mask from 
    Sw_DOY_ma = Sw_DOY_ma.astype(elev_ma.dtype)          # change data type to elev_ma's.
    with rio.open(os.path.join(opath,ofile_raster), 'w', **out_meta) as outf:
        outf.write(Sw_DOY_ma, 1) 
    
    print('Done')
