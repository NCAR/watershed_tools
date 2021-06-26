#!/usr/bin/env python
# coding: utf-8

"""
Created on Fri Oct 16 09:35:11 2020

@author: hongli
"""

import os

# Function to extract a given setting from the configuration file
def read_from_control(control_file, setting):
    
    # Open 'control_active.txt' and ...
    with open(control_file) as contents:
        for line in contents:
            read_setting = line.split('|',1)[0].strip()
            
            # find the line with the requested setting
            if (read_setting == setting) and (not line.startswith('#')):
                break
    # Extract the setting's value
    substring = line.split('|',1)[1]      # Remove the setting's name (split into 2 based on '|', keep only 2nd part)
    substring = substring.split('#',1)[0] # Remove comments, does nothing if no '#' is found
    substring = substring.strip()         # Remove leading and trailing whitespace, tabs, newlines                
       
    # Return this value    
    return substring

# Specify the default path if required
def specify_file_path(control_file, setting):
    path = read_from_control(control_file, setting)
    
    if path == 'default':
        
        if setting == 'domain_gru_shp':
            file_name = 'gru.shp'
        elif setting == 'domain_hucid_txt':
            file_name = 'huc12Ids.txt'
        elif setting == 'domain_gru_corr_txt':
            file_name = 'gruNo_HUC12_corr.txt'
        elif setting == 'domain_gru_prj_shp':
            file_name = 'gru_prj.shp'
        elif setting == 'domain_gru_raster':
            file_name = 'gru.tif'
            
        elif setting == 'domain_dem_raster':
            file_name = 'dem.tif'
        elif setting == 'domain_slope_raster':
            file_name = 'slope.tif'
        elif setting == 'domain_aspect_raster':
            file_name = 'aspect.tif'
        elif setting == 'domain_soil_raster':
            file_name = 'soil.tif'
        elif setting == 'domain_radiation_raster':
            file_name = 'radiation.tif'
        elif setting == 'domain_stream_shp':
            file_name = 'stream.shp'
           
        elif setting == 'domain_landcover_raster':
            file_name = 'landcover.tif'
        elif setting == 'domain_landcover_resample_raster':
            file_name = 'landcover_resample.tif'
        elif setting == 'domain_landcover_class_raster':
            file_name = 'landcover_class.tif'            

        root_path = read_from_control(control_file, 'root_path')
        domain_name = read_from_control(control_file, 'domain_name')
        geoPath = os.path.join(root_path, domain_name, file_name)
    else: 
        geoPath = path 
    return geoPath

def specify_refraster_path(control_file):
    path = read_from_control(control_file, 'refraster')
    
    if path == 'default':
        geoPath = specify_file_path(control_file, 'domain_dem_raster')
    else:
        geoPath = path
    return geoPath
        