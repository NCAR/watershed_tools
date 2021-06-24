#!/usr/bin/env python
# coding: utf-8

"""
@authors: hongli, 2020; rewritten AW Wood, 2021
"""

import os

# Function to extract a given setting from the configuration file
def read_from_control(control_file, setting):
    
    # Open and read control file
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

# Specify the default filename (including directory path)
def set_filename(control_file, setting):
    fileName        = read_from_control(control_file, setting)
    basin_data_path = read_from_control(control_file, 'basin_data_path')
    gis_path        = basin_data_path + 'gis/'
    
    if setting == 'basin_hucId_txt':
        fileName = basin_data_path + fileName            
    elif setting == 'basin_gruNo_hucId_txt':
        fileName = basin_data_path + 'gruNo_hucId.txt'            
    elif setting == 'basin_gru_raster':
        fileName = gis_path + 'gru.tif'            
            
    elif setting == 'basin_dem_raster':
        fileName = gis_path + 'dem.tif'            
    elif setting == 'basin_slope_raster':
        fileName = gis_path + 'slope.tif'            
    elif setting == 'basin_aspect_raster':
        fileName = gis_path + 'aspect.tif'            
    elif setting == 'basin_soiltype_raster':
        fileName = gis_path + 'soiltype.tif'    
    elif setting == 'basin_radiation_raster':
        fileName = gis_path + 'radiation.tif'            
    elif setting == 'basin_flowlines_shp':
        fileName = gis_path + 'flowlines.shp'            
           
    elif setting == 'basin_landcover_raster':
        fileName = gis_path + 'landcover.tif'            
    elif setting == 'basin_landcover_resample_raster':
        fileName = gis_path + 'landcover_resample.tif'            
    elif setting == 'basin_landcover_class_raster':
        fileName = gis_path + 'landcover_class.tif'     
    elif setting == 'basin_canopy_class_raster':
        fileName = gis_path + 'canopy_class.tif'            
    elif setting == 'refraster':
        # if a different raster is not given, default to the basin DEM
        if(fileName == 'default'):
            fileName = gis_path + 'dem.tif'
           
    return fileName
