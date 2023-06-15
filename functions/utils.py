#!/usr/bin/env python
# coding: utf-8

"""
Utility functions supporting 'Watershed_Tools' code
@authors: orig hongli liu, 2020; rewritten AW Wood, 2021, amended by D. Casson, 2022
"""

import os, sys
import datetime
import logging
import logging.config
from pathlib import Path
import rasterio as rio
import json
import pprint
logger = logging.getLogger(__name__)

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


def set_filename(control_file, setting):
    """Specify the default filename (including directory path)"""

    filename        = read_from_control(control_file, setting)
    basin_data_path = read_from_control(control_file, 'basin_data_path')
    gis_path        = basin_data_path + 'gis/'

    # files in main basin_data directory
    if setting in ['basin_gruId_txt', 'basin_gruNo_gruId_txt']:
        filename = basin_data_path + filename

    # files in GIS directory
    elif setting in ['basin_gru_shp', 'basin_gru_raster', 'basin_dem_raster', 'basin_slope_raster',  
                     'basin_aspect_raster', 'basin_soiltype_raster', 'basin_radiation_raster',  
                     'basin_landcover_raster', 'basin_landcover_resample_raster', 'refraster', 
                     'basin_canopy_class_raster', 'canopy_class.tif', 'basin_flowlines_shp' ]:
        
        # special case for refraster
        if setting == 'refraster' and filename == 'default':
            filename = read_from_control(control_file, 'basin_dem_raster')

        # special case for basin_gru_shp
        if setting == 'basin_gru_shp' and filename == 'default':
            filename = 'gru.shp'     # create this file rather than read it

        filename = gis_path + filename
        # derived filenames


    else:
        logger.debug(f'Set_filename() was not successful for file {setting}')
        #print('STOP')     # warning in jupyter notebook; use exit() in normal script
            
    return filename

def set_epsg(settings):
    settings['dest_crs'] = rio.crs.CRS.from_epsg(settings['epsg'])
    return

def read_complete_control_file(control_file, logging=False):
    """Read the control file to a dictionary, with the option to log"""
    if logging == True:
       start_logging()

    settings = {}
    with open(control_file) as contents:
      for line in contents:
        line_strp = line.strip()
        if line_strp.startswith('#') or line_strp.startswith('\n') or (len(line_strp) == 0):
          continue
        key = line.split('|', 1)[0]
        val = line.split('|', 1)[1]
        val = val.split('#', 1)[0]
        #Add the custom file path setting defined in set_filename functions

        key = key.strip()
        val = val.strip()
        if key != 'basin_data_bath':
            val = set_filename(control_file,key)
        settings[key] = val

        if logging == True:
            logger.info(f'Setting: {key} = {val}')

    #set_epsg(settings)

    #Add additional definitions
    #From prepare_dem_grids
    settings['dem_prj_raster']        = settings['fulldom_dem_raster'].split('.tif')[0]+'_prj.tif'
    settings['basin_gru_prj_shp']     = settings['basin_gru_shp'].split('.shp')[0]+'_prj.shp'

    return settings

def start_logging(file_name='log_file'):

    '''Create a .\logs directory and write the logging file to it.
    It uses the logger_config.ini to set properties

    Parameters
    ----------
    file_name : str
        optional argument for name to be given to logfile

    Returns
    -------
    logger : logging object
        logger can be called with logger.debug("") etc to log to the console and output file

    '''

    #Set unique file output name
    now = datetime.datetime.now()
    log_file_name = f'{file_name}_{now.strftime("%Y-%m-%d_%H:%M:%S")}.log'

    #Create the output directory (.\logs\logfile.log)
    working_folder = os.path.dirname(os.path.abspath(__file__))
    Path(working_folder, './../logs').mkdir(parents=True, exist_ok=True)
    logfile = os.path.join(working_folder, './../logs', log_file_name)

    #Read settings from logger_config.ini
    logging.config.fileConfig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../control_files/logger_config.ini'),
                              defaults={'logfilename': logfile}, disable_existing_loggers=False)

    #Set additional options and log the creation of the file
    logger = logging.getLogger(__name__)
    logging.getLogger('matplotlib.font_manager').disabled = True

    logger.info(f'Log File Generated: {logfile}')

    return logger

def read_control_json(json_file):
    """
    Reads in json file to dictionary, also defining paths.

    Parameters
    ----------
    json file: file path
        complete path to json file

    Returns
    -------
    settings : dict
        dictionary of settings
    """
    f = open(control_json_file)
    settings = json.load(f)

    set_paths(settings,'fulldom_data', file_expected=True)

    return settings

def set_paths(settings, path_type, file_expected = False):
    """
    Update all paths in settings dictionary

    Parameters
    ----------
    settings: dict
        dictionary of run settings read from json
    path_type: str
        path type to be updated, example "basin_data"
    file_expected: bool
        check if the files are expected, such as needed input data files.

    Returns
    -------
    settings : dict
        update settings dictionary

    """

    full_path = path_type +'_path'

    for key in settings[path_type]:

        if settings[path_type][key]['path'] == full_path:
           settings[path_type][key]['loc'] = settings['paths'][full_path] + settings[path_type][key]['name']

        else:
            settings[path_type][key]['loc'] = settings[path_type][key]['path'] + settings[path_type][key]['name']

        if file_expected and not os.path.exists(settings[path_type][key]['loc']):
            logger.error(f"Expected file {settings[path_type][key]['loc']} not found. Check path and file location")

    return settings

if __name__ == '__main__':

    control_json_file = '/Users/drc858/GitHub/watershed_tools/test_cases/taylorpark/control.taylorpark.json'

    settings = read_control_json(control_json_file)
    pprint.pprint(settings)
