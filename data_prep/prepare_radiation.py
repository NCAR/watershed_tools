#!/usr/bin/env python
# coding: utf-8

"""
Prepare radiation input for discretization

Code to calculate daily radiation. Copy this code to the command folder for use later.
This calculation code is based on the reference:Allen, R.G., Trezza, R. and Tasumi, M., 2006.
Analytical integrated functions for daily solar radiation on slopes. Agricultural and Forest Meteorology, 139(1-2), pp.55-73.

Calculate average daily radiation over the snow-melting period. This step includes:
1. calculate daily radiation based on dem, slope, aspect and the day of year (DOY). One output per DOY.
2. calculate the period mean daily radiation.
3. check the correctness of radiation.<br>

Note that it is better to run this code on high-performance computers because the 1st step is time-and-memory consuming.



@authors: orig hongli liu, ncar, 2021
          debugged/revised by andy wood, ncar, 2021
          amended by dave casson, usask, 2022

"""

import os, shutil, sys
sys.path.append('../')
import functions.utils as ut
import functions.geospatial_analysis as ga
import functions.geospatial_plot as gp
import geopandas as gpd
import rasterio as rio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logging
logger = logging.getLogger(__name__)

def prepare_radiation(settings):

    settings['script_file'] = os.path.join(settings['main_path'], 'functions/calculate_radiation.py')

    prepare_radiation_dir(settings)
    create_radiation_job_submissions(settings)
    submit_radiation_job_submissions(settings)
    calculate_avg_daily_rad(settings)
    check_and_plot_of_avg_daily_rad(settings)

def prepare_radiation_dir(settings):

    settings['basin_gru_prj_shp'] = settings['basin_gru_shp'].split('.shp')[0] + '_prj.shp'

    if not os.path.exists(settings['tmpfile_path']):
        os.makedirs(settings['tmpfile_path'])

    # path to store daily radiation. One output per DOY.
    settings['srad_output_dir'] = os.path.join(settings['basin_data_path'], 'radiation_doy/')
    if not os.path.exists(settings['srad_output_dir']):
        os.makedirs(settings['srad_output_dir'])

    # path to store job submission files.
    settings['command_dir'] = os.path.join(settings['basin_data_path'], 'radiation_doy_commands/')
    if os.path.exists(settings['command_dir']):
        shutil.rmtree(settings['command_dir'])
    os.makedirs(settings['command_dir'])

    return

def read_shell_script_template(settings):

    shell_script_cmd = []
    try:
        shell_cmd_complete = settings['shell_script_template']
        with open(shell_cmd_complete) as f:
            lines = f.readlines()
        for line in lines:
            shell_script_cmd.append(line)
    except:
        logger.warning(f'No shell script commands provide, using local python installation. Add shell_script_template to configuration file to add commands')

    return shell_script_cmd

def create_radiation_job_submissions(settings):
    # create job submission files for radiation calculation
    shell_script_cmd = read_shell_script_template(settings)
    # first set radiation calculation period
    settings['ndays'] = int(settings['solrad_end_DOY']) - int(settings['solrad_start_DOY']) + 1  # number of days in radiation calculation period
    # calculate 1 radiation estimate per week in period
    for i in tqdm(np.arange(0, settings['ndays'] + 4, 7)):
        DOY = int(settings['solrad_start_DOY']) + i
        command_filename = os.path.join(settings['command_dir'], 'qsub_DOY' + str(DOY) + '.sh')
        if os.path.exists(command_filename):
            os.remove(command_filename)


        with open(command_filename, 'w') as f:
            for cmd in shell_script_cmd:
                f.write(cmd)
            f.write("%s %s %s %s %s %s %d %s\n" %
                    ('python',
                     settings['script_file'], settings['basin_dem_raster'], settings['basin_slope_raster'],
                     settings['basin_aspect_raster'],
                     str(DOY), int(settings['epsg']), settings['srad_output_dir']))

    logger.info(f'Shell script created {command_filename}')

    return

def submit_radiation_job_submissions(settings):
    """
    Submit jobs, one per day in range (or run consecutively)
    for large domains (more than a few square degrees, submission is likely faster; otherwise sequential may be fine)
    """
    for i in tqdm(np.arange(0, settings['ndays'] + 4, 7)):
        DOY = int(settings['solrad_start_DOY']) + i
        command_filename = os.path.join(settings['command_dir'], 'qsub_DOY' + str(DOY) + '.sh')

        if settings['run_local_or_hpc'] == 'hpc':
            os.system('qsub ' + command_filename) # run on cluster   (uncomment to run ... need to add check on existing files so that this isn't kicked off if the outputs exist already)
        else:
            os.system('chmod +x '+command_filename)
            os.system(command_filename)         # alternatively, run sequentially on a local machine
    logger.info(f'Shell script {command_filename} submitted')

def calculate_avg_daily_rad(settings):
    """Calculate average daily radiation"""
    # read daily radiation over the period and calculate the period mean
    ndays_eff = len(np.arange(0, settings['ndays'] + 4, 7))  # number of days that will be averaged at weekly frequency
    for i in np.arange(0, settings['ndays'] + 4, 7):
        DOY = int(settings['solrad_start_DOY']) + i
        settings['sw_DOY_raster'] = os.path.join(settings['srad_output_dir'], 'sw_DOY' + str(DOY) + '.tif')

        # read DOY radiation
        with rio.open(settings['sw_DOY_raster']) as ff:
            sw = ff.read(1)
            mask = ff.read_masks(1)
            sw_ma = np.ma.masked_array(sw, mask == 0)
            out_meta = ff.meta.copy()
            nodatavals = ff.nodatavals

        # create matching 2D array to sum up DOY radiation (creating the mean)
        if i == 0:
            sw_avg = sw_ma * 0
        # update value with prorated addition of each new day in period
        sw_avg = sw_avg + (sw_ma / ndays_eff)

    # save to raster
    sw_avg_value = sw_avg.filled(
        fill_value=nodatavals)  # return a copy of self, with masked values filled with a given value.
    sw_avg_ma = np.ma.masked_array(sw_avg_value, mask == 0)  # assign mask from sw_DOY
    sw_avg_ma = sw_avg_ma.astype(sw_ma.dtype)  # change data type to sw_ma's.
    with rio.open(settings['basin_radiation_raster'], 'w', **out_meta) as outf:
        outf.write(sw_avg_ma, 1)

    return
def check_and_plot_of_avg_daily_rad(settings):
    # 3a. Check CDF of average daily radiation ####
    try:
        num_bins = settings['num_bins']
    except:
        logging.info('Number of radiation bins not specified in config file, using default value')
        num_bins = 100

    # raw sw and its area-based cdf
    with rio.open(settings['basin_radiation_raster']) as ff:
        sw = ff.read(1)
        mask = ff.read_masks(1)
    origin_counts, origin_bin_edges = np.histogram(sw[mask != 0], bins=num_bins)

    cum_counts = np.cumsum(origin_counts)
    total_count = cum_counts[-1]
    origin_cdf = cum_counts / float(total_count)

    # Plot comparatives cdf
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(origin_bin_edges[1:], origin_cdf, '-k', label='Average daily radiation')
    plt.xlabel("Radiation (W m$^{-2}$)")
    plt.ylabel("CDF")
    plt.legend(loc='best')
    plt.show()

    # #### 3b. Check distribution of average daily radiation ###
    plt.clf()

    plt.figure()
    f, ax = plt.subplots(1, 2, figsize=(15, 15))

    asp_ma = ga.read_raster(settings['basin_aspect_raster'])
    gp.plot_locatable_axes(asp_ma, ax[0])
    ax[0].set_title('Aspect')

    sw_ma = ga.read_raster(settings['basin_radiation_raster'])
    gp.plot_locatable_axes(sw_ma, ax[1])
    ax[1].set_title('Radiation')

    plt.tight_layout()
    plt.show()

    return

if __name__ == '__main__':

    control_file = '/Users/drc858/GitHub/watershed_tools/test_cases/tuolumne/control_tuolumne.txt'
    settings = ut.read_complete_control_file(control_file, logging=True)

    prepare_radiation(settings)











