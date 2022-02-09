#!/usr/bin/env python
# coding: utf-8
"""
DEM preperation codes supporting scripts in 'watershed_tools' repository
  for discretizing watersheds based on elevation, vegetation type, and solar radiation

Prepare basin DEM inputs for discretization process

This script requires that the target basin GRU and flowline shapefiles exist
If they do not, run 'prepare_basin_shapefiles' script first.

Script tasks (each executed only if needed):
1. project larger full-domain DEM raster into a specified projection system
2. extract DEM for the basin area<br>
3. rasterize basin GRU shapefile based on basin DEM raster
4. calculate slope and aspect for the basin area
5. visualize (plot) slope and aspect results

@authors: orig hongli liu, ncar, 2021
          debugged/revised by andy wood, ncar, 2021
          amended by dave casson, usask, 2022

"""

import os, sys 
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.warp import Resampling
import matplotlib.pyplot as plt 

sys.path.append('../')   # libraries from this repo
import functions.utils as ut
import functions.geospatial_analysis as ga
import functions.geospatial_plot as gp
import logging
logger = logging.getLogger(__name__)


def prepare_dem_grids(settings):
    # Check whether required input shapefiles exist -- if not, exit
    if not (os.path.exists(settings['basin_flowlines_shp']) and os.path.exists(settings['basin_gru_prj_shp'])):
        logging.error(f'Required input shapefiles do not exist: {basin_gru_prj_shp} and, {basin_flowlines_shp}')
        logging.error(f'Run prepare_basin_shapefiles script to create them')
        exit()  # warning in jupyter notebook; use exit() in normal script

    # Extract basin DEM
    if not os.path.exists(settings['basin_dem_raster']):
        if not os.path.exists(settings['dem_prj_raster']):
            # reproject large-domain DEM to equal-area projection if it doesn't already exist (can take time (minutes) if the DEM is large)
            ga.reproject_raster(settings['fulldom_dem_raster'], settings['dem_prj_raster'], settings['dest_crs'],
                                Resampling.bilinear)  # cannot use 'nearest' - results in striations
            logging.info(f're-projected full domain raster ( epsg = {settings["epsg"]}), {settings["dem_prj_raster"]}')

        ga.crop_raster(settings['dem_prj_raster'], settings['basin_gru_prj_shp'], settings['basin_dem_raster'])
        logging.info(f'Basin DEM raster: {settings["basin_dem_raster"]}')

    # Rasterize basin GRU shapefile based on domain DEM raster ####
    # The basin GRU raster will be used in HRU generation and for calculating zonal statistics
    if not (os.path.exists(settings['basin_gru_raster']) and os.path.exists(settings['basin_gruNo_gruId_txt'])):
        ga.rasterize_gru_vector(settings['basin_gru_prj_shp'], settings['gru_fieldname'], settings['gruId_fieldname'],
                                settings['gruNo_fieldname'], settings['gruNo_field_dtype'],
                                settings['refraster'], settings['basin_gru_raster'], settings['basin_gruNo_gruId_txt'])
        logging.info(f'Rasterized GRU shapefile:{settings["basin_gru_raster"]}')
        logging.info(f'Wrote gruNo to gruId file {settings["basin_gruNo_gruId_txt"]}')

    # Calculate domain slope and aspect, and classify aspect into 8 directions ####
    # use numpy module (takes a little while - can skip if these rasters exist)
    if not (os.path.exists(settings['basin_slope_raster']) and os.path.exists(settings['basin_aspect_raster'])):
        ga.calculate_slope_and_aspect(settings['basin_dem_raster'], settings['basin_slope_raster'],
                                      settings['basin_aspect_raster'])
        logging.info(
            f'Calculating slope and aspect from dem. Output to {settings["basin_slope_raster"]} and {settings["basin_aspect_raster"]}')

    # classify aspect into 8 classes (can skip if this exists) -- this can also be done in 'analysis' phase
    settings['aspect_class_raster'] = settings['basin_aspect_raster'].split('.tif')[0] + '_class.tif'
    if not os.path.exists(settings['aspect_class_raster']):
        ga.classify_aspect(settings['basin_aspect_raster'], 8, settings['aspect_class_raster'])

        logging.info(f'Classifying aspect raster. Output to {settings["aspect_class_raster"]}')
        # alternate method: use gdal module
        # from osgeo import gdal
        # gdal.UseExceptions()
        # gdal.DEMProcessing(basin_slope_raster, basin_dem_raster, 'slope', computeEdges=True)
        # gdal.DEMProcessing(basin_aspect_raster, basin_dem_raster, 'aspect', zeroForFlat=True)

    return

def plot_dem_slope_aspect(settings):
    """
    :param settings: dictionary containing all run settings
    :output plot of dem, slope and aspect
    :return None
    """
    plt.clf()
    plt.figure()
    f, ax = plt.subplots(1, 3, figsize=(20, 20))

    dem_ma = ga.read_raster(settings['basin_dem_raster'])
    gp.plot_locatable_axes(dem_ma, ax[0])
    ax[0].set_title('DEM')

    slp_ma = ga.read_raster(settings['basin_slope_raster'])
    gp.plot_locatable_axes(slp_ma, ax[1])
    ax[1].set_title('slope')

    asp_ma = ga.read_raster(settings['basin_aspect_raster'])
    gp.plot_locatable_axes(asp_ma, ax[2])
    ax[2].set_title('aspect')

    plt.tight_layout()
    output_fig_file = os.path.join(settings['basin_data_path'], 'plots/dem_slope_aspect.png')
    plt.savefig(output_fig_file)

    return

def prepare_dem_grids_plot(settings):
    """
    :param settings: dictionary containing all run settings
    :output plot of prepared dem plots with stream boundaries
    :return: None
    """
    # define legend dictionary. dist[raster_value]=list(color,label)
    legend_dict = {0: ["black", "Flat (0)"],
                   1: ["red", "North (337.5 - 22.5)"],
                   2: ["orange", 'Northeast (22.5 - 67.5)'],
                   3: ["yellow", 'East (67.5 - 112.5)'],
                   4: ["lime", 'Southeast (112.5 - 157.5)'],
                   5: ["cyan", 'South (157.5 - 202.5)'],
                   6: ["cornflowerblue", 'Southwest (202.5 - 247.5)'],
                   7: ["blue", 'West (247.5 - 292.5)'],
                   8: ["purple", 'Northwest (292.5 - 337.5)']}

    label_dict = {0: "Flat (0)",
                  1: "North (337.7 - 22.5)",
                  2: 'Northeast (22.5 - 67.5)',
                  3: 'East (67.5 - 112.5)',
                  4: 'Southeast (112.5 - 157.5)',
                  5: 'South (157.5 - 202.5)',
                  6: 'Southwest (202.5 - 247.5)',
                  7: 'West (247.5 - 292.5)',
                  8: 'Northwest (292.5 - 337.5)'}

    wgs_epsg = 4326
    figsize = (15, 15 * 0.6)  # width, height in inches
    title = settings['basin_name'].capitalize() + ' aspect class'
    leg_ncol = 2
    leg_loc = 'upper center'
    leg_bbox_to_anchor = (0.5, -0.1)

    # plot classified aspect
    output_fig_file = os.path.join(settings['basin_data_path'], 'plots/aspect_class_and_bound_stream.png')
    gp.plot_raster_and_bound_stream(settings['aspect_class_raster'], settings['basin_gru_prj_shp'],
                                    settings['basin_flowlines_shp'], wgs_epsg, 'user',
                                    legend_dict, figsize, title, leg_loc, leg_bbox_to_anchor, leg_ncol, output_fig_file)

    return


if __name__ == '__main__':

    control_file                      = '../test_cases/kananaskis/control_kananaskis.txt'
    settings                          = ut.read_complete_control_file(control_file, logging=True)

    settings['dem_prj_raster']        = settings['fulldom_dem_raster'].split('.tif')[0]+'_prj.tif'
    settings['basin_gru_prj_shp']     = settings['basin_gru_shp'].split('.shp')[0]+'_prj.shp'

    prepare_dem_grids(settings)
    plot_dem_slope_aspect(settings)
    prepare_dem_grids_plot(settings)
