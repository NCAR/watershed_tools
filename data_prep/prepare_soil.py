#!/usr/bin/env python
# coding: utf-8
"""
Prepare soiltype input for discretization

Prepare soiltype input data for the basin area. This step includes:
1. project full-domain soiltype raster into a specified projection system
2. extract soiltype input for the basin area
3. check basin soiltype


@authors: orig hongli liu, ncar, 2021
          debugged/revised by andy wood, ncar, 2021
          amended by dave casson, usask, 2022

"""

import os,sys
sys.path.append('../')
import functions.geospatial_analysis as ga
import functions.geospatial_plot as gp
import functions.utils as ut
import rasterio as rio
from rasterio.warp import Resampling
import logging
logger = logging.getLogger(__name__)

def prepare_soil(settings):
    """Prepare soil settings"""
    # derived filenames
    settings['plot_path']       = os.path.join(settings['basin_data_path'], 'plots/')

    settings['basin_gru_prj_shp']     = settings['basin_gru_shp'].split('.shp')[0]+'_prj.shp'
    settings['soiltype_prj_raster']   = settings['fulldom_soiltype_raster'].split('.tif')[0]+'_prj.tif'
    ##### 1. Reproject full-domain soiltype ####
    # if this is slow, the file can be prepared externally using gdalwarp -t_srs EPSG:<epsg> <input_tif> <output_tif>
    # (issue: the reprojected filesize may balloons by a factor of 200 due to substandard compression)
    if os.path.exists(settings['fulldom_soiltype_raster']):
        ga.reproject_raster(settings['fulldom_soiltype_raster'], settings['soiltype_prj_raster'],
                            settings['dest_crs'], Resampling.nearest)
    logger.info(f'Reprojected soiltype raster: {settings["soiltype_prj_raster"]}')

    # #### 2. Extract basin soiltype ####
    if not os.path.exists(settings['basin_soiltype_raster']):
        ga.crop_raster(settings['soiltype_prj_raster'], settings['basin_gru_prj_shp'], settings['basin_soiltype_raster'])
    logger.info(f'Cropped basin soiltype raster: {settings["basin_soiltype_raster"]}')

    # #### 3. Check domain soil ####
    plot_soils(settings)
    logger.info(f'Plotting soils')

    return

def plot_soils(settings):
    # Plot settings
    wgs_epsg = 4326
    figsize = (15, 15 * 0.6)  # width, height in inches
    title = settings['basin_name'].capitalize() + ' soil class'
    leg_ncol = 2
    leg_loc = 'upper center'
    leg_bbox_to_anchor = (0.5, -0.15)

    legend_dict = {0: "NoData",
                   1: 'CLAY',
                   2: 'CLAY LOAM',
                   3: 'LOAM',
                   4: 'LOAMY SAND',
                   5: 'SAND',
                   6: 'SANDY CLAY',
                   7: 'SANDY CLAY LOAM',
                   8: 'SANDY LOAM',
                   9: 'SILT',
                   10: 'SILTY CLAY',
                   11: 'SILTY CLAY LOAM',
                   12: 'SILT LOAM'}

    # plot classified aspect
    output_fig_file = os.path.join(settings['plot_path'], 'soiltype_and_bound_stream.png')
    cmap_str = 'jet'

    gp.plot_raster_and_bound_stream(settings['basin_soiltype_raster'], settings['basin_gru_prj_shp'],
                                    settings['basin_flowlines_shp'], wgs_epsg, cmap_str,legend_dict,
                                    figsize, title, leg_loc, leg_bbox_to_anchor, leg_ncol, output_fig_file)

    logging.info(f'Exporting soil plot to {output_fig_file}')

if __name__ == '__main__':

    control_file    = '../test_cases/kananaskis/control_kananaskis.txt'
    settings = ut.read_complete_control_file(control_file, logging=True)

    prepare_soil(settings)




