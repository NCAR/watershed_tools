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
from pathlib import Path
sys.path.append('../')
import functions.geospatial_analysis as ga
import functions.geospatial_plot as gp
import functions.wt_utils as ut
import rasterio as rio
from rasterio.warp import Resampling
import logging
import yaml
logger = logging.getLogger(__name__)

def prepare_soil(settings):
    """Prepare soil settings"""

    ##### 1. Reproject full-domain soiltype ####
    # if this is slow, the file can be prepared externally using gdalwarp -t_srs EPSG:<epsg> <input_tif> <output_tif>
    # (issue: the reprojected filesize may balloons by a factor of 200 due to substandard compression)
    if not os.path.exists(settings["soiltype_prj_raster"]):
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
    wgs_epsg = rio.crs.CRS.from_epsg(settings['epsg'])
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

    config_file = '/Users/drc858/GitHub/summa_snakemake/snakemake/config/summa_snakemake_config.yaml'

    #read yaml file to dict using python package yaml.safe_load
    with open(config_file) as file:
        config = yaml.safe_load(file)

    control_file    = config['watershed_tools_paths']['control_file']
    wt_config          = ut.read_complete_control_file(control_file, logging=False)
    
    prepare_soil(wt_config)


