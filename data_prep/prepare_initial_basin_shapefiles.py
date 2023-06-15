"""
Prepare basin GRU (hydrologic unit or GRU) and flowline shapefiles
If these don't both pre-exist, run this script before all other scripts

This script includes:
1a. if needed, extract basin GRU shapefile from a large-domain GRU shapefile.<br>
1b. write basin gruId.txt list
2. extract basin flowlines shapefile from a large-domain flowlines shapefile.<br>
3. reproject basin GRU and flowlines shapefiles to a common equal coordinate system. <br>
The script also makes some directories used in the discretization process.

@authors: orig hongli liu, ncar, 2021
          debugged/revised by andy wood, ncar, 2021
          amended by dave casson, usask, 2022

"""


# Import libraries
import os, sys
sys.path.append('../')
import functions.geospatial_analysis as ga
import functions.utils as ut
import geopandas as gpd
import functions.ogr2ogr as ogr2ogr
import numpy as np

import logging
logger = logging.getLogger(__name__)

def prepare_initial_basin_shapefiles(settings):
    # Make standard directories
    create_dir_structure(settings)

    if settings['merge_cat_riv_file'] == 'True':
        # If using seperate data files where upstream hru is in the rivfile,
        # Perform an initial merge of the shp files to combine attributes based on spatial location

        ga.merge_shp_spatial_join(settings['cat_file'],
                                  settings['riv_file'],
                                  settings['fulldom_gru_shpfile'],
                                  merge_attr='COMID')

    # Derived filenames
    settings['basin_gru_prj_shp'] = settings['basin_gru_shp'].split('.shp')[0] + '_prj.shp'

    # If the basin shapefile doesn't exist, it needs to be extracted from another larger GRU shapefile
    if not os.path.exists(settings['basin_gru_shp']):
        set_basin_GRU_shp(settings)

    # Reproject basin GRU shapefile if it doesn't exist
    if not os.path.exists(settings['basin_gru_prj_shp']):
        new_epsg = settings['epsg']
        ga.reproject_vector(settings['basin_gru_shp'], settings['basin_gru_prj_shp'], new_epsg)
        logger.info(f'Reprojected basin GRUs: {settings["basin_gru_prj_shp"]}')

    # Extract the baseline flow lines
    if not os.path.exists(settings['basin_flowlines_shp']):
        extract_basin_flowline_shp(settings)
        logger.info('Extracting basin flowlines shape')

def create_dir_structure(settings):
    """Create directory structure based on control file settings """

    basin_data_path = settings['basin_data_path']

    # Make standard directories
    if not os.path.exists(basin_data_path):
        os.makedirs(basin_data_path)
    plot_path  = os.path.join(basin_data_path, 'plots/')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    gis_path  = os.path.join(basin_data_path, 'gis/')
    if not os.path.exists(gis_path):
        os.makedirs(gis_path)
    logger.info(f'Setting directory paths {gis_path} and {plot_path}')

    return

def set_basin_GRU_shp(settings):
    """ Extract basin GRU shapefile and ID list from a larger full-domain GRU shapefile
    Set basin GRU shapefile (extract from larger full-domain if needed) """

    # read filename and other necessary info
    fulldom_gru_shp = settings['fulldom_gru_shpfile']
    outlet_gruId    = int(settings['basin_outlet_gruId'])
    toGRU_fieldname = settings['toGRU_fieldname']
    gru_fieldname   = settings['gru_fieldname']
    basin_gru_shp   = settings['basin_gru_shp']
    data = gpd.read_file(fulldom_gru_shp)

    # check whether two useful columns (gru_field, toGRU_field) are in gru_shp.
    if not gru_fieldname in data.columns.values:
        exit(gru_fieldname + ' column does not exist in shapefile.')
    else:
        grus = data[gru_fieldname].values
    if not toGRU_fieldname in data.columns.values:
        exit(toGRU_fieldname + ' column does not exist in shapefile.')
    else:
        togrus = data[toGRU_fieldname].values

    # extract only the useful columns to save data memory.
    data = data[[gru_fieldname, toGRU_fieldname, 'geometry']]

    # ---- search upstream GRUs ----
    # method 1: search upstream grus base on the most downstream gruId
    upstream_grus = [outlet_gruId]  # list of upstream grus. initiate with outlet_gruid
    gru_found = np.unique(
        grus[np.where(togrus == outlet_gruId)])  # find all the upstream grus that drain to outlet_gruid.
    upstream_grus.extend(list(gru_found))  # add the found upstream grus of outlet_gruid to upstream_grus list
    round_num = 0  # record the round number of searching.

    while len(gru_found) != 0:  # terminate searching upstream grus until no one can be found any more.
        round_num = round_num + 1
        print("Round %d: %d GRUs found." % (round_num, len(upstream_grus)))

        # search upstream grus
        gru_found_next = []
        for gru_i in gru_found:
            gru_found_next.extend(list(grus[np.where(togrus == gru_i)]))
        gru_found_next = np.unique(gru_found_next)

        # identify if the found GRUs exist in upstrm_grus
        gru_found = [gru for gru in gru_found_next if not gru in upstream_grus]
        upstream_grus.extend(gru_found)

        # alternate method, not used: manually add upstream_grus when the list of upstream grus is known.
        # upstream_grus= np.loadtxt('/glade/u/home/andywood/proj/SHARP/wreg/bighorn/prep/lists/gruIds.06279940.txt',dtype=int)

    # ---- save upstream GRU shapefile ----
    upstream_grus = np.unique(upstream_grus)

    data[data[gru_fieldname].isin(upstream_grus)].to_file(basin_gru_shp)

    return


def write_gruId_list(settings):
    """Output gruId list to txt file"""

    # set basin shapefiles
    basin_gru_shp = settings['basin_gru_shp']  # may exist
    gru_fieldname = settings['gru_fieldname'] # gru fieldname and text file

    # read the basin shapefile and write gruId list
    data = gpd.read_file(basin_gru_shp)
    if not gru_fieldname in data.columns.values:
        exit(gru_fieldname + ' column does not exist in shapefile ' + basin_gru_shp)
    else:
        grus = data[gru_fieldname].values

    if 'int' in str(grus.dtype):
        np.savetxt(basin_gruId_txt, grus, fmt='%d')
    else:
        np.savetxt(basin_gruId_txt, grus, fmt='%s')
    logging.info('Wrote gruId file for the target basin %s: %s' % (basin_name, basin_gruId_txt))


def extract_basin_flowline_shp(settings):
    """ Extract basin flowlines from full-dom flowlines file if it doesn't exist
        Note that the basin flowlines shapefile will be in the common projected coordinates (new_epsg)
        this step can take a few minutes (wait for 'done')"""

    # May need to reproject full-domain flowlines shapefile first
    flowlines_shp = settings['fulldom_flowlines_shp']
    basin_gru_prj_shp = settings['basin_gru_prj_shp']
    basin_flowlines_shp = settings['basin_flowlines_shp']
    flowlines_prj_shp = flowlines_shp.split('.shp')[0] + '_prj.shp'

    new_epsg = settings['epsg']
    if not os.path.exists(flowlines_prj_shp):
        ga.reproject_vector(flowlines_shp, flowlines_prj_shp, new_epsg)
        print('reprojected full domain streams:', flowlines_prj_shp)

    # read stream and boundary files (projected)
    flowlines_gpd = gpd.read_file(flowlines_prj_shp)
    basin_gru_gpd = gpd.read_file(basin_gru_prj_shp)
    print('read reprojected shapefiles for clipping flowlines')

    # create basin outer boundary shapefile
    #tmp_gpd = basin_gru_gpd[['geometry']]
    basin_gru_gpd['tmp_col'] = 0  # create null column for dissolve
    basin_boundary_gpd = basin_gru_gpd.dissolve(by='tmp_col')
    basin_boundary_prj_shp = basin_gru_prj_shp.split('.shp')[0] + '_boundary.shp'
    basin_boundary_gpd.to_file(basin_boundary_prj_shp)
    print('wrote basin boundary shapefile to use in stream clipping:', basin_boundary_prj_shp)

    # clip full-dom reprojected flowlines with basin boundary
    #   note: if geopandas version < 0.7, cannot use clip(), so instead use ogr2ogr
    if float(gpd.__version__.split(".")[0] + "." + gpd.__version__.split(".")[1]) >= 0.7:
        in_gpd_clip = gpd.clip(flowlines_gpd, basin_boundary_gpd)
        in_gpd_clip.to_file(basin_flowlines_shp)
    else:
        print('Note: using ogr2ogr to clip streams to basin')
        driverName = 'ESRI Shapefile'  # can later be upgraded to work with geopackages (eg 'GPKG')
        ogr2ogr.main(["", "-f", driverName, "-clipsrc", basin_boundary_prj_shp, basin_flowlines_shp, flowlines_prj_shp])

    print('wrote basin-clipped stream shapefile:', basin_flowlines_shp)
    print('done')

if __name__ == '__main__':

    control_file    = '/Users/drc858/GitHub/watershed_tools/test_cases/tuolumne/control_tuolumne.txt'
    settings = ut.read_complete_control_file(control_file, logging=True)

    prepare_initial_basin_shapefiles(settings)







