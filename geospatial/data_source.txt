Data sources of discretization work

----------
Contact
Hongli Liu (hongli@ucar.edu)

----------
DEM: Multi-Error-Removed Improved-Terrain Digital Elevation Model (MERIT DEM) (Yamazaki et al., 2017) at 3sec resolution (~90m at the equator). 
/glade/u/home/andywood/rhap/common_data/geophysical/MERIT_Hydro_DEM_NLDAS/MERIT_Hydro_dem_NLDAS.tif

----------
HUC: watershed boundary dataset for 12-digit hydrologic units (U.S. Geological Survey, 2006)
/glade/work/andywood/wreg/gis/shapes/wUS_HUC_12_simplified

----------
Land cover: 20-category International Geosphere - Biosphere Programme (IGBP) land cover dataset at 1/160 degree resolution (IGBP, 1990) (http://www.eomf.ou.edu/static/IGBP.pdf)
/glade/work/mizukami/data/geophysical/modis/no_qc/160_MCD12Q1/NLDAS/annual_climate_lc.tif

----------
soilTypeIndex
- Wouter Knoben derived from SOILGRIDS data (Hengl et al., 2019) that show sand/silt/clay % for each pixel.
- SOILGRIDS sand/silt/clay % converted to soil class using USDA soil class definitions (Benham et al., 2009).
Indices refer to ROSETTA Table soil classes.
Files "usda_soilclass_sl[1-7]_NA_250m_ll.tif" contain soil class per pixel for each of 7 depths for which SOILGRIDS provides data.
File "usda_mode_soilclass_vCompressed_NA_250m_ll.tif" contains the soil class per pixel corresponding to the mode value for that pixel for the 7 provided depths.
/glade/u/home/wouter/geospatial_veg_soil_indices_north_america/usda_mode_soilclass_vCompressed_NA_250m_ll.tif

----------
River:  
/glade/work/andywood/wreg/gis/shapes/merit/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_wUS.shp
/glade/u/home/hongli/data/shapefile/riv_pfaf_7_MERIT_westUS/

----------
Others from Andy Wood (maybe useful)
# ==== config file for spatial (HUC) attribute plot script 'plot_HUC_attribute.py' ====
geo_file    = '/glade/work/andywood/wreg/gis/shapes/wUShuc12/wUS_HUC12_ext_v4.simpl-0.002.shp'
nc_file     = '/glade/u/home/andywood/proj/SHARP/wreg/pnnl/sf_flathead/settings/attributes.sf_flathead.v1.nc'
output_dir   = './'
GSHHS_coastline = '/glade/p/ral/hap/common_data/GIS/GMT/GSHHS_shp/i/GSHHS_i_L1.shp'      # GSHHS_shp coastlines path
WDBII_country  = '/glade/p/ral/hap/common_data/GIS/GMT/WDBII_shp/i/WDBII_border_i_L1.shp'  # WDBII_shp country border path
WDBII_state   = '/glade/p/ral/hap/common_data/GIS/GMT/WDBII_shp/i/WDBII_border_i_L2.shp'  # WDBII_shp state border path