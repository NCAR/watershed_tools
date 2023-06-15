#!/usr/bin/env python
# coding: utf-8

# ### Generate HRU at different complexity levels ###

# In[1]:


import os, sys
import numpy as np
import rasterio as rio
import geopandas as gpd
import matplotlib.pyplot as plt 
sys.path.append('../')
import functions.geospatial_analysis as ga
import functions.utils as ut


# In[2]:


# common paths
control_file    = '/Users/drc858/GitHub/watershed_tools/test_cases/tuolumne/control_tuolumne.txt'
basin_data_path = ut.read_from_control(control_file, 'basin_data_path')
basin_name      = ut.read_from_control(control_file, 'basin_name')
main_path       = ut.read_from_control(control_file, 'main_path')
results_path    = os.path.join(basin_data_path, 'results/')
if not os.path.exists(results_path):
    os.makedirs(results_path)


# In[3]:


# basin data files and fields
basin_gru_shp             = ut.set_filename(control_file, 'basin_gru_shp')  
basin_gru_raster          = ut.set_filename(control_file, 'basin_gru_raster')
basin_gruNo_gruId_txt     = ut.set_filename(control_file, 'basin_gruNo_gruId_txt')
basin_dem_raster          = ut.set_filename(control_file, 'basin_dem_raster')  
basin_slope_raster        = ut.set_filename(control_file, 'basin_slope_raster')  
basin_aspect_raster       = ut.set_filename(control_file, 'basin_aspect_raster')
basin_soiltype_raster     = ut.set_filename(control_file, 'basin_soiltype_raster')
basin_radiation_raster    = ut.set_filename(control_file, 'basin_radiation_raster')
refraster                 = ut.set_filename(control_file, 'refraster')
basin_canopy_class_raster = ut.set_filename(control_file, 'basin_canopy_class_raster')

# derived filenames
basin_gru_prj_shp = basin_gru_shp.split('.shp')[0]+'_prj.shp' 


# In[4]:


# gru fieldnames
gruNo_fieldname           = ut.read_from_control(control_file, 'gruNo_fieldname')
gruNo_field_dtype         = ut.read_from_control(control_file, 'gruNo_field_dtype')
gruId_fieldname           = ut.read_from_control(control_file, 'gruId_fieldname')

# hru field names
hruNo_fieldname           = ut.read_from_control(control_file, 'hruNo_fieldname')
hruNo_field_dtype         = ut.read_from_control(control_file, 'hruNo_field_dtype')         # used to save hruNo raster (cannot be int64)
hruId_fieldname           = ut.read_from_control(control_file, 'hruId_fieldname')           # field name of hru name, e.g., 10080012010101, 100800120102. 
    
elev_class_fieldname      = ut.read_from_control(control_file, 'elev_class_fieldname')      # field name of the elevation class column in HRU. 
land_class_fieldname      = ut.read_from_control(control_file, 'land_class_fieldname')      # field name of the land class column in HRU. 
radiation_class_fieldname = ut.read_from_control(control_file, 'radiation_class_fieldname') # field name of the radiation class column in HRU. 
hru_area_fieldname        = ut.read_from_control(control_file, 'hru_area_fieldname')        # field name of the HRU area.

hru_threshold_type        = ut.read_from_control(control_file, 'hru_threshold_type')        # use a fraction or area value to eliminate small HRUs.
hru_threshold             = float(ut.read_from_control(control_file, 'hru_threshold'))      # if hru_thld_type = 'fraction', hru_thld = partial of the gru area.                                                                                   ## if hru_thld_type = 'value', hru_thld = elimination area.


# In[5]:


# basename for dem and radiation classification in a hru level.
dem_class_basename = 'dem_class' # basename for DEM class files (eg, 0:low elevation. 1: high elevation).
dem_value_basename = 'dem_value' # basename for DEM value files (eg, average DEM per class).

rad_class_basename = 'rad_class' # basename for radiation class files (eg, 0:low. 1:high).
rad_value_basename = 'rad_value' # basename for radiation value files (eg, average radiation per class).


# #### Define HRU complexity levels ####
# level 0: GRU = HRU. <br>
# level 1a: use only elevation bands in HRU generation.<br>
# level 1b: use only canopy class in HRU generation.<br>
# level 1c: use only radiation class in HRU generation.<br>
# level 2a: use elevation bands and canopy class in HRU generation.<br>
# level 2b: use elevation bands and radiation class in HRU generation.<br>
# level 2c: use canopy class and radiation class in HRU generation.<br>
# level 3: use elevation bands, radiation class, canopy class in HRU generation.<br>

# In[10]:


level_list = ['1a']


# #### 1. Discretize HRU ####

# In[15]:


for level in level_list:

    print('\n--- Complexity level %s ---' %(level))
    
    #  --- PART 1. define output files of discretization--- 
    hru_str         = 'hru_lev' + str(level)
    hru_elmn_str    = hru_str + '_elmn'     
    
    hru_raster      = os.path.join(results_path, hru_str+'.tif')       # original HRU
    hru_vector      = os.path.join(results_path, hru_str+'.shp')
    hru_raster_elmn = os.path.join(results_path, hru_elmn_str+'.tif')  # simplified HRU
    hru_vector_elmn = os.path.join(results_path, hru_elmn_str+'.shp')    

    # these settings should be in control file
    dem_classif_trigger = 300 # Elvation difference value per GRU used to trigger elevation classification.
    dem_bins            = 'median' # Elevation classification method. 'median' means using the median value per GRU as the classification threhold.
    dem_class_raster    = os.path.join(results_path, dem_class_basename+'_lev'+str(level)+'.tif')
    dem_value_raster    = os.path.join(results_path, dem_value_basename+'_lev'+str(level)+'.tif')
      
    rad_classif_trigger = 50 #None # Radiation difference value per GRU used to trigger elevation classification.
    rad_bins            = 'median' # Radiation classification method. 'median' means using the median value per GRU as the classification threhold.
    rad_class_raster    = os.path.join(results_path, rad_class_basename+'_lev'+str(level)+'.tif')
    rad_value_raster    = os.path.join(results_path, rad_value_basename+'_lev'+str(level)+'.tif')
    
    #  --- PART 2. define inputs of discretization ---
    #   raster_list: a list of raster inputs that are used to define HRU.
    #   fieldname_list: a list of field names corresponding to raster_list.
    #   note:  elevation and radiation class rasters may need generation at each step, while
    #          the landcover/canopy class raster already exists from prepare_landcover script

    print('  classifying input rasters')

    # level 0: GRU = HRU (benchmark). 
    if level == '0': 
        # (1) define input files for hru discretization
        raster_list    = [basin_gru_raster]
        fieldname_list = [gruNo_fieldname]  

    # level 1a: use only elevation bands in HRU generation.
    if level == '1a': 
        # (1) classify elevation raster per gru
        if not os.path.exists(dem_class_raster):
            ga.classify_raster(basin_dem_raster, basin_gru_raster, dem_classif_trigger, dem_bins,
                               dem_class_raster, dem_value_raster)        
        # (2) define input files for hru discretization
        raster_list    = [basin_gru_raster, dem_class_raster]
        fieldname_list = [gruNo_fieldname, elev_class_fieldname]
        print(dem_bins)
    
    # level 1b: use only canopy classes in HRU generation.
    if level == '1b': 
        # (1) canopy class raster already exists from prepare_landcover script
        # (2) define input files for hru discretization
        raster_list    = [basin_gru_raster, basin_canopy_class_raster]
        fieldname_list = [gruNo_fieldname, land_class_fieldname]
 
    # level 1c: use only radiation class in HRU generation.
    if level == '1c': 
        # (1) classify radiation raster per gru
        if not os.path.exists(rad_class_raster):
            ga.classify_raster(basin_radiation_raster, basin_gru_raster, rad_classif_trigger, rad_bins, 
                               rad_class_raster, rad_value_raster)        
        # (2) define input files for hru discretization
        raster_list    = [basin_gru_raster, rad_class_raster]
        fieldname_list = [gruNo_fieldname, radiation_class_fieldname]
 
    # level 2a: use elevation bands and landcover classes in HRU generation.
    elif level == '2a': 
        # (1) classify elevation raster per gru (landcover class already exists)
        if not os.path.exists(dem_class_raster):
            ga.classify_raster(basin_dem_raster, basin_gru_raster, dem_classif_trigger, dem_bins,
                               dem_class_raster, dem_value_raster)        
        # (2) define input files for hru discretization
        raster_list    = [basin_gru_raster, dem_class_raster, basin_canopy_class_raster]
        fieldname_list = [gruNo_fieldname, elev_class_fieldname, land_class_fieldname]

    # level 2b: use elevation bands and radiation class in HRU generation.
    elif level == '2b': 
        # (1) classify elevation raster per gru
        if not os.path.exists(dem_class_raster):
            ga.classify_raster(basin_dem_raster, basin_gru_raster, dem_classif_trigger, dem_bins,
                               dem_class_raster, dem_value_raster)        
        #      classify radiation raster per gru
        if not os.path.exists(rad_class_raster):
            ga.classify_raster(basin_radiation_raster, basin_gru_raster, rad_classif_trigger, rad_bins, 
                               rad_class_raster, rad_value_raster)        
        # (2) define input files for hru discretization
        raster_list    = [basin_gru_raster, dem_class_raster, rad_class_raster]
        fieldname_list = [gruNo_fieldname, elev_class_fieldname, radiation_class_fieldname]
    
    # level 2c: use landcover class and radiation class in HRU generation.
    elif level == '2c': 
        # (1) classify radiation raster per gru (landcover class already exists)
        if not os.path.exists(rad_class_raster):
            ga.classify_raster(basin_radiation_raster, basin_gru_raster, rad_classif_trigger, rad_bins, 
                               rad_class_raster, rad_value_raster)        
        # (2) define input files for hru discretization
        raster_list    = [basin_gru_raster, basin_canopy_class_raster, rad_class_raster]
        fieldname_list = [gruNo_fieldname, land_class_fieldname, radiation_class_fieldname]
    
    # level 3: use elevation bands, radiation bands, landcover classes in HRU generation.
    elif level == '3':
        # (1) classify elevation raster per gru
        if not os.path.exists(dem_class_raster):
            ga.classify_raster(basin_dem_raster, basin_gru_raster, dem_classif_trigger, dem_bins,
                               dem_class_raster, dem_value_raster)        
        #     classify radiation raster per gru       
        if not os.path.exists(rad_class_raster):
            ga.classify_raster(basin_radiation_raster, basin_gru_raster, rad_classif_trigger, rad_bins, 
                               rad_class_raster, rad_value_raster)              
        # (2) define input files for hru discretization
        raster_list    = [basin_gru_raster, dem_class_raster, rad_class_raster, basin_canopy_class_raster]
        fieldname_list = [gruNo_fieldname, elev_class_fieldname, radiation_class_fieldname, land_class_fieldname]

    # --- PART 3. generate HRU based on gru and elevation class ---
    print('  defining new HRUs')
    print(raster_list, fieldname_list, basin_gru_raster, basin_gruNo_gruId_txt, gruNo_fieldname, gruId_fieldname,
                  hru_raster, hru_vector, hruNo_fieldname, hruNo_field_dtype, hruId_fieldname)
    ga.define_hru(raster_list, fieldname_list, basin_gru_raster, basin_gruNo_gruId_txt, gruNo_fieldname, gruId_fieldname,
                  hru_raster, hru_vector, hruNo_fieldname, hruNo_field_dtype, hruId_fieldname)
    print('  wrote shapefile', hru_vector)
    
    
    # --- PART 4. calculate HRU area ---
    print('  adding HRU areas to shapefile')
    in_gpd                     = gpd.read_file(hru_vector)
    in_gpd[hru_area_fieldname] = in_gpd.area
    in_gpd.to_file(hru_vector)

    # --- PART 5. eliminate small area HRUs ---
    # method 1: change HRU attribute to the most dominant HRU within the GRU. Use function ga.eliminate_small_hrus_dominant
    # method 2: change HRU attribute to its largest neighbor's HRU
    print('  eliminating small HRU fractions')
    if not (os.path.exists(hru_vector_elmn) and os.path.exists(hru_raster_elmn)):
        ga.eliminate_small_hrus_neighbor(hru_vector, hru_threshold_type, hru_threshold, gruNo_fieldname, gruId_fieldname, 
                                         hruNo_fieldname, hruNo_field_dtype, hruId_fieldname, hru_area_fieldname, 
                                         fieldname_list, refraster, hru_vector_elmn, hru_raster_elmn)
        print('  wrote new shapefile', hru_vector_elmn)
    else:
        print('  shapefile exists, skipped:', hru_vector_elmn)
    
    #ga.plot_vector(hru_vector_elmn, hruName_field) # quick plot for check


# #### 2. Calculate HRU zonal statistics ####

# In[8]:


for level in level_list:

    print('--- Complexity level %s ---' %(level))
    
    #  --- PART 1. define hru complexity level dependent files --- 
    hru_str      = 'hru_lev' + str(level)
    hru_elmn_str = hru_str + '_elmn'     
    
    hru_vector      = os.path.join(results_path, hru_str+'.shp')
    hru_vector_elmn = os.path.join(results_path, hru_elmn_str+'.shp')    
    
    # --- PART 2. calculate zonal statistics ---
    for invector in [hru_vector, hru_vector_elmn]:
        
        # (1) define invector dependent files 
        invector_field       = hruNo_fieldname
        invector_field_dtype = hruNo_field_dtype
        
        attrb_elev = os.path.join(results_path, invector.split('.shp')[0]+'_attrb_elevation.tif')        
        attrb_slp  = os.path.join(results_path, invector.split('.shp')[0]+'_attrb_slope.tif')        
        attrb_asp  = os.path.join(results_path, invector.split('.shp')[0]+'_attrb_aspect.tif')        
        attrb_lc   = os.path.join(results_path, invector.split('.shp')[0]+'_attrb_landcover.tif')        
        attrb_soil = os.path.join(results_path, invector.split('.shp')[0]+'_attrb_soil.tif')        
        
        # (2) elevation zonal statistics 
        ga.zonal_statistic(basin_dem_raster, invector, invector_field, invector_field_dtype, 
                           refraster, 'mean', attrb_elev, output_column_prefix='elev')

        # (3) slope zonal statistics 
        ga.zonal_statistic(basin_slope_raster, invector, invector_field, invector_field_dtype, refraster, 
                           'mean', attrb_slp, output_column_prefix='slope')

        # (4) aspect zonal statistics 
        ga.zonal_statistic(basin_aspect_raster, invector, invector_field, invector_field_dtype, refrraster,
                           'mean_aspect', attrb_asp, output_column_prefix='aspect')

        # (5) landcover zonal statistics 
        ga.zonal_statistic(basin_canopy_class_raster, invector, invector_field, invector_field_dtype, refraster, 
                           'mode', attrb_lc, output_column_prefix='vegType')        
        
        # (6) soil zonal statistics 
        ga.zonal_statistic(basin_soiltype_raster, invector, invector_field, invector_field_dtype, refraster, 
                           'mode', attrb_soil, output_column_prefix='soilType')        
        
        # -------- post-process attributes for SUMMA ---------
        # (7) landcover and soil types 
        # convert landcover int to [1,17] range 
        # change soilType from float to int (because source soilType is float)
        in_gpd             = gpd.read_file(invector)
        in_gpd['vegType']  = in_gpd['vegType']+1
        in_gpd['soilType'] = in_gpd['soilType'].astype('int')
        in_gpd.to_file(invector)
        
        # (8) convert landcover int to string for easy understanding
        lcClass_list = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 255]
        lcValue_list = ['Evergreen needleleaf forests', 'Evergreen broadleaf forests', 'Deciduous needleleaf forests',
                        'Deciduous broadleaf forests', 'Mixed forests', 'Closed shrublands', 'Open shrublands', 
                        'Woody savannas', 'Savannas', 'Grasslands', 'Permanent wetlands', 'Croplands', 
                        'Urban and built-up lands', 'Cropland/natural vegetation mosaics', 'Snow and ice', 
                        'Barren', 'Water bodies', 'None']
        in_gpd              = gpd.read_file(invector)
        in_gpd['landcover'] = ""
        for irow, row in in_gpd.iterrows():
            lcClass = in_gpd.loc[irow,'vegType'] 
            lcValue = lcValue_list[lcClass_list.index(lcClass)]
            in_gpd.at[irow,'landcover'] = lcValue
        in_gpd['landcover'] = in_gpd['landcover'].astype('str')
        in_gpd.to_file(invector)

        # (9) convert ROSETTA soil to STAS and add string for easy understanding
        soilClass_list = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        soilValue_list = ['OTHER(land-ice)', 'CLAY', 'CLAY LOAM', 'LOAM', 'LOAMY SAND', 'SAND', 'SANDY CLAY', 
                          'SANDY CLAY LOAM', 'SANDY LOAM', 'SILT','SILTY CLAY', 'SILTY CLAY LOAM', 'SILT LOAM']
        soilClass_list_STAS = [16, 12, 9, 6, 2, 1, 10, 7, 3, 5, 11, 8, 4]
        
        in_gpd = gpd.read_file(invector)
        in_gpd['soilROSETTA'] = in_gpd['soilType']
        in_gpd['soilSTAS']    = ""
        in_gpd['soil']        = ""
        for irow, row in in_gpd.iterrows():
            
            soilClass = in_gpd.loc[irow,'soilType'] 
            if soilClass==0:
                lcClass = in_gpd.loc[irow,'vegType'] 
                #print('hruNo = %d, soilType_ROSETTA = 0, and vegType = %s.'%(in_gpd.loc[irow,'hruNo'],lcClass))
            
            soilValue      = soilValue_list[soilClass_list.index(soilClass)]
            soilClass_STAS = soilClass_list_STAS[soilClass_list.index(soilClass)]
            
            in_gpd.at[irow,'soil']     = soilValue
            in_gpd.at[irow,'soilSTAS'] = soilClass_STAS
            
        in_gpd['soil']     = in_gpd['soil'].astype('str')
        in_gpd['soilSTAS'] = in_gpd['soilSTAS'].astype('int')
        in_gpd['soilType'] = in_gpd['soilSTAS']
        in_gpd = in_gpd.drop(columns=['soilSTAS'])
        in_gpd.to_file(invector)

        # (10) convert slope to tan_slope 
        in_gpd              = gpd.read_file(invector)
        in_gpd['tan_slope'] = np.tan(np.radians(in_gpd['slope']))
        in_gpd.to_file(invector)

        # (11) calculate contourLength (meter)
        # assuming the hru area is a circle and taking the radius as contourLength.
        in_gpd                  = gpd.read_file(invector)
        in_gpd['contourLength'] = np.power(in_gpd['areaSqm']/np.pi,0.5)
        in_gpd.to_file(invector)

        # (12) calculate centroid lat/lon (degree)
        def getXY(pt):
            return (pt.x, pt.y)
        in_gpd         = gpd.read_file(invector)
        in_gpd_wgs84   = in_gpd.copy()
        in_gpd_wgs84   = in_gpd_wgs84.to_crs(epsg=4326) #"EPSG:4326"
        centroidseries = in_gpd_wgs84['geometry'].centroid
        in_gpd['longitude'],in_gpd['latitude'] = [list(t) for t in zip(*map(getXY, centroidseries))]
        in_gpd.to_file(invector)

    # --- PART 3. save HRU with attributes into gpkg ---
    invector_gpkg = invector.split('.shp')[0]+'.gpkg'
    in_gpd        = gpd.read_file(invector)
    in_gpd.to_file(invector_gpkg, driver="GPKG")


# #### 3. Calculate GRU zonal statistics (optional) ####

# In[9]:


print('setting filenames and reading basin shapefile')
invector             = basin_gru_prj_shp
invector_field       = gruNo_fieldname
invector_field_dtype = gruNo_field_dtype

attrb_elev = os.path.join(results_path, invector.split('.shp')[0]+'_attrb_elevation.tif')        
attrb_slp  = os.path.join(results_path, invector.split('.shp')[0]+'_attrb_slope.tif')        
attrb_asp  = os.path.join(results_path, invector.split('.shp')[0]+'_attrb_aspect.tif')        
attrb_lc   = os.path.join(results_path, invector.split('.shp')[0]+'_attrb_landcover.tif')        
attrb_soil = os.path.join(results_path, invector.split('.shp')[0]+'_attrb_soil.tif')        

outvector = invector.split('.shp')[0]+'_attrb.gpkg'   
in_gpd    = gpd.read_file(invector)
in_gpd.to_file(outvector, driver="GPKG")
invector  = outvector # avoid process gru_shp_prj. Work on gpkg.

# (1) calculate zonal area ---
print('calculating zonal area for shapefiles')

in_gpd            = gpd.read_file(invector)
in_gpd['areaSqm'] = in_gpd.area
in_gpd.to_file(invector)

# (2) elevation zonal statistics 
ga.zonal_statistic(basin_dem_raster, invector, invector_field, invector_field_dtype, refraster, 'mean', attrb_elev,
                   output_column_prefix='elev')

# (3) slope zonal statistics 
ga.zonal_statistic(basin_slope_raster, invector, invector_field, invector_field_dtype, refraster, 'mean', attrb_slp,
                   output_column_prefix='slope')

# (4) aspect zonal statistics 
ga.zonal_statistic(basin_aspect_raster, invector, invector_field, invector_field_dtype, refraster, 'mean_aspect', attrb_asp,
                   output_column_prefix='aspect')

# (5) landcover zonal statistics 
ga.zonal_statistic(basin_canopy_class_raster, invector, invector_field, invector_field_dtype, refraster, 'mode', attrb_lc,
                   output_column_prefix='vegType')        

# (6) soil zonal statistics 
ga.zonal_statistic(basin_soiltype_raster, invector, invector_field, invector_field_dtype, refraster, 'mode', attrb_soil,
                   output_column_prefix='soilType')        

# -------- post-process attributes for SUMMA ---------
print('calculating zonal landcover and soil types')

# (7) landcover and soil types 
# convert landcover int to [1,17] range 
# change soilType from float to int (because source soilType is float)
in_gpd             = gpd.read_file(invector)
in_gpd['vegType']  = in_gpd['vegType']+1
in_gpd['soilType'] = in_gpd['soilType'].astype('int')
in_gpd.to_file(invector)

# (8) convert landcover int to string for easy understanding
lcClass_list = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 255]
lcValue_list = ['Evergreen needleleaf forests', 'Evergreen broadleaf forests', 'Deciduous needleleaf forests',
                'Deciduous broadleaf forests', 'Mixed forests', 'Closed shrublands', 'Open shrublands', 
                'Woody savannas', 'Savannas', 'Grasslands', 'Permanent wetlands', 'Croplands', 
                'Urban and built-up lands', 'Cropland/natural vegetation mosaics', 'Snow and ice', 
                'Barren', 'Water bodies', 'None']
in_gpd              = gpd.read_file(invector)
in_gpd['landcover'] = ""
for irow, row in in_gpd.iterrows():
    lcClass = in_gpd.loc[irow,'vegType'] 
    lcValue = lcValue_list[lcClass_list.index(lcClass)]
    in_gpd.at[irow,'landcover'] = lcValue
in_gpd['landcover'] = in_gpd['landcover'].astype('str')
in_gpd.to_file(invector)

# (9) convert ROSETTA soil to STAS and add string for easy understanding
soilClass_list = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
soilValue_list = ['OTHER(land-ice)', 'CLAY', 'CLAY LOAM', 'LOAM', 'LOAMY SAND', 'SAND', 'SANDY CLAY', 
                  'SANDY CLAY LOAM', 'SANDY LOAM', 'SILT','SILTY CLAY', 'SILTY CLAY LOAM', 'SILT LOAM']
soilClass_list_STAS = [16, 12, 9, 6, 2, 1, 10, 7, 3, 5, 11, 8, 4]

in_gpd                = gpd.read_file(invector)
in_gpd['soilROSETTA'] = in_gpd['soilType']
in_gpd['soilSTAS']    = ""
in_gpd['soil']        = ""
for irow, row in in_gpd.iterrows():

    soilClass = in_gpd.loc[irow,'soilType'] 
    if soilClass==0:
        lcClass = in_gpd.loc[irow,'vegType'] 
        #print('hruNo = %d, soilType_ROSETTA = 0, and vegType = %s.' % (in_gpd.loc[irow,'hruNo'],lcClass))

    soilValue      = soilValue_list[soilClass_list.index(soilClass)]
    soilClass_STAS = soilClass_list_STAS[soilClass_list.index(soilClass)]

    in_gpd.at[irow,'soil']     = soilValue
    in_gpd.at[irow,'soilSTAS'] = soilClass_STAS

in_gpd['soil']     = in_gpd['soil'].astype('str')
in_gpd['soilSTAS'] = in_gpd['soilSTAS'].astype('int')
in_gpd['soilType'] = in_gpd['soilSTAS']
in_gpd             = in_gpd.drop(columns=['soilSTAS'])
in_gpd.to_file(invector)

# (10) convert slope to tan_slope 
in_gpd              = gpd.read_file(invector)
in_gpd['tan_slope'] = np.tan(np.radians(in_gpd['slope']))
in_gpd.to_file(invector)

# (11) calculate contourLength (meter)
# assuming the hru area is a circle and taking the radius as contourLength.
in_gpd                  = gpd.read_file(invector)
in_gpd['contourLength'] = np.power(in_gpd['areaSqm']/np.pi,0.5)
in_gpd.to_file(invector)

# (12) calculate centroid lat/lon (degree)
def getXY(pt):
    return (pt.x, pt.y)
in_gpd         = gpd.read_file(invector)
in_gpd_wgs84   = in_gpd.copy()
in_gpd_wgs84   = in_gpd_wgs84.to_crs(epsg=4326) #"EPSG:4326"
centroidseries = in_gpd_wgs84['geometry'].centroid
in_gpd['longitude'],in_gpd['latitude'] = [list(t) for t in zip(*map(getXY, centroidseries))]
in_gpd.to_file(invector)

print('added zonal statistics to GRU shapefile')


# In[ ]:




