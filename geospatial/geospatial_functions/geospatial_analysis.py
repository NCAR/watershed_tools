#!/usr/bin/env python
# coding: utf-8

"""
Created on Fri Oct 16 09:35:11 2020

@author: hongli

Note: in python, there are two types of nodata mask for raster files.
The first is in the format of GDAL. 0 is invalid region, 255 is valid region. 
The second is in the foramt of Numpy masked array. True is invalid region, False is valid region.
When read mask use ff.read_masks(1), it by default returns mask in the first format.
(reference: https://rasterio.readthedocs.io/en/latest/topics/masks.html)
"""

import os, fiona
import numpy as np
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
import fiona.crs 
import rasterio.mask
import pandas as pd
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt 

def reproject_raster(inraster, outraster, dst_crs, resampling_method):
    '''inraster: input, raster, the raster to be re-projected.
    outraster: output, raster, path of output raster.
    dst_crs: input, proj, destinate/output crs. 
    resampling_method: input, rasterio.warp.Resampling method. 
    # resampling_method must be one of the followings:
    # Resampling.nearest, Resampling.bilinear, Resampling.cubic, Resampling.cubic_spline, Resampling.lanczos, Resampling.average, Resampling.mode, 
    # Resampling.max (GDAL >= 2.2), Resampling.min (GDAL >= 2.2), Resampling.med (GDAL >= 2.2), Resampling.q1 (GDAL >= 2.2), Resampling.q3 (GDAL >= 2.2)
    # reference: https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/reproject-raster/  
    # reference: https://rasterio.readthedocs.io/en/latest/api/rasterio.warp.html'''

    with rio.open(inraster) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rio.open(outraster, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling_method)
    return
                
def reproject_vector(invector, outvector, dst_crs):
    '''invector: input, vector, the vector to be re-projected.
    outvector: output, vector, path of output vector.
    dst_crs: input, proj, output crs. '''
    # reference:https://www.earthdatascience.org/courses/use-data-open-source-python/intro-vector-data-python/vector-data-processing/reproject-vector-data-in-python/
    in_gdf = gpd.read_file(invector)            # read vector file
    in_gdf_prj = in_gdf.to_crs(dst_crs)         # convert projection   
    in_gdf_prj.to_file(outvector)               # save projected geodataframe
    return 
    
def rasterize_gru_vector(invector,infield,gruName_field,gruNo_field,gruNo_field_dtype,refraster,outraster,gru_corr_txt):
    # reference: https://gis.stackexchange.com/questions/151339/rasterize-a-shapefile-with-geopandas-or-fiona-python
    '''
    invector: input, vector, the vector to be rasterized.
    infield: input, str, attribute field of input vector that is used as a burn-in value.
    gruName_field: input, str, gru name field, add to the updated invector.
    gruNo_field: input, str, field name of the gru number column, e.g.,1,2,3....
    gruNo_field_dtype: input, str, data type of gruNo_field, used to specify dtype of output gru raster.
    refraster: input, raster, reference raster to get meta. 
    outraster: output, raster, path of output raster.
    gru_corr_txt: output, txt, gruNo-gruName correspondence relationship.    '''
    
    # open input vector
    in_gdf = gpd.read_file(invector)  
    in_gdf = in_gdf.sort_values(by=[infield])

    # add a gruName_field to gdf
    in_gdf[gruName_field] = in_gdf[infield]
    in_gdf[gruName_field] = pd.to_numeric(in_gdf[gruName_field], errors='coerce')

    # add a gruNo_field to gdf
    in_gdf[gruNo_field] = np.arange(1,len(in_gdf)+1)
    in_gdf[gruNo_field] = pd.to_numeric(in_gdf[gruNo_field], errors='coerce')

    # save the correspondence between gruName and gruNo
    in_gdf[[gruNo_field,gruName_field]].to_csv(gru_corr_txt, sep=',', index=False)    

    # copy and update the metadata from the input raster for the output
    # avaialble dtypes for rasterio: 'int16', 'int32', 'float32', 'float64' 
    # reference: https://test2.biogeo.ucdavis.edu/rasterio/_modules/rasterio/dtypes.html
    with rio.open(refraster) as src:                 
        ref_mask = src.read_masks(1)
        meta = src.meta.copy()
        nodatavals = src.nodatavals
    meta.update(count=1, dtype=gruNo_field_dtype, compress='lzw')

    with rio.open(outraster, 'w+', **meta) as out:
        out_arr = out.read(1)

        # burn the features into the raster and write it out
        # this is where we create a generator of geom, value pairs to use in rasterizing
        shapes = ((geom,value) for geom, value in zip(in_gdf['geometry'], in_gdf[gruNo_field]))  
        # Areas not covered by input geometries are replaced with an optional fill value, which defaults to 0.
        burned = rio.features.rasterize(shapes=shapes, out=out_arr, fill=nodatavals, transform=out.transform)
        burned_ma = np.ma.masked_array(burned, ref_mask==0)
        out.write(burned_ma,1) 
    
    # save updated gru shapefile
    in_gdf.to_file(invector) 
    return

def rasterize_vector(invector,infield,infield_dtype,refraster,outraster):
    # reference: https://gis.stackexchange.com/questions/151339/rasterize-a-shapefile-with-geopandas-or-fiona-python
    '''
    invector: input, vector, the vector to be rasterized.
    infield: input, string. field of input vector to use as a burn-in value.
    infield_dtype: input, string. data type of infield, used to rasterize invector.
    Note: Avaialble dtypes for rasterio: 'int16', 'int32', 'float32', 'float64'(reference: https://test2.biogeo.ucdavis.edu/rasterio/_modules/rasterio/dtypes.html)
    refraster: input, raster, reference raster to get meta. 
    outraster: output, raster, path of output raster.    '''
    
    # open input vector
    in_gdf = gpd.read_file(invector)    
        
    # copy and update the metadata from the refraster for the output
    with rio.open(refraster) as src:                 
        ref_mask = src.read_masks(1)
        meta = src.meta.copy()
        nodatavals = src.nodatavals
    meta.update(count=1, dtype=infield_dtype, compress='lzw') 
    
    with rio.open(outraster, 'w+', **meta) as out:
        out_arr = out.read(1)
    
        # burn the features into the raster and write it out
        # this is where we create a generator of geom, value pairs to use in rasterizing
        shapes = ((geom,value) for geom, value in zip(in_gdf['geometry'], in_gdf[infield]))  
        # Areas not covered by input geometries are replaced with an fill value.
        # Note that this process may induce pixels missing if the pixels' center is not within the polygon.
        # reference: https://rasterio.readthedocs.io/en/latest/api/rasterio.features.html
        burned = rio.features.rasterize(shapes=shapes, out=out_arr, fill=nodatavals, transform=out.transform)
        burned_ma = np.ma.masked_array(burned, ref_mask==0)
        out.write(burned_ma,1)    
    return

def crop_raster(inraster,invector,outraster):
    '''
    inraster: input, raster, rater to be cropped.
    invector: input, vector, provide crop extent.
    outraster: output, raster, output raster after crop.'''
    
    # open crop extent
    in_gdf = gpd.read_file(invector) 
    shapes = in_gdf['geometry']    
    
    # crop raster with the invector extent
    with rio.open(inraster) as src:
        if src.crs == in_gdf.crs:     # check projection consistency
            out_arr, out_transform = rio.mask.mask(src, shapes, crop=True)
            out_meta = src.meta.copy()
        else:
            print('Crop failed because input raster and vector are in different projections.')
     
    # update the new cropped meta info
    out_meta.update({"driver": "GTiff",
                     "height": out_arr.shape[1],
                     "width": out_arr.shape[2],
                     "transform": out_transform})
    
    # write cropped raster data
    with rio.open(outraster, 'w', **out_meta) as outf:
        out_arr_ma = np.ma.masked_array(out_arr)
        outf.write(out_arr_ma)            
    return

def calculate_slope_and_aspect(dem_raster,slope_raster,aspect_raster):
    '''dem_raster: input, DEM raster.
    slope_raster: output, slope raster.
    aspect_raster: output, aspect raster.'''
    # reference: https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-slope-works.htm
    # reference: https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-aspect-works.htm

    # read dem input
    with rio.open(dem_raster) as ff:
        dem  = ff.read(1)
        mask = ff.read_masks(1)
        out_meta = ff.meta.copy()
        nodatavals = ff.nodatavals
        pixelSizeX, pixelSizeY  = ff.res

    # # calculate slope and aspect
    slope = dem.copy()
    aspect = dem.copy()
    (nx,ny) = np.shape(dem)
    # X is latitude, Y is longitude.
    #                             j(ny)
    #     |------------------------->
    #     |            North
    #     |        ------------
    #     |         a | b | c
    #     |        ------------
    #     |  West   d | e | f    East
    #     |        ------------
    #     |         g | h | i
    #  i  |        ------------
    # (nx)↓           South

    for i in range(nx): # latitude (north -> south)
        for j in range(ny): # longitude (west -> east)
            if (dem[i,j] != nodatavals):
                # neighbor index
                isouth = i+1
                inorth = i-1
                jwest = j-1
                jeast = j+1

                if (inorth < 0):
                    inorth = 0
                if (isouth > nx-1):
                    isouth = nx-1
                if (jwest < 0):
                    jwest = 0
                if (jeast > ny-1):
                    jeast = ny-1

                # neighbor elevation            
                dem_a = dem[inorth,jwest]
                dem_b = dem[inorth,j]
                dem_c = dem[inorth,jeast]
                dem_d = dem[i,jwest]
                dem_e = dem[i,j]
                dem_f = dem[i,jeast]
                dem_g = dem[isouth,jwest]
                dem_h = dem[isouth,j]
                dem_i = dem[isouth,jeast]

                # dz/dx and dz/dy
                dzdx = ((dem_c+2*dem_f+dem_i) - (dem_a+2*dem_d+dem_g))/(8*pixelSizeX)
                dzdy = ((dem_g+2*dem_h+dem_i) - (dem_a+2*dem_b+dem_c))/(8*pixelSizeY)

                # slope in rise_run (radian)
                rise_run = (dzdx**2 + dzdy**2)**0.5            
                # slope in degree [0, 90]
                slope[i,j] = np.degrees(np.arctan(rise_run))

                # aspect in degree [-180, 180]
                aspect0 = np.degrees(np.arctan2(dzdy,-dzdx))
                # convert to compass direction values (0-360 degrees)
                if aspect0 < 0:
                    cell = 90.0 - aspect0
                elif aspect0 > 90.0:
                    cell = 360.0 - aspect0 + 90.0
                else:
                    cell = 90.0 - aspect0
                aspect[i,j] = cell
    
    # let aspect = 0 for flat grids (slope=0).
    aspect[slope==0.0] = 0
    
    # save slope and aspect into raster
    with rio.open(slope_raster, 'w', **out_meta) as outf:
        slope_ma = np.ma.masked_array(slope, mask==0)
        outf.write(slope_ma,1)       
    with rio.open(aspect_raster, 'w', **out_meta) as outf:
        aspect_ma = np.ma.masked_array(aspect, mask==0)
        outf.write(aspect_ma,1)  
    return


def resample_raster(inraster,refraster,outraster):
    '''
    inraster: input, raster, rater to be cropped.
    refraster: input, raster, provide reference of array shape.
    outraster: output, raster, output raster after crop.'''
    
    # read refraster to get band size
    with rio.open(refraster) as ff:
        ref_crs = ff.crs
        ref_mask = ff.read_masks(1)
        ref_height, ref_width = ff.height, ff.width
        
    # method 1. use rasterio
    with rio.open(inraster) as ff:
        data  = ff.read(1)
        out_meta = ff.meta.copy()

        # reference: https://rasterio.readthedocs.io/en/stable/topics/resampling.html
        with rio.open(inraster) as ff:
            # resample data to target shape
            data = ff.read(
                out_shape=(int(ref_height),int(ref_width)),
                resampling=rio.enums.Resampling.nearest)

            # scale image transform
            transform = ff.transform * ff.transform.scale(
                (ff.width / data.shape[-1]),
                (ff.height / data.shape[-2]))

    # write resampled raster data
    out_meta.update(transform=transform, driver='GTiff', 
                    height=ref_height, width=ref_width, crs=ref_crs)

    with rio.open(outraster, 'w', **out_meta) as outf:
        out_arr_ma = np.ma.masked_array(data, ref_mask==0)
        outf.write(out_arr_ma) 

    # method 2. use gdal 
    # reference: https://gis.stackexchange.com/questions/271226/up-sampling-increasing-resolution-raster-image-using-gdal
    return

def resample_raster_scale(inraster,scale_factor,outraster):
    # This function is used when the scale_factor is provided by users. 
    # The function has not been used in this project yet, but it's put here in case it can be used later.
    with rio.open(inraster) as dataset:

        # resample data to target shape
        data = dataset.read(
            out_shape=(dataset.count,
                       int(dataset.height * scale_factor),
                       int(dataset.width * scale_factor)),
            resampling=Resampling.bilinear)

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2]))

        # write resampled raster data
        out_meta = dataset.meta.copy()
        out_meta.update(transform=transform, driver='GTiff', 
                        height=int(dataset.height * scale_factor),
                        width=int(dataset.width * scale_factor))

        with rio.open(outraster, 'w', **out_meta) as outf:
            out_arr_ma = np.ma.masked_array(data, data==dataset.nodatavals)
            outf.write(out_arr_ma) 
        return
    
    
def classify_raster(inraster, bound_raster, classif_trigger, bins, class_outraster, value_outraster):
    '''
    bound_raster: input, raster, boundary as mask to classify inraster per boundary unit (eg, GRU raster).
    inraster: input, raster, continuous raster file.
    classif_trigger: input, numeric, inraster value difference to trigger classification. 
        - When the raster value difference is beyond classif_trigger value, implement classification. 
        - Otherwise, do not implement the classification.
    bins: input, Three types:
        - If bins is an int, it defines the number of equal-width bins in the given range. 
        - If bins is a sequence, it defines a monotonically increasing array of bin edges, 
        including the rightmost edge, allowing for non-uniform bin widths.
        - If bins is a string, it defines the method used to calculate the bin edge sequence.
    class_outraster: output, raster, output raster of class.
    value_outraster: output, raster, output raster of mean value per class. '''
    
    # read inraster data
    with rio.open(inraster) as ff:
        data  = ff.read(1)
        data_mask = ff.read_masks(1)
        out_meta = ff.meta.copy()

    # read bound_raster and identify unique bounds
    with rio.open(bound_raster) as ff:
        bounds  = ff.read(1)
        bounds_mask = ff.read_masks(1)
    unique_bounds = np.unique(bounds[bounds_mask!=0])

    # define array in the same shape of data, and specify dtype!
    data_class = data.copy()
    data_value = data.copy()

    # reclassify raster 
    # (1) if 'bins' is a string
    if isinstance(bins, str):
        bin_name = bins

        if bin_name == 'median':
            # loop through bounds
            for bound in unique_bounds:
                smask = bounds == bound            
                # Bin the raster data based on the data median
                smin,smedian,smax=np.min(data[smask]), np.median(data[smask]), np.max(data[smask])
                # use smedian to do classification in two conditions: 
                # when not given classif_trigger;
                # when given classif_trigger, and smax-smin is larger than classif_trigger. 
#                 print('gru %d, raster value diff %f.'%(bound,smax-smin))
                if (classif_trigger is None) or \
                ((classif_trigger is not None) and ((smax-smin)>=classif_trigger)):
                    bins = [smin,smedian,smax]
                else:
                    bins = [smin,smax]
                (hist,bin_edges) = np.histogram(data[smask],bins=bins)            
                #Place the mean elev and elev band
                for ibin in np.arange(len(bin_edges)-1):
                    if ibin!= (len(bin_edges)-1-1):
                        smask = (bounds == bound) & (data >= bin_edges[ibin]) & (data < bin_edges[ibin+1])
                    else: 
                        # All but the last (righthand-most) bin is half-open.  
                        # reference: https://github.com/numpy/numpy/blob/v1.19.0/numpy/lib/histograms.py#L677-L928
                        smask = (bounds == bound) & (data >= bin_edges[ibin]) & (data <= bin_edges[ibin+1])
                    data_class[smask] = ibin+1  # ele_band starts from one, not zero.
                    data_value[smask] = np.mean(data[smask])     

    # (2) if 'bins' is an integer or a sequence
    elif (np.ndim(bins) == 0) or (np.ndim(bins) == 1): 
        # loop through bounds
        for bound in unique_bounds:
            smask = bounds == bound
            # Bin the raster data based on bins
            (hist,bin_edges) = np.histogram(data[smask],bins=bins)
            #Place the mean elev and elev band
            for ibin in np.arange(len(bin_edges)-1):
                if ibin!= (len(bin_edges)-1-1):
                    smask = (bounds == bound) & (data >= bin_edges[ibin]) & (data < bin_edges[ibin+1])
                else: 
                    smask = (bounds == bound) & (data >= bin_edges[ibin]) & (data <= bin_edges[ibin+1])
                data_class[smask] = ibin+1  # ele_band starts from one, not zero.
                data_value[smask] = np.mean(data[smask])     

    # save data_class_ma and data_value_ma into rasters
    # update data type of meta and save raster
    data_class = data_class.astype('int32') 
    out_meta.update(count=1, dtype='int32', compress='lzw')
    with rio.open(class_outraster, 'w', **out_meta) as outf:
        data_class_ma = np.ma.masked_array(data_class,data_mask==0)
        outf.write(data_class_ma, 1)    

    # update data type of meta and save raster
    data_value = data_value.astype('float64') 
    out_meta.update(count=1, dtype='float64', compress='lzw')
    with rio.open(value_outraster, 'w', **out_meta) as outf:
        data_value_ma = np.ma.masked_array(data_value,data_mask==0) 
        outf.write(data_value_ma, 1)
    return

def classify_landcover(lc_raster, lc_class_raster):
    '''
    lc_raster: input, raster, landcover raster file.
    lc_class_raster: output, raster, raster class file.'''
    
    # read landcover data
    with rio.open(lc_raster) as ff:
        lc  = ff.read(1)
        lc_mask = ff.read_masks(1)
        out_meta = ff.meta.copy()

    # define array and specify dtype!
    lc_class = lc.copy()

    # specify landcover class
    # lc_class 1: crop. 2: non-crop. 
    ## Crop class includes: 0 Evergreen needleleaf forests, 1 Evergreen broadleaf forests, 
    ## 2 Deciduous needleleaf forests, 3 Deciduous broadleaf forests, 4 Mixed forests, 5 Closed shrublands, 7 Woody savannas.  
    lc_class[(lc<=5) | (lc==7)]=1  # crop
    lc_class[(lc>5) & (lc!=7) & (lc!=255)]=2 # non-crop    

    # convert class dtype from float to int
    # avaialble dtypes for rasterio: 'int16', 'int32', 'float32', 'float64' 
    # reference: https://test2.biogeo.ucdavis.edu/rasterio/_modules/rasterio/dtypes.html
    lc_class = lc_class.astype('int32') 
    out_meta.update(count=1, dtype='int32', compress='lzw')
    
    # save data_class to raster
    with rio.open(lc_class_raster, 'w', **out_meta) as outf:
        lc_class_ma = np.ma.masked_array(lc_class,lc_mask==0)
        outf.write(lc_class_ma, 1)    
    return

def classify_aspect(aspect_raster, class_num, class_outraster):
    '''
    inraster: input, raster, continuous raster file.
    class_num: input, int, it defines the number of desired aspect classes. 
    class_outraster: output, raster, output raster of class.
    value_outraster: output, raster, output raster of mean value per class. '''
    
    # read aspect_raster
    with rio.open(aspect_raster) as ff:
        data  = ff.read(1)
        mask = ff.read_masks(1)
        nodatavals = ff.nodatavals
        out_meta = ff.meta.copy()
    data_ma = np.ma.masked_array(data, mask==0)

    # define array in the same shape of data, and specify dtype!
    data_class = data.copy()

    # reclassify aspect
    # reference: https://www.neonscience.org/resources/learning-hub/tutorials/classify-raster-thresholds-py
    if class_num == 4:
        data_class[np.where((data_ma==0))] = 0                    # flat
        data_class[np.where((data_ma>0) & (data_ma<=45))] = 1     # north
        data_class[np.where((data_ma>45) & (data_ma<=135))] = 2   # east
        data_class[np.where((data_ma>135) & (data_ma<=225))] = 3  # south
        data_class[np.where((data_ma>225) & (data_ma<=315))] = 4  # west
        data_class[np.where((data_ma>315) & (data_ma<=360))] = 1  # north
    elif class_num == 8:
        data_class[np.where((data_ma==0))] = 0                       # flat
        data_class[np.where((data_ma>0) & (data_ma<=22.5))] = 1      # north
        data_class[np.where((data_ma>22.5) & (data_ma<=67.5))] = 2   # northeast
        data_class[np.where((data_ma>67.5) & (data_ma<=112.5))] = 3  # east
        data_class[np.where((data_ma>112.5) & (data_ma<=157.5))] = 4 # southeast
        data_class[np.where((data_ma>157.5) & (data_ma<=202.5))] = 5 # south
        data_class[np.where((data_ma>202.5) & (data_ma<=247.5))] = 6 # southwest
        data_class[np.where((data_ma>247.5) & (data_ma<=292.5))] = 7 # west
        data_class[np.where((data_ma>292.5) & (data_ma<=337.5))] = 8 # northwest
        data_class[np.where((data_ma>337.5) & (data_ma<=360))] = 1   # north
    
    # convert class dtype from float to int
    data_class = data_class.astype('int32') 
    out_meta.update(count=1, dtype='int32', compress='lzw')
    
    # save data_class to raster
    with rio.open(class_outraster, 'w', **out_meta) as outf:
        data_class_ma = np.ma.masked_array(data_class,mask=0)
        outf.write(data_class_ma, 1)    
    return

def polygonize_raster(inraster, outvector, attrb_field, attrb_field_dtype):
    '''
    inraster: input, raster, source to convert to vector.
    outvector: output, vector, output vector.
    attrb_field: input, str, attribute field name of vector attribtue table to save raster values.
    attrb_field_dtype: input, str, type of the output attribute field. For example, int'''
    # reference: https://programtalk.com/vs2/?source=python/7910/rasterio/examples/rasterio_polygonize.py
    
    driver='ESRI Shapefile'     
    with rio.Env():
        
        # read raster
        with rio.open(inraster) as src:
            image = src.read(1)
            mask = src.read_masks(1)
            
        # extract shapes of raster features
        results = (
            {'properties': {attrb_field: v}, 'geometry': s}
            for i, (s, v) in enumerate(rio.features.shapes(image, mask=mask, transform=src.transform)))
        
        # write GeoJSON style geometry dict to shapefile
        with fiona.open(
                outvector, 'w',
                driver=driver,
                crs=fiona.crs.to_string(src.crs),
                schema={'properties': [(attrb_field, attrb_field_dtype)],
                        'geometry': 'Polygon'}) as dst:
            dst.writerecords(results)      
    return

# Create HRU by overlaying a list of provided input rasters.
def define_hru(raster_list, fieldname_list, gru_raster, gru_corr_txt, gruNo_field, gruName_field,
               outraster, outvector, hruNo_field, hruNo_field_dtype, hruName_field):
    '''
    raster_list: input, list, a list raster inputs that are used to define HRU.
    fieldname_list: input, str, a list of field names corresponding to raster_list.
    gru_raster: input, raster, gru input raster.
    gruNo_field: input, string, field name of the gru number column, e.g.,1,2,3... 
    gru_corr_txt: input, text, gruNo-HUC12 correspondence txt file.
    outraster: output, raster, HRU raster with integer value.
    outvector: output, raster, HRU vector with HRU_full and HRU_int fields. '''
    
    # remove output files if exist
    for file in [outraster,outvector]:
        if os.path.exists(file): 
            os.remove(file)
        
    # --- PART 1. rasterize HRU via raster overlay (array concatenation) ---
    raster_str_max_len_list = [] # recrod the max str lenght of input class str
    for iraster, raster in enumerate(raster_list):
        # read raster
        with rio.open(raster) as ff:
            raster_value  = ff.read(1)
            raster_mask = ff.read_masks(1)

        # convert raster value to str, and format str to the same length (i.e., the max str length) 
        raster_str = raster_value.astype(str)
        max_len = max(map(len,raster_str[raster_mask!=0]))
        raster_str_max_len_list.append(max_len)
        raster_str_fmt = np.char.zfill(raster_str,max_len)

        # concatenate element-wise two arrays to generate HRU
        if iraster == 0:
            hru_str_fmt = raster_str_fmt
            hru_mask = (raster_mask!=0)
        else:
            hru_str_fmt = np.char.add(hru_str_fmt, raster_str_fmt)
            hru_mask = (hru_mask & raster_mask)

    # assign the unique integer to unique HRU (HRU_str_fmt)
    # note: the defined numpy array dtype needs to be consistent with the available rasterio dytpes (e.g., int32,float32,float64).
    # note: rasterio dytpes doesn't have int64.
    hru_int = np.zeros(np.shape(hru_str_fmt), dtype=np.int32)
    unique_hrus_str = np.unique(hru_str_fmt[hru_mask!=0]) 
    for ihru, hru in enumerate(unique_hrus_str):
        hru_mask = hru_str_fmt == hru
        hru_int[hru_mask] = int(ihru)+1
    hru_int_ma = np.ma.masked_array(hru_int,hru_mask==0)

    # save hru_int_ma into raster based on gru_raster 
    with rio.open(gru_raster) as src:
        dst_meta = src.meta.copy()
    with rio.open(outraster, 'w', **dst_meta) as dst:
        dst.write(hru_int_ma, 1)

    # --- PART 2. polgonize HRU and add attributes ---
    # polygonize hru rater
    polygonize_raster(outraster, outvector, hruNo_field, hruNo_field_dtype)

    # dissolve and exclude HRU=0 blank area
    hru_gpd = gpd.read_file(outvector)
    # use buffer(0) to fix self-intersection issue
    hru_gpd['geometry'] = hru_gpd.geometry.buffer(0)
    
    hru_gpd_disv = hru_gpd.dissolve(by=hruNo_field)
    hru_gpd_disv = hru_gpd_disv.reset_index()
    hru_gpd_disv = hru_gpd_disv.loc[hru_gpd_disv.loc[:,hruNo_field] > 0]
    hru_gpd_disv = hru_gpd_disv.reset_index()

    # add other fields: HRU and detailed input raster classes
    # Note: since we want to name HRU as HUC12+hru# (xxxx01, xxxx02), 
    # we need to find out the string location of gru among the compelte hru str.
    gru_index = raster_list.index(gru_raster)
    gru_str_start = sum(raster_str_max_len_list[0:gru_index])
    gru_str_len = raster_str_max_len_list[gru_index]

    # read gruNo-HUC12 corresponding text file to get gruNo <-> HUC12.
    corr_df = pd.read_csv(gru_corr_txt, sep=",")

    # loop through HRUs to add attributes
    for irow, row in hru_gpd_disv.iterrows():

        # (a) add new field columns, define other needed variables.
        if irow == 0:
            hru_gpd_disv[hruName_field] = ""    # simplified HRU name, e.g., hruNo + HRU#.
            hru_gpd_disv[gruName_field] = ""    # HUC12 int
            for jfield, field in enumerate(fieldname_list): # other fields
                hru_gpd_disv[field] = ""          
            hru_num_per_gru = 0           # count the number of HRUs per gru
            gruNo_before = ''             # gruNo before this irow to update hru_num_per_gru
        else:
            gruNo_before = gruNo_current

        # (b) identify current gru ID (gruNo_current: 1,2,3...)
        hru_str = unique_hrus_str[row[hruNo_field]-1]
        gruNo_current = hru_str[gru_str_start:gru_str_start+gru_str_len] 

        # (c) update the number of HRUs whtin gruNo_current (hru_num_per_gru: 1,2)
        if gruNo_current == gruNo_before:  # if the current gruNo is the same as before, cumulate hru num.
            hru_num_per_gru = hru_num_per_gru+1
        else: # otherwise, reset hru num as zero.
            hru_num_per_gru = 0

        # (d) fill hruName_field = gruNo + hru_count in gruNo_current, and HUC12 field.   
        gruName = corr_df[corr_df[gruNo_field]==int(gruNo_current)].reset_index().loc[0,gruName_field]
        hru_gpd_disv.loc[irow,hruName_field] = str(gruName) + str(hru_num_per_gru+1).zfill(2) # two-digit HRU#
        hru_gpd_disv.loc[irow,gruName_field] = str(gruName) 

        # (e) fill other fields for class checks 
        for jfield, field in enumerate(fieldname_list):
            field_str_start = sum(raster_str_max_len_list[0:jfield])
            field_str_len = raster_str_max_len_list[jfield]
            hru_gpd_disv.at[irow,field] = int(hru_str[field_str_start:field_str_start+field_str_len])          
    
    # change dtypes of name and class fields to be int64
    hru_gpd_disv[hruName_field] = pd.to_numeric(hru_gpd_disv[hruName_field], errors='coerce')
    hru_gpd_disv[gruName_field] = pd.to_numeric(hru_gpd_disv[gruName_field], errors='coerce')
    for field in fieldname_list:   
        hru_gpd_disv[field] = pd.to_numeric(hru_gpd_disv[field], errors='coerce')
        
    # save gpd to shapefile
    hru_gpd_disv.to_file(outvector, index=False)
    return

# Eliminate small HRUs by merging with the most dominant HRU within the same GRU.
def eliminate_small_hrus_dominant(hru_vector, hru_area_thld, gruNo_field, gruName_field, 
                                  hruNo_field, hruNo_field_dtype, hruName_field, hruArea_field, 
                                  fieldname_list, refraster, hru_vector_disv, hru_raster_disv):
    '''
    hru_vector: input, HRU shapefile.
    hru_area_thld: input, number, small HRU area threthold below which the HRU will be merged with another HRU.
    gruNo_field: input, string, field name of the gru number column, e.g.,1,2,3... 
    gruName_field: input, string, field name of gru name, e.g., 100800120101. 
    hruNo_field: input, string, field name of the hru number column, e.g.,1,2,3...
    hruNo_field_dtype: input, string, data type of hruNo_field, used to save hru raster.
    hruName_field: input, string, field name of the hru name column, e.g., 10080012010101, 100800120102. 
    hruArea_field: input, string, field name of the HRU area, used in small HRU elimination.
    fieldname_list: input, string list, a list of filed names used in hru generation.
    refraster: input, raster, a reference raster to get meta when rasterizing outvector.
    outraster: output, raster, HRU raster with integer value.
    outvector: output, raster, HRU vector with HRU_full and HRU_int fields. '''

    in_gpd = gpd.read_file(hru_vector)
    in_gpd_disv = in_gpd.copy()

    # PART 1. eliminate small HRUs based on HRU area threshold
    grus = np.unique(in_gpd_disv[gruNo_field].values)
    pbar = tqdm(total=len(grus))
    for gru in grus:
        in_gpd_disv[in_gpd_disv[gruNo_field]==gru][hruArea_field]
        max_index = in_gpd_disv[in_gpd_disv[gruNo_field]==gru][hruArea_field].argmax()
        flt1 = (in_gpd_disv[gruNo_field]==gru)                 # filter 1: gru
        flt2 = (in_gpd_disv[hruArea_field]<=hru_area_thld)     # filter 2: HRU area    
        flt = (flt1 & flt2)
        # change attributes to the most dominant one's
        for field in fieldname_list:
            in_gpd_disv.at[flt,field]=in_gpd_disv.loc[max_index,field]
        in_gpd_disv.at[flt,hruName_field]=in_gpd_disv.loc[max_index,hruName_field]
        pbar.update(1)
    pbar.close()

    in_gpd_disv = in_gpd_disv.dissolve(by=hruName_field)
    in_gpd_disv = in_gpd_disv.reset_index() 
    in_gpd_disv[hruArea_field] = in_gpd_disv.area       

    # PART 2. update HRUId and HRUNo based on elimination result
    # identify the max number of HRUs among all grus and its integer length
    max_hru_num = [len(in_gpd_disv[in_gpd_disv[gruNo_field]==gru]) for gru in grus]
    max_hru_str_len = max([len(str(max(max_hru_num))),2]) # str_len min is 2 (e.g, 01,02,...) 
    # loop through HRU to update HRUNo and HRUId
    for irow, row in in_gpd_disv.iterrows():
        target_gru = in_gpd_disv.loc[irow,gruNo_field]
        cum_df = in_gpd_disv[0:irow+1]
        hru_num = len(cum_df[cum_df[gruNo_field]==target_gru])
        hru_num_str = str(hru_num).zfill(max_hru_str_len)
        # update two fields        
        in_gpd_disv.at[irow,hruNo_field]=irow+1    
        in_gpd_disv.at[irow,hruName_field]=int(str(in_gpd_disv.loc[irow,gruName_field])+hru_num_str)

    # change dtypes of No, Name and class fields to be int64
    for field in [hruNo_field,gruNo_field,hruName_field,gruName_field]:   
        in_gpd_disv[field] = pd.to_numeric(in_gpd_disv[field], errors='coerce')
    for field in fieldname_list:   
        in_gpd_disv[field] = pd.to_numeric(in_gpd_disv[field], errors='coerce')

    # drop index column
    if 'index' in in_gpd_disv.columns:
        in_gpd_disv = in_gpd_disv.drop(columns=['index'])
    in_gpd_disv.to_file(hru_vector_disv, index=False)

    # PART 3. convert hru_vector_disv to raster for zonal statistics
    rasterize_vector(hru_vector_disv,hruNo_field, hruNo_field_dtype,refraster,hru_raster_disv)
    return

# Eliminate small HRUs by merging with the largest neighbor HRU within the same GRU.
def eliminate_small_hrus_neighbor(hru_vector, hru_thld_type, hru_thld, gruNo_field, gruName_field, 
                                  hruNo_field, hruNo_field_dtype, hruName_field,hruArea_field, 
                                  fieldname_list, refraster, hru_vector_disv, hru_raster_disv):
    '''
    hru_vector: input, HRU shapefile.
    hru_area_thld: input, number, small HRU area threthold below which the HRU will be merged with another HRU.
    gruNo_field: input, string, field name of the gru number column, e.g.,1,2,3... 
    gruName_field: input, string, field name of gru name, e.g., 100800120101. 
    hruNo_field: input, string, field name of the hru number column, e.g.,1,2,3...
    hruNo_field_dtype: input, string, data type of hruNo_field, used to save hru raster.
    hruName_field: input, string, field name of the hru name column, e.g., 10080012010101, 100800120102. 
    hruArea_field: input, string, field name of the HRU area, used in small HRU elimination.
    fieldname_list: input, string list, a list of filed names used in hru generation.
    refraster: input, raster, a reference raster used to rasterize outvector.
    outraster: output, raster, HRU raster with integer value.
    outvector: output, raster, HRU vector with HRU_full and HRU_int fields. '''

    in_gpd = gpd.read_file(hru_vector)
    in_gpd_disv = in_gpd.copy()
    in_gpd_disv = in_gpd_disv.to_crs(in_gpd.crs)  # convert projection   

    # PART 1. eliminate small HRUs based on HRU area threshold
    grus = np.unique(in_gpd_disv[gruNo_field].values)
    pbar = tqdm(total=len(grus))
    for gru in grus:
        flt1 = (in_gpd_disv[gruNo_field]==gru)               # filter 1: gru
        gru_df = in_gpd_disv[flt1]                           # gru dataframe
        dom_hru_idx = gru_df[hruArea_field].idxmax()         # the most dominate HRU index in gru_df
        dom_hru = gru_df.loc[dom_hru_idx,hruName_field]      # the most dominate HRU hruName in gru_df
        
        # update hru_area_thld if hru_thld_type is a fraction
        if hru_thld_type == 'fraction':
            gru_area = gru_df[hruArea_field].sum()
            hru_area_thld = hru_thld*gru_area
        elif hru_thld_type == 'value':
            hru_area_thld = hru_thld        
        flt2 = (in_gpd_disv[hruArea_field]<=hru_area_thld)     # filter 2: HRU area    
        elim_df = in_gpd_disv[flt1 & flt2]                     # to-be eliminated HRU dataframe    

        row_num = len(gru_df)                                  # total number of HRUs within the gru
        elim_num = len(elim_df)                                # number of to-be eliminated HRUs
        iter_num = 0                                           # elimination iteration number 
        
        # elimination
        while row_num>1 and elim_num != 0 and iter_num<10:
#             print(('GRU=%s, Iteration=%d, Target=%d/%d HRUs.')%(gru,iter_num+1,elim_num,row_num))

            for irow, row in elim_df.iterrows():
                # identify the target HRU's name and its neighbours' index in gru_df
                target_hruName = elim_df.loc[irow,hruName_field]
                nbhds_idx = gru_df[gru_df.geometry.touches(row['geometry'])].index.tolist() 

                # update hruName_field depending on its neighbors
                if len(nbhds_idx)>0:
                    # when there are neightbors, take the attribute of the largest neighboring HRU.
                    larg_nbhd_idx= gru_df.loc[nbhds_idx,hruArea_field].idxmax()
                    for field in fieldname_list:
                        in_gpd_disv.at[in_gpd_disv[hruName_field]==target_hruName,field]=gru_df.loc[larg_nbhd_idx,field]
                    in_gpd_disv.at[in_gpd_disv[hruName_field]==target_hruName,hruName_field] = gru_df.loc[larg_nbhd_idx,hruName_field] 
                else:
                    # when there is no neightbor, take the attribute of the most dominant HRU within the gru. 
                    print('no neighbors')
                    for field in fieldname_list:
                        in_gpd_disv.at[in_gpd_disv[hruName_field]==target_hruName,field]=gru_df.loc[dom_hru_idx,field]
                    in_gpd_disv.at[in_gpd_disv[hruName_field]==target_hruName,hruName_field] = dom_hru 

            # dissolve in_gpd_disv based on hruName_field column and change hruName_field from index to column
            in_gpd_disv = in_gpd_disv.dissolve(by=hruName_field)
            in_gpd_disv = in_gpd_disv.reset_index() 

            # update HRU area and filters
            in_gpd_disv[hruArea_field] = in_gpd_disv.area       
            flt1 = (in_gpd_disv[gruNo_field]==gru) 
            flt2 = (in_gpd_disv[hruArea_field]<=hru_area_thld)
            gru_df = in_gpd_disv[flt1]
            elim_df = in_gpd_disv[flt1 & flt2] 
            dom_hru_idx = gru_df[hruArea_field].idxmax()   
            dom_hru = gru_df.loc[dom_hru_idx,hruName_field] 

            # udpdate counts
            row_num = len(gru_df) 
            elim_num = len(elim_df) 
            iter_num = iter_num+1 
        pbar.update(1)
    pbar.close()

    # PART 2. update HRUId and HRUNo based on elimination result
    # identify the max number of HRUs among all grus and its integer length
    max_hru_num = [len(in_gpd_disv[in_gpd_disv[gruNo_field]==gru]) for gru in grus]
    max_hru_str_len = max([len(str(max(max_hru_num))),2]) # str_len min is 2 (e.g, 01,02,...) 
    # loop through HRU to update HRUNo and HRUId
    for irow, row in in_gpd_disv.iterrows():
        target_gru = in_gpd_disv.loc[irow,gruNo_field]
        cum_df = in_gpd_disv[0:irow+1]
        hru_num = len(cum_df[cum_df[gruNo_field]==target_gru])
        hru_num_str = str(hru_num).zfill(max_hru_str_len)        
        # update two fields
        in_gpd_disv.at[irow,hruNo_field]=irow+1    
        in_gpd_disv.at[irow,hruName_field]=int(str(in_gpd_disv.loc[irow,gruName_field])+hru_num_str)

    # change dtypes of No, Name and class fields to be int64
    for field in [hruNo_field,gruNo_field,hruName_field,gruName_field]:   
        in_gpd_disv[field] = pd.to_numeric(in_gpd_disv[field], errors='coerce')
    for field in fieldname_list:   
        in_gpd_disv[field] = pd.to_numeric(in_gpd_disv[field], errors='coerce')

    if 'index' in in_gpd_disv.columns:
        in_gpd_disv = in_gpd_disv.drop(columns=['index'])
    in_gpd_disv.to_file(hru_vector_disv, index=False)

    # PART 3. convert hru_vector_disv to raster for zonal statistics
    rasterize_vector(hru_vector_disv,hruNo_field, hruNo_field_dtype,refraster,hru_raster_disv)
    return

# Calculate zonal statistics for a given input vector and raster attribute.
def zonal_statistic(attr_raster, invector, infield, infield_dtype, refraster, metric, out_raster, *args, **kwargs):
    '''
    attr_raster: input, raster. Raster to analyze.
    invector: input, vector. Vector with zones boundaries.
    infield: input, string. Field name of invector that is used to identify zone of invector.
    infield_dtype: input, string. data type of infield, used to rasterize invector.
    metric: input, string. metric to calcualte zonal statistics. Choose from 'mean', 'median', 'max', 'min'.
    out_raster: output, raster. Raster to save the zonal attribtue value of invector. 
    raster_band: optional input, integer. Number of raster band to analyze.
    output_column_prefix: optional input, str. Prefix for output fields.'''

    output_column_prefix = kwargs.get('output_column_prefix', 'DN')
    raster_band = kwargs.get('raster_band', 1)
        
    # --- PART 1. rasterize invector based on the relative resolution of attr_raster ---
    # read attr_raster and refraster resolutions
    with rio.open(attr_raster) as ff:
        attr_SizeX, attr_SizeY  = ff.res
    with rio.open(refraster) as ff:
        ref_SizeX, ref_SizeY  = ff.res
        
    # choose the raster that has a smaller resolution from attr_size and refraster.
    # This process is to avoid converting invector into a resolution that is larger than refraster.
    # Converting invector to a larger resolution will cause no-data metric value in certain grids of in_raster.
    if attr_SizeX <= ref_SizeX:
        refraster = attr_raster
        # no need to resample attribtue raster
    else:
        refraster = refraster 
        # need to resample attribute raster
        attr_raster_resample = os.path.join(os.path.dirname(attr_raster), os.path.basename(attr_raster).split('.')[0]+'_resample.tif')
        resample_raster(attr_raster,refraster,attr_raster_resample)
        # change attr_raster_resample as the new attr_raster for the following calculation
        attr_raster=attr_raster_resample
    # convert invector into the chosen refraster resolution
    in_raster = os.path.join(os.path.dirname(invector), os.path.basename(invector).split('.')[0]+'tmp.shp')    
    rasterize_vector(invector,infield,infield_dtype,refraster,in_raster)

    # --- PART 2. loop through HRU to get attribute value ---
    # read invector
    in_gpd = gpd.read_file(invector)
    with rio.open(in_raster) as ff:
        in_raster_value  = ff.read(1)
        in_raster_mask = ff.read_masks(1)
    # read new attr_raster 
    with rio.open(attr_raster) as ff:
        attr_raster_value  = ff.read(raster_band)
        out_meta = ff.meta.copy()

    # initialize the output column prefix 
    if output_column_prefix in in_gpd.columns:
        in_gpd = in_gpd.drop(columns=[output_column_prefix])
    in_gpd[output_column_prefix] = "" 

    # initialize the output array
    output_array = attr_raster_value.copy()
    
    # loop through invector polygons
    unique_field_value = np.unique(in_raster_value[in_raster_mask!=0])
    for field_value in unique_field_value:
        smask = in_raster_value == field_value         
        attr_smask = attr_raster_value[smask]
        if isinstance(metric, str):
            if metric == 'mean':
                zonal_value = np.nanmean(attr_smask)
            elif metric == 'mean_aspect':    
                # reference: https://www.reddit.com/r/gis/comments/hulxk2/calculating_mean_aspect_for_a_polygon/
                # reference: https://geo.libretexts.org/Courses/University_of_California_Davis/UCD_GEL_56_-_Introduction_to_Geophysics/Geophysics_is_everywhere_in_geology.../zz%3A_Back_Matter/Arctan_vs_Arctan2
                # Method: decompose the aspect angle into a 2D vector using sin and cos, average the vectors, and then recompose them into an angle with arctan2.
                # Note: arctan is the 2-quadrant inverse tangent, it takes only one input value x/y. Its result is in [-90,90].
                # Note: arctan2 is the 4-quadrant inverse tangent, it takes two input arguments x and y. Its result is in [-180,180].
                attr_smask = np.where(attr_smask==0.0,np.nan,attr_smask) # exclude flat grids (flat aspect=0)
                
                y = np.sin(np.radians(attr_smask)) # north/south vector (positive to N)
                x = np.cos(np.radians(attr_smask)) # west/east vector (positive to E)
                zonal_y = np.nanmean(y)
                zonal_x = np.nanmean(x)
                zonal_value = np.degrees(np.arctan2(zonal_y, zonal_x))
                if zonal_value<0:
                    zonal_value = 360.0+zonal_value
            elif metric == 'mode':
                zonal_value = stats.mode(attr_smask,nan_policy='omit')[0]

            in_gpd.loc[in_gpd[infield]==field_value, output_column_prefix] = zonal_value
            output_array[smask] = zonal_value
        else: 
            print("Invalid metric string. Please choose from 'mean', 'median', 'max', 'min'.")
    
    # change astype as the same as attr_raster dtype
    in_gpd[output_column_prefix] = in_gpd[output_column_prefix].astype(out_meta['dtype'])
    output_array=output_array.astype(out_meta['dtype'])
    
    # save vector
    in_gpd.to_file(invector)
    os.remove(in_raster)

    # save outraster
    with rio.open(out_raster, 'w', **out_meta) as outf:
        out_arr_ma = np.ma.masked_array(output_array, in_raster_mask==0)
        outf.write(out_arr_ma,1)             
    return

def plot_vector(invector, column):    
    in_gpd = gpd.read_file(invector)
    fig, ax = plt.subplots(figsize=(10, 6))
    in_gpd.plot(column=column, ax=ax)
    ax.set_axis_off()
    # plt.axis('equal')
    plt.show()   
    return
