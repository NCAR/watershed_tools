#!/usr/bin/env python
# coding: utf-8

"""
Created on Fri Oct 16 09:35:11 2020

@author: hongli

"""
import os
import numpy as np
import geopandas as gpd
import rasterio as rio
import rasterio.shutil
import rasterio.plot 
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Plot GRU boundary
def plot_gru_bound(gru_shp,stream_shp,wgs_crs,title,ofile):
    '''gru_shp: input, gru shapefile.
    stream_shp: input, streamline shapefile.
    wgs_crs: input, string, projection system for plot (eg, 'epsg:4326').
    title: input, string, figure title.
    ofile: output, path of the plot figure file.'''
    
    fig, ax = plt.subplots(figsize=(9,9*0.5),constrained_layout=True) 
    fig.suptitle(title, weight='bold') 

    # (1) plot gru
    sub_gpd = gpd.read_file(gru_shp)
    sub_gpd_prj = sub_gpd.to_crs(wgs_crs)
    sub_gpd_prj.geometry.boundary.plot(color=None,edgecolor='k',linewidth=1.0,ax=ax,label='GRU') 
    del sub_gpd, sub_gpd_prj

    # (2) plot stream
    stream_gpd = gpd.read_file(stream_shp)
    stream_gpd_prj = stream_gpd.to_crs(wgs_crs)
    stream_gpd_prj.plot(color='b', linewidth=1.0, ax=ax, label='Stream')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='best', framealpha=0.6, facecolor=None)
    fig.savefig(ofile, bbox_inches='tight',dpi=150)    

    plt.show()   
    return

# Plot GRU and HRU boundary with elevation as background
def plot_gru_hru_bound(gru_shp,hru_shp,stream_shp,dem_raster,wgs_crs,title,ofile):
    '''gru_shp: input, gru shapefile.
    hru_shp: input, hru shapefile.
    stream_shp: input, streamline shapefile.
    dem_raster: input, dem raster.
    wgs_crs: input, string, projection system for plot (eg, 'epsg:4326').
    title: input, string, figure title.
    ofile: output, path of the plot figure file.'''
    
    fig, ax = plt.subplots(figsize=(9,9*0.5),constrained_layout=True) 
    fig.suptitle(title, weight='bold') 

    # (1) plot hru
    hru_gpd = gpd.read_file(hru_shp)
    hru_gpd_prj = hru_gpd.to_crs(wgs_crs)
    hru_gpd_prj.geometry.boundary.plot(color=None,edgecolor='goldenrod',linewidth=1.0,ax=ax,label='HRU') 
    del hru_gpd, hru_gpd_prj

    # (2) plot gru
    sub_gpd = gpd.read_file(gru_shp)
    sub_gpd_prj = sub_gpd.to_crs(wgs_crs)
    sub_gpd_prj.geometry.boundary.plot(color=None,edgecolor='r',linewidth=1.0,ax=ax,label='GRU') 
    del sub_gpd, sub_gpd_prj

    # (3) plot stream
    stream_gpd = gpd.read_file(stream_shp)
    stream_gpd_prj = stream_gpd.to_crs(wgs_crs)
    stream_gpd_prj.plot(color='b', linewidth=1.0, ax=ax, label='Stream')

    # (4) plot DEM
    # - reproject raster by creating a VRT file, which is merely a ASCII txt file 
    # that just contains reference to the referred file. This is usefult o avoid duplicating raster files.
    # reference: https://geohackweek.github.io/raster/04-workingwithrasters/
    dem_vrt_file = os.path.join(os.path.dirname(dem_raster), 
                                os.path.basename(dem_raster).split('.')[0]+'_vrt.tif')
    with rio.open(dem_raster) as src:
        with rio.vrt.WarpedVRT(src, crs=wgs_crs, resampling=rio.enums.Resampling.nearest) as vrt:
            rio.shutil.copy(vrt, dem_vrt_file, driver='VRT')
    
    # - plot the reprojected DEM
    with rio.open(dem_vrt_file,nodata=np.nan) as src:
        data  = src.read(1)
        nodatavals = src.nodatavals
        data[data==nodatavals]=np.nan # reprojection caused nondata value

        # firstly, use ax.imshow so that we have something to map the colorbar to.
        image_hidden = ax.imshow(data, cmap='Greys')
        cbar = plt.colorbar(image_hidden, ax=ax, fraction=0.1,aspect=25)
        cbar.ax.set_ylabel('Elevation (m)')

        # secondly, use rasterio.plot.show to plot coordinate.
        dem_image = rasterio.plot.show(data,ax=ax,transform=src.transform,cmap='Greys') #'Greys' #'terrain'    

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='best', framealpha=0.6, facecolor=None)
    fig.savefig(ofile, bbox_inches='tight',dpi=150)    

    plt.show()   
    return

# Plot HRU
def plot_hru(level_str, hru_shp, gru_shp, stream_shp, wgs_crs, fieldname_list, cmap_str, input_dict,
             figsize,title,leg_loc,leg_bbox_to_anchor,leg_ncol,ofile):
    '''level_str: input, string, complexity level, used in title plot.
    hru_shp: input, hru shapefile.
    gru_shp: input, gru shapefile.
    stream_shp: input, streamline shapefile.
    wgs_crs: input string, projection system for plot (eg, 'epsg:4326').
    fieldname_list: input, a list of field names corresponding to the inputs that are used to define HRU (except GRU input).
    cmap_str: input string, color map. Can be Python built-in colormap (eg, 'jet'), or 'user'.
    input_dict: input dictionary. When cmap_str is 'user', user needs to define each HRU type value corresponding 
    color (for plot) and label (for legend) following the format: dict[hru_type]=list(color,label). For example, 
    input_dict = {1:["black", "Low Elev HRU"],
                  2:["white", "High Elev HRU"]}     
    When cmap_str is a Python built-in colormap (eg, 'jet'), user needs to define each hru type corresponding label (for legend)
    following the format: dict[hru_type]=label. For example,
    input_dict = {1: "Low Elev HRU",
                  2: 'High Elev HRU'}   
    figsize: input, tuple, figure size (eg, (9,9*0.75)).
    title: input, string, figure title.
    leg_loc: input, int, number of legend columns. 
    leg_bbox_to_anchor, legend location relative to 
    leg_ncol: input, int, number of legend columns. 
    ofile: output, output figure path.
    '''    
    # 1. identify HRU classes per gru
    hru_gpd = gpd.read_file(hru_shp)
    hru_gpd_prj = hru_gpd.to_crs(wgs_crs)
    hru_num = len(hru_gpd_prj)

    group_column = 'hru_type'
    hru_gpd_prj[group_column]=''
    for field in fieldname_list:
        hru_gpd_prj[group_column]=hru_gpd_prj[group_column]+hru_gpd_prj[field].astype('str')
    data_unique,data_counts= np.unique(hru_gpd_prj[group_column].values,return_counts=True) # unique values and counts

    # 2. create colormap, norm and legend (two options)
    # NOTE: use colormap only when number of HRUs > number of GRUs
    if len(fieldname_list)>0: 
        # method 1. use user-specified cmap
        if cmap_str!='user':
            step = 1/float(len(data_unique)-1)
            vals = np.arange(0,1+step,step)
            colors =  mpl.cm.get_cmap(cmap_str)
            cols = colors(vals)

            legend_labels = {}
            count_record = []
            for data_i in data_unique:
                data_i_color = cols[np.where(data_unique==data_i)][0]
                data_i_label = input_dict[data_i]
                legend_labels[data_i]=[data_i_color,data_i_label]
                count_record.append([data_i,data_i_label,int(data_counts[data_unique==data_i])])

        # method 2. use user-defined colors
        elif cmap_str=='user':

            legend_labels = {} # used to create legend
            count_record = []  # usde to record class count
            colors = []        # used to create cmap
            for data_i in data_unique:
                data_i_color = input_dict[data_i][0] #cols[np.where(unique==data_i)]
                data_i_label = input_dict[data_i][1]
                legend_labels[data_i]=[data_i_color,data_i_label]
                count_record.append([data_i,data_i_label,int(data_counts[data_unique==data_i])])
                colors.append(data_i_color)

    # 3 plot 
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig.suptitle(title, weight='bold') 

    # 3.1 plot HRU
    if len(fieldname_list)>0: 
        for ctype, data in hru_gpd_prj.groupby(group_column):
            color = legend_labels[ctype][0]
            label = legend_labels[ctype][1]     
            data.plot(color=color,ax=ax,alpha=0.8)    
    del hru_gpd, hru_gpd_prj

    # 3.2. plot basin boundary
    gru_gpd = gpd.read_file(gru_shp)
    gru_gpd_prj = gru_gpd.to_crs(wgs_crs)
    gru_num = len(gru_gpd_prj)
    gru_gpd_prj.geometry.boundary.plot(color=None,edgecolor='k',linewidth=1.5,ax=ax) 
    del gru_gpd, gru_gpd_prj

    # 3.3. plot streamline
    stream_gpd = gpd.read_file(stream_shp)
    stream_gpd_prj = stream_gpd.to_crs(wgs_crs)
    stream_gpd_prj.plot(color='darkblue', linewidth=1.5, ax=ax)
    del stream_gpd, stream_gpd_prj

    # 3.4. plot legend
    if len(fieldname_list)==0:
        patches = []
    else:
        patches = [Patch(color=legend_labels[key][0], label=legend_labels[key][1]) for key in legend_labels]

    basin_bound = mpl.patches.Patch(edgecolor='black', linewidth=1, fill=False, label='GRU')
    patches.append(basin_bound)

    stream_line = mpl.lines.Line2D([], [], color='darkblue', linewidth=1.5, marker=None, label='Streamline')
    patches.append(stream_line)
    plt.legend(handles=patches, bbox_to_anchor=leg_bbox_to_anchor, loc=leg_loc, ncol=leg_ncol, fancybox=True)

    # 3.5 plot figure title
    title_update = title+' at complexity level '+str(level_str)+'\n'+' (#GRUs='+str(gru_num)+'. #HRUs='+str(hru_num)+')'
    fig.suptitle(title_update, weight='bold') 

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.savefig(ofile, bbox_inches='tight',dpi=150)   
    plt.show()
    
    return

# Plot raster (eg, dem, aspect, radiation)
def plot_raster(inraster,wgs_crs,cmap_str,input_dict,
                figsize,title,leg_loc,leg_bbox_to_anchor,leg_ncol,ofile):
    '''inraster: input, raster to be plot.
    wgs_crs: input string, projection system for plot (eg, 'epsg:4326').
    cmap_str: input string, color map. Can be Python built-in colormap (eg, 'jet'), or 'user'.
    input_dict: input dictionary. When cmap_str is 'user', user needs to define each raster value corresponding 
    color (for plot) and label (for legend) following the format: dict[raster_value]=list(color,label). For example, for aspect:
    input_dict = {0:["black", "Flat (0)"],
                 1:["red", "North (337.5 - 22.5)"],
                 2:["orange", 'Northeast (22.5 - 67.5)'],
                 3:["yellow", 'East (67.5 - 112.5)'], 
                 4:["lime", 'Southeast (112.5 - 157.5)'], 
                 5:["cyan", 'South (157.5 - 202.5)'], 
                 6:["cornflowerblue", 'Southwest (202.5 - 247.5)'], 
                 7:["blue", 'West (247.5 - 292.5)'], 
                 8:["purple", 'Northwest (292.5 - 337.5)']}     
    When cmap_str is a Python built-in colormap (eg, 'jet'), user needs to define each raster corresponding label (for legend)
    following the format: dict[raster_value]=label. For example,for soil,
    input_dict = {0: "NoData",
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
    figsize: input, tuple, figure size (eg, (9,9*0.75)).
    title: input, string, figure title.
    leg_loc: input, int, number of legend columns. 
    leg_bbox_to_anchor, legend location relative to 
    leg_ncol: input, int, number of legend columns. 
    ofile: output, output figure path.
    '''
    ## Part 1. pre-process raster data, color, legend
    #  1. reproject raster by creating a VRT file, which is merely a ASCII txt file --- 
    # that just contains reference to the referred file. This is useful to avoid duplicating raster files.
    # reference: https://geohackweek.github.io/raster/04-workingwithrasters/
    raster_vrt_file = os.path.join(os.path.dirname(inraster), 
                                os.path.basename(inraster).split('.')[0]+'_vrt.tif')
    with rio.open(inraster) as src:
        with rio.vrt.WarpedVRT(src, crs=wgs_crs, resampling=rio.enums.Resampling.nearest) as vrt:
            rio.shutil.copy(vrt, raster_vrt_file, driver='VRT')

    #  2. read the reprojected raster 
    with rio.open(raster_vrt_file) as src:
        data  = src.read(1)
        mask = src.read_masks(1)
        nodatavals = src.nodatavals

    data_ma = np.ma.masked_array(data, mask==0)
    data_unique,data_counts= np.unique(data[data!=nodatavals],return_counts=True) # unique values and counts

    # 3. create colormap, norm and legend (two options)
    # method 1. use user-specified cmap
    if cmap_str!='user':
        step = 1/float(len(data_unique)-1)
        vals = np.arange(0,1+step,step)
        colors =  mpl.cm.get_cmap(cmap_str)
        cols = colors(vals)
        cmap = mpl.colors.ListedColormap(cols, int(data_unique.max())+1)

        legend_labels = {}
        count_record = []
        for data_i in data_unique:
            data_i_color = cols[np.where(data_unique==data_i)][0]
            data_i_label = input_dict[data_i]
            legend_labels[data_i]=[data_i_color,data_i_label]
            count_record.append([data_i,data_i_label,int(data_counts[data_unique==data_i])])

    # method 2. use user-defined colors
    elif cmap_str=='user':

        legend_labels = {} # used to create legend
        count_record = []  # usde to record class count
        colors = []        # used to create cmap
        for data_i in data_unique:
            data_i_color = input_dict[data_i][0] 
            data_i_label = input_dict[data_i][1]
            legend_labels[data_i]=[data_i_color,data_i_label]
            count_record.append([data_i,data_i_label,int(data_counts[data_unique==data_i])])
            colors.append(data_i_color)
        cmap = ListedColormap(colors, len(data_unique)) # generate your own cmap    
    print(legend_labels)

    ## Part 2. plot
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig.suptitle(title, weight='bold') 

    # 2.1. plot raster using rasterio.plot.show in order to show coordinate
    raster_image = rasterio.plot.show(data_ma,ax=ax,cmap=cmap,transform=src.transform)

    # 2.2. plot legend
    patches = [Patch(color=legend_labels[key][0], label=legend_labels[key][1]) for key in legend_labels]
    plt.legend(handles=patches, bbox_to_anchor=leg_bbox_to_anchor, loc=leg_loc, ncol=leg_ncol, fancybox=True)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.savefig(ofile, bbox_inches='tight',dpi=150)   
    plt.show()

    # 2.5. save count record to txt --- 
    count_ofile = os.path.join(os.path.dirname(ofile), os.path.basename(ofile).split('.')[0]+'.txt')
    count_sum = np.sum(data_counts)
    with open(count_ofile,'w') as f:
        f.write('#RasterValue,Label,Count,Proportion\n')
        for i in range(len(count_record)):
            f.write('%d,%s,%d,%.4f\n'%(count_record[i][0],count_record[i][1],count_record[i][2],
                                       count_record[i][2]/float(count_sum)))   
    return

# Plot raster (eg, dem, aspect, radiation) with basin boundary and streamline
def plot_raster_and_bound_stream(inraster,gru_shp,stream_shp,wgs_crs,cmap_str,input_dict,
                                 figsize,title,leg_loc,leg_bbox_to_anchor,leg_ncol,ofile):
    '''inraster: input, raster to be plot.
    gru_shp: input, vector with basin bound.
    stream_shp: input, vector of basin streamline.
    wgs_crs: input string, projection system for plot (eg, 'epsg:4326').
    cmap_str: input string, color map. Can be Python built-in colormap (eg, 'jet'), or 'user'.
    input_dict: input dictionary. When cmap_str is 'user', user needs to define each raster value corresponding 
    color (for plot) and label (for legend) following the format: dict[raster_value]=list(color,label). For example, for aspect:
    input_dict = {0:["black", "Flat (0)"],
                 1:["red", "North (337.5 - 22.5)"],
                 2:["orange", 'Northeast (22.5 - 67.5)'],
                 3:["yellow", 'East (67.5 - 112.5)'], 
                 4:["lime", 'Southeast (112.5 - 157.5)'], 
                 5:["cyan", 'South (157.5 - 202.5)'], 
                 6:["cornflowerblue", 'Southwest (202.5 - 247.5)'], 
                 7:["blue", 'West (247.5 - 292.5)'], 
                 8:["purple", 'Northwest (292.5 - 337.5)']}     
    When cmap_str is a Python built-in colormap (eg, 'jet'), user needs to define each raster corresponding label (for legend)
    following the format: dict[raster_value]=label. For example,for soil,
    input_dict = {0: "NoData",
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
    figsize: input, tuple, figure size (eg, (9,9*0.75)).
    title: input, string, figure title.
    leg_loc: input, int, number of legend columns. 
    leg_bbox_to_anchor, legend location relative to 
    leg_ncol: input, int, number of legend columns. 
    ofile: output, output figure path.
    '''
    ## Part 1. pre-process raster data, color, legend
    #  1. reproject raster by creating a VRT file, which is merely a ASCII txt file --- 
    # that just contains reference to the referred file. This is useful to avoid duplicating raster files.
    # reference: https://geohackweek.github.io/raster/04-workingwithrasters/
    raster_vrt_file = os.path.join(os.path.dirname(inraster), 
                                os.path.basename(inraster).split('.')[0]+'_vrt.tif')
    with rio.open(inraster) as src:
        with rio.vrt.WarpedVRT(src, crs=wgs_crs, resampling=rio.enums.Resampling.nearest) as vrt:
            rio.shutil.copy(vrt, raster_vrt_file, driver='VRT')

    #  2. read the reprojected raster 
    with rio.open(raster_vrt_file) as src:
        data  = src.read(1)
        mask = src.read_masks(1)
        nodatavals = src.nodatavals

    data_ma = np.ma.masked_array(data, mask==0)
    data_unique,data_counts= np.unique(data[data!=nodatavals],return_counts=True) # unique values and counts

    # 3. create colormap, norm and legend (two options)
    # method 1. use user-specified cmap
    if cmap_str!='user':
        vals = np.arange(len(data_unique)+1)/float(len(data_unique))
        colors =  mpl.cm.get_cmap(cmap_str)
        cols = colors(vals)
        cmap = mpl.colors.ListedColormap(cols, int(data_unique.max())+1)

        legend_labels = {}
        count_record = []
        for data_i in data_unique:
            data_i_color = cols[np.where(data_unique==data_i)][0]
            data_i_label = input_dict[data_i]
            legend_labels[data_i]=[data_i_color,data_i_label]
            count_record.append([data_i,data_i_label,int(data_counts[data_unique==data_i])])

    # method 2. use user-defined colors
    elif cmap_str=='user':

        legend_labels = {} # used to create legend
        count_record = []  # usde to record class count
        colors = []        # used to create cmap
        for data_i in data_unique:
            data_i_color = input_dict[data_i][0] #cols[np.where(unique==data_i)]
            data_i_label = input_dict[data_i][1]
            legend_labels[data_i]=[data_i_color,data_i_label]
            count_record.append([data_i,data_i_label,int(data_counts[data_unique==data_i])])
            colors.append(data_i_color)
        cmap = ListedColormap(colors, len(data_unique)) # Define the colors you want based on raster values    

    ## Part 2. plot
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig.suptitle(title, weight='bold') 

    # 2.1. plot raster using rasterio.plot.show in order to show coordinate
    raster_image = rasterio.plot.show(data_ma,ax=ax,cmap=cmap,transform=src.transform)

    # 2.2. plot basin boundary
    gru_gpd = gpd.read_file(gru_shp)
    gru_gpd_prj = gru_gpd.to_crs(wgs_crs)
    gru_gpd_prj['new_column'] = 0
    gpd_new = gru_gpd_prj.dissolve(by='new_column')
    gpd_new.boundary.plot(color=None,edgecolor='k',linewidth=1,ax=ax) 

    # 2.3. plot streamline
    stream_gpd = gpd.read_file(stream_shp)
    stream_gpd_prj = stream_gpd.to_crs(wgs_crs)
    stream_gpd_prj.plot(color='darkblue', linewidth=1.5, ax=ax)

    # 2.3. plot legend
    patches = [Patch(color=legend_labels[key][0], label=legend_labels[key][1]) for key in legend_labels]
    basin_bound = mpl.patches.Patch(edgecolor='black', linewidth=1, fill=False, label='Basin')
    patches.append(basin_bound)

    stream_line = mpl.lines.Line2D([], [], color='darkblue', linewidth=1.5, marker=None, label='Streamline')
    patches.append(stream_line)

    plt.legend(handles=patches, bbox_to_anchor=leg_bbox_to_anchor, loc=leg_loc, ncol=leg_ncol, fancybox=True)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.savefig(ofile, bbox_inches='tight',dpi=150)   
    plt.show()

    # 2.5. save count record to txt --- 
    count_ofile = os.path.join(os.path.dirname(ofile), os.path.basename(ofile).split('.')[0]+'.txt')
    count_sum = np.sum(data_counts)
    with open(count_ofile,'w') as f:
        f.write('#RasterValue,Label,Count,Proportion\n')
        for i in range(len(count_record)):
            f.write('%d,%s,%d,%.4f\n'%(count_record[i][0],count_record[i][1],count_record[i][2],
                                       count_record[i][2]/float(count_sum)))  
    return