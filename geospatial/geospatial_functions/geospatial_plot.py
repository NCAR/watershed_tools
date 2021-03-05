#!/usr/bin/env python
# coding: utf-8

"""
Created on Fri Oct 16 09:35:11 2020

@author: hongli

"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import rasterio.shutil
import rasterio.plot 
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def plot_sub_hru_bound(sub_vector,hru_vector,stream_vector,dem_raster,wgs_crs,title,ofile):
    
    fig, ax = plt.subplots(figsize=(9,9*0.5),constrained_layout=True) 
    fig.suptitle(title, weight='bold') 

    # (1) plot hru
    hru_gpd = gpd.read_file(hru_vector)
    hru_gpd_prj = hru_gpd.to_crs(wgs_crs)
    hru_gpd_prj.geometry.boundary.plot(color=None,edgecolor='goldenrod',linewidth=1.0,ax=ax,label='HRU') 
    del hru_gpd, hru_gpd_prj

    # (2) plot gru
    sub_gpd = gpd.read_file(sub_vector)
    sub_gpd_prj = sub_gpd.to_crs(wgs_crs)
    sub_gpd_prj.geometry.boundary.plot(color=None,edgecolor='r',linewidth=1.0,ax=ax,label='GRU') 
    del sub_gpd, sub_gpd_prj

    # (3) plot stream
    stream_gpd = gpd.read_file(stream_vector)
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
        data_mask = src.read_masks(1)
        data[data==-9999]=np.nan # reprojection caused nondata value

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

def plot_sub_bound(sub_vector,stream_vector,wgs_crs,title,ofile):
    
    fig, ax = plt.subplots(figsize=(9,9*0.5),constrained_layout=True) 
    fig.suptitle(title, weight='bold') 

    # (1) plot gru
    sub_gpd = gpd.read_file(sub_vector)
    sub_gpd_prj = sub_gpd.to_crs(wgs_crs)
    sub_gpd_prj.geometry.boundary.plot(color=None,edgecolor='k',linewidth=1.0,ax=ax,label='GRU') 
    del sub_gpd, sub_gpd_prj

    # (2) plot stream
    stream_gpd = gpd.read_file(stream_vector)
    stream_gpd_prj = stream_gpd.to_crs(wgs_crs)
    stream_gpd_prj.plot(color='b', linewidth=1.0, ax=ax, label='Stream')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='best', framealpha=0.6, facecolor=None)
    fig.savefig(ofile, bbox_inches='tight',dpi=150)    

    plt.show()   
    return

def plot_hru_level1_area_old(sub_vector,hru_vector,stream_vector,wgs_crs,title,ofile):

    fig, ax = plt.subplots(figsize=(9,9*0.5), constrained_layout=True)
    fig.suptitle(title, weight='bold') 

    # (1) plot hru
    hru_gpd = gpd.read_file(hru_vector)
    hru_gpd_prj = hru_gpd.to_crs(wgs_crs)

#     hruColors = {'1': 'lightsteelblue', '2': 'cornflowerblue'}
    hruColors = {'1': 'wheat', '2': 'goldenrod'}
    hruLables = {'1': 'Low HRU', '2': 'High HRU'}
    for ctype, data in hru_gpd_prj.groupby('elevClass'):
        label = hruLables[ctype]    
        color = hruColors[ctype]
        data.plot(color=color,ax=ax,alpha=0.8)    
    del hru_gpd, hru_gpd_prj

    # (2) plot subbasin
    sub_gpd = gpd.read_file(sub_vector)
    sub_gpd_prj = sub_gpd.to_crs(wgs_crs)
    sub_gpd_prj.geometry.boundary.plot(color=None,edgecolor='r',linewidth=1.0,ax=ax) 
    del sub_gpd, sub_gpd_prj

    # (3) plot stream
    instream_gpd = gpd.read_file(stream_vector)
    instream_gpd_prj = instream_gpd.to_crs(wgs_crs)
    stream_plt = instream_gpd_prj.plot(color='b', linewidth=1.0, ax=ax)
    del instream_gpd, instream_gpd_prj

    # (4) customize legend
    # reference: https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/custom_legends.html
    legend_elements = [Line2D([0], [0], color='b', lw=1, label='Stream'),
                       Line2D([0], [0], color='r', lw=1, label='GRU'),
                       Patch(facecolor='wheat', edgecolor=None, label='Low HRU'),
                       Patch(facecolor='goldenrod', edgecolor=None, label='High HRU')]
    ax.legend(handles=legend_elements, loc='lower left', framealpha=0.6, ncol=1, facecolor=None)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.savefig(ofile,bbox_inches='tight',dpi=150)    

    plt.show()   
    return

def plot_hru_level1_area(sub_vector,hru_vector,stream_vector,wgs_crs,cmap_str,
                         figsize,title,leg_loc,leg_bbox_to_anchor,leg_ncol,ofile):
    # 1. identify HRU classes per subbasin
    inhru_gpd = gpd.read_file(hru_vector)
    inhru_gpd_prj = inhru_gpd.to_crs(wgs_crs)
    inhru_num = len(inhru_gpd_prj)

    group_column = 'hru_types_per_sub'
    inhru_gpd_prj[group_column]=inhru_gpd_prj['elevClass'].astype('str')
    
    # 2. define HRU plot colors and legend labels
    colors =  mpl.cm.get_cmap(cmap_str,2)
#     hruColors = {'1': 'wheat', '2': 'goldenrod'}
    hruColors = {'1': colors(0), '2': colors(1)}
    hruLables = {'1': 'LowE HRU', '2': 'HighE HRU'}

    # 3. plot
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # 3.1 plot subbasin
    insub_gpd = gpd.read_file(sub_vector)
    insub_gpd_prj = insub_gpd.to_crs(wgs_crs)
    insub_num = len(insub_gpd_prj)
    insub_gpd_prj.geometry.boundary.plot(color=None,edgecolor='k',linewidth=1.5,ax=ax) 
    del insub_gpd, insub_gpd_prj

    # 3.2 plot stream
    instream_gpd = gpd.read_file(stream_vector)
    instream_gpd_prj = instream_gpd.to_crs(wgs_crs)
    stream_plt = instream_gpd_prj.plot(color='b', linewidth=1.5, ax=ax)
    del instream_gpd, instream_gpd_prj

    # 3.3 plot HRU
    for ctype, data in inhru_gpd_prj.groupby(group_column):
        label = hruLables[ctype]    
        color = hruColors[ctype]
        data.plot(color=color,ax=ax,alpha=0.8)    
    # inhru_gpd_prj.plot(column = 'elevClass', categorical=True,legend=True,ax=ax) 
    ## for quick plot with legend, but have to discard the subbasin and stream legends.
    ## becasue geopandas has a bug here. It cannot pass the subbasin polygon label to legend.
    # del inhru_gpd, inhru_gpd_prj

    # 3.4 plot figure title
    title_update = title+' at complexity level 1\n'+' (#GRUs='+str(insub_num)+'. #HRUs='+str(inhru_num)+')'
    fig.suptitle(title_update, weight='bold') 

    # 3.5 plot customized legend
    # reference: https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/custom_legends.html
    legend_elements = [Line2D([0], [0], color='b', lw=1.5, label='Stream'),
                       Patch(facecolor='white', edgecolor='k', label='GRU'),
                       Patch(facecolor=colors(0), label='LowE HRU'),
                       Patch(facecolor=colors(1),label='HighE HRU')]
    plt.legend(handles=legend_elements, bbox_to_anchor=leg_bbox_to_anchor, loc=leg_loc, ncol=leg_ncol, fancybox=True)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.savefig(ofile,bbox_inches='tight',dpi=150)    

    plt.show()  
    return
def plot_hru_level2_area(sub_vector,hru_vector,stream_vector,wgs_crs,cmap_str,
                         figsize,title,leg_loc,leg_bbox_to_anchor,leg_ncol,ofile):
    # 1. identify HRU classes per subbasin
    inhru_gpd = gpd.read_file(hru_vector)
    inhru_gpd_prj = inhru_gpd.to_crs(wgs_crs)
    inhru_num = len(inhru_gpd_prj)

    group_column = 'hru_types_per_sub'
    inhru_gpd_prj[group_column]=inhru_gpd_prj['elevClass'].astype('str')+inhru_gpd_prj['lcClass'].astype('str')
    
    # 2. define HRU plot colors and legend labels
    colors =  mpl.cm.get_cmap(cmap_str,4)
#     hruColors = {'11': 'lightgreen', '12': 'forestgreen', '21': 'wheat', '22': 'goldenrod'}
    hruColors = {'11': colors(0), '12': colors(1), '21': colors(2), '22': colors(3)}
    hruLables = {'11': 'LowE, C HRU', '12': 'LowE, NC HRU', '21': 'HighE, C HRU', '22': 'HighE, NC HRU'}

    # 3. plot
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # 3.1 plot subbasin
    insub_gpd = gpd.read_file(sub_vector)
    insub_gpd_prj = insub_gpd.to_crs(wgs_crs)
    insub_num = len(insub_gpd_prj)
    insub_gpd_prj.geometry.boundary.plot(color=None,edgecolor='k',linewidth=1.5,ax=ax) 
    del insub_gpd, insub_gpd_prj

    # 3.2 plot stream
    instream_gpd = gpd.read_file(stream_vector)
    instream_gpd_prj = instream_gpd.to_crs(wgs_crs)
    stream_plt = instream_gpd_prj.plot(color='b', linewidth=1.5, ax=ax)
    del instream_gpd, instream_gpd_prj

    # 3.3 plot HRU
    for ctype, data in inhru_gpd_prj.groupby(group_column):
        label = hruLables[ctype]    
        color = hruColors[ctype]
        data.plot(color=color,ax=ax,alpha=0.8)    
    # inhru_gpd_prj.plot(column = 'elevClass', categorical=True,legend=True,ax=ax) 
    ## for quick plot with legend, but have to discard the subbasin and stream legends.
    ## becasue geopandas has a bug here. It cannot pass the subbasin polygon label to legend.
    # del inhru_gpd, inhru_gpd_prj

    # 3.4 plot figure title
    title_update = title+' at complexity level 2\n'+' (#GRUs='+str(insub_num)+'. #HRUs='+str(inhru_num)+')'
    fig.suptitle(title_update, weight='bold') 

    # 3.5 plot customized legend
    # reference: https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/custom_legends.html
    legend_elements = [Line2D([0], [0], color='b', lw=1.5, label='Stream'),
                       Patch(facecolor='white', edgecolor='k', label='GRU'),
                       Patch(facecolor=colors(0), label='LowE, C HRU'),
                       Patch(facecolor=colors(1),label='LowE, NC HRU'),
                       Patch(facecolor=colors(2), label='HighE, C HRU'),
                       Patch(facecolor=colors(3),label='HighE, NC HRU')]
    plt.legend(handles=legend_elements, bbox_to_anchor=leg_bbox_to_anchor, loc=leg_loc, ncol=leg_ncol, fancybox=True)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.savefig(ofile,bbox_inches='tight',dpi=150)    

    plt.show()   
    return

def plot_hru_level3_area(sub_vector,hru_vector,stream_vector,wgs_crs,cmap_str,
                         figsize,title,leg_loc,leg_bbox_to_anchor,leg_ncol,ofile):
    
    # 1. identify HRU classes per subbasin
    inhru_gpd = gpd.read_file(hru_vector)
    inhru_gpd_prj = inhru_gpd.to_crs(wgs_crs)
    inhru_num = len(inhru_gpd_prj)

    group_column = 'hru_types_per_sub'
    inhru_gpd_prj[group_column]=inhru_gpd_prj['elevClass'].astype('str')+inhru_gpd_prj['radClass'].astype('str')+inhru_gpd_prj['lcClass'].astype('str')
    
    # 2. define HRU plot colors and legend labels
    colors =  mpl.cm.get_cmap(cmap_str,8)
#     hruColors = {'111': 'lightgreen', '112': 'forestgreen', '121': 'wheat', '122': 'goldenrod',
#                  '211': 'lightgreen', '212': 'forestgreen', '221': 'wheat', '222': 'goldenrod'}
    hruColors = {'111': colors(0), '112': colors(1), '121': colors(2), '122': colors(3),
                 '211': colors(4), '212': colors(5), '221': colors(6), '222': colors(7)}
    hruLables = {'111': 'Low elev, low rad, canopy HRU', '112': 'Low elev, low rad, non-canopy HRU', 
                 '121': 'Low elev, high rad, canopy HRU', '122': 'Low elev, high rad, non-canopy HRU',
                 '211': 'High elev, low rad, canopy HRU', '212': 'High elev, low rad, non-canopy HRU', 
                 '221': 'High elev, high rad, canopy HRU', '222': 'High elev, high rad, non-canopy HRU'}

    # 3. plot
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # 3.1 plot subbasin
    insub_gpd = gpd.read_file(sub_vector)
    insub_gpd_prj = insub_gpd.to_crs(wgs_crs)
    insub_num = len(insub_gpd_prj)
    insub_gpd_prj.geometry.boundary.plot(color=None,edgecolor='k',linewidth=1.5,ax=ax) 
    del insub_gpd, insub_gpd_prj

    # 3.2 plot stream
    instream_gpd = gpd.read_file(stream_vector)
    instream_gpd_prj = instream_gpd.to_crs(wgs_crs)
    stream_plt = instream_gpd_prj.plot(color='b', linewidth=1.5, ax=ax)
    del instream_gpd, instream_gpd_prj

    # 3.3 plot HRU
    for ctype, data in inhru_gpd_prj.groupby(group_column):
        label = hruLables[ctype]    
        color = hruColors[ctype]
        data.plot(color=color,ax=ax,alpha=0.8)    
    # inhru_gpd_prj.plot(column = 'elevClass', categorical=True,legend=True,ax=ax) 
    ## for quick plot with legend, but have to discard the subbasin and stream legends.
    ## becasue geopandas has a bug here. It cannot pass the subbasin polygon label to legend.
    # del inhru_gpd, inhru_gpd_prj

    # 3.4 plot figure title
    title_update = title+' at complexity level 3\n'+' (#GRUs='+str(insub_num)+'. #HRUs='+str(inhru_num)+')'
    fig.suptitle(title_update, weight='bold') 

    # 3.5 plot customized legend
    # reference: https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/custom_legends.html
    legend_elements = [Line2D([0], [0], color='b', lw=1.5, label='Stream'),
                       Patch(facecolor='white', edgecolor='k', label='GRU'),
                       Patch(facecolor=colors(0), label='LowE, LowR, C HRU'),
                       Patch(facecolor=colors(1),label='LowE, LowR, NC HRU'),
                       Patch(facecolor=colors(2), label='LowE, HighR, C HRU'),
                       Patch(facecolor=colors(3),label='LowE, HighR, NC HRU'),
                       Patch(facecolor=colors(4), label='HighE, LowR, C HRU'),
                       Patch(facecolor=colors(5),label='HighE, LowR, NC HRU'),
                       Patch(facecolor=colors(6), label='HighE, HighR, C HRU'),
                       Patch(facecolor=colors(7),label='HighE, HighR, NC HRU')]
    plt.legend(handles=legend_elements, bbox_to_anchor=leg_bbox_to_anchor, loc=leg_loc, ncol=leg_ncol, fancybox=True)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.savefig(ofile,bbox_inches='tight',dpi=150)    

    plt.show()   
    return

def plot_discrete_raster(inraster,bound_vector,wgs_crs,cmap_str,input_dict,
                         figsize,title,leg_loc,leg_bbox_to_anchor,leg_ncol,ofile):
    
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
    with rio.open(raster_vrt_file,nodata=np.nan) as src:
        data  = src.read(1)
        data_mask = src.read_masks(1)

        data = data.astype('float64') 
        data[data==255]=np.nan # nondata value
        data[data==-9999]=np.nan # reprojection caused nondata value
        data_unique,data_counts= np.unique(data[~np.isnan(data)],return_counts=True) # unique values and counts

    # 3. create colormap, norm and legend (two options)
    if cmap_str!='user':
        # method 1. use user-specified cmap
        vals = np.arange(int(data_unique.max()+1))/float(data_unique.max())
        colors =  mpl.cm.get_cmap(cmap_str)
        cols = colors(vals)
        cmap = mpl.colors.ListedColormap(cols, int(data_unique.max())+1)
        norm=mpl.colors.Normalize(vmin=data_unique.min()-0.5, vmax=data_unique.max()+0.5)

        legend_labels = {}
        count_record = []
        for data_i in data_unique:
            data_i_color = cols[np.where(data_unique==data_i)][0]
            data_i_label = input_dict[data_i]
            legend_labels[data_i]=[data_i_color,data_i_label]
            count_record.append([data_i,data_i_label,int(data_counts[data_unique==data_i])])
    elif cmap_str=='user':
        # method 2. use user-defined colors
        colors = []
        for key in input_dict:
            colors.append(input_dict[key][0])   
        cmap = ListedColormap(colors, len(data_unique)) # Define the colors you want based on raster values    
        norm = mpl.colors.Normalize(vmin=data_unique.min()-0.5, vmax=data_unique.max()+0.5)

        legend_labels = {}
        count_record = []
        for data_i in data_unique:
            data_i_color = input_dict[data_i][0] #cols[np.where(unique==data_i)]
            data_i_label = input_dict[data_i][1]
            legend_labels[data_i]=[data_i_color,data_i_label]
            count_record.append([data_i,data_i_label,int(data_counts[data_unique==data_i])])

    ## Part 2. plot
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig.suptitle(title, x=0.6, y=1.05, weight='bold') 

    #  2.1. plot basin boundary
    bound_gpd = gpd.read_file(bound_vector)
    bound_gpd_prj = bound_gpd.to_crs(wgs_crs)
    bound_gpd_prj['new_column'] = 0
    gpd_new = bound_gpd_prj.dissolve(by='new_column')
    gpd_new.geometry.boundary.plot(color=None,edgecolor='k',linewidth=0.5,ax=ax) 

    # 2.2. plot raster using imshow
    im = ax.imshow(data, cmap=cmap, norm=norm)
    cb = fig.colorbar(im, ax=ax,  norm=norm)
    cb.set_ticks(data_unique)

    # 2.3. plot legend
    patches = [Patch(color=legend_labels[key][0], label=legend_labels[key][1]) for key in legend_labels]
    plt.legend(handles=patches, bbox_to_anchor=leg_bbox_to_anchor, loc=leg_loc, ncol=leg_ncol, fancybox=True)

    # 2.4. plot raster using rasterio.plot.show in order to show coordinate
    raster_image = rasterio.plot.show(data,ax=ax,transform=src.transform,cmap=cmap)     

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