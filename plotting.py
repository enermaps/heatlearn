#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot tiles.

Created on Thu Feb  4 09:13:07 2021

@author: giuseppeperonato
"""

import matplotlib.pyplot as plt
import matplotlib
import geopandas as gpd
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import numpy as np
from shapely import wkt
from shapely.geometry import box

pd.options.mode.chained_assignment = 'warn'
matplotlib.rcParams.update({'font.size': 12})
def tileshow(fishnet, image,grid_type, tile_type, tile_size, tile_id, path=""):
    """
    Plot tiles with proper formatting.

    Parameters
    ----------
    fishnet : DataFrame
        tiles
    image : str
        "esm", "height", "idc", "mask"
    grid_type : str
        "statbl" or "hotmaps".
    tile_type : str
        "contiguous" or "overlapped".
    tile_size : int
        size of tile in m.
    tile_id : int
        ID of the tile.
    path : str
        Optional path for saving fig.
    Returns
    -------
    None.

    """
    fishnet = pd.read_csv("data/{}_{}_{}.csv".format(grid_type,tile_size,tile_type),index_col=0)
    fishnet['geometry'] = fishnet['geometry'].apply(wkt.loads)
    fishnet.crs = "EPSG:2056"
    fishnet = gpd.GeoDataFrame(fishnet)

    if image == "esm":
        base_folder = "data/200km_2p5m_N26E38"
    elif image == "height" or image =="footprints":
        base_folder = "data/GML_CAD_BATIMENT_HORSOL"
    elif image == "mask":
        base_folder = "data/GDB_SCANE_INDICE_MOYENNES_3_ANS"

    
    
    if image != "idc":
        matrix = plt.imread("{}/tiles_{}_{}/tiles_{}/{}.tif".format(base_folder,
                                                           grid_type,
                                                           tile_type,
                                                           tile_size,
                                                           tile_id))
    
    if image == "esm":
        col_dict={1:"#70a2ff", #water
                  2:"#666666",#railways
                  10:"#f2f2f2",#NBU Area - Open Space
                  20:"#dde6cf",#NBU Area - Green ndvix
                  30:"#e1e1e1",#BU Area - Open Space
                  40:"#b5cc8e",#BU Area - Green ndvix
                  41:"#c8e6a1",#BU Area - Green Urban Atlas
                  50:"#807d79",#BU Area - Built-up
                  }
        
        labels = ["Water",
                  "Railways",
                  "Non-built - Open Space",
                  "Non-built - Green ndvix",
                  "Built - Open Space",
                  "Built - Green ndvix",
                  "Built - Green Urban Atlas",
                  "Built - Built-up",
                  ]
    elif image == "mask":
        col_dict={0:"grey",  
                  1:"yellow",
                  }
        
        labels = ["",
                  r"$\neg$ IDC",
                  ]
    if image == "mask" or image =="esm":
        # Plotting from https://stackoverflow.com/a/60870122
        # We create a colormar from our list of colors
        cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
        
        len_lab = len(labels)
        
        # prepare normalizer
        ## Prepare bins for the normalizer
        norm_bins = np.sort([*col_dict.keys()]) + 0.5
        norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
        
        ## Make normalizer and formatter
        norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
        fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
        
        # Plot our figure
        fig,ax = plt.subplots()
        im = ax.imshow(matrix, cmap=cm, norm=norm)
        
        # Create a Rectangle patch
        if matrix.shape[0] > 40:
            rect = patches.Rectangle((matrix.shape[0]/2-20, matrix.shape[0]/2-20), 40, 40, linewidth=1, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
        
        diff = norm_bins[1:] - norm_bins[:-1]
        tickz = norm_bins[:-1] + diff / 2
        cb = fig.colorbar(im, format=fmt, ticks=tickz, fraction=0.0458, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

        
    elif image == "height":
        # Plot our figure
        fig,ax = plt.subplots()
        im = ax.imshow(matrix)
        cb = fig.colorbar(im)
        cb.set_label('Height (m)') 
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

        
    elif image == "footprints":
        # Plot our figure
        fig,ax = plt.subplots()
        im = ax.imshow(matrix==0,cmap="gray")
        # Create a Rectangle patch
        if matrix.shape[0] > 40:
            rect = patches.Rectangle((matrix.shape[0]/2-20, matrix.shape[0]/2-20), 40, 40, linewidth=1, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
        
    elif image == "idc":
        bounds = fishnet.loc[tile_id,:].geometry.bounds
        boundary = box(*bounds)
        bldgs = gpd.read_file("data/GML_CAD_BATIMENT_HORSOL/CAD_BATIMENT_HORSOL.gml")
        idc = gpd.read_file("data/GDB_SCANE_INDICE_MOYENNES_3_ANS/SCANE_INDICE_MOYENNES_3_ANS.gdb")
        idc = idc.loc[idc.ANNEE == 2016,:]
        fig,ax = plt.subplots()
        bldgs_intile = bldgs[bldgs.intersects(boundary)]
        idc_intile = idc[idc.intersects(boundary)]
        bldgs_intile.plot(color="grey",ax=ax)
        idc_intile["INDICE"] /= 3.6 # to KWh
        idc_intile.plot(column="INDICE",legend=True, ax=ax, legend_kwds={"label": "kWh/m$^2$"})
        geo_size = bounds[2]-bounds[0]
        if geo_size > 100:
            rect = patches.Rectangle((boundary.centroid.coords[0][0]-50, boundary.centroid.coords[0][1]-50), 100, 100, linewidth=1, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
        plt.xlim((bounds[0],bounds[2]))
        plt.ylim((bounds[1],bounds[3]))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
        
    if len(path)> 0:
        fig.tight_layout()
        fig.savefig(path)

def getLatLong(tile_id, fishnet):
    fishnet_4326 = fishnet.to_crs("EPSG:4326")
    tile = fishnet_4326.loc[tile_id]
    print("Lat, long of tile {}".format(tile.name))
    print(tile.geometry.centroid.coords[0][1],tile.geometry.centroid.coords[0][0])
    

if __name__ == "__main__":  
    
    print("Use this script to plot tiles.")
