#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 14:16:34 2021

@author: giuseppeperonato
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Polygon
from shapely import wkt
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import rasterio
import rasterio.mask
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.plot import show
import numpy as np
import shutil
import os
from osgeo import gdal
from multiprocessing import  Pool

def load():
    print("Loading datasets")
    idc = gpd.read_file("data/GDB_SCANE_INDICE_MOYENNES_3_ANS/SCANE_INDICE_MOYENNES_3_ANS.gdb")
    bldg = gpd.read_file("data/GML_CAD_BATIMENT_HORSOL/CAD_BATIMENT_HORSOL.gml")
    
    idc = idc.loc[idc.ANNEE == 2016,:]
    bounds = idc.total_bounds
    bound = box(*bounds)
        
    hotmaps = gpd.GeoDataFrame(pd.read_csv("data/hotmaps_heat_density.csv",index_col=0))
    hotmaps['geometry'] = hotmaps['geometry'].apply(wkt.loads)
    hotmaps.crs = "EPSG:2056"
    
    hotmaps = hotmaps.loc[hotmaps.within(bound)]

    statbl = pd.read_csv("data/ag-b-00/GWS2019_HA.csv")
    statbl = gpd.GeoDataFrame(
    statbl,
    geometry=gpd.points_from_xy(statbl.E_KOORD+50, statbl.N_KOORD+50), #centered points
    crs="EPSG:2056")
    
    statbl = statbl.loc[statbl.within(bound)]
    
    cantons = gpd.read_file("data/BOUNDARIES_2021/DATEN/swissTLMRegio/SHAPEFILE_LV95/swissTLMRegio_KANTONSGEBIET_LV95.shp")
    ge = cantons[cantons.NAME=="Genève"].unary_union
    
    return statbl, bldg, idc, hotmaps, ge
   

def makeGrid(bounds, size):
    # https://gis.stackexchange.com/a/316460
    xmin,ymin,xmax,ymax = bounds

    wide = size
    length = size
    
    
    cols = list(range(int(np.floor(xmin)), int(np.floor(xmax)-(xmax-xmin)%size), wide))
    rows = list(range(int(np.floor(ymin)), int(np.floor(ymax)-(ymax-ymin)%size), length))
    
    decimal_x = xmin - int(xmin)
    decimal_y = ymin - int(ymin)
    
    # rows.reverse()

    polygons = []
    for x in cols:
        for y in rows:
            polygons.append( Polygon([(x+decimal_x, y+decimal_y),
                                      (x+wide+decimal_x, y+decimal_y),
                                      (x+decimal_x+wide, y+decimal_y+length),
                                      (x+decimal_x, y+decimal_y+length)]) )
    
    grid = gpd.GeoDataFrame({'geometry':polygons})
    return grid

def tiling(grid, size = 100, contiguous=True):
    # Create tiles from statistical 100x100 grid
    fishnet  = gpd.GeoDataFrame(grid.copy())
    bounds = fishnet.total_bounds
    bounds_off = [bounds[0]-100/2,bounds[1]-100/2,bounds[2]+100/2,bounds[3]+100/2]
    
    if size == 100 or not contiguous:
        fishnet["geometry"] = fishnet.apply(lambda x: box(minx=x.geometry.coords[0][0]-size/2,
                                                      miny=x.geometry.coords[0][1]-size/2,
                                                      maxx=x.geometry.coords[0][0]+size/2,
                                                      maxy=x.geometry.coords[0][1]+size/2)
                                        ,axis=1)
        fishnet = fishnet[["geometry"]]
        fishnet = gpd.GeoDataFrame(fishnet)
    else:
        fishnet = makeGrid(bounds_off, size)
        fishnet.crs = "EPSG:2056"
    return fishnet

def makeTiles(fishnet, bldg, idc,
              input_folder = "200km_2p5m_N26E38",
              input_file="200km_2p5m_N26E38_2056.TIF",
              output_folder="tiles",
              size = 40):
    if not os.path.exists("data/{}/{}".format(input_folder,output_folder)):
        os.mkdir("data/{}/{}".format(input_folder,output_folder))
    with rasterio.open("data/{}/{}".format(input_folder,input_file)) as src:
        esm, esm_transform = rasterio.mask.mask(src, [bound], crop=True)
        esm_meta = src.meta
        for i, row in fishnet.iterrows():
            tile, tile_transform = rasterio.mask.mask(src, [row.geometry], crop=True, all_touched=False)
            tile = np.squeeze(tile)
            tile = replace_with_dict(tile,ESM_dic)
            tile = tile.astype(np.uint8)
            with rasterio.open(
            'data/{}/{}/{}.tif'.format(input_folder,output_folder,i),
            'w',
            driver='GTiff',
            # hardcoded size, to be improved 
            height=size, 
            width=size,
            #
            count=1,
            dtype=rasterio.uint8,
            crs=esm_meta["crs"],
            transform=tile_transform,
            ) as dst:
                dst.write(tile, 1)
  
def makeTilesGDAL(fishnet, bldg, idc,
              input_folder = "200km_2p5m_N26E38",
              input_file="200km_2p5m_N26E38_2056.TIF",
              output_folder="tiles"):
    if not os.path.exists("data/{}/{}".format(input_folder,output_folder)):
        os.mkdir("data/{}/{}".format(input_folder,output_folder))
    src = gdal.Open("data/{}/{}".format(input_folder,input_file))
    print("data/{}/{}".format(input_folder,input_file))
    for i, row in fishnet.iterrows():
        ulx = row.geometry.bounds[0] 
        uly = row.geometry.bounds[3] 
        lrx = row.geometry.bounds[2]
        lry = row.geometry.bounds[1]
        gdal.Translate('data/{}/{}/{}.tif'.format(input_folder,output_folder,i),
                       src,
                       projWin=[ulx, uly, lrx, lry]) 
        

def makeMasks(fishnet, bldgs, idc, input_file="200km_2p5m_N26E38_2056.TIF",
              input_folder= "200km_2p5m_N26E38", output_folder="masks", pixels = 40):
    if not os.path.exists("data/{}/{}".format(input_folder,output_folder)):
        os.mkdir("data/{}/{}".format(input_folder,output_folder))
    for i, row in fishnet.iterrows():
        # if i == 49601166:
        bounds = row.geometry.bounds
        boundary = box(*bounds)
        bldgs_intile = bldgs[bldgs.intersects(boundary)]
        idc_intile = idc[idc.intersects(boundary)]
        
        not_idc = bldgs_intile.loc[~bldgs_intile.EGID.isin(idc_intile.EGID)] #without EGID
        # not_idc_intile = gpd.clip(not_idc.reset_index(), boundary)

        shape = pixels, pixels
        transform = rasterio.transform.from_bounds(*bounds, *shape)
        shapes = [(s, 1) for s in not_idc.geometry]

        if len(shapes) > 0:
            rasterized = rasterize(
                shapes,
                out_shape=shape,
                transform=transform,
                fill=0,
                all_touched=False,
                dtype=rasterio.uint8)
        # else:
        #     rasterized = np.zeros([pixels,pixels])
            with rasterio.open(
            'data/{}/{}/{}.tif'.format(input_folder,output_folder,i),
            'w',
            driver='GTiff',
            # hardcoded size, to be improved 
            height=pixels, 
            width=pixels,
            count=1,
            dtype=rasterio.uint8,
            crs="EPSG:2056",
            transform=transform,
            ) as dst:
                dst.write(rasterized, 1)
        
if __name__ == "__main__":
    # Choose parameters
    size = 300
    grid_type = "statbl" #hotmaps or statbl or grid50
    tile_type = "contiguous" #contiguous, overlapped
    n_cores = 10 #for multiprocessing stats
    makeTiles = True
    keep_only_idc = True
    #####################
    
    contiguous = False
    if tile_type == "contiguous":
        contiguous = True
    
    # Prepare data
    if not os.path.exists("data/200km_2p5m_N26E38/200km_2p5m_N26E38_2056.TIF"):
        from osgeo import gdal
        ds = gdal.Open("data/200km_2p5m_N26E38/200km_2p5m_N26E38.TIF")
        gdal.Warp("data/200km_2p5m_N26E38/200km_2p5m_N26E38_2056.TIF", ds, dstSRS="EPSG:2056")
        
    statbl, bldg, idc, hotmaps, ge = load()
    statbl = statbl.set_index("RELI")
    
    
    grids = {}
    grids["hotmaps"] = hotmaps
    grids["statbl"] = statbl
    grids["grid80"] = gpd.GeoDataFrame(geometry=makeGrid(idc.total_bounds, 80).centroid,crs="EPSG:2056")
    grids["grid50"] = gpd.GeoDataFrame(geometry=makeGrid(idc.total_bounds, 50).centroid,crs="EPSG:2056")
    
    for directory in ["200km_2p5m_N26E38","GML_CAD_BATIMENT_HORSOL","GDB_SCANE_INDICE_MOYENNES_3_ANS"]:
        if not os.path.exists("data/{}/tiles_{}_{}".format(directory,grid_type,tile_type)):
            os.mkdir("data/{}/tiles_{}_{}".format(directory,grid_type,tile_type))
        if not os.path.exists("data/{}/tiles_{}_{}/tiles_{}".format(directory,grid_type,tile_type,size)):
            os.mkdir("data/{}/tiles_{}_{}/tiles_{}".format(directory,grid_type,tile_type,size))
        
    
    for g in grids.keys():
        bounds = grids[g].total_bounds
        # Make sure the bounds are within the original grid
        bounds[2] -= (bounds[2]-bounds[0])%size
        bounds[3] -= (bounds[3]-bounds[1])%size
        
        bound = box(*bounds) 

        grids[g] = grids[g].loc[grids[g].within(bound)]
    
    fishnet = tiling(grids[grid_type], size=size, contiguous=contiguous)
    
    # Keep only tiles with idc
    if keep_only_idc:
        fishnet = gpd.GeoDataFrame(gpd.sjoin(fishnet,idc,op="intersects",how="inner"))
        fishnet = fishnet[~fishnet.index.duplicated(keep='first')]
        fishnet.drop(labels=fishnet.columns.drop("geometry"),axis=1,inplace=True)

    
    #Keep only tiles within Geneva canton
    fishnet = fishnet.loc[fishnet.within(ge)]

    
    # # Create tiles
    if makeTiles:
        print("Making ESM tiles")
        makeTilesGDAL(fishnet, bldg, idc,
                  input_folder="200km_2p5m_N26E38",
                  input_file="200km_2p5m_N26E38_2056.TIF",
                  output_folder="tiles_{}_{}/tiles_{}".format(grid_type,tile_type,size))
        
        print("Making Height tiles")
        makeTilesGDAL(fishnet, bldg, idc,
                  input_folder="GML_CAD_BATIMENT_HORSOL",
                  input_file="CAD_BATIMENT_HORSOL.TIF",
                  output_folder="tiles_{}_{}/tiles_{}".format(grid_type,tile_type,size))
        
        print("Making masks ")
        makeTilesGDAL(fishnet, bldg, idc,
                  input_folder="GDB_SCANE_INDICE_MOYENNES_3_ANS",
                  input_file="not_idc_2016.tif",
                  output_folder="tiles_{}_{}/tiles_{}".format(grid_type,tile_type,size))
    

    print("Calculate aggregated metrics")
    # Calculate non-normalized idc
    idc["idc"] = idc.INDICE * idc.SRE
    idc["idc_footprint"] = (idc.INDICE * idc.SHAPE_Area) #MJ/m2floor * m2footprint
    
    def calcStatistics(fishnet):
        stats = fishnet.copy()
        count = 0
        for i, row in fishnet.iterrows():
            # Dealing with buildings that are only partially within the tile
            crossing = (idc.intersects(row.geometry)&~idc.within(row.geometry))
            try:
                crossing_clipped = gpd.clip(idc.loc[crossing,:],row.geometry)
            except:
                print("Invalid geometry in tile {}".format(i))
                crossing_clipped = gpd.GeoDataFrame(columns=idc.columns)
            crossing_clipped["ratio_inside"] = 1-(crossing_clipped["SHAPE_Area"]- crossing_clipped.area) \
    /crossing_clipped["SHAPE_Area"]
            intile = idc.intersects(row.geometry)
            idc_mod  = idc.loc[intile,:].copy()
            # Weighting this buildings by their footprint area within the tile
            idc_mod.loc[crossing,["SHAPE_Area","SRE","INDICE","idc","idc_footprint"]] = \
                idc_mod.loc[crossing,["SHAPE_Area","SRE","INDICE","idc","idc_footprint"]].multiply(
                    crossing_clipped["ratio_inside"],axis=0)
            # Statistics
            # Calculate how many buildings with IDC and in total there are in each tile
            stats.loc[i,"nIDC"] = idc_mod.shape[0]
            stats.loc[i,"nBLDG"] = bldg.intersects(row.geometry).sum()
            stats.loc[i,"heated_area"] = idc_mod["SRE"].sum()
            stats.loc[i,"footprint_area"] = idc_mod["SHAPE_Area"].sum()
            # Footprint-weighted average normalized IDC (MJ/m2 floor area)
            stats.loc[i,"idc_norm"] = idc_mod["idc_footprint"].sum()/stats.loc[i,"footprint_area"]
            # Average normalized IDC per footprint area (MJ/m2 footprint area)
            stats.loc[i,"idc_norm_foot"] = idc_mod["idc"].sum()/stats.loc[i,"footprint_area"]
            # Absolute IDC (MJ)
            stats.loc[i,"idc_abs"] = idc_mod["idc"].sum()
            if count % 100 == 0:
                print("Progress:", np.round(count/fishnet.shape[0] * 100, 2),"%")
            count += 1
        return stats


    fishnet_split = np.array_split(fishnet, n_cores)
    pool = Pool(n_cores)
    with pool:
        stats = pd.concat(pool.map(calcStatistics,fishnet_split))
    
    stats["idc_abs"] /= 3600
    stats["idc_norm"] /= 3600
    stats["idc_norm_foot"] /= 3600
        
    stats.to_csv("data/{}_{}_{}.csv".format(grid_type,size,tile_type,))
    
