#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 09:38:36 2021

@author: giuseppeperonato
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = 'raise'
ESM_dic={1:1, #water
        2:2,#railways
        10:3,#NBU Area - Open Space
        20:4,#NBU Area - Green ndvix
        30:5,#BU Area - Open Space
        40:6,#BU Area - Green ndvix
        41:7,#BU Area - Green Urban Atlas
        50:0,#BU Area - Built-up
        }

def replace_with_dict(ar, dic):
    # https://stackoverflow.com/a/47171600
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))
    
    # Get argsort indices
    sidx = k.argsort()
    
    # Drop the magic bomb with searchsorted to get the corresponding
    # places for a in keys (using sorter since a is not necessarily sorted).
    # Then trace it back to original order with indexing into sidx
    # Finally index into values for desired output.
    return v[sidx[np.searchsorted(k,ar,sorter=sidx)]]


def getX(parameters):

    image_size = int(parameters["tile_size"]/parameters["pixel_size"]) 
    
    tiles = pd.read_csv("data/{}_{}_{}.csv".format(parameters["grid_type"],
                                                   parameters["tile_size"],
                                                   parameters["tile_type"]),index_col=0)
        
    if parameters["predicting_on_tile100"]:
        print("Predicting on tile 100")
        tiles_100 = pd.read_csv("data/{}_{}_{}.csv".format(parameters["grid_type"],
                                                           100,
                                                           "contiguous"),index_col=0)
        tiles = tiles_100.loc[tiles_100.index.isin(tiles.index),:]
        
    # Keep only tiles which have some buildings with IDC
    tiles_sel = tiles.loc[tiles.nIDC >= tiles.nBLDG * parameters["ratio_with_idc"],:].copy()



    # Compute building_coverage_ratio (of buildings with idc)
    if parameters["predicting_on_tile100"]:
        tile_size = 100
    else:
        tile_size = parameters["tile_size"]
    tiles_sel.loc[:,"idc_coverage_ratio"] = tiles_sel.loc[:,"footprint_area"] / tile_size**2

    
    # Create arrays
    # X = np.empty([tiles_sel.shape[0],image_size*image_size]) # single matrix 
    X1 = np.empty([tiles_sel.shape[0],image_size,image_size,1]) # multidimensional data
    X2 = np.empty([tiles_sel.shape[0],image_size,image_size,2]) # multidimensional data with mask
    X3 = np.empty([tiles_sel.shape[0],image_size,image_size,3]) # multidimensional data with mask and height
    
    count = 0
    coverage_ratio = np.empty([tiles_sel.shape[0],2])
    for i, row in tiles_sel.iterrows():
        #print(i, row.RELI)
        matrix = plt.imread("data/200km_2p5m_N26E38/tiles_{}_{}/tiles_{}/{}.tif".format(parameters["grid_type"],
                                                                                        parameters["tile_type"],
                                                                                        parameters["tile_size"],
                                                                                        i))
        matrix = np.squeeze(matrix)
        matrix = replace_with_dict(matrix,ESM_dic)
        
        # Calculate statistics of coverage ratio
        matrix_height = plt.imread("data/GML_CAD_BATIMENT_HORSOL/tiles_{}_{}/tiles_{}/{}.tif".format(parameters["grid_type"],
                                                                                                  parameters["tile_type"],
                                                                                                  parameters["tile_size"],
                                                                                                  i))
        matrix_height = np.squeeze(matrix_height)
        
        if parameters["predicting_on_tile100"]:
            s = int((matrix.shape[0]-40)//2)
            coverage_ratio[count,0] = np.sum(matrix[s:s+40,s:s+40] == 0)/np.sum(matrix[s:s+40,s:s+40]>=0)
            coverage_ratio[count,1] = np.sum(matrix_height[s:s+40,s:s+40] > 0)/np.sum(matrix_height[s:s+40,s:s+40]>=0)
        else:
            coverage_ratio[count,0] = np.sum(matrix == 0) / np.sum(matrix>=0)
            coverage_ratio[count,1] = np.sum(matrix_height > 0) / np.sum(matrix_height>=0)

        
        X1[count,:,:,0] = matrix
        
        mask = plt.imread("data/GDB_SCANE_INDICE_MOYENNES_3_ANS/tiles_{}_{}/tiles_{}/{}.tif".format(parameters["grid_type"],
                                                                                                parameters["tile_type"],
                                                                                                parameters["tile_size"],
                                                                                                i))
        X2[count,:,:,0] = matrix
        X2[count,:,:,1] = mask
        if parameters["use_heights"]:
            if parameters["modeled_height"]:
                matrix_height = plt.imread("data/height_generated/{}_{}/tiles_{}/{}.tif".format(parameters["grid_type"],
                                                                                             parameters["tile_type"],
                                                                                             parameters["tile_size"],
                                                                                             i))
            else:
                matrix_height = plt.imread("data/GML_CAD_BATIMENT_HORSOL/tiles_{}_{}/tiles_{}/{}.tif".format(parameters["grid_type"],
                                                                                                          parameters["tile_type"],
                                                                                                          parameters["tile_size"],
                                                                                                          i))
            X3[count,:,:,0] = matrix
            X3[count,:,:,1] = mask
            X3[count,:,:,2] = matrix_height
        # if parameters["predicting_on_tile100"]:
        #     X3[count,:,:,0] = matrix
        #     X3[count,:,:,1] = mask
        #     X3[count,:,:,2] = np.pad(np.zeros([40,40]),int((image_size-40)/2),constant_values=1)
    
        count += 1
    # Keep only tiles with a minimum building_coverage_ratio on idc and difference between raster and vector datasets 
    tiles_sel.loc[:,"building_coverage_ratio_raster"] = coverage_ratio[:,0]
    tiles_sel.loc[:,"building_coverage_ratio_vector"] = coverage_ratio[:,1]
    tiles_sel.loc[:,"difference_ratio"] = np.abs((tiles_sel["building_coverage_ratio_raster"]-tiles_sel["building_coverage_ratio_vector"])/tiles_sel["building_coverage_ratio_vector"])
    if parameters["difference_ratio"] == 999:
        diff_between_raster_and_vector = np.ones((tiles_sel.shape[0]), dtype=bool)
    else:
        diff_between_raster_and_vector = tiles_sel.loc[:,"difference_ratio"]
    conditions = np.logical_and(tiles_sel.loc[:,"idc_coverage_ratio"] >= parameters["idc_coverage_ratio"],
                                diff_between_raster_and_vector <= parameters["difference_ratio"])
        
    tiles_sel = tiles_sel.loc[conditions,:]
    X1 = X1[conditions,:,:,:]
    X2 = X2[conditions,:,:,:]
    X3 = X3[conditions,:,:,:]

            
    # Shuffle
    seed = 42
    np.random.seed(seed)
    shuffled = np.arange(X1.shape[0])
    np.random.shuffle(shuffled)
    tiles_sel = tiles_sel.iloc[shuffled]
    X1 = X1[shuffled]
    X2 = X2[shuffled]
    X3 = X3[shuffled]
    
        
    # Normalize
    X1 = X1/7
    X2 = X2/7
    X3 = X3/7
    
    # Separating train, val, test sets
    if parameters["use_heights"]:    
        # ESM + Height
        X_train = X3[:int(X1.shape[0]*parameters["training_ratio"]),:,:,:]
        X_val = X3[int(X1.shape[0]*parameters["training_ratio"]):int(X1.shape[0]*(parameters["training_ratio"]+parameters["validation_ratio"])),:,:]
        X_test = X3[int(X1.shape[0]*(parameters["training_ratio"]+parameters["validation_ratio"])):,:,:]
    else: # Only ESM + mask
        X_train = X2[:int(X1.shape[0]*parameters["training_ratio"]),:,:,:]
        X_val = X2[int(X1.shape[0]*parameters["training_ratio"]):int(X1.shape[0]*(parameters["training_ratio"]+parameters["validation_ratio"])),:,:]
        X_test = X2[int(X1.shape[0]*(parameters["training_ratio"]+parameters["validation_ratio"])):,:,:]
    
    tiles_sel = tiles_sel.copy()
    tiles_sel["set"] = ""
    tiles_sel.loc[tiles_sel.index[:int(tiles_sel.shape[0]*parameters["training_ratio"])],"set"] = "train"
    tiles_sel.loc[tiles_sel.index[int(tiles_sel.shape[0]*parameters["training_ratio"]):int(tiles_sel.shape[0]*(parameters["training_ratio"]+parameters["validation_ratio"]))],"set"] = "validation"
    tiles_sel.loc[tiles_sel.index[int(tiles_sel.shape[0]*(parameters["training_ratio"]+parameters["validation_ratio"]))]:,"set"] = "test"

    # Augmenting training set
    def flip(X_train,tiles_train,axis=2):
        X_train_dup = np.flip(X_train,axis)
        tiles_train_dup = tiles_train.copy()
        tiles_train_dup.index = tiles_train_dup.index.astype(str) + "_flip{}".format(axis)
        return X_train_dup, tiles_train_dup
        
    if parameters.get("flipping"):
        X_train_flip, tiles_train_flip = flip(X_train,tiles_sel.loc[tiles_sel["set"] == "train",:],axis=2)
        X_train = np.append(X_train,X_train_flip,axis=0)
        tiles_sel = pd.concat([tiles_sel.loc[tiles_sel["set"] == "train",:],
                               tiles_train_flip,
                               tiles_sel.loc[tiles_sel["set"] != "train",:]], axis=0)        
    if parameters.get("flipping_xy"):
        X_train_flip2, tiles_train_flip2 = flip(X_train,tiles_sel.loc[tiles_sel["set"] == "train",:],axis=2)
        X_train_flip1, tiles_train_flip1 = flip(X_train,tiles_sel.loc[tiles_sel["set"] == "train",:],axis=1)
        X_train = np.concatenate([X_train,X_train_flip1,X_train_flip2],axis=0)
        tiles_sel = pd.concat([tiles_sel.loc[tiles_sel["set"] == "train",:],
                               tiles_train_flip1,
                               tiles_train_flip2,
                               tiles_sel.loc[tiles_sel["set"] != "train",:]], axis=0)

    
    return tiles_sel, X_train, X_val, X_test

def performanceStats(predictor, real):
    diff = predictor  - real
    percentDiff = (diff / real) * 100
    absPercentDiff = np.abs(percentDiff)
    
    
    plt.boxplot(percentDiff)
    plt.xticks([1],["percentDiff"])
    plt.ylabel("%")
    plt.show()
    
    # compute the mean and standard deviation of the absolute percentage
    # difference
    mean = np.mean(absPercentDiff)
    std = np.std(absPercentDiff)
    print("Mean: {0:0.3f}%".format(mean))
    print("STD: {0:0.3f}%".format(std))
    print("")
        
