#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:11:55 2021

@author: giuseppeperonato, rboghetti

Original template from https://www.pyimagesearch.com/2019/01/28/keras-regression-and-cnns/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import uuid
import json
import os
import glob
import sys
import shutil

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Cropping2D, Dense, Concatenate, Dropout, Activation, UpSampling2D, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import prepareModel

def create_cnn(width, height, depth, filters=(16, 32, 64, 64), regress=False):
    # Original CNN from https://www.pyimagesearch.com/2019/01/28/keras-regression-and-cnns/
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1
	# define the model input
	inputs = Input(shape=inputShape)
	# loop over the number of filters
	for (i, f) in enumerate(filters):
		# if this is the first CONV layer then set the input
		# appropriately
		if i == 0:
			x = inputs
		# CONV => RELU => BN => POOL
		x = Conv2D(f, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)    
		x = BatchNormalization(axis=chanDim)(x)
        
	# flatten the volume, then FC => RELU => BN => DROPOUT
	x = Flatten()(x)
	
	x = Dense(64)(x)
	x = Activation("relu")(x)
	x = Dropout(0.4)(x)
	x = Dense(64)(x)
	x = Activation("relu")(x)
	x = Dropout(0.4)(x)
	# apply another FC layer, this one to match the number of nodes
	# coming out of the MLP
	x = Dense(8)(x)
	x = Activation("relu")(x)
	# check to see if the regression node should be added
	if regress:
		x = Dense(1, activation="linear")(x)
	# construct the CNN
	model = Model(inputs, x)
	# return the CNN
	return model

def create_context_aware_cnn(conv_filters = 256, hidden_neurons = 128,
                             branch_hidden_neurons = 256, 
                             height = 40, width = 40, channels = 2):
    # Input layer
    input_layer = Input(shape=(height, width, channels), name= "tile_input") 
    
    # Convolutions - full context
    xfull = Conv2D(conv_filters//4, (3, 3), activation='relu')(input_layer)
    xfull = MaxPooling2D((2, 2))(xfull)
    xfull = BatchNormalization(axis=-1)(xfull)
    xfull = Dropout(0.4)(xfull)
    xfull = Conv2D(conv_filters//2, (3, 3), activation='relu')(xfull)
    xfull = MaxPooling2D((2, 2))(xfull)
    xfull = BatchNormalization(axis=-1)(xfull)
    xfull = Dropout(0.4)(xfull)
    xfull = Conv2D(conv_filters//2, (3, 3), activation='relu')(xfull)
    xfull = MaxPooling2D((2, 2))(xfull)
    xfull = BatchNormalization(axis=-1)(xfull)
    xfull = Dropout(0.4)(xfull)
    xfull = Conv2D(conv_filters, (3, 3), activation='relu')(xfull)
    xfull = MaxPooling2D((2, 2))(xfull)
    xfull = BatchNormalization(axis=-1)(xfull)
    xfull = Dropout(0.4)(xfull)
    xfull = Conv2D(conv_filters, (3, 3), activation='relu')(xfull)
    xfull = BatchNormalization(axis=-1)(xfull)
    xfull = Dropout(0.4)(xfull)
    xfull = Flatten()(xfull)
    xfull = Dense(branch_hidden_neurons)(xfull)
    xfull = Dropout(0.4)(xfull)
    
    # Convolutions - image center
    s = int((height - 40)//2)
    xcenter = Cropping2D(cropping=s)(input_layer)
    xcenter = Conv2D(conv_filters//2, (3, 3), activation='relu')(xcenter)
    xcenter = MaxPooling2D((2, 2))(xcenter)               
    xcenter = BatchNormalization(axis=-1)(xcenter)
    xcenter = Dropout(0.4)(xcenter)
    xcenter = Conv2D(conv_filters, (3, 3), activation='relu')(xcenter)
    xcenter = MaxPooling2D((2, 2))(xcenter)
    xcenter = BatchNormalization(axis=-1)(xcenter)
    xcenter = Dropout(0.4)(xcenter)
    xcenter = Conv2D(conv_filters, (3, 3), activation='relu')(xcenter)
    xcenter = MaxPooling2D((2, 2))(xcenter)
    xcenter = BatchNormalization(axis=-1)(xcenter)
    xcenter = Dropout(0.4)(xcenter)
    xcenter = Flatten()(xcenter)
    xcenter = Dense(branch_hidden_neurons)(xcenter)
    xcenter = Dropout(0.4)(xcenter)

    # Concatenation
    concatenate = Concatenate()([xfull, xcenter])
    
    # Hidden layer
    x = Dense(hidden_neurons, activation="relu", name="dense_conc")(concatenate)
    x = Dropout(0.4)(x)
    
    # Hidden layer
    x = Dense(4, activation="relu", name="dense_conc")(concatenate)
    
    # Output
    output_layer = Dense(1, activation="relu", name="output")(x)

    # Model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

if __name__ == "__main__":
    
    #######  INPUT
    if len(sys.argv) > 1:
        param_file = sys.argv[1]
    else:
        param_file = "parameters.json"
    
    with open(param_file,"r") as f:
        parameters = json.loads(f.read())
    
    image_size = int(parameters["tile_size"]/parameters["pixel_size"])
    #######
    
    # Creating results data
    # Check whether the results already exist
    Run = True
    for file in glob.glob('results/*/parameters.json'):
        with open(file,"r") as f:
            if parameters == json.loads(f.read()):
                overwrite = "n"
                overwrite = input('Overwrite? (type y to continue, something else to load the model)   ')
                if overwrite != "y":
                    Run = False
                this_uuid = os.path.basename(os.path.dirname(file))
                break
            else:
                this_uuid = str(uuid.uuid1())
    if len(glob.glob('results/*/parameters.json')) == 0:
        this_uuid = str(uuid.uuid1())
        
    # Create results folder and json parameters            
    dest_folder = os.path.join("results","{}".format(this_uuid))
    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)
        
    
    with open(os.path.join(dest_folder,"parameters.json"),"w") as f:
        f.write(json.dumps(parameters,indent=4))
    
    
    tiles, X_train, X_val, X_test = prepareModel.getX(parameters)
    
    # Select parameter                                                                          
    y_train = tiles.loc[tiles.set=="train",parameters["predicting_variable"]].values 
    y_val = tiles.loc[tiles.set=="validation",parameters["predicting_variable"]].values 
    y_test = tiles.loc[tiles.set=="test",parameters["predicting_variable"]].values 
     
    if Run:
        if parameters['predicting_on_tile100'] and parameters.get("context_aware"):
            model = create_context_aware_cnn(height = parameters["tile_size"] // parameters["pixel_size"], 
                                          width = parameters["tile_size"] // parameters["pixel_size"])
        else:
            model = create_cnn(image_size, image_size, depth=X_train.shape[3], filters=(16, 32, 32, 64, 64), regress=True)
        
        opt = Adam(lr=1e-3, decay=1e-3 / 200)
        model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
        
        checkpoint = ModelCheckpoint(
        filepath=os.path.join("results",this_uuid,"checkpoint"),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch',
        )
        
        early_stopping_monitor = EarlyStopping(
            monitor='val_loss',
            min_delta=0.1,
            patience=25,
            verbose=1,
            mode='auto',
            baseline=None,
            restore_best_weights=True
        )
        
        # train the model
        print("[INFO] training model...")
        model.fit(x=X_train, y=y_train, 
            validation_data=(X_val, y_val),
            epochs=1000, batch_size=8, callbacks=[checkpoint, early_stopping_monitor])
        
        # Saving results
        model.save("results/{}/model".format(this_uuid))
        try:
            os.remove("results/{}/checkpoint".format(this_uuid))
        except:
            shutil.rmtree("results/{}/checkpoint".format(this_uuid))
        pd.DataFrame(model.history.history).sort_values("val_loss").to_csv(os.path.join("results",this_uuid,"losses.txt"),
                                                                           sep="\t",
                                                                           float_format='%.2f')
        tiles["set"].to_csv(os.path.join("results",this_uuid,"samples.csv"),index=True,header=True)
        
        # Plotting loss function
        pd.DataFrame(model.history.history).plot()
        plt.ylim([10,120])
        plt.show()
    
    else:
        model = load_model(os.path.join("results",this_uuid,"model"))
    
    
    print("[INFO] predicting  IDC...")
    preds = model.predict(X_test)
    
    
    print("Model")
    prepareModel.performanceStats(preds.flatten(),y_test)




