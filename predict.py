#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 20:30:21 2023

@author: gw
"""

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def predict_efficiency(input_data, model_path):
    # Load the input data
    data = pd.read_csv(input_data,header=None)
    
    # Data normalization
    scaler = StandardScaler()
    turbine_para = data.iloc[:, :7]
    print(turbine_para)
    X_scaled = scaler.fit_transform(turbine_para)
    
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    print(predictions)
    # Return the predictions
    #return predictions

# Specify the input data and model path
input_data = 'turbin1-machinelearning.csv'
model_path = 'trained_model.h5'

# Make predictions using the trained model
#predictions = predict_efficiency(input_data, model_path)

# Print the predictions
#print(predictions)