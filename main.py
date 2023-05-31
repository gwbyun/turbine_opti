#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 21:51:09 2023

@author: gw
"""

import mlp
#import rbf
import predict

input_data ='turbin1-machinelearning.csv'
input_data_predict ='turbin1-machinelearning.csv'
output_model_data ='trained_model.h5'

model = mlp.mlp
epoch = 1000

model(input_data, output_model_data,epoch)

predict.predict_efficiency(input_data_predict,output_model_data)