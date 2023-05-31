#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 21:06:40 2023

@author: gw
"""

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def rbf(input_data, output_data,epoch):
    # csv file load
    data = pd.read_csv(input_data, header=None)

    # data divide
    turbine_para = data.iloc[:, :7]
    effi = data.iloc[:, 9]

    # data normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(turbine_para)

    # train set and test set
    turbine_para_train, turbine_para_test, effi_train, effi_test = train_test_split(X_scaled, effi, test_size=0.2, random_state=42)

    # TensorFlow model creation and training
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(100, activation='relu', input_shape=(7,)))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(turbine_para_train, effi_train, epochs=epoch, batch_size=64)

    # predict test data
    effi_pred = model.predict(turbine_para_test)

    # evaluate the model
    score = model.evaluate(turbine_para_test, effi_test)

    # plot predicted vs actual values
    plt.scatter(effi_test, effi_pred)
    plt.xlabel('Actual Efficiency')
    plt.ylabel('Predicted Efficiency')
    plt.title('Efficiency Prediction')
    plt.show()

    print("모델 평가 점수: {:.2f}".format(score))

    # save the model
    
    model.save(output_data)
    
rbf('turbin1-machinelearning.csv','trained_model.h5')
