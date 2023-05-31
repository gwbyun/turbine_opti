#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 17:30:40 2023

@author: gw
"""
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def mlp(input_data, output_data,epoch):
    #csv file load
    data = pd.read_csv(input_data,header=None)

    #data divide
    print(data)
    turbine_para = data.iloc[:, :7]
    effi= data.iloc[:, 9]
    #print(effi)

    #data nomalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(turbine_para)
    #train set and test set
    turbine_para_train, turbine_para_test, effi_train, effi_test = train_test_split(X_scaled,effi,test_size=0.2, random_state = 42)
    '''
    #mlp model creation and training
    mlp = MLPRegressor(hidden_layer_sizes=(100,100), activation = 'relu', random_state =42)
    mlp.fit(turbine_para_train,effi_train)
    '''
    
    # TensorFlow 모델 생성과 훈련
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(250, activation='relu', input_shape=(7,)))
    model.add(tf.keras.layers.Dense(250, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(turbine_para_train, effi_train, epochs=epoch, batch_size=64)
    loss_history = model.history.history['loss']
    #predict test data
    #effi_pred = model.predict(turbine_para_test)
    
    score = model.evaluate(turbine_para_test,effi_test)
    # 손실 값 그래프 그리기
    plt.plot(loss_history)
    plt.title('Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    print("모델 평가 점수: {:.2f}".format(score))
    #model save
    model.save(output_data)
    
#mlp('turbin1-machinelearning.csv','trained_model.h5')



