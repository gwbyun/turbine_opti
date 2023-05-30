#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:45:42 2023

@author: gw
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler

def normalize(input_data,output_data):
    # CSV 파일 읽기
    data = pd.read_csv(input_data, header = None)
    #print(data)


    # 특성(피처) 값 추출
    features = data.iloc[:, 0:9].astype(float).values
    print(features)
    # 정규화를 위한 Scaler 객체 생성
    scaler = MinMaxScaler()
    standardized_features = scaler.fit_transform(features)
    standardized_scaler = MinMaxScaler()
    standardized_data = standardized_scaler.fit_transform(standardized_features)
    #print(normalized_features)
    # 정규화된 특성 값을 DataFrame으로 변환
    standardized_df = pd.DataFrame(standardized_data, columns=data.columns[0:9])
    print(standardized_df)
    
    standardized_df.to_csv(output_data, index=False)
    
normalize("turbin1-machinelearning.csv","turbin1-machinelearning_normal2.csv")