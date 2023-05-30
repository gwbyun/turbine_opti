#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:09:01 2023

@author: gw
"""

import numpy as np
import pandas as pd
from pyDOE2 import lhs
np.set_printoptions(precision=6, suppress=True)

def latinhypercube_sampling(input_data, output_data):
    data = pd.read_csv(input_data, header=None)
    features = data.iloc[:, :-3]

    min_values = np.min(features, axis=0)
    max_values = np.max(features, axis=0)
    
    min_values=min_values.values
    max_values=max_values.values
    print("min:",min_values)
    print("max:", max_values)

    # 샘플링할 특성(변수) 개수
    num_features = features.shape[1]
    # 생성할 샘플 개수
    num_samples = 1000

    #sampling = lhs(num_features, samples=num_samples, criterion='center')
    sampling = lhs(num_features, samples=num_samples, criterion='maximin')
    print(sampling)
    scaled_sampling = min_values + (max_values - min_values) * sampling
    
    #print(scaled_sampling.__str__())

    sampling_df = pd.DataFrame(scaled_sampling, columns=features.columns)
    sampling_df.to_csv(output_data, index=False)

    print(f"샘플링 결과가 저장되었습니다: {output_data}")

latinhypercube_sampling("turbin1-machinelearning.csv", "turbin1-machinelearning_normal3.csv")