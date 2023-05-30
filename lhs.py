#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:09:01 2023

@author: gw
"""

import numpy as np
from scipy.stats import uniform
import pandas as pd

data = pd.read_csv("turbin1-machinelearning.csv",header =None)
features = data.iloc[:, :-2]
print(features)
'''
# 난수 생성을 위한 설정
np.random.seed(0)
n_samples = 100  # 생성할 데이터 개수
n_features = 7  # 특성 데이터 열 수

# 특성 데이터의 범위 설정 (0부터 1까지)
ranges = np.array([[0, 1]] * n_features)

# lhs를 생성하여 특성 데이터 생성
lhs_samples = uniform(loc=ranges[:, 0], scale=ranges[:, 1] - ranges[:, 0]).rvs((n_samples, n_features))

# 특성 데이터를 DataFrame 생성
df = pd.DataFrame(lhs_samples, columns=[f"Column {i}" for i in range(1, n_features + 1)])

# DataFrame을 CSV 파일로 저장
df.to_csv("data.csv", index=False)
'''