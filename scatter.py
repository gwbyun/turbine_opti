#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:10:46 2023

@author: gw
"""

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix, heatmap
import numpy as np


df = pd.read_csv("turbin1-machinelearning_normal.csv",header=None)
df.columns=['a','b','c','d','e','f','g' ,'power','effi']
df.head()
print(df)

cols=['a','b','c','d','e','f','g','power','effi']
#scatterplotmatrix(df[cols].values, figsize=(10,8), names = cols, alpha = 0.5)
#plt.tight_layout()
#plt.show()

cm=np.corrcoef(df[cols].values.T)
hm = heatmap(cm,
             row_names=cols,
             column_names=cols)
plt.show()