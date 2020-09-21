# encoding: utf-8
# Author: zTaylor

import os
import math
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from scipy.stats import pearsonr
from tools import distcorr


data = pd.read_csv('./Processed_Data/Data_PreProcessed.csv')
y = pd.read_csv('./Processed_Data/Data_y.csv')


print(data.shape)       # 299


# selected by var
# cols = list(data.columns)
# cols.remove('RON')
cols = data.columns[11:]
for col in cols:
    s = data[col]
    s = (data[col] - min(s)) / (max(s) - min(s))
    if s.var() < 0.01:
        print(col)
        data = data.drop([col], axis=1)
print(data.shape)   # 269
# data.to_csv('./Processed_Data/Data_rm_by_Var.csv', float_format="%.10f", na_rep='NaN', index=False)


# selected by corr
delta = 0.8
corr = (data.corr() >= delta) | (data.corr() <= -delta)
# cols = list(data.columns)
# cols.remove('RON')
cols = data.columns[11:]
for i in range(len(cols)):
    s = corr.iloc[i]
    l = list(range(11))
    l.extend(range(i+1, len(cols)))
    for j in l:       # for A and B which has strong correlation, we should remove only one of them
        if s.iloc[j]:
            print(cols[i])
            data = data.drop([cols[i]], axis=1)
            break
print(data.shape)   # 139


# selected by d_corr
# cols = list(data.columns)
# cols.remove('RON')
cols = data.columns[11:]
for i in range(len(cols)):
    l = list(range(11))
    l.extend(range(i+1, len(cols)))
    for j in l:
        if(cols[i] in data.columns) and (cols[j] in data.columns):
            s1 = data[cols[i]]
            s2 = data[cols[j]]
            dcorr = distcorr(s1.to_numpy(), s2.to_numpy())
            if dcorr > 0.8:
                print(cols[i], cols[j])
                data = data.drop([cols[i]], axis=1)
                break
print(data.shape)   # 96
# data.to_csv('./Processed_Data/Data_rm_by_Corr.csv', float_format="%.10f", na_rep='NaN', index=False)
print('\n\n')


import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
sns.heatmap(data.corr())
plt.show()


# selected RON feature by RF
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=6,
    max_features=30,
    bootstrap=True,
    random_state=30,
    verbose=True,
    max_samples=0.5,
)
rf = rf.fit(data.to_numpy(), y.to_numpy()[:, 1])
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
top30 = []
for f in range(30):
    print("%2d) %-*s %f" % (f + 1, 30, data.columns[indices[f]], importances[indices[f]]))
    top30.append(data.columns[indices[f]])
# print(top30, '\n\n')
# data[top30].to_csv('./Processed_Data/RON_Feature_Selected_by_RF.csv', float_format="%.10f", na_rep='NaN', index=False)
print('\n\n')

