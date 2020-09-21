# encoding: utf-8
# Author: zTaylor

import os
import math
import numpy as np
import pandas as pd
from pandas import DataFrame


data_285 = pd.read_excel('./Data/Data3.xlsx', sheet_name=0)
data_313 = pd.read_excel('./Data/Data3.xlsx', sheet_name=1)

data = [data_285, data_313][1]

# remove cols which NaN data or duplicate data take 60% of total
for col in data.columns[1:]:
    series = data[col]
    series = series.replace(0, np.nan)
    if series.isna().sum() / 40 == 1:
        print("nan    ", col, series.isna().sum() /40)
        data = data.drop(col, axis=1)
    else:
        series.fillna(series.mean())
        data[col] = series
print('\n\n')


# filter out of (min, max)
data4 = pd.read_excel('./Data/Data4_Copy.xlsx')
for i in range(data4.shape[0]):
    col = data4.iloc[i]['name']
    min = data4.iloc[i]['min']
    max = data4.iloc[i]['max']
    if col in data.columns[1:]:
        s = data[col]
        for i in range(s.size):
            if s[i] < min:
                # s[i] = min
                s[i] = 0
            elif s[i] > max:
                # s[i] = max
                s[i] = 0
        data[col] = s


# remove cols which NaN data or duplicate data take 60% of total
for col in data.columns[1:]:
    series = data[col]
    series = series.replace(0, np.nan)
    if series.isna().sum() / 40 == 1:
        print("nan    ", col, series.isna().sum() /40)
        data = data.drop(col, axis=1)
    else:
        series.fillna(series.mean())
        data[col] = series
print('\n\n')


# 3seg filter
for col in data.columns[1:]:
    s = data[col]
    s.replace(0, np.nan)
    seg = math.sqrt(sum((s - s.mean())**2) / (s.size - 1)) * 3
    data[col] = s[(s - s.mean()).abs() <= seg]

data = data.iloc[:, 1:]
avg = []
for col in data.columns:
    s = data[col]
    avg.append(s.mean())

s = pd.Series(avg, index=list(data.columns))
data = data.append(s, ignore_index=True)

# data.to_csv('./Data/Process_285.csv', float_format="%.10f", na_rep='NaN', index=False)
# data.to_csv('./Data/Process_313.csv', float_format="%.10f", na_rep='NaN', index=False)
