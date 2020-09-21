# encoding: utf-8
# Author: zTaylor

import os
import math
import numpy as np
import pandas as pd
from pandas import DataFrame


data = pd.read_excel('./Data/Data1_Copy_Copy_Copy.xlsx')
print(data.shape)             # (325, 340)


# remove time col
removed_cols = pd.DataFrame(columns=['time'])
removed_cols['time'] = data.iloc[:, 0]
data = data.iloc[:, 1:]       # 339


# remove 'total' cols
for col in data.columns:
    if 'TOTAL' in col:
        data = data.drop(col, axis=1)
        print(col)
print(data.shape)             # (325, 311)


# remove cols which NaN data or duplicate data take 60% of total
for col in data.columns[14:]:
    series = data[col]
    if series.isna().sum() / 325 >= 0.6:
        print("nan    ", col, series.isna().sum() / 325)
        removed_cols[col] = data[col]
        data = data.drop(col, axis=1)
    elif series.drop_duplicates().size / series.size < 0.4:
        print("duplicate    ", col, series.drop_duplicates().size / series.size)
        removed_cols[col] = data[col]
        data = data.drop(col, axis=1)
    if col in data.columns[14:]:
        series = series.fillna(series.mean())
        data[col] = series

print(removed_cols.shape)     # (325, 1 + 1 + 7)
print(data.shape)             # (325, 303)


# filter out of (min, max)
data4 = pd.read_excel('./Data/Data4_Copy.xlsx')
for i in range(data4.shape[0]):
    col = data4.iloc[i]['name']
    min = data4.iloc[i]['min']
    max = data4.iloc[i]['max']
    if col in data.columns:
        s = data[col]
        # s = s[(s >= min) & (s <= max)]
        if sum((s >= min) & (s <= max)) / s.size <= 0.3:
            print(col)
            data = data.drop(col, axis=1)
            continue
        else:
            for i in range(s.size):
                if s[i] < min:
                    s[i] = min
                elif s[i] > max:
                    s[i] = max
            data[col] = s
print(data.shape)             # (325, 302)


## 3seg filter
# for i in range(15, 320):
#     s = data.iloc[:, i]
#     seg = math.sqrt(sum((s - s.mean())**2) / (s.size - 1)) * 3
#     data.iloc[:, i] = s[(s - s.mean()).abs() <= seg]
# print(data.iloc[:, 15])
# print(data.columns)
# print(len(data.columns))
# # exit()
# print(data)


exit()

y = data.iloc[:, 7:10]
y.to_csv('./Processed_Data/Data_y.csv', float_format="%.10f", na_rep='NaN', index=False)

data = data.drop(columns=data.columns[7:10])
data.to_csv('./Processed_Data/Data_PreProcessed.csv', float_format="%.10f", na_rep='NaN', index=False)