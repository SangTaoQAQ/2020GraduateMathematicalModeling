# encoding: utf-8
# Author: zTaylor

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

from SA import SA


data = pd.read_csv('./Processed_Data/RON_Feature_Selected_by_RF.csv')

new_col = ['Index', 'Time', 'RON_Loss', 'S']
new_col.extend(data.columns)

optimal_data = pd.DataFrame(columns=new_col)
# print(optimal_data.columns)

sa = SA()       # 模拟退火


for idx in range(data.shpae[0]):
    time, RON_LOSS_Pred, S_Pred, new_x = sa.getNewX(idx)
    s = [idx, time, RON_LOSS_Pred, S_Pred]
    for col in new_x.index:
        s.append(new_x[col])
    s = pd.Series(s, index=new_col)
    optimal_data = optimal_data.append(s, ignore_index=True)

    optimal_data.to_csv('./Processed_Data/Optimal_Data_by_SA.csv', float_format="%.10f", na_rep='NaN', index=False)