# encoding: utf-8
# Author: zTaylor

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import random

random.seed(5)
np.random.seed(5)


class SA:
    def __init__(self):
        self.data = pd.read_csv('./Processed_Data/RON_Feature_Selected_by_RF.csv')
        self.y = pd.read_csv('./Processed_Data/Data_y.csv')
        self.data4 = pd.read_excel('./Data/Data4_Copy.xlsx', index_col=0)
        self.init()

    def init(self):
        self.RON_model_list = []
        for i in range(3):
            with open('./saved_model/RON_model_{}.pickle'.format(i), 'rb') as f:
                self.RON_model_list.append(pickle.load(f))

        self.S_model_list = []
        for i in range(3):
            with open('./saved_model/S_model_{}.pickle'.format(i), 'rb') as f:
                self.S_model_list.append(pickle.load(f))

        self.val_Mins = {}
        self.val_Maxs = {}
        self.val_delta = {}
        for col in self.data4.index:
            self.val_Mins[col] = self.data4.loc[col]['min']
            self.val_Maxs[col] = self.data4.loc[col]['max']
            self.val_delta[col] = self.data4.loc[col]['delta值']

        self.k = 10
        self.vals = {}  # vals    325
        self.vals_rec = {}  # record vals when changed    max: k for evey val
        col_name_lt = self.data.columns
        for i in range(self.data.shape[0]):
            # if i not in self.vals:
            self.vals_rec[i] = {}
            for j in range(self.k):
                self.vals_rec[i][j] = self.data.loc[i].copy(deep=True)  # k copy
                # for col in data4.index:
                #     vals_rec[i][j][col] = getRangeRandom(val_Mins[col], val_Maxs[col])
            self.vals[i] = {}
            for col_name in col_name_lt:
                self.vals[i][col_name] = self.data.loc[i][col_name]

    def getRONPred(self, single_x):
        x_new = np.zeros((1, len(self.RON_model_list) - 1))
        for i, model in enumerate(self.RON_model_list[:-1]):
            x_new[0][i] = model.predict(single_x.to_numpy().reshape(1, -1))[0]
        return self.RON_model_list[-1].predict(x_new)

    def getSPred(self, single_x):
        x_new = np.zeros((1, len(self.S_model_list) - 1))
        for i, model in enumerate(self.S_model_list[:-1]):
            x_new[0][i] = model.predict(single_x.to_numpy().reshape(1, -1))[0]
        return self.S_model_list[-1].predict(x_new)

    def getRandom(self, x, time):
        for col in x.index:
            if col in self.data4.index:
                if time >= 5:
                    x[col] = random.uniform(self.val_Mins[col], self.val_Maxs[col])
                else:
                    x[col] = x[col] + random.uniform(-self.val_Mins[col], self.val_Mins[col])
                    if x[col] < self.val_Mins[col]:
                        x[col] = self.val_Mins[col]
                    if x[col] > self.val_Maxs[col]:
                        x[col] = self.val_Maxs[col]
        return x

    def getRangeRandom(self, min, max):
        return min + (max - min) * random.random()

    def getNewX(self, idx=0, t=100, k=10, step=0.9, TMin=1e-3):
        # print(self.data.loc[idx], '\n\n')
        org_pred = self.y.to_numpy()[idx][2]
        target_pred = org_pred * 0.7
        time = 0
        flag = False
        while t > TMin:
            for j in range(k):
                RON_pred = self.getRONPred(self.vals_rec[idx][j])
                S_pred = self.getSPred(self.vals_rec[idx][j])
                x_new = self.getRandom(self.vals_rec[idx][j], time)
                RON_pred_new = self.getRONPred(x_new)
                S_pred_new = self.getSPred(x_new)
                if RON_pred_new < RON_pred and S_pred_new[0] < 5:
                    self.vals_rec[idx][j] = x_new
                else:
                    if (t - TMin) > random.randint(0, 99):
                        self.vals_rec[idx][j] = x_new
                if RON_pred_new < target_pred and S_pred_new[0] < 5:
                    flag = True
                    break

            if flag:
                break
            t = t * step

            # if time % 100 == 0:
            #     RON_pred = 10000
            #     x_temp = []
            #     for j in range(k):
            #         RON_pred_new = self.getRONPred(self.vals_rec[idx][j])
            #         if RON_pred_new < RON_pred:
            #             RON_pred = RON_pred_new
            #             x_temp.append(self.vals_rec[idx][j])
            #     print(time)
            #     print(RON_pred)
            #     print(x_temp[-1])

            time += 1

        RON_pred = 10000
        S_pred = 100
        x_temp = []
        for j in range(k):
            RON_pred_new = self.getRONPred(self.vals_rec[idx][j])
            if RON_pred_new < RON_pred:
                RON_pred = RON_pred_new
                S_pred = self.getSPred(self.vals_rec[idx][j])
                x_temp.append(self.vals_rec[idx][j])
        print("\nans:")
        print(time)
        print(RON_pred)
        print(S_pred)
        print(x_temp[-1])
        return time, RON_pred[0], S_pred[0], x_temp[-1]


if __name__ == '__main__':
    SA().getNewX(53)