# encoding: utf-8
# Author: zTaylor


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import random

random.seed(5)
np.random.seed(5)


class getVisualData:
    def __init__(self, idx = 132):
        self.org_x = pd.read_csv('./Processed_Data/RON_Feature_Selected_by_RF.csv').loc[idx]
        self.target_x = pd.read_csv('./Processed_Data/Optimal_Data_133.csv').loc[0][2:]
        self.org_y = pd.read_csv('./Processed_Data/Data_y.csv').loc[idx]
        self.target_y = pd.read_csv('./Processed_Data/Optimal_Data_133.csv').loc[0][:2]
        self.data4 = pd.read_excel('./Data/Data4_Copy.xlsx', index_col=0)
        self.todo_col_lt = [item for item in self.org_x.index if item in self.data4.index]
        self.finished_col = []
        self.flag = False
        self.step = 0
        print(self.org_x, '\n')
        print(self.target_x, '\n')
        print(self.org_y, '\n')
        print(self.target_y, '\n')
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

        self.new_col = ['Step', 'RON_Pred', 'S_Pred']
        self.new_col.extend(list(self.todo_col_lt))
        self.data_by_step = pd.DataFrame(columns=self.new_col)
        d = [self.step, self.org_y['RON_2'], self.org_y['Liu_2']]
        for col in self.todo_col_lt:
            d.append(self.org_x[col])
        d = pd.Series(d, index=self.new_col)
        self.data_by_step = self.data_by_step.append(d, ignore_index=True)
        self.data_by_step.to_csv('./Processed_Data/133_Data_by_Step.csv', float_format="%.10f", na_rep='NaN', index=False)

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

    def next(self):
        if self.flag:
            return

        self.step += 1
        for col in self.todo_col_lt:
            if self.org_x[col] != self.target_x[col]:
                if self.org_x[col] > self.target_x[col]:
                    self.org_x[col] = max(self.org_x[col] - self.val_delta[col], self.target_x[col])
                elif self.org_x[col] < self.target_x[col]:
                    self.org_x[col] = min(self.org_x[col] + self.val_delta[col], self.target_x[col])
            if self.org_x[col] == self.target_x[col] and col not in self.finished_col:
                self.finished_col.append(col)
                if len(self.todo_col_lt) == len(self.finished_col):
                    self.flag = True

        RON_pred_new = self.getRONPred(self.org_x)[0]
        S_pred_new = self.getSPred(self.org_x)[0]

        d = [self.step, self.org_x['RON'] - RON_pred_new, S_pred_new]
        for col in self.todo_col_lt:
            d.append(self.org_x[col])
        d = pd.Series(d, index=self.new_col)
        self.data_by_step = self.data_by_step.append(d, ignore_index=True)
        self.data_by_step.to_csv('./Processed_Data/133_Data_by_Step.csv', float_format="%.10f", na_rep='NaN', index=False)

        return self.step, self.org_x['RON'] - RON_pred_new, S_pred_new, self.org_x


if __name__ == '__main__':
    v = getVisualData()

    while not v.flag:
        step, RON_new, S_new, X_new = v.next()
        print(step)
        print(RON_new)
        print(S_new)