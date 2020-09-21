# encoding: utf-8
# Author: zTaylor

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

import pickle


data = pd.read_csv('./Processed_Data/RON_Feature_Selected_by_RF.csv')
y = pd.read_csv('./Processed_Data/Data_y.csv')


test_ratio = 0.2
is_shuffle = True
random_state = 20
x_train, x_test, y_train, y_test = train_test_split(data.to_numpy(), y.to_numpy()[:, 2],
                                                    test_size=test_ratio, shuffle=is_shuffle, random_state=random_state)


from sklearn.model_selection import KFold
_N_FOLDS = 5    # 采用5折交叉验证
kf = KFold(n_splits=_N_FOLDS, shuffle=True, random_state=10)
# X = data.to_numpy()
# Y = y.to_numpy()[:, 2]
bst = XGBRegressor(
    learning_rate=0.01,
    n_estimators=300,
    max_delta=10,
    # min_child_weight=6,
    # gamma=0.1,
    subsample=0.5,
    colsample_btree=0.8,
    # objective='multi:softmax',
    # scale_pos_weight=1,
    # random_state=27
)
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    # max_features=30,
    bootstrap=True,
    # random_state=30,
    verbose=True,
    # max_samples=0.5,
)

model_list = [bst, rf]

def getStkTrain(model):
    stk_train = np.zeros((x_train.shape[0], 1))
    stk_test = np.empty((_N_FOLDS, x_test.shape[0], 1))
    for i, (train_index, val_index) in enumerate(kf.split(x_train)):
        X_train = x_train[train_index]
        Y_train = y_train[train_index]
        X_val = x_train[val_index]
        Y_val = y_train[val_index]

        model.fit(X_train, Y_train)
        # model = model.predict(X_val)
        stk_train[val_index] = model.predict(X_val).reshape(-1, 1)
        stk_test[i, :] = model.predict(x_test).reshape(-1, 1)
        # print("{} tims, XGB MSE_LOSS on validation dataset: {}".format(i, mean_squared_error(Y_val, bst_y_pred)))

    stk_test = stk_test.mean(axis=0)
    return stk_train, stk_test


new_train, new_test = [], []
for model in model_list:
    stk_train, stk_test = getStkTrain(model)
    new_train.append(stk_train)
    new_test.append(stk_test)


new_train = np.concatenate(new_train, axis=1)
new_test = np.concatenate(new_test, axis=1)


model = LinearRegression()
model.fit(new_train, y_train)
model.predict(new_test)
print("MSE_LOSS on test dataset: {}".format(mean_squared_error(y_test, model.predict(new_test))))
model_list.append(model)

plt.plot(range(len(x_test)), y_test, 'b-', label='y_true')
plt.plot(range(len(x_test)), model.predict(new_test), 'r-', label='y_pred')
plt.legend(labels=['y_true','y_pred'],loc='best')
plt.xlabel('Sample NO.')
plt.ylabel('RON_LOSS')
plt.show()

is_save = False
if is_save:
    for i, model in enumerate(model_list):
        with open('./saved_model/RON_model_{}.pickle'.format(i), 'wb') as f:
            pickle.dump(model, f)
