#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 22:38:21 2017

@author: LinZhang
"""
import numpy as np # linear algebra
import scipy as sp
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV,PredefinedSplit

from sklearn_evaluation import plot

#
def imp_print(info,slen=20):
    print ("="*slen)
    print (info)
    print ("="*slen)

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
def evaluate_test(model,topk=50):
    # 每天计算分数最低top50，平均后再按天平均
    model.fit(train.values,y_train)
    y_test_pred = model.predict(test.values)
    ##### mse Evaluation
    mse = mean_squared_error(y_test_pred, y_test)
    print("Test mse score: {:.10f}".format(mse))
    ##### stock Evaluation
    # take top 50 stocks and calculate avg(label)
    test_withPred = test.copy()
    test_withPred["pred_test"] = y_test_pred 
    test_withPred.insert(0, 'csv_index', test_csv_index.values)
    csv_indexs = test_withPred['csv_index'].unique()
    # 每个index单独评分
    test_withPred_sorted = test_withPred.sort_values("pred_test",ascending = True)
    avg_label =  test_withPred_sorted[:topk][test_withPred_sorted.columns[0]].mean()
    print("Test Dataset avg label score:{:.4f}".format(test[test.columns[0]].mean()))
    print("Test top"+str(topk)+" avg label score: {:.4f}".format(avg_label))
    
#####################
## define global parameters
Params = {}
Params['algo'] = ['lasso']
#####################
# Read the data: 选择数据的时间段
#####################
imp_print("Data Loading...",40)
# 数据格式 hdf5
train_raw = pd.read_hdf('DataSet/train_1331_1333.h5')
test_raw = pd.read_hdf('DataSet/test_1331_1333.h5')
# 选择数据时间段：todo
train = train_raw
test=test_raw
#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))
#Save the 'csv_index' column
train_csv_index = train[train.columns[0]]
test_csv_index = test[train.columns[0]]
#Save the 'Id' column
train_ID = train[train.columns[1]]
test_ID = test[train.columns[1]]
#Now drop the  'csv_index' & 'Id' colum since it's unnecessary for  the prediction process.
train.drop(train.columns[0:2], axis = 1, inplace = True)
test.drop(test.columns[0:2], axis = 1, inplace = True)
#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))

# 
#####################
# Preprocess: 处理成训练和测试集合
#####################
imp_print("Data Processing...",40)
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train[train.columns[0]].values
y_test = test[test.columns[0]].values
all_data = pd.concat((train, test)).reset_index(drop=True)
y_all_data = all_data[all_data.columns[0]].values
all_data.drop(train.columns[0], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))
# missing data
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)
print("missing data column numbers:{}".format(missing_data.shape[0]))
if(missing_data.shape[0] == 0):
    imp_print("no missing data, go to next step...")
else:
    imp_print("Need filling missing data...")

#####################
# Modeling: 建模
#####################
imp_print("Modeling...",40)
# get the train and val and test data
train = all_data[:ntrain]
test = all_data[ntrain:]
######
# Lasso Regression
######
scaler = StandardScaler()
#scaler = RobustScaler()
lasso = Pipeline(steps=[('scaler',scaler),
                      ('lasso',Lasso(alpha =0.05, random_state=1))])
#####################
# # Test: 测试获取评价结果
#####################
# grid search params
param_grid = dict(scaler=[StandardScaler(),RobustScaler()]
                  ,lasso__alpha=[0.03,0.04,0.05,0.06,0.07])
# grid_search
custom_cv = PredefinedSplit([-1]*ntrain+[1]*ntest)
reggressor = GridSearchCV(lasso
                           , param_grid=param_grid
                           ,scoring = 'neg_mean_squared_error'
                           ,cv = custom_cv
                           ,n_jobs=-1, verbose=1)
reggressor.fit(all_data.values,y_all_data)

print('grid search result:')
print('best score:'+ str(reggressor.best_score_))
print('best params:'+ str(reggressor.best_params_))
print('detailed results:',report(reggressor.cv_results_))

imp_print("Testing...",40)
if('lasso' in Params['algo']):
    imp_print("lasso:",10)
    evaluate_test(lasso)

print ("Finished...")