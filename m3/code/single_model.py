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

from sklearn.metrics import make_scorer

import lightgbm as lgb


#User defined functions
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
            print("Mean validation score: {0:.5f} (std: {1:.5f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def report_params(results,rank,rank_parmas):
    candidates = np.flatnonzero(results['rank_test_score'] == rank)
    for candidate in candidates:
        rank_parmas.loc[len(rank_parmas)] = [rank
                                            ,results['mean_test_score'][candidate]
                                            ,results['params'][candidate]]

def top50_avg_loss(estimator, X, y):
#    print(estimator.get_params())
    y_pred = estimator.predict(X)
    result_df = pd.DataFrame(columns=['csv_index'
                                      ,'pred_avg'])
    for csv_index in test_csvIndexs:
#        print('y '+str(csv_index)+':'+ str(y[(test_csv_index==csv_index).values][:10]))
        temp_df = pd.DataFrame()
        temp_df['temp_y_pred'] = y_pred[(test_csv_index==csv_index).values]
        temp_df['temp_y'] = y[(test_csv_index==csv_index).values]
        temp_df_sorted = temp_df.sort_values("temp_y_pred",ascending = True)
        temp_select = temp_df_sorted[:Params['topK']]['temp_y']
#        print('temp_select len:'+ str(len(temp_select)))
        result_df.loc[len(result_df)] = [str(csv_index),temp_select.mean()]
    
#    print(result_df)
#    print(result_df['pred_avg'].mean())
    result_list.append(result_df['pred_avg'].mean())
    return -result_df['pred_avg'].mean()
    

            
def evaluate_test(model,params,topks=[50,30,10]):
    # 设置训练参数：
    model.set_params(**params)
    # 每天计算分数最低top50，平均后再按天平均
    model.fit(train.values,y_train)
    y_test_pred = model.predict(test.values)
    ##### mse Evaluation
    mse = mean_squared_error(y_test_pred, y_test)
    print("Test mse score: {:.10f}".format(mse))
    ##### stock Evaluation
    # take top 50 stocks and calculate avg(label)
    test_withPred = pd.DataFrame()
    test_withPred['y_test'] = y_test
    test_withPred["pred_test"] = y_test_pred 
    test_withPred.insert(0, 'csv_index', test_csv_index.values)
    csv_indexs = test_withPred['csv_index'].unique()
    # 每个index单独评分
    eval_df = pd.DataFrame(columns=['csv_index'
                                    ,'topk'
                                    ,'pred_avg'
                                    ,'pred_std'
                                    ,'pred_min'
                                    ,'pred_max'
                                    ,'simple_avg'])
    for csv_index in csv_indexs:
        temp_test = test_withPred[test_withPred['csv_index'] == csv_index]
        temp_test_sorted = temp_test.sort_values("pred_test",ascending = True)
        for topk in topks:
            temp_select = temp_test_sorted[:topk]['y_test']
            temp_avgLabel = temp_select.mean()
            temp_stdLabel = temp_select.std()
            temp_min = temp_select.min()
            temp_max = temp_select.max()
            temp_simple_avg = temp_test[['y_test']].mean()
            eval_df.loc[len(eval_df)] = [str(csv_index),topk,temp_avgLabel,temp_stdLabel
                                        ,temp_min,temp_max,temp_simple_avg]
    return eval_df

#####################
## define global parameters
Params = {}
Params['algo'] = ['model_lgb'] # 可选参数：lasso,model_lgb
# lasso params
Params['lasso_grid_params'] = dict(scaler=[StandardScaler()]
                              ,lasso__alpha=[0.0001,0.0005,0.001,0.002,0.005,0.01,0.05])
# lgb params
Params['model_lgb_grid_params'] = {
    'learning_rate': [0.01,0.02]
    ,'n_estimators': [8,24,48]
    ,'num_leaves': [6,12]
    ,'boosting_type' : ['gbdt']
    ,'objective' : ['regression']
    ,'seed' : [500]
    ,'colsample_bytree' : [0.65, 0.75]
    ,'subsample' : [0.7,0.75]
#    ,'reg_alpha' : [1,2,6]
#    ,'reg_lambda' : [1,2,6]
    }
Params['topK'] = 50 # 选股个数
Params['topK_params'] = 1 # 前k个参数用于实际测试 

result_list = []
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
                      ('lasso',Lasso(alpha = 0.01,random_state=1))])
######
# LightGBM
######
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
#grid search params
for algo in Params['algo']:
    imp_print(algo,20)
    param_grid = Params[algo + '_grid_params']
    estimator = eval(algo)
    # grid_search
    # each csv is one test set
    custom_cv = PredefinedSplit([-1]*ntrain+[1]*ntest)
    customized_cv = [(range(ntrain),range(ntrain+ntest))]
    # get csv_index in the test set for gridsearch evaluation
    test_csvIndexs = test_csv_index.unique()
    
    reggressor = GridSearchCV(estimator
                               , param_grid=param_grid
    #                           ,scoring = 'neg_mean_squared_error'
                               ,scoring = top50_avg_loss
                               ,cv = custom_cv
                               ,refit = False
                               ,n_jobs=-1
                               ,verbose=1
                               ,return_train_score=False)
    reggressor.fit(all_data.values,y_all_data)
    
    print('grid search result:')
    print('best score:'+ str(reggressor.best_score_))
    print('best params:'+ str(reggressor.best_params_))
    print('detailed results:',report(reggressor.cv_results_))
    rank_params = pd.DataFrame(columns=['Rank'
                                        ,'CV_score'
                                        ,'params'])
    for rank in range(1,Params['topK_params']+1):
        report_params(reggressor.cv_results_,rank,rank_params)
           
    #####################
    # # Test: 测试获取评价结果
    #####################
    imp_print("Testing...",40)
    for rank in range(Params['topK_params']):
        temp_params = rank_params['params'][rank]
        print('CV rank {} model:\n\
              params:{}'.format(rank+1,temp_params))
        eval_df = evaluate_test(model=estimator,params=temp_params)
    
        print('simple_avg:{}'.format(eval_df['simple_avg'].mean()))
        
        for topk in eval_df['topk'].unique():
            print('top'+str(int(topk))+' avg:{}'.format(str(eval_df['pred_avg'][eval_df['topk']==topk].mean())))
    
print ("Finished...")