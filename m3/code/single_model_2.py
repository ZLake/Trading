#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 22:38:21 2017

@author: LinZhang
"""
import numpy as np # linear algebra
import scipy as sp
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dask.dataframe as dd

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV,PredefinedSplit
from sklearn.metrics import make_scorer
from sklearn.externals import joblib # solve OOM problem


import multiprocessing
from time import localtime, strftime

import lightgbm as lgb
import xgboost as xgb

import time
import gc
import os
import warnings


from simple_functions import imp_print
from outlier_detection import outlier_detection
from evaluation import evaluate_test,store_result


def training():
    #####################
    ## define global parameters
    Params = {}
    Params['theme'] = 'ODTest'# 本次运行的目的
    ########## Outlier detection params
    IF_Params = {'max_samples':0.7
                 ,'n_estimators':100
                 ,'contamination':0.1} # 0.1
    LOF_Params = {'n_neighbors':20
                  ,'algorithm':'ball_tree'
                  ,'leaf_size':30
                  ,'metric':'minkowski'
                  ,'p':2
                  ,'contamination':0.1
                    }
    Params['Outlier_Detector'] = {'algo':'IF'                 # None,IF:IsolationForest,LOF
                                  ,'apply_on_test':True
                                  ,'IF_Params':IF_Params
                                  ,'LOF_Params':LOF_Params}
    ########## Modeling parmas
    Params['algo'] = ['lasso'] # 可选参数： lasso,model_lgb
    ######## 注意： 因为内存不够问题，lasso的StandardScaler(copy=False)，会对train 做inplace 替换！！！
    # lasso params
    Params['lasso_grid_params'] = dict(scaler=[StandardScaler()]
                                  ,lasso__alpha=[0.0001,0.0005,0.001,0.002,0.005,0.01,0.05])
    # lgb params
    Params['model_lgb_grid_params'] = {
        'learning_rate': [0.02]
        ,'n_estimators': [48]
        ,'num_leaves': [18]
        ,'boosting_type' : ['gbdt']
        ,'objective' : ['regression']
        ,'seed' : [500]
        ,'colsample_bytree' : [0.6]
        ,'subsample' : [0.75]
    #    ,'reg_alpha' : [1,2,6]
    #    ,'reg_lambda' : [1,2,6]
        }
    ########## Evaluation params
    Params['topK'] = 50 # 选股个数
    rng = np.random.RandomState(42)
    if multiprocessing.cpu_count() >=60:
        num_threads = multiprocessing.cpu_count()//2
    else:
        num_threads = multiprocessing.cpu_count()
    #####################
    # Read the data: 选择数据的时间段
    #####################
    imp_print("Data Loading...",40)
    read_start = time.time()
    # 数据格式 hdf5
    # Sample 100 rows of data to determine dtypes.
#    df_test = pd.read_hdf('DataSet/train_1200_1333.h5', nrows=10)
#    float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
#    float32_cols = {c: np.float32 for c in float_cols}
#    chunk_size = 10**5
#    train =pd.concat(chunck_df for chunck_df in pd.read_hdf('DataSet/train_1331_1333.h5',iterator=True, chunksize=chunk_size))
#    test = pd.concat(chunck_df for chunck_df in pd.read_hdf('DataSet/test_1331_1333.h5',iterator=True, chunksize=chunk_size))
    if(multiprocessing.cpu_count() >=60):
        train_name_raw = 'train_1200_1333.h5'
        test_name_raw ='test_1200_1333.h5'
    else:
        train_name_raw = 'train_1331_1333.h5'
        test_name_raw ='test_1331_1333.h5'
    train= pd.read_hdf('DataSet/'+ train_name_raw,engine = 'c')
    test = pd.read_hdf('DataSet/'+ test_name_raw,engine = 'c')

#    train= dd.read_csv('../input/*.csv')
#    test = dd.read_csv('../input/*.csv')
    # 选择数据时间段：todo
#    train = train_raw
#    test = test_raw
#    del train_raw,test_raw
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
    read_end = time.time()
    print("\nThe train data size after dropping Id feature is : {} ".format(train.shape))
    print("The test data size after dropping Id feature is : {} ".format(test.shape))
    #
#    gc.collect()
#    if(gc.collect()>0):
#        print('garbage collection:')
#        print(gc.collect())
    #####################
    # Preprocess: 处理成训练和测试集合
    #####################
    imp_print("Data Processing...",40)
    proc_start = time.time()
    #############
    ntrain = train.shape[0]
    ntest = test.shape[0]
    y_train = train[train.columns[0]].values
    y_test = test[test.columns[0]].values
#    all_data = pd.concat((train, test)).reset_index(drop=True)
#    y_all_data = all_data[all_data.columns[0]].values
    train.drop(train.columns[0], axis=1, inplace=True)
    test.drop(test.columns[0], axis=1, inplace=True)
#    all_data.drop(train.columns[0], axis=1, inplace=True)
#    del train,test
#    print("all_data size is : {}".format(all_data.shape))
    # missing data
#    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
#    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
#    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
#    missing_data.head(20)
#    print("missing data column numbers:{}".format(missing_data.shape[0]))
#    if(missing_data.shape[0] == 0):
#        imp_print("no missing data, go to next step...")
#    else:
#        imp_print("Need filling missing data...")
    # Outlier Detection
    if(Params['Outlier_Detector']['algo']!='None'):
        train,y_train,test,y_test,test_csv_index = outlier_detection(train_name_raw,test_name_raw
                                                                 ,Params['Outlier_Detector']['algo'],Params['Outlier_Detector']
                                                                 ,train,y_train,test,y_test,test_csv_index
                                                                 ,apply_on_test = Params['Outlier_Detector']['apply_on_test']
                                                                 ,num_threads = num_threads)
    else:
        print('None outlier detection is applied...')

#
    proc_end = time.time()
    #
#    gc.collect()
#    if(gc.collect()>0):
#        print('garbage collection:')
#        print(gc.collect())
    #####################
    # Modeling: 建模
    #####################
    imp_print("Modeling...",40)
    model_start = time.time()
    # get the train and val and test data
    train = train.values
    test = test.values
#    del all_data
    ######
    # Lasso Regression
    ######
    scaler = StandardScaler(copy=False)
    #scaler = RobustScaler()
    lasso = Pipeline(steps=[('scaler',scaler),
                          ('lasso',Lasso(alpha = 0.01,random_state=rng
                                         ,copy_X = False))])

    ######
    # LightGBM
    ######

    print('number of thread in training lgb:{}'.format(num_threads))
    model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                                  learning_rate=0.02, n_estimators=1500,
                                  max_bin = 55, bagging_fraction = 0.8,
                                  bagging_freq = 5, feature_fraction = 0.2319,
                                  feature_fraction_seed=9, bagging_seed=9,
                                  min_data_in_leaf =6, min_sum_hessian_in_leaf = 11,
                                  num_threads = num_threads,
                                  reg_alpha=1, reg_lambda=1,
                                  )
    #grid search params
    for algo in Params['algo']:
        imp_print(algo,20)
        estimator = eval(algo)
        #####################
        # # Test: 测试获取评价结果
        #####################
        imp_print("Testing...",40)
        eval_df = evaluate_test(estimator,train,y_train,test,y_test,test_csv_index)

        print('simple_avg:{}'.format(eval_df['simple_avg'].mean()))

        for topk in eval_df['topk'].unique():
            print('top'+str(int(topk))+' avg:{}'.format(str(eval_df['pred_avg'][eval_df['topk']==topk].mean())))
        store_result(Params,algo,eval_df,estimator,train_name_raw,test_name_raw,Params['theme'])

    model_end = time.time()

    imp_print('Execution Time:')
    print("Reading data time cost: {}s".format(read_end - read_start))
    print("Processing data time cost: {}s".format(proc_end - proc_start))
    print("Modeling & Test data time cost: {}s".format(model_end - model_start))

if __name__ == "__main__":
    training()
    print ("Finished...")
