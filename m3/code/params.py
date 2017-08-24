#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:51:13 2017

@author: lakezhang
"""
import multiprocessing
from sklearn.preprocessing import RobustScaler,StandardScaler

def get_params():
    #####################
    ## define global parameters
    Params = {}
    # data files
    if(multiprocessing.cpu_count() >=60):
        Params['train_name_raw'] = 'train_1200_1333.h5'
        Params['test_name_raw'] = 'test_1200_1333.h5'
    else:
        Params['train_name_raw'] ='train_1331_1333.h5'
        Params['test_name_raw'] = 'test_1331_1333.h5'
    # theme
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
                                  ,'apply_on_test':False
                                  ,'IF_Params':IF_Params
                                  ,'LOF_Params':LOF_Params}
    ########## Modeling parmas
    Params['algo'] = ['model_lgb'] # 可选参数： lasso,model_lgb
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
    return Params
