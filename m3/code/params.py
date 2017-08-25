#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:51:13 2017

@author: lakezhang
"""
import multiprocessing
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.model_selection import ParameterGrid
import pandas as pd
import os

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
    IF_Grid_Params = {'max_samples':[0.7,0.8]}
    LOF_Params = {'n_neighbors':20
                  ,'algorithm':'ball_tree'
                  ,'leaf_size':30
                  ,'metric':'minkowski'
                  ,'p':2
                  ,'contamination':0.1
                    }
    LOF_Grid_Params = {}
    Params['Outlier_Detector'] = {'algo':'None'                 # None,IF:IsolationForest,LOF
                                  ,'apply_on_test':True
                                  ,'IF_Params':IF_Params
                                  ,'IF_Grid_Params':IF_Grid_Params
                                  ,'LOF_Params':LOF_Params
                                  ,'LOF_Grid_Params':LOF_Grid_Params}
    ########## Modeling parmas
    Params['algo'] = ['lasso'] # 可选参数： lasso,model_lgb
    ######## 注意： 因为内存不够问题，lasso的StandardScaler(copy=False)，会对train 做inplace 替换！！！
    # lasso params
    Params['lasso_default_params'] = {}
    Params['lasso_grid_params'] = dict(scaler=[StandardScaler()]
                                  ,lasso__alpha=[0.001,0.002,0.005])
    # lgb params
    Params['model_lgb_default_params'] = {} 
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

def paramGridSearch(params):
    params_combs = ParameterGrid(params)
    return params_combs

def load_params_combs(theme,train_name_raw,params,continue_mode = True):
    file_name = train_name_raw.strip('.h5').strip('train_')
    full_path = 'Result/'+file_name+'/'+theme+'__' + 'param_combs.h5'
    if not os.path.exists('Result/'+file_name):
        os.makedirs('Result/'+file_name)
    if (os.path.exists(full_path) and continue_mode):
        print('Get parameter combinations from: ' + full_path)
        param_combs_df = pd.read_hdf(full_path,engine = 'c')
    else:
        print('Generating parameter combinations...')
        param_combs_df = pd.DataFrame(columns=['NO.','params','status'])
        param_combs = paramGridSearch(params)
        ind = 1
        for param in param_combs:
            param_combs_df.loc[len(param_combs_df)] = [ind,param,0]
            ind+=1
    return param_combs_df
        
if __name__ == "__main__":
    params_od = {'max_samples':[0.7,0.8]
                 ,'n_estimators':[100]
                 ,'contamination':[0.1]} # 0.1
    params_lasso = dict(scaler=[StandardScaler()]
                                  ,lasso__alpha=[0.0001,0.0005,0.001,0.002,0.005,0.01,0.05])
    params ={}
    params_combs = load_params_combs('test','train_1331_1333.h5',params)
    
    print ("Finished...")
