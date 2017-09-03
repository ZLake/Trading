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
import warnings


def get_params():
    #####################
    ## define global parameters
    Params = {}
    # grid search continue or reset:
    Params['OD_continue']= True
    Params['Algo_continue']= True
    # data files
    if(multiprocessing.cpu_count() >=60):
        Params['train_name_raw'] = 'train_1200_1333.h5'
        Params['test_name_raw'] = 'test_1200_1333.h5'
    else:
        Params['train_name_raw'] ='train_1331_1333.h5'
        Params['test_name_raw'] = 'test_1331_1333.h5'
    # theme
    Params['theme'] = 'OD_None_Test_Algo_model_lgb_rate_nTree'# 本次运行的目的
    # OD_IF_Test_Algo_lasso
    ########## Outlier detection params
    IF_Params = {'max_samples':0.7
                 ,'n_estimators':100
                 ,'contamination':0.1} # 0.1
    IF_Grid_Params = {'max_samples':[0.7]
                        ,'n_estimators':[100]}
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
    Params['algo'] = ['model_lgb'] # 可选参数： lasso,model_lgb
    # lasso params
    Params['lasso_default_params'] = {'scaler':StandardScaler()                     
                                        ,'lasso__alpha': 0.01
                                        ,'max_iter':2000
                                        }
    Params['lasso_grid_params'] = dict(scaler=[StandardScaler()]  # None,StandardScaler()
                                  ,lasso__alpha=[0.001,0.002,0.005,0.01,0.02,0.05,0.08]
                                  )
    # lgb params
    Params['model_lgb_default_params'] = {'objective':'regression'
                                          ,'max_bin':50
                                          ,'boosting_type' : 'gbdt'
                                          ,'save_binary':True
                                          ,'bagging_fraction':0.8
                                          ,'bagging_freq':5
                                          ,'min_data_in_leaf':100
                                          ,'min_sum_hessian_in_leaf':10
                                          } 
    Params['model_lgb_grid_params'] = {
        'learning_rate': [0.02,0.05,0.08]
        ,'n_estimators': [1000,1300,1600,2000]
        ,'num_leaves': [30,45,60]
        ,'objective' : ['regression']
        ,'feature_fraction':[0.6,0.8]
        ,'reg_alpha' : [2]
        ,'reg_lambda' : [1]
        }
    ########## Evaluation params
    Params['topK'] = 50 # 选股个数
    return Params

def get_params2():
    #####################
    ## define global parameters
    Params = {}
    # grid search continue or reset:
    Params['OD_continue']= True
    Params['Algo_continue']= True
    # data files
    if(multiprocessing.cpu_count() >=60):
        Params['train_name_raw'] = 'train_1200_1333.h5'
        Params['test_name_raw'] = 'test_1200_1333.h5'
    else:
        Params['train_name_raw'] ='train_1331_1333.h5'
        Params['test_name_raw'] = 'test_1331_1333.h5'
    # theme
    Params['theme'] = 'OD_IF_Test_Algo_model_lgb'# 本次运行的目的
    # OD_IF_Test_Algo_lasso
    ########## Outlier detection params
    IF_Params = {'max_samples':0.7
                 ,'n_estimators':100
                 ,'contamination':0.1} # 0.1
    IF_Grid_Params = {'max_samples':[0.7]
                        ,'n_estimators':[100]
                        ,'contamination':[0.08,0.1,0.12]}
    LOF_Params = {'n_neighbors':20
                  ,'algorithm':'ball_tree'
                  ,'leaf_size':30
                  ,'metric':'minkowski'
                  ,'p':2
                  ,'contamination':0.1
                    }
    LOF_Grid_Params = {}
    Params['Outlier_Detector'] = {'algo':'IF'                 # None,IF:IsolationForest,LOF
                                  ,'apply_on_test':True
                                  ,'IF_Params':IF_Params
                                  ,'IF_Grid_Params':IF_Grid_Params
                                  ,'LOF_Params':LOF_Params
                                  ,'LOF_Grid_Params':LOF_Grid_Params}
    ########## Modeling parmas
    Params['algo'] = ['model_lgb'] # 可选参数： lasso,model_lgb
    # lasso params
    Params['lasso_default_params'] = {'scaler':StandardScaler()                     
                                        ,'lasso__alpha': 0.01
                                        ,'max_iter':2000
                                        }
    Params['lasso_grid_params'] = dict(scaler=[StandardScaler()]  # None,StandardScaler()
                                  ,lasso__alpha=[0.001,0.002,0.005,0.01,0.02,0.05,0.08]
                                  )
    # lgb params
    Params['model_lgb_default_params'] = {'objective':'regression'
                                          ,'max_bin':50
                                          ,'boosting_type' : 'gbdt'
                                          ,'save_binary':True
                                          ,'bagging_fraction':0.8
                                          ,'bagging_freq':5
                                          ,'min_data_in_leaf':100
                                          ,'min_sum_hessian_in_leaf':10
                                          } 
    Params['model_lgb_grid_params'] = {
        'learning_rate': [0.02]
        ,'n_estimators': [1500]
        ,'num_leaves': [30,45]
        ,'objective' : ['regression']
        ,'colsample_bytree' : [0.6,0.8]
        ,'feature_fraction':[0.4,0.6]
        ,'reg_alpha' : [1,2]
        ,'reg_lambda' : [1,2]
        }
    ########## Evaluation params
    Params['topK'] = 50 # 选股个数
    return Params

def paramGridSearch(params):
    params_combs = ParameterGrid(params)
    return params_combs

def load_params_combs(theme,stage,train_name_raw,params,continue_mode = True):
    file_name = train_name_raw.strip('.h5').strip('train_')
    full_path = 'Result/'+file_name+'/'+theme+'__' + stage + '_param_combs.h5'
    if not os.path.exists('Result/'+file_name):
        os.makedirs('Result/'+file_name)
    if (os.path.exists(full_path) and continue_mode):
        print('Get parameter combinations from: ' + full_path)
        param_combs_df = pd.read_hdf(full_path,engine = 'c')
    else:
        print('Generating parameter combinations...')
        param_combs_df = pd.DataFrame(columns=['NO.','params','status']) # status: 0:undone; 1:done
        param_combs = paramGridSearch(params)
        ind = 1
        for param in param_combs:
            param_combs_df.loc[len(param_combs_df)] = [ind,param,0]
            ind+=1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            param_combs_df.to_hdf(full_path,'grid_search_params')
    return param_combs_df
        
def update_params_combs(theme,train_name_raw,stage,ind):
    # add done flag to finished grid param
    file_name = train_name_raw.strip('.h5').strip('train_')
    full_path = 'Result/'+file_name+'/'+theme+'__' + stage + '_param_combs.h5'
    param_combs_df = pd.read_hdf(full_path,engine = 'c')
    param_combs_df.loc[param_combs_df['NO.']==ind,'status'] = 1 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        param_combs_df.to_hdf(full_path,'grid_search_params')
    print('stage:{} NO.{} is done...'.format(stage,ind))
    
if __name__ == "__main__":
    params_od = {'max_samples':[0.7,0.8,0.9]
                 ,'n_estimators':[100]
                 ,'contamination':[0.1]} # 0.1
    params_lasso = dict(scaler=[StandardScaler()]
                                  ,lasso__alpha=[0.0001,0.0005,0.001,0.002,0.005,0.01,0.05])
    params ={}
    params_combs = load_params_combs('test','train_1331_1333.h5',params)
    
    print ("Finished...")
