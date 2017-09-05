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
import os,gc
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
    Params['theme'] = 'OD_None_Test_Algo_model_lgb_rate_nTree_leave_loss_mbin'# 本次运行的目的
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
                                          ,'boosting_type' : 'gbdt'
                                          ,'save_binary':True
                                          ,'bagging_fraction':0.8
                                          ,'bagging_freq':5
                                          ,'min_data_in_leaf':100
                                          ,'min_sum_hessian_in_leaf':10
                                          #### found:
                                          ,'feature_fraction':0.6
                                          ,'reg_alpha':2
                                          ,'reg_lambda':1
                                          } 
    Params['model_lgb_grid_params'] = {
        'learning_rate': [0.01,0.02]
        ,'n_estimators': [1300,1600,2000]
        ,'num_leaves': [45,60,75]
        ,'objective' : ['regression','huber']
        ,'max_bin':[50,100]
        }
    Params['model_lgb_grid_params_filter'] = [
            {'learning_rate':[0.01],'objective':['huber']},
            {'num_leaves':[75],'objective':['huber']},
            {'max_bin':[100],'objective':['huber']}
            ]
    '''
    untested params:
        subsample
        subsample_freq
        colsample_bytree: not useful?
    '''
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
    Params['theme'] = 'MultiVSsingle_test'# 本次运行的目的
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
                                          ,'min_data_in_leaf':100
                                          ,'min_sum_hessian_in_leaf':10
                                          } 
    Params['model_lgb_grid_params'] = {
        'learning_rate': [0.02]
        ,'n_estimators': [1500]
        ,'num_leaves': [45]
        ,'objective' : ['regression']
        ,'feature_fraction':[0.4,0.6]
        ,'reg_alpha' : [1]
        ,'reg_lambda' : [1]
        ,'subsample':[1]
        ,'subsample_freq':[1]
        ,'colsample_bytree':[1]
        ,'bagging_fraction':[1]
        ,'bagging_freq':[5]
        }
    Params['model_lgb_grid_params_filter'] = [
            {'learning_rate':[0.02],'n_estimators':[800,1000]},
            {'learning_rate':[0.05,0.08],'n_estimators':[1600,2000]}
            ]
    ########## Evaluation params
    Params['topK'] = 50 # 选股个数
    return Params

def paramGridSearch(params):
    params_combs = ParameterGrid(params)
    return params_combs

def load_params_combs(theme,stage,train_name_raw,params,params_filter=[],continue_mode = True):
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
        if params_filter: # if needs parameter filter
            ind = 1
            for param in param_combs:
                filtered = 0
                for params_filter_comb in params_filter:
                    filter_flag = 1
                    for filter_key in params_filter_comb.keys():
                        filter_flag = filter_flag * (param[filter_key] in params_filter_comb[filter_key])
                    if(filter_flag == 1):
                        filtered = 1
                        continue
                if (filtered == 0):
                    param_combs_df.loc[len(param_combs_df)] = [ind,param,0]
                    ind+=1
        else:
            ind = 1
            for param in param_combs:
                param_combs_df.loc[len(param_combs_df)] = [ind,param,0]
                ind+=1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            param_combs_df.to_hdf(full_path,'grid_search_params')
            del param_combs
    print('load_params_combs garbage collection:{}'.format(gc.collect()))
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
    del param_combs_df
    print('update_params_combs garbage collection:{}'.format(gc.collect()))
    
if __name__ == "__main__":
    params_od = {'max_samples':[0.7,0.8,0.9]
                 ,'n_estimators':[100]
                 ,'contamination':[0.1]} # 0.1
    params_lasso = dict(scaler=[StandardScaler()]
                                  ,lasso__alpha=[0.0001,0.0005,0.001,0.002,0.005,0.01,0.05])
    params ={}
    params_combs = load_params_combs('test','train_1331_1333.h5',params,[])
    
    print ("Finished...")
