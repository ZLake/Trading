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
    # theme
    Params['theme'] = 'time_weight_decay_1268_1311'# 本次运行的目的
    # grid search continue or reset:
    Params['OD_continue']= True
    Params['Algo_continue']= True
    # data files
    if(multiprocessing.cpu_count() >=60):
        Params['train_name_raw'] = 'train_1268_1311.h5'
        Params['test_name_raw'] = 'test_1268_1311.h5'
    else:
        Params['train_name_raw'] ='train_1332_1333.h5'
        Params['test_name_raw'] = 'test_1332_1333.h5'
    # preprocess suffix
    Params['Proc'] = False
    Params['procSuffix'] = '_normalized_fea_label' # _normalized_fea_label,
    # feature selection
    Params['FeaSelect'] = False
    Params['IMPDF'] =  'Preprocess/feature_selection/New_data_gridSearch_4__1268_1311_Model_19_feaImp.h5'


    # OD_IF_Test_Algo_lasso
    ########## Use Sample Weight
    Params['Sample_weight'] = True
    Params['Decay_algo'] = 'exp' # exp
    Params['Decay_params'] = {'decay_constant':[0,0.007,0.0008,0.009]} #0.0008,0.0012
    Params['Sample_weight_algo'] = ['model_lgb']#支持样本权重的算法
    ########## Select Train data start time
    Params['Train_start_time'] = [0] # 请从小到大输入，否则会出现问题
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
                                          ,'min_data_in_leaf':100
                                          ,'min_sum_hessian_in_leaf':10
                                          #### found:
                                          ,'reg_alpha':3
                                          ,'reg_lambda':1
                                          ,'max_bin':100  # can be more
                                          ,'n_estimators': 2000# can be more
                                          ,'learning_rate':0.02 # can be less
                                          ,'num_leaves':45
                                          ,'bagging_fraction':0.8
                                          ,'bagging_freq':5
                                          ,'feature_fraction':0.6
                                          } 
    Params['model_lgb_grid_params'] = {
            'n_estimators':[1600,2000,2200]
            ,'learning_rate':[0.02,0.025]
            ,'num_leaves':[45,60]
        }
    Params['model_lgb_grid_params_filter'] = [
            {'n_estimators':[1200],'learning_rate':[0.02]}
            ,{'n_estimators':[2000,2500],'learning_rate':[0.06]}
#            ,{'n_estimators':[2500],'reg_lambda':[0.5]}
            ]
    '''
    untested params:
        subsample: not useful?
        subsample_freq: not useful?
        colsample_bytree: not useful?
        subsample_for_bin:not useful?
    '''
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
