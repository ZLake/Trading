#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 22:38:21 2017

# 可以GridSearch最优参数
@author: LinZhang
"""
import sys
sys.path.insert(0, 'functions')

import numpy as np # linear algebra
import scipy as sp
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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

from params import get_params,get_params2,load_params_combs,update_params_combs
from simple_functions import imp_print
from outlier_detection import outlier_detection,outlier_detection_grid
from models import get_model
from evaluation import evaluate_test,store_result

def training():
    Params = get_params2()
    Params['theme'] = Params['theme'] + '2'
    rng = np.random.RandomState(42)
    if multiprocessing.cpu_count() >=60:
        num_threads = multiprocessing.cpu_count()//2
    else:
        num_threads = multiprocessing.cpu_count()
    # get dataset filename
    train_name_raw = Params['train_name_raw']
    test_name_raw =Params['test_name_raw']
    #####################
    # Get Outlier Detection Parameters：获取OD参数
    #####################
    if (Params['Outlier_Detector']['algo']!= 'None'):
        OD_Grid_Params =  Params['Outlier_Detector'][Params['Outlier_Detector']['algo']+'_Grid_Params']
        OD_Grid_Params_combs = load_params_combs(Params['theme']
                                                ,'OD'
                                                ,train_name_raw
                                                ,OD_Grid_Params
                                                ,[]
                                                ,continue_mode = Params['OD_continue'])
        OD_Grid_Params_combs_undone = OD_Grid_Params_combs[OD_Grid_Params_combs['status']==0]
    else:
        OD_Grid_Params = {'algo':['None']}
        OD_Grid_Params_combs = load_params_combs(Params['theme']
                                                ,'OD_None'
                                                ,train_name_raw
                                                ,OD_Grid_Params
                                                ,[]
                                                ,continue_mode = Params['OD_continue'])
        OD_Grid_Params_combs_undone = pd.DataFrame(columns=['NO.','params','status']);
        OD_Grid_Params_combs_undone.loc[len(OD_Grid_Params_combs_undone)] = [1,'None',0]
    #### Loop the ODParams:
    for index,OD_row in OD_Grid_Params_combs_undone.iterrows():
        OD_grid_param = OD_row['params']
        imp_print('The {}th OD round'.format(OD_row['NO.']))
        print('OD Param:{}'.format(OD_grid_param))
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
    
        train= pd.read_hdf('DataSet/'+ train_name_raw,engine = 'c',memory_map=True)
        test = pd.read_hdf('DataSet/'+ test_name_raw,engine = 'c',memory_map=True)
    
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
        train_csv_index = train[train.columns[0]].copy()
        test_csv_index = test[train.columns[0]].copy()
        #Save the 'Id' column
        train_ID = train[train.columns[1]].copy()
        test_ID = test[train.columns[1]].copy()
        #Now drop the  'csv_index' & 'Id' colum since it's unnecessary for  the prediction process.
        train.drop(train.columns[0:2], axis = 1, inplace = True)
        test.drop(test.columns[0:2], axis = 1, inplace = True)
        #check again the data size after dropping the 'Id' variable
        read_end = time.time()
        print("\nThe train data size after dropping Id feature is : {} ".format(train.shape))
        print("The test data size after dropping Id feature is : {} ".format(test.shape))
        
        print('garbage collection:{}'.format(gc.collect()))
    
        #####################
        # Preprocess: 处理成训练和测试集合
        #####################
        imp_print("Data Processing...",40)
        proc_start = time.time()
        #############
    #    ntrain = train.shape[0]
    #    ntest = test.shape[0]
        y_train = train[train.columns[0]].copy().values
        y_test = test[test.columns[0]].copy().values
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
    
        print('garbage collection:{}'.format(gc.collect()))
        # Outlier Detection
        if(Params['Outlier_Detector']['algo']!='None'):
            train,y_train,test,y_test,test_csv_index = outlier_detection_grid(train_name_raw,test_name_raw
                                                                     ,Params['Outlier_Detector']['algo']
                                                                     ,Params['Outlier_Detector']
                                                                     ,OD_grid_param
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
        print('garbage collection:{}'.format(gc.collect()))
    
        #grid search params
        for algo in Params['algo']:
            # get algo param grid
            algo_Grid_Params = Params[algo+'_grid_params']
            algo_Grid_Params_filter = Params[algo+'_grid_params_filter']
            algo_param_combs = load_params_combs(Params['theme']
                                                ,'OD_{}_Model'.format(OD_row['NO.'])
                                                ,train_name_raw
                                                ,algo_Grid_Params
                                                ,algo_Grid_Params_filter
                                                ,continue_mode = (Params['Algo_continue'] or Params['OD_continue']))
            algo_Grid_Params_combs_undone = algo_param_combs[algo_param_combs['status']==0]
            #### Loop the ODParams:
            for index,algo_row in algo_Grid_Params_combs_undone.iterrows():
                algo_grid_param = algo_row['params']
                imp_print('The {}th Algo({}) round'.format(algo_row['NO.'],algo))
                print('Algo({}) Param:{}'.format(algo,algo_grid_param))
                temp_time_start = time.time()
                imp_print(algo,20)
                #define classifier:
                default_param = Params[algo+'_default_params']
                estimator = get_model(algo,default_param,rng,num_threads)
                estimator.set_params(**algo_grid_param)
#                estimator = eval(algo)
                #####################
                # # Test: 测试获取评价结果
                #####################
                imp_print("Testing...",40)
                eval_df = evaluate_test(estimator,train,y_train,test,y_test,test_csv_index)
        
                print('simple_avg:{}'.format(eval_df['simple_avg'].mean()))
        
                for topk in eval_df['topk'].unique():
                    print('top'+str(int(topk))+' avg:{}'.format(str(eval_df['pred_avg'][eval_df['topk']==topk].mean())))
                temp_time_end = time.time()
                cost_time = (temp_time_end-temp_time_start)/60                # min
                store_result(Params,algo_grid_param,algo,eval_df,estimator
                             ,train_name_raw,test_name_raw,Params['theme'],cost_time)
                print('Cost time:{}'.format(cost_time))
                #update done info for grid search:algo
                update_params_combs(Params['theme'],train_name_raw
                                    ,'OD_{}_Model'.format(OD_row['NO.'])
                                    ,algo_row['NO.'])
                del estimator,eval_df
                print('garbage collection:{}'.format(gc.collect()))

            imp_print('Grid search on algo:{} is finished...'.format(algo))
            print('garbage collection:{}'.format(gc.collect()))
        
        model_end = time.time()

        imp_print('Execution Time:')
        print("Reading data time cost: {}s".format(read_end - read_start))
        print("Processing data time cost: {}s".format(proc_end - proc_start))
        print("Modeling & Test data time cost: {}s".format(model_end - model_start))
        
                
        #update done info for grid search:outlier detection
        if (Params['Outlier_Detector']['algo']!= 'None'):
            update_params_combs(Params['theme'],train_name_raw,'OD',OD_row['NO.'])
        else:
            update_params_combs(Params['theme'],train_name_raw,'OD_None',OD_row['NO.'])
        
        
if __name__ == "__main__":
    training()
    print ("Finished...")
