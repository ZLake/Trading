#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 10:37:35 2017

@author: LinZhang
"""

#测试内存使用
import sys
sys.path.insert(0, 'functions')

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

from params import get_params
from simple_functions import imp_print
from outlier_detection import outlier_detection
from evaluation import evaluate_test,store_result
def training():
    Params = get_params()
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

    train_name_raw = Params['train_name_raw']
    test_name_raw =Params['test_name_raw']
    train= pd.read_hdf('DataSet/'+ train_name_raw,engine = 'c',memory_map=True)
    test = pd.read_hdf('DataSet/'+ test_name_raw,engine = 'c',memory_map=True)
    
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

    print('Data loaded...')
    time.sleep(3600)

if __name__ == "__main__":
    training()
    print ("Finished...")
