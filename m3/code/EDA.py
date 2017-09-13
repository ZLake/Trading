#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:45:43 2017

@author: lakezhang
"""
import sys
sys.path.insert(0, 'functions')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

from simple_functions import imp_print
from params import get_params
# Explore Data Analysis

def EDA():
    Params = get_params()
    rng = np.random.RandomState(42)
    imp_print("Data Loading...",40)
    # get dataset filename
    train_name_raw = Params['train_name_raw']
    test_name_raw =Params['test_name_raw']
    train= pd.read_hdf('DataSet/'+ train_name_raw,engine = 'c',memory_map=True)
    test = pd.read_hdf('DataSet/'+ test_name_raw,engine = 'c',memory_map=True)
    all_data = pd.concat([train,test])
    #Save the 'csv_index' column
    train_csv_index = train[train.columns[0]].copy()
    test_csv_index = test[train.columns[0]].copy()
    #Save the 'Id' column
    train_ID = train[train.columns[1]].copy()
    test_ID = test[train.columns[1]].copy()
    #Now drop the  'csv_index' & 'Id' colum since it's unnecessary for  the prediction process.
#    train.drop(train.columns[0:2], axis = 1, inplace = True)
#    test.drop(test.columns[0:2], axis = 1, inplace = True)
    # 多少个stock
    train_stocks = train_ID.unique()
    test_stocks = test_ID.unique()
    print('Train set contain stock number:{}'.format(len(train_stocks)))
    print('Test set contain stock number:{}'.format(len(test_stocks)))
    print('Train & Test set both contain stock number:{}'.format(len(set(train_stocks)&set(test_stocks))))
    # 计算每天(全量数据)的target 均值，std
    # 这里是全部数据，包含测试，因为通过csv_index区分了不同的集合
    market_df = all_data[['csv_index', 2]].groupby('csv_index').agg([np.mean, np.std, len]).reset_index()
    # plot 
    t      = market_df['csv_index'].astype(int)
    y_mean = np.array(market_df[2]['mean'])
    y_std  = np.array(market_df[2]['std'])
    n      = np.array(market_df[2]['len'])
    
    plt.figure()
    plt.plot(t, y_mean, '.')
    plt.xlabel('csv_index')
    plt.ylabel('mean of y')
    
    plt.figure()
    plt.plot(t, y_std, '.')
    plt.xlabel('csv_index')
    plt.ylabel('std of y')
    
    plt.figure()
    plt.plot(t, n, '.')
    plt.xlabel('csv_index')
    plt.ylabel('portfolio size')
    plt.show()
    
    # 计算特征和target的相关系数
    train_person = train.corr(method='pearson')
    train_spearman = train.corr(method='pearson')
    print('')
if __name__ == "__main__":
    EDA()
    print ("Finished...")
    