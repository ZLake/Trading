#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 09:59:04 2017

@author: lakezhang
"""
import pandas as pd
from params import get_params
from read_data import restore_with_chunks
import numpy as np
import os,gc

# 处理data，并保存起来
def normalize_fea_label(data,fname,train_mode = 0):
    """
    data:归一化数据输入
    train_mode:是否需要对label做归一化: 
        0: 训练模式，归一化lable;
        1：cv测试模式，不归一化label;
        2:真实test模式，pred用
    """
    if(train_mode == 0):
        print('Train mode:')
        # 统计数据，为归一化做准备
        file_name = fname.strip('.h5').strip('train_')
        if not os.path.exists('Result/'+file_name):
            os.makedirs('Result/'+file_name)        
        train_label_stat = data.groupby([data.columns[0]])[data.columns[2:]].agg(['mean','std','min','max'])
        train_label_stat.to_hdf('Result/'+file_name + '/{}_train_label_stat.h5'.format(fname.strip('.h5')),fname,append=False)
        print('Stat file generated')
        counter = 0
        for column_name in data.columns[2:]:
            idx = np.where(data.columns.values==column_name)[0][0]
            train_label_withStat = pd.merge(data[data.columns[[0,idx]]],train_label_stat[column_name],how='left',left_on = 'csv_index',right_index=True)
            data[column_name] = (train_label_withStat[column_name].sub(train_label_withStat['mean'])).divide(train_label_withStat['std'])
            counter +=1
            if counter %100 == 0:
                print('Now processing column:{}'.format(column_name))
        if not os.path.exists("DataSet/"):
            os.makedirs("DataSet/")  
        data.to_hdf('DataSet/'+fname.strip('.h5')+'_normalized_fea_label.h5','train_normalized_fea_label',append=False)
    elif(train_mode == 1):
        print('CvTest mode:')
        # 统计数据，为归一化做准备
        file_name = fname.strip('.h5').strip('train_')
        if not os.path.exists('Result/'+file_name):
            os.makedirs('Result/'+file_name)        
        test_label_stat = data.groupby([data.columns[0]])[data.columns[3:]].agg(['mean','std'])
        test_label_stat.to_hdf('Result/'+file_name + '/{}_train_label_stat.h5'.format(fname.strip('.h5')),fname,append=False)
        print('Stat file generated')
        counter = 0
        for column_name in data.columns[3:]:
            idx = np.where(data.columns.values==column_name)[0][0]
            test_label_withStat = pd.merge(data[data.columns[[0,idx]]],test_label_stat[column_name],how='left',left_on = 'csv_index',right_index=True)
            data[column_name] = (test_label_withStat[column_name].sub(test_label_withStat['mean'])).divide(test_label_withStat['std'])
            counter +=1
            if counter %100 == 0:
                print('Now processing column:{}'.format(column_name))
        if not os.path.exists("DataSet/"):
            os.makedirs("DataSet/")  
        data.to_hdf('DataSet/'+fname.strip('.h5')+'_normalized_fea_label.h5','train_normalized_fea_label',append=False)

    elif(train_mode == 2):
        print('Pred mode:')
    
    gc.collect()
    print('Finshed...')
    
    

if __name__ == "__main__":
    Params = get_params()
    train_name_raw = Params['train_name_raw']
    test_name_raw =Params['test_name_raw']
    train = restore_with_chunks(train_name_raw)
    test = pd.read_hdf('DataSet/'+ test_name_raw,engine = 'c',memory_map=True)
    normalize_fea_label(train,train_name_raw,train_mode=0)
    gc.collect()
    normalize_fea_label(test,test_name_raw,train_mode=1)
    gc.collect()
#    
    tt = pd.read_hdf('Result/1332_1333/train_1332_1333_train_label_stat.h5',engine = 'c',memory_map=True)
    print('Finished...')