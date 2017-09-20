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
from subprocess import check_output


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
        train_label_stat = data.groupby([data.columns[0]])[data.columns[2:]].agg(['mean','std'])
        train_label_stat.to_hdf('Result/'+'_'.join(file_name.split('_')[:2]) + '/{}_train_fea_label_stat.h5'.format(fname.strip('.h5')),fname,append=False)
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
        fname_list = fname.strip('.h5').split('_')
        data.to_hdf('DataSet/'+''+'_'.join(fname_list[:3])+'_normalized_fea_label_'+'_'.join(fname_list[3:])+'.h5','train_normalized_fea_label',append=False)
    elif(train_mode == 1):
        print('CvTest mode:')
        # 统计数据，为归一化做准备
        file_name = fname.strip('.h5').strip('test_')
        if not os.path.exists('Result/'+file_name):
            os.makedirs('Result/'+file_name)        
        test_label_stat = data.groupby([data.columns[0]])[data.columns[3:]].agg(['mean','std'])
        test_label_stat.to_hdf('Result/'+file_name + '/{}_test_fea_label_stat.h5'.format(fname.strip('.h5')),fname,append=False)
        print('Stat file generated')
        counter = 0
        for column_name in data.columns[3:]:
            idx = np.where(data.columns.values==column_name)[0][0]
            test_label_withStat = pd.merge(data[data.columns[[0,idx]]],test_label_stat[column_name],how='left',left_on = 'csv_index',right_index=True)
            data[column_name] = (test_label_withStat[column_name].sub(test_label_withStat['mean'])).divide(test_label_withStat['std'])
            counter +=1
            if counter %1000 == 0:
                print('Now processing column:{}'.format(column_name))
        if not os.path.exists("DataSet/"):
            os.makedirs("DataSet/")  
        data.to_hdf('DataSet/'+fname.strip('.h5')+'_normalized_fea_label.h5','train_normalized_fea_label',append=False)

    elif(train_mode == 2):
        print('Pred mode:')
    
    gc.collect()    
    
def preprocess_withChunks(file_name,dest='DataSet/',chunk_size = 100):
    files_str = check_output(["ls", dest]).decode("utf-8")
    files_list = [file_str for file_str in files_str.strip('\n').split('\n')]
    files_list.sort()
    chunk_prefix = '{}_{}_'.format(file_name.strip('.h5'),chunk_size)
    file_str_filter = [x for x in files_list if chunk_prefix in str(x)]
    print('{} is restored from:\n{}'.format(file_name,'\n'.join(file_str_filter)))
    num_chunks = len(file_str_filter)
    file_list = list([None] * num_chunks)
    print('len:{}'.format(len(file_list)))
    for proc_idx in range(num_chunks):
        fname = 'DataSet/'+ chunk_prefix+str(proc_idx)+'.h5'
        chunk_data = pd.read_hdf('DataSet/'+ chunk_prefix+str(proc_idx)+'.h5',engine = 'c',memory_map=True)
        normalize_fea_label(chunk_data,chunk_prefix+str(proc_idx)+'.h5',train_mode=0)
        del chunk_data
        gc.collect()
    print('done...')
    
if __name__ == "__main__":
    Params = get_params()
    train_name_raw = Params['train_name_raw']
    test_name_raw =Params['test_name_raw']
#    train = restore_with_chunks(train_name_raw)
    test = pd.read_hdf('DataSet/'+ test_name_raw,engine = 'c',memory_map=True)
    preprocess_withChunks(train_name_raw)
    gc.collect()
    normalize_fea_label(test,test_name_raw,train_mode=1)
    gc.collect()
#    
#    tt = pd.read_hdf('Result/1332_1333/train_1332_1333_train_label_stat.h5',engine = 'c',memory_map=True)
    print('Finished...')