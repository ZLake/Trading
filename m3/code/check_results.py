#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:55:19 2017

@author: lakezhang
"""
import pandas as pd


def get_result(theme,train_name_raw,test_name_raw):
    file_name = train_name_raw.strip('.h5').strip('train_')
    full_path = 'Result/'+file_name+'/'+theme+'__' + file_name+'_result.h5'
    result_df = pd.read_hdf(full_path,engine = 'c')
    return result_df
    
    print('reading success')
if __name__ == "__main__":
    theme = 'ODTest'
    train_name_raw = 'train_1331_1333.h5'
    test_name_raw = 'test_1331_1333.h5'
    result_df = get_result(theme,train_name_raw,test_name_raw)
    
    print ("Finished...")