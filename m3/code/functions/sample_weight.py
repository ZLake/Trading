#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 10:40:43 2017

@author: lakezhang
"""
import pandas as pd
import numpy as np
import functools

def simple_minus(x,max_number):
    return max_number-x

def exp_decay(x,max_number,decay_constant=0.001):
    return np.exp(-(max_number-x) * decay_constant)

def get_sample_weight(train,cal_column_name,decay_algo,decay_param):
    """
    train: train set dataframe 
    cal_column: 用于计算sample_weight的列名
    """
    result = pd.DataFrame(columns=['cal_column_name','sample_weight'])
    cal_column = train[cal_column_name].unique();
    cal_column.sort()
    max_number = cal_column[-1]
    params = {}
    params['max_number'] = max_number
    params.update(decay_param)
    result['cal_column_name'] = cal_column
    switcher = {
            'exp':exp_decay
        }        
    # Get the function from switcher dictionary
    decay_func = functools.partial(switcher.get(decay_algo, lambda: "nothing"),**params)    
    result['sample_weight'] = result['cal_column_name'].apply(decay_func)
    return result
    
    
    
    
    