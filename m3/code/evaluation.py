#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 10:06:17 2017

@author: lakezhang
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def evaluate_test(model,train,y_train,test,y_test,test_csv_index,topks=[50,30,10]):
    # 每天计算分数最低top50，平均后再按天平均
    model.fit(train,y_train)
    y_test_pred = model.predict(test)
    ##### mse Evaluation
    mse = mean_squared_error(y_test_pred, y_test)
    print("Test mse score: {:.10f}".format(mse))
    ##### stock Evaluation
    # take top 50 stocks and calculate avg(label)
    test_withPred = pd.DataFrame()
    test_withPred['y_test'] = y_test
    test_withPred["pred_test"] = y_test_pred 
    test_withPred.insert(0, 'csv_index', test_csv_index.values)
    csv_indexs = test_withPred['csv_index'].unique()
    # 每个index单独评分
    eval_df = pd.DataFrame(columns=['csv_index'
                                    ,'topk'
                                    ,'pred_avg'
                                    ,'pred_std'
                                    ,'pred_min'
                                    ,'pred_max'
                                    ,'above_039'
                                    ,'under_039'
                                    ,'simple_avg'])
    for csv_index in csv_indexs:
        temp_test = test_withPred[test_withPred['csv_index'] == csv_index]
        temp_test_sorted = temp_test.sort_values("pred_test",ascending = True)
        for topk in topks:
            temp_select = temp_test_sorted[:topk]['y_test']
            temp_avgLabel = temp_select.mean()
            temp_stdLabel = temp_select.std()
            temp_min = temp_select.min()
            temp_max = temp_select.max()
            temp_above_039 = np.sum([temp_select > 0.39])
            temp_under_039 = np.sum([temp_select <= 0.39])
            temp_simple_avg = temp_test[['y_test']].mean()
            eval_df.loc[len(eval_df)] = [str(csv_index),topk,temp_avgLabel,temp_stdLabel
                                        ,temp_min,temp_max,temp_above_039,temp_under_039
                                        ,temp_simple_avg]
    return eval_df

    