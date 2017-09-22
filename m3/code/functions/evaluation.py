#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 10:06:17 2017

@author: lakezhang
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import gc
import os
from time import localtime, strftime
import warnings

def evaluation(y_test_pred, y_test,test_csv_index,topks):
    ##### mse Evaluation
    mse = mean_squared_error(y_test_pred, y_test)
    print("Test mse score: {:.10f}".format(mse))
    ##### stock Evaluation
    # take top 50 stocks and calculate avg(label)
    test_withPred = pd.DataFrame()
    test_withPred['y_test'] = y_test.copy()
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
    print('evaluate_test garbage collection:{}'.format(gc.collect()))
    return eval_df

def evaluate_test_sampleWeight(model,train,y_train,test,y_test,test_csv_index
                  ,sample_weight,topks=[50,30,10]):
    model.fit(train,y_train,sample_weight = sample_weight)
    y_test_pred = model.predict(test)
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    else:
        feature_importance = None
        
    return evaluation(y_test_pred, y_test,test_csv_index,topks),feature_importance
    
def evaluate_test(model,train,y_train,test,y_test,test_csv_index,topks=[50,30,10]):
    # 每天计算分数最低top50，平均后再按天平均
    model.fit(train,y_train)
    y_test_pred = model.predict(test)
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    else:
        feature_importance = None

    return evaluation(y_test_pred,y_test,test_csv_index,topks),feature_importance

def store_result(Params,algo_grid_param,algo
                 ,sample_weight_algo,sample_weight_param
                 ,eval_df,estimator,train_name_raw,test_name_raw
                 ,theme,cost_time
                 ,feature_importance
                 ,feaImp_row):
    # store result
    temp_result = []   
    daytime = strftime("%Y-%m-%d %H:%M:%S", localtime())
    #get time 
    temp_result.append(daytime)
    temp_result.append(cost_time)
    temp_result.append(train_name_raw.strip('.h5'))
    temp_result.append(test_name_raw.strip('.h5'))
    temp_result.append(feaImp_row['model'])
    temp_result.append(feaImp_row['topk'])
    temp_result.append(Params['Outlier_Detector']['algo'])
    if(Params['Outlier_Detector']['algo'] == 'None'):
        temp_result.append('None')
        temp_result.append(False)
    else:
        temp_result.append(Params['Outlier_Detector'][Params['Outlier_Detector']['algo'] + '_'+'Params'])
        temp_result.append(Params['Outlier_Detector']['apply_on_test'])
    temp_result.append(sample_weight_algo)
    temp_result.append(sample_weight_param)
    temp_result.append(algo)    #'estimator algo'
    temp_result.append(algo_grid_param)      #'estimator input params'
    if(algo == 'lasso'):
        temp_result.append(dict(estimator.get_params()['steps']))
    elif(algo == 'model_lgb'):
        temp_result.append(estimator.get_params())
    else:
        temp_result.append({'info':'unknown'})
    # performance metric
    temp_result.append(eval_df['pred_avg'][eval_df['topk']==50].mean())
    temp_result.append(eval_df['pred_avg'][eval_df['topk']==50].std())
    temp_result.append(eval_df['pred_avg'][eval_df['topk']==30].mean())
    temp_result.append(eval_df['pred_avg'][eval_df['topk']==30].std())
    temp_result.append(eval_df['pred_avg'][eval_df['topk']==10].mean())
    temp_result.append(eval_df['pred_avg'][eval_df['topk']==10].std())
    temp_result.append(eval_df['under_039'][eval_df['topk']==50].mean())
    temp_result.append(eval_df['under_039'][eval_df['topk']==30].mean())
    temp_result.append(eval_df['under_039'][eval_df['topk']==10].mean())
    temp_result.append(eval_df)
    temp_result.append(eval_df['simple_avg'].mean())
    temp_result.append(feature_importance)
    #读取之前的记录
    # Generate file name for storage
    file_name = train_name_raw.strip('.h5').strip('train_')
    full_path = 'Result/'+file_name+'/'+theme+'__' + file_name+'_result.h5'
    if not os.path.exists('Result/'+file_name):
        os.makedirs('Result/'+file_name)
    if os.path.exists(full_path):
        print('Appending result to existing file: ' + full_path)
        final_result = pd.read_hdf(full_path,engine = 'c')
        final_result.loc[len(final_result)] = temp_result
    else:
        print('Writting result to new file:' + full_path)
        final_result = pd.DataFrame(columns=['date'                 #记录日期
                                         ,'cost_time(min)'
                                         ,'train_period'
                                         ,'test_period'
                                         ,'fea_model'
                                         ,'fea_topk'
                                         ,'OD algo'
                                         ,'OD params'
                                         ,'OD apply_on_test'
                                         ,'sample weight algo'
                                         ,'sample weight params'
                                         ,'estimator algo'
                                         ,'estimator input params'
                                         ,'estimator all params'
                                         ,'top50_avg'
                                         ,'top50_std'
                                         ,'top30_avg'
                                         ,'top30_std'
                                         ,'top10_avg'
                                         ,'top10_std'
                                         ,'top50_under039_avg'
                                         ,'top30_under039_avg'
                                         ,'top10_under039_avg'
                                         ,'details'              #存eval_df
                                         ,'simple_avg'
                                         ,'feature_importance']) 
        final_result.loc[len(final_result)] = temp_result
    with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_result.to_hdf(full_path,'result',append=False)
            del final_result
    del temp_result,eval_df
    print('store_result garbage collection:{}'.format(gc.collect()))
    #加上新纪录并保存
    