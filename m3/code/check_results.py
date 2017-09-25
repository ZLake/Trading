#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:55:19 2017

@author: lakezhang
"""
import pandas as pd
import json
import numpy as np
from tabulate import tabulate
import warnings
from matplotlib import pyplot as plt
import os


def get_result(theme,train_name_raw,test_name_raw):
    file_name = train_name_raw.strip('.h5').strip('train_')
    full_path = 'Result/'+file_name+'/'+theme+'__' + file_name+'_result.h5'
    print('Reading result:{}'.format(full_path))
    result_df = pd.read_hdf(full_path,engine = 'c')
    return result_df
    print('reading success')
    
def check_model_params(result_df,params,metric):
    """
    params: list(strs),params to be compared
    """
    # delete the param from the params so that we can rank them later
    temp_result_df = result_df.copy()
    uncare_params = [] # 去除关注params之后余下的params
    care_params = [] # 关注params
    for index,row in result_df.iterrows():
        all_params = row['estimator input params'].copy()
        uncare_param = all_params.copy()
        care_param = {}
        for param_name in all_params.keys():
            if param_name in params:
                care_param[param_name] = all_params[param_name]
                del uncare_param[param_name]
        
        uncare_params.append(json.dumps(uncare_param))
        care_params.append(json.dumps(care_param))
    temp_result_df['uncare_params']=uncare_params
    temp_result_df['care_params']=care_params
    temp_result_df_uncare_sorted = temp_result_df.groupby('uncare_params').mean().sort_values('top50_avg')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # top1
        top1_df = temp_result_df[temp_result_df['uncare_params'] == (temp_result_df_uncare_sorted.index[0])]
        top1_df['uncare_params_number_1'] = 1
        top1_df_avg = top1_df.groupby('care_params')[metric].mean()
        top1_df_avg.columns = [x+'_1' for x in top1_df_avg.columns.values]
        top1_df_avg['uncare_params_number_1'] = 1
        result_df = top1_df_avg.copy()
        # top 10%
        num10p = int(np.ceil(len(temp_result_df_uncare_sorted) * 0.1));
        if num10p>1:
            top10p_df = temp_result_df[temp_result_df['uncare_params'].isin(temp_result_df_uncare_sorted.index[0:num10p])]
            top10p_df['uncare_params_number_10p'] = num10p
            top10p_df_avg = top10p_df.groupby('care_params')[metric].mean()
            top10p_df_avg.columns = [x+'_10p' for x in top10p_df_avg.columns.values]
            top10p_df_avg['uncare_params_number_10p'] = num10p
            result_df = pd.merge(result_df, top10p_df_avg, how='inner',
                                 left_index=True, right_index=True, 
                                 sort=False,copy=True, indicator=False)
        # top 30%
        num30p = int(np.ceil(len(temp_result_df_uncare_sorted) * 0.3));
        if num30p>1:
            top30p_df = temp_result_df[temp_result_df['uncare_params'].isin(temp_result_df_uncare_sorted.index[0:num30p])]
            top30p_df['uncare_params_number_30p'] = num30p
            top30p_df_avg = top30p_df.groupby('care_params')[metric].mean()
            top30p_df_avg.columns = [x+'_30p' for x in top30p_df_avg.columns.values]
            top30p_df_avg['uncare_params_number_30p'] = num30p
            result_df = pd.merge(result_df, top30p_df_avg, how='inner',
                                 left_index=True, right_index=True, 
                                 sort=False,copy=True, indicator=False)
        # top 50%
        num50p = int(np.ceil(len(temp_result_df_uncare_sorted) * 0.5));
        if num50p>1:
            top50p_df = temp_result_df[temp_result_df['uncare_params'].isin(temp_result_df_uncare_sorted.index[0:num50p])]
            top50p_df['uncare_params_number_50p'] = num50p
            top50p_df_avg = top50p_df.groupby('care_params')[metric].mean()
            top50p_df_avg.columns = [x+'_50p' for x in top50p_df_avg.columns.values]
            top50p_df_avg['uncare_params_number_50p'] = num50p
            result_df = pd.merge(result_df, top50p_df_avg, how='inner',
                                 left_index=True, right_index=True, 
                                 sort=False,copy=True, indicator=False)
        # top 75%
        num75p = int(np.ceil(len(temp_result_df_uncare_sorted) * 0.75));
        if num75p>1:
            top75p_df = temp_result_df[temp_result_df['uncare_params'].isin(temp_result_df_uncare_sorted.index[0:num75p])]
            top75p_df['uncare_params_number_75p'] = num75p
            top75p_df_avg = top75p_df.groupby('care_params')[metric].mean()
            top75p_df_avg.columns = [x+'_75p' for x in top75p_df_avg.columns.values]
            top75p_df_avg['uncare_params_number_75p'] = num75p
            result_df = pd.merge(result_df, top75p_df_avg, how='inner',
                                 left_index=True, right_index=True, 
                                 sort=False,copy=True, indicator=False)
        # 100%
        num100p = int(np.ceil(len(temp_result_df_uncare_sorted) * 1));
        if num100p>1:
            top100p_df = temp_result_df[temp_result_df['uncare_params'].isin(temp_result_df_uncare_sorted.index[0:num100p])]
            top100p_df['uncare_params_number_100p'] = num100p
            top100p_df_avg = top100p_df.groupby('care_params')[metric].mean()
            top100p_df_avg.columns = [x+'_100p' for x in top100p_df_avg.columns.values]
            top100p_df_avg['uncare_params_number_100p'] = num100p
            result_df = pd.merge(result_df, top100p_df_avg, how='inner',
                                 left_index=True, right_index=True, 
                                 sort=False,copy=True, indicator=False)
        
    return result_df

def plt_hist_fea_imp(result_df,idx):
    fea_imp = result_df.iloc[idx]['feature_importance']
    plt.hist(range(0, len(fea_imp)), weights=fea_imp.tolist(), bins=np.max(fea_imp)+1)
    plt.title('feature imp value plot of {}'.format(idx))
    plt.xticks(np.arange(0, len(fea_imp), 500.0))
    plt.ylabel('feature imp')
    plt.xlabel('feature idx')
    plt.show()
    plt.hist(fea_imp.tolist(), bins=np.max(fea_imp)+1)
    plt.title('feature imp histogram plot of {}'.format(idx))
    plt.xticks(np.arange(0, np.max(fea_imp)+1, 20.0))
    plt.ylabel('histogram')
    plt.xlabel('feature imp')
    plt.show()

def get_topk_fea(result_df,idx,top=100):
    fea_imp = result_df.iloc[idx]['feature_importance']
    return np.argsort(-fea_imp)[:top]

def store_topk_fea(theme,train_name_raw,result_df,idx,topks):
    file_name = train_name_raw.strip('.h5').strip('train_')
    if not os.path.exists("Preprocess/feature_selection/"):
        os.makedirs("Preprocess/feature_selection/")
    return_df = pd.DataFrame(columns = ['model','topk','idx']) 
    final_name = 'Preprocess/feature_selection/'+theme+'__' + file_name +'_Model_{}_feaImp.h5'.format(idx)

    for topk in topks:
        fea_imp_topk_idx = get_topk_fea(result_df,idx,topk)
        return_df.loc[len(return_df)] = [theme+'__' + file_name +'_Model_{}_feaImp_top{}'.format(idx,topk),topk,fea_imp_topk_idx]
    with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return_df.to_hdf(final_name,'fea_imp',append=False)
    return return_df

if __name__ == "__main__":
    theme = 'New_data_gridSearch_4'
    train_name_raw = 'train_1268_1311.h5'
    test_name_raw = 'test_1332_1333.h5'
    start_time = 0
    if start_time > 0:
        train_name_raw = str(start_time) + '_' + train_name_raw                    

    result_df = get_result(theme,train_name_raw,test_name_raw)
    topks = [20,33,50,100,200,400,600,800,1000
             ,1200,1400,1600,1800,2000
             ,2500,3000,3500,4286,4561]
#    store_topk_fea(theme,train_name_raw,result_df,19,topks)
    print('Possible params to watch:{}'.format(result_df.iloc[0]['estimator input params'].keys()))
    # check one parameter performance
    result_df_analysis = check_model_params(result_df,['reg_lambda', 'reg_alpha'],['top50_avg'])    
#    print (tabulate(result_df_analysis, headers='keys', tablefmt='psql'))
    
    print ("Finished...")