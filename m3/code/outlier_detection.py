#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:33:46 2017

@author: lakezhang
"""
import os.path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def proc_name(file_name_raw,clf_params):
    file_name_proc = file_name_raw.strip('.h5').strip(file_name_raw.split('_')[0]+'_') +'/' + file_name_raw.strip('.h5') +'_' + clf_params['algo']
    Params = clf_params[clf_params['algo'] + '_'+'Params']
    for key in Params:
        if (key == 'contamination' and clf_params['algo'] == 'IF'):
            continue
        else:
            file_name_proc = file_name_proc + '_' + str(Params[key])
        
    file_name_proc = file_name_proc + '.h5'
    if (os.path.exists('DataSet/'+ file_name_proc)):
        return file_name_proc,False
    else:
        return file_name_proc,True
    
def outlier_detection(train_name_raw,test_name_raw
                      ,clf_name,clf_params,train,y_train,test,y_test,test_csv_index
                      ,apply_on_test = False,num_threads = 1):
    if(clf_name == 'None'):
        return train,y_train,test,y_test,test_csv_index

    # Generate file name for storage
    if not os.path.exists("DataSet/"+train_name_raw.strip('.h5').strip('train_')+'/'):
        os.makedirs("DataSet/"+train_name_raw.strip('.h5').strip('train_')+'/')
    train_name_proc,train_need_retrain = proc_name(train_name_raw,clf_params)
    test_name_proc,test_need_retrain = proc_name(test_name_raw,clf_params)
    need_retrain = train_need_retrain or test_need_retrain
    if (need_retrain):# 未运行过当前outlierdetect，需要训练
        print('Retrain is required, fitting the model...')    
        # outlier detection
        rng = np.random.RandomState(42)
        if(num_threads > 30):
            num_threads = 2
            print('Outlier Detection nthreads:{}'.format(num_threads))
        elif(clf_name == 'IF'):
            print('IsolationForest is applied:')
            clf = IsolationForest(max_samples=0.7
                              ,max_features =1.0
                              ,random_state=rng
                              ,n_jobs = num_threads
                              ,n_estimators = 100
                              ,contamination=0.1
                              )
            clf.set_params(**clf_params['IF_Params'])
        elif(clf_name == 'LOF'):
            print('LOF is applied:')
            clf = LocalOutlierFactor(n_neighbors=20
                                     ,n_jobs = num_threads)
            clf.set_params(**clf_params['LOF_Params'])
        else:
            print('Invalid outlier detector, take it as None')
            return train,y_train,test,y_test,test_csv_index
        if(clf_name =='LOF'):
            # fit the model
            train_pred_outliers = clf.fit_predict(train.values)
            train_pred_outliers_df = pd.DataFrame()
            train_pred_outliers_df['LOF_outliers'] = train_pred_outliers
            train_pred_outliers_df.to_hdf('DataSet/'+train_name_proc,'train_LOF_outliers',append=False)
            # 去除test里的outlier
            test_pred_outliers = clf.fit_predict(test.values)
            test_pred_outliers_df = pd.DataFrame()
            test_pred_outliers_df['LOF_outliers'] = test_pred_outliers
            test_pred_outliers_df.to_hdf('DataSet/'+test_name_proc,'test_LOF_outliers',append=False)
        elif(clf_name == 'IF'):
            # fit the model
            clf.fit(train.values)
            # train
            train_pred_values = clf.decision_function(train.values)
            train_pred_values_df = pd.DataFrame()
            train_pred_values_df['IF_values'] = train_pred_values
            train_pred_values_df.to_hdf('DataSet/'+train_name_proc,'train_IF_values',append=False)
            # test
            test_pred_values = clf.decision_function(test.values)
            test_pred_values_df = pd.DataFrame()
            test_pred_values_df['IF_values'] = test_pred_values
            test_pred_values_df.to_hdf('DataSet/'+test_name_proc,'test_IF_values',append=False)
            
            temp = train_pred_values.copy()
            temp.sort()
            threshold = temp[round(np.ceil(len(temp)*clf_params['IF_Params']['contamination']))]
            train_pred_outliers = (train_pred_values>threshold).astype(int) * 2 - 1
            test_pred_outliers = (test_pred_values>threshold).astype(int) * 2 - 1
#            train_pred_outliers = clf.predict(train.values)
            
    else:# 不需要retrain，直接读之前的outlier detection结果
        print('No retrain is required, restoring from previous results...')    
        if(clf_name == 'IF'):
            print('train_pred_values from:'+'DataSet/'+train_name_proc)
            print('test_pred_values from:'+'DataSet/'+test_name_proc)
            train_pred_values = pd.read_hdf('DataSet/'+train_name_proc,engine = 'c').values[:,0]
            test_pred_values = pd.read_hdf('DataSet/'+test_name_proc,engine = 'c').values[:,0]
            temp = train_pred_values.copy()
            temp.sort()
            threshold = temp[int(np.ceil(len(temp)*clf_params['IF_Params']['contamination']))]
            train_pred_outliers = (train_pred_values>=threshold).astype(int) * 2 - 1
            test_pred_outliers = (test_pred_values>=threshold).astype(int) * 2 - 1
        elif(clf_name == 'LOF'):
            train_pred_outliers = pd.read_hdf('DataSet/'+train_name_proc,engine = 'c').values[:,0]
            test_pred_outliers = pd.read_hdf('DataSet/'+test_name_proc,engine = 'c').values[:,0]

    # 去除train里的outlier
    train = train[train_pred_outliers == 1]
    y_train = y_train[train_pred_outliers == 1]
    print('Train Set: outlier number:{}, percentage:{:.2f}%'.format((train_pred_outliers == -1).sum(),(train_pred_outliers == -1).sum()*100/len(train)))
    if(apply_on_test):
        # 去除test里的outlier
        test = test[test_pred_outliers == 1]
        print('Test Set: outlier number:{}, percentage:{:.2f}%'.format((test_pred_outliers == -1).sum(),(test_pred_outliers == -1).sum()*100/len(test)))
        y_test = y_test[test_pred_outliers == 1]
        test_csv_index = test_csv_index[test_pred_outliers == 1]         
        
            
        
    return train,y_train,test,y_test,test_csv_index