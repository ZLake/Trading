#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 14:46:15 2017

@author: lakezhang
"""
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.preprocessing import RobustScaler,StandardScaler
import lightgbm as lgb
import xgboost as xgb


def init_lasso(default_param,rng,num_threads,need_sample_weight):
    clf = Pipeline(steps=[('scaler',StandardScaler(copy=True)),
                         ('lasso',Lasso(alpha = 0.01,random_state=rng,
                                        copy_X=False))])
    clf.set_params(**default_param)
    return clf

def init_model_lgb(default_param,rng,num_threads,need_sample_weight):
    print('number of thread in training lgb:{}'.format(num_threads))
#    clf = lgb.LGBMRegressor(objective='regression',num_leaves=5,
#                                      learning_rate=0.02, n_estimators=1500,
#                                      max_bin = 55, bagging_fraction = 0.8,
#                                      bagging_freq = 5, feature_fraction = 0.2319,
#                                      feature_fraction_seed=9, bagging_seed=9,
#                                      min_data_in_leaf =6, min_sum_hessian_in_leaf = 11,
#                                      num_threads = num_threads,
#                                      reg_alpha=1, reg_lambda=1,
#                                      save_binary = True
#                                      )
    clf = lgb.LGBMRegressor(n_jobs = num_threads)
    clf.set_params(**default_param)
#    if(need_sample_weight):
#        clf.set_params(weight='0') #第一列是weight_column
    return clf



#def init_model_lgb_general(default_param,rng,num_threads):
#    print('number of thread in training lgb_general:{}'.format(num_threads))
#    clf = lgb.LGBMModel(objective='regression'
#                        ,n_jobs = num_threads)
#    clf.set_params(**default_param)
#    return clf

def get_model(model_name,default_param,rng,num_threads,sample_weight=False):
    switcher = {
        'lasso': init_lasso,
        'model_lgb': init_model_lgb
        }        
    # Get the function from switcher dictionary
    model_initializer = switcher.get(model_name, lambda: "nothing")
    
    return model_initializer(default_param,rng,num_threads,sample_weight)
