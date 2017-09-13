# coding: utf-8
get_ipython().system('echo $DISPLAY')
import sys
sys.path.insert(0, 'functions')

import numpy as np # linear algebra
import scipy as sp
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV,PredefinedSplit
from sklearn.metrics import make_scorer
from sklearn.externals import joblib # solve OOM problem


import multiprocessing
from time import localtime, strftime

import lightgbm as lgb
import xgboost as xgb

import time
import gc
import os
import warnings

from params import get_params,get_params2,load_params_combs,update_params_combs,paramGridSearch
from sample_weight import get_sample_weight
from simple_functions import imp_print
from outlier_detection import outlier_detection,outlier_detection_grid
from models import get_model
from evaluation import evaluate_test,evaluate_test_sampleWeight,store_result
Params = get_params()
rng = np.random.RandomState(42)
if multiprocessing.cpu_count() >=60:
    num_threads = -1
else:
    num_threads = multiprocessing.cpu_count()
train_name_raw = Params['train_name_raw']
test_name_raw =Params['test_name_raw']

train= pd.read_hdf('DataSet/'+ train_name_raw,engine = 'c',memory_map=True)
test = pd.read_hdf('DataSet/'+ test_name_raw,engine = 'c',memory_map=True)
train_csv_index = train[train.columns[0]].copy()
test_csv_index = test[train.columns[0]].copy()
#Save the 'Id' column
train_ID = train[train.columns[1]].copy()
test_ID = test[train.columns[1]].copy()
y_train = train[train.columns[2]].copy()
y_test = test[test.columns[2]].copy()
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib')
