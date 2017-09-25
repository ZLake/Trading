# coding: utf-8
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

from params import get_params,load_params_combs,update_params_combs,paramGridSearch
from read_data import restore_with_chunks
from sample_weight import get_sample_weight
from simple_functions import imp_print
from outlier_detection import outlier_detection,outlier_detection_grid
from models import get_model
from evaluation import evaluate_test,evaluate_test_sampleWeight,store_result

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

test_name_raw = 'test_1200_1333.h5'
test = pd.read_hdf('DataSet/'+ test_name_raw,engine = 'c',memory_map=True)
test_label_stat = test.groupby([test.columns[0]])[test.columns[2]].agg(['mean','std'])
