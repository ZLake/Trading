#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 10:27:25 2017

@author: lakezhang
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore') #Supress unnecessary warnings for readability
from scipy import stats
from scipy.stats import norm, skew, boxcox #for some statistics
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory
from scipy.special import boxcox1p

def imp_print(info,slen=20):
    print ("="*slen)
    print (info)
    print ("="*slen)
    
  
#####################
# Data Loading
#####################
imp_print("Data Loading...",40)
train = pd.read_csv('../input/1331.csv',header=None)
val = pd.read_csv('../input/1332.csv',header=None)
test = pd.read_csv('../input/1333.csv',header=None)
#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The val data size before dropping Id feature is : {} ".format(val.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))
#Save the 'Id' column
train_ID = train[0]
val_ID = val[0]
test_ID = test[0]
#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop(train.columns[0], axis = 1, inplace = True)
val.drop(val.columns[0], axis = 1, inplace = True)
test.drop(test.columns[0], axis = 1, inplace = True)
#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The val data size after dropping Id feature is : {} ".format(val.shape))
print("The test data size after dropping Id feature is : {} ".format(test.shape))
#####################
# Data Processing
#####################
imp_print("Data Processing...",40)
ntrain = train.shape[0]
nval = val.shape[0]
ntest = test.shape[0]
y_train = train[train.columns[0]].values
y_val = val[val.columns[0]].values
y_test = test[test.columns[0]].values
all_data = pd.concat((train, val, test)).reset_index(drop=True)
all_data.drop(train.columns[0], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))
# missing data
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)
print("missing data column numbers:{}".format(missing_data.shape[0]))
if(missing_data.shape[0] == 0):
    imp_print("no missing data, go to next step...")
else:
    imp_print("Need filling missing data...")
######
# Data Correlation
######
# corrmat = train.corr()
######
# Imputing missing values
######
# No missing value, skip
######
# Feature Engineering
######
### Skewed features
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
print(skewness.head(10))
#==============================================================================
## plot the skewed varible 
#colRank = 3
#colname = skewness.head(50).index[colRank]
#sns.distplot(train[colname] , fit=norm)
## Get the fitted parameters used by the function
#(mu, sigma) = norm.fit(train[colname])
#print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#            loc='best')
#plt.ylabel('Frequency')
#plt.title(str(colname) + ' distribution')
#fig = plt.figure()
#res = stats.probplot(train[colname], plot=plt)
#plt.show()
##plt.close()
#==============================================================================
#skewness = skewness[abs(skewness) > 0.75]
##print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
#skewed_features = skewness.index
#lam = 0.15
#for feat in skewed_features:
#    all_data[feat] = boxcox1p(all_data[feat], lam)
#    all_data[feat] += 1
######
# Categorical Encoding
######
# No categorical feature, skip
#####################
# Modelling
#####################
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

# get the train and val and test data
train = all_data[:ntrain]
val = all_data[ntrain:nval+ntrain]
test = all_data[nval+ntrain:]
# kfold on train set evaluation
n_folds = 5
def rmse_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
######
# BASIC MODELS
######
## Notice that here the outliers are removed by RobustScaler() in the pipeline!
######
# Lasso Regression
######
scaler = StandardScaler()
#scaler = RobustScaler()
lasso = make_pipeline(scaler, Lasso(alpha =0.05, random_state=1))
######
# Elastic Net Regression :
######
ENet = make_pipeline(scaler, ElasticNet(alpha=0.05, l1_ratio=0.8, random_state=5))
######
# Kernel Ridge Regression :
######
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
######
# Gradient Boosting Regression :
######
## With huber loss that makes it robust to outliers
GBoost = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
######
# Xgboost :
######
model_xgb = xgb.XGBRegressor(colsample_bytree=0.2, gamma=0.0, 
                             learning_rate=0.05, max_depth=6, 
                             min_child_weight=1.5, n_estimators=1000,
                             reg_alpha=0.9, reg_lambda=0.6,
                             subsample=0.2,seed=42, silent=1,
                             random_state =7)
######
# LightGBM
######
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
######
# Base models scores: 
######
#==============================================================================
def evaluate_val(model,topk=50):
    model.fit(train.values,y_train)
    y_val_pred = model.predict(val.values)
    ##### mse Evaluation
    mse = mean_squared_error(y_val_pred, y_val)
    print("Val mse score: {:.4f}".format(mse))
    ##### stock Evaluation
    # take top 50 stocks and calculate avg(label)
    val_withPred = val.copy()
    val_withPred["pred_val"] = y_val_pred 
    val_withPred_sorted = val_withPred.sort_values("pred_val",ascending = True)
    avg_label =  val_withPred_sorted[:topk][val_withPred_sorted.columns[0]].mean()
    print("Val Dataset avg label score:{:.4f}".format(val[val.columns[0]].mean()))
    print("Val top"+str(topk)+" avg label score: {:.4f}".format(avg_label))
def evaluate_test(model,topk=50):
    model.fit(val.values,y_val)
    y_test_pred = model.predict(test.values)
    ##### mse Evaluation
    mse = mean_squared_error(y_test_pred, y_test)
    print("Test mse score: {:.4f}".format(mse))
    ##### stock Evaluation
    # take top 50 stocks and calculate avg(label)
    test_withPred = test.copy()
    test_withPred["pred_val"] = y_test_pred 
    test_withPred_sorted = test_withPred.sort_values("pred_val",ascending = True)
    avg_label =  test_withPred_sorted[:topk][test_withPred_sorted.columns[0]].mean()
    print("Test Dataset avg label score:{:.4f}".format(val[val.columns[0]].mean()))
    print("Test top"+str(topk)+" avg label score: {:.4f}".format(avg_label))
#==============================================================================
imp_print("lasso:",10)
evaluate_val(lasso)
evaluate_test(lasso)
imp_print("Enet:",10)
evaluate_val(ENet)
evaluate_test(ENet)
imp_print("GBoost:",10)
evaluate_val(GBoost)
evaluate_test(GBoost)
imp_print("model_xgb:",10)
evaluate_val(model_xgb)
evaluate_test(model_xgb)
imp_print("model_lgb:",10)
evaluate_val(model_lgb)
evaluate_test(model_lgb)
#==============================================================================


print ("Finished...")