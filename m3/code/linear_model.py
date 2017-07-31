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
#print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
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
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


print ("Finished...")