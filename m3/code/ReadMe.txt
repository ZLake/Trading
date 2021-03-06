## 文件夹：
Dataset文件夹：存储处理过的train和test Set及Outlieer Detection处理结果
functions文件夹：存储用到的自定义函数
Result文件夹：存储运行结果和一些参数
## 文件
read_data.py:处理批量csv文件为train和test set
params.py：控制运行参数及模型参数
single_model_2.py:单模型测试
Model_gridSearch.py:模型grid_search脚本
## 训练测试集合
# 训练：<=1200,1253,1268
# 测试：>1200-1333,1253-1333,1268-1311

## 模型参数：
### lasso：
scaler:StandardScaler(),
alpha:0.001,0.002,0.005,0.01,0.02,0.05,0.08

### model_lgb
objective = 'regression',
num_leaves=15,
max_depth = 10
learning_rate=0.02, 0.05
n_estimators=500,1000
max_bin = 50
boosting_type = 'gbdt',
bagging_fraction = 0.8,
bagging_freq = 5,
colsample_bytree = 0.6,0.8 
feature_fraction = 0.4,0.6
min_data_in_leaf = 100
min_sum_hessian_in_leaf = 10
reg_alpha=1,2
reg_lambda=1,2
num_iterations = 100,                      #boosting iters
feature_fraction_seed=9, 
bagging_seed=9,
save_binary = True

# Test1:    OD_None_Test_Algo_model_lgb
Data time:-------------
    Params['Train_start_time'] = [0]
OD:--------------------
None
Model:-----------------
Default:
    Params['model_lgb_default_params'] = {'objective':'regression'
                                          ,'max_bin':50
                                          ,'boosting_type' : 'gbdt'
                                          ,'save_binary':True
                                          ,'bagging_fraction':0.8
                                          ,'bagging_freq':5
                                          ,'min_data_in_leaf':100
                                          ,'min_sum_hessian_in_leaf':10
                                          }
Grid:
    Params['model_lgb_grid_params'] = {
        'learning_rate': [0.02,0.05]
        ,'n_estimators': [1000,1500]
        ,'num_leaves': [15,30,45]
        ,'objective' : ['regression']
        ,'feature_fraction':[0.4,0.6]
        ,'reg_alpha' : [1,2]
        ,'reg_lambda' : [1,2]
        }
:-------------------------
# Test1.1:    OD_None_Test_Algo_model_lgb_rate_nTree_feaFrac
Data time:-------------
    Params['Train_start_time'] = [0]
OD:--------------------
None
Model:-----------------
Default:
    Params['model_lgb_default_params'] = {'objective':'regression'
                                          ,'max_bin':50
                                          ,'boosting_type' : 'gbdt'
                                          ,'save_binary':True
                                          ,'bagging_fraction':0.8
                                          ,'bagging_freq':5
                                          ,'min_data_in_leaf':100
                                          ,'min_sum_hessian_in_leaf':10
                                          } 
Grid:
    Params['model_lgb_grid_params'] = {
        'learning_rate': [0.02,0.05,0.08]
        ,'n_estimators': [1000,1300,1600,2000]
        ,'num_leaves': [30,45,60]
        ,'objective' : ['regression']
        ,'feature_fraction':[0.6,0.8]
        ,'reg_alpha' : [2]
        ,'reg_lambda' : [1]
        }
    Params['model_lgb_grid_params_filter'] = [
            {'learning_rate':[0.02],'n_estimators':[800,1000]},
            {'learning_rate':[0.05,0.08],'n_estimators':[1600,2000]}
            ]
:-------------------------
# Test1.2: OD_None_Test_Algo_model_lgb_rate_nTree_leave_loss_mbin
Data time:-------------
    Params['Train_start_time'] = [0]
OD:--------------------
None
Model:-----------------
Default:
    Params['model_lgb_default_params'] = {'objective':'regression'
                                          ,'boosting_type' : 'gbdt'
                                          ,'save_binary':True
                                          ,'bagging_fraction':0.8
                                          ,'bagging_freq':5
                                          ,'min_data_in_leaf':100
                                          ,'min_sum_hessian_in_leaf':10
                                          #### found:
                                          ,'feature_fraction':0.6
                                          ,'reg_alpha':2
                                          ,'reg_lambda':1
                                          } 
Grid:
    Params['model_lgb_grid_params'] = {
        'learning_rate': [0.01,0.02]
        ,'n_estimators': [1300,1600,2000]
        ,'num_leaves': [45,60,75]
        ,'objective' : ['regression','huber']
        ,'max_bin':[50,100]
        }
    Params['model_lgb_grid_params_filter'] = [
            {'learning_rate':[0.01],'objective':['huber']},
            {'num_leaves':[75],'objective':['huber']},
            {'max_bin':[100],'objective':['huber']}
            ]
:-------------------------
# Test1.3: OD_None_Test_Algo_model_lgb_dataTime_bag_feaRatio
Data time:-------------
    Params['Train_start_time'] = [0,100,200]
OD:--------------------
None
Model:-----------------
Default:
    Params['model_lgb_default_params'] = {'objective':'regression'
                                          ,'boosting_type' : 'gbdt'
                                          ,'save_binary':True
                                          ,'min_data_in_leaf':100
                                          ,'min_sum_hessian_in_leaf':10
                                          #### found:
                                          ,'reg_alpha':2
                                          ,'reg_lambda':1
                                          ,'max_bin':100
                                          ,'n_estimators':1 # can be more
                                          ,'learning_rate':0.02 # can be less
                                          ,'num_leaves':45
                                          } 
Grid:
    Params['model_lgb_grid_params'] = {
            'bagging_fraction':[0.7,0.8,0.9]
            ,'bagging_freq':[3,5,7]
            ,'feature_fraction':[0.5,0.6,0.7]
        }
    Params['model_lgb_grid_params_filter'] = [
            ]   
    
:-------------------------
# Test1.4: OD_None_Test_Algo_model_lgb_timeDecay_regularization_$  # $ = 1,2 两个版本，不同Decay_params
Data time:-------------
    Params['Train_start_time'] = [0] 
Time Decay:------------
    Params['Sample_weight'] = True
    Params['Decay_algo'] = 'exp' # exp
    Params['Decay_params'] = {'decay_constant':[0,0.0006,0.0007,0.0008,0.0009,0.001,0.0012]} #0.0008,0.0012
OD:--------------------
None
Model:-----------------
Default:
    Params['model_lgb_default_params'] = {'objective':'regression'
                                          ,'boosting_type' : 'gbdt'
                                          ,'save_binary':True
                                          ,'min_data_in_leaf':100
                                          ,'min_sum_hessian_in_leaf':10
                                          #### found:
                                          ,'reg_alpha':2
                                          ,'reg_lambda':1
                                          ,'max_bin':100  # can be more
                                          ,'n_estimators': 2000# can be more
                                          ,'learning_rate':0.02 # can be less
                                          ,'num_leaves':45
                                          ,'bagging_fraction':0.8
                                          ,'bagging_freq':5
                                          ,'feature_fraction':0.6
                                          } 
Grid:
    Params['model_lgb_grid_params'] = {
            'reg_alpha':[1,2,3]
            ,'reg_lambda':[0.5,1,2]

        }
    Params['model_lgb_grid_params_filter'] = [
            ]   
    
:-------------------------
# Test1.5: new,old train,test data verification:new_train > old train on new_test
# Test1.5:
Data:
    Params['train_name_raw'] = 'train_1253_1333.h5'
    Params['test_name_raw'] = 'test_1253_1333.h5'
Data time:-------------
    Params['Train_start_time'] = [0] 
Time Decay:------------
    Params['Sample_weight'] = False
OD:--------------------
None
Model:-----------------
Default:
    Params['model_lgb_default_params'] = {'objective':'regression'
                                          ,'boosting_type' : 'gbdt'
                                          ,'save_binary':True
                                          ,'min_data_in_leaf':100
                                          ,'min_sum_hessian_in_leaf':10
                                          #### found:
                                          ,'reg_alpha':3
                                          ,'reg_lambda':1
                                          ,'max_bin':100  # can be more
                                          ,'n_estimators': 2000# can be more
                                          ,'learning_rate':0.02 # can be less
                                          ,'num_leaves':45
                                          ,'bagging_fraction':0.8
                                          ,'bagging_freq':5
                                          ,'feature_fraction':0.6
                                          } 
Grid:
    Params['model_lgb_grid_params'] = {
            'n_estimators':[2000,2500]
            ,'min_data_in_leaf':[20,100,200]
            ,'min_sum_hessian_in_leaf':[10,50,100]
        }
    Params['model_lgb_grid_params_filter'] = [
            ]      
    
# undone ideas:
select feature with imp
normalized feature and label by day    
    
    
    
# =====================================================================
# Test2: OD_IF_Test_Algo_model_lgb: 测试IF OD及不同contamination, Undone
OD:--------------------
IF:
'algo':'IF'
    IF_Params = {'max_samples':0.7
                 ,'n_estimators':100
                 ,'contamination':0.1} # 0.1
Grid:
    IF_Grid_Params = {'max_samples':[0.7]
                        ,'n_estimators':[100]
                        ,'contamination':[0.08,0.1,0.12]}
Model:-----------------
Default:
    Params['model_lgb_default_params'] = {'objective':'regression'
                                          ,'max_bin':50
                                          ,'boosting_type' : 'gbdt'
                                          ,'save_binary':True
                                          ,'bagging_fraction':0.8
                                          ,'bagging_freq':5
                                          ,'min_data_in_leaf':100
                                          ,'min_sum_hessian_in_leaf':10
                                          }
Grid:
    Params['model_lgb_grid_params'] = {
        'learning_rate': [0.02]
        ,'n_estimators': [1500]
        ,'num_leaves': [30,45]
        ,'objective' : ['regression']
        ,'feature_fraction':[0.4,0.6]
        ,'reg_alpha' : [1,2]
        ,'reg_lambda' : [1,2]
        }
:-------------------------