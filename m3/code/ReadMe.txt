## 文件夹：
Dataset文件夹：存储处理过的train和test Set及Outlieer Detection处理结果
functions文件夹：存储用到的自定义函数
Result文件夹：存储运行结果和一些参数
## 文件
read_data.py:处理批量csv文件为train和test set
params.py：控制运行参数及模型参数
single_model_2.py:单模型测试
Model_gridSearch.py:模型grid_search脚本

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