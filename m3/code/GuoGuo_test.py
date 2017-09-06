#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 18:01:02 2017

@author: lakezhang
"""

import numpy as np
from scipy.stats import pearsonr
#np.random.seed(0)
#size = 300
#x = np.random.normal(0, 1, size)
#print ("Lower noise", pearsonr(x, x + np.random.normal(0, 1, size)))
#print ("Higher noise", pearsonr(x, x + np.random.normal(0, 10, size)))

def normalize(data):
    for key in data.keys():
        data[key] = [value/sum(data[key]) for value in data[key]]
    return data

def print_data(data):
    for key in data.keys():
        print(key+':'+str([round(x,4) for x in data[key]]))

def cal_pearsonr(base_data,data):
    for key in data.keys():
        print('Pearson coef with {}:{}'.format(key,pearsonr(base_data,data[key])))


if __name__ == "__main__":
    base_data = [102.65,302.24,22.03,56.25 ,5.27 ,44.90 ,26.56]
    
    data = {    'hz' : [102.65,302.24,22.03,56.25 ,5.27 ,44.90 ,26.56],
                'bj' : [241.74,469.51,71.35,118.20,7.36 ,54.88 ,66.02],
                'sh' : [237.78,495.77,60.50,93.35 ,12.27,103.94,36.21],
                'fs' : [80.46 ,156.65,24.86,29.51 ,2.81 ,44.22 ,13.87],
                'gz' : [215.15,296.22,49.73,51.19 ,10.05,92.00 ,32.63],
                'fz' : [69.04 ,180.57,23.89,17.47 ,3.29 ,27.01 ,9.26 ],
                'nj' : [105.64,212.80,24.74,82.25 ,3.61 ,58.16 ,38.34],
                'tj' : [98.77 ,109.54,38.45,28.89 ,2.01 ,23.34 ,20.70]
    }
    print('每个行的各个业务量：')
    print_data(data)
    normalized_data = normalize(data.copy())
    print('每个行的业务量占比：')
    print_data(normalized_data)
    print('每个行和杭州行的皮尔逊相关系数：(相关系数，p-value)')
    cal_pearsonr(base_data,normalized_data)
    