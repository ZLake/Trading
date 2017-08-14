#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 14:58:43 2017

@author: lakezhang
"""
"""This file reads the data, 1-1200.csv as training data, 1201+ as test data.
"""
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import os, errno
import sys


def imp_print(info,slen=20):
    print ("="*slen)
    print (info)
    print ("="*slen)
def imp_print2(info,slen=10):
    print ("="*slen + info + "="*slen)

def argParser():
    print ("arg len:" + str(len(sys.argv)))
    # init args
    args = {}
    if len(sys.argv)==2:
        args["split_num"] = int(sys.argv[1])
    elif len(sys.argv)==3:
        args["split_num"] = int(sys.argv[1])
        args["dest"] = sys.argv[2]
    elif len(sys.argv)>3:
        raise Exception("more then 2 args, only split_num and dest are required")

    return args

def read(split_num=1331,dest='../input/'):
    """
    split_num:data <= split_num.csv => training set
              data >  spllt_num.csv => testing  set
    dest: dir path contains all csv files
    """
    imp_print2("Running Info",15)
    print("CSV data files loc:"+dest)
    print("Training set:data <= "+str(split_num)+".csv")
    print("Testing  set:data >  "+str(split_num)+".csv")
    files_str = check_output(["ls", dest]).decode("utf-8")
#    print(files_str)
    files_list = [int(file_str.strip('.csv')) for file_str in files_str.strip('\n').split('\n')]
    files_list.sort()
    # train
    train = pd.DataFrame()
    # test
    test = pd.DataFrame()
    imp_print2("Start Reading Data")
    imp_print2("Train set",3)
    
    train = pd.DataFrame()
    train_list = []
    for i in range(files_list.index(split_num)+1):
        temp_train = pd.read_csv(dest+str(files_list[i])+".csv",header=None)
        temp_train.insert(0, 'csv_index', files_list[i])
        train_list.append(temp_train)
        print('Training set now processing: '+ str(files_list[i])+".csv, sample number: "+ str(len(temp_train)))
    train = pd.concat(train_list)
    
    imp_print2("Test set",3)
    test = pd.DataFrame()
    test_list = []
    for j in range(files_list.index(split_num)+1,len(files_list)):
        temp_test = pd.read_csv(dest+str(files_list[j])+".csv",header=None)
        temp_test.insert(0, 'csv_index', files_list[j])
        test_list.append(temp_test)
        print('Testing  set now processing: '+ str(files_list[j])+".csv, sample number: "+ str(len(temp_test)))
    test = pd.concat(test_list)

    imp_print2("End Reading Data",11)
    print ("train set size:{}".format(train.shape))
    print ("test set size:{}".format(test.shape))
    imp_print2("Saving Data",13)
    if not os.path.exists("DataSet/"):
        os.makedirs("DataSet/")
    train_filename = "DataSet/"+"train_"+str(split_num)+"_"+str(max(files_list))+'.h5'
    test_filename = "DataSet/"+"test_"+str(split_num)+"_"+str(max(files_list))+'.h5'
    train.to_hdf(train_filename,'train',format='table',append=False)
    test.to_hdf(test_filename,'test',format='table',append=False)
    print("train data: " +train_filename)
    print("test  data: " +test_filename )
    imp_print2("Done",15)

if __name__ == "__main__":
    args = argParser()
    print("args len:{}".format(len(args)))
    if (len(args)>=1):
        read(**args)
    else:
        read()

