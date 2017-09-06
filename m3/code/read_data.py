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
import gc
import numpy as np

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
    elif len(sys.argv)==4:
        args["split_num"] = int(sys.argv[1])
        args["dest"] = sys.argv[2]
        args["train_test"] = int(sys.argv[3])
    elif len(sys.argv)>4:
        raise Exception("more then 2 args, only split_num and dest are required")

    return args

def read(split_num=1332,dest='../input/',train_test = 0):
    """
    split_num:data <= split_num.csv => training set
              data >  spllt_num.csv => testing  set
    dest: dir path contains all csv files
    train_test: 0, generate train and test set;1,only train set;2,only test set
    """
    imp_print2("Running Info",15)
    print("CSV data files loc:"+dest)
    print("Training set:data <= "+str(split_num)+".csv")
    print("Testing  set:data >  "+str(split_num)+".csv")
    files_str = check_output(["ls", dest]).decode("utf-8")
#    print(files_str)
    files_list = [int(file_str.strip('.csv')) for file_str in files_str.strip('\n').split('\n')]
    files_list.sort()
    
    imp_print2("Start Reading Data")
    if(train_test == 0):
        do_train = True
        do_test = True
        print('Generate both train and test set...')
    elif(train_test == 1):
        do_train = True
        do_test = False
        print('Generate only train set...')
    elif(train_test == 2):
        do_train = False
        do_test = True
        print('Generate only test set...')
    # set data types
    # train
    if(do_train):
        imp_print2("Train set",3)
        train = pd.DataFrame()
        train_list = []
        for i in range(files_list.index(split_num)+1):
            temp_train = pd.read_csv(dest+str(files_list[i])+".csv"
                                     ,header=None
                                     )
            temp_train.insert(0, 'csv_index', files_list[i])
            train_list.append(temp_train)
            print('Training set now processing: '+ str(files_list[i])+".csv, sample number: "+ str(len(temp_train)))
        train = pd.concat(train_list,ignore_index=True)
    # test
    if(do_test):
        imp_print2("Test set",3)
        test = pd.DataFrame()
        test_list = []
        for j in range(files_list.index(split_num)+1,len(files_list)):
            temp_test = pd.read_csv(dest+str(files_list[j])+".csv"
                                    ,header=None
                                    )
            temp_test.insert(0, 'csv_index', files_list[j])
            test_list.append(temp_test)
            print('Testing  set now processing: '+ str(files_list[j])+".csv, sample number: "+ str(len(temp_test)))
        test = pd.concat(test_list,ignore_index=True)

    imp_print2("End Reading Data",11)
    if(do_train):
        print ("train set size:{}".format(train.shape))
    if(do_test):
        print ("test set size:{}".format(test.shape))
    imp_print2("Saving Data",13)
    if not os.path.exists("DataSet/"):
        os.makedirs("DataSet/")
    if(do_train):
        train_filename = "DataSet/"+"train_"+str(split_num)+"_"+str(max(files_list))+'.h5'
        train.to_hdf(train_filename,'train',append=False)
        print("train data: " +train_filename)
        del train
    if(do_test):
        test_filename = "DataSet/"+"test_"+str(split_num)+"_"+str(max(files_list))+'.h5'
        test.to_hdf(test_filename,'test',append=False)
        del test
        print("test  data: " +test_filename )
    imp_print2("Done",15)

if __name__ == "__main__":
    args = argParser()
    print("args len:{}".format(len(args)))
    if (len(args)>=1):
        read(**args)
    else:
        read()

