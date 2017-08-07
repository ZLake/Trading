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

def read(split_num=1200,dest='../input/'):
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
    # train
    train = pd.DataFrame()
    # test
    test = pd.DataFrame()
    imp_print2("Start Reading Data")
    imp_print2("Train set",3)
    for i in range(files_list.index(split_num)+1):
        train = pd.concat((train,pd.read_csv(dest+str(files_list[i])+".csv",header=None))).reset_index(drop=True)
        print('Training set now processing: '+ str(files_list[i])+".csv, after train sample number: "+ str(len(train)))
    imp_print2("Test set",3)
    for j in range(files_list.index(split_num)+1,len(files_list)):
        test = pd.concat((test,pd.read_csv(dest+str(files_list[i])+".csv",header=None))).reset_index(drop=True)
        print('Testing  set now processing: '+ str(files_list[j])+".csv, after test sample number: "+ str(len(test)))
    imp_print2("End Reading Data",11)
    print ("train set size:{}".format(train.shape))
    print ("test set size:{}".format(test.shape))
    imp_print2("Saving Data",13)
    if not os.path.exists("DataSet/"):
        os.makedirs("DataSet/")
    train_filename = "train_"+str(split_num)+"_"+str(max(files_list))
    test_filename = "test_"+str(split_num)+"_"+str(max(files_list))
    train.to_pickle("DataSet/"+train_filename)
    test.to_pickle("DataSet/"+test_filename)
    print("train data: DataSet/" +train_filename)
    print("test  data: DataSet/" +test_filename )
    imp_print2("Done",15)

if __name__ == "__main__":
    args = argParser()
    print("args len:{}".format(len(args)))
    if (len(args)>=1):
        read(**args)
    else:
        read()

