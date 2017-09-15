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
import multiprocessing
from multiprocessing import Manager
import functools



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
    elif len(sys.argv)==5:
        args["split_num"] = int(sys.argv[1])
        args["dest"] = sys.argv[2]
        args["train_test"] = int(sys.argv[3])
        args["chunk_size"] = int(sys.argv[4])
    elif len(sys.argv)>5:
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
    gc.collect()
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

def read_data_chunks(proc_idx,start_pos,train_filename,dest,files_list,end_num,chunk_size):
    train_chunk = pd.DataFrame()
    train_list = []
    start_idx = proc_idx*chunk_size + start_pos
    end_idx = min((proc_idx+1)*chunk_size+ start_pos,files_list.index(end_num)+1) 
    print('start_idx:{},end_idx:{}'.format(start_idx,end_idx))
    for i in range(start_idx,end_idx):
        temp_train = pd.read_csv(dest+str(files_list[i])+".csv"
                                 ,header=None
                                 )
        temp_train.insert(0, 'csv_index', files_list[i])
        train_list.append(temp_train)
        print('Training set chunk '+ str(proc_idx) +' now processing: '+ str(files_list[i])+".csv, sample number: "+ str(len(temp_train)))
    train_chunk = pd.concat(train_list,ignore_index=True)
    train_filename_proc = train_filename + '_{}_{}'.format(chunk_size,proc_idx)+'.h5'
    train_chunk.to_hdf(train_filename_proc,'train',append=False)
    print("Train chunk {} is stored in:{}".format(proc_idx,train_filename_proc))
    print("Train chunk {}:{} is finished".format(proc_idx,train_filename))
    print ("Train chunk {} size:{}".format(proc_idx,train_chunk.shape))
    del train_chunk
    
def read_withChunks(split_num=1332,dest='../input/',train_test = 0,chunk_size=100):
    """
    split_num:data <= split_num.csv => training set
              data >  spllt_num.csv => testing  set
    dest: dir path contains all csv files
    train_test: 0, generate train and test set;1,only train set;2,only test set
    chunk_size: 每chunk_size个csv存为一个.hs文件
    """
    imp_print2("Running Info",15)
    print("Read data with chunks...")
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
    # make dir if necessary
    if not os.path.exists("DataSet/"):
        os.makedirs("DataSet/")
    # train
    if(do_train):
        imp_print2("Train set",3)
        train_filename = "DataSet/"+"train_"+str(split_num)+"_"+str(max(files_list))
        num_process = int(np.ceil((files_list.index(split_num) + 1)/chunk_size))
        params_train={}
        params_train['start_pos'] = 0 # the index of the 1st csv file 
        params_train['train_filename'] = train_filename
        params_train['files_list'] = files_list
        params_train['dest']=dest
        params_train['end_num'] = split_num
        params_train['chunk_size'] = chunk_size
        read_data_chunks_partial = functools.partial(read_data_chunks,**params_train) 
#        read_data_chunks_partial(0)
        print('Train set number of chunks:{}'.format(num_process))
        pool = multiprocessing.Pool(processes=num_process)
        for proc_idx in range(num_process):
            pool.apply_async(read_data_chunks_partial, args=(proc_idx,))
        pool.close()
        pool.join()
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
        test_filename = "DataSet/"+"test_"+str(split_num)+"_"+str(max(files_list))+'.h5'
        test.to_hdf(test_filename,'test',append=False)
        print("Test  data: " +test_filename )
        print ("test set size:{}".format(test.shape))
        del test
        
    gc.collect()
    imp_print2("Done",15)
    
def init(l):
    global lock
    lock = l

def restore_single_chunks(file_list,proc_idx,chunk_prefix):
    chunk_data = pd.read_hdf('DataSet/'+ chunk_prefix+str(proc_idx)+'.h5',engine = 'c',memory_map=True)
    file_list[proc_idx] = chunk_data
    print('done...')
    
def restore_with_chunks(file_name,dest='DataSet/',chunk_size = 100):
    files_str = check_output(["ls", dest]).decode("utf-8")
    files_list = [file_str for file_str in files_str.strip('\n').split('\n')]
    files_list.sort()
    chunk_prefix = '{}_{}_'.format(file_name.strip('.h5'),chunk_size)
    file_str_filter = [x for x in files_list if chunk_prefix in str(x)]
    print('{} is restored from:\n{}'.format(file_name,'\n'.join(file_str_filter)))
    num_chunks = len(file_str_filter)
    manager = Manager()
    file_list = manager.list([None] * num_chunks)
    param = {}
    param['chunk_prefix'] = chunk_prefix
    restore_single_chunks_partial = functools.partial(restore_single_chunks,**param) 
    print('len:{}'.format(len(file_list)))
#    restore_single_chunks_partial(0)
    l = multiprocessing.Lock()
    pool = multiprocessing.Pool(processes=len(file_str_filter),initializer=init, initargs=(l,))
    for proc_idx in range(num_chunks):
        pool.apply_async(restore_single_chunks_partial, args=(file_list,proc_idx))
    pool.close()
    pool.join()
    data_final = pd.concat(file_list,ignore_index=True)
    print('data_final shape:{}'.format(data_final.shape))
    print('Restore data done...')
    gc.collect()
    return data_final

if __name__ == "__main__":
    args = argParser()
    print("args len:{}".format(len(args)))
    if (len(args)>=1):
        read_withChunks(**args)
    else:
        read_withChunks()
#        restore_with_chunks('train_1332_1333.h5')
        

