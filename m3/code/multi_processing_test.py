#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:52:38 2017

@author: lakezhang
"""
import multiprocessing
import time

def target(item):
    # Do cool stuff
    if (item>0):
        lock1.acquire()
        # Write to stdout or logfile, etc.
        for i in range(100):
            print('this is item {}'.format(item))
        lock1.release()
    time.sleep(5)
    print('item {} is finished...'.format(item))

def init(l1,l2):
    global lock1,lock2
    lock1 = l1
    lock2 = l2

def main():
    iterable = [1, 2, 3, 4, 5]
    l1 = multiprocessing.Lock()
    l2 = multiprocessing.Lock()

    pool = multiprocessing.Pool(processes=2,initializer=init, initargs=(l1,l2,))
    for item in iterable:
        pool.apply_async(target, args=(item,))
    pool.close()
    pool.join()
    
if __name__ == "__main__":
    main()
    print('finished...')