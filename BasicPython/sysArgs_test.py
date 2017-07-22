# -*- coding: utf-8 -*-

import sys

def argParser():
    print "arg len:" + str(len(sys.argv))
    if len(sys.argv)<=3:
        raise Exception,u"arguments needed"
    # init args
    args = {}
    args["1st arg"] = "Default 1st arg"
    args["2nd arg"] = "Default 2nd arg"
    args["3rd arg"] = "Default 3rd arg"
    
    # set args
    args["1st arg"] = sys.argv[1]
    args["2nd arg"] = sys.argv[2]
    args["3rd arg"] = sys.argv[3]

    if len(sys.argv)>4:
        raise Exception,"more then 3 args, 4th and later args are not used"

    return args
    

if __name__=='__main__':
    args = argParser()
    print "Parsed arg len:" + str(len(args))
    for key,value in args.items():
        print "key:" + key +",value:" + value
