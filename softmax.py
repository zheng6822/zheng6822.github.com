# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 20:29:50 2012

@author: zhengxin
"""
import cudamat as cm

def softmax(eta):
    #temp = cm.empty((eta.shape[0],1))
    temp = cm.empty((1,eta.shape[1]))
    # this is considered to be potential numerical problem
    if True:
        eta.max(axis = 0, target = temp)
        #print eta.shape
        #print temp.shape
        temp.mult(-1)
        eta.add_row_vec(temp)
        cm.exp(eta)
        eta.sum(axis = 0, target = temp)
        temp.reciprocal()
        eta.mult_by_row(temp)
#    else:
#        cm.exp(eta)
#        eta.sum(axis = 0, target = temp)
#        temp.reciprocal()
#        eta.mult_by_col(temp)
        