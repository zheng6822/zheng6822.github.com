# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:44:21 2012

@author: zhengxin
"""

class options:
    def __init__(self, type='bb'):
        self.method = 'cd'
        if type == 'bb':
            self.eta = 0.02
        elif type == 'gb':
            self.eta = 0.002
        else:
            self.eta = 0.0002
        self.num_hid = 1000
        self.momentum = 0.5
        self.maxepoch = 100
        self.avglast = 5
        self.batchsize = 256
        self.verbose = True
        