# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:52:30 2012

@author: zhengxin
"""
#==============================================================================
#     This file contains three(four) kinds of RBMs: Gaussian-Bernoulli RBM,
#   Bernoulli-Bernoulli RBM, Discriminative RBM
#==============================================================================
from options import options
import numpy as np
import cudamat as cm
from softmax import softmax
import scipy.io
import os

class GaussianRBM(object):
    def __init__(self, datafolder=None, num_hid=None, options=None):
        if datafolder == None:
            return
        self.datafolder = datafolder
        self.datalist = os.listdir(datafolder)
        self.num_batchdata = len(self.datalist)
        mdict = scipy.io.loadmat(os.path.join(datafolder, self.datalist[0]))
        tempdata = mdict['data']
        self.options = options
        self.num_vis = tempdata.shape[0]
        self.num_hid = num_hid
#        print self.num_vis
#        print self.num_hid
        self.num_batches = tempdata.shape[1]/self.options.batchsize
        self.batch_size = self.options.batchsize
        self.doPCD = False
        self.cdstep = 1
        
        # initialize weights
        self.W = cm.CUDAMatrix(0.01 * np.random.randn(self.num_vis, self.num_hid))
        self.vb = cm.CUDAMatrix(np.zeros((self.num_vis,1)))# for gaussian rbm, v_bias we mean the mean of visible layer
        self.hb = cm.CUDAMatrix(np.zeros((self.num_hid,1)))
        
        # initialize weights updates
        self.dW = cm.CUDAMatrix(np.zeros((self.num_vis, self.num_hid)))
        self.dvb = cm.CUDAMatrix(np.zeros((self.num_vis, 1)))
        self.dhb = cm.CUDAMatrix(np.zeros((self.num_hid, 1)))
        self.W_inc = cm.CUDAMatrix(np.zeros((self.num_vis, self.num_hid)))
        self.vb_inc = cm.CUDAMatrix(np.zeros((self.num_vis,1)))
        self.hb_inc = cm.CUDAMatrix(np.zeros((self.num_hid,1)))
        
        # initialize temporary storage
        self.v = cm.empty((self.num_vis, self.batch_size))# a batch of data
        self.vm = cm.empty((self.num_vis, self.batch_size))# temp storage of data-vb
        self.h = cm.empty((self.num_hid, self.batch_size))
        self.r = cm.empty((self.num_hid, self.batch_size))# store random number in positive phase
        self.r2 = cm.empty((self.num_vis, self.batch_size))# store random number in negative phase
        
    def getCurrentBatch(self,mdict,batch):
        # get current batch
        batchdata = mdict['data'][:,batch*self.batch_size:(batch+1)*self.batch_size]
        self.v = cm.CUDAMatrix(batchdata)
        self.v_true = cm.CUDAMatrix(batchdata)
        
    def applyMomentum(self):
        # apply momentum
        # maybe we can change it while proccessing
        self.dW.mult(0)
        self.dvb.mult(0)
        self.dhb.mult(0)
        self.W_inc.mult(self.options.momentum)
        self.vb_inc.mult(self.options.momentum)
        self.hb_inc.mult(self.options.momentum)
        
    def hidActProb(self,vis, target):
        # positive phase
#        print self.W.shape
#        print vis.shape
#        print target.shape
        cm.dot(self.W.T, vis, target = target)
        target.add_col_vec(self.hb)
        target.apply_sigmoid()
        
    def visActProb(self):
        # negative phase
        cm.dot(self.W, self.h, target = self.v)
        self.v.add_col_vec(self.vb)#now v = Wh + c
        
    def CDstats(self, vis, hid, posphase=True):
        multiplier = 1.0 if posphase else -1.0
        self.dhb.add_sums(hid, 1, mult=multiplier)
        
        if posphase:
            #print 'posphase'
            self.dW.add_dot(vis, hid.T)
            self.vm.assign(vis)
            self.vb.mult(-1)
            self.vm.add_col_vec(self.vb)
            self.vb.mult(-1)
            self.dvb.add_sums(self.vm, 1, mult=multiplier)
        else:
            #print 'negphase'
            self.dW.subtract_dot(vis,hid.T)
            self.vm.assign(vis)
            self.vb.mult(-1)
            self.vm.add_col_vec(self.vb)
            self.vb.mult(-1)
            self.dvb.add_sums(self.vm, 1, mult=multiplier)
            
    def sampleHid(self,r,target):
        # sample hiddens
        r.fill_with_rand()
        r.less_than(target, target = target)
        
    def sampleVis(self):
        self.r2.fill_with_randn()
        self.v.add(self.r2)
    
    def CDn(self):
        n = self.cdstep
        self.hidActProb(self.v, self.h)
        self.CDstats(self.v, self.h)
        for i in range(n):
            self.sampleHid(self.r,self.h)
            self.visActProb()
            self.sampleVis()
            self.hidActProb(self.v, self.h)
        self.CDstats(self.v, self.h, False)
        
    def doOneStep(self):
        if self.doPCD:
            self.PCD()
        else:
            self.CDn()
        self.updateWeights()
        
    def updateWeights(self):
        self.W_inc.add_mult(self.dW, self.options.eta/self.batch_size)
        self.vb_inc.add_mult(self.dvb, self.options.eta/self.batch_size)
        self.hb_inc.add_mult(self.dhb, self.options.eta/self.batch_size)
                
        # update weights
        self.W.add(self.W_inc)
        self.vb.add(self.vb_inc)
        self.hb.add(self.hb_inc)
        
    def getReconErr(self):
        self.v.subtract(self.v_true)
        return self.v.euclid_norm()**2
        
    def train(self):    
        for epoch in range(self.options.maxepoch):
            err = []
            for batchdata in range(self.num_batchdata):
                mdict = scipy.io.loadmat(os.path.join(self.datafolder, self.datalist[batchdata]))
                #data = mdict['data']
                for batch in range(self.num_batches):
                    self.getCurrentBatch(mdict,batch)
                    self.doOneStep()
                    self.applyMomentum()
                    err.append(self.getReconErr()/(self.num_vis*self.batch_size))
            print "Epoch " + str(epoch + 1)+" "+"Mean squared error: " + str(np.mean(err))
        
    def getDataUpwards(self,loadfolder,savefolder):
        # push data of visible layer upwards to form a set of new data
        # because of memory issues, we have to each batch data to disc and read and combine them later
        # batch mode receive data from cpu and return a matrix on cpu
        datalist = os.listdir(loadfolder)
        batchsize = 4096
        n = 0
        for dataname in datalist:
            name = os.path.join(loadfolder,dataname)
            mdict = scipy.io.loadmat(name)
            data = mdict['data']
            labels = mdict['label']
#            print labels.shape
            numbatch = data.shape[1]/batchsize
            for batch in range(numbatch):
                #print 'batch %d/%d'%(n, numbatch*len(datalist))
                batchdata = data[:,batch*batchsize:(batch+1)*batchsize]
                batchlabels = labels[batch*batchsize:(batch+1)*batchsize]
                temp = cm.empty((self.num_hid,batchdata.shape[1]))
                vis = cm.CUDAMatrix(batchdata)
                self.hidActProb(vis, temp)
                temp.copy_to_host()
                #topdata[:,batch*batchsize:(batch+1)*batchsize] = temp.numpy_array
                mdict = {}
                mdict['data'] = temp.numpy_array
                mdict['label'] = batchlabels
                scipy.io.savemat('%s/%d.mat'%(savefolder,n),mdict)
                n = n+1
                
    def getTestDataUpwards(self,data):
        batchsize = 4096
        numbatch = data.shape[1]/batchsize
        topdata = np.zeros((self.num_hid,data.shape[1]))
        for batch in range(numbatch):
            batchdata = data[:,batch*batchsize:(batch+1)*batchsize]
            temp = cm.empty((self.num_hid,batchdata.shape[1]))
            vis = cm.CUDAMatrix(batchdata)
            self.hidActProb(vis, temp)
            temp.copy_to_host()
            topdata[:,batch*batchsize:(batch+1)*batchsize] = temp.numpy_array
        return topdata
                    
    def save(self,filename):
        self.W.copy_to_host()
        self.vb.copy_to_host()
        self.hb.copy_to_host()
        mdict = {}
        mdict['type']='gauss'
        mdict['W']=self.W.numpy_array
        mdict['vb']=self.vb.numpy_array
        mdict['hb']=self.hb.numpy_array
        scipy.io.savemat(filename,mdict)
        
    def load(self, filename):
        mdict = scipy.io.loadmat(filename)
        self.W = cm.CUDAMatrix(mdict['W'])
        self.vb = cm.CUDAMatrix(mdict['vb'])
        self.hb = cm.CUDAMatrix(mdict['hb'])
        (self.num_vis, self.num_hid) = self.W.shape
        
class BinaryRBM(GaussianRBM):
    def visActProb(self):
        GaussianRBM.visActProb(self)
        self.v.apply_sigmoid()
        
    def CDstats(self, vis, hid, posphase=True):
        multiplier = 1.0 if posphase else -1.0
        self.dhb.add_sums(hid, 1, mult=multiplier)
        self.dvb.add_sums(vis, 1, mult=multiplier)
        if posphase:
            self.dW.add_dot(vis, hid.T)
        else:
            self.dW.subtract_dot(vis,hid.T)
            
    def sampleVis(self):
        # sample hiddens
        self.r2.fill_with_rand()
        self.r2.less_than(self.v, target = self.v)# now h = phstates
        
class SoftmaxRBM(BinaryRBM):
    def hidActProb(self,vis, target):
        cm.dot(self.W.T, vis, target = target)
        target.add_col_vec(self.hb)
        softmax(target)
        
class DiscriminativeRBM(GaussianRBM):
    def __init__(self,datafolder=None,labels=None,numhid=None,options=None):
        # the labels here is just used for calculating number of labels, no other practical use.
        super(DiscriminativeRBM,self).__init__(datafolder,numhid,options)
        self.labels = labels
        self.num_class = self.getClassNum(labels)
        #self.targets = self.getTargets()
        if datafolder == None:
            return
        self.cW = cm.CUDAMatrix(0.01 * np.random.randn(self.num_class,self.num_hid))
        self.cb = cm.CUDAMatrix(np.zeros((self.num_class,1)))
        self.dcW = cm.CUDAMatrix(np.zeros((self.num_class,self.num_hid)))
        self.dcb = cm.CUDAMatrix(np.zeros((self.num_class,1)))
        self.cW_inc = cm.CUDAMatrix(np.zeros((self.num_class,self.num_hid)))
        self.cb_inc = cm.CUDAMatrix(np.zeros((self.num_class,1)))
        self.c = cm.empty((self.num_class,self.batch_size))
        
    def getClassNum(self,labels):
        self.labellist = np.unique(labels)
        return len(self.labellist)
        
    def getTargets(self,labels):
        # create targets
        targets = np.zeros((self.num_class,len(labels)))
        #print targets.shape
        for i in range(self.num_class):
            for j in range(len(labels)):
                if labels[j] == self.labellist[i]:
                    targets[i,j] = True
        return targets
        
    def applyMomentum(self):
        super(DiscriminativeRBM,self).applyMomentum()
        self.dcW.mult(0)
        self.dcb.mult(0)
        self.cW_inc.mult(self.options.momentum)
        self.cb_inc.mult(self.options.momentum)
        
    def hidActProb(self,vis, target):
        # positive phase
        cm.dot(self.W.T, vis, target = target)
        target.add_dot(self.cW.T, self.c)
        target.add_col_vec(self.hb)
        target.apply_sigmoid()
        
    def getCurrentBatch(self,mdict,batch):
        super(DiscriminativeRBM,self).getCurrentBatch(mdict,batch)
        #print mdict['label'].shape
        batchlabels = mdict['label'][batch*self.batch_size:(batch+1)*self.batch_size]
        batchtargets = self.getTargets(batchlabels)
        self.c = cm.CUDAMatrix(batchtargets)
        
    def CDstats(self, vis, hid, posphase=True):
        multiplier = 1.0 if posphase else -1.0
        self.dhb.add_sums(hid, 1, mult=multiplier)
        self.dvb.add_sums(vis, 1, mult=multiplier)
        if posphase:
            self.dW.add_dot(vis, hid.T)
            self.dcb.add_sums(self.c, 1, mult=1.0)
            self.dcW.add_dot(self.c, hid.T)
        else:
            self.dW.subtract_dot(vis,hid.T)
            self.dcb.add_sums(self.c, 1, mult=-1.0)
            self.dcW.subtract_dot(self.c,hid.T)
            
    def visActProb(self):
        # negative phase
        super(DiscriminativeRBM,self).visActProb()
        self.v.apply_sigmoid()
        cm.dot(self.cW, self.h, target = self.c)
        self.c.add_col_vec(self.cb)
        softmax(self.c)
        
    def updateWeights(self):
        super(DiscriminativeRBM,self).updateWeights()
        self.cW_inc.add_mult(self.dcW, self.options.eta/self.batch_size)
        self.cb_inc.add_mult(self.dcb, self.options.eta/self.batch_size)
        self.cW.add(self.cW_inc)
        self.cb.add(self.cb_inc)
        
    def save(self,filename):
        self.W.copy_to_host()
        self.vb.copy_to_host()
        self.hb.copy_to_host()
        self.cb.copy_to_host()
        self.cW.copy_to_host()
        mdict = {}
        mdict['type']='discriminative'
        mdict['W']=self.W.numpy_array
        mdict['vb']=self.vb.numpy_array
        mdict['hb']=self.hb.numpy_array
        mdict['cb']=self.cb.numpy_array
        mdict['cW']=self.cW.numpy_array
        scipy.io.savemat(filename,mdict)
        
    def load(self, filename):
        mdict = scipy.io.loadmat(filename)
        self.W = cm.CUDAMatrix(mdict['W'])
        self.vb = cm.CUDAMatrix(mdict['vb'])
        self.hb = cm.CUDAMatrix(mdict['hb'])
        self.cb = cm.CUDAMatrix(mdict['cb'])
        self.cW = cm.CUDAMatrix(mdict['cW'])
        
#    def train(self):    
#        for epoch in range(self.options.maxepoch):
#            #err = 0
#            err = []
#            for batchdata in range(self.num_batchdata):
#                #print 'batchdata'+str(batchdata)
#                #print self.datalist[batchdata]
#                mdict = scipy.io.loadmat(os.path.join(self.datafolder, self.datalist[batchdata]))
#                data = mdict['data']
#                for batch in range(self.num_batches):
#                    #print 'batch'+str(batch)
#                    self.getCurrentBatch(data,batch)
#                    self.doOneStep()
#                    self.applyMomentum()
#                    err.append(self.getReconErr()/(self.num_vis*self.batch_size))      
#            print "Epoch " + str(epoch + 1)+" "+"Mean squared error: " + str(np.mean(err))
#        