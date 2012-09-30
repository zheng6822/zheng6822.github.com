# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 09:19:31 2012

@author: zhengxin
"""
from dbn import DBN
import os
import scipy.io
import cudamat as cm
import numpy as np
from softmax import softmax
import learn
import sys

class NeuralNet(DBN):
    def __init__(self,data, labels, testdata, testlabels, devdata, devlabels, featurefolder, postfeaturefolder, config):
        DBN.__init__(self,data,labels,testdata,testlabels,config)
        self.devdata = devdata
        self.devlabels = devlabels
        self.batchsize = 256
        self.momentum = 0.5
        self.featurefolder = featurefolder
        self.postfeaturefolder = postfeaturefolder
        self.eta = 0.01
        self.W = []
        self.b = []
        self.dW = []
        self.db = []
        self.W_inc = []
        self.b_inc = []
        self.bestW = []
        self.bestb = []
        self.vis = []
        self.num_class = self.getClassNum(labels)
        self.h = []# a list of pre-allocated CUDAMatrix in order to save gpu RAM
        
    def load(self, modelnamelist):
        DBN.load(self,modelnamelist)
        for i in range(self.H):
#            print type(self.model[i].W)
            self.W.append(self.model[i].W)
#            self.W[i].copy_to_host()
#            print self.W[i].numpy_array
            self.dW.append(cm.CUDAMatrix(np.zeros(self.model[i].W.shape)))
            self.W_inc.append(cm.CUDAMatrix(np.zeros(self.model[i].W.shape)))
            self.b.append(self.model[i].hb)
            self.db.append(cm.CUDAMatrix(np.zeros(self.model[i].hb.shape)))
            self.b_inc.append(cm.CUDAMatrix(np.zeros(self.model[i].hb.shape)))
            self.h.append(cm.empty((self.model[i].num_hid,self.batchsize)))
    
    def applyMomentum(self):
        for i in range(self.H):
            self.dW[i].mult(0)
            self.db[i].mult(0)
            self.W_inc[i].mult(self.momentum)
            self.b_inc[i].mult(self.momentum)
            
    def updateWeights(self):
#        print 'update weights'
#        assert(len(self.W)==len(self.b) and len(self.W_inc) == len(self.b_inc) and len(self.dW)==len(self.db) and len(self.W)==self.H and len(self.W_inc)==self.H and len(self.dW) == self.H)
        for i in range(self.H):
            self.W_inc[i].add_mult(self.dW[i], self.eta/self.batchsize)
            self.b_inc[i].add_mult(self.db[i], self.eta/self.batchsize)
            self.W[i].subtract(self.W_inc[i])
            self.b[i].subtract(self.b_inc[i])
#            self.W_inc[i].copy_to_host()
#            print self.W_inc[i].numpy_array
            
    def getError(self, data, labels, batch):
#        print 'get error'
        batchdata = data[:,batch*self.batchsize:(batch+1)*self.batchsize]
        batchlabels = labels[batch*self.batchsize:(batch+1)*self.batchsize]
        batchtargets = self.getTargets(batchlabels)
        results = self.forwardProp(batchdata)
        softmax(results)
        targets = cm.CUDAMatrix(batchtargets)
        results.subtract(targets)
        return results
        
    def getValidError(self):
#        print 'calculating validation error'
        num_batches = self.devdata.shape[1]/self.batchsize
        num_error = 0
        for batch in range(num_batches):
            batchdata = self.devdata[:,batch*self.batchsize:(batch+1)*self.batchsize]
            batchlabels = self.devlabels[batch*self.batchsize:(batch+1)*self.batchsize]
            results = self.forwardProp(batchdata)
            results.copy_to_host()
            results = results.numpy_array
            prediction = results.argmax(0)
            for i in range(self.batchsize):
                if self.labellist[prediction[i]]!=batchlabels[i]:
                    num_error = num_error+1
        error = num_error*1.0/len(self.devlabels)
        print 'validation error: %f'% error
        return error
        
    def postFeature(self):
        featurelist = os.listdir(self.featurefolder)
        for featurefile in featurelist:
            filename = os.path.join(self.featurefolder,featurefile)
            feature = scipy.io.loadmat(filename)
            featuremat = feature['data'].T# each column is a feature
            pre_softmax = self.forwardProp(featuremat)
            postfeature_gpu = softmax(pre_softmax)
            postfeature_gpu.copy_to_host()
            postfeature = postfeature_gpu.numpy_array
            d = {}
            d['data']=postfeature
            savefilename = os.path.join(self.postfeaturefolder, featurefile)
            scipy.io.savemat(savefilename, d)
        
    def getPostProb(self,data):
        # get the posterior prob. of a feature file
        results = self.forwarProp(data)# pre-softmax results
        softmax(results)
        return results
                    
    def forwardProp(self, batchdata):
#        print 'forward propagation'
        # feed forward till right before output, so we can get what we want as output outside this function
        self.vis = []
        tempdata = cm.CUDAMatrix(batchdata)
#        tempdata.copy_to_host()
        self.vis.append(tempdata)
        # feed forward through binary layers
        for i in range(self.H-1):
            cm.dot(self.W[i].T,tempdata, self.h[i])
#            self.W[i].copy_to_host()
#            if np.any(np.isnan(self.W[i].numpy_array)):
#                print 'fatal error'
#                sys.exit(0)
            self.h[i].add_col_vec(self.b[i])
            self.h[i].apply_sigmoid()
#            self.h[i].copy_to_host()
            tempdata = cm.empty(self.h[i].shape)
            tempdata.assign(self.h[i])
            self.vis.append(tempdata)
        # top softmax layer
        cm.dot(self.W[self.H-1].T,tempdata,self.h[self.H-1])
        self.h[self.H-1].add_col_vec(self.b[self.H-1])
        return self.h[self.H-1]
                
    def backProp(self, error):
#        print 'back propagation'
        self.dW[self.H-1].add_dot(self.vis[self.H-1],error.T)
#        print 'self.vis'
#        self.vis[self.H-1].copy_to_host()
#        print self.vis[self.H-1].numpy_array
#        print 'self.dW'
#        self.dW[self.H-1].copy_to_host()
#        print self.dW[self.H-1].numpy_array
#        print 'error 2'
#        error.copy_to_host()
#        print error.numpy_array
        self.db[self.H-1].add_sums(error,axis =1 )
        for i in list(reversed(range(self.H-1))):
            delta = cm.empty((self.W[i+1].shape[0],error.shape[1]))
            cm.dot(self.W[i+1],error,target = delta)# delta : 2000*256
            learn.mult_by_sigmoid_deriv(delta, self.vis[i+1])
            self.dW[i].add_dot(self.vis[i], delta.T)
            self.db[i].add_sums(delta, axis = 1)
            error = delta
        
    def fineTuning(self,datafolder,num_epoch,batchsize = 256):
        patience = 16*370*100
        patience_inc = 2
#        valid_freq = 16*370
        improve_thresh = 0.995
        epoch = 0
        done = False
        best_valid_err = 1e60
        datalist = os.listdir(datafolder)
        #num_batchdata = len(datalist)
        while epoch < num_epoch and not done:
            epoch = epoch+1
            training_error = 0
            for i in range(len(datalist)):
                mdict = scipy.io.loadmat(os.path.join(datafolder,datalist[i]))
                data = mdict['data']
                labels = mdict['label']
                num_batches = data.shape[1]/self.batchsize
                for batch in range(num_batches):
                    num_iter = epoch*num_batches*len(datalist)+i*len(datalist)+batch
                    error = self.getError(data, labels , batch)
#                    print 'error is'
#                    error.copy_to_host()
#                    print error.numpy_array
                    self.backProp(error)
                    self.updateWeights()
                    self.applyMomentum()
#                    if num_iter % valid_freq == 0:
#                        valid_err = self.getValidError()
#                        if valid_err<best_valid_err:
#                            if valid_err<best_valid_err*improve_thresh:
#                                patience = max(patience, num_iter*patience_inc)
#                            best_valid_err = valid_err
#                            self.saveWeights()
                    if patience<=num_iter:
                        done = True
                        break
                    training_error += error.euclid_norm()**2
            print 'epoch %d, trining error : %f'% (epoch,training_error)
            valid_err = self.getValidError()
            if valid_err<best_valid_err:
                if valid_err<best_valid_err*improve_thresh:
                    patience = max(patience, num_iter*patience_inc)
                    best_valid_err = valid_err
                    self.saveWeights()
                    
    def saveNN(self,filename):
        d ={}
        d['H'] = self.H
        for i in range(self.H):
            d['W%d'%i] = self.bestW[i]
            d['b%d'%i] = self.bestb[i]
        scipy.io.savemat(filename,d)
        
    def loadNN(self,filename):
        d = scipy.io.loadmat(filename)
        for i in range(self.H):
            self.W.append(cm.CUDAMatrix(d['W%d'%i]))
            self.b.append(cm.CUDAMatrix(d['b%d'%i]))
            self.dW.append(cm.CUDAMatrix(np.zeros(self.W[i].shape)))
            self.W_inc.append(cm.CUDAMatrix(np.zeros(self.W[i].shape)))
            self.db.append(cm.CUDAMatrix(np.zeros(self.b[i].shape)))
            self.b_inc.append(cm.CUDAMatrix(np.zeros(self.b[i].shape)))
            self.h.append(cm.empty((self.W[i].shape[1],self.batchsize)))
                    
    def saveWeights(self):
        self.bestW = []
        self.bestb = []
        for i in range(self.H):
            self.W[i].copy_to_host()
            self.b[i].copy_to_host()
            self.bestW.append(self.W[i].numpy_array)
            self.bestb.append(self.b[i].numpy_array)
    
    def getClassNum(self,labels):
        self.labellist = np.unique(labels)
        return len(self.labellist)
        
    def getTargets(self,labels):
        # create targets
        targets = np.zeros((self.num_class,len(labels)))
#        print targets.shape
        for i in range(self.num_class):
            for j in range(len(labels)):
                if labels[j] == self.labellist[i]:
                    targets[i,j] = True
        return targets