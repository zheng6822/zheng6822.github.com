# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 18:46:09 2012

@author: zhengxin
"""
from rbm import *

class DBN:
    def __init__(self, data, labels, testdata, testlabels, config=None):
        self.data = data
        self.labels = labels
        self.testdata = testdata
        self.testlabels = testlabels
        #self.devdata = devdata
        #self.devlabels = devlabels
        if config!=None:
            self.config = config
            self.H = len(config)
        #print self.H
        
    def fit(self):
        self.model = []
        H = self.H
        self.model.append(GaussianRBM(self.data, self.config[0].num_hid, self.config[0]))#bottom layer
        print 'training layer 1'
        self.model[0].train()#train the rbm
        data=self.model[0].getDataUpwards(self.data)# get data for layer above
        for i in range(H-2):
            #data = self.model[i].topdata
            self.model.append(BinaryRBM(data, self.config[i+1].num_hid, self.config[i+1]))
            self.model[i+1].train()
            data=self.model[i+1].getDataUpwards(data)
        #data = self.model[H-2].topdata
        self.model.append(DiscriminativeRBM(data, self.labels, self.config[H-1].num_hid, self.config[H-1]))
        self.model[H-1].train()
        
    def getPredict(self):
        #topdata = cm.CUDAMatrix(self.testdata)
        topdata = self.testdata
        print topdata.shape
        for i in xrange(self.H-1):
            topdata = self.model[i].getTestDataUpwards(topdata)
            print topdata.shape
#            tempdata = cm.empty((self.model[i].num_hid,self.testdata.shape[1]))
#            self.model[i].hidActProb(tempdata,temp)
#            tempdata.assign(temp)
        return self.getRBMPredict(topdata)
    
    # consider moving this funtion to rbm.py
    def getRBMPredict(self,data):
        F = self.getFreeEnergy(data)
        predic = F.argmax(0)
        prediction = np.zeros(predic.shape)
        for i in xrange(len(prediction)):
            prediction[i] = self.model[self.H-1].labellist[predic[i]]
        return prediction
        
    def getSoftmax(self):
        topdata = self.testdata
        for i in xrange(self.H-1):
            topdata = self.model[i].getDataUpwards(topdata)
#            tempdata = cm.empty((self.model[i].num_hid,self.testdata.shape[1]))
#            self.model[i].hidActProb(tempdata,temp)
#            tempdata.assign(temp)
        return self.getRBMSoftmax(topdata)
    
    # consider moving this funtion to rbm.py
    def getRBMSoftmax(self,data):
        F = self.getFreeEnergy(data)# numclasses*numcases
        sm = softmax(F)# also numclasses*numcases
        return sm
        
    def load(self,modelnamelist):
        self.H = len(modelnamelist)
        self.model = []
        for name in modelnamelist:
#            mdict = scipy.io.loadmat(name)
#            #gauss or binary
#            #if mdict['type']=='gauss':
            if name == modelnamelist[0]:
                grbm = GaussianRBM()
                grbm.load(name)
                self.model.append(grbm)
            elif name == modelnamelist[-1]:
                drbm = SoftmaxRBM()
                drbm.load(name)
                self.model.append(drbm)
            else:
                brbm = BinaryRBM()
                brbm.load(name)
                self.model.append(brbm)
                

    
    # consider moving this funtion to rbm.py
#    def getFreeEnergy(self,data):
#        #data: num_vis*numcases
#        #W: num_vis*num_hid
#        #cW: numclass*num_hid
#        #X: numclass*numcases
#        H = self.H
#        numclasses = self.model[H-1].cW.shape[0]
#        numcases = data.shape[1]
#        num_hid = self.model[H-1].W.shape[1]
#        batchsize = 256
#        temp = np.zeros((numclasses,batchsize))
#        tempnet = cm.CUDAMatrix(np.zeros((num_hid, batchsize)))
#        tempexp = cm.empty((num_hid,batchsize))
#        tempnet = np.zeros((num_hid,batchsize))
#        F = np.zeros((numclasses, numcases))
#        # move to cpu
#        self.model[H-1].W.copy_to_host()
#        self.model[H-1].cW.copy_to_host()
#        self.model[H-1].hb.copy_to_host()
#        self.model[H-1].cb.copy_to_host()
#        W = self.model[H-1].W.numpy_array
#        cW = self.model[H-1].cW.numpy_array
#        hb = self.model[H-1].hb.numpy_array
#        cb = self.model[H-1].cb.numpy_array
#        #tempones = cm.CUDAMatrix(np.ones(1,numcases))
#        #set every class bit in turn and find -free energy of the configuration
#        for i in xrange(numclasses):
#            X = np.zeros((numclasses, numcases))
#            X[i,:] = 1            
#            tempnet = np.dot(W.T,data)+np.dot(cW.T,X)+hb
#            tempnet = np.log(exp(tempnet)+1)
#            
##            cm.dot(self.model[H-1].W.T, data, target = tempnet)
##            tempnet.add_dot(self.model[H-1].cW.T, X)
##            tempnet.add_col_vec(self.model[H-1].hb)
##            cm.exp(tempnet,target=tempexp)
##            tempexp.copy_to_host()
##            tempexp.numpy_array+=1
##            templog = np.log(tempexp.numpy_array)
##            self.model[H-1].cb.copy_to_host()
##            cb = self.model[H-1].cb.numpy_array
##            # this might be incorrect
#            F[i,:]= np.tile(cb[i],(1,numcases))+sum(tempnet,0)
#        return F
        
    def getFreeEnergy(self,data):
        #data: num_vis*numcases
        #W: num_vis*num_hid
        #cW: numclass*num_hid
        #X: numclass*numcases
        H = self.H
        numclasses = self.model[H-1].cW.shape[0]
        numcases = data.shape[1]
        num_hid = self.model[H-1].W.shape[1]
        batchsize = 512
        batchnum = numcases/batchsize
        
        tempnet = cm.CUDAMatrix(np.zeros((num_hid, batchsize)))
        tempexp = cm.empty((num_hid,batchsize))
        F = np.zeros((numclasses, numcases))
        # move to cpu
#        self.model[H-1].W.copy_to_host()
#        self.model[H-1].cW.copy_to_host()
#        self.model[H-1].hb.copy_to_host()
        self.model[H-1].cb.copy_to_host()
#        W = self.model[H-1].W.numpy_array
#        cW = self.model[H-1].cW.numpy_array
#        hb = self.model[H-1].hb.numpy_array
        cb = self.model[H-1].cb.numpy_array
        #tempones = cm.CUDAMatrix(np.ones(1,numcases))
        #set every class bit in turn and find -free energy of the configuration
        for i in xrange(numclasses):
            temp = np.zeros((numclasses,batchsize))
            temp[i,:]=1
            X = cm.CUDAMatrix(temp)
            for batch in range(batchnum):
                #print '1'
                #tempnet = cm.empty((num_hid, batchsize))
                #tempexp = cm.empty((num_hid,batchsize))
                batchdata = data[:,batch*batchsize:(batch+1)*batchsize]
                tempdata = cm.CUDAMatrix(batchdata)
                cm.dot(self.model[H-1].W.T, tempdata, target = tempnet)
                tempnet.add_dot(self.model[H-1].cW.T, X)
                tempnet.add_col_vec(self.model[H-1].hb)
                cm.exp(tempnet,target=tempexp)
                tempexp.copy_to_host()
                tempexp.numpy_array+=1
                templog = np.log(tempexp.numpy_array)
                F[i,batch*batchsize:(batch+1)*batchsize]= np.tile(cb[i],(1,batchdata.shape[1]))+sum(templog,0)
#            cm.dot(self.model[H-1].W.T, data, target = tempnet)
#            tempnet.add_dot(self.model[H-1].cW.T, X)
#            tempnet.add_col_vec(self.model[H-1].hb)
#            cm.exp(tempnet,target=tempexp)
#            tempexp.copy_to_host()
#            tempexp.numpy_array+=1
#            templog = np.log(tempexp.numpy_array)
#            self.model[H-1].cb.copy_to_host()
#            cb = self.model[H-1].cb.numpy_array
#            # this might be incorrect
            
        return F