# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 20:23:12 2012

@author: zhengxin
"""

from dbn import DBN
from rbm import *
import scipy.io
from options import options
import cudamat as cm
from neuralnet import NeuralNet

#datadict = scipy.io.loadmat('timit_MFCC_0_D_A_11fr_norm_mini.mat')
#data = datadict['minidata'].T
#labels = datadict['minilabel']
datadict = scipy.io.loadmat('timit_MFCC_0_D_A_11fr_norm.mat')
data = datadict['data'].T
labels = datadict['label']
testdatadict = scipy.io.loadmat('timit_MFCC_0_D_A_11fr_coretest_norm.mat')
testdata = testdatadict['testdata'].T
testlabels = testdatadict['testlabel']
devdatadict = scipy.io.loadmat('timit_MFCC_0_D_A_11fr_dev_norm.mat')
devdata = devdatadict['data'].T
devlabels = devdatadict['label']


#batchsize = 4096
#numbatch = data.shape[1]/batchsize
#for batch in range(numbatch):
#    batchdata = data[:,batch*batchsize:(batch+1)*batchsize]
#    batchlabels = labels[batch*batchsize:(batch+1)*batchsize]
#    mdict = {}
#    mdict['data'] = batchdata
#    mdict['label'] = batchlabels
#    #print batchlabels.shape
#    scipy.io.savemat('%s/%d.mat'%('data',batch),mdict)
#print 'data loaded'

#config = []
#option1 = options()
#option1.eta = 0.002
#option1.maxepoch=50
#option1.num_hid = 2100
#option2 = options()
#option2.eta = 0.02
#option2.maxepoch=100
#option2.num_hid = 2000
#option3 = options()
#option3.eta = 0.02
#option3.maxepoch=100
#option3.num_hid=2000
#option4 = options()
#option4.eta = 0.02
#option4.maxepoch = 20
#option4.num_hid= 3000
#option5 = options()
#option5.eta = 0.02
#option5.maxepoch = 100
#option5.num_hid =61
#config.append(option1)
#config.append(option5)
#config.append(option3)

# initialize CUDA
print 'initializing CUDA'
cm.cublas_init()
cm.CUDAMatrix.init_random(1)
print 'CUDA initialized'

config = []
option0 = options()
option0.eta = 0.002
option0.maxepoch = 200
option0.num_hid = 2000
option1 = options()
option1.eta = 0.02
option1.maxepoch = 100
option1.num_hid = 2000
option2 = options()
option2.eta = 0.01
option2.maxepoch = 200
option2.num_hid = 61
config.append(option1)#gaussian layer
config.append(option1)#binary layer2
config.append(option1)#binary layer3
config.append(option1)#binary layer4
config.append(option1)#binary layer5
config.append(option2)#softmax layer6

#model1 = GaussianRBM('data', option0.num_hid, option0)
#model1.train()
#model1.save('gauss_2000.mat')
#model1.getDataUpwards('data', 'gauss')
#model2 = BinaryRBM('gauss',option1.num_hid, option1)
#model2.train()
#model2.save('layer2_2000.mat')
#model2.getDataUpwards('gauss','layer2')
#model3 = BinaryRBM('layer2',option1.num_hid, option1)
#model3.train()
#model3.save('layer3_2000.mat')
#model3.getDataUpwards('layer2','layer3')
#model4 = BinaryRBM('layer3',option1.num_hid, option1)
#model4.train()
#model4.save('layer4_2000.mat')
#model4.getDataUpwards('layer3','layer4')
#model4 = BinaryRBM('layer4',option1.num_hid, option1)
#model4.train()
#model4.save('layer5_2000.mat')
#model4.getDataUpwards('layer4','layer5')
#model = SoftmaxRBM('layer5',option2.num_hid,option2)
#model.train()
#model.save('smtop6_2000.mat')

modelnamelist = []
modelnamelist.append('gauss_2000.mat')
#for i in range(2,6):
#    modelnamelist.append('layer%d_2000.mat'%i)
#modelnamelist.append('smtop6_2000.mat')
#nn = NeuralNet(data, labels, testdata, testlabels, devdata, devlabels, 'fe', 'pf', config)
#nn.load(modelnamelist)
#nn.fineTuning('data',100)
#nn.save('nn6_2000.mat')
nn = NeuralNet(data, labels, testdata, testlabels, devdata, devlabels, 'fe', 'pf', config)
nn.loadNN('nn6_2000.mat')
nn.fineTuning('data',100)
nn.save('nn6_2000_200.mat')

#print 'model saved'
#model1.load('gauss_2100.mat')
#model1.getDataUpwards(data)


#model3 = BinaryRBM('layer2',option2.num_hid, option2)
#model3.train()
#model3.save('layer3_2000.mat')
#model3.getDataUpwards('layer2','layer3')

#
#model = DiscriminativeRBM('gauss',labels, option4.num_hid, option4)
#model.train()
#model.save('top2_3000.mat')
#
#model = SoftmaxRBM('gauss',option5.num_hid,option5)
#model.train()
#model.save('smtop2_2100.mat')
#==============================================================================
#  the code below is for fine tuning
#==============================================================================
#modelnamelist = []
#modelnamelist.append('gauss_2100.mat')
#modelnamelist.append('smtop2_2100.mat')
#nn = NeuralNet(data, labels, testdata, testlabels, devdata, devlabels, 'fe', 'pf', config)
#nn.load(modelnamelist)
#nn.fineTuning('data',100)
#
#modelnamelist = []
#modelnamelist.append('gauss_2000.mat')
#modelnamelist.append('top2_3000.mat')
#dbn = DBN(data,labels,testdata,testlabels)
#dbn.load(modelnamelist)
#print 'models loaded'
#prediction = dbn.getPredict()
#print sum(prediction==testlabels[:,0])*1.0/len(prediction)


##modelnamelist.append('layer3_2000.mat')
#modelnamelist.append('top2_3000.mat')
#dbn = DBN(data,labels,testdata,testlabels)
#dbn.load(modelnamelist)
#print 'models loaded'
#prediction = dbn.getPredict()
#print sum(prediction==testlabels[:,0])*1.0/len(prediction)

#dbn = dbn(data, labels, testdata, testlabels, config =config)
#print 'training...'
#dbn.fit()
#print 'predicting...'
#prediction = dbn.getPredict()
#print prediction.shape
#print testlabels.shape
#
#print sum(prediction==testlabels[:,0])*1.0/len(prediction)

cm.cublas_shutdown()