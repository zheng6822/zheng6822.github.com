# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 20:25:45 2012

@author: zhengxin
"""

import time
import numpy as np
import cudamat as cm
import util
import scipy.io

# initialize CUDA
cm.cublas_init()
cm.CUDAMatrix.init_random(1)

#load data
mat_dict = scipy.io.loadmat('/home/zhengxin/Work/TIMIT/timit_MFCC_0_D_A_11fr_norm_mini.mat')
dev_dat = mat_dict['minidata']
#mat_dict = scipy.io.loadmat('/home/zhengxin/Work/TIMIT/timit_MFCC_0_D_A_11fr_norm.mat')
#dev_dat = mat_dict['data']
dev_dat = dev_dat.T
print dev_dat.shape
#util.load('mnist.dat', globals())
#dev_dat = cm.CUDAMatrix(cm.reformat(dat/255.))

# training parameters
epsilon = 0.002
momentum = 0.9

num_epochs = 300
batch_size = 128
num_batches = dev_dat.shape[1]/batch_size

# model parameters
num_vis = dev_dat.shape[0]
num_hid = 1096

# initialize weights
w_vh = cm.CUDAMatrix(0.1 * np.random.randn(num_vis, num_hid))
w_v = cm.CUDAMatrix(np.zeros((num_vis, 1)))
#w_vsigma = cm.CUDAMatrix(np.ones((num_vis,1)))
#w_vds = cm.CUDAMatrix(np.zeros((num_vis,num_hid)))
w_h = cm.CUDAMatrix(-4.*np.ones((num_hid, 1)))

# initialize weight updates
wu_vh = cm.CUDAMatrix(np.zeros((num_vis, num_hid)))
wu_v = cm.CUDAMatrix(np.zeros((num_vis, 1)))
#wu_vsigma = cm.CUDAMatrix(np.zeros((num_vis, 1)))
wu_h = cm.CUDAMatrix(np.zeros((num_hid, 1)))

vh_inc = cm.CUDAMatrix(np.zeros((num_vis, num_hid)))
v_inc = cm.CUDAMatrix(np.zeros((num_vis,1)))
h_inc = cm.CUDAMatrix(np.zeros((num_hid,1)))

# initialize temporary storage
v = cm.empty((num_vis, batch_size))
vm = cm.empty((num_vis, batch_size))
h = cm.empty((num_hid, batch_size))
r = cm.empty((num_hid, batch_size))
r2 = cm.empty((num_vis, batch_size))

start_time = time.time()
for epoch in range(num_epochs):
    #print "Epoch " + str(epoch + 1)
    err = 0

    for batch in range(num_batches):
        # get current minibatch
        batchdat = dev_dat[:,batch*batch_size:(batch+1)*batch_size]
        #print batchdat
        #print batchdat.shape
        v = cm.CUDAMatrix(batchdat)
        #v.copy_to_host()
        #print v.numpy_array.T
        v_true = cm.CUDAMatrix(batchdat)
        #v_true = dev_dat.slice(batch*batch_size,(batch + 1)*batch_size)
        #v.assign(v_true)

        # apply momentum
#        wu_vh.mult(momentum)
#        wu_v.mult(momentum)
#        wu_h.mult(momentum)
        wu_vh.mult(0)
        wu_v.mult(0)
        wu_h.mult(0)
        vh_inc.mult(momentum)
        v_inc.mult(momentum)
        h_inc.mult(momentum)

        # positive phase
  #      cm.divide(v, w_vsigma, target = w_vds)
        cm.dot(w_vh.T, v, target = h)
        h.add_col_vec(w_h)
        h.apply_sigmoid()#now h = sigmoid(W'v + b)

        wu_vh.add_dot(v, h.T)#first term of dW
        
        vm.assign(v)
        w_v.mult(-1)
        vm.add_col_vec(w_v)
        w_v.mult(-1)#these 4 lines make vm = data-c and c unchanged
        
        wu_v.add_sums(vm, axis = 1)#first term of dc
        wu_h.add_sums(h, axis = 1)#first term of db

        # sample hiddens
        r.fill_with_rand()
        r.less_than(h, target = h)# now h = phstates

        # negative phase
        cm.dot(w_vh, h, target = v)
        v.add_col_vec(w_v)#now v = Wh + c
        r2.fill_with_randn()
#        print type(r)
#        print type(r.T)
        v.add(r2)# now v = negdatastates
        #v.apply_sigmoid()

        cm.dot(w_vh.T, v, target = h)
        h.add_col_vec(w_h)
        h.apply_sigmoid()# h = sigmoid(W*negstates+b) namely nh

        wu_vh.subtract_dot(v, h.T)
        vm.assign(v)
        w_v.mult(-1)
        vm.add_col_vec(w_v)
        w_v.mult(-1)
        wu_v.add_sums(vm, axis = 1, mult = -1.)
        wu_h.add_sums(h, axis = 1, mult = -1.)
        vh_inc.add_mult(wu_vh, epsilon/batch_size)
        v_inc.add_mult(wu_v, epsilon/batch_size)
        h_inc.add_mult(wu_h, epsilon/batch_size)

        # update weights
#        w_vh.add_mult(vh_inc, epsilon/batch_size)
#        w_v.add_mult(v_inc, epsilon/batch_size)
#        w_h.add_mult(h_inc, epsilon/batch_size)
        w_vh.add(vh_inc)
        w_v.add(v_inc)
        w_h.add(h_inc)

        # calculate reconstruction error
        v.subtract(v_true)
        #print v.euclid_norm()
        #err.append(v.euclid_norm()**2/(num_vis*batch_size))
        #err.append(v.euclid_norm()**2/(num_vis*batch_size))
        err = err + v.euclid_norm()

    print "Ended epoch %i/%i. Reconstruction error is %f\n"% epoch, num_epochs, err
#    print "Mean squared error: " + str(err)
    #print "Time: " + str(time.time() - start_time)

#w_vh.copy_to_host()
#util.save('weights.dat', 'w_vh', {'w_vh': w_vh.numpy_array})

cm.cublas_shutdown()