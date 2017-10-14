#!/usr/bin/env python

from config import *
from feature import feature
import numpy as np
from numpy import random 
import tensorflow as tf
import tflearn

# using GPU numbered 0
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

def weight_variable(shape, name='W'):
    #with tf.variable_scope(name, initializer=tf.random_normal_initializer(),regularizer=regularizer):
    return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(), regularizer=regularizer)

def bias_variable(shape, name='b'):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name)

def print_shape(net):
    print net.get_shape().as_list()

def tfmgrid(L):
    v = tf.range(0, L, 1)
    x, v = tf.meshgrid(v, v)
    return tf.stack([v, v])

#incoming shape (batch_size, L(seqLen), feature_num)
#output[:, i, j, :] = incoming[:. i, :] + incoming[:, j, :] + incoming[:, (i+j)/2, :]
def seq2pairwise(incoming):
    L = tf.shape(incoming)[1]
    #save the indexes of each position
    index = tfmgrid(L)
    i = index[0]
    j = index[1]
    m = (i+j)/2
    #switch batch dim with L dim to put L at first
    incoming2 = tf.transpose(incoming, perm=[1, 0, 2])
    #full matrix i with element in incomming2 indexed i[i][j]
    out1 = tf.nn.embedding_lookup(incoming2, i)
    out2 = tf.nn.embedding_lookup(incoming2, j)
    out3 = tf.nn.embedding_lookup(incoming2, m)
    #concatante final feature dim together
    out = tf.concat([out1, out2, out3], axis=3)
    #return to original dims
    output = tf.transpose(out, perm=[2, 0, 1, 3])
    return output

def res_block_1d(incoming, out_channels, filter_size, block_name='res_block_1d'):
    net = incoming
    in_channels = incoming.get_shape().as_list()[-1]
    ident = net
    net = tflearn.conv_1d(net, out_channels, filter_size)
    net = tflearn.conv_1d(net, out_channels, filter_size)
    if in_channels != out_channels:
        ch = (out_channels - in_channels)//2
        remain = out_channels-in_channels-ch
        ident = tf.pad(ident, [[0, 0], [0, 0], [ch, remain]])
        in_channels = out_channels
    # Add the original featrues to result
    net = net + ident
    return net

def res_block_2d(incoming, out_channels, filter_size, block_name='res_block_2d'):
    net = incoming
    in_channels = incoming.get_shape().as_list()[-1]
    ident = net
    # 1st conv layer in residual block
    W1 = weight_variable([filter_size, filter_size, in_channels, out_channels], block_name+'W1')
    b1 = bias_variable([out_channels], block_name+'b1')
    net = tf.nn.relu(tf.nn.conv2d(net, W1, strides=[1,1,1,1], padding='SAME') + b1)
    # 2nd conv layer in residual block
    W2 = weight_variable([filter_size, filter_size, out_channels, out_channels], block_name+'W2')
    b2 = bias_variable([out_channels], block_name+'b2')
    net = tf.nn.relu(tf.nn.conv2d(net, W2, strides=[1,1,1,1], padding='SAME') + b2)  
    #print "in_channel = %d" %in_channels
    #print "out_channel = %d" %out_channels
    if in_channels != out_channels:
        ch = (out_channels - in_channels)//2
        remain = out_channels-in_channels-ch
        ident = tf.pad(ident, [[0, 0], [0, 0], [0, 0], [ch, remain]])
        in_channels = out_channels
    # Add the original featrues to result
    net = net + ident
    return net

def tf_model():

    x_seq = tf.placeholder("float", shape=[None, None, 26])
    x_pair = tf.placeholder("float", shape=[None, None, None, 5])
    
    #some basic parameters here
    block_num = 10
    filter_size = 3
    channel_step = 3
   
    #L2 norm

    ######## 1d Residual Network ##########
    net = x_seq
    out_channels = net.get_shape().as_list()[-1]
    for i in xrange(block_num):    #1D-residual blocks building
        out_channels += channel_step
        net = res_block_1d(net, out_channels, filter_size, 'resBLOCK_1D'+str(i+1))
    #######################################
    
    #conversion of sequential to pairwise feature
    net = seq2pairwise(net) 

    #merge coevolution info(pairwise potential) and above feature
    net = tf.concat([net, x_pair], axis=3)
    out_channels = net.get_shape().as_list()[-1]
    
    ######## 2d Residual Network ##########
    for i in xrange(block_num):    #2D-residual blocks building
        out_channels += channel_step
        net = res_block_2d(net, out_channels, filter_size, 'resBLOCK_2D'+str(i+1))
    #######################################

    # softmax channels of each pair into a score
    W_conv2 = weight_variable([1, 1, out_channels, 1], 'w2')
    b_conv2 = bias_variable([1], 'b2')
    y_conv = tf.nn.softmax(tf.nn.conv2d(net, W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2)
    saver = tf.train.Saver()

    y_conv = tf.squeeze(y_conv, squeeze_dims=3)
    #tf.reshape(y_conv, [-1])
    #tf.reshape(y_, [-1])
    y_ = tf.placeholder("float", shape=None)
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    
    #objective function: minimize the sum of loss function and the L2 norm
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    loss = cross_entropy + reg_term
    
    #training step
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    #correct_prediction = tf.greater(tf.argmax(y_conv,1), tf.argmax(y_,1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    #save_path = saver.save(sess, '/home/zhangqi/workspace/contact_map/tmp/model.ckpt')

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        F = feature(train_file, valid_file, test_file)
        train_data, valid_data, test_data = F.get_feature()
        X1 = train_data[0]
        X2 = train_data[1]
        Y = train_data[2]
        # training with one sample in a batch
        for i in xrange(1000):
            #batch = F.next_batch(1)
            x1 = X1[i][np.newaxis]
            x2 = X2[i][np.newaxis]
            y = Y[i][np.newaxis]
            sess.run(train_step, feed_dict={x_seq: x1, x_pair: x2, y_: y})
            if i % 5 == 0:
                train_loss = loss.eval(feed_dict={x_seq: x1, x_pair: x2, y_: y})
                print "step %d, loss = %g" %(i, train_loss)
                if i % 200 == 50:
                    print y_conv.eval(feed_dict={x_seq: x1, x_pair: x2, y_: y})
        #print "test accuracy = %g"%accuracy.eval(feed_dict={x_seq: aaa, x_pair: bbb, y_: ccc})


def main():
    F = feature(train_file, valid_file, test_file)
    train_data, valid_data, test_data = F.get_feature()
    for i in xrange(1):
        x1, x2, y = F.next_batch(2)
        print x1.shape, x2.shape, y.shape



tf_model()
