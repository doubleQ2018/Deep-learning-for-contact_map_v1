#!/usr/bin/env python

from config import *
from feature import feature
from evaluation import *
from operation import *
import numpy as np
from numpy import random 
import time
import tensorflow as tf
import tflearn

# using GPU numbered 0
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

class TFmodel(object):
    def __init__(self, filter_size_1d=17, filter_size_2d=3, block_1d=5, block_2d=5, regulation=False, batch_normalization=False):
        self.filter_size_1d = filter_size_1d
        self.filter_size_2d = filter_size_2d
        self.block_1d = block_1d
        self.block_2d = block_2d
        self.regulation = regulation
        self.BN = batch_normalization
        self.block1d_num = 0
        self.block2d_num = 0

    def TFmodelBuild(self):
        #tf.logging.set_verbosity(tf.logging.INFO)
        regularizer = None
        if self.regulation:
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

        x_seq = tf.placeholder("float", shape=[None, None, 26])
        x_pair = tf.placeholder("float", shape=[None, None, None, 5])
        y_ = tf.placeholder("float", shape=None)


        channel_step = 2
        with tf.name_scope('input_1d'):
            net = x_seq
        print "Input channels = %d" %net.get_shape().as_list()[-1] 

        ######## 1d Residual Network ##########
        out_channels = net.get_shape().as_list()[-1]
        with tf.name_scope('conv_1d'):
            for i in xrange(self.block_1d):    #1D-residual blocks building
                self.block1d_num += 1
                out_channels += channel_step
                with tf.name_scope("block-" + str(self.block1d_num)):
                    net = res_block_1d(net, out_channels, self.filter_size_1d, regularizer, batch_norm=self.BN, block_name='resBLOCK_1D'+str(i+1))
                
        #######################################
        
        print "After conv_1d channels = %d" %net.get_shape().as_list()[-1] 

        with tf.name_scope('1d_to_2d'):
            #conversion of sequential to pairwise feature
            net = seq2pairwise(net) 

            #merge coevolution info(pairwise potential) and above feature
        if self.block_1d == 0:
            net = x_pair
        else:
            net = tf.concat([net, x_pair], axis=3)
        out_channels = net.get_shape().as_list()[-1]
        
        print "Add 1d to 2d, channels = %d" %net.get_shape().as_list()[-1] 

        ######## 2d Residual Network ##########
        with tf.name_scope('conv_2d'):
            for i in xrange(self.block_2d):    #2D-residual blocks building
                self.block2d_num += 1
                out_channels += channel_step
                with tf.name_scope("block-" + str(self.block1d_num)):
                    net = res_block_2d(net, out_channels, self.filter_size_2d, regularizer, batch_norm=self.BN, block_name='BLOCK_2D'+str(i+1))
                    #tf.summary.scalar('output', net)
        #######################################

        print "After conv_2d channels = %d" %net.get_shape().as_list()[-1] 

        # softmax channels of each pair into a score
        with tf.name_scope('softmax_layer'):
            W_out = weight_variable([1, 1, out_channels, 2], regularizer, 'W')
            b_out = bias_variable([2], 'b')
            output_prob = tf.nn.softmax(tf.nn.conv2d(net, W_out, strides=[1,1,1,1], padding='SAME') + b_out)
            #tf.summary.scalar('output_prob', output_prob)
        
        with tf.name_scope('loss_function'):
            loss = loss1(output_prob, y_)
            if self.regulation:
                reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
                loss += reg_term
            tf.summary.scalar('loss', loss)

        with tf.name_scope('training'):
            train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        #top_k_indeces = self.TOP_k(output_prob, y_)
    
        saver = tf.train.Saver() 
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            merged_summary_op = tf.summary.merge_all() 
            summary_writer = tf.summary.FileWriter(logs_file + time.strftime("%Y-%m-%d-%H-%M", time.localtime()), sess.graph)

            ### Loading database
            F = feature(train_file, valid_file, test_file)
            train_data, valid_data, test_data = F.get_feature()
            X1 = train_data[0]
            X2 = train_data[1]
            Y = train_data[2]
            # training with one sample in a batch
            for i in xrange(len(X1)):
                x1 = X1[i][np.newaxis]
                x2 = X2[i][np.newaxis]
                y = Y[i][np.newaxis]
                print "L = %d" %y.shape[-2]
                sess.run(train_step, feed_dict={x_seq: x1, x_pair: x2, y_: y})
                train_loss = loss.eval(feed_dict={x_seq: x1, x_pair: x2, y_: y})
                print "step %d, loss = %g" %(i, train_loss)
                if i % 10 == 0:
                    summary_str, y_out = sess.run([merged_summary_op, output_prob], feed_dict={x_seq: x1, x_pair: x2, y_: y})
                    summary_writer.add_summary(summary_str, i)
                    print "training %d, accuracy = %f" %(i+1, accuracy(y_out, y, 2))

            saver.save(sess, "Models/model.ckpt")

            X1_test = test_data[0]
            X2_test = test_data[1]
            Y_test = test_data[2]
            for i in xrange(len(X1_test)):
                x1 = X1_test[i][np.newaxis]
                x2 = X2_test[i][np.newaxis]
                y = Y_test[i][np.newaxis]
                y_out = sess.run(output_prob, feed_dict={x_seq: x1, x_pair: x2, y_: y})
                print "testing %d, accuracy = %f" %(i+1, accuracy(y_out, y, 2))


M = TFmodel(filter_size_1d=17, filter_size_2d=3, block_1d=3, block_2d=5, regulation=True, batch_normalization=True) 
#M = TFmodel(filter_size_1d=17, filter_size_2d=3, block_1d=0, block_2d=20, regulation=True, batch_normalization=True) # Training without 1d feature
M.TFmodelBuild()
