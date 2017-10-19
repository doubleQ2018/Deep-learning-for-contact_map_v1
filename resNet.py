#!/usr/bin/env python

from config import *
from feature import feature
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

def accuracy(y_out, y, k):
    L = y.shape[-2]
    top_k = L/k
    y_out = y_out[:,:,:,1:]
    y = y[:,:,:,1:] ### when input shape of y is (, L, L, 2)
    y_flat = y_out.flatten()
    y = y.flatten()
    indeces = [[index, val] for index, val in enumerate(y_flat)]
    indeces = sorted(indeces, key=lambda x: x[1], reverse=True)
    right = 0
    for i in range(top_k):
        right += y[indeces[i][0]]
    return 1.0 * right / top_k

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


    def weight_variable(self, shape, regularizer, name='W'):
        if regularizer == None:
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)
        else:
            return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(), regularizer=regularizer)


    def bias_variable(self, shape, name='b'):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name)


    #achieve np.mgrid with tensorflow
    def tfmgrid(self, L):
        v = tf.range(0, L, 1)
        x, v = tf.meshgrid(v, v)
        return tf.stack([v, v])

    ### Incoming shape (batch_size, L(seqLen), feature_num)
    ### Output[:, i, j, :] = incoming[:. i, :] + incoming[:, j, :] + incoming[:, (i+j)/2, :]
    def seq2pairwise(self, incoming):
        L = tf.shape(incoming)[1]
        #save the indexes of each position
        index = self.tfmgrid(L)
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

    def res_block_1d(self, incoming, out_channels, filter_size, regularizer, block_name='BLOCK_1D'):
        net = incoming
        in_channels = incoming.get_shape().as_list()[-1]
        ident = net
        # 1st conv layer in residual block
        W1 = self.weight_variable([filter_size, in_channels, out_channels], regularizer, block_name+'_W1')
        #variable_summaries(W1)
        b1 = self.bias_variable([out_channels], 'b1')
        #variable_summaries(b1)
        net = tf.nn.conv1d(net, W1, stride=1, padding='SAME') + b1
        ### Add batch nomalization
        if self.BN:
            net = tf.contrib.layers.batch_norm(net)
        net = tf.nn.relu(net)
        # 2nd conv layer in residual block
        W2 = self.weight_variable([filter_size, out_channels, out_channels], regularizer, block_name+'_W2')
        #variable_summaries(W2)
        b2 = self.bias_variable([out_channels], 'b2')
        #variable_summaries(b2)
        net = tf.nn.conv1d(net, W2, stride=1, padding='SAME') + b2
        ### Add batch nomalization
        if self.BN:
            net = tf.contrib.layers.batch_norm(net)
        net = tf.nn.relu(net)  
        if in_channels != out_channels:
            ch = (out_channels - in_channels)//2
            remain = out_channels-in_channels-ch
            ident = tf.pad(ident, [[0, 0], [0, 0], [ch, remain]])
            in_channels = out_channels
        # Add the original featrues to result, identify
        net = net + ident
        return net

    def res_block_2d(self, incoming, out_channels, filter_size, regularizer, block_name='BLOCK_2D'):
        net = incoming
        in_channels = incoming.get_shape().as_list()[-1]
        ident = net
        # 1st conv layer in residual block
        W1 = self.weight_variable([filter_size, filter_size, in_channels, out_channels], regularizer, block_name+'_W1')
        #variable_summaries(W1)
        b1 = self.bias_variable([out_channels], 'b1')
        #variable_summaries(b1)
        net = tf.nn.conv2d(net, W1, strides=[1,1,1,1], padding='SAME') + b1
        ### Add batch nomalization
        if self.BN:
            net = tf.contrib.layers.batch_norm(net)
        net = tf.nn.relu(net)
        ### 2nd conv layer in residual block
        W2 = self.weight_variable([filter_size, filter_size, out_channels, out_channels], regularizer, block_name+'_W2')
        #variable_summaries(W2)
        b2 = self.bias_variable([out_channels], 'b2')
        #variable_summaries(b2)
        net = tf.nn.conv2d(net, W2, strides=[1,1,1,1], padding='SAME') + b2
        ### Add batch nomalization
        if self.BN:
            net = tf.contrib.layers.batch_norm(net)
        net = tf.nn.relu(net)  
        if in_channels != out_channels:
            ch = (out_channels - in_channels)//2
            remain = out_channels-in_channels-ch
            ident = tf.pad(ident, [[0, 0], [0, 0], [0, 0], [ch, remain]])
            in_channels = out_channels
        ### Add the original featrues to result
        net = net + ident
        return net

    def loss(self, output_prob, y, weight=None):
        y_flat = tf.reshape(y, [-1])
        if weight is not None:
            w_flat = tf.reshape(weight, [-1])

        n_labels = tf.shape(output_prob)[3]
        out_flat = tf.reshape(output_prob, [-1, n_labels])
        
        # Achieve advance indexing throught tensorflow operator
        batch_nums = tf.range(0, limit=tf.shape(out_flat)[0])
        indices = tf.stack((batch_nums, y_flat), axis=1)
        if weight is not None:
            los = -tf.sum(tf.mul(w_flat, tf.gather_nd(tf.log(tf.clip_by_value(out_flat,1e-10,1.0)), indices))) / tf.sum(w_flat)
        else:
            los = -tf.reduce_mean(tf.gather_nd(tf.log(tf.clip_by_value(out_flat,1e-10,1.0)), indices))
        return los
                
    def loss1(self, output_prob, y, weight=None):
        return -tf.reduce_mean(tf.multiply(tf.log(tf.clip_by_value(output_prob,1e-10,1.0)), y))

    def TFmodelBuild(self):
        tf.logging.set_verbosity(tf.logging.INFO)
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
                    net = self.res_block_1d(net, out_channels, self.filter_size_1d, regularizer, 'resBLOCK_1D'+str(i+1))
                    #tf.summary.scalar('output', net)
                
        #######################################
        
        print "After conv_1d channels = %d" %net.get_shape().as_list()[-1] 

        with tf.name_scope('1d_to_2d'):
            #conversion of sequential to pairwise feature
            net = self.seq2pairwise(net) 

            #merge coevolution info(pairwise potential) and above feature
            net = tf.concat([net, x_pair], axis=3)
            #tf.summary.histogram('output', net)
        out_channels = net.get_shape().as_list()[-1]
        
        print "Add 1d to 2d, channels = %d" %net.get_shape().as_list()[-1] 

        ######## 2d Residual Network ##########
        with tf.name_scope('conv_2d'):
            for i in xrange(self.block_2d):    #2D-residual blocks building
                self.block2d_num += 1
                out_channels += channel_step
                with tf.name_scope("block-" + str(self.block1d_num)):
                    net = self.res_block_2d(net, out_channels, self.filter_size_2d, regularizer, 'BLOCK_2D'+str(i+1))
                    #tf.summary.scalar('output', net)
        #######################################

        print "After conv_2d channels = %d" %net.get_shape().as_list()[-1] 

        # softmax channels of each pair into a score
        with tf.name_scope('softmax_layer'):
            W_out = self.weight_variable([1, 1, out_channels, 2], regularizer, 'W')
            b_out = self.bias_variable([2], 'b')
            output_prob = tf.nn.softmax(tf.nn.conv2d(net, W_out, strides=[1,1,1,1], padding='SAME') + b_out)
            #tf.summary.scalar('output_prob', output_prob)
        
        with tf.name_scope('loss_function'):
            loss = self.loss1(output_prob, y_)
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


M = TFmodel(filter_size_1d=17, filter_size_2d=3, block_1d=1, block_2d=15, regulation=True, batch_normalization=True)
M.TFmodelBuild()
