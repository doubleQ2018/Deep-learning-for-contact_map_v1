#!/usr/bin/env python

import tensorflow as tf

def weight_variable(shape, regularizer, name="W"):
    if regularizer == None:
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name)
    else:
        return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(), regularizer=regularizer)


def bias_variable(shape, name="b"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name)

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

#achieve np.mgrid with tensorflow
def tfmgrid(L):
    v = tf.range(0, L, 1)
    x, v = tf.meshgrid(v, v)
    return tf.stack([v, v])

### Incoming shape (batch_size, L(seqLen), feature_num)
### Output[:, i, j, :] = incoming[:. i, :] + incoming[:, j, :] + incoming[:, (i+j)/2, :]
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

def res_block_1d(incoming, out_channels, filter_size, regularizer, batch_norm=False, scope=None, name="ResidualBlock_1d"):
    net = incoming
    in_channels = incoming.get_shape().as_list()[-1]
    ident = net
    with tf.variable_scope(scope, default_name = name, values=[incoming]) as scope:
        # 1st conv layer in residual block
        W1 = weight_variable([filter_size, in_channels, out_channels], regularizer, name="W1")
        #variable_summaries(W1)
        b1 = bias_variable([out_channels], name="b1")
        #variable_summaries(b1)
        net = tf.nn.conv1d(net, W1, stride=1, padding='SAME') + b1
        ### Add batch nomalization
        if batch_norm:
            net = tf.contrib.layers.batch_norm(net)
        net = tf.nn.relu(net)
        # 2nd conv layer in residual block
        W2 = weight_variable([filter_size, out_channels, out_channels], regularizer, name="W2")
        #variable_summaries(W2)
        b2 = bias_variable([out_channels], name="b2")
        #variable_summaries(b2)
        net = tf.nn.conv1d(net, W2, stride=1, padding='SAME') + b2
        ### Add batch nomalization
        if batch_norm:
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

def res_block_2d(incoming, out_channels, filter_size, regularizer, batch_norm=False, scope=None, name="ResidualBlock_2d"):
    net = incoming
    in_channels = incoming.get_shape().as_list()[-1]
    ident = net
    with tf.variable_scope(scope, default_name = name, values=[incoming]) as scope:
        # 1st conv layer in residual block
        W1 = weight_variable([filter_size, filter_size, in_channels, out_channels], regularizer, name="W1")
        #variable_summaries(W1)
        b1 = bias_variable([out_channels], name="b1")
        #variable_summaries(b1)
        net = tf.nn.conv2d(net, W1, strides=[1,1,1,1], padding='SAME') + b1
        ### Add batch nomalization
        if batch_norm:
            net = tf.contrib.layers.batch_norm(net)
        net = tf.nn.relu(net)
        ### 2nd conv layer in residual block
        W2 = weight_variable([filter_size, filter_size, out_channels, out_channels], regularizer, name="W2")
        #variable_summaries(W2)
        b2 = bias_variable([out_channels], name="b2")
        #variable_summaries(b2)
        net = tf.nn.conv2d(net, W2, strides=[1,1,1,1], padding='SAME') + b2
        ### Add batch nomalization
        if batch_norm:
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

def loss_function(output_prob, y, weight=None):
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
            
def loss1(output_prob, y, weight=None):
    return -tf.reduce_mean(tf.multiply(tf.log(tf.clip_by_value(output_prob,1e-10,1.0)), y))
        #los = -tf.reduce_mean(tf.multiply(weight, tf.multiply(tf.log(tf.clip_by_value(output_prob,1e-10,1.0)), y)))

