"""Contains the definition of the Inception Resnet V1 architecture.
paper:    http://arxiv.org/abs/1602.07261.
abstract: Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
authors:  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

# Inception-Renset-A
# (3 branches)
def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 35x35 resnet block."""
    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv    = slim.conv2d(net,           32, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net,           32, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net,           32, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 32, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 32, 3, scope='Conv2d_0c_3x3')
        # tensor dimension: NxWxHxC, concat at dim-c
        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
        # output_num of up should be equal to input_num of layer
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net

# Inception-Renset-B
# (2 branches)
def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 17x17 resnet block."""
    with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv    = slim.conv2d(net,           128, 1,      scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net,           128, 1,      scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 128, [1, 7], scope='Conv2d_0b_1x7')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 128, [7, 1], scope='Conv2d_0c_7x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net

# Inception-Resnet-C
# (2 branches)
def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 8x8 resnet block."""
    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv    = slim.conv2d(net,           192, 1,      scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net,           192, 1,      scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 192, [1, 3], scope='Conv2d_0b_1x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [3, 1], scope='Conv2d_0c_3x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net
  
# 35x35x256 -> 17x17x896
# (3 branches)
def reduction_a(net, k, l, m, n):
    with tf.variable_scope('Branch_0'):
        tower_conv    = slim.conv2d(net,           n, 3, scope='Conv2d_1a_3x3', stride=2, padding='VALID')
    with tf.variable_scope('Branch_1'):
        tower_conv1_0 = slim.conv2d(net,           k, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1_0, l, 3, scope='Conv2d_0b_3x3')
        tower_conv1_2 = slim.conv2d(tower_conv1_1, m, 3, scope='Conv2d_1a_3x3', stride=2, padding='VALID')
    with tf.variable_scope('Branch_2'):
        tower_pool    = slim.max_pool2d(net, 3, scope='MaxPool_1a_3x3', stride=2, padding='VALID')
    net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
    return net

# 17x17x896 -> 8x8x1792
# (4 braches)
def reduction_b(net):
    with tf.variable_scope('Branch_0'):
        tower_conv    = slim.conv2d(net,           256, 1, scope='Conv2d_0a_1x1')
        tower_conv_1  = slim.conv2d(tower_conv,    384, 3, scope='Conv2d_1a_3x3', stride=2, padding='VALID')
    with tf.variable_scope('Branch_1'):
        tower_conv1   = slim.conv2d(net,           256, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1,   256, 3, scope='Conv2d_1a_3x3', stride=2, padding='VALID')
    with tf.variable_scope('Branch_2'):
        tower_conv2   = slim.conv2d(net,           256, 1, scope='Conv2d_0a_1x1')
        tower_conv2_1 = slim.conv2d(tower_conv2,   256, 3, scope='Conv2d_0b_3x3')
        tower_conv2_2 = slim.conv2d(tower_conv2_1, 256, 3, scope='Conv2d_1a_3x3', stride=2, padding='VALID')
    with tf.variable_scope('Branch_3'):
        tower_pool = slim.max_pool2d(net, 3, scope='MaxPool_1a_3x3', stride=2, padding='VALID')
    net = tf.concat([tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], 3)
    return net

def inception_resnet_v1(inputs,
        is_training = True,
        keep_prob = 0.8,
        bottleneck_size = 128,
        reuse = None,
        scope = 'Inception-Resnet-v1'):
    """Creates the Inceptiion Resnet V1 model.
    Args:
        inputs: a 4-D tensor of size [batch_size, height, width, 3].
        is_training: training phase flag.
        keep_prob: float, dropout probability.
        reuse: whether or not the network and its variables should be reused.
        scope: Optional variable_scope.
    Returns:
        logits: the logits outputs of the model.
        end_points: the set of end_points rrom the inception model.
    """
    end_points = {}
    
    with tf.variable_scope(scope, 'Inception-Resnet-v1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                    stride=1, padding='SAME'):

                net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                end_points['Conv2d_1a_3x3'] = net
                net = slim.conv2d(net, 32, 3, padding='VALID', scope='Conv2d_2a_3x3')
                end_points['Conv2d_2a_3x3'] = net
                net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                end_points['Conv2d_2b_3x3'] = net
                net = slim.max_pool2d(net, 3, stride=2,padding='VALID', scope='MaxPool_3a_3x3')
                end_points['MaxPool_3a_3x3'] = net
                net = slim.conv2d(net, 80, 1, scope='Conv2d_3b_1x1')
                end_points['Conv2d_3b_1x1'] = net
                net = slim.conv2d(net, 192, 3, padding='VALID', scope='Conv2d_4a_3x3')
                end_points['Conv2d_4a_3x3'] = net
                net = slim.conv2d(net, 256, 3, stride=2, padding='VALID', scope='Conv2d_4b_3x3')
                end_points['Conv2d_4b_3x3'] = net

                net = slim.repeat(net, 5, block35, scale=0.17)  # scale: 0.1~0.3

                with tf.variable_scope('Mixed_6a'):
                    net = reduction_a(net, 192, 192, 256, 384)
                end_points['Mixed_6a'] = net

                net = slim.repeat(net, 10, block17, scale=0.10)

                with tf.variable_scope('Mixed_7a'):
                    net = reduction_b(net)
                end_points['Mixed_7a'] = net

                net = slim.repeat(net, 5, block8, scale=0.20)
                #net = block8(net, activation_fn=None)

                with tf.variable_scope('AvgPool'):
                    end_points['PrePool'] = net
                    # average_pool
                    # ** net.get_shape()      -> (n,w,h,c)
                    # ** net.get_shape()[1:3] -> (w,h)
                    # flatten
                    # ** (n,1,1,c) -> (n,c)
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)
                    net = slim.dropout(net, keep_prob, is_training=is_training, scope='Dropout')
                    end_points['PreLogitsFlatten'] = net
                # Fc1
                net = slim.fully_connected(net, bottleneck_size, activation_fn=None, scope='Bottleneck', reuse=False)

    return net, end_points

def inference(images, keep_prob,
        bottleneck_size=128,
        phase_train=True, 
        weight_decay=0.0,
        reuse=None):
    batch_norm_params = {
        'decay': 0.995,
        'epsilon': 0.001,
        'updates_collections': None,
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES] }
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            weights_regularizer=slim.l2_regularizer(weight_decay),
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        return inception_resnet_v1(images, is_training=phase_train, keep_prob=keep_prob, bottleneck_size=bottleneck_size, reuse=reuse)
