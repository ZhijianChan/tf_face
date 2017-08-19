"""Contains the definition of the Inception ResNet V1 architecture.
paper:    http://arxiv.org/abs/1602.07261.
abstract: Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
authors:  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


# Inception-ResNet-A
# (3 branches)
def block35(x, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 35x35 resx block."""
    with tf.variable_scope(scope, 'Block35', [x], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(x, 32, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(x, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(x, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
        # tensor dimension: NxWxHxC, concat at dim-c
        mixed = tf.concat(values=[tower_conv, tower_conv1_1, tower_conv2_2], axis=3)
        # output_num of up should be equal to input_num of layer
        # "We used batch-normalization only on top of the traditional layers,
        #  not on top of the summations." --3.2, para3
        up = slim.conv2d(mixed, x.get_shape()[3], 1,
                         normalizer_fn=None,
                         activation_fn=None,
                         scope='Conv2d_1x1')
        x += scale * up
        if activation_fn:
            x = activation_fn(x)
    return x


# Inception-ResNet-B
# (2 branches)
def block17(x, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 17x17 ResNet block."""
    with tf.variable_scope(scope, 'Block17', [x], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(x, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(x, 128, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7], scope='Conv2d_0b_1x7')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1], scope='Conv2d_0c_7x1')
        mixed = tf.concat(values=[tower_conv, tower_conv1_2], axis=3)
        up = slim.conv2d(mixed, x.get_shape()[3], 1,
                         normalizer_fn=None,
                         activation_fn=None,
                         scope='Conv2d_1x1')
        x += scale * up
        if activation_fn:
            x = activation_fn(x)
    return x


# Inception-ResNet-C
# (2 branches)
def block8(x, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 8x8 ResNet block."""
    with tf.variable_scope(scope, 'Block8', [x], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(x, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(x, 192, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3], scope='Conv2d_0b_1x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1], scope='Conv2d_0c_3x1')
        mixed = tf.concat(values=[tower_conv, tower_conv1_2], axis=3)
        up = slim.conv2d(mixed, x.get_shape()[3], 1,
                         normalizer_fn=None,
                         activation_fn=None,
                         scope='Conv2d_1x1')
        x += scale * up
        if activation_fn:
            x = activation_fn(x)
    return x


# Reduce-A
# (3 branches)
def reduction_a(x, k, l, m, n):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(x, n, 3, scope='Conv2d_1a_3x3', stride=2, padding='VALID')
    with tf.variable_scope('Branch_1'):
        tower_conv1_0 = slim.conv2d(x, k, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1_0, l, 3, scope='Conv2d_0b_3x3')
        tower_conv1_2 = slim.conv2d(tower_conv1_1, m, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_pool = slim.max_pool2d(x, 3, stride=2, padding='VALID', scope='MaxPool_1a_3x3')
    x = tf.concat(values=[tower_conv, tower_conv1_2, tower_pool], axis=3)
    return x


# Reduce-B
# (4 branches)
def reduction_b(x):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(x, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv_1 = slim.conv2d(tower_conv, 384, 3, scope='Conv2d_1a_3x3', stride=2, padding='VALID')
    with tf.variable_scope('Branch_1'):
        tower_conv1 = slim.conv2d(x, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1, 256, 3, scope='Conv2d_1a_3x3', stride=2, padding='VALID')
    with tf.variable_scope('Branch_2'):
        tower_conv2 = slim.conv2d(x, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv2_1 = slim.conv2d(tower_conv2, 256, 3, scope='Conv2d_0b_3x3')
        tower_conv2_2 = slim.conv2d(tower_conv2_1, 256, 3, scope='Conv2d_1a_3x3', stride=2, padding='VALID')
    with tf.variable_scope('Branch_3'):
        tower_pool = slim.max_pool2d(x, 3, scope='MaxPool_1a_3x3', stride=2, padding='VALID')
    x = tf.concat(values=[tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], axis=3)
    return x


def inception_resnet_v2(inputs,
                        is_training=True,
                        keep_prob=0.8,
                        bottleneck_size=128,
                        reuse=None,
                        scope='Inception-ResNet-v2'):
    """Creates the Inception ResNet V2 model.

    :param inputs:  a 4-D tensor of size [batch_size, height, width, 3]
    :param is_training: whether is training or not
    :param keep_prob:  float, the fraction to keep before final layer
    :param bottleneck_size: bottleneck feature dimension
    :param reuse: whether or not the network and its variables should be reused.
    :param scope: optional variable_scope
    :return: prelogits, end_points
    """
    with tf.variable_scope(scope, 'Inception-ResNet-v2', [inputs], reuse=reuse):
        with slim.arg_scope(
                [slim.batch_norm, slim.dropout],
                is_training=is_training):
            with slim.arg_scope(
                    [slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                    stride=1, padding='SAME'):
                # 149x149x32
                x = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                # 147x147x32
                x = slim.conv2d(x, 32, 3, padding='VALID', scope='Conv2d_2a_3x3')
                # 147x147x64
                x = slim.conv2d(x, 64, 3, scope='Conv2d_2b_3x3')
                # 73x73x160
                with tf.variable_scope('Mixed_3a'):
                    x1 = slim.max_pool2d(x, 3, stride=2, padding='VALID', scope='MaxPool_3a_3x3')
                    x2 = slim.conv2d(x, 96, 3, stride=2, padding='VALID', scope='Conv2d_3a_3x3')
                    x = tf.concat(values=[x1, x2], axis=3)
                # 71x71x192
                with tf.variable_scope('Mixed_4a'):
                    with tf.variable_scope('Branch_0'):
                        x1 = slim.conv2d(x, 64, 1, scope='Conv2d_0a_1x1')
                        x1 = slim.conv2d(x1, 96, 3, padding='VALID', scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_1'):
                        x2 = slim.conv2d(x, 64, 1, scope='Conv2d_0a_1x1')
                        x2 = slim.conv2d(x2, 64, [7, 1], scope='Conv2d_0b_7x1')
                        x2 = slim.conv2d(x2, 64, [1, 7], scope='Conv2d_0c_1x7')
                        x2 = slim.conv2d(x2, 96, 3, padding='VALID', scope='Conv2d_0d_3x3')
                    x = tf.concat(values=[x1, x2], axis=3)
                # 35x35x384
                with tf.variable_scope('Mixed_5a'):
                    x1 = slim.conv2d(x, 192, 3, stride=2, padding='VALID', scope='Conv2d_0a_3x3')
                    x2 = slim.max_pool2d(x, 3, stride=2, padding='VALID', scope='MaxPool_0b_3x3')
                    x = tf.concat(values=[x1, x2], axis=3)
                # 35x35x384
                x = slim.repeat(x, 5, block35, scale=0.17)  # scale: 0.1~0.3
                # 17x17x1152
                with tf.variable_scope('Mixed_6a'):
                    x = reduction_a(x, 256, 256, 384, 384)
                # 17x17x1152
                x = slim.repeat(x, 10, block17, scale=0.10)
                # 8x8x2144
                with tf.variable_scope('Mixed_7a'):
                    x = reduction_b(x)
                # 8x8x2144
                x = slim.repeat(x, 6, block8, scale=0.20)
                with tf.variable_scope('AvgPool'):
                    # average_pool
                    # ** x.get_shape()      -> (n,w,h,c)
                    # ** x.get_shape()[1:3] -> (w,h)
                    # flatten
                    # ** (n,1,1,c) -> (n,c)
                    x = slim.avg_pool2d(x, x.get_shape()[1:3], padding='VALID', scope='AvgPool_1a_8x8')
                    x = slim.flatten(x)
                    x = slim.dropout(x, keep_prob, is_training=is_training, scope='Dropout')
                x = slim.fully_connected(
                    x, bottleneck_size,
                    activation_fn=None,
                    # normalizer_fn=None, #[test3: not add bn after fc
                    scope='Bottleneck',
                    reuse=False)
    return x, None


def inference(inputs, keep_prob,
              bottleneck_size=128,
              phase_train=True,
              weight_decay=0.0,
              reuse=None):
    batch_norm_params = {
        'decay': 0.995,
        'epsilon': 0.001,
        'updates_collections': None,
        # 'scale': True,  # [test1]
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES]}  # [test2: removed from 'trainable_variables']
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            weights_regularizer=slim.l2_regularizer(weight_decay),
            biases_regularizer=slim.l2_regularizer(weight_decay),  # [test4: add weight_decay to biases]):
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        return inception_resnet_v2(
            inputs,
            is_training=phase_train,
            keep_prob=keep_prob,
            bottleneck_size=bottleneck_size,
            reuse=reuse)
