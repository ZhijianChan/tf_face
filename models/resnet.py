import tensorflow as tf
import numpy as np
import sys

sys.path.insert(0, '../YelloFin')
from YellowFin.tuner_utils.yellowfin import YFOptimizer

from tensorflow.python.training import moving_averages
from collections import namedtuple

HParams = namedtuple('HParams',
                     'batch_size, num_residual_units, '
                     'use_bottleneck, relu_leakiness')


class ResNet(object):
    """ResNet constructor"""

    def __init__(self, hps, phase_train=True):
        self.hps = hps
        self.phase_train = phase_train
        self._extract_train_ops = []
        self.global_step = None

    @staticmethod
    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d"""
        return [1, stride, stride, 1]

    @staticmethod
    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]  # shape of 'x': [n, w, h, c] -> params_shape: c
            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

            if self.phase_train == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
                moving_mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32))
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32))
                self._extract_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self._extract_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)
                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)

            # elipson used to be 1e-5. Maybe 1e-3 solves Nan problem in deeper net?
            y = tf.nn.batch_normalization(
                x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    @staticmethod
    def _relu(self, x, leakiness=0.0):
        """leakly relu?"""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    @staticmethod
    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    @staticmethod
    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

    def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                             activate_before_residual=False):
        """Bottleneck residual unit with 3 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('common_bn_relu'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope('residual_bn_relu'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 1, in_filter, out_filter / 4, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv('conv2', x, 3, out_filter / 4, out_filter / 4, [1, 1, 1, 1])

        with tf.variable_scope('sub3'):
            x = self._batch_norm('bn3', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv('conv3', x, 1, out_filter / 4, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
            x += orig_x

        tf.logging.info('image after unit %s', x.get_shape())
        return x

    def _residual(self, x, in_filter, out_filter, stride,
                  activation_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activation_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0], [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
            x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def decay(self, weight_decay):
        """L2 weight decay loss."""
        costs = []
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
        return tf.multiply(weight_decay, tf.add_n(costs))

    def fully_connected(self, x, out_dim):
        """fully connected layer"""
        x = tf.reshape(x, [self.hps.batch_size, -1])
        w = tf.get_variable(
            'DW', [x.get_shape()[1], out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def inference(self, image_batch):
        x = self._conv('init_conv', image_batch, 3, 3, 16, self._stride_arr(1))

        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        if self.hps.use_bottleneck:
            res_func = self._bottleneck_residual
            filters = [16, 64, 128, 256]
        else:
            res_func = self._residual
            filters = [16, 16, 32, 64]

        with tf.variable_scope('unit_1_0'):
            x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]), activate_before_residual[0])
        for i in range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_1_%d' % i):
                x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

        with tf.variable_scope('unit_2_0'):
            x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]), activate_before_residual[1])
        for i in range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_2_%d' % i):
                x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

        with tf.variable_scope('unit_3_0'):
            x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]), activate_before_residual[2])
        for i in range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_3_%d' % i):
                x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

        with tf.variable_scope('unit_last'):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._global_avg_pool(x)

        return x

    def get_train_op(self, total_loss, global_step, lrn_rate):
        lrn_rate = tf.constant(lrn_rate, tf.float32)
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        grads = tf.gradients(total_loss, var_list)

        if self.hps.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(lrn_rate)
            apply_op = optimizer.apply_gradients(
                zip(grads, var_list),
                global_step=global_step,
                name='train_step')

        elif self.hps.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(lrn_rate, 0.9)
            apply_op = optimizer.apply_gradients(
                zip(grads, var_list),
                global_step=global_step,
                name='train_step')

        elif self.hps.optimizer == 'YF':
            optimizer = YFOptimizer(learning_rate=1.0, momentum=0.0)
            apply_op = optimizer.apply_gradients(
                zip(grads, var_list))

        train_ops = [apply_op] + self._extract_train_ops
        return tf.group(*train_ops)
