from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as fp:
        for line in fp:
            par = line.strip().split(':')
            e = int(par[0])
            lr = float(par[1])
            if e <= epoch:
                learning_rate = lr
            else:
                return learning_rate

def _add_loss_summaries(total_loss):
    """Add summaries for losses.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
  
    Args:
      total_loss: total losses including regularity
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    ema = tf.train.ExponentialMovingAverage(0.9, name='avg')
    # [notice: 'losses' collection here only contains cross entropy]
    losses = tf.get_collection('losses')
    # [notice: 'apply' maintains moving averages of variables]
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name+'(raw)', l)
        # [notice: 'average(var)' returns the variable holding the average of 'var']
        tf.summary.scalar(l.op.name, loss_averages_op.average(l))
    return loss_averages_op

def get_fusion_train_op(total_loss, global_step, optimizer,
    lr_base, var_list1, lr_fusion, var_list2,
    moving_average_decay, log_histograms = False):
    # maintain moving average of total_losses
    # [notice: this may not be very necessary]
    #loss_averages_op = _add_loss_summaries(total_loss)

    # compute gradients
    #with tf.control_dependencies([loss_averages_op]):
    if optimizer == 'ADAGRAD':
        opt1 = tf.train.AdagradOptimizer(lr_base)
        opt2 = tf.train.AdagradOptimizer(lr_fusion)
    elif optimizer == 'ADADELTA':
        opt1 = tf.train.AdadeltaOptimizer(lr_base,   rho=0.9, epsilon=1e-6)
        opt2 = tf.train.AdadeltaOptimizer(lr_fusion, rho=0.9, epsilon=1e-6)
    elif optimizer == 'ADAM':
        opt1 = tf.train.AdamOptimizer(lr_base,   beta1=0.9, beta2=0.999, epsilon=0.1)
        opt2 = tf.train.AdamOptimizer(lr_fusion, beta1=0.9, beta2=0.999, epsilon=0.1)
    elif optimizer == 'RMSPROP':
        opt1 = tf.train.RMSPropOptimizer(lr_base,   decay=0.9, momentum=0.9, epsilon=1.0)
        opt2 = tf.train.RMSPropOptimizer(lr_fusion, decay=0.9, momentum=0.9, epsilon=1.0)
    elif optimizer == 'MOM':
        opt1 = tf.train.MomentumOptimizer(lr_base,   0.9, use_nesterov=True)
        opt2 = tf.train.MomentumOptimizer(lr_fusion, 0.9, use_nesterov=True)
    else:
        raise ValueError('Invalid optimization algorithm')

    grads = tf.gradients(total_loss, var_list1 + var_list2)
    grads1 = grads[:len(var_list1)]
    grads2 = grads[len(var_list1):]
    train_op1 = opt1.apply_gradients(zip(grads1, var_list1), global_step=global_step)
    train_op2 = opt2.apply_gradients(zip(grads2, var_list2), global_step=global_step)
    train_op_ = tf.group(train_op1, train_op2)
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name+'/grad', grad)
    # maintain the moving averages of all trainable variables
    ema = tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
    var_avg_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_op_, var_avg_op]):
        train_op = tf.no_op(name='train')
    return train_op

def get_train_op(total_loss, global_step,
    optimizer, lr, moving_average_decay,
    var_list, log_histograms = False):
    # maintain moving average of total_losses
    # [notice: this may not be very necessary]
    #loss_averages_op = _add_loss_summaries(total_loss)

    # compute gradients
    #with tf.control_dependencies([loss_averages_op]):
    if optimizer == 'ADAGRAD':
        opt = tf.train.AdagradOptimizer(lr)
    elif optimizer == 'ADADELTA':
        opt = tf.train.AdadeltaOptimizer(lr, rho=0.9, epsilon=1e-6)
    elif optimizer == 'ADAM':
        opt = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=0.1)
    elif optimizer == 'RMSPROP':
        opt = tf.train.RMSPropOptimizer(lr, decay=0.9, momentum=0.9, epsilon=1.0)
    elif optimizer == 'MOM':
        opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
    else:
        raise ValueError('Invalid optimization algorithm')
    grads = opt.compute_gradients(total_loss, var_list)
    train_op_ = opt.apply_gradients(grads, global_step=global_step)
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name+'/grad', grad)
    # maintain the moving averages of all trainable variables
    ema = tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
    var_avg_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_op_, var_avg_op]):
        train_op = tf.no_op(name='train')
    return train_op

def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper
    
    args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    pos_dist = tf.reduce_sum( tf.square( tf.subtract( anchor, positive ) ), 1 )
    neg_dist = tf.reduce_sum( tf.square( tf.subtract( anchor, negative ) ), 1 )
    tot_dist = tf.sum( tf.subtract( pos_dist, neg_dist ), alpha )
    loss = tf.reduce_mean( tf.maximum( tot_dist, 0.0 ), 0)
    return loss

def center_loss(features, label, alpha, num_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    dim_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, dim_features], dtype = tf.float32, initializer = tf.constant_initializer(0), trainable = False)
    label = tf.reshape(label, [-1])
    center_feats = tf.gather(centers, label)
    diff = (1 - alpha) * tf.subtract( center_feats, features )
    centers = tf.scatter_sub( centers, label, diff )
    loss = tf.nn.l2_loss(features - center_feats)
    return loss, centers

def get_datasets(data_dir, imglist_path):
    """Load imglist, which contains 'img_path label' in each line"""
    mdict = {}
    image_list = []
    label_list = []
    with open(imglist_path) as fp:
        for line in fp:
            items = line.strip().split()
            imgpath = os.path.join(data_dir, items[0])
            label   = int(items[1])
            mdict[label] = 1
            image_list.append(imgpath)
            label_list.append(label)
    return image_list, label_list, len(mdict.keys())
