from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

# ============================================= #
# train utils
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
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg') #momentumn?
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses+[total_loss])
    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses+[total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name+'(raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op

def get_train_op(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, log_histograms = True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # compute gradients
    with tf.control_dependencies([loss_averages_op]):
        if optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer == 'ADAM':
            opt=tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer == 'RMSPROP':
            opt=tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer == 'MOM':
            opt=tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')
        grads = opt.compute_gradients(total_loss, update_gradient_vars)

    # update parameters
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    if log_histograms:
        # trainable variables
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        # gradients for variables
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name+'/grad', grad)

    # Track the moving averages of all trainable variables
    variables_average = tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
    variables_average_op = variables_average.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_average_op]):
        train_op = tf.no_op(name='train')
    return train_op

# ============================================= #
# loss utils
def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
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

# ============================================= #
# dataset utils
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
