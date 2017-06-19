
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.model_selection import KFold
from tensorflow.python.platform import gfile
from scipy import interpolate
from scipy import misc

import re
import os
import numpy as np
import tensorflow as tf

# =========================== #
# tf utils
# =========================== #
def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in %s' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('More than one meta files found in %s' % model_dir)
    meta_file = meta_files[0]
    ckpt_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in ckpt_files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step >= max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def load_model(model_path):
    if (os.path.isfile(model_path)):
        # A protobuf file with a frozen graph
        print('Model filename: %s' % model_path)
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        # A directory containing a metagraph file and a checkpoint file
        print('Model directory: %s' % model_path)
        meta_file, ckpt_file = get_model_filenames(model_path)
        print('Metagraph  file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
        saver = tf.train.import_meta_graph(os.path.join(model_path, meta_file), clear_devices=True)
        saver.restore(tf.get_default_session(), os.path.join(model_path, ckpt_file))

# =========================== #
# statistic utils
# =========================== #
def calculate_acc(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum( np.logical_and( predict_issame, actual_issame ))
    fp = np.sum( np.logical_and( predict_issame, np.logical_not( actual_issame )))
    tn = np.sum( np.logical_and( np.logical_not( predict_issame ), np.logical_not( actual_issame )))
    fn = np.sum( np.logical_and( np.logical_not( predict_issame ), actual_issame ))
    tpr = 0 if (tp+fn == 0) else float(tp) / float(tp+fn)
    fpr = 0 if (tn+fp == 0) else float(fp) / float(tn+fp)
    acc = float(tp+tn) / dist.size
    return tpr, fpr, acc

def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum( np.logical_and( predict_issame, actual_issame ))
    fp = np.sum( np.logical_and( predict_issame, np.logical_not( actual_issame )))
    p  = np.sum( actual_issame )
    n  = actual_issame.size - p
    val= 0 if (p == 0) else float(tp) / float(p)
    far= 0 if (n == 0) else float(fp) / float(n)
    return val, far

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, num_folds = 10):
    """Calculate TPR and FPR under different threshold, accuracy under the best threshold"""
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    num_pairs = min(len(actual_issame), embeddings1.shape[0])
    num_threshold = len(thresholds)
    k_fold = KFold(n_splits=num_folds, shuffle=False)
    
    tprs = np.zeros((num_folds, num_threshold))
    fprs = np.zeros((num_folds, num_threshold))
    acc  = np.zeros((num_folds))

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(num_pairs)

    for fold_id, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold
        acc_train = np.zeros((num_threshold))
        for thres_id, thres in enumerate(thresholds):
            _, _, acc_train[thres_id] = calculate_acc(thres, dist[train_set], actual_issame[train_set])
        best_id = np.argmax(acc_train)
        # Calculate tprs and fprs on test set
        for thres_id, thres in enumerate(thresholds):
            tprs[fold_id, thres_id], fprs[fold_id, thres_id], _ = calculate_acc(thres, dist[test_set], actual_issame[test_set])
        # Use the best threshold to calculate accuracy
        _, _, acc[fold_id] = calculate_acc(thresholds[best_id], dist[test_set], actual_issame[test_set])

    tpr = np.mean(tprs, 0)  # true  positive rate under different threshold
    fpr = np.mean(fprs, 0)  # false positive rate under different threshold
    return tpr, fpr, acc

def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, num_folds = 10):
    """Calculate the VR, VR_std and FAR @ FAR_target"""
    print('evaluating vr@far=0.001:')
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    num_pairs = min(len(actual_issame), embeddings1.shape[0])
    num_threshold = len(thresholds)
    k_fold = KFold(n_splits=num_folds, shuffle=False)

    val = np.zeros(num_folds)
    far = np.zeros(num_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(num_pairs)
    for fold_id, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Use the best threshold to calculate accuracy
        far_train = np.zeros(num_threshold)
        for thres_id, thres in enumerate(thresholds):
            _, far_train[thres_id] = calculate_val_far(thres, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            thres = f(far_target)
        else:
            thres = 0.0  # What if far is 0???
        val[fold_id], far[fold_id] = calculate_val_far(thres, dist[test_set], actual_issame[test_set])
        print('[%d] thres:%.2f val:%.2f far:%.2f' % (fold_id, thres, val[fold_id], far[fold_id]))

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std  = np.std(val)
    return val_mean, val_std, far_mean

# =========================== #
# lfw test utils
# =========================== #
def evaluate(embeddings, actual_issame, num_folds = 10):
    """Evaluate TPR, FPR, ACC under different threshold && VAR, VAR_std, FAR @ FAR=0.001"""
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, acc = calculate_roc(thresholds,embeddings1,embeddings2,np.asarray(actual_issame),num_folds = num_folds)

    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far  = calculate_val(thresholds,embeddings1,embeddings2,np.asarray(actual_issame),1e-3,num_folds = num_folds)
    return tpr, fpr, acc, val, val_std, far

# =========================== #
# lfw file utils
# =========================== #
def read_pairs(pairs_filename):
    with open(pairs_filename, 'r') as f:
        f.readline()
        pairs = [line.strip().split() for line in f.readlines()]
    return np.array(pairs)

def get_paths(lfw_dir, pairs, file_ext):
    num_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+file_ext)
            path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+file_ext)
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+file_ext)
            path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+file_ext)
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            num_skipped_pairs += 1
    if num_skipped_pairs > 0:
        print('Skipped %d image pairs' % num_skipped_pairs)
    return path_list, issame_list

# =========================== #
# image utils
# =========================== #
def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w,h,3), dtype=np.uint8)
    ret[:,:,0] = ret[:,:,1] = ret[:,:,2] = img
    return ret

def prewhiten(x):
    mean = np.mean(x)  # scalar
    std  = np.std(x)   # scalar
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)  # substract the image from mean, then divide the std
    return y

def crop(image, random_crop, image_size):
    if image.shape[1] > image_size:
        sz1 = int(image.shape[1] // 2)
        sz2 = int(image_size // 2)
        if random_crop:
            diff = sz1 - sz2
            (h,v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
        else:  # crop from center
            (h,v) = (0,0)
        image = image[(sz1-sz2+v):(sz1+sz2+v), (sz1-sz2+h):(sz1+sz2+h),:]
    return image

def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

def load_data(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten = True):
    # -- prewhiten is necessary !! -- #
    num_samples = len(image_paths)
    images = np.zeros((num_samples, image_size, image_size, 3))
    for i in range(num_samples):
        img = misc.imread(image_paths[i])
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        # img = crop(img, do_random_crop, image_size)
        # img = flip(img, do_random_flip)
        images[i,:,:,:] = img
    return images

# =========================== #
# Model ensemble utils
# =========================== #
def get_model_list(model_list):
    with open(model_list) as fp:
        model_list = [line.strip().split() for line in fp]
    return model_list
