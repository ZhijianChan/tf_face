from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import test_utils
import math
import sys
import argparse

from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate

def main(args):
    pairs = test_utils.read_pairs(args.lfw_pairs)
    model_list = test_utils.get_model_list(args.model_list)
    for t,model in enumerate(model_list):
        # get lfw pair filename
        paths, actual_issame = test_utils.get_paths(args.lfw_dir, pairs, model[1])
        with tf.device('/gpu:%d' % (t + 1)):
            gpu_options = tf.GPUOptions(allow_growth=True)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                log_device_placement=False, allow_soft_placement=True))
            with sess.as_default():
                print("[%d] model: %s"  % (t, model[1]))
                # restore model
                test_utils.load_model(model[0])
                # load data tensor
                images_pl      = tf.get_default_graph().get_tensor_by_name('input:0')
                embeddings     = tf.get_default_graph().get_tensor_by_name('embeddings:0')
                phase_train_pl = tf.get_default_graph().get_tensor_by_name('phase_train:0')
                image_size     = args.image_size
                emb_size       = embeddings.get_shape()[1]
                # extract feature
                batch_size = args.lfw_batch_size
                num_images = len(paths)
                num_batches = num_images // batch_size
                emb_arr  = np.zeros((num_images, emb_size))
                for i in range(num_batches):
                    print('process %d/%d' % (i+1, num_batches), end='\r')
                    beg_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, num_images)
                    images  = test_utils.load_data(paths[beg_idx:end_idx], False, False, image_size)
                    emb = sess.run(embeddings, feed_dict={images_pl:images, phase_train_pl:False})
                    emb_arr[beg_idx:end_idx,:] = emb
        # get lfw pair filename
        print("\ndone.")
        # concate feaure
        if t == 0:
            emb_ensemble = emb_arr*math.sqrt(float(model[2]))
        else:
            emb_ensemble = np.concatenate((emb_ensemble, emb_arr*math.sqrt(float(model[2]))), axis=1)
        print("ensemble feature:", emb_ensemble.shape)
            
    '''
    norm = np.linalg.norm(emb_ensemble, axis=1)
    for i in range(emb_ensemble.shape[0]):
        emb_ensemble[i] = emb_ensemble[i] / norm[i]
    '''

    tpr, fpr, acc, vr, vr_std, far = test_utils.evaluate(emb_ensemble, actual_issame, num_folds=args.num_folds)
    # display
    auc = metrics.auc(fpr, tpr)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr,tpr)(x), 0., 1.)
    print('Acc:           %1.3f+-%1.3f' % (np.mean(acc), np.std(acc)))
    print('VR@FAR=%2.5f:  %2.5f+-%2.5f' % (far, vr, vr_std))
    print('AUC:           %1.3f' % auc)
    print('EER:           %1.3f' % eer)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('lfw_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.',
        default='../files/pairs.txt')
    parser.add_argument('model_list', type=str,
        help='A file containing model path and corrsponding file type')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Mini-batch size', default=20)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default = 160)
    parser.add_argument('--num_folds', type=int,
        help='Number of folders to do cross validation', default=10)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
