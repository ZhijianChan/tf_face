from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import time
import numpy as np
import tensorflow as tf
from scipy import misc
from models import inception_resnet_v1

import test_utils
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate

def crop(image, image_size):
    if image.shape[1] > image_size:
        sz1 = int(image.shape[1] // 2)
        sz2 = int(image_size // 2)
        image = image[(sz1 - sz2):(sz1 + sz2), (sz1 - sz2):(sz1 + sz2), :]
    return image


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def load_data(imglist):
    num_images = len(imglist)
    images = np.zeros((num_images, 160, 160, 3))
    for i in range(num_images):
        img = misc.imread(imglist[i])
        if img.ndim == 2:
            img = to_rgb(img)
        img = crop(img, 160)
        img = prewhiten(img)
        images[i, :, :, :] = img
    return images


def main(args):
    if not os.path.exists(args.pretrained_model):
        print('invalid pretrained model path')
        return
    weights = np.load(args.pretrained_model)

    pairs = test_utils.read_pairs('/exports_data/czj/data/lfw/files/pairs.txt')
    imglist, labels = test_utils.get_paths('/exports_data/czj/data/lfw/lfw_aligned/', pairs, '_face_.jpg')
    total_images = len(imglist)

    # ---- build graph ---- #
    input = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='image_batch')
    prelogits, _ = inception_resnet_v1.inference(input, 1, phase_train=False)
    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10)

    # ---- extract ---- #
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(
        config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True))
    with sess.as_default():
        beg_time = time.time()
        to_assign = [v.assign(weights[()][v.name][0]) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
        sess.run(to_assign)
        print('restore parameters: %.2fsec' % (time.time() - beg_time))

        beg_time = time.time()
        images = load_data(imglist)
        print('load images: %.2fsec' % (time.time() - beg_time))

        beg_time = time.time()
        batch_size = 32
        beg = 0
        end = 0
        features = np.zeros((total_images, 128))
        while end < total_images:
            end = min(beg + batch_size, total_images)
            features[beg:end] = sess.run(embeddings, {input:images[beg:end]})
            beg = end
        print('extract features: %.2fsec' % (time.time() - beg_time))

    tpr, fpr, acc, vr, vr_std, far = test_utils.evaluate(features, labels, num_folds=10)
    # display
    auc = metrics.auc(fpr, tpr)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Acc:           %1.3f+-%1.3f' % (np.mean(acc), np.std(acc)))
    print('VR@FAR=%2.5f:  %2.5f+-%2.5f' % (far, vr, vr_std))
    print('AUC:           %1.3f' % auc)
    print('EER:           %1.3f' % eer)
    sess.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('pretrained_model', type=str, help='path of pretrained model')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
