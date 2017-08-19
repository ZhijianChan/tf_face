from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import time
import json
#import matio
import importlib
import numpy as np
import tensorflow as tf
from scipy import misc
from models import resnet_v2
import tensorflow.contrib.slim as slim


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


def load_data(data_root, imglist):
    num_images = len(imglist)
    images = np.zeros((num_images, 160, 160, 3))
    for i in range(num_images):
        img = misc.imread(os.path.join(data_root, imglist[i]))
        if img.ndim == 2:
            img = to_rgb(img)
        img = crop(img, 160)
        img = prewhiten(img)
        images[i, :, :, :] = img
    return images


def main(args):
    if not os.path.exists(args.imglist):
        print('invalid imglist files')
        return

    if not os.path.exists(args.pretrained_model):
        print('invalid pretrained model path')
        return

    with open(args.imglist) as fp:
        imglist = [line.strip() for line in fp.readlines()]

    total_images = len(imglist)
    weights = np.load(args.pretrained_model)
    #model_module = importlib.import_module(args.model_def)

    print("total images:", total_images)
    #print("file_ext:", args.file_ext)

    # ---- build graph ---- #
    with tf.device('/gpu:%d' % args.gpu_id):
        input = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='image_batch')
        with slim.arg_scope(resnet_v2.resnet_arg_scope(is_training=False)):
            prelogits, _ = resnet_v2.resnet_v2_50(input, phase_train=False)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10)

    # ---- extract ---- #
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(
        config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True))
    with sess.as_default():
        beg_time = time.time()
        to_assign = [v.assign(weights[()][v.name][0]) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
        sess.run(to_assign)
        print('restore parameters: %.2fsec' % (time.time()-beg_time))

        batch_size = 32
        beg_time = time.time()
        images = load_data(args.data_root, imglist)
        print('load images: %.2fsec' % (time.time()-beg_time))
        beg_time = time.time()
        beg = 0
        end = 0
        features = np.zeros((total_images, 128))
        while end < total_images:
            end = min(beg + batch_size, total_images)
            print('extract: [%d][%d]' % (end, total_images))
            features[beg:end] = sess.run(embeddings, {input:images[beg:end]})
            beg = end
        print('done: %.2fsec' % (time.time()-beg_time))
        print('saving ...')
        np.savetxt(args.savepath, features, delimiter=',')
        print('done: %.2fsec' % (time.time()-beg_time))
    sess.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', type=str)
    parser.add_argument('imglist', type=str)
    parser.add_argument('pretrained_model', type=str)
    #parser.add_argument('model_def', type=str)
    #parser.add_argument('file_ext', type=str)
    parser.add_argument('gpu_id', type=int)
    parser.add_argument('savepath', type=str)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
