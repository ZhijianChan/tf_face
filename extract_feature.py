from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import time
import json
import matio
import importlib
import numpy as np
import tensorflow as tf
from scipy import misc
from models import inception_resnet_v1


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
    if not os.path.exists(args.templates_files):
        print('invalid template files')
        return

    if not os.path.exists(args.pretrained_model):
        print('invalid pretrained model path')
        return

    with open(args.templates_files) as fp:
        imglist = json.load(fp)['path']
        #tmplist = []
        #for i in range(len(imglist)):
        #    if not os.path.exists(os.path.join(args.data_root, imglist[i]) + args.file_ext):
        #        tmplist.append(imglist[i])

    #imglist = tmplist
    total_images = len(imglist)
    weights = np.load(args.pretrained_model)
    model_module = importlib.import_module(args.model_def)

    print("total images:", total_images)
    print("file_ext:", args.file_ext)

    # ---- build graph ---- #
    with tf.device('/gpu:%d' % args.gpu_id):
        input = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='image_batch')
        prelogits, _ = model_module.inference(input, 1, phase_train=False)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10)
        #embeddings = prelogits

    # ---- extract ---- #
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(
        config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True))
    with sess.as_default():
        beg_time = time.time()
        to_assign = [v.assign(weights[()][v.name][0]) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
        sess.run(to_assign)
        print('restore parameters: %.2fsec' % (time.time()-beg_time))

        images_batch_size = 10000
        images_batch_num = total_images // images_batch_size
        if total_images % images_batch_size != 0:
            images_batch_num = images_batch_num + 1
        batch_size = 32
        for i in range(images_batch_num):
            beg_time = time.time()
            images_beg = i * images_batch_size
            images_end = min(images_beg + images_batch_size, total_images)
            to_process = images_end - images_beg
            images = load_data(args.data_root, imglist[images_beg: images_end])
            print('load images: %.2fsec' % (time.time()-beg_time))
            beg_time = time.time()
            beg = 0
            end = 0
            features = np.zeros((to_process, 128),dtype =np.float32)
            while end < to_process:
                end = min(beg + batch_size, to_process)
                print('extract: [%d][%d][%d]' % (i, end, to_process))
                features[beg:end] = sess.run(embeddings, {input:images[beg:end]})
                '''
                for j in range(end - beg):
                    feats[j,:] = feats[j,:] / np.linalg.norm(feats[j,:])
                features[beg:end] = feats
                '''
                beg = end
            print('done: %.2fsec' % (time.time()-beg_time))
            print('saving ...')
            beg_time = time.time()
            for i in range(to_process):
                savepath = os.path.join(args.data_root, imglist[images_beg + i]) + args.file_ext
                with open(savepath, 'wb') as fp:
                    matio.write_mat(fp, features[i,:])
            print('done: %.2fsec' % (time.time()-beg_time))
    sess.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', type=str)
    parser.add_argument('templates_files', type=str)
    parser.add_argument('pretrained_model', type=str)
    parser.add_argument('model_def', type=str)
    parser.add_argument('file_ext', type=str)
    parser.add_argument('gpu_id', type=int)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
