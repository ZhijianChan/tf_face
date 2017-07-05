from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import sys
import argparse
import numpy as np


def main(args):
    if args.meta_file == None or not os.path.exists(args.meta_file):
        print("Invalid tensorflow meta-graph file:", args.meta_file)
        return

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
        allow_soft_placement=True))
    with sess.as_default():
        # ---- load pretrained parameters ---- #
        saver = tf.train.import_meta_graph(args.meta_file)
        saver.restore(tf.get_default_session(), args.ckpt_file)
        pretrained = {}
        var_ = tf.trainable_variables()
        print("total:", len(var_))
        for v in var_:
            print("process:", v.name)
            # [notice: the name of parameter is like 'Resnet/conv2d/bias:0',
            #  here we should remove the prefix name, and get '/conv2d/bias:0']
            v_name = v.name
            v_name = v_name[v_name.find('/'):]
            pretrained[v_name] = sess.run([v])
    np.save(args.save_path, pretrained)
    print("done:", len(pretrained.keys()))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # ---- pretrained model ---- #
    parser.add_argument("--meta_file", type=str,
                        help="path to tensorflow meta-graph file")
    parser.add_argument("--ckpt_file", type=str,
                        help="path to tensorflow checkpoint file")
    parser.add_argument("--save_path", type=str,
                        help="path to save .npy model file", default="pretrained/pretrained.npy")
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
