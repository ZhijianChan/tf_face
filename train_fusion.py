from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import train_utils
import test_utils

import random
import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from models import inception_resnet_v1


def snapshot(sess, saver, model_dir, model_name, step):
    print('snapshot...')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)

    # save trainable variables of sess into file
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    duration = time.time() - start_time
    print('snapshot finisned in %.2f seconds' % duration)

    metagraph_path = os.path.join(model_dir, 'model-%s.meta' % model_name)
    if not os.path.exists(metagraph_path):
        print('saving metagraph...')
        start_time = time.time()
        saver.export_meta_graph(metagraph_path)
        duration = time.time() - start_time
        print('metagraph saved in %.2f seconds' % duration)


def evaluate(sess, enque_op,
             face_pl, nose_pl, lefteye_pl, rightmouth_pl, labels_pl,
             phase_train_pl, batch_size_pl,
             embeddings, labels,
             image_list, actual_issame,
             batch_size, num_folds,
             log_dir, step, summary_writer):
    print("evaluating on lfw...")
    start_time = time.time()

    face_array = np.array(image_list)
    nose_array = np.array(
        [ss.replace('face', 'nose') for ss in face_array])
    lefteye_array = np.array(
        [ss.replace('face', 'lefteye') for ss in face_array])
    rightmouth_array = np.array(
        [ss.replace('face', 'rightmouth') for ss in face_array])

    labels_array = np.expand_dims(np.arange(0, len(image_list)), 1)
    face_array = np.expand_dims(face_array, 1)
    nose_array = np.expand_dims(nose_array, 1)
    lefteye_array = np.expand_dims(lefteye_array, 1)
    rightmouth_array = np.expand_dims(rightmouth_array, 1)

    sess.run(enque_op, {
        face_pl: face_array,
        nose_pl: nose_array,
        lefteye_pl: lefteye_array,
        rightmouth_pl: rightmouth_array,
        labels_pl: labels_array
    })

    embeddings_dim = embeddings.get_shape()[1]
    num_images = len(actual_issame) * 2
    assert num_images % batch_size == 0, 'Num of sample must be even.'

    num_batches = num_images // batch_size
    emb_array = np.zeros((num_images, embeddings_dim))
    lab_array = np.zeros((num_images,))
    for i in range(num_batches):
        feed_dict = {
            phase_train_pl: False,
            batch_size_pl: batch_size
        }
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab] = emb

    assert np.array_equal(lab_array, np.arange(num_images)) == True, 'Wrong labels used for evaluation'
    _, _, acc, val, val_std, far = test_utils.evaluate(emb_array, actual_issame, num_folds=num_folds)

    print('acc: %1.3f+-%1.3f' % (np.mean(acc), np.std(acc)))
    print('vr : %2.5f+=%2.5f @ FAR=%2.5F' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Summary
    summary = tf.Summary()
    summary.value.add(tag='lfw/acc', simple_value=np.mean(acc))
    summary.value.add(tag='lfw/vr', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir, 'lfw_result.txt'), 'at') as fp:
        fp.write('%d\t%.5f\t%.5f\n' % (step, np.mean(acc), val))


def run_epoch(args, sess, epoch,
              image_list, label_list,
              deque_op, enque_op,
              face_pl, nose_pl, lefteye_pl, rightmouth_pl, labels_pl,
              lr_fusion_pl, phase_train_pl, batch_size_pl,
              global_step, total_loss, reg_loss,
              train_op, summary_op, summary_writer):
    batch_num = 0
    '''
    if args.lr_base > 0.0:
        lr_base = args.lr_base
    else:
        lr_base = train_utils.get_learning_rate_from_file(args.lr_base_schedule_file, epoch)
    '''

    if args.lr_fusion > 0.0:
        lr_fusion = args.lr_fusion
    else:
        lr_fusion = train_utils.get_learning_rate_from_file(args.lr_fusion_schedule_file, epoch)

    index_epoch = sess.run(deque_op)
    label_epoch = np.array(label_list)[index_epoch]
    face_epoch = np.array(image_list)[index_epoch]  # face
    nose_epoch = np.array(  # nose
        [ss.replace('face', 'nose') for ss in face_epoch])
    lefteye_epoch = np.array(  # lefteye
        [ss.replace('face', 'lefteye') for ss in face_epoch])
    rightmouth_epoch = np.array(  # rightmouth
        [ss.replace('face', 'rightmouth') for ss in face_epoch])

    labels_array = np.expand_dims(label_epoch, 1)
    face_array = np.expand_dims(face_epoch, 1)
    nose_array = np.expand_dims(nose_epoch, 1)
    lefteye_array = np.expand_dims(lefteye_epoch, 1)
    rightmouth_array = np.expand_dims(rightmouth_epoch, 1)

    sess.run(enque_op, {
        face_pl: face_array,
        nose_pl: nose_array,
        lefteye_pl: lefteye_array,
        rightmouth_pl: rightmouth_array,
        labels_pl: labels_array
    })

    train_time = 0
    while batch_num < args.epoch_size:
        start_time = time.time()
        feed_dict = {
            # lr_base_pl : lr_base,
            lr_fusion_pl: lr_fusion,
            phase_train_pl: True,
            batch_size_pl: args.batch_size
        }
        if batch_num == 0 or (batch_num + 1) % 100 == 0:
            err, _, step, reg, summary_str = sess.run([total_loss,
                                                       train_op, global_step, reg_loss, summary_op],
                                                      feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
        else:
            err, _, step, reg = sess.run([total_loss,
                                          train_op, global_step, reg_loss],
                                         feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime: %.3f\tTotal Loss: %2.3f\tRegLoss: %2.3f' %
              (epoch, batch_num + 1, args.epoch_size, duration, err, np.sum(reg)))
        batch_num += 1
        train_time += duration

    summary = tf.Summary()
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, step)
    return step


def main(args):
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(args.logs_base_dir, subdir, 'logs')
    model_dir = os.path.join(args.logs_base_dir, subdir, 'models')
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    print('log   dir: %s' % log_dir)
    print('model dir: %s' % model_dir)

    # build the graph
    # ---- load pretrained model ---- #
    pretrained = {}
    # Face model
    pretrained['Face'] = np.load(args.face_model)[()]
    # Nose model
    pretrained['Nose'] = np.load(args.nose_model)[()]
    # Lefteye model
    pretrained['Lefteye'] = np.load(args.lefteye_model)[()]
    # Rightmouth model
    pretrained['Rightmouth'] = np.load(args.rightmouth_model)[()]
    # ---- data preparation ---- #
    image_list, label_list, num_classes = train_utils.get_datasets(args.data_dir, args.imglist)
    range_size = len(image_list)

    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        pairs = test_utils.read_pairs(args.lfw_pairs)
        lfw_paths, actual_issame = test_utils.get_paths(args.lfw_dir, pairs, args.lfw_file_ext)

    with tf.Graph().as_default():
        # random indices producer
        indices_que = tf.train.range_input_producer(range_size)
        dequeue_op = indices_que.dequeue_many(args.batch_size * args.epoch_size, 'index_dequeue')

        tf.set_random_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        global_step = tf.Variable(0, trainable=False)
        # lr_base_pl    = tf.placeholder(tf.float32, name='base_learning_rate')
        lr_fusion_pl = tf.placeholder(tf.float32, name='fusion_learning_rate')
        batch_size_pl = tf.placeholder(tf.int32, name='batch_size')
        phase_train_pl = tf.placeholder(tf.bool, name='phase_train')
        face_pl = tf.placeholder(tf.string, name='image_paths1')  # face images
        nose_pl = tf.placeholder(tf.string, name='image_paths2')  # nose images
        lefteye_pl = tf.placeholder(tf.string, name='image_paths3')  # left eye images
        rightmouth_pl = tf.placeholder(tf.string, name='image_paths4')  # right mouth images
        labels_pl = tf.placeholder(tf.int64, name='labels')

        # define a filename queue
        input_queue = tf.FIFOQueue(
            # [notice: capacity > bach_size*epoch_size]
            capacity=100000,
            dtypes=[tf.string, tf.string, tf.string, tf.string, tf.int64],
            shapes=[(1,), (1,), (1,), (1,), (1,)],
            shared_name=None, name='input_que')
        enque_op = input_queue.enqueue_many(
            [face_pl, nose_pl, lefteye_pl, rightmouth_pl, labels_pl],
            name='enque_op')
        # define 4 readers
        num_threads = 4
        threads_input_list = []
        for _ in range(num_threads):
            imgpath1, imgpath2, imgpath3, imgpath4, label = input_queue.dequeue()
            images1 = []
            images2 = []
            images3 = []
            images4 = []
            # face
            for img_path in tf.unstack(imgpath1):
                img_contents = tf.read_file(img_path)
                img = tf.image.decode_jpeg(img_contents)
                # [notice: random crop only used in face image]
                if args.random_crop:
                    img = tf.random_crop(img, [160, 160, 3])
                else:
                    img = tf.image.resize_image_with_crop_or_pad(img, 160, 160)
                # [notice: flip only used in face image or nose patch]
                if args.random_flip:
                    img = tf.image.random_flip_left_right(img)
                img.set_shape((160, 160, 3))
                images1.append(tf.image.per_image_standardization(img))
            # Nose
            for img_path in tf.unstack(imgpath2):
                img_contents = tf.read_file(img_path)
                img = tf.image.decode_jpeg(img_contents)
                # [notice: flip only used in face image or nose patch]
                if args.random_flip:
                    img = tf.image.random_flip_left_right(img)
                img.set_shape((160, 160, 3))
                images2.append(tf.image.per_image_standardization(img))
            # Lefteye
            for img_path in tf.unstack(imgpath3):
                img_contents = tf.read_file(img_path)
                img = tf.image.decode_jpeg(img_contents)
                img.set_shape((160, 160, 3))
                images3.append(tf.image.per_image_standardization(img))
            # Rightmouth
            for img_path in tf.unstack(imgpath4):
                img_contents = tf.read_file(img_path)
                img = tf.image.decode_jpeg(img_contents)
                img.set_shape((160, 160, 3))
                images4.append(tf.image.per_image_standardization(img))
            threads_input_list.append([images1, images2, images3, images4, label])

        # define 4 buffer queue
        face_batch, nose_batch, lefteye_batch, rightmouth_batch, label_batch = tf.train.batch_join(
            threads_input_list,
            # [notice: here is 'batch_size_pl', not 'batch_size'!!]
            batch_size=batch_size_pl,
            shapes=[
                # [notice: shape of each element should be assigned, otherwise it raises
                # "tensorflow queue shapes must have the same length as dtype" exception]
                (args.image_size, args.image_size, 3),
                (args.image_size, args.image_size, 3),
                (args.image_size, args.image_size, 3),
                (args.image_size, args.image_size, 3),
                ()],
            enqueue_many=True,
            # [notice: how long the prefetching is allowed to fill the queue]
            capacity=4 * num_threads * args.batch_size,
            allow_smaller_final_batch=True)
        print('Total classes: %d' % num_classes)
        print('Total images:  %d' % range_size)
        tf.summary.image('face_images', face_batch, 10)
        tf.summary.image('nose_images', nose_batch, 10)
        tf.summary.image('lefteye_images', lefteye_batch, 10)
        tf.summary.image('rightmouth_images', rightmouth_batch, 10)

        # ---- build graph ---- #
        with tf.variable_scope('BaseModel'):
            with tf.device('/gpu:%d' % args.gpu_id1):
                # embeddings for face model
                features1, _ = inception_resnet_v1.inference(
                    face_batch,
                    args.keep_prob,
                    phase_train=phase_train_pl,
                    weight_decay=args.weight_decay,
                    scope='Face')
            with tf.device('/gpu:%d' % args.gpu_id2):
                # embeddings for nose model
                features2, _ = inception_resnet_v1.inference(
                    nose_batch,
                    args.keep_prob,
                    phase_train=phase_train_pl,
                    weight_decay=args.weight_decay,
                    scope='Nose')
            with tf.device('/gpu:%d' % args.gpu_id3):
                # embeddings for left eye model
                features3, _ = inception_resnet_v1.inference(
                    lefteye_batch,
                    args.keep_prob,
                    phase_train=phase_train_pl,
                    weight_decay=args.weight_decay,
                    scope='Lefteye')
            with tf.device('/gpu:%d' % args.gpu_id4):
                # embeddings for right mouth model
                features4, _ = inception_resnet_v1.inference(
                    rightmouth_batch,
                    args.keep_prob,
                    phase_train=phase_train_pl,
                    weight_decay=args.weight_decay,
                    scope='Rightmouth')
        with tf.device('/gpu:%d' % args.gpu_id5):
            with tf.variable_scope("Fusion"):
                # ---- concatenate ---- #
                concated_features = tf.concat([features1, features2, features3, features4], 1)
                # prelogits
                prelogits = slim.fully_connected(
                    concated_features,
                    args.fusion_dim,
                    activation_fn=None,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    weights_regularizer=slim.l2_regularizer(args.weight_decay),
                    scope='prelogits',
                    reuse=False)
                # logits
                logits = slim.fully_connected(
                    prelogits,
                    num_classes,
                    activation_fn=None,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    weights_regularizer=slim.l2_regularizer(args.weight_decay),
                    scope='logits',
                    reuse=False)
                # normalized feaures
                # [notice: used in test stage]
                embeddings = tf.nn.l2_normalize(
                    prelogits, 1, 1e-10, name='embeddings')

            # ---- define loss & train op ---- #
            cross_entropy = - tf.reduce_sum(
                tf.one_hot(indices=tf.cast(label_batch, tf.int32), depth=num_classes) * tf.log(
                    tf.nn.softmax(logits) + 1e-10),
                reduction_indices=[1])
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
            tf.add_to_collection('losses', cross_entropy_mean)
            # weight decay
            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            # total loss: cross_entropy + weight_decay
            total_loss = tf.add_n([cross_entropy_mean] + reg_loss, name='total_loss')
            '''
            lr_base = tf.train.exponential_decay(lr_base_pl,
                global_step,
                args.lr_decay_epochs * args.epoch_size,
                args.lr_decay_factor,
                staircase = True)
            '''
            lr_fusion = tf.train.exponential_decay(lr_fusion_pl,
                                                   global_step,
                                                   args.lr_decay_epochs * args.epoch_size,
                                                   args.lr_decay_factor,
                                                   staircase=True)
            # tf.summary.scalar('base_learning_rate', lr_base)
            tf.summary.scalar('fusion_learning_rate', lr_fusion)
            var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='BaseModel')
            var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Fusion')
            '''
            train_op = train_utils.get_fusion_train_op(
                total_loss, global_step, args.optimizer,
                lr_base, var_list1, lr_fusion, var_list2,
                args.moving_average_decay)
            '''
            train_op = train_utils.get_train_op(total_loss,
                                                global_step,
                                                args.optimizer,
                                                lr_fusion,
                                                args.moving_average_decay,
                                                var_list2)

        # ---- training ---- #
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options,
            log_device_placement=False,
            # [notice: 'allow_soft_placement' will switch to cpu automatically
            #  when some operations are not supported by GPU]
            allow_soft_placement=True))
        saver = tf.train.Saver(var_list1 + var_list2)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            # ---- restore pre-trained parameters ---- #
            to_assign = []
            print("restore pretrained parameters...")
            print("total:", len(var_list1))
            for v in var_list1:
                v_name = v.name  # 'BaseModel/Face/xxx'
                v_name = v_name[v_name.find('/') + 1:]  # 'Face/xxx'
                v_name_1 = v_name[:v_name.find('/')]  # 'Face'
                v_name_2 = v_name[v_name.find('/'):]  # '/xxx'
                print("precess: %s" % v_name, end=" ")
                if v_name_1 in pretrained:
                    to_assign.append(v.assign(pretrained[v_name_1][v_name_2][0]))
                    print("[ok]")
                else:
                    print("[no found]")
                    v.assign(pretrained[v_name_1][v_name_2][0])
                    print("done")
            sess.run(to_assign)

            print("start training ...")
            epoch = 0
            while epoch < args.max_num_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size

                # run one epoch
                run_epoch(args, sess, epoch,
                          image_list, label_list,
                          dequeue_op, enque_op,
                          face_pl, nose_pl, lefteye_pl, rightmouth_pl, labels_pl,
                          lr_fusion_pl, phase_train_pl, batch_size_pl,
                          global_step, total_loss, reg_loss,
                          train_op, summary_op, summary_writer)

                # snapshot for currently learnt weights
                snapshot(sess, saver, model_dir, subdir, step)

                # evaluate on LFW
                if args.lfw_dir:
                    evaluate(sess, enque_op,
                             face_pl, nose_pl, lefteye_pl, rightmouth_pl, labels_pl,
                             phase_train_pl, batch_size_pl,
                             embeddings, label_batch,
                             lfw_paths, actual_issame,
                             args.lfw_batch_size, args.lfw_num_folds,
                             log_dir, step, summary_writer)
    sess.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # ---- file related ---- #
    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='logs')

    # ---- pretrained models ---- #
    parser.add_argument("--face_model", type=str,
                        help="path to to pretrained .npy face model",
                        default="pretrained/pretrained_face.npy")
    parser.add_argument("--nose_model", type=str,
                        help="path to to pretrained .npy nose model",
                        default="pretrained/pretrained_nose.npy")
    parser.add_argument("--lefteye_model", type=str,
                        help="path to to pretrained .npy lefteye model",
                        default="pretrained/pretrained_lefteye.npy")
    parser.add_argument("--rightmouth_model", type=str,
                        help="path to to pretrained .npy rightmouth model",
                        default="pretrained/pretrained_rightmouth.npy")
    parser.add_argument('--imglist', type=str,
                        help='Training images list.',
                        default='/export_data/czj/data/casia/files/train_set.txt')

    # ---- training related ---- #
    parser.add_argument('--seed', type=int,
                        help='Random seed.',
                        default=666)
    parser.add_argument('--max_num_epochs', type=int,
                        help='Number of epochs to run.',
                        default=80)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.',
                        default=10)
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.',
                        default=1000)
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned faces.',
                        default='/exports_data/czj/data/lfw/lfw_aligned/')
    parser.add_argument('--gpu_id1', type=int,
                        help='available gpu id 1',
                        default=1)
    parser.add_argument('--gpu_id2', type=int,
                        help='available gpu id 2',
                        default=2)
    parser.add_argument('--gpu_id3', type=int,
                        help='available gpu id 3',
                        default=3)
    parser.add_argument('--gpu_id4', type=int,
                        help='available gpu id 4',
                        default=4)
    parser.add_argument('--gpu_id5', type=int,
                        help='available gpu id 5',
                        default=5)
    parser.add_argument('--random_crop',
                        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
                             'If the size of the images in the data directory is equal to image_size no cropping is performed',
                        action='store_true')
    parser.add_argument('--random_flip',
                        help='Performs random horizontal flipping of training images.',
                        action='store_true')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.',
                        default=160)

    # ---- regularity ---- #
    parser.add_argument('--keep_prob', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).',
                        default=1.0)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.',
                        default=0.0)

    # ---- learning strategy ---- #
    parser.add_argument('--lr_base', type=float,
                        help='Learning rate for base model',
                        default=0.0001)
    parser.add_argument('--lr_fusion', type=float,
                        help='Learning rate for fusion model',
                        default=0.01)
    parser.add_argument('--lr_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.',
                        default=100)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.',
                        default=0.9999)
    parser.add_argument('--lr_decay_factor', type=float,
                        help='Learning rate decay factor.',
                        default=1.0)
    parser.add_argument('--optimizer', type=str,
                        choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use',
                        default='ADAGRAD')
    parser.add_argument('--lr_base_schedule_file', type=str,
                        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.',
                        default='lr_decay_base.txt')
    parser.add_argument('--lr_fusion_schedule_file', type=str,
                        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.',
                        default='lr_decay_fusion.txt')

    # ---- log related ---- #
    parser.add_argument('--log_histograms',
                        help='Enables logging of weight/bias histograms in tensorboard.',
                        action='store_true')

    # ---- LFW related ---- #
    parser.add_argument('--lfw_pairs', type=str,
                        help='LFW pairs file.',
                        default='')
    parser.add_argument('--lfw_file_ext', type=str,
                        help='The file extension for the LFW dataset.',
                        default='_face_.jpg')
    parser.add_argument('--lfw_dir', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        default=10)
    parser.add_argument('--lfw_batch_size', type=int,
                        help='Number of images to process in a batch in the LFW test set.',
                        default=10)
    parser.add_argument('--lfw_num_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.',
                        default=10)

    # ---- hyper parameters ---- #
    parser.add_argument('--fusion_dim', type=int,
                        help='Dimension of fusion features',
                        default=128)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
