# -*- coding: UTF-8 -*-
import collections  # 原生的collections库
import tensorflow as tf

slim = tf.contrib.slim  # 使用方便的contrib.slim库来辅助创建ResNet


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    '''
    定义一个典型的Block，需要输入三个参数：
    scope：Block的名称
    unit_fn：ResNet V2中的残差学习单元
    args：Block的args。
    '''


########定义一个降采样的方法########
def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


########创建卷积层########
def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1,
                           padding='SAME', scope=scope)
    else:  # 如果不为1，则显式的pad zero，pad zero总数为kernel_size - 1
        # kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,  # 对输入变量进行补零操作
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           padding='VALID', scope=scope)


########定义堆叠Blocks的函数########
@slim.add_arg_scope
def stack_blocks_dense(net, blocks,
                       outputs_collections=None):
    # 使用两层循环，逐个Residual Unit地堆叠
    for block in blocks:  # 先使用两个tf.variable_scope将残差学习单元命名为block1/unit_1的形式
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    # 在第2层循环中，我们拿到每个block中每个Residual Unit的args并展开为下面四个参数
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    net = block.unit_fn(net,  # 使用残差学习单元的生成函数顺序的创建并连接所有的残差学习单元
                                        depth=unit_depth,
                                        depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)  # 将输出net添加到collections中

    return net  # 当所有block中的所有Residual Unit都堆叠完成之后，再返回最后的net作为stack_blocks_dense


# 创建ResNet通用的arg_scope,arg_scope用来定义某些函数的参数默认值
def resnet_arg_scope(is_training=True,  # 训练标记
                     weight_decay=0.0001,  # 权重衰减速率
                     batch_norm_decay=0.997,  # BN的衰减速率
                     batch_norm_epsilon=1e-5,  # BN的epsilon默认1e-5
                     batch_norm_scale=True):  # BN的scale默认值

    batch_norm_params = {  # 定义batch normalization（标准化）的参数字典
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': None,
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }

    with slim.arg_scope(  # 通过slim.arg_scope将[slim.conv2d]的几个默认参数设置好
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),  # 权重正则器设置为L2正则
            weights_initializer=slim.variance_scaling_initializer(),  # 权重初始化器
            activation_fn=tf.nn.relu,  # 激活函数
            normalizer_fn=slim.batch_norm,  # 标准化器设置为BN
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:  # ResNet原论文是VALID模式，SAME模式可让特征对齐更简单
                return arg_sc  # 最后将基层嵌套的arg_scope作为结果返回


# 定义核心的bottleneck残差学习单元
@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride,
               outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)  # 可以限定最少为四个维度
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')

        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')
            # 如果不一样就按步长和1*1的卷积改变其通道数，使得输入、输出通道数一致

        # 先是一个1*1尺寸，步长1，输出通道数为depth_bottleneck的卷积
        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')
        # 然后是3*3尺寸，步长为stride，输出通道数为depth_bottleneck的卷积
        residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                               scope='conv2')
        # 最后是1*1卷积，步长1，输出通道数depth的卷积，得到最终的residual。最后一层没有正则项也没有激活函数
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv3')

        output = shortcut + residual  # 将降采样的结果和residual相加

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.name,
                                                output)


########定义生成resnet_v2的主函数########
def resnet_v2(inputs,  # A tensor of size [batch, height_in, width_in, channels].输入
              blocks,  # 定义好的Block类的列表
              bottleneck_size=None,  # 最后输出的类数
              global_pool=True,  # 是否加上最后的一层全局平均池化
              is_training=True,
              drop_probability=1,
              include_root_block=True,  # 是否加上ResNet网络最前面通常使用的7*7卷积和最大池化
              reuse=None,  # 是否重用
              scope=None):  # 整个网络的名称
    # 在函数体先定义好variable_scope和end_points_collection
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'  # 定义end_points_collection
        with slim.arg_scope([slim.conv2d, bottleneck,
                             stack_blocks_dense],
                            outputs_collections=end_points_collection):  # 将三个参数的outputs_collections默认设置为end_points_collection
            net = inputs
            if include_root_block:  # 根据标记值
                with slim.arg_scope([slim.conv2d],
                                    activation_fn=None, normalizer_fn=None):
                    net = conv2d_same(net, 64, 7, stride=2, scope='conv1')  # 创建resnet最前面的64输出通道的步长为2的7*7卷积
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')  # 然后接最大池化
                # 经历过两个步长为2的层图片缩为1/4
            net = stack_blocks_dense(net, blocks)  # 将残差学习模块组生成好
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)  # 将collection转化为python的dict
            if global_pool:  # 根据标记添加全局平均池化层
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)  # tf.reduce_mean实现全局平均池化效率比avg_pool高
            net = slim.flatten(net)
            net = slim.dropout(net, drop_probability, is_training=is_training, scope='Dropout')
            end_points['PreLogitsFlatten'] = net
            net = slim.fully_connected(net, bottleneck_size, activation_fn=None, scope='Bottleneck', reuse=False)
            '''if num_classes is not None:  # 是否有通道数
              net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, # 无激活函数和正则项
                                normalizer_fn=None, scope='logits') # 添加一个输出通道num_classes的1*1的卷积'''
            end_points['PreLogits'] = net
            '''if num_classes is not None:
              end_points['predictions'] = slim.softmax(net, scope='predictions') # 输出网络结果'''
            return net, end_points


def resnet_v2_50(inputs,
                 drop_probability=1,
                 bottleneck_size=128,
                 global_pool=True,
                 phase_train=True,
                 reuse=None,  # 是否重用
                 scope='resnet_v2_50'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),

        Block(
            'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, bottleneck_size, global_pool, is_training=phase_train,
                     drop_probability=drop_probability,
                     include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_101(inputs,  # unit提升的主要场所是block3
                  bottleneck_size=None,
                  global_pool=True,
                  phase_train=True,
                  reuse=None,
                  scope='resnet_v2_101'):
    blocks = [
        Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, bottleneck_size, global_pool, is_training=phase_train,
                     include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_152(inputs,  # unit提升的主要场所是block3
                  bottleneck_size=None,
                  global_pool=True,
                  phase_train=True,
                  reuse=None,
                  scope='resnet_v2_152'):
    """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, bottleneck_size, global_pool, is_training=phase_train,
                     include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_200(inputs,  # unit提升的主要场所是block2
                  bottleneck_size=None,
                  global_pool=True,
                  phase_train=True,
                  reuse=None,
                  scope='resnet_v2_200'):
    """ResNet-200 model of [2]. See resnet_v2() for arg and return description."""
    blocks = [
        Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, bottleneck_size, global_pool, is_training=phase_train,
                     include_root_block=True, reuse=reuse, scope=scope)
