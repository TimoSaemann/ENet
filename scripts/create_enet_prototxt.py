#!/usr/bin/env python
# -*- coding: utf-8 -*
"""
This script creates the prototxt files for the ENet architecture. It requires the path to your trainings data
file (text file of white-space separated paths to images (.jpeg or .png) and corresponding label images alternatively)
and also the mode in which it should operate (--mode). The following modes are available:
 --mode train_encoder: creates the prototxt file for training the encoder architecture (first stage)
 --mode train_encoder_decoder: creates the prototxt file for training ENet end-to-end (second stage)
 --mode test: creates the prototxt file for testing ENet (deploy)
(see -h for information about further arguments)
"""
import argparse
import os
caffe_root = 'ENet/caffe-enet/'  # Change this to the absolute directory to SegNet Caffe
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L
from caffe import params as P

__author__ = 'Timo Sämann'
__university__ = 'Aschaffenburg University of Applied Sciences'
__email__ = 'Timo.Saemann@gmx.de'
__data__ = '1th May, 2017'


def data_layer_train(n, label_divide_factor):
    n.data, n.label = L.DenseImageData(dense_image_data_param=dict(source=args.source, new_height=args.new_height,
                                                                   new_width=args.new_width, batch_size=args.batch_size,
                                                                   shuffle=args.shuffle,
                                                                   label_divide_factor=label_divide_factor), ntop=2)
    return n.to_proto()


def data_layer_test(n):
    n.data = L.Input(input_param=dict(shape=dict(dim=[1, 3, args.input_size[0], args.input_size[1]])))
    return n.to_proto()


def initial_block(n):
    bn_mode = 0
    if args.mode == 'test':
        bn_mode = 1
    n.conv0_1 = L.Convolution(n.data, num_output=13, bias_term=1, pad=1, kernel_size=3, stride=2,
                              weight_filler=dict(type='msra'))
    n.pool0_1 = L.Pooling(n.data, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.concat0_1 = L.Concat(n.conv0_1, n.pool0_1, axis=1)
    n.bn0_1 = L.BN(n.concat0_1, scale_filler=dict(type='constant', value=1), bn_mode=bn_mode,
                   shift_filler=dict(type='constant', value=0.001), param=[dict(lr_mult=1, decay_mult=1),
                                                                           dict(lr_mult=1, decay_mult=0)])

    n.prelu0_1 = L.PReLU(n.bn0_1)
    last_layer = 'prelu0_1'
    return n.to_proto(), last_layer


def bottleneck(n, prev_layer, stage, num_bottle, num_output, type, param_add=None):
    scale_factor = 4
    input_layer = prev_layer  # save input layer of this bottleneck
    module = 0
    phase = 'TRAIN'
    bn_mode = 0
    if args.mode == 'test':
        phase = 'TEST'
        bn_mode = 1

    param_str = str({'phase': phase, 'p': '0.01'})

    # first module
    conv_name = 'conv{}_{}_{}'.format(stage, num_bottle, module)
    bn_name = 'bn{}_{}_{}'.format(stage, num_bottle, module)
    prelu_name = 'prelu{}_{}_{}'.format(stage, num_bottle, module)
    kernel_size = 1
    stride = 1

    if stage >= 2:
        param_str = str({'phase': phase, 'p': '0.1'})

    if type == 'downsampling':
        kernel_size = 2
        stride = 2

    setattr(n, conv_name, L.Convolution(getattr(n, prev_layer), num_output=int(num_output/scale_factor), bias_term=0,
                                        kernel_size=kernel_size, stride=stride, weight_filler=dict(type='msra')))
    setattr(n, bn_name, L.BN(getattr(n, conv_name), scale_filler=dict(type='constant', value=1), bn_mode=bn_mode,
                             shift_filler=dict(type='constant', value=0.001), param=[dict(lr_mult=1, decay_mult=1),
                                                                                     dict(lr_mult=1, decay_mult=0)]))
    if param_add == 'relu':
        setattr(n, prelu_name, L.ReLU(getattr(n, bn_name)))
    else:
        setattr(n, prelu_name, L.PReLU(getattr(n, bn_name)))
    prev_layer = getattr(n, prelu_name)

    # second module conv
    conv_name = 'conv{}_{}_{}'.format(stage, num_bottle, module+1)
    bn_name = 'bn{}_{}_{}'.format(stage, num_bottle, module+1)
    prelu_name = 'prelu{}_{}_{}'.format(stage, num_bottle, module+1)

    if type == 'dilated':
        setattr(n, conv_name, L.Convolution(prev_layer, num_output=int(num_output/scale_factor), bias_term=1, kernel_size=3,
                                            stride=1, pad=param_add, dilation=param_add,
                                            weight_filler=dict(type='msra')))
    elif type == 'asymmetric':
        conv_name2 = 'conv{}_{}_{}_a'.format(stage, num_bottle, module+1)
        setattr(n, conv_name2, L.Convolution(prev_layer, num_output=int(num_output/scale_factor), bias_term=0,
                                             kernel_h=param_add, kernel_w=1, stride=1, pad=1,
                                             weight_filler=dict(type='msra')))
        setattr(n, conv_name, L.Convolution(getattr(n, conv_name2), num_output=int(num_output/scale_factor), bias_term=1,
                                            kernel_h=1, kernel_w=param_add, stride=1, pad=1,
                                            weight_filler=dict(type='msra')))
    elif type == 'upsampling':
        conv_name = 'deconv{}_{}_{}'.format(stage, num_bottle, module+1)
        setattr(n, conv_name, L.Deconvolution(prev_layer, convolution_param=dict(num_output=int(num_output/scale_factor),
                                                                                 bias_term=1, kernel_size=2, stride=2)))
    else:
        setattr(n, conv_name, L.Convolution(prev_layer, num_output=int(num_output/scale_factor), bias_term=1,
                                            kernel_size=3, stride=1, pad=1, weight_filler=dict(type='msra')))

    setattr(n, bn_name, L.BN(getattr(n, conv_name), scale_filler=dict(type='constant', value=1), bn_mode=bn_mode,
                             shift_filler=dict(type='constant', value=0.001), param=[dict(lr_mult=1, decay_mult=1),
                                                                                     dict(lr_mult=1, decay_mult=0)]))
    if param_add == 'relu':
        setattr(n, prelu_name, L.ReLU(getattr(n, bn_name)))
    else:
        setattr(n, prelu_name, L.PReLU(getattr(n, bn_name)))
    prev_layer = getattr(n, prelu_name)

    # third module 1x1
    conv_name = 'conv{}_{}_{}'.format(stage, num_bottle, module+2)
    bn_name = 'bn{}_{}_{}'.format(stage, num_bottle, module+2)
    prelu_name = 'prelu{}_{}_{}'.format(stage, num_bottle, module+2)
    setattr(n, conv_name, L.Convolution(prev_layer, num_output=num_output, bias_term=0, kernel_size=1, stride=1,
                                        weight_filler=dict(type='msra')))
    setattr(n, bn_name, L.BN(getattr(n, conv_name), scale_filler=dict(type='constant', value=1), bn_mode=bn_mode,
                             shift_filler=dict(type='constant', value=0.001), param=[dict(lr_mult=1, decay_mult=1),
                                                                                     dict(lr_mult=1, decay_mult=0)]))

    prev_layer = getattr(n, bn_name)

    # regularizer (fourth module)
    drop_name = 'drop{}_{}_{}'.format(stage, num_bottle, module+3)
    setattr(n, drop_name, L.Python(prev_layer, python_param=dict(module="spatial_dropout",
                                                                 layer="SpatialDropoutLayer", param_str=param_str)))
    prev_layer1 = getattr(n, drop_name)

    eltwise_name = 'eltwise{}_{}_{}'.format(stage, num_bottle, module+4)
    prelu_name = 'prelu{}_{}_{}'.format(stage, num_bottle, module+4)

    # main branch; pool and pad, just for type == downsampling
    if type == 'downsampling':
        pool_name = 'pool{}_{}_{}'.format(stage, num_bottle, module+4)
        conv_name = 'conv{}_{}_{}'.format(stage, num_bottle, module+4)
        bn_name = 'bn{}_{}_{}'.format(stage, num_bottle, module+4)

        if stage == 1 and args.mode != 'train_encoder':
            n.pool1_0_4, n.pool1_0_4_mask = L.Pooling(getattr(n, input_layer), kernel_size=2, stride=2,
                                                      pool=P.Pooling.MAX, ntop=2)
        elif stage == 2 and args.mode != 'train_encoder':
            n.pool2_0_4, n.pool2_0_4_mask = L.Pooling(getattr(n, input_layer), kernel_size=2, stride=2,
                                                      pool=P.Pooling.MAX, ntop=2)
        elif stage == 1 and args.mode == 'train_encoder':
            n.pool1_0_4 = L.Pooling(getattr(n, input_layer), kernel_size=2, stride=2, pool=P.Pooling.MAX)

        elif stage == 2 and args.mode == 'train_encoder':
            n.pool2_0_4 = L.Pooling(getattr(n, input_layer), kernel_size=2, stride=2, pool=P.Pooling.MAX)

        else:
            print ("downsampling is just available for stage 1 and 2")

        setattr(n, conv_name,
                L.Convolution(getattr(n, pool_name), num_output=num_output, bias_term=0, kernel_size=1,
                              stride=1, weight_filler=dict(type='msra')))
        setattr(n, bn_name, L.BN(getattr(n, conv_name), scale_filler=dict(type='constant', value=1), bn_mode=bn_mode,
                                 shift_filler=dict(type='constant', value=0.001), param=[dict(lr_mult=1, decay_mult=1),
                                                                                         dict(lr_mult=1,
                                                                                              decay_mult=0)]))
        prev_layer2 = getattr(n, bn_name)

    elif type == 'upsampling':
        conv_name = 'conv{}_{}_{}'.format(stage, num_bottle, module+4)
        bn_name = 'bn{}_{}_{}'.format(stage, num_bottle, module+4)
        upsample_name = 'upsample{}_{}_{}'.format(stage, num_bottle, module+4)

        setattr(n, conv_name, L.Convolution(getattr(n, input_layer), num_output=num_output, bias_term=0, kernel_size=1,
                                            stride=1, weight_filler=dict(type='msra')))
        setattr(n, bn_name, L.BN(getattr(n, conv_name), scale_filler=dict(type='constant', value=1), bn_mode=bn_mode,
                                 shift_filler=dict(type='constant', value=0.001), param=[dict(lr_mult=1, decay_mult=1),
                                                                                         dict(lr_mult=1,
                                                                                              decay_mult=0)]))
        if stage == 4:
            setattr(n, upsample_name, L.Upsample(getattr(n, bn_name), n.pool2_0_4_mask, scale=2))
        elif stage == 5:
            setattr(n, upsample_name, L.Upsample(getattr(n, bn_name), n.pool1_0_4_mask, scale=2))
        else:
            print ("upsampling is just available for stage 4 and 5")

        prev_layer2 = getattr(n, upsample_name)

    else:
        prev_layer2 = getattr(n, input_layer)  # if not type==downsampling: bottom layer of eltwise is input layer of
        # bottleneck

    setattr(n, eltwise_name, L.Eltwise(prev_layer1, prev_layer2))
    if param_add == 'relu':
        setattr(n, prelu_name, L.ReLU(getattr(n, eltwise_name)))
    else:
        setattr(n, prelu_name, L.PReLU(getattr(n, eltwise_name)))
    last_layer = prelu_name

    return n.to_proto(), last_layer


def fullconv(n, prev_layer, stage, num_bottle, num_of_classes):
    module = 0
    kernel_size = 2
    stride = 2
    conv_name = 'deconv{}_{}_{}'.format(stage, num_bottle, module)
    if args.mode == 'train_encoder':
        kernel_size = 1
        stride = 1
        conv_name = 'deconv_encoder{}_{}_{}'.format(stage, num_bottle, module)

    setattr(n, conv_name, L.Deconvolution(getattr(n, prev_layer),
                                          convolution_param=dict(num_output=num_of_classes, bias_term=1,
                                                                 kernel_size=kernel_size, stride=stride)))
    last_layer = conv_name
    return n.to_proto(), last_layer


def loss_layer(n, prev_layer):
    n.loss = L.SoftmaxWithLoss(getattr(n, prev_layer), n.label, loss_param=dict(ignore_label=args.ignore_label,
                                                                                weight_by_label_freqs=0))
    n.accuracy = L.Accuracy(getattr(n, prev_layer), n.label, top='per_class_accuracy')

    return n.to_proto()

"""
def argmax_layer(n, prev_layer):
    n.argmax = L.ArgMax(getattr(n, prev_layer), argmax_param=dict(axis=1))

    return n.to_proto()
"""

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='absolute path to your data file')
    parser.add_argument('--mode', type=str, required=True, help='train_encoeder, train_encoder_decoder and test mode '
                                                                'available')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--shuffle', type=str, default=1)
    parser.add_argument('--new_height', type=int, default='512', help='reshape input size')
    parser.add_argument('--new_width', type=int, default='1024', help='reshape input size')
    parser.add_argument('--ignore_label', type=int, default='255', help='label of ground truth that should be ignored '
                                                                        'during training')
    parser.add_argument('--num_of_classes', type=int, default='19', help='number of output classes')
    parser.add_argument('--input_size', nargs='*', type=list, default=[512, 1024],
                        help='size of input image for deploy network. [h, w]')
    parser.add_argument('--out_dir', type=str, default="/"+os.path.join(*os.path.realpath(__file__).split("/")[:-2])
                                                       + "/prototxts/", help='output directory in which the prototxt '
                                                                             'file should be stored')
    return parser

if __name__ == '__main__':
    parser1 = make_parser()
    args = parser1.parse_args()

    n = caffe.NetSpec()

    if args.mode == 'train_encoder_decoder':
        network = data_layer_train(n, 1)
        out_directory = args.out_dir + 'enet_train_encoder_decoder.prototxt'
    elif args.mode == 'train_encoder':
        network = data_layer_train(n, 8)
        out_directory = args.out_dir + 'enet_train_encoder.prototxt'
    elif args.mode == 'test':
        network = data_layer_test(n)
        out_directory = args.out_dir + 'enet_deploy.prototxt'
    else:
        raise Exception("Wrong mode! Just train_encoeder, train_encoder_decoder and test mode available, "
                        "but received {}.".format(args.mode))

    network, prev_layer = initial_block(n)

    network, prev_layer = bottleneck(n, prev_layer, 1, 0, 64, 'downsampling')  # stage, number_bottleneck, num_input,
    #  type,

    for i in range(1, 5):
        network, prev_layer = bottleneck(n, prev_layer, 1, i, 64, 'regular')

    network, prev_layer = bottleneck(n, prev_layer, 2, 0, 128, 'downsampling')

    for j in range(2, 4):
        network, prev_layer = bottleneck(n, prev_layer, j, 1, 128, 'regular')
        network, prev_layer = bottleneck(n, prev_layer, j, 2, 128, 'dilated', 2)
        network, prev_layer = bottleneck(n, prev_layer, j, 3, 128, 'asymmetric', 5)
        network, prev_layer = bottleneck(n, prev_layer, j, 4, 128, 'dilated', 4)
        network, prev_layer = bottleneck(n, prev_layer, j, 5, 128, 'regular')
        network, prev_layer = bottleneck(n, prev_layer, j, 6, 128, 'dilated', 8)
        network, prev_layer = bottleneck(n, prev_layer, j, 7, 128, 'asymmetric', 5)
        network, prev_layer = bottleneck(n, prev_layer, j, 8, 128, 'dilated', 16)

    if args.mode == 'train_encoder_decoder' or args.mode == 'test':
        network, prev_layer = bottleneck(n, prev_layer, 4, 0, 64, 'upsampling', 'relu')  # last one = additional flag,
        # that relu is used instead of prelu
        network, prev_layer = bottleneck(n, prev_layer, 4, 1, 64, 'regular', 'relu')
        network, prev_layer = bottleneck(n, prev_layer, 4, 2, 64, 'regular', 'relu')

        network, prev_layer = bottleneck(n, prev_layer, 5, 0, 16, 'upsampling', 'relu')
        network, prev_layer = bottleneck(n, prev_layer, 5, 1, 16, 'regular', 'relu')

    network, prev_layer = fullconv(n, prev_layer, 6, 0, args.num_of_classes)

    if args.mode == 'train_encoder_decoder' or args.mode == 'train_encoder':
        network = loss_layer(n, prev_layer)
    # elif args.mode == 'test':
    #     network = argmax_layer(n, prev_layer)

    with open(out_directory, 'w') as f:
        f.write('name: "ENet"\n')
        f.write(str(network))

    print ("Done!")
