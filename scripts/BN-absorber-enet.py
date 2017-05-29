#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
After ENet has been trained, run compute_bn_statistics.py script and then BN-absorber.py.

For inference batch normalization and dropout layer can be merged into convolutional kernels, to
speed up the network. All three layers applies a linear transformation. For that reason
the batch normalization and dropout layer can be absorbed in the previous convolutional layer
by modifying its weights and biases. That is exactly what the script does.
"""
import os
import numpy as np
from argparse import ArgumentParser
from os.path import join
import argparse
import sys
caffe_root = 'ENet/caffe-enet/'  # Change this to the absolute directory to SegNet Caffe
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import google.protobuf as pb
from caffe import layers as L


__author__ = 'Timo Sämann'
__university__ = 'Aschaffenburg University of Applied Sciences'
__email__ = 'Timo.Saemann@gmx.de'
__data__ = '6th May, 2017'


def copy_double(data):
    return np.array(data, copy=True, dtype=np.double)


def drop_absorber_weights(model, net):
    # load the prototxt file as a protobuf message
    with open(model) as f:
        str2 = f.read()
    msg = caffe_pb2.NetParameter()
    text_format.Merge(str2, msg)

    # iterate over all layers of the network
    for i, layer in enumerate(msg.layer):

        if not layer.type == 'Python':
            continue

        conv_layer = msg.layer[i - 2].name  # conv layers are always two layers behind dropout

        # get some necessary sizes
        kernel_size = 1
        shape_of_kernel_blob = net.params[conv_layer][0].data.shape
        number_of_feature_maps = list(shape_of_kernel_blob[0:1])
        shape_of_kernel_blob = list(shape_of_kernel_blob[1:4])
        for x in shape_of_kernel_blob:
            kernel_size *= x

        weight = copy_double(net.params[conv_layer][0].data)
        bias = copy_double(net.params[conv_layer][1].data)

        # get p from dropout layer
        python_param_str = eval(msg.layer[i].python_param.param_str)
        p = float(python_param_str['p'])
        scale = 1/(1-p)

        # manipulate the weights and biases over all feature maps:
        for j in xrange(number_of_feature_maps[0]):
            net.params[conv_layer][0].data[j] = weight[j] * scale
            net.params[conv_layer][1].data[j] = bias[j] * scale

        return net


def bn_absorber_weights(model, weights):

    # load the prototxt file as a protobuf message
    with open(model) as f:
        str2 = f.read()
    msg = caffe_pb2.NetParameter()
    text_format.Merge(str2, msg)

    # load net
    net = caffe.Net(model, weights, caffe.TEST)

    # iterate over all layers of the network
    for i, layer in enumerate(msg.layer):

        if not layer.type == 'BN':
            continue

        # check the special case that the bn layer comes right after concat layer
        if msg.layer[i].name == 'bn0_1':
            continue

        if msg.layer[i - 1].type == 'Deconvolution':  # do not merge into deconv layer
            continue

        bn_layer = msg.layer[i].name
        conv_layer = msg.layer[i - 1].name

        # get some necessary sizes
        kernel_size = 1
        shape_of_kernel_blob = net.params[conv_layer][0].data.shape
        number_of_feature_maps = list(shape_of_kernel_blob[0:1])
        shape_of_kernel_blob = list(shape_of_kernel_blob[1:4])
        for x in shape_of_kernel_blob:
            kernel_size *= x

        weight = copy_double(net.params[conv_layer][0].data)
        bias = copy_double(net.params[conv_layer][1].data)

        # receive new_gamma and new_beta which was already calculated by the compute_bn_statistics.py script
        new_gamma = net.params[bn_layer][0].data[...]
        new_beta = net.params[bn_layer][1].data[...]

        # manipulate the weights and biases over all feature maps:
        # weight_new = weight * gamma_new
        # bias_new = bias * gamma_new + beta_new
        # for more information see https://github.com/alexgkendall/caffe-segnet/issues/109
        for j in xrange(number_of_feature_maps[0]):

            net.params[conv_layer][0].data[j] = weight[j] * np.repeat(new_gamma.item(j), kernel_size).reshape(
                net.params[conv_layer][0].data[j].shape)
            net.params[conv_layer][1].data[j] = bias[j] * new_gamma.item(j) + new_beta.item(j)

        # set the no longer needed bn params to zero
        net.params[bn_layer][0].data[:] = 0
        net.params[bn_layer][1].data[:] = 0

    return net


def bn_absorber_prototxt(model):

    # load the prototxt file as a protobuf message
    with open(model) as k:
        str1 = k.read()
    msg1 = caffe_pb2.NetParameter()
    text_format.Merge(str1, msg1)

    # search for bn layer and remove them
    for i, l in enumerate(msg1.layer):
        if l.type == "BN":
            if msg1.layer[i].name == 'bn0_1':
                continue
            if msg1.layer[i - 1].type == 'Deconvolution':
                continue
            msg1.layer.remove(l)
            msg1.layer[i].bottom.append(msg1.layer[i-1].top[0])

            if len(msg1.layer[i].bottom) == 2:
                msg1.layer[i].bottom.remove(msg1.layer[i].bottom[0])
            elif len(msg1.layer[i].bottom) == 3:
                if ('bn' in msg1.layer[i].bottom[0]) is True:  # to remove just the layers with 'bn' in the name
                    msg1.layer[i].bottom.remove(msg1.layer[i].bottom[0])
                elif ('bn' in msg1.layer[i].bottom[1]) is True:
                    msg1.layer[i].bottom.remove(msg1.layer[i].bottom[1])
                else:
                    raise Exception("no bottom blob with name 'bn' present in {} layer".format(msg1.layer[i]))

            else:
                raise Exception("bn absorber does not support more than 2 input blobs for layer {}"
                                .format(msg1.layer[i]))

            if msg1.layer[i].type == 'Upsample':
                temp = msg1.layer[i].bottom[0]
                msg1.layer[i].bottom[0] = msg1.layer[i].bottom[1]
                msg1.layer[i].bottom[1] = temp
                # l.bottom.append(l.top[0]) #msg1.layer[i-1].top

    return msg1


def drop_absorber_prototxt(msg1):

    # search for bn layer and remove them
    for i, l in enumerate(msg1.layer):
        if l.type == "Python":
            msg1.layer.remove(l)
            if msg1.layer[i].type == 'Pooling' or msg1.layer[i].type == 'Convolution':
                msg1.layer[i+2].bottom.append(msg1.layer[i-1].top[0])
                if ('drop' in msg1.layer[i+2].bottom[0]) is True:  # to remove just the layers with 'drop' in the name
                    msg1.layer[i + 2].bottom.remove(msg1.layer[i+2].bottom[0])
                elif ('drop' in msg1.layer[i+2].bottom[1]) is True:
                    msg1.layer[i + 2].bottom.remove(msg1.layer[i+2].bottom[1])
                else:
                    raise Exception("no bottom blob with name 'drop' present in {} layer".format(msg1.layer[i+2]))

            else:
                msg1.layer[i].bottom.append(msg1.layer[i - 1].top[0])
                msg1.layer[i].bottom.remove(msg1.layer[i].bottom[0])

    return msg1


def add_bias_to_conv(model, weights, out_dir):
    # load the prototxt file as a protobuf message
    with open(model) as n:
        str1 = n.read()
    msg2 = caffe_pb2.NetParameter()
    text_format.Merge(str1, msg2)

    for l2 in msg2.layer:
        if l2.type == "Convolution":
            if l2.convolution_param.bias_term is False:
                l2.convolution_param.bias_term = True
                l2.convolution_param.bias_filler.type = 'constant'
                l2.convolution_param.bias_filler.value = 0.0  # actually default value

    model_temp = os.path.join(out_dir, "model_temp.prototxt")
    print "Saving temp model..."
    with open(model_temp, 'w') as m:
        m.write(text_format.MessageToString(msg2))

    net_src = caffe.Net(model, weights, caffe.TEST)
    net_des = caffe.Net(model_temp, caffe.TEST)

    for l3 in net_src.params.keys():
        for i in range(len(net_src.params[l3])):

            net_des.params[l3][i].data[:] = net_src.params[l3][i].data[:]

    # save weights with bias
    weights_temp = os.path.join(out_dir, "weights_temp.caffemodel")
    print "Saving temp weights..."
    net_des.save(weights_temp)

    return model_temp, weights_temp


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='.prototxt file which you want to use for inference')
    parser.add_argument('--weights', type=str, required=True, help='.caffemodel file in which the batch normalization '
                                                                   'and convolutional layer should be merged')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='output directory in which the modified model and weights should be stored')
    parser.add_argument('--gpu', type=str, default='0', help='0: gpu mode active, else gpu mode inactive')
    return parser


if __name__ == '__main__':
    parser1 = make_parser()
    args = parser1.parse_args()
    if args.gpu == 0:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    # check if output directory exist
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    model_temp, weights_temp = add_bias_to_conv(args.model, args.weights, args.out_dir)
    network = bn_absorber_weights(model_temp, weights_temp)  # merge bn layer into conv kernel
    network = drop_absorber_weights(model_temp, network)  # merge dropout layer into conv kernel
    msg_proto = bn_absorber_prototxt(model_temp)  # remove bn layer from prototxt file
    msg_proto = drop_absorber_prototxt(msg_proto)  # remove dropout layer from prototxt file

    os.remove(model_temp)
    os.remove(weights_temp)

    # save prototxt for inference
    print "Saving inference prototxt file..."
    path = os.path.join(args.out_dir, "bn_conv_merged_model.prototxt")
    with open(path, 'w') as m:
        m.write(text_format.MessageToString(msg_proto))

    # save weights
    print "Saving new weights..."
    network.save(os.path.join(args.out_dir, "bn_conv_merged_weights.caffemodel"))
    print "Done!"
