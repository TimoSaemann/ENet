#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script visualize the semantic segmentation of ENet.
"""
import os
import numpy as np
from argparse import ArgumentParser
from os.path import join
import argparse
import sys
caffe_root = 'ENet/caffe-enet/'  # Change this to the absolute directory to ENet Caffe
sys.path.insert(0, caffe_root + 'python')
import caffe
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2


__author__ = 'Timo Sämann'
__university__ = 'Aschaffenburg University of Applied Sciences'
__email__ = 'Timo.Saemann@gmx.de'
__data__ = '24th May, 2017'


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='.prototxt file for inference')
    parser.add_argument('--weights', type=str, required=True, help='.caffemodel file')
    parser.add_argument('--colours', type=str, required=True, help='label colours')
    parser.add_argument('--input_image', type=str, required=True, help='input image path')
    parser.add_argument('--out_dir', type=str, default=None, help='output directory in which the segmented images '
                                                                   'should be stored')
    parser.add_argument('--gpu', type=str, default='0', help='0: gpu mode active, else gpu mode inactive')

    return parser


if __name__ == '__main__':
    parser1 = make_parser()
    args = parser1.parse_args()
    if args.gpu == 0:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(args.model, args.weights, caffe.TEST)

    input_shape = net.blobs['data'].data.shape
    output_shape = net.blobs['deconv6_0_0'].data.shape

    label_colours = cv2.imread(args.colours, 1).astype(np.uint8)
    input_image = cv2.imread(args.input_image, 1).astype(np.float32)

    input_image = cv2.resize(input_image, (input_shape[3], input_shape[2]))
    input_image = input_image.transpose((2, 0, 1))
    input_image = np.asarray([input_image])

    out = net.forward_all(**{net.inputs[0]: input_image})

    prediction = net.blobs['deconv6_0_0'].data[0].argmax(axis=0)

    prediction = np.squeeze(prediction)
    prediction = np.resize(prediction, (3, input_shape[2], input_shape[3]))
    prediction = prediction.transpose(1, 2, 0).astype(np.uint8)

    prediction_rgb = np.zeros(prediction.shape, dtype=np.uint8)
    label_colours_bgr = label_colours[..., ::-1]
    cv2.LUT(prediction, label_colours_bgr, prediction_rgb)

    cv2.imshow("ENet", prediction_rgb)
    key = cv2.waitKey(0)

    if args.out_dir is not None:
        input_path_ext = args.input_image.split(".")[-1]
        input_image_name = args.input_image.split("/")[-1:][0].replace('.' + input_path_ext, '')
        out_path_im = args.out_dir + input_image_name + '_enet' + '.' + input_path_ext
        out_path_gt = args.out_dir + input_image_name + '_enet_gt' + '.' + input_path_ext

        cv2.imwrite(out_path_im, prediction_rgb)
        # cv2.imwrite(out_path_gt, prediction) #  label images, where each pixel has an ID that represents the class






