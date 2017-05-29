#!/usr/bin/env python
# -*- coding: utf-8 -*
"""
This script calculates the class weighting for the "SoftmaxWithLoss" layer.
cf. https://arxiv.org/pdf/1411.4734.pdf
"we weight each pixel by αc = median freq/freq(c) where freq(c) is the number of pixels of class c divided by the total
number of pixels in images where c is present, and median freq is the median of these frequencies."
"""
import argparse
import numpy as np
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2


__author__ = 'Timo Sämann'
__university__ = 'Aschaffenburg University of Applied Sciences'
__email__ = 'Timo.Saemann@gmx.de'
__data__ = '16th May, 2017'


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='absolute path to your data file')
    parser.add_argument('--num_classes', type=int, required=True, help='absolute path to your data file')
    return parser


if __name__ == '__main__':
    parser1 = make_parser()
    args = parser1.parse_args()

    classes, freq, class_weights, present_in_data, a = ([0 for i in xrange(args.num_classes)] for i in xrange(5))
    image_nr = 0
    median_freq = 0

    with open(args.source) as inf:
        for line in inf:
            print 'progress: {}'.format(image_nr+1)
            image_nr += 1
            columns = line.split()  # split line into columns
            if len(columns) != 2:
                raise Exception("The file must have 2 columns (first column: image; second column: labels). "
                                "Received columns = {}".format(len(columns)))

            labels = cv2.imread(columns[1], 0)
            for i in xrange(args.num_classes):
                if ((labels == i).sum()) == 0:
                    pass
                else:
                    classes[i] += (labels == i).sum()  # sum up all pixels that belongs to a certain class
                    present_in_data[i] += 1  # how often the class is present in the dataset

        for l in xrange(args.num_classes):
            if present_in_data[l] == 0:
                raise Exception("The class {} is not present in the dataset".format(l+1))

            freq[l] = float(classes[l]) / float(present_in_data[l])  # calculate freq per class
            median_freq = 0.5*sum(freq)/(len(classes))

        for c in xrange(args.num_classes):
            a[c] = float(median_freq) / float(freq[c])

        for m in xrange(args.num_classes):
            print '    class_weighting: {:.4f}'.format(a[m])

        print "Done!"
