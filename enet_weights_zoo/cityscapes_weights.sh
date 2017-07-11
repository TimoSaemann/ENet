#!/bin/bash
DIR="$( cd "$(dirname "$0")" ; pwd -P )"
wget -O $DIR/weights.caffemodel https://www.dropbox.com/sh/5wjmb5twfr5b0wo/AADEpW5a8-GSSt5pfJYCKxoOa?dl=1
wget -O $DIR/weights2.caffemodel https://www.dropbox.com/s/3i367gsl7sspeo1/cityscapes_weights_before_bn_merge.caffemodel?dl=1
unzip $DIR/weights.caffemodel 
unzip $DIR/weights2.caffemodel 
rm $DIR/weights.caffemodel
rm $DIR/weights2.caffemodel

