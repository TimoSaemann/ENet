#!/bin/bash
DIR="$( cd "$(dirname "$0")" ; pwd -P )"
wget -O $DIR/weights.caffemodel https://www.dropbox.com/sh/5wjmb5twfr5b0wo/AADEpW5a8-GSSt5pfJYCKxoOa?dl=1
unzip $DIR/weights.caffemodel 
rm $DIR/weights.caffemodel
