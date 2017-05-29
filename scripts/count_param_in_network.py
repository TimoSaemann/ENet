"""
This script is from kaushikpavani:
https://gist.github.com/kaushikpavani/a6a32bd87fdfe5529f0e908ed743f779

It calculates the number of parameters in Caffe model.
"""
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
caffe_root = 'ENet/caffe-enet/' # Change this to the absolute directoy to SegNet Caffe
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_cpu()
import numpy as np
from numpy import prod, sum
from pprint import pprint


def print_net_parameters(deploy_file):
    print "Net: " + deploy_file
    net = caffe.Net(deploy_file, caffe.TEST)
    print "Layer-wise parameters: "
    pprint([(k, v[0].data.shape) for k, v in net.params.items()])
    print "Total number of parameters: " + str(sum([prod(v[0].data.shape) for k, v in net.params.items()]))


deploy_file = "ENet/bn_conv_merged_model.prototxt"
print_net_parameters(deploy_file)
