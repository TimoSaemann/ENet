# ENet in Caffe

![Alt text](/example_image/image_enet.PNG?raw=true "image_enet")


## Execution times and hardware requirements

| Network | 1024x512 | 1280x720 | Parameters | Model size (fp32) |
|:--------:|:----------------:|:------------------:|:------------:|:--------------:|
| ENet | 20.4 ms | 32.9 ms | 0.36 M | 1.5 MB |
| SegNet | 66.5 ms | 114.3 ms | 29.4 M | 117.8 MB |

A comparison of computational time, number of parameters and model size required for ENet and SegNet. The caffe time command was used to compute time requirement averaged over 100 iterations. Hardware setup: Intel Xeon E5-1620v3, Titan X Pascal with cuDNN v5.1. 

## Tutorial

For a detailed introduction on how to train and test ENet please see the [tutorial](https://github.com/TimoSaemann/ENet/tree/master/Tutorial).


## Publication

This work has been published in arXiv: [`ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation`](https://arxiv.org/abs/1606.02147).

## ModelDepot

Also available on [ModelDepot](https://modeldepot.io/browse).

## License

This software is released under a creative commons license which allows for personal and research use only. You can view a license summary here: http://creativecommons.org/licenses/by-nc/4.0/
