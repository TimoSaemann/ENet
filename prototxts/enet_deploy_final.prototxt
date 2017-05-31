name: "ENet"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param {
    shape {
      dim: 1
      dim: 3
      dim: 512
      dim: 1024
    }
  }
}
layer {
  name: "conv0_1"
  type: "Convolution"
  bottom: "data"
  top: "conv0_1"
  convolution_param {
    num_output: 13
    bias_term: true
    pad: 1
    kernel_size: 3
    stride: 2
    weight_filler {
      type: "msra"
    }
  }
}
layer {
  name: "pool0_1"
  type: "Pooling"
  bottom: "data"
  top: "pool0_1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "concat0_1"
  type: "Concat"
  bottom: "conv0_1"
  bottom: "pool0_1"
  top: "concat0_1"
  concat_param {
    axis: 1
  }
}
layer {
  name: "bn0_1"
  type: "BN"
  bottom: "concat0_1"
  top: "bn0_1"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 1.0
    decay_mult: 0.0
  }
  bn_param {
    scale_filler {
      type: "constant"
      value: 1.0
    }
    shift_filler {
      type: "constant"
      value: 0.001
    }
    bn_mode: INFERENCE
  }
}
layer {
  name: "prelu0_1"
  type: "PReLU"
  bottom: "bn0_1"
  top: "prelu0_1"
}
layer {
  name: "conv1_0_0"
  type: "Convolution"
  bottom: "prelu0_1"
  top: "conv1_0_0"
  convolution_param {
    num_output: 16
    bias_term: true
    kernel_size: 2
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu1_0_0"
  type: "PReLU"
  bottom: "conv1_0_0"
  top: "prelu1_0_0"
}
layer {
  name: "conv1_0_1"
  type: "Convolution"
  bottom: "prelu1_0_0"
  top: "conv1_0_1"
  convolution_param {
    num_output: 16
    bias_term: true
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
  }
}
layer {
  name: "prelu1_0_1"
  type: "PReLU"
  bottom: "conv1_0_1"
  top: "prelu1_0_1"
}
layer {
  name: "conv1_0_2"
  type: "Convolution"
  bottom: "prelu1_0_1"
  top: "conv1_0_2"
  convolution_param {
    num_output: 64
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "pool1_0_4"
  type: "Pooling"
  bottom: "prelu0_1"
  top: "pool1_0_4"
  top: "pool1_0_4_mask"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv1_0_4"
  type: "Convolution"
  bottom: "pool1_0_4"
  top: "conv1_0_4"
  convolution_param {
    num_output: 64
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise1_0_4"
  type: "Eltwise"
  bottom: "conv1_0_4"
  bottom: "conv1_0_2"
  top: "eltwise1_0_4"
}
layer {
  name: "prelu1_0_4"
  type: "PReLU"
  bottom: "eltwise1_0_4"
  top: "prelu1_0_4"
}
layer {
  name: "conv1_1_0"
  type: "Convolution"
  bottom: "prelu1_0_4"
  top: "conv1_1_0"
  convolution_param {
    num_output: 16
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu1_1_0"
  type: "PReLU"
  bottom: "conv1_1_0"
  top: "prelu1_1_0"
}
layer {
  name: "conv1_1_1"
  type: "Convolution"
  bottom: "prelu1_1_0"
  top: "conv1_1_1"
  convolution_param {
    num_output: 16
    bias_term: true
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
  }
}
layer {
  name: "prelu1_1_1"
  type: "PReLU"
  bottom: "conv1_1_1"
  top: "prelu1_1_1"
}
layer {
  name: "conv1_1_2"
  type: "Convolution"
  bottom: "prelu1_1_1"
  top: "conv1_1_2"
  convolution_param {
    num_output: 64
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise1_1_4"
  type: "Eltwise"
  bottom: "prelu1_0_4"
  bottom: "conv1_1_2"
  top: "eltwise1_1_4"
}
layer {
  name: "prelu1_1_4"
  type: "PReLU"
  bottom: "eltwise1_1_4"
  top: "prelu1_1_4"
}
layer {
  name: "conv1_2_0"
  type: "Convolution"
  bottom: "prelu1_1_4"
  top: "conv1_2_0"
  convolution_param {
    num_output: 16
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu1_2_0"
  type: "PReLU"
  bottom: "conv1_2_0"
  top: "prelu1_2_0"
}
layer {
  name: "conv1_2_1"
  type: "Convolution"
  bottom: "prelu1_2_0"
  top: "conv1_2_1"
  convolution_param {
    num_output: 16
    bias_term: true
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
  }
}
layer {
  name: "prelu1_2_1"
  type: "PReLU"
  bottom: "conv1_2_1"
  top: "prelu1_2_1"
}
layer {
  name: "conv1_2_2"
  type: "Convolution"
  bottom: "prelu1_2_1"
  top: "conv1_2_2"
  convolution_param {
    num_output: 64
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise1_2_4"
  type: "Eltwise"
  bottom: "prelu1_1_4"
  bottom: "conv1_2_2"
  top: "eltwise1_2_4"
}
layer {
  name: "prelu1_2_4"
  type: "PReLU"
  bottom: "eltwise1_2_4"
  top: "prelu1_2_4"
}
layer {
  name: "conv1_3_0"
  type: "Convolution"
  bottom: "prelu1_2_4"
  top: "conv1_3_0"
  convolution_param {
    num_output: 16
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu1_3_0"
  type: "PReLU"
  bottom: "conv1_3_0"
  top: "prelu1_3_0"
}
layer {
  name: "conv1_3_1"
  type: "Convolution"
  bottom: "prelu1_3_0"
  top: "conv1_3_1"
  convolution_param {
    num_output: 16
    bias_term: true
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
  }
}
layer {
  name: "prelu1_3_1"
  type: "PReLU"
  bottom: "conv1_3_1"
  top: "prelu1_3_1"
}
layer {
  name: "conv1_3_2"
  type: "Convolution"
  bottom: "prelu1_3_1"
  top: "conv1_3_2"
  convolution_param {
    num_output: 64
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise1_3_4"
  type: "Eltwise"
  bottom: "prelu1_2_4"
  bottom: "conv1_3_2"
  top: "eltwise1_3_4"
}
layer {
  name: "prelu1_3_4"
  type: "PReLU"
  bottom: "eltwise1_3_4"
  top: "prelu1_3_4"
}
layer {
  name: "conv1_4_0"
  type: "Convolution"
  bottom: "prelu1_3_4"
  top: "conv1_4_0"
  convolution_param {
    num_output: 16
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu1_4_0"
  type: "PReLU"
  bottom: "conv1_4_0"
  top: "prelu1_4_0"
}
layer {
  name: "conv1_4_1"
  type: "Convolution"
  bottom: "prelu1_4_0"
  top: "conv1_4_1"
  convolution_param {
    num_output: 16
    bias_term: true
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
  }
}
layer {
  name: "prelu1_4_1"
  type: "PReLU"
  bottom: "conv1_4_1"
  top: "prelu1_4_1"
}
layer {
  name: "conv1_4_2"
  type: "Convolution"
  bottom: "prelu1_4_1"
  top: "conv1_4_2"
  convolution_param {
    num_output: 64
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise1_4_4"
  type: "Eltwise"
  bottom: "prelu1_3_4"
  bottom: "conv1_4_2"
  top: "eltwise1_4_4"
}
layer {
  name: "prelu1_4_4"
  type: "PReLU"
  bottom: "eltwise1_4_4"
  top: "prelu1_4_4"
}
layer {
  name: "conv2_0_0"
  type: "Convolution"
  bottom: "prelu1_4_4"
  top: "conv2_0_0"
  convolution_param {
    num_output: 32
    bias_term: true
    kernel_size: 2
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu2_0_0"
  type: "PReLU"
  bottom: "conv2_0_0"
  top: "prelu2_0_0"
}
layer {
  name: "conv2_0_1"
  type: "Convolution"
  bottom: "prelu2_0_0"
  top: "conv2_0_1"
  convolution_param {
    num_output: 32
    bias_term: true
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
  }
}
layer {
  name: "prelu2_0_1"
  type: "PReLU"
  bottom: "conv2_0_1"
  top: "prelu2_0_1"
}
layer {
  name: "conv2_0_2"
  type: "Convolution"
  bottom: "prelu2_0_1"
  top: "conv2_0_2"
  convolution_param {
    num_output: 128
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "pool2_0_4"
  type: "Pooling"
  bottom: "prelu1_4_4"
  top: "pool2_0_4"
  top: "pool2_0_4_mask"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv2_0_4"
  type: "Convolution"
  bottom: "pool2_0_4"
  top: "conv2_0_4"
  convolution_param {
    num_output: 128
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise2_0_4"
  type: "Eltwise"
  bottom: "conv2_0_4"
  bottom: "conv2_0_2"
  top: "eltwise2_0_4"
}
layer {
  name: "prelu2_0_4"
  type: "PReLU"
  bottom: "eltwise2_0_4"
  top: "prelu2_0_4"
}
layer {
  name: "conv2_1_0"
  type: "Convolution"
  bottom: "prelu2_0_4"
  top: "conv2_1_0"
  convolution_param {
    num_output: 32
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu2_1_0"
  type: "PReLU"
  bottom: "conv2_1_0"
  top: "prelu2_1_0"
}
layer {
  name: "conv2_1_1"
  type: "Convolution"
  bottom: "prelu2_1_0"
  top: "conv2_1_1"
  convolution_param {
    num_output: 32
    bias_term: true
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
  }
}
layer {
  name: "prelu2_1_1"
  type: "PReLU"
  bottom: "conv2_1_1"
  top: "prelu2_1_1"
}
layer {
  name: "conv2_1_2"
  type: "Convolution"
  bottom: "prelu2_1_1"
  top: "conv2_1_2"
  convolution_param {
    num_output: 128
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise2_1_4"
  type: "Eltwise"
  bottom: "prelu2_0_4"
  bottom: "conv2_1_2"
  top: "eltwise2_1_4"
}
layer {
  name: "prelu2_1_4"
  type: "PReLU"
  bottom: "eltwise2_1_4"
  top: "prelu2_1_4"
}
layer {
  name: "conv2_2_0"
  type: "Convolution"
  bottom: "prelu2_1_4"
  top: "conv2_2_0"
  convolution_param {
    num_output: 32
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu2_2_0"
  type: "PReLU"
  bottom: "conv2_2_0"
  top: "prelu2_2_0"
}
layer {
  name: "conv2_2_1"
  type: "Convolution"
  bottom: "prelu2_2_0"
  top: "conv2_2_1"
  convolution_param {
    num_output: 32
    bias_term: true
    pad: 2
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
    dilation: 2
  }
}
layer {
  name: "prelu2_2_1"
  type: "PReLU"
  bottom: "conv2_2_1"
  top: "prelu2_2_1"
}
layer {
  name: "conv2_2_2"
  type: "Convolution"
  bottom: "prelu2_2_1"
  top: "conv2_2_2"
  convolution_param {
    num_output: 128
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise2_2_4"
  type: "Eltwise"
  bottom: "prelu2_1_4"
  bottom: "conv2_2_2"
  top: "eltwise2_2_4"
}
layer {
  name: "prelu2_2_4"
  type: "PReLU"
  bottom: "eltwise2_2_4"
  top: "prelu2_2_4"
}
layer {
  name: "conv2_3_0"
  type: "Convolution"
  bottom: "prelu2_2_4"
  top: "conv2_3_0"
  convolution_param {
    num_output: 32
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu2_3_0"
  type: "PReLU"
  bottom: "conv2_3_0"
  top: "prelu2_3_0"
}
layer {
  name: "conv2_3_1_a"
  type: "Convolution"
  bottom: "prelu2_3_0"
  top: "conv2_3_1_a"
  convolution_param {
    num_output: 32
    bias_term: true
    pad: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
    kernel_h: 5
    kernel_w: 1
  }
}
layer {
  name: "conv2_3_1"
  type: "Convolution"
  bottom: "conv2_3_1_a"
  top: "conv2_3_1"
  convolution_param {
    num_output: 32
    bias_term: true
    pad: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    kernel_h: 1
    kernel_w: 5
  }
}
layer {
  name: "prelu2_3_1"
  type: "PReLU"
  bottom: "conv2_3_1"
  top: "prelu2_3_1"
}
layer {
  name: "conv2_3_2"
  type: "Convolution"
  bottom: "prelu2_3_1"
  top: "conv2_3_2"
  convolution_param {
    num_output: 128
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise2_3_4"
  type: "Eltwise"
  bottom: "prelu2_2_4"
  bottom: "conv2_3_2"
  top: "eltwise2_3_4"
}
layer {
  name: "prelu2_3_4"
  type: "PReLU"
  bottom: "eltwise2_3_4"
  top: "prelu2_3_4"
}
layer {
  name: "conv2_4_0"
  type: "Convolution"
  bottom: "prelu2_3_4"
  top: "conv2_4_0"
  convolution_param {
    num_output: 32
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu2_4_0"
  type: "PReLU"
  bottom: "conv2_4_0"
  top: "prelu2_4_0"
}
layer {
  name: "conv2_4_1"
  type: "Convolution"
  bottom: "prelu2_4_0"
  top: "conv2_4_1"
  convolution_param {
    num_output: 32
    bias_term: true
    pad: 4
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
    dilation: 4
  }
}
layer {
  name: "prelu2_4_1"
  type: "PReLU"
  bottom: "conv2_4_1"
  top: "prelu2_4_1"
}
layer {
  name: "conv2_4_2"
  type: "Convolution"
  bottom: "prelu2_4_1"
  top: "conv2_4_2"
  convolution_param {
    num_output: 128
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise2_4_4"
  type: "Eltwise"
  bottom: "prelu2_3_4"
  bottom: "conv2_4_2"
  top: "eltwise2_4_4"
}
layer {
  name: "prelu2_4_4"
  type: "PReLU"
  bottom: "eltwise2_4_4"
  top: "prelu2_4_4"
}
layer {
  name: "conv2_5_0"
  type: "Convolution"
  bottom: "prelu2_4_4"
  top: "conv2_5_0"
  convolution_param {
    num_output: 32
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu2_5_0"
  type: "PReLU"
  bottom: "conv2_5_0"
  top: "prelu2_5_0"
}
layer {
  name: "conv2_5_1"
  type: "Convolution"
  bottom: "prelu2_5_0"
  top: "conv2_5_1"
  convolution_param {
    num_output: 32
    bias_term: true
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
  }
}
layer {
  name: "prelu2_5_1"
  type: "PReLU"
  bottom: "conv2_5_1"
  top: "prelu2_5_1"
}
layer {
  name: "conv2_5_2"
  type: "Convolution"
  bottom: "prelu2_5_1"
  top: "conv2_5_2"
  convolution_param {
    num_output: 128
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise2_5_4"
  type: "Eltwise"
  bottom: "prelu2_4_4"
  bottom: "conv2_5_2"
  top: "eltwise2_5_4"
}
layer {
  name: "prelu2_5_4"
  type: "PReLU"
  bottom: "eltwise2_5_4"
  top: "prelu2_5_4"
}
layer {
  name: "conv2_6_0"
  type: "Convolution"
  bottom: "prelu2_5_4"
  top: "conv2_6_0"
  convolution_param {
    num_output: 32
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu2_6_0"
  type: "PReLU"
  bottom: "conv2_6_0"
  top: "prelu2_6_0"
}
layer {
  name: "conv2_6_1"
  type: "Convolution"
  bottom: "prelu2_6_0"
  top: "conv2_6_1"
  convolution_param {
    num_output: 32
    bias_term: true
    pad: 8
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
    dilation: 8
  }
}
layer {
  name: "prelu2_6_1"
  type: "PReLU"
  bottom: "conv2_6_1"
  top: "prelu2_6_1"
}
layer {
  name: "conv2_6_2"
  type: "Convolution"
  bottom: "prelu2_6_1"
  top: "conv2_6_2"
  convolution_param {
    num_output: 128
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise2_6_4"
  type: "Eltwise"
  bottom: "prelu2_5_4"
  bottom: "conv2_6_2"
  top: "eltwise2_6_4"
}
layer {
  name: "prelu2_6_4"
  type: "PReLU"
  bottom: "eltwise2_6_4"
  top: "prelu2_6_4"
}
layer {
  name: "conv2_7_0"
  type: "Convolution"
  bottom: "prelu2_6_4"
  top: "conv2_7_0"
  convolution_param {
    num_output: 32
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu2_7_0"
  type: "PReLU"
  bottom: "conv2_7_0"
  top: "prelu2_7_0"
}
layer {
  name: "conv2_7_1_a"
  type: "Convolution"
  bottom: "prelu2_7_0"
  top: "conv2_7_1_a"
  convolution_param {
    num_output: 32
    bias_term: true
    pad: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
    kernel_h: 5
    kernel_w: 1
  }
}
layer {
  name: "conv2_7_1"
  type: "Convolution"
  bottom: "conv2_7_1_a"
  top: "conv2_7_1"
  convolution_param {
    num_output: 32
    bias_term: true
    pad: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    kernel_h: 1
    kernel_w: 5
  }
}
layer {
  name: "prelu2_7_1"
  type: "PReLU"
  bottom: "conv2_7_1"
  top: "prelu2_7_1"
}
layer {
  name: "conv2_7_2"
  type: "Convolution"
  bottom: "prelu2_7_1"
  top: "conv2_7_2"
  convolution_param {
    num_output: 128
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise2_7_4"
  type: "Eltwise"
  bottom: "prelu2_6_4"
  bottom: "conv2_7_2"
  top: "eltwise2_7_4"
}
layer {
  name: "prelu2_7_4"
  type: "PReLU"
  bottom: "eltwise2_7_4"
  top: "prelu2_7_4"
}
layer {
  name: "conv2_8_0"
  type: "Convolution"
  bottom: "prelu2_7_4"
  top: "conv2_8_0"
  convolution_param {
    num_output: 32
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu2_8_0"
  type: "PReLU"
  bottom: "conv2_8_0"
  top: "prelu2_8_0"
}
layer {
  name: "conv2_8_1"
  type: "Convolution"
  bottom: "prelu2_8_0"
  top: "conv2_8_1"
  convolution_param {
    num_output: 32
    bias_term: true
    pad: 16
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
    dilation: 16
  }
}
layer {
  name: "prelu2_8_1"
  type: "PReLU"
  bottom: "conv2_8_1"
  top: "prelu2_8_1"
}
layer {
  name: "conv2_8_2"
  type: "Convolution"
  bottom: "prelu2_8_1"
  top: "conv2_8_2"
  convolution_param {
    num_output: 128
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise2_8_4"
  type: "Eltwise"
  bottom: "prelu2_7_4"
  bottom: "conv2_8_2"
  top: "eltwise2_8_4"
}
layer {
  name: "prelu2_8_4"
  type: "PReLU"
  bottom: "eltwise2_8_4"
  top: "prelu2_8_4"
}
layer {
  name: "conv3_1_0"
  type: "Convolution"
  bottom: "prelu2_8_4"
  top: "conv3_1_0"
  convolution_param {
    num_output: 32
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu3_1_0"
  type: "PReLU"
  bottom: "conv3_1_0"
  top: "prelu3_1_0"
}
layer {
  name: "conv3_1_1"
  type: "Convolution"
  bottom: "prelu3_1_0"
  top: "conv3_1_1"
  convolution_param {
    num_output: 32
    bias_term: true
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
  }
}
layer {
  name: "prelu3_1_1"
  type: "PReLU"
  bottom: "conv3_1_1"
  top: "prelu3_1_1"
}
layer {
  name: "conv3_1_2"
  type: "Convolution"
  bottom: "prelu3_1_1"
  top: "conv3_1_2"
  convolution_param {
    num_output: 128
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise3_1_4"
  type: "Eltwise"
  bottom: "prelu2_8_4"
  bottom: "conv3_1_2"
  top: "eltwise3_1_4"
}
layer {
  name: "prelu3_1_4"
  type: "PReLU"
  bottom: "eltwise3_1_4"
  top: "prelu3_1_4"
}
layer {
  name: "conv3_2_0"
  type: "Convolution"
  bottom: "prelu3_1_4"
  top: "conv3_2_0"
  convolution_param {
    num_output: 32
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu3_2_0"
  type: "PReLU"
  bottom: "conv3_2_0"
  top: "prelu3_2_0"
}
layer {
  name: "conv3_2_1"
  type: "Convolution"
  bottom: "prelu3_2_0"
  top: "conv3_2_1"
  convolution_param {
    num_output: 32
    bias_term: true
    pad: 2
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
    dilation: 2
  }
}
layer {
  name: "prelu3_2_1"
  type: "PReLU"
  bottom: "conv3_2_1"
  top: "prelu3_2_1"
}
layer {
  name: "conv3_2_2"
  type: "Convolution"
  bottom: "prelu3_2_1"
  top: "conv3_2_2"
  convolution_param {
    num_output: 128
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise3_2_4"
  type: "Eltwise"
  bottom: "prelu3_1_4"
  bottom: "conv3_2_2"
  top: "eltwise3_2_4"
}
layer {
  name: "prelu3_2_4"
  type: "PReLU"
  bottom: "eltwise3_2_4"
  top: "prelu3_2_4"
}
layer {
  name: "conv3_3_0"
  type: "Convolution"
  bottom: "prelu3_2_4"
  top: "conv3_3_0"
  convolution_param {
    num_output: 32
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu3_3_0"
  type: "PReLU"
  bottom: "conv3_3_0"
  top: "prelu3_3_0"
}
layer {
  name: "conv3_3_1_a"
  type: "Convolution"
  bottom: "prelu3_3_0"
  top: "conv3_3_1_a"
  convolution_param {
    num_output: 32
    bias_term: true
    pad: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
    kernel_h: 5
    kernel_w: 1
  }
}
layer {
  name: "conv3_3_1"
  type: "Convolution"
  bottom: "conv3_3_1_a"
  top: "conv3_3_1"
  convolution_param {
    num_output: 32
    bias_term: true
    pad: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    kernel_h: 1
    kernel_w: 5
  }
}
layer {
  name: "prelu3_3_1"
  type: "PReLU"
  bottom: "conv3_3_1"
  top: "prelu3_3_1"
}
layer {
  name: "conv3_3_2"
  type: "Convolution"
  bottom: "prelu3_3_1"
  top: "conv3_3_2"
  convolution_param {
    num_output: 128
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise3_3_4"
  type: "Eltwise"
  bottom: "prelu3_2_4"
  bottom: "conv3_3_2"
  top: "eltwise3_3_4"
}
layer {
  name: "prelu3_3_4"
  type: "PReLU"
  bottom: "eltwise3_3_4"
  top: "prelu3_3_4"
}
layer {
  name: "conv3_4_0"
  type: "Convolution"
  bottom: "prelu3_3_4"
  top: "conv3_4_0"
  convolution_param {
    num_output: 32
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu3_4_0"
  type: "PReLU"
  bottom: "conv3_4_0"
  top: "prelu3_4_0"
}
layer {
  name: "conv3_4_1"
  type: "Convolution"
  bottom: "prelu3_4_0"
  top: "conv3_4_1"
  convolution_param {
    num_output: 32
    bias_term: true
    pad: 4
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
    dilation: 4
  }
}
layer {
  name: "prelu3_4_1"
  type: "PReLU"
  bottom: "conv3_4_1"
  top: "prelu3_4_1"
}
layer {
  name: "conv3_4_2"
  type: "Convolution"
  bottom: "prelu3_4_1"
  top: "conv3_4_2"
  convolution_param {
    num_output: 128
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise3_4_4"
  type: "Eltwise"
  bottom: "prelu3_3_4"
  bottom: "conv3_4_2"
  top: "eltwise3_4_4"
}
layer {
  name: "prelu3_4_4"
  type: "PReLU"
  bottom: "eltwise3_4_4"
  top: "prelu3_4_4"
}
layer {
  name: "conv3_5_0"
  type: "Convolution"
  bottom: "prelu3_4_4"
  top: "conv3_5_0"
  convolution_param {
    num_output: 32
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu3_5_0"
  type: "PReLU"
  bottom: "conv3_5_0"
  top: "prelu3_5_0"
}
layer {
  name: "conv3_5_1"
  type: "Convolution"
  bottom: "prelu3_5_0"
  top: "conv3_5_1"
  convolution_param {
    num_output: 32
    bias_term: true
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
  }
}
layer {
  name: "prelu3_5_1"
  type: "PReLU"
  bottom: "conv3_5_1"
  top: "prelu3_5_1"
}
layer {
  name: "conv3_5_2"
  type: "Convolution"
  bottom: "prelu3_5_1"
  top: "conv3_5_2"
  convolution_param {
    num_output: 128
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise3_5_4"
  type: "Eltwise"
  bottom: "prelu3_4_4"
  bottom: "conv3_5_2"
  top: "eltwise3_5_4"
}
layer {
  name: "prelu3_5_4"
  type: "PReLU"
  bottom: "eltwise3_5_4"
  top: "prelu3_5_4"
}
layer {
  name: "conv3_6_0"
  type: "Convolution"
  bottom: "prelu3_5_4"
  top: "conv3_6_0"
  convolution_param {
    num_output: 32
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu3_6_0"
  type: "PReLU"
  bottom: "conv3_6_0"
  top: "prelu3_6_0"
}
layer {
  name: "conv3_6_1"
  type: "Convolution"
  bottom: "prelu3_6_0"
  top: "conv3_6_1"
  convolution_param {
    num_output: 32
    bias_term: true
    pad: 8
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
    dilation: 8
  }
}
layer {
  name: "prelu3_6_1"
  type: "PReLU"
  bottom: "conv3_6_1"
  top: "prelu3_6_1"
}
layer {
  name: "conv3_6_2"
  type: "Convolution"
  bottom: "prelu3_6_1"
  top: "conv3_6_2"
  convolution_param {
    num_output: 128
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise3_6_4"
  type: "Eltwise"
  bottom: "prelu3_5_4"
  bottom: "conv3_6_2"
  top: "eltwise3_6_4"
}
layer {
  name: "prelu3_6_4"
  type: "PReLU"
  bottom: "eltwise3_6_4"
  top: "prelu3_6_4"
}
layer {
  name: "conv3_7_0"
  type: "Convolution"
  bottom: "prelu3_6_4"
  top: "conv3_7_0"
  convolution_param {
    num_output: 32
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu3_7_0"
  type: "PReLU"
  bottom: "conv3_7_0"
  top: "prelu3_7_0"
}
layer {
  name: "conv3_7_1_a"
  type: "Convolution"
  bottom: "prelu3_7_0"
  top: "conv3_7_1_a"
  convolution_param {
    num_output: 32
    bias_term: true
    pad: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
    kernel_h: 5
    kernel_w: 1
  }
}
layer {
  name: "conv3_7_1"
  type: "Convolution"
  bottom: "conv3_7_1_a"
  top: "conv3_7_1"
  convolution_param {
    num_output: 32
    bias_term: true
    pad: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    kernel_h: 1
    kernel_w: 5
  }
}
layer {
  name: "prelu3_7_1"
  type: "PReLU"
  bottom: "conv3_7_1"
  top: "prelu3_7_1"
}
layer {
  name: "conv3_7_2"
  type: "Convolution"
  bottom: "prelu3_7_1"
  top: "conv3_7_2"
  convolution_param {
    num_output: 128
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise3_7_4"
  type: "Eltwise"
  bottom: "prelu3_6_4"
  bottom: "conv3_7_2"
  top: "eltwise3_7_4"
}
layer {
  name: "prelu3_7_4"
  type: "PReLU"
  bottom: "eltwise3_7_4"
  top: "prelu3_7_4"
}
layer {
  name: "conv3_8_0"
  type: "Convolution"
  bottom: "prelu3_7_4"
  top: "conv3_8_0"
  convolution_param {
    num_output: 32
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu3_8_0"
  type: "PReLU"
  bottom: "conv3_8_0"
  top: "prelu3_8_0"
}
layer {
  name: "conv3_8_1"
  type: "Convolution"
  bottom: "prelu3_8_0"
  top: "conv3_8_1"
  convolution_param {
    num_output: 32
    bias_term: true
    pad: 16
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
    dilation: 16
  }
}
layer {
  name: "prelu3_8_1"
  type: "PReLU"
  bottom: "conv3_8_1"
  top: "prelu3_8_1"
}
layer {
  name: "conv3_8_2"
  type: "Convolution"
  bottom: "prelu3_8_1"
  top: "conv3_8_2"
  convolution_param {
    num_output: 128
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise3_8_4"
  type: "Eltwise"
  bottom: "prelu3_7_4"
  bottom: "conv3_8_2"
  top: "eltwise3_8_4"
}
layer {
  name: "prelu3_8_4"
  type: "PReLU"
  bottom: "eltwise3_8_4"
  top: "prelu3_8_4"
}
layer {
  name: "conv4_0_0"
  type: "Convolution"
  bottom: "prelu3_8_4"
  top: "conv4_0_0"
  convolution_param {
    num_output: 16
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu4_0_0"
  type: "ReLU"
  bottom: "conv4_0_0"
  top: "prelu4_0_0"
}
layer {
  name: "deconv4_0_1"
  type: "Deconvolution"
  bottom: "prelu4_0_0"
  top: "deconv4_0_1"
  convolution_param {
    num_output: 16
    bias_term: true
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "bn4_0_1"
  type: "BN"
  bottom: "deconv4_0_1"
  top: "bn4_0_1"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 1.0
    decay_mult: 0.0
  }
  bn_param {
    scale_filler {
      type: "constant"
      value: 1.0
    }
    shift_filler {
      type: "constant"
      value: 0.001
    }
    bn_mode: INFERENCE
  }
}
layer {
  name: "prelu4_0_1"
  type: "ReLU"
  bottom: "bn4_0_1"
  top: "prelu4_0_1"
}
layer {
  name: "conv4_0_2"
  type: "Convolution"
  bottom: "prelu4_0_1"
  top: "conv4_0_2"
  convolution_param {
    num_output: 64
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "conv4_0_4"
  type: "Convolution"
  bottom: "prelu3_8_4"
  top: "conv4_0_4"
  convolution_param {
    num_output: 64
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "upsample4_0_4"
  type: "Upsample"
  bottom: "conv4_0_4"
  bottom: "pool2_0_4_mask"
  top: "upsample4_0_4"
  upsample_param {
    scale: 2
  }
}
layer {
  name: "eltwise4_0_4"
  type: "Eltwise"
  bottom: "upsample4_0_4"
  bottom: "conv4_0_2"
  top: "eltwise4_0_4"
}
layer {
  name: "prelu4_0_4"
  type: "ReLU"
  bottom: "eltwise4_0_4"
  top: "prelu4_0_4"
}
layer {
  name: "conv4_1_0"
  type: "Convolution"
  bottom: "prelu4_0_4"
  top: "conv4_1_0"
  convolution_param {
    num_output: 16
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu4_1_0"
  type: "ReLU"
  bottom: "conv4_1_0"
  top: "prelu4_1_0"
}
layer {
  name: "conv4_1_1"
  type: "Convolution"
  bottom: "prelu4_1_0"
  top: "conv4_1_1"
  convolution_param {
    num_output: 16
    bias_term: true
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
  }
}
layer {
  name: "prelu4_1_1"
  type: "ReLU"
  bottom: "conv4_1_1"
  top: "prelu4_1_1"
}
layer {
  name: "conv4_1_2"
  type: "Convolution"
  bottom: "prelu4_1_1"
  top: "conv4_1_2"
  convolution_param {
    num_output: 64
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise4_1_4"
  type: "Eltwise"
  bottom: "prelu4_0_4"
  bottom: "conv4_1_2"
  top: "eltwise4_1_4"
}
layer {
  name: "prelu4_1_4"
  type: "ReLU"
  bottom: "eltwise4_1_4"
  top: "prelu4_1_4"
}
layer {
  name: "conv4_2_0"
  type: "Convolution"
  bottom: "prelu4_1_4"
  top: "conv4_2_0"
  convolution_param {
    num_output: 16
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu4_2_0"
  type: "ReLU"
  bottom: "conv4_2_0"
  top: "prelu4_2_0"
}
layer {
  name: "conv4_2_1"
  type: "Convolution"
  bottom: "prelu4_2_0"
  top: "conv4_2_1"
  convolution_param {
    num_output: 16
    bias_term: true
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
  }
}
layer {
  name: "prelu4_2_1"
  type: "ReLU"
  bottom: "conv4_2_1"
  top: "prelu4_2_1"
}
layer {
  name: "conv4_2_2"
  type: "Convolution"
  bottom: "prelu4_2_1"
  top: "conv4_2_2"
  convolution_param {
    num_output: 64
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise4_2_4"
  type: "Eltwise"
  bottom: "prelu4_1_4"
  bottom: "conv4_2_2"
  top: "eltwise4_2_4"
}
layer {
  name: "prelu4_2_4"
  type: "ReLU"
  bottom: "eltwise4_2_4"
  top: "prelu4_2_4"
}
layer {
  name: "conv5_0_0"
  type: "Convolution"
  bottom: "prelu4_2_4"
  top: "conv5_0_0"
  convolution_param {
    num_output: 4
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu5_0_0"
  type: "ReLU"
  bottom: "conv5_0_0"
  top: "prelu5_0_0"
}
layer {
  name: "deconv5_0_1"
  type: "Deconvolution"
  bottom: "prelu5_0_0"
  top: "deconv5_0_1"
  convolution_param {
    num_output: 4
    bias_term: true
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "bn5_0_1"
  type: "BN"
  bottom: "deconv5_0_1"
  top: "bn5_0_1"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 1.0
    decay_mult: 0.0
  }
  bn_param {
    scale_filler {
      type: "constant"
      value: 1.0
    }
    shift_filler {
      type: "constant"
      value: 0.001
    }
    bn_mode: INFERENCE
  }
}
layer {
  name: "prelu5_0_1"
  type: "ReLU"
  bottom: "bn5_0_1"
  top: "prelu5_0_1"
}
layer {
  name: "conv5_0_2"
  type: "Convolution"
  bottom: "prelu5_0_1"
  top: "conv5_0_2"
  convolution_param {
    num_output: 16
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "conv5_0_4"
  type: "Convolution"
  bottom: "prelu4_2_4"
  top: "conv5_0_4"
  convolution_param {
    num_output: 16
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "upsample5_0_4"
  type: "Upsample"
  bottom: "conv5_0_4"
  bottom: "pool1_0_4_mask"
  top: "upsample5_0_4"
  upsample_param {
    scale: 2
  }
}
layer {
  name: "eltwise5_0_4"
  type: "Eltwise"
  bottom: "upsample5_0_4"
  bottom: "conv5_0_2"
  top: "eltwise5_0_4"
}
layer {
  name: "prelu5_0_4"
  type: "ReLU"
  bottom: "eltwise5_0_4"
  top: "prelu5_0_4"
}
layer {
  name: "conv5_1_0"
  type: "Convolution"
  bottom: "prelu5_0_4"
  top: "conv5_1_0"
  convolution_param {
    num_output: 4
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "prelu5_1_0"
  type: "ReLU"
  bottom: "conv5_1_0"
  top: "prelu5_1_0"
}
layer {
  name: "conv5_1_1"
  type: "Convolution"
  bottom: "prelu5_1_0"
  top: "conv5_1_1"
  convolution_param {
    num_output: 4
    bias_term: true
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
  }
}
layer {
  name: "prelu5_1_1"
  type: "ReLU"
  bottom: "conv5_1_1"
  top: "prelu5_1_1"
}
layer {
  name: "conv5_1_2"
  type: "Convolution"
  bottom: "prelu5_1_1"
  top: "conv5_1_2"
  convolution_param {
    num_output: 16
    bias_term: true
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "eltwise5_1_4"
  type: "Eltwise"
  bottom: "prelu5_0_4"
  bottom: "conv5_1_2"
  top: "eltwise5_1_4"
}
layer {
  name: "prelu5_1_4"
  type: "ReLU"
  bottom: "eltwise5_1_4"
  top: "prelu5_1_4"
}
layer {
  name: "deconv6_0_0"
  type: "Deconvolution"
  bottom: "prelu5_1_4"
  top: "deconv6_0_0"
  convolution_param {
    num_output: 19
    bias_term: true
    kernel_size: 2
    stride: 2
  }
}
