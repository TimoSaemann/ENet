# Tutorial on how to train and test ENet on Cityscapes dataset

## Installation

First, please clone the ENet repository by running:

	$ git clone --recursive https://github.com/TimoSaemann/ENet.git

Please compile the modified Caffe framework __caffe-enet__ (It supports all necessary layers for ENet):

	$ cd ENet/caffe-enet
	$ mkdir build && cd build
	$ cmake ..
	$ make all -j8 && make pycaffe

You can also consult the generic [Caffe installation guide](http://caffe.berkeleyvision.org/installation.html) for further help. If you like to compile with __make__, please uncomment the following line in the Makefile.config 

	$ WITH_PYTHON_LAYER := 1 
	
Please make sure that the python layer (spatial_dropout.py) is defined in your PYTHONPATH:

	$ export PYTHONPATH="$CAFFE_PATH/python:$PYTHONPATH"

## Preparation

Please download the fine labeled Cityscapes dataset __leftImg8bit_trainvaltest.zip (11GB)__ and the corresponding ground truth __gtFine_trainvaltest.zip (241MB)__ from the [Cityscapes website](https://www.cityscapes-dataset.com). In addition, clone the __cityscapesScripts__ repository: 

	$ git clone https://github.com/mcordts/cityscapesScripts.git
	
After that, run the `/preparation/createTrainIdLabelImags.py` script, to convert annotations in polygonal format to png images with label IDs, where pixels encode "train IDs" (that you can define in `labels.py`). Since we use the default 19 classes, you do not need to change anything in the `labels.py` script.
The input data layer which is used requires a text file of white-space separated paths to the images and the corresponding ground truth.
For this reason, please modify `ENet/dataset/train_fine_cityscapes.txt` to your absolute path of the data.

Next, change __caffe_root__ to the absolute path of __caffe-enet__ in the following scripts:
 - ENet/scripts/BN-absorber-enet.py
 - ENet/scripts/compute_bn_statistics.py
 - ENet/scripts/create_enet_prototxt.py
 - ENet/scripts/test_segmentation.py

Furthermore, change all relative paths to the absolute path in both solver files:
 - ENet/prototxts/enet_solver_encoder.prototxt
 - ENet/prototxts/enet_solver_encoder_decoder.prototxt

## Training ENet 

The training of ENet is performed in two stages: 
 - training the encoder architecture
 - training the encoder and decoder architecture jointly

### Let's start with the encoder training:

First, create the prototxt file `enet_train_encoder.prototxt` by running:

	$ python create_enet_prototxt.py --source ENet/dataset/train_fine_cityscapes.txt --mode train_encoder
	
This prototxt file includes the encoder architecture of ENet with some default settings you can customize according your needs. For example, the input images are resized to 1024x512, because of GPU memory restrictions. For more details have a look in the prototxt file or in the python file.

The next step is optional:
To improve the quality of ENet predictions in small classes (traffic sign, pole, etc.), you can add __class_weighting__ to the __SoftmaxWithLoss__ layer. 

	$ python calculate_class_weighting.py --source ENet/dataset/train_fine_cityscapes.txt --num_classes 19
	
Copy the __class_weightings__ from the terminal in the `enet_train_encoder.prototxt` and `enet_train_encoder_decoder.prototxt` under __weight_by_label_freqs__ and set this flag from __false__ to __true__. 
 
Now you are ready to start the training:

	$ ENet/caffe-enet/build/tools/caffe train -solver /ENet/prototxts/enet_solver_encoder.prototxt

If the GPU memory is not enough (error == cudasuccess), reduce the batch size in `ENet/prototxt/enet_train_encoder_decoder.prototxt`.

### After the encoder-training is finished you can continue with the training of encoder + decoder:

First create the `enet_train_encoder_decoder.prototxt` by running:

	$ python create_enet_prototxt.py --source ENet/dataset/train_fine_cityscapes.txt --mode train_encoder_decoder

Start the training of the encoder + decoder and use the pretrained weights as initialization of the encoder:

	$ ENet/caffe-enet/build/tools/caffe train -solver ENet/prototxts/enet_solver_encoder_decoder.prototxt -weights ENet/weights/snapshots_encoder/NAME.caffemodel

Replace the place holder __NAME__ to the name of your weights.

After about 100 epochs you should see it converge (2975 images * 100 epochs / batch size = ~75k iterations). You should be looking for greater than 80 % training accuracy.

## Testing ENet

The Batch Normalisation layers [1] in ENet shift the input feature maps according to their mean and variance
statistics for each mini batch during training. At test time we must use the statistics for the entire dataset.
For this reason run __compute_bn_statistics.py__ to calculate the new weights called __test_weights.caffemodel__:

	$ python compute_bn_statistics.py 	ENet/prototxt/enet_train_encoder_decoder.prototxt \
						ENet/weights/snapshots_decoder/NAME.caffemodel \
						ENet/weights/weights_bn/ 

The script saves the __bn-weights__ in the output directory `ENet/weights/weights_bn/bn_weights.caffemodel`.

For inference batch normalization and dropout layer can be merged into convolutional kernels, to
speed up the network. You can do this by running:

	$ python BN-absorber-enet.py 	--model ENet/prototxts/enet_deploy.prototxt \
					--weights ENet/weights/bn_weights.caffemodel \
					--out_dir ENet/final_model_weigths/

It also deletes the corresponding batch normalization and dropout layers from the prototxt file. The final model (prototxt file) and weights are saved in the folder __final_model_and_weights__. 

### Visualize the prediction with python

You can visualize the prediction of ENet by running:

	$ python test_segmentation.py 	--model ENet/final_model_weigths/bn_conv_merged_model.prototxt \
					--weights ENet/final_model_weigths/bn_conv_merged_weights.caffemodel \
					--colours /ENet/scripts/cityscapes19.png \
					--input_image ENet/example_image/munich_000000_000019_leftImg8bit.png \
					--out_dir /ENet/example_image/ 


### Visualize the prediction with C++

If you like to visualize the prediction of ENet with C++ code:

	$ cd ENet/caffe-enet/build/examples/ENet_with_C++
	$ ./test_segmentation 	ENet/final_model_weights/bn_conv_merged_model.prototxt \
				ENet/final_model_weights/bn_conv_merged_weights.caffemodel \
				ENet/example_image/munich_000000_000019_leftImg8bit.png \
				ENet/scripts/cityscapes19.png



### Kick-start

If you just want to test the quality with the trained weights in the [enet-weights-zoo](https://github.com/TimoSaemann/ENet/tree/master/enet_weights_zoo), run:
	
	$ sh cityscapes_weights.sh
	
to download the weights and then:

	$ python test_segmentation.py 	--model ENet/prototxts/enet_deploy_final.prototxt \						
					--weights ENet/enet_weights_zoo/cityscapes_weights.caffemodel \
					--colours ENet/scripts/cityscapes19.png \
					--input_image ENet/example_image/munich_000000_000019_leftImg8bit.png \
					--out_dir ENet/example_image/

You should be able to reproduce the following example:

![Alt text](/example_image/image_enet.PNG?raw=true "image_enet")

### Measurement of execution time

To measure ENet execution time layer-by-layer run:

	$ ENet/caffe-enet/build/tools/caffe time -model ENet/prototxts/enet_deploy_final.prototxt -gpu 0 -iterations 100


[1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing
internal covariate shift." arXiv preprint arXiv:1502.03167 (2015)
