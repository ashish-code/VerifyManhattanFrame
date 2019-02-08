# Verfiy the existence of a 'Manhattan Frame Assumption' in an image.

## Installation
This program uses CUDA (8.0) and Torch.
Please install torch using the instructions in: http://torch.ch/docs/getting-started.html
The code assumes that torch is stored in home directory, ie. torch is installed in ~/torch.
The provided installation uses LuaJIT for run .lua files.
After installation of torch, please install required libraries using luarocks.
luarocks install nn
luarocks install  cutorch
luarocks install  cunn
luarocks install  cudnn
luarocks install  nngraph
luarocks install  optim
luarocks install  image

A compressed Torch7 model file is present in ./model/ which needs to be uncompressed (Max. file size issue with GitHub)
Kindly run $tar -xvf ./model/sync_physic_nyufinetune.t7.tar.gz
Ensure you have a "sync_physic_nyufinetune.t7" in ./model/ before continuing.


## Usage:
$python verifyManhattan.py [options] args
Begin with:
$python verifyManhattan.py --help

## Demo:

## Single Image
For a single image: (sample image is provided in ./image/ directory)
1. $python verifyManhattan.py -t i

2. Typical usage for single image (sample image provided for demo)
$python verifyManhattan.py -t i -f ./image/ship.png

## Txt file with image urls
For text file containing list of paths to images
$python verifyManhattan.py -t l -l ./image/image_list.txt

## Directory containing images
For folder containing images
$python verifyManhattan.py -t d -d ./image/


## Output:
The results by default are stored in ./log/ directory in file: ./log/manhattan_existence.txt
The file has comma separated result:
file_path_1,[0/1]
file_path_2,[0/1]
...

If a manhattan frame is present then result is 1. Non existence of manhattan frame is 0.

## Usage 2:
Select the images in ./log/manhattan_existence.txt or <result-folder>/manhattan_existence.txt with associated score of 1.

## Note:
The pre-trained torch7 CNN model (in ./model/) is based on older version of cudaNN (https://developer.nvidia.com/cudnn). The required libcudnn.so.5 file is provided. It should work for cuDNN ver 5.1 for CUDA 8.

## Description

A ‘manhattan world’ is the presence of at least one set of three mutually orthogonal surfaces in an image. Images in the wild can range from having 0 to multiple occurrences of ‘Manhattan Frame’ in it. A ‘Manhattan frame’ is a 3-dimensional geometric structure. The challenge is estimating this 3D structure from a 2D image. This is currently an open research problem under active research. Existing literature in this domain typically focus on problems like indoor navigation, in Robotics, which have the benefit of depth information through stereo or RGB-D cameras, or they utilized structure-from-motion (SfM) methods on a sequence of images to infer 3D information.

In the MediFor task, we deal with a single image from the wild. Consequently, we rely on use of 3D structural primitives to estimate 3D from 2D. Briefly, this is a model trained on 2D image features, where the 3D geometric properties of the image are known. This establishes a correspondence between 2D visual features and their implicit 3D properties, primarily the direction of normal to the surface at each pixel in the image. We utilize this model to compute normal for every pixel in each query image. The normal are clustered in roughly 3 mutually orthogonal groups. For the purposes of identifying ‘Manhattan Frame’, we focus on the smallest of the 3 groups. The histogram of normal within this group is used a discriminative feature. A block diagram of this approach is shown in Figure 1.

<img src="https://github.com/ashish-code/VerifyManhattanFrame/blob/master/manhattan_12072017.png" width="1000">
Figure 1: 'Manhattan World' classifier

The key issue at the point in time is the lack of annotated ‘Non-manhattan Frame’ images towards training the classifier. We have manually scraped the internet for about 100 images of both classes. The trained classifier is shown in Figure 2. Visual inspection of the results does indicate that the approach we have developed is functional. Training images which definitely contain a ‘manhattan frame’ are highly separated from images which are definitely ‘flat’ and do not contain a ‘manhattan frame’. This means we can achieve a much better classification performance than current results by a simple modification of the classifier parameters, without altering the over-all approach.

<img src="https://github.com/ashish-code/VerifyManhattanFrame/blob/master/trained_classifier.png" width="800">
Figure 2: Linear and Quadratic Discriminant Analysis of Manhattan and Non-manhattan frame training images.

## Contact
Ashish Gupta, PhD
ashish.gupta@rit.edu

