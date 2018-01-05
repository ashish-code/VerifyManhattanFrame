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

## Contact
Ashish Gupta, PhD
ashish.gupta@rit.edu

