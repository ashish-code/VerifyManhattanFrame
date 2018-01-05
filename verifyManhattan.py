# Copyright (C) 2017 Ashish Gupta
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# author:   Ashish Gupta
# email:    ashish.gupta@rit.edu
# version:  0.1.0

from __future__ import print_function

# import libraries
import os
import pickle
from skimage import io
from skimage import color
import numpy as np
import argparse
from skimage import transform

# change global variable for desired set of valid image file extentions
valid_image_file_extentions = ['jpg', 'png', 'pgm', 'bmp']


def is_image_manhattan(image_url, output_folder=None, verbose=False):
    """
    Analyzes image to verify existence of Manhattan Frame Assumption.
    We assume that an image with a Manhattan Frame has sufficient pixels in the image associated with
    3 mutually orthogonal planes. In comparison, a non-manhattan frame image is predominantly 2D, which
    has fewer than 3 normal planes.
    Caveats: The estimation of normal at each pixel is an open research problem. The existence of objects
    in the image scene with complex geometric shapes introduces error in estimation of normal planes.
    Therefore the efficacy of this program in detecting Manhattan Frames is limited, pending availability
    of a properly annotated dataset with sufficient examples of images with and without manhattan frames.

    :param verbose: print to console during execution : boolean
    :param output_folder: directory to store results in : string
    :param image_url: path to image file : string
    :return: manhattan frame existence : boolean
    """

    """ load the pre-trained classifier model stored in pickle file """
    pkl_filename = './qda_model.pkl'
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    """ temporary data is stored to local directory to interact with lua code """
    temp_result_file = './temp.jpg'
    temp_file_list = './temp.txt'

    temp_input_file = './temp_input.jpg'
    img = io.imread(image_url)
    try:
        img_resized = transform.resize(img, (256, 256), mode='reflect')
    except:
        img_resized = img

    try:
        io.imsave(temp_input_file, img_resized)
    except:
        return 0

    with open(temp_file_list, 'w') as tf:
        tf.write(temp_input_file)

    try:
        _t = os.system("~/torch/install/bin/th norm_one_image.lua")
        image = io.imread(temp_result_file)                         # read the image with normals at pixels
        if len(image.shape) == 2:
            image = color.gray2rgb(image)
        _total_pixels = image.shape[0] * image.shape[1]
        _hist = np.histogram(image[:, :, 2], bins=256)[0]
        _hist = [h / _total_pixels for h in _hist]
        X = np.zeros((1, 2))
        X[0, :] = _hist[-3:-1]                                      # we are interested in the 3rd ortho plane

        pred = pickle_model.predict(X)[0]
        # pred = int(not pred)                                        # 1-manhattan, 0-non manhattan
        os.remove(temp_result_file)                                 # remove temporary files
    except:
        pred = 0
    finally:
        if verbose:
            print('{},{}\n'.format(image_url, pred))
        """ file ouput for single image is provide as convenience """
        if output_folder:
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            result_file_name = image_url.split('/')[-1].split('.')[0]+'.txt'
            result_file_url = os.path.join(output_folder, result_file_name)
            with open(result_file_url, 'w') as f:
                f.write('{}'.format(pred))
        os.remove(temp_file_list)                                   # remove temporary files
    return pred


def is_image_list_url_manhattan(image_list_url, output_folder='./log/'):
    """
    Analyzes all images in the file containing image paths for existence of Manhattan Frame Assumption
    :param image_list_url: string
    :param output_folder: string
    :return: result file url : string
    """

    with open(image_list_url, 'r') as list_file:
        image_list = list_file.readlines()
        image_list = [image_path.strip() for image_path in image_list if image_path.strip() != '']
        result_path = is_image_list_manhattan(image_list, output_folder)

    # DEBUG:
    print(result_path)


def is_image_list_manhattan(image_list, output_folder='./log/'):
    """

    :param image_list: python list of image paths : array
    :param ouput_folder: destination folder : string
    :return: result path
    """
    default_prediction = 0  # ouput is non-manhattan by default

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    result_file_url = os.path.join(output_folder, 'manhattan_existence.txt')
    # iterate over each image in the image list file
    with open(result_file_url, 'w') as result_file:
        # check if the image has proper image type extention
        for image_url in image_list:
            if image_url.split('.')[-1] not in valid_image_file_extentions:
                result_file.write('{},{}\n'.format(image_url, default_prediction))
                continue
            # check if the image file is readable
            try:
                _image = io.imread(image_url)
                prediction = is_image_manhattan(image_url)
                result_file.write('{},{}\n'.format(image_url, prediction))
            except:
                result_file.write('{},{}\n'.format(image_url, default_prediction))
                continue
    return result_file_url


def is_image_folder_manhattan(image_folder_url, output_folder='./log/'):
    """
    Analyzes all images in the given folder for existence of Manhattan Frame Assumption
    :param image_folder_url: string
    :param output_folder: string
    :return: result file url : string
    """
    # list all the files in the input directory
    image_list = os.listdir(image_folder_url)

    image_list = [os.path.join(image_folder_url,image_path) for image_path in image_list if image_path.split('.')[-1] in valid_image_file_extentions]
    # call method to verify manhattan frame for image list
    result_path = is_image_list_manhattan(image_list, output_folder)

    # DEBUG:
    print(result_path)


def main():
    usage = "usage: %prog [option] arg"
    parser = argparse.ArgumentParser(description="Verify existence of manhattan frame in image")
    parser.add_argument("-t", "--type", dest="type", default='i', help="type of input: i=image, l=list, d=directory.")
    parser.add_argument("-f", "--file", dest="filename", default='./image/pizza.png', help='image file path')
    parser.add_argument("-l", "--list", dest="filelist", default='./image/image_list.txt', help='path to text file with list of image paths')
    parser.add_argument("-d", "--dir", dest="directory", default='./image/', help='directory containing image files')
    parser.add_argument("-r", "--result", dest="result", default='./log/', help='directory where the result is stored.')

    args = parser.parse_args()
    input_type = args.type
    result_folder = args.result
    if input_type == 'i':
        image_url = args.filename
        is_image_manhattan(image_url, result_folder)
        return
    elif input_type == 'l':
        image_list_url = args.filelist
        is_image_list_url_manhattan(image_list_url, result_folder)
        return
    elif input_type == 'd':
        image_folder = args.directory
        is_image_folder_manhattan(image_folder, result_folder)
        return
    else:
        print('Input type (image/list/directory) incorrect. Please try again with -t (i/l/d).')
        return


if __name__ == '__main__':
    main()
