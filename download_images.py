"""
Download images listed in a file
The image file names is the image-id on the medifor.rankone.io database.
This program parses the image list file, formulates the web query, and handles data request
and local storage.
"""

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

import os
import errno
import sys
import argparse
import json
import requests
import multiprocessing
from urllib import request
from skimage import io
import csv

url_prefix = "https://s3.amazonaws.com/medifor/images/"


def prune_image_list_file(args):
    """
    Read the csv file, download the images, write the successfully acquired images and label to file
    :param args:
    :return:
    """
    input_csv_path = args.filelist
    output_csv_path = args.filename
    image_directory = args.directory

    with open(output_csv_path, 'w') as output_file:
        with open(input_csv_path, 'r') as input_file:
            input_iter = csv.reader(input_file, delimiter=',')
            for _row in input_iter:
                image_name = _row[0]
                image_label = int(not(int(_row[1])))
                image_url = url_prefix + image_name
                image_path = os.path.join(args.directory, image_name)
                try:
                    request.urlretrieve(image_url, image_path)
                    try:
                        _img = io.imread(image_path)
                        out_str = '{},{}\n'.format(image_name, image_label)
                        output_file.write(out_str)
                        print(out_str)
                    except:
                        print('I/O failed on {}. Deleting file'.format(image_name))
                        os.remove(image_path)
                except:
                    print('unable to retrieve {}'.format(image_name))
    pass


if __name__ == "__main__":
    usage = "usage: %prog [option] arg"
    parser = argparse.ArgumentParser(description="Verify existence of manhattan frame in image")
    parser.add_argument("-f", "--file", dest="filename", default='./unifi_list.csv', help='image list file path')
    parser.add_argument("-l", "--list", dest="filelist", default='./unifi.csv',
                        help='path to text file with list of image paths')
    parser.add_argument("-d", "--dir", dest="directory", default='/home/ashish/Data/MediFor/UniFi/', help='directory containing image files')
    parser.add_argument("-r", "--result", dest="result", default='./log/', help='directory where the result is stored.')
    parser.add_argument("-u", "--username", dest="username", default='agupta', help='username')
    parser.add_argument("-p", "--password", dest="password", default='javaBean123', help='password')
    # Login to get access token
    args = parser.parse_args()

    response = requests.post('https://medifor.rankone.io/api/login/', {'username': args.username, 'password': args.password})
    if response.status_code == requests.codes.ok:
        token = response.json()['key']
    else:
        print("Bad credentials!")
        sys.exit(1)

    prune_image_list_file(args)



