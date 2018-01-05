"""
Train discriminative classifier between Manhattan and Non-manhattan images
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
from skimage import io
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score
import pickle
import csv
from skimage import color
from skimage import transform
from skimage import exposure

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# change global variable for desired set of valid image file extentions
valid_image_file_extentions = ['jpg', 'png', 'pgm', 'bmp']

# Colormap
cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)


# Plot functions
def plot_data(lda, X, y, y_pred, fig_index):
    splot = plt.subplot(1, 2, fig_index)
    if fig_index == 1:
        plt.title('LDA')
    elif fig_index == 2:
        plt.title('QDA')

    tp = (y == y_pred)  # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]

    alpha = 0.5

    # class 0: dots
    plt.plot(X0_tp[:, 0], X0_tp[:, 1], 'o', alpha=alpha,
             color='red', markeredgecolor='k')
    plt.plot(X0_fp[:, 0], X0_fp[:, 1], '*', alpha=alpha,
             color='#990000', markeredgecolor='k')  # dark red

    # class 1: dots
    plt.plot(X1_tp[:, 0], X1_tp[:, 1], 'o', alpha=alpha,
             color='blue', markeredgecolor='k')
    plt.plot(X1_fp[:, 0], X1_fp[:, 1], '*', alpha=alpha,
             color='#000099', markeredgecolor='k')  # dark blue

    # class 0 and 1 : areas
    nx, ny = 200, 200
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                   norm=colors.Normalize(0., 1.))
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='k')

    # means
    plt.plot(lda.means_[0][0], lda.means_[0][1],
             'o', color='black', markersize=10, markeredgecolor='k')
    plt.plot(lda.means_[1][0], lda.means_[1][1],
             'o', color='black', markersize=10, markeredgecolor='k')

    red_patch = mpatches.Patch(color='red', label='manhattan')
    blue_patch = mpatches.Patch(color='blue', label='non-manhattan')
    plt.legend(handles=[red_patch, blue_patch])

    return splot


def plot_ellipse(splot, mean, cov, color):
    v, w = linalg.eigh(cov)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    # filled Gaussian at 2 standard deviation
    ell = mpl.patches.Ellipse(mean, 2 * v[0] ** 0.5, 2 * v[1] ** 0.5,
                              180 + angle, facecolor=color,
                              edgecolor='yellow',
                              linewidth=2, zorder=2)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.5)
    splot.add_artist(ell)
    splot.set_xticks(())
    splot.set_yticks(())


def plot_lda_cov(lda, splot):
    plot_ellipse(splot, lda.means_[0], lda.covariance_, 'red')
    plot_ellipse(splot, lda.means_[1], lda.covariance_, 'blue')


def plot_qda_cov(qda, splot):
    plot_ellipse(splot, qda.means_[0], qda.covariance_[0], 'red')
    plot_ellipse(splot, qda.means_[1], qda.covariance_[1], 'blue')

data_directory = '/home/ashish/Data/MediFor/UniFi/'

label = []

train_image_list = './unifi_list.csv'

# parse the list to create data vector and label vector

data = np.empty((0, 2))

with open(train_image_list, 'r') as csvfile:
    input_iter = csv.reader(csvfile, delimiter=',')
    for row in input_iter:
        image_path = os.path.join(data_directory, row[0])
        print(image_path)
        image_label = int(row[1])

        """ temporary data is stored to local directory to interact with lua code """
        temp_result_file = './temp.jpg'
        temp_file_list = './temp.txt'

        temp_input_file = './temp_input.jpg'
        img = io.imread(image_path)
        try:
            img_resized = transform.resize(img, (256, 256), mode='reflect')
        except:
            img_resized = img

        try:
            io.imsave(temp_input_file, img_resized)
        except:
            continue

        with open(temp_file_list, 'w') as tf:
            tf.write(temp_input_file)

        _t = os.system("~/torch/install/bin/th norm_one_image.lua")


        image = io.imread(temp_result_file)  # read the image with normals at pixels
        if len(image.shape) == 2:
            image = color.gray2rgb(image)
        _total_pixels = image.shape[0] * image.shape[1]
        _hist = np.histogram(image[:, :, 2], bins=256)[0]
        _hist = [h / _total_pixels for h in _hist]
        X = np.zeros((1, 2))
        X[0, :] = _hist[-3:-1]  # we are interested in the 3rd ortho plane
        data = np.vstack((data, X))
        label.append(image_label)


print(data.shape)
label = np.array(label)
print(label.shape)

X = data
y = label


plt.figure(1, figsize=(16,8))

# Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
y_pred = lda.fit(X, y).predict(X)
print(accuracy_score(y, y_pred))
splot = plot_data(lda, X, y, y_pred, fig_index=1)
plot_lda_cov(lda, splot)
plt.axis('tight')

# Quadratic Discriminant Analysis
qda = QuadraticDiscriminantAnalysis(store_covariance=True)
y_pred = qda.fit(X, y).predict(X)
print(accuracy_score(y, y_pred))
splot = plot_data(qda, X, y, y_pred, fig_index=2)
plot_qda_cov(qda, splot)
plt.axis('tight')

plt.suptitle('Manhattan World')
plt.show()

pkl_filename = './lda_model.pkl'
with open(pkl_filename, 'wb') as file:
    pickle.dump(lda, file)

pkl_filename = './qda_model.pkl'
with open(pkl_filename, 'wb') as file:
    pickle.dump(qda, file)

with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

score = pickle_model.score(X,y)
print(score)