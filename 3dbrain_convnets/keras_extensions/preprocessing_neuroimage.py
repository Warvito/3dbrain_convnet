'''Fairly basic set of tools for real-time data augmentation on image data.
Can easily be extended to include new transformations,
new preprocessing methods, etc...
'''
from __future__ import absolute_import
from __future__ import print_function

from scipy import ndimage
from skimage import exposure
import scipy.ndimage as ndi
import os
import threading
import warnings
import cv2

import keras.backend as K

import numpy as np
import os
import threading
import re
import math
import glob

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def sort_nicely(l):
    return sorted(l, key=alphanum_key)

def load_img(path):
    file_npz = np.load(path)
    img = file_npz['image']
    label = file_npz['label']
    return img, label

def unit_vector(data, axis=None, out=None):
    if out is None:
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def rotation_transf_matrix(angle, direction):
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])

    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[ 0.0,         -direction[2],  direction[1]],
                      [ direction[2], 0.0,          -direction[0]],
                      [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    return M

def random_rx(x, angle_range,row_index=1, col_index=2, dep_index=3, channel_index=0, fill_mode='constant', cval=0., order=0):
    h, w, d = x.shape[row_index], x.shape[col_index], x.shape[dep_index]
    xaxis = [1, 0, 0]
    alpha = np.pi / 180 * np.random.uniform(-angle_range, angle_range)
    Rx = rotation_transf_matrix(alpha, xaxis)
    transform_matrix = transform_matrix_offset_center(Rx, h, w, d)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval, order)
    return x

def random_ry(x, angle_range,row_index=1, col_index=2, dep_index=3, channel_index=0, fill_mode='constant', cval=0., order=0):
    h, w, d = x.shape[row_index], x.shape[col_index], x.shape[dep_index]
    xaxis = [0, 1, 0]
    alpha = np.pi / 180 * np.random.uniform(-angle_range, angle_range)
    Ry = rotation_transf_matrix(alpha, xaxis)
    transform_matrix = transform_matrix_offset_center(Ry, h, w, d)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval, order)
    return x

def random_rz(x, angle_range,row_index=1, col_index=2, dep_index=3, channel_index=0, fill_mode='constant', cval=0., order=0):
    h, w, d = x.shape[row_index], x.shape[col_index], x.shape[dep_index]
    xaxis = [0, 0, 1]
    alpha = np.pi / 180 * np.random.uniform(-angle_range, angle_range)
    Rz = rotation_transf_matrix(alpha, xaxis)
    transform_matrix = transform_matrix_offset_center(Rz, h, w, d)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval, order)
    return x


def random_shift(x, wrg, hrg, drg, row_index=1, col_index=2, dep_index=3, channel_index=0,
                 fill_mode='constant', cval=0., order=0):
    h, w, d = x.shape[row_index], x.shape[col_index], x.shape[dep_index]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    tz = np.random.uniform(-drg, drg) * d

    translation_matrix = np.array([[1, 0, 0, tx],
                                   [0, 1, 0, ty],
                                   [0, 0, 1, tz],
                                   [0, 0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval, order)
    return x


def random_zoom(x, zoom_range, row_index=1, col_index=2, dep_index=3, channel_index=0, fill_mode='constant', cval=0., order=0):
    if len(zoom_range) != 2:
        raise ValueError('zoom_range should be a tuple or list of two floats. '
                         'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy, zz = 1, 1, 1
    else:
        zx, zy, zz = np.random.uniform(zoom_range[0], zoom_range[1], 3)
    zoom_matrix = np.array([[zx, 0, 0, 0],
                            [0, zy, 0, 0],
                            [0, 0, zz, 0],
                            [0, 0, 0, 1]])

    h, w, d = x.shape[row_index], x.shape[col_index], x.shape[dep_index]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w, d)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval, order)
    return x

def random_streching_x(x, streching_range, row_index=1, col_index=2, dep_index=3, channel_index=0, fill_mode='constant', cval=0., order=0):
    if len(streching_range) != 2:
        raise ValueError('zoom_range should be a tuple or list of two floats. '
                         'Received arg: ', streching_range)

    if streching_range[0] == 1 and streching_range[1] == 1:
        zx = 1
    else:
        zx = (np.random.uniform(streching_range[0], streching_range[1], 1))[0]
    zoom_matrix = np.array([[zx, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

    h, w, d = x.shape[row_index], x.shape[col_index], x.shape[dep_index]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w, d)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval, order)
    return x

def random_streching_y(x, streching_range, row_index=1, col_index=2, dep_index=3, channel_index=0, fill_mode='constant', cval=0., order=0):
    if len(streching_range) != 2:
        raise ValueError('zoom_range should be a tuple or list of two floats. '
                         'Received arg: ', streching_range)

    if streching_range[0] == 1 and streching_range[1] == 1:
        zy = 1
    else:
        zy = (np.random.uniform(streching_range[0], streching_range[1], 1))[0]
    zoom_matrix = np.array([[1, 0, 0, 0],
                            [0, zy, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

    h, w, d = x.shape[row_index], x.shape[col_index], x.shape[dep_index]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w, d)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval, order)
    return x

def random_streching_z(x, streching_range, row_index=1, col_index=2, dep_index=3, channel_index=0, fill_mode='constant', cval=0., order=0):
    if len(streching_range) != 2:
        raise ValueError('zoom_range should be a tuple or list of two floats. '
                         'Received arg: ', streching_range)

    if streching_range[0] == 1 and streching_range[1] == 1:
        zz = 1
    else:
        zz = (np.random.uniform(streching_range[0], streching_range[1], 1))[0]
    zoom_matrix = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, zz, 0],
                            [0, 0, 0, 1]])

    h, w, d = x.shape[row_index], x.shape[col_index], x.shape[dep_index]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w, d)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval, order)
    return x


def random_channel_shift(x, intensity, channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def transform_matrix_offset_center(matrix, x, y, z):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    o_z = float(z) / 2 + 0.5
    offset_matrix = np.array([[1, 0, 0, o_x], [0, 1, 0, o_y], [0, 0, 1, o_z], [0, 0, 0, 1]])
    reset_matrix = np.array([[1, 0, 0, -o_x], [0, 1, 0, -o_y], [0, 0, 1, -o_z], [0, 0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x, transform_matrix, channel_index=0, fill_mode='constant', cval=0., order=0):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:3, :3]
    final_offset = transform_matrix[:3, 3]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=order, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def add_gaussian_noise(x, std):
    x = x + np.random.normal(loc= 0, scale = std, size = x.shape)
    return x

def multiply_value(x, intensity, channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel * np.random.uniform(intensity[0], intensity[1]), min_x, max_x) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x

def adaptive_equalization(x, channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    x_channel = x[0]
    orig_shape = x_channel.shape
    x_channel=x_channel.reshape((x_channel.shape[0], -1))
    x_channel= exposure.equalize_adapthist(x_channel, clip_limit=0.01)
    x_channel=x_channel.reshape(orig_shape)
    x = np.expand_dims(x_channel, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x

def equalize_histogram(x, channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    x_channel = x[0]
    orig_shape = x_channel.shape
    x_channel=x_channel.reshape((x_channel.shape[0], -1))
    x_channel= exposure.equalize_hist(x_channel)
    x_channel=x_channel.reshape(orig_shape)
    x = np.expand_dims(x_channel, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x

def contrast_stretching(x, channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    x_channel = x[0]
    orig_shape = x_channel.shape
    x_channel=x_channel.reshape((x_channel.shape[0], -1))
    p2, p98 = np.percentile(x_channel, (2, 98))
    x_channel = exposure.rescale_intensity(x_channel, in_range=(p2, p98))
    x_channel= x_channel.reshape(orig_shape)
    x = np.expand_dims(x_channel, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x

def gaussian_filter(x, sigma=(0.0,1.0), channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    sigma = np.random.uniform(sigma[0], sigma[1])
    x_channel = x[0]
    orig_shape = x_channel.shape
    x_channel=x_channel.reshape((x_channel.shape[0], -1))
    x_channel = ndimage.gaussian_filter(x_channel, sigma)
    x_channel=x_channel.reshape(orig_shape)
    x = np.expand_dims(x_channel, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x

def average_filter(x, sigma=(5,5), channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    x_channel = x[0]
    orig_shape = x_channel.shape
    x_channel = x_channel.reshape((x_channel.shape[0], -1))
    x_channel = cv2.blur(x_channel, sigma)
    x_channel=x_channel.reshape(orig_shape)
    x = np.expand_dims(x_channel, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x

def median_filter(x, sigma=5, channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    x_channel = x[0]
    orig_shape = x_channel.shape
    x_channel = x_channel.reshape((x_channel.shape[0], -1))
    x_channel = cv2.medianBlur(x_channel, sigma)
    x_channel=x_channel.reshape(orig_shape)
    x = np.expand_dims(x_channel, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x



class DataGenerator(object):
    '''Generate minibatches with
    real-time data augmentation.
    # Arguments
        do_ZMUV
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".
    '''
    def __init__(self,
                 do_ZMUV=False,
                 contrast_stretching=False,  #####
                 histogram_equalization=False,  #####
                 adaptive_equalization=False,  #####
                 rotation_x_range=0.,
                 rotation_y_range=0.,
                 rotation_z_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 depth_shift_range=0.,
                 streching_x_range=0.,
                 streching_y_range=0.,
                 streching_z_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 gaussian_noise=0.,
                 eq_prob=0.,
                 fill_mode='constant',
                 cval=0.,
                 order=0,
                 image_shape=(256, 256, 256),
                 dim_ordering='default'):

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.__dict__.update(locals())
        self.zmuv = do_ZMUV
        self.image_shape = image_shape
        self.mean = np.zeros((1,) + image_shape, dtype='float32')
        self.std = np.zeros((1,) + image_shape, dtype='float32')
        self.rotation_x_range = rotation_x_range
        self.rotation_y_range = rotation_y_range
        self.rotation_z_range = rotation_z_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.depth_shift_range = depth_shift_range
        self.channel_shift_range = channel_shift_range
        self.gaussian_noise = gaussian_noise
        self.eq_prob = eq_prob
        self.fill_mode = fill_mode
        self.cval = cval
        self.order = order


        if dim_ordering not in {'tf', 'th'}:
            raise ValueError('dim_ordering should be "tf" (channel after row and '
                             'column) or "th" (channel before row and column). '
                             'Received arg: ', dim_ordering)
        self.dim_ordering = dim_ordering
        if dim_ordering == 'th':
            self.channel_index = 1
            self.row_index = 2
            self.col_index = 3
            self.dep_index = 4
        if dim_ordering == 'tf':
            self.channel_index = 4
            self.row_index = 1
            self.col_index = 2
            self.dep_index = 3

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('zoom_range should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)


        if np.isscalar(streching_x_range):
            self.streching_x_range = [1 - streching_x_range, 1 + streching_x_range]
        elif len(streching_x_range) == 2:
            self.streching_x_range = [streching_x_range[0], streching_x_range[1]]
        else:
            raise ValueError('streching_x_range should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', streching_x_range)

        if np.isscalar(streching_y_range):
            self.streching_y_range = [1 - streching_y_range, 1 + streching_y_range]
        elif len(streching_y_range) == 2:
            self.streching_y_range = [streching_y_range[0], streching_y_range[1]]
        else:
            raise ValueError('streching_y_range should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', streching_y_range)

        if np.isscalar(streching_z_range):
            self.streching_z_range = [1 - streching_z_range, 1 + streching_z_range]
        elif len(streching_z_range) == 2:
            self.streching_z_range = [streching_z_range[0], streching_z_range[1]]
        else:
            raise ValueError('streching_z_range should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', streching_z_range)


    def flow_from_directory(self, directory,
                            batch_size=32, subject_index=None,
                            nb_class=None,
                            shuffle=True, seed=None):
        return DirectoryIterator(directory, self,
                                 batch_size=batch_size, subject_index=subject_index,
                                 image_shape = (1,) + self.image_shape,
                                 nb_class=nb_class,
                                 shuffle=shuffle, seed=seed)


    def fit(self, directory, subject_index=None, seed=None):

        nb_train_samples = len(subject_index)

        paths_train = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".npz"):
                    paths_train.append(os.path.join(root, file))
        filenames = sort_nicely(paths_train)

        for i in range(nb_train_samples):
            fname = filenames[subject_index[i]]
            img, label = load_img(fname)
            self.mean += np.true_divide(img,nb_train_samples)

        for i in range(nb_train_samples):
            fname = filenames[subject_index[i]]
            img, label = load_img(fname)
            self.std += np.true_divide(np.square(img-self.mean),nb_train_samples)

        self.std = np.sqrt(self.std)


    def standardize(self, x):
        if self.zmuv:
            x -= self.mean
            x /= (self.std + 1e-7)
        return x

    # https://github.com/aleju/imgaug
    # TODO: histogram equalization (3d to 2d) *, Gaussian blur (gaussian_filter)*, erosion (binary_erosion), salt & pepper,
    # TODO: Add*, multiply*, AverageBlur*, MedianBlur*, occlusion

    def random_transform(self, x):
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_dep_index = self.dep_index - 1
        img_channel_index = self.channel_index - 1

        xaxis, yaxis, zaxis = [1, 0, 0], [0, 1, 0], [0, 0, 1]
        if self.rotation_x_range:
            alpha = np.pi / 180 * np.random.uniform(-self.rotation_x_range, self.rotation_x_range)
        else:
            alpha = 0
        Rx = rotation_transf_matrix(alpha, xaxis)

        if self.rotation_y_range:
            beta = np.pi / 180 * np.random.uniform(-self.rotation_y_range, self.rotation_x_range)
        else:
            beta = 0
        Ry = rotation_transf_matrix(beta, yaxis)

        if self.rotation_z_range:
            gamma = np.pi / 180 * np.random.uniform(-self.rotation_z_range, self.rotation_z_range)
        else:
            gamma = 0
        Rz = rotation_transf_matrix(gamma, xaxis)

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_index]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_index]
        else:
            ty = 0

        if self.depth_shift_range:
            tz = np.random.uniform(-self.depth_shift_range, self.depth_shift_range) * x.shape[img_dep_index]
        else:
            tz = 0

        translation_matrix = np.array([[1, 0, 0, tx],
                                       [0, 1, 0, ty],
                                       [0, 0, 1, tz],
                                       [0, 0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy, zz = 1, 1, 1
        else:
            zx, zy, zz = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 3)
        zoom_matrix = np.array([[zx, 0, 0, 0],
                                [0, zy, 0, 0],
                                [0, 0, zz, 0],
                                [0, 0, 0, 1]])

        if self.streching_x_range[0] == 1 and self.streching_x_range[1] == 1:
            zx = 1
        else:
            zx = (np.random.uniform(self.streching_x_range[0], self.streching_x_range[1], 1))[0]
        streching_x_matrix = np.array([[zx, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]])

        if self.streching_y_range[0] == 1 and self.streching_y_range[1] == 1:
            zy = 1
        else:
            zy = (np.random.uniform(self.streching_y_range[0], self.streching_y_range[1], 1))[0]
        streching_y_matrix = np.array([[1, 0, 0, 0],
                                       [0, zy, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]])

        if self.streching_z_range[0] == 1 and self.streching_z_range[1] == 1:
            zz = 1
        else:
            zz = (np.random.uniform(self.streching_z_range[0], self.streching_z_range[1], 1))[0]
        streching_z_matrix = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, zz, 0],
                                       [0, 0, 0, 1]])

        scaling_matrix = np.dot(np.dot(np.dot(zoom_matrix,streching_x_matrix),streching_y_matrix),streching_z_matrix)

        transform_matrix = (np.dot(np.dot(np.dot(np.dot(Rx,Ry),Rz),translation_matrix), scaling_matrix)) #Gimball lock problem

        h, w, d = x.shape[img_row_index], x.shape[img_col_index], x.shape[img_dep_index]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w, d)
        x = apply_transform(x, transform_matrix, img_channel_index, fill_mode=self.fill_mode, cval=self.cval, order=self.order)

        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_index)

        if self.gaussian_noise != 0:
            x = add_gaussian_noise(x,self.gaussian_noise)

        prob = np.random.uniform(0,1)
        if prob < self.eq_prob:
            x = adaptive_equalization(x)

        prob = np.random.uniform(0,1)
        if prob < self.eq_prob:
            x = equalize_histogram(x)

        prob = np.random.uniform(0,1)
        if prob < self.eq_prob:
            x = contrast_stretching(x)

        prob = np.random.uniform(0,1)
        if prob < self.eq_prob:
            x = gaussian_filter(x, sigma=2)

        prob = np.random.uniform(0,1)
        if prob < self.eq_prob:
            x = average_filter(x, sigma=(2,2))

        prob = np.random.uniform(0,1)
        if prob < self.eq_prob:
            x = median_filter(x, sigma=2)

        return x


class Iterator(object):

    def __init__(self, N, batch_size, shuffle, seed):
        self.N = N
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(N, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(N)
                if shuffle:
                    index_array = np.random.permutation(N)

            current_index = (self.batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

class DirectoryIterator(Iterator):

    def __init__(self, directory, image_data_generator, subject_index=None, image_shape=(256, 256, 256), nb_class = None, batch_size=32, shuffle=True, seed=None):
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.nb_class = nb_class
        self.image_shape = image_shape

        # first, count the number of samples and classes
        self.nb_sample = len(subject_index)
        self.filenames = []
        self.subject_index = subject_index

        paths_train = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".npz"):
                    paths_train.append(os.path.join(root, file))

        self.filenames = sort_nicely(paths_train)

        print('Found %d neuroimages in the directory.' % (self.nb_sample))

        super(DirectoryIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[self.subject_index[j]]
            img, label = load_img(fname)
            img = self.image_data_generator.random_transform(img)
            img = self.image_data_generator.standardize(img)
            batch_x[i] = img
            batch_y[i, label] = 1.
        return batch_x, batch_y

    def get_labels(self):
        labels = np.zeros((self.nb_sample,))
        for i,j in enumerate(self.subject_index):
            fname = self.filenames[j]
            img, label = load_img(fname)
            labels[i] =label
        return labels

    def get_names(self):
        fnames = []
        for i in self.subject_index:
            fnames.append(os.path.basename(self.filenames[i]))
        return fnames