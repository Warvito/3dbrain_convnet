'''
Fairly basic set of tools for real-time data augmentation on image data.
Can easily be extended to include new transformations,
new preprocessing methods, etc...
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re
import math
from skimage import exposure
import scipy.ndimage as ndi
import os
import threading
import cv2

import keras.backend as K


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
    # TODO: Documentation
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
                or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
                target size is different from that of the loaded image.
                Supported methods are "nearest", "bilinear", and "bicubic".
                If PIL version 1.1.3 or newer is installed, "lanczos" is also
                supported. If PIL version 3.4.0 or newer is installed, "box" and
                "hamming" are also supported. By default, "nearest" is used.
    # Returns
         A PIL Image instance.
    # Raises
         ImportError: if PIL is not available.
         ValueError: if interpolation method is not supported.
    """
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


def random_rx(x, rg, row_index=1, col_index=2, dep_index=3, channel_index=0,
              fill_mode='constant', cval=0., order=1):
    # TODO: Documentation
    """Performs a random rotation of a Numpy image tensor.
        # Arguments
            x: Input tensor. Must be 3D.
            rg: Rotation range, in degrees.
            row_axis: Index of axis for rows in the input tensor.
            col_axis: Index of axis for columns in the input tensor.
            channel_axis: Index of axis for channels in the input tensor.
            fill_mode: Points outside the boundaries of the input
                are filled according to the given mode
                (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
            cval: Value used for points outside the boundaries
                of the input if `mode='constant'`.
        # Returns
            Rotated Numpy image tensor.
    """
    h, w, d = x.shape[row_index], x.shape[col_index], x.shape[dep_index]
    xaxis = [1, 0, 0]
    alpha = np.deg2rad(np.random.uniform(-rg, rg))
    Rx = rotation_transf_matrix(alpha, xaxis)
    transform_matrix = transform_matrix_offset_center(Rx, h, w, d)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval, order)
    return x


def random_ry(x, angle_range,row_index=1, col_index=2, dep_index=3, channel_index=0,
              fill_mode='constant', cval=0., order=1):
    # TODO: Documentation
    """Performs a random rotation of a Numpy image tensor.
        # Arguments
            x: Input tensor. Must be 3D.
            rg: Rotation range, in degrees.
            row_axis: Index of axis for rows in the input tensor.
            col_axis: Index of axis for columns in the input tensor.
            channel_axis: Index of axis for channels in the input tensor.
            fill_mode: Points outside the boundaries of the input
                are filled according to the given mode
                (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
            cval: Value used for points outside the boundaries
                of the input if `mode='constant'`.
        # Returns
            Rotated Numpy image tensor.
    """
    h, w, d = x.shape[row_index], x.shape[col_index], x.shape[dep_index]
    yaxis = [0, 1, 0]
    alpha = np.deg2rad(np.random.uniform(-angle_range, angle_range))
    Ry = rotation_transf_matrix(alpha, yaxis)
    transform_matrix = transform_matrix_offset_center(Ry, h, w, d)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval, order)
    return x


def random_rz(x, angle_range,row_index=1, col_index=2, dep_index=3, channel_index=0,
              fill_mode='constant', cval=0., order=1):
    # TODO: Documentation
    """Performs a random rotation of a Numpy image tensor.
        # Arguments
            x: Input tensor. Must be 3D.
            rg: Rotation range, in degrees.
            row_axis: Index of axis for rows in the input tensor.
            col_axis: Index of axis for columns in the input tensor.
            channel_axis: Index of axis for channels in the input tensor.
            fill_mode: Points outside the boundaries of the input
                are filled according to the given mode
                (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
            cval: Value used for points outside the boundaries
                of the input if `mode='constant'`.
        # Returns
            Rotated Numpy image tensor.
    """
    h, w, d = x.shape[row_index], x.shape[col_index], x.shape[dep_index]
    zaxis = [0, 0, 1]
    alpha = np.deg2rad(np.random.uniform(-angle_range, angle_range))
    Rz = rotation_transf_matrix(alpha, zaxis)
    transform_matrix = transform_matrix_offset_center(Rz, h, w, d)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval, order)
    return x


def random_shift(x, wrg, hrg, drg, row_index=1, col_index=2, dep_index=3, channel_index=0,
                 fill_mode='constant', cval=0., order=1):
    # TODO: Documentation
    """Performs a random spatial shift of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Shifted Numpy image tensor.
    """
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


def random_zoom(x, zoom_range, row_index=1, col_index=2, dep_index=3, channel_index=0,
                fill_mode='constant', cval=0., order=1):
    # TODO: Documentation
    """Performs a random spatial zoom of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        zoom_range: Tuple of floats; zoom range for width and height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Zoomed Numpy image tensor.
    # Raises
        ValueError: if `zoom_range` isn't a tuple.
    """
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


def random_streching_x(x, streching_range, row_index=1, col_index=2, dep_index=3, channel_index=0,
                       fill_mode='constant', cval=0., order=1):
    # TODO: Different message
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


def random_streching_y(x, streching_range, row_index=1, col_index=2, dep_index=3, channel_index=0,
                       fill_mode='constant', cval=0., order=1):
    # TODO: Different message
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


def random_streching_z(x, streching_range, row_index=1, col_index=2, dep_index=3, channel_index=0,
                       fill_mode='constant', cval=0., order=1):
    # TODO: Different message
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


def apply_transform(x, transform_matrix, channel_index=0,
                    fill_mode='constant', cval=0., order=1):
    # TODO: Documentation
    """Apply the image transformation specified by a matrix.
    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:3, :3]
    final_offset = transform_matrix[:3, 3]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=order,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x


def add_gaussian_noise(x, std):
    x = x + np.random.normal(loc=0, scale=std, size=x.shape)
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
    x_channel = ndi.gaussian_filter(x_channel, sigma)
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
    # TODO: Documentation
    """Generate minibatches with real-time data augmentation.

    # Arguments
        do_ZMUV
        rotation_x: degrees (0 to 180).
        rotation_y: degrees (0 to 180).
        rotation_z: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        depth_shift_range: fraction of total depth.
        streching_x_range:
        streching_x_range:
        streching_x_range:
        zoom_range:
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
            Points outside the boundaries of the input are filled according to the given mode:
                'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                'nearest':  aaaaaaaa|abcd|dddddddd
                'reflect':  abcddcba|abcd|dcbaabcd
                'wrap':  abcdabcd|abcd|abcdabcd
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".
        order:
        image_shape:
        gaussian_noise:
        eq_prob:
        contrast_stretching:
        histogram_equalization:
        adaptive_equalization:

    """
    def __init__(self,
                 do_zmuv=False,
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
                 fill_mode='constant',
                 cval=0.,
                 order=1,
                 data_format=None,
                 image_shape=(256, 256, 256),
                 gaussian_noise=0.,
                 eq_prob=0.,
                 contrast_stretching=False,  #####
                 histogram_equalization=False,  #####
                 adaptive_equalization=False,  #####
                 ):

        if data_format is None:
            data_format = K.image_data_format()
        self.zmuv = do_zmuv
        self.rotation_x_range = rotation_x_range
        self.rotation_y_range = rotation_y_range
        self.rotation_z_range = rotation_z_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.depth_shift_range = depth_shift_range

        self.image_shape = image_shape

        self.mean = None
        self.std = None

        self.channel_shift_range = channel_shift_range
        self.gaussian_noise = gaussian_noise
        self.eq_prob = eq_prob
        self.fill_mode = fill_mode
        self.cval = cval
        self.order = order


        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('`data_format` should be `"channels_last"` (channel after row and '
                             'column) or `"channels_first"` (channel before row and column). '
                             'Received arg: ', data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
            self.dep_axis = 4

        if data_format == 'channels_last':
            self.channel_axis = 4
            self.row_axis = 1
            self.col_axis = 2
            self.dep_axis = 3


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
                            shuffle=True, seed=None,
                            permuted_labels=None):
        return DirectoryIterator(directory, self,
                                 batch_size=batch_size, subject_index=subject_index,
                                 image_shape = (1,) + self.image_shape,
                                 nb_class=nb_class,
                                 labels_permuted=permuted_labels,
                                 shuffle=shuffle, seed=seed)


    def fit(self, directory, subject_index=None):
        """Fits internal statistics to some sample data.
        """
        self.mean = np.zeros((1,) + self.image_shape, dtype=K.floatx())
        self.std = np.zeros((1,) + self.image_shape, dtype=K.floatx())

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
            if self.std is not None:
                x -= self.mean
                x /= (self.std + 1e-7)
        return x

    # https://github.com/aleju/imgaug
    # TODO: histogram equalization (3d to 2d) *, Gaussian blur (gaussian_filter)*, erosion (binary_erosion), salt & pepper,
    # TODO: Add*, multiply*, AverageBlur*, MedianBlur*, occlusion

    def random_transform(self, x):
        """Randomly augment a single image tensor.
        # Arguments
             x: 3D tensor, single image.
             seed: random seed.
        # Returns
             A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_dep_axis = self.dep_axis - 1
        img_channel_axis = self.channel_axis - 1

        xaxis, yaxis, zaxis = [1, 0, 0], [0, 1, 0], [0, 0, 1]

        if self.rotation_x_range:
            alpha = np.deg2rad(np.random.uniform(-self.rotation_x_range, self.rotation_x_range))
        else:
            alpha = 0
        Rx = rotation_transf_matrix(alpha, xaxis)

        if self.rotation_y_range:
            beta = np.deg2rad(np.random.uniform(-self.rotation_y_range, self.rotation_x_range))
        else:
            beta = 0
        Ry = rotation_transf_matrix(beta, yaxis)

        if self.rotation_z_range:
            gamma = np.deg2rad(np.random.uniform(-self.rotation_z_range, self.rotation_z_range))
        else:
            gamma = 0
        Rz = rotation_transf_matrix(gamma, xaxis)

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        if self.depth_shift_range:
            tz = np.random.uniform(-self.depth_shift_range, self.depth_shift_range) * x.shape[img_dep_axis]
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

        h, w, d = x.shape[img_row_axis], x.shape[img_col_axis], x.shape[img_dep_axis]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w, d)
        x = apply_transform(x, transform_matrix, img_channel_axis, fill_mode=self.fill_mode, cval=self.cval, order=self.order)

        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_axis)

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
    """Base class for image data iterators.
    Every `Iterator` must implement the `_get_batches_of_transformed_samples`
    method.
    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)



class DirectoryIterator(Iterator):
    """Iterator capable of reading images from a directory on disk.
    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        subject_index:
        image_shape:
        nb_class:
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.

    """
    def __init__(self, directory, image_data_generator,
                 subject_index=None,
                 image_shape=(256, 256, 256),
                 nb_class = None,
                 labels_permuted=None,
                 batch_size=32, shuffle=True, seed=None):
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

        if labels_permuted is not None:
            self.labels_permuted = labels_permuted

        super(DirectoryIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)


    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        batch_y = np.zeros((len(batch_x), self.nb_class), dtype=K.floatx())

        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[self.subject_index[j]]
            img, label = load_img(fname)
            if self.labels_permuted is not None:
                label = self.labels_permuted[j]
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