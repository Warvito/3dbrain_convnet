from __future__ import print_function

import os
import imp
import glob
import argparse
import numpy as np
import nibabel as nib

from keras_extensions.preprocessing_neuroimage import sort_nicely


def create_npy(args):
    config_name = args.config_name

    try:
        config_module = imp.load_source('config', config_name)

    except IOError:
        print('Cannot open ', config_name,
              '. Please specify the correct path of the configuration file. Example: python create_dataset.py ./config/config_test.py')

    paths = config_module.path_files
    input_data_type = config_module.input_data_type
    experiment_name = config_module.experiment_name
    labels_file = paths["labels_file"]
    data_dir = paths["raw_images_dir"]

    print("Saving 3d images using .npz fromat for experiment: ", experiment_name)

    save_dir = "./experiments_files/" + experiment_name + "/img_npz/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Reading labels from %s" % labels_file)
    labels = np.genfromtxt(labels_file, delimiter=',', dtype='int8')
    print("   # of labels samples: %d " % len(labels))

    print("Reading images with format {} from: %s".format(input_data_type, data_dir))
    img_paths = glob.glob(data_dir + "/*" + input_data_type)
    img_paths = sort_nicely(img_paths)

    n_samples = len(labels)
    if n_samples != len(img_paths):
        raise ValueError('Different number of labels and images files')

    # IF YOU WANT TO REMOVE BLANK VOXELS, UNCOMMENT THIS SECTION
    print("Calculating 3d dimensions to remove blank voxels and reduce the 3d volume.")
    min_x = 1000
    max_x = 0
    min_y = 1000
    max_y = 0
    min_z = 1000
    max_z = 0

    for k, path in enumerate(img_paths):
        img = nib.load(path)
        img = img.get_data()
        img = np.asarray(img, dtype='float32')
        img = np.nan_to_num(img)
        img_shape = img.shape

    #     X
        for i in range(0,img_shape[0]):
            if np.max(img[i,:,:]) > 0:
                break
        if min_x > i:
            min_x = i

        for i in range(img_shape[0]-1,0,-1):
            if np.max(img[i,:,:]) > 0:
                break
        if max_x < i:
            max_x = i

    #     Y
        for i in range(0,img_shape[1]):
            if np.max(img[:,i,:]) > 0:
                break
        if min_y > i:
            min_y = i

        for i in range(img_shape[1]-1,0,-1):
            if np.max(img[:,i,:]) > 0:
                break
        if max_y < i:
            max_y = i

    #     Z
        for i in range(0,img_shape[2]):
            if np.max(img[:,:,i]) > 0:
                break
        if min_z > i:
            min_z = i

        for i in range(img_shape[2]-1,0,-1):
            if np.max(img[:,:,i]) > 0:
                break
        if max_z < i:
            max_z = i

    print(max_x,max_y,max_z)
    print(min_z,min_y,min_x)

    print("Loading images")
    print("   # of images samples: %d " % len(img_paths))
    print("")
    print("{:<5}  {:100s} {:15s}\tCLASS\tMIN    - MAX VALUES".format('#', 'FILENAME', 'DIMENSIONS'))
    for k, path in enumerate(img_paths):
        label = labels[k]
        img = nib.load(path)
        img = img.get_data()
        img = np.asarray(img, dtype='float32')
        img = np.nan_to_num(img)
        img = img[min_x:max_x,min_y:max_y,min_z:max_z]
        print("{:<5}  {:100s} ({:3}, {:3}, {:3})\t{:}\t{:6.4} - {:6.4}".format((k + 1), os.path.basename(os.path.normpath(path)), img.shape[0], img.shape[1], img.shape[2], labels[k], np.min(img), np.max(img)))
        img = np.true_divide(img,np.max(img))
        img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        f_name = os.path.splitext(os.path.basename(path))[0]
        np.savez(save_dir + f_name, image=img, label=label)
        del img
    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to create dataset files.')
    parser.add_argument("config_name", type=str, help="The name of file .py with configurations, e.g., ./config/config_test.py")
    args = parser.parse_args()
    create_npy(args)
