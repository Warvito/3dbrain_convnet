"""

"""
from __future__ import print_function
import sys
import os
import imp
import glob
import argparse
import numpy as np
import nibabel as nib

sys.path.insert(0, './keras_extensions/')
from utils import sort_nicely


def create_kernel(args):
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
    metadata_file = paths["metadata_file"]
    data_dir = paths["raw_images_dir"]

    print("Creating precomputed linear kernels for experiment: ", experiment_name)

    save_dir = "./results/" + experiment_name + "/SVM/precomputed_kernel/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    kernel_file = "precomputed_kernel.npz"

    print("Reading labels from %s" % labels_file)
    labels = np.genfromtxt(labels_file, delimiter=',', dtype='int8')
    print("   # of labels samples: %d " % len(labels))

    print("Reading covariates from %s" % labels_file)
    metadata = np.genfromtxt(metadata_file, delimiter=',', dtype='int8')
    print("   # of labels samples: %d " % len(labels))

    print("Reading images with format {} from: %s".format(input_data_type, data_dir))
    img_paths = glob.glob(data_dir + "/*" + input_data_type)
    img_paths = sort_nicely(img_paths)

    img_names = []
    for img_name in img_paths:
        img_names.append(os.path.splitext(os.path.basename(img_name))[0])

    n_samples = len(labels)
    if n_samples != len(img_paths):
        raise ValueError('Different number of labels and images files')

    print("Loading images")
    print("   # of images samples: %d " % len(img_paths))

    n_samples = len(img_paths)

    print(n_samples)

    K = np.float64(np.zeros((n_samples, n_samples)))
    step_size = 30

    # outer loop
    for i in range(int(np.ceil(n_samples / np.float(step_size)))):

        it = i + 1
        max_it = int(np.ceil(n_samples / np.float(step_size)))
        print(" outer loop iteration: %d of %d." % (it, max_it))

        # generate indices and then paths for this block
        start_ind_1 = i * step_size
        stop_ind_1 = min(start_ind_1 + step_size, n_samples)
        block_paths_1 = img_paths[start_ind_1:stop_ind_1]
        block_metadata_1 = metadata[start_ind_1:stop_ind_1, :]

        # read in the images in this block
        images_1 = []
        for k, path in enumerate(block_paths_1):
            img = nib.load(path)
            img = img.get_data()
            img = np.asarray(img, dtype='float64')
            img = np.nan_to_num(img)
            img_vec = np.reshape(img, np.product(img.shape))
            img_vec = np.append(img_vec, block_metadata_1[k, :])
            images_1.append(img_vec)
            del img

        images_1 = np.array(images_1)
        for j in range(i + 1):
            it = j + 1
            max_it = i + 1

            print(" inner loop iteration: %d of %d." % (it, max_it))

            # if i = j, then sets of image data are the same - no need to load
            if i == j:
                start_ind_2 = start_ind_1
                stop_ind_2 = stop_ind_1
                images_2 = images_1

            # if i !=j, read in a different block of images
            else:
                start_ind_2 = j * step_size
                stop_ind_2 = min(start_ind_2 + step_size, n_samples)
                block_paths_2 = img_paths[start_ind_2:stop_ind_2]
                block_metadata_2 = metadata[start_ind_2:stop_ind_2, :]

                images_2 = []
                for k, path in enumerate(block_paths_2):
                    img = nib.load(path)
                    img = img.get_data()
                    img = np.asarray(img, dtype='float64')
                    img = np.nan_to_num(img)
                    img_vec = np.reshape(img, np.product(img.shape))
                    img_vec = np.append(img_vec, block_metadata_2[k, :])
                    images_2.append(img_vec)
                    del img
                images_2 = np.array(images_2)

            block_K = np.dot(images_1, np.transpose(images_2))
            K[start_ind_1:stop_ind_1, start_ind_2:stop_ind_2] = block_K
            K[start_ind_2:stop_ind_2, start_ind_1:stop_ind_1] = np.transpose(block_K)

    print("")
    print("Saving Dataset")
    print("   Kernel+Labels:" + kernel_file)
    np.savez(save_dir + kernel_file, kernel=K, labels=labels, names=img_names)
    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to create dataset files.')
    parser.add_argument("config_name", type=str,
                        help="The name of file .py with configurations, e.g., ./config/config_test.py")
    args = parser.parse_args()
    create_kernel(args)
