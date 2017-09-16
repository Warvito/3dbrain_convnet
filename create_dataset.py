import argparse
import glob
import os
import imp

import numpy as np
import nibabel as nib


def create_npy(args):
    config_name = args.config_name

    try:
        config_module = imp.load_source('config', "./config/" + config_name + ".py")
    except IOError:
        print(
            "Cannot open {}. Please specify the correct name of the configuration file (at the directory ./config). Make sure that the name of the file doesn't have any invalid character and the filename is without the suffix .py at the command. Correct example: python train.py config_test".format(
                config_name))

    paths = config_module.path_files

    labels_file = paths["labels_file"]
    images_dir = paths["raw_images_dir"]
    save_dir = paths["npy_dir"]

    print "Reading labels from %s" % labels_file
    labels = np.genfromtxt(labels_file, delimiter=',', dtype='int8')
    print "   # of labels samples: %d " % len(labels)

    input_data_type = config_module.input_data_type
    print "Reading images with format {} from: %s".format(input_data_type, images_dir)
    paths_train = glob.glob(images_dir + "/*" + input_data_type)
    paths_train.sort()

    n_samples = len(labels)
    if n_samples != len(paths_train):
        raise ValueError('Different number of labels and images files')

    print "Loading images"
    print "   # of images samples: %d " % len(paths_train)
    print ""
    print "{:<5}  {:50s} {:15s}\tCLASS\tMIN    - MAX VALUES".format('#', 'FILENAME', 'DIMENSIONS')
    for k, path in enumerate(paths_train):
        label = labels[k]
        img = nib.load(path)
        img = img.get_data()
        img = np.asarray(img, dtype='float16')
        print "{:<5}  {:50s} ({:3}, {:3}, {:3},{:3})\t{:}\t{:6.4} - {:6.4}".format((k + 1), os.path.basename(os.path.normpath(path)),
                                                                                   img.shape[3], img.shape[0], img.shape[1], img.shape[2], labels[k],
                                                                   np.min(img), np.max(img))
        if img.ndim == 4:
            img = np.reshape(img, (img.shape[3], img.shape[0], img.shape[1], img.shape[2], 1))
        else:
            print "Not 4D nifti"

        if config_module.nb_timesteps != img.shape[0]:
            raise ValueError('Different number of declared timesteps at config file.')

        for i in xrange(img.shape[0]):
            f_name = os.path.splitext(os.path.basename(path))[0] + "_t_"+str(i)
            print "Saving "+f_name

            np.savez(save_dir+f_name, image=img[i,...], label=label)

        del img
        print "Done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to create dataset files.')
    parser.add_argument("config_name", type=str, help="The name of file .py with configurations, e.g., Alexnet")
    args = parser.parse_args()
    create_npy(args)