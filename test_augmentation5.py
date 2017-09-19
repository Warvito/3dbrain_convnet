from __future__ import print_function
import argparse
import imp
import numpy as np
import os

from keras_extentions.preprocessing_neuroimage import *

from sklearn.model_selection import train_test_split

def main(args):
    config_name = args.config_name

    try:
        config_module = imp.load_source('config', "./config/" + config_name + ".py")
    except IOError:
        print(
            "Cannot open {}. Please specify the correct name of the configuration file (at the directory ./config). Example: python train.py config_test".format(
                config_name))

    np.random.seed(config_module.N_SEED)

    paths = config_module.path_files
    img_dir = paths["npy_dir"]
    labels_file = paths["labels_file"]
    output_dir = paths["save_file"]

    print("Loading data from : ", img_dir)
    labels = np.genfromtxt(labels_file, delimiter=',', dtype='int8')
    nb_samples = len(labels)

    nb_classes = config_module.nb_classes
    batch_size = config_module.batch_size
    nb_epoch = config_module.nb_epoch
    image_dimension = config_module.image_dimension
    do_zmuv = config_module.do_zmuv
    test_size = config_module.test_size

    train_index, test_index = train_test_split(range(0,nb_samples), test_size = test_size, random_state = 42, stratify=labels)

    nb_train_samples = len(train_index)
    nb_test_samples = len(test_index)


# ----------------------- Training -----------------------

    print("")
    print("Training with %d subjects." % (len(train_index)))
    print("Using %d neuroimages." % (len(train_index))) #No augmentation
    print("")

    train_datagen = DataGenerator(do_ZMUV=do_zmuv,
                            image_shape=image_dimension,
                            rotation_x_range=30,
                            rotation_y_range=30,
                            rotation_z_range=30,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            depth_shift_range=0.2,
                            zoom_range=0.3,
                            channel_shift_range=0.1
                            )
    filenames = []
    for fname in sort_nicely(os.listdir(img_dir)):
        if fname.lower().endswith('.npz'):
            filenames.append(os.path.join(img_dir, fname))

    file_npz = np.load(filenames[0])
    img = file_npz['image']
    label = file_npz['label']
    img = np.asarray(img,dtype='float32')

    import matplotlib.pyplot as plt
    def show_slices(slices):
        fig, axes = plt.subplots(1, len(slices))
        for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, cmap="gray", origin="lower")
    print(img.shape)
    slice_0 = img[0,50, :, :]
    slice_1 = img[0,:, 50, :]
    slice_2 = img[0,:, :, 50]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("ORIGINAL")
    plt.show()

    # img_sx = random_streching_x(img, [0.7,1.3])
    # slice_0 = img_sx[0,50, :, :]
    # slice_1 = img_sx[0,:, 50, :]
    # slice_2 = img_sx[0,:, :, 50]
    # show_slices([slice_0, slice_1, slice_2])
    # plt.suptitle("STRECHED X")
    # plt.show()
    #
    # img_sy = random_streching_y(img, [0.7,1.3])
    # slice_0 = img_sy[0,50, :, :]
    # slice_1 = img_sy[0,:, 50, :]
    # slice_2 = img_sy[0,:, :, 50]
    # show_slices([slice_0, slice_1, slice_2])
    # plt.suptitle("STRECHED Y")
    # plt.show()
    #
    # img_sz = random_streching_z(img, [0.7,1.3])
    # slice_0 = img_sz[0,50, :, :]
    # slice_1 = img_sz[0,:, 50, :]
    # slice_2 = img_sz[0,:, :, 50]
    # show_slices([slice_0, slice_1, slice_2])
    # plt.suptitle("STRECHED Z")
    # plt.show()
    #
    # img_rx = random_rx(img, 45)
    # slice_0 = img_rx[0,50, :, :]
    # slice_1 = img_rx[0,:, 50, :]
    # slice_2 = img_rx[0,:, :, 50]
    # show_slices([slice_0, slice_1, slice_2])
    # plt.suptitle("ROTATION X")
    # plt.show()
    #
    # img_ry = random_ry(img, 45)
    # slice_0 = img_ry[0,50, :, :]
    # slice_1 = img_ry[0,:, 50, :]
    # slice_2 = img_ry[0,:, :, 50]
    # show_slices([slice_0, slice_1, slice_2])
    # plt.suptitle("ROTATION Y")
    # plt.show()
    #
    # img_rz = random_rz(img, 45)
    # slice_0 = img_rz[0,50, :, :]
    # slice_1 = img_rz[0,:, 50, :]
    # slice_2 = img_rz[0,:, :, 50]
    # show_slices([slice_0, slice_1, slice_2])
    # plt.suptitle("ROTATION Z")
    # plt.show()
    #
    # img_ch_shifted = random_channel_shift(img, 0.5)
    # slice_0 = img_ch_shifted[0,50, :, :]
    # slice_1 = img_ch_shifted[0,:, 50, :]
    # slice_2 = img_ch_shifted[0,:, :, 50]
    # show_slices([slice_0, slice_1, slice_2])
    # plt.suptitle("CHANNEL SHIFTED")
    # plt.show()
    #
    # img_zoom = random_zoom(img, [0.8,1.2])
    # slice_0 = img_zoom[0,50, :, :]
    # slice_1 = img_zoom[0,:, 50, :]
    # slice_2 = img_zoom[0,:, :, 50]
    # show_slices([slice_0, slice_1, slice_2])
    # plt.suptitle("ZOOMED")
    # plt.show()
    #
    # img_shifted = random_shift(img, 0.2,0.2,0.2)
    # slice_0 = img_shifted[0,50, :, :]
    # slice_1 = img_shifted[0,:, 50, :]
    # slice_2 = img_shifted[0,:, :, 50]
    # show_slices([slice_0, slice_1, slice_2])
    # plt.suptitle("SHIFTED")
    # plt.show()

    # img_shifted = add_gaussian_noise(img, 0.02)
    # slice_0 = img_shifted[0,50, :, :]
    # slice_1 = img_shifted[0,:, 50, :]
    # slice_2 = img_shifted[0,:, :, 50]
    # show_slices([slice_0, slice_1, slice_2])
    # plt.suptitle("Gaussian noise")
    # plt.show()

    # img_shifted = multiply_value(img, (0.2,1.5))
    # slice_0 = img_shifted[0,50, :, :]
    # slice_1 = img_shifted[0,:, 50, :]
    # slice_2 = img_shifted[0,:, :, 50]
    # show_slices([slice_0, slice_1, slice_2])
    # plt.suptitle("multiply")
    # plt.show()
    #
    #
    # img_shifted = adaptive_equalization(img)
    # slice_0 = img_shifted[0,50, :, :]
    # slice_1 = img_shifted[0,:, 50, :]
    # slice_2 = img_shifted[0,:, :, 50]
    # show_slices([slice_0, slice_1, slice_2])
    # plt.suptitle("ad equa")
    # plt.show()
    #
    # img_shifted = equalize_histogram(img)
    # slice_0 = img_shifted[0,50, :, :]
    # slice_1 = img_shifted[0,:, 50, :]
    # slice_2 = img_shifted[0,:, :, 50]
    # show_slices([slice_0, slice_1, slice_2])
    # plt.suptitle("equa hist")
    # plt.show()
    #
    # img_shifted = contrast_stretching(img)
    # slice_0 = img_shifted[0,50, :, :]
    # slice_1 = img_shifted[0,:, 50, :]
    # slice_2 = img_shifted[0,:, :, 50]
    # show_slices([slice_0, slice_1, slice_2])
    # plt.suptitle("contrast_stretching")
    # plt.show()

    img_shifted = gaussian_filter(img,sigma=[0.7, 1.3])
    slice_0 = img_shifted[0,50, :, :]
    slice_1 = img_shifted[0,:, 50, :]
    slice_2 = img_shifted[0,:, :, 50]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("gaussian filter")
    plt.show()

    img_shifted = average_filter(img,sigma=(2,2))
    slice_0 = img_shifted[0,50, :, :]
    slice_1 = img_shifted[0,:, 50, :]
    slice_2 = img_shifted[0,:, :, 50]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("average filter")
    plt.show()

    img_shifted = median_filter(img,sigma=3)
    slice_0 = img_shifted[0,50, :, :]
    slice_1 = img_shifted[0,:, 50, :]
    slice_2 = img_shifted[0,:, :, 50]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("median filter")
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train model.')
    parser.add_argument("config_name", type=str, help="The name of file .py with configurations, e.g., Alexnet")
    args = parser.parse_args()
    main(args)