from __future__ import print_function
import argparse
import imp
import numpy as np
import os

from keras_extentions.preprocessing_neuroimage import DataGenerator, sort_nicely, random_channel_shift,random_zoom,random_shift,random_rx, random_ry, random_rz, random_streching_x, random_streching_y, random_streching_z

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

    print("Loading data from : ", img_dir)
    labels = np.genfromtxt(labels_file, delimiter=',', dtype='int8')
    nb_samples = len(labels)

    nb_classes = config_module.nb_classes
    batch_size = config_module.batch_size
    nb_epoch = config_module.nb_epoch
    image_dimension = config_module.image_dimension
    do_zmuv = config_module.do_zmuv
    test_size = 0.2

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

    import matplotlib.pyplot as plt
    def show_slices(slices):
        fig, axes = plt.subplots(1, len(slices))
        for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, cmap="gray", origin="lower")
    print(img.shape)
    slice_0 = img[26, :, :,0]
    slice_1 = img[:, 30, :,0]
    slice_2 = img[:, :, 16,0]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("ORIGINAL")
    plt.show()

    img_sx = random_streching_x(img, [1.5,2])
    slice_0 = img_sx[26, :, :,0]
    slice_1 = img_sx[:, 30, :,0]
    slice_2 = img_sx[:, :, 16,0]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("STRECHED X")
    plt.show()

    img_sy = random_streching_y(img, [1.5,2])
    slice_0 = img_sy[26, :, :,0]
    slice_1 = img_sy[:, 30, :,0]
    slice_2 = img_sy[:, :, 16,0]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("STRECHED Y")
    plt.show()

    img_sz = random_streching_z(img, [1.5,2])
    slice_0 = img_sz[26, :, :,0]
    slice_1 = img_sz[:, 30, :,0]
    slice_2 = img_sz[:, :, 16,0]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("STRECHED Z")
    plt.show()

    img_rx = random_rx(img, 10)
    slice_0 = img_rx[26, :, :,0]
    slice_1 = img_rx[:, 30, :,0]
    slice_2 = img_rx[:, :, 16,0]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("ROTATION X")
    plt.show()

    img_ry = random_ry(img, 10)
    slice_0 = img_ry[26, :, :,0]
    slice_1 = img_ry[:, 30, :,0]
    slice_2 = img_ry[:, :, 16,0]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("ROTATION Y")
    plt.show()

    img_rz = random_rz(img, 10)
    slice_0 = img_rz[26, :, :,0]
    slice_1 = img_rz[:, 30, :,0]
    slice_2 = img_rz[:, :, 16,0]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("ROTATION Z")
    plt.show()

    img_ch_shifted = random_channel_shift(img, 0.5)
    slice_0 = img_ch_shifted[26, :, :,0]
    slice_1 = img_ch_shifted[:, 30, :,0]
    slice_2 = img_ch_shifted[:, :, 16,0]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("CHANNEL SHIFTED")
    plt.show()

    img_zoom = random_zoom(img, [1.1,1.2])
    slice_0 = img_zoom[26, :, :,0]
    slice_1 = img_zoom[:, 30, :,0]
    slice_2 = img_zoom[:, :, 16,0]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("ZOOMED")
    plt.show()

    img_shifted = random_shift(img, 0.2,0.2,0.2)
    slice_0 = img_shifted[26, :, :,0]
    slice_1 = img_shifted[:, 30, :,0]
    slice_2 = img_shifted[:, :, 16,0]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("SHIFTED")
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train model.')
    parser.add_argument("config_name", type=str, help="The name of file .py with configurations, e.g., Alexnet")
    args = parser.parse_args()
    main(args)