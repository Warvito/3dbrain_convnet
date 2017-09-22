
experiment_name = "TEST_1"

path_files = {
    "raw_images_dir": "./example/data_test/DATASET_1/", # Path to the directory with the raw data (.nii or .img and .hdr). Path used in create_dataset.py.
    "labels_file": "./example/data_test/DATASET_1/labels_dataset1.csv", # Path to the csv with the labels (need to be sorted as the raw data at the directory). Path used in create_dataset.py.
    "npy_dir": "./example/data_test/DATASET_1/", # Path to save/load the python file of the dataset. This scipt allows faster load of the dataset during the training  (need to be sorted in the same order as the raw data at the directory "raw_images_dir"). Path used in create_dataset.py, train.py and saliency.py.
    "save_file": "./example/output_test/",
}

# Type of input data. Specify ".nii" or ".img"
input_data_type = ".nii"

# Perform the zero mean unit variance during the training.
do_zmuv = True

# Define the random seed. This value will affect the random process during the code
N_SEED = 1

# k-fold cross-validation
n_folds = 3

# Size of test set (only in train_one_model)
test_size = 0.4

# Number of classes for classification
nb_classes = 2

# Minibatch size
batch_size = 2

# Number of training epochs
nb_epoch = 20

# Dimension of each volume in voxels
image_dimension = (57, 67, 56)

# REAL TIME AUGMENTATION PARAMETERS
# Set everything 0. for no augmentation
rotation_x_range = 30
rotation_y_range = 30
rotation_z_range = 30
streching_x_range = 0.1
streching_y_range = 0.1
streching_z_range = 0.1
width_shift_range = 0.2
height_shift_range = 0.2
depth_shift_range = 0.2
zoom_range = 0.3
channel_shift_range = 0.1

cropping_shape = (140,240,210)
occlusion_size = 10.,
noise_sigma = 0.1,


def get_model():
    from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Convolution3D, MaxPooling3D
    from keras.regularizers import l2
    from keras.models import Sequential

    img_channels = 1

    # ---------------------------------------- START MODEL ------------------------------------------------------------
    model = Sequential()
    model.add(Convolution3D(64, (7, 7, 7), strides=(2,2,2),padding = 'valid', activation='relu', input_shape=(img_channels,) + image_dimension))
    model.add(MaxPooling3D(pool_size=(5, 5, 5), strides=(2, 2 , 2)))
    model.add(Convolution3D(64, (5, 5, 5) ,padding = 'same', activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(5, 5, 5), strides=(2, 2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(2, activation='softmax'))
    # ------------------------------------------- END MODEL ------------------------------------------------------------
    return model


# Select the optimizer. Please check https://keras.io/optimizers/
def get_optimizer():
    import keras.optimizers as opt
    optimizer = opt.rmsprop(lr=0.05, decay=1e-6)
    return optimizer
