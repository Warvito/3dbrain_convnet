"""CONFIGURATION FILE"""

experiment_name = "TEST1"

path_files = {
    "raw_images_dir": "./data/PPMI/", # Path to the directory with the raw data (.nii or .img and .hdr). Path used in create_dataset.py.
    "labels_file": "./data/labels.csv", # Path to the csv with the labels (need to be sorted as the raw data at the directory). Path used in create_dataset.py.
}

# Type of input data. Specify ".nii" or ".img"
input_data_type = ".nii"

C = 1
n_permutations = 1000

# Perform the zero mean unit variance during the training.
do_zmuv = True

# Define the random seed. This value will affect the random process during the code
N_SEED = 1

# k-fold cross-validation
n_folds = 5

# Size of test set (only in train_one_model)
test_size = 0.4

# Number of classes for classification
nb_classes = 2

# Minibatch size
batch_size = 2

# Number of training epochs
nb_epoch = 20

# Dimension of each volume in voxels
image_dimension = (95, 133, 104)

# REAL TIME AUGMENTATION PARAMETERS
# Set everything 0. for no augmentation
rotation_x_range = 0
rotation_y_range = 0
rotation_z_range = 0

width_shift_range = 0
height_shift_range = 0
depth_shift_range = 0

streching_x_range = 0
streching_y_range = 0
streching_z_range = 0

zoom_range = 0
channel_shift_range = 0
gaussian_noise = 0
eq_prob = 0


def get_model():
    from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Convolution3D, MaxPooling3D
    from keras.regularizers import l2
    from keras.models import Sequential

    img_channels = 1

    # ---------------------------------------- START MODEL ------------------------------------------------------------
    model = Sequential()
    model.add(Convolution3D(64, (7, 7, 7), strides=(2,2,2),padding = 'valid', activation='relu', input_shape=(img_channels,) + image_dimension))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(5, 5, 5), strides=(2, 2 , 2)))

    model.add(Convolution3D(64, (5, 5, 5) ,padding = 'same', activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(5, 5, 5), strides=(2, 2, 2)))

    model.add(Convolution3D(64, (5, 5, 5), padding='same', activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(5, 5, 5), strides=(3, 3, 3)))

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
