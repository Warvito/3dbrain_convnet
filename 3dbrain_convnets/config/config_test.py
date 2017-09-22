path_files = {
    "raw_images_dir": "./data_test", # Path to the directory with the raw data (.nii or .img and .hdr). Path used in create_dataset.py.
    "labels_file": "./data_test/labels.csv", # Path to the csv with the labels (need to be sorted as the raw data at the directory). Path used in create_dataset.py.
    "npy_dir": "./data_test/", # Path to save/load the python file of the dataset. This scipt allows faster load of the dataset during the training  (need to be sorted in the same order as the raw data at the directory "raw_images_dir"). Path used in create_dataset.py, train.py and saliency.py.
}

# Type of input data. Specify ".nii" or ".img"
input_data_type = ".nii"

# Perform the zero mean unit variance during the training.
do_zmuv = True

# k-fold cross-validation
n_folds = 5
n_repetitions = 30


# Define the random seed. This value will affect the random process during the code
N_SEED = 1

# Please, insert the number of the timesteps from the data.
nb_timesteps = 2

# Number of classes for classification
nb_classes = 2

# Minibatch size
batch_size = 10

# Number of classes for classification
nb_epoch = 20

# Dimension of each volume in voxels
image_dimension = (128, 96, 24)


def get_model():
    # Keras layers. Please check https://keras.io/layers/core/
    from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D
    from keras.models import Sequential

    print("")
    print("Building model...")
    img_channels = 1

    # ----------------------------------------- START MODEL ------------------------------------------------------------
    model = Sequential()
    model.add(Conv3D(16, (7, 7, 7), strides=(2,2,2), activation='relu', input_shape=image_dimension+(img_channels,)))
    model.add(MaxPooling3D(pool_size=(5, 5, 5), strides=(3, 3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv3D(32, (3, 3, 3), padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(nb_classes,activation='softmax'))
    # ------------------------------------------- END MODEL ------------------------------------------------------------
    return model


# Select the optimizer. Please check https://keras.io/optimizers/
def get_optimizer():
    import keras.optimizers as opt
    optimizer = opt.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    return optimizer