"""CONFIGURATION FILE"""

experiment_name = "TEST"

path_files = {
    "raw_images_dir": "./data/PPMI/",  # Path to the directory with the raw data (.nii or .img and .hdr).
                                       #  Path used in create_dataset.py.

    "labels_file": "./data/labels_autoencoder.csv",  # Path to the csv with the labels
                                         # (need to be sorted as the raw data at the directory).
                                         #  Path used in create_dataset.py.

    "metadata_file": "./data/metadata.csv",  # Path to the csv with the metadata
                                             # (need to be sorted as the raw data at the directory and each column
                                             #  represents one covariate). Path used in create_kernel_with_metadata.py.

    "mask_file": "" #Path to the file with the binary mask (.nii) Path used in create_dataset.py.
                    #  Leave it blank ("") if there is no mask.
    }
# ./data/mask.nii
# Type of input data. Specify ".nii" or ".img"
input_data_type = ".nii"

C = 1
n_permutations = 10

# Perform the zero mean unit variance during the training.
do_zmuv = True

# Define the random seed. This value will affect the random process during the code
N_SEED = 1

# k-fold cross-validation (n_folds = 10 recommended)
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
image_dimension = (70, 90, 74)

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

def get_autoencoder_model():
    from keras.layers import Dense, Dropout, Convolution3D, MaxPooling3D, UpSampling3D, GaussianNoise, Cropping3D
    from keras.regularizers import l2
    from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
    from keras.models import Model
    from keras.models import Sequential

    img_channels = 1
    noise_factor = 0.02

    # ---------------------------------------- START MODEL ------------------------------------------------------------
    input_img = Input(shape=(img_channels,) + image_dimension)
    x = GaussianNoise(noise_factor)(input_img)
    x = Convolution3D(64, (7, 7, 7), padding='same', activation='relu')(x)
    x = MaxPooling3D(strides=(2, 2, 2), padding='same')(x)

    x = Convolution3D(32, (5, 5, 5), padding='same', activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = MaxPooling3D(strides=(2, 2, 2), padding='same')(x)

    x = Convolution3D(32, (5, 5, 5), padding='same', activation='relu', kernel_regularizer=l2(0.0001))(x)
    encoded = MaxPooling3D(strides=(2, 2, 2), padding='same')(x)

    # Bottleneck

    x = Convolution3D(32, (5, 5, 5), padding='same', activation='relu', kernel_regularizer=l2(0.0001))(encoded)
    x = UpSampling3D(size=(2, 2, 2))(x)

    x = Convolution3D(32, (5, 5, 5), padding='same', activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = UpSampling3D(size=(2, 2, 2))(x)

    x = Convolution3D(64, (5, 5, 5), padding='same', activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = UpSampling3D(size=(2, 2, 2))(x)

    decoded = Convolution3D(1, (7, 7, 7), padding='same')(x)
    decoded_cropped = Cropping3D(((1, 1), (3, 3), (3, 3)))(decoded)


    encoder = Model(input_img, encoded)
    autoencoder = Model(input_img, decoded_cropped)

    # ------------------------------------------- END MODEL ------------------------------------------------------------
    return encoder, autoencoder

# Select the optimizer. Please check https://keras.io/optimizers/
def get_optimizer():
    import keras.optimizers as opt
    optimizer = opt.rmsprop(lr=0.05, decay=1e-6)
    return optimizer
