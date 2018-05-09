"""
Train Convolution Neural Network.

"""
# TODO: FIX CHANNEL LAST
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import imp
import numpy as np
import random as rn
import argparse
import tensorflow as tf

from keras.callbacks import TensorBoard
from keras import backend as K

sys.path.insert(0, './keras_extensions/')
from preprocessing_neuroimage_autoencoder import DataGenerator

def main(config_module):
    # ------------------------------- Reproducibility -----------------------------------------------
    seed = config_module.N_SEED
    os.environ['PYTHONHASHSEED'] = '0'

    np.random.seed(seed)
    rn.seed(seed)
    tf.set_random_seed(seed)

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    # ------------------------------- Experiment data --------------------------------------------------
    experiment_name = config_module.experiment_name

    # ------------------------------- Hiperparameters --------------------------------------------------
    batch_size = config_module.batch_size
    nb_epoch = config_module.nb_epoch
    image_dimension = config_module.image_dimension
    do_zmuv = config_module.do_zmuv

    # ---------------- Data Augmentation --------------------------------------------
    rotation_x_range = config_module.rotation_x_range
    rotation_y_range = config_module.rotation_y_range
    rotation_z_range = config_module.rotation_z_range
    streching_x_range = config_module.streching_x_range
    streching_y_range = config_module.streching_y_range
    streching_z_range = config_module.streching_z_range
    width_shift_range = config_module.width_shift_range
    height_shift_range = config_module.height_shift_range
    depth_shift_range = config_module.depth_shift_range
    zoom_range = config_module.zoom_range
    channel_shift_range = config_module.channel_shift_range
    gaussian_noise = config_module.gaussian_noise
    eq_prob = config_module.eq_prob

    # ------------------------------- Loading data --------------------------------------------------
    img_dir = "./results/" + experiment_name + "/CNN/img_npz/"
    print("Loading data from : ", img_dir)

    # ------------------------------- Reproducibility -----------------------------------------------
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    # ------------------------------ Model -------------------------------------------------
    print("Building autoencoder model")
    encoder, model = config_module.get_autoencoder_model()
    print(model.summary())

    # ------------------------------ Learning algorithm ---------------------------------------------------
    model.compile(loss='mse',
                  optimizer=config_module.get_optimizer(),
                  metrics=['accuracy'])

    # ------------------------------ Data generator ---------------------------------------------------
    print("")
    train_datagen = DataGenerator(do_zmuv=do_zmuv,
                                  image_shape=image_dimension,
                                  rotation_x_range=rotation_x_range,
                                  rotation_y_range=rotation_y_range,
                                  rotation_z_range=rotation_z_range,
                                  streching_x_range=streching_x_range,
                                  streching_y_range=streching_y_range,
                                  streching_z_range=streching_z_range,
                                  width_shift_range=width_shift_range,
                                  height_shift_range=height_shift_range,
                                  depth_shift_range=depth_shift_range,
                                  zoom_range=zoom_range,
                                  channel_shift_range=channel_shift_range,
                                  gaussian_noise=gaussian_noise,
                                  eq_prob=eq_prob)

    train_generator = train_datagen.flow_from_directory(img_dir, batch_size=batch_size)

    # ------------------------------- Normalization ------------------------------------------------
    if do_zmuv:
        print("")
        print("Calculating mean and std for ZMUV Normalization...")
        train_datagen.fit(img_dir)

    # ------------------------------ Tensorboard ---------------------------------------------------
    if not os.path.exists("./results/" + experiment_name + "/CNN/tensorboard/"):
        os.makedirs("./results/" + experiment_name + "/CNN/tensorboard/")

    tb = TensorBoard(log_dir="./results/" + experiment_name + "/CNN/tensorboard/run_AE")

    # ------------------------------ Traning ---------------------------------------------------
    model.fit_generator(train_generator,
                        steps_per_epoch=train_datagen.nb_sample/batch_size,
                        epochs=nb_epoch,
                        callbacks=[tb],
                        verbose=1)

    # -------------------------- Testing --------------------------------------------
    print("")
    y_predicted = encoder.predict_generator(train_generator, steps=train_datagen.nb_sample/batch_size)
    y_predicted = np.reshape(y_predicted, (y_predicted.shape[0], y_predicted.shape[1] * y_predicted.shape[2] * y_predicted.shape[3] * y_predicted.shape[4]))

    fnames = train_generator.get_names()
    c = np.column_stack([fnames,y_predicted])
    np.savetxt("./results/" + experiment_name + "/CNN/transformed_data.csv", c, delimiter=',', fmt='%s')

    # -------------------------- Saving models -------------------------------------------
    if not os.path.exists("./results/" + experiment_name + "/CNN/models/"):
        os.makedirs("./results/" + experiment_name + "/CNN/models/")
    model.save("./results/" + experiment_name + "/CNN/models/model_AE.h5")

    np.savez("./results/" + experiment_name + "/CNN/models/model_normalization_AE.npz",
             mean = train_datagen.mean, std = train_datagen.std)

    tf.reset_default_graph()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train model.')
    parser.add_argument("config_name", type=str, help="The name of file .py with configurations, e.g., Alexnet")
    args = parser.parse_args()

    config_name = args.config_name
    try:
        config_module = imp.load_source('config', config_name)
    except IOError:
        print('Cannot open ', config_name,
              '. Please specify the correct path of the configuration file. '
              'Example: python create_dataset.py ./config/config_test.py')

    main(config_module)