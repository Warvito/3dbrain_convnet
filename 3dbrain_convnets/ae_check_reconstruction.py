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
import copy

from keras import backend as K
from keras.models import load_model

sys.path.insert(0, './keras_extensions/')

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
    image_test = config_module.image_test
    image_dimension = config_module.image_dimension

    # ------------------------------- Hiperparameters --------------------------------------------------
    do_zmuv = config_module.do_zmuv

    # ------------------------------- Loading data --------------------------------------------------
    img_dir = "./results/" + experiment_name + "/CNN/img_npz/"
    print("Loading data from : ", img_dir)

    # ------------------------------- Reproducibility -----------------------------------------------
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    # ------------------------------ Model -------------------------------------------------
    print("Building autoencoder model")
    model = load_model("./results/" + experiment_name + "/CNN/models/model_AE.h5")
    print(model.summary())

    # ------------------------------ IMAGE TEST --------------------------------------------
    img_file = np.load("./results/" + experiment_name + "/CNN/img_npz/" + image_test)
    img = img_file["image"]

    # ------------------------------- Normalization ------------------------------------------------
    if do_zmuv:
        norm_file = np.load("./results/" + experiment_name + "/CNN/models/model_normalization_AE.npz")
        norm_mean = norm_file["mean"]
        norm_std = norm_file["std"]

        print("")
        print("Calculating mean and std for ZMUV Normalization...")
        img_normalizada = copy.deepcopy(img)
        img_normalizada -= norm_mean
        img_normalizada /= (norm_std + 1e-7)
    else:
        img_normalizada = img


    # -------------------------- Testing --------------------------------------------
    print("")
    img_reconstructed = model.predict(img_normalizada)


    # -------------------------- Ploting --------------------------------------------
    img_reconstructed *= (norm_std + 1e-7)
    img_reconstructed += norm_mean

    _,y_slice, _ = image_dimension

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img[:,np.floor(y_slice/2),:], cmap="gray", origin="lower")
    axes[1].imshow(img_reconstructed[:, np.floor(y_slice / 2), :], cmap="gray", origin="lower")
    plt.show()


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