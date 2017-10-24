"""

"""
from __future__ import print_function
import sys
import os
import csv
import imp
import numpy as np
import random as rn
import argparse
import tensorflow as tf

from keras.callbacks import TensorBoard
from keras import backend as K

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, './keras_extensions/')
from preprocessing_neuroimage import DataGenerator

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

    # ------------------------------- Hiperparametros --------------------------------------------------
    n_folds = config_module.n_folds
    nb_classes = config_module.nb_classes
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
    img_dir = "./experiments_files/" + experiment_name + "/CNN/img_npz/"
    print("Loading data from : ", img_dir)
    paths = config_module.path_files
    labels_file = paths["labels_file"]
    labels = np.genfromtxt(labels_file, delimiter=',', dtype='int8')

    # ----------------- Cross validation ---------------------------------
    print("")
    print("Performing k-fold cross validation")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    cv_test_bac = np.zeros((n_folds,))
    cv_test_sens = np.zeros((n_folds,))
    cv_test_spec = np.zeros((n_folds,))
    cv_error_rate = np.zeros((n_folds,))
    # ---------------------------------------------------------------

    for i_fold, (train_index, test_index) in enumerate(skf.split(labels, labels)):
        # ------------------------------- Reproducibility -----------------------------------------------
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

        # ---------------------------------------------------------------------------------------------
        nb_train_samples = len(train_index)
        nb_test_samples = len(test_index)

        print("")
        print("k-fold: ", i_fold + 1)
        print("")

        # ------------------------------ Model -------------------------------------------------
        print("Building model")
        model = config_module.get_model()
        print(model.summary())

        # ------------------------------ Learning algorithm ---------------------------------------------------
        model.compile(loss='categorical_crossentropy',
                      optimizer=config_module.get_optimizer(),
                      metrics=['accuracy'])

        # ------------------------------ Data generator ---------------------------------------------------
        print("")
        print("Training with %d subjects." % (len(train_index)))
        train_datagen = DataGenerator(do_ZMUV=do_zmuv,
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

        train_generator = train_datagen.flow_from_directory(
            img_dir,
            subject_index=train_index,
            nb_class=nb_classes,
            batch_size=batch_size)

        test_datagen = DataGenerator(do_ZMUV=do_zmuv,
                                     image_shape=image_dimension)

        test_generator = test_datagen.flow_from_directory(
            img_dir,
            subject_index=test_index,
            nb_class=nb_classes,
            batch_size=1)

        # ------------------------------- Normalization ------------------------------------------------
        if do_zmuv:
            print("")
            print("Calculating mean and std for ZMUV Normalization...")
            train_datagen.fit(img_dir, subject_index=train_index)
            test_datagen.mean = train_datagen.mean
            test_datagen.std = train_datagen.std

        # ------------------------------ Tensorboard ---------------------------------------------------
        if not os.path.exists("./experiments_files/" + experiment_name + "/CNN/tensorboard/"):
            os.makedirs("./experiments_files/" + experiment_name + "/CNN/tensorboard/")

        tb = TensorBoard(log_dir="./experiments_files/" + experiment_name + "/CNN/tensorboard/run_"+str(i_fold))

        # ------------------------------ Traning ---------------------------------------------------
        model.fit_generator(train_generator,
                            steps_per_epoch=nb_train_samples/batch_size,
                            epochs=nb_epoch,
                            validation_data=test_generator,
                            validation_steps=nb_test_samples/batch_size,
			                callbacks=[tb],
                            verbose=1)

        # -------------------------- Testing --------------------------------------------
        print("")
        print("Testing with %d subjects." % (nb_test_samples))

        y_predicted = model.predict_generator(test_generator, nb_test_samples)
        y_test = test_generator.get_labels()
        fnames = test_generator.get_names()

        # -------------------------- Error analysis --------------------------------------------
        if not os.path.exists("./experiments_files/" + experiment_name + "/CNN/error_analysis/"):
            os.makedirs("./experiments_files/" + experiment_name + "/CNN/error_analysis/")

        if i_fold == 0:
            file_predictions = open("./experiments_files/" + experiment_name + "/CNN/error_analysis/predictions.csv", 'w')
            wr = csv.writer(file_predictions)
            wr.writerow(['NAME', 'TRUE LABEL', 'PREDICTED'])
        else:
            file_predictions = open("./experiments_files/" + experiment_name + "/CNN/error_analysis/predictions.csv", 'a')
            wr = csv.writer(file_predictions)
        for j, fname in enumerate(fnames):
            wr.writerow([(str(fname)).encode('utf-8'),(str( y_test[j])).encode('utf-8'),(str((np.argmax(y_predicted, axis=1))[j])).encode('utf-8')])
        wr.writerow(['-', '-', '-'])
        file_predictions.close()

        # -------------------------- Performance metrics -------------------------------------------
        print("")
        print("Confusion matrix")
        cm = confusion_matrix(y_test, np.argmax(y_predicted, axis=1))
        print(cm)

        test_bac = np.sum(np.true_divide(np.diagonal(cm), np.sum(cm, axis=1))) / cm.shape[1]
        test_sens = np.true_divide(cm[1, 1], np.sum(cm[1, :]))
        test_spec = np.true_divide(cm[0, 0], np.sum(cm[0, :]))
        error_rate = np.true_divide(cm[0, 1] + cm[1, 0], np.sum(np.sum(cm)))

        print("Balanced acc: %.4f " % (test_bac))
        print("Sensitivity: %.4f " % (test_sens))
        print("Specificity: %.4f " % (test_spec))
        print("Error Rate: %.4f " % (error_rate))

        cv_test_bac[i_fold] = test_bac
        cv_test_sens[i_fold] = test_sens
        cv_test_spec[i_fold] = test_spec
        cv_error_rate[i_fold] = error_rate

        # -------------------------- Saving models -------------------------------------------
        if not os.path.exists("./experiments_files/" + experiment_name + "/CNN/models/"):
            os.makedirs("./experiments_files/" + experiment_name + "/CNN/models/")
        model.save("./experiments_files/" + experiment_name + "/CNN/models/model_%d.h5" % i_fold)

        tf.reset_default_graph()

    print("")
    print("")
    print("Cross-validation balanced acc: %.4f +- %.4f" % (cv_test_bac.mean(), cv_test_bac.std()))
    print("Cross-validation Sensitivity: %.4f +- %.4f" % (cv_test_sens.mean(), cv_test_sens.std()))
    print("Cross-validation Specificity: %.4f +- %.4f" % (cv_test_spec.mean(), cv_test_spec.std()))
    print("Cross-validation Error Rate: %.4f +- %.4f" % (cv_error_rate.mean(), cv_error_rate.std()))

    # -------------------------- Saving performance -------------------------------------------
    if not os.path.exists("./experiments_files/" + experiment_name + "/CNN/summary/"):
        os.makedirs("./experiments_files/" + experiment_name + "/CNN/summary/")
    np.savez("./experiments_files/" + experiment_name + "/CNN/summary/cv_results.npz",
             bac=cv_test_bac, sens=cv_test_sens, spec=cv_test_spec, error_rate = cv_error_rate )


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