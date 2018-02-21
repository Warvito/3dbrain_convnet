"""
Performs the permutation test for the CNN method.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import imp
import csv
import random
import argparse
import numpy as np
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from keras import backend as K

sys.path.insert(0, './keras_extensions/')
from preprocessing_neuroimage import DataGenerator

def main(config_module):

    N_SEED = config_module.N_SEED
    experiment_name = config_module.experiment_name
    img_dir = "./results/" + experiment_name + "/CNN/img_npz/"
    paths = config_module.path_files
    labels_file = paths["labels_file"]
    n_folds = config_module.n_folds
    nb_classes = config_module.nb_classes
    batch_size = config_module.batch_size
    nb_epoch = config_module.nb_epoch
    image_dimension = config_module.image_dimension
    do_zmuv = config_module.do_zmuv

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

    n_permutations=config_module.n_permutations

    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(N_SEED)
    np.random.seed(N_SEED)

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(N_SEED)

    print("Loading data from : ", img_dir)
    labels = np.genfromtxt(labels_file, delimiter=',', dtype='int8')

    accumulated_BAC = np.zeros((n_permutations,))
    accumulated_SENS = np.zeros((n_permutations,))
    accumulated_SPEC = np.zeros((n_permutations,))
    accumulated_ERROR = np.zeros((n_permutations,))

    for i_rep in range(n_permutations):
        # CV  VARIABLES -----------------------------------------------------------------------------------------------
        cv_test_bac = np.zeros((n_folds,))
        cv_test_sens = np.zeros((n_folds,))
        cv_test_spec = np.zeros((n_folds,))
        cv_error_rate = np.zeros((n_folds,))

        np.random.seed(config_module.N_SEED + i_rep)
        permuted_labels = np.random.permutation(labels)

        skf = StratifiedKFold(n_splits=n_folds, random_state=config_module.N_SEED+i_rep, shuffle=True)

        for i_fold, (train_index, test_index) in enumerate(skf.split(permuted_labels, permuted_labels)):
            sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
            K.set_session(sess)

            labels_train, labels_test = permuted_labels[train_index], labels[test_index]

            print("REPETITION :", i_rep + 1)
            print("k-fold: ", i_fold + 1)
            print("")

            # CREATING MODEL ----------------------------------------------------------------------------------------
            print("Building model")
            model = config_module.get_model()
            print(model.summary())
            model.compile(loss='categorical_crossentropy',
                          optimizer=config_module.get_optimizer(),
                          metrics=['accuracy'])

            #  TRAINING  --------------------------------------------------------------------------------------------
            print("")
            print("Training with %d subjects." % (len(train_index)))
            print("TRAINING")
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
                batch_size=batch_size,
                permuted_labels = labels_train)

            print("")
            print("Training with %d neuroimages." % (len(train_index)))
            print("")

            model.fit_generator(train_generator,
                                steps_per_epoch=len(train_index)/batch_size,
                                epochs=nb_epoch)

            # TESTING --------------------------------------------------------------------------------------------------
            test_datagen = DataGenerator(do_zmuv=do_zmuv,
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
                                         image_shape=image_dimension,
                                         gaussian_noise=gaussian_noise,
                                         eq_prob=eq_prob)

            test_generator = test_datagen.flow_from_directory(
                img_dir,
                subject_index=test_index,
                nb_class=nb_classes,
                batch_size=batch_size)


            print("Testing with %d subjects." % (len(test_index)))
            y_predicted = model.predict_generator(test_generator, float(len(test_index))/float(batch_size))
            cm = confusion_matrix(labels_test, y_predicted)

            test_bac = np.sum(np.true_divide(np.diagonal(cm), np.sum(cm, axis=1))) / cm.shape[1]
            test_sens = np.true_divide(cm[1, 1], np.sum(cm[1, :]))
            test_spec = np.true_divide(cm[0, 0], np.sum(cm[0, :]))
            error_rate = np.true_divide(cm[0, 1] + cm[1, 0], np.sum(np.sum(cm)))

            cv_test_bac[i_fold] = test_bac
            cv_test_sens[i_fold] = test_sens
            cv_test_spec[i_fold] = test_spec
            cv_error_rate[i_fold] = error_rate

        accumulated_BAC[i_rep] = np.mean(cv_test_bac)
        accumulated_SENS[i_rep] = np.mean(cv_test_sens)
        accumulated_SPEC[i_rep] = np.mean(cv_test_spec)
        accumulated_ERROR[i_rep] = np.mean(cv_error_rate)
        print("PERMUTATION: ", i_rep + 1, " BAC: ", np.mean(cv_test_bac), ' SENS: ', np.mean(cv_test_sens), ' SPEC: ',np.mean(cv_test_spec) ,' ERROR: ', np.mean(cv_error_rate))

    if not os.path.exists("./results/" + experiment_name + "/SVM/permutation_test/"):
        os.makedirs("./results/" + experiment_name + "/SVM/permutation_test/")
    np.savetxt("./results/" + experiment_name + "/SVM/permutation_test/perm_SVM_bac.csv", np.asarray(accumulated_BAC), delimiter=",")
    np.savetxt("./results/" + experiment_name + "/SVM/permutation_test/perm_SVM_sens.csv", np.asarray(accumulated_SENS), delimiter=",")
    np.savetxt("./results/" + experiment_name + "/SVM/permutation_test/perm_SVM_spec.csv", np.asarray(accumulated_SPEC), delimiter=",")
    np.savetxt("./results/" + experiment_name + "/SVM/permutation_test/perm_SVM_error.csv", np.asarray(accumulated_ERROR), delimiter=",")

    print("")
    print("Reading CV PERFORMANCE " + python_files)
    file_npz = np.load("./results/" + experiment_name + "/SVM/summary/cv_results.npz")
    bac = file_npz['bac']
    sens = file_npz['sens']
    spec = file_npz['spec']
    error = file_npz['error_rate']

    print("BAC P-VALUE", (np.sum((accumulated_BAC > np.mean(bac)).astype('int')) + 1.) / (n_permutations + 1.))
    print("SENS P-VALUE", (np.sum((accumulated_SENS > np.mean(sens)).astype('int')) + 1.) / (n_permutations + 1.))
    print("SPEC P-VALUE", (np.sum((accumulated_SPEC > np.mean(spec)).astype('int')) + 1.) / (n_permutations + 1.))
    print("ERROR P-VALUE", (np.sum((accumulated_ERROR < np.mean(error)).astype('int')) + 1.) / (n_permutations + 1.))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train model.')
    parser.add_argument("config_name", type=str,
                        help="The name of file .py with configurations, e.g., Alexnet")
    args = parser.parse_args()

    config_name = args.config_name

    try:
        config_module = imp.load_source('config', config_name)

    except IOError:
        print('Cannot open ', config_name,
              '. Please specify the correct path of the configuration file. Example: python create_dataset.py ./config/config_test.py')


    main(config_module)
