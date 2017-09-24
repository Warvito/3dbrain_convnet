from __future__ import print_function

import os
import imp
import csv
import random
import argparse
import numpy as np
from sklearn import svm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix


def main(args):
    config_name = args.config_name

    try:
        config_module = imp.load_source('config', config_name)

    except IOError:
        print('Cannot open ', config_name,
              '. Please specify the correct path of the configuration file. Example: python create_dataset.py ./config/config_test.py')

    N_SEED = config_module.N_SEED
    experiment_name = config_module.experiment_name
    C = config_module.C
    n_folds = config_module.n_folds
    n_permutations=config_module.n_permutations

    random.seed(N_SEED)
    np.random.seed(N_SEED)

    kernel_file = "precomputed_kernel.npz"
    save_dir = "./experiments_files/" + experiment_name + "/SVM/precomputed_kernel/"
    python_files = save_dir+kernel_file

    print("")
    print("Loading data from " + python_files)
    file_npz = np.load(python_files)
    data = file_npz['kernel']
    labels = file_npz['labels']

    print("")
    print("Hyperparameters")
    print("  C: %f" % C)
    print("")
    print("")
    print("    Dataset size: %d" % len(data))
    print("")
    print("Stratified K-fold Cross Validation....")

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
            train_y, test_y = permuted_labels[train_index], permuted_labels[test_index]
            train_x, test_x,  = data[train_index, :][:, train_index], data[test_index,:][:, train_index]

            clf = svm.SVC(C=C, kernel='precomputed')
            clf.fit(train_x, train_y)

            # ----------------------- Testing -----------------------

            y_predicted = clf.predict(test_x)
            cm = confusion_matrix(test_y, y_predicted)

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

    if not os.path.exists("./experiments_files/" + experiment_name + "/SVM/permutation_test/"):
        os.makedirs("./experiments_files/" + experiment_name + "/SVM/permutation_test/")
    np.savetxt("./experiments_files/" + experiment_name + "/SVM/permutation_test/perm_SVM_bac.csv", np.asarray(accumulated_BAC), delimiter=",")
    np.savetxt("./experiments_files/" + experiment_name + "/SVM/permutation_test/perm_SVM_sens.csv", np.asarray(accumulated_SENS), delimiter=",")
    np.savetxt("./experiments_files/" + experiment_name + "/SVM/permutation_test/perm_SVM_spec.csv", np.asarray(accumulated_SPEC), delimiter=",")
    np.savetxt("./experiments_files/" + experiment_name + "/SVM/permutation_test/perm_SVM_error.csv", np.asarray(accumulated_ERROR), delimiter=",")

    print("")
    print("Reading CV PERFORMANCE " + python_files)
    file_npz = np.load("./experiments_files/" + experiment_name + "/SVM/summary/cv_results.npz")
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
    main(args)
