from __future__ import print_function

import argparse
import imp
import random
from imblearn.over_sampling import RandomOverSampler

import numpy as np
from sklearn import svm

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix


def main(args):
    config_name = args.config_name
    try:
        config_module = imp.load_source('config', "./config/" + config_name + ".py")
    except IOError:
        print('Cannot open {}'.format(config_name))

    random.seed(config_module.N_SEED)
    np.random.seed(config_module.N_SEED)

    python_files = config_module.path_files['python_files']
    print("")
    print("Loading data from " + python_files)
    file_npz = np.load(python_files)
    data = file_npz['kernel']
    labels = file_npz['labels']

    C = config_module.C
    print("")
    print("Hyperparameters")
    print("  C: %f" % C)
    print("")
    print("")
    print("    Dataset size: %d" % len(data))
    print("")
    print("Stratified K-fold Cross Validation....")
    n_folds = config_module.n_folds
    n_repetitions=config_module.n_repetitions

    accumulated_BAC = []
    accumulated_SENS = []
    accumulated_SPEC = []
    accumulated_ERROR = []
    for i_rep in range(n_repetitions):
        # CV  VARIABLES -----------------------------------------------------------------------------------------------
        cv_test_bac = np.zeros((n_folds,))
        cv_test_sens = np.zeros((n_folds,))
        cv_test_spec = np.zeros((n_folds,))
        cv_error_rate = np.zeros((n_folds,))

        skf = StratifiedKFold(n_splits=n_folds, random_state=config_module.N_SEED+i_rep, shuffle=True)
        i_fold = 0

        np.random.seed(config_module.N_SEED + i_rep)
        permuted_labels = np.random.permutation(labels)


        for train_index, test_index in skf.split(labels, labels):
            train_y, test_y = permuted_labels[train_index], permuted_labels[test_index]
            train_x, test_x,  = data[train_index, :][:, train_index], data[test_index,:][:, train_index]

            print("")
            print("REPETITION: ", i_rep + 1)

            print("k-fold: ", i_fold + 1)
            print("")

            print("SIZE TRAINING SET")
            print("          CLASS 0:", (len(train_y) - np.count_nonzero(train_y)))
            print("          CLASS 1:", np.count_nonzero(train_y))
            print("")
            print("SIZE TEST SET")
            print("          CLASS 0:", (len(test_y) - np.count_nonzero(test_y)))
            print("          CLASS 1:", np.count_nonzero(test_y))
            print("")

            # Oversampling  -----------------------------------------------------------------------------------------------
            ros = RandomOverSampler()
            data_train_resampled, labels_train_resampled = ros.fit_sample(train_x, train_y)

            print("SIZE TRAINING SET AFTER OVERSAMPLING")
            print("          CLASS 0:", (len(labels_train_resampled) - np.count_nonzero(labels_train_resampled)))
            print("          CLASS 1:", np.count_nonzero(labels_train_resampled))
            print("")

            train_x = train_x.reshape((len(train_x), -1))
            test_x = test_x.reshape((len(test_x), -1))

            print("TRAINING")
            clf = svm.SVC(C=C, kernel='precomputed')
            clf.fit(train_x, train_y)

            test_pred = clf.predict(test_x)
            print("")
            print("Confusion matrix")
            cm = confusion_matrix(test_y, test_pred)
            print(cm)
            print("")

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

            i_fold += 1

        np.savetxt("./logs/perm_SVM_bac_rep{}.csv".format(i_rep), cv_test_bac, delimiter=",")
        np.savetxt("./logs/perm_SVM_sens_rep{}.csv".format(i_rep), cv_test_sens, delimiter=",")
        np.savetxt("./logs/perm_SVM_spec_rep{}.csv".format(i_rep), cv_test_spec, delimiter=",")
        np.savetxt("./logs/perm_SVM_error_rep{}.csv".format(i_rep), cv_error_rate, delimiter=",")

        accumulated_BAC.extend([np.mean(cv_test_bac)])
        accumulated_SENS.extend([np.mean(cv_test_sens)])
        accumulated_SPEC.extend([np.mean(cv_test_spec)])
        accumulated_ERROR.extend([np.mean(cv_error_rate)])

    np.savetxt("./logs/_accumulated_perm_SVM_bac.csv", np.asarray(accumulated_BAC), delimiter=",")
    np.savetxt("./logs/_accumulated_perm_SVM_sens.csv", np.asarray(accumulated_SENS), delimiter=",")
    np.savetxt("./logs/_accumulated_perm_SVM_spec.csv", np.asarray(accumulated_SPEC), delimiter=",")
    np.savetxt("./logs/_accumulated_perm_SVM_error.csv", np.asarray(accumulated_ERROR), delimiter=",")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train model.')
    parser.add_argument("config_name", type=str,
                        help="The name of file .py with configurations, e.g., Alexnet")
    args = parser.parse_args()
    main(args)
