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
from sklearn.externals import joblib

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

    random.seed(N_SEED)
    np.random.seed(N_SEED)

    kernel_file = "precomputed_kernel.npz"
    save_dir = "./experiments_files/" + experiment_name + "/precomputed_kernel/"
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

    # CV  VARIABLES -----------------------------------------------------------------------------------------------
    cv_test_bac = np.zeros((n_folds,))
    cv_test_sens = np.zeros((n_folds,))
    cv_test_spec = np.zeros((n_folds,))
    cv_error_rate = np.zeros((n_folds,))

    skf = StratifiedKFold(n_splits=n_folds, random_state=config_module.N_SEED, shuffle=True)

    accumulated_predicted_class = []
    accumulated_true_class = []

    for i_fold, (train_index, test_index) in enumerate(skf.split(labels, labels)):
        train_y, test_y = labels[train_index], labels[test_index]
        train_x, test_x,  = data[train_index, :][:, train_index], data[test_index,:][:, train_index]

        print("")
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

        print("TRAINING")
        clf = svm.SVC(C=C, kernel='precomputed')
        clf.fit(train_x, train_y)

        # ----------------------- Testing -----------------------
        print("")
        print("Testing with %d subjects." % (len(test_x)))

        y_predicted = clf.predict(test_x)
        fnames = file_npz['names']
        fnames = fnames[test_index]

        if not os.path.exists("./experiments_files/" + experiment_name + "/error_analysis/"):
            os.makedirs("./experiments_files/" + experiment_name + "/error_analysis/")

        if i_fold == 0:
            file_predictions = open("./experiments_files/" + experiment_name + "/error_analysis/predictions.csv", 'wb')
            wr = csv.writer(file_predictions)
            wr.writerow(['NAME', 'TRUE LABEL', 'PREDICTED'])
        else:
            file_predictions = open("./experiments_files/" + experiment_name + "/error_analysis/predictions.csv", 'a')
            wr = csv.writer(file_predictions)
        for j, fname in enumerate(fnames):
            wr.writerow([(str(fname)).encode('utf-8'),(str(test_y[j])).encode('utf-8'),(str(y_predicted[j])).encode('utf-8')])
        file_predictions.close()


        print("")
        print("Confusion matrix")
        cm = confusion_matrix(test_y, y_predicted)
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

        if not os.path.exists("./experiments_files/" + experiment_name + "/models/"):
            os.makedirs("./experiments_files/" + experiment_name + "/models/")
        joblib.dump(clf, "./experiments_files/" + experiment_name + "/models/model_%d.pkl" % i_fold)

    print("")
    print("")
    print("Cross-validation Balanced acc: %.4f +- %.4f" % (cv_test_bac.mean(), cv_test_bac.std()))
    print("Cross-validation Sensitivity: %.4f +- %.4f" % (cv_test_sens.mean(), cv_test_sens.std()))
    print("Cross-validation Specificity: %.4f +- %.4f" % (cv_test_spec.mean(), cv_test_spec.std()))
    print("Cross-validation Error Rate: %.4f +- %.4f" % (cv_error_rate.mean(), cv_error_rate.std()))

    if not os.path.exists("./experiments_files/" + experiment_name + "/summary/"):
        os.makedirs("./experiments_files/" + experiment_name + "/summary/")
    np.savez("./experiments_files/" + experiment_name + "/summary/cv_results.npz", bac=cv_test_bac, sens=cv_test_sens, spec=cv_test_spec, error_rate = cv_error_rate )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train model.')
    parser.add_argument("config_name", type=str, help="The name of file .py with configurations, e.g., ./config/config_test.py")
    args = parser.parse_args()
    main(args)
