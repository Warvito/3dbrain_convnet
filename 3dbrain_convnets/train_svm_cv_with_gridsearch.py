"""
Train Support Vector Machine Using gridsearch

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import imp
import csv
import random
import argparse
import numpy as np
from sklearn import svm

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.externals import joblib

def main(config_module):
    N_SEED = config_module.N_SEED
    experiment_name = config_module.experiment_name
    C = [2e3, 2e1, 2e-0, 2e-1, 2e-3, 2e-5, 2e-7, 2e-9]
    param_grid = dict(C=C)
    n_folds = config_module.n_folds

    random.seed(N_SEED)
    np.random.seed(N_SEED)

    kernel_file = "precomputed_kernel.npz"
    save_dir = "./results/" + experiment_name + "/SVM/precomputed_kernel/"
    python_files = save_dir+kernel_file

    print("")
    print("Loading data from " + python_files)
    file_npz = np.load(python_files)
    data = file_npz['kernel']
    labels = file_npz['labels']

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
        def balanced_accuracy_score(actual, prediction):
            cm = confusion_matrix(actual,prediction)
            bac = np.sum(np.true_divide(np.diagonal(cm), np.sum(cm, axis=1))) / cm.shape[1]
            return bac
        grid_scorer = make_scorer(balanced_accuracy_score, greater_is_better=True)

        nested_skf = StratifiedKFold(n_splits=10)

        clf = svm.SVC(kernel='precomputed')
        grid = GridSearchCV(estimator=clf, param_grid=param_grid, scoring = grid_scorer, cv = nested_skf, verbose=1)

        print("Performing gridSearchCV...")
        grid_result = grid.fit(train_x, train_y)
        print("gridSearchCV done!")

        print("Sumary of gridSearch")
        # ------------------------------------------------------------------ SUMMARIZE RESULTS --------------------------------------------------------------------------------------
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        print("Re-training the model using the whole training set and evaluating on test set")
        print("Building model")
        best_c = grid_result.best_params_["C"]

        model = svm.SVC(C=best_c, kernel='precomputed')
        model.fit(train_x, train_y)
        print("Best model")
        print(model)


        # ----------------------- Testing -----------------------
        print("")
        print("Testing with %d subjects." % (len(test_x)))

        y_predicted = model.predict(test_x)
        fnames = file_npz['names']
        fnames = fnames[test_index]

        if not os.path.exists("./results/" + experiment_name + "/SVM/error_analysis/"):
            os.makedirs("./results/" + experiment_name + "/SVM/error_analysis/")

        if i_fold == 0:
            file_predictions = open("./results/" + experiment_name + "/SVM/error_analysis/predictions.csv", 'w')
            wr = csv.writer(file_predictions)
            wr.writerow(['NAME', 'TRUE LABEL', 'PREDICTED'])
        else:
            file_predictions = open("./results/" + experiment_name + "/SVM/error_analysis/predictions.csv", 'a',
                                    encoding='utf8')
            wr = csv.writer(file_predictions)
        for j, fname in enumerate(fnames):
            wr.writerow([str(fname),str(test_y[j]),str(y_predicted[j])])
        wr.writerow(['-', '-', '-'])
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

        if not os.path.exists("./results/" + experiment_name + "/SVM/models/"):
            os.makedirs("./results/" + experiment_name + "/SVM/models/")
        joblib.dump(model, "./results/" + experiment_name + "/SVM/models/model_%d.pkl" % i_fold)

    print("")
    print("")
    print("Cross-validation Balanced acc: %.4f +- %.4f" % (cv_test_bac.mean(), cv_test_bac.std()))
    print("Cross-validation Sensitivity: %.4f +- %.4f" % (cv_test_sens.mean(), cv_test_sens.std()))
    print("Cross-validation Specificity: %.4f +- %.4f" % (cv_test_spec.mean(), cv_test_spec.std()))
    print("Cross-validation Error Rate: %.4f +- %.4f" % (cv_error_rate.mean(), cv_error_rate.std()))

    if not os.path.exists("./results/" + experiment_name + "/SVM/summary/"):
        os.makedirs("./results/" + experiment_name + "/SVM/summary/")
    np.savez("./results/" + experiment_name + "/SVM/summary/cv_results.npz",
             bac=cv_test_bac, sens=cv_test_sens, spec=cv_test_spec, error_rate=cv_error_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train model.')
    parser.add_argument("config_name", type=str,
                        help="The name of file .py with configurations, e.g., ./config/config_test.py")
    args = parser.parse_args()

    config_name = args.config_name

    try:
        config_module = imp.load_source('config', config_name)

    except IOError:
        print('Cannot open ', config_name,
              '. Please specify the correct path of the configuration file.'
              ' Example: python create_dataset.py ./config/config_test.py')


    main(config_module)
