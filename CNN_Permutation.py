from __future__ import print_function
import argparse
import imp
import numpy as np

from keras_extentions.preprocessing_Du import DataGenerator
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler


def main(args):
    config_name = args.config_name

    try:
        config_module = imp.load_source('config', "./config/" + config_name + ".py")
    except IOError:
        print(
            "Cannot open {}. Please specify the correct name of the configuration file (at the directory ./config). Example: python train.py config_test".format(
                config_name))


    # READING DATA -----------------------------------------------------------------------------------------------
    paths = config_module.path_files
    labels_file = paths["labels_file"]
    img_dir = paths["npy_dir"]
    labels = np.genfromtxt(labels_file, delimiter=',', dtype='int8')
    print("Loading data from : ", img_dir)

    # CV PARAMETERS -----------------------------------------------------------------------------------------------
    seed = config_module.N_SEED
    np.random.seed(seed)
    n_repetitions = config_module.n_repetitions
    n_folds = config_module.n_folds
    nb_timesteps = config_module.nb_timesteps
    nb_classes = config_module.nb_classes
    batch_size = config_module.batch_size
    nb_epoch = config_module.nb_epoch
    image_dimension = config_module.image_dimension
    do_zmuv = config_module.do_zmuv

    accumulated_BAC = []
    accumulated_SENS = []
    accumulated_SPEC = []
    accumulated_ERROR = []

    for i_rep in range(n_repetitions):
        # CV  VARIABLES -----------------------------------------------------------------------------------------------
        cv_test_bac_subj = np.zeros((n_folds,))
        cv_test_sens_subj = np.zeros((n_folds,))
        cv_test_spec_subj = np.zeros((n_folds,))
        cv_error_rate_subj = np.zeros((n_folds,))

        # CV  -----------------------------------------------------------------------------------------------
        skf = StratifiedKFold(n_splits=n_folds, random_state=seed + i_rep, shuffle=True)
        i_fold = 0

        np.random.seed(seed + i_rep)
        permuted_labels = np.random.permutation(labels)

        for train_index, test_index in skf.split(labels, labels):
            # SPLITTING DATA  -----------------------------------------------------------------------------------------------
            labels_train, labels_test = permuted_labels[train_index], permuted_labels[test_index]

            print("")
            print("REPETITION: ", i_rep + 1)
            print("k-fold: ", i_fold + 1)
            print("")

            print("SIZE TRAINING SET")
            print("          CLASS 0:", (len(labels_train) - np.count_nonzero(labels_train)))
            print("          CLASS 1:", np.count_nonzero(labels_train))
            print("")
            print("SIZE TEST SET")
            print("          CLASS 0:", (len(labels_test) - np.count_nonzero(labels_test)))
            print("          CLASS 1:", np.count_nonzero(labels_test))
            print("")

            # CREATING MODEL -----------------------------------------------------------------------------------------------
            model = config_module.get_model()
            print(model.summary())
            model.compile(loss='categorical_crossentropy',
                          optimizer=config_module.get_optimizer(),
                          metrics=['accuracy'])

            # Oversampling  -----------------------------------------------------------------------------------------------
            ros = RandomOverSampler()
            train_index_resampled, labels_train_resampled = ros.fit_sample(train_index.reshape((len(train_index),1)), labels_train)



            print("SIZE TRAINING SET AFTER OVERSAMPLING")
            print("          CLASS 0:", (len(labels_train_resampled) - np.count_nonzero(labels_train_resampled)))
            print("          CLASS 1:", np.count_nonzero(labels_train_resampled))
            print("")

            # TRAINING  ----------------------------------------------------------------------------------------------------
            nb_train_samples = len(train_index_resampled) * nb_timesteps
            datagen = DataGenerator(image_dimension, do_zmuv)
            train_generator = datagen.flow_from_directory(
                img_dir,
                subject_index=train_index_resampled,
                nb_timesteps=nb_timesteps,
                nb_class=nb_classes,
                batch_size=batch_size,
                permuted_labels = labels_train_resampled)

            if do_zmuv:
                print("Calculating mean and std for ZMUV Normalization...")
                datagen.fit(img_dir, subject_index=train_index, nb_timesteps=nb_timesteps)

            print('ZMUV DONE')
            nb_test_samples = len(test_index) * nb_timesteps
            test_generator = datagen.flow_from_directory(
                img_dir,
                subject_index=test_index,
                nb_timesteps=nb_timesteps,
                nb_class=nb_classes,
                batch_size=batch_size)

            print("")
            print("Training with %d subjects." % (len(train_index)))
            print("Using %d neuroimages." % (nb_train_samples))
            print("")

            model.fit_generator(train_generator,
                                steps_per_epoch=nb_train_samples/batch_size,
                                epochs=nb_epoch)

            # TESTING ------------------------------------------------------------------------------------------------------
            print("")
            print("Testing with %d subjects." % (len(test_index)))
            print("Using %d samples." % (nb_test_samples))

            predictions = model.predict_generator(test_generator, float(nb_test_samples)/float(batch_size))
            y_test = test_generator.get_labels()

            print(predictions)
            print(y_test)

            from sklearn.metrics import confusion_matrix
            print("")
            print("SAMPLES confusion matrix")
            cm = confusion_matrix(y_test, np.argmax(predictions, axis=1))
            print(cm)

            nb_subj_test = (len(predictions) / nb_timesteps)
            subj_pred = np.zeros((nb_subj_test,))
            subj_class = np.zeros((nb_subj_test,))
            for j in xrange(nb_subj_test):
                subj_prob = np.mean(predictions[j * nb_timesteps:(j + 1) * nb_timesteps])
                subj_class[j] = y_test[j * nb_timesteps]
                if subj_prob >= 0.5:
                    subj_pred[j] = 1
                else:
                    subj_pred[j] = 0
            print("")
            print("SUBJECT confusion matrix")
            cm_subj = confusion_matrix(subj_class, subj_pred)
            print(cm_subj)
            test_bac_subj = np.sum(np.true_divide(np.diagonal(cm_subj), np.sum(cm_subj, axis=1))) / cm_subj.shape[1]
            test_sens_subj = np.true_divide(cm_subj[1, 1], np.sum(cm_subj[1, :]))
            test_spec_subj = np.true_divide(cm_subj[0, 0], np.sum(cm_subj[0, :]))
            error_rate_subj = np.true_divide(cm_subj[0, 1] + cm_subj[1, 0], np.sum(np.sum(cm_subj)))

            print("Balanced acc: %.4f " % (test_bac_subj))
            print("Sensitivity: %.4f " % (test_sens_subj))
            print("Specificity: %.4f " % (test_spec_subj))
            print("Error Rate: %.4f " % (error_rate_subj))

            cv_test_bac_subj[i_fold] = test_bac_subj
            cv_test_sens_subj[i_fold] = test_sens_subj
            cv_test_spec_subj[i_fold] = test_spec_subj
            cv_error_rate_subj[i_fold] = error_rate_subj

            i_fold += 1

        np.savetxt("./logs/perm_CNN_bac_rep{}.csv".format(i_rep), cv_test_bac_subj, delimiter=",")
        np.savetxt("./logs/perm_CNN_sens_rep{}.csv".format(i_rep), cv_test_sens_subj, delimiter=",")
        np.savetxt("./logs/perm_CNN_spec_rep{}.csv".format(i_rep), cv_test_spec_subj, delimiter=",")
        np.savetxt("./logs/perm_CNN_error_rep{}.csv".format(i_rep), cv_error_rate_subj, delimiter=",")

        accumulated_BAC.extend([np.mean(cv_test_bac_subj)])
        accumulated_SENS.extend([np.mean(cv_test_sens_subj)])
        accumulated_SPEC.extend([np.mean(cv_test_spec_subj)])
        accumulated_ERROR.extend([np.mean(cv_error_rate_subj)])

    np.savetxt("./logs/_accumulated_perm_CNN_bac.csv", np.asarray(accumulated_BAC), delimiter=",")
    np.savetxt("./logs/_accumulated_perm_CNN_sens.csv", np.asarray(accumulated_SENS), delimiter=",")
    np.savetxt("./logs/_accumulated_perm_CNN_spec.csv", np.asarray(accumulated_SPEC), delimiter=",")
    np.savetxt("./logs/_accumulated_perm_CNN_error.csv", np.asarray(accumulated_ERROR), delimiter=",")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train model.')
    parser.add_argument("config_name", type=str, help="The name of file .py with configurations, e.g., Alexnet")
    args = parser.parse_args()
    main(args)

