from __future__ import print_function
import argparse
import imp
import numpy as np
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import StratifiedKFold


from keras_extentions.preprocessing_neuroimage import DataGenerator


def main(args):
    config_name = args.config_name
    try:
        config_module = imp.load_source('config', "./config/" + config_name + ".py")
    except IOError:
        print(
            "Cannot open {}. Please specify the correct name of the configuration file (at the directory ./config). Example: python train.py config_test".format(
                config_name))
    np.random.seed(config_module.N_SEED)

    paths = config_module.path_files
    img_dir = paths["npy_dir"]
    labels_file = paths["labels_file"]
    output_dir = paths["save_file"]

    print("Loading data from : ", img_dir)
    labels = np.genfromtxt(labels_file, delimiter=',', dtype='int8')

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

# -----------------CV VARIABLES ---------------------------------
    print("")
    print("Performing k-fold cross validation")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config_module.N_SEED)


    cv_test_bac = np.zeros((n_folds,))
    cv_test_sens = np.zeros((n_folds,))
    cv_test_spec = np.zeros((n_folds,))
    cv_error_rate = np.zeros((n_folds,))
# ---------------------------------------------------------------
    i = 0
    for train_index, test_index in skf.split(labels, labels):
        nb_train_samples = len(train_index)
        nb_test_samples = len(test_index)

        print("")
        print("k-fold: ", i + 1)
        print("")
        print("Building model")
        model = config_module.get_model()
        print(model.summary())
        model.compile(loss='categorical_crossentropy',
                      optimizer=config_module.get_optimizer(),
                      metrics=['accuracy'])

        # ----------------------- Training -----------------------
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
                                      channel_shift_range=channel_shift_range)

        train_generator = train_datagen.flow_from_directory(
            img_dir,
            subject_index=train_index,
            nb_class=nb_classes,
            batch_size=batch_size)

        print("")
        print("VALIDATION/TEST")  # No Augmentation
        test_datagen = DataGenerator(do_ZMUV=do_zmuv,
                                     image_shape=image_dimension)

        test_generator = test_datagen.flow_from_directory(
            img_dir,
            subject_index=test_index,
            nb_class=nb_classes,
            batch_size=1)

        if do_zmuv:
            print("")
            print("Calculating mean and std for ZMUV Normalization...")
            train_datagen.fit(img_dir, subject_index=train_index)
            test_datagen.mean = train_datagen.mean
            test_datagen.std = train_datagen.std

        print("")
        print("Compiling....")

        #-------------------------------------------------------
        model.fit_generator(train_generator,
                            steps_per_epoch=nb_train_samples/batch_size,
                            epochs=nb_epoch,
                            validation_data=test_generator,
                            validation_steps=nb_test_samples/batch_size,
                            verbose=1)

        # ----------------------- Testing -----------------------
        print("")
        print("Testing with %d subjects." % (nb_test_samples))

        y_predicted = model.predict_generator(test_generator, nb_test_samples)
        y_test = test_generator.get_labels()
        fnames = test_generator.get_names()

        import csv
        if i == 0:
            file_predictions = open(output_dir + 'predictions.csv', 'wb')
            wr = csv.writer(file_predictions)
            wr.writerow(['NAME', 'TRUE LABEL', 'PREDICTED'])
        else:
            file_predictions = open(output_dir + 'predictions.csv', 'a')
        for j, fname in enumerate(fnames):
            wr = csv.writer(file_predictions)
            wr.writerow([fname, y_test[j],(np.argmax(y_predicted, axis=1))[j]])
        file_predictions.close()


        from sklearn.metrics import confusion_matrix
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

        cv_test_bac[i] = test_bac
        cv_test_sens[i] = test_sens
        cv_test_spec[i] = test_spec
        cv_error_rate[i] = error_rate
        #model.save_weights(output_dir+('model_%d.h5' % i))
        i += 1
    print("")
    print("")
    print("Cross-validation balanced acc: %.4f +- %.4f" % (cv_test_bac.mean(), cv_test_bac.std()))
    print("Cross-validation Sensitivity: %.4f +- %.4f" % (cv_test_sens.mean(), cv_test_sens.std()))
    print("Cross-validation Specificity: %.4f +- %.4f" % (cv_test_spec.mean(), cv_test_spec.std()))
    print("Cross-validation Error Rate: %.4f +- %.4f" % (cv_error_rate.mean(), cv_error_rate.std()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train model.')
    parser.add_argument("config_name", type=str, help="The name of file .py with configurations, e.g., Alexnet")
    args = parser.parse_args()
    main(args)
