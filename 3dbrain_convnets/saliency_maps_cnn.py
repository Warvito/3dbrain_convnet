"""
Saliency using guided bacKprop + smoothgrad

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import imp
import time

import tensorflow as tf
import numpy as np
import random as rn
import keras.backend as K
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold

import re
import os
import nibabel as nib

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def sort_nicely(l):
    return sorted(l, key=alphanum_key)



class GuidedBackprop():
    """A SaliencyMask class that computes saliency masks with GuidedBackProp.

    This implementation copies the TensorFlow graph to a new graph with the ReLU
    gradient overwritten as in the paper:
    https://arxiv.org/abs/1412.6806
    """

    GuidedReluRegistered = False

    def __init__(self, model, output_index=0, custom_loss=None):
        """Constructs a GuidedBackprop SaliencyMask."""

        if GuidedBackprop.GuidedReluRegistered is False:
            @tf.RegisterGradient("GuidedRelu")
            def _GuidedReluGrad(op, grad):
                gate_g = tf.cast(grad > 0, "float32")
                gate_y = tf.cast(op.outputs[0] > 0, "float32")
                return gate_y * gate_g * grad
        GuidedBackprop.GuidedReluRegistered = True

        """ 
            Create a dummy session to set the learning phase to 0 (test mode in keras) without 
            inteferring with the session in the original keras model. This is a workaround
            for the problem that tf.gradients returns error with keras models that contains 
            Dropout or BatchNormalization.

            Basic Idea: save keras model => create new keras model with learning phase set to 0 => save
            the tensorflow graph => create new tensorflow graph with ReLU replaced by GuiededReLU.
        """
        model.save('/tmp/gb_keras.h5')
        with tf.Graph().as_default():
            with tf.Session().as_default():
                K.set_learning_phase(0)
                load_model('/tmp/gb_keras.h5', custom_objects={"custom_loss": custom_loss})
                session = K.get_session()
                tf.train.export_meta_graph()

                saver = tf.train.Saver()
                saver.save(session, '/tmp/guided_backprop_ckpt')

        self.guided_graph = tf.Graph()
        with self.guided_graph.as_default():
            self.guided_sess = tf.Session(graph=self.guided_graph)

            with self.guided_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
                saver = tf.train.import_meta_graph('/tmp/guided_backprop_ckpt.meta')
                saver.restore(self.guided_sess, '/tmp/guided_backprop_ckpt')

                self.imported_y = self.guided_graph.get_tensor_by_name(model.output.name)[0][output_index]
                self.imported_x = self.guided_graph.get_tensor_by_name(model.input.name)

                self.guided_grads_node = tf.gradients(self.imported_y, self.imported_x)

    def get_mask(self, input_image):
        """Returns a GuidedBackprop mask."""
        x_value = np.expand_dims(input_image, axis=0)
        guided_feed_dict = {}
        guided_feed_dict[self.imported_x] = x_value

        gradients = self.guided_sess.run(self.guided_grads_node, feed_dict=guided_feed_dict)[0][0]

        return gradients

    def get_smoothed_mask(self, input_image, stdev_spread=.2, nsamples=50):
        """Returns a mask that is smoothed with the SmoothGrad method.

        Args:
            input_image: input image with shape (H, W, 3).
        """
        stdev = stdev_spread * (np.max(input_image) - np.min(input_image))

        total_gradients = np.zeros_like(input_image)
        for i in range(nsamples):
            noise = np.random.normal(0, stdev, input_image.shape)
            x_value_plus_noise = input_image + noise

            total_gradients += self.get_mask(x_value_plus_noise)

        return total_gradients / nsamples



def main(config_module):
    # ------------------------------- Reproducibility -----------------------------------------------
    os.environ['PYTHONHASHSEED'] = '0'

    seed = 1
    np.random.seed(seed)
    rn.seed(seed)
    tf.set_random_seed(seed)

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    # ------------------------------- Experiment data --------------------------------------------------
    experiment_name = config_module.experiment_name
    print("Calculating saliency maps")

    img_npz_dir = "./results/" + experiment_name + "/CNN/img_npz/"
    image_dimension = config_module.image_dimension

    paths_train = []
    for root, dirs, files in os.walk(img_npz_dir):
        for file in files:
            if file.endswith(".npz"):
                paths_train.append(os.path.join(root, file))
    filenames = sort_nicely(paths_train)

    paths = config_module.path_files
    labels_file = paths["labels_file"]
    labels = np.genfromtxt(labels_file, delimiter=',', dtype='int8')

    # ------------------------------ Cross validation ----------------------------------------------
    n_folds = 10
    print("")
    print("Performing k-fold cross validation")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # ---------------------------------------------------------------------------------------------
    final_saliency = []

    for i, (train_index, test_index) in enumerate(skf.split(labels, labels)):
        start_time = time.time()
        print("fold ", i)
        # ------------------------------- Reproducibility -----------------------------------------------
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

        # ---------------------------------------------------------------------------------------------
        model = load_model( "./results/" + experiment_name + "/CNN/models/model_"+str(i)+".h5")
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        guided_bprop = GuidedBackprop(model,output_index=1)

        masks = np.zeros((len(filenames),)+image_dimension,dtype='float32')

        for i in range(len(filenames)):
            print(i)
            fname = filenames[i]
            file_npz = np.load(fname)
            img = file_npz['image']
            mask = guided_bprop.get_smoothed_mask(img)
            masks[i] = np.reshape(mask, (mask.shape[0], mask.shape[1], mask.shape[2]))

        saliency_maps = np.mean(masks, axis=0)
        saliency_maps -= np.min(saliency_maps)
        saliency_maps /= np.max(saliency_maps)
        saliency_maps *= 255
        img_nib = nib.Nifti1Image(saliency_maps, affine=np.eye(4))
        nib.save(img_nib, "./results/" + experiment_name + "/CNN/saliency/map_"+str(i) +".nii")
        final_saliency.append(saliency_maps)

        tf.reset_default_graph()
        stop_time = time.time()
        print("ETA: ", (i - n_folds) * (stop_time - start_time), " seconds")

    final_saliency = np.asarray(final_saliency)
    img_nib = nib.Nifti1Image(np.mean(final_saliency, axis=0), affine=np.eye(4))
    nib.save(img_nib, "./results/" + experiment_name + "/CNN/saliency/final_saliency_map.nii")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train model.')
    parser.add_argument("config_name", type=str, help="The name of file .py with configurations, e.g., Alexnet")
    args = parser.parse_args()
    config_name = args.config_name
    try:
        config_module = imp.load_source('config', "./config/" + config_name + ".py")
    except IOError:
        print(
            "Cannot open {}. Please specify the correct name of the configuration file (at the directory ./config). Example: python train.py config_test".format(
                config_name))


    main(config_module)