# Saliency using guided bacKprop + smoothgrad
from __future__ import print_function
import argparse
import imp
import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.models import load_model
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
    GuidedReluRegistered = False
    def __init__(self, model, output_index=0, custom_loss=None):
        if GuidedBackprop.GuidedReluRegistered is False:
            @tf.RegisterGradient("GuidedRelu")
            def _GuidedReluGrad(op, grad):
                gate_g = tf.cast(grad > 0., tf.float32)
                gate_y = tf.cast(op.outputs[0] > 0., tf.float32)
                return gate_y * gate_g * grad
        GuidedBackprop.GuidedReluRegistered = True

        model.save('/tmp/gb_keras.h5')
        with tf.Graph().as_default():
            with tf.Session().as_default():
                K.set_learning_phase(0)
                load_model('/tmp/gb_keras.h5', custom_objects={"custom_loss": custom_loss})
                session = K.get_session()
                session.run(tf.global_variables_initializer())

                saver = tf.train.Saver()
                saver.save(session, '/tmp/guided_backprop_ckpt')

        self.guided_graph = tf.Graph()
        with self.guided_graph.as_default():
            self.guided_sess = tf.Session(graph=self.guided_graph)

            with self.guided_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
                tf.import_graph_def(session.graph_def, name='')
                saver.restore(self.guided_sess, '/tmp/guided_backprop_ckpt')

                self.imported_y = self.guided_graph.get_tensor_by_name(model.output.name)[0][output_index]
                self.imported_x = self.guided_graph.get_tensor_by_name(model.input.name)

                self.guided_grads_node = tf.gradients(self.imported_y, self.imported_x)

    def get_mask(self, input_image):
        x_value = np.expand_dims(input_image, axis=0)
        guided_feed_dict = {}
        guided_feed_dict[self.imported_x] = x_value

        gradients = self.guided_sess.run(self.guided_grads_node, feed_dict=guided_feed_dict)[0][0]

        return gradients

    def get_smoothed_mask(self, input_image, stdev_spread=.2, nsamples=1):
        stdev = stdev_spread * (np.max(input_image) - np.min(input_image))

        total_gradients = np.zeros_like(input_image)
        for i in range(nsamples):
            noise = np.random.normal(0, stdev, input_image.shape)
            x_value_plus_noise = input_image + noise

            total_gradients += self.get_mask(x_value_plus_noise)

        return total_gradients / nsamples


def main(args):
    config_name = args.config_name
    try:
        config_module = imp.load_source('config', "./config/" + config_name + ".py")
    except IOError:
        print(
            "Cannot open {}. Please specify the correct name of the configuration file (at the directory ./config). Example: python train.py config_test".format(
                config_name))

    paths = config_module.path_files

    save_dir = paths["npy_dir"]
    image_dimension = config_module.image_dimension

    paths_train = []
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            if file.endswith(".npz"):
                paths_train.append(os.path.join(root, file))
    filenames = sort_nicely(paths_train)

    model = load_model('./models/model_1.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    guided_bprop_healthy = GuidedBackprop(model,output_index=0)
    guided_bprop_patient = GuidedBackprop(model, output_index=1)

    labels = np.zeros((len(filenames)),dtype='float32')
    masks = np.zeros((len(filenames),)+image_dimension,dtype='float32')

    for i in range(len(filenames)):
        print(i)
        fname = filenames[i]
        file_npz = np.load(fname)
        img = file_npz['image']
        label = file_npz['label']
        if label == 0:
            mask = guided_bprop_healthy.get_smoothed_mask(img)
        else:
            mask = guided_bprop_patient.get_smoothed_mask(img)

        labels[i] = label
        masks[i] = np.reshape(mask, (mask.shape[0], mask.shape[1], mask.shape[2]))

    heathy_img = np.mean(masks[labels ==0], axis=0)
    heathy_img -= np.min(heathy_img)
    heathy_img /= np.max(heathy_img)
    heathy_img *= 255
    img_nib = nib.Nifti1Image(heathy_img, affine=np.eye(4))
    nib.save(img_nib, './saliency/heathy.nii')

    patient_img = np.mean(masks[labels == 1], axis=0)
    patient_img -= np.min(patient_img)
    patient_img /= np.max(patient_img)
    patient_img *= 255
    img_nib = nib.Nifti1Image(patient_img, affine=np.eye(4))
    nib.save(img_nib, './saliency/patient.nii')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train model.')
    parser.add_argument("config_name", type=str, help="The name of file .py with configurations, e.g., Alexnet")
    args = parser.parse_args()
    main(args)