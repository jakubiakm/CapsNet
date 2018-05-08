from __future__ import division, print_function, unicode_literals
import tensorflow as tf
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from config import cfg
from utils import load_data

def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                    keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector

def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                    keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)
class CapsNet:
    def __init__(self):
        
        tf.reset_default_graph()
        np.random.seed(42)
        tf.set_random_seed(42)



    
        # n_samples = 15

        # plt.figure(figsize=(n_samples * 2, 3))
        # for index in range(n_samples):
        #     plt.subplot(1, n_samples, index + 1)
        #     sample_image = mnist.train.images[index].reshape(28, 28)
        #     plt.imshow(sample_image, cmap="binary")
        #     plt.axis("off")

        # plt.show()

        # mnist.train.labels[:n_samples]


        #placeholder na literki 28x28 pixeli
        self.X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")

        #The first layer will be composed of 32 maps of 6×6 capsules each, where each capsule will output an 8D activation vector:
        caps1_n_maps = 32
        caps1_n_caps = caps1_n_maps * 6 * 6  # 1152 primary capsules
        caps1_n_dims = 8

        #To compute their outputs, we first apply two regular convolutional layers:
        conv1_params = {
            "filters": 256,
            "kernel_size": 9,
            "strides": 1,
            "padding": "valid",
            "activation": tf.nn.relu,
        }

        conv2_params = {
            "filters": caps1_n_maps * caps1_n_dims, # 256 convolutional filters
            "kernel_size": 9,
            "strides": 2,
            "padding": "valid",
            "activation": tf.nn.relu
        }

        conv1 = tf.layers.conv2d(self.X, name="conv1", **conv1_params)
        conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)
        #Note: since we used a kernel size of 9 and no padding (for some reason, that's what `"valid"` means), the image shrunk by 9-1=8 pixels after each convolutional layer (28×28 to 20×20, then 20×20 to 12×12), and since we used a stride of 2 in the second convolutional layer, the image size was divided by 2. This is how we end up with 6×6 feature maps.
        #Next, we reshape the output to get a bunch of 8D vectors representing the outputs of the primary capsules. The output of `conv2` is an array containing 32×8=256 feature maps for each instance, where each feature map is 6×6. So the shape of this output is (_batch size_, 6, 6, 256). We want to chop the 256 into 32 vectors of 8 dimensions each. We could do this by reshaping to (_batch size_, 6, 6, 32, 8). However, since this first capsule layer will be fully connected to the next capsule layer, we can simply flatten the 6×6 grids. This means we just need to reshape to (_batch size_, 6×6×32, 8).

        caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims],name="caps1_raw")

        caps1_output = squash(caps1_raw, name="caps1_output")

        init_sigma = 0.1

        caps2_n_caps = 10
        caps2_n_dims = 16

        W_init = tf.random_normal(
            shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
            stddev=init_sigma, dtype=tf.float32, name="W_init")
        W = tf.Variable(W_init, name="W")

        #cfg.batch_size = tf.shape(X)[0]
        W_tiled = tf.tile(W, [cfg.batch_size, 1, 1, 1, 1], name="W_tiled")

        caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                            name="caps1_output_expanded")
        caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                        name="caps1_output_tile")
        caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],
                                    name="caps1_output_tiled")

        caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name="caps2_predicted")


        raw_weights = tf.zeros([cfg.batch_size, caps1_n_caps, caps2_n_caps, 1, 1], dtype=np.float32, name="raw_weights")

        routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")

        weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                        name="weighted_predictions")
        weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                                    name="weighted_sum")
        caps2_output_round_1 = squash(weighted_sum, axis=-2,
                                    name="caps2_output_round_1")

        caps2_output_round_1_tiled = tf.tile(
            caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],
            name="caps2_output_round_1_tiled")

        agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                            transpose_a=True, name="agreement")

        raw_weights_round_2 = tf.add(raw_weights, agreement,
                                    name="raw_weights_round_2")

        routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
                                                dim=2,
                                                name="routing_weights_round_2")
        weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
                                                caps2_predicted,
                                                name="weighted_predictions_round_2")
        weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
                                            axis=1, keep_dims=True,
                                            name="weighted_sum_round_2")
        caps2_output_round_2 = squash(weighted_sum_round_2,
                                    axis=-2,
                                    name="caps2_output_round_2")

        self.caps2_output = caps2_output_round_2

        y_proba = safe_norm(self.caps2_output, axis=-2, name="y_proba")

        y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")

        self.y_pred = tf.squeeze(y_proba_argmax, axis=[1,2], name="y_pred")

        self.y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

        m_plus = 0.9
        m_minus = 0.1
        lambda_ = 0.5

        T = tf.one_hot(self.y, depth=caps2_n_caps, name="T")

        caps2_output_norm = safe_norm(self.caps2_output, axis=-2, keep_dims=True,
                                    name="caps2_output_norm")

        present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),
                                    name="present_error_raw")
        present_error = tf.reshape(present_error_raw, shape=(-1, 10),
                                name="present_error")

        absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),
                                    name="absent_error_raw")
        absent_error = tf.reshape(absent_error_raw, shape=(-1, 10),
                                name="absent_error")

        L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
                name="L")

        margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

        self.mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                                    name="mask_with_labels")

        reconstruction_targets = tf.cond(self.mask_with_labels, # condition
                                        lambda: self.y,        # if True
                                        lambda: self.y_pred,   # if False
                                        name="reconstruction_targets")

        reconstruction_mask = tf.one_hot(reconstruction_targets,
                                        depth=caps2_n_caps,
                                        name="reconstruction_mask")

        reconstruction_mask_reshaped = tf.reshape(
            reconstruction_mask, [-1, 1, caps2_n_caps, 1, 1],
            name="reconstruction_mask_reshaped")

        caps2_output_masked = tf.multiply(
            self.caps2_output, reconstruction_mask_reshaped,
            name="caps2_output_masked")

        decoder_input = tf.reshape(caps2_output_masked,
                                [-1, caps2_n_caps * caps2_n_dims],
                                name="decoder_input")

        n_hidden1 = 512
        n_hidden2 = 1024
        n_output = 28 * 28

        with tf.name_scope("decoder"):
            hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                                    activation=tf.nn.relu,
                                    name="hidden1")
            hidden2 = tf.layers.dense(hidden1, n_hidden2,
                                    activation=tf.nn.relu,
                                    name="hidden2")
            self.decoder_output = tf.layers.dense(hidden2, n_output,
                                            activation=tf.nn.sigmoid,
                                            name="decoder_output")

        X_flat = tf.reshape(self.X, [-1, n_output], name="X_flat")
        squared_difference = tf.square(X_flat - self.decoder_output,
                                    name="squared_difference")
        reconstruction_loss = tf.reduce_mean(squared_difference,
                                            name="reconstruction_loss")

        alpha = 0.0005

        self.loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")

        correct = tf.equal(self.y, self.y_pred, name="correct")
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

        optimizer = tf.train.AdamOptimizer()
        self.training_op = optimizer.minimize(self.loss, name="training_op")
        self.saver = tf.train.Saver()
        