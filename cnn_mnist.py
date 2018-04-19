"""
Adapted from TensorFlow MNIST tutorial available: https://www.tensorflow.org/tutorials/layers
Author: Ryan Bury
Class: CMPSC 450 Penn State University

Description: To assess the performance improvement of image recognition convolutional
neural networks on CPU and GPU devices.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.info)

if __name__ == "__main__":
    tf.app.run()

def cnn_model_fn(features, labels, mode):
    #Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu
    )

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

    # Convolutional layer #2 and Pooling layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu
    )

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

    #Dense Layer
    
