#  Copyright 2020 Maruan Al-Shedivat. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  =============================================================================
"""VGG16 networks."""

import tensorflow as tf

from tensorflow.keras.applications import vgg16


__all__ = ["VGG16", "VGG16_MNIST"]


def VGG16(pooling=None, weights=None):
    """Builds the standard VGG16 network with optional top dense layers.

    Args:
        pooling : str, None (default: None)
            See `keras.applications.vgg16`.
        weights : str, None (default: None)
            Whether to initialize the network with pre-trained weights.
            Can be either 'imagenet' or a full path to weights.

    Returns:
        network: function
            Takes input tensors and builds output tensors.
    """

    def network(inputs):
        previous = vgg16.VGG16(
            include_top=False,
            input_tensor=inputs,
            pooling=pooling,
            weights=weights,
        ).output
        return tf.keras.layers.Flatten(name="flatten")(previous)

    return network


def VGG16_MNIST():
    """Reduced VGG16 architecture for MNIST.

    Source: https://github.com/kkweon/mnist-competition/blob/master/vgg16.py.
    Note: to reproduce their results requires data augmentation.

    Returns:
        network: function
            Takes input tensors and builds output tensors.
    """

    def two_conv_pool(x, F1, F2, name):
        x = tf.keras.layers.Conv2D(
            F1, (3, 3), activation=None, padding="same", name=f"{name}_conv1"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(
            F2, (3, 3), activation=None, padding="same", name=f"{name}_conv2"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D(
            (2, 2), strides=(2, 2), name=f"{name}_pool"
        )(x)
        return x

    def three_conv_pool(x, F1, F2, F3, name):
        x = tf.keras.layers.Conv2D(
            F1, (3, 3), activation=None, padding="same", name=f"{name}_conv1"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(
            F2, (3, 3), activation=None, padding="same", name=f"{name}_conv2"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(
            F3, (3, 3), activation=None, padding="same", name=f"{name}_conv3"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D(
            (2, 2), strides=(2, 2), name=f"{name}_pool"
        )(x)
        return x

    def network(inputs):
        previous = inputs
        previous = two_conv_pool(previous, 64, 64, "block1")
        previous = two_conv_pool(previous, 128, 128, "block2")
        previous = three_conv_pool(previous, 256, 256, 256, "block3")
        previous = three_conv_pool(previous, 512, 512, 512, "block4")
        return tf.keras.layers.Flatten()(previous)

    return network
