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
"""A collection of ResNet models."""

import tensorflow as tf

from tensorflow.keras.applications import resnet50


__all__ = ["ResNet50"]


def ResNet50(pooling=None, weights=None):
    """Builds the standard ResNet50 network with optional top dense layers.

    Args:
        pooling: str, None (default: None)
            See `keras.applications.vgg16`.
        weights: str, None (default: None)
            Whether to initialize the network with pre-trained weights.
            Can be either 'imagenet' or a full path to weights.

    Returns:
        network: function
            Takes input tensors and builds output tensors.
    """

    def network(inputs):
        previous = resnet50.ResNet50(
            include_top=False,
            input_tensor=inputs,
            pooling=pooling,
            weights=weights,
        ).output
        return tf.keras.layers.Flatten(name="flatten")(previous)

    return network
