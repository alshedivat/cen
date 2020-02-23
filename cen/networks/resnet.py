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
