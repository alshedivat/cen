"""A collection of DenseNet models."""

import tensorflow as tf

from tensorflow.keras.applications import densenet


__all__ = ["DenseNet"]


def DenseNet(blocks=[6, 12, 24, 16], pooling=None, weights=None):
    """Builds the standard ResNet50 network with optional top dense layers.

    Args:
        blocks: list of ints (default: [6, 12, 24, 16])
            Numbers of dense blocks.
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
        previous = densenet.DenseNet(
            blocks=blocks,
            include_top=False,
            input_tensor=inputs,
            pooling=pooling,
            weights=weights,
        ).output
        return tf.keras.layers.Flatten(name="flatten")(previous)

    return network
