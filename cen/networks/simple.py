"""A collection of simple networks."""

import tensorflow as tf


__all__ = ["CNN", "MLP", "Identity"]


def CNN(
    blocks=1,
    filters=64,
    kernel_size=(3, 3),
    conv_strides=(1, 1),
    pool_strides=None,
    pool_size=(2, 2),
    padding="valid",
    activation="relu",
    dropout=0.0,
):
    """Simple multi-layer conv net."""

    def network(inputs):
        previous = inputs
        for _ in range(blocks):
            previous = tf.keras.layers.Conv2D(
                filters=filters,
                activation=activation,
                kernel_size=kernel_size,
                strides=conv_strides,
                padding=padding,
            )(previous)
            previous = tf.keras.layers.Conv2D(
                filters=filters,
                activation=activation,
                kernel_size=kernel_size,
                strides=conv_strides,
                padding=padding,
            )(previous)
            previous = tf.keras.layers.MaxPool2D(
                pool_size=pool_size, padding=padding, strides=pool_strides
            )(previous)
            previous = tf.keras.layers.Dropout(dropout)(previous)
        return tf.keras.layers.Flatten()(previous)

    return network


def MLP(blocks=2, units=64, activation="relu", dropout=0.0):
    """Simple multi-layer MLP."""

    def network(inputs):
        previous = inputs
        for _ in range(blocks):
            previous = tf.keras.layers.Dense(
                units=units, activation=activation
            )(previous)
            previous = tf.keras.layers.Dropout(dropout)(previous)
        return previous

    return network


def Identity():
    """Simple identity encoder."""

    def network(inputs):
        return inputs

    return network
