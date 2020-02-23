"""CEN models."""

import numpy as np
import tensorflow as tf

from .. import layers


def build_model(
    encoder,
    input_dtypes,
    input_shapes,
    output_shape,
    explainer_name,
    explainer_kwargs,
    output_activation="softmax",
    top_dense_layers=1,
    top_dense_units=128,
    top_dense_activation="relu",
    top_dense_dropout=0.5,
):
    # Input nodes.
    context = tf.keras.layers.Input(
        shape=input_shapes["C"], dtype=input_dtypes["C"], name="C"
    )
    features = tf.keras.layers.Input(
        shape=input_shapes["X"], dtype=input_dtypes["X"], name="X"
    )

    # Build encoded context.
    encodings = encoder(context)
    for _ in range(top_dense_layers):
        encodings = tf.keras.layers.Dense(
            top_dense_units, activation=top_dense_activation
        )(encodings)
        encodings = tf.keras.layers.Dropout(top_dense_dropout)(encodings)
    encodings = tf.keras.layers.Lambda(lambda x: x, name="E")(encodings)

    # Build contextual explanation layer.
    features_flat = tf.keras.layers.Flatten()(features)
    explainer = layers.get_contextual(
        explainer_name,
        units=np.prod(output_shape),
        activation=output_activation,
        **explainer_kwargs,
    )
    outputs = explainer((encodings, features_flat))
    outputs = tf.keras.layers.Reshape(output_shape, name="Y")(outputs)

    # Create a Keras model.
    model = tf.keras.models.Model(inputs=(context, features), outputs=outputs)

    info = {"context": context, "encodings": encodings}

    return model, info
