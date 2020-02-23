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
"""Baseline models."""

import numpy as np
import tensorflow as tf


def build_model(
    encoder,
    input_dtypes,
    input_shapes,
    output_shape,
    output_activation="softmax",
    top_dense_layers=1,
    top_dense_units=128,
    top_dense_activation="relu",
    top_dense_dropout=0.5,
):
    # Input nodes.
    inputs = tf.keras.layers.Input(
        shape=input_shapes["C"], dtype=input_dtypes["C"], name="C"
    )

    # Build top dense layers.
    previous = encoder(inputs)
    for _ in range(top_dense_layers):
        previous = tf.keras.layers.Dense(
            top_dense_units, activation=top_dense_activation
        )(previous)
        previous = tf.keras.layers.Dropout(top_dense_dropout)(previous)

    # Build outputs.
    outputs = tf.keras.layers.Dense(
        np.prod(output_shape), activation=output_activation
    )(previous)
    outputs = tf.keras.layers.Reshape(output_shape, name="Y")(outputs)

    # Create a Keras model.
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model
