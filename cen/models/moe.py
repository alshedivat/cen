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
"""Mixtures of contextual linear experts."""

import numpy as np
import tensorflow as tf

from .. import layers


def build_model(
    encoder,
    input_dtypes,
    input_shapes,
    output_shape,
    num_experts,
    expert_kwargs,
    mixture_kwargs,
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

    # Experts.
    experts = [
        tf.keras.layers.Dense(np.prod(output_shape), **expert_kwargs)
        for _ in range(num_experts)
    ]

    # Build contextual mixture of experts layer.
    features_flat = tf.keras.layers.Flatten()(features)
    mixture = layers.get_contextual(
        "mixture", experts=experts, **mixture_kwargs
    )
    outputs = mixture((encodings, features_flat))
    outputs = tf.keras.layers.Reshape(output_shape, name="Y")(outputs)

    # Create a Keras model.
    model = tf.keras.models.Model(inputs=(context, features), outputs=outputs)

    return model
