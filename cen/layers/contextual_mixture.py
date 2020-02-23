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
"""Contextual Mixture of Experts."""

import tensorflow as tf

from tensorflow.python.framework import tensor_shape

from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import InputSpec
from tensorflow.python.keras.layers import Layer


class ContextualMixture(Layer):
    """
    The layer that represents a contextual mixture.

    Internally, the gating mechanism is represented by a dense layer built on
    the contextual inputs which softly combines the outputs of each expert.
    Each expert can be an arbitrary network as long as it is compatible with
    the features as inputs.
    """

    def __init__(
        self,
        experts,
        activity_regularizer=None,
        gate_use_bias=False,
        gate_kernel_initializer="glorot_uniform",
        gate_bias_initializer="zeros",
        gate_kernel_regularizer=None,
        gate_bias_regularizer=None,
        gate_kernel_constraint=None,
        gate_bias_constraint=None,
        **kwargs,
    ):
        super(ContextualMixture, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs,
        )

        # Sanity check.
        self.experts = tuple(experts)
        for expert in self.experts:
            if not isinstance(expert, tf.Module):
                raise ValueError(
                    "Please initialize `{name}` expert with a "
                    "`tf.Module` instance. You passed: {input}".format(
                        name=self.__class__.__name__, input=expert
                    )
                )

        # Regularizers and constraints for the weight generator.
        self.gate_use_bias = gate_use_bias
        self.gate_kernel_initializer = initializers.get(gate_kernel_initializer)
        self.gate_bias_initializer = initializers.get(gate_bias_initializer)
        self.gate_kernel_regularizer = regularizers.get(gate_kernel_regularizer)
        self.gate_bias_regularizer = regularizers.get(gate_bias_regularizer)
        self.gate_kernel_constraint = constraints.get(gate_kernel_constraint)
        self.gate_bias_constraint = constraints.get(gate_bias_constraint)

        self.supports_masking = True
        self.input_spec = [
            InputSpec(min_ndim=2),  # Context input spec.
            InputSpec(min_ndim=2),  # Features input spec.
        ]

        # Instantiate contextual attention for gating.
        self.gating_attention = Dense(
            len(self.experts), activation=tf.nn.softmax, name="attention"
        )

        # Internals.
        self.context_shape = None
        self.feature_shape = None

    def _build_sanity_check(self, context_shape, feature_shape):
        dtype = tf.dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "Unable to build `ContextualDense` layer with "
                f"non-floating point dtype {dtype}."
            )
        context_shape = tensor_shape.TensorShape(context_shape)
        if tensor_shape.dimension_value(context_shape[-1]) is None:
            raise ValueError(
                "The last dimension of the context "
                "should be defined. Found `None`."
            )
        feature_shape = tensor_shape.TensorShape(feature_shape)
        if tensor_shape.dimension_value(feature_shape[-1]) is None:
            raise ValueError(
                "The last dimension of the features "
                "should be defined. Found `None`."
            )

        # Ensure that output shapes of all experts are identical.
        expected_shape = self.experts[0].compute_output_shape(feature_shape)
        for expert in self.experts[1:]:
            expert_output_shape = expert.compute_output_shape(feature_shape)
            if expert_output_shape != expected_shape:
                raise ValueError(
                    f"Experts have different output shapes! "
                    f"Expected shape: {expected_shape}. "
                    f"Found expert {expert} with shape: {expert_output_shape}."
                )

    def build(self, input_shape=None):
        context_shape, feature_shape = input_shape

        # Sanity checks.
        self._build_sanity_check(context_shape, feature_shape)

        # Build gating attention.
        self.gating_attention.build(context_shape)

        # Build the wrapped layer.
        for expert in self.experts:
            expert.build(feature_shape)

        self.built = True

    def call(self, inputs, **kwargs):
        context, features = inputs
        context = tf.convert_to_tensor(context)
        features = tf.convert_to_tensor(features)

        # Compute outputs for each expert.
        # <float32> [batch_size, num_experts, units].
        expert_outputs = tf.stack(
            [expert(features) for expert in self.experts], axis=1
        )

        # Compute gating attention.
        # <float32> [batch_size, num_experts, 1].
        gating_attention = tf.expand_dims(self.gating_attention(context), -1)

        # Compute output as attention-weighted linear combination.
        # <float32> [batch_size, units].
        outputs = tf.reduce_sum(gating_attention * expert_outputs, axis=1)

        return outputs

    def compute_output_shape(self, input_shape):
        return self.experts[0].compute_output_shape(input_shape)
