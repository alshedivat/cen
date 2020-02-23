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
"""Contextual Dense layers."""

import tensorflow as tf

from tensorflow.python.framework import tensor_shape

from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import InputSpec
from tensorflow.python.keras.layers import Layer

__all__ = ["ContextualDense", "ContextualAffineDense", "ContextualConvexDense"]


class ContextualDense(Layer):
    """
    The base class for contextual Dense layers.

    The weights of the layer (kernel and bias) are tensor-valued functions of
    the context representation.
    """

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        activity_regularizer=None,
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super(ContextualDense, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs,
        )

        self.units = int(units)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.regularizers = {
            "kernels": regularizers.get(kernel_regularizer),
            "biases": regularizers.get(bias_regularizer),
        }
        self.constraints = {
            "kernels": constraints.get(kernel_constraint),
            "biases": constraints.get(bias_constraint),
        }

        self.supports_masking = True
        self.input_spec = [
            InputSpec(min_ndim=2),  # Context input spec.
            InputSpec(min_ndim=2),  # Features input spec.
        ]

        # Internals.
        self.context_dim = None
        self.feature_dim = None

    def build(self, input_shape=None):
        context_shape, feature_shape = input_shape

        # Sanity checks.
        self._build_sanity_check(context_shape, feature_shape)

        # Update input spec.
        self.context_dim = tensor_shape.dimension_value(context_shape[-1])
        self.feature_dim = tensor_shape.dimension_value(feature_shape[-1])
        self.input_spec[0] = InputSpec(min_ndim=2, axes={-1: self.context_dim})
        self.input_spec[1] = InputSpec(min_ndim=2, axes={-1: self.feature_dim})

        # Build contextual weight generator.
        self.build_weight_generator(context_shape, feature_shape)

        self.built = True

    def _build_sanity_check(self, context_shape, feature_shape):
        dtype = tf.dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "Unable to build `ContextualDense` layer with "
                "non-floating point dtype %s" % (dtype,)
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

    def build_weight_generator(self, context_shape, feature_shape):
        raise NotImplementedError("Abstract Method")

    def generate_contextual_weights(self, context):
        raise NotImplementedError("Abstract Method")

    def contextual_dense_outputs(self, contextual_weights, features):
        """Computes contextual outputs."""
        outputs = tf.einsum(
            "ijk,ij->ik", contextual_weights["kernels"], features
        )
        if self.use_bias:
            outputs = tf.add(outputs, contextual_weights["biases"])
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def call(self, inputs, **kwargs):
        context, features = inputs
        context = tf.convert_to_tensor(context)
        features = tf.convert_to_tensor(features)

        # Build contextual weights:
        #   kernels: <float32> [batch_size, feature_dim, units].
        #   biases: <float32> [batch_size, units].
        contextual_weights = self.generate_contextual_weights(context)

        # Add regularizers.
        for name in ["kernels", "biases"]:
            if self.regularizers[name] is not None:
                self.add_loss(self.regularizers[name](contextual_weights[name]))

        # Compute outputs: <float32> [batch_size, units].
        return self.contextual_dense_outputs(contextual_weights, features)

    def compute_output_shape(self, input_shape):
        input_shapes = dict(zip(["context", "features"], input_shape))
        for key, shape in input_shapes.items():
            shape = tensor_shape.TensorShape(shape)
            shape = shape.with_rank_at_least(2)
            if shape[-1].value is None:
                raise ValueError(
                    "The innermost dimension of %s_shape must be defined, "
                    "but saw: %s" % (key, shape)
                )
            input_shapes[key] = shape
        return input_shapes["features"][:-1].concatenate(self.output_dim)

    def get_config(self):
        config = {
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "activity_regularizer": regularizers.serialize(
                self.activity_regularizer
            ),
            "kernel_regularizer": regularizers.serialize(
                self.regularizers["kernels"]
            ),
            "bias_regularizer": regularizers.serialize(
                self.regularizers["biases"]
            ),
            "kernel_constraint": constraints.serialize(
                self.constraints["kernels"]
            ),
            "bias_constraint": constraints.serialize(
                self.constraints["biases"]
            ),
        }
        base_config = super(ContextualDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ContextualAffineDense(ContextualDense):
    """
    Contextual Dense layer that generates weights using affine functions:
            kernel(context) = affine_func_kernel(context)
            bias(context) = affine_func_bias(context)
    """

    def __init__(
        self,
        units,
        gen_use_bias=False,
        gen_kernel_initializer="glorot_uniform",
        gen_bias_initializer="glorot_uniform",
        gen_kernel_regularizer=None,
        gen_bias_regularizer=None,
        gen_kernel_constraint=None,
        gen_bias_constraint=None,
        **kwargs,
    ):
        super(ContextualAffineDense, self).__init__(units, **kwargs)
        self.gen_use_bias = gen_use_bias

        # Regularizers and constraints for the weight generator.
        self.gen_kernel_initializer = initializers.get(gen_kernel_initializer)
        self.gen_bias_initializer = initializers.get(gen_bias_initializer)
        self.gen_kernel_regularizer = regularizers.get(gen_kernel_regularizer)
        self.gen_bias_regularizer = regularizers.get(gen_bias_regularizer)
        self.gen_kernel_constraint = constraints.get(gen_kernel_constraint)
        self.gen_bias_constraint = constraints.get(gen_bias_constraint)

        # Internals.
        self.gen_kernel_weights = None
        self.gen_bias_weights = None

    def build_weight_generator(self, context_shape, feature_shape):
        # Build generator kernels.
        self.gen_kernel_weights = {
            "kernels": self.add_weight(
                "gen_kernel_kernel",
                shape=(self.context_dim, self.feature_dim * self.units),
                initializer=self.gen_kernel_initializer,
                regularizer=self.gen_kernel_regularizer,
                constraint=self.gen_kernel_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        }
        if self.use_bias:
            self.gen_kernel_weights["biases"] = self.add_weight(
                "gen_kernel_bias",
                shape=(self.context_dim, self.units),
                initializer=self.gen_kernel_initializer,
                regularizer=self.gen_kernel_regularizer,
                constraint=self.gen_kernel_constraint,
                dtype=self.dtype,
                trainable=True,
            )

        # Build generator biases, if necessary.
        if self.gen_use_bias:
            self.gen_bias_weights = {
                "kernels": self.add_weight(
                    f"gen_bias_kernel",
                    shape=(self.feature_dim * self.units,),
                    initializer=self.gen_bias_initializer,
                    regularizer=self.gen_bias_regularizer,
                    constraint=self.gen_bias_constraint,
                    dtype=self.dtype,
                    trainable=True,
                )
            }
            if self.use_bias:
                self.gen_bias_weights["biases"] = self.add_weight(
                    "gen_bias_bias",
                    shape=(self.units,),
                    initializer=self.gen_bias_initializer,
                    regularizer=self.gen_bias_regularizer,
                    constraint=self.gen_bias_constraint,
                    dtype=self.dtype,
                    trainable=True,
                )

        self.built = True

    def generate_contextual_weights(self, context):
        """Generates contextual weights for the Dense layer."""
        # <float32> [batch_size, context_dim].
        context = tf.convert_to_tensor(context)

        contextual_weights = {
            # <float32> [batch_size, kernel_dim].
            name: tf.tensordot(context, kernel, [[-1], [0]])
            for name, kernel in self.gen_kernel_weights.items()
        }
        if self.gen_use_bias:
            for name, bias in self.gen_bias_weights.items():
                # kernel: <float32> [batch_size, feature_dim, units].
                # bias: <float32> [batch_size, units].
                contextual_weights[name] = tf.add(
                    contextual_weights[name], tf.expand_dims(bias, 0)
                )

        # Reshape contextual kernels appropriately.
        contextual_weights["kernels"] = tf.reshape(
            contextual_weights["kernels"], (-1, self.feature_dim, self.units)
        )

        return contextual_weights

    def get_config(self):
        config = {
            "gen_use_bias": self.gen_use_bias,
            "gen_kernel_initializer": initializers.serialize(
                self.gen_kernel_initializer
            ),
            "gen_bias_initializer": initializers.serialize(
                self.gen_bias_initializer
            ),
            "gen_kernel_regularizer": regularizers.serialize(
                self.gen_kernel_regularizer
            ),
            "gen_bias_regularizer": regularizers.serialize(
                self.gen_bias_regularizer
            ),
            "gen_kernel_constrain": regularizers.serialize(
                self.gen_kernel_constraint
            ),
            "gen_bias_constraint": regularizers.serialize(
                self.gen_bias_constraint
            ),
        }
        base_config = super(ContextualAffineDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ContextualConvexDense(ContextualDense):
    """
    Contextual Dense layer that generates weights using a convex combination
    of Dense models from a dictionary.
    """

    def __init__(
        self,
        units,
        dict_size,
        dict_kernel_initializer="glorot_uniform",
        dict_bias_initializer="glorot_uniform",
        dict_kernel_regularizer=None,
        dict_bias_regularizer=None,
        dict_kernel_constraint=None,
        dict_bias_constraint=None,
        selector_use_bias=True,
        **kwargs,
    ):
        super(ContextualConvexDense, self).__init__(units, **kwargs)

        # Regularizers and constraints for the weight generator.
        self.dict_size = dict_size
        self.dict_kernel_initializer = initializers.get(dict_kernel_initializer)
        self.dict_bias_initializer = initializers.get(dict_bias_initializer)
        self.dict_kernel_regularizer = regularizers.get(dict_kernel_regularizer)
        self.dict_bias_regularizer = regularizers.get(dict_bias_regularizer)
        self.dict_kernel_constraint = constraints.get(dict_kernel_constraint)
        self.dict_bias_constraint = constraints.get(dict_bias_constraint)
        self.selector_use_bias = selector_use_bias

        # Contextual (soft) model selector from the dictionary.
        self.selector = Dense(
            self.dict_size,
            use_bias=self.selector_use_bias,
            activation="softmax",
            name="selector",
        )

        # Internal.
        self.dict_weights = None

    def build_weight_generator(self, context_shape, feature_shape):
        # Build attention.
        self.selector.build(context_shape)

        # Build dictionary of weights.
        self.dict_weights = {
            "kernels": self.add_weight(
                "kernels",
                shape=(self.dict_size, self.feature_dim * self.units),
                initializer=self.dict_kernel_initializer,
                regularizer=self.dict_kernel_regularizer,
                constraint=self.dict_kernel_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        }

        # Build dictionary of biases, if necessary.
        if self.use_bias:
            self.dict_weights["biases"] = self.add_weight(
                "biases",
                shape=(self.dict_size, self.units),
                initializer=self.dict_bias_initializer,
                regularizer=self.dict_bias_regularizer,
                constraint=self.dict_bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )

        self.built = True

    def generate_contextual_weights(self, context):
        context = tf.convert_to_tensor(context)

        # Compute attention over the dictionary elements.
        attention = self.selector(context)

        # Compute contextual weights.
        contextual_weights = {
            # <float32> [batch_size, weights_dim].
            name: tf.tensordot(attention, weights, [[-1], [0]])
            for name, weights in self.dict_weights.items()
        }

        # Reshape contextual kernels appropriately.
        contextual_weights["kernels"] = tf.reshape(
            contextual_weights["kernels"], (-1, self.feature_dim, self.units)
        )

        return contextual_weights

    def get_config(self):
        config = {
            "dict_size": self.dict_size,
            "dict_kernel_initializer": initializers.serialize(
                self.dict_kernel_initializer
            ),
            "dict_bias_initializer": initializers.serialize(
                self.dict_bias_initializer
            ),
            "dict_kernel_regularizer": regularizers.serialize(
                self.dict_kernel_regularizer
            ),
            "dict_bias_regularizer": regularizers.serialize(
                self.dict_bias_regularizer
            ),
            "dict_kernel_constrain": regularizers.serialize(
                self.dict_kernel_constraint
            ),
            "dict_bias_constraint": regularizers.serialize(
                self.dict_bias_constraint
            ),
        }
        base_config = super(ContextualConvexDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
