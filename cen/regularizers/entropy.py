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
"""Entropy-based activity regularizers."""

import tensorflow as tf

from tensorflow.python.keras.regularizers import Regularizer


class ContextConditionalNegativeEntropy(Regularizer):
    """Encourages models with higher context-conditional entropy."""

    def __init__(self, coeff=0., num_samples=256, stddev=2e-1, epsilon=1e-6):
        self.coeff = coeff
        self.stddev = stddev
        self.epsilon = epsilon
        self.num_samples = num_samples

    def __call__(self, x):
        if self.coeff == 0.:
            return tf.constant(0.)

        # Unpack inputs.
        # contextual_weights:
        #   kernels: <float32> [batch_size, feature_dim, num_classes].
        #   biases: <float32> [batch_size, num_classes].
        # features: <float32> [batch_size, feature_dim].
        # outputs: <float32> [batch_size, num_classes].
        contextual_weights, features, outputs = x

        # Generate features from P(x | c).
        # <float32> [batch_size, num_samples, feature_dim].
        features_shape = tf.shape(features)
        features_noise = tf.random.normal(
            shape=(features_shape[0], self.num_samples, features_shape[1]),
            stddev=self.stddev
        )
        # <float32> [batch_size, num_samples, feature_dim].
        features_prime = tf.expand_dims(features, axis=1) + features_noise

        # Compute log mean_j P(Y | x_j, c_i).
        # <float32> [batch_size, num_samples, num_classes].
        logits = tf.einsum(
            "ipk,ijp->ijk", contextual_weights["kernels"], features_prime
        )
        if "biases" in contextual_weights:
            # <float32> [batch_size, num_samples, units].
            biases = tf.expand_dims(contextual_weights["biases"], axis=1)
            logits = tf.add(logits, biases)
        # <float32> [batch_size, num_classes].
        probs = tf.reduce_mean(tf.nn.softmax(logits), axis=1) + self.epsilon
        probs_sum = tf.reduce_sum(probs, axis=-1, keepdims=True)
        log_probs = tf.math.log(probs / probs_sum)

        # Compute loss.
        loss = -tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.nn.softmax(outputs), logits=log_probs
        )
        return self.coeff * tf.reduce_mean(loss)

    def __str__(self):
        config = self.get_config()
        return "{name:s}({coeff:f})".format(**config)

    def get_config(self):
        return {"name": self.__class__.__name__, "coeff": float(self.coeff)}


# Aliases.


def ctx_cond_neg_ent(coeff=0., num_samples=32, stddev=.1, epsilon=1e-6):
    return ContextConditionalNegativeEntropy(
        coeff=coeff, num_samples=num_samples, stddev=stddev, epsilon=epsilon
    )
