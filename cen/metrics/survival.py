"""Custom metrics for survival analysis."""

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

from functools import partial

import tensorflow as tf

from tensorflow.python.keras.metrics import MeanMetricWrapper


class SurvivalAccuracyScore(MeanMetricWrapper):
    """
    Accuracy score for survival probability prediction.
    """

    def __init__(self, time_step, censored_indicator=1.0, dtype=None):
        super(SurvivalAccuracyScore, self).__init__(
            partial(
                accuracy_score,
                t=time_step,
                censored_indicator=censored_indicator,
            ),
            name=f"survival_acc_at_{time_step}",
            dtype=dtype,
        )


class SurvivalBrierScore(MeanMetricWrapper):
    """
    Brier score for survival probability prediction.
    """

    def __init__(self, time_step, censored_indicator=1.0, dtype=None):
        super(SurvivalBrierScore, self).__init__(
            partial(
                brier_score, t=time_step, censored_indicator=censored_indicator
            ),
            name=f"survival_brier_at_{time_step}",
            dtype=dtype,
        )


def survival_probs(logits, t):
    """Computes survival probabilities.

    Args:
        logits: <float32> [batch_size, time_steps].
        t: int.
    """
    # <float32> [batch_size].
    logits_cumsum = tf.cumsum(logits, axis=-1, reverse=True)
    # <float32> [batch_size, time_steps + 1].
    logits_cumsum_padded = tf.pad(logits_cumsum, paddings=[[0, 0], [0, 1]])
    # <float32> [batch_size].
    lognum = tf.reduce_logsumexp(logits_cumsum_padded[:, t:], axis=-1)
    # <float32> [batch_size].
    logdenom = tf.reduce_logsumexp(logits_cumsum_padded, axis=-1)
    # <float32> [batch_size].
    return tf.exp(lognum - logdenom)


def accuracy_score(y_true, logits, t, censored_indicator):
    """Implements accuracy score for the survival probability prediction.

    Args:
        y_true: <float32> [batch_size, time_steps, 2]
            y_true[i, t, 0] should indicate whether the instance i was censored
            at time t; y_true[i, t, 1] indicates occurrence of the event for
            instance i at time t_event <= t.
        logits: <float32> [batch_size, time_steps, 1]
        t: float
        censored_indicator: float

    Returns:
        accuracy: <float32> [num_non_censored].
    """
    # Resolve inputs.
    # <float32> [batch_size, time_steps].
    y_true_c = y_true[:, :, 0]
    # <float32> [batch_size, time_steps].
    y_true_e = y_true[:, :, 1]
    # <float32> [batch_size, time_steps].
    logits = logits[:, :, 0]

    # Find the non censored instances.
    # <float32> [batch_size].
    not_censored_at_t = tf.not_equal(y_true_c[:, t], censored_indicator)
    # <float32> [num_non_censored, time_steps].
    logits_uc = tf.boolean_mask(logits, not_censored_at_t)

    # Compute survival probabilities.
    # <float32> [num_non_censored].
    survival_prob = survival_probs(logits_uc, t)

    # Compute accuracy on the non-censored instances.
    # <float32> [num_non_censored].
    survived = 1 - tf.boolean_mask(y_true_e, not_censored_at_t)[:, t]
    # <float32> [num_non_censored].
    return tf.keras.metrics.binary_accuracy(survived, survival_prob)


def brier_score(y_true, logits, t, censored_indicator):
    """Brier score of the survival probability predictions.

    Args:
        y_true : <float32> [batch_size, time_steps, 2]
            y_true[i, t, 0] should indicate whether the instance i was censored
            at time t; y_true[i, t, 1] indicates occurrence of the event for
            instance i at time t_event <= t.
        logits : <float32> [batch_size, time_steps, 1]
        t : float
        censored_indicator : float

    Returns:
        score : <float32> [num_non_censored].
    """
    # Resolve inputs.
    # <float32> [batch_size, time_steps].
    y_true_c = y_true[:, :, 0]
    # <float32> [batch_size, time_steps].
    y_true_e = y_true[:, :, 1]
    # <float32> [batch_size, time_steps].
    logits = logits[:, :, 0]

    # Find the non censored instances.
    # <float32> [num_non_censored].
    not_censored_at_t = tf.not_equal(y_true_c[:, t], censored_indicator)
    # <float32> [num_non_censored, time_steps].
    logits_uc = tf.boolean_mask(logits, not_censored_at_t)

    # Compute survival probabilities.
    # <float32> [num_non_censored].
    survival_prob = survival_probs(logits_uc, t)

    # Compute brier score.
    # <float32> [num_non_censored].
    survived = 1 - tf.boolean_mask(y_true_e, not_censored_at_t)[:, t]
    # <float32> [num_non_censored].
    return tf.keras.metrics.mean_squared_error(survived, survival_prob)
