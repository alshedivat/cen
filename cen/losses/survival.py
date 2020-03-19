"""Custom loss functions for survival analysis."""

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

import numpy as np
import tensorflow as tf


class NegativeLogLikelihoodSurvival(tf.keras.losses.Loss):
    """
    The negative log-likelihood loss for survival analysis. Requires survival
    times to be quantized and the problem converted into a multi-task binary
    classification.

    Reference:
        [1] Yu, C.-N., et al.: "Learning patient-specific cancer survival
            distributions as a sequence of dependent regressors." NIPS 2011.
    """

    def __init__(self, censored_indicator=1.0, **kwargs):
        super(NegativeLogLikelihoodSurvival, self).__init__(**kwargs)
        self.censored_indicator = censored_indicator
        self.from_logits = True

    def call(self, y_true, logits):
        """
        Computes negative log-likelihood given censored ground truth sequences
        and prediction probabilities (or logits).

        Args:
            y_true: <float32> [batch_size, time_steps, 2]
                y_true[i, t, 0] should indicate whether the instance i was
                censored at time t; y_true[i, t, 1] indicates occurrence of the
                event for instance i at time t_event <= t.
            logits: <float32> [batch_size, time_steps, 1].
                The log-probability of surviving at each given time step.

        Returns:
            loss: <float32> [batch_size].
        """
        # Resolve inputs.
        # <float32> [batch_size, time_steps].
        y_true_c = tf.equal(y_true[:, :, 0], self.censored_indicator)
        # <float32> [batch_size, time_steps].
        y_true_e = y_true[:, :, 1]
        # <float32> [batch_size, time_steps].
        logits = logits[:, :, 0]

        # Compute reverse cumulative sum of logits.
        # <float32> [batch_size, time_steps].
        # Note: these are the scores of the sequence with the events occurring
        #       in each of the possible time intervals.
        logits_cumsum = tf.cumsum(logits, axis=-1, reverse=True)

        # Compute the log numerator for censored data.
        # Notes:
        #   1) Mask out time steps based on when each event was censored.
        #   2) The added last time step corresponds to the open time interval,
        #      from the last time step to infinity, [t_last, +inf).
        # Example (censored at step 2): [[-inf, -inf, x, y, z, ..., 0.]].
        improbable_logits = tf.fill(tf.shape(logits), -np.inf)
        # <float32> [batch_size, time_steps + 1].
        logits_cumsum_masked = tf.pad(
            tf.where(y_true_c, logits_cumsum, improbable_logits),
            paddings=[[0, 0], [0, 1]],
        )
        # <float32> [batch_size].
        lognum_censored = tf.reduce_logsumexp(logits_cumsum_masked, axis=-1)

        # Compute the log numerator for uncensored data.
        # Note: No marginalization for uncensored events.
        # <float32> [batch_size].
        lognum_uncensored = tf.reduce_sum(y_true_e * logits, axis=1)

        # Select correct numerator for each instance.
        # <float32> [batch_size].
        lognum = tf.where(y_true_c[:, -1], lognum_censored, lognum_uncensored)

        # Compute the log denominator.
        # Note: The extra time step for the [t_last, +inf) interval.
        # <float32> [batch_size, time_steps + 1].
        logits_cumsum_padded = tf.pad(logits_cumsum, paddings=[[0, 0], [0, 1]])
        # <float32> [batch_size].
        logdenom = tf.reduce_logsumexp(logits_cumsum_padded, axis=-1)

        # <float32> [batch_size].
        return logdenom - lognum
