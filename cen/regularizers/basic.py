"""Custom basic regularizers."""
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

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.regularizers import Regularizer


class L1L2TV(Regularizer):
    """Regularizer for L1, L2, and TV regularization.

    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
        tv: Float; TV regularization factor.
    """

    def __init__(self, l1=0.0, l2=0.0, tv=0.0):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.tv = K.cast_to_floatx(tv)

    def __call__(self, x):
        regularization = 0
        if self.l1:
            regularization += K.sum(self.l1 * K.abs(x))
        if self.l2:
            regularization += K.sum(self.l2 * K.square(x))
        if self.tv:
            regularization += K.sum(self.tv * K.square(x[:, 1:] - x[:, :-1]))
        return regularization

    def __str__(self):
        config = self.get_config()
        return "{name:s}(l1={l1:f}, l2={l2:f}, tv={tv:f})".format(**config)

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "l1": float(self.l1),
            "l2": float(self.l2),
            "tv": float(self.tv),
        }


# Aliases.


def l1_l2_tv(l1=0.01, l2=0.01, tv=0.01):
    return L1L2TV(l1=l1, l2=l2, tv=tv)
