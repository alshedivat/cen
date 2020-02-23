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

import omegaconf

from . import contextual_dense
from . import contextual_mixture


def get_contextual(name, **kwargs):
    for key, value in kwargs.items():
        if isinstance(value, omegaconf.dictconfig.DictConfig):
            kwargs[key] = dict(value)
    if name == "affine":
        return contextual_dense.ContextualAffineDense(**kwargs)
    elif name == "convex":
        return contextual_dense.ContextualConvexDense(**kwargs)
    elif name == "mixture":
        return contextual_mixture.ContextualMixture(**kwargs)
    else:
        raise ValueError(f"Unknown contextual layer: {name}")
