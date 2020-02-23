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

from . import densenet
from . import resnet
from . import rnn
from . import simple
from . import vgg16


def get(name, **kwargs):
    if name == "bert":
        return rnn.Bert(**kwargs)
    elif name == "bilstm":
        return rnn.BiLSTM(**kwargs)
    elif name == "densenet":
        return densenet.DenseNet(**kwargs)
    elif name == "resent50":
        return resnet.ResNet50(**kwargs)
    elif name == "simple_cnn":
        return simple.CNN(**kwargs)
    elif name == "simple_mlp":
        return simple.MLP(**kwargs)
    elif name == "identity":
        return simple.Identity()
    else:
        raise ValueError(f"Unknown network: {name}")
