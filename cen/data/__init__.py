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

from . import fashion_mnist
from . import imdb
from . import mnist
from . import physionet
from . import satellite
from . import support2


def load(
    name,
    context_kwargs=None,
    feature_kwargs=None,
    max_train_size=None,
    merge_train_valid=False,
    permute=True,
):
    """Loads the dataset as a dictionary of subsets."""
    if name == "mnist":
        load_data = mnist.load_data
        load_interp_features = mnist.load_interp_features
        non_test_size = mnist.TRAIN_SIZE + mnist.VALID_SIZE
    elif name == "fashion_mnist":
        load_data = fashion_mnist.load_data
        load_interp_features = fashion_mnist.load_interp_features
        non_test_size = fashion_mnist.TRAIN_SIZE + fashion_mnist.VALID_SIZE
    elif name == "imdb":
        load_data = imdb.load_data
        load_interp_features = imdb.load_interp_features
        non_test_size = imdb.TRAIN_SIZE + imdb.VALID_SIZE
    elif name == "satellite":
        load_data = satellite.load_data
        load_interp_features = satellite.load_interp_features
        non_test_size = satellite.TRAIN_SIZE + satellite.VALID_SIZE
    elif name == "support2":
        load_data = support2.load_data
        load_interp_features = support2.load_interp_features
        non_test_size = support2.TRAIN_SIZE + support2.VALID_SIZE
    elif name == "physionet":
        load_data = physionet.load_data
        load_interp_features = physionet.load_interp_features
        non_test_size = physionet.TRAIN_SIZE + physionet.VALID_SIZE
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Permute training and validation, if necessary.
    order = None
    if permute:
        order = np.random.permutation(non_test_size)

    # Load context features and targets.
    context_kwargs = context_kwargs or {}
    data = load_data(order=order, **context_kwargs)

    # Load interpretable features, if necessary.
    if feature_kwargs is not None:
        data_interp = load_interp_features(order=order, **feature_kwargs)
        for set_name, value in data_interp.items():
            data[set_name][0]["X"] = value

    # Limit the size of the training set, if necessary.
    if max_train_size is not None:
        for key, value in data["train"][0].items():
            data["train"][0][key] = value[:max_train_size]
        data["train"][1] = data["train"][1][:max_train_size]

    # Merge train and valid, if necessary.
    if merge_train_valid:
        data["train"] = merge(data, ("train", "valid"))
        data["valid"] = data["test"]

    return data


def merge(data, set_names, weights=None):
    """Merges training, validation, and test data."""
    inputs = {
        key: np.concatenate([data[name][0][key] for name in set_names], axis=0)
        for key in data[set_names[0]][0].keys()
    }
    labels = np.concatenate(
        [data[set_name][1] for set_name in set_names], axis=0
    )
    if weights is None:
        return inputs, labels
    else:
        weights = np.concatenate(
            [
                np.full(data[name][1].shape[0], weights[i])
                for i, name in enumerate(set_names)
            ],
            axis=0,
        )
        return inputs, labels, weights


def split(data, train_ids, test_ids, valid_ids=None):
    """Split data into train, test (and validation) subsets."""
    datasets = {
        "train": (
            tuple(map(lambda x: x[train_ids], data[0])),
            data[1][train_ids],
        ),
        "test": (tuple(map(lambda x: x[test_ids], data[0])), data[1][test_ids]),
    }
    if valid_ids is not None:
        datasets["valid"] = (
            tuple(map(lambda x: x[valid_ids], data[0])),
            data[1][valid_ids],
        )
    else:
        datasets["valid"] = None
    return datasets
