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
"""Interface for the satellite imagery data."""

import logging
import os

import numpy as np

from tensorflow.python.keras.utils import np_utils

logger = logging.getLogger(__name__)

# Data parameters
TRAIN_SIZE, VALID_SIZE, TEST_SIZE = 400, 100, 142
NB_CLASSES = 2


def load_data(
    datapath=None,
    survey="LSMS",
    country="uganda",
    standardize=True,
    permute=False,
    seed=42,
):
    """Load satellite imagery features and survey targets.

    Args:
        datapath : str or None (default: None)
        country : str (default: "uganda")
        permute : bool (default: True)
        seed : uint (default: 42)

    Returns:
        data: tuple (X, y) of ndarrays
    """
    if datapath is None:
        datapath = "$DATA_PATH/Satellite"
    datapath = os.path.join(datapath, survey, country)
    datapath = os.path.expandvars(datapath)

    X = np.load(os.path.join(datapath, "conv_features.npy")).astype(np.float32)
    y = np.load(os.path.join(datapath, "survey2.npy"))[:, 0]

    # Sanity checks.
    assert len(X) == len(y) == TRAIN_SIZE + VALID_SIZE + TEST_SIZE

    # Convert labels to one-hot.
    y = np_utils.to_categorical(y, NB_CLASSES)

    if standardize:
        X -= X.mean(axis=0)
        nonconst = np.where(np.logical_not(np.isclose(X.std(axis=0), 0.0)))[0]
        X[:, nonconst] /= X[:, nonconst].std(axis=0)

    if permute:
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(X))
        X, y = X[order], y[order]

    # Split data into train, valid, test.
    X_train = X[:TRAIN_SIZE]
    y_train = y[:TRAIN_SIZE]
    X_valid = X[TRAIN_SIZE : (TRAIN_SIZE + VALID_SIZE)]
    y_valid = y[TRAIN_SIZE : (TRAIN_SIZE + VALID_SIZE)]
    X_test = X[-TEST_SIZE:]
    y_test = y[-TEST_SIZE:]

    logger.debug(f"X shape: {X_train.shape[1:]}")
    logger.debug(f"Y shape: {y_train.shape[1:]}")
    logger.debug(f"{len(X_train)} train samples")
    logger.debug(f"{len(X_valid)} validation samples")
    logger.debug(f"{len(X_test)} test samples")

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def load_interp_features(
    datapath=None,
    survey="LSMS",
    country="uganda",
    standardize=True,
    permute=False,
    seed=42,
):
    """Load survey data.

    Args:
        datapath : str or None (default: None)
        country : str (default: "uganda")
        permute : bool (default: True)
        seed : uint (default: 42)

    Returns:
        Z: ndarray
    """
    if datapath is None:
        datapath = "$DATA_PATH/Satellite"
    datapath = os.path.join(datapath, survey, country)
    datapath = os.path.expandvars(datapath)

    Z = np.load(os.path.join(datapath, "survey2.npy"))[:, 1:].astype(np.float32)

    # Sanity checks.
    assert len(Z) == TRAIN_SIZE + VALID_SIZE + TEST_SIZE

    if standardize:
        Z_min, Z_max = Z.min(axis=0), Z.max(axis=0)
        Z = (Z - Z_min) / (Z_max - Z_min)

    if permute:
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(Z))
        Z = Z[order]

    # Split data into train, valid, test.
    Z_train = Z[:TRAIN_SIZE]
    Z_valid = Z[TRAIN_SIZE : (TRAIN_SIZE + VALID_SIZE)]
    Z_test = Z[-TEST_SIZE:]

    logger.debug(f"Z shape: {Z_train.shape[1:]}")
    logger.debug(f"{Z_train.shape[0]} train samples")
    logger.debug(f"{Z_valid.shape[0]} validation samples")
    logger.debug(f"{Z_test.shape[0]} test samples")

    return Z_train, Z_valid, Z_test
