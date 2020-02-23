"""Loader and preprocessors for Fashion MNIST data."""

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

import logging
import os

import numpy as np

from tensorflow.python.keras.utils import np_utils

from . import utils

logger = logging.getLogger(__name__)

# Data parameters
TRAIN_SIZE, VALID_SIZE, TEST_SIZE = 50000, 10000, 10000
IMG_ROWS, IMG_COLS, IMG_CHANNELS = 28, 28, 1
NB_CLASSES = 10


def load_data(
    datapath=None, standardize=False, padding=None, permute=False, seed=42
):
    """Load FASHION MNIST data.

    Args:
        datapath: str or None (default: None)
        padding: tuple of int or None (default: None)
        permute: bool (default: False)
        seed: uint (default: 42)

    Returns:
        data: tuples (X, y) of nd.arrays
    """
    if datapath is None:
        datapath = "$DATA_PATH/FASHION_MNIST/data.npz"
    datapath = os.path.expandvars(datapath)

    # the data, shuffled and split between train and test sets
    data = np.load(datapath)
    X_train, y_train = data["x_train"], data["y_train"]
    X_test, y_test = data["x_test"], data["y_test"]

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255

    if standardize:
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0)
        X_train -= X_mean
        X_train /= X_std
        X_test -= X_mean
        X_test /= X_std

    if permute:
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(X_train))
        X_train = X_train[order]
        y_train = y_train[order]

    # Split train into train and validation
    X_valid = X_train[-VALID_SIZE:]
    X_train = X_train[:-VALID_SIZE]
    y_valid = y_train[-VALID_SIZE:]
    y_train = y_train[:-VALID_SIZE]

    # Sanity checks
    assert X_train.shape[0] == TRAIN_SIZE
    assert X_valid.shape[0] == VALID_SIZE
    assert X_test.shape[0] == TEST_SIZE

    input_shape = [IMG_ROWS, IMG_COLS, IMG_CHANNELS]

    # Add padding (if necessary)
    if padding is not None:
        assert isinstance(padding, (list, tuple)) and len(padding) == 2
        pad_width = [(0, 0)]
        for i in range(2):
            pad_width.append((padding[i], padding[i]))
            input_shape[i] += padding[i] * 2
        X_train = np.pad(
            X_train, pad_width=pad_width, mode="constant", constant_values=0.0
        )
        X_valid = np.pad(
            X_valid, pad_width=pad_width, mode="constant", constant_values=0.0
        )
        X_test = np.pad(
            X_test, pad_width=pad_width, mode="constant", constant_values=0.0
        )

    # Reshape
    X_train = X_train.reshape(TRAIN_SIZE, *input_shape)
    X_valid = X_valid.reshape(VALID_SIZE, *input_shape)
    X_test = X_test.reshape(TEST_SIZE, *input_shape)

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
    Y_valid = np_utils.to_categorical(y_valid, NB_CLASSES)
    Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

    logger.debug(f"X shape: {X_train.shape[1:]}")
    logger.debug(f"Y shape: {y_train.shape[1:]}")
    logger.debug(f"{len(X_train)} train samples")
    logger.debug(f"{len(X_valid)} validation samples")
    logger.debug(f"{len(X_test)} test samples")

    return (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)


def load_interp_features(
    datapath=None,
    feature_type="pixels16x16",
    feature_subset_per=None,
    remove_const_features=True,
    standardize=True,
    whiten=False,
    permute=True,
    signal_to_noise=None,
    seed=42,
):
    """Load an interpretable representation for FASHION MNIST.

    Args:
        datapath: str or None (default: None)
        feature_type: str (default: 'pixels16x16')
            Possible values are:
            {'pixels16x16', 'pixels20x20', 'pixels28x28', 'hog3x3'}.
        standardize: bool (default: True)
        whiten: bool (default: False)
        permute: bool (default: True)
        signal_to_noise: float or None (default: None)
            If not None, adds white noise to each feature with a specified SNR.
        seed: uint (default: 42)

    Returns:
        data: tuple (Z_train, Z_valid, Z_test) of nd.arrays
    """
    if datapath is None:
        datapath = "$DATA_PATH/FASHION_MNIST/feat.interp.%s.npz" % feature_type
    datapath = os.path.expandvars(datapath)

    data = np.load(datapath)
    Z_train, Z_test = data["Z_train"], data["Z_test"]

    if feature_type.startswith("pixels"):
        Z_train = Z_train.astype("float32")
        Z_test = Z_test.astype("float32")
        Z_train /= 255
        Z_test /= 255

    Z_train = Z_train.reshape((TRAIN_SIZE + VALID_SIZE, -1))
    Z_test = Z_test.reshape((TEST_SIZE, -1))

    if remove_const_features:
        Z_std = Z_train.std(axis=0)
        nonconst = np.where(Z_std > 1e-5)[0]
        Z_train = Z_train[:, nonconst]
        Z_test = Z_test[:, nonconst]

    if standardize:
        Z_mean = Z_train.mean(axis=0)
        Z_std = Z_train.std(axis=0)
        nonconst = np.where(Z_std > 1e-5)[0]
        Z_train -= Z_mean
        Z_train[:, nonconst] /= Z_std[nonconst]
        Z_test -= Z_mean
        Z_test[:, nonconst] /= Z_std[nonconst]

    if whiten:
        WM = utils.get_zca_whitening_mat(Z_train)
        Z_train = utils.zca_whiten(Z_train, WM)
        Z_test = utils.zca_whiten(Z_test, WM)

    if permute:
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(Z_train))
        Z_train = Z_train[order]

    if feature_subset_per is not None:
        assert feature_subset_per > 0.0 and feature_subset_per <= 1.0
        feature_subset_size = int(Z_train.shape[1] * feature_subset_per)
        rng = np.random.RandomState(seed)
        feature_idx = rng.choice(
            Z_train.shape[1], size=feature_subset_size, replace=False
        )
        Z_train = Z_train[:, feature_idx]
        Z_test = Z_test[:, feature_idx]

    if signal_to_noise is not None and signal_to_noise > 0.0:
        rng = np.random.RandomState(seed)
        N_train = np.random.normal(
            scale=1.0 / signal_to_noise, size=Z_train.shape
        )
        N_test = np.random.normal(
            scale=1.0 / signal_to_noise, size=Z_test.shape
        )
        Z_train += N_train
        Z_test += N_test

    # Split train into train and validation
    Z_valid = Z_train[-VALID_SIZE:]
    Z_train = Z_train[:-VALID_SIZE]

    # Sanity checks
    assert Z_train.shape[0] == TRAIN_SIZE
    assert Z_valid.shape[0] == VALID_SIZE
    assert Z_test.shape[0] == TEST_SIZE

    logger.debug(f"Z shape: {Z_train.shape[1:]}")
    logger.debug(f"{Z_train.shape[0]} train samples")
    logger.debug(f"{Z_valid.shape[0]} validation samples")
    logger.debug(f"{Z_test.shape[0]} test samples")

    return Z_train, Z_valid, Z_test
