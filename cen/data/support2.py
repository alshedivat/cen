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
"""Loaders and preprocessors for SUPPORT2 data."""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Data parameters
TRAIN_SIZE, VALID_SIZE, TEST_SIZE = 7105, 1000, 1000
EXCLUDE_FEATURES = [
    "aps",
    "sps",
    "surv2m",
    "surv6m",
    "prg2m",
    "prg6m",
    "dnr",
    "dnrday",
    "hospdead",
    "dzclass",
    "edu",
    "scoma",
    "totmcst",
    "charges",
    "totcst",
]
TARGETS = ["death", "d.time"]
AVG_VALUES = {
    "alb": 3.5,
    "bili": 1.01,
    "bun": 6.51,
    "crea": 1.01,
    "pafi": 333.3,
    "wblc": 9.0,
    "urine": 2502.0,
}


def load_data(
    datapath=None,
    nb_intervals=156,
    interval_len=7,
    fill_na="avg",
    na_value=0.0,
    death_indicator=1.0,
    censorship_indicator=1.0,
    inputs_as_sequences=False,
    inputs_pad_mode="constant",
    permute=True,
    seed=42,
):
    """Load and preprocess the SUPPORT2 dataset.

    Args:
        datapath : str or None
        nb_intervals : uint (default: 100)
            Number of intervals to split the time line.
        interval_len : uint (default: 20)
            The length of the interval in days.
        fill_na : str (default: "avg")
        na_value : float (default: -1.0)
        death_indicator : float (default: 1.0)
        censorship_indicator : float (default: -1.0)
        permute: bool (default: True)
        seed : uint (default: 42)

    Returns:
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
    """
    if datapath is None:
        datapath = "$DATA_PATH/SUPPORT2/support2.csv"
    datapath = os.path.expandvars(datapath)

    # Exclude unnecessary columns
    df = pd.read_csv(datapath)
    columns = sorted(list(set(df.columns) - set(EXCLUDE_FEATURES)))
    df = df[columns]

    # Split into features and targets
    targets = df[TARGETS]
    features = df[list(set(df.columns) - set(TARGETS))]

    # Convert categorical columns into one-hot format
    cat_columns = features.columns[features.dtypes == "object"]
    features = pd.get_dummies(features, dummy_na=False, columns=cat_columns)

    # Scale and impute real-valued features
    features[["num.co", "slos", "hday"]] = features[
        ["num.co", "slos", "hday"]
    ].astype(np.float)
    float_cols = features.columns[features.dtypes == np.float]
    features[float_cols] = (
        features[float_cols] - features[float_cols].min()
    ) / (features[float_cols].max() - features[float_cols].min())
    if fill_na == "avg":
        for key, val in AVG_VALUES.items():
            features[[key]] = features[[key]].fillna(val)
    features.fillna(na_value, inplace=True)

    X = features.values
    X[:, 33] = np.random.rand(X.shape[0])
    X = X.astype(np.float32)

    # Preprocess targets
    T = targets.values
    Y = np.zeros((len(targets), nb_intervals, 2))
    for i, (death, days) in enumerate(T):
        intervals = days // interval_len
        if death and intervals < nb_intervals:
            Y[i, intervals:, 1] = death_indicator
        if not death and intervals < nb_intervals:
            Y[i, intervals:, 0] = censorship_indicator

    # Convert inputs into sequences if necessary
    if inputs_as_sequences:
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        if inputs_pad_mode == "constant":
            X = np.pad(
                X,
                [(0, 0), (0, Y.shape[1] - 1), (0, 0)],
                mode=inputs_pad_mode,
                constant_values=0.0,
            )
        else:
            X = np.pad(
                X, [(0, 0), (0, Y.shape[1] - 1), (0, 0)], mode=inputs_pad_mode
            )

    # Shuffle & split the data into sets.
    if permute:
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(X))
        X, Y = X[order], Y[order]

    X_train = X[:TRAIN_SIZE]
    y_train = Y[:TRAIN_SIZE]
    X_valid = X[TRAIN_SIZE : TRAIN_SIZE + VALID_SIZE]
    y_valid = Y[TRAIN_SIZE : TRAIN_SIZE + VALID_SIZE]
    X_test = X[-TEST_SIZE:]
    y_test = Y[-TEST_SIZE:]

    logger.debug(f"X shape: {X_train.shape[1:]}")
    logger.debug(f"Y shape: {y_train.shape[1:]}")
    logger.debug(f"{len(X_train)} train samples")
    logger.debug(f"{len(X_valid)} validation samples")
    logger.debug(f"{len(X_test)} test samples")

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def load_interp_features(
    datapath=None, fill_na="avg", na_value=0.0, permute=True, seed=42
):
    if datapath is None:
        datapath = "$DATA_PATH/SUPPORT2/support2.csv"
    datapath = os.path.expandvars(datapath)

    # Exclude unnecessary columns.
    df = pd.read_csv(datapath)
    columns = sorted(list(set(df.columns) - set(EXCLUDE_FEATURES)))
    df = df[columns]

    # Convert categorical columns into one-hot format.
    features = df[list(set(df.columns) - set(TARGETS))]
    cat_columns = features.columns[features.dtypes == "object"]
    features = pd.get_dummies(features, dummy_na=False, columns=cat_columns)

    # Scale and impute real-valued features.
    features[["num.co", "slos", "hday"]] = features[
        ["num.co", "slos", "hday"]
    ].astype(np.float)
    float_cols = features.columns[features.dtypes == np.float]
    features[float_cols] = (
        features[float_cols] - features[float_cols].min()
    ) / (features[float_cols].max() - features[float_cols].min())
    if fill_na == "avg":
        for key, val in AVG_VALUES.items():
            features[[key]] = features[[key]].fillna(val)
    features.fillna(na_value, inplace=True)

    Z = features.values
    Z[:, 33] = np.random.rand(Z.shape[0])
    Z = Z.astype(np.float32)

    # Shuffle & split the data into sets
    if permute:
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(Z))
        Z = Z[order]

    Z_train = Z[:TRAIN_SIZE]
    Z_valid = Z[TRAIN_SIZE : TRAIN_SIZE + VALID_SIZE]
    Z_test = Z[-TEST_SIZE:]

    logger.debug(f"Z shape: {Z_train.shape[1:]}")
    logger.debug(f"{Z_train.shape[0]} train samples")
    logger.debug(f"{Z_valid.shape[0]} validation samples")
    logger.debug(f"{Z_test.shape[0]} test samples")

    return Z_train, Z_valid, Z_test
