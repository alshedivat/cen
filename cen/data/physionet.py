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
"""Loaders and preprocessors for PhysioNet data."""

import logging
import os

import numpy as np
import pandas as pd

from concurrent import futures
from functools import partial

from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.preprocessing import sequence

logger = logging.getLogger(__name__)

# Data parameters
META_PARAMS = ["RecordID", "Age", "Gender", "Weight", "ICUType"]
META_PARAMS_DROP = ["RecordID", "Age", "Gender", "Height", "ICUType"]
META_PARAMS_FILLNA = {"Gender": lambda x: 0.0, "Weight": lambda x: np.mean(x)}
MEASUREMENTS = [
    "Albumin",
    "ALP",
    "ALT",
    "AST",
    "Bilirubin",
    "BUN",
    "Cholesterol",
    "Creatinine",
    "DiasABP",
    "FiO2",
    "GCS",
    "Glucose",
    "HCO3",
    "HCT",
    "HR",
    "K",
    "Lactate",
    "Mg",
    "MAP",
    "MechVent",
    "Na",
    "NIDiasABP",
    "NIMAP",
    "NISysABP",
    "PaCO2",
    "PaO2",
    "pH",
    "Platelets",
    "RespRate",
    "SaO2",
    "SysABP",
    "Temp",
    "TroponinI",
    "TroponinT",
    "Urine",
    "WBC",
    "Weight",
]
INTERP_STATISTICS = [
    # lambda x: np.nanmin(x, axis=1, keepdims=False),
    # lambda x: np.nanmax(x, axis=1, keepdims=False),
    # lambda x: np.nanmean(x, axis=1, keepdims=False),
    lambda x: np.nanmedian(x, axis=1, keepdims=False)
]

TRAIN_SIZE, VALID_SIZE, TEST_SIZE = 2400, 800, 800
UNLABELED_SIZE = 4000


def proc_case(fpath, fill_na, na_value, resample_agg, resample_freq):
    """Loads and preprocesses a single case."""
    case = pd.read_csv(fpath, na_values="-1")
    case.Time = pd.TimedeltaIndex([t + ":00" for t in case.Time])
    case = pd.pivot_table(
        case, dropna=False, index="Time", columns="Parameter", values="Value"
    )
    case = case.resample(resample_freq).agg(resample_agg)

    # Save meta-data & drop the corresponding columns
    meta = {pname: case[pname].mean() for pname in META_PARAMS}
    case.drop(META_PARAMS_DROP, 1, inplace=True)

    # Add missing columns (cases have only subsets of measurements)
    columns = set(case.columns.tolist())
    missing_columns = set(MEASUREMENTS) - columns
    case = case.assign(**{col: np.nan for col in missing_columns})

    # Sanity check & re-arrange columns
    assert set(case.columns.tolist()) == set(MEASUREMENTS)
    case = case[MEASUREMENTS]

    # Fill NaNs (if necessary)
    if fill_na:
        case.fillna(na_value, inplace=True)

    return case.as_matrix(), meta


def load_cases_and_outcomes(
    datapath,
    task="survival",
    resample_freq="10T",
    resample_agg="mean",
    fill_na=True,
    na_value=-1.0,
    num_proc=None,
    use_cached=True,
):
    cache_path = os.path.join(
        datapath,
        f"data-{resample_freq}-{resample_agg}-{task}-na_value_{int(na_value)}.npz",
    )

    if use_cached and os.path.isfile(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        cases = data["cases"].item()
        outcomes = data["outcomes"]
    else:
        # Load cases.
        cases, metadata = {}, {}
        for set_name in ["set-a", "set-b"]:
            logger.debug(f"processing {set_name}")
            cases[set_name] = []
            metadata[set_name] = {pname: [] for pname in META_PARAMS}
            with futures.ProcessPoolExecutor(num_proc) as executor:
                fpaths = [
                    os.path.join(datapath, set_name, fname)
                    for fname in os.listdir(os.path.join(datapath, set_name))
                ]
                fn = partial(
                    proc_case,
                    fill_na=fill_na,
                    na_value=na_value,
                    resample_agg=resample_agg,
                    resample_freq=resample_freq,
                )
                for i, (c, m) in enumerate(executor.map(fn, fpaths), 1):
                    if i % 1000 == 0:
                        logger.debug(f"...processed {i}/{len(fpaths)} cases")
                    cases[set_name].append(c)
                    for pname in META_PARAMS:
                        metadata[set_name][pname].append(m[pname])

            # Fill NaNs in metadata (if necessary).
            for pname, pval in metadata[set_name].items():
                metadata[set_name][pname] = np.asarray(pval)
                pval = metadata[set_name][pname]
                if fill_na and (pname in META_PARAMS_FILLNA):
                    nanidx = np.isnan(pval)
                    notnanidx = np.logical_not(nanidx)
                    fill_value = META_PARAMS_FILLNA[pname](pval[notnanidx])
                    pval[nanidx] = fill_value

        # Load outcomes (only for set-a; set-b outcomes are not available).
        outcomes_path = os.path.join(datapath, "Outcomes-a.txt")
        usecols = [0, 3, 4] if task == "survival" else [0, 5]
        outcomes = pd.read_csv(outcomes_path, usecols=usecols, index_col=0)
        outcomes = outcomes.ix[metadata["set-a"]["RecordID"]].as_matrix()

        logger.debug("Saving data...")
        np.savez_compressed(cache_path, cases=cases, outcomes=outcomes)

    return cases, outcomes


def load_data(
    datapath=None,
    task="survival",
    resample_freq="10T",
    resample_agg="mean",
    maxlen=-1,
    interval_len=1,
    nb_intervals=60,
    death_indicator=1.0,
    censored_indicator=1.0,
    fill_na=True,
    na_value=-1.0,
    num_proc=None,
    use_cached=True,
    permute=True,
    seed=42,
):
    """Load and preprocess the PhysioNet 2012 challenge dataset.

    Args:
        task: str (default: "survival")
            The prediction task. One of {"survival", "in-hospital-death"},
            where the former is survival analysis task and the latter is binary
            classification of the in-hospital deaths.
        resample_freq: str (default: "10T")
        resample_agg: str (default: "mean")
        maxlen: int (default: -1)
        fill_na: bool (default: True)
        na_value: float (default: -1.0)
        use_cached: bool (default: True)
        permute: bool (default: True)
        seed: uint (default: 42)

    Returns:
        data: tuples (X, y) of ndarrays
        X_unsup: ndarray
    """
    if datapath is None:
        datapath = "$DATA_PATH/PhysioNetChallenge"
    datapath = os.path.expandvars(datapath)

    assert task in {"survival", "death"}

    # Load cases and outcomes.
    cases, outcomes = load_cases_and_outcomes(
        datapath,
        task=task,
        fill_na=fill_na,
        na_value=na_value,
        resample_agg=resample_agg,
        resample_freq=resample_freq,
        use_cached=use_cached,
        num_proc=num_proc,
    )

    # Select X's and Y's.
    X = cases["set-a"]
    # X_unsup = cases["set-b"]
    if task == "survival":
        Y = np.zeros((len(outcomes), nb_intervals, 2))
        for i, (in_hospital_days, survival_days) in enumerate(outcomes):
            intervals = in_hospital_days // interval_len
            if survival_days < 0 or survival_days > in_hospital_days:
                Y[i, intervals:, 0] = censored_indicator
            else:  # survival_days <= in_hospital_days:
                Y[i, intervals:, 1] = death_indicator
    else:
        Y = np_utils.to_categorical(outcomes)

    # Shuffle the data before splitting.
    if permute:
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(X))
        X, Y = [X[i] for i in order], Y[order]

    # Padding.
    maxlen = max([x.shape[0] for x in X]) if maxlen < 0 else maxlen
    X = sequence.pad_sequences(
        X, maxlen=maxlen, value=na_value, dtype=np.float32
    )

    # Split data.
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
    datapath=None,
    task="survival",
    resample_freq="10T",
    resample_agg="mean",
    fill_na=True,
    na_value=-1.0,
    maxlen=-1,
    num_proc=None,
    permute=True,
    seed=42,
):
    if datapath is None:
        datapath = "$DATA_PATH/PhysioNetChallenge"
    datapath = os.path.expandvars(datapath)

    # Load cases.
    cases, _ = load_cases_and_outcomes(
        datapath,
        task=task,
        resample_freq=resample_freq,
        resample_agg=resample_agg,
        fill_na=fill_na,
        na_value=na_value,
        use_cached=True,
        num_proc=num_proc,
    )
    X = cases["set-a"]

    # Shuffle the data before splitting.
    if permute:
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(X))
        X = [X[i] for i in order]

    # Padding.
    maxlen = max([x.shape[0] for x in X]) if maxlen < 0 else maxlen
    X = sequence.pad_sequences(
        X, maxlen=maxlen, value=na_value, dtype=np.float32
    )

    # Split data.
    X_train = X[:TRAIN_SIZE]
    X_valid = X[TRAIN_SIZE : TRAIN_SIZE + VALID_SIZE]
    X_test = X[-TEST_SIZE:]

    # Construct Z.
    Z = []
    for X in [X_train, X_valid, X_test]:
        X_ = X.copy()
        X_[X_ == na_value] = np.nan
        Z_ = np.concatenate([stat(X_) for stat in INTERP_STATISTICS], axis=-1)
        Z_[np.isnan(Z_)] = na_value
        Z.append(Z_)

    Z_train, Z_valid, Z_test = Z
    logger.debug(f"Z shape: {Z_train.shape[1:]}")
    logger.debug(f"{Z_train.shape[0]} train samples")
    logger.debug(f"{Z_valid.shape[0]} validation samples")
    logger.debug(f"{Z_test.shape[0]} test samples")

    return Z_train, Z_valid, Z_test
