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
"""Loader and preprocessors for IMDB data."""

import logging
import os

import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from tensorflow.python.keras.utils import np_utils

logger = logging.getLogger(__name__)

# Data parameters
TRAIN_SIZE, VALID_SIZE, TEST_SIZE = 20000, 5000, 25000
NB_CLASSES = 2


def load_imdb(path):
    """Loads the IMDB dataset.
    Taken from `keras.datasets.imdb` and edited as necessary.
    """
    f = np.load(path, allow_pickle=True)
    x_train = f["x_train"]
    labels_train = f["y_train"]
    x_test = f["x_test"]
    labels_test = f["y_test"]
    x_unsup = f["x_unsup"]

    # Convert labels to one-hot.
    labels = np.concatenate([labels_train, labels_test])
    labels = np_utils.to_categorical(labels, NB_CLASSES)
    y_train = np.array(labels[: len(x_train)])
    y_test = np.array(labels[len(x_train) :])

    return (x_train, y_train), (x_test, y_test), x_unsup


def load_data(datapath=None, order=None):
    """Load the sequential representation of the data.

    Args:
        datapath: str or None (default: None)
        order: np.ndarray (default: None)

    Returns:
        data: tuples (X, y) of np.ndarrays.
    """
    if datapath is None:
        datapath = "$DATA_PATH/IMDB/imdb_bert_tokenized.npz"
    datapath = os.path.expandvars(datapath)

    (X_train, y_train), (X_test, y_test), X_unsup = load_imdb(datapath)

    # Re-order instances before splitting into train and valid.
    if order is not None:
        X_train = X_train[order]
        y_train = y_train[order]

    # Split test into valid and test if no specific set is requested.
    X_valid = X_train[-VALID_SIZE:]
    y_valid = y_train[-VALID_SIZE:]
    X_train = X_train[:-VALID_SIZE]
    y_train = y_train[:-VALID_SIZE]

    # Sanity check
    assert len(X_train) == TRAIN_SIZE
    assert len(X_valid) == VALID_SIZE
    assert len(X_test) == TEST_SIZE

    data = {
        "train": ({"C": X_train}, y_train),
        "valid": ({"C": X_valid}, y_valid),
        "test": ({"C": X_test}, y_test),
        "unsup": ({"C": X_unsup}, None),
    }

    logger.debug(f"X shape: {X_train.shape[1:]}")
    logger.debug(f"Y shape: {y_train.shape[1:]}")
    logger.debug(f"{len(X_train)} train sequences")
    logger.debug(f"{len(X_valid)} validation sequences")
    logger.debug(f"{len(X_test)} test sequences")

    return data


def load_interp_features(
    datapath=None,
    feature_type="topics",
    nb_topics=50,
    bow_vocab_size=2000,
    topic_vocab_size=20000,
    extended=False,
    remove_const_features=True,
    standardize=True,
    order=None,
    feature_subset_ratio=None,
    signal_to_noise=None,
):
    """Load the interpretable representation of the data.

    Args:
        datapath: str or None (default: None)
        feature_type: str (default: "topics")
            Interpretable representation (one of {"topics", "BoW", "both"}).
        nb_topics: int (default: 50)
        bow_vocab_size: int (default: 20000)
        topic_vocab_size: int (default: 20000)
        extended: bool (default: False)
            Whether to load the representation produced by a topic model
            trained on the extended dataset.
        order: np.ndarray (default: None)
        feature_subset_ratio: float (default: None)
        signal_to_noise: float or None (default: None)
            If not None, adds white noise to each feature with a specified SNR.

    Returns:
        data: tuple (Z_train, Z_valid, Z_test) of ndarrays.
    """
    if datapath is None:
        datapath = "$DATA_PATH/IMDB/"
    datapath = os.path.expandvars(datapath)

    if feature_type in {"BoW", "both"}:
        with open(os.path.join(datapath, "bow", "imdb.vocab")) as fp:
            vocab = [word.strip() for word in fp.readlines()]
        bow_train, _ = load_svmlight_file(
            os.path.join(datapath, "bow", "bow_train.feat"),
            n_features=len(vocab),
        )
        bow_test, _ = load_svmlight_file(
            os.path.join(datapath, "bow", "bow_test.feat"),
            n_features=len(vocab),
        )

        # Pre-process the data (reduce the vocabulary size).
        word_ids = np.nonzero([w not in ENGLISH_STOP_WORDS for w in vocab])[0]
        bow_train = bow_train[:, word_ids[:bow_vocab_size]].toarray()
        bow_test = bow_test[:, word_ids[:bow_vocab_size]].toarray()

    if feature_type in {"topics", "both"}:
        prefix = "utrain" if extended else "train"
        train_path = os.path.join(
            datapath,
            "topics",
            "%s_%d_%d.npy" % (prefix, nb_topics, topic_vocab_size),
        )
        topics_train = np.load(train_path)

        prefix = "utest" if extended else "test"
        test_path = os.path.join(
            datapath,
            "topics",
            "%s_%d_%d.npy" % (prefix, nb_topics, topic_vocab_size),
        )
        topics_test = np.load(test_path)

    if feature_type == "both":
        Z_train = np.hstack([topics_train, bow_train])
        Z_test = np.hstack([topics_test, bow_test])
    elif feature_type == "BoW":
        Z_train, Z_test = bow_train, bow_test
    elif feature_type == "topics":
        Z_train, Z_test = topics_train, topics_test

    if remove_const_features:
        Z_std = Z_train.std(axis=0)
        nonconst = np.where(Z_std > 1e-5)[0]
        Z_train = Z_train[:, nonconst]
        Z_test = Z_test[:, nonconst]

    if standardize:
        Z_mean = Z_train.mean(axis=0)
        Z_std = Z_train.std(axis=0)
        Z_train -= Z_mean
        Z_train /= Z_std
        Z_test -= Z_mean
        Z_test /= Z_std

    # Cast dtype to float32.
    Z_train = Z_train.astype(np.float32)
    Z_test = Z_test.astype(np.float32)

    # Remove unsupervised data, if was used for learning topics.
    if extended:
        Z_unsup = Z_train[(TRAIN_SIZE + VALID_SIZE) :]
        Z_train = Z_train[: (TRAIN_SIZE + VALID_SIZE)]
    else:
        Z_unsup = None

    # Permute the train data, if necessary.
    if order is not None:
        Z_train = Z_train[order]

    if feature_subset_ratio is not None:
        assert feature_subset_ratio > 0.0 and feature_subset_ratio <= 1.0
        feature_subset_size = int(Z_train.shape[1] * feature_subset_ratio)
        rng = np.random
        feature_idx = rng.choice(
            Z_train.shape[1], size=feature_subset_size, replace=False
        )
        Z_train = Z_train[:, feature_idx]
        Z_test = Z_test[:, feature_idx]

    if signal_to_noise is not None and signal_to_noise > 0.0:
        rng = np.random  # seeded externally.
        N_train = rng.normal(scale=1.0 / signal_to_noise, size=Z_train.shape)
        N_test = rng.normal(scale=1.0 / signal_to_noise, size=Z_test.shape)
        Z_train += N_train
        Z_test += N_test

    Z_valid = Z_train[-VALID_SIZE:]
    Z_train = Z_train[:-VALID_SIZE]

    # Sanity check
    assert len(Z_train) == TRAIN_SIZE
    assert len(Z_valid) == VALID_SIZE
    assert len(Z_test) == TEST_SIZE

    data = {
        "train": Z_train,
        "valid": Z_valid,
        "test": Z_test,
        "unsup": Z_unsup,
    }

    logger.debug(f"Z shape: {Z_train.shape[1:]}")
    logger.debug(f"{Z_train.shape[0]} train samples")
    logger.debug(f"{Z_valid.shape[0]} validation samples")
    logger.debug(f"{Z_test.shape[0]} test samples")

    return data
