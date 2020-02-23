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
"""Cross-validation."""

import logging

import numpy as np

from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

from .. import data
from .train import train
from .eval import evaluate

logger = logging.getLogger(__name__)


def cross_validate(cfg, datasets):
    # Merge train, validation, and test data.
    inputs, labels = data.merge(datasets)

    # Build stratified k-fold cross-validation splitter.
    skf = StratifiedKFold(
        n_splits=cfg.crossval.splits,
        shuffle=cfg.crossval.shuffle,
        random_state=cfg.crossval.seed,
    )
    cv_splits = skf.split(inputs[0], labels.nonzero()[1])

    if cfg.crossval.verbose:
        logger.info(f"Cross-validating...")

    # Run cross-validation.
    all_metrics = defaultdict(dict)
    for i, (train_ids, test_ids) in enumerate(cv_splits):
        datasets = data.split(
            (inputs, labels), train_ids=train_ids, test_ids=test_ids
        )
        # Train.
        train(cfg, datasets["train"], validation_data=datasets["valid"])
        # Evaluate.
        metrics = evaluate(cfg, datasets)
        # Aggregate.
        for set_name, metrics_dict in metrics.items():
            for name, value in metrics_dict.items():
                all_metrics[set_name][name] = all_metrics[set_name].get(
                    name, []
                ) + [value]
        # Log.
        if cfg.crossval.verbose:
            logger.info(f"Fold {i + 1}/{cfg.crossval.splits}:")
            for set_name, metrics_dict in all_metrics.items():
                metrics_mean_std = zip(
                    map(np.mean, metrics_dict.values()),
                    map(np.std, metrics_dict.values()),
                )
                metrics_mean_std_str = map(
                    lambda x: "{:.2f} +/- {:.2f}".format(*x), metrics_mean_std
                )
                metrics_mean_std_dict = dict(
                    zip(metrics_dict.keys(), metrics_mean_std_str)
                )
                logger.info(f"{set_name} metrics: {metrics_mean_std_dict}")

    return all_metrics
