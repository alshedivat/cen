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
"""Entry point for running experiments."""

import hydra
import logging
import os
import pickle
import random

import numpy as np
import tensorflow as tf

from . import data
from .experiment import train, evaluate, cross_validate

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs/config.yaml", strict=False)
def main(cfg):
    logger.info("Experiment config:\n" + cfg.pretty())

    # Set random seeds.
    random.seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)
    tf.random.set_seed(cfg.run.seed)

    logger.info("Loading data...")
    datasets = data.load(**cfg.dataset)

    # Limit GPU memory growth.
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    # Cross-validation.
    if cfg.crossval:
        metrics = cross_validate(cfg, datasets)
        save_path = os.path.join(os.getcwd(), "cv.metrics.pkl")
        with open(save_path, "wb") as fp:
            pickle.dump(metrics, fp)

    # Supervised training.
    else:
        model = None
        # Train.
        if cfg.train:
            model, info = train(cfg, datasets["train"], datasets["valid"])
            save_path = os.path.join(os.getcwd(), "train.history.pkl")
            with open(save_path, "wb") as fp:
                pickle.dump(info["history"], fp)
        # Evaluate.
        if cfg.eval:
            metrics = evaluate(cfg, datasets, model=model)
            save_path = os.path.join(os.getcwd(), "eval.metrics.pkl")
            with open(save_path, "wb") as fp:
                pickle.dump(metrics, fp)


if __name__ == "__main__":
    main()
