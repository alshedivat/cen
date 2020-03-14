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
"""Integration test for training and evaluation on MNIST."""

import logging
import pytest
import os
import random
import tempfile

import numpy as np
import tensorflow as tf

from hydra._internal.hydra import GlobalHydra, Hydra

from cen import data
from cen.experiment import train, evaluate

logger = logging.getLogger(__name__)

AVAILABLE_MODELS = [
    "baseline",
    "cen_convex_hog",
    "cen_convex_pxl",
    "cen_affine_hog",
    "cen_affine_pxl",
    "moe_hog",
    "moe_pxl",
]


def get_hydra():
    global_hydra = GlobalHydra()
    if not global_hydra.is_initialized():
        return Hydra.create_main_hydra_file_or_module(
            calling_file=__file__,
            calling_module=None,
            config_dir="configs",
            strict=False,
        )
    else:
        return global_hydra.hydra


@pytest.mark.parametrize('model', AVAILABLE_MODELS)
def test_mnist(model):

    with tempfile.TemporaryDirectory() as dir_path:
        # Set data path.
        os.environ["DATA_PATH"] = dir_path
        mnist_dir = os.path.join(dir_path, "MNIST")
        os.makedirs(mnist_dir, exist_ok=False)

        # Parse hydra configs.
        hydra = get_hydra()
        cfg = hydra.compose_config(
            "config.yaml",
            overrides=[
                f"experiment=train_eval",
                f"problem=mnist",
                f"encoder=mnist/cnn",
                f"model=mnist/{model}",
                f"optimizer=adam",
                f"train.epochs=1",
                f"train.batch_size=128",
                f"train.checkpoint_kwargs=null",
                f"train.tensorboard=null",
                "train.verbose=2",
                "eval.verbose=2",
            ],
            strict=False
        )

        # Set random seeds.
        random.seed(cfg.run.seed)
        np.random.seed(cfg.run.seed)
        tf.random.set_seed(cfg.run.seed)

        # Load datasets.
        logger.info("Loading data...")
        datasets = data.load(**cfg.dataset)

        # Truncate datasets.
        max_sizes = {"train": 256, "valid": 128, "test": 128}
        for set_name in ["train", "valid", "test"]:
            max_size = max_sizes[set_name]
            inputs = {
                key: value[:max_size]
                for key, value in datasets[set_name][0].items()
            }
            labels = datasets[set_name][1][:max_size]
            datasets[set_name] = inputs, labels

        # Test train and eval.
        model, _ = train(cfg, datasets["train"], datasets["valid"])
        evaluate(cfg, datasets, model=model)
