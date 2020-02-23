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
"""Supervised training."""

import logging
import os

import tensorflow as tf

from . import utils

logger = logging.getLogger(__name__)


def train(cfg, train_data, validation_data=None):
    logger.info("Building...")

    # Build the model.
    input_dtypes = utils.get_input_dtypes(train_data)
    input_shapes = utils.get_input_shapes(train_data)
    output_shape = utils.get_output_shape(train_data)
    model, info = utils.build(
        cfg,
        input_dtypes=input_dtypes,
        input_shapes=input_shapes,
        output_shape=output_shape,
        mode=utils.ModeKeys.TRAIN,
    )

    # Build datasets.
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train_data)
        .shuffle(
            cfg.train.shuffle_buffer_size,
            reshuffle_each_iteration=True,
            seed=cfg.run.seed,
        )
        .batch(cfg.train.batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    validation_dataset = None
    if validation_data is not None:
        validation_dataset = (
            tf.data.Dataset.from_tensor_slices(validation_data)
            .batch(cfg.train.batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

    logger.info("Training...")

    history = model.fit(
        train_dataset,
        callbacks=info["callbacks"],
        epochs=cfg.train.epochs,
        validation_data=validation_dataset,
        verbose=cfg.train.verbose,
    )
    info["history"] = history.history

    checkpoint_path = os.path.join(os.getcwd(), "checkpoint")
    if not os.path.exists(checkpoint_path):
        # Save model weights if checkpointing was off.
        model.save_weights(checkpoint_path)
    else:
        # Load the best weights if checkpointing was on.
        model.load_weights(checkpoint_path).expect_partial()

    return model, info
