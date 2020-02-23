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
"""Experiment utils."""

import os

import tensorflow as tf

from .. import losses
from .. import metrics
from .. import models
from .. import networks


class ModeKeys(object):
    TRAIN = "train"
    EVAL = "eval"
    INFER = "infer"


def get_input_dtypes(data):
    """Returns input shapes."""
    return {k: str(v.dtype) for k, v in data[0].items()}


def get_input_shapes(data):
    """Returns input shapes."""
    return {k: v.shape[1:] for k, v in data[0].items()}


def get_output_shape(data):
    """Returns output shapes."""
    return data[1].shape[1:]


def build(cfg, input_dtypes, input_shapes, output_shape, mode=ModeKeys.TRAIN):
    """Builds model and callbacks for training or evaluation."""
    tf.keras.backend.clear_session()

    # Build model.
    net = networks.get(**cfg.network)
    model, info = models.get(
        cfg.model.name,
        encoder=net,
        input_dtypes=input_dtypes,
        input_shapes=input_shapes,
        output_shape=output_shape,
        **cfg.model.kwargs,
    )

    # Build loss and optimizer.
    loss = losses.get(**cfg.train.loss)
    opt = tf.keras.optimizers.get(dict(**cfg.optimizer))

    # Build metrics.
    metrics_list = None
    if cfg.eval.metrics:
        metrics_list = [metrics.get(**v) for _, v in cfg.eval.metrics.items()]

    # Compile model for training.
    if mode == ModeKeys.TRAIN:
        model.compile(optimizer=opt, loss=loss, metrics=metrics_list)
        callbacks = []
        if cfg.train.checkpoint_kwargs:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(os.getcwd(), "checkpoint"),
                    **cfg.train.checkpoint_kwargs,
                )
            )
        if cfg.train.tensorboard:
            callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(os.getcwd(), "tensorboard"),
                    **cfg.train.tensorboard,
                )
            )
        info["callbacks"] = callbacks
        return model, info

    # Compile model for evaluation or inference.
    else:
        model.compile(loss=loss, optimizer=opt, metrics=metrics_list)
        checkpoint_path = os.path.join(os.getcwd(), "checkpoint")
        model.load_weights(checkpoint_path).expect_partial()
        return model, info
