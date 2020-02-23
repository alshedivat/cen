"""Evaluation."""

import logging

import tensorflow as tf

from . import utils

logger = logging.getLogger(__name__)


def evaluate(cfg, datasets, model=None):
    if model is None:
        logger.info("Building...")

        input_dtypes = utils.get_input_dtypes(datasets["test"])
        input_shapes = utils.get_input_shapes(datasets["test"])
        output_shape = utils.get_output_shape(datasets["test"])
        model = utils.build(
            cfg,
            input_dtypes=input_dtypes,
            input_shapes=input_shapes,
            output_shape=output_shape,
            mode=utils.ModeKeys.EVAL,
        )

    logger.info("Evaluating...")

    metrics = {}
    for set_name, dataset in datasets.items():
        if dataset is None or dataset[1] is None:
            continue
        metric_names = ["loss"] + list(cfg.eval.metrics.keys())
        metric_values = model.evaluate(
            *dataset, batch_size=cfg.eval.batch_size, verbose=cfg.eval.verbose
        )
        metrics[set_name] = dict(zip(metric_names, metric_values))
        logger.info(f"{set_name} metrics: {metrics[set_name]}")

    return metrics
