#!/usr/bin/env bash

PROBLEM=mnist

CUDA_VISIBLE_DEVICES=0 \
python -u -m cen.run \
  experiment=train_eval \
  problem=${PROBLEM} \
  encoder=${PROBLEM}/cnn \
  model=${PROBLEM}/baseline \
  optimizer=adam \
  run.seed=1
