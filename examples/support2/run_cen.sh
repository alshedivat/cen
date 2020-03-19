#!/usr/bin/env bash

PROBLEM=support2

CUDA_VISIBLE_DEVICES= \
python -u -m cen.run \
  experiment=train_eval \
  problem=${PROBLEM} \
  encoder=${PROBLEM}/mlp \
  model=${PROBLEM}/cen_convex \
  optimizer=adam
