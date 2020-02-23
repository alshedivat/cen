#!/usr/bin/env bash

PROBLEM=imdb

CUDA_VISIBLE_DEVICES=0 \
python -u -m cen.run \
  experiment=train_eval \
  problem=${PROBLEM} \
  encoder=${PROBLEM}/bilstm \
  model=${PROBLEM}/cen_convex \
  optimizer=adam \
  run.seed=1
