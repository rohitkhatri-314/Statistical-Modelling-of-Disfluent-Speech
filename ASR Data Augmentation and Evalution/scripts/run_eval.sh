#!/usr/bin/env bash
set -e

python src/evaluate.py \
  --config configs/eval.yaml
