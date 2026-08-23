#!/usr/bin/env bash
set -e

python src/asr.py \
  --config configs/eval.yaml
