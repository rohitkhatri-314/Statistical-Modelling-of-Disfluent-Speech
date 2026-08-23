#!/usr/bin/env bash
set -e

bash scripts/run_ingest.sh
bash scripts/run_augment.sh
bash scripts/run_tts.sh
bash scripts/run_asr.sh
bash scripts/run_eval.sh
