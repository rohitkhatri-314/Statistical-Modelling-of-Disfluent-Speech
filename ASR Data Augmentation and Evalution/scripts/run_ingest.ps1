$ErrorActionPreference = "Stop"

python src/ingest.py `
  --dataset openslr/librispeech_asr `
  --config clean `
  --split validation `
  --max_samples 100 `
  --min_words 4 `
  --max_words 30 `
  --output data/processed/clean_manifest.jsonl
