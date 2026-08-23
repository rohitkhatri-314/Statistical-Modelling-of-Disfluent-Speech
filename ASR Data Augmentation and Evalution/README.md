# ASR Data Augmentation: Stuttered Speech Pipeline

This project is a complete pipeline for creating synthetic stuttered speech from clean speech transcripts (using LibriSpeech) and then testing how well an Automatic Speech Recognition (ASR) model understands it.

## Features Included

- **Data Ingestion:** Downloads clean text transcripts from the LibriSpeech dataset.
- **Text Augmentation:** Adds realistic stuttering effects to the text (like word repetitions, phrase repetitions, and filler words like "uh" or "um").
- **Text-to-Speech (TTS):** Converts the clean and stuttered text into audio files using Microsoft's SpeechT5 model and various speaker voices.
- **ASR Evaluation:** Runs the generated audio through an ASR model (`facebook/wav2vec2-base-960h`) to see what text the model predicts.
- **Metrics Calculation:** Calculates Word Error Rate (WER) and BERTScore to see how much the stuttering affects the ASR model's performance compared to clean speech.

## Prerequisites

Before running the code, make sure you have Python installed on your computer.

### 1. Setup the Environment

Open your terminal (or command prompt in Windows), navigate to this folder, and run these commands to set up the necessary tools:

```bash
# Create a virtual environment named .venv
python -m venv .venv

# Activate the virtual environment
# On Windows use: 
.venv\Scripts\activate
# (On Mac/Linux use: source .venv/bin/activate)

# Install the required dependencies
pip install -r requirements.txt
```

### 2. How to Run the Pipeline

The easiest way to run the entire process from start to finish is using the provided `run_all.sh` shell script. If you are on Windows, you can use Git Bash or WSL to run the `.sh` files, or simply run the python commands found inside the scripts.

#### Option A: Run Everything Automatically

```bash
# This will run data download, augmentation, TTS, ASR, and Evaluation sequentially
bash scripts/run_all.sh
```

#### Option B: Run Step-by-Step

If you want to see what happens at each step, you can run them one by one.

**1. Get Clean Data:** 
```bash
bash scripts/run_ingest.sh
```
*(Saves clean transcripts in `data/processed/clean_manifest.jsonl`)*

**2. Add Stuttering (Augment):** 
```bash
bash scripts/run_augment.sh
```
*(Saves the stuttered text versions in `data/augmented/manifests/stuttered_manifest_initial.jsonl`)*

**3. Generate Audio (TTS):** 
```bash
bash scripts/run_tts.sh
```
*(Creates `.flac` audio files for clean and stuttered speech in `data/augmented/audio/`)*

**4. Transcribe Audio (ASR Inference):** 
```bash
bash scripts/run_asr.sh
```
*(Saves the ASR model's text predictions in `outputs/asr_predictions/predictions.jsonl`)*

**5. Evaluate Results:** 
```bash
bash scripts/run_eval.sh
```
*(Calculates metrics and saves reports in `outputs/metrics/`)*

### Outputs Generated

After the pipeline finishes, you'll be able to find your results here:
- **`data/augmented/audio/clean/`**: Audio files of clean, fluent speech.
- **`data/augmented/audio/stuttered/`**: Audio files with the stuttering effects applied.
- **`outputs/metrics/per_utterance_metrics.csv`**: A detailed spreadsheet showing the error rates (WER and BERTScore) for every single audio file.
- **`outputs/metrics/summary_metrics.json`**: A quick summary comparing how the ASR model performed overall on clean speech versus stuttered speech.

## Advanced Usage

By default, the pipeline runs a small test batch to make sure everything works quickly. If you want to run a larger experiment:

1. Open `scripts/run_ingest.sh` and increase the `--max_samples` value.
2. If you want a more extreme stuttering effect, run the expanded augmentation config instead of the initial one:
   ```bash
   python src/augment.py --config configs/augment_expanded.yaml
   ```
3. Update `configs/tts.yaml` to point `input_manifest` to the newly generated `stuttered_manifest_expanded.jsonl`, and run the rest of the steps as normal.
