# ASR Baseline and Fine-tuning Implementation Report

This report summarizes the implementation of the baseline work based on the paper **"Inclusive ASR for Disfluent Speech" (Interspeech 2024)**.

## 1. What was the missing "Baseline" work?

The initial codebase successfully implemented the synthetic data generation pipeline (LibriSpeech text → Augmentation → TTS → ASR). However, to fully replicate the paper's core contributions and baseline, we needed to implement:

1. **Evaluation on Real Stuttered Speech:** The paper evaluates its models on the **FluencyBank dataset** (real humans who stutter), not just synthetic audio.
2. **Accuracy Bias Evaluation (FluencyBank-N):** The paper measures "accuracy bias" by comparing ASR performance on real stuttered speech vs. a synthesized clean version of the exact same speech (called FluencyBank-N). 
3. **Fine-Tuning Loop:** The main contribution of the paper is fine-tuning the `wav2vec2-base-960h` model on augmented data. The training script was missing.

## 2. What we added and fixed

Without disrupting your existing code, we successfully implemented these missing components:

### A. Code Fixes
* **`src/augment.py`**: Fixed a critical typo in the `clamp_int` function.
* **`src/evaluate.py`**: Updated the BERTScore parameter `rescale_with_layers` to the correct modern standard `rescale_with_baseline`.
* **`src/ingest.py`**: Updated LibriSpeech ingestion to properly extract the `speaker_id` so that synthetic voices are assigned consistently per speaker.

### B. New Components Added
* **`src/ingest_fluencybank.py`**: A new script to ingest real FluencyBank metadata (from a CSV file) and format it into the JSONL manifest required by the pipeline.
* **`configs/tts_fluencybank_n.yaml`**: A new configuration file. Running `tts.py` with this config will generate the **FluencyBank-N** dataset (the synthetic clean counterpart to the real stuttered audio) using 10 different speaker voices, exactly as the paper did.
* **`src/finetune.py`**: The core missing piece. This script uses HuggingFace `Trainer` and a custom CTC (Connectionist Temporal Classification) data collator to fine-tune the `wav2vec2-base-960h` model on the augmented stuttered datasets.

---

## 3. How to collect all metrics for your Professor

To present a complete set of baseline metrics for your project, follow these steps locally.

### Step 1: Collect Synthetic Baseline Metrics
Run the existing scripts to evaluate how the off-the-shelf model performs on clean vs. augmented synthetic speech:
```powershell
.\scripts\run_all.ps1
```
* **What to show:** Open `outputs/metrics/summary_metrics.json`. Show the professor the `mean_delta_wer` and `mean_delta_fbert`, which prove that stuttered speech heavily degrades standard ASR performance.

### Step 2: Ingest Real FluencyBank Data
*(You will need to create a CSV file named `fluencybank_meta.csv` with columns: `utterance_id, speaker_id, audio_path, clean_text, stuttered_text` based on your downloaded dataset)*
```powershell
python src/ingest_fluencybank.py --input_csv fluencybank_meta.csv --output_manifest data/processed/fluencybank_manifest.jsonl
```

### Step 3: Generate FluencyBank-N (Accuracy Bias Baseline)
Generate the clean synthetic counterpart to the real dataset:
```powershell
python src/tts.py --config configs/tts_fluencybank_n.yaml
```

### Step 4: Evaluate Baseline on Real Data
Evaluate the original wav2vec2 model on the real FluencyBank dataset:
```powershell
python src/asr.py --config <create_a_config_pointing_to_fluencybank_manifest>
python src/evaluate.py --config <create_a_config_for_evaluation>
```

### Step 5: Run Fine-Tuning (The Main Contribution)
Finally, fine-tune the model on the augmented data you generated in Step 1:
```powershell
python src/finetune.py --manifest data/augmented/manifests/stuttered_manifest_initial.jsonl
```
* **What to show:** After training finishes, point the `asr.py` script to use your new fine-tuned model (located in `./outputs/wav2vec2-stuttered`). Run evaluation again on FluencyBank. Show the professor the drop in Word Error Rate (WER) between the baseline model and your new fine-tuned model!
