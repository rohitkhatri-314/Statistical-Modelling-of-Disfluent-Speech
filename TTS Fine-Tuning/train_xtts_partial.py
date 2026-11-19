"""
train_xtts_partial.py
Partial fine-tuning of XTTS-v2 on stuttered/disfluent speech.
Structure mirrors the official Coqui recipe:
  recipes/ljspeech/xtts_v2/train_gpt_xtts.py
Adds selective layer freezing on top of GPTTrainer.
"""

import os
import argparse
import torch
from pathlib import Path

# ── Imports (match official recipe exactly) ──────────────────────────────────
from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import (
    GPTArgs,
    GPTTrainer,
    GPTTrainerConfig,
)
# XttsAudioConfig location differs between coqui-tts pip package and GitHub source
try:
    from TTS.tts.layers.xtts.trainer.gpt_trainer import XttsAudioConfig
except ImportError:
    from TTS.tts.models.xtts import XttsAudioConfig
from TTS.utils.manage import ModelManager


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="XTTS-v2 Partial Fine-Tuning — Stuttered Speech")
    p.add_argument("--dataset-dir",    default="/content/dataset_disfluent",
                   help="Extracted dataset folder (must contain metadata_train.csv + wavs/)")
    p.add_argument("--output-path",    default="/content/drive/MyDrive/xtts_stutter_checkpoints",
                   help="Where checkpoints are saved (use Drive path in Colab)")
    p.add_argument("--base-model-path", default=None,
                   help="Local XTTS-v2 folder (dvae.pth, mel_stats.pth, model.pth, vocab.json). "
                        "Leave blank to auto-download.")
    p.add_argument("--batch-size",     type=int,   default=2,    help="Per-step batch size (2–3 on T4)")
    p.add_argument("--grad-accum",     type=int,   default=84,   help="Gradient accumulation steps")
    p.add_argument("--lr",             type=float, default=5e-6, help="Learning rate")
    p.add_argument("--unfreeze-gpt-layers", type=int, default=6,
                   help="How many top GPT transformer layers to unfreeze")
    p.add_argument("--continue-path",  type=str,   default=None,
                   help="Path to an existing run folder in Google Drive to resume interrupted training")
    return p.parse_args()


# ── Partial Layer Freezing ────────────────────────────────────────────────────
def apply_partial_freezing(model, unfreeze_layers=6):
    """
    Freeze HiFi-GAN vocoder + text embeddings + lower GPT layers.
    Train only the top `unfreeze_layers` transformer blocks.
    """
    print("\n--- Configuring Partial Layer Freezing ---")

    # 1. Freeze HiFi-GAN vocoder
    if hasattr(model, "xtts") and hasattr(model.xtts, "hifigan_decoder"):
        for p in model.xtts.hifigan_decoder.parameters():
            p.requires_grad = False
        print("[Frozen] HiFi-GAN Vocoder")

    # 2. Freeze DVAE
    if hasattr(model, "xtts") and hasattr(model.xtts, "dvae"):
        for p in model.xtts.dvae.parameters():
            p.requires_grad = False
        print("[Frozen] DVAE")

    # 3. Freeze speaker encoder
    if hasattr(model, "xtts") and hasattr(model.xtts, "gpt") and hasattr(model.xtts.gpt, "conditioning_encoder"):
        for p in model.xtts.gpt.conditioning_encoder.parameters():
            p.requires_grad = False
        print("[Frozen] Speaker Conditioning Encoder")

    # 4. Freeze text embeddings
    if hasattr(model, "xtts") and hasattr(model.xtts, "gpt") and hasattr(model.xtts.gpt, "text_embedding"):
        for p in model.xtts.gpt.text_embedding.parameters():
            p.requires_grad = False
        print("[Frozen] GPT Text Embeddings")

    # 5. Freeze lower GPT transformer layers, unfreeze top N
    gpt_core = None
    if hasattr(model, "xtts") and hasattr(model.xtts, "gpt"):
        gpt_obj = model.xtts.gpt
        for attr in ("gpt", "transformer"):
            if hasattr(gpt_obj, attr):
                gpt_core = getattr(gpt_obj, attr)
                break

    blocks = None
    if gpt_core is not None:
        for attr in ("layers", "blocks", "h"):
            if hasattr(gpt_core, attr):
                blocks = getattr(gpt_core, attr)
                break

    if blocks is not None:
        total = len(blocks)
        freeze_up_to = max(0, total - unfreeze_layers)
        for i in range(freeze_up_to):
            for p in blocks[i].parameters():
                p.requires_grad = False
        for i in range(freeze_up_to, total):
            for p in blocks[i].parameters():
                p.requires_grad = True
        print(f"[Partial] Froze bottom {freeze_up_to} / {total} GPT layers. Training top {unfreeze_layers}.")
    else:
        print("[Notice] Could not locate GPT layer list — training full GPT backbone.")

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = (trainable_params / total_params * 100) if total_params > 0 else 0
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,} ({pct:.2f}%)")
    print("-------------------------------------------\n")


# ── Main Training Function ────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── 1. Resolve / download base model files ───────────────────────────────
    #   Official recipe downloads dvae.pth, mel_stats.pth, model.pth, vocab.json
    #   separately from Coqui's CDN. ModelManager().download_model() may place
    #   them in a different layout, so we replicate the recipe approach.
    CHECKPOINTS_OUT = os.path.join(args.output_path, "XTTS_v2.0_original_model_files")
    os.makedirs(CHECKPOINTS_OUT, exist_ok=True)

    CDN = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main"
    FILE_URLS = {
        "config.json":   f"{CDN}/config.json",
        "dvae.pth":      f"{CDN}/dvae.pth",
        "mel_stats.pth": f"{CDN}/mel_stats.pth",
        "model.pth":     f"{CDN}/model.pth",
        "vocab.json":    f"{CDN}/vocab.json",
    }

    if args.base_model_path:
        # User supplied a local directory — use it directly
        base_dir = Path(args.base_model_path)
        dvae_checkpoint = str(base_dir / "dvae.pth")
        mel_norm_file   = str(base_dir / "mel_stats.pth")
        xtts_checkpoint = str(base_dir / "model.pth")
        tokenizer_file  = str(base_dir / "vocab.json")
    else:
        # Download each file from CDN if not already present (exactly like official recipe)
        need_download = [
            url for name, url in FILE_URLS.items()
            if not os.path.isfile(os.path.join(CHECKPOINTS_OUT, name))
        ]
        if need_download:
            print(f"Downloading {len(need_download)} XTTS-v2 model file(s)...")
            ModelManager._download_model_files(need_download, CHECKPOINTS_OUT, progress_bar=True)
        else:
            print("All XTTS-v2 model files already cached.")

        dvae_checkpoint = os.path.join(CHECKPOINTS_OUT, "dvae.pth")
        mel_norm_file   = os.path.join(CHECKPOINTS_OUT, "mel_stats.pth")
        xtts_checkpoint = os.path.join(CHECKPOINTS_OUT, "model.pth")
        tokenizer_file  = os.path.join(CHECKPOINTS_OUT, "vocab.json")

    for f in [dvae_checkpoint, mel_norm_file, xtts_checkpoint, tokenizer_file]:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Required model file not found: {f}")
    print(f"Model files: {os.path.dirname(dvae_checkpoint)}")

    # ── 2. Dataset configuration ──────────────────────────────────────────────
    dataset_path = Path(args.dataset_dir).resolve()
    print(f"Configuring dataset from: {dataset_path}")

    if not (dataset_path / "metadata_train.csv").exists():
        raise FileNotFoundError(
            f"metadata_train.csv not found in {dataset_path}.\n"
            "Make sure you extracted dataset_disfluent.zip to /content/ first:\n"
            "  !unzip -q /content/dataset_disfluent.zip -d /content/"
        )

    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="fluencybank_stuttered",
        path=str(dataset_path),
        meta_file_train="metadata_train.csv",
        meta_file_val="metadata_eval.csv",
        language="en",
    )

    DATASETS_CONFIG_LIST = [config_dataset]

    # ── 3. Model arguments ────────────────────────────────────────────────────
    model_args = GPTArgs(
        max_conditioning_length=132300,       # 6 s at 22050 Hz
        min_conditioning_length=66150,        # 3 s
        debug_loading_failures=False,
        max_wav_length=255995,                # ~11.6 s (XTTS token limit)
        max_text_length=200,
        mel_norm_file=mel_norm_file,
        dvae_checkpoint=dvae_checkpoint,
        xtts_checkpoint=xtts_checkpoint,
        tokenizer_file=tokenizer_file,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    # ── 4. Audio configuration ────────────────────────────────────────────────
    audio_config = XttsAudioConfig(
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000,
    )

    # ── 5. GPTTrainerConfig — mirrors official recipe exactly ─────────────────
    #      NOTE: datasets are NOT a field here; they go into load_tts_samples()
    OUT_PATH = args.output_path
    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name="xtts_stutter_partial",
        project_name="XTTS_trainer",
        run_description="XTTS-v2 partial fine-tuning on FluencyBank disfluent speech",
        dashboard_logger="tensorboard",
        logger_uri=None,
        audio=audio_config,
        batch_size=args.batch_size,
        batch_group_size=48,
        eval_batch_size=args.batch_size,
        num_loader_workers=4,          # 2 CPUs in Colab — keep at 4 max
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=1000,                # save more often than default (10000) for Colab sessions
        save_n_checkpoints=2,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=args.lr,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={
            "milestones": [50000 * 18, 150000 * 18, 300000 * 18],
            "gamma": 0.5,
            "last_epoch": -1,
        },
    )

    # ── 6. Load dataset samples ───────────────────────────────────────────────
    #   Installed coqui-tts expects load_tts_samples(config_obj, ...) where
    #   config_obj.datasets exists.  GitHub source takes a bare list instead.
    #   We handle both by attaching the list to the config object.
    config.datasets = DATASETS_CONFIG_LIST
    print("Loading audio-transcript pairs...")
    train_samples, eval_samples = load_tts_samples(
        config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    print(f"  → {len(train_samples)} training samples, {len(eval_samples)} eval samples")

    # ── 7. Initialise model (official pattern: init_from_config) ──────────────
    print("Initialising GPTTrainer...")
    model = GPTTrainer.init_from_config(config)

    # ── 8. Apply partial freezing ─────────────────────────────────────────────
    apply_partial_freezing(model, unfreeze_layers=args.unfreeze_gpt_layers)

    # ── 9. Build Trainer and start ────────────────────────────────────────────
    trainer = Trainer(
        TrainerArgs(
            continue_path=args.continue_path,
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=True,
            grad_accum_steps=args.grad_accum,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print("\nStarting partial fine-tuning — good luck! 🎙️\n")
    trainer.fit()


if __name__ == "__main__":
    main()
