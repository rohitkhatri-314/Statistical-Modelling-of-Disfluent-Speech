"""
inference_xtts.py
Standalone inference script for fine-tuned XTTS-v2 stuttered speech model.
Run on Google Colab (T4 GPU) or any CUDA-enabled machine.

Usage:
  python inference_xtts.py --text "I I I want to say something" --ref-wav /path/to/reference.wav
  python inference_xtts.py --text "Up until last year I was a co co covert stutterer." --output /content/output.wav
  python inference_xtts.py --interactive   # interactive mode: type sentences, hear them spoken
"""

import os
import sys
import argparse
import torch
import soundfile as sf
import numpy as np


# ── 0. Patch gpt_inference.py for transformers KV-cache compatibility ────────
def patch_gpt_inference():
    """Fix tuple vs DynamicCache incompatibility in installed coqui-tts."""
    candidates = [
        "/usr/local/lib/python3.13/dist-packages/TTS/tts/layers/xtts/gpt_inference.py",
        "/usr/local/lib/python3.12/dist-packages/TTS/tts/layers/xtts/gpt_inference.py",
        "/usr/local/lib/python3.11/dist-packages/TTS/tts/layers/xtts/gpt_inference.py",
    ]
    # Also try to find it dynamically
    try:
        import TTS
        pkg_dir = os.path.dirname(TTS.__file__)
        candidates.insert(0, os.path.join(pkg_dir, "tts", "layers", "xtts", "gpt_inference.py"))
    except ImportError:
        pass

    for py_path in candidates:
        if os.path.isfile(py_path):
            with open(py_path, "r") as f:
                code = f.read()
            old = "past_key_values.get_seq_length()"
            safe = '(past_key_values.get_seq_length() if hasattr(past_key_values, "get_seq_length") else past_key_values[0][0].shape[-2])'
            if old in code:
                code = code.replace(old, safe)
                with open(py_path, "w") as f:
                    f.write(code)
                print(f"[Patch] Fixed KV-cache compatibility in {py_path}")
            return
    print("[Patch] gpt_inference.py not found — skipping (may already be compatible)")


# Apply patch before importing TTS
patch_gpt_inference()

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager


# ── 1. Model Loading ─────────────────────────────────────────────────────────
def load_model(checkpoint_root, checkpoint_path=None, device="cuda"):
    """
    Load base XTTS-v2 architecture and inject fine-tuned GPT weights.

    Args:
        checkpoint_root: Path to xtts_stutter_checkpoints directory
                         (contains XTTS_v2.0_original_model_files/)
        checkpoint_path: Direct path to fine-tuned best_model.pth
        device: "cuda" or "cpu"

    Returns:
        (model, config) tuple ready for synthesis
    """
    orig_dir = os.path.join(checkpoint_root, "XTTS_v2.0_original_model_files")
    os.makedirs(orig_dir, exist_ok=True)

    # Download base config.json if not present
    base_config_path = os.path.join(orig_dir, "config.json")
    if not os.path.isfile(base_config_path):
        import urllib.request
        print("Downloading official XTTS-v2 config.json...")
        urllib.request.urlretrieve(
            "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json",
            base_config_path,
        )

    # Download base model files if not present
    CDN = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main"
    for fname in ["dvae.pth", "mel_stats.pth", "model.pth", "vocab.json"]:
        fpath = os.path.join(orig_dir, fname)
        if not os.path.isfile(fpath):
            print(f"Downloading {fname}...")
            ModelManager._download_model_files([f"{CDN}/{fname}"], orig_dir, progress_bar=True)

    # 1. Load base architecture
    print("Loading base XTTS-v2 architecture...")
    config = XttsConfig()
    config.load_json(base_config_path)
    model = Xtts.init_from_config(config)

    model.load_checkpoint(
        config,
        checkpoint_dir=orig_dir,
        vocab_path=os.path.join(orig_dir, "vocab.json"),
        eval=True,
        use_deepspeed=False,
    )

    # 2. Resolve fine-tuned checkpoint path
    best_model_path = checkpoint_path
    if not best_model_path or not os.path.isfile(best_model_path):
        # Known default run path
        default_path = os.path.join(
            checkpoint_root,
            "xtts_stutter_partial-November-16-2026_10+08AM-0000000",
            "best_model.pth"
        )
        if os.path.isfile(default_path):
            best_model_path = default_path
        else:
            import glob
            run_dirs = sorted(
                glob.glob(os.path.join(checkpoint_root, "run-*"))
                + glob.glob(os.path.join(checkpoint_root, "xtts_stutter_partial-*"))
            )
            if run_dirs:
                latest_run = run_dirs[-1]
                cand = os.path.join(latest_run, "best_model.pth")
                if os.path.isfile(cand):
                    best_model_path = cand
                else:
                    ckpts = sorted(glob.glob(os.path.join(latest_run, "checkpoint_*.pth")))
                    if ckpts:
                        best_model_path = ckpts[-1]

    if not best_model_path or not os.path.isfile(best_model_path):
        print("[Warning] No fine-tuned checkpoint found — using base model only.")
        model.eval()
        model.to(device)
        return model, config

    # 3. Inject fine-tuned weights (strip 'xtts.' prefix from GPTTrainer keys)
    print(f"Injecting fine-tuned weights from: {best_model_path}")
    ckpt = torch.load(best_model_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    clean_state = {}
    for k, v in state_dict.items():
        key = k[5:] if k.startswith("xtts.") else k
        clean_state[key] = v

    model.load_state_dict(clean_state, strict=False)
    model.eval()
    model.to(device)

    print(f"✓ Fine-tuned model loaded on {device.upper()}")
    return model, config


# ── 2. Synthesis ──────────────────────────────────────────────────────────────
def synthesize(model, config, text, ref_wav, language="en",
               temperature=0.7, repetition_penalty=10.0, top_k=50, top_p=0.85):
    """
    Generate speech from text using fine-tuned XTTS-v2.

    Args:
        model: Loaded Xtts model
        config: XttsConfig
        text: Input text (can include verbatim disfluencies like "I I I want")
        ref_wav: Path to a reference WAV file for voice cloning (3-10 seconds)
        language: Language code (default "en")
        temperature: Sampling temperature (0.5=conservative, 0.85=expressive)
        repetition_penalty: Penalise repeated tokens (higher = less repetition)

    Returns:
        numpy array of audio samples at 24000 Hz
    """
    outputs = model.synthesize(
        text,
        config,
        speaker_wav=ref_wav,
        language=language,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        top_p=top_p,
    )
    return np.array(outputs["wav"])


# ── 3. CLI ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="XTTS-v2 Fine-Tuned Inference — Stuttered Speech")
    parser.add_argument("--checkpoint-root",
                        default="/content/drive/MyDrive/xtts_stutter_checkpoints",
                        help="Root directory containing XTTS_v2.0_original_model_files/ and run-*/")
    parser.add_argument("--checkpoint-path", type=str,
                        default="/content/drive/MyDrive/xtts_stutter_checkpoints/xtts_stutter_partial-November-16-2026_10+08AM-0000000/best_model.pth",
                        help="Direct path to best_model.pth")
    parser.add_argument("--text", type=str, default=None,
                        help="Text to synthesize (include verbatim disfluencies)")
    parser.add_argument("--ref-wav", type=str,
                        default="/content/dataset_disfluent/wavs/24fb_PAR_000_29.240_34.830.wav",
                        help="Reference speaker WAV for voice cloning")
    parser.add_argument("--output", type=str, default="/content/stutter_output.wav",
                        help="Output WAV file path")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (0.5-0.85)")
    parser.add_argument("--repetition-penalty", type=float, default=10.0)
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode: type sentences and hear them spoken")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model, config = load_model(args.checkpoint_root, checkpoint_path=args.checkpoint_path, device=device)

    if args.interactive:
        print("\n" + "=" * 60)
        print("  XTTS-v2 Stuttered Speech — Interactive Mode")
        print("  Type a sentence and press Enter to generate speech.")
        print("  Include disfluencies in the text for stuttered output.")
        print("  Type 'quit' or 'exit' to stop.")
        print("=" * 60 + "\n")

        count = 0
        while True:
            try:
                text = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text or text.lower() in ("quit", "exit", "q"):
                break

            count += 1
            out_path = f"/content/stutter_output_{count:03d}.wav"
            wav = synthesize(model, config, text, args.ref_wav,
                             language=args.language, temperature=args.temperature,
                             repetition_penalty=args.repetition_penalty)
            sf.write(out_path, wav, 24000)
            print(f"  → Saved: {out_path} ({len(wav)/24000:.1f}s)\n")

        print("Done.")
    else:
        if not args.text:
            args.text = "Up until last year I was a co co covert stutterer."

        print(f"\nSynthesizing: \"{args.text}\"")
        wav = synthesize(model, config, args.text, args.ref_wav,
                         language=args.language, temperature=args.temperature,
                         repetition_penalty=args.repetition_penalty)
        sf.write(args.output, wav, 24000)
        print(f"✓ Audio saved: {args.output} ({len(wav)/24000:.1f}s)")


if __name__ == "__main__":
    main()
