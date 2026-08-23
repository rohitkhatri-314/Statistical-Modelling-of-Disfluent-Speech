import argparse
from pathlib import Path
from typing import Dict

import torch
from transformers import pipeline

from normalize import normalize_text
from utils import ensure_dir, load_yaml, read_jsonl, write_jsonl


def get_torch_device(device_setting: str) -> int:
    if device_setting == "cuda":
        return 0 if torch.cuda.is_available() else -1

    if device_setting == "cpu":
        return -1

    # auto
    return 0 if torch.cuda.is_available() else -1


def create_asr_pipeline(cfg: Dict):
    asr_model = cfg["asr_model"]
    device = get_torch_device(cfg.get("device", "auto"))

    chunk_length_s = cfg.get("chunk_length_s", 15)
    stride_length_s = cfg.get("stride_length_s", 2)

    try:
        asr = pipeline(
            "automatic-speech-recognition",
            model=asr_model,
            device=device,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
        )
    except TypeError:
        asr = pipeline(
            "automatic-speech-recognition",
            model=asr_model,
            device=device,
        )

    return asr


def transcribe_audio(asr, audio_path: str) -> str:
    result = asr(str(audio_path))
    if isinstance(result, dict):
        return result.get("text", "")
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ASR inference on generated clean and stuttered audio."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to evaluation YAML config.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    audio_manifest = cfg["audio_manifest"]
    predictions_path = cfg["predictions_path"]

    ensure_dir(Path(predictions_path).parent)

    print("Loading ASR pipeline...")
    asr = create_asr_pipeline(cfg)

    predictions = []

    for record in read_jsonl(audio_manifest):
        record_id = record["record_id"]
        clean_text = record.get("clean_text", "")
        stuttered_text = record.get("stuttered_text", "")
        speaker_id = record.get("speaker_id", "")

        clean_audio_path = record.get("clean_audio_path")
        stuttered_audio_path = record.get("stuttered_audio_path")

        if clean_audio_path and Path(clean_audio_path).exists():
            hypothesis_raw = transcribe_audio(asr, clean_audio_path)
            hypothesis = normalize_text(hypothesis_raw)

            predictions.append(
                {
                    "record_id": record_id,
                    "variant": "clean",
                    "speaker_id": speaker_id,
                    "audio_path": clean_audio_path,
                    "reference_clean_text": clean_text,
                    "intended_text": clean_text,
                    "hypothesis_raw": hypothesis_raw,
                    "hypothesis": hypothesis,
                }
            )
        else:
            print(f"Missing clean audio for {record_id}")

        if stuttered_audio_path and Path(stuttered_audio_path).exists():
            hypothesis_raw = transcribe_audio(asr, stuttered_audio_path)
            hypothesis = normalize_text(hypothesis_raw)

            predictions.append(
                {
                    "record_id": record_id,
                    "variant": "stuttered",
                    "speaker_id": speaker_id,
                    "audio_path": stuttered_audio_path,
                    "reference_clean_text": clean_text,
                    "intended_text": stuttered_text,
                    "hypothesis_raw": hypothesis_raw,
                    "hypothesis": hypothesis,
                }
            )
        else:
            print(f"Missing stuttered audio for {record_id}")

    write_jsonl(predictions_path, predictions)

    print(f"Saved ASR predictions: {len(predictions)}")
    print(f"Output: {predictions_path}")


if __name__ == "__main__":
    main()
