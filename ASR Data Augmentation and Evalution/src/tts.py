import argparse
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Tuple

import soundfile as sf
import torch
from datasets import load_dataset
from transformers import (
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
)

from utils import ensure_dir, load_yaml, read_jsonl, write_jsonl


def safe_filename_part(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", str(text))


def stable_mod(text: str, mod: int) -> int:
    return int(hashlib.md5(str(text).encode("utf-8")).hexdigest(), 16) % mod


def get_device(device_setting: str) -> torch.device:
    if device_setting == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_setting == "cpu":
        return torch.device("cpu")

    if device_setting == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    return torch.device(device_setting)


def load_speaker_embeddings(cfg: Dict) -> List[Tuple[str, torch.Tensor]]:
    dataset_name = cfg.get(
        "speaker_embeddings_dataset",
        "Matthijs/cmu-arctic-xvectors",
    )
    split = cfg.get("speaker_embeddings_split", "validation")
    requested_speaker_ids = cfg.get("speaker_ids", [])
    num_speakers = cfg.get("num_speakers", 8)

    ds = load_dataset(dataset_name, split=split)

    # Remove heavy columns if present.
    keep_cols = []
    for col in ds.column_names:
        if col in {"xvector", "speaker_id", "speaker", "id", "name"}:
            keep_cols.append(col)

    if keep_cols:
        remove_cols = [col for col in ds.column_names if col not in keep_cols]
        if remove_cols:
            ds = ds.remove_columns(remove_cols)

    def row_matches(row: Dict, speaker_id: str) -> bool:
        for key in ["speaker_id", "speaker", "id", "name"]:
            if key in row and str(row[key]).lower() == str(speaker_id).lower():
                return True
        return False

    available = []

    # First try requested speaker IDs.
    for speaker_id in requested_speaker_ids:
        row = None

        for r in ds:
            if row_matches(r, speaker_id):
                row = r
                break

        if row is None and str(speaker_id).isdigit():
            idx = int(speaker_id)
            if idx < len(ds):
                row = ds[idx]

        if row is not None and "xvector" in row:
            available.append(
                (
                    str(speaker_id),
                    torch.tensor(row["xvector"]).float(),
                )
            )

    # Fallback: use first N available speaker embeddings.
    if not available:
        for i, row in enumerate(ds):
            if i >= num_speakers:
                break

            if "xvector" not in row:
                continue

            speaker_id = (
                row.get("speaker_id")
                or row.get("speaker")
                or row.get("id")
                or row.get("name")
                or f"spk_{i}"
            )

            available.append(
                (
                    str(speaker_id),
                    torch.tensor(row["xvector"]).float(),
                )
            )

    if not available:
        raise ValueError(
            "Could not load speaker embeddings from "
            f"{dataset_name} split {split}"
        )

    return available


def synthesize_text(
    text: str,
    processor: SpeechT5Processor,
    model: SpeechT5ForTextToSpeech,
    vocoder: SpeechT5HifiGan,
    speaker_embedding: torch.Tensor,
    device: torch.device,
):
    inputs = processor(text=text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    speaker_embedding = speaker_embedding.unsqueeze(0).to(device)

    with torch.no_grad():
        speech = model.generate_speech(
            input_ids,
            speaker_embedding,
            vocoder=vocoder,
        )

    return speech.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate clean and stuttered synthetic speech."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to TTS YAML config.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    input_manifest = cfg["input_manifest"]
    output_manifest = cfg["output_manifest"]
    audio_dir = Path(cfg["audio_dir"])

    clean_dir = audio_dir / "clean"
    stuttered_dir = audio_dir / "stuttered"

    ensure_dir(clean_dir)
    ensure_dir(stuttered_dir)

    sample_rate = cfg.get("sample_rate", 16000)
    max_text_chars = cfg.get("max_text_chars", 250)
    overwrite = cfg.get("overwrite", False)

    device = get_device(cfg.get("device", "auto"))

    print("Loading SpeechT5 models...")
    processor = SpeechT5Processor.from_pretrained(cfg["tts_model"])
    model = SpeechT5ForTextToSpeech.from_pretrained(cfg["tts_model"]).to(device)
    model.eval()

    vocoder = SpeechT5HifiGan.from_pretrained(cfg["vocoder"]).to(device)
    vocoder.eval()

    print("Loading speaker embeddings...")
    speakers = load_speaker_embeddings(cfg)
    speaker_map = {speaker_id: embedding for speaker_id, embedding in speakers}

    print(f"Using {len(speakers)} speakers:")
    for speaker_id, _ in speakers:
        print(f"  - {speaker_id}")

    audio_records = []

    for record in read_jsonl(input_manifest):
        record_id = record.get("augmented_id") or record.get("utterance_id")
        clean_text = record.get("clean_text", "").strip()
        stuttered_text = record.get("stuttered_text", "").strip()

        if not clean_text or not stuttered_text:
            continue

        if len(clean_text) > max_text_chars or len(stuttered_text) > max_text_chars:
            print(f"Skipping {record_id}: text too long for TTS config.")
            continue

        speaker_id = record.get("speaker_id", None)

        if speaker_id not in speaker_map:
            speaker_index = stable_mod(record_id, len(speakers))
            speaker_id = speakers[speaker_index][0]

        speaker_embedding = speaker_map[speaker_id]

        safe_record_id = safe_filename_part(record_id)
        safe_speaker_id = safe_filename_part(speaker_id)

        clean_path = clean_dir / f"{safe_record_id}_clean_spk_{safe_speaker_id}.flac"
        stuttered_path = stuttered_dir / f"{safe_record_id}_stuttered_spk_{safe_speaker_id}.flac"

        try:
            if overwrite or not clean_path.exists():
                clean_speech = synthesize_text(
                    clean_text,
                    processor,
                    model,
                    vocoder,
                    speaker_embedding,
                    device,
                )
                sf.write(clean_path, clean_speech, sample_rate)
                clean_duration = len(clean_speech) / sample_rate
            else:
                clean_duration = sf.info(clean_path).duration

            if overwrite or not stuttered_path.exists():
                stuttered_speech = synthesize_text(
                    stuttered_text,
                    processor,
                    model,
                    vocoder,
                    speaker_embedding,
                    device,
                )
                sf.write(stuttered_path, stuttered_speech, sample_rate)
                stuttered_duration = len(stuttered_speech) / sample_rate
            else:
                stuttered_duration = sf.info(stuttered_path).duration

            audio_records.append(
                {
                    "record_id": record_id,
                    "original_utterance_id": record.get(
                        "original_utterance_id",
                        record.get("utterance_id"),
                    ),
                    "clean_text": clean_text,
                    "stuttered_text": stuttered_text,
                    "speaker_id": speaker_id,
                    "clean_audio_path": str(clean_path),
                    "stuttered_audio_path": str(stuttered_path),
                    "clean_duration_seconds": clean_duration,
                    "stuttered_duration_seconds": stuttered_duration,
                    "is_disfluent": True,
                    "disfluency_events": record.get("disfluency_events", []),
                    "primary_disfluency_type": record.get(
                        "primary_disfluency_type",
                        None,
                    ),
                }
            )

        except Exception as e:
            print(f"Failed TTS for {record_id}: {e}")
            continue

    write_jsonl(output_manifest, audio_records)

    print(f"Generated/verified audio records: {len(audio_records)}")
    print(f"Audio manifest: {output_manifest}")


if __name__ == "__main__":
    main()
