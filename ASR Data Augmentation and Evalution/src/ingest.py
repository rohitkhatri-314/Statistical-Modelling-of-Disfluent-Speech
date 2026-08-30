import argparse
from typing import Any, Dict, Iterable, List, Optional

import datasets

from normalize import normalize_text
from utils import write_jsonl


def normalize_dataset_args(
    dataset_name: str,
    config_name: Optional[str],
    split: str,
):
    config_name = config_name if config_name else None

    # Compatibility aliases for common LibriSpeech ASR configs/splits.
    if dataset_name == "openslr/librispeech_asr":
        if config_name == "clean.100":
            config_name = "clean"

        if split == "validation.clean":
            split = "validation"
        elif split == "test.clean":
            split = "test"

    return dataset_name, config_name, split


def load_hf_dataset_stream(
    dataset_name: str,
    config_name: Optional[str],
    split: str,
):
    dataset_name, config_name, split = normalize_dataset_args(
        dataset_name,
        config_name,
        split,
    )

    try:
        if config_name:
            ds = datasets.load_dataset(
                dataset_name,
                name=config_name,
                split=split,
                streaming=True,
            )
        else:
            ds = datasets.load_dataset(
                dataset_name,
                split=split,
                streaming=True,
            )
    except TypeError:
        if config_name:
            ds = datasets.load_dataset(
                dataset_name,
                config_name,
                split=split,
                streaming=True,
            )
        else:
            ds = datasets.load_dataset(
                dataset_name,
                split=split,
                streaming=True,
            )

    return ds


def disable_example_decoding(ds):
    """
    Try to disable HF Datasets audio/image decoding.

    This is needed because the LibriSpeech ASR dataset has an audio column,
    and newer datasets versions try to decode audio using torchcodec.
    """

    try:
        ds = ds.cast_column("audio", datasets.Audio(decode=False))
    except Exception:
        pass

    try:
        ds.set_format("python", decode=False)
        return ds
    except Exception:
        pass

    try:
        ds.set_format(decode=False)
        return ds
    except Exception:
        pass

    try:
        return ds.with_format("python", decode=False)
    except Exception:
        pass

    try:
        return ds.with_format(decode=False)
    except Exception:
        pass

    return ds


def drop_audio_columns(ds):
    """
    Remove audio columns if possible.
    """

    # Try using known features.
    try:
        features = getattr(ds, "features", None)

        if features:
            audio_cols = []

            for name, feature in features.items():
                feature_type = str(getattr(feature, "_type", "")).lower()
                class_name = feature.__class__.__name__.lower()

                if "audio" in feature_type or "audio" in class_name:
                    audio_cols.append(name)

            if audio_cols:
                return ds.remove_columns(audio_cols)
    except Exception:
        pass

    # Fallback using column names.
    try:
        column_names = getattr(ds, "column_names", None)

        if column_names:
            audio_cols = [
                col
                for col in column_names
                if str(col).lower() in {"audio", "speech", "waveform"}
            ]

            if audio_cols:
                return ds.remove_columns(audio_cols)
    except Exception:
        pass

    # Final fallback: try removing the standard audio column.
    try:
        return ds.remove_columns(["audio"])
    except Exception:
        return ds


def keep_only_columns(ds, keep_columns: Optional[List[str]]):
    """
    Keep only transcript/id columns if possible.
    """

    if not keep_columns:
        return ds

    try:
        column_names = getattr(ds, "column_names", None)

        if column_names:
            keep_existing = [
                col for col in keep_columns if col in column_names
            ]

            if keep_existing:
                remove_cols = [
                    col
                    for col in column_names
                    if col not in keep_existing
                ]

                if remove_cols:
                    return ds.remove_columns(remove_cols)

                return ds
    except Exception:
        pass

    try:
        return ds.select_columns(keep_columns)
    except Exception:
        return ds


def iter_rows(
    dataset_name: str,
    config_name: str,
    split: str,
    keep_columns: Optional[List[str]] = None,
) -> Iterable[Dict[str, Any]]:
    ds = load_hf_dataset_stream(dataset_name, config_name, split)

    # Important order:
    # 1. Disable decoding first.
    # 2. Drop audio columns.
    # 3. Keep only needed columns.
    ds = disable_example_decoding(ds)
    ds = drop_audio_columns(ds)
    ds = keep_only_columns(ds, keep_columns)

    for row in ds:
        yield row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest clean LibriSpeech transcripts."
    )

    parser.add_argument(
        "--dataset",
        default="openslr/librispeech_asr",
        help="HuggingFace dataset name.",
    )
    parser.add_argument(
        "--config",
        default="clean",
        help="HuggingFace dataset config name.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Maximum number of clean utterances to keep.",
    )
    parser.add_argument(
        "--min_words",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max_words",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--text_field",
        default="text",
    )
    parser.add_argument(
        "--id_field",
        default="id",
    )
    parser.add_argument(
        "--output",
        default="data/processed/clean_manifest.jsonl",
    )

    args = parser.parse_args()

    keep_columns = [args.text_field]

    if args.id_field and args.id_field != args.text_field:
        keep_columns.append(args.id_field)

    records = []
    seen = 0

    for row in iter_rows(
        args.dataset,
        args.config,
        args.split,
        keep_columns=keep_columns,
    ):
        seen += 1

        raw_text = row.get(args.text_field, "")
        normalized_text = normalize_text(raw_text)
        words = normalized_text.split()

        if len(words) < args.min_words:
            continue

        if len(words) > args.max_words:
            continue

        utterance_id = row.get(args.id_field, None)

        if utterance_id is None:
            utterance_id = f"utt-{len(records):06d}"

        records.append(
            {
                "utterance_id": str(utterance_id),
                "raw_text": raw_text,
                "clean_text": normalized_text,
                "source_dataset": args.dataset,
                "source_config": args.config,
                "source_split": args.split,
            }
        )

        if len(records) >= args.max_samples:
            break

    write_jsonl(args.output, records)

    print(f"Seen rows: {seen}")
    print(f"Saved clean utterances: {len(records)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()