import argparse
from typing import Any, Dict, Iterable

import datasets

from normalize import normalize_text
from utils import write_jsonl


def load_hf_dataset_stream(
    dataset_name: str,
    config_name: str,
    split: str,
):
    config_name = config_name if config_name else None

    try:
        ds = datasets.load_dataset(
            dataset_name,
            config_name,
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
    except TypeError:
        ds = datasets.load_dataset(
            dataset_name,
            config_name,
            split=split,
            streaming=True,
        )

    return ds


def iter_rows(
    dataset_name: str,
    config_name: str,
    split: str,
) -> Iterable[Dict[str, Any]]:
    ds = load_hf_dataset_stream(dataset_name, config_name, split)
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
        default="clean.100",
        help="HuggingFace dataset config name.",
    )
    parser.add_argument(
        "--split",
        default="validation.clean",
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

    records = []
    seen = 0

    for row in iter_rows(args.dataset, args.config, args.split):
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
