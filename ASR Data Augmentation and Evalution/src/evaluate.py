import argparse
import json
from pathlib import Path
from typing import Dict, List

import jiwer
import numpy as np
import pandas as pd
from bert_score import score as bert_score

from normalize import normalize_text
from utils import ensure_dir, load_yaml, read_jsonl


def compute_wer_metrics(reference: str, hypothesis: str) -> Dict[str, float]:
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)

    if not reference:
        return {
            "wer": 0.0 if not hypothesis else 1.0,
            "substitutions": 0.0,
            "deletions": 0.0,
            "insertions": float(len(hypothesis.split())) if hypothesis else 0.0,
            "hits": 0.0,
        }

    measures = jiwer.compute_measures(reference, hypothesis)

    return {
        "wer": float(measures["wer"]),
        "substitutions": float(measures["substitutions"]),
        "deletions": float(measures["deletions"]),
        "insertions": float(measures["insertions"]),
        "hits": float(measures["hits"]),
    }


def compute_bertscores(
    references: List[str],
    candidates: List[str],
    model_type: str,
    batch_size: int,
) -> List[float]:
    scores = []

    references = [r if r.strip() else " " for r in references]
    candidates = [c if c.strip() else " " for c in candidates]

    for i in range(0, len(references), batch_size):
        ref_batch = references[i : i + batch_size]
        cand_batch = candidates[i : i + batch_size]

        _, _, f_scores = bert_score(
            cand_batch,
            ref_batch,
            model_type=model_type,
            lang="en",
            rescale_with_layers=True,
            verbose=False,
        )

        scores.extend(f_scores.tolist())

    return scores


def json_default(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return str(obj)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute WER and BERTScore for ASR predictions."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to evaluation YAML config.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    predictions_path = cfg["predictions_path"]
    metrics_dir = Path(cfg["metrics_dir"])
    bertscore_model = cfg.get("bertscore_model", "microsoft/deberta-large-mnli")
    batch_size = cfg.get("batch_size", 8)

    ensure_dir(metrics_dir)

    predictions = list(read_jsonl(predictions_path))

    if not predictions:
        raise ValueError(f"No predictions found in {predictions_path}")

    references = []
    candidates = []

    for pred in predictions:
        reference = normalize_text(pred.get("reference_clean_text", ""))
        hypothesis = normalize_text(pred.get("hypothesis", ""))

        references.append(reference)
        candidates.append(hypothesis)

    print("Computing WER...")
    wer_records = []

    for pred, reference, hypothesis in zip(predictions, references, candidates):
        wer_metrics = compute_wer_metrics(reference, hypothesis)

        wer_records.append(
            {
                "record_id": pred.get("record_id"),
                "variant": pred.get("variant"),
                "speaker_id": pred.get("speaker_id"),
                "reference_clean_text": reference,
                "hypothesis": hypothesis,
                **wer_metrics,
            }
        )

    print("Computing BERTScore...")
    bert_scores = compute_bertscores(
        references=references,
        candidates=candidates,
        model_type=bertscore_model,
        batch_size=batch_size,
    )

    for record, fbert in zip(wer_records, bert_scores):
        record["fbert"] = float(fbert)

    df = pd.DataFrame(wer_records)

    per_utterance_path = metrics_dir / "per_utterance_metrics.csv"
    df.to_csv(per_utterance_path, index=False)

    summary = {
        "num_predictions": int(len(df)),
        "by_variant": {},
    }

    for variant, group in df.groupby("variant"):
        summary["by_variant"][str(variant)] = {
            "mean_wer": float(group["wer"].mean()),
            "mean_fbert": float(group["fbert"].mean()),
            "count": int(len(group)),
        }

    clean_df = df[df["variant"] == "clean"].set_index("record_id")
    stuttered_df = df[df["variant"] == "stuttered"].set_index("record_id")

    common_ids = clean_df.index.intersection(stuttered_df.index)

    if len(common_ids) > 0:
        delta_wer = (
            stuttered_df.loc[common_ids, "wer"]
            - clean_df.loc[common_ids, "wer"]
        ).mean()

        delta_fbert = (
            stuttered_df.loc[common_ids, "fbert"]
            - clean_df.loc[common_ids, "fbert"]
        ).mean()

        summary["paired_comparison"] = {
            "num_pairs": int(len(common_ids)),
            "mean_delta_wer": float(delta_wer),
            "mean_delta_fbert": float(delta_fbert),
        }

    summary_path = metrics_dir / "summary_metrics.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=json_default)

    print(f"Saved per-utterance metrics: {per_utterance_path}")
    print(f"Saved summary metrics: {summary_path}")


if __name__ == "__main__":
    main()
