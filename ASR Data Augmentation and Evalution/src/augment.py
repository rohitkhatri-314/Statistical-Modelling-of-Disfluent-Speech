import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils import load_yaml, read_jsonl, write_jsonl


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(highigh))


def choose_event_type(cfg: Dict[str, Any], rng: random.Random) -> str:
    probs = cfg.get("event_probabilities", {})
    if not probs:
        return rng.choice(
            [
                "word_repetition",
                "phrase_repetition",
                "interjection",
            ]
        )

    total = float(sum(probs.values()))
    r = rng.random() * total
    cumulative = 0.0

    for event_type, prob in probs.items():
        cumulative += float(prob)
        if r <= cumulative:
            return event_type

    return list(probs.keys())[-1]


def add_word_repetition(
    tokens: List[str],
    cfg: Dict[str, Any],
    rng: random.Random,
    max_insertions: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    if not tokens or max_insertions <= 0:
        return tokens, []

    n = len(tokens)

    num_select_min = cfg.get("num_words_to_select_min", 1)
    num_select_max = cfg.get("num_words_to_select_max", 3)
    rep_min = cfg.get("additional_repetitions_min", 1)
    rep_max = cfg.get("additional_repetitions_max", 4)

    num_select = rng.randint(num_select_min, num_select_max)
    num_select = min(num_select, n)

    if num_select <= 0:
        return tokens, []

    selected_indices = rng.sample(range(n), num_select)
    selected = {}
    remaining_insertions = max_insertions

    for idx in sorted(selected_indices):
        if remaining_insertions <= 0:
            break

        reps = rng.randint(rep_min, rep_max)
        reps = min(reps, remaining_insertions)

        if reps <= 0:
            continue

        selected[idx] = reps
        remaining_insertions -= reps

    if not selected:
        return tokens, []

    new_tokens = []
    events = []

    for i, token in enumerate(tokens):
        if i in selected:
            reps = selected[i]
            new_tokens.extend([token] * (reps + 1))
            events.append(
                {
                    "type": "word_repetition",
                    "original_token_index": i,
                    "target": token,
                    "additional_repetitions": reps,
                }
            )
        else:
            new_tokens.append(token)

    return new_tokens, events


def add_phrase_repetition(
    tokens: List[str],
    cfg: Dict[str, Any],
    rng: random.Random,
    max_insertions: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    if not tokens or max_insertions <= 0:
        return tokens, []

    n = len(tokens)

    phrase_len_min = cfg.get("phrase_length_min", 2)
    phrase_len_max = cfg.get("phrase_length_max", 4)
    rep_min = cfg.get("additional_repetitions_min", 1)
    rep_max = cfg.get("additional_repetitions_max", 3)

    phrase_len_max = min(phrase_len_max, n)

    if phrase_len_min > phrase_len_max:
        return tokens, []

    phrase_len = rng.randint(phrase_len_min, phrase_len_max)

    if phrase_len > n:
        return tokens, []

    start_max = n - phrase_len
    if start_max < 0:
        return tokens, []

    start = rng.randint(0, start_max)

    reps = rng.randint(rep_min, rep_max)

    # Each additional phrase repetition inserts phrase_len tokens.
    reps = min(reps, max_insertions // phrase_len)

    if reps <= 0:
        return tokens, []

    phrase = tokens[start : start + phrase_len]

    new_tokens = (
        tokens[:start]
        + phrase * (reps + 1)
        + tokens[start + phrase_len :]
    )

    events = [
        {
            "type": "phrase_repetition",
            "original_start_index": start,
            "phrase": " ".join(phrase),
            "phrase_length": phrase_len,
            "additional_repetitions": reps,
        }
    ]

    return new_tokens, events


def add_interjections(
    tokens: List[str],
    cfg: Dict[str, Any],
    rng: random.Random,
    max_insertions: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    if not tokens or max_insertions <= 0:
        return tokens, []

    n = len(tokens)

    interjections = cfg.get("interjections", ["uh", "um"])
    loc_min = cfg.get("num_insertion_locations_min", 1)
    loc_max = cfg.get("num_insertion_locations_max", 4)
    rep_min = cfg.get("additional_repetitions_min", 1)
    rep_max = cfg.get("additional_repetitions_max", 4)

    num_locations = rng.randint(loc_min, loc_max)

    # There are n + 1 possible insertion positions.
    # Also cap by max_insertions because each location needs at least one token.
    num_locations = min(num_locations, n + 1, max_insertions)

    if num_locations <= 0:
        return tokens, []

    positions = rng.sample(range(n + 1), num_locations)
    insert_map = {}
    remaining_insertions = max_insertions

    for pos in sorted(positions):
        if remaining_insertions <= 0:
            break

        reps = rng.randint(rep_min, rep_max)
        reps = min(reps, remaining_insertions)

        if reps <= 0:
            continue

        interjection = rng.choice(interjections)
        insert_map[pos] = [interjection] * reps
        remaining_insertions -= reps

    if not insert_map:
        return tokens, []

    new_tokens = []

    for i in range(n + 1):
        if i in insert_map:
            new_tokens.extend(insert_map[i])
        if i < n:
            new_tokens.append(tokens[i])

    events = []

    for pos, inserted_tokens in insert_map.items():
        events.append(
            {
                "type": "interjection",
                "insertion_position": pos,
                "interjection": inserted_tokens[0],
                "repetitions": len(inserted_tokens),
            }
        )

    return new_tokens, events


def augment_record(
    record: Dict[str, Any],
    cfg: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    clean_text = record["clean_text"]
    tokens = clean_text.split()

    if len(tokens) < 2:
        return None

    max_total_tokens = cfg.get("max_total_tokens", 80)

    # Paper says upper bounds are adjusted to fit short utterances.
    # Here we cap total inserted tokens by both utterance length and
    # maximum total token length for TTS stability.
    max_insertions = max_total_tokens - len(tokens)
    max_insertions = min(max_insertions, len(tokens))
    max_insertions = max(0, max_insertions)

    if max_insertions <= 0:
        return None

    event_type = choose_event_type(cfg, rng)

    if event_type == "word_repetition":
        new_tokens, events = add_word_repetition(
            tokens,
            cfg.get("word_repetition", {}),
            rng,
            max_insertions,
        )
    elif event_type == "phrase_repetition":
        new_tokens, events = add_phrase_repetition(
            tokens,
            cfg.get("phrase_repetition", {}),
            rng,
            max_insertions,
        )
    elif event_type == "interjection":
        new_tokens, events = add_interjections(
            tokens,
            cfg.get("interjection", {}),
            rng,
            max_insertions,
        )
    else:
        return None

    if not events:
        return None

    stuttered_text = " ".join(new_tokens)

    if stuttered_text == clean_text:
        return None

    augmented = dict(record)
    augmented["original_utterance_id"] = record["utterance_id"]
    augmented["stuttered_text"] = stuttered_text
    augmented["is_disfluent"] = True
    augmented["disfluency_events"] = events
    augmented["primary_disfluency_type"] = event_type

    return augmented


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment clean transcripts with stutter-like disfluencies."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to augmentation YAML config.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    input_manifest = cfg["input_manifest"]
    output_manifest = cfg["output_manifest"]

    seed = cfg.get("seed", 42)
    max_augmented_samples = cfg.get("max_augmented_samples", None)

    rng = random.Random(seed)

    clean_records = list(read_jsonl(input_manifest))

    if not clean_records:
        raise ValueError(f"No clean records found in {input_manifest}")

    target_count = max_augmented_samples or len(clean_records)

    augmented_records = []
    attempts = 0
    max_attempts = target_count * 20 + 100

    while len(augmented_records) < target_count and attempts < max_attempts:
        attempts += 1

        record = rng.choice(clean_records)
        augmented = augment_record(record, cfg, rng)

        if augmented is None:
            continue

        augmented["augmented_id"] = (
            f"{augmented['original_utterance_id']}-"
            f"aug-{len(augmented_records):05d}"
        )

        augmented_records.append(augmented)

    write_jsonl(output_manifest, augmented_records)

    print(f"Input clean records: {len(clean_records)}")
    print(f"Generated augmented records: {len(augmented_records)}")
    print(f"Output: {output_manifest}")


if __name__ == "__main__":
    main()
