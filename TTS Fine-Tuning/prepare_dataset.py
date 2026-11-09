
import argparse
import csv
import json
import random
import re
import shutil
import subprocess
from pathlib import Path

# If two recordings belong to the SAME participant, map them here.
# Example:
# SPEAKER_OVERRIDES = {"24fa": "participant_24", "24fb": "participant_24"}
SPEAKER_OVERRIDES = {}

SAMPLE_RATE = 22050
MIN_CLIP_DURATION = 0.5
EVAL_RATIO = 0.15
RANDOM_SEED = 42

# CHAT timestamp format: 0x15 start_end 0x15
TIMESTAMP_RE = re.compile(r"\x15\s*(\d+)_(\d+)\s*\x15")
TIER_RE = re.compile(r"(?m)^\*([^\s:]+):\s*")
UNINTELLIGIBLE_RE = re.compile(r"\b(?:xxx|yyy|www)\b", re.I)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--wav-dir", required=True, type=Path)
    parser.add_argument("--cha-dir", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("XTTS_dataset")
    )
    parser.add_argument("--padding", type=float, default=0.0)
    parser.add_argument("--eval-ratio", type=float, default=EVAL_RATIO)
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def find_files(folder, extension):
    """Find files recursively and match them using filename stems."""

    files = {}

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() == extension:
            key = path.stem.casefold()

            if key in files:
                raise ValueError(
                    f"Duplicate filename found: {path.stem}"
                )

            files[key] = path

    return files


def get_speaker_info(cha_text):
    """Extract available PAR information from the @ID header."""

    for line in cha_text.splitlines():

        if line.startswith("@ID:"):
            fields = line.split(":", 1)[1].strip().split("|")

            if len(fields) >= 3 and fields[2].strip().upper() == "PAR":

                fields += [""] * (10 - len(fields))

                return {
                    "language": fields[0].strip(),
                    "corpus": fields[1].strip(),
                    "speaker_code": fields[2].strip(),
                    "age": fields[3].strip(),
                    "sex": fields[4].strip(),
                    "group": fields[5].strip(),
                    "ses": fields[6].strip(),
                    "role": fields[7].strip(),
                    "education": fields[8].strip(),
                    "other": fields[9].strip()
                }

    return {"speaker_code": "PAR"}


def clean_transcript(text):
    """Remove CHAT markup while retaining spoken words."""

    # Remove timestamps
    text = TIMESTAMP_RE.sub(" ", text)

    # Convert fillers: &-um -> um, &-uh -> uh
    text = re.sub(r"&[-+](?=[A-Za-z])", "", text)

    # Remove non-speech event tags such as &=laugh
    text = re.sub(r"&=[A-Za-z0-9_-]+", " ", text)

    # Keep words inside angle brackets, remove only brackets
    text = text.replace("<", " ").replace(">", " ")

    # Remove CHAT annotations such as [//], [/], [=! ...]
    text = re.sub(r"\[[^\]]*\]", " ", text)

    # Remove CHAT suffix tags such as word@u
    text = re.sub(r"(?<=\w)@[A-Za-z][A-Za-z0-9_-]*", "", text)

    # Replace underscores and pipe characters
    text = text.replace("_", " ").replace("|", " ")

    # Normalize spaces and punctuation
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_par_utterances(cha_text):
    """
    Extract each PAR utterance and its global timestamps.

    Returns:
        raw_text, start_ms, end_ms
    """

    tiers = list(TIER_RE.finditer(cha_text))

    for i, match in enumerate(tiers):

        speaker = match.group(1).upper()

        if speaker != "PAR":
            continue

        # Everything until the next speaker tier belongs to this utterance
        block_end = (
            tiers[i + 1].start()
            if i + 1 < len(tiers)
            else len(cha_text)
        )

        block = cha_text[match.end():block_end]

        spoken_lines = []

        for line in block.splitlines():

            line = line.strip()

            # Ignore dependent tiers and header lines
            if not line or line.startswith(("%", "@")):
                continue

            spoken_lines.append(line)

        raw_text = " ".join(spoken_lines)

        timestamps = [
            (int(start), int(end))
            for start, end in TIMESTAMP_RE.findall(block)
        ]

        if timestamps:
            start_ms = min(start for start, end in timestamps)
            end_ms = max(end for start, end in timestamps)
        else:
            start_ms = end_ms = None

        yield raw_text, start_ms, end_ms


def get_audio_duration(wav_path):
    """Read the original WAV duration using FFprobe."""

    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(wav_path)
        ],
        capture_output=True,
        text=True,
        check=True
    )

    return float(result.stdout.strip())


def write_xtts_metadata(path, rows):
    """
    XTTS demo metadata format:
    audio_file|text|speaker_name
    """

    with open(path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["audio_file", "text", "speaker_name"],
            delimiter="|"
        )

        writer.writeheader()
        writer.writerows(rows)


def write_csv(path, rows):

    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8-sig") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


def main():

    args = parse_args()

    wav_dir = args.wav_dir.resolve()
    cha_dir = args.cha_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not wav_dir.is_dir() or not cha_dir.is_dir():
        raise FileNotFoundError("Check your WAV and CHA folder paths.")

    if args.padding < 0:
        raise ValueError("Padding cannot be negative.")

    if not 0 <= args.eval_ratio < 1:
        raise ValueError("Evaluation ratio must be between 0 and 1.")

    wav_files = find_files(wav_dir, ".wav")
    cha_files = find_files(cha_dir, ".cha")

    matched = sorted(set(wav_files) & set(cha_files))

    print("WAV files:", len(wav_files))
    print("CHA files:", len(cha_files))
    print("Matched pairs:", len(matched))

    if not matched:
        raise RuntimeError("No matching WAV and CHA filenames found.")

    missing_cha = sorted(set(wav_files) - set(cha_files))
    missing_wav = sorted(set(cha_files) - set(wav_files))

    if missing_cha:
        print("WAVs without CHA:", missing_cha)

    if missing_wav:
        print("CHAs without WAV:", missing_wav)

    if not args.dry_run:

        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise RuntimeError("Install FFmpeg and add it to PATH.")

        (output_dir / "wavs").mkdir(parents=True, exist_ok=True)

    trainable = []
    audit = []
    skipped = []
    speaker_metadata = {}

    for key in matched:

        wav_path = wav_files[key]
        cha_path = cha_files[key]

        recording = wav_path.stem

        speaker_name = SPEAKER_OVERRIDES.get(
            recording,
            f"{recording}_PAR"
        )

        cha_text = cha_path.read_text(
            encoding="utf-8-sig",
            errors="replace"
        )

        speaker_info = get_speaker_info(cha_text)

        if speaker_name not in speaker_metadata:

            speaker_metadata[speaker_name] = {
                **speaker_info,
                "speaker_name": speaker_name,
                "recordings": []
            }

        speaker_metadata[speaker_name]["recordings"].append({
            "wav": wav_path.name,
            "cha": cha_path.name
        })

        audio_duration = (
            None if args.dry_run
            else get_audio_duration(wav_path)
        )

        utterance_number = 0

        for raw_text, start_ms, end_ms in extract_par_utterances(cha_text):

            utterance_number += 1

            text = clean_transcript(raw_text)
            reason = ""

            if start_ms is None or end_ms is None:
                reason = "missing_timestamp"

            elif end_ms <= start_ms:
                reason = "invalid_timestamp"

            elif not text:
                reason = "empty_transcript"

            elif UNINTELLIGIBLE_RE.search(text):
                reason = "unintelligible_transcript_requires_review"

            if start_ms is not None and end_ms is not None and end_ms > start_ms:

                start = max(0, start_ms / 1000 - args.padding)
                end = end_ms / 1000 + args.padding
                duration = end - start

                if audio_duration is not None and end > audio_duration + 0.05:
                    reason = "timestamp_exceeds_audio_duration"

                if duration < MIN_CLIP_DURATION and not reason:
                    reason = "clip_too_short"

            else:
                start = end = duration = 0.0

            clip_name = (
                f"{recording}_PAR_{utterance_number:04d}_"
                f"{start:.3f}_{end:.3f}.wav"
            )

            relative_path = f"wavs/{clip_name}"

            status = "excluded" if reason else "pending"

            # Clip audio
            if not reason and not args.dry_run:

                output_wav = output_dir / relative_path

                command = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-y",
                    "-i", str(wav_path),
                    "-ss", f"{start:.3f}",
                    "-t", f"{duration:.3f}",
                    "-map", "0:a:0",
                    "-vn",
                    "-ac", "1",
                    "-ar", str(SAMPLE_RATE),
                    "-c:a", "pcm_s16le",
                    str(output_wav)
                ]

                try:

                    subprocess.run(
                        command,
                        check=True,
                        capture_output=True,
                        text=True
                    )

                    if not output_wav.exists() or output_wav.stat().st_size == 0:
                        raise RuntimeError("Empty output audio file.")

                    status = "created"

                except (subprocess.CalledProcessError, RuntimeError) as error:

                    reason = f"ffmpeg_error: {error}"
                    status = "excluded"

            if reason:

                skipped.append({
                    "recording": recording,
                    "utterance_number": utterance_number,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "reason": reason,
                    "raw_text": raw_text,
                    "cleaned_text": text
                })

            elif not args.dry_run:

                trainable.append({
                    "audio_file": relative_path,
                    "text": text,
                    "speaker_name": speaker_name
                })

            audit.append({
                "clip": clip_name if not reason else "",
                "speaker_name": speaker_name,
                "source_wav": wav_path.name,
                "source_cha": cha_path.name,
                "utterance_number": utterance_number,
                "start_ms": start_ms if start_ms is not None else "",
                "end_ms": end_ms if end_ms is not None else "",
                "start_sec": f"{start:.3f}" if start_ms is not None else "",
                "end_sec": f"{end:.3f}" if end_ms is not None else "",
                "duration_sec": f"{duration:.3f}" if start_ms is not None else "",
                "raw_chat_text": raw_text,
                "cleaned_text": text,
                "status": status,
                "reason": reason
            })

        print(f"Processed {recording}: {utterance_number} PAR utterances")

    if args.dry_run:

        print("\nDry run complete. No clips or metadata were created.")
        return

    # Reproducible train/evaluation split
    random.Random(RANDOM_SEED).shuffle(trainable)

    eval_count = 0

    if len(trainable) >= 2 and args.eval_ratio > 0:

        eval_count = max(
            1,
            int(len(trainable) * args.eval_ratio)
        )

        eval_count = min(eval_count, len(trainable) - 1)

    eval_rows = trainable[:eval_count]
    train_rows = trainable[eval_count:]

    # XTTS metadata files
    write_xtts_metadata(
        output_dir / "metadata_train.csv",
        train_rows
    )

    write_xtts_metadata(
        output_dir / "metadata_eval.csv",
        eval_rows
    )

    # Additional audit and speaker information
    write_csv(output_dir / "metadata_audit.csv", audit)

    write_csv(output_dir / "skipped_utterances.csv", skipped)

    with open(
        output_dir / "speaker_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            speaker_metadata,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\nDataset preparation complete!")
    print("Usable clips:", len(trainable))
    print("Training samples:", len(train_rows))
    print("Evaluation samples:", len(eval_rows))
    print("Skipped samples:", len(skipped))
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
