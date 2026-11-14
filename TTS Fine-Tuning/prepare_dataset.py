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

# XTTS fine-tuning audio specifications
SAMPLE_RATE = 22050
MIN_CLIP_DURATION = 0.5
MAX_CLIP_DURATION = 11.6
EVAL_RATIO = 0.15
RANDOM_SEED = 42

UNINTELLIGIBLE_RE = re.compile(r"\b(?:xxx|yyy|www)\b", re.I)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare XTTS dataset with disfluent verbatim transcripts from FluencyBank Romana metadata."
    )

    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("Romana/TimeStamped/metadata.csv"),
        help="Path to Romana TimeStamped metadata.csv containing verbatim transcripts and timestamps."
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        default=Path("WAVs"),
        help="Directory containing full raw interview WAV recordings."
    )
    parser.add_argument(
        "--cha-dir",
        type=Path,
        default=Path("CHAs"),
        help="Directory containing CHAT .cha transcript files (used for speaker demographics)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_disfluent"),
        help="Output directory for XTTS dataset."
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=MIN_CLIP_DURATION,
        help="Minimum clip duration in seconds (XTTS minimum accepted is 0.5s)."
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=MAX_CLIP_DURATION,
        help="Maximum clip duration in seconds (XTTS training token limit is ~11.6s)."
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help="Optional padding in seconds added to both ends of each segment."
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=EVAL_RATIO,
        help="Proportion of data allocated to evaluation set (default: 0.15)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate dataset processing without cutting audio files."
    )

    return parser.parse_args()


def find_files(folder, extension):
    """Find files recursively and match them using lowercase filename stems."""
    files = {}
    if not folder.is_dir():
        return files

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() == extension:
            key = path.stem.casefold()
            if key in files:
                raise ValueError(f"Duplicate filename found: {path.stem}")
            files[key] = path

    return files


def get_speaker_info(cha_text):
    """Extract available PAR information from the @ID header in a .cha file."""
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


def clean_verbatim_text(text):
    """
    Clean verbatim transcript while strictly preserving all stuttered disfluencies
    (repetitions, partial words, filled pauses, false starts).
    """
    if not text:
        return ""

    # Normalize multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Normalize stray CHAT annotations if any leak through
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = text.replace("<", " ").replace(">", " ").replace("_", " ").replace("|", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Capitalize the first letter of the sentence for standard TTS reading
    if text:
        text = text[0].upper() + text[1:]
        # Ensure ending punctuation if not present
        if not text.endswith((".", "!", "?")):
            text = text + "."

    return text


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
    XTTS training metadata format:
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

    metadata_csv_path = args.metadata_csv.resolve()
    wav_dir = args.wav_dir.resolve()
    cha_dir = args.cha_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not metadata_csv_path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found at: {metadata_csv_path}")
    if not wav_dir.is_dir():
        raise FileNotFoundError(f"WAV directory not found at: {wav_dir}")

    if not 0 <= args.eval_ratio < 1:
        raise ValueError("Evaluation ratio must be between 0 and 1.")

    wav_files = find_files(wav_dir, ".wav")
    cha_files = find_files(cha_dir, ".cha") if cha_dir.is_dir() else {}

    print(f"Loaded master WAV files: {len(wav_files)}")
    print(f"Loaded CHA files (for demographics): {len(cha_files)}")
    print(f"Reading disfluent metadata: {metadata_csv_path}")

    if not args.dry_run:
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise RuntimeError("Install FFmpeg/FFprobe and ensure they are added to PATH.")
        (output_dir / "wavs").mkdir(parents=True, exist_ok=True)

    # Cache WAV durations so we don't call ffprobe repeatedly
    wav_durations = {}

    trainable = []
    audit = []
    skipped = []
    speaker_metadata = {}

    with open(metadata_csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        segments = list(reader)

    print(f"Total candidate segments in metadata: {len(segments)}")

    disfluent_count = 0

    for row_idx, row in enumerate(segments, start=1):
        segid = row.get("segid", "").strip()
        raw_start = row.get("segstart", "").strip()
        raw_end = row.get("segend", "").strip()
        verbatim = row.get("verbatim", "").strip()
        intended = row.get("intended", "").strip()

        # segid format: e.g. 24fb_000 -> speaker: 24fb, utterance: 000
        parts = segid.split("_", 1)
        recording = parts[0]
        utterance_id = parts[1] if len(parts) > 1 else f"{row_idx:04d}"

        speaker_name = SPEAKER_OVERRIDES.get(
            recording,
            f"{recording}_PAR"
        )

        # Track speaker metadata from CHA header if available
        if speaker_name not in speaker_metadata:
            speaker_info = {"speaker_code": "PAR"}
            cha_key = recording.casefold()
            if cha_key in cha_files:
                cha_text = cha_files[cha_key].read_text(encoding="utf-8-sig", errors="replace")
                speaker_info = get_speaker_info(cha_text)

            speaker_metadata[speaker_name] = {
                **speaker_info,
                "speaker_name": speaker_name,
                "recordings": []
            }
            if recording.casefold() in wav_files:
                speaker_metadata[speaker_name]["recordings"].append({
                    "wav": wav_files[recording.casefold()].name,
                    "cha": cha_files.get(cha_key, Path(f"{recording}.cha")).name
                })

        text = clean_verbatim_text(verbatim)
        is_disfluent = (verbatim.casefold() != intended.casefold())
        if is_disfluent:
            disfluent_count += 1

        reason = ""

        # Validation: Check raw master WAV existence
        wav_key = recording.casefold()
        if wav_key not in wav_files:
            reason = "missing_source_wav"
            wav_path = None
        else:
            wav_path = wav_files[wav_key]

        # Validation: Parse timestamps
        try:
            start_sec = float(raw_start)
            end_sec = float(raw_end)
        except (ValueError, TypeError):
            reason = "invalid_timestamp"
            start_sec = end_sec = 0.0

        if not reason:
            if end_sec <= start_sec:
                reason = "end_time_not_greater_than_start"
            elif not text:
                reason = "empty_transcript"
            elif UNINTELLIGIBLE_RE.search(text):
                reason = "unintelligible_transcript"

        start = max(0.0, start_sec - args.padding)
        end = end_sec + args.padding
        duration = end - start

        if not reason and wav_path:
            if wav_key not in wav_durations and not args.dry_run:
                try:
                    wav_durations[wav_key] = get_audio_duration(wav_path)
                except Exception:
                    wav_durations[wav_key] = None

            audio_total_dur = wav_durations.get(wav_key)
            if audio_total_dur and end > audio_total_dur + 0.05:
                reason = "timestamp_exceeds_audio_duration"
            elif duration < args.min_duration:
                reason = "clip_too_short"
            elif duration > args.max_duration:
                reason = "clip_too_long"

        clip_name = f"{recording}_PAR_{utterance_id}_{start:.3f}_{end:.3f}.wav"
        relative_path = f"wavs/{clip_name}"
        status = "excluded" if reason else "pending"

        # Slice audio clip
        if not reason and not args.dry_run:
            output_wav = output_dir / relative_path

            # FFmpeg: slice and convert to mono 22050Hz 16-bit PCM WAV (XTTS standard)
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-ss", f"{start:.3f}",
                "-t", f"{duration:.3f}",
                "-i", str(wav_path),
                "-map", "0:a:0",
                "-vn",
                "-ac", "1",
                "-ar", str(SAMPLE_RATE),
                "-c:a", "pcm_s16le",
                str(output_wav)
            ]

            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                if not output_wav.exists() or output_wav.stat().st_size == 0:
                    raise RuntimeError("Output audio file is empty or missing.")
                status = "created"
            except (subprocess.CalledProcessError, RuntimeError) as error:
                reason = f"ffmpeg_error: {error}"
                status = "excluded"

        if reason:
            skipped.append({
                "recording": recording,
                "utterance_id": utterance_id,
                "start_sec": f"{start_sec:.3f}",
                "end_sec": f"{end_sec:.3f}",
                "duration_sec": f"{duration:.3f}",
                "reason": reason,
                "verbatim": verbatim,
                "intended": intended,
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
            "segid": segid,
            "speaker_name": speaker_name,
            "source_wav": wav_path.name if wav_path else "",
            "utterance_id": utterance_id,
            "start_sec": f"{start:.3f}",
            "end_sec": f"{end:.3f}",
            "duration_sec": f"{duration:.3f}",
            "is_disfluent": is_disfluent,
            "verbatim_text": verbatim,
            "intended_text": intended,
            "cleaned_text": text,
            "status": status,
            "reason": reason
        })

    print(f"Segments with disfluencies: {disfluent_count} / {len(segments)}")

    if args.dry_run:
        trainable_count = len([a for a in audit if a["status"] != "excluded"])
        print("\n--- Dry Run Summary ---")
        print(f"Total candidate segments: {len(segments)}")
        print(f"Valid trainable clips: {trainable_count}")
        print(f"Skipped segments: {len(skipped)}")
        return

    # Reproducible train/eval split (85% train, 15% eval)
    random.Random(RANDOM_SEED).shuffle(trainable)

    eval_count = 0
    if len(trainable) >= 2 and args.eval_ratio > 0:
        eval_count = max(1, int(len(trainable) * args.eval_ratio))
        eval_count = min(eval_count, len(trainable) - 1)

    eval_rows = trainable[:eval_count]
    train_rows = trainable[eval_count:]

    # Write XTTS metadata files
    write_xtts_metadata(output_dir / "metadata_train.csv", train_rows)
    write_xtts_metadata(output_dir / "metadata_eval.csv", eval_rows)

    # Write audit, skipped logs, and speaker metadata
    write_csv(output_dir / "metadata_audit.csv", audit)
    write_csv(output_dir / "skipped_utterances.csv", skipped)

    with open(output_dir / "speaker_metadata.json", "w", encoding="utf-8") as f:
        json.dump(speaker_metadata, f, indent=4, ensure_ascii=False)

    print("\nDataset preparation complete!")
    print(f"Total usable clips: {len(trainable)}")
    print(f"Training samples: {len(train_rows)}")
    print(f"Evaluation samples: {len(eval_rows)}")
    print(f"Skipped samples: {len(skipped)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
