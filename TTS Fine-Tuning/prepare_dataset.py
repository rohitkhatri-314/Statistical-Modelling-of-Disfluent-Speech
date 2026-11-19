import argparse
import csv
import json
import random
import re
import shutil
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf

# If two recordings belong to the SAME participant, map them here.
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
        description="XTTS Dataset Pipeline with Selective Denoising for Stuttered Speech."
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("Romana/TimeStamped/metadata.csv"),
        help="Path to Romana TimeStamped metadata.csv."
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
        help="Directory containing CHAT .cha transcript files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_disfluent"),
        help="Output directory for XTTS dataset."
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        default=True,
        help="Enable selective audio noise-profiling and gentle speech enhancement."
    )
    parser.add_argument(
        "--snr-threshold",
        type=float,
        default=16.0,
        help="SNR threshold in dB below which audio is treated as noisy (default: 16.0 dB)."
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=MIN_CLIP_DURATION,
        help="Minimum clip duration in seconds."
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=MAX_CLIP_DURATION,
        help="Maximum clip duration in seconds."
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
        help="Proportion of data allocated to evaluation set."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate dataset processing without writing audio files."
    )
    return parser.parse_args()


def calculate_snr(wav_path, sample_seconds=45):
    """
    Computes mathematical WADA-SNR from the audio signal to estimate noise floor.
    """
    try:
        data, sr = sf.read(str(wav_path), stop=int(16000 * sample_seconds))
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        data = data - np.mean(data)
        mean_abs = np.mean(np.abs(data))
        if mean_abs <= 1e-9:
            return 0.0
        gamma = np.mean(data ** 2) / (mean_abs ** 2)
        if gamma <= 1.57:
            return 0.0
        snr = 10 * np.log10(max(1e-4, (gamma - 1.57) / (np.pi / 2)))
        return round(float(np.clip(snr * 3.5 + 10.0, -5.0, 45.0)), 2)
    except Exception:
        return 15.0  # default to moderate if unreadable


def find_files(folder, extension):
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
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = text.replace("<", " ").replace(">", " ").replace("_", " ").replace("|", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if text:
        text = text[0].upper() + text[1:]
        if not text.endswith((".", "!", "?")):
            text = text + "."
    return text


def get_audio_duration(wav_path):
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
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
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

    wav_files = find_files(wav_dir, ".wav")
    cha_files = find_files(cha_dir, ".cha") if cha_dir.is_dir() else {}

    print(f"Loaded master WAV files: {len(wav_files)}")
    print(f"Loaded CHA files (for demographics): {len(cha_files)}")
    print(f"Reading metadata: {metadata_csv_path}")

    # Step 1: Profile noise / SNR across all master recordings
    noise_profiles = {}
    print("\n[Step 1] Analyzing acoustic noise profiles for master recordings...")
    for stem, path in sorted(wav_files.items()):
        snr = calculate_snr(path)
        needs_denoise = snr < args.snr_threshold
        noise_profiles[stem] = {
            "snr_db": snr,
            "needs_denoise": needs_denoise,
            "category": "NOISY (Will Denoise)" if needs_denoise else "CLEAN (Pass Through)"
        }
        print(f"  • {path.name:<12} | SNR: {snr:5.2f} dB | {noise_profiles[stem]['category']}")

    if not args.dry_run:
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise RuntimeError("Install FFmpeg/FFprobe and ensure they are added to PATH.")
        (output_dir / "wavs").mkdir(parents=True, exist_ok=True)

    wav_durations = {}
    trainable = []
    audit = []
    skipped = []
    speaker_metadata = {}

    with open(metadata_csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        segments = list(reader)

    print(f"\n[Step 2] Processing {len(segments)} segments into XTTS dataset...")

    for row_idx, row in enumerate(segments, start=1):
        segid = row.get("segid", "").strip()
        raw_start = row.get("segstart", "").strip()
        raw_end = row.get("segend", "").strip()
        verbatim = row.get("verbatim", "").strip()
        intended = row.get("intended", "").strip()

        parts = segid.split("_", 1)
        recording = parts[0]
        utterance_id = parts[1] if len(parts) > 1 else f"{row_idx:04d}"

        speaker_name = SPEAKER_OVERRIDES.get(recording, f"{recording}_PAR")

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
                    "cha": cha_files.get(cha_key, Path(f"{recording}.cha")).name,
                    "snr_db": noise_profiles.get(recording.casefold(), {}).get("snr_db", 0.0)
                })

        text = clean_verbatim_text(verbatim)
        is_disfluent = (verbatim.casefold() != intended.casefold())
        reason = ""

        wav_key = recording.casefold()
        if wav_key not in wav_files:
            reason = "missing_source_wav"
            wav_path = None
        else:
            wav_path = wav_files[wav_key]

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

        # Selective Audio Slicing & Enhancement
        if not reason and not args.dry_run:
            output_wav = output_dir / relative_path

            # Determine audio filter chain
            profile = noise_profiles.get(wav_key, {})
            needs_denoising = args.denoise and profile.get("needs_denoise", False)
            
            # Base audio filter: low-cut sub-75Hz rumble and EBU R128 loudness normalization
            if needs_denoising:
                # Gentle FFT denoiser with low reduction (nr=10dB, noise floor -45dB)
                # carefully calibrated to avoid stripping stuttering prolongations
                audio_filter = "highpass=f=75,afftdn=nr=10:nf=-45:tn=1,loudnorm=I=-20:TP=-1.5:LRA=11"
                proc_type = "denoised_and_normalized"
            else:
                # Clean audio: highpass rumble cut + loudness normalization only (zero spectral artifact)
                audio_filter = "highpass=f=75,loudnorm=I=-20:TP=-1.5:LRA=11"
                proc_type = "normalized_only"

            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-ss", f"{start:.3f}",
                "-t", f"{duration:.3f}",
                "-i", str(wav_path),
                "-af", audio_filter,
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
                proc_type = "failed"
        else:
            proc_type = "none"

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
            "duration_sec": f"{duration:.3f}",
            "snr_db": noise_profiles.get(wav_key, {}).get("snr_db", ""),
            "processing": proc_type,
            "is_disfluent": is_disfluent,
            "verbatim_text": verbatim,
            "cleaned_text": text,
            "status": status,
            "reason": reason
        })

    if args.dry_run:
        trainable_count = len([a for a in audit if a["status"] != "excluded"])
        print("\n--- Dry Run Summary ---")
        print(f"Total candidate segments: {len(segments)}")
        print(f"Valid trainable clips:    {trainable_count}")
        print(f"Skipped segments:         {len(skipped)}")
        return

    # Train / Eval Split (85% / 15%)
    random.Random(RANDOM_SEED).shuffle(trainable)
    eval_count = 0
    if len(trainable) >= 2 and args.eval_ratio > 0:
        eval_count = max(1, int(len(trainable) * args.eval_ratio))
        eval_count = min(eval_count, len(trainable) - 1)

    eval_rows = trainable[:eval_count]
    train_rows = trainable[eval_count:]

    write_xtts_metadata(output_dir / "metadata_train.csv", train_rows)
    write_xtts_metadata(output_dir / "metadata_eval.csv", eval_rows)
    write_csv(output_dir / "metadata_audit.csv", audit)
    write_csv(output_dir / "skipped_utterances.csv", skipped)

    with open(output_dir / "speaker_metadata.json", "w", encoding="utf-8") as f:
        json.dump(speaker_metadata, f, indent=4, ensure_ascii=False)

    print("\nDataset preparation with selective denoising complete!")
    print(f"Total usable clips:  {len(trainable)}")
    print(f"Training samples:    {len(train_rows)}")
    print(f"Evaluation samples:  {len(eval_rows)}")
    print(f"Skipped samples:     {len(skipped)}")
    print(f"Output directory:    {output_dir}")


if __name__ == "__main__":
    main()
