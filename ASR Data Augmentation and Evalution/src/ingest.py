import argparse
import urllib.request
import tarfile
import io
from pathlib import Path

from normalize import normalize_text
from utils import write_jsonl, ensure_dir

def main():
    parser = argparse.ArgumentParser(description="Ingest clean LibriSpeech transcripts without heavy audio downloads.")
    parser.add_argument("--url", default="http://www.openslr.org/resources/12/dev-clean.tar.gz", help="URL to a LibriSpeech tar.gz (e.g., dev-clean, 337MB).")
    parser.add_argument("--max_samples", type=int, default=200, help="Maximum number of clean utterances to keep.")
    parser.add_argument("--min_words", type=int, default=4)
    parser.add_argument("--max_words", type=int, default=30)
    parser.add_argument("--output", default="data/processed/clean_manifest.jsonl")

    args = parser.parse_args()

    print(f"Downloading LibriSpeech from {args.url} (This is only ~337 MB instead of 30 GB!)...")
    
    # We will open the URL as a stream and extract ONLY the text files (.txt) on the fly!
    req = urllib.request.Request(args.url, headers={'User-Agent': 'Mozilla/5.0'})
    
    records = []
    seen = 0

    with urllib.request.urlopen(req) as response:
        # Open the tarfile from the stream
        with tarfile.open(fileobj=response, mode="r|gz") as tar:
            for member in tar:
                # We only care about the transcript files, skip all the heavy .flac audio files
                if member.name.endswith(".txt") and "transcripts" not in member.name:
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    
                    content = f.read().decode("utf-8")
                    lines = content.strip().split("\n")
                    
                    for line in lines:
                        if not line.strip():
                            continue
                            
                        # LibriSpeech transcripts look like: 1272-128104-0000 TEXT GOES HERE
                        parts = line.split(" ", 1)
                        if len(parts) < 2:
                            continue
                            
                        utterance_id = parts[0].strip()
                        raw_text = parts[1].strip()
                        normalized_text = normalize_text(raw_text)
                        words = normalized_text.split()
                        
                        if len(words) < args.min_words or len(words) > args.max_words:
                            continue
                            
                        speaker_id = utterance_id.split("-")[0]
                        
                        records.append({
                            "utterance_id": utterance_id,
                            "raw_text": raw_text,
                            "clean_text": normalized_text,
                            "source_dataset": "LibriSpeech-Direct",
                            "source_config": "dev-clean",
                            "source_split": "validation",
                            "speaker_id": speaker_id,
                        })
                        seen += 1
                        
                        if len(records) >= args.max_samples:
                            break
                            
                if len(records) >= args.max_samples:
                    break

    write_jsonl(args.output, records)

    print(f"Processed valid rows: {seen}")
    print(f"Saved clean utterances: {len(records)}")
    print(f"Output: {args.output}")

if __name__ == "__main__":
    main()