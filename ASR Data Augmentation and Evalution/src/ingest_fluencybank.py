import argparse
import pandas as pd
from pathlib import Path
from normalize import normalize_text
from utils import write_jsonl, ensure_dir

def main():
    parser = argparse.ArgumentParser(description="Ingest FluencyBank transcripts/audio paths for evaluation.")
    parser.add_argument("--input_csv", required=True, help="Path to CSV containing FluencyBank metadata (utterance_id, speaker_id, audio_path, clean_text, stuttered_text).")
    parser.add_argument("--output_manifest", required=True, help="Path to output JSONL manifest.")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    
    records = []
    for _, row in df.iterrows():
        record_id = str(row.get("utterance_id", f"fb-{len(records):05d}"))
        clean_text = str(row.get("clean_text", ""))
        stuttered_text = str(row.get("stuttered_text", ""))
        
        # We need normalized text for evaluation
        norm_clean = normalize_text(clean_text)
        norm_stuttered = normalize_text(stuttered_text)
        
        record = {
            "record_id": record_id,
            "speaker_id": str(row.get("speaker_id", "unknown")),
            "stuttered_audio_path": str(row.get("audio_path", "")),
            "clean_text": norm_clean,
            "stuttered_text": norm_stuttered,
            "is_disfluent": True,
            "source_dataset": "FluencyBank"
        }
        records.append(record)
        
    write_jsonl(args.output_manifest, records)
    print(f"Ingested {len(records)} FluencyBank records.")
    print(f"Output saved to {args.output_manifest}")

if __name__ == "__main__":
    main()
