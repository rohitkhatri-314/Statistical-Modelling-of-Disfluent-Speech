import argparse
import pandas as pd
import os
import glob
try:
    import pylangacq
except ImportError:
    print("Please install pylangacq: pip install pylangacq")
    exit(1)

def extract_metadata(data_dir, output_csv):
    """
    Extracts utterance metadata from FluencyBank .cha files and saves to CSV.
    """
    records = []
    
    # Find all CHAT files
    cha_files = glob.glob(os.path.join(data_dir, '**', '*.cha'), recursive=True)
    print(f"Found {len(cha_files)} .cha files in {data_dir}")
    
    for cha_file in cha_files:
        try:
            chat = pylangacq.read_chat(cha_file)
            base_name = os.path.splitext(os.path.basename(cha_file))[0]
            
            # Get corresponding audio file path (assuming same name, .wav extension)
            audio_dir = os.path.dirname(cha_file)
            audio_path = os.path.join(audio_dir, f"{base_name}.wav")
            
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found for {cha_file}. Using path anyway: {audio_path}")
            
            for i, utterance in enumerate(chat.utterances()):
                # Usually we only care about the person who stutters, often marked as *PAR or *INV
                speaker = utterance.participant
                
                # Raw text with disfluencies (stuttered_text)
                stuttered_text = utterance.tiers[speaker]
                
                # pylangacq provides words() which often strips out the raw CHAT symbols, 
                # giving us a cleaner version (clean_text)
                clean_text = " ".join(utterance.words())
                
                records.append({
                    "utterance_id": f"{base_name}_{i:04d}",
                    "speaker_id": speaker,
                    "audio_path": audio_path,
                    "clean_text": clean_text,
                    "stuttered_text": stuttered_text
                })
        except Exception as e:
            print(f"Error parsing {cha_file}: {e}")
            
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(records)} utterances to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract FluencyBank metadata to CSV")
    parser.add_argument("--data_dir", required=True, help="Path to the downloaded FluencyBank data folder (containing .cha and .wav files)")
    parser.add_argument("--output_csv", default="fluencybank_meta.csv", help="Output CSV path")
    
    args = parser.parse_args()
    extract_metadata(args.data_dir, args.output_csv)
