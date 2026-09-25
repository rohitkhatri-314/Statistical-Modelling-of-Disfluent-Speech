import argparse
import hashlib
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import soundfile as sf
import torch
from datasets import load_dataset

from utils import ensure_dir, load_yaml, read_jsonl, write_jsonl

def safe_filename_part(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", str(text))

def stable_mod(text: str, mod: int) -> int:
    return int(hashlib.md5(str(text).encode("utf-8")).hexdigest(), 16) % mod

def get_device(device_setting: str) -> torch.device:
    if device_setting == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_setting == "cpu":
        return torch.device("cpu")
    if device_setting == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_setting)

# -----------------------------------------------------------------------------
# TTS Providers
# -----------------------------------------------------------------------------

class TTSProvider(ABC):
    @abstractmethod
    def synthesize_to_file(self, text: str, output_path: Path, speaker_id: str = None) -> float:
        """Synthesize text to audio and save to output_path. Returns the duration in seconds."""
        pass


class SpeechT5Provider(TTSProvider):
    def __init__(self, cfg: Dict):
        from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

        self.device = get_device(cfg.get("device", "auto"))
        print(f"Loading SpeechT5 models on {self.device}...")
        self.processor = SpeechT5Processor.from_pretrained(cfg.get("tts_model", "microsoft/speecht5_tts"))
        self.model = SpeechT5ForTextToSpeech.from_pretrained(cfg.get("tts_model", "microsoft/speecht5_tts")).to(self.device)
        self.model.eval()
        self.vocoder = SpeechT5HifiGan.from_pretrained(cfg.get("vocoder", "microsoft/speecht5_hifigan")).to(self.device)
        self.vocoder.eval()
        
        self.sample_rate = cfg.get("sample_rate", 16000)
        print("Loading speaker embeddings...")
        self.speakers = self._load_speaker_embeddings(cfg)
        self.speaker_map = {speaker_id: embedding for speaker_id, embedding in self.speakers}
        
        print(f"Loaded SpeechT5 with {len(self.speakers)} speakers.")

    def _load_speaker_embeddings(self, cfg: Dict) -> List[Tuple[str, torch.Tensor]]:
        dataset_name = cfg.get("speaker_embeddings_dataset", "Matthijs/cmu-arctic-xvectors")
        split = cfg.get("speaker_embeddings_split", "validation")
        num_speakers = cfg.get("num_speakers", 8)
        available = []

        try:
            ds = load_dataset(dataset_name, split=split)
            for i, row in enumerate(ds):
                if i >= num_speakers:
                    break
                if "xvector" not in row:
                    continue
                speaker_id = row.get("speaker_id") or row.get("speaker") or f"spk_{i}"
                available.append((str(speaker_id), torch.tensor(row["xvector"]).float()))
        except Exception:
            import urllib.request
            import zipfile
            import io
            import numpy as np
            print("Falling back to downloading zip file directly...")
            url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main/spkrec-xvect.zip"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                    speaker_files = {}
                    for name in z.namelist():
                        if name.endswith(".npy"):
                            spk = name.split("/")[-1].split("-")[0]
                            if spk not in speaker_files:
                                speaker_files[spk] = name
                    for spk, name in list(speaker_files.items())[:num_speakers]:
                        with z.open(name) as f:
                            arr = np.load(f)
                            available.append((spk, torch.tensor(arr).float()))
                            
        if not available:
            raise ValueError(f"Could not load speaker embeddings from {dataset_name}")
        return available

    def synthesize_to_file(self, text: str, output_path: Path, speaker_id: str = None) -> float:
        if speaker_id not in self.speaker_map:
            speaker_index = stable_mod(str(speaker_id), len(self.speakers))
            speaker_id = self.speakers[speaker_index][0]
            
        speaker_embedding = self.speaker_map[speaker_id]
        
        inputs = self.processor(text=text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        speaker_embedding = speaker_embedding.unsqueeze(0).to(self.device)

        with torch.no_grad():
            speech = self.model.generate_speech(input_ids, speaker_embedding, vocoder=self.vocoder)

        audio_np = speech.cpu().numpy()
        sf.write(output_path, audio_np, self.sample_rate)
        return len(audio_np) / self.sample_rate


class OpenAIProvider(TTSProvider):
    def __init__(self, cfg: Dict):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
            
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = cfg.get("openai_model", "tts-1-hd")
        # Available voices: alloy, echo, fable, onyx, nova, shimmer
        self.voices = cfg.get("openai_voices", ["nova", "alloy", "echo", "fable", "onyx", "shimmer"])
        print(f"Loaded OpenAI TTS using model: {self.model}")

    def synthesize_to_file(self, text: str, output_path: Path, speaker_id: str = None) -> float:
        # Deterministically map the speaker_id to one of the available voices
        voice_index = stable_mod(str(speaker_id), len(self.voices))
        selected_voice = self.voices[voice_index]
        
        # We save as flac since it's lossless and used elsewhere in the script
        response = self.client.audio.speech.create(
            model=self.model,
            voice=selected_voice,
            input=text,
            response_format="flac"
        )
        
        response.stream_to_file(output_path)
        return sf.info(output_path).duration


class ElevenLabsProvider(TTSProvider):
    def __init__(self, cfg: Dict):
        try:
            from elevenlabs.client import ElevenLabs
        except ImportError:
            raise ImportError("Please install elevenlabs: pip install elevenlabs")
            
        self.client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        self.model = cfg.get("elevenlabs_model", "eleven_multilingual_v2")
        
        # ElevenLabs requires Voice IDs. Here are some default premade voice IDs if none provided.
        # Rachel, Drew, Clyde, Mimi, Fin, Bella
        self.voices = cfg.get("elevenlabs_voices", [
            "21m00Tcm4TlvDq8ikWAM", 
            "29vD33N1CtxCmqQRPOHJ", 
            "2EiwWnXFnvU5JabPnv8n", 
            "zrHiDhphv9ZnVXBqCLjz", 
            "D38z5RcWu1voky8WS1ja", 
            "EXAVITQu4vr4xnSDxMaL"
        ])
        print(f"Loaded ElevenLabs TTS using model: {self.model}")

    def synthesize_to_file(self, text: str, output_path: Path, speaker_id: str = None) -> float:
        # Deterministically map the speaker_id to one of the available voices
        voice_index = stable_mod(str(speaker_id), len(self.voices))
        selected_voice = self.voices[voice_index]
        
        import time
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # In elevenlabs v1, we use text_to_speech.convert 
                # We request WAV format so it can be read natively by soundfile for duration info
                audio_stream = self.client.text_to_speech.convert(
                    text=text,
                    voice_id=selected_voice,
                    model_id=self.model,
                    output_format="wav_16000"
                )
                
                # We need to change the extension to .wav since elevenlabs does not natively return flac
                temp_wav_path = output_path.with_suffix(".wav")
                with open(temp_wav_path, "wb") as f:
                    for chunk in audio_stream:
                        if chunk:
                            f.write(chunk)
                            
                break # If we made it here, it succeeded! Break out of the retry loop.
                
            except Exception as e:
                if "10054" in str(e) or "forcibly closed" in str(e):
                    # Sometimes the connection pool gets poisoned on Windows, 
                    # so we re-initialize the client to get a fresh connection.
                    from elevenlabs.client import ElevenLabs
                    self.client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
                    
                if attempt < max_retries - 1:
                    print(f"Network connection dropped (attempt {attempt + 1}/{max_retries}). Retrying in 3 seconds...")
                    time.sleep(3)
                else:
                    raise e # Re-raise the error if we failed 3 times
                    
        # Now use soundfile to read the WAV and save as true FLAC
        data, samplerate = sf.read(temp_wav_path)
        sf.write(output_path, data, samplerate)
        
        # Clean up temp wav
        if temp_wav_path.exists():
            temp_wav_path.unlink()
            
        return len(data) / samplerate


class XTTSProvider(TTSProvider):
    def __init__(self, cfg: Dict):
        try:
            from TTS.api import TTS
        except ImportError:
            raise ImportError("Please install TTS: pip install TTS")
            
        self.device = get_device(cfg.get("device", "auto"))
        # You can use the free XTTS v2 model
        self.model_name = cfg.get("xtts_model", "tts_models/multilingual/multi-dataset/xtts_v2")
        print(f"Loading XTTS model {self.model_name} on {self.device}...")
        
        # Load the TTS model
        self.tts = TTS(self.model_name).to(self.device)

        # tts.speakers is a property on an nn.Module subclass — if it raises an
        # AttributeError internally, nn.Module.__getattr__ swallows it and re-raises
        # its own AttributeError. Access speaker names directly from the model instead.
        try:
            speaker_manager = self.tts.synthesizer.tts_model.speaker_manager
            self.available_speakers = list(speaker_manager.speaker_names)
        except AttributeError:
            self.available_speakers = []
        if not self.available_speakers:
            raise ValueError("No built-in speakers found in this XTTS model.")
            
        print(f"Loaded XTTS with {len(self.available_speakers)} built-in speakers.")
        self.language = cfg.get("xtts_language", "en")

    def synthesize_to_file(self, text: str, output_path: Path, speaker_id: str = None) -> float:
        # Deterministically select a built-in speaker based on the ID
        speaker_index = stable_mod(str(speaker_id), len(self.available_speakers))
        selected_speaker = self.available_speakers[speaker_index]
        
        temp_wav = output_path.with_suffix(".wav")
        
        # Generate speech
        self.tts.tts_to_file(
            text=text,
            speaker=selected_speaker,
            language=self.language,
            file_path=str(temp_wav)
        )
        
        # Convert to flac
        data, samplerate = sf.read(temp_wav)
        sf.write(output_path, data, samplerate)
        
        if temp_wav.exists():
            temp_wav.unlink()
            
        return len(data) / samplerate


def get_tts_provider(cfg: Dict) -> TTSProvider:
    provider_type = cfg.get("provider", "speecht5").lower()
    
    if provider_type == "openai":
        return OpenAIProvider(cfg)
    elif provider_type == "elevenlabs":
        return ElevenLabsProvider(cfg)
    elif provider_type == "speecht5":
        return SpeechT5Provider(cfg)
    elif provider_type == "xtts":
        return XTTSProvider(cfg)
    else:
        raise ValueError(f"Unknown TTS provider: {provider_type}. Choose from 'speecht5', 'openai', 'elevenlabs', 'xtts'")

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clean and stuttered synthetic speech.")
    parser.add_argument("--config", required=True, help="Path to TTS YAML config.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    input_manifest = cfg["input_manifest"]
    base_output_manifest = Path(cfg["output_manifest"])
    base_audio_dir = Path(cfg["audio_dir"])

    provider_type = cfg.get("provider", "speecht5").lower()
    
    # Create a separate folder for each provider to prevent overwriting
    audio_dir = base_audio_dir / provider_type
    clean_dir = audio_dir / "clean"
    stuttered_dir = audio_dir / "stuttered"
    
    # Also create a separate manifest file for each provider
    output_manifest = base_output_manifest.with_name(f"{base_output_manifest.stem}_{provider_type}{base_output_manifest.suffix}")

    ensure_dir(clean_dir)
    ensure_dir(stuttered_dir)

    max_text_chars = cfg.get("max_text_chars", 250)
    overwrite = cfg.get("overwrite", False)

    # Initialize the correct TTS provider based on config
    provider = get_tts_provider(cfg)

    audio_records = []

    for record in read_jsonl(input_manifest):
        if len(audio_records) >= 50:
            break
        record_id = record.get("augmented_id") or record.get("utterance_id")
        clean_text = record.get("clean_text", "").strip()
        stuttered_text = record.get("stuttered_text", "").strip()

        if not clean_text or not stuttered_text:
            continue

        if len(clean_text) > max_text_chars or len(stuttered_text) > max_text_chars:
            print(f"Skipping {record_id}: text too long for TTS config.")
            continue

        # Use record_id as fallback speaker_id if not present for deterministic voice mapping
        speaker_id = record.get("speaker_id", record_id)

        safe_record_id = safe_filename_part(record_id)
        safe_speaker_id = safe_filename_part(speaker_id)

        clean_path = clean_dir / f"{safe_record_id}_clean_spk_{safe_speaker_id}.flac"
        stuttered_path = stuttered_dir / f"{safe_record_id}_stuttered_spk_{safe_speaker_id}.flac"

        try:
            if overwrite or not clean_path.exists():
                clean_duration = provider.synthesize_to_file(clean_text, clean_path, speaker_id)
            else:
                clean_duration = sf.info(clean_path).duration

            if overwrite or not stuttered_path.exists():
                stuttered_duration = provider.synthesize_to_file(stuttered_text, stuttered_path, speaker_id)
            else:
                stuttered_duration = sf.info(stuttered_path).duration

            audio_records.append({
                "record_id": record_id,
                "original_utterance_id": record.get("original_utterance_id", record.get("utterance_id")),
                "clean_text": clean_text,
                "stuttered_text": stuttered_text,
                "speaker_id": speaker_id,
                "clean_audio_path": str(clean_path),
                "stuttered_audio_path": str(stuttered_path),
                "clean_duration_seconds": clean_duration,
                "stuttered_duration_seconds": stuttered_duration,
                "is_disfluent": True,
                "disfluency_events": record.get("disfluency_events", []),
                "primary_disfluency_type": record.get("primary_disfluency_type", None),
            })
            
        except Exception as e:
            print(f"Failed TTS for {record_id}: {e}")
            continue

    write_jsonl(output_manifest, audio_records)

    print(f"Generated/verified audio records: {len(audio_records)}")
    print(f"Audio manifest: {output_manifest}")

if __name__ == "__main__":
    main()
