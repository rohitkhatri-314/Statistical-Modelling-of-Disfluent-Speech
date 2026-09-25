import os
from elevenlabs.client import ElevenLabs

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

audio = client.text_to_speech.convert(
    text="Hello, this is a test.",
    voice_id="21m00Tcm4TlvDq8ikWAM",
    model_id="eleven_multilingual_v2",
    output_format="wav_16000"
)

with open("test.wav", "wb") as f:
    for chunk in audio:
        if chunk:
            f.write(chunk)

print("TTS successful")