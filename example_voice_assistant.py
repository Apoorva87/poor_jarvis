# Voice Assistant: Whisper + Coqui TTS + Ollama (LLM)

import os
import sounddevice as sd
import numpy as np
import whisper
from TTS.api import TTS
import tempfile
import subprocess
import ollama
import soundfile as sf

import asyncio
import threading

# ========== CONFIGURATION ==========
SAMPLE_RATE = 16000
DURATION = 5  # seconds
DURATION_SMALL = 0.5
OLLAMA_MODEL="qwen3:8b"
TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"

# ========== INIT MODELS ============
print("Loading Whisper model...")
stt_model = whisper.load_model("base.en")

print("Loading Coqui TTS model...")
tts = TTS(TTS_MODEL).to("cpu")

def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Done recording.")
    return np.squeeze(audio)

def transcribe(audio):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, SAMPLE_RATE)
        result = stt_model.transcribe(f.name)
    os.remove(f.name)
    return result["text"]

def query_llm(prompt):
    response = ollama.chat(model=OLLAMA_MODEL, 
                           messages=[{"role": "user", 
                                      "content": prompt}],
                           think=False)
    return response['message']['content']

def speak(text):
    wav = tts.tts(text)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, wav, samplerate=tts.synthesizer.output_sample_rate)
        subprocess.run(["afplay", f.name])  # macOS only; use "play" or "ffplay" for Linux/Windows
        os.remove(f.name)

# ========== MAIN LOOP ============
print("Voice assistant ready. Press Ctrl+C to stop.")

def play_audio_nonblocking(user_text, samplerate=22050):
    def _play():
        response = query_ollama(user_text)               # sync or async
        print(f"{response}")
        audio = tts.tts(response)
        sd.play(audio, samplerate=samplerate)
        sd.wait()
    threading.Thread(target=_play).start()

import webrtcvad
vad = webrtcvad.Vad(3)
turn_detect=False

# --- Configuration Constants ---
SAMPLE_RATE = 16000  # WebRTC VAD only supports 8kHz, 16kHz, 32kHz, and 48kHz
FRAME_DURATION_MS = 10  # Duration of a frame in milliseconds (10, 20, or 30)
VAD_MODE = 3  # VAD aggressiveness mode: 0 (least aggressive) to 3 (most aggressive)
PADDING_DURATION_MS = 300 # Additional padding around detected speech to capture context

# Calculate frame size based on sample rate and frame duration
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)



async def transcribe_microphone_async(old_audio=None):
    old_audio = old_audio
    DURATION_SMALL=(10/1000)
    print(f"Listening..{DURATION_SMALL}")
    try:
        # audio = sd.rec(int(DURATION_SMALL * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        audio =  sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        if vad.is_speech(audio, SAMPLE_RATE):
            old_audio += audio
            transcribe_microphone_async(old_audio)
            turn_detect = True
        else:
            if not turn_detect:
                return None
    except Exception as e:
        print(f"Exception as {e=}")
        return None

    printf("passing to transcribe..")
    user_text = transcribe(old_audio)
    print(f"Passing text..{user_text}")
    return user_text

async def conversation_loop():
    while True:
        try:
            user_text = await transcribe_microphone_async()  # your async STT
        except Exception as e:
            print(f"{e=}")
        if user_text:
            play_audio_nonblocking(user_text)

asyncio.run (conversation_loop())
#try:
#    while True:
#        audio = record_audio()
#        user_text = transcribe(audio)
#        print(f"User: {user_text}")
#
#        if user_text.strip().lower() in ["exit", "quit", "stop"]:
#            print("Exiting.")
#            break
#
#        response = query_llm(user_text)
#        print(f"Assistant: {response}")
#
#        speak(response)
#
#except KeyboardInterrupt:
#    print("Stopped by user.")
#
