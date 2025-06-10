import collections
import sys
import webrtcvad
import sounddevice as sd
import numpy as np
import wave
import os
import requests
import torch
import io
from TTS.api import TTS
import ollama
import threading
import asyncio
from collections import deque


# --- Configuration Constants ---
SAMPLE_RATE = 16000  # Audio sample rate for VAD, STT, and TTS (WebRTC VAD supports 8k, 16k, 32k, 48k)
FRAME_DURATION_MS = 30  # Duration of a frame in milliseconds (10, 20, or 30 for WebRTC VAD)
VAD_MODE = 3  # VAD aggressiveness mode: 0 (least aggressive) to 3 (most aggressive)
PADDING_DURATION_MS = 500  # Additional padding around detected speech to capture context

# Calculate frame size based on sample rate and frame duration
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)

# Whisper STT Configuration
WHISPER_MODEL_NAME = "base.en"  # or "base.en", "small.en" for better accuracy. "large" is very slow.
WHISPER_MODEL = None # Will be loaded dynamically

# Coqui TTS Configuration
TTS_MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC" # Default English model
TTS_MODEL_NAME="tts_models/en/vctk/vits"
TTS_MODEL = None # Will be loaded dynamically
TTS_SPEAKER_NAME="p225" #None # Can be specified for multi-speaker models
TTS_LANGUAGE="en"

# Ollama LLM Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b" # Ensure this model is pulled and running in Ollama

# Set max history length (e.g., last 10 messages total = 5 exchanges)
MAX_HISTORY = 10

# Conversation history using deque for bounded memory
chat_history = deque(maxlen=MAX_HISTORY)

# --- Helper Function: Audio Frame Generator ---
def audio_frame_generator(stream, frame_size):
    """
    Generates audio frames from the input stream.
    Each frame is a numpy array of integers.
    """
    while True:
        try:
            data, overflowed = stream.read(frame_size)
            if overflowed:
                print("Warning: Audio input buffer overflowed!", file=sys.stderr)
            # Convert numpy array to bytes for WebRTC VAD.
            # Ensure the data type is int16 and convert to bytes in little-endian format.
            yield data.astype(np.int16).tobytes()
        except Exception as e:
            print(f"Error reading from audio stream: {e}", file=sys.stderr)
            break # Exit generator on error


# --- Helper Function: VAD Collector ---
def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """
    Filters out non-speech frames using VAD and collects speech segments.
    
    Args:
        sample_rate (int): The audio sample rate (e.g., 16000 Hz).
        frame_duration_ms (int): The duration of each audio frame in milliseconds.
        padding_duration_ms (int): Additional padding around detected speech to capture context.
        vad (webrtcvad.Vad): An initialized WebRTC VAD instance.
        frames (iterable): An iterable of audio frames (byte strings).

    Yields:
        bytes: A complete audio segment (bytes) when speech ends.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    
    for frame_index, frame in enumerate(frames):
        is_speech = vad.is_speech(frame, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            if is_speech:
                triggered = True
                # Add frames from the ring buffer (padding) to the voiced_frames
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
                print(f"\n[VAD] Speech detected. Starting recording...")
        else: # Triggered
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            if not is_speech:
                # If current frame is not speech, check if we've left a speech segment
                # and if all padding frames are also non-speech.
                num_unvoiced = len([f for f, s in ring_buffer if not s])
                if num_unvoiced == ring_buffer.maxlen:
                    # All padding frames are unvoiced, end of speech segment
                    triggered = False
                    # Remove the unvoiced padding frames from voiced_frames
                    for _ in range(ring_buffer.maxlen):
                        voiced_frames.pop()
                    print("\n[VAD] Speech segment ended. Processing...")
                    yield b''.join(voiced_frames)
                    voiced_frames = []
                    ring_buffer.clear()
    
    # If the stream ends while triggered, yield the remaining voiced frames
    if voiced_frames:
        print("\n[VAD] Stream ended while speech was active. Processing last segment...")
        yield b''.join(voiced_frames)

# --- Speech-to-Text (Whisper) ---
async def transcribe_audio(audio_segment_bytes, model):
    """
    Transcribes an audio segment using the Whisper model.
    Args:
        audio_segment_bytes (bytes): The raw audio bytes (int16 mono).
        model (whisper.Whisper): The loaded Whisper model.
    Returns:
        str: The transcribed text.
    """
    if not audio_segment_bytes:
        return ""
    
    print("[STT] Transcribing audio...")
    # Whisper expects audio as a 16kHz float32 numpy array
    audio_np = np.frombuffer(audio_segment_bytes, dtype=np.int16).flatten().astype(np.float32) / 32768.0
    
    try:
        result = model.transcribe(audio_np, fp16=torch.cuda.is_available())
        transcription = result["text"].strip()
        print(f"[STT] Transcription: '{transcription}'")
        return transcription
    except Exception as e:
        print(f"[STT] Error during transcription: {e}", file=sys.stderr)
        return ""

import re
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r'', text)


def add_message(role, content):
    """Add a message to the chat history."""
    chat_history.append({"role": role, "content": content})

add_message("system", "You are a helpful personal voice Assistant. Do not generate formatter answers. Do not generate emoticons and keeps answers concise.")

# --- Local LLM Interaction (Ollama) ---
def get_llm_response(prompt):
    """
    Sends a prompt to the local Ollama LLM and returns the response.
    Args:
        prompt (str): The input prompt for the LLM.
    Returns:
        str: The LLM's generated response.
    """
    if not prompt:
        return "Please say something."
    
    print(f"[LLM] Sending prompt to Ollama: '{prompt}'")

    try:
        print(list(chat_history))
        response = ollama.chat(model=OLLAMA_MODEL, 
                               messages=list(chat_history)+[{'role':'user','content':prompt}],
                               think=False
                               )
        add_message("user", prompt)
    except Exception as e:
        print(f"Exception as {e}")
        return "Sorry. My brain isnt working, try again. Pardon me."

    print(f"[LLM] Ollama response: '{response}'")

    text = remove_emojis(response['message']['content'])
    add_message(response['message']['role'], text)

    return response['message']['content']


# --- Text-to-Speech (Coqui TTS) ---
def synthesize_speech(text, tts_model, play_audio=True):
    """
    Synthesizes text into speech using Coqui TTS and optionally plays it.
    Args:
        text (str): The text to synthesize.
        tts_model (TTS.api.TTS): The loaded Coqui TTS model.
        play_audio (bool): Whether to play the audio back.
    Returns:
        numpy.ndarray: The synthesized audio as a numpy array, or None if error.
    """
    if not text:
        return None
    
    print(f"[TTS] Synthesizing speech for: '{text}'")
    try:
        # Coqui TTS generates output as a numpy array
        # Note: Ensure the `speaker` and `language` parameters match your model if needed.
        # For 'tts_models/en/ljspeech/tacotron2-DDC', speaker is not usually needed.
        # If using a multi-speaker model, you would set TTS_SPEAKER_NAME.
        #wav = tts_model.tts(text=text, speaker=TTS_SPEAKER_NAME, language=TTS_LANGUAGE)
        wav = tts_model.tts(text=text, speaker=TTS_SPEAKER_NAME)
        
        # Convert to numpy array if not already, and ensure correct dtype for sounddevice
        #audio_out = np.array(wav).astype(np.float32)
        
        if play_audio:
            print("[TTS] Playing synthesized speech...")
            sd.play(wav, 24000)
            sd.wait() # Wait until playback is finished
        
        return True
    except Exception as e:
        print(f"[TTS] Error during speech synthesis or playback: {e}", file=sys.stderr)
        return None


async def get_llm_response_async(text):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, get_llm_response, text)

async def synthesize_speech_async(text, tts_model):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, synthesize_speech, text, tts_model)


# --- Main Execution ---
async def main():
    global WHISPER_MODEL, TTS_MODEL

    # Initialize WebRTC VAD
    vad = webrtcvad.Vad(VAD_MODE)
    print(f"[INIT] WebRTC VAD initialized with mode: {VAD_MODE}")
    print(f"[INIT] Audio Config: Sample Rate: {SAMPLE_RATE} Hz, Frame Duration: {FRAME_DURATION_MS} ms")

    # Load Whisper Model
    try:
        import whisper
        print(f"[INIT] Loading Whisper model: {WHISPER_MODEL_NAME}...")
        WHISPER_MODEL = whisper.load_model(WHISPER_MODEL_NAME)
        print(f"[INIT] Whisper model loaded.")
    except Exception as e:
        print(f"[INIT] Error loading Whisper model: {e}", file=sys.stderr)
        #print("Tip: If you're encountering CUDA errors, try running on CPU (fp16=False in t r anscribe adio) or ensure PyTorch is correctly installed with CUDA support.", file=sys.stderr)
        sys.exit(1)

    # Load Coqui TTS Model
    try:
        print(f"[INIT] Loading Coqui TTS model: {TTS_MODEL_NAME}...")
        TTS_MODEL = TTS(model_name=TTS_MODEL_NAME).to("cpu")
        print(f"[INIT] Coqui TTS model loaded.")
    except Exception as e:
        print(f"[INIT] Error loading Coqui TTS model: {e}", file=sys.stderr)
        print("Tip: Ensure you have downloaded the necessary Coqui TTS model files. Check TTS documentation for common issues.", file=sys.stderr)
        sys.exit(1)
    
    print("\n[SYSTEM] Ready! Listening for your voice...")
    print("Press Ctrl+C to stop.")

    try:
        # Start the sound device input stream
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16') as stream:
            # Create a generator for audio frames
            frames = audio_frame_generator(stream, FRAME_SIZE)
            
            # Use the VAD collector to process frames and identify speech segments
            for audio_segment in vad_collector(
                SAMPLE_RATE, FRAME_DURATION_MS, PADDING_DURATION_MS, vad, frames
            ):
                if audio_segment:
                    print(f"\n[SYSTEM] Detected speech segment ({len(audio_segment) / (2 * SAMPLE_RATE):.2f} seconds).")
                    # 1. Transcribe the speech segment
                    transcribed_text = await transcribe_audio(audio_segment, WHISPER_MODEL)

                    if transcribed_text.strip().lower() in ["exit", "quit", "stop"]:
                        print("Exiting.")
                        break
                  
                    print(f"Passing this down to llm : {transcribed_text}")
                    if transcribed_text:
                        llm_response =  await get_llm_response_async(transcribed_text)
                        if llm_response:
                            await synthesize_speech_async(llm_response, TTS_MODEL)
                    else:
                        print("[SYSTEM] No clear speech detected for transcription.")
                else:
                    print("[SYSTEM] No audio segment captured by VAD.")

    except KeyboardInterrupt:
        print("\n[SYSTEM] Stopping conversational AI system.")
    except Exception as e:
        print(f"\n[SYSTEM] An unexpected error occurred: {e}", file=sys.stderr)
        print("[SYSTEM] Please ensure your microphone is connected and working.", file=sys.stderr)
        print("[SYSTEM] Also check sounddevice documentation and your system's audio settings.", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(main())


