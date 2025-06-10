import numpy as np
import torch
import whisper
from .base import STTAgent

class WhisperSTTAgent(STTAgent):
    def __init__(self, model_name: str = "base.en"):
        self.model = whisper.load_model(model_name)

    async def transcribe(self, audio_segment_bytes: bytes) -> str:
        """Transcribe audio using Whisper model."""
        if not audio_segment_bytes:
            return ""
        
        print("[STT] Transcribing audio...")
        try:
            audio_np = np.frombuffer(audio_segment_bytes, dtype=np.int16).flatten().astype(np.float32) / 32768.0
            result = self.model.transcribe(audio_np, fp16=torch.cuda.is_available())
            transcription = result["text"].strip()
            print(f"[STT] Transcription: '{transcription}'")
            return transcription
        except Exception as e:
            print(f"[STT] Error during transcription: {e}")
            return "" 