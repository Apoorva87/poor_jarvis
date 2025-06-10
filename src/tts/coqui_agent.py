import sounddevice as sd
import numpy as np
from TTS.api import TTS
from .base import TTSAgent
import asyncio

class CoquiTTSAgent(TTSAgent):
    def __init__(self, model_name: str = "tts_models/en/vctk/vits", speaker_name: str = "p225"):
        self.model = TTS(model_name=model_name).to("cpu")
        self.speaker_name = speaker_name

    async def synthesize(self, text: str, play_audio: bool = True) -> bool:
        """Synthesize text to speech using Coqui TTS."""
        if not text:
            return False
        
        print(f"[TTS] Synthesizing speech for: '{text}'")
        try:
            # Run TTS in a thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            wav = await loop.run_in_executor(
                None, 
                lambda: self.model.tts(text=text, speaker=self.speaker_name)
            )
            
            if play_audio:
                print("[TTS] Playing synthesized speech...")
                # Run audio playback in a thread pool
                await loop.run_in_executor(
                    None,
                    lambda: (sd.play(wav, 24000), sd.wait())
                )
            
            return True
        except Exception as e:
            print(f"[TTS] Error during speech synthesis or playback: {e}")
            return False 