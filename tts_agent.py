from abc import ABC, abstractmethod
import sounddevice as sd
import numpy as np
from TTS.api import TTS

class TTSAgent(ABC):
    @abstractmethod
    def synthesize(self, text: str, play_audio: bool = True) -> bool:
        """Synthesize text to speech and optionally play it."""
        pass

class CoquiTTSAgent(TTSAgent):
    def __init__(self, model_name: str = "tts_models/en/vctk/vits", speaker_name: str = "p225"):
        self.model = TTS(model_name=model_name).to("cpu")
        self.speaker_name = speaker_name

    def synthesize(self, text: str, play_audio: bool = True) -> bool:
        """Synthesize text to speech using Coqui TTS."""
        if not text:
            return False
        
        print(f"[TTS] Synthesizing speech for: '{text}'")
        try:
            wav = self.model.tts(text=text, speaker=self.speaker_name)
            
            if play_audio:
                print("[TTS] Playing synthesized speech...")
                sd.play(wav, 24000)
                sd.wait()
            
            return True
        except Exception as e:
            print(f"[TTS] Error during speech synthesis or playback: {e}")
            return False 