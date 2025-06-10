from abc import ABC, abstractmethod

class TTSAgent(ABC):
    @abstractmethod
    async def synthesize(self, text: str, play_audio: bool = True) -> bool:
        """Synthesize text to speech and optionally play it."""
        pass 