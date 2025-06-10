from abc import ABC, abstractmethod

class STTAgent(ABC):
    @abstractmethod
    async def transcribe(self, audio_segment_bytes: bytes) -> str:
        """Transcribe audio bytes to text."""
        pass 