from dataclasses import dataclass, field
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class OpenAIConfig:
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", "150"))

@dataclass
class OllamaConfig:
    model: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

@dataclass
class TTSConfig:
    model_name: str = os.getenv("TTS_MODEL", "tts_models/en/vctk/vits")
    speaker_name: str = os.getenv("TTS_SPEAKER", "p225")

@dataclass
class STTConfig:
    model_name: str = os.getenv("STT_MODEL", "base.en")

@dataclass
class VADConfig:
    sample_rate: int = int(os.getenv("VAD_SAMPLE_RATE", "16000"))
    frame_duration_ms: int = int(os.getenv("VAD_FRAME_DURATION_MS", "30"))
    padding_duration_ms: int = int(os.getenv("VAD_PADDING_DURATION_MS", "500"))
    mode: int = int(os.getenv("VAD_MODE", "3"))

@dataclass
class Config:
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    max_history: int = int(os.getenv("MAX_HISTORY", "10"))

# Create a global config instance
config = Config()
