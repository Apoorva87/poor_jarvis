from abc import ABC, abstractmethod
from collections import deque
import re

class LLMAgent(ABC):
    def __init__(self, max_history: int = 10):
        self.chat_history = deque(maxlen=max_history)
        self._initialize_chat_history()

    def _initialize_chat_history(self):
        """Initialize chat history with system message."""
        self.add_message("system", "You are a helpful personal voice Assistant. Do not generate formatted answers. Do not generate emoticons and keeps answers concise.")

    def add_message(self, role: str, content: str):
        """Add a message to the chat history."""
        self.chat_history.append({"role": role, "content": content})

    def _clean_text(self, text: str) -> str:
        """Remove emojis and formatting characters from text."""
        # Remove emojis
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub(r'', text)
        
        # Remove markdown and formatting characters
        formatting_pattern = re.compile(r'[*_~`#]|\[|\]|\(\)|```|`|>|#+')
        text = formatting_pattern.sub('', text)
        
        # Remove multiple spaces and trim
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    @abstractmethod
    async def get_response(self, prompt: str) -> str:
        """Get response from the LLM."""
        pass 