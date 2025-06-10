import ollama
from collections import deque
import re

class OllamaAgent:
    def __init__(self, model_name: str = "qwen3:8b", max_history: int = 10):
        self.model_name = model_name
        self.chat_history = deque(maxlen=max_history)
        self._initialize_chat_history()

    def _initialize_chat_history(self):
        """Initialize chat history with system message."""
        self.add_message("system", "You are a helpful personal voice Assistant. Do not generate formatter answers. Do not generate emoticons and keeps answers concise.")

    def add_message(self, role: str, content: str):
        """Add a message to the chat history."""
        self.chat_history.append({"role": role, "content": content})

    def _remove_emojis(self, text: str) -> str:
        """Remove emojis from text."""
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

    def get_response(self, prompt: str) -> str:
        """Get response from Ollama LLM."""
        if not prompt:
            return "Please say something."
        
        print(f"[LLM] Sending prompt to Ollama: '{prompt}'")
        try:
            print(list(self.chat_history))
            response = ollama.chat(
                model=self.model_name,
                messages=list(self.chat_history) + [{'role': 'user', 'content': prompt}],
                think=False
            )
            self.add_message("user", prompt)
        except Exception as e:
            print(f"Exception as {e}")
            return "Sorry. My brain isnt working, try again. Pardon me."

        print(f"[LLM] Ollama response: '{response}'")
        text = self._remove_emojis(response['message']['content'])
        self.add_message(response['message']['role'], text)
        return response['message']['content'] 