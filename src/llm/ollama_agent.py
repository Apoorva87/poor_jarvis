import ollama
from collections import deque
import re
from .base import LLMAgent
import asyncio

class OllamaAgent(LLMAgent):
    def __init__(self, model_name: str = "qwen3:8b", max_history: int = 10):
        super().__init__(max_history)
        self.model_name = model_name
        self._initialize_chat_history()

    def _initialize_chat_history(self):
        """Initialize chat history with system message."""
        self.add_message("system", 
                         " You are a helpful personal voice Assistant. Make sure your responses have no text formatting, no hi-lights or bullets and do not generate emoticons and keep answers concise.")

    def add_message(self, role: str, content: str):
        """Add a message to the chat history."""
        self.chat_history.append({"role": role, "content": content})

    def _remove_emojis(self, text: str) -> str:
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

    async def get_response(self, prompt: str) -> str:
        """Get response from Ollama LLM."""
        if not prompt:
            return "Please say something."
        
        print(f"[LLM] Sending prompt to Ollama: '{prompt}'")
        try:
            print(list(self.chat_history))
            # Run Ollama in a thread pool since it's not async
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: ollama.chat(
                    model=self.model_name,
                    messages=list(self.chat_history) + [{'role': 'user', 'content': prompt}],
                    think=False
                )
            )
            self.add_message("user", prompt)
        except Exception as e:
            print(f"Exception as {e}")
            return "Sorry. My brain isnt working, try again. Pardon me."

        print(f"[LLM] Ollama response: '{response}'")
        text = self._remove_emojis(response['message']['content'])
        self.add_message(response['message']['role'], text)
        return response['message']['content'] 