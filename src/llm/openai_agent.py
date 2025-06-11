from openai import AsyncOpenAI
from .base import LLMAgent
from ..config import config

class OpenAIAgent(LLMAgent):
    def __init__(self):
        super().__init__(max_history=config.max_history)
        self.model_name = config.openai.model
        self.client = AsyncOpenAI(api_key=config.openai.api_key)

    async def get_response(self, prompt: str) -> str:
        """Get response from OpenAI LLM."""
        if not prompt:
            return "Please say something."
        
        print(f"[LLM] Sending prompt to OpenAI: '{prompt}'")
        try:
            print(list(self.chat_history))
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=list(self.chat_history) + [{'role': 'user', 'content': prompt}],
                temperature=config.openai.temperature,
                max_tokens=config.openai.max_tokens
            )
            self.add_message("user", prompt)
        except Exception as e:
            print(f"Exception as {e}")
            return "Sorry. My brain isnt working, try again. Pardon me."

        print(f"[LLM] OpenAI response: '{response}'")
        text = self._clean_text(response.choices[0].message.content)
        self.add_message(response.choices[0].message.role, text)
        return text 