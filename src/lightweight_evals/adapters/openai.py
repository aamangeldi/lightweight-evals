import os

from openai import OpenAI

from .base import BaseAdapter


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI models."""

    name: str = "openai"
    version: str = "1.0"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(self, prompt: str, *, max_tokens: int = 256, temperature: float = 0.2) -> str:
        """Generate a response using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"Error generating response: {str(e)}"
