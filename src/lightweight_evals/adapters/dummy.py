import random

from .base import BaseAdapter


class DummyAdapter(BaseAdapter):
    """A dummy adapter that returns randomized canned responses for testing."""

    name: str = "dummy"
    version: str = "1.0"

    def __init__(self, seed: int = 123) -> None:
        self.seed = seed
        random.seed(seed)

        self.responses: list[str] = [
            "I am sorry, I am a dummy and can't answer that.",
            "This is a dummy response for testing purposes.",
            "I cannot provide a meaningful response as I'm just a test adapter.",
            "Dummy adapter activated - no real processing available.",
            "This is a placeholder response from the dummy model.",
            "I'm a dummy adapter and don't have real capabilities.",
            "Test response: This adapter is for development only.",
            "Dummy mode: Cannot generate actual content.",
        ]

    def generate(self, prompt: str, *, max_tokens: int = 256, temperature: float = 0.2) -> str:
        """Generate a random canned response."""
        # Use prompt hash to ensure deterministic responses for the same prompt
        prompt_hash = hash(prompt) % len(self.responses)
        return self.responses[prompt_hash]
