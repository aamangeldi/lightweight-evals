from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    """Abstract base class for all model adapters."""

    name: str = "base"
    version: str = "0.0"

    @abstractmethod
    def generate(self, prompt: str, *, max_tokens: int = 256, temperature: float = 0.2) -> str:
        """Generate a response to the given prompt."""
        raise NotImplementedError
