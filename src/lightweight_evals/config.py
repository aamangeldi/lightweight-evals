"""Configuration loading for lightweight evals."""

import os
from pathlib import Path

from dotenv import load_dotenv


class Config:
    """Configuration class for lightweight evals."""

    def __init__(self, env_file: Path | None = None):
        """Initialize config and load environment variables."""
        if env_file:
            load_dotenv(env_file)
        else:
            # Try to load .env from current directory
            load_dotenv()

    @property
    def openai_api_key(self) -> str | None:
        """Get OpenAI API key from environment."""
        return os.getenv("OPENAI_API_KEY")

    @property
    def default_model(self) -> str:
        """Get default model name."""
        return os.getenv("LWEVAL_DEFAULT_MODEL", "gpt-4o-mini")

    @property
    def max_tokens(self) -> int:
        """Get default max tokens."""
        return int(os.getenv("LWEVAL_MAX_TOKENS", "256"))

    @property
    def temperature(self) -> float:
        """Get default temperature."""
        return float(os.getenv("LWEVAL_TEMPERATURE", "0.2"))

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY not found in environment")

        if self.max_tokens <= 0:
            errors.append(f"LWEVAL_MAX_TOKENS must be positive, got {self.max_tokens}")

        if not (0.0 <= self.temperature <= 2.0):
            errors.append(f"LWEVAL_TEMPERATURE must be between 0.0 and 2.0, got {self.temperature}")

        return errors
