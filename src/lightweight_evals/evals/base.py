import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..adapters.base import BaseAdapter

if TYPE_CHECKING:
    from ..scoring import LLMJudge


@dataclass(frozen=True)
class EvalItem:
    id: str
    prompt: str
    expected_behavior: str  # "refuse" | "comply" | "consistent"
    metadata: dict[str, Any]


@dataclass
class EvalResult:
    item_id: str
    prompt: str
    response: str
    passed: bool
    scores: dict[str, float]
    notes: str | None = None


class BaseEval(ABC):
    """Abstract base for all evaluation suites."""

    suite_name: str = "base"

    def __init__(self, data_path: Path, seed: int = 123, judge: "LLMJudge | None" = None):
        self.data_path = Path(data_path)
        self.seed = seed
        self.items: list[EvalItem] = []

    def load_items(self) -> list[EvalItem]:
        """Load eval items from a JSONL file."""
        self.items = []
        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.items.append(
                    EvalItem(
                        id=obj["id"],
                        prompt=obj["prompt"],
                        expected_behavior=obj["expected_behavior"],
                        metadata=obj.get("metadata", {}),
                    )
                )
        return self.items

    @abstractmethod
    def score(self, item: EvalItem, response: str) -> EvalResult:
        """Subclass must implement: how to decide pass/fail for one item."""
        raise NotImplementedError

    def run(self, adapter: BaseAdapter, *, max_tokens: int = 256, temperature: float = 0.2) -> list[EvalResult]:
        """Run all eval items through a model adapter."""
        if not self.items:
            self.load_items()
        results: list[EvalResult] = []
        for it in self.items:
            response = adapter.generate(it.prompt, max_tokens=max_tokens, temperature=temperature)
            results.append(self.score(it, response))
        return results
