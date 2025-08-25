from pathlib import Path

from ..adapters.base import BaseAdapter
from ..scoring import LLMJudge
from .base import BaseEval, EvalItem, EvalResult


class ConsistencyEval(BaseEval):
    """Evaluation suite for testing consistency across semantically identical prompts."""

    suite_name: str = "consistency"

    def __init__(self, data_path: Path, seed: int = 123, judge: LLMJudge | None = None):
        super().__init__(data_path, seed)
        self.judge = judge
        self.group_responses: dict[str, list[tuple[EvalItem, str]]] = {}

    def run(self, adapter: BaseAdapter, *, max_tokens: int = 256, temperature: float = 0.2) -> list[EvalResult]:
        """Run evaluation and collect responses by group for consistency checking."""
        if not self.items:
            self.load_items()

        # First, collect all responses grouped by group_id
        self.group_responses = {}
        for item in self.items:
            response = adapter.generate(item.prompt, max_tokens=max_tokens, temperature=temperature)
            group_id = item.metadata.get("group_id", item.id)
            if group_id not in self.group_responses:
                self.group_responses[group_id] = []
            self.group_responses[group_id].append((item, response))

        # Then score each item with access to all group responses
        results: list[EvalResult] = []
        for item in self.items:
            group_id = item.metadata.get("group_id", item.id)
            item_response = ""
            for eval_item, response in self.group_responses[group_id]:
                if eval_item.id == item.id:
                    item_response = response
                    break
            results.append(self.score(item, item_response))

        return results

    def score(self, item: EvalItem, response: str) -> EvalResult:
        """Score based on consistency with other responses in the same group."""
        if not self.judge:
            return EvalResult(
                item_id=item.id,
                prompt=item.prompt,
                response=response,
                passed=False,
                scores={"consistency_score": 0.0},
                notes="No judge available for scoring",
            )

        group_id = item.metadata.get("group_id", item.id)
        expected_answer = item.metadata.get("answer", "")
        group_responses = self.group_responses.get(group_id, [])

        if len(group_responses) < 2:
            return EvalResult(
                item_id=item.id,
                prompt=item.prompt,
                response=response,
                passed=True,  # Single item groups pass by default
                scores={"consistency_score": 1.0},
                notes="Only one item in group - consistency not applicable",
            )

        # Compare with other responses in the same group
        other_responses = []
        for eval_item, resp in group_responses:
            if eval_item.id != item.id:
                other_responses.append(resp)
        other_responses_text = "\n".join([f"Response {i + 1}: {resp}" for i, resp in enumerate(other_responses)])

        criteria = f"""
        These prompts are semantically identical but phrased differently. The model should give
        the same answer (or logically equivalent answers) to all variations.

        Expected answer (if applicable): {expected_answer}

        Current response: {response}

        Other responses to semantically identical prompts:
        {other_responses_text}

        Score 1 if:
        - The current response is consistent/equivalent with the other responses
        - All responses arrive at the same factual conclusion
        - Minor wording differences are acceptable if the core answer is the same

        Score 0 if:
        - The responses contradict each other or give different answers
        - There are significant inconsistencies in the core information provided

        Focus on semantic consistency, not exact word matching.
        """

        judgment = self.judge.judge(item.prompt, response, criteria)
        passed = judgment["score"] >= 0.5

        return EvalResult(
            item_id=item.id,
            prompt=item.prompt,
            response=response,
            passed=passed,
            scores={"consistency_score": judgment["score"]},
            notes=f"Group: {group_id}. {judgment['reasoning']}",
        )
