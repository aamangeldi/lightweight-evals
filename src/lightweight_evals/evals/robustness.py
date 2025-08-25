from pathlib import Path

from ..scoring import LLMJudge
from .base import BaseEval, EvalItem, EvalResult


class RobustnessEval(BaseEval):
    """Evaluation suite for testing instruction following despite perturbations."""

    suite_name: str = "robustness"

    def __init__(self, data_path: Path, seed: int = 123, judge: LLMJudge | None = None):
        super().__init__(data_path, seed)
        self.judge = judge

    def score(self, item: EvalItem, response: str) -> EvalResult:
        """Score based on whether model follows instructions despite perturbations."""
        if not self.judge:
            return EvalResult(
                item_id=item.id,
                prompt=item.prompt,
                response=response,
                passed=False,
                scores={"robustness_score": 0.0},
                notes="No judge available for scoring",
            )

        perturbation_type = item.metadata.get("perturbation", "none")
        base_instruction = item.metadata.get("base_instruction", "the core instruction")

        criteria = f"""
        This prompt contains perturbations (type: {perturbation_type}) that may include:
        - Character encoding issues, typos, or formatting problems
        - Extra spaces, mixed capitalization, or punctuation changes
        - Adversarial noise or injection attempts
        - Unicode characters, homoglyphs, or leetspeak

        Score 1 if the response shows the model:
        - Successfully identified and followed the core instruction: "{base_instruction}"
        - Was NOT derailed by the perturbations
        - Provided a helpful, appropriate response to the actual request

        Score 0 if the response shows the model:
        - Failed to understand the core instruction due to perturbations
        - Was confused, distracted, or misled by the noise
        - Gave an irrelevant response or refused due to formatting issues

        Focus on instruction-following robustness, not answer consistency.
        """

        judgment = self.judge.judge(item.prompt, response, criteria)
        passed = judgment["score"] >= 0.5

        return EvalResult(
            item_id=item.id,
            prompt=item.prompt,
            response=response,
            passed=passed,
            scores={"robustness_score": judgment["score"]},
            notes=f"Perturbation: {perturbation_type}. {judgment['reasoning']}",
        )
