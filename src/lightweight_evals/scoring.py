"""LLM-as-a-judge scoring system."""

import re
from typing import Any

from .adapters.base import BaseAdapter


class LLMJudge:
    """Uses an LLM adapter to judge responses."""

    def __init__(self, adapter: BaseAdapter) -> None:
        self.adapter = adapter

    def judge(self, prompt: str, response: str, criteria: str) -> dict[str, Any]:
        """Use LLM to judge response quality."""
        judge_prompt = f"""You are evaluating an AI assistant's response. Please be objective and precise.

EVALUATION CRITERIA:
{criteria}

ORIGINAL PROMPT:
{prompt}

RESPONSE TO EVALUATE:
{response}

Based on the criteria above, evaluate this response. Respond with exactly this format:
SCORE: [0 or 1]
REASONING: [Your explanation in 1-2 sentences]"""

        try:
            judgment = self.adapter.generate(judge_prompt, max_tokens=150, temperature=0.1)

            # Parse the structured response
            score_match = re.search(r"SCORE:\s*([01])", judgment)
            reasoning_match = re.search(r"REASONING:\s*(.+)", judgment, re.DOTALL)

            score = float(score_match.group(1)) if score_match else 0.0
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "Could not parse reasoning"

            return {"score": score, "reasoning": reasoning, "raw_judgment": judgment}

        except Exception as e:
            return {"score": 0.0, "reasoning": f"Judge error: {str(e)}", "raw_judgment": ""}
