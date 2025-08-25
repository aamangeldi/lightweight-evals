"""Main runner for lightweight evals."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .adapters.base import BaseAdapter
from .evals.base import BaseEval, EvalResult
from .evals.consistency import ConsistencyEval
from .evals.harmlessness import HarmlessnessEval
from .evals.robustness import RobustnessEval
from .scoring import LLMJudge
from .utils import format_timestamp, generate_run_id, set_seed


@dataclass
class RunConfig:
    """Configuration for an evaluation run."""

    adapter_name: str
    eval_suite: str
    seed: int = 123
    max_tokens: int = 256
    temperature: float = 0.2
    output_dir: Path = Path("./reports")


@dataclass
class RunResult:
    """Results from an evaluation run."""

    run_id: str
    timestamp: str
    config: RunConfig
    adapter_info: dict[str, str]
    eval_results: list[EvalResult]
    summary_stats: dict[str, Any]


class EvalRunner:
    """Main runner for lightweight evaluations."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data"

        # Registry of available eval suites
        self.eval_registry: dict[str, type[BaseEval]] = {
            "harmlessness": HarmlessnessEval,
            "robustness": RobustnessEval,
            "consistency": ConsistencyEval,
        }

    def get_available_suites(self) -> list[str]:
        """Get list of available evaluation suites."""
        return list(self.eval_registry.keys())

    def run_eval(self, adapter: BaseAdapter, suite_name: str, config: RunConfig, judge_adapter: BaseAdapter | None = None) -> RunResult:
        """Run a single evaluation suite."""
        set_seed(config.seed)

        if suite_name not in self.eval_registry:
            raise ValueError(f"Unknown eval suite: {suite_name}")

        # Set up judge if provided
        judge = LLMJudge(judge_adapter) if judge_adapter else None

        # Load eval suite
        eval_class = self.eval_registry[suite_name]
        data_path = self.data_dir / f"{suite_name}.jsonl"
        eval_suite = eval_class(data_path, config.seed, judge)

        # Generate run metadata
        timestamp = format_timestamp()
        run_id = generate_run_id(
            adapter_name=adapter.name, adapter_version=adapter.version, eval_suite_name=suite_name, data_path=data_path, timestamp=timestamp
        )

        # Run evaluation
        eval_results = eval_suite.run(adapter, max_tokens=config.max_tokens, temperature=config.temperature)

        # Calculate summary stats
        total_items = len(eval_results)
        passed_items = sum(1 for r in eval_results if r.passed)
        pass_rate = passed_items / total_items if total_items > 0 else 0.0

        # Average scores
        all_scores: dict[str, list[float]] = {}
        for result in eval_results:
            for score_name, score_value in result.scores.items():
                if score_name not in all_scores:
                    all_scores[score_name] = []
                all_scores[score_name].append(score_value)

        avg_scores = {name: sum(values) / len(values) if values else 0.0 for name, values in all_scores.items()}

        summary_stats = {"total_items": total_items, "passed_items": passed_items, "pass_rate": pass_rate, "average_scores": avg_scores}

        return RunResult(
            run_id=run_id,
            timestamp=timestamp,
            config=config,
            adapter_info={"name": adapter.name, "version": adapter.version},
            eval_results=eval_results,
            summary_stats=summary_stats,
        )

    def run_multiple_suites(
        self, adapter: BaseAdapter, suite_names: list[str], config: RunConfig, judge_adapter: BaseAdapter | None = None
    ) -> list[RunResult]:
        """Run multiple evaluation suites."""
        results = []
        for suite_name in suite_names:
            suite_config = RunConfig(
                adapter_name=config.adapter_name,
                eval_suite=suite_name,
                seed=config.seed,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                output_dir=config.output_dir,
            )
            result = self.run_eval(adapter, suite_name, suite_config, judge_adapter)
            results.append(result)
        return results

    def save_results(self, result: RunResult) -> Path:
        """Save results to JSON file."""
        result.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        serializable_result = {
            "run_id": result.run_id,
            "timestamp": result.timestamp,
            "config": asdict(result.config),
            "adapter_info": result.adapter_info,
            "eval_results": [asdict(r) for r in result.eval_results],
            "summary_stats": result.summary_stats,
        }

        # Convert Path objects to strings for JSON serialization
        if isinstance(serializable_result["config"], dict):
            serializable_result["config"]["output_dir"] = str(result.config.output_dir)

        filename = f"run_{result.timestamp}_{result.run_id}.json"
        output_path = result.config.output_dir / filename

        with open(output_path, "w") as f:
            json.dump(serializable_result, f, indent=2)

        return output_path
