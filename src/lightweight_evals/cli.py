"""Command-line interface for lightweight evals."""

from pathlib import Path

import click

from .adapters.base import BaseAdapter
from .adapters.dummy import DummyAdapter
from .adapters.openai import OpenAIAdapter
from .config import Config
from .reporting.report_builder import ReportBuilder
from .runner import EvalRunner, RunConfig


@click.group()
@click.version_option()
def main() -> None:
    """Lightweight Evals - A minimal LLM evaluation library."""
    pass


@main.command()
def list_suites() -> None:
    """List available evaluation suites."""
    runner = EvalRunner()
    suites = runner.get_available_suites()

    click.echo("Available evaluation suites:")
    for suite in suites:
        click.echo(f"  - {suite}")


@main.command()
def list_adapters() -> None:
    """List available model adapters."""
    adapters = ["dummy", "openai"]

    click.echo("Available model adapters:")
    for adapter in adapters:
        click.echo(f"  - {adapter}")


@main.command()
@click.option("--adapter", required=True, type=click.Choice(["dummy", "openai"]), help="Model adapter to use")
@click.option("--suite", required=True, help="Evaluation suite to run (or 'all')")
@click.option("--out", default="./reports", help="Output directory for reports")
@click.option("--seed", default=123, help="Random seed for reproducibility")
@click.option("--model", default=None, help="Model name (for OpenAI adapter)")
@click.option("--max-tokens", default=256, help="Maximum tokens to generate")
@click.option("--temperature", default=0.2, help="Temperature for generation")
@click.option("--judge-adapter", default=None, type=click.Choice(["dummy", "openai"]), help="Adapter to use for LLM-as-a-judge scoring")
def run(adapter: str, suite: str, out: str, seed: int, model: str | None, max_tokens: int, temperature: float, judge_adapter: str | None) -> None:
    """Run evaluation suite(s) against a model adapter."""
    config = Config()

    # Validate config for OpenAI adapter
    if adapter == "openai" or judge_adapter == "openai":
        errors = config.validate()
        if errors:
            for error in errors:
                click.echo(f"Error: {error}", err=True)
            raise click.Abort()

    # Set up model adapter
    model_adapter: BaseAdapter
    if adapter == "dummy":
        model_adapter = DummyAdapter(seed=seed)
    elif adapter == "openai":
        model_name = model or config.default_model
        model_adapter = OpenAIAdapter(model=model_name, api_key=config.openai_api_key)
    else:
        raise click.BadParameter(f"Unknown adapter: {adapter}")

    # Set up judge adapter if specified
    judge_model_adapter: BaseAdapter | None = None
    if judge_adapter:
        if judge_adapter == "dummy":
            judge_model_adapter = DummyAdapter(seed=seed)
        elif judge_adapter == "openai":
            judge_model_name = model or config.default_model
            judge_model_adapter = OpenAIAdapter(model=judge_model_name, api_key=config.openai_api_key)

    # Set up runner
    runner = EvalRunner()
    output_dir = Path(out)

    run_config = RunConfig(adapter_name=adapter, eval_suite=suite, seed=seed, max_tokens=max_tokens, temperature=temperature, output_dir=output_dir)

    # Determine suites to run
    if suite == "all":
        suite_names = runner.get_available_suites()
    else:
        if suite not in runner.get_available_suites():
            click.echo(f"Error: Unknown evaluation suite '{suite}'", err=True)
            click.echo(f"Available suites: {', '.join(runner.get_available_suites())}")
            raise click.Abort()
        suite_names = [suite]

    # Run evaluations
    click.echo(f"Running {len(suite_names)} evaluation suite(s) with {adapter} adapter...")

    if judge_model_adapter:
        click.echo(f"Using {judge_adapter} adapter for LLM-as-a-judge scoring...")
    else:
        click.echo("Warning: No judge adapter specified - evaluations will not be scored")

    results = runner.run_multiple_suites(adapter=model_adapter, suite_names=suite_names, config=run_config, judge_adapter=judge_model_adapter)

    # Save results and generate reports
    report_builder = ReportBuilder()

    for result in results:
        # Save JSON
        json_path = runner.save_results(result)
        click.echo(f"Saved results: {json_path}")

        # Generate HTML report
        html_path = output_dir / f"report_{result.config.eval_suite}_{result.timestamp}.html"
        report_builder.generate_html_report(result, html_path)
        click.echo(f"Generated HTML report: {html_path}")

        # Generate Markdown report
        md_path = output_dir / f"report_{result.config.eval_suite}_{result.timestamp}.md"
        report_builder.generate_markdown_report(result, md_path)
        click.echo(f"Generated Markdown report: {md_path}")

        # Print summary
        stats = result.summary_stats
        click.echo(f"\n{result.config.eval_suite.title()} Results:")
        click.echo(f"  Pass Rate: {stats['pass_rate']:.1%}")
        click.echo(f"  Passed: {stats['passed_items']}/{stats['total_items']}")

        if stats.get("average_scores"):
            click.echo("  Average Scores:")
            for score_name, score_value in stats["average_scores"].items():
                click.echo(f"    {score_name}: {score_value:.2f}")


@main.command()
@click.option("--json", "json_path", required=True, help="Path to JSON results file")
@click.option("--format", "output_format", default="html", type=click.Choice(["html", "markdown", "both"]), help="Output format for the report")
def report(json_path: str, output_format: str) -> None:
    """Generate report from existing JSON results."""
    json_file = Path(json_path)

    if not json_file.exists():
        click.echo(f"Error: File not found: {json_path}", err=True)
        raise click.Abort()

    report_builder = ReportBuilder()

    try:
        result = report_builder.load_result_from_json(json_file)
    except Exception as e:
        click.echo(f"Error loading results: {e}", err=True)
        raise click.Abort() from e

    output_dir = json_file.parent
    base_name = json_file.stem

    if output_format in ["html", "both"]:
        html_path = output_dir / f"{base_name}.html"
        report_builder.generate_html_report(result, html_path)
        click.echo(f"Generated HTML report: {html_path}")

    if output_format in ["markdown", "both"]:
        md_path = output_dir / f"{base_name}.md"
        report_builder.generate_markdown_report(result, md_path)
        click.echo(f"Generated Markdown report: {md_path}")


if __name__ == "__main__":
    main()
