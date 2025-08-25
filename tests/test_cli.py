"""Tests for CLI interface."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from lightweight_evals.cli import main


class TestCLI:
    """Tests for CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_list_suites(self, runner):
        """Test list-suites command."""
        result = runner.invoke(main, ["list-suites"])

        assert result.exit_code == 0
        assert "Available evaluation suites:" in result.output
        assert "harmlessness" in result.output
        assert "robustness" in result.output
        assert "consistency" in result.output

    def test_list_adapters(self, runner):
        """Test list-adapters command."""
        result = runner.invoke(main, ["list-adapters"])

        assert result.exit_code == 0
        assert "Available model adapters:" in result.output
        assert "dummy" in result.output
        assert "openai" in result.output

    @patch("lightweight_evals.cli.EvalRunner")
    @patch("lightweight_evals.cli.DummyAdapter")
    @patch("lightweight_evals.cli.ReportBuilder")
    def test_run_dummy_adapter(self, mock_report_builder, mock_dummy_adapter, mock_eval_runner, runner):
        """Test run command with dummy adapter."""
        # Mock the runner and its methods
        mock_runner_instance = Mock()
        mock_eval_runner.return_value = mock_runner_instance
        mock_runner_instance.get_available_suites.return_value = ["harmlessness", "robustness", "consistency"]

        # Mock run result
        mock_result = Mock()
        mock_result.config.eval_suite = "harmlessness"
        mock_result.timestamp = "20240101_120000"
        mock_result.summary_stats = {"pass_rate": 0.8, "passed_items": 4, "total_items": 5, "average_scores": {"refusal_score": 0.8}}
        mock_runner_instance.run_multiple_suites.return_value = [mock_result]
        mock_runner_instance.save_results.return_value = Path("test.json")

        # Mock report builder
        mock_builder_instance = Mock()
        mock_report_builder.return_value = mock_builder_instance

        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(main, ["run", "--adapter", "dummy", "--suite", "harmlessness", "--out", temp_dir, "--seed", "42"])

        assert result.exit_code == 0
        assert "Running 1 evaluation suite(s)" in result.output
        assert "dummy adapter" in result.output

        # Verify mocks were called
        mock_dummy_adapter.assert_called_once_with(seed=42)
        mock_runner_instance.run_multiple_suites.assert_called_once()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("lightweight_evals.cli.EvalRunner")
    @patch("lightweight_evals.cli.OpenAIAdapter")
    @patch("lightweight_evals.cli.ReportBuilder")
    def test_run_openai_adapter(self, mock_report_builder, mock_openai_adapter, mock_eval_runner, runner):
        """Test run command with OpenAI adapter."""
        # Mock the runner
        mock_runner_instance = Mock()
        mock_eval_runner.return_value = mock_runner_instance
        mock_runner_instance.get_available_suites.return_value = ["harmlessness"]

        # Mock run result
        mock_result = Mock()
        mock_result.config.eval_suite = "harmlessness"
        mock_result.timestamp = "20240101_120000"
        mock_result.summary_stats = {"pass_rate": 1.0, "passed_items": 5, "total_items": 5}
        mock_runner_instance.run_multiple_suites.return_value = [mock_result]
        mock_runner_instance.save_results.return_value = Path("test.json")

        # Mock report builder
        mock_builder_instance = Mock()
        mock_report_builder.return_value = mock_builder_instance

        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(main, ["run", "--adapter", "openai", "--suite", "harmlessness", "--out", temp_dir, "--model", "gpt-3.5-turbo"])

        assert result.exit_code == 0
        mock_openai_adapter.assert_called_once()

    def test_run_missing_openai_key(self, runner):
        """Test run command fails without OpenAI API key."""
        with patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(main, ["run", "--adapter", "openai", "--suite", "harmlessness"])

        assert result.exit_code != 0
        assert "OPENAI_API_KEY not found" in result.output

    def test_run_unknown_suite(self, runner):
        """Test run command with unknown evaluation suite."""
        with patch("lightweight_evals.cli.EvalRunner") as mock_eval_runner:
            mock_runner_instance = Mock()
            mock_eval_runner.return_value = mock_runner_instance
            mock_runner_instance.get_available_suites.return_value = ["harmlessness", "robustness"]

            result = runner.invoke(main, ["run", "--adapter", "dummy", "--suite", "unknown_suite"])

        assert result.exit_code != 0
        assert "Unknown evaluation suite" in result.output

    @patch("lightweight_evals.cli.EvalRunner")
    @patch("lightweight_evals.cli.DummyAdapter")
    @patch("lightweight_evals.cli.ReportBuilder")
    def test_run_all_suites(self, mock_report_builder, mock_dummy_adapter, mock_eval_runner, runner):
        """Test run command with 'all' suites."""
        mock_runner_instance = Mock()
        mock_eval_runner.return_value = mock_runner_instance
        mock_runner_instance.get_available_suites.return_value = ["harmlessness", "robustness"]

        # Mock multiple results
        mock_results = []
        for suite in ["harmlessness", "robustness"]:
            mock_result = Mock()
            mock_result.config.eval_suite = suite
            mock_result.timestamp = "20240101_120000"
            mock_result.summary_stats = {"pass_rate": 0.5, "passed_items": 1, "total_items": 2}
            mock_results.append(mock_result)

        mock_runner_instance.run_multiple_suites.return_value = mock_results
        mock_runner_instance.save_results.return_value = Path("test.json")

        mock_builder_instance = Mock()
        mock_report_builder.return_value = mock_builder_instance

        result = runner.invoke(main, ["run", "--adapter", "dummy", "--suite", "all"])

        assert result.exit_code == 0
        assert "Running 2 evaluation suite(s)" in result.output

    @patch("lightweight_evals.cli.ReportBuilder")
    def test_report_command(self, mock_report_builder, runner):
        """Test report generation command."""
        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            test_data = {
                "run_id": "test123",
                "timestamp": "20240101_120000",
                "config": {
                    "adapter_name": "dummy",
                    "eval_suite": "harmlessness",
                    "seed": 42,
                    "max_tokens": 256,
                    "temperature": 0.2,
                    "output_dir": "/tmp",
                },
                "adapter_info": {"name": "dummy", "version": "1.0"},
                "eval_results": [],
                "summary_stats": {"total_items": 0, "passed_items": 0, "pass_rate": 0.0},
            }
            json.dump(test_data, f)
            temp_path = Path(f.name)

        try:
            # Mock report builder
            mock_builder_instance = Mock()
            mock_report_builder.return_value = mock_builder_instance
            mock_builder_instance.load_result_from_json.return_value = Mock()

            result = runner.invoke(main, ["report", "--json", str(temp_path), "--format", "html"])

            assert result.exit_code == 0
            assert "Generated HTML report" in result.output

            mock_builder_instance.load_result_from_json.assert_called_once_with(temp_path)
            mock_builder_instance.generate_html_report.assert_called_once()

        finally:
            temp_path.unlink()

    def test_report_missing_file(self, runner):
        """Test report command with missing JSON file."""
        result = runner.invoke(main, ["report", "--json", "/nonexistent/file.json"])

        assert result.exit_code != 0
        assert "File not found" in result.output

    @patch("lightweight_evals.cli.ReportBuilder")
    def test_report_both_formats(self, mock_report_builder, runner):
        """Test report generation with both formats."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"test": "data"}, f)
            temp_path = Path(f.name)

        try:
            mock_builder_instance = Mock()
            mock_report_builder.return_value = mock_builder_instance
            mock_builder_instance.load_result_from_json.return_value = Mock()

            result = runner.invoke(main, ["report", "--json", str(temp_path), "--format", "both"])

            assert result.exit_code == 0

            # Should call both generation methods
            mock_builder_instance.generate_html_report.assert_called_once()
            mock_builder_instance.generate_markdown_report.assert_called_once()

        finally:
            temp_path.unlink()
