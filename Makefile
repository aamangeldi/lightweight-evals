.PHONY: install lint test run-local report clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  install     - Install dependencies with uv"
	@echo "  lint        - Run ruff linter and mypy type checker"
	@echo "  test        - Run pytest tests"
	@echo "  run-local   - Run evaluations with dummy adapter"
	@echo "  report      - Rebuild latest HTML report from JSON"
	@echo "  clean       - Clean up generated files"
	@echo "  help        - Show this help message"

# Install dependencies
install:
	uv sync --all-extras

# Lint and type check
lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy src

# Run tests
test:
	uv run pytest tests/ -v

# Run local evaluation with dummy adapter
run-local:
	mkdir -p reports
	uv run lweval run --adapter dummy --suite all --out reports --seed 123

# Rebuild report from latest JSON
report:
	@latest_json=$$(ls -t reports/run_*.json 2>/dev/null | head -1); \
	if [ -n "$$latest_json" ]; then \
		echo "Rebuilding report from $$latest_json"; \
		uv run lweval report --json "$$latest_json" --format both; \
	else \
		echo "No JSON files found in reports/ directory"; \
		echo "Run 'make run-local' first to generate reports"; \
	fi

# Clean generated files
clean:
	rm -rf reports/
	rm -rf .pytest_cache/
	rm -rf src/lightweight_evals/__pycache__/
	rm -rf tests/__pycache__/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +