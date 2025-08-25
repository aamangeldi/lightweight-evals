"""Utility functions for lightweight evals."""

import hashlib
import random
from datetime import datetime
from pathlib import Path


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)


def generate_run_id(
    adapter_name: str, adapter_version: str, eval_suite_name: str, data_path: Path, code_version: str = "0.1.0", timestamp: str | None = None
) -> str:
    """Generate a deterministic run ID based on evaluation parameters."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    # Calculate hash of data file
    data_sha = calculate_file_hash(data_path)

    # Create hash input string
    hash_input = f"{adapter_name}:{adapter_version}:{eval_suite_name}:{data_sha}:{code_version}:{timestamp}"

    # Generate short hash using SHA-256
    run_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:8]

    return run_hash


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()[:16]  # Use first 16 chars


def format_timestamp() -> str:
    """Generate a formatted timestamp for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
