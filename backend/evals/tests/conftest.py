"""Evals tests are isolated from the regression suite.

`backend/pyproject.toml` has `testpaths = ["tests"]`, so these are picked up
only when invoked explicitly: `uv run pytest evals/tests/`.
"""
