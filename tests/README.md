# Tests

This directory contains the automated test suite for the Fusion scoring pipeline.

## Scope

- `conftest.py`: shared pytest fixtures and SQLite-backed inference fixture
- `test_config.py`: config loading, override merge, default config writing
- `test_data_processing.py`: normalization, keyword cleaning, domain parsing, batch cleaning
- `test_scoring.py`: score scaling, neutral-weight override, behavioral/final score rules
- `test_inference.py`: single-message input shaping and inference output contract

## Run

From the repository root:

```bash
pytest -q tests
```

## Notes

- Tests disable the semantic adapter where possible to keep runs deterministic and fast.
- The suite uses temporary paths and a fixture SQLite database, so it does not depend on the full production dataset.
