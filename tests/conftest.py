from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion_pipeline.config import DEFAULT_CONFIG
from fusion_pipeline.data_processing import compute_text_hash_scalar, normalize_text_scalar


@pytest.fixture
def base_config(tmp_path: Path) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    config["semantic_adapter"]["enabled"] = False
    config["paths"]["input_parquet"] = str(tmp_path / "input.parquet")
    config["paths"]["clean_output_parquet"] = str(tmp_path / "clean.parquet")
    config["paths"]["batch_sqlite_db"] = str(tmp_path / "fusion.sqlite")
    config["paths"]["author_scores_parquet"] = str(tmp_path / "author_scores.parquet")
    config["paths"]["scored_messages_parquet"] = str(tmp_path / "scored_messages.parquet")
    config["paths"]["manifest_path"] = str(tmp_path / "manifest.json")
    config["paths"]["author_feature_stage_parquet"] = str(tmp_path / "author_stage.parquet")
    config["runtime"]["enable_progress_logs"] = False
    return config


@pytest.fixture
def sqlite_score_fixture(base_config: dict) -> tuple[dict, dict]:
    from fusion_pipeline.artifacts import get_sqlite_connection

    db_path = Path(base_config["paths"]["batch_sqlite_db"])
    normalized_text = normalize_text_scalar("Bot bot bot #promo !!!")
    assert normalized_text is not pd.NA
    text_hash = compute_text_hash_scalar(normalized_text)
    date = pd.Timestamp("2024-01-01T00:00:00Z")
    bucket = int(date.value // (base_config["thresholds"]["hard_bot_time_window_sec"] * 1_000_000_000))

    conn = get_sqlite_connection(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE cleaned_messages (
                text_length_chars INTEGER,
                keyword_count INTEGER,
                hashtag_count INTEGER,
                hashtag_density_chars REAL,
                hashtag_density_tokens REAL,
                max_token_frequency INTEGER,
                max_token_ratio REAL,
                repeated_token_count_over_2 INTEGER
            );
            CREATE TABLE text_clusters (
                text_hash TEXT,
                repeat_count INTEGER,
                unique_author_count INTEGER
            );
            CREATE TABLE text_window_clusters (
                text_hash TEXT,
                time_window_bucket INTEGER,
                window_count INTEGER
            );
            CREATE TABLE author_scores (
                author_hash TEXT PRIMARY KEY,
                author_score REAL,
                author_hard_hourly_flag INTEGER
            );
            """
        )
        conn.execute(
            """
            INSERT INTO cleaned_messages (
                text_length_chars,
                keyword_count,
                hashtag_count,
                hashtag_density_chars,
                hashtag_density_tokens,
                max_token_frequency,
                max_token_ratio,
                repeated_token_count_over_2
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (19, 2, 1, 0.05, 0.25, 3, 0.75, 1),
        )
        conn.execute(
            "INSERT INTO text_clusters(text_hash, repeat_count, unique_author_count) VALUES (?, ?, ?)",
            (text_hash, 4, 2),
        )
        conn.execute(
            "INSERT INTO text_window_clusters(text_hash, time_window_bucket, window_count) VALUES (?, ?, ?)",
            (text_hash, bucket, 3),
        )
        conn.execute(
            "INSERT INTO author_scores(author_hash, author_score, author_hard_hourly_flag) VALUES (?, ?, ?)",
            ("author-1", 0.82, 0),
        )
        conn.commit()
    finally:
        conn.close()

    runtime_result = {"paths": {"sqlite_db": str(db_path)}}
    return base_config, runtime_result
