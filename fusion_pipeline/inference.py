from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import compute_message_refs_from_sqlite, get_sqlite_connection
from .data_processing import build_message_features, clean_batch, compute_text_hash_scalar
from .scoring import apply_final_score_weighting, compute_behavioral_score, compute_final_scores, compute_message_scores

def prepare_single_message_input(
    message_text: str,
    language: str | None,
    url: str | None,
    date: str | pd.Timestamp | None,
    author_hash: str | None,
    english_keywords: str | None,
    primary_theme: str | None,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "original_text": message_text,
                "language": language,
                "url": url,
                "date": date,
                "author_hash": author_hash,
                "english_keywords": english_keywords,
                "primary_theme": primary_theme,
                "sentiment": pd.NA,
                "main_emotion": pd.NA,
            }
        ]
    )


def score_single_message(
    message_row: pd.DataFrame,
    result: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    clean_single = clean_batch(message_row, config=config, rare_languages=set())
    clean_single["text_hash"] = clean_single["normalized_text"].map(compute_text_hash_scalar)
    clean_single["date"] = pd.to_datetime(clean_single["date"], utc=True, errors="coerce")
    window_seconds = config["thresholds"]["hard_bot_time_window_sec"]
    date_ns = clean_single["date"].astype("int64", copy=False)
    clean_single["time_window_bucket"] = date_ns.floordiv(window_seconds * 1_000_000_000).where(clean_single["date"].notna(), pd.NA)

    if "clean_df" in result and "normalization_refs" in result:
        combined = pd.concat([result["clean_df"], clean_single], ignore_index=True)
        message_features = build_message_features(combined, config).tail(len(clean_single)).copy()
        message_scores = compute_message_scores(message_features, result["normalization_refs"], config)
        if clean_single["author_type"].iloc[0] == "identified":
            author_hash = clean_single["author_hash"].iloc[0]
            author_scores = result["author_scores"]
            author_row = author_scores.loc[
                author_scores["author_hash"] == author_hash,
                ["author_hash", "author_score", "author_hard_hourly_flag"],
            ]
        else:
            author_row = pd.DataFrame(columns=["author_hash", "author_score", "author_hard_hourly_flag"])
        final = compute_final_scores(clean_single, author_row, message_scores, config)
    else:
        conn = get_sqlite_connection(Path(result["paths"]["sqlite_db"]))
        try:
            text_hash = clean_single["text_hash"].iloc[0]
            time_window_bucket = clean_single["time_window_bucket"].iloc[0]
            cluster_row = conn.execute(
                """
                SELECT
                    COALESCE(tc.repeat_count, 0),
                    COALESCE(tc.unique_author_count, 0),
                    COALESCE(twc.window_count, 0)
                FROM text_clusters tc
                LEFT JOIN text_window_clusters twc
                    ON twc.text_hash = tc.text_hash AND twc.time_window_bucket = ?
                WHERE tc.text_hash = ?
                """,
                [None if pd.isna(time_window_bucket) else int(time_window_bucket), text_hash],
            ).fetchone()
            repeat_count, unique_author_count, window_count = cluster_row if cluster_row else (0, 0, 0)
            clean_single["same_text_repeat_count"] = int(repeat_count)
            clean_single["same_text_unique_author_count"] = int(unique_author_count)
            clean_single["same_text_time_window_count"] = int(window_count)
            clean_single["spam_pattern_flag"] = (
                (clean_single["same_text_repeat_count"] >= config["thresholds"]["spam_repeat_threshold"])
                | (clean_single["same_text_unique_author_count"] >= config["thresholds"]["spam_multi_author_threshold"])
                | (clean_single["same_text_time_window_count"] >= config["thresholds"]["spam_time_cluster_threshold"])
            ).astype("int8")
            clean_single["hard_bot_cluster_flag"] = (
                (clean_single["same_text_repeat_count"] >= config["thresholds"]["hard_bot_repeat_threshold"])
                & (clean_single["same_text_unique_author_count"] >= config["thresholds"]["hard_bot_multi_author_threshold"])
                & (clean_single["same_text_time_window_count"] >= config["thresholds"]["hard_bot_time_cluster_threshold"])
            ).astype("int8")
            if clean_single["author_type"].iloc[0] == "identified":
                author_hash = clean_single["author_hash"].iloc[0]
                author_row = conn.execute(
                    "SELECT author_score, author_hard_hourly_flag FROM author_scores WHERE author_hash = ?",
                    [author_hash],
                ).fetchone()
                clean_single["author_score"] = float(author_row[0]) if author_row else 0.0
                clean_single["author_hard_hourly_flag"] = int(author_row[1]) if author_row else 0
            else:
                clean_single["author_score"] = np.nan
                clean_single["author_hard_hourly_flag"] = 0
        finally:
            conn.close()

        refs = compute_message_refs_from_sqlite(config)
        message_scores = compute_message_scores(clean_single, refs, config)
        final = message_scores.copy()
        final["behavioral_score"] = compute_behavioral_score(final, config)
        final = apply_final_score_weighting(final, config)

    explanation_columns = [
        "author_type",
        "message_score",
        "author_score",
        "behavioral_score",
        "roberta_score",
        "behavioral_confidence_weight",
        "roberta_confidence_weight",
        "behavioral_effective_weight",
        "roberta_effective_weight",
        "final_score_before_rules",
        "final_score",
        "same_text_repeat_component",
        "spam_pattern_component",
        "hashtag_spam_component",
        "token_repetition_component",
        "long_text_component",
        "keyword_signal_component",
        "author_hard_hourly_flag",
        "hard_bot_cluster_flag",
        "hard_same_text_repeat_flag",
    ]
    return final[explanation_columns]
