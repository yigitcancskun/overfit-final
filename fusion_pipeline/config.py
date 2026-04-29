from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "input_parquet": "data/datathonFINAL.parquet",
        "clean_output_parquet": "data/datathonFINAL_formula_cleaned.parquet",
        "batch_sqlite_db": "data/fusion_batch_store.sqlite",
        "author_scores_parquet": "data/fusion_author_scores.parquet",
        "scored_messages_parquet": "data/fusion_scored_messages.parquet",
        "manifest_path": "data/fusion_manifest.json",
        "author_feature_stage_parquet": "data/fusion_author_feature_stage.parquet",
    },
    "runtime": {
        "mode": "full",
        "build_mode": "lean",
        "batch_size": 50_000,
        "max_batches": None,
        "sample_n_rows": None,
        "overwrite_outputs": True,
        "enable_progress_logs": True,
        "progress_every_batches": 5,
        "top_n_domain_context": 128,
        "author_batch_size": 5_000,
        "message_batch_size": 100_000,
    },
    "semantic_adapter": {
        "enabled": True,
        "model_name": "junaid1993/distilroberta-bot-detection",
        "supported_languages": ["en"],
        "unsupported_language_score": 0.50,
        "max_length": 512,
        "batch_size": 32,
        "device": "auto",
    },
    "thresholds": {
        "min_text_len": 5,
        "rare_language_threshold": 100,
        "long_text_start": 30,
        "hourly_penalty_start": 10,
        "language_penalty_start": 4,
        "hard_bot_time_window_sec": 300,
        "hard_bot_repeat_threshold": 5,
        "hard_bot_multi_author_threshold": 2,
        "hard_bot_time_cluster_threshold": 3,
        "spam_repeat_threshold": 3,
        "spam_multi_author_threshold": 2,
        "spam_time_cluster_threshold": 3,
    },
    "rules": {
        "long_text_requires_spam": False,
    },
    "dynamic_final_weighting": {
        "enabled": True,
        "min_confidence_weight": 0.20,
        "power": 2.0,
        "sigmoid_steepness": 8.0,
    },
    "neutral_score_policy": {
        "neutral_score": 0.50,
        "epsilon": 1e-6,
    },
    "dominant_signal_policy": {
        "enabled": True,
        "mode": "floor",
        "threshold": 0.68,
        "repeat_hard_threshold": 5,
        "scope": {
            "author": True,
            "message": True,
        },
    },
    "derived_thresholds": {},
    "weights": {
        "behavioral_vs_semantic": {
            "behavioral": 0.60,
            "semantic": 0.40,
        },
        "author_vs_message": {
            "author": 0.70,
            "message": 0.30,
        },
        "author_components": {
            "activity": 0.35,
            "timing": 0.25,
            "repetition": 0.30,
            "diversity": 0.10,
            "metadata": 0.00,
        },
        "message_components": {
            "same_text_repeat": 0.24,
            "spam_pattern": 0.25,
            "hashtag_spam": 0.18,
            "token_repetition": 0.15,
            "long_text": 0.10,
            "keyword_signal": 0.08,
        },
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    if config_path is None:
        return deepcopy(DEFAULT_CONFIG)
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return deep_merge(DEFAULT_CONFIG, payload)


def write_default_config(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
