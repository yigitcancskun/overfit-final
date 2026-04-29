from __future__ import annotations

import pandas as pd

from fusion_pipeline.data_processing import (
    clean_batch,
    clean_keywords_scalar,
    compute_row_fingerprint,
    count_keywords_scalar,
    normalize_missing_series,
    normalize_text_scalar,
    parse_domain_parts_cached,
)


def test_normalize_missing_series_replaces_common_placeholders() -> None:
    series = pd.Series([" nan ", "None", "", "value"])
    normalized = normalize_missing_series(series)
    assert normalized.isna().tolist() == [True, True, True, False]


def test_normalize_text_scalar_and_keyword_dedup() -> None:
    assert normalize_text_scalar("  Hello   world \r\n next  ") == "Hello world\nnext"
    assert clean_keywords_scalar("bot, bot, spam  , spam") == "bot, spam"
    assert count_keywords_scalar("bot, spam") == 2


def test_parse_domain_parts_cached_extracts_registered_domain() -> None:
    raw, registered, subdomain = parse_domain_parts_cached("https://news.example.com/path")
    assert raw == "news.example.com"
    assert registered == "example.com"
    assert subdomain == "news"


def test_compute_row_fingerprint_is_stable_for_identical_rows() -> None:
    frame = pd.DataFrame(
        {
            "original_text": ["same", "same"],
            "author_hash": ["a", "a"],
            "date": ["2024-01-01", "2024-01-01"],
            "url": ["x.com", "x.com"],
        }
    )
    fingerprints = compute_row_fingerprint(frame)
    assert fingerprints.iloc[0] == fingerprints.iloc[1]


def test_clean_batch_with_semantic_disabled(base_config: dict) -> None:
    batch = pd.DataFrame(
        {
            "original_text": ["Bot bot bot #promo !!!"],
            "english_keywords": ["bot, promo, promo"],
            "sentiment": [0.1],
            "main_emotion": ["joy"],
            "primary_theme": [pd.NA],
            "language": ["en"],
            "url": ["https://news.example.com/path"],
            "author_hash": [pd.NA],
            "date": ["2024-01-01T00:00:00Z"],
        }
    )

    cleaned = clean_batch(batch, config=base_config, rare_languages={"xx"})
    row = cleaned.iloc[0]

    assert row["normalized_text"] == "Bot bot bot #promo !!!"
    assert row["english_keywords_clean"] == "bot, promo"
    assert row["keyword_count"] == 2
    assert row["author_type"] == "anonymous"
    assert row["registered_domain"] == "example.com"
    assert row["roberta_score"] == 0.5
    assert row["semantic_model_applied_flag"] == 0
