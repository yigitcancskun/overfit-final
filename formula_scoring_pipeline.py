from __future__ import annotations

import gc
import hashlib
import math
import os
import re
import sqlite3
import time
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None

try:
    import tldextract  # type: ignore
except ImportError:
    tldextract = None


MISSING_PLACEHOLDERS = {"", "nan", "null", "none", "n/a"}
BLANK_SENSITIVE_COLUMNS = ["author_hash", "original_text", "english_keywords", "primary_theme"]
COMMON_MULTI_PART_SUFFIXES = {
    "co.uk",
    "org.uk",
    "gov.uk",
    "ac.uk",
    "com.au",
    "net.au",
    "org.au",
    "co.jp",
    "com.br",
    "com.tr",
    "com.mx",
    "co.in",
    "com.cn",
}


def get_available_memory_bytes() -> int:
    if psutil is not None:
        return int(psutil.virtual_memory().available)
    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_AVPHYS_PAGES" in os.sysconf_names:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return page_size * available_pages
    return 4 * 1024**3


def estimate_batch_size(
    min_rows: int = 40_000,
    max_rows: int = 150_000,
    estimated_bytes_per_row: int = 2_500,
    target_memory_fraction: float = 0.03,
) -> int:
    available = get_available_memory_bytes()
    target_bytes = max(int(available * target_memory_fraction), estimated_bytes_per_row * min_rows)
    estimated_rows = target_bytes // estimated_bytes_per_row
    return max(min_rows, min(max_rows, estimated_rows))


def normalize_missing_scalar(value: Any):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in MISSING_PLACEHOLDERS:
            return pd.NA
        return stripped
    return value


def normalize_missing_series(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    return s.mask(s.str.lower().isin(MISSING_PLACEHOLDERS), pd.NA)


def normalize_text_scalar(value: Any):
    value = normalize_missing_scalar(value)
    if pd.isna(value):
        return pd.NA
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = text.strip()
    return text if text else pd.NA


def normalize_text_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_text_scalar)


def clean_keywords_scalar(value: Any):
    value = normalize_missing_scalar(value)
    if pd.isna(value):
        return pd.NA
    parts: list[str] = []
    seen: set[str] = set()
    for item in str(value).split(","):
        token = " ".join(item.split()).strip()
        if not token:
            continue
        if token not in seen:
            seen.add(token)
            parts.append(token)
    return ", ".join(parts) if parts else pd.NA


def clean_keywords_series(series: pd.Series) -> pd.Series:
    return series.map(clean_keywords_scalar)


def count_keywords_scalar(value: Any) -> int:
    if pd.isna(value):
        return 0
    return len([token for token in str(value).split(",") if token.strip()])


TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)
HASHTAG_PATTERN = re.compile(r"(?<!\w)#[\w_]+", flags=re.UNICODE)


def extract_text_stats_scalar(value: Any) -> dict[str, float]:
    if pd.isna(value):
        return {
            "token_count": 0.0,
            "unique_token_count": 0.0,
            "max_token_frequency": 0.0,
            "max_token_ratio": 0.0,
            "repeated_token_count_over_2": 0.0,
            "hashtag_count": 0.0,
            "hashtag_density_chars": 0.0,
            "hashtag_density_tokens": 0.0,
        }

    text = str(value)
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
    token_count = len(tokens)
    if token_count:
        counts = pd.Series(tokens).value_counts()
        max_token_frequency = float(counts.iloc[0])
        unique_token_count = float(len(counts))
        max_token_ratio = max_token_frequency / token_count
        repeated_token_count_over_2 = float((counts >= 3).sum())
    else:
        max_token_frequency = 0.0
        unique_token_count = 0.0
        max_token_ratio = 0.0
        repeated_token_count_over_2 = 0.0

    hashtag_count = float(len(HASHTAG_PATTERN.findall(text)))
    text_len = max(len(text), 1)
    hashtag_density_chars = hashtag_count / text_len
    hashtag_density_tokens = hashtag_count / max(token_count, 1)

    return {
        "token_count": float(token_count),
        "unique_token_count": unique_token_count,
        "max_token_frequency": max_token_frequency,
        "max_token_ratio": max_token_ratio,
        "repeated_token_count_over_2": repeated_token_count_over_2,
        "hashtag_count": hashtag_count,
        "hashtag_density_chars": hashtag_density_chars,
        "hashtag_density_tokens": hashtag_density_tokens,
    }


def extract_text_stats_series(series: pd.Series) -> pd.DataFrame:
    stats = series.map(extract_text_stats_scalar)
    return pd.DataFrame(stats.tolist(), index=series.index)


@lru_cache(maxsize=50_000)
def parse_domain_parts_cached(raw_value: str) -> tuple[str | pd.NA, str | pd.NA, str | pd.NA]:
    value = normalize_missing_scalar(raw_value)
    if pd.isna(value):
        return (pd.NA, pd.NA, pd.NA)

    candidate = str(value).strip()
    parsed = urlparse(candidate if "://" in candidate else f"https://{candidate}")
    host = (parsed.hostname or candidate).lower().strip(".")
    if not host:
        return (pd.NA, pd.NA, pd.NA)

    if tldextract is not None:
        ext = tldextract.extract(host)
        registered_domain = ".".join(part for part in [ext.domain, ext.suffix] if part) or host
        subdomain = ext.subdomain or pd.NA
        return (host, registered_domain, subdomain)

    parts = [part for part in host.split(".") if part]
    if len(parts) < 2:
        return (host, host, pd.NA)

    tail_two = ".".join(parts[-2:])
    tail_three = ".".join(parts[-3:]) if len(parts) >= 3 else ""
    if tail_two in COMMON_MULTI_PART_SUFFIXES and len(parts) >= 3:
        return (host, tail_three, ".".join(parts[:-3]) or pd.NA)
    return (host, tail_two, ".".join(parts[:-2]) or pd.NA)


def extract_domain_columns(series: pd.Series) -> pd.DataFrame:
    parts = series.astype("string").map(parse_domain_parts_cached)
    return pd.DataFrame(parts.tolist(), columns=["raw_url_domain", "registered_domain", "subdomain"], index=series.index)


def _get_first_series(frame: pd.DataFrame, column: str) -> pd.Series:
    value = frame[column]
    if isinstance(value, pd.DataFrame):
        return value.iloc[:, 0]
    return value


def compute_row_fingerprint(frame: pd.DataFrame) -> pd.Series:
    original_text = _get_first_series(frame, "original_text").astype("string").fillna("<NA>")
    author_hash = _get_first_series(frame, "author_hash").astype("string").fillna("<NA>")
    date = _get_first_series(frame, "date").astype("string").fillna("<NA>")
    url = _get_first_series(frame, "url").astype("string").fillna("<NA>")
    row_key = original_text + "\x1f" + author_hash + "\x1f" + date + "\x1f" + url
    return pd.util.hash_pandas_object(row_key, index=False)


def iter_pandas_batches(path: Path, batch_size: int):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield batch.to_pandas(types_mapper=pd.ArrowDtype)


def collect_language_counts(path: Path, batch_size: int) -> pd.Series:
    language_counts = pd.Series(dtype="int64")
    for batch_df in iter_pandas_batches(path, batch_size=batch_size):
        batch_lang = normalize_missing_series(batch_df["language"]).value_counts(dropna=False)
        language_counts = language_counts.add(batch_lang, fill_value=0)
        del batch_df, batch_lang
        gc.collect()
    return language_counts.sort_values(ascending=False).astype("int64")


def clean_batch(batch_df: pd.DataFrame, config: dict[str, Any], rare_languages: set[str]) -> pd.DataFrame:
    batch_df = batch_df.copy()
    batch_df.columns = [str(col) for col in batch_df.columns]

    for column in BLANK_SENSITIVE_COLUMNS:
        batch_df[column] = normalize_missing_series(batch_df[column])

    for column in ["language", "url", "main_emotion"]:
        if column in batch_df.columns:
            batch_df[column] = normalize_missing_series(batch_df[column])

    batch_df["date"] = pd.to_datetime(batch_df["date"], utc=True, errors="coerce")
    batch_df["normalized_text"] = normalize_text_series(batch_df["original_text"])
    batch_df["english_keywords_clean"] = clean_keywords_series(batch_df["english_keywords"])
    batch_df["primary_theme"] = batch_df["primary_theme"].fillna("unknown_theme")

    batch_df["author_type"] = np.where(batch_df["author_hash"].isna(), "anonymous", "identified")
    batch_df["author_hash_missing_flag"] = batch_df["author_hash"].isna().astype("int8")
    batch_df["is_empty_text"] = batch_df["normalized_text"].isna().astype("int8")
    batch_df["text_length_chars"] = batch_df["normalized_text"].str.len().fillna(0).astype("int32")
    batch_df["is_short_text"] = batch_df["text_length_chars"].lt(config["thresholds"]["min_text_len"]).astype("int8")
    batch_df["is_long_text_flag"] = batch_df["text_length_chars"].gt(config["thresholds"]["long_text_start"]).astype("int8")
    batch_df["is_rare_language"] = batch_df["language"].isin(rare_languages).astype("int8")
    batch_df["keyword_count"] = batch_df["english_keywords_clean"].map(count_keywords_scalar).astype("int16")
    text_stats = extract_text_stats_series(batch_df["normalized_text"])
    for column in text_stats.columns:
        batch_df[column] = text_stats[column]
    batch_df["token_count"] = batch_df["token_count"].astype("int32")
    batch_df["unique_token_count"] = batch_df["unique_token_count"].astype("int32")
    batch_df["max_token_frequency"] = batch_df["max_token_frequency"].astype("int32")
    batch_df["repeated_token_count_over_2"] = batch_df["repeated_token_count_over_2"].astype("int32")
    batch_df["hashtag_count"] = batch_df["hashtag_count"].astype("int16")

    domain_parts = extract_domain_columns(batch_df["url"])
    batch_df = batch_df.drop(columns=["raw_url_domain", "registered_domain", "subdomain"], errors="ignore")
    batch_df = pd.concat([batch_df, domain_parts], axis=1)
    return batch_df


def clean_dataset(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    input_path = Path(config["paths"]["input_parquet"])
    output_path = Path(config["paths"]["clean_output_parquet"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = config["runtime"]["batch_size"] or estimate_batch_size()
    rare_threshold = config["thresholds"]["rare_language_threshold"]
    language_counts = collect_language_counts(input_path, batch_size=batch_size)
    rare_languages = set(language_counts[language_counts < rare_threshold].index.astype("string").tolist())

    if output_path.exists() and config["runtime"]["overwrite_outputs"]:
        output_path.unlink()

    summary = {
        "raw_rows": 0,
        "clean_rows": 0,
        "technical_duplicates_removed": 0,
        "missing_author_rows": 0,
        "empty_text_rows": 0,
        "short_text_rows": 0,
        "bad_date_rows": 0,
        "rare_language_rows": 0,
        "batch_size": batch_size,
    }

    seen_fingerprints: set[int] = set()
    writer = None
    cleaned_frames: list[pd.DataFrame] = []
    max_batches = config["runtime"].get("max_batches")
    max_rows = config["runtime"].get("sample_n_rows")

    start = time.time()
    try:
        for batch_id, batch_df in enumerate(iter_pandas_batches(input_path, batch_size=batch_size), start=1):
            if max_batches and batch_id > max_batches:
                break
            summary["raw_rows"] += len(batch_df)

            cleaned_batch = clean_batch(batch_df, config=config, rare_languages=rare_languages)
            fingerprints = compute_row_fingerprint(cleaned_batch)
            is_duplicate = fingerprints.isin(seen_fingerprints)
            seen_fingerprints.update(fingerprints.loc[~is_duplicate].tolist())
            cleaned_batch = cleaned_batch.loc[~is_duplicate].copy()
            cleaned_batch["technical_duplicate_flag"] = 0

            summary["technical_duplicates_removed"] += int(is_duplicate.sum())
            summary["clean_rows"] += len(cleaned_batch)
            summary["missing_author_rows"] += int(cleaned_batch["author_hash_missing_flag"].sum())
            summary["empty_text_rows"] += int(cleaned_batch["is_empty_text"].sum())
            summary["short_text_rows"] += int(cleaned_batch["is_short_text"].sum())
            summary["bad_date_rows"] += int(cleaned_batch["date"].isna().sum())
            summary["rare_language_rows"] += int(cleaned_batch["is_rare_language"].sum())

            cleaned_frames.append(cleaned_batch)
            table = pa.Table.from_pandas(cleaned_batch, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)

            current_rows = sum(len(frame) for frame in cleaned_frames)
            if max_rows and current_rows >= max_rows:
                break

            del batch_df, fingerprints, is_duplicate, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()

    clean_df = pd.concat(cleaned_frames, ignore_index=True) if cleaned_frames else pd.DataFrame()
    if max_rows and len(clean_df) > max_rows:
        clean_df = clean_df.head(max_rows).copy()

    summary["elapsed_min"] = round((time.time() - start) / 60, 2)
    summary["rare_languages"] = sorted(rare_languages)
    return clean_df, summary


def compute_top_domain_coverage(clean_df: pd.DataFrame, top_n: int = 128) -> dict[str, Any]:
    counts = clean_df["registered_domain"].value_counts(dropna=True)
    if counts.empty:
        return {"unique_domains": 0, "coverage_share": 0.0, "top_domains": counts}
    coverage = float(counts.head(top_n).sum() / counts.sum())
    return {"unique_domains": int(len(counts)), "coverage_share": coverage, "top_domains": counts}


def build_author_features(clean_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    author_df = clean_df[clean_df["author_type"] == "identified"].dropna(subset=["author_hash", "date"]).copy()
    if author_df.empty:
        return pd.DataFrame()

    author_df["date"] = pd.to_datetime(author_df["date"], utc=True, errors="coerce")
    author_df = author_df.dropna(subset=["date"]).copy()
    author_df["day_bucket"] = author_df["date"].dt.floor("D")
    author_df["hour_bucket"] = author_df["date"].dt.floor("h")

    base = author_df.groupby("author_hash").agg(
        total_posts=("author_hash", "size"),
        active_days=("day_bucket", "nunique"),
        active_hours=("hour_bucket", "nunique"),
        language_nunique=("language", lambda s: s.nunique(dropna=True)),
        theme_nunique=("primary_theme", lambda s: s.nunique(dropna=True)),
        sentiment_std=("sentiment", lambda s: pd.to_numeric(s, errors="coerce").std()),
    )
    base["posts_per_day"] = base["total_posts"] / base["active_days"].clip(lower=1)
    base["posts_per_active_hour"] = base["total_posts"] / base["active_hours"].clip(lower=1)

    hourly_counts = author_df.groupby(["author_hash", "hour_bucket"]).size().rename("hour_post_count")
    base["max_posts_one_hour"] = hourly_counts.groupby("author_hash").max()
    base["hourly_post_distribution"] = hourly_counts.groupby("author_hash").apply(lambda s: s.tolist())

    author_df = author_df.sort_values(["author_hash", "date"]).copy()
    author_df["interpost_sec"] = author_df.groupby("author_hash")["date"].diff().dt.total_seconds()

    timing = author_df.groupby("author_hash").agg(
        mean_interpost_sec=("interpost_sec", "mean"),
        median_interpost_sec=("interpost_sec", "median"),
        p10_interpost_sec=("interpost_sec", lambda s: s.quantile(0.10)),
        std_interpost_sec=("interpost_sec", "std"),
    )
    base = base.join(timing)
    base["interval_cv"] = base["std_interpost_sec"] / base["mean_interpost_sec"].replace(0, np.nan)

    same_text_author_counts = (
        author_df.groupby(["author_hash", "normalized_text"]).size().rename("same_text_author_count").reset_index()
    )
    repeated_text = same_text_author_counts["same_text_author_count"].gt(1)
    repeat_ratio = (
        same_text_author_counts.loc[repeated_text]
        .groupby("author_hash")["same_text_author_count"]
        .sum()
        .div(base["total_posts"])
    )
    base["same_text_repeat_ratio"] = repeat_ratio.fillna(0.0)
    base["same_text_repeat_max"] = same_text_author_counts.groupby("author_hash")["same_text_author_count"].max().fillna(0.0)

    global_text_authors = (
        author_df.groupby("normalized_text")["author_hash"].nunique(dropna=True).rename("same_text_unique_author_count")
    )
    author_df = author_df.join(global_text_authors, on="normalized_text")
    multi_author_ratio = (
        author_df.groupby("author_hash")["same_text_unique_author_count"]
        .apply(lambda s: float(s.gt(1).mean()) if len(s) else 0.0)
        .fillna(0.0)
    )
    base["multi_author_repeat_ratio"] = multi_author_ratio
    base["sentiment_std"] = base["sentiment_std"].fillna(0.0)
    base["mean_interpost_sec"] = base["mean_interpost_sec"].fillna(np.inf)
    base["median_interpost_sec"] = base["median_interpost_sec"].fillna(np.inf)
    base["p10_interpost_sec"] = base["p10_interpost_sec"].fillna(np.inf)
    base["interval_cv"] = base["interval_cv"].replace([np.inf, -np.inf], np.nan).fillna(np.inf)
    return base.reset_index()


def build_message_features(clean_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    msg_df = clean_df.copy()
    msg_df["date"] = pd.to_datetime(msg_df["date"], utc=True, errors="coerce")
    msg_df["same_text_repeat_count"] = msg_df.groupby("normalized_text")["normalized_text"].transform("size").fillna(0).astype("int32")
    unique_authors = (
        msg_df.dropna(subset=["author_hash"])
        .groupby("normalized_text")["author_hash"]
        .nunique(dropna=True)
        .rename("same_text_unique_author_count")
    )
    msg_df = msg_df.join(unique_authors, on="normalized_text")
    msg_df["same_text_unique_author_count"] = msg_df["same_text_unique_author_count"].fillna(0).astype("int32")

    window_seconds = config["thresholds"]["hard_bot_time_window_sec"]
    date_ns = msg_df["date"].astype("int64", copy=False)
    msg_df["time_window_bucket"] = date_ns.floordiv(window_seconds * 1_000_000_000).where(msg_df["date"].notna(), pd.NA)
    msg_df["same_text_time_window_count"] = (
        msg_df.groupby(["normalized_text", "time_window_bucket"])["normalized_text"]
        .transform("size")
        .fillna(0)
        .astype("int32")
    )

    spam_thresholds = config["thresholds"]
    msg_df["spam_pattern_flag"] = (
        (msg_df["same_text_repeat_count"] >= spam_thresholds["spam_repeat_threshold"])
        | (msg_df["same_text_time_window_count"] >= spam_thresholds["spam_time_cluster_threshold"])
        | (
            msg_df["same_text_unique_author_count"]
            >= spam_thresholds["spam_multi_author_threshold"]
        )
    ).astype("int8")

    msg_df["hard_bot_cluster_flag"] = (
        (msg_df["same_text_repeat_count"] >= spam_thresholds["hard_bot_repeat_threshold"])
        & (msg_df["same_text_unique_author_count"] >= spam_thresholds["hard_bot_multi_author_threshold"])
        & (msg_df["same_text_time_window_count"] >= spam_thresholds["hard_bot_time_cluster_threshold"])
    ).astype("int8")
    return msg_df


def _series_quantiles(series: pd.Series, low_q: float, high_q: float) -> tuple[float, float]:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return (0.0, 1.0)
    lower = float(clean.quantile(low_q))
    upper = float(clean.quantile(high_q))
    if math.isclose(lower, upper):
        upper = lower + 1.0
    return lower, upper


def fit_normalization_reference(author_features: pd.DataFrame, message_features: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    refs: dict[str, Any] = {"author": {}, "message": {}}

    for column in [
        "posts_per_day",
        "posts_per_active_hour",
        "theme_nunique",
        "sentiment_std",
        "same_text_repeat_ratio",
        "same_text_repeat_max",
        "multi_author_repeat_ratio",
    ]:
        refs["author"][column] = _series_quantiles(author_features[column], 0.05, 0.99) if column in author_features else (0.0, 1.0)

    for column in ["mean_interpost_sec", "median_interpost_sec", "p10_interpost_sec", "interval_cv"]:
        refs["author"][column] = _series_quantiles(author_features[column], 0.01, 0.95) if column in author_features else (1.0, 2.0)

    if "max_posts_one_hour" in author_features and not author_features.empty:
        refs["author"]["max_posts_one_hour_max"] = float(pd.to_numeric(author_features["max_posts_one_hour"], errors="coerce").fillna(0).max())
    else:
        refs["author"]["max_posts_one_hour_max"] = float(config["thresholds"]["hourly_penalty_start"] + 1)

    if "language_nunique" in author_features and not author_features.empty:
        refs["author"]["language_nunique_max"] = float(pd.to_numeric(author_features["language_nunique"], errors="coerce").fillna(0).max())
    else:
        refs["author"]["language_nunique_max"] = float(config["thresholds"]["language_penalty_start"] + 1)

    for column in [
        "text_length_chars",
        "same_text_repeat_count",
        "same_text_unique_author_count",
        "same_text_time_window_count",
        "keyword_count",
        "hashtag_count",
        "hashtag_density_chars",
        "hashtag_density_tokens",
        "max_token_frequency",
        "max_token_ratio",
        "repeated_token_count_over_2",
    ]:
        refs["message"][column] = _series_quantiles(message_features[column], 0.05, 0.99) if column in message_features else (0.0, 1.0)

    if "text_length_chars" in message_features and not message_features.empty:
        long_excess = (pd.to_numeric(message_features["text_length_chars"], errors="coerce") - config["thresholds"]["long_text_start"]).clip(lower=0)
        refs["message"]["long_text_excess_max"] = float(long_excess.max()) if not long_excess.empty else 1.0
    else:
        refs["message"]["long_text_excess_max"] = 1.0
    return refs


def bounded_scale(value: pd.Series | float, lower: float, upper: float) -> pd.Series | float:
    if upper <= lower:
        upper = lower + 1.0
    result = (value - lower) / (upper - lower)
    return result.clip(0, 1) if hasattr(result, "clip") else float(max(0.0, min(1.0, result)))


def inverse_bounded_scale(value: pd.Series | float, lower: float, upper: float) -> pd.Series | float:
    if upper <= lower:
        upper = lower + 1.0
    result = (upper - value) / (upper - lower)
    return result.clip(0, 1) if hasattr(result, "clip") else float(max(0.0, min(1.0, result)))


def log_penalty(series: pd.Series, start_threshold: float, max_reference: float) -> pd.Series:
    excess = (series - start_threshold).clip(lower=0)
    denom = math.log1p(max(max_reference - start_threshold, 1.0))
    if denom <= 0:
        denom = 1.0
    return (np.log1p(excess) / denom).clip(0, 1)


def detect_hourly_knee_threshold(values: pd.Series, minimum_floor: int) -> int:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    clean = clean[clean > 0].sort_values().reset_index(drop=True)
    if clean.empty:
        return minimum_floor
    if len(clean) == 1:
        return max(int(clean.iloc[0]), minimum_floor)

    tail = clean[clean >= minimum_floor]
    if tail.empty:
        return minimum_floor
    if len(tail) < 5:
        return max(int(tail.max()), minimum_floor)

    tail_unique = np.sort(tail.unique())[::-1].astype(float)
    x = np.linspace(0.0, 1.0, len(tail_unique))
    y = np.log1p(tail_unique)
    y = (y - y.min()) / max(y.max() - y.min(), 1e-9)
    baseline = 1.0 - x
    distance = y - baseline
    knee_idx = int(np.argmax(distance))
    knee_value = int(tail_unique[knee_idx])
    tail_guard = int(tail.quantile(0.99))
    return max(knee_value, tail_guard, minimum_floor)


def compute_author_scores(author_features: pd.DataFrame, refs: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    if author_features.empty:
        return author_features.copy()

    scores = author_features.copy()
    weights = config["weights"]["author_components"]
    thresholds = config["thresholds"]
    author_refs = refs["author"]
    derived_thresholds = config.setdefault("derived_thresholds", {})

    scores["activity_posts_per_day_risk"] = bounded_scale(scores["posts_per_day"], *author_refs["posts_per_day"])
    scores["activity_posts_per_hour_risk"] = bounded_scale(scores["posts_per_active_hour"], *author_refs["posts_per_active_hour"])
    hourly_knee = detect_hourly_knee_threshold(scores["max_posts_one_hour"], thresholds["hourly_penalty_start"])
    derived_thresholds["hourly_hard_knee"] = int(hourly_knee)
    scores["author_hard_hourly_flag"] = scores["max_posts_one_hour"].ge(hourly_knee).astype("int8")
    scores["activity_hourly_penalty_risk"] = log_penalty(
        scores["max_posts_one_hour"],
        thresholds["hourly_penalty_start"],
        author_refs["max_posts_one_hour_max"],
    )
    scores["activity_risk"] = (
        0.20 * scores["activity_posts_per_day_risk"]
        + 0.25 * scores["activity_posts_per_hour_risk"]
        + 0.55 * scores["activity_hourly_penalty_risk"]
    )

    scores["timing_mean_gap_risk"] = inverse_bounded_scale(scores["mean_interpost_sec"], *author_refs["mean_interpost_sec"])
    scores["timing_median_gap_risk"] = inverse_bounded_scale(scores["median_interpost_sec"], *author_refs["median_interpost_sec"])
    scores["timing_p10_gap_risk"] = inverse_bounded_scale(scores["p10_interpost_sec"], *author_refs["p10_interpost_sec"])
    scores["timing_cv_risk"] = inverse_bounded_scale(scores["interval_cv"], *author_refs["interval_cv"])
    scores["timing_risk"] = (
        0.15 * scores["timing_mean_gap_risk"]
        + 0.25 * scores["timing_median_gap_risk"]
        + 0.35 * scores["timing_p10_gap_risk"]
        + 0.25 * scores["timing_cv_risk"]
    )

    scores["repetition_same_text_ratio_risk"] = bounded_scale(scores["same_text_repeat_ratio"], *author_refs["same_text_repeat_ratio"])
    scores["repetition_same_text_max_risk"] = bounded_scale(scores["same_text_repeat_max"], *author_refs["same_text_repeat_max"])
    scores["repetition_multi_author_risk"] = bounded_scale(scores["multi_author_repeat_ratio"], *author_refs["multi_author_repeat_ratio"])
    scores["repetition_risk"] = (
        0.30 * scores["repetition_same_text_ratio_risk"]
        + 0.25 * scores["repetition_same_text_max_risk"]
        + 0.45 * scores["repetition_multi_author_risk"]
    )

    language_excess = (scores["language_nunique"] - thresholds["language_penalty_start"]).clip(lower=0)
    language_max_excess = max(author_refs["language_nunique_max"] - thresholds["language_penalty_start"], 1.0)
    scores["diversity_language_risk"] = bounded_scale(language_excess, 0.0, language_max_excess)
    scores["diversity_theme_risk"] = bounded_scale(scores["theme_nunique"], *author_refs["theme_nunique"])
    scores["diversity_sentiment_risk"] = bounded_scale(scores["sentiment_std"], *author_refs["sentiment_std"])
    scores["diversity_risk"] = (
        0.40 * scores["diversity_language_risk"]
        + 0.30 * scores["diversity_theme_risk"]
        + 0.30 * scores["diversity_sentiment_risk"]
    )

    scores["metadata_risk"] = 0.0
    scores["author_score"] = (
        weights["activity"] * scores["activity_risk"]
        + weights["timing"] * scores["timing_risk"]
        + weights["repetition"] * scores["repetition_risk"]
        + weights["diversity"] * scores["diversity_risk"]
        + weights["metadata"] * scores["metadata_risk"]
    ).clip(0, 1)
    return scores


def compute_message_scores(message_features: pd.DataFrame, refs: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    if message_features.empty:
        return message_features.copy()

    scores = message_features.copy()
    weights = config["weights"]["message_components"]
    thresholds = config["thresholds"]
    msg_refs = refs["message"]

    scores["same_text_repeat_risk"] = bounded_scale(scores["same_text_repeat_count"], *msg_refs["same_text_repeat_count"])
    scores["same_text_multi_author_risk"] = bounded_scale(scores["same_text_unique_author_count"], *msg_refs["same_text_unique_author_count"])
    scores["same_text_time_window_risk"] = bounded_scale(scores["same_text_time_window_count"], *msg_refs["same_text_time_window_count"])
    scores["same_text_repeat_component"] = (
        0.45 * scores["same_text_repeat_risk"]
        + 0.35 * scores["same_text_multi_author_risk"]
        + 0.20 * scores["same_text_time_window_risk"]
    )

    scores["spam_pattern_component"] = scores["spam_pattern_flag"].astype(float)

    scores["hashtag_count_risk"] = bounded_scale(scores["hashtag_count"], *msg_refs["hashtag_count"])
    scores["hashtag_density_chars_risk"] = bounded_scale(scores["hashtag_density_chars"], *msg_refs["hashtag_density_chars"])
    scores["hashtag_density_tokens_risk"] = bounded_scale(scores["hashtag_density_tokens"], *msg_refs["hashtag_density_tokens"])
    scores["hashtag_spam_component"] = (
        0.40 * scores["hashtag_count_risk"]
        + 0.30 * scores["hashtag_density_chars_risk"]
        + 0.30 * scores["hashtag_density_tokens_risk"]
    )

    scores["token_frequency_risk"] = bounded_scale(scores["max_token_frequency"], *msg_refs["max_token_frequency"])
    scores["token_ratio_risk"] = bounded_scale(scores["max_token_ratio"], *msg_refs["max_token_ratio"])
    scores["token_repeat_cluster_risk"] = bounded_scale(
        scores["repeated_token_count_over_2"], *msg_refs["repeated_token_count_over_2"]
    )
    scores["token_repetition_component"] = (
        0.35 * scores["token_frequency_risk"]
        + 0.45 * scores["token_ratio_risk"]
        + 0.20 * scores["token_repeat_cluster_risk"]
    )

    long_excess = (scores["text_length_chars"] - thresholds["long_text_start"]).clip(lower=0)
    scores["long_text_component"] = log_penalty(
        long_excess + thresholds["long_text_start"],
        thresholds["long_text_start"],
        thresholds["long_text_start"] + max(msg_refs["long_text_excess_max"], 1.0),
    )
    if config["rules"]["long_text_requires_spam"]:
        scores["long_text_component"] = scores["long_text_component"] * scores["spam_pattern_component"]

    scores["keyword_signal_component"] = bounded_scale(scores["keyword_count"], *msg_refs["keyword_count"])
    scores["message_score"] = (
        weights["same_text_repeat"] * scores["same_text_repeat_component"]
        + weights["spam_pattern"] * scores["spam_pattern_component"]
        + weights["hashtag_spam"] * scores["hashtag_spam_component"]
        + weights["token_repetition"] * scores["token_repetition_component"]
        + weights["long_text"] * scores["long_text_component"]
        + weights["keyword_signal"] * scores["keyword_signal_component"]
    ).clip(0, 1)
    return scores


def compute_final_scores(
    clean_df: pd.DataFrame,
    author_scores: pd.DataFrame,
    message_scores: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    df = message_scores.copy()
    merge_columns = ["author_hash", "author_score", "author_hard_hourly_flag"]
    if not author_scores.empty:
        df = df.merge(author_scores[merge_columns], on="author_hash", how="left")
    else:
        df["author_score"] = np.nan
        df["author_hard_hourly_flag"] = 0

    author_weight = config["weights"]["author_vs_message"]["author"]
    message_weight = config["weights"]["author_vs_message"]["message"]
    df["behavioral_score"] = np.where(
        df["author_type"].eq("anonymous"),
        df["message_score"],
        (author_weight * df["author_score"].fillna(0.0) + message_weight * df["message_score"]).clip(0, 1),
    )
    df["final_score"] = np.where(
        df["author_hard_hourly_flag"].fillna(0).eq(1) | df["hard_bot_cluster_flag"].eq(1),
        1.0,
        df["behavioral_score"],
    )
    return df


def run_formula_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    clean_df, summary = clean_dataset(config)
    domain_info = compute_top_domain_coverage(clean_df, top_n=config["runtime"]["top_n_domain_context"])
    author_features = build_author_features(clean_df, config)
    message_features = build_message_features(clean_df, config)
    refs = fit_normalization_reference(author_features, message_features, config)
    author_scores = compute_author_scores(author_features, refs, config)
    message_scores = compute_message_scores(message_features, refs, config)
    scored_df = compute_final_scores(clean_df, author_scores, message_scores, config)
    return {
        "clean_df": clean_df,
        "summary": summary,
        "domain_info": domain_info,
        "author_features": author_features,
        "message_features": message_features,
        "normalization_refs": refs,
        "author_scores": author_scores,
        "message_scores": message_scores,
        "scored_df": scored_df,
    }


def build_qa_tables(result: dict[str, Any], config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    scored_df = result["scored_df"]
    author_scores = result["author_scores"]
    domain_counts = result["domain_info"]["top_domains"]

    tables = {
        "summary": pd.Series(result["summary"]).to_frame("value"),
        "top_domains": domain_counts.head(20).rename_axis("registered_domain").reset_index(name="rows"),
        "top128_coverage": pd.DataFrame(
            {
                "metric": ["unique_domains", "top128_coverage_share"],
                "value": [result["domain_info"]["unique_domains"], result["domain_info"]["coverage_share"]],
            }
        ),
        "hourly_heavy_authors": author_scores.loc[author_scores["max_posts_one_hour"] > config["thresholds"]["hourly_penalty_start"]]
        .sort_values("max_posts_one_hour", ascending=False)
        .head(50),
        "language_diversity_authors": author_scores.loc[author_scores["language_nunique"] > config["thresholds"]["language_penalty_start"]]
        .sort_values("language_nunique", ascending=False)
        .head(50),
        "hard_bot_examples": scored_df.loc[scored_df["hard_bot_cluster_flag"] == 1, [
            "author_hash",
            "normalized_text",
            "same_text_repeat_count",
            "same_text_unique_author_count",
            "same_text_time_window_count",
            "final_score",
        ]].head(50),
        "rapid_fire_examples": author_scores.sort_values("median_interpost_sec", ascending=True).head(50),
    }
    return tables


def plot_hourly_penalty_curve(author_scores: pd.DataFrame, config: dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if author_scores.empty:
        return fig
    x = np.arange(1, int(author_scores["max_posts_one_hour"].max()) + 1)
    y = log_penalty(
        pd.Series(x, dtype="float64"),
        config["thresholds"]["hourly_penalty_start"],
        float(author_scores["max_posts_one_hour"].max()),
    )
    ax.plot(x, y, color="#dc2626", linewidth=2)
    ax.axvline(config["thresholds"]["hourly_penalty_start"], color="#0f172a", linestyle="--", linewidth=1)
    knee = config.get("derived_thresholds", {}).get("hourly_hard_knee")
    if knee is not None:
        ax.axvline(knee, color="#7c3aed", linestyle=":", linewidth=1.5)
    ax.set_title("Hourly Penalty Curve")
    ax.set_xlabel("Max posts in one hour")
    ax.set_ylabel("Penalty")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig


def plot_hourly_distribution(author_scores: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if author_scores.empty:
        return fig
    ax.hist(author_scores["max_posts_one_hour"], bins=40, color="#2563eb", edgecolor="white")
    ax.set_title("Max Posts In One Hour Distribution")
    ax.set_xlabel("Posts")
    ax.set_ylabel("Author count")
    ax.set_yscale("log")
    fig.tight_layout()
    return fig


def plot_sentiment_theme_distributions(author_scores: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if author_scores.empty:
        return fig
    axes[0].hist(author_scores["theme_nunique"], bins=30, color="#7c3aed", edgecolor="white")
    axes[0].set_title("Theme Nunique")
    axes[1].hist(author_scores["sentiment_std"], bins=30, color="#059669", edgecolor="white")
    axes[1].set_title("Sentiment Std")
    fig.tight_layout()
    return fig


def validate_keyword_signal(scored_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if scored_df.empty:
        return pd.DataFrame()
    heavy_mask = scored_df["same_text_repeat_count"].ge(config["thresholds"]["spam_repeat_threshold"]) | scored_df["spam_pattern_flag"].eq(1)
    result = (
        scored_df.assign(is_heavy_message=heavy_mask)
        .groupby("is_heavy_message")
        .agg(
            avg_keyword_count=("keyword_count", "mean"),
            median_keyword_count=("keyword_count", "median"),
            avg_text_length=("text_length_chars", "mean"),
            rows=("keyword_count", "size"),
        )
        .reset_index()
    )
    return result


def summarize_score_bands(scored_df: pd.DataFrame) -> pd.DataFrame:
    if scored_df.empty:
        return pd.DataFrame(columns=["band", "rows", "share"])
    bins = [-0.001, 0.4, 0.6, 0.7, 0.85, 0.999999, 1.000001]
    labels = ["0.00-0.40", "0.40-0.60", "0.60-0.70", "0.70-0.85", "0.85-<1.0", "1.0"]
    bands = pd.cut(scored_df["final_score"], bins=bins, labels=labels, include_lowest=True, right=True)
    counts = bands.value_counts(sort=False).rename_axis("band").reset_index(name="rows")
    counts["share"] = counts["rows"] / max(len(scored_df), 1)
    return counts


def compute_text_hash_scalar(value: Any) -> str:
    if pd.isna(value):
        return hashlib.sha1(b"<NA>").hexdigest()
    return hashlib.sha1(str(value).encode("utf-8", errors="ignore")).hexdigest()


def _sqlite_path_from_config(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["batch_sqlite_db"])


def _author_scores_path_from_config(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["author_scores_parquet"])


def _scored_messages_path_from_config(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["scored_messages_parquet"])


def get_sqlite_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def initialize_batch_store(config: dict[str, Any]) -> None:
    db_path = _sqlite_path_from_config(config)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists() and config["runtime"]["overwrite_outputs"]:
        db_path.unlink()

    author_scores_path = _author_scores_path_from_config(config)
    if author_scores_path.exists() and config["runtime"]["overwrite_outputs"]:
        author_scores_path.unlink()

    scored_messages_path = _scored_messages_path_from_config(config)
    if scored_messages_path.exists() and config["runtime"]["overwrite_outputs"]:
        scored_messages_path.unlink()

    conn = get_sqlite_connection(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cleaned_messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                row_fingerprint TEXT NOT NULL UNIQUE,
                original_text TEXT,
                normalized_text TEXT,
                text_hash TEXT,
                english_keywords_clean TEXT,
                keyword_count INTEGER,
                primary_theme TEXT,
                sentiment REAL,
                main_emotion TEXT,
                language TEXT,
                url TEXT,
                raw_url_domain TEXT,
                registered_domain TEXT,
                subdomain TEXT,
                date TEXT,
                time_window_bucket INTEGER,
                author_hash TEXT,
                author_type TEXT,
                author_hash_missing_flag INTEGER,
                text_length_chars INTEGER,
                token_count INTEGER,
                unique_token_count INTEGER,
                max_token_frequency INTEGER,
                max_token_ratio REAL,
                repeated_token_count_over_2 INTEGER,
                hashtag_count INTEGER,
                hashtag_density_chars REAL,
                hashtag_density_tokens REAL,
                is_short_text INTEGER,
                is_long_text_flag INTEGER,
                is_empty_text INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_cleaned_author_date ON cleaned_messages(author_hash, date);
            CREATE INDEX IF NOT EXISTS idx_cleaned_text_hash ON cleaned_messages(text_hash);
            CREATE INDEX IF NOT EXISTS idx_cleaned_window ON cleaned_messages(text_hash, time_window_bucket);
            CREATE INDEX IF NOT EXISTS idx_cleaned_domain ON cleaned_messages(registered_domain);
            """
        )
        conn.commit()
    finally:
        conn.close()


def _prepare_sqlite_rows(batch_df: pd.DataFrame, config: dict[str, Any]) -> list[tuple[Any, ...]]:
    df = batch_df.copy()
    df["text_hash"] = df["normalized_text"].map(compute_text_hash_scalar)
    df["row_fingerprint"] = compute_row_fingerprint(df).astype("string")
    window_seconds = config["thresholds"]["hard_bot_time_window_sec"]
    valid_date = pd.to_datetime(df["date"], utc=True, errors="coerce")
    date_ns = valid_date.astype("int64", copy=False)
    df["time_window_bucket"] = date_ns.floordiv(window_seconds * 1_000_000_000).where(valid_date.notna(), pd.NA)

    columns = [
        "row_fingerprint",
        "original_text",
        "normalized_text",
        "text_hash",
        "english_keywords_clean",
        "keyword_count",
        "primary_theme",
        "sentiment",
        "main_emotion",
        "language",
        "url",
        "raw_url_domain",
        "registered_domain",
        "subdomain",
        "date",
        "time_window_bucket",
        "author_hash",
        "author_type",
        "author_hash_missing_flag",
        "text_length_chars",
        "token_count",
        "unique_token_count",
        "max_token_frequency",
        "max_token_ratio",
        "repeated_token_count_over_2",
        "hashtag_count",
        "hashtag_density_chars",
        "hashtag_density_tokens",
        "is_short_text",
        "is_long_text_flag",
        "is_empty_text",
    ]
    rows: list[tuple[Any, ...]] = []
    for record in df[columns].itertuples(index=False, name=None):
        converted = []
        for value in record:
            if pd.isna(value):
                converted.append(None)
            elif isinstance(value, pd.Timestamp):
                converted.append(value.isoformat())
            else:
                converted.append(value)
        rows.append(tuple(converted))
    return rows


def run_batch_pass1(config: dict[str, Any]) -> dict[str, Any]:
    initialize_batch_store(config)
    input_path = Path(config["paths"]["input_parquet"])
    db_path = _sqlite_path_from_config(config)
    batch_size = config["runtime"]["batch_size"] or estimate_batch_size()
    max_batches = config["runtime"].get("max_batches")
    max_rows = config["runtime"].get("sample_n_rows")

    summary = {
        "raw_rows": 0,
        "clean_rows": 0,
        "technical_duplicates_removed": 0,
        "missing_author_rows": 0,
        "empty_text_rows": 0,
        "short_text_rows": 0,
        "bad_date_rows": 0,
        "batch_size": batch_size,
    }
    conn = get_sqlite_connection(db_path)
    start = time.time()
    try:
        insert_sql = """
            INSERT OR IGNORE INTO cleaned_messages (
                row_fingerprint, original_text, normalized_text, text_hash, english_keywords_clean, keyword_count,
                primary_theme, sentiment, main_emotion, language, url, raw_url_domain, registered_domain, subdomain,
                date, time_window_bucket, author_hash, author_type, author_hash_missing_flag, text_length_chars,
                token_count, unique_token_count, max_token_frequency, max_token_ratio, repeated_token_count_over_2,
                hashtag_count, hashtag_density_chars, hashtag_density_tokens, is_short_text, is_long_text_flag, is_empty_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        for batch_id, batch_df in enumerate(iter_pandas_batches(input_path, batch_size=batch_size), start=1):
            if max_batches and batch_id > max_batches:
                break
            summary["raw_rows"] += len(batch_df)
            cleaned_batch = clean_batch(batch_df, config=config, rare_languages=set())
            rows = _prepare_sqlite_rows(cleaned_batch, config=config)
            before_changes = conn.total_changes
            conn.executemany(insert_sql, rows)
            conn.commit()
            inserted_rows = conn.total_changes - before_changes
            duplicate_rows = len(rows) - inserted_rows

            summary["clean_rows"] += inserted_rows
            summary["technical_duplicates_removed"] += duplicate_rows
            summary["missing_author_rows"] += int(cleaned_batch["author_hash_missing_flag"].sum())
            summary["empty_text_rows"] += int(cleaned_batch["is_empty_text"].sum())
            summary["short_text_rows"] += int(cleaned_batch["is_short_text"].sum())
            summary["bad_date_rows"] += int(cleaned_batch["date"].isna().sum())

            if max_rows and summary["clean_rows"] >= max_rows:
                break

            del batch_df, cleaned_batch, rows
            gc.collect()

        summary["elapsed_min"] = round((time.time() - start) / 60, 2)
        summary["sqlite_db"] = str(db_path)
        return summary
    finally:
        conn.close()


def materialize_batch_cluster_tables(config: dict[str, Any]) -> None:
    conn = get_sqlite_connection(_sqlite_path_from_config(config))
    try:
        conn.executescript(
            """
            DROP TABLE IF EXISTS text_clusters;
            CREATE TABLE text_clusters AS
            SELECT
                text_hash,
                MIN(normalized_text) AS sample_text,
                COUNT(*) AS repeat_count,
                COUNT(DISTINCT author_hash) AS unique_author_count
            FROM cleaned_messages
            GROUP BY text_hash;
            CREATE INDEX IF NOT EXISTS idx_text_clusters_hash ON text_clusters(text_hash);

            DROP TABLE IF EXISTS text_window_clusters;
            CREATE TABLE text_window_clusters AS
            SELECT
                text_hash,
                time_window_bucket,
                COUNT(*) AS window_count
            FROM cleaned_messages
            WHERE time_window_bucket IS NOT NULL
            GROUP BY text_hash, time_window_bucket;
            CREATE INDEX IF NOT EXISTS idx_text_window_clusters ON text_window_clusters(text_hash, time_window_bucket);
            """
        )
        conn.commit()
    finally:
        conn.close()


def _fetch_author_batch_frame(conn: sqlite3.Connection, author_ids: list[str]) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in author_ids)
    query = f"""
        SELECT
            cm.author_hash,
            cm.date,
            cm.language,
            cm.primary_theme,
            cm.sentiment,
            cm.text_hash,
            COALESCE(tc.unique_author_count, 0) AS same_text_unique_author_count
        FROM cleaned_messages cm
        LEFT JOIN text_clusters tc ON tc.text_hash = cm.text_hash
        WHERE cm.author_hash IN ({placeholders})
        ORDER BY cm.author_hash, cm.date
    """
    return pd.read_sql_query(query, conn, params=author_ids)


def _compute_author_features_from_frame(author_df: pd.DataFrame) -> pd.DataFrame:
    if author_df.empty:
        return pd.DataFrame()
    author_df["date"] = pd.to_datetime(author_df["date"], utc=True, errors="coerce")
    author_df = author_df.dropna(subset=["author_hash", "date"]).copy()
    author_df["day_bucket"] = author_df["date"].dt.floor("D")
    author_df["hour_bucket"] = author_df["date"].dt.floor("h")

    base = author_df.groupby("author_hash").agg(
        total_posts=("author_hash", "size"),
        active_days=("day_bucket", "nunique"),
        active_hours=("hour_bucket", "nunique"),
        language_nunique=("language", lambda s: s.nunique(dropna=True)),
        theme_nunique=("primary_theme", lambda s: s.nunique(dropna=True)),
        sentiment_std=("sentiment", lambda s: pd.to_numeric(s, errors="coerce").std()),
    )
    base["posts_per_day"] = base["total_posts"] / base["active_days"].clip(lower=1)
    base["posts_per_active_hour"] = base["total_posts"] / base["active_hours"].clip(lower=1)

    hourly_counts = author_df.groupby(["author_hash", "hour_bucket"]).size().rename("hour_post_count")
    base["max_posts_one_hour"] = hourly_counts.groupby("author_hash").max()
    base["hourly_post_distribution"] = hourly_counts.groupby("author_hash").apply(lambda s: s.tolist())

    author_df = author_df.sort_values(["author_hash", "date"]).copy()
    author_df["interpost_sec"] = author_df.groupby("author_hash")["date"].diff().dt.total_seconds()
    timing = author_df.groupby("author_hash").agg(
        mean_interpost_sec=("interpost_sec", "mean"),
        median_interpost_sec=("interpost_sec", "median"),
        p10_interpost_sec=("interpost_sec", lambda s: s.quantile(0.10)),
        std_interpost_sec=("interpost_sec", "std"),
    )
    base = base.join(timing)
    base["interval_cv"] = base["std_interpost_sec"] / base["mean_interpost_sec"].replace(0, np.nan)

    author_text_counts = author_df.groupby(["author_hash", "text_hash"]).size().rename("author_text_count").reset_index()
    repeated = author_text_counts["author_text_count"].gt(1)
    repeat_ratio = (
        author_text_counts.loc[repeated]
        .groupby("author_hash")["author_text_count"]
        .sum()
        .div(base["total_posts"])
    )
    base["same_text_repeat_ratio"] = repeat_ratio.fillna(0.0)
    base["same_text_repeat_max"] = author_text_counts.groupby("author_hash")["author_text_count"].max().fillna(0.0)
    multi_author_ratio = (
        author_df.groupby("author_hash")["same_text_unique_author_count"]
        .apply(lambda s: float(pd.Series(s).gt(1).mean()) if len(s) else 0.0)
        .fillna(0.0)
    )
    base["multi_author_repeat_ratio"] = multi_author_ratio
    base["sentiment_std"] = base["sentiment_std"].fillna(0.0)
    base["mean_interpost_sec"] = base["mean_interpost_sec"].fillna(np.inf)
    base["median_interpost_sec"] = base["median_interpost_sec"].fillna(np.inf)
    base["p10_interpost_sec"] = base["p10_interpost_sec"].fillna(np.inf)
    base["interval_cv"] = base["interval_cv"].replace([np.inf, -np.inf], np.nan).fillna(np.inf)
    return base.reset_index()


def run_batch_pass2_author(config: dict[str, Any]) -> pd.DataFrame:
    conn = get_sqlite_connection(_sqlite_path_from_config(config))
    author_scores_path = _author_scores_path_from_config(config)
    author_batch_size = config["runtime"].get("author_batch_size", 5000)
    writer = None
    preview_frames: list[pd.DataFrame] = []
    try:
        materialize_batch_cluster_tables(config)
        author_cursor = conn.execute(
            "SELECT DISTINCT author_hash FROM cleaned_messages WHERE author_type = 'identified' AND author_hash IS NOT NULL ORDER BY author_hash"
        )
        author_frames: list[pd.DataFrame] = []
        while True:
            rows = author_cursor.fetchmany(author_batch_size)
            if not rows:
                break
            author_ids = [row[0] for row in rows if row[0] is not None]
            author_batch_frame = _fetch_author_batch_frame(conn, author_ids)
            batch_features = _compute_author_features_from_frame(author_batch_frame)
            author_frames.append(batch_features)
        all_author_features = pd.concat(author_frames, ignore_index=True) if author_frames else pd.DataFrame()
        refs = fit_normalization_reference(all_author_features, pd.DataFrame(), config)
        author_scores = compute_author_scores(all_author_features, refs, config)
        table = pa.Table.from_pandas(author_scores, preserve_index=False)
        writer = pq.ParquetWriter(author_scores_path, table.schema, compression="snappy")
        writer.write_table(table)

        conn.execute("DROP TABLE IF EXISTS author_scores")
        conn.execute(
            """
            CREATE TABLE author_scores (
                author_hash TEXT PRIMARY KEY,
                author_score REAL,
                author_hard_hourly_flag INTEGER,
                max_posts_one_hour REAL,
                language_nunique REAL,
                theme_nunique REAL,
                sentiment_std REAL,
                median_interpost_sec REAL
            )
            """
        )
        conn.executemany(
            "INSERT INTO author_scores(author_hash, author_score, author_hard_hourly_flag, max_posts_one_hour, language_nunique, theme_nunique, sentiment_std, median_interpost_sec) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            author_scores[
                ["author_hash", "author_score", "author_hard_hourly_flag", "max_posts_one_hour", "language_nunique", "theme_nunique", "sentiment_std", "median_interpost_sec"]
            ].itertuples(index=False, name=None),
        )
        conn.commit()
        preview_frames.append(author_scores.head(50))
        return author_scores
    finally:
        if writer is not None:
            writer.close()
        conn.close()


def run_batch_pass2_message(config: dict[str, Any], refs: dict[str, Any]) -> dict[str, Any]:
    conn = get_sqlite_connection(_sqlite_path_from_config(config))
    scored_path = _scored_messages_path_from_config(config)
    message_batch_size = config["runtime"].get("message_batch_size", 100_000)
    writer = None
    last_id = 0
    hard_bot_examples: list[pd.DataFrame] = []
    preview_frames: list[pd.DataFrame] = []
    heavy_keyword_stats = {"heavy_keyword_sum": 0.0, "heavy_rows": 0, "light_keyword_sum": 0.0, "light_rows": 0}
    score_band_accumulator: list[pd.DataFrame] = []
    hashtag_examples: list[pd.DataFrame] = []
    token_examples: list[pd.DataFrame] = []
    try:
        while True:
            query = """
                SELECT
                    cm.message_id,
                    cm.original_text,
                    cm.normalized_text,
                    cm.author_hash,
                    cm.author_type,
                    cm.date,
                    cm.keyword_count,
                    cm.text_length_chars,
                    cm.token_count,
                    cm.unique_token_count,
                    cm.max_token_frequency,
                    cm.max_token_ratio,
                    cm.repeated_token_count_over_2,
                    cm.hashtag_count,
                    cm.hashtag_density_chars,
                    cm.hashtag_density_tokens,
                    cm.is_long_text_flag,
                    cm.time_window_bucket,
                    COALESCE(tc.repeat_count, 0) AS same_text_repeat_count,
                    COALESCE(tc.unique_author_count, 0) AS same_text_unique_author_count,
                    COALESCE(twc.window_count, 0) AS same_text_time_window_count,
                    COALESCE(a.author_score, NULL) AS author_score,
                    COALESCE(a.author_hard_hourly_flag, 0) AS author_hard_hourly_flag
                FROM cleaned_messages cm
                LEFT JOIN text_clusters tc ON tc.text_hash = cm.text_hash
                LEFT JOIN text_window_clusters twc ON twc.text_hash = cm.text_hash AND twc.time_window_bucket = cm.time_window_bucket
                LEFT JOIN author_scores a ON a.author_hash = cm.author_hash
                WHERE cm.message_id > ?
                ORDER BY cm.message_id
                LIMIT ?
            """
            batch_df = pd.read_sql_query(query, conn, params=[last_id, message_batch_size])
            if batch_df.empty:
                break
            last_id = int(batch_df["message_id"].max())
            batch_df["spam_pattern_flag"] = (
                (batch_df["same_text_repeat_count"] >= config["thresholds"]["spam_repeat_threshold"])
                | (batch_df["same_text_unique_author_count"] >= config["thresholds"]["spam_multi_author_threshold"])
                | (batch_df["same_text_time_window_count"] >= config["thresholds"]["spam_time_cluster_threshold"])
            ).astype("int8")
            batch_df["hard_bot_cluster_flag"] = (
                (batch_df["same_text_repeat_count"] >= config["thresholds"]["hard_bot_repeat_threshold"])
                & (batch_df["same_text_unique_author_count"] >= config["thresholds"]["hard_bot_multi_author_threshold"])
                & (batch_df["same_text_time_window_count"] >= config["thresholds"]["hard_bot_time_cluster_threshold"])
            ).astype("int8")

            message_scores = compute_message_scores(batch_df, refs, config)
            final_df = message_scores.copy()
            final_df["behavioral_score"] = np.where(
                final_df["author_type"].eq("anonymous"),
                final_df["message_score"],
                (
                    config["weights"]["author_vs_message"]["author"] * final_df["author_score"].fillna(0.0)
                    + config["weights"]["author_vs_message"]["message"] * final_df["message_score"]
                ).clip(0, 1),
            )
            final_df["final_score"] = np.where(
                final_df["author_hard_hourly_flag"].eq(1) | final_df["hard_bot_cluster_flag"].eq(1),
                1.0,
                final_df["behavioral_score"],
            )

            heavy_mask = final_df["spam_pattern_flag"].eq(1) | final_df["same_text_repeat_count"].ge(config["thresholds"]["spam_repeat_threshold"])
            heavy_keyword_stats["heavy_keyword_sum"] += float(final_df.loc[heavy_mask, "keyword_count"].sum())
            heavy_keyword_stats["heavy_rows"] += int(heavy_mask.sum())
            heavy_keyword_stats["light_keyword_sum"] += float(final_df.loc[~heavy_mask, "keyword_count"].sum())
            heavy_keyword_stats["light_rows"] += int((~heavy_mask).sum())

            if not preview_frames:
                preview_frames.append(final_df.head(20).copy())
            hard_bot_sample = final_df.loc[final_df["hard_bot_cluster_flag"].eq(1)].head(50)
            if not hard_bot_sample.empty and len(hard_bot_examples) < 3:
                hard_bot_examples.append(hard_bot_sample.copy())
            hashtag_sample = final_df.sort_values("hashtag_spam_component", ascending=False).head(20)
            if len(hashtag_examples) < 2 and not hashtag_sample.empty:
                hashtag_examples.append(hashtag_sample.copy())
            token_sample = final_df.sort_values("token_repetition_component", ascending=False).head(20)
            if len(token_examples) < 2 and not token_sample.empty:
                token_examples.append(token_sample.copy())
            score_band_accumulator.append(summarize_score_bands(final_df))

            output_columns = [
                "message_id",
                "author_hash",
                "author_type",
                "normalized_text",
                "same_text_repeat_count",
                "same_text_unique_author_count",
                "same_text_time_window_count",
                "hashtag_count",
                "max_token_frequency",
                "max_token_ratio",
                "spam_pattern_flag",
                "hard_bot_cluster_flag",
                "author_hard_hourly_flag",
                "author_score",
                "message_score",
                "behavioral_score",
                "final_score",
            ]
            table = pa.Table.from_pandas(final_df[output_columns], preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(scored_path, table.schema, compression="snappy")
            writer.write_table(table)
            del batch_df, message_scores, final_df, table
            gc.collect()

        keyword_validation = pd.DataFrame(
            [
                {
                    "is_heavy_message": False,
                    "avg_keyword_count": heavy_keyword_stats["light_keyword_sum"] / max(heavy_keyword_stats["light_rows"], 1),
                    "rows": heavy_keyword_stats["light_rows"],
                },
                {
                    "is_heavy_message": True,
                    "avg_keyword_count": heavy_keyword_stats["heavy_keyword_sum"] / max(heavy_keyword_stats["heavy_rows"], 1),
                    "rows": heavy_keyword_stats["heavy_rows"],
                },
            ]
        )
        if score_band_accumulator:
            score_bands = pd.concat(score_band_accumulator, ignore_index=True).groupby("band", as_index=False)["rows"].sum()
            score_bands["share"] = score_bands["rows"] / max(score_bands["rows"].sum(), 1)
        else:
            score_bands = pd.DataFrame(columns=["band", "rows", "share"])
        return {
            "scored_preview": pd.concat(preview_frames, ignore_index=True) if preview_frames else pd.DataFrame(),
            "hard_bot_examples": pd.concat(hard_bot_examples, ignore_index=True).head(50) if hard_bot_examples else pd.DataFrame(),
            "keyword_validation": keyword_validation,
            "score_bands": score_bands,
            "hashtag_spam_examples": pd.concat(hashtag_examples, ignore_index=True).head(20) if hashtag_examples else pd.DataFrame(),
            "token_spam_examples": pd.concat(token_examples, ignore_index=True).head(20) if token_examples else pd.DataFrame(),
        }
    finally:
        if writer is not None:
            writer.close()
        conn.close()


def build_batch_summary_tables(config: dict[str, Any], pass1_summary: dict[str, Any], author_scores: pd.DataFrame, message_artifacts: dict[str, Any]) -> dict[str, pd.DataFrame]:
    conn = get_sqlite_connection(_sqlite_path_from_config(config))
    try:
        top_domains = pd.read_sql_query(
            """
            SELECT registered_domain, COUNT(*) AS rows
            FROM cleaned_messages
            WHERE registered_domain IS NOT NULL
            GROUP BY registered_domain
            ORDER BY rows DESC
            LIMIT 20
            """,
            conn,
        )
        coverage_df = pd.read_sql_query(
            """
            WITH domain_counts AS (
                SELECT registered_domain, COUNT(*) AS rows
                FROM cleaned_messages
                WHERE registered_domain IS NOT NULL
                GROUP BY registered_domain
            ),
            ranked AS (
                SELECT registered_domain, rows,
                       ROW_NUMBER() OVER (ORDER BY rows DESC) AS rn
                FROM domain_counts
            )
            SELECT
                (SELECT COUNT(*) FROM domain_counts) AS unique_domains,
                1.0 * (SELECT COALESCE(SUM(rows), 0) FROM ranked WHERE rn <= 128)
                    / NULLIF((SELECT COALESCE(SUM(rows), 0) FROM domain_counts), 0) AS top128_coverage_share
            """,
            conn,
        )
        summary_frame = pd.Series(pass1_summary).to_frame("value")
    finally:
        conn.close()

    tables = {
        "summary": summary_frame,
        "derived_thresholds": pd.DataFrame(
            {
                "metric": ["hourly_hard_knee"],
                "value": [config.get("derived_thresholds", {}).get("hourly_hard_knee")],
            }
        ),
        "top_domains": top_domains,
        "top128_coverage": coverage_df,
        "hourly_heavy_authors": author_scores.loc[author_scores["max_posts_one_hour"] > config["thresholds"]["hourly_penalty_start"]]
        .sort_values("max_posts_one_hour", ascending=False)
        .head(50),
        "hourly_hard_authors": author_scores.loc[author_scores["author_hard_hourly_flag"].eq(1)]
        .sort_values("max_posts_one_hour", ascending=False)
        .head(50),
        "language_diversity_authors": author_scores.loc[author_scores["language_nunique"] > config["thresholds"]["language_penalty_start"]]
        .sort_values("language_nunique", ascending=False)
        .head(50),
        "rapid_fire_examples": author_scores.sort_values("median_interpost_sec", ascending=True).head(50),
        "hard_bot_examples": message_artifacts["hard_bot_examples"],
        "keyword_validation": message_artifacts["keyword_validation"],
        "score_bands": message_artifacts["score_bands"],
        "hashtag_spam_examples": message_artifacts["hashtag_spam_examples"],
        "token_spam_examples": message_artifacts["token_spam_examples"],
    }
    return tables


def compute_message_refs_from_sqlite(config: dict[str, Any]) -> dict[str, Any]:
    conn = get_sqlite_connection(_sqlite_path_from_config(config))
    try:
        refs = {
            "author": {},
            "message": {
                "text_length_chars": (0.0, 1.0),
                "same_text_repeat_count": (0.0, 1.0),
                "same_text_unique_author_count": (0.0, 1.0),
                "same_text_time_window_count": (0.0, 1.0),
                "keyword_count": (0.0, 1.0),
                "hashtag_count": (0.0, 1.0),
                "hashtag_density_chars": (0.0, 1.0),
                "hashtag_density_tokens": (0.0, 1.0),
                "max_token_frequency": (0.0, 1.0),
                "max_token_ratio": (0.0, 1.0),
                "repeated_token_count_over_2": (0.0, 1.0),
                "long_text_excess_max": 1.0,
            },
        }
        row = conn.execute(
            """
            SELECT
                COALESCE(MAX(text_length_chars), 1),
                COALESCE(MAX(keyword_count), 1),
                COALESCE(MAX(hashtag_count), 1),
                COALESCE(MAX(hashtag_density_chars), 1),
                COALESCE(MAX(hashtag_density_tokens), 1),
                COALESCE(MAX(max_token_frequency), 1),
                COALESCE(MAX(max_token_ratio), 1),
                COALESCE(MAX(repeated_token_count_over_2), 1)
            FROM cleaned_messages
            """
        ).fetchone()
        (
            text_length_max,
            keyword_count_max,
            hashtag_count_max,
            hashtag_density_chars_max,
            hashtag_density_tokens_max,
            max_token_frequency_max,
            max_token_ratio_max,
            repeated_token_count_over_2_max,
        ) = row
        cluster_row = conn.execute(
            """
            SELECT
                COALESCE(MAX(repeat_count), 1),
                COALESCE(MAX(unique_author_count), 1)
            FROM text_clusters
            """
        ).fetchone()
        repeat_count_max, unique_author_count_max = cluster_row
        window_row = conn.execute(
            """
            SELECT COALESCE(MAX(window_count), 1)
            FROM text_window_clusters
            """
        ).fetchone()
        window_count_max = window_row[0]

        refs["message"]["text_length_chars"] = (0.0, float(text_length_max))
        refs["message"]["same_text_repeat_count"] = (0.0, float(repeat_count_max))
        refs["message"]["same_text_unique_author_count"] = (0.0, float(unique_author_count_max))
        refs["message"]["same_text_time_window_count"] = (0.0, float(window_count_max))
        refs["message"]["keyword_count"] = (0.0, float(keyword_count_max))
        refs["message"]["hashtag_count"] = (0.0, float(hashtag_count_max))
        refs["message"]["hashtag_density_chars"] = (0.0, float(hashtag_density_chars_max))
        refs["message"]["hashtag_density_tokens"] = (0.0, float(hashtag_density_tokens_max))
        refs["message"]["max_token_frequency"] = (0.0, float(max_token_frequency_max))
        refs["message"]["max_token_ratio"] = (0.0, float(max_token_ratio_max))
        refs["message"]["repeated_token_count_over_2"] = (0.0, float(repeated_token_count_over_2_max))
        refs["message"]["long_text_excess_max"] = float(max(text_length_max - config["thresholds"]["long_text_start"], 1.0))
        return refs
    finally:
        conn.close()


def run_formula_pipeline_two_pass(config: dict[str, Any]) -> dict[str, Any]:
    pass1_summary = run_batch_pass1(config)
    author_scores = run_batch_pass2_author(config)
    refs = compute_message_refs_from_sqlite(config)
    message_artifacts = run_batch_pass2_message(config, refs)
    tables = build_batch_summary_tables(config, pass1_summary, author_scores, message_artifacts)
    return {
        "summary": pass1_summary,
        "tables": tables,
        "author_scores": author_scores,
        "scored_preview": message_artifacts["scored_preview"],
        "paths": {
            "sqlite_db": str(_sqlite_path_from_config(config)),
            "author_scores_parquet": str(_author_scores_path_from_config(config)),
            "scored_messages_parquet": str(_scored_messages_path_from_config(config)),
        },
    }


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
            author_row = author_scores.loc[author_scores["author_hash"] == author_hash, ["author_hash", "author_score"]]
        else:
            author_row = pd.DataFrame(columns=["author_hash", "author_score"])
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
        final["behavioral_score"] = np.where(
            final["author_type"].eq("anonymous"),
            final["message_score"],
            (
                config["weights"]["author_vs_message"]["author"] * final["author_score"].fillna(0.0)
                + config["weights"]["author_vs_message"]["message"] * final["message_score"]
            ).clip(0, 1),
        )
        final["final_score"] = np.where(
            final["author_hard_hourly_flag"].eq(1) | final["hard_bot_cluster_flag"].eq(1),
            1.0,
            final["behavioral_score"],
        )

    explanation_columns = [
        "author_type",
        "message_score",
        "author_score",
        "behavioral_score",
        "final_score",
        "same_text_repeat_component",
        "spam_pattern_component",
        "hashtag_spam_component",
        "token_repetition_component",
        "long_text_component",
        "keyword_signal_component",
        "author_hard_hourly_flag",
        "hard_bot_cluster_flag",
    ]
    return final[explanation_columns]
