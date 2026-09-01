from __future__ import annotations

import gc
import hashlib
import os
import re
import time
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .constants import BLANK_SENSITIVE_COLUMNS, COMMON_MULTI_PART_SUFFIXES, MISSING_PLACEHOLDERS

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None

try:
    import tldextract  # type: ignore
except ImportError:
    tldextract = None

try:
    import torch  # type: ignore
except ImportError:
    torch = None

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
except ImportError:
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

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
SEMANTIC_URL_PATTERN = re.compile(r"http\S+|www\.\S+", flags=re.IGNORECASE)
SEMANTIC_USER_TAG_PATTERN = re.compile(r"[@#]", flags=re.UNICODE)
SEMANTIC_PUNCT_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)
SEMANTIC_NUMBER_PATTERN = re.compile(r"\d+", flags=re.UNICODE)


def extract_text_stats_scalar(value: Any) -> dict[str, float]:
    if pd.isna(value):
        return {
            "token_count": 0.0,
            "unique_token_count": 0.0,
            "max_token_frequency": 0.0,
            "max_token_ratio": 0.0,
            "repeated_token_count_over_2": 0.0,
            "hashtag_count": 0.0,
            "exclamation_count": 0.0,
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
    exclamation_count = float(text.count("!"))
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
        "exclamation_count": exclamation_count,
        "hashtag_density_chars": hashtag_density_chars,
        "hashtag_density_tokens": hashtag_density_tokens,
    }


def extract_text_stats_series(series: pd.Series) -> pd.DataFrame:
    stats = series.map(extract_text_stats_scalar)
    return pd.DataFrame(stats.tolist(), index=series.index)


def preprocess_semantic_text_scalar(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = SEMANTIC_URL_PATTERN.sub(" ", text)
    text = SEMANTIC_USER_TAG_PATTERN.sub("", text)
    text = SEMANTIC_PUNCT_PATTERN.sub(" ", text)
    text = SEMANTIC_NUMBER_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def detect_torch_device(config: dict[str, Any]) -> str:
    semantic_cfg = resolve_semantic_adapter_config(config)
    preferred = semantic_cfg.get("device", "auto")
    if preferred != "auto":
        return preferred
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_semantic_adapter_config(config: dict[str, Any]) -> dict[str, Any]:
    semantic_cfg = dict(config["semantic_adapter"])
    model_presets = semantic_cfg.pop("models", None)
    selected_model_key = semantic_cfg.pop("selected_model_key", None)
    if not model_presets or not selected_model_key:
        return semantic_cfg
    if selected_model_key not in model_presets:
        available = ", ".join(sorted(model_presets))
        raise KeyError(
            f"semantic_adapter.selected_model_key={selected_model_key!r} not found in semantic_adapter.models. "
            f"Available keys: {available}"
        )
    resolved_cfg = dict(semantic_cfg)
    resolved_cfg.update(model_presets[selected_model_key])
    resolved_cfg["selected_model_key"] = selected_model_key
    return resolved_cfg


@lru_cache(maxsize=2)
def _load_semantic_adapter_cached(model_name: str, device: str):
    if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
        raise ImportError(
            "transformers and torch are required for semantic adapter. Install requirements and rerun."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return tokenizer, model


def compute_roberta_scores_for_batch(batch_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    semantic_cfg = resolve_semantic_adapter_config(config)
    neutral_score = float(semantic_cfg["unsupported_language_score"])
    output = pd.DataFrame(
        {
            "roberta_score": np.full(len(batch_df), neutral_score, dtype="float32"),
            "semantic_model_applied_flag": np.zeros(len(batch_df), dtype="int8"),
        },
        index=batch_df.index,
    )
    if not semantic_cfg.get("enabled", True):
        return output

    selected_model_key = str(semantic_cfg.get("selected_model_key", ""))
    model_name = str(semantic_cfg["model_name"])
    if selected_model_key == "xlmr_base_multilingual" and model_name == "FacebookAI/xlm-roberta-base":
        raise ValueError(
            "The xlmr_base_multilingual preset points to the raw XLM-R base checkpoint, which is not a bot "
            "classification model. Set semantic_adapter.models['xlmr_base_multilingual']['model_name'] to a "
            "fine-tuned XLM-R sequence-classification checkpoint before enabling this preset."
        )

    supported_languages_value = semantic_cfg.get("supported_languages", ["en"])
    if supported_languages_value in ("all", "*"):
        supported_mask = pd.Series(True, index=batch_df.index)
    else:
        supported_languages = {str(lang).lower() for lang in supported_languages_value}
        language_series = batch_df["language"].astype("string").fillna("").str.lower()
        supported_mask = language_series.isin(supported_languages)
    if not bool(supported_mask.any()):
        return output

    texts = batch_df.loc[supported_mask, "original_text"].map(preprocess_semantic_text_scalar)
    non_empty_mask = texts.str.len().fillna(0).gt(0)
    if not bool(non_empty_mask.any()):
        return output

    device = detect_torch_device(config)
    tokenizer, model = _load_semantic_adapter_cached(model_name, device)
    batch_size = int(semantic_cfg["batch_size"])
    max_length = int(semantic_cfg["max_length"])
    inference_indices = texts.index[non_empty_mask]
    inference_texts = texts.loc[non_empty_mask].tolist()
    score_values: list[float] = []

    for start in range(0, len(inference_texts), batch_size):
        stop = start + batch_size
        text_batch = inference_texts[start:stop]
        encoded = tokenizer(
            text_batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)[:, 1]
        score_values.extend(probabilities.detach().cpu().tolist())

    output.loc[inference_indices, "roberta_score"] = np.array(score_values, dtype="float32")
    output.loc[inference_indices, "semantic_model_applied_flag"] = 1
    return output


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
    batch_df["exclamation_count"] = batch_df["exclamation_count"].astype("int16")

    domain_parts = extract_domain_columns(batch_df["url"])
    batch_df = batch_df.drop(columns=["raw_url_domain", "registered_domain", "subdomain"], errors="ignore")
    batch_df = pd.concat([batch_df, domain_parts], axis=1)
    semantic_scores = compute_roberta_scores_for_batch(batch_df, config)
    batch_df["roberta_score"] = semantic_scores["roberta_score"]
    batch_df["semantic_model_applied_flag"] = semantic_scores["semantic_model_applied_flag"].astype("int8")
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
def compute_text_hash_scalar(value: Any) -> str:
    if pd.isna(value):
        return hashlib.sha1(b"<NA>").hexdigest()
    return hashlib.sha1(str(value).encode("utf-8", errors="ignore")).hexdigest()
