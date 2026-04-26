from __future__ import annotations

import argparse
import gc
import os
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")

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
    max_rows: int = 120_000,
    estimated_bytes_per_row: int = 2_500,
    target_memory_fraction: float = 0.025,
) -> int:
    available = get_available_memory_bytes()
    target_bytes = max(int(available * target_memory_fraction), estimated_bytes_per_row * min_rows)
    estimated_rows = target_bytes // estimated_bytes_per_row
    return max(min_rows, min(max_rows, estimated_rows))


def normalize_missing_scalar(value):
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


def normalize_text_scalar(value):
    value = normalize_missing_scalar(value)
    if pd.isna(value):
        return pd.NA
    value = str(value).replace("\r\n", "\n").replace("\r", "\n")
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"[^\S\n]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = value.strip()
    return value if value else pd.NA


def normalize_text_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_text_scalar)


def clean_keywords_scalar(value):
    value = normalize_missing_scalar(value)
    if pd.isna(value):
        return pd.NA
    parts = []
    seen = set()
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


def update_counter(counter: Counter, series: pd.Series) -> None:
    clean_series = series.astype("string").dropna()
    if clean_series.empty:
        return
    counter.update(clean_series.value_counts().to_dict())


def update_pair_counter(counter: Counter, left: pd.Series, right: pd.Series) -> None:
    pair_df = pd.DataFrame({"left": left.astype("string"), "right": right.astype("string")}).dropna()
    if pair_df.empty:
        return
    counts = pair_df.value_counts().to_dict()
    counter.update({(k[0], k[1]): int(v) for k, v in counts.items()})


@dataclass
class EDAAccumulator:
    sentiment_bins: np.ndarray = field(default_factory=lambda: np.linspace(-1.0, 1.0, 21))
    text_length_bins: np.ndarray = field(
        default_factory=lambda: np.array([0, 20, 40, 80, 120, 180, 260, 400, 600, 1000, 2000, 5000, 20000])
    )
    language_counts: Counter = field(default_factory=Counter)
    platform_counts: Counter = field(default_factory=Counter)
    theme_counts: Counter = field(default_factory=Counter)
    emotion_counts: Counter = field(default_factory=Counter)
    daily_counts: Counter = field(default_factory=Counter)
    hour_counts: Counter = field(default_factory=Counter)
    lang_platform_counts: Counter = field(default_factory=Counter)
    theme_emotion_counts: Counter = field(default_factory=Counter)
    sentiment_hist: np.ndarray = field(init=False)
    text_length_hist: np.ndarray = field(init=False)
    sentiment_sum: float = 0.0
    sentiment_count: int = 0
    unique_authors: set = field(default_factory=set)

    def __post_init__(self) -> None:
        self.sentiment_hist = np.zeros(len(self.sentiment_bins) - 1, dtype=np.int64)
        self.text_length_hist = np.zeros(len(self.text_length_bins) - 1, dtype=np.int64)

    def update(self, frame: pd.DataFrame) -> None:
        update_counter(self.language_counts, frame["language"])
        update_counter(self.platform_counts, frame["registered_domain"])
        update_counter(self.theme_counts, frame["primary_theme"])
        update_counter(self.emotion_counts, frame["main_emotion"])
        update_pair_counter(self.lang_platform_counts, frame["language"], frame["registered_domain"])
        update_pair_counter(self.theme_emotion_counts, frame["primary_theme"], frame["main_emotion"])

        valid_dates = frame["date"].dropna()
        if not valid_dates.empty:
            hour_counts = valid_dates.dt.hour.value_counts().to_dict()
            self.hour_counts.update({int(k): int(v) for k, v in hour_counts.items()})

            day_counts = valid_dates.dt.floor("D").dt.strftime("%Y-%m-%d").value_counts().to_dict()
            self.daily_counts.update(day_counts)

        sentiment = pd.to_numeric(frame["sentiment"], errors="coerce").dropna()
        if not sentiment.empty:
            hist, _ = np.histogram(sentiment.to_numpy(dtype=float), bins=self.sentiment_bins)
            self.sentiment_hist += hist
            self.sentiment_sum += float(sentiment.sum())
            self.sentiment_count += int(sentiment.size)

        text_lengths = frame["normalized_text"].str.len().dropna()
        if not text_lengths.empty:
            hist, _ = np.histogram(text_lengths.to_numpy(dtype=float), bins=self.text_length_bins)
            self.text_length_hist += hist

        authors = frame["author_hash"].dropna().astype("string").unique().tolist()
        self.unique_authors.update(authors)


def collect_language_counts(path: Path, batch_size: int) -> pd.Series:
    language_counts = pd.Series(dtype="int64")
    for batch_df in iter_pandas_batches(path, batch_size=batch_size):
        batch_lang = normalize_missing_series(batch_df["language"]).value_counts(dropna=False)
        language_counts = language_counts.add(batch_lang, fill_value=0)
        del batch_df, batch_lang
        gc.collect()
    return language_counts.sort_values(ascending=False).astype("int64")


def clean_batch(batch_df: pd.DataFrame, min_text_len: int, rare_languages: set[str]) -> pd.DataFrame:
    batch_df.columns = [str(col) for col in batch_df.columns]
    for column in BLANK_SENSITIVE_COLUMNS:
        batch_df[column] = normalize_missing_series(batch_df[column])

    batch_df["language"] = normalize_missing_series(batch_df["language"])
    batch_df["url"] = normalize_missing_series(batch_df["url"])
    batch_df["main_emotion"] = normalize_missing_series(batch_df["main_emotion"])

    batch_df["date"] = pd.to_datetime(batch_df["date"], utc=True, errors="coerce")
    batch_df["normalized_text"] = normalize_text_series(batch_df["original_text"])
    batch_df["english_keywords_clean"] = clean_keywords_series(batch_df["english_keywords"])
    batch_df["primary_theme"] = batch_df["primary_theme"].fillna("unknown_theme")

    batch_df["author_hash_missing_flag"] = batch_df["author_hash"].isna().astype("int8")
    batch_df["is_empty_text"] = batch_df["normalized_text"].isna().astype("int8")
    batch_df["text_length_chars"] = batch_df["normalized_text"].str.len().fillna(0).astype("int32")
    batch_df["is_short_text"] = batch_df["text_length_chars"].lt(min_text_len).astype("int8")
    batch_df["is_rare_language"] = batch_df["language"].isin(rare_languages).astype("int8")

    domain_parts = extract_domain_columns(batch_df["url"])
    batch_df = batch_df.drop(columns=["raw_url_domain", "registered_domain", "subdomain"], errors="ignore")
    batch_df = pd.concat([batch_df, domain_parts], axis=1)
    return batch_df


def clean_write_and_collect(
    input_path: Path,
    clean_output_path: Path,
    batch_size: int,
    min_text_len: int,
    rare_languages: set[str],
    report_dir: Path,
    max_batches: int | None,
) -> tuple[dict, EDAAccumulator]:
    seen_fingerprints = set()
    writer = None
    summary = {
        "raw_rows": 0,
        "cleaned_rows": 0,
        "technical_duplicates_removed": 0,
        "missing_author_rows": 0,
        "empty_text_rows": 0,
        "short_text_rows": 0,
        "bad_date_rows": 0,
        "rare_language_rows": 0,
    }
    eda = EDAAccumulator()

    if clean_output_path.exists():
        clean_output_path.unlink()

    start = time.time()
    try:
        for batch_id, batch_df in enumerate(iter_pandas_batches(input_path, batch_size=batch_size), start=1):
            if max_batches is not None and batch_id > max_batches:
                break

            summary["raw_rows"] += len(batch_df)
            batch_df = clean_batch(batch_df, min_text_len=min_text_len, rare_languages=rare_languages)

            fingerprints = compute_row_fingerprint(batch_df)
            is_duplicate = fingerprints.isin(seen_fingerprints)
            new_fingerprints = fingerprints.loc[~is_duplicate].tolist()
            seen_fingerprints.update(new_fingerprints)

            cleaned_batch = batch_df.loc[~is_duplicate].copy()
            cleaned_batch["technical_duplicate_flag"] = 0

            summary["technical_duplicates_removed"] += int(is_duplicate.sum())
            summary["cleaned_rows"] += len(cleaned_batch)
            summary["missing_author_rows"] += int(cleaned_batch["author_hash_missing_flag"].sum())
            summary["empty_text_rows"] += int(cleaned_batch["is_empty_text"].sum())
            summary["short_text_rows"] += int(cleaned_batch["is_short_text"].sum())
            summary["bad_date_rows"] += int(cleaned_batch["date"].isna().sum())
            summary["rare_language_rows"] += int(cleaned_batch["is_rare_language"].sum())

            eda.update(cleaned_batch)

            table = pa.Table.from_pandas(cleaned_batch, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(clean_output_path, table.schema, compression="snappy")
            writer.write_table(table)

            if batch_id % 5 == 0:
                elapsed = (time.time() - start) / 60
                print(
                    f"[clean+eda] batch={batch_id} cleaned_rows={summary['cleaned_rows']:,} "
                    f"duplicates_removed={summary['technical_duplicates_removed']:,} elapsed_min={elapsed:.2f}"
                )

            del batch_df, fingerprints, is_duplicate, new_fingerprints, cleaned_batch, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()

    elapsed = (time.time() - start) / 60
    print(f"[clean+eda] completed in {elapsed:.2f} min")
    print(f"[clean+eda] cleaned_output={clean_output_path}")
    print(f"[clean+eda] report_dir={report_dir}")
    return summary, eda


def counter_to_series(counter: Counter, name: str) -> pd.Series:
    if not counter:
        return pd.Series(dtype="int64", name=name)
    series = pd.Series(counter, name=name).sort_values(ascending=False)
    series.index = series.index.astype(str)
    return series.astype("int64")


def pair_counter_to_frame(counter: Counter, left_name: str, right_name: str, value_name: str) -> pd.DataFrame:
    if not counter:
        return pd.DataFrame(columns=[left_name, right_name, value_name])
    rows = [{left_name: k[0], right_name: k[1], value_name: int(v)} for k, v in counter.items()]
    return pd.DataFrame(rows).sort_values(value_name, ascending=False)


def save_bar_chart(series: pd.Series, title: str, output_path: Path, xlabel: str, top_n: int = 15) -> None:
    if series.empty:
        return
    top = series.head(top_n).sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(top.index, top.values, color="#2563eb")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def save_daily_posts_chart(series: pd.Series, output_path: Path) -> None:
    if series.empty:
        return
    daily = series.sort_index()
    plt.figure(figsize=(12, 4.8))
    plt.plot(daily.index, daily.values, color="#0f766e", linewidth=1.4)
    plt.title("Daily Post Volume")
    plt.xlabel("Date")
    plt.ylabel("Posts")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def save_histogram_from_counts(bin_edges: np.ndarray, counts: np.ndarray, title: str, output_path: Path, xlabel: str) -> None:
    if counts.sum() == 0:
        return
    widths = np.diff(bin_edges)
    centers = bin_edges[:-1]
    plt.figure(figsize=(10, 5))
    plt.bar(centers, counts, width=widths, align="edge", color="#7c3aed", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def save_hour_chart(hour_counts: pd.Series, output_path: Path) -> None:
    if hour_counts.empty:
        return
    hours = pd.Series(0, index=range(24), dtype="int64")
    hours.loc[hour_counts.index.astype(int)] = hour_counts.values
    plt.figure(figsize=(10, 4.5))
    plt.bar(hours.index.astype(str), hours.values, color="#ea580c")
    plt.title("Posting Hours (UTC)")
    plt.xlabel("Hour")
    plt.ylabel("Posts")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def save_heatmap(frame: pd.DataFrame, index_col: str, column_col: str, value_col: str, output_path: Path, title: str) -> None:
    if frame.empty:
        return
    top_index = frame.groupby(index_col)[value_col].sum().nlargest(10).index
    top_columns = frame.groupby(column_col)[value_col].sum().nlargest(10).index
    filtered = frame[frame[index_col].isin(top_index) & frame[column_col].isin(top_columns)]
    if filtered.empty:
        return
    pivot = filtered.pivot_table(index=index_col, columns=column_col, values=value_col, aggfunc="sum", fill_value=0)
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot.values, aspect="auto", cmap="Blues")
    plt.title(title)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label="Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def write_summary_markdown(
    report_dir: Path,
    summary: dict,
    rare_languages: set[str],
    eda: EDAAccumulator,
    clean_output_path: Path,
    batch_size: int,
) -> None:
    language_series = counter_to_series(eda.language_counts, "posts")
    platform_series = counter_to_series(eda.platform_counts, "posts")
    theme_series = counter_to_series(eda.theme_counts, "posts")
    emotion_series = counter_to_series(eda.emotion_counts, "posts")
    daily_series = counter_to_series(eda.daily_counts, "posts")

    mean_sentiment = eda.sentiment_sum / eda.sentiment_count if eda.sentiment_count else 0.0
    missing_author_ratio = summary["missing_author_rows"] / max(summary["cleaned_rows"], 1)
    empty_text_ratio = summary["empty_text_rows"] / max(summary["cleaned_rows"], 1)
    short_text_ratio = summary["short_text_rows"] / max(summary["cleaned_rows"], 1)
    rare_language_ratio = summary["rare_language_rows"] / max(summary["cleaned_rows"], 1)

    lines = [
        "# Fast Cleaning + EDA Report",
        "",
        "This report was generated with a RAM-friendly streaming pipeline based on the project markdown decisions.",
        "",
        "## Run Config",
        "",
        f"- `clean_output`: `{clean_output_path}`",
        f"- `batch_size`: `{batch_size:,}`",
        f"- `rare_language_threshold`: languages appearing fewer than configured threshold rows are flagged",
        f"- `unique_authors_seen`: `{len(eda.unique_authors):,}`",
        "",
        "## Cleaning Summary",
        "",
        f"- `raw_rows`: `{summary['raw_rows']:,}`",
        f"- `cleaned_rows`: `{summary['cleaned_rows']:,}`",
        f"- `technical_duplicates_removed`: `{summary['technical_duplicates_removed']:,}`",
        f"- `missing_author_rows`: `{summary['missing_author_rows']:,}` ({missing_author_ratio:.2%})",
        f"- `empty_text_rows`: `{summary['empty_text_rows']:,}` ({empty_text_ratio:.2%})",
        f"- `short_text_rows`: `{summary['short_text_rows']:,}` ({short_text_ratio:.2%})",
        f"- `bad_date_rows`: `{summary['bad_date_rows']:,}`",
        f"- `rare_language_rows`: `{summary['rare_language_rows']:,}` ({rare_language_ratio:.2%})",
        "",
        "## Quick Findings",
        "",
        f"- Mean sentiment: `{mean_sentiment:.4f}`",
        f"- Top language: `{language_series.index[0] if not language_series.empty else 'n/a'}`",
        f"- Top platform: `{platform_series.index[0] if not platform_series.empty else 'n/a'}`",
        f"- Top theme: `{theme_series.index[0] if not theme_series.empty else 'n/a'}`",
        f"- Date coverage points: `{len(daily_series):,}` days",
        f"- Rare language labels identified: `{len(rare_languages):,}`",
        "",
        "## Top Languages",
        "",
    ]

    for idx, value in language_series.head(10).items():
        lines.append(f"- `{idx}`: `{int(value):,}`")

    lines.extend(["", "## Top Platforms", ""])
    for idx, value in platform_series.head(10).items():
        lines.append(f"- `{idx}`: `{int(value):,}`")

    lines.extend(["", "## Top Themes", ""])
    for idx, value in theme_series.head(10).items():
        lines.append(f"- `{idx}`: `{int(value):,}`")

    lines.extend(["", "## Top Emotions", ""])
    for idx, value in emotion_series.head(10).items():
        lines.append(f"- `{idx}`: `{int(value):,}`")

    lines.extend(
        [
            "",
            "## Charts",
            "",
            "![Languages](language_top15.png)",
            "",
            "![Platforms](platform_top15.png)",
            "",
            "![Themes](theme_top15.png)",
            "",
            "![Emotions](emotion_top15.png)",
            "",
            "![Daily Posts](daily_posts.png)",
            "",
            "![Posting Hours](posting_hours_utc.png)",
            "",
            "![Sentiment Histogram](sentiment_hist.png)",
            "",
            "![Text Length Histogram](text_length_hist.png)",
            "",
            "![Language Platform Heatmap](language_platform_heatmap.png)",
            "",
            "![Theme Emotion Heatmap](theme_emotion_heatmap.png)",
            "",
        ]
    )

    (report_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_report(report_dir: Path, summary: dict, rare_languages: set[str], eda: EDAAccumulator, clean_output_path: Path, batch_size: int) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    language_series = counter_to_series(eda.language_counts, "posts")
    platform_series = counter_to_series(eda.platform_counts, "posts")
    theme_series = counter_to_series(eda.theme_counts, "posts")
    emotion_series = counter_to_series(eda.emotion_counts, "posts")
    daily_series = counter_to_series(eda.daily_counts, "posts")
    hour_series = counter_to_series(eda.hour_counts, "posts")
    lang_platform_frame = pair_counter_to_frame(eda.lang_platform_counts, "language", "platform", "posts")
    theme_emotion_frame = pair_counter_to_frame(eda.theme_emotion_counts, "theme", "emotion", "posts")

    save_bar_chart(language_series, "Top Languages", report_dir / "language_top15.png", "Posts")
    save_bar_chart(platform_series, "Top Platforms", report_dir / "platform_top15.png", "Posts")
    save_bar_chart(theme_series, "Top Themes", report_dir / "theme_top15.png", "Posts")
    save_bar_chart(emotion_series, "Top Emotions", report_dir / "emotion_top15.png", "Posts")
    save_daily_posts_chart(daily_series, report_dir / "daily_posts.png")
    save_hour_chart(hour_series, report_dir / "posting_hours_utc.png")
    save_histogram_from_counts(eda.sentiment_bins, eda.sentiment_hist, "Sentiment Distribution", report_dir / "sentiment_hist.png", "Sentiment")
    save_histogram_from_counts(
        eda.text_length_bins, eda.text_length_hist, "Normalized Text Length Distribution", report_dir / "text_length_hist.png", "Text length"
    )
    save_heatmap(lang_platform_frame, "language", "platform", "posts", report_dir / "language_platform_heatmap.png", "Language x Platform")
    save_heatmap(theme_emotion_frame, "theme", "emotion", "posts", report_dir / "theme_emotion_heatmap.png", "Theme x Emotion")

    language_series.head(25).rename_axis("language").reset_index(name="posts").to_csv(report_dir / "language_top25.csv", index=False)
    platform_series.head(25).rename_axis("platform").reset_index(name="posts").to_csv(report_dir / "platform_top25.csv", index=False)
    theme_series.head(25).rename_axis("theme").reset_index(name="posts").to_csv(report_dir / "theme_top25.csv", index=False)
    emotion_series.head(25).rename_axis("emotion").reset_index(name="posts").to_csv(report_dir / "emotion_top25.csv", index=False)
    lang_platform_frame.head(100).to_csv(report_dir / "language_platform_top100.csv", index=False)
    theme_emotion_frame.head(100).to_csv(report_dir / "theme_emotion_top100.csv", index=False)

    write_summary_markdown(report_dir, summary, rare_languages, eda, clean_output_path, batch_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast cleaning + EDA pipeline for Dataleague final")
    parser.add_argument("--input", type=Path, default=Path("data/datathonFINAL.parquet"))
    parser.add_argument("--clean-output", type=Path, default=Path("data/datathonFINAL_cleaned_fast.parquet"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/fast_eda"))
    parser.add_argument("--batch-size", type=int, default=0, help="0 means auto")
    parser.add_argument("--min-text-len", type=int, default=5)
    parser.add_argument("--rare-language-threshold", type=int, default=100)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means full run")
    return parser.parse_known_args()[0]


def main() -> None:
    args = parse_args()
    batch_size = args.batch_size or estimate_batch_size()
    max_batches = args.max_batches or None

    parquet_file = pq.ParquetFile(args.input)
    print(f"input={args.input}")
    print(f"rows={parquet_file.metadata.num_rows:,}")
    print(f"row_groups={parquet_file.metadata.num_row_groups}")
    print(f"available_memory_gb={get_available_memory_bytes() / 1024**3:.2f}")
    print(f"batch_size={batch_size:,}")
    print(f"max_batches={max_batches if max_batches is not None else 'full'}")
    print(f"tldextract_enabled={tldextract is not None}")

    start = time.time()
    language_counts = collect_language_counts(args.input, batch_size=batch_size if max_batches is None else min(batch_size, 50_000))
    rare_languages = set(language_counts[language_counts < args.rare_language_threshold].index.astype("string").tolist())
    print(f"rare_language_count={len(rare_languages):,}")

    summary, eda = clean_write_and_collect(
        input_path=args.input,
        clean_output_path=args.clean_output,
        batch_size=batch_size,
        min_text_len=args.min_text_len,
        rare_languages=rare_languages,
        report_dir=args.report_dir,
        max_batches=max_batches,
    )
    build_report(args.report_dir, summary, rare_languages, eda, args.clean_output, batch_size)

    elapsed = (time.time() - start) / 60
    print("\nsummary")
    print(pd.Series(summary).to_string())
    print(f"\nreport_markdown={args.report_dir / 'summary.md'}")
    print(f"total_elapsed_min={elapsed:.2f}")


if __name__ == "__main__":
    main()
