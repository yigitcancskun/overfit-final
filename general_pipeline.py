from __future__ import annotations

import argparse
import gc
import os
import re
import time
import unicodedata
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

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

    # Conservative fallback for systems where memory can't be inspected.
    return 4 * 1024**3


def estimate_batch_size(
    min_rows: int = 50_000,
    max_rows: int = 200_000,
    estimated_bytes_per_row: int = 2_500,
    target_memory_fraction: float = 0.03,
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
    s = series.astype("string")
    s = s.str.strip()
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
        registered_domain = tail_three
        subdomain = ".".join(parts[:-3]) or pd.NA
    elif ".".join(parts[-2:]) in COMMON_MULTI_PART_SUFFIXES and len(parts) >= 3:
        registered_domain = tail_three
        subdomain = ".".join(parts[:-3]) or pd.NA
    else:
        registered_domain = tail_two
        subdomain = ".".join(parts[:-2]) or pd.NA

    return (host, registered_domain, subdomain)


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
    row_key = (
        original_text
        + "\x1f"
        + author_hash
        + "\x1f"
        + date
        + "\x1f"
        + url
    )
    return pd.util.hash_pandas_object(row_key, index=False)


def iter_pandas_batches(path: Path, batch_size: int):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield batch.to_pandas(types_mapper=pd.ArrowDtype)


def collect_language_counts(path: Path, batch_size: int) -> pd.Series:
    language_counts = pd.Series(dtype="int64")
    start = time.time()
    for batch_id, batch_df in enumerate(iter_pandas_batches(path, batch_size=batch_size), start=1):
        batch_lang = batch_df["language"].astype("string").str.strip().value_counts(dropna=False)
        language_counts = language_counts.add(batch_lang, fill_value=0)
        if batch_id % 10 == 0:
            print(f"[pass1] batch={batch_id} unique_languages={len(language_counts):,}")
        del batch_df, batch_lang
        gc.collect()
    elapsed = (time.time() - start) / 60
    print(f"[pass1] completed in {elapsed:.2f} min")
    return language_counts.sort_values(ascending=False).astype("int64")


def clean_and_write(
    input_path: Path,
    output_path: Path,
    batch_size: int,
    min_text_len: int,
    rare_languages: set,
) -> dict:
    seen_fingerprints = set()
    writer = None
    summary = {
        "raw_rows": 0,
        "written_rows": 0,
        "technical_duplicates_removed": 0,
        "missing_author_rows": 0,
        "empty_text_rows": 0,
        "short_text_rows": 0,
        "bad_date_rows": 0,
        "rare_language_rows": 0,
    }

    start = time.time()
    if output_path.exists():
        output_path.unlink()

    try:
        for batch_id, batch_df in enumerate(iter_pandas_batches(input_path, batch_size=batch_size), start=1):
            batch_df.columns = [str(col) for col in batch_df.columns]
            summary["raw_rows"] += len(batch_df)

            for column in BLANK_SENSITIVE_COLUMNS:
                batch_df[column] = normalize_missing_series(batch_df[column])

            batch_df["date"] = pd.to_datetime(batch_df["date"], utc=True, errors="coerce")
            batch_df["normalized_text"] = normalize_text_series(batch_df["original_text"])
            batch_df["english_keywords_clean"] = clean_keywords_series(batch_df["english_keywords"])
            batch_df["primary_theme"] = batch_df["primary_theme"].fillna("unknown_theme")

            batch_df["author_hash_missing_flag"] = batch_df["author_hash"].isna().astype("int8")
            batch_df["is_empty_text"] = batch_df["normalized_text"].isna().astype("int8")
            batch_df["is_short_text"] = batch_df["normalized_text"].str.len().fillna(0).lt(min_text_len).astype("int8")
            batch_df["is_rare_language"] = batch_df["language"].isin(rare_languages).astype("int8")

            domain_parts = extract_domain_columns(batch_df["url"])
            batch_df = batch_df.drop(columns=["raw_url_domain", "registered_domain", "subdomain"], errors="ignore")
            batch_df = pd.concat([batch_df, domain_parts], axis=1)

            fingerprints = compute_row_fingerprint(batch_df)
            is_duplicate = fingerprints.isin(seen_fingerprints)
            batch_df["technical_duplicate_flag"] = is_duplicate.astype("int8")

            new_fingerprints = fingerprints.loc[~is_duplicate].tolist()
            seen_fingerprints.update(new_fingerprints)

            cleaned_batch = batch_df.loc[~is_duplicate].copy()

            summary["technical_duplicates_removed"] += int(is_duplicate.sum())
            summary["missing_author_rows"] += int(cleaned_batch["author_hash_missing_flag"].sum())
            summary["empty_text_rows"] += int(cleaned_batch["is_empty_text"].sum())
            summary["short_text_rows"] += int(cleaned_batch["is_short_text"].sum())
            summary["bad_date_rows"] += int(cleaned_batch["date"].isna().sum())
            summary["rare_language_rows"] += int(cleaned_batch["is_rare_language"].sum())
            summary["written_rows"] += len(cleaned_batch)

            table = pa.Table.from_pandas(cleaned_batch, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)

            if batch_id % 5 == 0:
                elapsed = (time.time() - start) / 60
                print(
                    f"[pass2] batch={batch_id} written={summary['written_rows']:,} "
                    f"duplicates_removed={summary['technical_duplicates_removed']:,} elapsed={elapsed:.2f} min"
                )

            del batch_df, domain_parts, fingerprints, is_duplicate, cleaned_batch, table, new_fingerprints
            gc.collect()
    finally:
        if writer is not None:
            writer.close()

    elapsed = (time.time() - start) / 60
    print(f"[pass2] completed in {elapsed:.2f} min")
    print(f"[pass2] output={output_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAM-dostu Dataleague cleaning pipeline")
    parser.add_argument("--input", type=Path, default=Path("datathonFINAL.parquet"))
    parser.add_argument("--output", type=Path, default=Path("datathonFINAL_cleaned.parquet"))
    parser.add_argument("--batch-size", type=int, default=0, help="0 ise otomatik hesaplanir")
    parser.add_argument("--min-text-len", type=int, default=5)
    parser.add_argument("--rare-language-threshold", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_size = args.batch_size or estimate_batch_size()

    parquet_file = pq.ParquetFile(args.input)
    print(f"input={args.input}")
    print(f"rows={parquet_file.metadata.num_rows:,}")
    print(f"row_groups={parquet_file.metadata.num_row_groups}")
    print(f"available_memory_gb={get_available_memory_bytes() / 1024**3:.2f}")
    print(f"batch_size={batch_size:,}")
    print(f"tldextract_enabled={tldextract is not None}")

    language_counts = collect_language_counts(args.input, batch_size=batch_size)
    rare_languages = set(language_counts[language_counts < args.rare_language_threshold].index.tolist())
    print(f"rare_language_count={len(rare_languages):,}")

    summary = clean_and_write(
        input_path=args.input,
        output_path=args.output,
        batch_size=batch_size,
        min_text_len=args.min_text_len,
        rare_languages=rare_languages,
    )

    print("\nsummary")
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
