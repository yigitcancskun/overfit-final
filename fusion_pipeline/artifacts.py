from __future__ import annotations

import gc
import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .constants import (
    CLEANED_MESSAGE_SCHEMA,
    DEBUG_EXTRA_STORE_COLUMNS,
    LEAN_STORE_COLUMNS,
    PIPELINE_VERSION,
    REQUIRED_AUTHOR_SCORE_COLUMNS,
    REQUIRED_BATCH_TABLES,
    REQUIRED_CLEANED_MESSAGE_COLUMNS,
    REQUIRED_SCORED_MESSAGE_COLUMNS,
    SCORE_SCHEMA_VERSION,
    STORE_SCHEMA_VERSION,
)
from .data_processing import clean_batch, compute_row_fingerprint, compute_text_hash_scalar, estimate_batch_size, iter_pandas_batches
from .scoring import apply_final_score_weighting, compute_author_scores, compute_behavioral_score, compute_message_scores, fit_normalization_reference, log_penalty

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
        "hard_bot_examples": scored_df.loc[
            scored_df["hard_bot_cluster_flag"].eq(1) | scored_df["hard_same_text_repeat_flag"].eq(1),
            [
            "author_hash",
            "normalized_text",
            "same_text_repeat_count",
            "same_text_unique_author_count",
            "same_text_time_window_count",
            "hard_same_text_repeat_flag",
            "final_score",
        ],
        ].head(50),
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


def _manifest_path_from_config(config: dict[str, Any]) -> Path:
    manifest_path = config["paths"].get("manifest_path", "data/fusion_manifest.json")
    return Path(manifest_path)


def _author_feature_stage_path_from_config(config: dict[str, Any]) -> Path:
    stage_path = config["paths"].get("author_feature_stage_parquet", "data/fusion_author_feature_stage.parquet")
    return Path(stage_path)


def get_store_columns(config: dict[str, Any]) -> tuple[str, ...]:
    build_mode = config["runtime"].get("build_mode", "lean")
    if build_mode == "debug":
        return LEAN_STORE_COLUMNS + DEBUG_EXTRA_STORE_COLUMNS
    return LEAN_STORE_COLUMNS


def log_progress(stage: str, message: str, *, config: dict[str, Any] | None = None) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    if config is not None and config["runtime"].get("enable_progress_logs", True):
        print(f"[{timestamp}] [{stage}] {message}")
    elif config is None:
        print(f"[{timestamp}] [{stage}] {message}")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _extract_store_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "paths": {
            "input_parquet": config["paths"]["input_parquet"],
        },
        "runtime": {
            "top_n_domain_context": config["runtime"].get("top_n_domain_context"),
            "build_mode": config["runtime"].get("build_mode", "lean"),
        },
        "thresholds": {
            "min_text_len": config["thresholds"]["min_text_len"],
            "rare_language_threshold": config["thresholds"]["rare_language_threshold"],
            "long_text_start": config["thresholds"]["long_text_start"],
            "hard_bot_time_window_sec": config["thresholds"]["hard_bot_time_window_sec"],
        },
        "rules": config.get("rules", {}),
        "semantic_adapter": config["semantic_adapter"],
        "versions": {
            "store_schema_version": STORE_SCHEMA_VERSION,
        },
        "store_columns": list(get_store_columns(config)),
    }


def _extract_scoring_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "thresholds": {
            "hourly_penalty_start": config["thresholds"]["hourly_penalty_start"],
            "hard_hourly_bot_threshold": config["thresholds"].get("hard_hourly_bot_threshold", 15),
            "language_penalty_start": config["thresholds"]["language_penalty_start"],
            "hard_bot_repeat_threshold": config["thresholds"]["hard_bot_repeat_threshold"],
            "hard_bot_multi_author_threshold": config["thresholds"]["hard_bot_multi_author_threshold"],
            "hard_bot_time_cluster_threshold": config["thresholds"]["hard_bot_time_cluster_threshold"],
            "spam_repeat_threshold": config["thresholds"]["spam_repeat_threshold"],
            "spam_multi_author_threshold": config["thresholds"]["spam_multi_author_threshold"],
            "spam_time_cluster_threshold": config["thresholds"]["spam_time_cluster_threshold"],
        },
        "rules": config.get("rules", {}),
        "neutral_score_policy": config.get("neutral_score_policy", {}),
        "dominant_signal_policy": config.get("dominant_signal_policy", {}),
        "weights": config["weights"],
        "versions": {
            "score_schema_version": SCORE_SCHEMA_VERSION,
        },
    }


def compute_config_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(_json_ready(payload), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found at {path}. Run run_formula_pipeline_two_pass(CONFIG).")
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_ready(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def raise_schema_mismatch(subject: str, missing_columns: list[str], remedy: str) -> None:
    missing_repr = ", ".join(repr(column) for column in missing_columns)
    raise ValueError(f"{subject}: missing [{missing_repr}]. {remedy}")


def _sqlite_table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row[0] for row in rows}


def _sqlite_table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [row[1] for row in rows]


def _parquet_columns(path: Path) -> list[str]:
    return list(pq.ParquetFile(path).schema_arrow.names)


def build_manifest_payload(
    config: dict[str, Any],
    *,
    derived_thresholds: dict[str, Any] | None = None,
    created_at: str | None = None,
    actual_store_columns: list[str] | None = None,
    actual_score_columns: list[str] | None = None,
) -> dict[str, Any]:
    created_value = created_at or datetime.now(timezone.utc).isoformat()
    store_columns = actual_store_columns or list(get_store_columns(config))
    score_columns = actual_score_columns or list(REQUIRED_SCORED_MESSAGE_COLUMNS)
    return {
        "pipeline_version": PIPELINE_VERSION,
        "store_schema_version": STORE_SCHEMA_VERSION,
        "score_schema_version": SCORE_SCHEMA_VERSION,
        "created_at": created_value,
        "sqlite_path": str(_sqlite_path_from_config(config)),
        "author_scores_path": str(_author_scores_path_from_config(config)),
        "scored_messages_path": str(_scored_messages_path_from_config(config)),
        "config_hash_for_store": compute_config_hash(_extract_store_config(config)),
        "config_hash_for_scoring": compute_config_hash(_extract_scoring_config(config)),
        "required_cleaned_message_columns": list(REQUIRED_CLEANED_MESSAGE_COLUMNS),
        "required_author_score_columns": list(REQUIRED_AUTHOR_SCORE_COLUMNS),
        "required_scored_message_columns": list(REQUIRED_SCORED_MESSAGE_COLUMNS),
        "store_columns": store_columns,
        "store_schema_fingerprint": compute_config_hash({"store_columns": sorted(store_columns)}),
        "score_schema_fingerprint": compute_config_hash({"scored_columns": sorted(score_columns)}),
        "derived_thresholds": _json_ready(derived_thresholds or config.get("derived_thresholds", {})),
    }


def validate_existing_store(config: dict[str, Any], manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    db_path = _sqlite_path_from_config(config)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite store not found at {db_path}. Run run_formula_pipeline_two_pass(CONFIG).")

    if manifest is not None:
        expected_store_hash = compute_config_hash(_extract_store_config(config))
        if manifest.get("store_schema_version") != STORE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Store schema version mismatch: manifest={manifest.get('store_schema_version')} code={STORE_SCHEMA_VERSION}. Full rebuild required."
            )
        if manifest.get("config_hash_for_store") != expected_store_hash:
            raise RuntimeError("Store config hash mismatch. Full rebuild required.")

    conn = get_sqlite_connection(db_path)
    try:
        table_names = _sqlite_table_names(conn)
        missing_tables = sorted(set(REQUIRED_BATCH_TABLES) - table_names)
        if missing_tables:
            raise RuntimeError(
                f"SQLite store is incomplete: missing tables {missing_tables}. Run run_formula_pipeline_two_pass(CONFIG)."
            )

        cleaned_columns = _sqlite_table_columns(conn, "cleaned_messages")
        missing_cleaned_columns = sorted(set(REQUIRED_CLEANED_MESSAGE_COLUMNS) - set(cleaned_columns))
        if missing_cleaned_columns:
            raise_schema_mismatch(
                "Store schema mismatch in cleaned_messages",
                missing_cleaned_columns,
                "Full rebuild required.",
            )

        if manifest is not None:
            actual_store_fingerprint = compute_config_hash(
                {"store_columns": sorted([column for column in cleaned_columns if column != "message_id"])}
            )
            if manifest.get("store_schema_fingerprint") != actual_store_fingerprint:
                raise RuntimeError("Store schema fingerprint mismatch. Full rebuild required.")

        return {
            "sqlite_path": str(db_path),
            "table_names": sorted(table_names),
            "cleaned_message_columns": cleaned_columns,
        }
    finally:
        conn.close()


def validate_scored_outputs(config: dict[str, Any], manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    author_path = _author_scores_path_from_config(config)
    scored_path = _scored_messages_path_from_config(config)
    if not author_path.exists() or not scored_path.exists():
        missing_paths = [str(path) for path in [author_path, scored_path] if not path.exists()]
        raise FileNotFoundError(
            f"Scored outputs missing: {missing_paths}. Run run_rescore_from_existing_store(CONFIG)."
        )

    if manifest is not None:
        expected_score_hash = compute_config_hash(_extract_scoring_config(config))
        if manifest.get("score_schema_version") != SCORE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Score schema version mismatch: manifest={manifest.get('score_schema_version')} code={SCORE_SCHEMA_VERSION}. Run run_rescore_from_existing_store(CONFIG)."
            )
        if manifest.get("config_hash_for_scoring") != expected_score_hash:
            raise RuntimeError("Scoring config hash mismatch. Run run_rescore_from_existing_store(CONFIG).")

    author_columns = _parquet_columns(author_path)
    missing_author_columns = sorted(set(REQUIRED_AUTHOR_SCORE_COLUMNS) - set(author_columns))
    if missing_author_columns:
        raise_schema_mismatch(
            "author_scores.parquet",
            missing_author_columns,
            "Run run_rescore_from_existing_store(CONFIG).",
        )

    scored_columns = _parquet_columns(scored_path)
    missing_scored_columns = sorted(set(REQUIRED_SCORED_MESSAGE_COLUMNS) - set(scored_columns))
    if missing_scored_columns:
        raise_schema_mismatch(
            "scored_messages.parquet",
            missing_scored_columns,
            "Run run_rescore_from_existing_store(CONFIG).",
        )

    if manifest is not None:
        actual_score_fingerprint = compute_config_hash({"scored_columns": sorted(scored_columns)})
        if manifest.get("score_schema_fingerprint") != actual_score_fingerprint:
            raise RuntimeError("Score schema fingerprint mismatch. Run run_rescore_from_existing_store(CONFIG).")

    return {
        "author_columns": author_columns,
        "scored_message_columns": scored_columns,
    }


def assert_artifacts_ready(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = _manifest_path_from_config(config)
    manifest = load_manifest(manifest_path)
    store_info = validate_existing_store(config, manifest)
    score_info = validate_scored_outputs(config, manifest)
    result["manifest"] = manifest
    result["store_validation"] = store_info
    result["score_validation"] = score_info
    return result


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

    manifest_path = _manifest_path_from_config(config)
    if manifest_path.exists() and config["runtime"]["overwrite_outputs"]:
        manifest_path.unlink()

    author_feature_stage_path = _author_feature_stage_path_from_config(config)
    if author_feature_stage_path.exists() and config["runtime"]["overwrite_outputs"]:
        author_feature_stage_path.unlink()

    conn = get_sqlite_connection(db_path)
    try:
        store_columns = get_store_columns(config)
        schema_parts = ["message_id INTEGER PRIMARY KEY AUTOINCREMENT"]
        for column in store_columns:
            schema_parts.append(f"{column} {CLEANED_MESSAGE_SCHEMA[column]}")
        create_table_sql = ",\n                ".join(schema_parts)
        conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS cleaned_messages (
                {create_table_sql}
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

    columns = list(get_store_columns(config))
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
        "semantic_applied_rows": 0,
        "batch_size": batch_size,
    }
    conn = get_sqlite_connection(db_path)
    start = time.time()
    try:
        store_columns = list(get_store_columns(config))
        insert_columns_sql = ", ".join(store_columns)
        insert_placeholders_sql = ", ".join("?" for _ in store_columns)
        insert_sql = """
            INSERT OR IGNORE INTO cleaned_messages (
                """ + insert_columns_sql + """
            ) VALUES (""" + insert_placeholders_sql + """)
        """
        progress_every = max(int(config["runtime"].get("progress_every_batches", 5)), 1)
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
            summary["semantic_applied_rows"] += int(cleaned_batch["semantic_model_applied_flag"].sum())

            if batch_id == 1 or batch_id % progress_every == 0:
                db_size_mb = round(db_path.stat().st_size / 1024**2, 1) if db_path.exists() else 0.0
                elapsed_min = round((time.time() - start) / 60, 2)
                log_progress(
                    "PASS1",
                    f"batch={batch_id} raw_rows={summary['raw_rows']:,} clean_rows={summary['clean_rows']:,} duplicates_removed={summary['technical_duplicates_removed']:,} semantic_applied_rows={summary['semantic_applied_rows']:,} db_size_mb={db_size_mb} elapsed_min={elapsed_min}",
                    config=config,
                )

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
    log_progress("CLUSTERS", "materializing text_clusters and text_window_clusters", config=config)
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
        log_progress("CLUSTERS", "cluster tables ready", config=config)
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


def _append_parquet_table(
    path: Path,
    frame: pd.DataFrame,
    writer: pq.ParquetWriter | None,
    *,
    compression: str = "snappy",
) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression=compression)
    writer.write_table(table)
    return writer


def run_batch_pass2_author(config: dict[str, Any], *, refresh_clusters: bool = True) -> pd.DataFrame:
    conn = get_sqlite_connection(_sqlite_path_from_config(config))
    author_scores_path = _author_scores_path_from_config(config)
    author_feature_stage_path = _author_feature_stage_path_from_config(config)
    author_batch_size = config["runtime"].get("author_batch_size", 5000)
    feature_writer = None
    score_writer = None
    preview_frames: list[pd.DataFrame] = []
    author_feature_frames_for_refs: list[pd.DataFrame] = []
    try:
        if refresh_clusters:
            materialize_batch_cluster_tables(config)
        if author_scores_path.exists():
            author_scores_path.unlink()
        if author_feature_stage_path.exists():
            author_feature_stage_path.unlink()
        author_cursor = conn.execute(
            "SELECT DISTINCT author_hash FROM cleaned_messages WHERE author_type = 'identified' AND author_hash IS NOT NULL ORDER BY author_hash"
        )
        batch_counter = 0
        progress_every = max(int(config["runtime"].get("progress_every_batches", 5)), 1)
        while True:
            rows = author_cursor.fetchmany(author_batch_size)
            if not rows:
                break
            batch_counter += 1
            author_ids = [row[0] for row in rows if row[0] is not None]
            author_batch_frame = _fetch_author_batch_frame(conn, author_ids)
            batch_features = _compute_author_features_from_frame(author_batch_frame)
            if not batch_features.empty:
                feature_writer = _append_parquet_table(author_feature_stage_path, batch_features, feature_writer)
                author_feature_frames_for_refs.append(
                    batch_features[
                        [
                            "posts_per_day",
                            "posts_per_active_hour",
                            "theme_nunique",
                            "sentiment_std",
                            "same_text_repeat_ratio",
                            "same_text_repeat_max",
                            "multi_author_repeat_ratio",
                            "mean_interpost_sec",
                            "median_interpost_sec",
                            "p10_interpost_sec",
                            "interval_cv",
                            "max_posts_one_hour",
                            "language_nunique",
                        ]
                    ].copy()
                )
            if batch_counter == 1 or batch_counter % progress_every == 0:
                log_progress(
                    "PASS2_AUTHOR_STAGE",
                    f"author_batches={batch_counter} staged_authors={sum(len(frame) for frame in author_feature_frames_for_refs):,}",
                    config=config,
                )
            del author_batch_frame, batch_features
            gc.collect()

        ref_source = pd.concat(author_feature_frames_for_refs, ignore_index=True) if author_feature_frames_for_refs else pd.DataFrame()
        refs = fit_normalization_reference(ref_source, pd.DataFrame(), config)
        if feature_writer is not None:
            feature_writer.close()
            feature_writer = None

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

        scored_batches = 0
        total_scored_authors = 0
        for feature_batch in iter_pandas_batches(author_feature_stage_path, batch_size=author_batch_size):
            scored_batches += 1
            scored_batch = compute_author_scores(feature_batch, refs, config)
            total_scored_authors += len(scored_batch)
            score_writer = _append_parquet_table(author_scores_path, scored_batch, score_writer)
            conn.executemany(
                "INSERT INTO author_scores(author_hash, author_score, author_hard_hourly_flag, max_posts_one_hour, language_nunique, theme_nunique, sentiment_std, median_interpost_sec) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                scored_batch[
                    ["author_hash", "author_score", "author_hard_hourly_flag", "max_posts_one_hour", "language_nunique", "theme_nunique", "sentiment_std", "median_interpost_sec"]
                ].itertuples(index=False, name=None),
            )
            conn.commit()
            if not preview_frames:
                preview_frames.append(scored_batch.head(50).copy())
            if scored_batches == 1 or scored_batches % progress_every == 0:
                log_progress(
                    "PASS2_AUTHOR_SCORE",
                    f"score_batches={scored_batches} scored_authors={total_scored_authors:,} hourly_hard_knee={config.get('derived_thresholds', {}).get('hourly_hard_knee')}",
                    config=config,
                )
            del feature_batch, scored_batch
            gc.collect()

        if score_writer is not None:
            score_writer.close()
            score_writer = None
        if author_feature_stage_path.exists():
            author_feature_stage_path.unlink()
        return pd.read_parquet(author_scores_path)
    finally:
        if feature_writer is not None:
            feature_writer.close()
        if score_writer is not None:
            score_writer.close()
        if author_feature_stage_path.exists():
            author_feature_stage_path.unlink()
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
        if scored_path.exists():
            scored_path.unlink()
        batch_counter = 0
        total_scored_rows = 0
        progress_every = max(int(config["runtime"].get("progress_every_batches", 5)), 1)
        while True:
            query = """
                SELECT
                    cm.message_id,
                    cm.normalized_text,
                    cm.roberta_score,
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
                    (LENGTH(cm.normalized_text) - LENGTH(REPLACE(cm.normalized_text, '!', ''))) AS exclamation_count,
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
            batch_counter += 1
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
            final_df["behavioral_score"] = compute_behavioral_score(final_df, config)
            final_df = apply_final_score_weighting(final_df, config)

            heavy_mask = final_df["spam_pattern_flag"].eq(1) | final_df["same_text_repeat_count"].ge(config["thresholds"]["spam_repeat_threshold"])
            heavy_keyword_stats["heavy_keyword_sum"] += float(final_df.loc[heavy_mask, "keyword_count"].sum())
            heavy_keyword_stats["heavy_rows"] += int(heavy_mask.sum())
            heavy_keyword_stats["light_keyword_sum"] += float(final_df.loc[~heavy_mask, "keyword_count"].sum())
            heavy_keyword_stats["light_rows"] += int((~heavy_mask).sum())

            if not preview_frames:
                preview_frames.append(final_df.head(20).copy())
            hard_bot_sample = final_df.loc[
                final_df["hard_bot_cluster_flag"].eq(1) | final_df["hard_same_text_repeat_flag"].eq(1)
            ].head(50)
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
                "exclamation_count",
                "max_token_frequency",
                "max_token_ratio",
                "spam_pattern_flag",
                "hard_bot_cluster_flag",
                "hard_same_text_repeat_flag",
                "author_hard_hourly_flag",
                "author_score",
                "message_score",
                "behavioral_score",
                "roberta_score",
                "behavioral_confidence_weight",
                "roberta_confidence_weight",
                "behavioral_effective_weight",
                "roberta_effective_weight",
                "final_score_before_rules",
                "final_score",
            ]
            table = pa.Table.from_pandas(final_df[output_columns], preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(scored_path, table.schema, compression="snappy")
            writer.write_table(table)
            total_scored_rows += len(final_df)
            if batch_counter == 1 or batch_counter % progress_every == 0:
                log_progress(
                    "PASS2_MESSAGE",
                    f"message_batches={batch_counter} scored_rows={total_scored_rows:,} last_message_id={last_id:,}",
                    config=config,
                )
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
        semantic_summary = pd.read_sql_query(
            """
            SELECT
                COUNT(*) AS rows,
                AVG(roberta_score) AS avg_roberta_score,
                MIN(roberta_score) AS min_roberta_score,
                MAX(roberta_score) AS max_roberta_score,
                SUM(semantic_model_applied_flag) AS semantic_applied_rows
            FROM cleaned_messages
            """,
            conn,
        )
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
        "semantic_adapter_summary": semantic_summary,
        "hashtag_spam_examples": message_artifacts["hashtag_spam_examples"],
        "token_spam_examples": message_artifacts["token_spam_examples"],
    }
    return tables


def build_manifest_summary_table(manifest: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"field": "pipeline_version", "value": manifest.get("pipeline_version")},
            {"field": "store_schema_version", "value": manifest.get("store_schema_version")},
            {"field": "score_schema_version", "value": manifest.get("score_schema_version")},
            {"field": "store_schema_fingerprint", "value": manifest.get("store_schema_fingerprint")},
            {"field": "score_schema_fingerprint", "value": manifest.get("score_schema_fingerprint")},
            {"field": "created_at", "value": manifest.get("created_at")},
            {"field": "last_rescore_at", "value": manifest.get("last_rescore_at")},
            {"field": "config_hash_for_store", "value": manifest.get("config_hash_for_store")},
            {"field": "config_hash_for_scoring", "value": manifest.get("config_hash_for_scoring")},
            {"field": "hourly_hard_knee", "value": manifest.get("derived_thresholds", {}).get("hourly_hard_knee")},
        ]
    )


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
    log_progress("PIPELINE", f"full build started build_mode={config['runtime'].get('build_mode', 'lean')}", config=config)
    pass1_summary = run_batch_pass1(config)
    author_scores = run_batch_pass2_author(config)
    refs = compute_message_refs_from_sqlite(config)
    message_artifacts = run_batch_pass2_message(config, refs)
    tables = build_batch_summary_tables(config, pass1_summary, author_scores, message_artifacts)
    manifest = build_manifest_payload(
        config,
        actual_store_columns=list(get_store_columns(config)),
        actual_score_columns=_parquet_columns(_scored_messages_path_from_config(config)),
    )
    manifest = write_manifest(_manifest_path_from_config(config), manifest)
    tables["manifest_summary"] = build_manifest_summary_table(manifest)
    return {
        "summary": pass1_summary,
        "tables": tables,
        "author_scores": author_scores,
        "scored_preview": message_artifacts["scored_preview"],
        "manifest": manifest,
        "paths": {
            "sqlite_db": str(_sqlite_path_from_config(config)),
            "author_scores_parquet": str(_author_scores_path_from_config(config)),
            "scored_messages_parquet": str(_scored_messages_path_from_config(config)),
            "manifest_json": str(_manifest_path_from_config(config)),
        },
    }


def run_rescore_from_existing_store(config: dict[str, Any]) -> dict[str, Any]:
    log_progress("PIPELINE", "rescore from existing store started", config=config)
    manifest_path = _manifest_path_from_config(config)
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        validate_existing_store(config, manifest)
    else:
        validate_existing_store(config, None)
        manifest = write_manifest(
            manifest_path,
            build_manifest_payload(
                config,
                actual_store_columns=list(get_store_columns(config)),
                actual_score_columns=_parquet_columns(_scored_messages_path_from_config(config)) if _scored_messages_path_from_config(config).exists() else None,
            ),
        )

    author_scores = run_batch_pass2_author(config, refresh_clusters=False)
    refs = compute_message_refs_from_sqlite(config)
    message_artifacts = run_batch_pass2_message(config, refs)

    conn = get_sqlite_connection(_sqlite_path_from_config(config))
    try:
        clean_rows = int(conn.execute("SELECT COUNT(*) FROM cleaned_messages").fetchone()[0])
        missing_author_rows = int(
            conn.execute(
                "SELECT COUNT(*) FROM cleaned_messages WHERE author_type = 'anonymous' OR author_hash_missing_flag = 1"
            ).fetchone()[0]
        )
        semantic_applied_rows = int(
            conn.execute("SELECT COALESCE(SUM(semantic_model_applied_flag), 0) FROM cleaned_messages").fetchone()[0]
        )
    finally:
        conn.close()

    summary = {
        "clean_rows": clean_rows,
        "missing_author_rows": missing_author_rows,
        "semantic_applied_rows": semantic_applied_rows,
        "technical_duplicates_removed": pd.NA,
        "rescore_only": True,
    }
    tables = build_batch_summary_tables(config, summary, author_scores, message_artifacts)
    manifest = build_manifest_payload(
        config,
        created_at=manifest.get("created_at"),
        actual_store_columns=list(get_store_columns(config)),
        actual_score_columns=_parquet_columns(_scored_messages_path_from_config(config)),
    )
    manifest["last_rescore_at"] = datetime.now(timezone.utc).isoformat()
    manifest = write_manifest(manifest_path, manifest)
    tables["manifest_summary"] = build_manifest_summary_table(manifest)

    validate_scored_outputs(config, manifest)
    return {
        "summary": summary,
        "tables": tables,
        "author_scores": author_scores,
        "scored_preview": message_artifacts["scored_preview"],
        "manifest": manifest,
        "paths": {
            "sqlite_db": str(_sqlite_path_from_config(config)),
            "author_scores_parquet": str(_author_scores_path_from_config(config)),
            "scored_messages_parquet": str(_scored_messages_path_from_config(config)),
            "manifest_json": str(manifest_path),
        },
    }

