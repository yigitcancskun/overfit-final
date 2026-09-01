from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .artifacts import (
    _author_scores_path_from_config,
    _manifest_path_from_config,
    _parquet_columns,
    _scored_messages_path_from_config,
    _sqlite_path_from_config,
    build_batch_summary_tables,
    build_manifest_payload,
    build_manifest_summary_table,
    compute_message_refs_from_sqlite,
    get_sqlite_connection,
    get_store_columns,
    load_manifest,
    log_progress,
    run_batch_pass1,
    run_batch_pass2_author,
    run_batch_pass2_message,
    validate_existing_store,
    validate_scored_outputs,
    write_manifest,
)
from .data_processing import build_author_features, build_message_features, clean_dataset, compute_top_domain_coverage
from .scoring import compute_author_scores, compute_final_scores, compute_message_scores, fit_normalization_reference

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

