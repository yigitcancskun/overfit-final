from __future__ import annotations

import json
from pathlib import Path


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CONFIG_CELL = """from pathlib import Path

CONFIG = {
    "paths": {
        "input_parquet": "data/datathonFINAL.parquet",
        "clean_output_parquet": "data/datathonFINAL_formula_cleaned.parquet",
        "batch_sqlite_db": "data/fusion_batch_store.sqlite",
        "author_scores_parquet": "data/fusion_author_scores.parquet",
        "scored_messages_parquet": "data/fusion_scored_messages.parquet",
        "manifest_path": "data/fusion_manifest.json",
    },
    "runtime": {
        "mode": "full",
        "batch_size": 50_000,
        "max_batches": None,
        "sample_n_rows": None,
        "overwrite_outputs": True,
        "top_n_domain_context": 128,
        "author_batch_size": 5_000,
        "message_batch_size": 100_000,
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
    "derived_thresholds": {},
    "weights": {
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

CONFIG
"""


NOTEBOOK_CELLS = [
    md_cell(
        """# Fusion Formula Tabanli Robot Skorlama Pipeline

Bu notebook iki ayri calisma modu sunar:

- `Full Build`: ilk kez SQLite store ve cluster artefact'larini uretir
- `Rescore From Existing Store`: mevcut `fusion_batch_store.sqlite` korunarak sadece skor katmanini yeniden hesaplar

Tum gorsellestirme hucreleri oncesinde artefact readiness kontrolu zorunludur. Eski-yeni uyumsuzluklarda notebook sessiz fallback yapmaz; yonlendirici hata uretir.
"""
    ),
    code_cell(CONFIG_CELL),
    code_cell(
        """import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq

from formula_scoring_pipeline import (
    assert_artifacts_ready,
    load_manifest,
    plot_hourly_distribution,
    plot_hourly_penalty_curve,
    plot_sentiment_theme_distributions,
    prepare_single_message_input,
    run_formula_pipeline_two_pass,
    run_rescore_from_existing_store,
    score_single_message,
    validate_scored_outputs,
)
"""
    ),
    md_cell(
        """## Input Kontrolu

Veri kaynagini, satir sayisini ve runtime ayarlarini burada kontrol edin.
"""
    ),
    code_cell(
        """input_path = Path(CONFIG["paths"]["input_parquet"])
parquet_file = pq.ParquetFile(input_path)

print(f"input={input_path}")
print(f"rows={parquet_file.metadata.num_rows:,}")
print(f"row_groups={parquet_file.metadata.num_row_groups}")
print(CONFIG)
"""
    ),
    md_cell(
        """## Full Build

Ilk kez store olustururken bu hucreyi calistirin. Bu adim `fusion_batch_store.sqlite` dosyasini yeniden kurar.
"""
    ),
    code_cell(
        """result = run_formula_pipeline_two_pass(CONFIG)
tables = result["tables"]

print("full build completed")
print(result["paths"])
"""
    ),
    md_cell(
        """## Rescore From Existing Store

Weight veya scoring threshold degistiginde bu hucreyi calistirin. Bu adim SQLite store'u yeniden olusturmaz; yalnizca `author_scores.parquet` ve `scored_messages.parquet` dosyalarini overwrite eder.
"""
    ),
    code_cell(
        """result = run_rescore_from_existing_store(CONFIG)
tables = result["tables"]

print("rescore completed")
print(result["paths"])
"""
    ),
    md_cell(
        """## Artefact Readiness Check

Bu hucre tum gorsellestirme ve tablo hucrelerinden once calismalidir. Burasi manifest, SQLite store ve scored outputs uyumlulugunu siki sekilde dogrular.
"""
    ),
    code_cell(
        """result = assert_artifacts_ready(result, CONFIG)
tables = result["tables"]
manifest = result["manifest"]

print("artifacts ready")
print({
    "pipeline_version": manifest["pipeline_version"],
    "store_schema_version": manifest["store_schema_version"],
    "score_schema_version": manifest["score_schema_version"],
    "created_at": manifest["created_at"],
    "last_rescore_at": manifest.get("last_rescore_at"),
})
"""
    ),
    md_cell(
        """## Manifest Ozeti"""
    ),
    code_cell(
        """tables["manifest_summary"]"""
    ),
    code_cell(
        """manifest"""
    ),
    md_cell(
        """## Cleaning ve QA Ozetleri"""
    ),
    code_cell(
        """tables["summary"]"""
    ),
    code_cell(
        """tables["derived_thresholds"]"""
    ),
    code_cell(
        """tables["top128_coverage"]"""
    ),
    code_cell(
        """tables["top_domains"].head(20)"""
    ),
    md_cell(
        """## Author Duzeyi Kontroller"""
    ),
    code_cell(
        """tables["hourly_heavy_authors"].head(20)"""
    ),
    code_cell(
        """tables["hourly_hard_authors"].head(20)"""
    ),
    code_cell(
        """fig = plot_hourly_distribution(result["author_scores"])
fig"""
    ),
    code_cell(
        """fig = plot_hourly_penalty_curve(result["author_scores"], CONFIG)
fig"""
    ),
    code_cell(
        """tables["language_diversity_authors"].head(20)"""
    ),
    code_cell(
        """fig = plot_sentiment_theme_distributions(result["author_scores"])
fig"""
    ),
    md_cell(
        """## Repeat / Spam / Hard Bot Kontrolleri"""
    ),
    code_cell(
        """tables["hard_bot_examples"].head(20)"""
    ),
    code_cell(
        """tables["rapid_fire_examples"].head(20)"""
    ),
    code_cell(
        """tables["score_bands"]"""
    ),
    md_cell(
        """## Keyword ve Spam Ornekleri"""
    ),
    code_cell(
        """tables["keyword_validation"]"""
    ),
    code_cell(
        """tables["hashtag_spam_examples"][[
    "normalized_text",
    "hashtag_count",
    "hashtag_spam_component",
    "message_score",
    "final_score",
]].head(20)"""
    ),
    code_cell(
        """tables["token_spam_examples"][[
    "normalized_text",
    "max_token_frequency",
    "max_token_ratio",
    "token_repetition_component",
    "message_score",
    "final_score",
]].head(20)"""
    ),
    md_cell(
        """## Skor Ciktilari"""
    ),
    code_cell(
        """result["scored_preview"][[
    "author_hash",
    "author_type",
    "normalized_text",
    "author_score",
    "message_score",
    "behavioral_score",
    "final_score",
    "hard_bot_cluster_flag",
]].head(20)"""
    ),
    md_cell(
        """## Final Score Dagilimi

Bu hucreler `scored_messages.parquet` icinden dagilimi ve band ayrisimini gosterir. Readiness check bu hucrelerden once calismis olmali.
"""
    ),
    code_cell(
        """validate_scored_outputs(CONFIG, manifest)
score_dist_path = Path(CONFIG["paths"]["scored_messages_parquet"])
score_dist = pd.read_parquet(
    score_dist_path,
    columns=["final_score", "behavioral_score", "hard_bot_cluster_flag", "author_hard_hourly_flag"],
)
score_dist.describe(include="all")
"""
    ),
    code_cell(
        """fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

axes[0].hist(score_dist["final_score"], bins=50, color="#2563eb", edgecolor="white")
axes[0].set_title("Final Score Histogram")
axes[0].set_xlabel("final_score")
axes[0].set_ylabel("rows")

axes[1].hist(score_dist["final_score"], bins=50, color="#dc2626", edgecolor="white")
axes[1].set_title("Final Score Histogram (Log Y)")
axes[1].set_xlabel("final_score")
axes[1].set_ylabel("rows")
axes[1].set_yscale("log")

fig.tight_layout()
fig
"""
    ),
    code_cell(
        """score_bands_full = pd.cut(
    score_dist["final_score"],
    bins=[-0.001, 0.4, 0.6, 0.7, 0.85, 0.999999, 1.000001],
    labels=["0.00-0.40", "0.40-0.60", "0.60-0.70", "0.70-0.85", "0.85-<1.0", "1.0"],
    include_lowest=True,
)

score_band_summary = score_bands_full.value_counts(sort=False).rename_axis("band").reset_index(name="rows")
score_band_summary["share"] = score_band_summary["rows"] / max(len(score_dist), 1)
score_band_summary
"""
    ),
    md_cell(
        """## Tek Mesaj Inference Scaffold"""
    ),
    code_cell(
        """single_message = prepare_single_message_input(
    message_text="Example message for inference testing.",
    language="en",
    url="x.com",
    date="2026-04-26T12:00:00Z",
    author_hash=None,
    english_keywords="example, test, inference",
    primary_theme="unknown_theme",
)

score_single_message(single_message, result, CONFIG)"""
    ),
]


NOTEBOOK = {
    "cells": NOTEBOOK_CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    output_path = Path("fusion.ipynb")
    output_path.write_text(json.dumps(NOTEBOOK, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
