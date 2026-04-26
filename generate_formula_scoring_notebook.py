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
        """# Formula Tabanli Robot Skorlama Pipeline

Bu notebook tek teslim akisina gore kuruldu:

- cleaning
- sadece formule girecek feature engineering
- behavioral skor hesaplama
- hard-bot override
- inference scaffold

Semantic katman ilk versiyonda yoktur. Nihai skor `0 -> insan`, `1 -> robot` mantigiyla uretilir.
"""
    ),
    code_cell(CONFIG_CELL),
    code_cell(
        """import pandas as pd
import pyarrow.parquet as pq

from formula_scoring_pipeline import (
    plot_hourly_distribution,
    plot_hourly_penalty_curve,
    plot_sentiment_theme_distributions,
    prepare_single_message_input,
    run_formula_pipeline_two_pass,
    score_single_message,
)
"""
    ),
    md_cell(
        """## Input Kontrolu

Veri kaynagini, satir sayisini ve mevcut runtime ayarlarini burada kontrol edin.
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
        """## Pipeline Calistir

Bu hucre cleaning, feature engineering ve formula scoring akisini birlikte calistirir.
"""
    ),
    code_cell(
        """result = run_formula_pipeline_two_pass(CONFIG)
tables = result["tables"]

print("pipeline completed")
print(f"clean_rows={result['summary']['clean_rows']:,}")
print(f"technical_duplicates_removed={result['summary']['technical_duplicates_removed']:,}")
print(f"anonymous_rows={result['summary']['missing_author_rows']:,}")
print(result["paths"])
"""
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
        """## Author Duzeyi Kontroller

Saatlik penalty, dil cesitliligi, theme cesitliligi ve sentiment oynakligi burada gorulur.
"""
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
        """## Keyword Dogrulama

Keyword sinyali final formulde dusuk/orta agirlikta tutuluyor. Bu hucre, repeat/spam agir mesajlarla farkini hizli kontrol etmek icindir.
"""
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
        """## Skor Ciktilari

Asagidaki tablo mesaj seviyesinde uretilecek temel risk ciktisini gosterir.
"""
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
        """## Tek Mesaj Inference Scaffold

Bu hucre yeni bir mesaji ayni pipeline mantigi ile skorlamak icindir. Anonymous mesajlar author-level kismini notr gecer.
"""
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
