.PHONY: compile help config test smoke verify

PYTHON ?= python
CONFIG_OUT ?= /tmp/fusion-config-verify.json

compile:
	$(PYTHON) -m py_compile fusion_pipeline/*.py main.py formula_scoring_pipeline.py tests/*.py

help:
	$(PYTHON) main.py --help

config:
	$(PYTHON) main.py --mode write-config --output-config $(CONFIG_OUT)

test:
	pytest -q

smoke:
	$(PYTHON) - <<'PY'
	import sqlite3
	import sys
	import tempfile
	from copy import deepcopy
	from pathlib import Path
	import pandas as pd
	sys.path.insert(0, str(Path('.').resolve()))
	from fusion_pipeline.config import DEFAULT_CONFIG
	from fusion_pipeline.data_processing import compute_text_hash_scalar, normalize_text_scalar
	from fusion_pipeline.inference import prepare_single_message_input, score_single_message
	with tempfile.TemporaryDirectory() as tmpdir:
	    tmp = Path(tmpdir)
	    config = deepcopy(DEFAULT_CONFIG)
	    config["semantic_adapter"]["enabled"] = False
	    config["runtime"]["enable_progress_logs"] = False
	    config["paths"]["batch_sqlite_db"] = str(tmp / "fusion.sqlite")
	    normalized_text = normalize_text_scalar("Bot bot bot #promo !!!")
	    text_hash = compute_text_hash_scalar(normalized_text)
	    date = pd.Timestamp("2024-01-01T00:00:00Z")
	    bucket = int(date.value // (config["thresholds"]["hard_bot_time_window_sec"] * 1_000_000_000))
	    conn = sqlite3.connect(config["paths"]["batch_sqlite_db"])
	    conn.executescript("""
	        CREATE TABLE cleaned_messages (
	            text_length_chars INTEGER,
	            keyword_count INTEGER,
	            hashtag_count INTEGER,
	            hashtag_density_chars REAL,
	            hashtag_density_tokens REAL,
	            max_token_frequency INTEGER,
	            max_token_ratio REAL,
	            repeated_token_count_over_2 INTEGER
	        );
	        CREATE TABLE text_clusters (
	            text_hash TEXT,
	            repeat_count INTEGER,
	            unique_author_count INTEGER
	        );
	        CREATE TABLE text_window_clusters (
	            text_hash TEXT,
	            time_window_bucket INTEGER,
	            window_count INTEGER
	        );
	        CREATE TABLE author_scores (
	            author_hash TEXT PRIMARY KEY,
	            author_score REAL,
	            author_hard_hourly_flag INTEGER
	        );
	    """)
	    conn.execute(
	        "INSERT INTO cleaned_messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
	        (19, 2, 1, 0.05, 0.25, 3, 0.75, 1),
	    )
	    conn.execute("INSERT INTO text_clusters VALUES (?, ?, ?)", (text_hash, 4, 2))
	    conn.execute("INSERT INTO text_window_clusters VALUES (?, ?, ?)", (text_hash, bucket, 3))
	    conn.execute("INSERT INTO author_scores VALUES (?, ?, ?)", ("author-1", 0.82, 0))
	    conn.commit()
	    conn.close()
	    result = {"paths": {"sqlite_db": config["paths"]["batch_sqlite_db"]}}
	    message = prepare_single_message_input(
	        message_text="Bot bot bot #promo !!!",
	        language="en",
	        url="https://news.example.com/path",
	        date="2024-01-01T00:00:00Z",
	        author_hash="author-1",
	        english_keywords="bot, promo",
	        primary_theme="campaign",
	    )
	    scored = score_single_message(message, result, config)
	    print(scored[["final_score", "behavioral_score", "roberta_score"]].to_json(orient="records", indent=2))
	PY

verify: compile help config test smoke
