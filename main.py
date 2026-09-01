from __future__ import annotations

import argparse
import json

from fusion_pipeline.config import load_config, write_default_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fusion risk skorlama hattı.")
    parser.add_argument(
        "--mode",
        choices=["build", "rescore", "validate", "score-single", "write-config"],
        default="build",
        help="Çalıştırma kipi.",
    )
    parser.add_argument("--config", type=str, default=None, help="JSON config override dosyası.")
    parser.add_argument(
        "--output-config",
        type=str,
        default="config.sample.json",
        help="write-config çıktı yolu.",
    )
    parser.add_argument("--message", type=str, default=None, help="score-single için mesaj metni.")
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--author-hash", type=str, default=None)
    parser.add_argument("--english-keywords", type=str, default=None)
    parser.add_argument("--primary-theme", type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "write-config":
        path = write_default_config(args.output_config)
        print(json.dumps({"written_config": str(path)}, indent=2))
        return

    config = load_config(args.config)

    if args.mode == "build":
        from fusion_pipeline.pipeline import run_formula_pipeline_two_pass

        result = run_formula_pipeline_two_pass(config)
        print(json.dumps(result["paths"], indent=2))
        return

    if args.mode == "rescore":
        from fusion_pipeline.pipeline import run_rescore_from_existing_store

        result = run_rescore_from_existing_store(config)
        print(json.dumps(result["paths"], indent=2))
        return

    if args.mode == "validate":
        from fusion_pipeline.artifacts import assert_artifacts_ready

        result = assert_artifacts_ready({"paths": {}}, config)
        payload = {
            "manifest_path": config["paths"]["manifest_path"],
            "sqlite_path": result["manifest"]["sqlite_path"],
            "author_scores_path": result["manifest"]["author_scores_path"],
            "scored_messages_path": result["manifest"]["scored_messages_path"],
        }
        print(json.dumps(payload, indent=2))
        return

    if not args.message:
        raise SystemExit("--message, --mode score-single için zorunlu")

    from fusion_pipeline.artifacts import assert_artifacts_ready
    from fusion_pipeline.inference import prepare_single_message_input, score_single_message

    validated = assert_artifacts_ready({"paths": {}}, config)
    runtime_result = {
        "paths": {
            "sqlite_db": validated["manifest"]["sqlite_path"],
        }
    }
    single_input = prepare_single_message_input(
        message_text=args.message,
        language=args.language,
        url=args.url,
        date=args.date,
        author_hash=args.author_hash,
        english_keywords=args.english_keywords,
        primary_theme=args.primary_theme,
    )
    scored = score_single_message(single_input, runtime_result, config)
    print(scored.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
