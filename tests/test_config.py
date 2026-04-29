from __future__ import annotations

import json

from fusion_pipeline import config as config_module


def test_default_config_has_required_sections() -> None:
    config = config_module.load_config()
    assert {"paths", "runtime", "semantic_adapter", "thresholds", "weights"}.issubset(config.keys())


def test_json_override_deep_merges_nested_values(tmp_path) -> None:
    override_path = tmp_path / "override.json"
    override_path.write_text(
        json.dumps(
            {
                "runtime": {"batch_size": 1234},
                "semantic_adapter": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )

    config = config_module.load_config(override_path)
    assert config["runtime"]["batch_size"] == 1234
    assert config["semantic_adapter"]["enabled"] is False
    assert "message_batch_size" in config["runtime"]
    assert "weights" in config


def test_write_default_config_round_trips(tmp_path) -> None:
    output_path = tmp_path / "config.sample.json"
    written = config_module.write_default_config(output_path)
    reloaded = config_module.load_config(written)

    assert written == output_path
    assert reloaded["paths"]["batch_sqlite_db"].endswith("fusion_batch_store.sqlite")
