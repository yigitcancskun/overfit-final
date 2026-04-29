from __future__ import annotations

import pandas as pd

from fusion_pipeline.scoring import (
    apply_final_score_weighting,
    apply_neutral_weight_override,
    bounded_scale,
    compute_behavioral_score,
    inverse_bounded_scale,
)


def test_bounded_scale_and_inverse_bounded_scale_clip_values() -> None:
    assert bounded_scale(5.0, 0.0, 10.0) == 0.5
    assert bounded_scale(-1.0, 0.0, 10.0) == 0.0
    assert inverse_bounded_scale(5.0, 0.0, 10.0) == 0.5
    assert inverse_bounded_scale(20.0, 0.0, 10.0) == 0.0


def test_neutral_weight_override_drops_neutral_side() -> None:
    left, right = apply_neutral_weight_override(
        left_score=pd.Series([0.5, 0.9]),
        right_score=pd.Series([0.9, 0.5]),
        left_weight=pd.Series([0.7, 0.7]),
        right_weight=pd.Series([0.3, 0.3]),
        neutral_score=0.5,
        epsilon=1e-6,
    )
    assert left.tolist() == [0.0, 1.0]
    assert right.tolist() == [1.0, 0.0]


def test_behavioral_score_uses_message_score_for_anonymous_author(base_config: dict) -> None:
    frame = pd.DataFrame(
        {
            "author_type": ["anonymous"],
            "author_score": [0.9],
            "message_score": [0.2],
            "hard_same_text_repeat_flag": [0],
        }
    )
    behavioral = compute_behavioral_score(frame, base_config)
    assert behavioral.iloc[0] == 0.2


def test_final_score_weighting_promotes_hard_rule(base_config: dict) -> None:
    frame = pd.DataFrame(
        {
            "behavioral_score": [0.4],
            "roberta_score": [0.2],
            "author_hard_hourly_flag": [0],
            "hard_bot_cluster_flag": [1],
            "hard_same_text_repeat_flag": [0],
            "hashtag_count": [0],
            "exclamation_count": [0],
        }
    )
    result = apply_final_score_weighting(frame, base_config)
    assert result["final_score"].iloc[0] == 1.0
