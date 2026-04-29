from __future__ import annotations

from fusion_pipeline.inference import prepare_single_message_input, score_single_message


def test_prepare_single_message_input_shape() -> None:
    frame = prepare_single_message_input(
        message_text="hello world",
        language="en",
        url="x.com",
        date="2024-01-01T00:00:00Z",
        author_hash="author-1",
        english_keywords="hello, world",
        primary_theme="test",
    )
    assert len(frame) == 1
    assert list(frame.columns) == [
        "original_text",
        "language",
        "url",
        "date",
        "author_hash",
        "english_keywords",
        "primary_theme",
        "sentiment",
        "main_emotion",
    ]


def test_score_single_message_with_sqlite_fixture(sqlite_score_fixture) -> None:
    config, runtime_result = sqlite_score_fixture
    single_input = prepare_single_message_input(
        message_text="Bot bot bot #promo !!!",
        language="en",
        url="https://news.example.com/path",
        date="2024-01-01T00:00:00Z",
        author_hash="author-1",
        english_keywords="bot, promo",
        primary_theme="campaign",
    )

    scored = score_single_message(single_input, runtime_result, config)
    expected_columns = [
        "author_type",
        "message_score",
        "author_score",
        "behavioral_score",
        "roberta_score",
        "behavioral_confidence_weight",
        "roberta_confidence_weight",
        "behavioral_effective_weight",
        "roberta_effective_weight",
        "final_score_before_rules",
        "final_score",
        "same_text_repeat_component",
        "spam_pattern_component",
        "hashtag_spam_component",
        "token_repetition_component",
        "long_text_component",
        "keyword_signal_component",
        "author_hard_hourly_flag",
        "hard_bot_cluster_flag",
        "hard_same_text_repeat_flag",
    ]

    assert list(scored.columns) == expected_columns
    assert scored.iloc[0]["author_type"] == "identified"
    assert 0.0 <= scored.iloc[0]["final_score"] <= 1.0
