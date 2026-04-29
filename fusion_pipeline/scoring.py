from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .data_processing import resolve_semantic_adapter_config

def _series_quantiles(series: pd.Series, low_q: float, high_q: float) -> tuple[float, float]:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return (0.0, 1.0)
    lower = float(clean.quantile(low_q))
    upper = float(clean.quantile(high_q))
    if math.isclose(lower, upper):
        upper = lower + 1.0
    return lower, upper


def fit_normalization_reference(author_features: pd.DataFrame, message_features: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    refs: dict[str, Any] = {"author": {}, "message": {}}

    for column in [
        "posts_per_day",
        "posts_per_active_hour",
        "theme_nunique",
        "sentiment_std",
        "same_text_repeat_ratio",
        "same_text_repeat_max",
        "multi_author_repeat_ratio",
    ]:
        refs["author"][column] = _series_quantiles(author_features[column], 0.05, 0.99) if column in author_features else (0.0, 1.0)

    for column in ["mean_interpost_sec", "median_interpost_sec", "p10_interpost_sec", "interval_cv"]:
        refs["author"][column] = _series_quantiles(author_features[column], 0.01, 0.95) if column in author_features else (1.0, 2.0)

    if "max_posts_one_hour" in author_features and not author_features.empty:
        refs["author"]["max_posts_one_hour_max"] = float(pd.to_numeric(author_features["max_posts_one_hour"], errors="coerce").fillna(0).max())
    else:
        refs["author"]["max_posts_one_hour_max"] = float(config["thresholds"]["hourly_penalty_start"] + 1)

    if "language_nunique" in author_features and not author_features.empty:
        refs["author"]["language_nunique_max"] = float(pd.to_numeric(author_features["language_nunique"], errors="coerce").fillna(0).max())
    else:
        refs["author"]["language_nunique_max"] = float(config["thresholds"]["language_penalty_start"] + 1)

    for column in [
        "text_length_chars",
        "same_text_repeat_count",
        "same_text_unique_author_count",
        "same_text_time_window_count",
        "keyword_count",
        "hashtag_count",
        "hashtag_density_chars",
        "hashtag_density_tokens",
        "max_token_frequency",
        "max_token_ratio",
        "repeated_token_count_over_2",
    ]:
        refs["message"][column] = _series_quantiles(message_features[column], 0.05, 0.99) if column in message_features else (0.0, 1.0)

    if "text_length_chars" in message_features and not message_features.empty:
        long_excess = (pd.to_numeric(message_features["text_length_chars"], errors="coerce") - config["thresholds"]["long_text_start"]).clip(lower=0)
        refs["message"]["long_text_excess_max"] = float(long_excess.max()) if not long_excess.empty else 1.0
    else:
        refs["message"]["long_text_excess_max"] = 1.0
    return refs


def bounded_scale(value: pd.Series | float, lower: float, upper: float) -> pd.Series | float:
    if upper <= lower:
        upper = lower + 1.0
    result = (value - lower) / (upper - lower)
    return result.clip(0, 1) if hasattr(result, "clip") else float(max(0.0, min(1.0, result)))


def inverse_bounded_scale(value: pd.Series | float, lower: float, upper: float) -> pd.Series | float:
    if upper <= lower:
        upper = lower + 1.0
    result = (upper - value) / (upper - lower)
    return result.clip(0, 1) if hasattr(result, "clip") else float(max(0.0, min(1.0, result)))


def log_penalty(series: pd.Series, start_threshold: float, max_reference: float) -> pd.Series:
    excess = (series - start_threshold).clip(lower=0)
    denom = math.log1p(max(max_reference - start_threshold, 1.0))
    if denom <= 0:
        denom = 1.0
    return (np.log1p(excess) / denom).clip(0, 1)


def apply_dominant_signal_floor(
    base_score: pd.Series,
    candidate_components: list[pd.Series],
    config: dict[str, Any],
    *,
    scope_key: str,
) -> pd.Series:
    policy = config.get("dominant_signal_policy", {})
    if not bool(policy.get("enabled", False)):
        return pd.to_numeric(base_score, errors="coerce").fillna(0.0).clip(0, 1)
    if str(policy.get("mode", "floor")) != "floor":
        return pd.to_numeric(base_score, errors="coerce").fillna(0.0).clip(0, 1)
    if not bool(policy.get("scope", {}).get(scope_key, False)):
        return pd.to_numeric(base_score, errors="coerce").fillna(0.0).clip(0, 1)
    if not candidate_components:
        return pd.to_numeric(base_score, errors="coerce").fillna(0.0).clip(0, 1)

    threshold = float(policy.get("threshold", 0.68))
    normalized_base = pd.to_numeric(base_score, errors="coerce").fillna(0.0).clip(0, 1)
    component_df = pd.concat(candidate_components, axis=1).apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(0, 1)
    dominant_component = component_df.max(axis=1)
    dominant_mask = dominant_component.ge(threshold)
    return normalized_base.where(~dominant_mask, np.maximum(normalized_base, dominant_component)).clip(0, 1)


def compute_author_scores(author_features: pd.DataFrame, refs: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    if author_features.empty:
        return author_features.copy()

    scores = author_features.copy()
    weights = config["weights"]["author_components"]
    thresholds = config["thresholds"]
    author_refs = refs["author"]
    derived_thresholds = config.setdefault("derived_thresholds", {})

    scores["activity_posts_per_day_risk"] = bounded_scale(scores["posts_per_day"], *author_refs["posts_per_day"])
    scores["activity_posts_per_hour_risk"] = bounded_scale(scores["posts_per_active_hour"], *author_refs["posts_per_active_hour"])
    hourly_hard_threshold = int(thresholds.get("hard_hourly_bot_threshold", 15))
    derived_thresholds["hourly_hard_knee"] = int(hourly_hard_threshold)
    scores["author_hard_hourly_flag"] = scores["max_posts_one_hour"].gt(hourly_hard_threshold).astype("int8")
    scores["activity_hourly_penalty_risk"] = log_penalty(
        scores["max_posts_one_hour"],
        thresholds["hourly_penalty_start"],
        author_refs["max_posts_one_hour_max"],
    )
    scores["activity_risk"] = (
        0.20 * scores["activity_posts_per_day_risk"]
        + 0.25 * scores["activity_posts_per_hour_risk"]
        + 0.55 * scores["activity_hourly_penalty_risk"]
    )

    scores["timing_mean_gap_risk"] = inverse_bounded_scale(scores["mean_interpost_sec"], *author_refs["mean_interpost_sec"])
    scores["timing_median_gap_risk"] = inverse_bounded_scale(scores["median_interpost_sec"], *author_refs["median_interpost_sec"])
    scores["timing_p10_gap_risk"] = inverse_bounded_scale(scores["p10_interpost_sec"], *author_refs["p10_interpost_sec"])
    scores["timing_cv_risk"] = inverse_bounded_scale(scores["interval_cv"], *author_refs["interval_cv"])
    scores["timing_risk"] = (
        0.15 * scores["timing_mean_gap_risk"]
        + 0.25 * scores["timing_median_gap_risk"]
        + 0.35 * scores["timing_p10_gap_risk"]
        + 0.25 * scores["timing_cv_risk"]
    )

    scores["repetition_same_text_ratio_risk"] = bounded_scale(scores["same_text_repeat_ratio"], *author_refs["same_text_repeat_ratio"])
    scores["repetition_same_text_max_risk"] = bounded_scale(scores["same_text_repeat_max"], *author_refs["same_text_repeat_max"])
    scores["repetition_multi_author_risk"] = bounded_scale(scores["multi_author_repeat_ratio"], *author_refs["multi_author_repeat_ratio"])
    scores["repetition_risk"] = (
        0.30 * scores["repetition_same_text_ratio_risk"]
        + 0.25 * scores["repetition_same_text_max_risk"]
        + 0.45 * scores["repetition_multi_author_risk"]
    )

    language_excess = (scores["language_nunique"] - thresholds["language_penalty_start"]).clip(lower=0)
    language_max_excess = max(author_refs["language_nunique_max"] - thresholds["language_penalty_start"], 1.0)
    scores["diversity_language_risk"] = bounded_scale(language_excess, 0.0, language_max_excess)
    scores["diversity_theme_risk"] = bounded_scale(scores["theme_nunique"], *author_refs["theme_nunique"])
    scores["diversity_sentiment_risk"] = bounded_scale(scores["sentiment_std"], *author_refs["sentiment_std"])
    scores["diversity_risk"] = (
        0.40 * scores["diversity_language_risk"]
        + 0.30 * scores["diversity_theme_risk"]
        + 0.30 * scores["diversity_sentiment_risk"]
    )

    scores["metadata_risk"] = 0.0
    scores["author_score"] = (
        weights["activity"] * scores["activity_risk"]
        + weights["timing"] * scores["timing_risk"]
        + weights["repetition"] * scores["repetition_risk"]
        + weights["diversity"] * scores["diversity_risk"]
        + weights["metadata"] * scores["metadata_risk"]
    ).clip(0, 1)
    scores["author_score"] = apply_dominant_signal_floor(
        scores["author_score"],
        [
            scores["activity_risk"],
            scores["timing_risk"],
            scores["repetition_risk"],
            scores["diversity_risk"],
        ],
        config,
        scope_key="author",
    )
    return scores


def compute_message_scores(message_features: pd.DataFrame, refs: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    if message_features.empty:
        return message_features.copy()

    scores = message_features.copy()
    weights = config["weights"]["message_components"]
    thresholds = config["thresholds"]
    msg_refs = refs["message"]

    scores["same_text_repeat_risk"] = bounded_scale(scores["same_text_repeat_count"], *msg_refs["same_text_repeat_count"])
    scores["same_text_multi_author_risk"] = bounded_scale(scores["same_text_unique_author_count"], *msg_refs["same_text_unique_author_count"])
    scores["same_text_time_window_risk"] = bounded_scale(scores["same_text_time_window_count"], *msg_refs["same_text_time_window_count"])
    scores["same_text_repeat_component"] = (
        0.45 * scores["same_text_repeat_risk"]
        + 0.35 * scores["same_text_multi_author_risk"]
        + 0.20 * scores["same_text_time_window_risk"]
    )

    scores["spam_pattern_component"] = scores["spam_pattern_flag"].astype(float)

    scores["hashtag_count_risk"] = bounded_scale(scores["hashtag_count"], *msg_refs["hashtag_count"])
    scores["hashtag_density_chars_risk"] = bounded_scale(scores["hashtag_density_chars"], *msg_refs["hashtag_density_chars"])
    scores["hashtag_density_tokens_risk"] = bounded_scale(scores["hashtag_density_tokens"], *msg_refs["hashtag_density_tokens"])
    scores["hashtag_spam_component"] = (
        0.40 * scores["hashtag_count_risk"]
        + 0.30 * scores["hashtag_density_chars_risk"]
        + 0.30 * scores["hashtag_density_tokens_risk"]
    )

    scores["token_frequency_risk"] = bounded_scale(scores["max_token_frequency"], *msg_refs["max_token_frequency"])
    scores["token_ratio_risk"] = bounded_scale(scores["max_token_ratio"], *msg_refs["max_token_ratio"])
    scores["token_repeat_cluster_risk"] = bounded_scale(
        scores["repeated_token_count_over_2"], *msg_refs["repeated_token_count_over_2"]
    )
    scores["token_repetition_component"] = (
        0.35 * scores["token_frequency_risk"]
        + 0.45 * scores["token_ratio_risk"]
        + 0.20 * scores["token_repeat_cluster_risk"]
    )

    long_excess = (scores["text_length_chars"] - thresholds["long_text_start"]).clip(lower=0)
    scores["long_text_component"] = log_penalty(
        long_excess + thresholds["long_text_start"],
        thresholds["long_text_start"],
        thresholds["long_text_start"] + max(msg_refs["long_text_excess_max"], 1.0),
    )
    if config["rules"]["long_text_requires_spam"]:
        scores["long_text_component"] = scores["long_text_component"] * scores["spam_pattern_component"]

    scores["keyword_signal_component"] = bounded_scale(scores["keyword_count"], *msg_refs["keyword_count"])
    repeat_hard_threshold = int(config.get("dominant_signal_policy", {}).get("repeat_hard_threshold", 5))
    scores["hard_same_text_repeat_flag"] = scores["same_text_repeat_count"].gt(repeat_hard_threshold).astype("int8")
    scores["message_score"] = (
        weights["same_text_repeat"] * scores["same_text_repeat_component"]
        + weights["spam_pattern"] * scores["spam_pattern_component"]
        + weights["hashtag_spam"] * scores["hashtag_spam_component"]
        + weights["token_repetition"] * scores["token_repetition_component"]
        + weights["long_text"] * scores["long_text_component"]
        + weights["keyword_signal"] * scores["keyword_signal_component"]
    ).clip(0, 1)
    scores["message_score"] = scores["message_score"].where(~scores["hashtag_count"].gt(5), 1.0)
    scores["message_score"] = scores["message_score"].where(~scores["exclamation_count"].gt(10), 1.0)
    scores["message_score"] = apply_dominant_signal_floor(
        scores["message_score"],
        [
            scores["same_text_repeat_component"],
            scores["spam_pattern_component"],
            scores["hashtag_spam_component"],
            scores["token_repetition_component"],
            scores["long_text_component"],
            scores["keyword_signal_component"],
        ],
        config,
        scope_key="message",
    )
    return scores


def confidence_weight_from_score(score: pd.Series, min_weight: float, power: float) -> pd.Series:
    score = pd.to_numeric(score, errors="coerce").fillna(0.5).clip(0, 1)
    distance_from_neutral = (score - 0.5).abs()
    normalized_distance = (distance_from_neutral / 0.5).clip(0, 1)
    return (min_weight + (1.0 - min_weight) * np.power(normalized_distance, power)).clip(min_weight, 1.0)


def neutral_mask_from_score(score: pd.Series, neutral_score: float, epsilon: float) -> pd.Series:
    numeric_score = pd.to_numeric(score, errors="coerce").fillna(neutral_score).clip(0, 1)
    return numeric_score.sub(float(neutral_score)).abs().le(float(epsilon))


def apply_neutral_weight_override(
    left_score: pd.Series,
    right_score: pd.Series,
    left_weight: pd.Series,
    right_weight: pd.Series,
    neutral_score: float,
    epsilon: float,
) -> tuple[pd.Series, pd.Series]:
    left_neutral_mask = neutral_mask_from_score(left_score, neutral_score, epsilon)
    right_neutral_mask = neutral_mask_from_score(right_score, neutral_score, epsilon)

    left_adjusted = left_weight.where(~left_neutral_mask, 0.0)
    right_adjusted = right_weight.where(~right_neutral_mask, 0.0)
    total_adjusted = left_adjusted + right_adjusted
    both_neutral_mask = left_neutral_mask & right_neutral_mask

    safe_total = total_adjusted.where(total_adjusted.gt(0), 1.0)
    left_final = (left_adjusted / safe_total).where(total_adjusted.gt(0), 0.5)
    right_final = (right_adjusted / safe_total).where(total_adjusted.gt(0), 0.5)
    left_final = left_final.where(~both_neutral_mask, 0.5)
    right_final = right_final.where(~both_neutral_mask, 0.5)
    return left_final.astype("float32"), right_final.astype("float32")


def sigmoid_gate_weights(left: pd.Series, right: pd.Series, steepness: float) -> tuple[pd.Series, pd.Series]:
    steepness = max(float(steepness), 1e-6)
    left_values = pd.to_numeric(left, errors="coerce").fillna(0.0)
    right_values = pd.to_numeric(right, errors="coerce").fillna(0.0)
    logits = np.clip(steepness * (right_values - left_values), -60, 60)
    right_weight = 1.0 / (1.0 + np.exp(-logits))
    left_weight = 1.0 - right_weight
    return left_weight, right_weight


def apply_final_score_weighting(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    weights = config["weights"]["behavioral_vs_semantic"]
    behavioral_prior = float(weights["behavioral"])
    roberta_prior = float(weights["semantic"])
    dynamic_cfg = config.get("dynamic_final_weighting", {})
    dynamic_enabled = bool(dynamic_cfg.get("enabled", False))
    semantic_cfg = resolve_semantic_adapter_config(config)
    neutral_policy_cfg = config.get("neutral_score_policy", {})
    neutral_score = float(neutral_policy_cfg.get("neutral_score", semantic_cfg["unsupported_language_score"]))
    neutral_epsilon = float(neutral_policy_cfg.get("epsilon", 1e-6))

    df["roberta_score"] = df["roberta_score"].fillna(neutral_score).clip(0, 1)
    behavioral_input_score = df["behavioral_score"].copy()

    if dynamic_enabled:
        min_weight = float(dynamic_cfg.get("min_confidence_weight", 0.2))
        power = float(dynamic_cfg.get("power", 2.0))
        df["behavioral_confidence_weight"] = confidence_weight_from_score(behavioral_input_score, min_weight, power)
        df["roberta_confidence_weight"] = confidence_weight_from_score(df["roberta_score"], min_weight, power)
    else:
        df["behavioral_confidence_weight"] = 1.0
        df["roberta_confidence_weight"] = 1.0

    behavioral_raw_weight = behavioral_prior * df["behavioral_confidence_weight"]
    roberta_raw_weight = roberta_prior * df["roberta_confidence_weight"]
    if dynamic_enabled:
        sigmoid_steepness = float(dynamic_cfg.get("sigmoid_steepness", 8.0))
        df["behavioral_effective_weight"], df["roberta_effective_weight"] = sigmoid_gate_weights(
            behavioral_raw_weight,
            roberta_raw_weight,
            sigmoid_steepness,
        )
    else:
        total_prior = max(behavioral_prior + roberta_prior, 1e-6)
        df["behavioral_effective_weight"] = behavioral_prior / total_prior
        df["roberta_effective_weight"] = roberta_prior / total_prior
    df["behavioral_effective_weight"], df["roberta_effective_weight"] = apply_neutral_weight_override(
        behavioral_input_score,
        df["roberta_score"],
        df["behavioral_effective_weight"],
        df["roberta_effective_weight"],
        neutral_score,
        neutral_epsilon,
    )
    df["final_score_before_rules"] = (
        df["behavioral_effective_weight"] * behavioral_input_score
        + df["roberta_effective_weight"] * df["roberta_score"]
    ).clip(0, 1)
    df["final_score"] = np.where(
        df["author_hard_hourly_flag"].fillna(0).eq(1)
        | df["hard_bot_cluster_flag"].fillna(0).eq(1)
        | df["hard_same_text_repeat_flag"].fillna(0).eq(1),
        1.0,
        df["final_score_before_rules"],
    )
    df["final_score"] = df["final_score"].where(~df["hashtag_count"].fillna(0).gt(5), 1.0)
    df["final_score"] = df["final_score"].where(~df["exclamation_count"].fillna(0).gt(10), 1.0)
    return df


def compute_behavioral_score(df: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    author_weight = config["weights"]["author_vs_message"]["author"]
    message_weight = config["weights"]["author_vs_message"]["message"]
    neutral_policy_cfg = config.get("neutral_score_policy", {})
    neutral_score = float(neutral_policy_cfg.get("neutral_score", 0.5))
    neutral_epsilon = float(neutral_policy_cfg.get("epsilon", 1e-6))
    author_input_score = pd.to_numeric(df["author_score"], errors="coerce").fillna(neutral_score).clip(0, 1)
    message_input_score = pd.to_numeric(df["message_score"], errors="coerce").fillna(neutral_score).clip(0, 1)
    author_weights = pd.Series(float(author_weight), index=df.index, dtype="float32")
    message_weights = pd.Series(float(message_weight), index=df.index, dtype="float32")
    behavioral_author_weight, behavioral_message_weight = apply_neutral_weight_override(
        author_input_score,
        message_input_score,
        author_weights,
        message_weights,
        neutral_score,
        neutral_epsilon,
    )
    behavioral_score = (
        behavioral_author_weight * author_input_score + behavioral_message_weight * message_input_score
    ).clip(0, 1)
    dominant_policy = config.get("dominant_signal_policy", {})
    dominant_threshold = float(dominant_policy.get("threshold", 0.68))
    dominant_message_mask = message_input_score.ge(dominant_threshold)
    if bool(dominant_message_mask.any()):
        behavioral_score = behavioral_score.where(
            ~dominant_message_mask,
            np.maximum(behavioral_score, message_input_score),
        )
    anonymous_mask = df["author_type"].eq("anonymous")
    if bool(anonymous_mask.any()):
        behavioral_score = behavioral_score.where(~anonymous_mask, message_input_score)
    if "hard_same_text_repeat_flag" in df.columns:
        hard_repeat_mask = df["hard_same_text_repeat_flag"].fillna(0).eq(1)
        if bool(hard_repeat_mask.any()):
            behavioral_score = behavioral_score.where(~hard_repeat_mask, 1.0)
    return behavioral_score


def compute_final_scores(
    clean_df: pd.DataFrame,
    author_scores: pd.DataFrame,
    message_scores: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    df = message_scores.copy()
    merge_columns = ["author_hash", "author_score", "author_hard_hourly_flag"]
    if not author_scores.empty:
        df = df.merge(author_scores[merge_columns], on="author_hash", how="left")
    else:
        df["author_score"] = np.nan
        df["author_hard_hourly_flag"] = 0

    df["behavioral_score"] = compute_behavioral_score(df, config)
    return apply_final_score_weighting(df, config)
