# Fusion Pipeline

`fusion_pipeline` is the core package for the Dataleague manipulation-risk model.

It is not a supervised bot classifier.

It is an explainable risk scoring pipeline for unsupervised social media manipulation detection.

## What the Model Tries to Solve

The dataset does not contain a ground-truth label such as:

- `bot`
- `human`
- `manipulative`
- `organic`

Because of that, the correct problem framing is:

- detect suspicious coordination patterns
- detect abnormal author behavior
- detect repeated or clustered messaging
- use text semantics as a supporting signal, not as the single source of truth

This package produces a `final_score` between `0.0` and `1.0`.

Interpretation:

- `0.0` means low manipulation risk
- `1.0` means high manipulation risk

This score is not a legal or absolute truth claim.

It is a ranked manipulation-risk signal.

## Modeling Philosophy

The most important assumption in this project is:

Text alone is not enough.

A human can write spam-looking text manually.
A real account can post emotional, repetitive, or promotional content.

Because of that, the strongest signal is behavioral context, especially:

- posting frequency
- inter-post timing
- same-text repetition
- multi-author reuse of the same text
- short-window message clustering
- language/theme diversity anomalies

The semantic model is useful, but it is secondary.

It helps when text content is suspicious, but behavior remains the core source of evidence.

## Final Score Structure

The final score is a fusion of:

- `behavioral_score`
- `roberta_score`

Conceptually:

```text
final_score = behavioral_weight * behavioral_score
            + semantic_weight   * roberta_score
```

The behavioral side is itself a fusion of:

- `author_score`
- `message_score`

Conceptually:

```text
behavioral_score = author_weight * author_score
                 + message_weight * message_score
```

### Author Score

`author_score` is the most important component in the system.

It uses:

- posts per day
- posts per active hour
- max posts in one hour
- mean / median / p10 inter-post time
- interval coefficient of variation
- same-text repeat ratio
- same-text repeat max
- multi-author repeat ratio
- language diversity
- theme diversity
- sentiment variance

This is the part that makes the model more than a text spam detector.

### Message Score

`message_score` captures message-level anomaly signals:

- same-text repeat count
- same-text unique author count
- same-text time-window count
- spam pattern flag
- hashtag density
- token repetition
- long text penalty
- keyword signal

This score is useful, but it is intentionally not the only decision source.

### Semantic Score

`roberta_score` comes from a sequence-classification model when enabled.

Its role:

- add language-level suspiciousness
- support behavioral evidence
- improve ranking when textual manipulation cues are strong

Its limits:

- it is not ground truth
- it may be language-restricted depending on config
- it should not be trusted alone for final decisions

## Hard Rules

Some cases are treated as high-confidence risk events.

Examples:

- extreme hourly posting burst
- heavy same-text repetition across multiple authors
- tight short-window cluster of repeated text

When those patterns trigger, the final score can be forced to `1.0`.

This exists because some coordination patterns are stronger than soft weighted signals.

## Package Layout

### `config.py`

Holds:

- default config
- deep merge logic
- JSON config loading
- sample config writing

### `constants.py`

Holds shared static definitions:

- schema versions
- required artifact columns
- SQLite table requirements
- store column definitions

### `data_processing.py`

Responsible for:

- missing-value normalization
- text normalization
- keyword cleaning
- text statistics extraction
- domain parsing
- semantic text preprocessing
- batch iteration
- dataset cleaning
- author/message feature preparation

### `scoring.py`

Responsible for:

- normalization references
- bounded scaling
- inverse scaling
- log penalties
- dominant signal logic
- author scoring
- message scoring
- behavioral weighting
- final score weighting

### `artifacts.py`

Responsible for:

- SQLite store setup
- batch pass orchestration
- manifest generation
- schema validation
- scored artifact validation
- QA tables
- diagnostic plots

### `inference.py`

Responsible for:

- single-message input formatting
- single-message scoring
- inference using either in-memory build results or SQLite-backed artifacts

### `pipeline.py`

High-level orchestration layer.

Main entry functions:

- `run_formula_pipeline`
- `run_formula_pipeline_two_pass`
- `run_rescore_from_existing_store`

### `legacy_impl.py`

Compatibility layer for older imports.

New code should use the domain modules directly.

## Main Execution Modes

### Full Build

Use when creating artifacts from scratch.

What it does:

- reads parquet input
- cleans messages
- writes SQLite store
- builds cluster tables
- scores authors
- scores messages
- writes final parquet outputs
- writes manifest

### Rescore From Existing Store

Use when scoring weights or thresholds change but the cleaned store does not need to be rebuilt.

What it does:

- reuses SQLite store
- avoids rebuilding the raw cleaned layer
- recomputes author/message/final scores
- rewrites scored parquet outputs
- updates manifest

### Single-Message Inference

Use during demos or hidden-message scoring.

What it does:

- accepts one message payload
- cleans it with the same pipeline logic
- looks up existing cluster/author context
- returns explanation-friendly score components

## Important Artifacts

Typical outputs:

- `fusion_batch_store.sqlite`
- `fusion_author_scores.parquet`
- `fusion_scored_messages.parquet`
- `fusion_manifest.json`

Purpose:

- SQLite store keeps reusable cleaned and clustered state
- author parquet stores author-level behavioral scores
- scored message parquet stores final message-level risk scores
- manifest stores schema/version/config compatibility

## Why This Model Is Credible

The credible part of this system is not “spam text means bot.”

The credible part is:

- repeated identical content
- synchronized reuse
- abnormal timing
- abnormal posting intensity
- author-level coordination patterns

That is why the behavioral side matters most.

The semantic side helps ranking, but the model should be explained primarily as a behavioral manipulation detector with semantic support.

## Limits

This package does not prove identity or intent.

It cannot guarantee:

- whether a message was written by a human by hand
- whether a suspicious post is truly automated
- whether a high-risk score means malicious intent with certainty

It can only say:

- this author/message pattern looks more or less consistent with manipulation risk

## Recommended Public Framing

When describing this project publicly, the strongest accurate claim is:

> We built an explainable unsupervised manipulation-risk scoring system that combines author behavior, coordination patterns, and semantic signals.

Avoid claiming:

> We built a perfect bot detector.

That claim is not defensible from this dataset.

## Supported Public Entry Points

For repo users, the active public surface is:

- `main.py`
- `fusion_pipeline/`
- `config.sample.json`
- `fusion.ipynb`

The package is designed so the code path is reusable even if the raw dataset is shared externally instead of stored in the repo.
