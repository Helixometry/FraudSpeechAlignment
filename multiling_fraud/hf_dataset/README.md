---
license: other
gated: true
task_categories:
  - audio-classification
  - text-classification
language:
  - en
  - hi
  - ko
tags:
  - fraud-detection
  - scam-call
  - conversational
  - synthetic
  - multilingual
  - audio-language-model
  - alignment
pretty_name: FraudAlign-MCS
size_categories:
  - 10K<n<100K
extra_gated_heading: "Request access to FraudAlign-MCS"
extra_gated_description: >
  FraudAlign-MCS is a multilingual and code-switched speech dataset
  for research on fraud safety and alignment in audio-language models.
  Access requests are manually reviewed.
extra_gated_prompt: >
  By requesting access, you agree to use FraudAlign-MCS only for
  legitimate research or educational purposes and to comply with
  the dataset terms of use.
extra_gated_fields:
  Full Name: text
  Institution / University / Organization: text
  Position or Role: text
  Country: country
  Intended use:
    type: select
    options:
      - Academic research
      - Educational research
      - Benchmark evaluation
      - Safety research
      - Commercial research
      - Other
  Briefly describe your research project: text
  I will not use this dataset to facilitate fraud or other harmful activity: checkbox
  I will not redistribute the dataset or provide unauthorized third-party access: checkbox
  I agree to cite the FraudAlign-MCS paper/dataset in publications using this resource: checkbox
  I agree to comply with the FraudAlign-MCS Terms of Use: checkbox
extra_gated_button_content: "Request access"
configs:
  - config_name: en
    data_files:
      - split: train
        path: data/en/train.jsonl
    default: true
  - config_name: hi
    data_files:
      - split: train
        path: data/hi/train.jsonl
  - config_name: ko
    data_files:
      - split: train
        path: data/ko/train.jsonl
  - config_name: hinglish
    data_files:
      - split: train
        path: data/hinglish/train.jsonl
  - config_name: all
    data_files:
      - split: train
        path:
          - data/en/train.jsonl
          - data/hi/train.jsonl
          - data/ko/train.jsonl
          - data/hinglish/train.jsonl
---

# FraudAlign-MCS

A **fraud-only** multilingual & code-switched dataset of scam-call dialogues, natively generated
(not translated) with `Qwen2.5-72B-Instruct-AWQ`. Modeled on the schema, fraud
taxonomy, and per-type proportions of the Chinese **TeleAntiFraud-28k** dataset,
regenerated from scratch in 4 languages: English (`en`), Hindi (`hi`), Korean (`ko`), Hinglish (Hindi-English code-switch) (`hinglish`).

**28,708 dialogues** total (7,177 per language), built to support
alignment of audio language models (ALMs) via preference pairs.

## Fraud taxonomy (per-type counts)

Seven fraud types, matching TeleAntiFraud's proportions:

| fraud_type_key | en | hi | ko | hinglish |
|---|---|---|---|---|
| customer_service | 2536 | 2536 | 2536 | 2536 |
| bank | 2039 | 2039 | 2039 | 2039 |
| investment | 984 | 984 | 984 | 984 |
| phishing | 555 | 555 | 555 | 555 |
| lottery | 524 | 524 | 524 | 524 |
| kidnapping | 407 | 407 | 407 | 407 |
| identity_theft | 132 | 132 | 132 | 132 |
| **total** | **7177** | **7177** | **7177** | **7177** |

## Fields

Each row is one dialogue:

| field | type | description |
|---|---|---|
| `id` | string | stable id, `{lang}_{fraud_type_key}_{00001}` |
| `language` | string | language code (`en`/`hi`/`ko`/`hinglish`) |
| `turns` | list | ordered `{"speaker": "caller"\|"callee", "text": ...}` |
| `fraud_type_key` | string | canonical type (english key, table above) |
| `fraud_type` | string | localized fraud-type label |
| `is_fraud` | bool | always `true` (fraud-only dataset) |
| `fraud_confidence` / `fraud_reason` | float / string | model's fraud judgement |
| `fraud_type_confidence` / `fraud_type_reason` | float / string | type judgement |
| `scene` / `scene_confidence` / `scene_reason` | string/float/string | scenario |
| `think` | string | model's reasoning trace |
| `caller_gender` / `callee_gender` | string | speaker genders (for TTS voices) |
| `audio_file` | string | relative path to the clip: `audio/{lang}/{id}.mp3` (Phase 2) |

## Configs

- `all` (default) — every language combined.
- `en`, `hi`, `ko`, `hinglish` — one language each.

```python
from datasets import load_dataset
ds = load_dataset("<repo>", "hi")          # Hindi only
ds = load_dataset("<repo>")                # all languages
```

## Roadmap / how this repo grows

This layout is designed so each phase is added **without rewriting earlier data**:

1. **Phase 1 — text (this release).** `data/<lang>/train.jsonl`.
2. **Phase 2 — audio.** TTS mp3s land in `audio/<lang>/<id>.mp3`; rows already
   carry the matching `audio_file` path. An `<lang>` audio config will be added.
3. **Phase 3 — preference pairs.** Chosen/rejected pairs for ALM alignment go
   under `preferences/<lang>/` as new configs.

## Provenance & license

Native generation with `Qwen2.5-72B-Instruct-AWQ` (vLLM). The Chinese
TeleAntiFraud-28k dataset supplied only the **schema, taxonomy, and proportions** —
no text was translated or copied. Released under **CC BY-NC 4.0**.

⚠️ **Intended use:** research on fraud/scam detection and audio-LM alignment.
All dialogues are synthetic; names, numbers, and stories are fabricated.
