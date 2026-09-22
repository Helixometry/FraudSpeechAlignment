#!/usr/bin/env python3
"""Assemble the multilingual fraud dialogues into a HuggingFace-ready dataset tree.

Layout produced under <root> (default: multiling_fraud/hf_dataset):

    <root>/
    ├── README.md                      # dataset card (YAML configs + docs)
    ├── data/<lang>/train.jsonl        # text dialogues, one JSON object per line
    ├── audio/<lang>/                   # (filled in Phase 2) <id>.mp3 clips
    │   └── README.md
    └── preferences/<lang>/             # (filled in Phase 3) preference pairs
        └── README.md

Design goals:
  * Each language is an independent HF *config* -> re-run this for one language
    (e.g. after the English regen) without touching the others.
  * Every row carries `audio_file = audio/<lang>/<id>.mp3` up front, so once the
    mp3s are uploaded the rows already point at them (no re-write of text needed).
  * JSONL is human-diffable and the Hub auto-converts it to Parquet for the viewer.

Usage:
    python scripts/build_hf_dataset.py                 # all languages
    python scripts/build_hf_dataset.py --langs en      # just refresh English
"""
import argparse, json, os, collections

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANGS = ["en", "hi", "ko", "hinglish"]
LANG_NAMES = {"en": "English", "hi": "Hindi", "ko": "Korean",
              "hinglish": "Hinglish (Hindi-English code-switch)"}
# HF ISO language tags for the card (hinglish -> hi+en multilingual)
LANG_TAGS = {"en": ["en"], "hi": ["hi"], "ko": ["ko"], "hinglish": ["hi", "en"]}
FRAUD_TYPES = ["customer_service", "bank", "investment", "phishing",
               "lottery", "kidnapping", "identity_theft"]


def load(lang):
    p = os.path.join(PROJ, "out", lang, "dialogues_full.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(rows, lang, root):
    outdir = os.path.join(root, "data", lang)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "train.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            rec = dict(r)
            # forward-compatible pointer to the (later-uploaded) audio clip
            rec["audio_file"] = f"audio/{lang}/{r['id']}.mp3"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def build_language(lang, root):
    rows = load(lang)
    path = write_jsonl(rows, lang, root)
    counts = collections.Counter(r["fraud_type_key"] for r in rows)
    print(f"[hf] {lang}: {len(rows)} rows -> {os.path.relpath(path, PROJ)}")
    return {"lang": lang, "n": len(rows), "counts": dict(counts)}


CONFIG_YAML_HEAD = """---
license: other
gated: true
task_categories:
  - audio-classification
  - text-classification
language:
{lang_tags}
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
{configs}
---
"""


def yaml_lang_tags(stats):
    tags = []
    for s in stats:
        for t in LANG_TAGS[s["lang"]]:
            if t not in tags:
                tags.append(t)
    return "\n".join(f"  - {t}" for t in tags)


def yaml_configs(stats):
    lines = []
    # per-language configs; English is the DEFAULT shown in the dataset viewer
    for s in stats:
        lines.append(f'  - config_name: {s["lang"]}')
        lines.append("    data_files:")
        lines.append(f'      - split: train\n        path: data/{s["lang"]}/train.jsonl')
        if s["lang"] == "en":
            lines.append("    default: true")
    # combined config across all languages: ONE split with a list of paths
    lines.append("  - config_name: all")
    lines.append("    data_files:")
    lines.append("      - split: train")
    lines.append("        path:")
    for s in stats:
        lines.append(f'          - data/{s["lang"]}/train.jsonl')
    return "\n".join(lines)


def counts_table(stats):
    header = "| fraud_type_key | " + " | ".join(s["lang"] for s in stats) + " |"
    sep = "|" + "---|" * (len(stats) + 1)
    rows = [header, sep]
    for ft in FRAUD_TYPES:
        rows.append("| " + ft + " | " +
                    " | ".join(str(s["counts"].get(ft, 0)) for s in stats) + " |")
    rows.append("| **total** | " + " | ".join(f'**{s["n"]}**' for s in stats) + " |")
    return "\n".join(rows)


def write_readme(stats, root):
    front = CONFIG_YAML_HEAD.format(lang_tags=yaml_lang_tags(stats),
                                    configs=yaml_configs(stats))
    total = sum(s["n"] for s in stats)
    langs_line = ", ".join(f'{LANG_NAMES[s["lang"]]} (`{s["lang"]}`)' for s in stats)
    body = f"""
# FraudAlign-MCS

A **fraud-only** multilingual & code-switched dataset of scam-call dialogues, natively generated
(not translated) with `Qwen2.5-72B-Instruct-AWQ`. Modeled on the schema, fraud
taxonomy, and per-type proportions of the Chinese **TeleAntiFraud-28k** dataset,
regenerated from scratch in {len(stats)} languages: {langs_line}.

**{total:,} dialogues** total ({stats[0]['n']:,} per language), built to support
alignment of audio language models (ALMs) via preference pairs.

## Fraud taxonomy (per-type counts)

Seven fraud types, matching TeleAntiFraud's proportions:

{counts_table(stats)}

## Fields

Each row is one dialogue:

| field | type | description |
|---|---|---|
| `id` | string | stable id, `{{lang}}_{{fraud_type_key}}_{{00001}}` |
| `language` | string | language code (`en`/`hi`/`ko`/`hinglish`) |
| `turns` | list | ordered `{{"speaker": "caller"\\|"callee", "text": ...}}` |
| `fraud_type_key` | string | canonical type (english key, table above) |
| `fraud_type` | string | localized fraud-type label |
| `is_fraud` | bool | always `true` (fraud-only dataset) |
| `fraud_confidence` / `fraud_reason` | float / string | model's fraud judgement |
| `fraud_type_confidence` / `fraud_type_reason` | float / string | type judgement |
| `scene` / `scene_confidence` / `scene_reason` | string/float/string | scenario |
| `think` | string | model's reasoning trace |
| `caller_gender` / `callee_gender` | string | speaker genders (for TTS voices) |
| `audio_file` | string | relative path to the clip: `audio/{{lang}}/{{id}}.mp3` (Phase 2) |

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
"""
    path = os.path.join(root, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(front + body)
    print(f"[hf] README.md -> {os.path.relpath(path, PROJ)}")


def write_placeholder(root, sub, note):
    for lang in LANGS:
        os.makedirs(os.path.join(root, sub, lang), exist_ok=True)
    with open(os.path.join(root, sub, "README.md"), "w", encoding="utf-8") as f:
        f.write(note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(PROJ, "hf_dataset"))
    ap.add_argument("--langs", nargs="*", default=LANGS)
    ap.add_argument("--no-readme", action="store_true",
                    help="skip README regen (use when refreshing one language)")
    args = ap.parse_args()

    os.makedirs(args.root, exist_ok=True)
    built = [build_language(l, args.root) for l in args.langs]

    # README + card always reflect the FULL set (load counts for any lang not rebuilt)
    if not args.no_readme:
        stats = []
        for l in LANGS:
            b = next((x for x in built if x["lang"] == l), None)
            if b is None:
                rows = load(l)
                b = {"lang": l, "n": len(rows),
                     "counts": dict(collections.Counter(r["fraud_type_key"] for r in rows))}
            stats.append(b)
        write_readme(stats, args.root)

    write_placeholder(args.root, "audio",
        "# Audio (Phase 2)\n\nTTS clips go here as `audio/<lang>/<id>.mp3`, "
        "matching each row's `audio_file`. Empty until synthesis is uploaded.\n")
    write_placeholder(args.root, "preferences",
        "# Preference pairs (Phase 3)\n\nALM alignment chosen/rejected pairs go "
        "here as `preferences/<lang>/train.jsonl`. Empty until Phase 3.\n")
    print("[hf] done ->", args.root)


if __name__ == "__main__":
    main()
