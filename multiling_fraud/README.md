# Multilingual Fraud Dataset (English) — a TeleAntiFraud-style, fraud-only, audio-text dataset

Generate a native **English** telecom-fraud dataset that mirrors the structure of the Chinese
**TeleAntiFraud-28k** dataset — same fraud taxonomy, same JSON/JSONL schema, same
scene → fraud → fraud-type task cascade, and real **spoken 2-speaker audio** — but produced
fresh in English (not translated) and containing **fraud calls only**.

Built to support **audio-LM (ALM) alignment**: the dialogues are the foundation for both the
audio and, later, `+`/`−` preference pairs.

---

## Status

| Stage | State |
| --- | --- |
| Pipeline (generate → TTS → assemble) | ✅ validated end-to-end (incl. Qwen2.5-72B on A100/H100) |
| Phase 1a — generate all 7,177 dialogues (text) | ▶️ running on held GPU → `out/dialogues_full.json` |
| Phase 1b — synthesize audio (2-voice, parallelizable) | ⏳ next |
| Phase 2 — build `+`/`−` preference pairs (from dialogue text) | 🗓️ planned |

---

## Approach

We treat TeleAntiFraud as a **blueprint**: reuse its *structure* (taxonomy, schema, task cascade,
conventions) and **regenerate the *content* natively in English** with a strong LLM, then
synthesize spoken audio. Fraud is culturally specific, so native generation beats translating
Chinese scripts. Details: **[PIPELINE.md](PIPELINE.md)**.

Every dialogue is a realistic scam in which the **victim is gradually manipulated and complies**
(the at-risk case a detector/ALM must learn), using well-known pressure tactics
(authority, fear, urgency, flattery, isolation). Safety guardrails: fake names/numbers only,
no working links/contacts, illustrative for detection — not an operational how-to.

---

## Repository layout

```
multiling_fraud/
├── README.md                     this file
├── PIPELINE.md                   full pipeline + how the Chinese dataset is used
├── config/
│   ├── taxonomy.py               SHARED: 7 fraud-type keys + original weights + N-distribution
│   ├── templates_en.py           English task templates + original-schema record builders
│   └── languages/
│       ├── __init__.py           get_language(code) loader
│       └── en.py                 English pack: language name, scenes, labels, TTS voices
├── scripts/                      (all take --lang, default en)
│   ├── generate_dialogues.py     Stage 1 — LLM → dialogues + annotations (vllm | hf backends)
│   ├── tts_synthesize.py         Stage 2 — XTTS-v2, 2-voice, multi-turn (shardable: --shard/--nshards)
│   ├── assemble_dataset.py       Stage 3 — emit original schema, train/test split
│   ├── env_setup.sh              build the two conda envs (fraudgen=vLLM, fraudtts=XTTS-v2)
│   ├── gen_only.sh <lang> <N>    generate ALL dialogues (text only) — Phase 1a
│   ├── run_full_fast.sh <lang> <N> <workers>   full run w/ parallel TTS (reuses dialogues if present)
│   ├── job_pilot_a100.sh         one-shot full pipeline on a whole A100
│   ├── job_hold_and_run.sh       persistent 3-day A100 hold (auto-runs validation, then holds)
│   └── job_hold_and_run_h100.sh  same, for H100
└── out/
    └── <lang>/dialogues_full.json    per-language dialogues — the single source of truth
```

Per-language output dataset → `/users/msingh/sharedscratch/TeleAntiFraud_<lang>/`
(`binary_classification/`, `sft/`, `audio/NEG-gen-<lang>/…`, `dataset_manifest.json`).

### Adding a new language

1. Add `config/languages/<code>.py` (copy `en.py`; set `LANGUAGE_NAME`, `SCENES`,
   `FRAUD_LABELS`, `VOICES`) and, if you want localized task prompts, a `templates_<code>.py`.
2. Run the pipeline with `--lang <code>` (or `gen_only.sh <code> <N>` / `run_full_fast.sh <code> <N> <workers>`).

---

## Fraud taxonomy (7 types, mirrors the original mix)

| Type | 中文 | Share of 7,177 |
| --- | --- | ---: |
| customer service fraud | 客服诈骗 | 2,536 |
| bank fraud | 银行诈骗 | 2,039 |
| investment fraud | 投资诈骗 | 984 |
| phishing fraud | 钓鱼诈骗 | 555 |
| lottery fraud | 彩票诈骗 | 524 |
| kidnapping fraud | 绑架诈骗 | 407 |
| identity theft | 身份盗窃 | 132 |

Each dialogue expands into the original task cascade: **scene** (2-turn) → **fraud** (4-turn) →
**fraud-type** (6-turn), plus a single-turn binary-classification record.

---

## Quickstart (Kelvin2 Slurm cluster)

```bash
# 0. one-time: build the conda envs on a GPU node
bash scripts/env_setup.sh

# 1a. generate all dialogues (text only) — fast, no audio
srun --jobid=<held-gpu-job> --overlap bash scripts/gen_only.sh en 7177
#    -> out/en/dialogues_full.json

# 1b. synthesize audio + assemble (parallel TTS workers; reuses the dialogues above)
srun --jobid=<held-gpu-job> --overlap bash scripts/run_full_fast.sh en 7177 4
#    -> /users/msingh/sharedscratch/TeleAntiFraud_en/
```

Grab a persistent GPU (queues once, holds 3 days, run work in via `--overlap`):
```bash
sbatch scripts/job_hold_and_run.sh        # A100
sbatch scripts/job_hold_and_run_h100.sh   # H100
```

---

## Notes

- **LLM:** `Qwen/Qwen2.5-72B-Instruct-AWQ` via vLLM (needs a full 80 GB card; **not** a MIG slice).
  Set `VLLM_USE_FLASHINFER_SAMPLER=0` (native sampler — avoids a CUDA JIT that needs `nvcc`).
- **TTS:** XTTS-v2, two built-in voices (`Damien Black` caller / `Ana Florence` callee).
  A single stream uses ~2 GB of 80 GB, so run several workers per GPU for speed
  (`--shard i --nshards N`).
- **Environments:** `fraudgen` (vLLM) and `fraudtts` (coqui-tts). Pin `transformers>=4.57,<5`,
  install `torchcodec`, and use Rust ≥1.85 for the vLLM build — see `env_setup.sh`.
```
