# PIPELINE.md — How the English fraud dataset is built from the Chinese one

This document explains, end to end, how we produce the English fraud dataset
(`/users/msingh/sharedscratch/TeleAntiFraud_en/`) using the Chinese
**TeleAntiFraud-28k** dataset as a *blueprint*.

---

## 0. Core idea: copy the structure, regenerate the content

We do **not** translate the Chinese audio or text. Fraud is culturally and
linguistically specific, so translated Chinese scripts would not sound like real
English scams. Instead we:

1. **Extract the blueprint** from the Chinese dataset — its fraud taxonomy, scene
   list, JSON/JSONL schema, the scene→fraud→fraud_type task cascade, and its file
   conventions.
2. **Generate brand-new English scam calls** (dialogue text) that fit that blueprint.
3. **Synthesize** each call to 2-voice spoken audio.
4. **Assemble** everything into the *exact same schema* as the Chinese dataset.

The result is a native English dataset structurally identical to the original.

Scope: **fraud class only** (the Chinese `NEG` class). The legitimate/`POS` class
is intentionally skipped.

---

## 1. What we reuse FROM the Chinese dataset

Derived by analysing the downloaded `TeleAntiFraud` dataset:

| Chinese-dataset element | Where it lives now | How it is reused |
| --- | --- | --- |
| 7 fraud types (客服/银行/投资/钓鱼/彩票/绑架/身份盗窃) + their train counts | `config/taxonomy.py` | Same 7 types as English labels; original counts become **weights** so the English mix mirrors the original distribution |
| 7 scene types (订餐服务, 咨询客服, …) | `config/taxonomy.py`, `config/templates_en.py` | English closed-set used by the scene task |
| 3 cascading SFT tasks (scene 2-turn, fraud 4-turn, fraud_type 6-turn) | `config/templates_en.py` | Reproduced in English, same turn structure |
| Exact schema `{messages,prompts,answers,audios}` and `{prompt,answer}` | `config/templates_en.py` builders | Emitted byte-compatible with the original |
| Output formats (`{scene,reason,confidence}`, `<think>…</think><answer>{fraud_type…}</answer>`) | `config/templates_en.py` | Same formats, English text |
| Conventions: `NEG-`=fraud, `audio/<cat>/<id>/<id>.mp3` | pipeline scripts | We use `NEG-gen-en/<id>/<id>.mp3` |

> Reference: the verified analysis of the Chinese dataset is in
> `/users/msingh/sharedscratch/TeleAntiFraud/DATASET_TEXT_ANNOTATIONS_GUIDE.md`.

---

## 2. The three stages

```
config/taxonomy.py      ← 7 fraud types + weights (from the Chinese data)
config/templates_en.py  ← English task templates + original-schema record builders
        │
 STAGE 1  generate_dialogues.py   → out/dialogues*.json
        │   LLM writes ONE English scam call per item: turns[caller/callee] +
        │   scene / fraud / fraud_type reasons + confidences.
        │   Labels are TRUE BY CONSTRUCTION (we request "bank fraud" -> label = bank fraud).
        │   Backends:  --backend vllm  (Qwen2.5-72B on a full A100, for scale)
        │              --backend hf    (transformers, for small/robust runs)
        │              (sample runs: dialogues authored directly, same JSON shape)
        ▼
 STAGE 2  tts_synthesize.py        → audio/NEG-gen-en/<id>/<id>.mp3  + *_withaudio.json
        │   XTTS-v2, two voices (caller=Damien Black, callee=Ana Florence);
        │   each turn synthesized, concatenated with short pauses into ONE call mp3.
        ▼
 STAGE 3  assemble_dataset.py      → binary_classification/ + sft/ + dataset_manifest.json
        │   Each dialogue -> 1 binary record + 3 SFT records (2/4/6-turn cascade),
        │   in the EXACT original schema; train/test split.
        ▼
 /users/msingh/sharedscratch/TeleAntiFraud_en/   (same layout as the Chinese dataset)
```

### Why labels need no manual annotation
Because we generate on demand: asking the LLM for a "bank fraud, customer-service
scene" call means `fraud_type=bank fraud`, `scene=customer service`,
`is_fraud=true` are correct by construction. The LLM also writes the `reason` /
`confidence` text that fills the assistant turns of the cascade.

---

## 3. Style requirement (for ALM alignment)

The downstream goal is **AI alignment of audio LMs** via +ve/-ve preference pairs.
So the dialogues must be **realistic, with the callee (victim) actually being
scammed** — trusting, susceptible, gradually manipulated and complying — not a
savvy sceptic who hangs up. The Stage-1 prompt enforces this
(authority / fear / urgency / flattery / isolation; victim complies), while keeping
safety guardrails: fake names/companies/numbers only, no real working
links/apps/contacts, illustrative for detection (not an operational how-to),
and no narration of completed theft in operational detail.

Phase plan:
- **Phase 1 (now):** build the dialogue + audio dataset (scam / negative style).
- **Phase 2 (later):** build +ve/-ve preference pairs (e.g. a matched "victim
  resists / handled safely" positive per negative) for preference tuning.

---

## 4. Environments (Kelvin2 Slurm cluster)

Two conda envs on scratch (`/mnt/scratch2/users/$USER/conda/envs`), built by
`scripts/env_setup.sh`:

- **fraudgen** — vLLM + transformers + autoawq (LLM generation). Needs modern Rust
  (`conda install -c conda-forge "rust>=1.85"`) before `pip install vllm`.
- **fraudtts** — coqui-tts (XTTS-v2) + torch + torchcodec + pydub + ffmpeg.
  Note: `pip install coqui-tts` does NOT pull torch; also pin `transformers>=4.57,<5`
  and install `torchcodec` (torch>=2.9 audio IO).

GPU notes:
- Full-card **A100 (80 GB)** for the 72B vLLM generation.
- **Do NOT run vLLM on a MIG slice** (2g.20gb): it mis-reads the slice as the full
  80 GB card and hangs during memory profiling. MIG is fine for **TTS** and for the
  `--backend hf` generation with small models.

---

## 5. How to run

```bash
# One-time env build (on an A100 node)
bash scripts/env_setup.sh

# Full pipeline on a whole A100 (generate 72B -> TTS -> assemble)
sbatch scripts/job_pilot_a100.sh 50        # pilot
sbatch scripts/job_pilot_a100.sh 5724      # full fraud-scale run

# TTS + assemble only, on a pre-written dialogues file (e.g. sample runs, MIG ok)
sbatch scripts/job_tts_assemble.sh out/dialogues_pilot.json

# TTS multi-turn smoke test (no LLM)
sbatch scripts/job_tts_smoke.sh
```

Outputs:
- dialogues JSON  → `multiling_fraud/out/`
- audio           → `TeleAntiFraud_en/audio/NEG-gen-en/<id>/<id>.mp3`
- dataset         → `TeleAntiFraud_en/{binary_classification,sft}/` + `dataset_manifest.json`

---

## 6. File map

```
multiling_fraud/
├── config/
│   ├── taxonomy.py         7 fraud types (+weights), 7 scenes, pilot distribution
│   └── templates_en.py     English task templates + original-schema record builders
├── scripts/
│   ├── env_setup.sh        build fraudgen + fraudtts envs
│   ├── generate_dialogues.py   Stage 1 (vllm | hf backends)
│   ├── tts_synthesize.py       Stage 2 (XTTS-v2, 2-voice, multi-turn)
│   ├── assemble_dataset.py     Stage 3 (emit original schema, split)
│   ├── tts_smoke_test.py       tiny multi-turn TTS check
│   ├── job_pilot_a100.sh       full pipeline on a whole A100 (generate->TTS->assemble)
│   ├── job_tts_assemble.sh     Stage 2+3 on a given dialogues file (MIG-ok)
│   └── job_tts_smoke.sh        smoke-test batch job
├── out/                    dialogue JSONs (intermediate)
├── README.md               quickstart
└── PIPELINE.md             this document
```
