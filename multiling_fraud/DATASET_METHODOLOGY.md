# DATASET_METHODOLOGY — how FraudAlign-MCS was actually built

A concrete, reproducible record of exactly how the complete dataset (dialogues +
audio, all four languages) was created, with the real models and parameters used.
For the design rationale see [PIPELINE.md](PIPELINE.md); for env/run commands see
[SETUP.md](SETUP.md).

Dataset: **4 languages × 7,177 dialogues = 28,708**, each with a 2-speaker audio clip.
Languages: English (`en`), Hindi (`hi`), Korean (`ko`), Hinglish (`hinglish`).

---

## Stage 0 — blueprint from TeleAntiFraud-28k (no content copied)
We reused only the *structure* of the Chinese TeleAntiFraud-28k dataset — its fraud
taxonomy, JSON schema, the scene→fraud→fraud-type task cascade, and the per-type
**proportions** — then regenerated all content natively per language. No Chinese text
or audio was translated or copied.

**Fraud taxonomy (7 types) and realized per-language counts** (identical in every
language, mirroring TeleAntiFraud's mix):

| fraud_type_key | count | share |
|---|---:|---:|
| customer_service | 2,536 | 35.3% |
| bank | 2,039 | 28.4% |
| investment | 984 | 13.7% |
| phishing | 555 | 7.7% |
| lottery | 524 | 7.3% |
| kidnapping | 407 | 5.7% |
| identity_theft | 132 | 1.8% |
| **total** | **7,177** | 100% |

---

## Stage 1 — dialogue generation (text)
**Script:** `scripts/generate_dialogues.py` · **Model:** `Qwen/Qwen2.5-72B-Instruct-AWQ`
(4-bit AWQ) served with **vLLM** on a full 80 GB GPU (`gpu_memory_utilization=0.92`,
`max_model_len=4096`, `VLLM_USE_FLASHINFER_SAMPLER=0`).

- **Sampling:** `temperature=1.0`, `top_p=0.95`, `max_tokens=1500`, no fixed per-seq seed
  (so identical prompts don't collapse to identical text; extra diversity from per-call
  profiles injected into the prompt).
- **Per call:** the model writes ONE realistic scam transcript of **6–10 turns**
  (`caller`/`callee`), plus the annotations that fill the task cascade — `scene`,
  `fraud_reason`, `fraud_type_reason`, their `*_confidence`, and a `think` trace.
- **Labels are correct by construction:** we *request* e.g. "bank fraud, customer-service
  scene", so `fraud_type`/`scene`/`is_fraud=true` are known without manual annotation.
- **Style requirement (for ALM alignment):** the callee is a believable victim who is
  gradually manipulated and complies, using well-known social-engineering pressure
  (authority, fear, urgency, reassurance, flattery, isolation).
- **Safety guardrails baked into the prompt:** fake names/companies/numbers only; no real
  working links/apps/contacts; illustrative for detection, not an operational how-to.
- **Per-language packs** (`config/languages/<code>.py`) supply the language name, scene
  list, localized fraud labels, local person-name pools, and TTS voices. Output is the
  single source of truth per language: `out/<lang>/dialogues_full.json`.
- **Robustness:** malformed JSON is repaired (`json_repair`) rather than dropped.

Fraud-only scope: we generate the TeleAntiFraud `NEG` (fraud) class only; the
legitimate/`POS` class is intentionally skipped.

---

## Stage 2 — audio synthesis (2-speaker calls)
Each dialogue → one `.mp3`: every turn is synthesized separately with a gender-matched
voice (caller vs callee always **different** speakers), then concatenated with a **350 ms**
pause between turns, hard-capped at **90 s** on a turn boundary, exported at **64 kbps mp3**.
Layout: `TeleAntiFraud_<lang>/audio/NEG-gen-<lang>/<id>/<id>.mp3`.

Two backends, split by language:

### English & Korean — XTTS-v2 (`scripts/tts_synthesize.py`, `fraudtts` env)
- **Model:** `tts_models/multilingual/multi-dataset/xtts_v2` (Coqui).
- **Voices:** gender-matched pool (English e.g. male *Damien Black, Craig Gutsy, …* /
  female *Ana Florence, Claribel Dervla, …*), rotated per dialogue so speakers vary.
- **Hindi-digit note:** XTTS expands digits via num2words (no Hindi) — for such languages
  digits are spelled out before synthesis to avoid crashes.
- **Long turns:** text above XTTS's ~400-token limit is split at sentence boundaries
  (Latin/CJK/Devanagari), synthesized in pieces, and re-joined — full text preserved.

### Hindi & Hinglish — Indic-Parler-TTS (`scripts/tts_parler.py`, `fraudparler` env)
- **Model:** `ai4bharat/indic-parler-tts` (gated; HF token required).
- **Voices:** named speakers — male *Rohit/Aman*, female *Divya/Rani* — gender-matched.
- **Voice control via style prompt:** caller = "expressive, persuasive, confident,
  slightly fast"; callee = "expressive, slightly anxious, hesitant, moderate" — so the
  scammer vs victim sound distinct.
- **Batched:** a call's turns are batched (fewer model calls) then concatenated.

### How it was run at scale
- Both backends are **shardable** (`--shard i --nshards N`, dialogue `i % N == shard`) and
  **resume-safe** (existing non-empty `<id>.mp3` is skipped), so work was split across
  multiple held GPUs and relaunched freely after timeouts.
- Throughput tuning observed: XTTS ~2–4 GB/worker → up to ~8 workers per 80 GB card;
  Indic-Parler ~12 GB/worker → ≤3–4 workers per 80 GB card (more → CUDA OOM).
- **Per-clip failure isolation** in both scripts: a clip that errors (bad text, transient
  OOM) is logged and skipped so it can't crash the rest of a shard; it's re-synthesized
  afterwards on its own.
- Every language was validated before upload: 7,177 clips, 0 zero-byte, unique ids, all
  dialogue ids matched, sample decode + duration/non-silence check.

---

## Stage 3 — assemble & publish
- `scripts/build_hf_dataset.py` writes `hf_dataset/data/<lang>/train.jsonl` (one JSON
  object per dialogue, with an `audio_file` pointer and, once labelled, the
  manipulation-mechanism annotations) and the dataset card.
- `scripts/upload_hf_dataset.py` pushes phase-by-phase (text, then audio) to the **gated**
  Hugging Face repo `ggirishg/MultiFraudAlign` via `upload_large_folder` (resumable).

## Per-dialogue fields (final schema)
`id, language, turns[{speaker,text}], fraud_type_key, fraud_type, is_fraud, scene,
scene_reason, scene_confidence, fraud_reason, fraud_confidence, fraud_type_reason,
fraud_type_confidence, think, caller_gender, callee_gender, audio_file`
— plus (added by `scripts/label_annotations.py`) `manipulation_tactics`,
`requested_action`, `compliance_level`. See [HANDOFF.md](HANDOFF.md) for that step.
