# PROJECT_STATUS — FraudAlign-MCS multilingual dataset

Snapshot of where the dataset build stands and the key decisions behind it.
(Setup/reproduction steps live in **[SETUP.md](SETUP.md)**; pipeline design in
**[PIPELINE.md](PIPELINE.md)**.)

Last updated: 2026-09-22.

## What this is
A **fraud-only** multilingual + code-switched scam-call dataset, natively generated
(not translated) with `Qwen2.5-72B-Instruct-AWQ`, mirroring the schema / fraud
taxonomy / per-type proportions of the Chinese **TeleAntiFraud-28k**. Built to support
alignment of audio-language models (ALMs). Published (gated) at
`ggirishg/MultiFraudAlign` on the HuggingFace Hub as **FraudAlign-MCS**.

Four languages, **7,177 dialogues each**: English (`en`), Hindi (`hi`),
Korean (`ko`), Hinglish (`hinglish`).

## Phase status

| Phase | Content | State |
|---|---|---|
| 1 — text | `data/<lang>/train.jsonl` (dialogues + labels + `audio_file` pointer) | ✅ all 4 languages on the Hub |
| 2 — audio | `audio/<lang>/<id>.mp3` (2-speaker TTS) | 🔄 en ✅, hi ✅, **ko ✅ (7,177)**, hinglish ▶️ generating |
| 3 — preference pairs | `preferences/<lang>/` (chosen/rejected for ALM alignment) | 🗓️ planned |

## Audio backends (per language)
- **English, Korean → XTTS-v2** (`fraudtts` env, `tts_range.sh` → `tts_synthesize.py`)
- **Hindi, Hinglish → Indic-Parler-TTS** (`fraudparler` env, `parler_range.sh` → `tts_parler.py`)

Both are shardable (`--shard/--nshards`) and **resume-safe** (skip existing clips).
Output: `TeleAntiFraud_<lang>/audio/NEG-gen-<lang>/<id>/<id>.mp3`.

## HuggingFace repo (`ggirishg/MultiFraudAlign`)
- **Gated: manual approval.** Access form (name, org, country, intended-use dropdown,
  4 agreement checkboxes) + card are generated into `hf_dataset/README.md` by
  `build_hf_dataset.py`.
- Viewer: `en` is the default config; `all` combines every language.
- Forward-compatible: each language is its own config and every text row already
  carries its `audio_file` path, so audio (Phase 2) and preferences (Phase 3) drop in
  **without rewriting earlier data**.

## Key engineering decisions / fixes
- **Multi-worker TTS across held GPUs** for throughput: XTTS scales to ~8 workers per
  80 GB card (~2-4 GB each); **Parler is memory-heavy → ≤3-4 workers/card** (more → OOM).
  Korean was accelerated from 2→16 workers across two A100s.
- **Long-turn chunking** in `tts_synthesize.py`: XTTS aborts above ~400 tokens. Turns
  are split at sentence boundaries (Latin/CJK/Devanagari) and re-concatenated — **full
  text preserved, no truncation**. This unblocked the last ~90 Korean clips.
- **Per-clip failure isolation**: one bad dialogue no longer crashes a whole shard.
- **Re-sharding to fill gaps**: because skipping is file-based, remaining clips can be
  fanned across more workers with a different `nshards` (e.g. leftovers of
  `nshards=16 shards 8-15` == `nshards=32 shards {8-15,24-31}`).
- **srun conda leak**: always `unset PYTHONHOME PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV`
  before launching, else `init_fs_encoding` failure.
- **3-day Slurm wall limit**: long audio runs can time out mid-way; resume-safety means
  just relaunch on a fresh hold — no data loss.

## Not in git (by design)
- `out/<lang>/dialogues_full.json` — the ~90 MB text source of truth (ignored;
  reproducible via the pipeline and published on HF).
- `*.mp3` / `*.wav` audio (published on the gated HF repo).
- HF token — read at runtime from `/users/msingh/GPU/.hf_token`.
