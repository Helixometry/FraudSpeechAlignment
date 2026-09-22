# SETUP — reproduce the multilingual fraud pipeline on another system

End-to-end guide to stand this project up from scratch on a new machine (built and
run on the **Kelvin2 Slurm cluster**, but the steps are portable to any Linux + CUDA
box with one or more 80 GB GPUs). Covers all four languages and both TTS backends.

> Pipeline in one line: **generate dialogue text (LLM) → synthesize 2-speaker audio
> (TTS) → assemble the HuggingFace dataset → push (gated) to the Hub.**

---

## 0. Languages & which engine makes their audio

| Lang code | Language | Text gen | **TTS backend** | conda env |
|---|---|---|---|---|
| `en` | English | Qwen2.5-72B | **XTTS-v2** | `fraudtts` |
| `ko` | Korean | Qwen2.5-72B | **XTTS-v2** | `fraudtts` |
| `hi` | Hindi | Qwen2.5-72B | **Indic-Parler-TTS** | `fraudparler` |
| `hinglish` | Hindi-English code-switch | Qwen2.5-72B | **Indic-Parler-TTS** | `fraudparler` |

Each language is **7,177 dialogues** (mirrors TeleAntiFraud-28k's fraud-type mix).

---

## 1. Prerequisites

- Linux + NVIDIA driver, CUDA-capable GPU(s). **One full 80 GB A100/H100** for the
  72B LLM (a MIG slice will *not* work for vLLM). TTS runs on any modern GPU.
- Anaconda/Miniconda.
- `ffmpeg` (pulled in by the env scripts via conda-forge).
- A HuggingFace account + **write token** (for pushing the dataset) and access to the
  gated `ai4bharat/indic-parler-tts` model (for Hindi/Hinglish).

### Paths this repo assumes (override via env vars / flags)
```
conda envs   : /mnt/scratch2/users/$USER/conda/envs/{fraudgen,fraudtts,fraudparler}
HF cache     : /mnt/scratch2/users/$USER/hf_cache
audio output : /users/msingh/sharedscratch/TeleAntiFraud_<lang>/
HF token file: /users/msingh/GPU/.hf_token       # NEVER committed; read at runtime
```

---

## 2. Build the three conda envs

vLLM and the two TTS stacks have conflicting deps, so they live in separate envs.

```bash
# on a GPU node
bash scripts/env_setup.sh          # -> fraudgen (vLLM+Qwen) + fraudtts (XTTS-v2)
bash scripts/env_setup_parler.sh   # -> fraudparler (Indic-Parler-TTS)
```

Gotchas baked into those scripts (worth knowing if you hand-build):
- **fraudgen**: vLLM pulls Rust-built deps needing edition2024 → `rust>=1.85` first.
  Run the 72B with `VLLM_USE_FLASHINFER_SAMPLER=0` (native sampler; avoids a CUDA JIT
  that needs `nvcc`).
- **fraudtts**: `pip install coqui-tts` does **not** install torch — install torch
  first, then `coqui-tts[codec]` (torchcodec for torch≥2.9 audio IO), and pin
  `transformers>=4.57,<5` (5.x removed a symbol coqui-tts needs).
- **fraudparler**: installs `parler-tts` from GitHub; the model
  `ai4bharat/indic-parler-tts` is **gated** → export your HF token (below) before first run.

### HF token (kept out of git)
```bash
mkdir -p /users/msingh/GPU
printf '%s' 'hf_xxxYOURTOKENxxx' > /users/msingh/GPU/.hf_token   # write+gated-read scope
chmod 600 /users/msingh/GPU/.hf_token
```
Every script reads it via `HF_TOKEN=$(cat /users/msingh/GPU/.hf_token)` — the token is
never hard-coded and `.hf_token` is outside the repo tree.

---

## 3. Grab a GPU (Slurm)

```bash
sbatch scripts/job_hold_and_run.sh        # hold one A100 for 3 days
sbatch scripts/job_hold_and_run_h100.sh   # or an H100
squeue -u $USER                           # wait for ST=R, note the JOBID + node
```

Attach and run work inside the held job with `--overlap`:
```bash
srun --jobid=<JOBID> --overlap --pty bash
```

> **Always** prefix non-interactive `srun` launches with
> `unset PYTHONHOME PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV;`
> or the launching shell's conda vars leak in and Python fails with
> `init_fs_encoding`.

---

## 4. Stage 1 — generate dialogue text

```bash
srun --jobid=<JOBID> --overlap bash scripts/gen_only.sh <lang> 7177
#   -> out/<lang>/dialogues_full.json     (the single source of truth for that language)
```
Labels are correct by construction (we ask the LLM for e.g. "bank fraud, customer-service
scene", so `fraud_type`/`scene`/`is_fraud` are known). See `PIPELINE.md` for details and
for **adding a new language** (add `config/languages/<code>.py`).

---

## 5. Stage 2 — synthesize audio (the part you scale across workers)

Both backends are **shardable and resume-safe**: each worker takes `--shard i
--nshards N` and processes dialogues where `index % N == i`; already-made clips
(`<id>.mp3` present and non-empty) are **skipped**, so re-running only fills gaps.
Output layout: `TeleAntiFraud_<lang>/audio/NEG-gen-<lang>/<id>/<id>.mp3`.

### English / Korean — XTTS-v2 (`tts_range.sh`)
```bash
# 8 workers on one GPU (nshards=16, this GPU does shards 0-7):
srun --jobid=<JOBID> --overlap bash -c \
  "unset PYTHONHOME PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV; \
   bash scripts/tts_range.sh ko 16 0 1 2 3 4 5 6 7"
# a SECOND GPU finishes the other half:
#   bash scripts/tts_range.sh ko 16 8 9 10 11 12 13 14 15
```
XTTS uses ~2-4 GB/worker → up to ~8 workers per 80 GB card. XTTS has a **400-token
per-call limit**; `tts_synthesize.py` auto-splits long turns at sentence boundaries
(full text preserved) and isolates any per-clip failure so it can't kill a shard.

### Hindi / Hinglish — Indic-Parler-TTS (`parler_range.sh`)
```bash
srun --jobid=<JOBID> --overlap bash -c \
  "unset PYTHONHOME PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV; \
   bash scripts/parler_range.sh hinglish 3 0 1 2"
```
Parler is **much heavier (~12 GB/worker)** → keep to **≤3-4 workers per 80 GB card**
(more → CUDA OOM). Speaker names in the Hindi/Hinglish packs (Rohit/Aman male,
Divya/Rani female) are Indic-Parler voice names.

### Re-sharding to fill a gap without redoing work
Because skipping is file-based, you can re-run with a *different* `nshards` to fan the
remaining clips across more workers. The already-done half is skipped instantly; e.g.
the clips left by `nshards=16 shards 8-15` are exactly `nshards=32 shards {8-15,24-31}`.

---

## 6. Stage 3 — build & push the HuggingFace dataset

```bash
conda activate .../fraudgen           # or any env with huggingface_hub
export HF_TOKEN=$(cat /users/msingh/GPU/.hf_token)

# assemble hf_dataset/ from out/<lang>/dialogues_full.json (text + audio_file pointers)
python scripts/build_hf_dataset.py                 # all languages
python scripts/build_hf_dataset.py --langs en      # refresh just one

# push incrementally (phases are independent):
python scripts/upload_hf_dataset.py --repo ggirishg/MultiFraudAlign \
    --include "README.md" "data/**"                # Phase 1: text
python scripts/upload_hf_dataset.py --repo ggirishg/MultiFraudAlign \
    --include "audio/**"                           # Phase 2: audio
```
The repo is **gated (manual approval)**; the access form and dataset card live in the
YAML front-matter that `build_hf_dataset.py` writes into `hf_dataset/README.md`.

### Staging audio into the HF tree
Rows point at `audio/<lang>/<id>.mp3`, while synthesis writes
`.../NEG-gen-<lang>/<id>/<id>.mp3`. Before pushing audio, flatten via symlinks:
```bash
for f in $TELE/TeleAntiFraud_<lang>/audio/NEG-gen-<lang>/*/*.mp3; do
  ln -sf "$f" hf_dataset/audio/<lang>/$(basename "$f"); done
```

---

## 7. Validate a language before upload
```bash
find $TELE/TeleAntiFraud_<lang> -name '*.mp3' | wc -l      # expect 7177
# unique ids, no zero-byte files:
find $TELE/TeleAntiFraud_<lang> -name '*.mp3' -size 0 | wc -l   # expect 0
```

---

## 8. Troubleshooting quick-ref

| Symptom | Cause / fix |
|---|---|
| `init_fs_encoding` on srun | conda vars leaked → `unset PYTHONHOME PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV` first |
| `XTTS can only generate ... 400 tokens` | long turn → handled by `chunk_text` in `tts_synthesize.py`; update if you see it |
| CUDA OOM during Parler | too many workers → drop to ≤3 per 80 GB card |
| vLLM hangs at memory profiling | running on a MIG slice → use a full 80 GB card |
| HF viewer "available soon" forever | malformed `configs:` YAML → one split with a path list (see `build_hf_dataset.py`) |
| Job vanished from `squeue` mid-run | hit the 3-day wall limit → resume is safe, just relaunch on a new hold |
