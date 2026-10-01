# HANDOFF — resume FraudAlign-MCS on a new server

Single-page context + resume guide for continuing this project on a different machine
(moving because the current cluster's 80 GB GPUs are backed up for ~a day). Read this
first; deeper docs: **[SETUP.md](SETUP.md)** (full env/pipeline), **[PIPELINE.md](PIPELINE.md)**
(design), **[PROJECT_STATUS.md](PROJECT_STATUS.md)** (state).

---

## 1. What this project is
**FraudAlign-MCS** — a fraud-only, multilingual & code-switched scam-call dataset,
natively generated (not translated) with `Qwen2.5-72B-Instruct-AWQ`, modelled on the
Chinese TeleAntiFraud-28k (same schema, 7 fraud types, per-type proportions). Built to
align audio-language models (ALMs) toward fraud-aware behavior.

4 languages × **7,177 dialogues** = **28,708** dialogues, each with a spoken 2-speaker clip.

## 2. Where everything lives
- **Code:** GitHub `Helixometry/FraudSpeechAlignment` (this repo). All pipeline code +
  docs. Branch: `fix/caller-name-diversity-and-topup`.
- **Data (the important part):** Hugging Face **`ggirishg/MultiFraudAlign`** — *gated*
  (manual approval). Contains everything and is the source of truth for the new server:
  - `data/<lang>/train.jsonl` — dialogues + all labels + `audio_file` pointers (4 langs ✅)
  - `audio/<lang>/*.mp3` — 7,177 clips per language (28,708 total ✅)
  - `README.md` — dataset card + gated-access form
- **NOT in git / not on HF:** `out/<lang>/dialogues_full.json` (90 MB source) — **not needed**:
  `train.jsonl` carries the same dialogue content (incl. `turns`), and the labeler reads it directly.

## 3. Status
| Phase | State |
|---|---|
| Text (4 langs) | ✅ on HF |
| Audio (4 langs, 28,708 clips) | ✅ on HF (en/ko = XTTS-v2; hi/hinglish = Indic-Parler-TTS) |
| **Annotation labeling (current task)** | ⏳ **TODO on new server** — see §5 |
| Preference pairs (Phase 3) | 🗓️ later |
| Per-turn timestamps | ⏸️ deferred (would need forced alignment; skipped for now) |

## 4. Schema — what each row already has, and what we're adding
Present now: `id, language, turns[{speaker,text}], fraud_type_key, fraud_type, is_fraud,
scene, *_confidence, *_reason, think, caller_gender, callee_gender, audio_file`.

Two taxonomies for the research question **P(unsafe behavior | manipulation strategy)**:
- **Taxonomy 1 — fraud outcome (what attack):** `fraud_type_key` ∈ {bank, customer_service,
  investment, phishing, lottery, kidnapping, identity_theft}. ✅ already present.
- **Taxonomy 2 — manipulation mechanism (how) + ask + outcome:** added by the labeling
  task below as `manipulation_tactics` (7 binary), `requested_action`, `compliance_level`.

## 5. THE CURRENT TASK — annotation labeling (run this on the new server)
Label all 28,708 dialogues in one pass of the same 72B model (greedy/deterministic),
emitting three fields per dialogue. Script: `scripts/label_annotations.py`. It is
**checkpointed every 512 and resume-safe**, and reads either the HF `train.jsonl` or a
local `dialogues_full.json`.

**Taxonomy it produces** (definitions are embedded in the script):
- `manipulation_tactics`: `{authority, urgency, fear, affinity, reward, isolation, credential_request}` → 0/1 each
- `requested_action`: one of `otp_or_verification_code, password_or_pin, card_or_bank_details,
  personal_identity_info, install_app_or_remote_access, transfer_or_pay_money,
  buy_gift_cards_or_vouchers, click_link_or_visit_site, other, none`
- `compliance_level`: `full | partial | none`

### Steps on the new server
```bash
# (a) env with vLLM + the model (see SETUP.md §2 for the full recipe)
#     conda env `fraudgen`: vllm, transformers, autoawq, json-repair, huggingface_hub
# (b) get the data from HF (gated -> use your approved token)
export HF_TOKEN=...                      # your write/read token (keep OUT of git)
huggingface-cli download ggirishg/MultiFraudAlign --repo-type dataset \
    --include "data/*/train.jsonl" --local-dir ./hf_data
# (c) SMOKE TEST FIRST (5 dialogues) and eyeball the labels vs transcripts
python scripts/label_annotations.py --lang en --limit 5 \
    --dialogues ./hf_data/data/en/train.jsonl --out out/en/annotations.json
# (d) full run, per language (needs ONE full 80GB GPU for 72B-AWQ)
for L in en hi ko hinglish; do
  python scripts/label_annotations.py --lang $L \
      --dialogues ./hf_data/data/$L/train.jsonl --out out/$L/annotations.json
done
```
GPU: a full **80 GB A100/H100** for `Qwen2.5-72B-Instruct-AWQ`. If only smaller cards are
free, classification is easy enough that `Qwen2.5-32B-AWQ` (~40 GB) or `-14B-AWQ` (~20 GB)
works well — set `FRAUD_LLM=Qwen/Qwen2.5-32B-Instruct-AWQ`. Use the smoke test to confirm
quality before committing to a model.

### After labeling → rebuild + re-upload
```bash
# merges manipulation_tactics / requested_action / compliance_level into every row
python scripts/build_hf_dataset.py            # reads out/<lang>/annotations.json if present
export HF_TOKEN=...
python scripts/upload_hf_dataset.py --repo ggirishg/MultiFraudAlign \
    --include "README.md" "data/**"           # re-push curated text (audio untouched)
```
Then verify, commit the code, and update PROJECT_STATUS.md.

## 6. Gotchas carried over
- Prefix non-interactive Slurm launches with
  `unset PYTHONHOME PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV;` (else `init_fs_encoding`).
- vLLM needs a **full** 80 GB card (not a MIG slice) for the 72B; set
  `VLLM_USE_FLASHINFER_SAMPLER=0`.
- HF token: read at runtime from a file (e.g. `~/.hf_token`), never commit it.
- Audio TTS (if ever re-run): XTTS auto-chunks >400-token turns; Parler ≤3-4 workers/80 GB card.

## 7. Open holds on the OLD cluster (cancel after migrating)
`scancel 10067740 10067695` — the A100 + (down) H100 holds queued here for this task.
