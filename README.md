# FraudSpeechAlignment

Tools, pipelines, and sample data for building **multilingual audio-text telecom-fraud
datasets** and using them to **align audio language models (ALMs)** toward fraud-aware,
protective behavior. Fraud-only, natively generated (not translated), modelled on the
Chinese *TeleAntiFraud-28k* dataset.

The full dataset — **FraudAlign-MCS** — is published (gated) on Hugging Face:
**[`ggirishg/MultiFraudAlign`](https://huggingface.co/datasets/ggirishg/MultiFraudAlign)**.

## 🌍 Languages

| Language | Code | Dialogues | Audio | TTS engine |
|---|---|---:|---:|---|
| English | `en` | 7,177 | ✅ | XTTS-v2 |
| Hindi | `hi` | 7,177 | ✅ | Indic-Parler-TTS |
| Korean | `ko` | 7,177 | ✅ | XTTS-v2 |
| Hinglish (Hindi-English) | `hinglish` | 7,177 | ✅ | Indic-Parler-TTS |

**28,708 dialogues** and **28,708 spoken 2-speaker clips**, across 7 fraud types
(bank, customer service, investment, phishing, lottery, kidnapping, identity theft).

## 🔊 Audio gallery

A simple page with an inline player for one sample per fraud type, in every language:
**https://helixometry.github.io/FraudSpeechAlignment/**

(If Pages isn't enabled yet: repo **Settings → Pages → Source: "Deploy from a branch" →
Branch: `main` / `root`**.) The curated sample clips live under `audio/NEG-gen-<lang>/`.

## 🛠️ Dataset pipeline — [`multiling_fraud/`](multiling_fraud/)

Reuse the TeleAntiFraud structure (same 7 fraud types, JSON/JSONL schema, and
scene → fraud → fraud-type task cascade) and regenerate fresh scam dialogues per language
with a strong LLM (Qwen2.5-72B), then synthesize spoken 2-speaker audio.

- **Set up on a new machine:** [`multiling_fraud/SETUP.md`](multiling_fraud/SETUP.md)
- **Current build state:** [`multiling_fraud/PROJECT_STATUS.md`](multiling_fraud/PROJECT_STATUS.md)
- **Design & how the Chinese dataset is used:** [`multiling_fraud/PIPELINE.md`](multiling_fraud/PIPELINE.md)

## Repository layout

```
FraudSpeechAlignment/
├── index.html, assets/        audio-gallery website (GitHub Pages)
├── audio/NEG-gen-<lang>/       curated sample fraud clips (.mp3), per language
└── multiling_fraud/           dataset generation pipeline (code + docs)
```

## Roadmap

1. **Dialogues** — generate all fraud dialogues (text). ✅ *(all 4 languages)*
2. **Audio** — synthesize 2-voice spoken calls for every dialogue. ✅ *(all 4 languages)*
3. **Preference pairs** — derive `+`/`−` preference data for ALM alignment. 🗓️ *next*

## Notes

Bulk generated data (full dialogue JSON, the complete audio set, archives) is intentionally
**not** committed — it is large and lives on the gated Hugging Face repo. Only the curated
gallery clips are kept. See `.gitignore`.
