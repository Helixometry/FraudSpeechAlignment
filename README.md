# FraudSpeechAlignment

Tools, pipelines, and sample data for building **audio-text telecom-fraud datasets** and using
them to **align audio language models (ALMs)** toward fraud-aware, protective behavior.
English-language, fraud-only, modelled on the Chinese *TeleAntiFraud-28k* dataset.

## 🔊 Audio gallery

A browsable page with an inline player for every sample clip is published via GitHub Pages:
**https://helixometry.github.io/FraudSpeechAlignment/**

(If Pages isn't enabled yet: repo **Settings → Pages → Build and deployment →
Source: "Deploy from a branch" → Branch: `main` / `root`**.)

The 10 clips under `audio/NEG-gen-en/` are synthetic re-enactments labelled `fraud`
(bank, customer-service, investment, phishing, lottery, kidnapping, identity-theft),
used to train and evaluate audio-based fraud-call classifiers.

## 🛠️ Dataset pipeline — [`multiling_fraud/`](multiling_fraud/)

Generate the full dataset natively in English: reuse the TeleAntiFraud structure
(same 7 fraud types, JSON/JSONL schema, and scene → fraud → fraud-type task cascade) and
regenerate fresh scam dialogues with a strong LLM (Qwen2.5-72B), then synthesize spoken
2-speaker audio with XTTS-v2.

- Usage: [`multiling_fraud/README.md`](multiling_fraud/README.md)
- Full design & how the Chinese dataset is used: [`multiling_fraud/PIPELINE.md`](multiling_fraud/PIPELINE.md)

## Repository layout

```
FraudSpeechAlignment/
├── index.html, assets/      audio-gallery website (GitHub Pages)
├── audio/NEG-gen-en/        10 curated sample fraud clips (.mp3)
└── multiling_fraud/         dataset generation pipeline (code + docs)
```

## Roadmap

1. **Dialogues** — generate all fraud dialogues (text). *(in progress)*
2. **Audio** — synthesize 2-voice spoken calls for every dialogue.
3. **Preference pairs** — derive `+`/`−` preference data from the dialogues for ALM alignment.

## Notes

Bulk generated data (full dialogue JSON, the complete audio set, archives) is intentionally
**not** committed — it is large and fully reproducible from the code. The 10 gallery clips are
kept on purpose. See `.gitignore`.
