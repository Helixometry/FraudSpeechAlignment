# FraudSpeechAlignment

Tools and pipelines for building **audio-text telecom-fraud datasets** and using them to
**align audio language models (ALMs)** toward fraud-aware, protective behavior.

## Contents

- **[`multiling_fraud/`](multiling_fraud/)** — generate a native **English**, fraud-only
  dataset that mirrors the structure of the Chinese *TeleAntiFraud-28k* dataset
  (same fraud taxonomy, schema, and scene → fraud → fraud-type task cascade), with real
  spoken 2-speaker audio. Dialogues are generated with a strong LLM (Qwen2.5-72B) and
  synthesized with XTTS-v2.
  - See [`multiling_fraud/README.md`](multiling_fraud/README.md) for usage and
    [`multiling_fraud/PIPELINE.md`](multiling_fraud/PIPELINE.md) for the full design.

## Roadmap

1. **Dialogues** — generate all fraud dialogues (text). *(in progress)*
2. **Audio** — synthesize 2-voice spoken calls for every dialogue.
3. **Preference pairs** — derive `+`/`−` preference data from the dialogues for ALM alignment.

## Notes

Generated data (dialogue JSON, audio, archives) is intentionally **not** committed — it is
large and fully reproducible from the code. See `.gitignore`.
