# FraudSpeechAlignment

Sample English-language telecom fraud-call recordings for speech/text
alignment and fraud-detection research (subset of the TeleAntiFraud
dataset).

## 🔊 Audio gallery

A browsable page with an inline player for every sample is published via
GitHub Pages: **https://helixometry.github.io/FraudSpeechAlignment/**

(If Pages isn't enabled yet: repo **Settings → Pages → Build and
deployment → Source: "Deploy from a branch" → Branch: `main` / `root`**.)

## Contents

```
audio/NEG-gen-en/
  en_bank_00001/               bank fraud
  en_bank_00002/                bank fraud
  en_customer_service_00001/    customer-service scam
  en_customer_service_00002/    customer-service scam
  en_identity_theft_00001/      identity-theft scam
  en_investment_00001/          investment scam
  en_investment_00002/          investment scam
  en_kidnapping_00001/          kidnapping scam
  en_lottery_00001/             lottery scam
  en_phishing_00001/            phishing call
```

All 10 clips are synthetic re-enactments labelled `fraud`, used to train
and evaluate audio-based fraud-call classifiers.
