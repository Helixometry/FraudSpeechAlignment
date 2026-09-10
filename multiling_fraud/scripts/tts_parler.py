#!/usr/bin/env python
"""
Multi-turn TTS backend using AI4Bharat Indic-Parler-TTS (natural, expressive Hindi).
Same orchestration as tts_synthesize.py (per-turn synth -> concatenate), but voices are
controlled by description prompts with consistent named speakers (Rohit=caller/male,
Divya=callee/female) so the call sounds spoken, not read.

Run (in the `fraudparler` env, on a GPU):
  python scripts/tts_parler.py --lang hi --dialogues out/hi/dialogues_sample6.json \
      --audio-root /users/msingh/sharedscratch/TeleAntiFraud_hi_parler
"""
import argparse, json, os, tempfile

MODEL = "ai4bharat/indic-parler-tts"

# Named Hindi speakers per gender (rotated across calls for speaker variety).
MALE_VOICES = ["Rohit", "Aman"]
FEMALE_VOICES = ["Divya", "Rani"]

# Role-specific manner (fixes the flat "reading" feel); {v} = chosen speaker name.
CALLER_STYLE = ("{v} speaks in an expressive, persuasive and confident tone at a slightly fast "
                "pace, sounding reassuring. The recording is very clear and close-sounding.")
CALLEE_STYLE = ("{v} speaks in an expressive, slightly anxious and hesitant tone at a moderate "
                "pace. The recording is very clear and close-sounding.")


def pick_voice(gender, idx):
    """Deterministically rotate a same-gender named speaker for variety across calls."""
    pool = FEMALE_VOICES if str(gender).lower().startswith("f") else MALE_VOICES
    return pool[idx % len(pool)]

_HI_DIGITS = {"0": "शून्य", "1": "एक", "2": "दो", "3": "तीन", "4": "चार",
              "5": "पाँच", "6": "छह", "7": "सात", "8": "आठ", "9": "नौ"}


def preprocess(text, lang):
    if lang == "hi":
        text = text.replace(",", "")
        text = "".join(f" {_HI_DIGITS[c]} " if c in _HI_DIGITS else c for c in text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", required=True)
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--audio-root", default="/users/msingh/sharedscratch/TeleAntiFraud_hi_parler")
    ap.add_argument("--subdir", default=None)
    ap.add_argument("--pause-ms", type=int, default=350)
    ap.add_argument("--out", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    if args.subdir is None:
        args.subdir = f"NEG-gen-{args.lang}"

    import torch, soundfile as sf, numpy as np
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    from pydub import AudioSegment

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[parler] loading {MODEL} on {dev}", flush=True)
    model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL).to(dev)
    tok = AutoTokenizer.from_pretrained(MODEL)
    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    sr = model.config.sampling_rate

    def enc_desc(text):
        return desc_tok(text, return_tensors="pt").to(dev)

    dialogues = json.load(open(args.dialogues))
    if args.nshards > 1:
        dialogues = [d for i, d in enumerate(dialogues) if i % args.nshards == args.shard]
    pause = AudioSegment.silent(duration=args.pause_ms)
    root = os.path.join(args.audio_root, "audio", args.subdir)

    def synth(text, desc_enc, path_wav):
        p = tok(preprocess(text, args.lang), return_tensors="pt").to(dev)
        with torch.no_grad():
            gen = model.generate(input_ids=desc_enc.input_ids, attention_mask=desc_enc.attention_mask,
                                 prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
        sf.write(path_wav, gen.cpu().numpy().squeeze().astype("float32"), sr)
        return AudioSegment.from_wav(path_wav)

    done = 0
    for idx, dlg in enumerate(dialogues):
        did = dlg["id"]; cdir = os.path.join(root, did); os.makedirs(cdir, exist_ok=True)
        # resume: skip clips already synthesized (survives GPU/hold restarts)
        _mp3 = os.path.join(cdir, f"{did}.mp3")
        if os.path.exists(_mp3) and os.path.getsize(_mp3) > 2000:
            dlg["audio"] = os.path.join("audio", args.subdir, did, f"{did}.mp3")
            done += 1
            continue
        # gender-matched, rotated voices; consistent within this call
        caller_v = pick_voice(dlg.get("caller_gender", "male"), idx)
        callee_v = pick_voice(dlg.get("callee_gender", "female"), idx)
        desc_enc = {
            "caller": enc_desc(CALLER_STYLE.format(v=caller_v)),
            "callee": enc_desc(CALLEE_STYLE.format(v=callee_v)),
        }
        call = AudioSegment.silent(duration=0)
        with tempfile.TemporaryDirectory() as tmp:
            for i, t in enumerate(dlg["turns"]):
                txt = (t.get("text") or "").strip()
                if not txt:
                    continue
                role = "caller" if t.get("speaker") == "caller" else "callee"
                call += synth(txt, desc_enc[role], os.path.join(tmp, f"{i}.wav")) + pause
        mp3 = os.path.join(cdir, f"{did}.mp3")
        call.export(mp3, format="mp3", bitrate="64k")
        dlg["audio"] = os.path.join("audio", args.subdir, did, f"{did}.mp3")
        done += 1
        print(f"[parler] {done}/{len(dialogues)}  {did}  ({len(call)/1000:.1f}s)", flush=True)

    out = args.out or args.dialogues.replace(".json", "_withaudio.json")
    json.dump(dialogues, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"[parler] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
