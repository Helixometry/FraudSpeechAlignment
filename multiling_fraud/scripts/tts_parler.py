#!/usr/bin/env python
"""
Multi-turn TTS with AI4Bharat Indic-Parler-TTS, BATCHED per call for speed:
all turns of a dialogue are synthesized in ONE batched model.generate() call
(~14x fewer calls than per-turn), then concatenated. Gender-matched, rotated named
speakers (Rohit/Aman male, Divya/Rani female). Resume-safe (skips done clips).

Run (fraudparler env, GPU; HF_TOKEN set for the gated model):
  python scripts/tts_parler.py --lang hi --dialogues out/hi/dialogues_full.json \
      --audio-root /users/msingh/sharedscratch/TeleAntiFraud_hi --shard 0 --nshards 2
"""
import argparse, json, os, re, tempfile

MODEL = "ai4bharat/indic-parler-tts"
MALE_VOICES = ["Rohit", "Aman"]
FEMALE_VOICES = ["Divya", "Rani"]
CALLER_STYLE = ("{v} speaks in an expressive, persuasive and confident tone at a slightly fast "
                "pace, sounding reassuring. The recording is very clear and close-sounding.")
CALLEE_STYLE = ("{v} speaks in an expressive, slightly anxious and hesitant tone at a moderate "
                "pace. The recording is very clear and close-sounding.")
_HI_DIGITS = {"0": "शून्य", "1": "एक", "2": "दो", "3": "तीन", "4": "चार",
              "5": "पाँच", "6": "छह", "7": "सात", "8": "आठ", "9": "नौ"}
# safety net: render unavoidable acronyms in Devanagari, drop any other Latin run so
# the Hindi model never voices Latin as gibberish ("speaking something else").
_HI_LATIN_MAP = {"otp": "ओटीपी", "pin": "पिन", "sms": "एसएमएस", "atm": "एटीएम",
                 "upi": "यूपीआई", "cvv": "सीवीवी", "id": "आईडी", "app": "ऐप",
                 "link": "लिंक", "email": "ईमेल", "kyc": "केवाईसी"}
_LATIN_RUN = re.compile(r'[A-Za-z][A-Za-z0-9._\-@]*')


def _strip_latin_hi(text):
    def repl(m):
        return _HI_LATIN_MAP.get(m.group(0).lower(), " ")   # map known, else drop
    return _LATIN_RUN.sub(repl, text)


def preprocess(text, lang):
    if lang == "hi":
        text = _strip_latin_hi(text)
        text = text.replace(",", "")
        text = "".join(f" {_HI_DIGITS[c]} " if c in _HI_DIGITS else c for c in text)
        text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


def _pool(gender):
    return FEMALE_VOICES if str(gender).lower().startswith("f") else MALE_VOICES


def assign_voices(caller_gender, callee_gender, idx):
    """Gender-matched voices for caller & callee, guaranteed to be DIFFERENT speakers
    (even when both are the same gender). Rotates by idx for cross-call variety."""
    cp, ep = _pool(caller_gender), _pool(callee_gender)
    caller = cp[idx % len(cp)]
    callee = ep[idx % len(ep)]
    if caller == callee:                        # same gender -> force the other speaker
        callee = ep[(idx + 1) % len(ep)]
    return caller, callee


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", required=True)
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--audio-root", default="/users/msingh/sharedscratch/TeleAntiFraud_hi")
    ap.add_argument("--subdir", default=None)
    ap.add_argument("--pause-ms", type=int, default=350)
    ap.add_argument("--out", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-batch", type=int, default=20, help="cap turns per generate call")
    ap.add_argument("--max-seconds", type=float, default=90.0, help="hard cap on clip length (turn boundary)")
    args = ap.parse_args()
    if args.subdir is None:
        args.subdir = f"NEG-gen-{args.lang}"

    import numpy as np, torch, soundfile as sf
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    from pydub import AudioSegment

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[parler] loading {MODEL} on {dev}", flush=True)
    model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL).to(dev)
    tok = AutoTokenizer.from_pretrained(MODEL)
    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    sr = model.config.sampling_rate
    pause = np.zeros(int(args.pause_ms / 1000 * sr), dtype=np.float32)

    dialogues = json.load(open(args.dialogues))
    if args.nshards > 1:
        dialogues = [d for i, d in enumerate(dialogues) if i % args.nshards == args.shard]
    root = os.path.join(args.audio_root, "audio", args.subdir)

    def trim(a, thr=0.01, pad_ms=60):
        idx = np.where(np.abs(a) > thr)[0]
        if len(idx) == 0:
            return a[: int(0.08 * sr)]
        return a[: min(len(a), idx[-1] + int(pad_ms / 1000 * sr))]

    def synth_batch(descs, prompts):
        di = desc_tok(descs, return_tensors="pt", padding=True).to(dev)
        pi = tok(prompts, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            gen = model.generate(input_ids=di.input_ids, attention_mask=di.attention_mask,
                                 prompt_input_ids=pi.input_ids, prompt_attention_mask=pi.attention_mask)
        arr = gen.cpu().numpy()
        if arr.ndim == 3:
            arr = arr.squeeze(1)
        return [trim(arr[k].astype("float32")) for k in range(arr.shape[0])]

    done = 0
    for dlg in dialogues:
        did = dlg["id"]; cdir = os.path.join(root, did); mp3 = os.path.join(cdir, f"{did}.mp3")
        if os.path.exists(mp3) and os.path.getsize(mp3) > 2000:   # resume
            dlg["audio"] = os.path.join("audio", args.subdir, did, f"{did}.mp3"); done += 1; continue
        os.makedirs(cdir, exist_ok=True)
        n = int(re.findall(r"\d+", did)[-1]) if re.findall(r"\d+", did) else done
        cv, ev = assign_voices(dlg.get("caller_gender", "male"), dlg.get("callee_gender", "female"), n)
        descs, prompts = [], []
        for t in dlg["turns"]:
            txt = (t.get("text") or "").strip()
            if not txt:
                continue
            style = CALLER_STYLE if t.get("speaker") == "caller" else CALLEE_STYLE
            descs.append(style.format(v=cv if t.get("speaker") == "caller" else ev))
            prompts.append(preprocess(txt, args.lang))
        # batch the clip's turns (split if above cap)
        segs = []
        for i in range(0, len(prompts), args.max_batch):
            segs += synth_batch(descs[i:i + args.max_batch], prompts[i:i + args.max_batch])
        pieces = []; total = 0; cap = int(args.max_seconds * sr)
        for s in segs:
            if total + len(s) > cap and pieces:   # stop at a turn boundary (never exceed cap)
                break
            pieces.append(s); pieces.append(pause); total += len(s) + len(pause)
        final = np.concatenate(pieces) if pieces else np.zeros(int(0.1 * sr), dtype="float32")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tw:
            sf.write(tw.name, final, sr)
            AudioSegment.from_wav(tw.name).export(mp3, format="mp3", bitrate="64k")
        dlg["audio"] = os.path.join("audio", args.subdir, did, f"{did}.mp3")
        done += 1
        print(f"[parler] {done}/{len(dialogues)}  {did}  ({len(final)/sr:.1f}s)", flush=True)

    out = args.out or args.dialogues.replace(".json", "_withaudio.json")
    json.dump(dialogues, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"[parler] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
