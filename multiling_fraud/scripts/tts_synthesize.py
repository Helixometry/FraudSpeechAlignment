#!/usr/bin/env python
"""
Stage 2: synthesize each dialogue into a 2-speaker call recording (.mp3),
mirroring the TeleAntiFraud audio layout: <audio_root>/NEG-gen-en/<id>/<id>.mp3

Two distinct XTTS-v2 built-in voices are used for caller vs callee, turns are
synthesized separately and concatenated with a short pause between them.

Run (on the A100 node, inside the `fraudtts` env):
    python scripts/tts_synthesize.py --dialogues out/dialogues_pilot.json \
        --audio-root /users/msingh/sharedscratch/TeleAntiFraud_en
"""
import argparse, json, os, sys, tempfile

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", required=True)
    ap.add_argument("--audio-root", default="/users/msingh/sharedscratch/TeleAntiFraud_en")
    ap.add_argument("--subdir", default="NEG-gen-en")  # NEG = fraud (mirrors original convention)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--voice-caller", default="Damien Black")
    ap.add_argument("--voice-callee", default="Ana Florence")
    ap.add_argument("--pause-ms", type=int, default=350)
    ap.add_argument("--out", default=None, help="dialogues file with audio paths added")
    ap.add_argument("--shard", type=int, default=0, help="this worker's index (0..nshards-1)")
    ap.add_argument("--nshards", type=int, default=1, help="total parallel workers")
    args = ap.parse_args()

    from TTS.api import TTS
    from pydub import AudioSegment
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[tts] loading {MODEL} on {dev}", flush=True)
    tts = TTS(MODEL).to(dev)

    dialogues = json.load(open(args.dialogues))
    if args.nshards > 1:
        dialogues = [d for i, d in enumerate(dialogues) if i % args.nshards == args.shard]
        print(f"[tts] shard {args.shard}/{args.nshards}: {len(dialogues)} dialogues", flush=True)
    pause = AudioSegment.silent(duration=args.pause_ms)
    audio_dir_root = os.path.join(args.audio_root, "audio", args.subdir)

    def synth_turn(text, speaker, path_wav):
        tts.tts_to_file(text=text, speaker=speaker, language=args.lang, file_path=path_wav)
        return AudioSegment.from_wav(path_wav)

    done = 0
    for d in dialogues:
        did = d["id"]
        clip_dir = os.path.join(audio_dir_root, did)
        os.makedirs(clip_dir, exist_ok=True)
        final_mp3 = os.path.join(clip_dir, f"{did}.mp3")
        call = AudioSegment.silent(duration=0)
        with tempfile.TemporaryDirectory() as tmp:
            for i, turn in enumerate(d["turns"]):
                spk = args.voice_caller if turn.get("speaker") == "caller" else args.voice_callee
                text = (turn.get("text") or "").strip()
                if not text:
                    continue
                seg = synth_turn(text, spk, os.path.join(tmp, f"t{i}.wav"))
                call += seg + pause
        call.export(final_mp3, format="mp3", bitrate="64k")
        # relative path exactly like original metadata ("audio/...")
        d["audio"] = os.path.join("audio", args.subdir, did, f"{did}.mp3")
        done += 1
        print(f"[tts] {done}/{len(dialogues)}  {did}  ({len(call)/1000:.1f}s)", flush=True)

    out = args.out or args.dialogues.replace(".json", "_withaudio.json")
    json.dump(dialogues, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"[tts] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
