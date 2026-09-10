#!/usr/bin/env python
"""
Stage 3: assemble generated+synthesized dialogues into the EXACT TeleAntiFraud schema
(fraud class only).

Emits under <out_root>:
  binary_classification/train.json, test.json      (single-turn, answer="fraud")
  sft/train.jsonl, test.jsonl                       (scene / fraud / fraud_type cascade)

Run (login node is fine, pure python):
    python scripts/assemble_dataset.py --dialogues out/dialogues_pilot_withaudio.json \
        --out-root /users/msingh/sharedscratch/TeleAntiFraud_en --test-frac 0.1
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.languages import get_language


def clean(s):
    return (s or "").replace("\n", " ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", required=True, help="dialogues file WITH audio paths")
    ap.add_argument("--lang", default="en", help="language pack code (config/languages/<lang>.py)")
    ap.add_argument("--out-root", default="/users/msingh/sharedscratch/TeleAntiFraud_en")
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    T = get_language(args.lang).templates
    build_binary_record = T.build_binary_record
    build_scene_sft = T.build_scene_sft
    build_fraud_sft = T.build_fraud_sft
    build_fraud_type_sft = T.build_fraud_type_sft

    dialogues = json.load(open(args.dialogues))
    dialogues = [d for d in dialogues if d.get("audio")]  # need audio
    random.Random(args.seed).shuffle(dialogues)
    n_test = max(1, int(len(dialogues) * args.test_frac)) if len(dialogues) > 1 else 0
    test, train = dialogues[:n_test], dialogues[n_test:]

    os.makedirs(os.path.join(args.out_root, "binary_classification"), exist_ok=True)
    os.makedirs(os.path.join(args.out_root, "sft"), exist_ok=True)

    def build(split):
        binary, sft = [], []
        for d in split:
            a = d["audio"]
            binary.append(build_binary_record(a))
            sc, scr, scc = d["scene"], clean(d.get("scene_reason")), d.get("scene_confidence", 0.9)
            fr, frc = clean(d.get("fraud_reason")), d.get("fraud_confidence", 0.9)
            th = clean(d.get("think"))
            ftr, ftc = clean(d.get("fraud_type_reason")), d.get("fraud_type_confidence", 0.9)
            ft = d["fraud_type"]
            _ = th  # generated reasoning; embed via templates if desired later
            sft.append(build_scene_sft(a, sc))
            sft.append(build_fraud_sft(a, sc, scr, scc))
            sft.append(build_fraud_type_sft(a, sc, scr, scc, fr, frc, ft))
        return binary, sft

    for name, split in [("train", train), ("test", test)]:
        binary, sft = build(split)
        with open(os.path.join(args.out_root, "binary_classification", f"{name}.json"), "w") as f:
            json.dump(binary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.out_root, "sft", f"{name}.jsonl"), "w") as f:
            for r in sft:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[asm] {name}: {len(binary)} binary, {len(sft)} sft ({len(split)} dialogues)")

    # manifest
    manifest = {
        "language": args.lang, "class": "fraud-only", "dialogues": len(dialogues),
        "train_dialogues": len(train), "test_dialogues": len(test),
    }
    json.dump(manifest, open(os.path.join(args.out_root, "dataset_manifest.json"), "w"), indent=2)
    print(f"[asm] out-root: {args.out_root}")


if __name__ == "__main__":
    main()
