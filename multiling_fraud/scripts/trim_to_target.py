#!/usr/bin/env python
"""Trim an over-generated dialogue set down to the EXACT per-fraud-type targets
(distribute(total)), renumbering ids sequentially per type. Flags any type that
came up short after the purity filter so we can top it up.

  python scripts/trim_to_target.py --in out/hi/dialogues_raw.json \
      --out out/hi/dialogues_full.json --lang hi --total 7177
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.taxonomy import distribute


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--total", type=int, default=7177)
    args = ap.parse_args()

    target = distribute(args.total)
    data = json.load(open(args.inp))
    by_type = {}
    for d in data:
        by_type.setdefault(d["fraud_type_key"], []).append(d)

    out, short = [], {}
    for key, tgt in target.items():
        got = by_type.get(key, [])
        if len(got) < tgt:
            short[key] = tgt - len(got)
        for i, d in enumerate(got[:tgt], 1):
            d["id"] = f"{args.lang}_{key}_{i:05d}"
            out.append(d)

    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[trim] kept {len(out)}/{args.total} -> {args.out}", flush=True)
    if short:
        print(f"[trim] SHORTFALL (need top-up): {short}", flush=True)
    else:
        print("[trim] all fraud types hit target — complete.", flush=True)


if __name__ == "__main__":
    main()
