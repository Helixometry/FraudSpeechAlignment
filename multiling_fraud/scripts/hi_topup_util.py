#!/usr/bin/env python
"""Helper for the Hindi zero-Latin top-up loop. Modes:
  init    <raw.json> <pool.json>              seed pool with strict-clean dialogues from raw
  deficit <pool.json> <counts.json> <total>   write buffered per-type deficit counts (or empty {} if done)
  merge   <pool.json> <round.json>            add round's strict-clean, deduped, to pool
  finalize<pool.json> <out.json> <lang> <total>  trim to per-type targets, renumber ids
A dialogue is strict-clean iff it contains NO Latin letter anywhere (no fused
words, emails, URLs or English) — the rule that guarantees intelligible Hindi TTS.
"""
import json, re, os, sys, math, collections
sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from config.taxonomy import distribute

_LAT = re.compile(r'[A-Za-z]')
_DEVA = re.compile(r'[ऀ-ॿ]')
# CLEAN_MODE=nonlatin (default): strict zero-Latin (Hindi/Korean/…).
# CLEAN_MODE=codemix: genuine code-switch — every dialogue must have BOTH
# Devanagari AND Latin, and no malformed turns (Hinglish).
_CLEAN_MODE = os.environ.get("CLEAN_MODE", "nonlatin")

def _ok_turns(x):
    ts = x.get("turns", [])
    return bool(ts) and all(isinstance(t, dict) for t in ts)

def clean(x):
    if not _ok_turns(x):
        return False
    text = " ".join((t.get("text") or "") for t in x["turns"])
    if _CLEAN_MODE == "codemix":
        return bool(_DEVA.search(text)) and bool(_LAT.search(text))
    if _CLEAN_MODE == "latin":        # Latin-script language (en): structural only
        return bool(text.strip())
    return not _LAT.search(text)      # nonlatin (default): strict zero-Latin

# top-up buffer divisor: assume ~this fraction of requests survive the filter.
# code-mix + hard categories drop much harder than plain non-Latin, so request more.
_SURV = float(os.environ.get("TOPUP_SURVIVAL", "0.75"))
def key(x):
    return tuple(t.get("text") or "" for t in x.get("turns", []))


def main():
    mode = sys.argv[1]
    if mode == "init":
        raw, pool = sys.argv[2], sys.argv[3]
        d = [x for x in json.load(open(raw)) if clean(x)]
        json.dump(d, open(pool, "w"), ensure_ascii=False)
        per = collections.Counter(x["fraud_type_key"] for x in d)
        print(f"[topup] init pool: {len(d)} strict-clean | {dict(per)}", flush=True)
    elif mode == "deficit":
        pool, out, total = sys.argv[2], sys.argv[3], int(sys.argv[4])
        tgt = distribute(total)
        per = collections.Counter(x["fraud_type_key"] for x in json.load(open(pool)))
        # buffer for the ~20% strict-drop rate, + small margin
        need = {k: math.ceil((tgt[k] - per.get(k, 0)) / _SURV) + 6
                for k in tgt if per.get(k, 0) < tgt[k]}
        json.dump(need, open(out, "w"))
        print(f"[topup] remaining deficit (raw): "
              f"{ {k: tgt[k]-per.get(k,0) for k in tgt if per.get(k,0)<tgt[k]} } "
              f"-> request {need}", flush=True)
    elif mode == "merge":
        pool, rnd = sys.argv[2], sys.argv[3]
        cur = json.load(open(pool)); seen = {key(x) for x in cur}
        add = [x for x in json.load(open(rnd)) if clean(x) and key(x) not in seen]
        cur += add
        json.dump(cur, open(pool, "w"), ensure_ascii=False)
        print(f"[topup] merged {len(add)} new strict-clean -> pool now {len(cur)}", flush=True)
    elif mode == "finalize":
        pool, out, lang, total = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
        tgt = distribute(total)
        by = {}
        for x in json.load(open(pool)):
            by.setdefault(x["fraud_type_key"], []).append(x)
        res, short = [], {}
        for k, want in tgt.items():
            got = by.get(k, [])
            if len(got) < want:
                short[k] = want - len(got)
            for i, x in enumerate(got[:want], 1):
                x["id"] = f"{lang}_{k}_{i:05d}"
                res.append(x)
        json.dump(res, open(out, "w"), ensure_ascii=False, indent=2)
        print(f"[topup] finalize: wrote {len(res)}/{total} -> {out}", flush=True)
        print(f"[topup] {'ALL TYPES COMPLETE' if not short else 'STILL SHORT: '+str(short)}", flush=True)


if __name__ == "__main__":
    main()
