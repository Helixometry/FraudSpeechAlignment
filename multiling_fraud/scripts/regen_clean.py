#!/usr/bin/env python
"""
Repair a language's dialogues_full.json: drop contaminated dialogues (cross-language
text, literal 'fake' placeholders, etc.) and regenerate clean replacements with the 72B
until each fraud type is back to its target count. One vLLM session.

Run (in fraudgen env, on a full-GPU node):
  python scripts/regen_clean.py --lang ko
"""
import argparse, json, os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_dialogues as G
from config.taxonomy import distribute
from config.languages import get_language


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--dialogues", default=None)
    ap.add_argument("--total", type=int, default=7177)
    ap.add_argument("--model", default=os.environ.get("FRAUD_LLM", "Qwen/Qwen2.5-72B-Instruct-AWQ"))
    ap.add_argument("--buffer", type=float, default=2.0)
    ap.add_argument("--max-new-tokens", type=int, default=1500)
    ap.add_argument("--min-turns", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=16)
    args = ap.parse_args()
    pack = get_language(args.lang)
    path = args.dialogues or f"out/{args.lang}/dialogues_full.json"

    d = json.load(open(path))
    target = distribute(args.total)                 # per-type target counts
    clean_by = {k: [] for k in target}
    dropped = 0
    for x in d:
        k = x.get("fraud_type_key")
        if k in clean_by and G.is_clean(x, args.lang):
            clean_by[k].append(x)
        else:
            dropped += 1
    deficit = {k: max(0, target[k] - len(clean_by[k])) for k in target}
    print(f"[regen] dropped {dropped} contaminated; deficits: { {k:v for k,v in deficit.items() if v} }", flush=True)
    if sum(deficit.values()) == 0:
        print("[regen] nothing to do — already clean & full.")
        return

    # build buffered prompts for deficit types
    counts = {k: math.ceil(v * args.buffer) for k, v in deficit.items() if v > 0}
    items = G.build_prompts(counts, pack, args.min_turns, args.max_turns)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [tok.apply_chat_template([{"role": "user", "content": it["prompt"]}],
                                       tokenize=False, add_generation_prompt=True) for it in items]
    texts = G.gen_vllm(args.model, prompts, 4096, 0, args.max_new_tokens)

    # collect clean candidates per type
    cand = {k: [] for k in counts}
    for it, txt in zip(items, texts):
        try:
            obj = G.extract_json(txt)
        except Exception:
            obj = None
        if not obj or "turns" not in obj or not G.is_clean(obj, args.lang):
            continue
        obj.setdefault("caller_gender", "male"); obj.setdefault("callee_gender", "female")
        obj["caller_gender"] = "female" if str(obj["caller_gender"]).lower().startswith("f") else "male"
        obj["callee_gender"] = "female" if str(obj["callee_gender"]).lower().startswith("f") else "male"
        obj.update({"language": args.lang, "fraud_type": it["fraud_label"],
                    "fraud_type_key": it["key"], "is_fraud": True})
        cand[it["key"]].append(obj)

    # rebuild: clean originals (capped at target) + regenerated, renumber ids
    final = []
    for k in target:
        keep = clean_by[k][:target[k]]
        need = target[k] - len(keep)
        keep += cand.get(k, [])[:need]
        got = len(keep)
        if got < target[k]:
            print(f"[regen] WARN {k}: {got}/{target[k]} (short by {target[k]-got}; rerun to top up)", flush=True)
        for i, x in enumerate(keep, 1):
            x["id"] = f"{args.lang}_{k}_{i:05d}"
            final.append(x)

    os.rename(path, path + ".contaminated.bak")
    json.dump(final, open(path, "w"), ensure_ascii=False, indent=2)
    print(f"[regen] wrote {len(final)} clean dialogues -> {path} (old kept as .contaminated.bak)", flush=True)


if __name__ == "__main__":
    main()
