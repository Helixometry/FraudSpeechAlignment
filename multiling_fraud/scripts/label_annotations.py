#!/usr/bin/env python
"""
Enrich every dialogue with research annotations, in a SINGLE pass of the same
model that wrote them (Qwen2.5-72B-Instruct-AWQ via vLLM, greedy/deterministic).

We capture everything the fraud-safety / ALM-alignment study needs at once, so
the expensive 72B never has to be re-run:

  1. manipulation_tactics  (7 binary flags)  -- HOW the victim is influenced
       authority urgency fear affinity reward isolation credential_request
  2. requested_action      (1 categorical)   -- the concrete UNSAFE ASK
  3. compliance_level      (none/partial/full)-- the OUTCOME (victim behavior)

Together with the existing `fraud_type_key` (WHAT the attack is) this gives the
two taxonomies plus the target variable, enabling:

      P(unsafe behavior | manipulation strategy).

Run (H100/A100, inside the `fraudgen` env):
    python scripts/label_annotations.py --lang en
    #   -> out/en/annotations.json   {id: {...}, ...}   (checkpointed, resume-safe)

Shardable (--shard/--nshards) to split one language across GPUs.
"""
import argparse, json, os, re

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("FRAUD_LLM", "Qwen/Qwen2.5-72B-Instruct-AWQ")

# ---- 1) manipulation-mechanism taxonomy (binary, multi-label) ---------------
TACTICS = {
    "authority":
        "The caller impersonates or invokes a trusted/official power to compel "
        "compliance — e.g. a bank, the police, a government/tax office, a court, a "
        "well-known company, tech support, or a superior/boss.",
    "urgency":
        "Time pressure: the caller insists the victim must act now / immediately, "
        "sets a deadline, or says the window/opportunity is closing.",
    "fear":
        "The caller threatens or alarms the victim with a bad consequence — account "
        "frozen/hacked, arrest or legal action, fines, money being lost/stolen, or "
        "harm to the victim or their family.",
    "affinity":
        "The caller builds rapport, trust or liking — friendliness, politeness, "
        "flattery, sympathy, or reassurance that they are helping/protecting the victim.",
    "reward":
        "The caller dangles a gain — a prize or lottery win, a refund, cashback, a gift, "
        "a discount, or high/guaranteed investment returns.",
    "isolation":
        "The caller pushes secrecy or separation — tells the victim not to tell anyone "
        "(bank, family, police), to keep it confidential, or to stay on the line / not hang up.",
    "credential_request":
        "The caller explicitly asks the victim to reveal or hand over sensitive data or "
        "access, or to move money — e.g. an OTP/verification code, password/PIN, full card "
        "number/CVV, bank-login details, installing a remote-access app, or making a "
        "transfer/payment to a given account.",
}
TACTIC_KEYS = list(TACTICS.keys())

# ---- 2) the concrete unsafe ask (single best-fit category) ------------------
ACTIONS = {
    "otp_or_verification_code": "asks the victim to read out an OTP / one-time / verification code.",
    "password_or_pin":          "asks for a password, PIN, or security answers.",
    "card_or_bank_details":     "asks for full card number / CVV / expiry or bank-account/login details.",
    "personal_identity_info":   "asks for identity data — ID/SSN/Aadhaar number, DOB, address.",
    "install_app_or_remote_access": "pushes the victim to install an app or grant remote access to a device.",
    "transfer_or_pay_money":    "pushes the victim to transfer, send, or pay money (incl. to a 'safe account').",
    "buy_gift_cards_or_vouchers": "tells the victim to buy gift cards / vouchers and share the codes.",
    "click_link_or_visit_site": "tells the victim to click a link or visit a (phishing) website/portal.",
    "other":                    "a different unsafe action not covered above.",
    "none":                     "no concrete unsafe action is actually requested in the call.",
}
ACTION_KEYS = list(ACTIONS.keys())

# ---- 3) outcome: how far the victim complied --------------------------------
COMPLIANCE = {
    "full":    "the victim carries out or clearly agrees to the main unsafe action.",
    "partial": "the victim gives some information or wavers / partially cooperates but does not fully complete the main ask.",
    "none":    "the victim refuses, stalls, or the call ends without any compliance.",
}
COMPLIANCE_KEYS = list(COMPLIANCE.keys())

SYS = (
    "You analyze fraudulent phone-call transcripts for a safety-research dataset. "
    "Judge ONLY from what is actually said in the transcript. Return a single compact "
    "JSON object, no prose, with exactly these keys:\n"
    "- \"manipulation_tactics\": object with keys "
    f"{TACTIC_KEYS}, each 0 or 1 (1 = the CALLER clearly uses it; several are usually 1).\n"
    f"- \"requested_action\": one string from {ACTION_KEYS} (the single main unsafe action "
    "the caller pushes the victim toward).\n"
    f"- \"compliance_level\": one string from {COMPLIANCE_KEYS} (how far the CALLEE/victim complied)."
)


def _defs(d):
    return "\n".join(f'- "{k}": {v}' for k, v in d.items())


def transcript_text(d, max_chars=6000):
    lines = []
    for t in d.get("turns", []):
        spk = "CALLER" if t.get("speaker") == "caller" else "CALLEE"
        txt = (t.get("text") or "").strip()
        if txt:
            lines.append(f"{spk}: {txt}")
    return "\n".join(lines)[:max_chars]


def build_prompt(tok, d):
    user = (
        "Manipulation tactics:\n" + _defs(TACTICS) + "\n\n"
        "requested_action options:\n" + _defs(ACTIONS) + "\n\n"
        "compliance_level options:\n" + _defs(COMPLIANCE) + "\n\n"
        "Transcript:\n" + transcript_text(d) + "\n\n"
        "Return the JSON object now."
    )
    return tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )


def load_dialogues(path):
    """Accept either a JSON array (out/<lang>/dialogues_full.json) or JSONL
    (the Hugging Face data/<lang>/train.jsonl) so this runs off the HF download
    directly on a fresh machine — no need to move the source file."""
    with open(path, encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).rsplit("```", 1)[0]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    raw = m.group(0) if m else text
    try:
        return json.loads(raw)
    except Exception:
        try:
            from json_repair import repair_json
            return json.loads(repair_json(raw))
        except Exception:
            return None


def _bin(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return 1 if v >= 1 else 0
    if isinstance(v, str):
        return 1 if v.strip().lower() in ("1", "true", "yes") else 0
    return 0


def coerce(obj):
    """Normalize to a clean annotation dict; missing/garbage -> safe defaults."""
    obj = obj or {}
    mt = obj.get("manipulation_tactics", {}) or {}
    tactics = {k: _bin(mt.get(k, 0)) for k in TACTIC_KEYS}
    act = str(obj.get("requested_action", "none")).strip().lower()
    if act not in ACTION_KEYS:
        act = "other" if act not in ("", "none") else "none"
    comp = str(obj.get("compliance_level", "none")).strip().lower()
    if comp not in COMPLIANCE_KEYS:
        comp = "none"
    return {"manipulation_tactics": tactics, "requested_action": act, "compliance_level": comp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en")
    ap.add_argument("--dialogues", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=512, help="checkpoint every N dialogues")
    ap.add_argument("--limit", type=int, default=0, help="label only the first N (smoke test)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    dial_path = args.dialogues or os.path.join(PROJ, "out", args.lang, "dialogues_full.json")
    out_path = args.out or os.path.join(PROJ, "out", args.lang, "annotations.json")
    if args.nshards > 1:
        out_path = out_path.replace(".json", f"_{args.shard}.json")

    dialogues = load_dialogues(dial_path)
    if args.nshards > 1:
        dialogues = [d for i, d in enumerate(dialogues) if i % args.nshards == args.shard]
    if args.limit:
        dialogues = dialogues[:args.limit]

    done = {}
    if os.path.exists(out_path):
        try:
            done = json.load(open(out_path))
        except Exception:
            done = {}
    todo = [d for d in dialogues if d["id"] not in done]
    print(f"[annot] lang={args.lang} shard={args.shard}/{args.nshards} "
          f"total={len(dialogues)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        print("[annot] nothing to do", flush=True)
        return

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.92,
              quantization="awq" if "AWQ" in args.model else None)
    sp = SamplingParams(temperature=0.0, max_tokens=200)   # greedy -> deterministic

    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        outs = llm.generate([build_prompt(tok, d) for d in chunk], sp)
        for d, o in zip(chunk, outs):
            done[d["id"]] = coerce(extract_json(o.outputs[0].text))
        tmp = out_path + ".tmp"
        json.dump(done, open(tmp, "w"), ensure_ascii=False, indent=0)
        os.replace(tmp, out_path)
        print(f"[annot] {min(i + args.batch, len(todo))}/{len(todo)} "
              f"(saved {len(done)}) -> {os.path.relpath(out_path, PROJ)}", flush=True)

    summarize(done)
    print(f"[annot] DONE {len(done)} dialogues -> {out_path}", flush=True)


def summarize(done):
    import collections
    n = len(done) or 1
    tac = collections.Counter()
    act = collections.Counter()
    comp = collections.Counter()
    for v in done.values():
        for k in TACTIC_KEYS:
            tac[k] += v.get("manipulation_tactics", {}).get(k, 0)
        act[v.get("requested_action", "none")] += 1
        comp[v.get("compliance_level", "none")] += 1
    print("[annot] tactic prevalence:", {k: f"{100*tac[k]/n:.0f}%" for k in TACTIC_KEYS}, flush=True)
    print("[annot] requested_action:", dict(act), flush=True)
    print("[annot] compliance_level:", dict(comp), flush=True)


if __name__ == "__main__":
    main()
