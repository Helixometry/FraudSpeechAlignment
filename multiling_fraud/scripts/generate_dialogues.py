#!/usr/bin/env python
"""
Stage 1: generate English fraud dialogues with a top open LLM (vLLM, offline batch).

Each dialogue is a realistic 2-speaker phone call for a given fraud type, plus all
the annotation fields needed for the cascading SFT tasks (scene / fraud / fraud_type
reasons + confidences). Labels are known by construction (we asked for that type).

Run (on the A100 node, inside the `fraudgen` env):
    python scripts/generate_dialogues.py --n 50 --out out/dialogues_pilot.json
    # tiny TTS check set:
    python scripts/generate_dialogues.py --n 3  --out out/dialogues_tiny.json
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.taxonomy import distribute
from config.languages import get_language

# One-line scenario per fraud type so the CONTENT is correct (esp. kidnapping, which the
# model otherwise turned into a generic customer-service scam).
SCENARIO_HINTS = {
    "customer_service": "poses as a company's customer-support agent (fake refund, account problem, remote-access app, or verification) and manipulates the victim into paying or sharing credentials",
    "bank": "poses as the victim's bank fraud/security team about suspicious activity and gets the victim to reveal a one-time code / card details or move money to a 'safe account'",
    "investment": "offers a fake high-return investment or scheme and pressures the victim to transfer money quickly",
    "phishing": "claims the victim's account is compromised and pushes them to a fake link / verification to capture their login and one-time code",
    "lottery": "claims the victim has won a lottery/prize and must pay a fee or tax up front to release the winnings",
    "kidnapping": "FALSELY claims to have kidnapped or detained the victim's relative and demands an urgent ransom, forbidding any contact with police (a 'virtual kidnapping' extortion) — this is NOT a customer-service call",
    "identity_theft": "poses as an official (tax office / government / utility) and, under threat of a penalty, pressures the victim to hand over personal identifiers (date of birth, ID/Aadhaar number, address)",
}

PROMPT = """You are helping build a *fraud-detection research dataset* used to TRAIN models that PROTECT people from telecom scams. Write ONE realistic {language_name} phone-call transcript that is an example of **{fraud_label}**.

Scenario: the caller {scenario}.

Rules:
- Two speakers only: "caller" (the fraudster) and "callee" (the target). Natural spoken {language_name}, {min_turns}-{max_turns} turns, alternating, starting with the caller.
- Write every word fully in the native script of {language_name}. Output ONLY in {language_name} — do NOT include any other language, transliteration, or meta-commentary. (A few universally-borrowed terms like OTP/PIN/SMS/app are fine.)
- If the scam references a link/website, use a short plausible but clearly fictitious domain written normally (e.g. secure-portal.co); NEVER write the literal words "fake", "fake-verification", or placeholder tokens.
- Give each speaker a definite gender and keep names + (in gendered languages) grammar consistent with it. The caller and callee may each be male or female — vary this naturally across calls.
- IMPORTANT — the callee is an ordinary, TRUSTING person who does NOT see through the scam. They are polite and a little anxious or excited, ask only normal everyday questions, and are gradually persuaded so the scam realistically PROGRESSES and they begin to comply (this is an at-risk/negative example). Do not make the callee a savvy sceptic who instantly refuses.
- The caller uses common, well-known social-engineering pressure: authority, fear, urgency, reassurance, flattery, isolation. Keep to widely-known tactics; do NOT invent novel techniques or a reusable step-by-step method that would materially help someone defeat real security controls.
- Use ONLY fake names, companies and numbers. No real institutions, no real personal data, no real working links/apps/phone numbers. The transcript is illustrative for DETECTION, not an operational how-to; you may show the victim starting to comply but do not narrate completed financial theft in operational detail.

Return STRICT JSON only (no markdown fence), with EXACTLY these keys:
{{
  "turns": [{{"speaker": "caller", "text": "..."}}, {{"speaker": "callee", "text": "..."}}, ...],
  "caller_gender": "male" or "female",
  "callee_gender": "male" or "female",
  "scene": one of {scenes},
  "scene_reason": "why the call superficially looks like this scene",
  "scene_confidence": float 0-1,
  "fraud_reason": "why this call IS fraud (red flags)",
  "fraud_confidence": float 0-1,
  "think": "short analysis of the tactics used",
  "fraud_type_reason": "why the fraud type is {fraud_label}",
  "fraud_type_confidence": float 0-1
}}
"""


def build_prompts(counts, pack, min_turns, max_turns):
    items = []
    for key, n in counts.items():
        label = pack.FRAUD_LABELS[key]
        for _ in range(n):
            p = PROMPT.format(
                language_name=pack.LANGUAGE_NAME,
                fraud_label=label,
                scenario=SCENARIO_HINTS[key],
                scenes=json.dumps(pack.SCENES, ensure_ascii=False),
                min_turns=min_turns, max_turns=max_turns,
            )
            items.append({"key": key, "fraud_label": label, "prompt": p})
    return items


# --- language-purity / contamination filter -------------------------------------
NON_LATIN_LANGS = {"hi", "ko", "zh", "ta", "bn", "te", "kn", "ml", "mr", "gu", "pa", "ur", "ja", "ru", "ar"}
_GERMAN = re.compile(r'\b(Bitte|korrigieren|Deutsch|abgebrochen|wurde|diesen|Satz)\b')
_LATIN_WORD = re.compile(r'[A-Za-z][A-Za-z\-]+')
_LATIN_OK = {"otp", "pin", "sms", "app", "kb", "nh", "id", "atm", "upi", "cvv", "http", "https",
             "com", "co", "url", "link", "netbanking", "sbi", "hdfc", "icici"}


def is_clean(obj, lang, max_latin=8):
    """Drop dialogues with cross-language contamination / literal placeholders."""
    text = " ".join((t.get("text") or "") for t in obj.get("turns", []))
    low = text.lower()
    if "fake-verification" in low or "fake-link" in low or "fake link" in low:
        return False
    if _GERMAN.search(text):
        return False
    if lang in NON_LATIN_LANGS:
        stray = [w for w in _LATIN_WORD.findall(text) if w.lower() not in _LATIN_OK]
        if len(stray) > max_latin:
            return False
    return True


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).rsplit("```", 1)[0]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    raw = m.group(0) if m else text
    try:
        return json.loads(raw)
    except Exception:
        # LLMs (esp. on exclamation-heavy scams like lottery) emit unescaped quotes /
        # missing commas -> repair the JSON rather than dropping the sample.
        try:
            from json_repair import repair_json
            return json.loads(repair_json(raw))
        except Exception:
            return None


def gen_vllm(model, prompts, max_model_len, seed, max_tokens=1500):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, max_model_len=max_model_len,
              gpu_memory_utilization=0.92, seed=seed,
              quantization="awq" if "AWQ" in model else None)
    sp = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=max_tokens, seed=seed)
    return [o.outputs[0].text for o in llm.generate(prompts, sp)]


def gen_hf(model, tok, prompts, batch_size=8, max_tokens=1500):
    """transformers backend — robust on MIG slices where vLLM mis-profiles memory."""
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", device_map="cuda")
    m.eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=max_tokens, do_sample=True,
                             temperature=0.9, top_p=0.95, pad_token_id=tok.pad_token_id)
        for j in range(out.shape[0]):
            gen = out[j, enc["input_ids"].shape[1]:]
            texts.append(tok.decode(gen, skip_special_tokens=True))
        print(f"[gen] hf batch {i // batch_size + 1} done ({min(i + batch_size, len(prompts))}/{len(prompts)})", flush=True)
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--lang", default="en", help="language pack code (config/languages/<lang>.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("FRAUD_LLM", "Qwen/Qwen2.5-72B-Instruct-AWQ"))
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm",
                    help="vllm (fast, full-GPU) or hf (robust on MIG slices)")
    ap.add_argument("--min-turns", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=1500, help="raise if outputs truncate (JSON parse fails)")
    ap.add_argument("--only-type", default=None, help="generate ONLY this fraud-type key (top-up); --n = count")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pack = get_language(args.lang)
    counts = {args.only_type: args.n} if args.only_type else distribute(args.n)
    print(f"[gen] lang={args.lang} backend={args.backend} model={args.model} distribution: {counts}", flush=True)
    items = build_prompts(counts, pack, args.min_turns, args.max_turns)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [tok.apply_chat_template([{"role": "user", "content": it["prompt"]}],
                                       tokenize=False, add_generation_prompt=True) for it in items]
    gen_texts = (gen_vllm(args.model, prompts, args.max_model_len, args.seed, args.max_new_tokens)
                 if args.backend == "vllm" else gen_hf(args.model, tok, prompts, max_tokens=args.max_new_tokens))

    dialogues, per_type = [], {}
    for it, text in zip(items, gen_texts):
        obj = None
        try:
            obj = extract_json(text)
        except Exception as e:
            print(f"[gen] parse fail ({it['key']}): {e}", flush=True)
        if not obj or "turns" not in obj:
            continue
        if not is_clean(obj, args.lang):
            print(f"[gen] contaminated, dropped ({it['key']})", flush=True)
            continue
        key = it["key"]
        per_type[key] = per_type.get(key, 0) + 1
        did = f"{args.lang}_{key}_{per_type[key]:05d}"
        obj.setdefault("caller_gender", "male")
        obj.setdefault("callee_gender", "female")
        obj["caller_gender"] = "female" if str(obj.get("caller_gender")).lower().startswith("f") else "male"
        obj["callee_gender"] = "female" if str(obj.get("callee_gender")).lower().startswith("f") else "male"
        obj.update({
            "id": did, "language": args.lang,
            "fraud_type": it["fraud_label"], "fraud_type_key": key, "is_fraud": True,
        })
        dialogues.append(obj)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dialogues, f, ensure_ascii=False, indent=2)
    print(f"[gen] wrote {len(dialogues)}/{len(items)} dialogues -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
