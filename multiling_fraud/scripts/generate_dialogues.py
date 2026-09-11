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

# Pools of sub-scenarios per fraud type -> within-type variety (pick one at random per call).
SUB_SCENARIOS = {
    "customer_service": [
        "poses as support processing a refund for an accidental double charge and needs remote access / a verification payment",
        "poses as support about a subscription auto-renewal the victim must cancel by confirming card details",
        "poses as a delivery/e-commerce agent about a failed delivery needing a small redelivery fee and card details",
        "poses as tech support claiming a virus was detected and the victim must install a remote-access app",
        "poses as support warning the account will be suspended today unless the victim verifies banking details",
        "poses as support offering a reward/cashback that requires a 'verification' transfer or app install",
        "poses as a utility/telecom agent about a billing error and a refund needing bank details",
    ],
    "bank": [
        "poses as the bank fraud team about a suspicious transaction and gets the OTP / card details",
        "claims the card is blocked and must be reactivated by confirming the full number and PIN",
        "claims urgent KYC re-verification is due or the account will be frozen",
        "offers a pre-approved loan / credit-limit increase needing account details and a processing fee",
        "claims a wrong transfer landed in the account and asks the victim to 'return' it",
        "warns of a new-device login and pushes the victim to move funds to a 'safe account'",
        "arranges a bank courier to collect the old card and PIN for 'secure destruction'",
    ],
    "investment": [
        "pitches a guaranteed high-return crypto scheme with a midnight deadline",
        "gives a 'hot' stock/IPO tip requiring an immediate deposit",
        "promotes a forex/trading platform and walks the victim through funding an account",
        "offers to recover the victim's past investment losses for an upfront fee",
        "pitches a mutual-fund/pension 'upgrade' with a limited-time bonus",
        "promotes a gold/real-estate scheme needing a booking amount now",
    ],
    "phishing": [
        "claims the email account is compromised and pushes a link to re-verify the login and OTP",
        "claims a social-media account is locked and needs re-login via a link",
        "sends a parcel-redelivery link that harvests card details",
        "claims a tax refund is waiting behind a verification link",
        "claims wallet/UPI KYC must be redone via a link and OTP",
        "claims a streaming subscription payment failed and links to a fake payment page",
    ],
    "lottery": [
        "claims a big national-lottery win that needs a tax/processing fee up front",
        "claims a lucky-draw prize from a shopping app needing a release fee",
        "claims a mobile-number contest win requiring bank details to deposit the prize",
        "claims a foreign lottery win needing customs/clearance charges",
        "claims a car/gadget prize that requires paying GST/tax first",
    ],
    "kidnapping": [
        "falsely claims to have kidnapped the victim's child and demands an urgent ransom, no police",
        "claims to hold the victim's sibling over a fabricated debt and demands payment",
        "poses as police saying a relative was arrested and needs immediate bail money",
        "runs a virtual kidnapping with staged screaming, demanding money transferred now",
        "claims the victim's spouse was in an accident and is 'held' until costs are paid",
    ],
    "identity_theft": [
        "poses as the tax office threatening a penalty unless the victim confirms DOB and ID number",
        "poses as a government-subsidy office needing the national ID (Aadhaar/SSN) to release funds",
        "poses as a utility threatening disconnection unless personal identifiers are confirmed",
        "poses as an employer/recruiter collecting ID documents for a fake job offer",
        "poses as a census/KYC survey collecting date of birth, ID number and address",
    ],
}

PROMPT = """You are helping build a *fraud-detection research dataset* used to TRAIN models that PROTECT people from telecom scams. Write ONE realistic {language_name} phone-call transcript that is an example of **{fraud_label}**.

Scenario: the caller {scenario}.

Profile for THIS specific call (make it clearly DIFFERENT from any other call): the caller is {caller_gender}; the victim (callee) is {callee_gender}, named {callee_name} ({context}). The unique detail to anchor this call: {seed_hint}. Invent fresh names, amounts, and story specifics — do NOT reuse a template.

Rules:
- Two speakers only: "caller" (the fraudster) and "callee" (the target). Natural spoken {language_name}, {min_turns}-{max_turns} turns, alternating, starting with the caller.
- Keep it BRIEF: each turn is ONE short spoken sentence (occasionally two). The whole call must be short enough to speak aloud in UNDER 90 seconds — get to the scam quickly, no long monologues or filler.
- Write every word fully in the native script of {language_name}. Output ONLY in {language_name} — do NOT include any other language, transliteration, or meta-commentary. (A few universally-borrowed terms like OTP/PIN/SMS/app are fine.)
- If the scam references a link/website, use a short plausible but clearly fictitious domain written normally (e.g. secure-portal.co); NEVER write the literal words "fake", "fake-verification", or placeholder tokens.
- Use pronouns and (in gendered languages) verb/adjective forms consistent with EACH speaker's gender as given in the profile above. The victim's name must match their stated gender.
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


import random

# generic persona/context + unique-anchor pools (English instructions; the LLM localizes)
CONTEXTS = [
    "a 28-year-old software worker in a big city", "a 55-year-old shopkeeper in a small town",
    "a 34-year-old schoolteacher", "a retired 68-year-old pensioner", "a 41-year-old homemaker",
    "a 23-year-old college student", "a 47-year-old bank clerk", "a 60-year-old farmer",
    "a 30-year-old nurse", "a 52-year-old small-business owner", "a 38-year-old delivery driver",
    "a 45-year-old office manager", "a 26-year-old call-centre agent", "a 63-year-old widow(er)",
]
SEED_HINTS = [
    "the victim just received their salary", "it is late evening and the victim is tired",
    "the victim recently shopped online", "a festival is coming up and money is tight",
    "the victim is at work and distracted", "the victim's phone battery is low",
    "the victim was expecting a delivery", "the victim recently changed their bank",
    "the victim is caring for a sick relative", "the victim just missed a call earlier",
    "the victim is new to smartphones", "the victim has a big payment due soon",
]


def build_prompts(counts, pack, min_turns, max_turns, rng=None):
    rng = rng or random.Random()
    names = getattr(pack, "NAMES", {"male": ["Alex"], "female": ["Sam"]})
    items = []
    for key, n in counts.items():
        label = pack.FRAUD_LABELS[key]
        for _ in range(n):
            cg = "male" if rng.random() < 0.60 else "female"      # callers skew male, but vary
            eg = "female" if rng.random() < 0.55 else "male"      # victims mixed both genders
            p = PROMPT.format(
                language_name=pack.LANGUAGE_NAME,
                fraud_label=label,
                scenario=SCENARIO_HINTS[key],
                scenes=json.dumps(pack.SCENES, ensure_ascii=False),
                min_turns=min_turns, max_turns=max_turns,
                caller_gender=cg, callee_gender=eg,
                callee_name=rng.choice(names[eg]), context=rng.choice(CONTEXTS),
                seed_hint=rng.choice(SEED_HINTS),
            )
            items.append({"key": key, "fraud_label": label, "prompt": p,
                          "caller_gender": cg, "callee_gender": eg})
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
    # NO fixed sampling seed -> each sequence samples independently (avoids identical outputs
    # for identical prompts). Diversity also comes from per-call profiles in the prompt.
    sp = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=max_tokens)
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
    ap.add_argument("--min-turns", type=int, default=6)
    ap.add_argument("--max-turns", type=int, default=10)
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
        obj.update({
            "id": did, "language": args.lang,
            # gender is what WE assigned (the LLM wrote grammar to match) — reliable for TTS
            "caller_gender": it["caller_gender"], "callee_gender": it["callee_gender"],
            "fraud_type": it["fraud_label"], "fraud_type_key": key, "is_fraud": True,
        })
        dialogues.append(obj)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dialogues, f, ensure_ascii=False, indent=2)
    print(f"[gen] wrote {len(dialogues)}/{len(items)} dialogues -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
