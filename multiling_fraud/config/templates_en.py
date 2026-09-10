"""
English instruction templates mirroring TeleAntiFraud's 3 cascading tasks, plus
builders that emit records in the EXACT original schema.

Original schema recap:
- binary_classification: {"prompt":[system, user(audio+text)], "answer":"fraud"}
- sft (jsonl): {"messages":[...ends on user...], "prompts":[copy], "answers":<gold>, "audios":[path]}
    * scene task     -> 2 msgs (system,user),                       answers = scene
    * fraud task     -> 4 msgs (system,user,assistant,user),        answers = "fraud"
    * fraud_type task-> 6 msgs (system,user,asst,user,asst,user),   answers = fraud_type
"""
from config.taxonomy import SCENES_EN, FRAUD_TYPE_CLOSED_SET_EN

SYSTEM = "You are a helpful audio understanding assistant."

AUDIO_TOKEN = "<audio>"

_SCENE_LIST = "\n".join(f"- {s}" for s in SCENES_EN)
_SCENE_SET = ", ".join(f'"{s}"' for s in SCENES_EN)
_TYPE_SET = ", ".join(f'"{t}"' for t in FRAUD_TYPE_CLOSED_SET_EN)

# ---- Round 1: scene classification -------------------------------------------
SCENE_USER = f"""{AUDIO_TOKEN}
**Task Description:**

Based on the audio you hear, determine which type of call scene this belongs to:

{_SCENE_LIST}

**Input:**

Call recording

**Output Format:**

Please output your judgment in the following format:

```
{{
  "scene": "<scene_type>",
  "reason": "<reason_for_judgment>",
  "confidence": <confidence_level>
}}
```

- `scene`: string, the identified call-scene type, chosen from the closed set: `[{_SCENE_SET}]`.
- `reason`: string, briefly explain your judgment.
- `confidence`: float in [0,1], your confidence in the judgment (1 = fully confident).
"""

# ---- Round 2: fraud detection ------------------------------------------------
FRAUD_USER = """
**Task Description:**

You are a professional audio model that analyzes audio content and judges whether it involves fraud. Based on the following, output your judgment:

1. The call scene from the first round of analysis.
2. The audio content.

**Output Format:**

Please output your judgment in the following format:

```
{
  "reason": "<reason_for_judgment>",
  "confidence": <confidence_level>,
  "is_fraud": <true/false>
}
```

- `reason`: string, briefly explain your judgment.
- `confidence`: float in [0,1], your confidence in the judgment.
- `is_fraud`: boolean, whether the audio involves fraud. `true` = fraud, `false` = not fraud.
"""

# ---- Round 3: fraud-type identification --------------------------------------
FRAUD_TYPE_USER = f"""
**Task Description:**

You are a professional audio model that analyzes audio content in detail and identifies the type of fraud involved. Based on the following, output your judgment:

1. The call scene from the first round.
2. The fraud judgment from the second round.
3. The audio content.

**Output Format:**

Please output your judgment in the following format:

<think>
...(detailed analysis)
</think>
<answer>
```
{{
  "fraud_type": "<fraud_type>",
  "reason": "<reason_for_judgment>",
  "confidence": <confidence_level>
}}
```
</answer>

- `fraud_type`: string, the identified fraud type, chosen from the closed set: `[{_TYPE_SET}]`.
- `reason`: string, briefly explain your judgment.
- `confidence`: float in [0,1], your confidence in the judgment.
"""

# Binary-classification single-turn user text (audio+text list style).
BINARY_TEXT = """Based on the audio content you hear, analyze whether this call involves fraud.

Please output in the following format:
"<thinking>
...(analysis)
</thinking>
<answer>
{
  "reason": "<reason_for_judgment>",
  "confidence": <confidence_level>,
  "is_fraud": <true/false>
}
</answer>"

- `reason`: string, briefly summarize the reason.
- `confidence`: float in [0,1], confidence in the judgment.
- `is_fraud`: boolean, whether the audio involves fraud. `true` = fraud, `false` = not fraud.
"""

BINARY_SYSTEM = "\nAnswer in the following format:\n<thinking>\n...\n</thinking>\n<answer>\n...\n</answer>\n"


def _json_block(obj_lines):
    return "\n\n```json\n" + obj_lines + "\n```"


def scene_answer(scene, reason, conf):
    return _json_block(
        f'{{\n  "scene": "{scene}",\n  "reason": "{reason}",\n  "confidence": {conf}\n}}'
    )


def fraud_answer(reason, conf, is_fraud=True):
    return _json_block(
        f'{{\n  "reason": "{reason}",\n  "confidence": {conf},\n  "is_fraud": {str(is_fraud).lower()}\n}}'
    )


def fraud_type_answer(think, fraud_type, reason, conf):
    return (
        f"\n<think>\n{think}\n</think>\n<answer>\n```json\n"
        f'{{\n  "fraud_type": "{fraud_type}",\n  "reason": "{reason}",\n  "confidence": {conf}\n}}'
        f"\n```\n</answer>"
    )


# ---- record builders ---------------------------------------------------------
def build_binary_record(audio_path):
    return {
        "prompt": [
            {"role": "system", "content": BINARY_SYSTEM},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": BINARY_TEXT},
            ]},
        ],
        "answer": "fraud",
    }


def build_scene_record(audio_path):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": SCENE_USER},
    ]
    return {"messages": msgs, "prompts": list(msgs), "answers": None, "audios": [audio_path]}
    # answers filled by caller (scene string)


def build_scene_sft(audio_path, scene):
    r = build_scene_record(audio_path)
    r["answers"] = scene
    return r


def build_fraud_sft(audio_path, scene, scene_reason, scene_conf):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": SCENE_USER},
        {"role": "assistant", "content": scene_answer(scene, scene_reason, scene_conf)},
        {"role": "user", "content": FRAUD_USER},
    ]
    return {"messages": msgs, "prompts": list(msgs), "answers": "fraud", "audios": [audio_path]}


def build_fraud_type_sft(audio_path, scene, scene_reason, scene_conf,
                         fraud_reason, fraud_conf, fraud_type):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": SCENE_USER},
        {"role": "assistant", "content": scene_answer(scene, scene_reason, scene_conf)},
        {"role": "user", "content": FRAUD_USER},
        {"role": "assistant", "content": fraud_answer(fraud_reason, fraud_conf, True)},
        {"role": "user", "content": FRAUD_TYPE_USER},
    ]
    return {"messages": msgs, "prompts": list(msgs), "answers": fraud_type, "audios": [audio_path]}
