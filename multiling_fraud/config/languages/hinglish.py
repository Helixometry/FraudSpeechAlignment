"""Hinglish (Hindi-English code-mixed) language pack.

Transcripts are natural code-switch: Hindi in Devanagari as the matrix language with
English words/phrases in Latin script mixed in, exactly as urban Indian scam calls sound.
Category labels/scenes are kept in English (how Hinglish speakers name them); victim
names are Devanagari Indian names (reused from the Hindi pack).

NOTE: `templates` points at the English task templates as a placeholder so DIALOGUE
GENERATION (which does not use templates) can run now. Add config/templates_hinglish.py
before ASSEMBLING the Hinglish dataset.
"""
from config import templates_en as templates   # TODO: dedicated templates for assembly
from config.taxonomy import (
    SCENES_EN, FRAUD_TYPE_CLOSED_SET_EN, FRAUD_TYPES, DEFAULT_SCENE_FOR_TYPE,
)
from config.languages.hi import NAMES           # Devanagari Indian name pools

CODE = "hinglish"
LANGUAGE_NAME = "Hinglish (Hindi-English code-mixed)"
SCENES = SCENES_EN
FRAUD_TYPE_CLOSED_SET = FRAUD_TYPE_CLOSED_SET_EN
FRAUD_LABELS = {t["key"]: t["en"] for t in FRAUD_TYPES}
SCENE_DEFAULT = DEFAULT_SCENE_FOR_TYPE

# Indic-Parler named speakers (Devanagari base carries the audio); gender pools for
# distinct caller/callee, matching the Hindi TTS path.
VOICES = {"caller": "Rohit", "callee": "Divya"}
VOICE_POOL = {"male": ["Rohit", "Aman"], "female": ["Divya", "Rani"]}
