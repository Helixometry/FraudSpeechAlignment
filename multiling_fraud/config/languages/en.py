"""English language pack."""
from config import templates_en as templates          # task templates + record builders
from config.taxonomy import (
    SCENES_EN, FRAUD_TYPE_CLOSED_SET_EN, FRAUD_TYPES, DEFAULT_SCENE_FOR_TYPE,
)

CODE = "en"
LANGUAGE_NAME = "English"           # used in the generation prompt ("write in {LANGUAGE_NAME}")
SCENES = SCENES_EN                  # scene closed-set (labels)
FRAUD_TYPE_CLOSED_SET = FRAUD_TYPE_CLOSED_SET_EN
FRAUD_LABELS = {t["key"]: t["en"] for t in FRAUD_TYPES}   # fraud-type key -> display label
SCENE_DEFAULT = DEFAULT_SCENE_FOR_TYPE                     # fraud-type key -> imitated scene

# XTTS-v2 built-in voices for the two roles
VOICES = {"caller": "Damien Black", "callee": "Ana Florence"}

# per-gender victim/caller name pools (for diverse, gender-consistent personas)
NAMES = {
    "male": ["James", "Robert", "David", "Michael", "William", "Thomas", "Daniel", "Paul",
             "Mark", "Andrew", "Kevin", "Brian", "George", "Steven", "Peter", "Richard"],
    "female": ["Mary", "Sarah", "Emily", "Jessica", "Laura", "Emma", "Olivia", "Sophie",
               "Rachel", "Hannah", "Anna", "Grace", "Claire", "Alice", "Megan", "Lucy"],
}
