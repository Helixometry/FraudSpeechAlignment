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

# XTTS-v2 built-in voices for the two roles (fallback)
VOICES = {"caller": "Damien Black", "callee": "Ana Florence"}

# gendered XTTS-v2 speaker pools -> gender-matched, distinct caller/callee, variety across calls
VOICE_POOL = {
    "male": ["Damien Black", "Craig Gutsy", "Aaron Dreschner", "Andrew Chipper", "Viktor Eka"],
    "female": ["Ana Florence", "Claribel Dervla", "Daisy Studious", "Gracie Wise", "Alison Dietlinde"],
}

# per-gender victim/caller name pools (for diverse, gender-consistent personas).
# Deliberately large + varied so neither the victim NOR the caller/agent name
# concentrates on a handful of defaults.
NAMES = {
    "male": ["James", "Robert", "David", "Michael", "William", "Thomas", "Daniel", "Paul",
             "Mark", "Andrew", "Kevin", "Brian", "George", "Steven", "Peter", "Richard",
             "John", "Christopher", "Matthew", "Anthony", "Joshua", "Nathan", "Simon",
             "Adam", "Jonathan", "Patrick", "Gary", "Dennis", "Frank", "Harold", "Ronald",
             "Carl", "Henry", "Douglas", "Arthur", "Roger", "Terry", "Keith", "Gerald",
             "Lawrence", "Ryan", "Ethan", "Nicholas", "Benjamin", "Samuel", "Gregory",
             "Sean", "Philip", "Vincent", "Marcus", "Dominic", "Oscar", "Leonard", "Wesley"],
    "female": ["Mary", "Sarah", "Emily", "Jessica", "Laura", "Emma", "Olivia", "Sophie",
               "Rachel", "Hannah", "Anna", "Grace", "Claire", "Alice", "Megan", "Lucy",
               "Elizabeth", "Katherine", "Rebecca", "Nicole", "Amanda", "Michelle",
               "Stephanie", "Melissa", "Christine", "Diane", "Carol", "Susan", "Karen",
               "Nancy", "Patricia", "Linda", "Barbara", "Sandra", "Donna", "Sharon",
               "Deborah", "Ruth", "Julia", "Victoria", "Charlotte", "Amelia", "Isabella",
               "Chloe", "Natalie", "Vanessa", "Teresa", "Gloria", "Paula", "Yvonne",
               "Bethany", "Fiona", "Heather", "Wendy"],
}
