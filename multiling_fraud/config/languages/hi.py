"""Hindi language pack.

NOTE: `templates` currently points at the English task templates as a placeholder so
DIALOGUE GENERATION (which does not use templates) can run now. Before ASSEMBLING the
Hindi dataset (Phase 1b), add a proper `config/templates_hi.py` (Hindi task prompts) and
switch the import below to it.
"""
from config import templates_en as templates   # TODO: replace with templates_hi for Hindi assembly

CODE = "hi"
LANGUAGE_NAME = "Hindi"

# Scene closed-set (Hindi)
SCENES = [
    "भोजन ऑर्डर", "ग्राहक सेवा", "अपॉइंटमेंट", "यातायात पूछताछ",
    "रोज़मर्रा की खरीदारी", "टैक्सी सेवा", "फ़ूड डिलीवरी",
]

# Fraud-type closed set (Hindi; 8 incl. email, mirrors original)
FRAUD_TYPE_CLOSED_SET = [
    "निवेश धोखाधड़ी", "फ़िशिंग धोखाधड़ी", "पहचान की चोरी", "लॉटरी धोखाधड़ी",
    "बैंक धोखाधड़ी", "अपहरण धोखाधड़ी", "ग्राहक सेवा धोखाधड़ी", "ईमेल धोखाधड़ी",
]

# fraud-type key -> display label (Hindi)
FRAUD_LABELS = {
    "customer_service": "ग्राहक सेवा धोखाधड़ी",
    "bank": "बैंक धोखाधड़ी",
    "investment": "निवेश धोखाधड़ी",
    "phishing": "फ़िशिंग धोखाधड़ी",
    "lottery": "लॉटरी धोखाधड़ी",
    "kidnapping": "अपहरण धोखाधड़ी",
    "identity_theft": "पहचान की चोरी",
}

# fraud-type key -> imitated scene (Hindi); most scams pose as customer service
SCENE_DEFAULT = {k: "ग्राहक सेवा" for k in FRAUD_LABELS}

# XTTS-v2 voices (built-in speakers work multilingually with language="hi")
VOICES = {"caller": "Damien Black", "callee": "Ana Florence"}
