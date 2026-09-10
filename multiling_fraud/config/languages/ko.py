"""Korean language pack.

NOTE: `templates` currently reuses the English task templates (placeholder) so DIALOGUE
GENERATION can run now. Add config/templates_ko.py (Korean task prompts) before ASSEMBLING
the Korean dataset, and switch the import below.
"""
from config import templates_en as templates   # TODO: templates_ko for Korean assembly

CODE = "ko"
LANGUAGE_NAME = "Korean"

# Scene closed-set (Korean)
SCENES = [
    "음식 주문", "고객 서비스", "예약", "교통 문의",
    "일상 쇼핑", "택시 서비스", "음식 배달",
]

# Fraud-type closed set (Korean; 8 incl. email, mirrors original)
FRAUD_TYPE_CLOSED_SET = [
    "투자 사기", "피싱 사기", "신원 도용", "복권 사기",
    "은행 사기", "납치 사기", "고객센터 사기", "이메일 사기",
]

# fraud-type key -> display label (Korean)
FRAUD_LABELS = {
    "customer_service": "고객센터 사기",
    "bank": "은행 사기",
    "investment": "투자 사기",
    "phishing": "피싱 사기",
    "lottery": "복권 사기",
    "kidnapping": "납치 사기",
    "identity_theft": "신원 도용",
}

# fraud-type key -> imitated scene (Korean); most scams pose as customer service
SCENE_DEFAULT = {k: "고객 서비스" for k in FRAUD_LABELS}

# TTS voices (placeholder; Korean audio backend chosen at audio phase — XTTS-v2 supports 'ko')
VOICES = {"caller": "Damien Black", "callee": "Ana Florence"}
