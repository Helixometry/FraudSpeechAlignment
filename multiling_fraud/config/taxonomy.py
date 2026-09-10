"""
Taxonomy for the multilingual (English) fraud dataset, mirroring TeleAntiFraud.
FRAUD CLASS ONLY. Keeps the original Chinese labels alongside English for fidelity.
"""

# 7 fraud types actually used in the original (email was in the closed set but unused).
# `weight` = original SFT-train count, used to distribute a pilot proportionally.
FRAUD_TYPES = [
    {"en": "customer service fraud", "zh": "客服诈骗", "key": "customer_service", "weight": 2022},
    {"en": "bank fraud",            "zh": "银行诈骗", "key": "bank",             "weight": 1626},
    {"en": "investment fraud",      "zh": "投资诈骗", "key": "investment",       "weight": 785},
    {"en": "phishing fraud",        "zh": "钓鱼诈骗", "key": "phishing",         "weight": 443},
    {"en": "lottery fraud",         "zh": "彩票诈骗", "key": "lottery",          "weight": 418},
    {"en": "kidnapping fraud",      "zh": "绑架诈骗", "key": "kidnapping",       "weight": 325},
    {"en": "identity theft",        "zh": "身份盗窃", "key": "identity_theft",   "weight": 105},
]

# Closed set the model must choose from for fraud_type (mirrors original: 8 incl. email).
FRAUD_TYPE_CLOSED_SET_EN = [
    "investment fraud", "phishing fraud", "identity theft", "lottery fraud",
    "bank fraud", "kidnapping fraud", "customer service fraud", "email fraud",
]

# 7 call scenes (the scene a fraud call *imitates*). Mirrors original scene closed set.
SCENES_EN = [
    "food ordering", "customer service", "appointment", "traffic inquiry",
    "daily shopping", "taxi service", "food delivery",
]
SCENES_ZH = ["订餐服务", "咨询客服", "预约服务", "交通咨询", "日常购物", "打车服务", "外卖服务"]

# Which scene each fraud type most plausibly imitates (LLM may override per dialogue).
DEFAULT_SCENE_FOR_TYPE = {
    "customer_service": "customer service",
    "bank": "customer service",
    "investment": "customer service",
    "phishing": "customer service",
    "lottery": "customer service",
    "kidnapping": "customer service",
    "identity_theft": "customer service",
}


def distribute(n_total):
    """Split n_total dialogues across fraud types proportional to original weights (>=1 each)."""
    total_w = sum(t["weight"] for t in FRAUD_TYPES)
    raw = {t["key"]: max(1, round(n_total * t["weight"] / total_w)) for t in FRAUD_TYPES}
    # adjust rounding drift to hit n_total exactly
    diff = n_total - sum(raw.values())
    order = sorted(FRAUD_TYPES, key=lambda t: -t["weight"])
    i = 0
    while diff != 0:
        k = order[i % len(order)]["key"]
        if diff > 0:
            raw[k] += 1; diff -= 1
        elif raw[k] > 1:
            raw[k] -= 1; diff += 1
        i += 1
    return raw


BY_KEY = {t["key"]: t for t in FRAUD_TYPES}

if __name__ == "__main__":
    import sys, json
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    print(f"Distribution for {n} dialogues:")
    print(json.dumps(distribute(n), indent=2))
