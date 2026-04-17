"""
NLP Engine for FoodMind AI
Keyword-based feature extraction + sentiment classification
Supports English, Hindi, and Kannada via keyword mapping
"""

import re
import json
import os
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Multilingual keyword maps
# ---------------------------------------------------------------------------

# TRANSLATION LAYER – maps Hindi/Kannada words → English concept tokens
MULTILINGUAL_MAP: Dict[str, str] = {
    # Hindi – oil
    "तेल": "oil",
    "tel": "oil",
    # Hindi – spice / chilli
    "मसाला": "spice",
    "masala": "spice",
    "तीखा": "spicy",
    "teekha": "spicy",
    "mirchi": "spice",
    "मिर्ची": "spice",
    # Hindi – quantity / portion
    "कम": "less",
    "zyada": "more",
    "ज़्यादा": "more",
    "quantity": "quantity",
    # Hindi – taste
    "स्वाद": "taste",
    "swad": "taste",
    "tasty": "taste",
    # Hindi – good / bad sentiment helpers
    "अच्छा": "good",
    "accha": "good",
    "acha": "good",
    "बुरा": "bad",
    "bura": "bad",
    "kharab": "bad",
    "खराब": "bad",
    # Kannada – oil
    "ಎಣ್ಣೆ": "oil",
    "enne": "oil",
    # Kannada – spice
    "ಮಸಾಲೆ": "spice",
    "masale": "spice",
    "ಖಾರ": "spicy",
    "khaara": "spicy",
    # Kannada – quantity
    "ಕಡಿಮೆ": "less",
    "kadime": "less",
    "ಜಾಸ್ತಿ": "more",
    "jaasti": "more",
    # Kannada – taste
    "ರುಚಿ": "taste",
    "ruchi": "taste",
    # Kannada – good / bad
    "ಚೆನ್ನಾಗಿದೆ": "good",
    "chennagide": "good",
    "ಕೆಟ್ಟದಾಗಿದೆ": "bad",
    "kettadagide": "bad",
}

# FEATURE KEYWORD SETS (after multilingual normalisation)
FEATURE_KEYWORDS: Dict[str, List[str]] = {
    "oil": [
        "oil",
        "oily",
        "greasy",
        "grease",
        "fatty",
        "fat",
        "dripping",
        "ghee",
        "butter",
    ],
    "spice": [
        "spice",
        "spicy",
        "spices",
        "spiced",
        "chili",
        "chilli",
        "pepper",
        "hot",
        "mild",
        "bland",
        "masala",
        "teekha",
        "khaara",
    ],
    "quantity": [
        "quantity",
        "portion",
        "amount",
        "serving",
        "size",
        "small",
        "large",
        "big",
        "less",
        "more",
        "half",
        "full",
        "insufficient",
    ],
    "taste": [
        "taste",
        "flavor",
        "flavour",
        "delicious",
        "yummy",
        "tasty",
        "bland",
        "good",
        "bad",
        "awful",
        "amazing",
        "terrible",
        "swad",
        "ruchi",
    ],
}

# SENTIMENT KEYWORD SETS
POSITIVE_WORDS: List[str] = [
    "good",
    "great",
    "excellent",
    "amazing",
    "awesome",
    "perfect",
    "loved",
    "love",
    "fantastic",
    "wonderful",
    "superb",
    "delicious",
    "yummy",
    "tasty",
    "fresh",
    "clean",
    "happy",
    "satisfied",
    "nice",
    "best",
    "accha",
    "acha",
    "chennagide",
    "liked",
    "like",
    "enjoyable",
    "enjoy",
    "recommend",
    "outstanding",
    "brilliant",
]

NEGATIVE_WORDS: List[str] = [
    "bad",
    "terrible",
    "awful",
    "horrible",
    "worst",
    "hate",
    "hated",
    "poor",
    "disappointing",
    "disgusting",
    "greasy",
    "oily",
    "too much",
    "too little",
    "bland",
    "stale",
    "cold",
    "overcooked",
    "undercooked",
    "complaint",
    "unhappy",
    "unhygienic",
    "bura",
    "kharab",
    "kettadagide",
    "not good",
    "not great",
    "not fresh",
    "rotten",
]

# NEGATION WORDS (flip sentiment)
NEGATION_WORDS: List[str] = [
    "not",
    "no",
    "never",
    "neither",
    "nope",
    "nah",
    "without",
    "nahi",
    "mat",
    "nahin",
]

# FEATURE-SPECIFIC COMPLAINT PATTERNS
COMPLAINT_PATTERNS: Dict[str, List[str]] = {
    "oil": [
        "too oily",
        "very oily",
        "excess oil",
        "too much oil",
        "dripping oil",
        "greasy",
        "too greasy",
        "lots of oil",
        "reduce oil",
    ],
    "spice": [
        "too spicy",
        "very spicy",
        "too hot",
        "too mild",
        "not spicy",
        "bland",
        "no spice",
        "less spice",
        "more spice",
        "too much spice",
    ],
    "quantity": [
        "small portion",
        "less quantity",
        "small serving",
        "not enough",
        "too little",
        "insufficient",
        "half portion",
        "small amount",
    ],
    "taste": [
        "bad taste",
        "no taste",
        "tasteless",
        "awful taste",
        "terrible taste",
        "not tasty",
        "not good",
        "poor taste",
        "horrible taste",
        "bland taste",
    ],
}

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


PRECISE_FEATURES: List[str] = [
    "oil_level",
    "spice_level",
    "quantity",
    "texture",
    "taste",
    "price_value",
    "freshness",
    "service",
    "hygiene",
]

PRECISE_TO_LEGACY = {
    "oil_level": "oil",
    "spice_level": "spice",
    "quantity": "quantity",
    "taste": "taste",
}

PRECISE_SENTIMENTS = {
    "strongly_negative",
    "negative",
    "neutral",
    "positive",
    "strongly_positive",
}


def _sentiment5_to_legacy(sentiment: str) -> str:
    if sentiment in {"strongly_negative", "negative"}:
        return "negative"
    if sentiment in {"strongly_positive", "positive"}:
        return "positive"
    return "neutral"


def _overall_to_legacy(overall: str, features: Dict[str, Dict]) -> str:
    if overall == "positive":
        return "positive"
    if overall == "negative":
        return "negative"

    pos = 0
    neg = 0
    for item in features.values():
        s = item.get("sentiment", "neutral")
        if s in {"positive", "strongly_positive"}:
            pos += 1
        elif s in {"negative", "strongly_negative"}:
            neg += 1

    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def _coerce_precise_schema(data: Dict) -> Dict:
    language = str(data.get("language", "mixed")).strip().lower()
    if language not in {"english", "hindi", "kannada", "mixed"}:
        language = "mixed"

    features_in = data.get("features", {}) or {}
    features_out: Dict[str, Dict] = {}
    for feature in PRECISE_FEATURES:
        if feature not in features_in:
            continue

        obj = features_in.get(feature, {}) or {}
        sentiment = str(obj.get("sentiment", "neutral"))
        if sentiment not in PRECISE_SENTIMENTS:
            sentiment = "neutral"

        evidence = str(obj.get("evidence", "")).strip()
        if not evidence:
            continue

        confidence = obj.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, round(confidence, 2)))

        features_out[feature] = {
            "sentiment": sentiment,
            "evidence": evidence,
            "confidence": confidence,
        }

    overall = str(data.get("overall", "mixed")).strip().lower()
    if overall not in {"positive", "negative", "mixed"}:
        overall = "mixed"

    key_complaint = data.get("key_complaint")
    key_praise = data.get("key_praise")
    if key_complaint is not None:
        key_complaint = str(key_complaint).strip() or None
    if key_praise is not None:
        key_praise = str(key_praise).strip() or None

    return {
        "language": language,
        "features": features_out,
        "overall": overall,
        "key_complaint": key_complaint,
        "key_praise": key_praise,
    }


def _analyse_review_openai_precise(text: str) -> Dict | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except Exception:
        return None

    system_prompt = (
        "You are a food review NLP expert. Analyze each review with surgical precision. "
        "Rules: detect language first (English/Hindi/Kannada/mixed), extract only explicitly mentioned features, "
        "feature-level sentiment only, confidence 0..1 per feature based on clarity, skip unmentioned features entirely. "
        "Features to extract: oil_level, spice_level, quantity, texture, taste, price_value, freshness, service, hygiene. "
        "Sentiment scale: strongly_negative, negative, neutral, positive, strongly_positive. "
        "Accuracy rules: 'too oily' => strongly_negative oil_level confidence 0.95; "
        "'little oily' => negative oil_level confidence 0.75; "
        "'perfect spice' => strongly_positive spice_level confidence 0.92; "
        "Sarcasm like 'oh great, swimming in oil' => strongly_negative oil_level; "
        "contradictory reviews split by feature and overall=mixed; "
        "short review 'good' => taste positive confidence 0.55 only. "
        "Return strict JSON only with schema: "
        '{"language":"...","features":{"oil_level":{"sentiment":"...","evidence":"exact phrase from review","confidence":0.0}},"overall":"positive|negative|mixed","key_complaint":"one line or null","key_praise":"one line or null"}."'
    )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    )

    content = response.choices[0].message.content or "{}"
    data = json.loads(content)
    return _coerce_precise_schema(data)


def _analyse_review_precise_fallback(text: str) -> Dict:
    norm = _normalise(text)
    features: Dict[str, Dict] = {}

    if "too oily" in norm:
        features["oil_level"] = {
            "sentiment": "strongly_negative",
            "evidence": "too oily",
            "confidence": 0.95,
        }
    elif "little oily" in norm:
        features["oil_level"] = {
            "sentiment": "negative",
            "evidence": "little oily",
            "confidence": 0.75,
        }
    elif any(
        k in norm
        for k in [
            "very oily",
            "swimming in oil",
            "excess oil",
            "oily",
            "greasy",
            "too much oil",
        ]
    ):
        evidence = (
            "swimming in oil"
            if "swimming in oil" in norm
            else (
                "very oily"
                if "very oily" in norm
                else (
                    "excess oil"
                    if "excess oil" in norm
                    else (
                        "too much oil"
                        if "too much oil" in norm
                        else ("greasy" if "greasy" in norm else "oily")
                    )
                )
            )
        )
        sentiment = (
            "strongly_negative"
            if evidence
            in {"swimming in oil", "very oily", "excess oil", "too much oil"}
            else "negative"
        )
        features["oil_level"] = {
            "sentiment": sentiment,
            "evidence": evidence,
            "confidence": 0.9 if sentiment == "strongly_negative" else 0.8,
        }

    if "perfect spice" in norm:
        features["spice_level"] = {
            "sentiment": "strongly_positive",
            "evidence": "perfect spice",
            "confidence": 0.92,
        }
    elif any(k in norm for k in ["too spicy", "very spicy", "not spicy", "bland"]):
        evidence = (
            "too spicy"
            if "too spicy" in norm
            else (
                "very spicy"
                if "very spicy" in norm
                else ("not spicy" if "not spicy" in norm else "bland")
            )
        )
        sentiment = (
            "strongly_negative"
            if evidence in {"too spicy", "very spicy"}
            else "negative"
        )
        features["spice_level"] = {
            "sentiment": sentiment,
            "evidence": evidence,
            "confidence": 0.9 if sentiment == "strongly_negative" else 0.8,
        }

    if any(
        k in norm
        for k in ["small portion", "less quantity", "not enough", "portion", "quantity"]
    ):
        evidence = (
            "small portion"
            if "small portion" in norm
            else (
                "less quantity"
                if "less quantity" in norm
                else (
                    "not enough"
                    if "not enough" in norm
                    else ("quantity" if "quantity" in norm else "portion")
                )
            )
        )
        sentiment = (
            "negative"
            if evidence in {"small portion", "less quantity", "not enough"}
            else "neutral"
        )
        features["quantity"] = {
            "sentiment": sentiment,
            "evidence": evidence,
            "confidence": 0.82 if sentiment == "negative" else 0.6,
        }

    if any(
        k in norm for k in ["crispy", "soggy", "overcooked", "undercooked", "texture"]
    ):
        evidence = (
            "soggy"
            if "soggy" in norm
            else (
                "overcooked"
                if "overcooked" in norm
                else (
                    "undercooked"
                    if "undercooked" in norm
                    else ("crispy" if "crispy" in norm else "texture")
                )
            )
        )
        if evidence in {"soggy", "overcooked", "undercooked"}:
            sentiment = "negative"
            conf = 0.85
        elif evidence == "crispy":
            sentiment = "positive"
            conf = 0.8
        else:
            sentiment = "neutral"
            conf = 0.6
        features["texture"] = {
            "sentiment": sentiment,
            "evidence": evidence,
            "confidence": conf,
        }

    if norm.strip() == "good":
        features["taste"] = {
            "sentiment": "positive",
            "evidence": "good",
            "confidence": 0.55,
        }
    elif any(
        k in norm
        for k in [
            "taste",
            "tasty",
            "delicious",
            "awful taste",
            "bad taste",
            "tasteless",
        ]
    ):
        evidence = (
            "awful taste"
            if "awful taste" in norm
            else (
                "bad taste"
                if "bad taste" in norm
                else (
                    "tasteless"
                    if "tasteless" in norm
                    else (
                        "delicious"
                        if "delicious" in norm
                        else ("tasty" if "tasty" in norm else "taste")
                    )
                )
            )
        )
        if evidence in {"awful taste", "bad taste", "tasteless"}:
            sentiment = "strongly_negative"
            conf = 0.9
        elif evidence in {"delicious", "tasty"}:
            sentiment = "positive"
            conf = 0.82
        else:
            sentiment = "neutral"
            conf = 0.6
        features["taste"] = {
            "sentiment": sentiment,
            "evidence": evidence,
            "confidence": conf,
        }

    if any(
        k in norm
        for k in ["expensive", "overpriced", "worth the price", "price", "value"]
    ):
        evidence = (
            "overpriced"
            if "overpriced" in norm
            else (
                "expensive"
                if "expensive" in norm
                else (
                    "worth the price"
                    if "worth the price" in norm
                    else ("price" if "price" in norm else "value")
                )
            )
        )
        sentiment = (
            "negative" if evidence in {"expensive", "overpriced"} else "positive"
        )
        features["price_value"] = {
            "sentiment": sentiment,
            "evidence": evidence,
            "confidence": 0.85 if sentiment == "negative" else 0.8,
        }

    if any(k in norm for k in ["fresh", "stale", "rotten", "not fresh"]):
        evidence = (
            "not fresh"
            if "not fresh" in norm
            else (
                "stale"
                if "stale" in norm
                else ("rotten" if "rotten" in norm else "fresh")
            )
        )
        sentiment = (
            "negative" if evidence in {"not fresh", "stale", "rotten"} else "positive"
        )
        features["freshness"] = {
            "sentiment": sentiment,
            "evidence": evidence,
            "confidence": 0.9 if sentiment == "negative" else 0.84,
        }

    if any(
        k in norm
        for k in [
            "service",
            "staff",
            "waiter",
            "server",
            "rude",
            "friendly",
            "hospitality",
            "slow service",
        ]
    ):
        evidence = (
            "slow service"
            if "slow service" in norm
            else (
                "rude"
                if "rude" in norm
                else (
                    "friendly"
                    if "friendly" in norm
                    else ("hospitality" if "hospitality" in norm else "service")
                )
            )
        )
        if evidence in {"slow service", "rude"}:
            sentiment = "negative"
            conf = 0.86
        elif evidence in {"friendly", "hospitality"}:
            sentiment = "positive"
            conf = 0.8
        else:
            sentiment = "neutral"
            conf = 0.6
        features["service"] = {
            "sentiment": sentiment,
            "evidence": evidence,
            "confidence": conf,
        }

    if any(
        k in norm
        for k in [
            "hygiene",
            "hygienic",
            "dirty",
            "unclean",
            "clean",
            "filthy",
            "sanitary",
        ]
    ):
        evidence = (
            "dirty"
            if "dirty" in norm
            else (
                "filthy"
                if "filthy" in norm
                else (
                    "unclean"
                    if "unclean" in norm
                    else (
                        "hygienic"
                        if "hygienic" in norm
                        else ("hygiene" if "hygiene" in norm else "clean")
                    )
                )
            )
        )
        if evidence in {"dirty", "filthy", "unclean"}:
            sentiment = "negative"
            conf = 0.88
        elif evidence in {"clean", "sanitary", "hygienic"}:
            sentiment = "positive"
            conf = 0.8
        else:
            sentiment = "neutral"
            conf = 0.6
        features["hygiene"] = {
            "sentiment": sentiment,
            "evidence": evidence,
            "confidence": conf,
        }

    has_hindi = any(
        ch in text for ch in "अआइईउऊएऐओऔकखगघचछजझटठडढतथदधनपफबभमयरलवशषसह़ािीुूेैोौंः"
    )
    has_kannada = any("\u0c80" <= ch <= "\u0cff" for ch in text)
    has_english = bool(re.search(r"[a-zA-Z]", text))
    if (
        (has_hindi and has_english)
        or (has_kannada and has_english)
        or (has_hindi and has_kannada)
    ):
        language = "mixed"
    elif has_hindi:
        language = "hindi"
    elif has_kannada:
        language = "kannada"
    else:
        language = "english"

    pos = 0
    neg = 0
    for item in features.values():
        s = item["sentiment"]
        if s in {"positive", "strongly_positive"}:
            pos += 1
        elif s in {"negative", "strongly_negative"}:
            neg += 1

    if pos > 0 and neg > 0:
        overall = "mixed"
    elif neg > 0:
        overall = "negative"
    elif pos > 0:
        overall = "positive"
    else:
        overall = "mixed"

    key_complaint = None
    key_praise = None
    for feat, obj in features.items():
        if (
            obj["sentiment"] in {"negative", "strongly_negative"}
            and key_complaint is None
        ):
            key_complaint = f"{feat} issue: {obj['evidence']}"
        if obj["sentiment"] in {"positive", "strongly_positive"} and key_praise is None:
            key_praise = f"{feat} praised: {obj['evidence']}"

    return {
        "language": language,
        "features": features,
        "overall": overall,
        "key_complaint": key_complaint,
        "key_praise": key_praise,
    }


def analyse_review_precise(text: str) -> Dict:
    try:
        llm_result = _analyse_review_openai_precise(text)
        if llm_result is not None:
            return llm_result
    except Exception:
        pass
    return _analyse_review_precise_fallback(text)


def _analyse_review_openai(text: str) -> Dict | None:
    """Use OpenAI precise schema and adapt it to legacy output fields."""
    precise = _analyse_review_openai_precise(text)
    if precise is None:
        return None

    legacy_features: Dict[str, bool] = {f: False for f in FEATURE_KEYWORDS}
    legacy_feature_sentiments: Dict[str, str] = {}

    for p_feature, payload in precise["features"].items():
        legacy_key = PRECISE_TO_LEGACY.get(p_feature)
        if not legacy_key:
            continue
        legacy_features[legacy_key] = True
        legacy_feature_sentiments[legacy_key] = _sentiment5_to_legacy(
            payload["sentiment"]
        )

    tags = []
    for feature, present in legacy_features.items():
        if present:
            tags.append(
                {
                    "feature": feature,
                    "sentiment": legacy_feature_sentiments.get(feature, "neutral"),
                }
            )

    return {
        "text": text,
        "sentiment": _overall_to_legacy(precise["overall"], precise["features"]),
        "features": legacy_features,
        "feature_sentiments": legacy_feature_sentiments,
        "tags": tags,
        "language": precise["language"],
        "precise_features": precise["features"],
        "overall": precise["overall"],
        "key_complaint": precise["key_complaint"],
        "key_praise": precise["key_praise"],
    }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, apply multilingual map."""
    text = text.lower()
    # Replace multilingual tokens with English equivalents
    for native, english in MULTILINGUAL_MAP.items():
        text = text.replace(native.lower(), f" {english} ")
    # Remove punctuation except spaces
    text = re.sub(r"[^\w\s]", " ", text)
    return text


def _tokenize(text: str) -> List[str]:
    return text.split()


def _window_contains_negation(tokens: List[str], idx: int, window: int = 3) -> bool:
    start = max(0, idx - window)
    return any(t in NEGATION_WORDS for t in tokens[start:idx])


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------


def classify_sentiment(text: str) -> str:
    """Return 'positive', 'negative', or 'neutral'."""
    norm = _normalise(text)
    tokens = _tokenize(norm)

    pos_score = 0
    neg_score = 0

    for i, token in enumerate(tokens):
        negated = _window_contains_negation(tokens, i)
        if token in POSITIVE_WORDS:
            if negated:
                neg_score += 1
            else:
                pos_score += 1
        elif token in NEGATIVE_WORDS:
            if negated:
                pos_score += 0.5
            else:
                neg_score += 1

    # Phrase-level boosts
    if "not good" in norm or "not great" in norm or "not fresh" in norm:
        neg_score += 1
    if "very good" in norm or "really good" in norm:
        pos_score += 1

    if pos_score > neg_score:
        return "positive"
    if neg_score > pos_score:
        return "negative"
    return "neutral"


def extract_features(text: str) -> Dict[str, bool]:
    """Return dict of which features are mentioned."""
    norm = _normalise(text)
    result: Dict[str, bool] = {}
    for feature, keywords in FEATURE_KEYWORDS.items():
        result[feature] = any(kw in norm for kw in keywords)
    return result


def extract_feature_sentiment(text: str) -> Dict[str, str]:
    """
    For each feature present in the text, classify whether the mention
    is positive, negative, or neutral.
    """
    norm = _normalise(text)
    tokens = _tokenize(norm)
    feature_sentiment: Dict[str, str] = {}

    for feature, keywords in FEATURE_KEYWORDS.items():
        mentioned_indices = [i for i, t in enumerate(tokens) if t in keywords]
        if not mentioned_indices:
            continue

        pos = 0
        neg = 0
        for idx in mentioned_indices:
            negated = _window_contains_negation(tokens, idx)
            # Check words within a small window after feature token
            context = tokens[idx : idx + 5]
            for ct in context:
                if ct in POSITIVE_WORDS:
                    if negated:
                        neg += 1
                    else:
                        pos += 1
                elif ct in NEGATIVE_WORDS:
                    if negated:
                        pos += 0.5
                    else:
                        neg += 1

            # Check complaint patterns
            for pattern in COMPLAINT_PATTERNS.get(feature, []):
                if pattern in norm:
                    neg += 1

        if neg > pos:
            feature_sentiment[feature] = "negative"
        elif pos > neg:
            feature_sentiment[feature] = "positive"
        else:
            feature_sentiment[feature] = "neutral"

    return feature_sentiment


def analyse_review(text: str) -> Dict:
    """Full analysis of a single review."""
    try:
        llm_result = _analyse_review_openai(text)
        if llm_result is not None:
            return llm_result
    except Exception:
        # Fallback to deterministic local analysis on API/network/parse failures.
        pass

    sentiment = classify_sentiment(text)
    features = extract_features(text)
    feature_sentiments = extract_feature_sentiment(text)

    tags = []
    for feature, present in features.items():
        if present:
            fs = feature_sentiments.get(feature, "neutral")
            tags.append({"feature": feature, "sentiment": fs})

    return {
        "text": text,
        "sentiment": sentiment,
        "features": features,
        "feature_sentiments": feature_sentiments,
        "tags": tags,
    }


# ---------------------------------------------------------------------------
# Aggregate analysis
# ---------------------------------------------------------------------------


def aggregate_insights(reviews: List[Dict]) -> Dict:
    """
    Accepts list of analyse_review() outputs.
    Returns complaint percentages, top complaint, alerts, recommendations,
    impact simulation, and menu customisations.
    """
    total = len(reviews)
    if total == 0:
        return {}

    # Count per-feature sentiment totals
    feature_counts: Dict[str, Dict[str, int]] = {
        f: {"positive": 0, "negative": 0, "neutral": 0, "total": 0}
        for f in FEATURE_KEYWORDS
    }

    sentiment_totals = {"positive": 0, "negative": 0, "neutral": 0}

    for r in reviews:
        sentiment_totals[r["sentiment"]] += 1
        for feature, present in r["features"].items():
            if present:
                feature_counts[feature]["total"] += 1
                fs = r["feature_sentiments"].get(feature, "neutral")
                feature_counts[feature][fs] += 1

    # Complaint percentages (negative mentions / total reviews)
    complaint_pct: Dict[str, float] = {}
    for feature, counts in feature_counts.items():
        if counts["total"] > 0:
            complaint_pct[feature] = round(counts["negative"] / total * 100, 1)
        else:
            complaint_pct[feature] = 0.0

    # Top complaint
    top_complaint_feature = max(complaint_pct, key=lambda f: complaint_pct[f])
    top_complaint_pct = complaint_pct[top_complaint_feature]

    # Alert if top complaint > 25%
    alerts = []
    for feature, pct in complaint_pct.items():
        if pct >= 25:
            alerts.append(
                {
                    "feature": feature,
                    "percentage": pct,
                    "message": f"⚠️ {feature.capitalize()} complaints are at {pct}% — needs immediate action!",
                    "level": "critical" if pct >= 40 else "warning",
                }
            )

    # Sentiment overview
    pos_pct = round(sentiment_totals["positive"] / total * 100, 1)
    neg_pct = round(sentiment_totals["negative"] / total * 100, 1)
    neu_pct = round(sentiment_totals["neutral"] / total * 100, 1)

    # Recommendations
    recommendations = _generate_recommendations(complaint_pct, feature_counts, total)

    # Impact simulation
    impact = _simulate_impact(complaint_pct, recommendations, total)

    # Menu customisations
    customisations = _generate_customisations(complaint_pct)

    # Trend (simple: split reviews into first/second half and compare)
    trend = _compute_trend(reviews)

    return {
        "total_reviews": total,
        "sentiment_overview": {
            "positive": pos_pct,
            "negative": neg_pct,
            "neutral": neu_pct,
            "counts": sentiment_totals,
        },
        "feature_complaints": complaint_pct,
        "feature_counts": feature_counts,
        "top_complaint": {
            "feature": top_complaint_feature,
            "percentage": top_complaint_pct,
        },
        "alerts": alerts,
        "recommendations": recommendations,
        "impact": impact,
        "customisations": customisations,
        "trend": trend,
    }


def _generate_recommendations(
    complaint_pct: Dict[str, float],
    feature_counts: Dict[str, Dict[str, int]],
    total: int,
) -> List[Dict]:
    recs = []

    # Sort features by complaint %
    sorted_features = sorted(complaint_pct.items(), key=lambda x: x[1], reverse=True)

    for feature, pct in sorted_features[:3]:
        if pct < 5:
            continue
        neg_count = feature_counts[feature]["negative"]
        confidence = min(round(40 + pct * 0.9 + (neg_count / total) * 20, 1), 98)

        if feature == "oil":
            action = "Reduce oil quantity by 15–20% across all fried items"
            detail = "Switch to air-frying or use non-stick pans"
        elif feature == "spice":
            action = "Introduce a customisable spice level (mild / medium / hot)"
            detail = "Use a standard spice base and add chilli oil on request"
        elif feature == "quantity":
            action = (
                "Increase portion size by 10% or offer an 'Extra Portion' add-on (+₹30)"
            )
            detail = "Standardise serving scoops to ensure consistency"
        elif feature == "taste":
            action = (
                "Revise base recipe — conduct a tasting panel with 5 focus customers"
            )
            detail = "Review seasoning ratios; consider adding a secret spice blend"
        else:
            action = f"Address {feature} quality concerns"
            detail = "Review preparation process for this attribute"

        recs.append(
            {
                "feature": feature,
                "action": action,
                "detail": detail,
                "confidence": confidence,
                "based_on": neg_count,
                "reasoning": (
                    f"{neg_count} out of {total} reviews ({pct}%) mention "
                    f"{feature} negatively. Confidence driven by review volume."
                ),
            }
        )

    return recs[:3]


def _simulate_impact(
    complaint_pct: Dict[str, float],
    recommendations: List[Dict],
    total: int,
) -> Dict:
    """
    Simple but believable impact simulation:
    - Each addressed complaint reduces negative reviews proportionally
    - Rating improvement modelled as: delta = sum(complaint_reduction * weight)
    """
    # Assume current average rating ~ 3.5 if negative > 40%, else 4.0
    avg_neg = sum(complaint_pct.values()) / max(len(complaint_pct), 1)
    current_rating = 3.2 if avg_neg > 40 else (3.6 if avg_neg > 20 else 4.0)

    # Each recommendation reduces its feature complaint by ~50%
    rating_delta = 0.0
    complaint_reduction: Dict[str, float] = {}
    for rec in recommendations:
        f = rec["feature"]
        pct = complaint_pct.get(f, 0)
        reduction = round(pct * 0.5, 1)
        complaint_reduction[f] = reduction
        # Each 10% complaint reduction → +0.1 star (capped)
        rating_delta += (reduction / 10) * 0.1

    rating_delta = round(min(rating_delta, 0.8), 2)
    predicted_rating = round(min(current_rating + rating_delta, 5.0), 1)

    # Before / after complaint numbers
    before: Dict[str, float] = {f: p for f, p in complaint_pct.items()}
    after: Dict[str, float] = {}
    for f, p in complaint_pct.items():
        reduction = complaint_reduction.get(f, 0)
        after[f] = max(round(p - reduction, 1), 0)

    return {
        "current_rating": current_rating,
        "predicted_rating": predicted_rating,
        "rating_increase": rating_delta,
        "complaint_reduction": complaint_reduction,
        "before_complaints": before,
        "after_complaints": after,
        "summary": (
            f"By implementing the top recommendations, expected rating "
            f"improvement is +{rating_delta} stars "
            f"(from {current_rating} → {predicted_rating})."
        ),
    }


def _generate_customisations(complaint_pct: Dict[str, float]) -> List[Dict]:
    options = []

    if complaint_pct.get("oil", 0) >= 10:
        options.append(
            {
                "name": "Light Oil Version",
                "description": "Prepared with 50% less oil using air-fry technique",
                "price_delta": "+₹0 (free upgrade)",
                "tag": "Healthy Choice 🥗",
            }
        )

    if complaint_pct.get("spice", 0) >= 10:
        options.append(
            {
                "name": "Custom Spice Level",
                "description": "Choose: Mild 🟢 / Medium 🟡 / Extra Hot 🔴",
                "price_delta": "+₹0 (free customisation)",
                "tag": "Personalised 🌶️",
            }
        )

    if complaint_pct.get("quantity", 0) >= 10:
        options.append(
            {
                "name": "Extra Protein Booster",
                "description": "Add extra egg / paneer / chicken to any dish",
                "price_delta": "+₹20 – ₹40",
                "tag": "Filling & Satisfying 💪",
            }
        )

    if not options:
        # Always offer at least one default customisation
        options.append(
            {
                "name": "Chef's Special Version",
                "description": "Freshly prepared with chef's premium seasoning blend",
                "price_delta": "+₹15",
                "tag": "Premium ⭐",
            }
        )

    return options[:3]


def _compute_trend(reviews: List[Dict]) -> Dict:
    """Compare first-half vs second-half negative sentiment rates."""
    n = len(reviews)
    if n < 4:
        return {"direction": "insufficient_data", "delta": 0}

    mid = n // 2
    first_half = reviews[:mid]
    second_half = reviews[mid:]

    def neg_rate(batch: List[Dict]) -> float:
        return sum(1 for r in batch if r["sentiment"] == "negative") / len(batch)

    r1 = neg_rate(first_half)
    r2 = neg_rate(second_half)
    delta = round((r2 - r1) * 100, 1)

    if delta > 5:
        direction = "worsening"
    elif delta < -5:
        direction = "improving"
    else:
        direction = "stable"

    return {
        "direction": direction,
        "delta": delta,
        "first_half_neg_rate": round(r1 * 100, 1),
        "second_half_neg_rate": round(r2 * 100, 1),
        "message": (
            f"Complaint trend is {direction} "
            f"({'↑' if delta > 0 else '↓' if delta < 0 else '→'} {abs(delta)}%)"
        ),
    }


def _feature_severity(sentiment: str) -> float:
    return {
        "strongly_negative": 1.0,
        "negative": 0.7,
        "neutral": 0.0,
        "positive": 0.0,
        "strongly_positive": 0.0,
    }.get(sentiment, 0.0)


def _feature_priority(score: float) -> str:
    if score >= 0.45:
        return "high"
    if score >= 0.2:
        return "medium"
    return "low"


def _effort_vs_impact(priority: str, score: float) -> str:
    if priority == "high" and score >= 0.45:
        return "high-impact / low-effort"
    if priority == "high":
        return "high-impact / medium-effort"
    if priority == "medium":
        return "medium-impact / medium-effort"
    return "low-impact / low-effort"


def _recommendation_for_feature(feature: str) -> str:
    fixes = {
        "oil_level": "Reduce oil by standardizing pan temperature, drain time, and portion-specific oil measurements.",
        "spice_level": "Add a spice calibration chart and offer selectable spice levels at order time.",
        "quantity": "Recalibrate serving scoops and weigh portions for consistency.",
        "texture": "Adjust cook time and holding process to preserve crispness and avoid sogginess.",
        "taste": "Review recipe balance and run a tasting loop on the base seasoning profile.",
        "price_value": "Align price with portion size or add a lower-cost value option.",
        "freshness": "Shorten prep-to-serve time and tighten ingredient rotation.",
        "service": "Train front-of-house staff on speed, order handling, and complaint recovery.",
        "hygiene": "Tighten kitchen sanitation checks, cleaning cadence, and visible prep discipline.",
    }
    return fixes.get(
        feature,
        "Review the preparation workflow and standardize the step that creates the issue.",
    )


def _build_issue_examples(
    reviews: List[Dict], feature: str, limit: int = 2
) -> List[str]:
    examples: List[str] = []
    for review in reviews:
        precise = review.get("precise_features") or review.get("features", {}) or {}
        if feature in precise and precise[feature]["sentiment"] in {
            "negative",
            "strongly_negative",
        }:
            evidence = precise[feature].get("evidence")
            if evidence and evidence not in examples:
                examples.append(evidence)
        if len(examples) >= limit:
            break
    return examples


def _aggregate_precise_reviews(reviews: List[Dict]) -> Dict:
    total = len(reviews)
    if total == 0:
        return {
            "insights": {},
            "top_issues": [],
            "recommendations": [],
            "impact_simulation": {},
            "before_vs_after": {},
        }

    language_counts: Dict[str, int] = {
        "english": 0,
        "hindi": 0,
        "kannada": 0,
        "mixed": 0,
    }
    sentiment_counts: Dict[str, int] = {
        "positive": 0,
        "negative": 0,
        "neutral": 0,
        "mixed": 0,
    }
    feature_stats: Dict[str, Dict[str, float]] = {
        feature: {
            "mentions": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "confidence_sum": 0.0,
            "negative_confidence_sum": 0.0,
            "positive_confidence_sum": 0.0,
        }
        for feature in PRECISE_FEATURES
    }

    for review in reviews:
        language = review.get("language", "mixed")
        if language not in language_counts:
            language = "mixed"
        language_counts[language] += 1

        overall = review.get("overall", "mixed")
        sentiment_counts[overall if overall in sentiment_counts else "mixed"] += 1

        for feature, payload in (review.get("features", {}) or {}).items():
            if feature not in feature_stats:
                continue
            feature_stats[feature]["mentions"] += 1
            confidence = float(payload.get("confidence", 0.5))
            feature_stats[feature]["confidence_sum"] += confidence
            sentiment = payload.get("sentiment", "neutral")
            if sentiment in {"negative", "strongly_negative"}:
                feature_stats[feature]["negative"] += 1
                feature_stats[feature]["negative_confidence_sum"] += confidence
            elif sentiment in {"positive", "strongly_positive"}:
                feature_stats[feature]["positive"] += 1
                feature_stats[feature]["positive_confidence_sum"] += confidence
            else:
                feature_stats[feature]["neutral"] += 1

    top_issues = []
    strengths = []
    for feature, stats in feature_stats.items():
        mentions = int(stats["mentions"])
        if mentions == 0:
            continue

        negative_mentions = int(stats["negative"])
        positive_mentions = int(stats["positive"])
        avg_conf = round(stats["confidence_sum"] / mentions, 2)
        avg_neg_conf = (
            round(stats["negative_confidence_sum"] / negative_mentions, 2)
            if negative_mentions
            else 0.0
        )
        avg_pos_conf = (
            round(stats["positive_confidence_sum"] / positive_mentions, 2)
            if positive_mentions
            else 0.0
        )

        frequency = negative_mentions / total if negative_mentions else 0.0
        severity = 0.0
        if negative_mentions:
            severity = 1.0 if avg_neg_conf >= 0.9 else 0.7
        impact_score = round(
            frequency * severity * (avg_neg_conf if negative_mentions else avg_conf), 3
        )

        if negative_mentions:
            top_issues.append(
                {
                    "feature": feature,
                    "frequency": frequency,
                    "negative_mentions": negative_mentions,
                    "severity": severity,
                    "confidence": avg_neg_conf or avg_conf,
                    "impact_score": impact_score,
                    "examples": _build_issue_examples(reviews, feature),
                }
            )

        if positive_mentions > negative_mentions:
            positive_frequency = positive_mentions / total
            strengths.append(
                {
                    "feature": feature,
                    "frequency": positive_frequency,
                    "positive_mentions": positive_mentions,
                    "confidence": avg_pos_conf or avg_conf,
                    "strength_score": round(
                        positive_frequency * (avg_pos_conf or avg_conf), 3
                    ),
                }
            )

    top_issues.sort(key=lambda item: item["impact_score"], reverse=True)
    strengths.sort(key=lambda item: item["strength_score"], reverse=True)

    recommendations = []
    for issue in top_issues[:4]:
        priority = _feature_priority(issue["impact_score"])
        recommendations.append(
            {
                "feature": issue["feature"],
                "fix": _recommendation_for_feature(issue["feature"]),
                "priority": priority,
                "effort_vs_impact": _effort_vs_impact(priority, issue["impact_score"]),
                "rationale": f"{issue['negative_mentions']} negative mentions out of {total} reviews.",
                "expected_rating_lift": round(
                    min(0.8, 0.15 + issue["impact_score"] * 1.2), 2
                ),
                "confidence": round(min(0.98, max(0.55, issue["confidence"])), 2),
            }
        )

    current_rating = round(
        max(
            1.0,
            min(
                5.0,
                4.8
                - (sentiment_counts["negative"] / total) * 2.0
                - (sentiment_counts["mixed"] / total) * 0.7
                + (sentiment_counts["positive"] / total) * 0.35,
            ),
        ),
        1,
    )
    projected_lift = round(
        sum(item["expected_rating_lift"] for item in recommendations[:3]), 2
    )
    predicted_rating = round(min(5.0, current_rating + projected_lift), 1)
    improvement_pct = (
        round(((predicted_rating - current_rating) / current_rating) * 100, 1)
        if current_rating
        else 0.0
    )

    before_complaints = {
        item["feature"]: round((item["negative_mentions"] / total) * 100, 1)
        for item in top_issues
    }
    after_complaints = {
        feature: round(max(0.0, pct - min(35.0, pct * 0.55)), 1)
        for feature, pct in before_complaints.items()
    }

    before_sentiment = {
        "positive": round((sentiment_counts["positive"] / total) * 100, 1),
        "negative": round((sentiment_counts["negative"] / total) * 100, 1),
        "neutral": round((sentiment_counts["neutral"] / total) * 100, 1),
        "mixed": round((sentiment_counts["mixed"] / total) * 100, 1),
    }
    after_sentiment = {
        "positive": round(
            min(100.0, before_sentiment["positive"] + projected_lift * 18), 1
        ),
        "negative": round(
            max(0.0, before_sentiment["negative"] - projected_lift * 18), 1
        ),
        "neutral": round(
            max(
                0.0,
                100.0 - (before_sentiment["positive"] + before_sentiment["negative"]),
            ),
            1,
        ),
        "mixed": round(max(0.0, before_sentiment["mixed"] - projected_lift * 4), 1),
    }

    insights = {
        "total_reviews": total,
        "languages": language_counts,
        "sentiment_distribution": before_sentiment,
        "feature_summary": {
            feature: {
                "mentions": int(stats["mentions"]),
                "positive": int(stats["positive"]),
                "negative": int(stats["negative"]),
                "neutral": int(stats["neutral"]),
                "avg_confidence": (
                    round(stats["confidence_sum"] / stats["mentions"], 2)
                    if stats["mentions"]
                    else 0.0
                ),
            }
            for feature, stats in feature_stats.items()
            if stats["mentions"]
        },
        "strengths": strengths[:4],
    }

    return {
        "insights": insights,
        "top_issues": top_issues[:5],
        "recommendations": recommendations[:4],
        "impact_simulation": {
            "current_rating": current_rating,
            "predicted_rating": predicted_rating,
            "improvement_pct": improvement_pct,
            "sentiment_shift": {"before": before_sentiment, "after": after_sentiment},
            "complaint_reduction": {
                feature: round(
                    before_complaints.get(feature, 0.0)
                    - after_complaints.get(feature, 0.0),
                    1,
                )
                for feature in before_complaints
            },
        },
        "before_vs_after": {
            "before": {
                "key_complaints": [
                    {
                        "feature": item["feature"],
                        "impact_score": item["impact_score"],
                        "examples": item["examples"],
                    }
                    for item in top_issues[:4]
                ],
                "sentiment_distribution": before_sentiment,
            },
            "after": {
                "expected_sentiment_distribution": after_sentiment,
                "reduced_complaints": after_complaints,
            },
        },
    }


def analyze_mvp_reviews(raw_reviews: List[str]) -> Dict:
    per_review = [analyse_review_precise(text) for text in raw_reviews]
    return _aggregate_precise_reviews(per_review)
