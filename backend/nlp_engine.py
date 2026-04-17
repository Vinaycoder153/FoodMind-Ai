"""
NLP Engine for FoodMind AI
Keyword-based feature extraction + sentiment classification
Supports English, Hindi, and Kannada via keyword mapping
"""

import re
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Multilingual keyword maps
# ---------------------------------------------------------------------------

# TRANSLATION LAYER – maps Hindi/Kannada words → English concept tokens
MULTILINGUAL_MAP: Dict[str, str] = {
    # Hindi – oil
    "तेल": "oil", "tel": "oil",
    # Hindi – spice / chilli
    "मसाला": "spice", "masala": "spice", "तीखा": "spicy", "teekha": "spicy",
    "mirchi": "spice", "मिर्ची": "spice",
    # Hindi – quantity / portion
    "कम": "less", "zyada": "more", "ज़्यादा": "more", "quantity": "quantity",
    # Hindi – taste
    "स्वाद": "taste", "swad": "taste", "tasty": "taste",
    # Hindi – good / bad sentiment helpers
    "अच्छा": "good", "accha": "good", "acha": "good",
    "बुरा": "bad", "bura": "bad", "kharab": "bad", "खराब": "bad",
    # Kannada – oil
    "ಎಣ್ಣೆ": "oil", "enne": "oil",
    # Kannada – spice
    "ಮಸಾಲೆ": "spice", "masale": "spice", "ಖಾರ": "spicy", "khaara": "spicy",
    # Kannada – quantity
    "ಕಡಿಮೆ": "less", "kadime": "less", "ಜಾಸ್ತಿ": "more", "jaasti": "more",
    # Kannada – taste
    "ರುಚಿ": "taste", "ruchi": "taste",
    # Kannada – good / bad
    "ಚೆನ್ನಾಗಿದೆ": "good", "chennagide": "good",
    "ಕೆಟ್ಟದಾಗಿದೆ": "bad", "kettadagide": "bad",
}

# FEATURE KEYWORD SETS (after multilingual normalisation)
FEATURE_KEYWORDS: Dict[str, List[str]] = {
    "oil": [
        "oil", "oily", "greasy", "grease", "fatty", "fat", "dripping",
        "ghee", "butter",
    ],
    "spice": [
        "spice", "spicy", "spices", "spiced", "chili", "chilli", "pepper",
        "hot", "mild", "bland", "masala", "teekha", "khaara",
    ],
    "quantity": [
        "quantity", "portion", "amount", "serving", "size", "small",
        "large", "big", "less", "more", "half", "full", "insufficient",
    ],
    "taste": [
        "taste", "flavor", "flavour", "delicious", "yummy", "tasty",
        "bland", "good", "bad", "awful", "amazing", "terrible", "swad",
        "ruchi",
    ],
}

# SENTIMENT KEYWORD SETS
POSITIVE_WORDS: List[str] = [
    "good", "great", "excellent", "amazing", "awesome", "perfect", "loved",
    "love", "fantastic", "wonderful", "superb", "delicious", "yummy",
    "tasty", "fresh", "clean", "happy", "satisfied", "nice", "best",
    "accha", "acha", "chennagide", "liked", "like", "enjoyable", "enjoy",
    "recommend", "outstanding", "brilliant",
]

NEGATIVE_WORDS: List[str] = [
    "bad", "terrible", "awful", "horrible", "worst", "hate", "hated",
    "poor", "disappointing", "disgusting", "greasy", "oily", "too much",
    "too little", "bland", "stale", "cold", "overcooked", "undercooked",
    "complaint", "unhappy", "unhygienic", "bura", "kharab", "kettadagide",
    "not good", "not great", "not fresh", "rotten",
]

# NEGATION WORDS (flip sentiment)
NEGATION_WORDS: List[str] = [
    "not", "no", "never", "neither", "nope", "nah", "without", "nahi",
    "mat", "nahin",
]

# FEATURE-SPECIFIC COMPLAINT PATTERNS
COMPLAINT_PATTERNS: Dict[str, List[str]] = {
    "oil": ["too oily", "very oily", "excess oil", "too much oil", "dripping oil",
            "greasy", "too greasy", "lots of oil", "reduce oil"],
    "spice": ["too spicy", "very spicy", "too hot", "too mild", "not spicy",
              "bland", "no spice", "less spice", "more spice", "too much spice"],
    "quantity": ["small portion", "less quantity", "small serving", "not enough",
                 "too little", "insufficient", "half portion", "small amount"],
    "taste": ["bad taste", "no taste", "tasteless", "awful taste", "terrible taste",
              "not tasty", "not good", "poor taste", "horrible taste", "bland taste"],
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
            context = tokens[idx: idx + 5]
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
            complaint_pct[feature] = round(
                counts["negative"] / total * 100, 1
            )
        else:
            complaint_pct[feature] = 0.0

    # Top complaint
    top_complaint_feature = max(complaint_pct, key=lambda f: complaint_pct[f])
    top_complaint_pct = complaint_pct[top_complaint_feature]

    # Alert if top complaint > 25%
    alerts = []
    for feature, pct in complaint_pct.items():
        if pct >= 25:
            alerts.append({
                "feature": feature,
                "percentage": pct,
                "message": f"⚠️ {feature.capitalize()} complaints are at {pct}% — needs immediate action!",
                "level": "critical" if pct >= 40 else "warning",
            })

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
            action = "Increase portion size by 10% or offer an 'Extra Portion' add-on (+₹30)"
            detail = "Standardise serving scoops to ensure consistency"
        elif feature == "taste":
            action = "Revise base recipe — conduct a tasting panel with 5 focus customers"
            detail = "Review seasoning ratios; consider adding a secret spice blend"
        else:
            action = f"Address {feature} quality concerns"
            detail = "Review preparation process for this attribute"

        recs.append({
            "feature": feature,
            "action": action,
            "detail": detail,
            "confidence": confidence,
            "based_on": neg_count,
            "reasoning": (
                f"{neg_count} out of {total} reviews ({pct}%) mention "
                f"{feature} negatively. Confidence driven by review volume."
            ),
        })

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
        options.append({
            "name": "Light Oil Version",
            "description": "Prepared with 50% less oil using air-fry technique",
            "price_delta": "+₹0 (free upgrade)",
            "tag": "Healthy Choice 🥗",
        })

    if complaint_pct.get("spice", 0) >= 10:
        options.append({
            "name": "Custom Spice Level",
            "description": "Choose: Mild 🟢 / Medium 🟡 / Extra Hot 🔴",
            "price_delta": "+₹0 (free customisation)",
            "tag": "Personalised 🌶️",
        })

    if complaint_pct.get("quantity", 0) >= 10:
        options.append({
            "name": "Extra Protein Booster",
            "description": "Add extra egg / paneer / chicken to any dish",
            "price_delta": "+₹20 – ₹40",
            "tag": "Filling & Satisfying 💪",
        })

    if not options:
        # Always offer at least one default customisation
        options.append({
            "name": "Chef's Special Version",
            "description": "Freshly prepared with chef's premium seasoning blend",
            "price_delta": "+₹15",
            "tag": "Premium ⭐",
        })

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
