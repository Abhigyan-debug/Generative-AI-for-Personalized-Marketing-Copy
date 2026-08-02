import re

from textblob import TextBlob

VOWELS = "aeiouy"


def word_count(text: str) -> int:
    return len(text.split())


def _count_syllables(word):
    word = word.lower().strip(".,!?;:\"'")
    if not word:
        return 0

    count = 0
    prev_was_vowel = False
    for char in word:
        is_vowel = char in VOWELS
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel

    if word.endswith("e") and count > 1:
        count -= 1

    return max(count, 1)


def readability_score(text: str) -> float:
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    words = text.split()

    if not sentences or not words:
        return 0.0

    syllable_count = sum(_count_syllables(w) for w in words)

    score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllable_count / len(words))
    return round(max(0.0, min(100.0, score)), 1)


def readability_label(score):
    if score >= 80:
        return "Easy"
    if score >= 60:
        return "Fairly easy"
    if score >= 40:
        return "Moderate"
    return "Difficult"


def sentiment_score(text: str) -> dict:
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.15:
        label = "Positive"
    elif polarity < -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {"polarity": round(polarity, 3), "label": label}


def personalization_score(email_sections: dict, customer_row) -> float:
    full_text = " ".join(email_sections.values()).lower()

    checks = [
        customer_row["first_name"].lower() in full_text,
        str(customer_row["city"]).lower() in full_text,
        str(customer_row["favorite_category"]).lower() in full_text,
        customer_row["loyalty_tier"].lower() in full_text,
    ]

    matched = sum(checks)
    return round((matched / len(checks)) * 100, 1)
