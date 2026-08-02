import json
import random

import pandas as pd

from src.config import BASE_PROMPT_TEMPLATE, BRAND_NAME, STYLE_GUIDES_JSON, VALID_STYLES
from src.recommender import ProductRecommender

OFFERS_BY_TIER = {
    "Bronze": ["5% off your next order", "free shipping on your next purchase"],
    "Silver": ["10% off your next order", "a free gift with your next purchase"],
    "Gold": ["15% off + free shipping", "early access to our next sale"],
    "Platinum": ["20% off + free shipping", "an exclusive early-access discount code"],
}


def load_style_guides():
    with open(STYLE_GUIDES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt_template():
    with open(BASE_PROMPT_TEMPLATE, "r", encoding="utf-8") as f:
        return f.read()


def pick_offer(loyalty_tier):
    options = OFFERS_BY_TIER.get(loyalty_tier, ["10% off your next order"])
    return random.choice(options)


def build_prompt(customer_row: pd.Series, style: str, recommender: ProductRecommender) -> str:
    style = style.lower().strip()
    if style not in VALID_STYLES:
        raise ValueError(f"Unknown style '{style}'. Pick one of: {VALID_STYLES}")

    style_guides = load_style_guides()
    style_info = style_guides[style]

    recommended = recommender.recommend(
        purchased_products=customer_row["purchased_products"],
        favorite_category=customer_row["favorite_category"],
        top_n=1,
    )[0]

    offer = pick_offer(customer_row["loyalty_tier"])
    template = load_prompt_template()

    return template.format(
        brand_name=BRAND_NAME,
        first_name=customer_row["first_name"],
        age_group=customer_row["age_group"],
        city=customer_row["city"],
        favorite_category=customer_row["favorite_category"],
        last_product=customer_row["last_product"],
        total_orders=int(customer_row["total_orders"]),
        loyalty_tier=customer_row["loyalty_tier"],
        style_label=style_info["label"],
        tone_description=style_info["tone_description"],
        recommended_product=recommended,
        offer_detail=offer,
    )
