from typing import Callable, List, Optional

import pandas as pd

from src import evaluation
from src.email_generator import EmailGenerator, generate_email_for_customer
from src.preprocessing import build_customer_profiles
from src.prompt_builder import build_prompt
from src.recommender import ProductRecommender


def generate_campaign(
    customers_df: pd.DataFrame,
    purchases_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    style: str,
    generator: EmailGenerator,
    limit: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[dict]:
    profiles_df = build_customer_profiles(customers_df, purchases_df)
    recommender = ProductRecommender(catalog_df)

    if limit is not None:
        profiles_df = profiles_df.head(limit)

    total = len(profiles_df)
    records = []

    for position, (_, customer_row) in enumerate(profiles_df.iterrows(), start=1):
        prompt = build_prompt(customer_row, style, recommender)
        sections = generate_email_for_customer(generator, prompt)
        full_text = " ".join(sections.values())
        sentiment = evaluation.sentiment_score(full_text)

        records.append({
            "customer_id": customer_row["customer_id"],
            "first_name": customer_row["first_name"],
            "city": customer_row["city"],
            "style": style,
            "subject": sections["SUBJECT"],
            "greeting": sections["GREETING"],
            "body": sections["BODY"],
            "recommendation": sections["RECOMMENDATION"],
            "offer": sections["OFFER"],
            "cta": sections["CTA"],
            "word_count": evaluation.word_count(full_text),
            "readability_score": evaluation.readability_score(full_text),
            "sentiment_label": sentiment["label"],
            "sentiment_polarity": sentiment["polarity"],
            "personalization_score": evaluation.personalization_score(sections, customer_row),
        })

        if progress_callback is not None:
            progress_callback(position, total)

    return records
