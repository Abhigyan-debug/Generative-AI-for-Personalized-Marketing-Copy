from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ProductRecommender:

    def __init__(self, catalog_df: pd.DataFrame):
        if catalog_df.empty:
            raise ValueError("Product catalog is empty - nothing to recommend from.")

        self.catalog_df = catalog_df.reset_index(drop=True)
        catalog_text = (self.catalog_df["category"] + " " + self.catalog_df["product_name"]).str.lower()

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.catalog_vectors = self.vectorizer.fit_transform(catalog_text)

    def recommend(self, purchased_products: List[str], favorite_category: str, top_n: int = 1) -> List[str]:
        profile_text = f"{favorite_category} " + " ".join(purchased_products)
        profile_vector = self.vectorizer.transform([profile_text.lower()])

        similarity_scores = cosine_similarity(profile_vector, self.catalog_vectors)[0]
        ranked_indices = similarity_scores.argsort()[::-1]

        already_bought = set(purchased_products)
        recommendations = []
        for idx in ranked_indices:
            product_name = self.catalog_df.loc[idx, "product_name"]
            if product_name in already_bought:
                continue
            recommendations.append(product_name)
            if len(recommendations) >= top_n:
                break

        if not recommendations:
            recommendations = [self.catalog_df.loc[ranked_indices[0], "product_name"]]

        return recommendations
