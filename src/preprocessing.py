import pandas as pd


def add_age_group(customers_df: pd.DataFrame) -> pd.DataFrame:
    customers_df = customers_df.copy()

    bins = [0, 25, 35, 50, 65, 200]
    labels = ["18-25", "26-35", "36-50", "51-65", "65+"]
    customers_df["age_group"] = pd.cut(customers_df["age"], bins=bins, labels=labels, right=True)
    customers_df["age_group"] = customers_df["age_group"].astype(str)

    return customers_df


def _loyalty_tier(total_spent, total_orders):
    if total_spent >= 15000 or total_orders >= 8:
        return "Platinum"
    if total_spent >= 8000 or total_orders >= 5:
        return "Gold"
    if total_spent >= 3000 or total_orders >= 2:
        return "Silver"
    return "Bronze"


def summarize_purchase_history(purchases_df: pd.DataFrame) -> pd.DataFrame:
    purchases_df = purchases_df.sort_values("purchase_date")

    summary_rows = []
    for customer_id, group in purchases_df.groupby("customer_id"):
        total_spent = float((group["price"] * group.get("quantity", 1)).sum())
        total_orders = len(group)
        favorite_category = group["category"].mode().iloc[0]
        last_row = group.iloc[-1]

        summary_rows.append({
            "customer_id": customer_id,
            "total_orders": total_orders,
            "total_spent": round(total_spent, 2),
            "favorite_category": favorite_category,
            "last_product": last_row["product_name"],
            "last_purchase_date": last_row["purchase_date"],
            "purchased_products": group["product_name"].tolist(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df["loyalty_tier"] = summary_df.apply(
        lambda row: _loyalty_tier(row["total_spent"], row["total_orders"]), axis=1
    )
    return summary_df


def build_customer_profiles(customers_df: pd.DataFrame, purchases_df: pd.DataFrame) -> pd.DataFrame:
    customers_df = add_age_group(customers_df)
    purchase_summary = summarize_purchase_history(purchases_df)

    merged = customers_df.merge(purchase_summary, on="customer_id", how="left")

    merged["total_orders"] = merged["total_orders"].fillna(0).astype(int)
    merged["total_spent"] = merged["total_spent"].fillna(0.0)
    merged["favorite_category"] = merged["favorite_category"].fillna("General")
    merged["last_product"] = merged["last_product"].fillna("nothing yet")
    merged["loyalty_tier"] = merged["loyalty_tier"].fillna("Bronze")
    merged["purchased_products"] = merged["purchased_products"].apply(
        lambda val: val if isinstance(val, list) else []
    )

    return merged


def segment_customers(profiles_df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    profiles_df = profiles_df.copy()
    n_clusters = min(n_clusters, len(profiles_df))
    if n_clusters < 2:
        profiles_df["segment"] = 0
        return profiles_df

    features = profiles_df[["age", "total_spent", "total_orders"]].fillna(0)
    scaled_features = StandardScaler().fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    profiles_df["segment"] = kmeans.fit_predict(scaled_features)

    return profiles_df
