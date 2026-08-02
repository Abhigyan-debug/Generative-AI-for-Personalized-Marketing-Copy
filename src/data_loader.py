from pathlib import Path
from typing import Union

import pandas as pd

CustomerColumns = {"customer_id", "first_name", "last_name", "age", "gender", "city"}
PurchaseColumns = {"order_id", "customer_id", "product_name", "category", "price", "purchase_date"}


def load_customer_data(source: Union[str, Path, "pd.io.common.BaseBuffer"]) -> pd.DataFrame:
    try:
        customers_df = pd.read_csv(source)
    except FileNotFoundError:
        raise FileNotFoundError(f"Couldn't find the customer data file at: {source}")
    except pd.errors.EmptyDataError:
        raise ValueError("The customer data file is empty.")
    except pd.errors.ParserError as exc:
        raise ValueError(f"Customer CSV doesn't look like valid CSV: {exc}")

    missing_cols = CustomerColumns - set(customers_df.columns)
    if missing_cols:
        raise ValueError(f"Customer data is missing required columns: {sorted(missing_cols)}")

    if customers_df.empty:
        raise ValueError("Customer data has no rows.")

    customers_df["customer_id"] = customers_df["customer_id"].astype(str).str.strip()
    customers_df["first_name"] = customers_df["first_name"].astype(str).str.strip()
    customers_df["last_name"] = customers_df["last_name"].astype(str).str.strip()

    duplicate_ids = customers_df["customer_id"][customers_df["customer_id"].duplicated()]
    if not duplicate_ids.empty:
        raise ValueError(f"Duplicate customer_id values found: {duplicate_ids.unique().tolist()}")

    return customers_df


def load_purchase_data(source: Union[str, Path, "pd.io.common.BaseBuffer"]) -> pd.DataFrame:
    try:
        purchases_df = pd.read_csv(source)
    except FileNotFoundError:
        raise FileNotFoundError(f"Couldn't find the purchase history file at: {source}")
    except pd.errors.EmptyDataError:
        raise ValueError("The purchase history file is empty.")
    except pd.errors.ParserError as exc:
        raise ValueError(f"Purchase history CSV doesn't look like valid CSV: {exc}")

    missing_cols = PurchaseColumns - set(purchases_df.columns)
    if missing_cols:
        raise ValueError(f"Purchase history is missing required columns: {sorted(missing_cols)}")

    if purchases_df.empty:
        raise ValueError("Purchase history has no rows.")

    purchases_df["customer_id"] = purchases_df["customer_id"].astype(str).str.strip()
    purchases_df["purchase_date"] = pd.to_datetime(purchases_df["purchase_date"], errors="coerce")

    bad_dates = purchases_df["purchase_date"].isna().sum()
    if bad_dates:
        print(f"Warning: {bad_dates} purchase record(s) had an unparseable date and will be dropped.")
        purchases_df = purchases_df.dropna(subset=["purchase_date"])

    invalid_prices = (purchases_df["price"] <= 0).sum()
    if invalid_prices:
        print(f"Warning: {invalid_prices} purchase record(s) had a non-positive price and were dropped.")
        purchases_df = purchases_df[purchases_df["price"] > 0]

    return purchases_df


def load_product_catalog(source: Union[str, Path]) -> pd.DataFrame:
    try:
        catalog_df = pd.read_csv(source)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=["product_name", "category", "price"])

    required = {"product_name", "category", "price"}
    if not required.issubset(catalog_df.columns):
        raise ValueError(f"Product catalog is missing required columns: {sorted(required - set(catalog_df.columns))}")

    return catalog_df
