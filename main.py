import argparse
import sys

from src import data_loader, export_utils
from src.config import (
    CUSTOMERS_CSV,
    DEFAULT_MODEL_NAME,
    PRODUCT_CATALOG_CSV,
    PURCHASE_HISTORY_CSV,
    VALID_STYLES,
)
from src.email_generator import EmailGenerator
from src.pipeline import generate_campaign


def parse_args():
    parser = argparse.ArgumentParser(description="Generate personalized marketing emails with a local LLM.")
    parser.add_argument("--style", default="friendly", choices=VALID_STYLES, help="Writing style to use.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Hugging Face model name to load.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N customers.")
    parser.add_argument("--customers", default=str(CUSTOMERS_CSV), help="Path to customer data CSV.")
    parser.add_argument("--purchases", default=str(PURCHASE_HISTORY_CSV), help="Path to purchase history CSV.")
    parser.add_argument("--catalog", default=str(PRODUCT_CATALOG_CSV), help="Path to product catalog CSV.")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        customers_df = data_loader.load_customer_data(args.customers)
        purchases_df = data_loader.load_purchase_data(args.purchases)
        catalog_df = data_loader.load_product_catalog(args.catalog)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading data: {exc}")
        sys.exit(1)

    print(f"Loaded {len(customers_df)} customers and {len(purchases_df)} purchase records.")
    print(f"Loading model '{args.model}' - this can take a while the first time it downloads...")

    generator = EmailGenerator(model_name=args.model)

    def show_progress(done, total):
        print(f"  generated {done}/{total} emails", end="\r")

    try:
        records = generate_campaign(
            customers_df,
            purchases_df,
            catalog_df,
            style=args.style,
            generator=generator,
            limit=args.limit,
            progress_callback=show_progress,
        )
    except RuntimeError as exc:
        print(f"\nError during generation: {exc}")
        sys.exit(1)

    print()
    output_path = export_utils.export_to_csv(records)
    print(f"Done. Wrote {len(records)} emails to {output_path}")


if __name__ == "__main__":
    main()
