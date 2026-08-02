from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CUSTOMERS_CSV = DATA_DIR / "customers.csv"
PURCHASE_HISTORY_CSV = DATA_DIR / "purchase_history.csv"
PRODUCT_CATALOG_CSV = DATA_DIR / "product_catalog.csv"

STYLE_GUIDES_JSON = PROMPTS_DIR / "style_guides.json"
BASE_PROMPT_TEMPLATE = PROMPTS_DIR / "base_prompt_template.txt"

BRAND_NAME = "Everdale"

DEFAULT_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DEFAULT_MAX_NEW_TOKENS = 350

VALID_STYLES = ("professional", "friendly", "luxury", "festive")
