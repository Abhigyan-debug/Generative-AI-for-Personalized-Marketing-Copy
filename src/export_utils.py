from datetime import datetime
from pathlib import Path
from typing import List, Union

import pandas as pd

from src.config import OUTPUTS_DIR

CSV_COLUMNS = [
    "customer_id", "first_name", "city", "style",
    "subject", "greeting", "body", "recommendation", "offer", "cta",
    "word_count", "readability_score", "sentiment_label", "sentiment_polarity",
    "personalization_score",
]


def records_to_dataframe(records):
    if not records:
        return pd.DataFrame(columns=CSV_COLUMNS)

    return pd.DataFrame(records)[CSV_COLUMNS]


def export_to_csv(records: List[dict], output_dir: Union[str, Path] = OUTPUTS_DIR, filename: str = None) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_emails_{timestamp}.csv"

    output_path = output_dir / filename
    df = records_to_dataframe(records)
    df.to_csv(output_path, index=False)

    return output_path
