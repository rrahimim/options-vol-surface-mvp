from __future__ import annotations

from pathlib import Path
import pandas as pd

# Project root = .../options-vol-surface-mvp/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "spy_2020_2022.csv"

# Minimal columns needed for: calls, IV, smile, surface
REQUIRED_COLS = [
    "QUOTE_DATE",
    "EXPIRE_DATE",
    "UNDERLYING_LAST",
    "STRIKE",
    "DTE",
    "C_BID",
    "C_ASK",
    "C_LAST",
    "P_BID",
    "P_ASK",
    "P_LAST",
]

NUM_COLS = [
    "UNDERLYING_LAST",
    "STRIKE",
    "DTE",
    "C_BID",
    "C_ASK",
    "C_LAST",
    "P_BID",
    "P_ASK",
    "P_LAST",
]


def load_options_data(path: str | Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """
    Load and clean options CSV.

    - Works whether you run from notebooks/ or project root
    - Fixes column names like "[QUOTE_DATE]" -> "QUOTE_DATE"
    - Strips whitespace from dates
    - Converts numeric columns safely
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"CSV not found at: {path}")

    df = pd.read_csv(path, low_memory=False)

    # Clean column names: "[QUOTE_DATE]" -> "QUOTE_DATE"
    df.columns = df.columns.str.strip().str.replace(r"^\[|\]$", "", regex=True)

    # Keep only required columns that exist
    keep = [c for c in REQUIRED_COLS if c in df.columns]
    df = df[keep].copy()

    # Strip whitespace in date columns (IMPORTANT)
    df["QUOTE_DATE"] = df["QUOTE_DATE"].astype(str).str.strip()
    df["EXPIRE_DATE"] = df["EXPIRE_DATE"].astype(str).str.strip()

    # Convert numerics
    for c in [c for c in NUM_COLS if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop missing critical values
    df = df.dropna(
        subset=["QUOTE_DATE", "EXPIRE_DATE", "UNDERLYING_LAST", "STRIKE", "DTE"]
    )

    return df


def most_liquid_quote_date(df: pd.DataFrame) -> str:
    """Quote date with the most rows (fast way to pick a good day)."""
    return df["QUOTE_DATE"].value_counts().idxmax()


def calls_for_date(
    df: pd.DataFrame,
    quote_date: str,
    use_last_price: bool = True,
    r: float = 0.01,
    q: float = 0.0,
) -> pd.DataFrame:
    """
    Return calls on a given quote date with standardized columns needed for IV & surface:

    Output columns include:
    - price (used for IV)
    - S, K, T, r, q
    - QUOTE_DATE, EXPIRE_DATE
    """
    d = df[df["QUOTE_DATE"] == str(quote_date).strip()].copy()

    if d.empty:
        raise ValueError(f"No rows found for QUOTE_DATE={quote_date!r}")

    if use_last_price:
        d = d[d["C_LAST"] > 0].copy()
        d["price"] = d["C_LAST"]
    else:
        d = d[(d["C_BID"] > 0) & (d["C_ASK"] > 0)].copy()
        d["price"] = 0.5 * (d["C_BID"] + d["C_ASK"])

    # Standardize fields
    d["S"] = d["UNDERLYING_LAST"]
    d["K"] = d["STRIKE"]
    d["T"] = d["DTE"] / 365.0
    d["r"] = float(r)
    d["q"] = float(q)
    d["option_type"] = "call"

    # Valid maturities only
    d = d[d["T"] > 0].copy()

    if d.empty:
        raise ValueError(
            "After filtering (positive prices and T>0), no call quotes remain."
        )

    return d

