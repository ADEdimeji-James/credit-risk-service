"""Credit-bureau features: one row of aggregates per applicant, left-joined to the application.

bureau.csv has one row per credit the applicant held with OTHER lenders, as reported to the
credit bureau (1.7M rows for ~306k applicants). The model needs one row per applicant, so
the credits are summarised per SK_ID_CURR.

Leakage:
- Rows dated after the application (DAYS_CREDIT > 0 or DAYS_CREDIT_UPDATE > 0) are removed:
  at decision time the lender could not have known them (EDA finding 10, 17 rows).
- Every aggregate uses only that applicant's own rows, so nothing is learned across
  applicants. Aggregating the whole file once (hold-out applicants included) is therefore
  safe: no hold-out row influences any training applicant's features.

Serving (Stage 8): the lending app sends these aggregates in the /predict request, already
computed from its own bureau report. aggregate_bureau() is the definition of that contract.
"""

import logging
from pathlib import Path

import pandas as pd

from credit_risk.features import safe_ratio

logger = logging.getLogger(__name__)

# Only the columns used below, with compact dtypes (the full file is ~480 MB as strings).
# The int columns have no missing values in the Kaggle file (checked in Stage 5).
BUREAU_DTYPES = {
    "SK_ID_CURR": "int64",
    "CREDIT_ACTIVE": "category",       # Closed / Active / Sold / Bad debt
    "DAYS_CREDIT": "int32",            # days before the application the credit was opened (<= 0)
    "CREDIT_DAY_OVERDUE": "int32",     # days overdue right now
    "CNT_CREDIT_PROLONG": "int32",     # times the credit was extended
    "AMT_CREDIT_SUM": "float64",       # credit amount
    "AMT_CREDIT_SUM_DEBT": "float64",  # outstanding debt (15% missing, a few negative)
    "AMT_CREDIT_SUM_OVERDUE": "float64",
    "DAYS_CREDIT_UPDATE": "int32",     # days before the application the record was last updated
}

RECENT_DAYS = 365  # "recent" credit = opened in the year before the application

# Counts and flags: an applicant with no bureau record really has 0 of these.
COUNT_COLS = [
    "BUREAU_N_CREDITS",
    "BUREAU_N_ACTIVE",
    "BUREAU_N_SOLD_OR_BAD",
    "BUREAU_N_CREDITS_LAST_YEAR",
    "BUREAU_HAS_OVERDUE",
    "BUREAU_N_PROLONGED",
]
# Amounts, ratios and recency stay NaN when there is no record: the value is unknown,
# not zero. HAS_BUREAU_RECORD carries the "no record" signal (EDA: 10.1% vs 7.7% default).


def load_bureau(path: Path) -> pd.DataFrame:
    """Read the bureau columns we use, with compact dtypes."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Download the Home Credit data into data/raw/.")
    df = pd.read_csv(path, usecols=list(BUREAU_DTYPES), dtype=BUREAU_DTYPES)
    logger.info("Loaded %s: %d rows", path.name, len(df))
    return df


def aggregate_bureau(bureau: pd.DataFrame, id_col: str = "SK_ID_CURR") -> pd.DataFrame:
    """One row per applicant (index = id_col) with the bureau aggregates."""
    known_at_application = bureau["DAYS_CREDIT"].le(0) & bureau["DAYS_CREDIT_UPDATE"].le(0)
    b = bureau.loc[known_at_application]
    logger.info("Bureau rows dated after the application removed: %d", int((~known_at_application).sum()))

    is_active = b["CREDIT_ACTIVE"].eq("Active")
    # Per-credit helper columns, then ONE groupby (fast, and each line is easy to defend).
    rows = pd.DataFrame(
        {
            id_col: b[id_col],
            "is_active": is_active.astype("int8"),
            "is_sold_or_bad": b["CREDIT_ACTIVE"].isin(["Sold", "Bad debt"]).astype("int8"),
            "is_recent": b["DAYS_CREDIT"].ge(-RECENT_DAYS).astype("int8"),
            "is_overdue": b["CREDIT_DAY_OVERDUE"].gt(0).astype("int8"),
            "days_overdue": b["CREDIT_DAY_OVERDUE"],
            "amt_overdue": b["AMT_CREDIT_SUM_OVERDUE"],
            # Exposure that still matters = active credits only. Negative debt (overpayment)
            # is clipped to 0. Missing debt on an active credit counts as 0 in the sum.
            "active_credit": b["AMT_CREDIT_SUM"].where(is_active, 0.0),
            "active_debt": b["AMT_CREDIT_SUM_DEBT"].clip(lower=0).where(is_active, 0.0),
            "days_credit": b["DAYS_CREDIT"],
            "prolonged": b["CNT_CREDIT_PROLONG"],
        }
    )
    agg = rows.groupby(id_col).agg(
        BUREAU_N_CREDITS=("is_active", "size"),
        BUREAU_N_ACTIVE=("is_active", "sum"),
        BUREAU_N_SOLD_OR_BAD=("is_sold_or_bad", "sum"),
        BUREAU_N_CREDITS_LAST_YEAR=("is_recent", "sum"),
        BUREAU_HAS_OVERDUE=("is_overdue", "max"),
        BUREAU_MAX_DAYS_OVERDUE=("days_overdue", "max"),
        BUREAU_OVERDUE_SUM=("amt_overdue", "sum"),
        BUREAU_ACTIVE_CREDIT_SUM=("active_credit", "sum"),
        BUREAU_ACTIVE_DEBT_SUM=("active_debt", "sum"),
        BUREAU_N_PROLONGED=("prolonged", "sum"),
        latest_days_credit=("days_credit", "max"),  # closest to 0 = most recent
    )
    agg["BUREAU_ACTIVE_DEBT_TO_CREDIT"] = safe_ratio(
        agg["BUREAU_ACTIVE_DEBT_SUM"], agg["BUREAU_ACTIVE_CREDIT_SUM"]
    )
    agg["BUREAU_DAYS_SINCE_LAST_CREDIT"] = -agg.pop("latest_days_credit")
    return agg.astype("float64")


def add_bureau_features(df: pd.DataFrame, bureau_agg: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Left-join the aggregates; applicants with no bureau record keep every row."""
    out = df.merge(bureau_agg, how="left", left_on=id_col, right_index=True, validate="one_to_one")
    has_record = out["BUREAU_N_CREDITS"].notna()
    extra = pd.DataFrame(
        {
            "HAS_BUREAU_RECORD": has_record.astype("int8"),
            # Bureau debt relative to declared income (row-wise, so no leakage).
            "BUREAU_DEBT_TO_INCOME": safe_ratio(out["BUREAU_ACTIVE_DEBT_SUM"], out["AMT_INCOME_TOTAL"]),
        },
        index=out.index,
    )
    out[COUNT_COLS] = out[COUNT_COLS].fillna(0.0)
    logger.info("Applicants with a bureau record: %.1f%%", 100 * has_record.mean())
    return pd.concat([out, extra], axis=1)
