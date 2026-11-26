"""Row-wise feature engineering and column selection (nothing here learns from data).

Every transform in add_features uses only the values in one row, so it cannot leak
information between rows, folds or the hold-out. Anything that learns from data
(medians, scaling, category lists) lives in the sklearn Pipeline (pipeline.py) and is
re-fitted inside every CV fold.
"""

from dataclasses import asdict, dataclass

import pandas as pd

from credit_risk.config import Config

# DAYS_EMPLOYED uses 365243 (about 1,000 years) as a code for "not employed"
# (EDA finding 3: 18% of rows, almost all pensioners). It is a fact about this dataset,
# not a tunable setting, so it is a named constant rather than a config value.
EMPLOYED_PLACEHOLDER_DAYS = 365243
DAYS_PER_YEAR = 365.25
HOUSING_SUFFIXES = ("_AVG", "_MODE", "_MEDI")


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """numerator / denominator, with NaN (not inf) where the denominator is 0 or missing."""
    return numerator / denominator.where(denominator != 0)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered columns decided in the EDA. Returns a new DataFrame."""
    is_placeholder = df["DAYS_EMPLOYED"].eq(EMPLOYED_PLACEHOLDER_DAYS)
    housing_cols = [c for c in df.columns if c.endswith(HOUSING_SUFFIXES)]

    new = pd.DataFrame(index=df.index)
    # Employment (finding 3): flag the placeholder, then express real values in years.
    new["EMPLOYED_PLACEHOLDER"] = is_placeholder.astype("int8")
    new["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"].mask(is_placeholder) / DAYS_PER_YEAR
    # Missingness carries signal (finding 2): applicants with no housing data default more.
    new["HAS_HOUSING_DATA"] = df[housing_cols].notna().any(axis=1).astype("int8")
    # Ratios (finding 5). Computed before AMT_GOODS_PRICE is dropped as redundant.
    new["CREDIT_TO_INCOME"] = safe_ratio(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    new["ANNUITY_TO_INCOME"] = safe_ratio(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    new["CREDIT_TO_GOODS"] = safe_ratio(df["AMT_CREDIT"], df["AMT_GOODS_PRICE"])

    # One concat instead of many single-column inserts (avoids pandas fragmentation).
    return pd.concat([df, new], axis=1)


def select_model_inputs(df: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.Series]:
    """Split into model inputs X and target y.

    Drops the ID, the target, protected attributes (kept elsewhere for the fairness
    audit) and the redundant columns listed in the config.
    """
    to_drop = [
        config.data.id_col,
        config.data.target_col,
        *config.features.protected_cols,
        *config.features.redundant_cols,
    ]
    missing = sorted(set(to_drop) - set(df.columns))
    if missing:
        raise KeyError(f"columns named in the config are not in the data: {missing}")
    X = df.drop(columns=to_drop)
    y = df[config.data.target_col].astype("int8")
    return X, y


@dataclass(frozen=True)
class ColumnGroups:
    """Explicit column lists for the ColumnTransformer, resolved once and logged."""

    money: list[str]
    numeric: list[str]
    categorical: list[str]

    def as_dict(self) -> dict[str, list[str]]:
        return asdict(self)


def resolve_column_groups(X: pd.DataFrame, config: Config) -> ColumnGroups:
    """Decide which columns get which preprocessing.

    Numbers vs text is decided with include/exclude="number", which works the same in
    pandas 2, 3 and 4 (unlike dtype=object, see tests/test_env_smoke.py). The resulting
    explicit lists are what the pipeline uses and what gets logged to MLflow.
    """
    money = list(config.features.money_cols)
    if config.bureau.enabled:
        money += config.bureau.money_cols
    missing = sorted(set(money) - set(X.columns))
    if missing:
        raise KeyError(f"features.money_cols not in the data: {missing}")

    numeric = [c for c in X.select_dtypes(include="number").columns if c not in money]
    categorical = list(X.select_dtypes(exclude="number").columns)

    covered = len(money) + len(numeric) + len(categorical)
    if covered != X.shape[1]:
        raise ValueError(f"column groups cover {covered} of {X.shape[1]} columns")
    return ColumnGroups(money=money, numeric=numeric, categorical=categorical)
