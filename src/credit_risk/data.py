"""Load application data and manage the fixed, stratified hold-out set.

The hold-out (test) set is chosen once, saved as a list of applicant IDs, and reused by
every later run. Saving IDs instead of rows:
- keeps one source of truth (the raw Kaggle file) and no second copy of licensed data;
- makes the split permanent: a seed only repeats a split with the same library version,
  a saved file repeats it with any version.

This module only loads and splits. Feature engineering lives in features.py. etc
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk.config import Config

logger = logging.getLogger(__name__)


def load_applications(path: Path) -> pd.DataFrame:
    """Read application_train.csv. pandas 3 loads text columns with the ``str`` dtype."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Download the Home Credit competition data into data/raw/ "
            "(see README)."
        )
    df = pd.read_csv(path)
    logger.info("Loaded %s: %d rows x %d columns", path.name, df.shape[0], df.shape[1])
    return df


def create_holdout_ids(df: pd.DataFrame, config: Config) -> pd.Series:
    """Pick the hold-out IDs: stratified on the target, seeded, sized by the config."""
    _, holdout_ids = train_test_split(
        df[config.data.id_col],
        test_size=config.split.holdout_size,
        stratify=df[config.data.target_col],  # same default rate (~8.07%) on both sides
        random_state=config.seed,
    )
    return holdout_ids.sort_values().reset_index(drop=True)


def _validate_holdout_ids(df: pd.DataFrame, holdout_ids: pd.Series, config: Config) -> None:
    """Fail loudly if the saved hold-out no longer matches the data or the config."""
    id_col = config.data.id_col
    if holdout_ids.duplicated().any():
        raise ValueError("saved hold-out file contains duplicate IDs")
    unknown = int((~holdout_ids.isin(df[id_col])).sum())
    if unknown:
        raise ValueError(f"{unknown} saved hold-out IDs are not in the data; wrong data file?")
    expected = len(df) * config.split.holdout_size
    if abs(len(holdout_ids) - expected) > 1:
        raise ValueError(
            f"saved hold-out has {len(holdout_ids)} IDs but split.holdout_size="
            f"{config.split.holdout_size} implies ~{expected:.0f}. Re-splitting changes the "
            f"test set: only delete {config.paths.holdout_ids_path.name} on purpose."
        )


def load_or_create_holdout_ids(df: pd.DataFrame, config: Config) -> pd.Series:
    """Return the saved hold-out IDs, creating and saving them on the very first run."""
    path = config.paths.holdout_ids_path
    id_col = config.data.id_col
    if path.exists():
        holdout_ids = pd.read_csv(path)[id_col]
        _validate_holdout_ids(df, holdout_ids, config)
        logger.info("Using saved hold-out: %d IDs from %s", len(holdout_ids), path)
        return holdout_ids

    holdout_ids = create_holdout_ids(df, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    holdout_ids.to_frame(id_col).to_csv(path, index=False)
    logger.info("Created hold-out: %d IDs saved to %s", len(holdout_ids), path)
    return holdout_ids


def split_train_holdout(
    df: pd.DataFrame, holdout_ids: pd.Series, id_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into (training part, hold-out part) using the saved IDs."""
    in_holdout = df[id_col].isin(holdout_ids)
    train_df = df.loc[~in_holdout].reset_index(drop=True)
    holdout_df = df.loc[in_holdout].reset_index(drop=True)
    return train_df, holdout_df
