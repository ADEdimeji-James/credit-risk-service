"""Build the model pipeline: preprocessing + estimator in ONE sklearn object.

Why one object:
- Leakage: in cross-validation the whole pipeline is re-fitted on each training fold, so
  medians, scaling and category lists never see the validation fold or the hold-out.
- Training/serving consistency: the API (Stage 8) loads this same fitted object, so the
  preprocessing at prediction time is exactly the preprocessing used in training.

Interview-critical (rewrite from memory once): the ColumnTransformer below.
"""

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from credit_risk.features import ColumnGroups


def build_preprocessor(groups: ColumnGroups) -> ColumnTransformer:
    """Preprocessing for a linear model (EDA findings 2, 6)."""
    # Money columns: heavy right tail (one income is 248x the 99th percentile), so log1p
    # before scaling. Missing indicator first: missingness predicts default.
    money = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    # Other numeric columns: median + missing indicator (no -999 sentinels), then scale
    # so the regularisation treats every coefficient on the same footing.
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    # Text columns: explicit "missing" category, then one-hot. handle_unknown="ignore":
    # a category never seen in training encodes as all zeros instead of crashing the API.
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("money", money, groups.money),
            ("numeric", numeric, groups.numeric),
            ("categorical", categorical, groups.categorical),
        ],
        remainder="drop",  # every input column is in a group (checked in features.py)
    )


def build_pipeline(model_name: str, params: dict[str, Any], groups: ColumnGroups) -> Pipeline:
    """Preprocessor + estimator. Only logistic regression exists in Stage 4."""
    if model_name == "logistic_regression":
        estimator = LogisticRegression(**params)
    else:
        raise ValueError(f"unknown model.name {model_name!r}")
    return Pipeline(steps=[("preprocess", build_preprocessor(groups)), ("model", estimator)])
