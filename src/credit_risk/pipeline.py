"""Build the model pipeline: preprocessing + estimator in ONE sklearn object.

Why one object:
- Leakage: in cross-validation the whole pipeline is re-fitted on each training fold, so
  medians, scaling and category lists never see the validation fold or the hold-out.
- Training/serving consistency: the API (Stage 8) loads this same fitted object, so the
  preprocessing at prediction time is exactly the preprocessing used in training.

Two preprocessors, because the two model families need different inputs:
- logistic regression: impute + indicator, log1p money, scale, one-hot (build_linear_preprocessor)
- LightGBM: numbers untouched with NaN kept, text -> integer codes (build_tree_preprocessor)

Interview-critical (rewrite from memory once): the linear ColumnTransformer below.
"""

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from credit_risk.features import ColumnGroups


def build_linear_preprocessor(groups: ColumnGroups) -> ColumnTransformer:
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


def build_tree_preprocessor(groups: ColumnGroups) -> ColumnTransformer:
    """Preprocessing for gradient-boosted trees.

    - Numbers pass through unchanged: trees split on order, so scaling and log1p change
      nothing, and LightGBM handles NaN natively (it learns which side missing values go).
      That keeps the "missing predicts default" signal without indicator columns.
    - Text becomes integer codes (OrdinalEncoder), used by LightGBM as ordinary numbers.
      Trade-off: the codes have an arbitrary order, so a category may need several splits
      to isolate; in return there is no high-cardinality overfitting (ORGANIZATION_TYPE has
      58 values) and no fit-time arguments to route through the Pipeline.
      Missing text and categories unseen in training both map to -1, so the API never
      crashes on a new category.
    - set_output("pandas") keeps column names, so LightGBM, feature importance and SHAP
      (Stage 6) report real feature names.
    """
    categorical = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", groups.money + groups.numeric),
            ("categorical", categorical, groups.categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,  # groups are disjoint, so plain names stay unique
    )
    return preprocessor.set_output(transform="pandas")


def build_pipeline(
    model_name: str, params: dict[str, Any], groups: ColumnGroups, seed: int
) -> Pipeline:
    """Preprocessor + estimator for the model named in the config."""
    params = {**params, "random_state": seed}  # one seed for everything, from the config
    if model_name == "logistic_regression":
        preprocessor = build_linear_preprocessor(groups)
        estimator = LogisticRegression(**params)
    elif model_name == "lightgbm":
        # Imported here so code that only uses logistic regression does not need lightgbm.
        from lightgbm import LGBMClassifier

        preprocessor = build_tree_preprocessor(groups)
        estimator = LGBMClassifier(**params)
    else:
        raise ValueError(f"unknown model.name {model_name!r}")
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
