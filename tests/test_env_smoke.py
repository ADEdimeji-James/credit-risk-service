"""Environment smoke test: scikit-learn handles pandas 3 data (str dtype, missing values)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def test_column_transformer_handles_pandas3_strings():
    # TODO 1: build a DataFrame of ~8 rows.
    # Features (X) and target (y) are kept separate, as in the real training code,
    # so the target can never reach the transformer by accident.
    X = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
            "text": ["A", "B", "A", "C", np.nan, "B", "C", "A"],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], name="target")

    # TODO 2: assert the text column's dtype is "str"
    assert X["text"].dtype == "str", f"expected str, got {X['text'].dtype}"

    # TODO 3: the trap. Selecting text columns by dtype=object still works in pandas 3.x
    # only through a deprecated backward-compatibility path (it warns). pandas 4 will
    # return no columns, silently dropping every text feature. That is why the real
    # pipeline selects columns by explicit name lists from the config, never by dtype.
    selector = make_column_selector(dtype_include=object)
    with pytest.warns(Warning, match="'str' dtypes are included by select_dtypes"):
        selector(X)

    # TODO 4: build the ColumnTransformer with EXPLICIT column-name lists:
    #   numeric: Pipeline([SimpleImputer(strategy="median", add_indicator=True), StandardScaler()])
    #   text:    OneHotEncoder(handle_unknown="ignore")  -- does it need an imputer first? find out
    default_transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ["numeric"],
            ),
            (
                "text",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),  # dense, so np.isnan always works
                    ]
                ),
                ["text"],
            ),
        ]
    )

    # TODO 5: fit_transform; assert the column count you worked out BY HAND beforehand;
    #         assert there is no NaN in the output
    X_transformed = default_transformer.fit_transform(X)
    # expected columns: 
    # numeric: 1 (missing_indicator) + 1 (original) = 2
    # text: 3 (A, B, C) + 1 (missing) = 4
    expected_columns = 2 + 4
    assert X_transformed.shape[1] == expected_columns, f"expected {expected_columns} columns, got {X_transformed.shape[1]}"
    assert not np.isnan(X_transformed).any(), "expected no NaN values in the transformed output"

    # TODO 6: transform a one-row DataFrame with a category not seen during fit; no error
    new_row_df = pd.DataFrame({"numeric": [9.0], "text": ["D"]})
    transformed_new_row = default_transformer.transform(new_row_df)
    assert transformed_new_row.shape == (1, expected_columns)
    # Output order: [numeric value, numeric missing-indicator, text one-hot x 4].
    # An unseen category matches none of A, B, C, "missing", so its one-hot block is all zeros.
    assert (transformed_new_row[0, 2:] == 0).all(), "unseen category should encode as all zeros"

    # TODO 7: put the ColumnTransformer + LogisticRegression in a Pipeline, fit it,
    #         assert every predict_proba row sums to 1 (use np.allclose)
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", default_transformer),
            ("classifier", LogisticRegression()),
        ]
    )
    model_pipeline.fit(X, y)
    proba = model_pipeline.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1), "expected each row of predict_proba to sum to 1"