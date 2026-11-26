"""Train and evaluate a model from a YAML config, and log the run to MLflow.

Usage (from anywhere, with the venv active):
    python -m credit_risk.train --config configs/base.yaml
    python -m credit_risk.train --config configs/base.yaml --set model.name=lightgbm --run-name lgbm_bureau
    python -m credit_risk.train --config configs/base.yaml --set bureau.enabled=false --run-name lr_baseline

Flow:
    config -> load data -> fixed hold-out (set aside, untouched until Stage 6)
    -> row-wise features (+ bureau aggregates if enabled) -> stratified K-fold CV on the training part
    -> metrics per fold (PR-AUC, ROC-AUC, naira profit vs approve-all)
    -> refit on the whole training part -> log everything to MLflow

Interview-critical (rewrite from memory once): cross_validate below.
"""

import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from credit_risk.bureau import add_bureau_features, aggregate_bureau, load_bureau
from credit_risk.config import Config, config_to_dict, load_config, project_root_for
from credit_risk.data import load_applications, load_or_create_holdout_ids, split_train_holdout
from credit_risk.evaluate import evaluate_predictions
from credit_risk.features import add_features, resolve_column_groups, select_model_inputs
from credit_risk.pipeline import build_pipeline

logger = logging.getLogger("credit_risk.train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="path to a YAML config file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one setting, e.g. split.cv_folds=3 (repeatable)",
    )
    parser.add_argument("--run-name", default=None, help="optional MLflow run name")
    return parser.parse_args(argv)


def cross_validate(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, config: Config
) -> list[dict[str, float]]:
    """Stratified K-fold CV. A fresh copy of the pipeline is fitted on each training fold,
    so every learned value (medians, scales, categories, coefficients) comes from that
    fold only, and the validation fold is scored as unseen data."""
    folds = StratifiedKFold(n_splits=config.split.cv_folds, shuffle=True, random_state=config.seed)
    results = []
    for fold, (train_idx, valid_idx) in enumerate(folds.split(X, y), start=1):
        start = time.perf_counter()
        model = clone(pipeline)  # unfitted copy: nothing carries over between folds
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        p_default = model.predict_proba(X.iloc[valid_idx])[:, 1]  # column 1 = P(TARGET=1)
        metrics = evaluate_predictions(
            y.iloc[valid_idx].to_numpy(), p_default, config.costs.threshold, config.costs
        )
        metrics["fit_seconds"] = time.perf_counter() - start
        results.append(metrics)
        logger.info(
            f"fold {fold}/{config.split.cv_folds}  PR-AUC {metrics['pr_auc']:.4f}  "
            f"ROC-AUC {metrics['roc_auc']:.4f}  profit/app {metrics['expected_profit']:,.0f} "
            f"(approve-all {metrics['approve_all_profit']:,.0f})  {metrics['fit_seconds']:.0f}s"
        )
    return results


def summarise(fold_results: list[dict[str, float]]) -> dict[str, float]:
    """Mean and standard deviation of each metric across folds."""
    summary = {}
    for name in fold_results[0]:
        values = np.array([r[name] for r in fold_results], dtype=np.float64)
        summary[f"cv_{name}_mean"] = float(values.mean())
        summary[f"cv_{name}_std"] = float(values.std(ddof=1))
    return summary


def _git_commit(config_path: str) -> str:
    """Current git commit, so every run can be traced back to the exact code."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root_for(Path(config_path)),
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _setup_mlflow(config: Config) -> None:
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    name = config.mlflow.experiment_name
    if mlflow.get_experiment_by_name(name) is None:
        # Pin the artifact folder to the project root (otherwise it depends on the cwd).
        config.mlflow.artifact_dir.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(name, artifact_location=config.mlflow.artifact_dir.as_uri())
    mlflow.set_experiment(name)


def _run_params(config: Config, X: pd.DataFrame, groups_sizes: dict[str, int]) -> dict[str, Any]:
    params = {
        "model": config.model.name,
        "bureau_enabled": config.bureau.enabled,
        "seed": config.seed,
        "holdout_size": config.split.holdout_size,
        "cv_folds": config.split.cv_folds,
        "c_fn": config.costs.c_fn,
        "c_fp": config.costs.c_fp,
        "threshold": round(config.costs.threshold, 4),
        "n_train_rows": len(X),
        "n_input_columns": X.shape[1],
    }
    params.update({f"model.{k}": v for k, v in config.model.active_params.items()})
    params.update({f"n_{k}_columns": v for k, v in groups_sizes.items()})
    return params


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    config = load_config(args.config, args.overrides)

    # 1. Data and the fixed hold-out. The hold-out rows are not used in this stage.
    df = load_applications(config.paths.train_path)
    holdout_ids = load_or_create_holdout_ids(df, config)
    train_df, _holdout_df = split_train_holdout(df, holdout_ids, config.data.id_col)
    logger.info(
        "Training part: %d rows (default rate %.4f); hold-out set aside: %d rows",
        len(train_df), train_df[config.data.target_col].mean(), len(holdout_ids),
    )

    # 2. Row-wise features (+ bureau aggregates) and explicit column groups.
    features_df = add_features(train_df)
    if config.bureau.enabled:
        bureau_agg = aggregate_bureau(load_bureau(config.paths.bureau_path), config.data.id_col)
        features_df = add_bureau_features(features_df, bureau_agg, config.data.id_col)
    X, y = select_model_inputs(features_df, config)
    groups = resolve_column_groups(X, config)
    group_sizes = {name: len(cols) for name, cols in groups.as_dict().items()}
    logger.info("Model inputs: %d columns %s", X.shape[1], group_sizes)

    pipeline = build_pipeline(config.model.name, config.model.active_params, groups, config.seed)

    # 3. Cross-validate, refit on the whole training part, log everything.
    _setup_mlflow(config)
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tags({"stage": "5", "git_commit": _git_commit(args.config)})
        mlflow.log_params(_run_params(config, X, group_sizes))
        mlflow.log_dict(config_to_dict(config), "config_resolved.yaml")  # includes --set overrides
        mlflow.log_dict(groups.as_dict(), "column_groups.json")

        fold_results = cross_validate(pipeline, X, y, config)
        for fold, metrics in enumerate(fold_results, start=1):
            mlflow.log_metrics({f"fold_{k}": v for k, v in metrics.items()}, step=fold)
        summary = summarise(fold_results)
        mlflow.log_metrics(summary)

        logger.info("Refitting on the whole training part (%d rows)", len(X))
        final_model = clone(pipeline).fit(X, y)
        # pickle: our own trusted model. (MLflow's default, skops, would need numpy.log1p
        # marked as a trusted type when loading.)
        mlflow.sklearn.log_model(final_model, name="model", serialization_format="pickle")

    print("\n=== CV summary (mean +/- std over folds) ===")
    for name in ("pr_auc", "roc_auc", "gini", "ks", "expected_profit", "approve_all_profit",
                 "profit_uplift", "approval_rate", "default_capture_rate"):
        mean, std = summary[f"cv_{name}_mean"], summary[f"cv_{name}_std"]
        print(f"{name:>22}: {mean:,.4f} +/- {std:,.4f}")
    print(f"\nthreshold p* = {config.costs.threshold:.4f} | MLflow run id: {run.info.run_id}")


if __name__ == "__main__":
    main()
