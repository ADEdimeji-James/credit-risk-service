"""Train and evaluate a model from a YAML config, logging the run to MLflow.

Usage:
    python -m credit_risk.train --config configs/base.yaml
    python -m credit_risk.train --config configs/base.yaml --set model.params.class_weight=balanced

For now this only loads and prints the resolved config. Data loading, the pipeline,
cross-validation and MLflow logging are added in items 4-8 of Stage 4.
"""

import argparse

import yaml

from credit_risk.config import config_to_dict, load_config


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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config, args.overrides)
    print(yaml.safe_dump(config_to_dict(config), sort_keys=False))
    print(f"decision threshold p* = {config.costs.threshold:.4f}")


if __name__ == "__main__":
    main()
