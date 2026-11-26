"""Load, override and validate the YAML run configuration.

Settings live in ``configs/*.yaml``. This module turns one YAML file, plus optional
command-line overrides such as ``split.cv_folds=3``, into a typed, validated, read-only
``Config`` object. Typos, missing keys and wrong types fail at load time, before any
data is read, instead of halfway through a training run.
"""

from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file or override is invalid."""


# --------------------------------------------------------------------------------------
# Typed sections. One dataclass per top-level YAML section.
# frozen=True: settings cannot be changed by accident after loading.
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PathsConfig:
    raw_dir: Path
    train_file: str
    bureau_file: str
    processed_dir: Path
    holdout_ids_file: str

    @property
    def train_path(self) -> Path:
        return self.raw_dir / self.train_file

    @property
    def bureau_path(self) -> Path:
        return self.raw_dir / self.bureau_file

    @property
    def holdout_ids_path(self) -> Path:
        return self.processed_dir / self.holdout_ids_file


@dataclass(frozen=True)
class DataConfig:
    id_col: str
    target_col: str


@dataclass(frozen=True)
class SplitConfig:
    holdout_size: float
    cv_folds: int

    def __post_init__(self) -> None:
        if not 0.0 < self.holdout_size < 1.0:
            raise ConfigError(f"split.holdout_size must be in (0, 1), got {self.holdout_size}")
        if self.cv_folds < 2:
            raise ConfigError(f"split.cv_folds must be >= 2, got {self.cv_folds}")


@dataclass(frozen=True)
class CostsConfig:
    c_fn: float  # cost of approving a borrower who defaults
    c_fp: float  # cost of declining a borrower who would have repaid

    def __post_init__(self) -> None:
        if self.c_fn <= 0 or self.c_fp <= 0:
            raise ConfigError("costs.c_fn and costs.c_fp must be positive")

    @property
    def threshold(self) -> float:
        """Cost-minimising decline threshold: decline when P(default) > c_fp / (c_fp + c_fn)."""
        return self.c_fp / (self.c_fp + self.c_fn)


@dataclass(frozen=True)
class FeaturesConfig:
    protected_cols: list[str]
    redundant_cols: list[str]
    money_cols: list[str]


@dataclass(frozen=True)
class BureauConfig:
    enabled: bool          # join bureau aggregates (bureau.py) or not
    money_cols: list[str]  # bureau sums that get log1p in the LR pipeline


@dataclass(frozen=True)
class ModelConfig:
    name: str
    # One free-form parameter block per model name; passed straight to the estimator.
    params: dict[str, dict[str, Any]]

    def __post_init__(self) -> None:
        if self.name not in self.params:
            raise ConfigError(
                f"model.name {self.name!r} has no block under model.params; "
                f"available: {sorted(self.params)}"
            )
        if not isinstance(self.params[self.name], dict):
            raise ConfigError(f"model.params.{self.name} must be a mapping")

    @property
    def active_params(self) -> dict[str, Any]:
        """Parameters of the selected model only."""
        return self.params[self.name]


@dataclass(frozen=True)
class MlflowConfig:
    experiment_name: str
    db_path: Path       # SQLite tracking database (MLflow 3's default backend)
    artifact_dir: Path  # where runs store models and files

    @property
    def tracking_uri(self) -> str:
        # Absolute path, so runs land in the same database from any working directory.
        return f"sqlite:///{self.db_path.as_posix()}"


@dataclass(frozen=True)
class Config:
    seed: int
    paths: PathsConfig
    data: DataConfig
    split: SplitConfig
    costs: CostsConfig
    features: FeaturesConfig
    bureau: BureauConfig
    model: ModelConfig
    mlflow: MlflowConfig


# --------------------------------------------------------------------------------------
# Building and validation
# --------------------------------------------------------------------------------------
def _check_type(value: Any, expected: Any, where: str) -> Any:
    """Check simple scalar types. Dataclasses do not do this at runtime."""
    if expected is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if expected is float:
        # Accept ints for floats (YAML reads 350000 as int), but not bools.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
    if expected is str and not isinstance(value, str):
        raise ConfigError(f"{where}: expected a string, got {value!r}")
    origin = getattr(expected, "__origin__", None)
    if origin is list and not isinstance(value, list):
        raise ConfigError(f"{where}: expected a list, got {value!r}")
    if origin is dict and not isinstance(value, dict):
        raise ConfigError(f"{where}: expected a mapping, got {value!r}")
    return value


def _build(cls: type, raw: Any, where: str, root: Path) -> Any:
    """Build dataclass ``cls`` from a YAML mapping, rejecting unknown and missing keys."""
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(raw).__name__}")

    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(raw) - set(known))
    missing = sorted(set(known) - set(raw))
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(known)}")
    if missing:
        raise ConfigError(f"{where}: missing key(s) {missing}")

    kwargs = {}
    for name, f in known.items():
        key = f"{where}.{name}" if where else name
        value = raw[name]
        if is_dataclass(f.type):
            kwargs[name] = _build(f.type, value, key, root)
        elif f.type is Path:
            path = Path(_check_type(value, str, key))
            kwargs[name] = path if path.is_absolute() else (root / path).resolve()
        else:
            kwargs[name] = _check_type(value, f.type, key)
    return cls(**kwargs)


# --------------------------------------------------------------------------------------
# Command-line overrides
# --------------------------------------------------------------------------------------
# Only these free-form mappings (and anything nested inside them) may gain keys that
# are not already in the YAML file.
_OPEN_MAPPINGS = {("model", "params")}


def _is_open(section: tuple[str, ...]) -> bool:
    return any(section[: len(prefix)] == prefix for prefix in _OPEN_MAPPINGS)


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``dotted.key=value`` overrides to the raw YAML mapping (in place).

    Values are parsed as YAML, so ``3`` -> int, ``0.5`` -> float, ``null`` -> None,
    ``balanced`` -> str. A key that does not exist raises, so a typo cannot silently
    add a new setting (except inside ``model.params``, e.g. ``model.params.lightgbm.max_depth``).
    """
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override {item!r} must look like section.key=value")
        dotted, text = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = raw
        for depth, key in enumerate(keys[:-1]):
            if not isinstance(node.get(key), dict):
                raise ConfigError(f"override {item!r}: no section {'.'.join(keys[: depth + 1])!r}")
            node = node[key]
        last = keys[-1]
        if last not in node and not _is_open(tuple(keys[:-1])):
            raise ConfigError(f"override {item!r}: unknown key {dotted!r}")
        node[last] = yaml.safe_load(text)
    return raw


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def project_root_for(config_path: Path) -> Path:
    """The project root is the folder that contains the configs/ folder."""
    return config_path.resolve().parent.parent


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Read a YAML config, apply overrides, and return a validated ``Config``."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path.resolve()}")
    with config_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    apply_overrides(raw, overrides or [])
    return _build(Config, raw, where="", root=project_root_for(config_path))


def config_to_dict(config: Config) -> dict[str, Any]:
    """Plain, YAML/JSON-serialisable dict of the resolved config (for MLflow logging)."""

    def convert(obj: Any) -> Any:
        if isinstance(obj, Path):
            return obj.as_posix()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    return convert(asdict(config))
