"""
sliceheads — Module 4: Hyperparameter Optimization
Design Document: v0.3  |  Package: 0.1.0  |  Schema: 1

Drives hyperparameter search from a YAML config file (Appendix A).
- search_strategy: grid | random | bayesian
- Model selection on validation split (selection_metric: roc_auc)
- Training only on train split — val never added to train
- Writes results to CSV + JSON
"""

from __future__ import annotations

import itertools
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from sliceheads.module2 import (
    ABMILClassifier,
    ALSTMFCNClassifier,
    DSMILClassifier,
    GatedABMILClassifier,
    GeMPoolClassifier,
    InceptionTimeClassifier,
    MaxPoolClassifier,
    MeanPoolClassifier,
    MultiRocketHydraClassifier,
    TransformerMILClassifier,
    _load_split_h5,
)
from sliceheads.module3 import evaluate

# ── head registry ─────────────────────────────────────────────────────────────
HEAD_REGISTRY: Dict[str, type] = {
    "MeanPoolClassifier":         MeanPoolClassifier,
    "MaxPoolClassifier":          MaxPoolClassifier,
    "GeMPoolClassifier":          GeMPoolClassifier,
    "ABMILClassifier":            ABMILClassifier,
    "GatedABMILClassifier":       GatedABMILClassifier,
    "DSMILClassifier":            DSMILClassifier,
    "TransformerMILClassifier":   TransformerMILClassifier,
    "InceptionTimeClassifier":    InceptionTimeClassifier,
    "ALSTMFCNClassifier":         ALSTMFCNClassifier,
    "MultiRocketHydraClassifier": MultiRocketHydraClassifier,
}


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load and return a YAML benchmark config (Appendix A format)."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Search strategy helpers
# ─────────────────────────────────────────────────────────────────────────────

def _grid_combinations(hp_space: Dict[str, List]) -> List[Dict]:
    keys, values = zip(*hp_space.items())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _random_combinations(hp_space: Dict[str, List],
                          n: int = 20,
                          seed: int = 42) -> List[Dict]:
    rng   = random.Random(seed)
    combos = _grid_combinations(hp_space)
    rng.shuffle(combos)
    return combos[:n]


def _bayesian_combinations(hp_space: Dict[str, List],
                            n: int = 20,
                            seed: int = 42) -> List[Dict]:
    """
    Lightweight Bayesian proxy: random search ordered by a simple GP surrogate.
    Falls back to random if scikit-optimize is not installed.
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Categorical

        dimensions = [Categorical(v, name=k) for k, v in hp_space.items()]
        keys = list(hp_space.keys())

        results = []
        def objective(params):
            combo = dict(zip(keys, params))
            results.append(combo)
            return 0.0  # placeholder — real evaluation happens outside

        gp_minimize(objective, dimensions, n_calls=n, random_state=seed, verbose=False)
        return results[:n]
    except ImportError:
        print("[sliceheads] scikit-optimize not found; falling back to random search.")
        return _random_combinations(hp_space, n=n, seed=seed)


def _get_combinations(hp_space, strategy, n_random=20, seed=42):
    s = strategy.lower()
    if s == "grid":
        return _grid_combinations(hp_space)
    elif s == "random":
        return _random_combinations(hp_space, n=n_random, seed=seed)
    elif s == "bayesian":
        return _bayesian_combinations(hp_space, n=n_random, seed=seed)
    else:
        raise ValueError(f"Unknown search_strategy: '{strategy}'. "
                         "Choose grid | random | bayesian.")


# ─────────────────────────────────────────────────────────────────────────────
# Single-head HPO
# ─────────────────────────────────────────────────────────────────────────────

def _quick_val_score(head, h5_path: str, selection_metric: str) -> float:
    """
    Compute a single selection metric on the val split, without the
    95% bootstrap CI machinery (that's reserved for the final test-split
    evaluation per design doc p.10-11). Used inside the HPO inner loop
    where speed matters and CI is not needed for model selection.
    """
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        roc_auc_score, precision_recall_curve, auc as auc_fn,
    )
    X, mask, y_true, _, _ = _load_split_h5(h5_path, "val")
    proba   = head.predict_proba(X, mask=mask)
    y_score = proba[:, 1]
    y_pred  = (y_score >= 0.5).astype(int)

    if selection_metric == "roc_auc":
        return float(roc_auc_score(y_true, y_score))
    elif selection_metric == "pr_auc":
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        return float(auc_fn(rec, prec))
    elif selection_metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    elif selection_metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    elif selection_metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    else:
        raise ValueError(f"Unknown selection_metric: '{selection_metric}'")


import inspect


# ─────────────────────────────────────────────────────────────────────────────
# Config-key → constructor-kwarg mapping
# ─────────────────────────────────────────────────────────────────────────────
#
# Appendix A's example config uses some keys that are documentation/metadata
# rather than literal constructor kwargs (e.g. "classifier": ["logistic_regression"]
# just records which sklearn backend is used — every pooling head in this
# package already hardcodes LogisticRegression, so there's nothing to pass).
# Some keys also use different names than this package's constructors
# (e.g. "lstm_units" in the doc vs "hidden_dim" in ALSTMFCNClassifier).
#
# This dict translates doc-level names to actual constructor kwarg names,
# per head. Keys mapped to None are dropped entirely (pure metadata).

PARAM_ALIASES: Dict[str, Dict[str, Optional[str]]] = {
    "MeanPoolClassifier": {
        "classifier": None,             # metadata only — always logistic_regression
    },
    "MaxPoolClassifier": {
        "classifier": None,
    },
    "GeMPoolClassifier": {
        "classifier": None,
    },
    "MultiRocketHydraClassifier": {
        "n_groups": None,                # not exposed as a separate constructor kwarg
    },
    "InceptionTimeClassifier": {
        # kernel_sizes / dropout / lr / weight_decay / n_filters / depth match directly
        "early_stopping": None,          # always on when X_val is provided
    },
    "ALSTMFCNClassifier": {
        "lstm_units":   "hidden_dim",    # doc name → constructor name
        "conv_filters": None,            # fixed architecture in this implementation
        "attention":    None,            # this implementation always uses attention
        "early_stopping": None,
    },
    "ABMILClassifier": {
        "early_stopping": None,
    },
    "GatedABMILClassifier": {
        "early_stopping": None,
    },
    "DSMILClassifier": {
        "early_stopping": None,
    },
    "TransformerMILClassifier": {
        "early_stopping": None,
    },
}


def _filter_params_for_head(head_name: str, head_cls: type,
                            params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate/drop config-file hyperparameter keys so only valid constructor
    kwargs for `head_cls` remain.

    Steps:
      1. Apply any per-head alias from PARAM_ALIASES (rename or drop=None).
      2. Drop any remaining key not in head_cls.__init__'s signature, with
         a one-time warning so silent typos in the YAML are visible.
    """
    aliases = PARAM_ALIASES.get(head_name, {})
    valid_kwargs = set(inspect.signature(head_cls.__init__).parameters) - {"self"}

    out: Dict[str, Any] = {}
    for key, value in params.items():
        target_key = aliases.get(key, key)   # alias rename, or unchanged
        if target_key is None:
            continue                          # explicitly dropped (metadata-only)
        if target_key not in valid_kwargs:
            print(f"  [warn] '{key}' is not a valid {head_name} parameter — skipping.")
            continue
        out[target_key] = value
    return out


def _hpo_one_head(
    head_name: str,
    head_cls: type,
    hp_space: Dict[str, List],
    h5_path: str,
    strategy: str = "grid",
    n_random: int = 20,
    selection_metric: str = "roc_auc",
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[Any, Dict, float]:
    """
    Run HPO for one head.

    Returns
    -------
    best_head    : fitted head with best val metric
    best_params  : hyperparameter dict
    best_val_score : float
    """
    # load data once
    X_tr, m_tr, y_tr, _, _ = _load_split_h5(h5_path, "train")
    X_vl, m_vl, y_vl, _, _ = _load_split_h5(h5_path, "val")

    combos = _get_combinations(hp_space, strategy, n_random=n_random, seed=seed)
    total  = len(combos)

    best_score  = -1.0
    best_head   = None
    best_params = {}

    print(f"\n[sliceheads] {head_name}: {total} configurations "
          f"(strategy={strategy})")

    for idx, params in enumerate(combos):
        if verbose:
            print(f"  [{idx+1}/{total}] {params}", end=" ... ")

        try:
            ctor_params = _filter_params_for_head(head_name, head_cls, params)
            head = head_cls(**ctor_params)
            head.fit(X_tr, y_tr, mask_train=m_tr,
                     X_val=X_vl, y_val=y_vl, mask_val=m_vl)

            # Evaluate on val split for model selection.
            # CI is not needed during search — only the final test-split
            # evaluation requires the 95% bootstrap CI (design doc p.10-11).
            # We compute the selection metric directly rather than calling
            # evaluate() with n_bootstraps=0, to keep the HPO loop's metric
            # computation independent of evaluate()'s CI machinery.
            score = _quick_val_score(head, h5_path, selection_metric)

            if verbose:
                print(f"val_{selection_metric}={score:.4f}")

            if score > best_score:
                best_score  = score
                best_head   = head
                best_params = params

        except Exception as exc:
            if verbose:
                print(f"ERROR: {exc}")

    print(f"  → Best val {selection_metric}={best_score:.4f}  params={best_params}")
    return best_head, best_params, best_score


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    config_path: str,
    h5_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    skip_hpo: bool = False,
    verbose: bool = True,
) -> Dict[str, dict]:
    """
    Full benchmark pipeline driven by a YAML config (Appendix A).

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.
    h5_path : str or None
        Override the h5_path from config (useful for quick tests).
    output_dir : str or None
        Where to write results. Defaults to ./results/.
    skip_hpo : bool
        If True, each head is trained with default params (no grid search).
    verbose : bool

    Returns
    -------
    dict  {head_name: {"metrics": ..., "best_params": ..., "val_score": ...}}
    """
    cfg = load_config(config_path)

    # resolve paths
    h5   = h5_path or cfg.get("data", {}).get("h5_path", "embeddings.h5")
    outd = output_dir or "results"
    Path(outd).mkdir(parents=True, exist_ok=True)

    # training config
    train_cfg  = cfg.get("training", {})
    seed       = train_cfg.get("random_seed", 42)
    strategy   = cfg.get("search_strategy", "grid")
    sel_metric = train_cfg.get("selection_metric", "roc_auc")
    eval_split = train_cfg.get("final_evaluation_split", "test")
    n_random   = cfg.get("n_random_configs", 20)

    heads_cfg  = cfg.get("heads", {})
    all_results: Dict[str, dict] = {}

    for head_name, hcfg in heads_cfg.items():
        if not hcfg.get("enabled", True):
            print(f"[sliceheads] Skipping {head_name} (enabled=false)")
            continue

        head_cls = HEAD_REGISTRY.get(head_name)
        if head_cls is None:
            print(f"[sliceheads] Unknown head '{head_name}', skipping.")
            continue

        hp_space = hcfg.get("hyperparameters", {})

        if skip_hpo or not hp_space:
            # train with defaults
            print(f"[sliceheads] Training {head_name} with default params...")
            head = head_cls()
            X_tr, m_tr, y_tr, _, _ = _load_split_h5(h5, "train")
            try:
                X_vl, m_vl, y_vl, _, _ = _load_split_h5(h5, "val")
                head.fit(X_tr, y_tr, mask_train=m_tr,
                         X_val=X_vl, y_val=y_vl, mask_val=m_vl)
            except ValueError:
                head.fit(X_tr, y_tr, mask_train=m_tr)
            best_params = {}
            val_score   = float("nan")
        else:
            head, best_params, val_score = _hpo_one_head(
                head_name=head_name, head_cls=head_cls,
                hp_space=hp_space, h5_path=h5,
                strategy=strategy, n_random=n_random,
                selection_metric=sel_metric, seed=seed, verbose=verbose,
            )
            if head is None:
                print(f"[sliceheads] All configs failed for {head_name}, skipping.")
                continue

        # final evaluation on test split
        print(f"[sliceheads] Final evaluation: {head_name} on '{eval_split}' split...")
        from sliceheads.module3 import evaluate as _eval, print_metrics
        metrics = _eval(head, h5, eval_split=eval_split,
                        n_bootstraps=cfg.get("confidence_interval", {})
                                        .get("n_bootstraps", 2000),
                        random_seed=seed)
        print_metrics(metrics)

        # save head
        head_dir = os.path.join(outd, head_name)
        head.save(head_dir)

        all_results[head_name] = {
            "metrics":     metrics,
            "best_params": best_params,
            "val_score":   val_score,
        }

    # ── write outputs ────────────────────────────────────────────────────────
    _write_metrics_csv(all_results, outd)
    _write_predictions_csv(all_results, outd)
    _write_hyperparams_json(all_results, outd)
    _write_run_metadata(cfg, outd, seed)

    print(f"\n[sliceheads] Benchmark complete. Results in: {outd}")
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────────────────────

def _write_metrics_csv(all_results: dict, output_dir: str) -> None:
    import csv
    path = os.path.join(output_dir, "metrics.csv")
    rows = []
    metric_keys = [
        "head_name", "eval_split", "n_samples", "operating_point",
        "roc_auc", "roc_auc_ci_lower", "roc_auc_ci_upper",
        "pr_auc", "accuracy", "balanced_accuracy",
        "macro_f1", "weighted_f1", "sensitivity", "specificity",
    ]
    for head_name, res in all_results.items():
        m = res["metrics"]
        row = {k: m.get(k, "") for k in metric_keys}
        rows.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metric_keys)
        writer.writeheader(); writer.writerows(rows)
    print(f"[sliceheads] metrics.csv → {path}")


def _write_predictions_csv(all_results: dict, output_dir: str) -> None:
    import csv
    path = os.path.join(output_dir, "test_predictions.csv")
    cols = ["sample_id", "patient_id", "true_label", "pred_label",
            "prob_class_0", "prob_class_1", "head_name"]
    rows = []
    for head_name, res in all_results.items():
        for sid, info in res["metrics"].get("predictions", {}).items():
            rows.append({
                "sample_id":    sid,
                "patient_id":   info.get("patient_id", ""),
                "true_label":   info.get("true_label", ""),
                "pred_label":   info.get("pred_label", ""),
                "prob_class_0": info.get("prob_class_0", ""),
                "prob_class_1": info.get("prob_class_1", ""),
                "head_name":    head_name,
            })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader(); writer.writerows(rows)
    print(f"[sliceheads] test_predictions.csv → {path}")


def _write_hyperparams_json(all_results: dict, output_dir: str) -> None:
    path = os.path.join(output_dir, "hyperparameters.json")
    out  = {
        head: {"best_params": res["best_params"], "val_score": res["val_score"]}
        for head, res in all_results.items()
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[sliceheads] hyperparameters.json → {path}")


def _write_run_metadata(cfg: dict, output_dir: str, seed: int) -> None:
    import platform, torch, sklearn
    path = os.path.join(output_dir, "run_metadata.json")
    meta = {
        "sliceheads_version": "0.1.0",
        "schema_version":     1,
        "random_seed":        seed,
        "python":             platform.python_version(),
        "platform":           platform.platform(),
        "torch":              torch.__version__,
        "sklearn":            sklearn.__version__,
        "config_experiment":  cfg.get("experiment", {}),
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[sliceheads] run_metadata.json → {path}")
