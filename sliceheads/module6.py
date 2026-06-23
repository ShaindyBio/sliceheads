"""
sliceheads — Module 6: Results Persistence
Design Document: v0.3  |  Package: 0.1.0  |  Schema: 1

Per design doc p.12, a benchmark run writes:

    results/
        metrics.csv
        test_predictions.csv
        hyperparameters.json
        importance.h5
        config_used.yaml
        run_metadata.json

- metrics.csv            records WHAT THE RESULT WAS.
- config_used.yaml        records WHAT EXPERIMENT WAS REQUESTED.
- run_metadata.json        records the software/hardware environment.
- importance.h5            single canonical filename, shared with
                            Module 5 / Appendix B.
- test_predictions.csv     columns: sample_id, patient_id, true_label,
                            pred_label, prob_class_0, prob_class_1, head_name.

This module is the single place that assembles a full results/ directory
from the outputs of Module 3 (evaluate), Module 4 (run_benchmark/HPO), and
Module 5 (run_explainability) — Module 4 already wrote partial versions of
some of these files; this module is the canonical, complete writer and is
safe to call standalone after any subset of modules 3-5 has run.
"""

from __future__ import annotations

import csv
import json
import os
import platform
from pathlib import Path
from typing import Dict, Optional

import yaml

IMPORTANCE_H5_FILENAME = "importance.h5"   # canonical name, shared with Module 5


# ─────────────────────────────────────────────────────────────────────────────
# metrics.csv
# ─────────────────────────────────────────────────────────────────────────────

def write_metrics_csv(all_results: Dict[str, dict], output_dir: str) -> str:
    """
    Write metrics.csv — one row per head, the classification metrics from
    Module 3's evaluate() (design doc p.10-11).

    Parameters
    ----------
    all_results : dict {head_name: evaluate()-style result dict}
        Each value must contain the keys produced by module3.evaluate():
        roc_auc, roc_auc_ci_lower/upper, pr_auc, accuracy, balanced_accuracy,
        macro_f1, weighted_f1, sensitivity, specificity, operating_point,
        eval_split, n_samples.
    output_dir : str

    Returns
    -------
    str — path to the written file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, "metrics.csv")

    fieldnames = [
        "head_name", "eval_split", "n_samples", "operating_point",
        "roc_auc", "roc_auc_ci_lower", "roc_auc_ci_upper",
        "pr_auc", "accuracy", "balanced_accuracy",
        "macro_f1", "weighted_f1", "sensitivity", "specificity",
    ]

    rows = []
    for head_name, res in all_results.items():
        # res may be the raw evaluate() dict, or {"metrics": {...}, ...}
        m = res.get("metrics", res)
        row = {"head_name": head_name}
        for key in fieldnames[1:]:
            row[key] = m.get(key, "")
        rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[sliceheads] metrics.csv → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# test_predictions.csv
# ─────────────────────────────────────────────────────────────────────────────

def write_predictions_csv(all_results: Dict[str, dict], output_dir: str) -> str:
    """
    Write test_predictions.csv (design doc p.12 exact column spec):
        sample_id, patient_id, true_label, pred_label,
        prob_class_0, prob_class_1, head_name

    Parameters
    ----------
    all_results : dict {head_name: evaluate()-style result dict}
        Reads the "predictions" sub-dict written by module3.evaluate().
    output_dir : str

    Returns
    -------
    str — path to the written file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, "test_predictions.csv")

    cols = ["sample_id", "patient_id", "true_label", "pred_label",
            "prob_class_0", "prob_class_1", "head_name"]

    rows = []
    for head_name, res in all_results.items():
        m = res.get("metrics", res)
        for sample_id, info in m.get("predictions", {}).items():
            rows.append({
                "sample_id":    sample_id,
                "patient_id":   info.get("patient_id", ""),
                "true_label":   info.get("true_label", ""),
                "pred_label":   info.get("pred_label", ""),
                "prob_class_0": info.get("prob_class_0", ""),
                "prob_class_1": info.get("prob_class_1", ""),
                "head_name":    head_name,
            })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[sliceheads] test_predictions.csv → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# hyperparameters.json
# ─────────────────────────────────────────────────────────────────────────────

def write_hyperparameters_json(all_results: Dict[str, dict], output_dir: str) -> str:
    """
    Write hyperparameters.json — best hyperparameter configuration and
    validation score found for each head during Module 4's search.

    Parameters
    ----------
    all_results : dict {head_name: {"best_params": ..., "val_score": ...}}
    output_dir : str

    Returns
    -------
    str — path to the written file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, "hyperparameters.json")

    out = {
        head_name: {
            "best_params": res.get("best_params", {}),
            "val_score":   res.get("val_score", None),
        }
        for head_name, res in all_results.items()
    }

    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[sliceheads] hyperparameters.json → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# config_used.yaml
# ─────────────────────────────────────────────────────────────────────────────

def write_config_used(config: dict, output_dir: str) -> str:
    """
    Write config_used.yaml — an exact copy of the experiment configuration
    that was actually used for this run (design doc p.12: "records what
    experiment was requested").

    Parameters
    ----------
    config : dict
        The parsed config (e.g. from module4.load_config()), or any dict
        describing the experiment that was run.
    output_dir : str

    Returns
    -------
    str — path to the written file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, "config_used.yaml")

    with open(path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"[sliceheads] config_used.yaml → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# run_metadata.json
# ─────────────────────────────────────────────────────────────────────────────

def write_run_metadata(output_dir: str, random_seed: Optional[int] = None,
                       extra: Optional[dict] = None) -> str:
    """
    Write run_metadata.json — the software/hardware environment the run
    executed in (design doc p.12).

    Parameters
    ----------
    output_dir : str
    random_seed : int or None
    extra : dict or None
        Any additional fields to merge in (e.g. experiment name/timestamp).

    Returns
    -------
    str — path to the written file.
    """
    import datetime

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, "run_metadata.json")

    meta = {
        "sliceheads_version": "0.1.0",
        "schema_version":     1,
        "random_seed":        random_seed,
        "timestamp":          datetime.datetime.utcnow().isoformat() + "Z",
        "python_version":     platform.python_version(),
        "platform":           platform.platform(),
    }

    # Best-effort version capture for key dependencies — each is optional
    # so a missing package doesn't break metadata writing.
    for pkg_name, import_name in [
        ("torch", "torch"), ("sklearn", "sklearn"),
        ("numpy", "numpy"), ("h5py", "h5py"),
        ("aeon", "aeon"), ("sktime", "sktime"),
    ]:
        try:
            mod = __import__(import_name)
            meta[f"{pkg_name}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            meta[f"{pkg_name}_version"] = "not installed"

    if extra:
        meta.update(extra)

    with open(path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[sliceheads] run_metadata.json → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Public API — assemble the full results/ directory
# ─────────────────────────────────────────────────────────────────────────────

def persist_results(
    all_results: Dict[str, dict],
    output_dir: str,
    config: Optional[dict] = None,
    importance_results: Optional[Dict[str, dict]] = None,
    random_seed: Optional[int] = None,
) -> Dict[str, str]:
    """
    Assemble the complete results/ directory specified on design doc p.12:

        results/
            metrics.csv
            test_predictions.csv
            hyperparameters.json
            importance.h5            (only if importance_results given)
            config_used.yaml         (only if config given)
            run_metadata.json

    This is the single entry point meant to be called once at the end of
    a benchmark run, after Module 3/4 (classification) and optionally
    Module 5 (explainability) have produced their results.

    Parameters
    ----------
    all_results : dict {head_name: evaluate()-style result, optionally
                        wrapped as {"metrics": ..., "best_params": ...,
                        "val_score": ...} as produced by module4.run_benchmark}
    output_dir : str
    config : dict or None
        The experiment config that was used (written verbatim to
        config_used.yaml). Pass None to skip this file.
    importance_results : dict or None
        Output of module5.run_explainability(). If given, each head's
        LOO importance (and native_attention where available) is written
        into output_dir/importance.h5 via Module 3's save_importance_h5
        (already called internally by Module 5 if its own output_dir was
        set — passing it here again is safe and idempotent per head).
    random_seed : int or None
        Recorded in run_metadata.json.

    Returns
    -------
    dict {file_name: path} for every file written.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    written["metrics.csv"]           = write_metrics_csv(all_results, output_dir)
    written["test_predictions.csv"]  = write_predictions_csv(all_results, output_dir)
    written["hyperparameters.json"]  = write_hyperparameters_json(all_results, output_dir)

    if config is not None:
        written["config_used.yaml"] = write_config_used(config, output_dir)

    written["run_metadata.json"] = write_run_metadata(
        output_dir, random_seed=random_seed,
        extra={"heads_evaluated": list(all_results.keys())},
    )

    if importance_results is not None:
        from sliceheads.module3 import save_importance_h5
        for head_name, res in importance_results.items():
            loo = res.get("loo_importance", res)   # tolerate either shape
            path = save_importance_h5(loo, head_name=head_name, output_dir=output_dir)
        written["importance.h5"] = os.path.join(output_dir, IMPORTANCE_H5_FILENAME)

    print(f"\n[sliceheads] Results persisted to: {output_dir}")
    for name, path in written.items():
        print(f"  ✅ {name}")

    return written


def load_results_directory(results_dir: str) -> Dict[str, object]:
    """
    Read back a previously persisted results/ directory.

    Returns
    -------
    dict with keys: metrics (list of dict rows), test_predictions (list of
    dict rows), hyperparameters (dict), config_used (dict or None),
    run_metadata (dict), importance_h5_path (str or None).
    """
    out: Dict[str, object] = {}

    metrics_path = os.path.join(results_dir, "metrics.csv")
    if os.path.exists(metrics_path):
        with open(metrics_path, newline="") as f:
            out["metrics"] = list(csv.DictReader(f))

    preds_path = os.path.join(results_dir, "test_predictions.csv")
    if os.path.exists(preds_path):
        with open(preds_path, newline="") as f:
            out["test_predictions"] = list(csv.DictReader(f))

    hp_path = os.path.join(results_dir, "hyperparameters.json")
    if os.path.exists(hp_path):
        with open(hp_path) as f:
            out["hyperparameters"] = json.load(f)

    cfg_path = os.path.join(results_dir, "config_used.yaml")
    out["config_used"] = None
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            out["config_used"] = yaml.safe_load(f)

    meta_path = os.path.join(results_dir, "run_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            out["run_metadata"] = json.load(f)

    imp_path = os.path.join(results_dir, IMPORTANCE_H5_FILENAME)
    out["importance_h5_path"] = imp_path if os.path.exists(imp_path) else None

    return out
