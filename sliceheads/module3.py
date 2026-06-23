"""
sliceheads — Module 3: Evaluation Metrics + Explainability (LOO)
Design Document: v0.3  |  Package: 0.1.0  |  Schema: 1

מה המודול הזה עושה (עמ' 10-11 + Appendix B עמ' 17-18):

Metrics (על test split בלבד כברירת מחדל):
  - Accuracy, Balanced Accuracy
  - Macro F1, Weighted F1
  - ROC-AUC, PR-AUC
  - Sensitivity (Recall class 1), Specificity (Recall class 0)
  - Confusion Matrix
  - 95% Bootstrap CI על ROC-AUC (n=2000)
  - operating_point = 0.5 (קבוע, מתועד במפורש)

Raw predictions dict: {sample_id → {patient_id, true_label, pred_label, prob_class_0, prob_class_1}}

LOO Slice Importance (Appendix B):
  - אותה שיטה לכל הראשים (model-agnostic)
  - כל slice מוחלף ב-x_bar (mean embedding מה-HDF5)
  - delta_k = p_0 - p_k  (signed)
  - importance = |delta| / (|delta|.sum() + 1e-9)  (non-negative, sums to 1)
  - signed_importance = delta
  - native_attention: None עבור pooling heads
  - נשמר ל-importance.h5 (Module 6 canonical filename)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    auc,
)

# ── canonical filename (shared with Module 6) ─────────────────────────────────
IMPORTANCE_H5_FILENAME = "importance.h5"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_split(
    h5_path: str,
    split: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load all samples of a given split from an HDF5 file.

    Returns
    -------
    X        : float32 [M, N_max, D]  — zero-padded embeddings
    mask     : int8    [M, N_max]     — 1=real slice, 0=padding
    y        : int64   [M]            — labels
    ids      : list[str]              — sample_ids  ("sample_001", …)
    pids     : list[str]              — patient_ids
    """
    X_list, y_list, ids, pids = [], [], [], []

    with h5py.File(h5_path, "r") as f:
        sample_keys = sorted(k for k in f.keys() if k.startswith("sample_"))
        for key in sample_keys:
            grp = f[key]
            sp  = grp.attrs.get("split", "train")
            if isinstance(sp, bytes):
                sp = sp.decode()
            if sp != split:
                continue
            X_list.append(grp["embeddings"][:].astype(np.float32))
            y_list.append(int(grp["label"][()]))
            ids.append(key)
            pid = grp.attrs.get("patient_id", key)
            pids.append(pid.decode() if isinstance(pid, bytes) else str(pid))

    if not X_list:
        raise ValueError(f"No samples found for split='{split}' in {h5_path}.")

    D     = X_list[0].shape[1]
    N_max = max(x.shape[0] for x in X_list)
    M     = len(X_list)

    X_pad = np.zeros((M, N_max, D), dtype=np.float32)
    mask  = np.zeros((M, N_max),    dtype=np.int8)
    for i, emb in enumerate(X_list):
        n = emb.shape[0]
        X_pad[i, :n, :] = emb
        mask[i, :n]     = 1

    return X_pad, mask, np.array(y_list, dtype=np.int64), ids, pids


def _load_mean_embedding(h5_path: str) -> np.ndarray:
    """Read the global mean embedding from HDF5 root (stored by Module 1)."""
    with h5py.File(h5_path, "r") as f:
        if "mean_embedding" not in f:
            raise KeyError(
                "'mean_embedding' not found in HDF5 root. "
                "Regenerate the HDF5 with Module 1 >= 0.1.0."
            )
        return f["mean_embedding"][:].astype(np.float32)


def _bootstrap_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstraps: int = 2000,
    ci: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, float]:
    """
    95% bootstrap confidence interval for ROC-AUC (design doc page 10).

    Returns
    -------
    (lower, upper) bounds of the CI.
    """
    rng   = np.random.default_rng(random_seed)
    aucs  = []
    n     = len(y_true)

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        s_b = y_score[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_b, s_b))
        except Exception:
            continue

    if not aucs:
        return (float("nan"), float("nan"))

    alpha = 1.0 - ci
    lower = float(np.percentile(aucs, 100 * alpha / 2))
    upper = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return lower, upper


# ─────────────────────────────────────────────────────────────────────────────
# Public API — evaluate()
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    head,
    h5_path: str,
    eval_split: str = "test",
    operating_point: float = 0.5,
    n_bootstraps: int = 2000,
    ci: float = 0.95,
    random_seed: int = 42,
) -> Dict:
    """
    Compute all classification metrics for a fitted head on one split.

    Metrics (design doc page 10-11)
    --------------------------------
    Threshold-free  : roc_auc, pr_auc
    Threshold-based : accuracy, balanced_accuracy, macro_f1, weighted_f1,
                      sensitivity, specificity, confusion_matrix
                      (all at fixed operating_point = 0.5)
    Uncertainty     : roc_auc_ci_lower, roc_auc_ci_upper
                      (95% bootstrap, n=2000)

    Parameters
    ----------
    head         : fitted BaseHead (MeanPoolClassifier or MaxPoolClassifier)
    h5_path      : str  — path to the HDF5 embeddings file
    eval_split   : str  — which split to evaluate on (default "test")
    operating_point : float — decision threshold (default 0.5, design doc page 10)
    n_bootstraps : int  — bootstrap iterations for CI (default 2000)
    ci           : float — confidence level (default 0.95)
    random_seed  : int

    Returns
    -------
    dict with keys:
        head_name, eval_split, n_samples, operating_point,
        roc_auc, pr_auc,
        roc_auc_ci_lower, roc_auc_ci_upper,
        accuracy, balanced_accuracy,
        macro_f1, weighted_f1,
        sensitivity, specificity,
        confusion_matrix  (2×2 list),
        predictions  (dict: sample_id → per-sample result)
    """
    # ── load split ────────────────────────────────────────────────────────
    X, mask, y_true, ids, pids = _load_split(h5_path, split=eval_split)

    if len(np.unique(y_true)) < 2:
        raise ValueError(
            f"Split '{eval_split}' has only one class — "
            "cannot compute AUC or most metrics."
        )

    # ── predict ───────────────────────────────────────────────────────────
    proba  = head.predict_proba(X, mask=mask)          # [M, 2]
    y_score = proba[:, 1]                               # positive class probability
    y_pred  = (y_score >= operating_point).astype(int)

    # ── threshold-free metrics ────────────────────────────────────────────
    roc_auc = float(roc_auc_score(y_true, y_score))

    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_score)
    pr_auc = float(auc(recall_vals, precision_vals))

    # ── bootstrap CI ──────────────────────────────────────────────────────
    ci_lower, ci_upper = _bootstrap_auc(
        y_true, y_score,
        n_bootstraps=n_bootstraps,
        ci=ci,
        random_seed=random_seed,
    )

    # ── threshold-based metrics ───────────────────────────────────────────
    accuracy          = float(accuracy_score(y_true, y_pred))
    balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1          = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    weighted_f1       = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    # sensitivity = TP / (TP + FN) = recall of class 1
    # specificity = TN / (TN + FP) = recall of class 0
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    # ── raw predictions dict (design doc page 10) ─────────────────────────
    predictions = {}
    for i, sid in enumerate(ids):
        predictions[sid] = {
            "patient_id":   pids[i],
            "true_label":   int(y_true[i]),
            "pred_label":   int(y_pred[i]),
            "prob_class_0": float(proba[i, 0]),
            "prob_class_1": float(proba[i, 1]),
            "split":        eval_split,
            "head_name":    head.__class__.__name__,
        }

    return {
        # identifiers
        "head_name":          head.__class__.__name__,
        "eval_split":         eval_split,
        "n_samples":          int(len(y_true)),
        "operating_point":    operating_point,       # explicit (design doc page 10)
        # threshold-free
        "roc_auc":            roc_auc,
        "pr_auc":             pr_auc,
        # bootstrap CI
        "roc_auc_ci_lower":   ci_lower,
        "roc_auc_ci_upper":   ci_upper,
        "n_bootstraps":       n_bootstraps,
        "ci_level":           ci,
        # threshold-based
        "accuracy":           accuracy,
        "balanced_accuracy":  balanced_accuracy,
        "macro_f1":           macro_f1,
        "weighted_f1":        weighted_f1,
        "sensitivity":        sensitivity,
        "specificity":        specificity,
        "confusion_matrix":   cm.tolist(),
        # raw predictions
        "predictions":        predictions,
    }


def print_metrics(results: Dict) -> None:
    """Pretty-print the output of evaluate()."""
    h  = results["head_name"]
    sp = results["eval_split"]
    n  = results["n_samples"]

    print(f"\n{'='*60}")
    print(f"  {h}  |  split={sp}  |  n={n}")
    print(f"{'='*60}")
    print(f"  ROC-AUC      : {results['roc_auc']:.4f}  "
          f"(95% CI: {results['roc_auc_ci_lower']:.4f} – {results['roc_auc_ci_upper']:.4f})")
    print(f"  PR-AUC       : {results['pr_auc']:.4f}")
    print(f"  Accuracy     : {results['accuracy']:.4f}")
    print(f"  Balanced Acc : {results['balanced_accuracy']:.4f}")
    print(f"  Macro F1     : {results['macro_f1']:.4f}")
    print(f"  Weighted F1  : {results['weighted_f1']:.4f}")
    print(f"  Sensitivity  : {results['sensitivity']:.4f}")
    print(f"  Specificity  : {results['specificity']:.4f}")
    print(f"  Confusion Matrix (TN FP / FN TP):")
    cm = results["confusion_matrix"]
    print(f"    [[{cm[0][0]:4d}  {cm[0][1]:4d}]")
    print(f"     [{cm[1][0]:4d}  {cm[1][1]:4d}]]")
    print(f"  operating_point = {results['operating_point']}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Public API — compute_loo_importance()
# ─────────────────────────────────────────────────────────────────────────────

def compute_loo_importance(
    head,
    h5_path: str,
    compute_on: str = "test",
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Leave-One-Out slice importance for a fitted head (Appendix B, pages 17-18).

    Algorithm (model-agnostic, identical for all head families):
    ------------------------------------------------------------
    Given embeddings X of shape [N, D] for one sample:
      1. p_0 = head.predict_proba(X)[positive_class]  (baseline)
      2. For k in 0..N-1:
           X_k = copy of X with row k replaced by x_bar (mean embedding)
           p_k = head.predict_proba(X_k)[positive_class]
           delta_k       = p_0 - p_k          (signed)
           abs_delta_k   = |delta_k|
      3. importance        = abs_delta / (abs_delta.sum() + 1e-9)
         signed_importance = delta

    Mean replacement, not deletion (Appendix B page 18):
      - keeps sequence length N constant
      - x_bar is the global mean computed over ALL training slices,
        stored in the HDF5 root as 'mean_embedding'

    Parameters
    ----------
    head        : fitted head (MeanPoolClassifier or MaxPoolClassifier)
    h5_path     : str — path to the embeddings HDF5 file
    compute_on  : "test" | "all"  (default "test", design doc page 18)
    verbose     : bool — print progress

    Returns
    -------
    dict  {sample_id: {
        "importance":        np.ndarray [N],  non-negative, sums to 1
        "signed_importance": np.ndarray [N],  raw signed delta
        "native_attention":  None,            (pooling heads have no native attention)
        "n_slices":          int,
        "split":             str,
        "patient_id":        str,
    }}
    """
    x_bar = _load_mean_embedding(h5_path)   # [D]

    importance_results: Dict[str, Dict] = {}

    with h5py.File(h5_path, "r") as f:
        sample_keys = sorted(k for k in f.keys() if k.startswith("sample_"))

        for key in sample_keys:
            grp = f[key]
            sp  = grp.attrs.get("split", "train")
            if isinstance(sp, bytes):
                sp = sp.decode()

            if compute_on != "all" and sp != compute_on:
                continue

            pid = grp.attrs.get("patient_id", key)
            if isinstance(pid, bytes):
                pid = pid.decode()

            embeddings = grp["embeddings"][:].astype(np.float32)   # [N, D]
            N, D       = embeddings.shape

            if verbose:
                print(f"  LOO  {key}  ({sp})  N={N} slices ...", end="  ")

            t0 = time.time()

            # ── baseline prediction ───────────────────────────────────────
            X0   = embeddings[np.newaxis, :, :]              # [1, N, D]
            mask0 = np.ones((1, N), dtype=np.int8)
            p0   = float(head.predict_proba(X0, mask=mask0)[0, 1])

            # ── per-slice perturbation ────────────────────────────────────
            deltas = np.zeros(N, dtype=np.float32)

            for k in range(N):
                X_k          = embeddings.copy()
                X_k[k]       = x_bar                         # replace with mean
                X_k_batch    = X_k[np.newaxis, :, :]         # [1, N, D]
                mask_k       = np.ones((1, N), dtype=np.int8)
                p_k          = float(head.predict_proba(X_k_batch, mask=mask_k)[0, 1])
                deltas[k]    = p0 - p_k                      # signed delta

            abs_deltas = np.abs(deltas)
            importance = (abs_deltas / (abs_deltas.sum() + 1e-9)).astype(np.float32)

            elapsed = time.time() - t0
            if verbose:
                print(f"done in {elapsed:.1f}s")

            importance_results[key] = {
                "importance":        importance,         # [N], non-negative, sums ~1
                "signed_importance": deltas.astype(np.float32),  # [N], signed
                "native_attention":  None,               # pooling heads: no native attn
                "n_slices":          N,
                "split":             sp,
                "patient_id":        pid,
            }

    if not importance_results:
        raise ValueError(
            f"No samples found for compute_on='{compute_on}' in {h5_path}."
        )

    return importance_results


# ─────────────────────────────────────────────────────────────────────────────
# Public API — save_importance_h5()
# ─────────────────────────────────────────────────────────────────────────────

def save_importance_h5(
    importance_dict: Dict[str, Dict],
    head_name: str,
    output_dir: str,
    filename: str = IMPORTANCE_H5_FILENAME,
) -> str:
    """
    Write LOO importance results to importance.h5 (design doc page 12 + Appendix B).

    Structure of importance.h5 (mirrors embeddings file):
        /sample_001/
            /MeanPoolClassifier/
                importance:        float32 [N]
                signed_importance: float32 [N]
                (native_attention: absent for pooling heads)

    If the file already exists (e.g. from a previous head), new head groups
    are appended without overwriting existing ones.

    Parameters
    ----------
    importance_dict : output of compute_loo_importance()
    head_name       : str  — used as the inner group name
    output_dir      : str  — directory to write importance.h5
    filename        : str  — default IMPORTANCE_H5_FILENAME = "importance.h5"

    Returns
    -------
    str  — full path to the written file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    mode = "a" if os.path.exists(out_path) else "w"

    with h5py.File(out_path, mode) as f:
        for sample_id, data in importance_dict.items():

            # create or open sample group
            if sample_id not in f:
                grp = f.create_group(sample_id)
                grp.attrs["patient_id"] = data["patient_id"]
                grp.attrs["split"]      = data["split"]
            else:
                grp = f[sample_id]

            # create or replace head subgroup
            if head_name in grp:
                del grp[head_name]
            head_grp = grp.create_group(head_name)

            head_grp.create_dataset(
                "importance",
                data=data["importance"],
                dtype="float32",
            )
            head_grp.create_dataset(
                "signed_importance",
                data=data["signed_importance"],
                dtype="float32",
            )
            # native_attention: omitted for pooling heads (design doc page 18)
            if data.get("native_attention") is not None:
                head_grp.create_dataset(
                    "native_attention",
                    data=data["native_attention"],
                    dtype="float32",
                )

    print(f"[sliceheads] importance.h5 written → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Public API — benchmark()
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(
    heads: Dict[str, object],
    h5_path: str,
    eval_split: str = "test",
    compute_importance: bool = False,
    importance_split: str = "test",
    output_dir: Optional[str] = None,
    n_bootstraps: int = 2000,
    random_seed: int = 42,
) -> Dict[str, Dict]:
    """
    Run evaluate() (and optionally LOO importance) for multiple heads at once.

    Parameters
    ----------
    heads : dict  {name: fitted_head}
        Example: {"MeanPool": mean_clf, "MaxPool": max_clf}

    h5_path : str
        Path to the HDF5 embeddings file.

    eval_split : str
        Split to evaluate on (default "test").

    compute_importance : bool
        If True, also run LOO importance for each head on importance_split.
        Warning: this can be slow (N forward passes per sample per head).

    importance_split : str
        Split to run LOO on (default "test").  Pass "all" for all splits.

    output_dir : str or None
        If provided, importance.h5 is written here.
        Required when compute_importance=True.

    n_bootstraps : int
        Bootstrap iterations for AUC CI (default 2000).

    random_seed : int

    Returns
    -------
    dict  {head_name: evaluate() result dict}
        If compute_importance=True, each result also contains
        "importance": {sample_id: ...} from compute_loo_importance().
    """
    if compute_importance and output_dir is None:
        raise ValueError(
            "output_dir is required when compute_importance=True."
        )

    all_results: Dict[str, Dict] = {}

    for name, head in heads.items():
        print(f"\n[sliceheads] Evaluating: {name}")

        result = evaluate(
            head,
            h5_path,
            eval_split=eval_split,
            n_bootstraps=n_bootstraps,
            random_seed=random_seed,
        )
        print_metrics(result)

        if compute_importance:
            print(f"[sliceheads] Computing LOO importance for {name} "
                  f"(split='{importance_split}') …")
            imp = compute_loo_importance(
                head,
                h5_path,
                compute_on=importance_split,
                verbose=True,
            )
            result["importance"] = imp

            if output_dir:
                save_importance_h5(imp, head_name=name, output_dir=output_dir)

        all_results[name] = result

    return all_results


def print_comparison_table(all_results: Dict[str, Dict]) -> None:
    """
    Print a side-by-side comparison table for all evaluated heads.
    """
    metrics = [
        "roc_auc", "pr_auc", "accuracy",
        "balanced_accuracy", "macro_f1", "sensitivity", "specificity",
    ]

    col_w = 14
    head_names = list(all_results.keys())

    header = f"{'metric':<22}" + "".join(f"{n:>{col_w}}" for n in head_names)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for m in metrics:
        row = f"{m:<22}"
        for name in head_names:
            val = all_results[name].get(m, float("nan"))
            row += f"{val:>{col_w}.4f}"
        print(row)

    # AUC CI row
    row = f"{'roc_auc_95%_CI':<22}"
    for name in head_names:
        lo = all_results[name].get("roc_auc_ci_lower", float("nan"))
        hi = all_results[name].get("roc_auc_ci_upper", float("nan"))
        ci_str = f"[{lo:.3f},{hi:.3f}]"
        row += f"{ci_str:>{col_w}}"
    print(row)
    print("=" * len(header))
