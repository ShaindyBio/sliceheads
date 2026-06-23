"""
sliceheads — Module 5: Explainability
Design Document: v0.3  |  Package: 0.1.0  |  Schema: 1

Per design doc p.11 + Appendix B (p.17-18), this module produces:
  - loo_importance       (computed by Module 3's compute_loo_importance)
  - native_attention      (computed by each head's native_attention(), where available)

Explainability evaluation (design doc p.11):
  - spearman_correlation between predicted importance and ground-truth
    annotations (important_slices field, when non-empty)
  - spearman_correlation between native_attention and loo_importance
    (diagnostic: does the head's learned attention agree with the
    operational LOO importance? — Appendix B p.18)
  - Top-10% slice recall

Note (design doc p.12): "because leave-one-out is resource-intensive, the
user should specify which heads to run this analysis on." This module's
public API always takes an explicit list of heads — there is no "run on
everything" default.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from scipy.stats import spearmanr

from sliceheads.module3 import compute_loo_importance, save_importance_h5, IMPORTANCE_H5_FILENAME


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_ground_truth_slices(h5_path: str, sample_id: str) -> Optional[np.ndarray]:
    """
    Read the important_slices ground-truth annotation for one sample.

    Returns
    -------
    np.ndarray [N] of {0,1}, or None if the field is empty (shape [0])
    — i.e. no annotation is available for this sample (design doc p.4).
    """
    with h5py.File(h5_path, "r") as f:
        if sample_id not in f or "important_slices" not in f[sample_id]:
            return None
        arr = f[sample_id]["important_slices"][:]
        if arr.shape[0] == 0:
            return None
        return arr.astype(np.int64)


def _top_k_recall(importance: np.ndarray, ground_truth: np.ndarray,
                  fraction: float = 0.10) -> float:
    """
    Top-k% slice recall (design doc p.11): of the slices the model ranks
    as most important (top `fraction` of N), what fraction of the true
    important slices (ground_truth == 1) are captured?

    Parameters
    ----------
    importance   : np.ndarray [N] — predicted importance scores
    ground_truth : np.ndarray [N] — binary annotation, 1 = truly important
    fraction     : float — top fraction to consider (default 0.10 = top-10%)

    Returns
    -------
    float in [0, 1], or NaN if ground_truth has zero positive slices.
    """
    n_true = int(ground_truth.sum())
    if n_true == 0:
        return float("nan")

    N = len(importance)
    k = max(1, int(np.ceil(fraction * N)))
    top_k_idx = np.argsort(importance)[::-1][:k]

    hits = ground_truth[top_k_idx].sum()
    return float(hits / n_true)


def _spearman_safe(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation, returning NaN instead of raising on edge cases."""
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    corr, _ = spearmanr(a, b)
    return float(corr)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_explainability(
    heads: Dict[str, object],
    h5_path: str,
    compute_on: str = "test",
    top_k_fraction: float = 0.10,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Run the full explainability pipeline for an explicit list of heads
    (design doc p.12: LOO is resource-intensive, so heads must be named
    explicitly — there is no "run on everything" default).

    For each head, this:
      1. Computes LOO importance (Module 3's compute_loo_importance).
      2. Computes native_attention where the head supports it.
      3. Computes spearman_correlation(loo_importance, ground_truth)
         and top_k_recall(loo_importance, ground_truth) wherever
         important_slices is non-empty for that sample.
      4. Computes spearman_correlation(native_attention, loo_importance)
         as a diagnostic (Appendix B p.18) wherever the head supports
         native attention.
      5. Writes importance.h5 (importance + signed_importance +
         native_attention where available) if output_dir is given.

    Parameters
    ----------
    heads : dict {head_name: fitted_head}
        Explicit list of heads to run LOO on. Required — see note above.
    h5_path : str
        Path to the embeddings HDF5 file.
    compute_on : str
        Split to run LOO importance on (default "test"; "all" for every split).
    top_k_fraction : float
        Top-k fraction for slice recall (default 0.10 = top-10%).
    output_dir : str or None
        If given, importance.h5 is written here for each head.
    verbose : bool

    Returns
    -------
    dict {head_name: {
        "loo_importance":   {sample_id: {...}},   # from compute_loo_importance
        "per_sample":       {sample_id: {
            "spearman_vs_ground_truth": float | nan,
            "top_k_recall":             float | nan,
            "spearman_attn_vs_loo":     float | nan,
            "has_ground_truth":         bool,
            "has_native_attention":     bool,
        }},
        "summary": {
            "mean_spearman_vs_ground_truth": float,
            "mean_top_k_recall":             float,
            "mean_spearman_attn_vs_loo":     float,
            "n_samples_with_ground_truth":   int,
            "n_samples_with_native_attention": int,
        },
    }}
    """
    all_results: Dict[str, Dict] = {}

    for head_name, head in heads.items():
        if verbose:
            print(f"\n[sliceheads] Explainability: {head_name} "
                  f"(compute_on='{compute_on}')")

        # ── 1. LOO importance ────────────────────────────────────────────
        loo = compute_loo_importance(head, h5_path, compute_on=compute_on,
                                     verbose=verbose)

        # ── 2 & 3 & 4. per-sample diagnostics ────────────────────────────
        per_sample: Dict[str, Dict] = {}
        spearman_gt_vals, recall_vals, spearman_attn_vals = [], [], []

        supports_attn = getattr(head, "supports_native_attention", False)

        with h5py.File(h5_path, "r") as f:
            for sample_id, loo_data in loo.items():
                importance = loo_data["importance"]
                N          = loo_data["n_slices"]

                # vs ground truth
                gt = _load_ground_truth_slices(h5_path, sample_id)
                has_gt = gt is not None
                spearman_gt = _spearman_safe(importance, gt) if has_gt else float("nan")
                recall_k    = _top_k_recall(importance, gt, top_k_fraction) if has_gt else float("nan")

                # vs native attention
                has_attn = False
                spearman_attn = float("nan")
                if supports_attn:
                    embeddings = f[sample_id]["embeddings"][:].astype(np.float32)
                    X0    = embeddings[np.newaxis, :, :]
                    mask0 = np.ones((1, N), dtype=np.int8)
                    try:
                        attn = head.native_attention(X0, mask=mask0)
                        if attn is not None:
                            has_attn = True
                            spearman_attn = _spearman_safe(importance, attn[0])
                    except Exception as exc:
                        if verbose:
                            print(f"  [warn] native_attention failed for "
                                 f"{sample_id}: {exc}")

                per_sample[sample_id] = {
                    "spearman_vs_ground_truth": spearman_gt,
                    "top_k_recall":             recall_k,
                    "spearman_attn_vs_loo":     spearman_attn,
                    "has_ground_truth":         has_gt,
                    "has_native_attention":     has_attn,
                }

                if has_gt:
                    if not np.isnan(spearman_gt): spearman_gt_vals.append(spearman_gt)
                    if not np.isnan(recall_k):     recall_vals.append(recall_k)
                if has_attn and not np.isnan(spearman_attn):
                    spearman_attn_vals.append(spearman_attn)

        summary = {
            "mean_spearman_vs_ground_truth":   float(np.mean(spearman_gt_vals)) if spearman_gt_vals else float("nan"),
            "mean_top_k_recall":               float(np.mean(recall_vals)) if recall_vals else float("nan"),
            "mean_spearman_attn_vs_loo":        float(np.mean(spearman_attn_vals)) if spearman_attn_vals else float("nan"),
            "n_samples_with_ground_truth":      len(spearman_gt_vals),
            "n_samples_with_native_attention":  len(spearman_attn_vals),
            "n_samples_total":                  len(loo),
            "top_k_fraction":                   top_k_fraction,
        }

        if verbose:
            print(f"  → mean spearman vs ground truth : "
                  f"{summary['mean_spearman_vs_ground_truth']:.4f} "
                  f"(n={summary['n_samples_with_ground_truth']})")
            print(f"  → mean top-{int(top_k_fraction*100)}% recall          : "
                  f"{summary['mean_top_k_recall']:.4f}")
            print(f"  → mean spearman attn-vs-LOO     : "
                  f"{summary['mean_spearman_attn_vs_loo']:.4f} "
                  f"(n={summary['n_samples_with_native_attention']})")

        all_results[head_name] = {
            "loo_importance": loo,
            "per_sample":     per_sample,
            "summary":        summary,
        }

        # ── 5. persist importance.h5 (with native_attention where available) ──
        if output_dir:
            _save_importance_with_attention(loo, per_sample, head, h5_path,
                                            head_name, output_dir)

    return all_results


def _save_importance_with_attention(
    loo: Dict[str, Dict],
    per_sample: Dict[str, Dict],
    head,
    h5_path: str,
    head_name: str,
    output_dir: str,
) -> str:
    """
    Extension of Module 3's save_importance_h5 that also writes
    native_attention into importance.h5 when the head supports it
    (design doc Appendix B, Output format, p.18).
    """
    enriched = {}
    supports_attn = getattr(head, "supports_native_attention", False)

    with h5py.File(h5_path, "r") as f:
        for sample_id, loo_data in loo.items():
            entry = dict(loo_data)   # copy: importance, signed_importance, n_slices, split, patient_id

            if supports_attn and per_sample[sample_id]["has_native_attention"]:
                N = loo_data["n_slices"]
                embeddings = f[sample_id]["embeddings"][:].astype(np.float32)
                X0    = embeddings[np.newaxis, :, :]
                mask0 = np.ones((1, N), dtype=np.int8)
                attn  = head.native_attention(X0, mask=mask0)
                entry["native_attention"] = attn[0].astype(np.float32)

            enriched[sample_id] = entry

    return save_importance_h5(enriched, head_name=head_name, output_dir=output_dir)


def print_explainability_summary(all_results: Dict[str, Dict]) -> None:
    """Pretty-print the summary section for each head."""
    print(f"\n{'='*70}")
    print(f"  Explainability Summary")
    print(f"{'='*70}")
    header = (f"{'head':<20} {'spearman_gt':>12} {'top_k_recall':>12} "
              f"{'spearman_attn':>14} {'n_gt':>6} {'n_attn':>7}")
    print(header)
    print("-" * len(header))
    for head_name, res in all_results.items():
        s = res["summary"]
        print(f"{head_name:<20} "
              f"{s['mean_spearman_vs_ground_truth']:>12.4f} "
              f"{s['mean_top_k_recall']:>12.4f} "
              f"{s['mean_spearman_attn_vs_loo']:>14.4f} "
              f"{s['n_samples_with_ground_truth']:>6d} "
              f"{s['n_samples_with_native_attention']:>7d}")
    print("=" * len(header))
