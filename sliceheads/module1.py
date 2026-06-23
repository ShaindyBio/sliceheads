"""
sliceheads — Module 1: Embedding Creation and HDF5 Storage
Design Document: v0.3  |  Package: 0.1.0  |  Schema: 1

שלוש דרכים להגדיר split:
  1. קובץ טקסט  (split_file=)
  2. רשימות ידניות (train_files=, val_files=, test_files=)
  3. ברירת מחדל — הכל "train"

פורמט קובץ טקסט (split_file):
  כל שורה: <שם_קובץ> <split>
  לדוגמה:
    study_001.nii.gz  train
    study_002.nii.gz  val
    study_003.nii.gz  test
  שורות ריקות ו-# מותרים כהערות.
  גם שם בלי סיומת עובד (study_001 → train).
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import h5py
import nibabel as nib
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# ── package constants ─────────────────────────────────────────────────────────
SLICEHEADS_VERSION = "0.1.0"
SCHEMA_VERSION     = 1
VALID_SPLITS       = {"train", "val", "test"}


# ── split helpers ─────────────────────────────────────────────────────────────

def parse_split_file(path: str) -> Dict[str, str]:
    """
    Parse a plain-text split file into a {filename: split} dict.

    Accepted formats (one entry per line):
        study_001.nii.gz    train
        study_002.nii.gz    val
        study_003.nii.gz    test
        study_004           test      ← without extension also works

    Lines starting with '#' and blank lines are ignored.
    Tabs or multiple spaces between name and split are fine.

    Parameters
    ----------
    path : str
        Path to the text file.

    Returns
    -------
    dict mapping filename (with or without extension) → split string.

    Raises
    ------
    ValueError  if an unknown split label is found.
    FileNotFoundError  if the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    result: Dict[str, str] = {}
    with open(p, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                print(f"  [warn] split file line {lineno} skipped (need 2 columns): {raw!r}")
                continue
            filename, split = parts[0], parts[1].lower()
            if split not in VALID_SPLITS:
                raise ValueError(
                    f"Split file line {lineno}: unknown split '{split}'. "
                    f"Expected one of {VALID_SPLITS}."
                )
            # store both with and without extension so matching is flexible
            result[filename] = split
            stem = filename.replace(".nii.gz", "").replace(".nii", "")
            if stem != filename:
                result[stem] = split

    print(f"[sliceheads] Loaded split file: {len(result)//2 or len(result)} entries from {p.name}")
    return result


def build_split_dict(
    train_files: Optional[List[str]] = None,
    val_files:   Optional[List[str]] = None,
    test_files:  Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Build a {filename: split} dict from explicit lists of filenames.

    Each list may contain filenames with or without the .nii / .nii.gz
    extension.  Duplicates across lists raise a ValueError.

    Parameters
    ----------
    train_files : list of str or None
    val_files   : list of str or None
    test_files  : list of str or None

    Returns
    -------
    dict  {filename → "train" | "val" | "test"}

    Example
    -------
    split_dict = build_split_dict(
        train_files=["study_001.nii.gz", "study_002.nii.gz"],
        val_files=["study_003.nii.gz"],
        test_files=["study_004.nii.gz"],
    )
    """
    result: Dict[str, str] = {}
    for split, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        if not files:
            continue
        for fname in files:
            fname = str(fname).strip()
            stem  = fname.replace(".nii.gz", "").replace(".nii", "")
            for key in (fname, stem):
                if key in result and result[key] != split:
                    raise ValueError(
                        f"Duplicate entry '{key}' assigned to both "
                        f"'{result[key]}' and '{split}'."
                    )
                result[key] = split

    total = len({v for k, v in result.items()
                 if not (k.endswith(".nii.gz") or k.endswith(".nii"))})
    print(f"[sliceheads] Manual split dict built: "
          f"train={sum(1 for v in result.values() if v=='train')//2 or 0}, "
          f"val={sum(1 for v in result.values() if v=='val')//2 or 0}, "
          f"test={sum(1 for v in result.values() if v=='test')//2 or 0} "
          f"(unique stems)")
    return result


def _resolve_split(
    nii_path: Path,
    split_dict: Dict[str, str],
    default: str = "train",
) -> str:
    """Return the split for a NIfTI file, trying name then stem."""
    stem = nii_path.name.replace(".nii.gz", "").replace(".nii", "")
    return (
        split_dict.get(nii_path.name)
        or split_dict.get(stem)
        or default
    )


# ── validation ────────────────────────────────────────────────────────────────

def validate_h5(h5_path: str) -> None:
    """
    Validates an HDF5 file produced by EmbeddingPipeline.run().

    Checks (per design doc page 4):
    - every sample has embeddings
    - label exists and is an integer scalar
    - label values are exactly {0, 1}  (binary-only, v0.1)
    - split exists
    - patient_id exists (warning otherwise)
    - embeddings shape is [N, D]
    - D == file-level embedding_dim
    - important_slices is length 0 or length N
    - active splits exist
    - no patient_id leakage across splits

    Raises ValueError on hard violations; prints warnings for soft ones.
    """
    print(f"[sliceheads] Validating: {h5_path}")

    with h5py.File(h5_path, "r") as f:

        embedding_dim = int(f.attrs.get("embedding_dim", -1))
        if embedding_dim < 0:
            raise ValueError("Missing root attribute: embedding_dim")
        if "mean_embedding" not in f:
            raise ValueError("Missing root dataset: mean_embedding")

        sample_keys = sorted(k for k in f.keys() if k.startswith("sample_"))
        if not sample_keys:
            raise ValueError("No sample groups found.")

        all_labels: list = []
        split_to_pids: Dict[str, list] = {}

        for key in sample_keys:
            grp = f[key]

            if "embeddings" not in grp:
                raise ValueError(f"{key}: missing 'embeddings'.")
            emb_shape = grp["embeddings"].shape
            if len(emb_shape) != 2:
                raise ValueError(f"{key}: embeddings must be 2-D [N,D], got {emb_shape}.")
            N, D = emb_shape
            if D != embedding_dim:
                raise ValueError(f"{key}: D={D} != embedding_dim={embedding_dim}.")

            if "label" not in grp:
                raise ValueError(f"{key}: missing 'label'.")
            all_labels.append(int(grp["label"][()]))

            sp = grp.attrs.get("split")
            if sp is None:
                raise ValueError(f"{key}: missing 'split'.")
            if isinstance(sp, bytes):
                sp = sp.decode()

            pid = grp.attrs.get("patient_id")
            if pid is None:
                print(f"  [warn] {key}: missing 'patient_id'.")
                pid = key
            if isinstance(pid, bytes):
                pid = pid.decode()
            split_to_pids.setdefault(sp, []).append(pid)

            if "important_slices" in grp:
                imp_len = grp["important_slices"].shape[0]
                if imp_len not in (0, N):
                    raise ValueError(
                        f"{key}: important_slices length={imp_len}, expected 0 or {N}."
                    )

        distinct = set(all_labels)
        if distinct != {0, 1}:
            raise ValueError(
                f"Expected label values {{0,1}}, found {distinct}. "
                "sliceheads v0.1 is binary-only."
            )

        active  = set(split_to_pids.keys())
        missing = {"train", "val", "test"} - active
        if missing:
            print(f"  [warn] Missing splits: {missing}. Found: {active}")

        split_names = list(split_to_pids.keys())
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                s1, s2  = split_names[i], split_names[j]
                overlap = set(split_to_pids[s1]) & set(split_to_pids[s2])
                if overlap:
                    raise ValueError(
                        f"patient_id leakage between '{s1}' and '{s2}': {overlap}"
                    )

    print(
        f"[sliceheads] ✅ Validation passed — "
        f"{len(sample_keys)} samples, labels={distinct}, splits={active}"
    )


# ── main pipeline ─────────────────────────────────────────────────────────────

class EmbeddingPipeline:
    """
    Extracts 2D slice embeddings from 3D NIfTI volumes using a HuggingFace
    vision encoder (e.g. DINOv2) and writes a structured HDF5 file.

    Parameters
    ----------
    model_source : str
        HuggingFace model ID or local directory.
    batch_size : int
        Slices per forward pass (default 16).
    device : str or None
        "cuda", "cpu", or None (auto-detect).
    """

    def __init__(
        self,
        model_source: str,
        batch_size: int = 16,
        device: Optional[str] = None,
    ):
        self.device       = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size   = batch_size
        self.model_source = str(model_source)

        print(f"[sliceheads] Device: {self.device}")
        print(f"[sliceheads] Loading encoder: {model_source} ...")

        self.processor = AutoImageProcessor.from_pretrained(model_source)
        self.model     = AutoModel.from_pretrained(model_source).to(self.device)
        self.model.eval()

        self.model_name = os.path.basename(str(model_source).rstrip("/"))
        self._processor_wants_rgb = (
            getattr(self.processor, "image_mean", None) is not None
            and len(getattr(self.processor, "image_mean", [])) == 3
        )
        self._backbone_revision = self._get_revision(model_source)

        print(f"[sliceheads] Ready — RGB={self._processor_wants_rgb}, "
              f"revision={self._backbone_revision}")

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _get_revision(model_source: str) -> str:
        try:
            ref = Path(model_source) / "refs" / "main"
            if ref.exists():
                return ref.read_text().strip()[:12]
        except Exception:
            pass
        return "unknown"

    def _preprocess_slice(
        self, slice_2d: np.ndarray, hu_min: float, hu_max: float
    ) -> Image.Image:
        """clip → [0,255] uint8 → PIL (grayscale or RGB per processor)."""
        clipped = np.clip(slice_2d, hu_min, hu_max)
        uint8   = np.round(
            (clipped - hu_min) / (hu_max - hu_min) * 255.0
        ).astype(np.uint8)
        if self._processor_wants_rgb:
            return Image.fromarray(
                np.stack([uint8, uint8, uint8], axis=-1), mode="RGB"
            )
        return Image.fromarray(uint8, mode="L")

    @torch.no_grad()
    def _extract_volume_embeddings(
        self, volume_3d: np.ndarray, hu_min: float, hu_max: float
    ) -> np.ndarray:
        """CLS-token embeddings for all axial slices → [N, D] float32."""
        z_dim  = volume_3d.shape[2]
        images = [
            self._preprocess_slice(volume_3d[:, :, z], hu_min, hu_max)
            for z in range(z_dim)
        ]
        chunks = []
        for start in range(0, z_dim, self.batch_size):
            batch   = images[start : start + self.batch_size]
            inputs  = self.processor(images=batch, return_tensors="pt")
            inputs  = {k: v.to(self.device) for k, v in inputs.items()}
            out     = self.model(**inputs)
            chunks.append(out.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)

    # ── public API ────────────────────────────────────────────────────────────

    def run(
        self,
        input_dirs: Union[str, List[str]],
        output_h5_path: str,
        dir_to_label: Optional[Dict[str, int]] = None,
        # ── split options (pick ONE) ──────────────────────────────────────
        split_file:   Optional[str] = None,
        train_files:  Optional[List[str]] = None,
        val_files:    Optional[List[str]] = None,
        test_files:   Optional[List[str]] = None,
        # ── other options ─────────────────────────────────────────────────
        dataset_name: str   = "MosMedData",
        hu_min:       float = -1000.0,
        hu_max:       float = 400.0,
    ) -> None:
        """
        Scan input directories, extract embeddings, write HDF5.

        Split assignment — three options (choose one):
        ------------------------------------------------
        1. split_file="path/to/splits.txt"
           Plain text, one entry per line: <filename>  <split>
           Example file content:
               study_001.nii.gz  train
               study_002.nii.gz  val
               study_003.nii.gz  test
               # comments and blank lines are fine

        2. train_files / val_files / test_files  (explicit lists)
           pipeline.run(
               ...
               train_files=["study_001.nii.gz", "study_002.nii.gz"],
               val_files=["study_003.nii.gz"],
               test_files=["study_004.nii.gz"],
           )

        3. Neither provided → all samples assigned to "train".

        Parameters
        ----------
        input_dirs : str or list of str
            Directory/directories containing .nii / .nii.gz files.

        output_h5_path : str
            Destination HDF5 path.

        dir_to_label : dict or None
            {"CT-0": 0, "CT-1": 1}  — maps folder name to integer label.
            If a directory is not in the dict, label defaults to 0.

        split_file : str or None
            Path to a plain-text split file (option 1 above).

        train_files / val_files / test_files : list or None
            Explicit filename lists (option 2 above).

        dataset_name : str
            Written to HDF5 root (default "MosMedData").

        hu_min, hu_max : float
            HU clipping window (default −1000 / 400).
        """
        # ── resolve split dict ────────────────────────────────────────────
        if split_file and (train_files or val_files or test_files):
            raise ValueError(
                "Provide either split_file OR train/val/test_files, not both."
            )

        if split_file:
            split_dict = parse_split_file(split_file)
        elif train_files or val_files or test_files:
            split_dict = build_split_dict(train_files, val_files, test_files)
        else:
            split_dict = {}
            print("[sliceheads] No split definition provided — all samples → 'train'.")

        # ── normalise other args ──────────────────────────────────────────
        if isinstance(input_dirs, str):
            input_dirs = [input_dirs]
        dir_to_label   = dir_to_label or {}
        output_h5_path = Path(output_h5_path)
        output_h5_path.parent.mkdir(parents=True, exist_ok=True)

        # ── collect NIfTI files ───────────────────────────────────────────
        file_entries: List[tuple] = []   # (nii_path, label)

        for dir_str in input_dirs:
            dir_path  = Path(dir_str)
            nii_files = sorted(
                list(dir_path.glob("*.nii")) + list(dir_path.glob("*.nii.gz"))
            )
            if not nii_files:
                print(f"[sliceheads] ⚠️  No NIfTI files in: {dir_str}")
                continue

            label = (
                dir_to_label.get(str(dir_path))
                or dir_to_label.get(dir_path.name)
                or 0
            )
            print(f"[sliceheads] {dir_path.name}: {len(nii_files)} volumes, label={label}")
            for nf in nii_files:
                file_entries.append((nf, label))

        if not file_entries:
            raise FileNotFoundError(f"No NIfTI files found in: {input_dirs}")

        # ── preview split assignment ──────────────────────────────────────
        if split_dict:
            preview: Dict[str, int] = {}
            unmatched = []
            for nf, _ in file_entries:
                sp = _resolve_split(nf, split_dict, default="train")
                preview[sp] = preview.get(sp, 0) + 1
                if sp == "train" and nf.name not in split_dict and \
                        nf.name.replace(".nii.gz","").replace(".nii","") not in split_dict:
                    unmatched.append(nf.name)
            print(f"[sliceheads] Split preview: {preview}")
            if unmatched:
                print(f"  [warn] {len(unmatched)} files not found in split definition "
                      f"→ defaulting to 'train'. First few: {unmatched[:5]}")

        print(f"\n[sliceheads] Processing {len(file_entries)} volumes → {output_h5_path}")

        # ── write HDF5 ────────────────────────────────────────────────────
        all_embeddings_for_mean: List[np.ndarray] = []

        with h5py.File(output_h5_path, "w") as h5:

            # root metadata
            h5.attrs["sliceheads_version"] = SLICEHEADS_VERSION
            h5.attrs["schema_version"]     = SCHEMA_VERSION
            h5.attrs["created_at"]         = datetime.datetime.utcnow().isoformat() + "Z"
            h5.attrs["backbone_name"]      = self.model_name
            h5.attrs["backbone_source"]    = self.model_source
            h5.attrs["backbone_revision"]  = self._backbone_revision
            h5.attrs["feature_type"]       = "cls_token"
            h5.attrs["hu_window_min"]      = hu_min
            h5.attrs["hu_window_max"]      = hu_max
            h5.attrs["resize_h"]           = 224
            h5.attrs["resize_w"]           = 224
            h5.attrs["slice_axis"]         = "z"
            h5.attrs["orientation"]        = "RAS"
            h5.attrs["dataset_name"]       = dataset_name
            h5.attrs["dataset_version"]    = "1.0"
            h5.attrs["dataset_license"]    = "CC BY-NC-ND 4.0"

            for idx, (nii_path, label) in enumerate(
                tqdm(file_entries, desc="Processing Volumes")
            ):
                stem      = nii_path.name.replace(".nii.gz", "").replace(".nii", "")
                sample_id = f"sample_{idx + 1:03d}"

                try:
                    img    = nib.load(str(nii_path))
                    header = img.header
                    vol    = img.get_fdata(dtype=np.float32)
                    zooms  = header.get_zooms()

                    if vol.ndim != 3:
                        print(f"  [skip] {stem}: ndim={vol.ndim}")
                        continue

                    embeddings = self._extract_volume_embeddings(vol, hu_min, hu_max)
                    all_embeddings_for_mean.append(embeddings)

                    grp = h5.create_group(sample_id)
                    grp.attrs["patient_id"]        = stem
                    grp.attrs["split"]             = _resolve_split(nii_path, split_dict)
                    grp.attrs["original_shape_h"]  = vol.shape[0]
                    grp.attrs["original_shape_w"]  = vol.shape[1]
                    grp.attrs["original_shape_z"]  = vol.shape[2]
                    grp.attrs["slice_spacing_mm_h"] = float(zooms[0]) if len(zooms) > 0 else 1.0
                    grp.attrs["slice_spacing_mm_w"] = float(zooms[1]) if len(zooms) > 1 else 1.0
                    grp.attrs["slice_spacing_mm_z"] = float(zooms[2]) if len(zooms) > 2 else 1.0
                    grp.attrs["scanner_manufacturer"] = _safe_header_str(
                        header, "manufacturer", "unknown"
                    )
                    grp.attrs["scanner_model"] = _safe_header_str(
                        header, "scanner_model", "unknown"
                    )

                    grp.create_dataset("embeddings",       data=embeddings,              dtype="float32")
                    grp.create_dataset("label",            data=np.int64(label),         dtype="int64")
                    grp.create_dataset("important_slices", data=np.zeros(0, dtype=np.uint8), dtype="uint8")

                except Exception as exc:
                    print(f"  [error] {stem}: {exc}")

            # mean embedding — root dataset (design doc page 4)
            if all_embeddings_for_mean:
                x_bar = np.mean(
                    np.concatenate(all_embeddings_for_mean, axis=0), axis=0
                ).astype(np.float32)
                h5.attrs["embedding_dim"] = int(x_bar.shape[0])
                h5.create_dataset("mean_embedding", data=x_bar, dtype="float32")

        print(f"\n[sliceheads] ✅ Module 1 complete → {output_h5_path}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_header_str(header, field: str, default: str = "unknown") -> str:
    try:
        val = header[field]
        if hasattr(val, "tobytes"):
            return val.tobytes().decode("utf-8", errors="ignore").strip("\x00").strip()
        return str(val).strip()
    except Exception:
        return default
