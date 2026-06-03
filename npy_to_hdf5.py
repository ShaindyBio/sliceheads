"""
Converts a directory tree of NPY files into a single HDF5 file.

Each NPY file represents one volume (vol_id). The output structure is:

  /<vol_id>/
      embeddings   — dataset, shape (N_slices, D), float32
      label        — attribute, integer class label  (0 for CT-0, 1 for CT-1+2)
      split        — attribute, string "train" / "val" / "test"

Label mapping:
  CT-0    → 0
  CT-1+2  → 1

Usage:
  python npy_to_hdf5.py --input_dir binary_dataset_normal --output dataset.h5
"""

import argparse
import numpy as np
import h5py
from pathlib import Path


# Integer label for each class folder name
LABEL_MAP = {
    "CT-0": 0,
    "CT-1+2": 1,
}


def convert(input_dir: str, output_path: str, compression: str = "gzip", compression_opts: int = 4):
    splits = ["train", "val", "test"]
    input_path = Path(input_dir)

    with h5py.File(output_path, "w") as hf:
        total_vols = 0
        seen_ids: set = set()

        for split in splits:
            for class_name, label in LABEL_MAP.items():
                folder = input_path / split / class_name

                if not folder.exists():
                    print(f"⚠️  Folder not found, skipping: {folder}")
                    continue

                npy_files = sorted(folder.glob("*.npy"))

                if not npy_files:
                    print(f"⚠️  No NPY files found in: {folder}")
                    continue

                print(f"\n📁 {split}/{class_name}  ({len(npy_files)} volumes)")

                for npy_file in npy_files:
                    vol_id = npy_file.stem  # use filename (without extension) as the volume ID

                    # Warn if the same vol_id appears more than once across splits/classes
                    if vol_id in seen_ids:
                        print(f"   ⚠️  Duplicate vol_id '{vol_id}' — skipping to avoid overwrite")
                        continue
                    seen_ids.add(vol_id)

                    # Load array and cast to float32
                    arr = np.load(npy_file).astype(np.float32)

                    # Ensure shape is 2-D: (N_slices, D)
                    if arr.ndim == 1:
                        arr = arr[np.newaxis, :]   # single-slice volume → (1, D)
                    elif arr.ndim != 2:
                        raise ValueError(
                            f"Expected a 1-D or 2-D array in {npy_file}, got shape {arr.shape}"
                        )

                    # Create the per-volume group
                    grp = hf.create_group(vol_id)

                    # embeddings dataset — shape (N_slices, D), float32
                    grp.create_dataset(
                        "embeddings",
                        data=arr,
                        dtype="float32",
                        compression=compression,
                        compression_opts=compression_opts if compression == "gzip" else None,
                    )

                    # label and split stored as group attributes (lightweight metadata)
                    grp.attrs["label"] = label
                    grp.attrs["split"] = split

                    total_vols += 1
                    print(f"   ✓ {vol_id}  embeddings={arr.shape}  label={label}  split={split}")

        print(f"\n✅ Done! {total_vols} volumes written to: {output_path}")
        print_structure(hf)


def print_structure(hf: h5py.File):
    """Print a concise summary of the HDF5 file contents."""
    print("\n📊 HDF5 structure (first 5 volumes shown):")
    for i, vol_id in enumerate(list(hf.keys())[:5]):
        grp = hf[vol_id]
        emb = grp["embeddings"]
        print(
            f"  /{vol_id}/"
            f"  embeddings={emb.shape}  dtype={emb.dtype}"
            f"  label={grp.attrs['label']}  split={grp.attrs['split']}"
        )
    if len(hf.keys()) > 5:
        print(f"  ... and {len(hf.keys()) - 5} more volumes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a directory tree of NPY files into a single HDF5 file."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="binary_dataset_normal",
        help="Path to the root dataset directory (default: binary_dataset_normal)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset.h5",
        help="Output HDF5 filename (default: dataset.h5)",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="gzip",
        choices=["gzip", "lzf", "none"],
        help="Compression algorithm (default: gzip)",
    )
    parser.add_argument(
        "--compression_opts",
        type=int,
        default=4,
        help="gzip compression level 1-9 (default: 4, ignored for lzf/none)",
    )

    args = parser.parse_args()
    compression = None if args.compression == "none" else args.compression

    convert(args.input_dir, args.output, compression, args.compression_opts)
