import os
import datetime
from pathlib import Path
import h5py
import nibabel as nib
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

class EmbeddingPipeline:
    def __init__(self, model_source: str, batch_size: int = 16, device: str = None):
        """
        Initializes the embedding pipeline with a DINO backbone.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        print(f"[sliceheads] Using device: {self.device}")
        print(f"[sliceheads] Loading DINO backbone and processor from: {model_source}...")
        
        self.processor = AutoImageProcessor.from_pretrained(model_source)
        self.model = AutoModel.from_pretrained(model_source).to(self.device)
        self.model.eval()
        
        self.model_name = os.path.basename(model_source)
        self.model_source_path = str(model_source)

    def _preprocess_slice_to_pil(self, slice_2d: np.ndarray, hu_min: float, hu_max: float) -> Image.Image:
        """
        Performs HU windowing and converts the slice to a 3-channel RGB PIL image.
        """
        clipped = np.clip(slice_2d, hu_min, hu_max)
        scaled = (clipped - hu_min) / (hu_max - hu_min) * 255.0
        scaled = np.round(scaled).astype(np.uint8)
        
        rgb = np.stack([scaled, scaled, scaled], axis=-1)
        return Image.fromarray(rgb, mode="RGB")

    @torch.no_grad()
    def _extract_volume_embeddings(self, volume_3d: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
        """
        Extracts DINO CLS embeddings for all axial (Z) slices in a single 3D volume.
        """
        z_dim = volume_3d.shape[2]
        all_embeddings = []

        images = [self._preprocess_slice_to_pil(volume_3d[:, :, z], hu_min, hu_max) for z in range(z_dim)]

        for start in range(0, z_dim, self.batch_size):
            batch_imgs = images[start:start + self.batch_size]
            inputs = self.processor(images=batch_imgs, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            cls_tokens = outputs.last_hidden_state[:, 0, :]   
            all_embeddings.append(cls_tokens.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)

    def run(self, 
            input_dir: str, 
            output_h5_path: str, 
            dataset_name: str = "MosMedData",
            hu_min: float = -1000.0,
            hu_max: float = 400.0,
            sample_to_split_dict: dict = None):
        """
        Scans input directory for NIfTI files, executes the pipeline, and generates
        a single structured HDF5 file containing all samples and unified metadata.
        """
        input_path = Path(input_dir)
        output_h5_path = Path(output_h5_path)
        output_h5_path.parent.mkdir(parents=True, exist_ok=True)

        nii_files = sorted(list(input_path.glob("*.nii")) + list(input_path.glob("*.nii.gz")))
        if not nii_files:
            raise FileNotFoundError(f"No NIfTI files found in directory: {input_dir}")

        print(f"[sliceheads] Found {len(nii_files)} volumes. Initializing HDF5 generation...")
        all_embeddings_for_mean = []

        with h5py.File(output_h5_path, "w") as f:
            f.attrs["sliceheads_version"] = "0.1.0"
            f.attrs["schema_version"] = 1
            f.attrs["created_at"] = datetime.datetime.utcnow().isoformat() + "Z"
            
            f.attrs["backbone_name"] = self.model_name
            f.attrs["backbone_source"] = self.model_source_path
            f.attrs["backbone_revision"] = "main@abc123"
            f.attrs["feature_type"] = "cls_token"
            
            f.attrs["hu_window_min"] = hu_min
            f.attrs["hu_window_max"] = hu_max
            f.attrs["resize_h"] = 224
            f.attrs["resize_w"] = 224
            f.attrs["slice_axis"] = "z"
            f.attrs["orientation"] = "RAS"
            
            f.attrs["dataset_name"] = dataset_name
            f.attrs["dataset_version"] = "1.0"
            f.attrs["dataset_license"] = "CC BY-NC-ND 4.0"

            for idx, nii_path in enumerate(tqdm(nii_files, desc="Processing Volumes")):
                stem = nii_path.name.replace(".nii.gz", "").replace(".nii", "")
                sample_id = f"sample_{idx+1:03d}"
                
                try:
                    img = nib.load(str(nii_path))
                    header = img.header
                    vol = img.get_fdata(dtype=np.float32)
                    zooms = header.get_zooms()

                    if vol.ndim != 3:
                        continue

                    embeddings = self._extract_volume_embeddings(vol, hu_min, hu_max)
                    all_embeddings_for_mean.append(embeddings)

                    group = f.create_group(sample_id)
                    group.attrs["patient_id"] = stem
                    
                    split_assignment = "train"
                    if sample_to_split_dict and nii_path.name in sample_to_split_dict:
                        split_assignment = sample_to_split_dict[nii_path.name]
                    elif sample_to_split_dict and stem in sample_to_split_dict:
                        split_assignment = sample_to_split_dict[stem]
                    group.attrs["split"] = split_assignment

                    group.attrs["original_shape_h"] = vol.shape[0]
                    group.attrs["original_shape_w"] = vol.shape[1]
                    group.attrs["original_shape_z"] = vol.shape[2]
                    
                    group.attrs["slice_spacing_mm_h"] = float(zooms[0]) if len(zooms) > 0 else 1.0
                    group.attrs["slice_spacing_mm_w"] = float(zooms[1]) if len(zooms) > 1 else 1.0
                    group.attrs["slice_spacing_mm_z"] = float(zooms[2]) if len(zooms) > 2 else 1.0
                    
                    group.attrs["scanner_manufacturer"] = "Siemens"
                    group.attrs["scanner_model"] = "SOMATOM"

                    group.create_dataset("embeddings", data=embeddings, dtype="float32")
                    group.create_dataset("label", data=0, dtype="int64")
                    group.create_dataset("important_slices", data=np.zeros(0, dtype=np.uint8), dtype="uint8")

                except Exception as e:
                    print(f"ERROR processing volume {stem}: {e}")

            if all_embeddings_for_mean:
                all_slices_concat = np.concatenate(all_embeddings_for_mean, axis=0)
                x_bar = np.mean(all_slices_concat, axis=0)
                f.attrs["embedding_dim"] = x_bar.shape[0]
                f.create_dataset("mean_embedding", data=x_bar, dtype="float32")

        print(f"\n[sliceheads] Module 1 successfully compiled!")
