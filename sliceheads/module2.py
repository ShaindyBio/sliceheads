"""
sliceheads — Module 2: Classification Heads
Design Document: v0.3  |  Package: 0.1.0  |  Schema: 1

Heads implemented (full head catalogue, design doc p.7):
  sklearn (native_mask):
    - MeanPoolClassifier
    - MaxPoolClassifier
    - GeMPoolClassifier
  PyTorch (native_mask):
    - InceptionTimeClassifier   (adapter_mask — global avg pool)
    - ALSTMFCNClassifier
    - ABMILClassifier
    - GatedABMILClassifier
    - DSMILClassifier
    - TransformerMILClassifier
  aeon/sktime (adapter_mask):
    - MultiRocketHydraClassifier
"""

from __future__ import annotations

import copy
import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple

import h5py
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ── constants ─────────────────────────────────────────────────────────────────
SLICEHEADS_VERSION = "0.1.0"
SCHEMA_VERSION     = 1
InputPolicy = Literal["native_mask", "adapter_mask"]


# ─────────────────────────────────────────────────────────────────────────────
# BaseHead
# ─────────────────────────────────────────────────────────────────────────────

class BaseHead(ABC):
    """Abstract base for all sliceheads classification heads (design doc pp.7-10)."""

    input_policy: InputPolicy = "native_mask"
    supports_native_attention: bool = False

    def _requires_mask(self) -> bool:
        return self.input_policy == "native_mask"

    def _validate_mask(self, X, mask, context: str = "input") -> None:
        if self._requires_mask() and mask is None:
            raise ValueError(
                f"{self.__class__.__name__} is a native-mask head and requires "
                f"a valid mask for {context}. Received mask=None."
            )

    @abstractmethod
    def fit(self, X_train, y_train, mask_train=None,
            X_val=None, y_val=None, mask_val=None) -> "BaseHead":
        self._validate_mask(X_train, mask_train, context="training")
        if X_val is not None:
            self._validate_mask(X_val, mask_val, context="validation")

    @abstractmethod
    def predict_proba(self, X, mask=None) -> np.ndarray:
        """Return [B, 2] probability array."""
        self._validate_mask(X, mask, context="prediction")

    def predict(self, X, mask=None) -> np.ndarray:
        return self.predict_proba(X, mask=mask).argmax(axis=1)

    def native_attention(self, X, mask=None) -> Optional[np.ndarray]:
        self._validate_mask(X, mask, context="native_attention")
        return None

    def get_params(self) -> Dict[str, Any]:
        return {}

    def set_params(self, **params) -> "BaseHead":
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # ── persistence ───────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        manifest = {
            "head_class":         f"{type(self).__module__}.{type(self).__qualname__}",
            "params":             self.get_params(),
            "input_policy":       self.input_policy,
            "sliceheads_version": SLICEHEADS_VERSION,
            "schema_version":     SCHEMA_VERSION,
        }
        with open(os.path.join(path, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        self._save_backend(path)

    @classmethod
    def load(cls, path: str) -> "BaseHead":
        with open(os.path.join(path, "manifest.json")) as f:
            manifest = json.load(f)
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version mismatch: saved={manifest['schema_version']}, "
                f"expected={SCHEMA_VERSION}."
            )
        head = cls(**manifest["params"])
        head.input_policy = manifest["input_policy"]
        head._load_backend(path)
        return head

    @abstractmethod
    def _save_backend(self, path: str) -> None: ...

    @abstractmethod
    def _load_backend(self, path: str) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 loading helpers (shared)
# ─────────────────────────────────────────────────────────────────────────────

def _load_split_h5(h5_path: str, split: str):
    X_list, y_list, ids, pids = [], [], [], []
    with h5py.File(h5_path, "r") as f:
        for key in sorted(k for k in f.keys() if k.startswith("sample_")):
            grp = f[key]
            sp  = grp.attrs.get("split", "train")
            if isinstance(sp, bytes): sp = sp.decode()
            if sp != split: continue
            X_list.append(grp["embeddings"][:].astype(np.float32))
            y_list.append(int(grp["label"][()]))
            ids.append(key)
            pid = grp.attrs.get("patient_id", key)
            pids.append(pid.decode() if isinstance(pid, bytes) else str(pid))
    if not X_list:
        raise ValueError(f"No samples for split='{split}' in {h5_path}.")
    D     = X_list[0].shape[1]
    N_max = max(x.shape[0] for x in X_list)
    M     = len(X_list)
    X_pad = np.zeros((M, N_max, D), dtype=np.float32)
    mask  = np.zeros((M, N_max),    dtype=np.int8)
    for i, emb in enumerate(X_list):
        n = emb.shape[0]; X_pad[i,:n,:] = emb; mask[i,:n] = 1
    return X_pad, mask, np.array(y_list, dtype=np.int64), ids, pids


def _load_all_h5(h5_path: str):
    X_list, y_list, split_list, ids, pids = [], [], [], [], []
    with h5py.File(h5_path, "r") as f:
        for key in sorted(k for k in f.keys() if k.startswith("sample_")):
            grp = f[key]
            sp  = grp.attrs.get("split", "train")
            if isinstance(sp, bytes): sp = sp.decode()
            X_list.append(grp["embeddings"][:].astype(np.float32))
            y_list.append(int(grp["label"][()]))
            split_list.append(sp); ids.append(key)
            pid = grp.attrs.get("patient_id", key)
            pids.append(pid.decode() if isinstance(pid, bytes) else str(pid))
    if not X_list:
        raise ValueError(f"No samples found in {h5_path}.")
    D     = X_list[0].shape[1]
    N_max = max(x.shape[0] for x in X_list)
    M     = len(X_list)
    X_pad = np.zeros((M, N_max, D), dtype=np.float32)
    mask  = np.zeros((M, N_max),    dtype=np.int8)
    for i, emb in enumerate(X_list):
        n = emb.shape[0]; X_pad[i,:n,:] = emb; mask[i,:n] = 1
    return X_pad, mask, np.array(y_list, dtype=np.int64), split_list, ids, pids


# ─────────────────────────────────────────────────────────────────────────────
# sklearn pooling base
# ─────────────────────────────────────────────────────────────────────────────

class _BasePoolClassifier(BaseHead):
    """Shared base for MeanPool / MaxPool / GeMPool (sklearn backend)."""

    input_policy: InputPolicy = "native_mask"

    def __init__(self, C=1.0, class_weight="balanced",
                 max_iter=1000, random_state=42):
        self.C            = C
        self.class_weight = class_weight
        self.max_iter     = max_iter
        self.random_state = random_state
        self._pipeline: Optional[Pipeline] = None
        self._is_fitted = False

    @abstractmethod
    def _pool_single(self, embeddings_real: np.ndarray) -> np.ndarray: ...

    def _pool_batch(self, X, mask):
        B = X.shape[0]
        pooled = np.empty((B, X.shape[2]), dtype=np.float32)
        mb = mask.astype(bool)
        for i in range(B):
            real = X[i][mb[i]]
            pooled[i] = self._pool_single(real if real.shape[0] > 0 else X[i])
        return pooled

    def fit(self, X_train, y_train, mask_train=None,
            X_val=None, y_val=None, mask_val=None):
        super().fit(X_train, y_train, mask_train, X_val, y_val, mask_val)
        feats = self._pool_batch(X_train, mask_train)
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                C=self.C, class_weight=self.class_weight,
                max_iter=self.max_iter, random_state=self.random_state,
                solver="lbfgs", multi_class="ovr")),
        ])
        self._pipeline.fit(feats, y_train)
        self._is_fitted = True
        n_pos = int(y_train.sum())
        print(f"[sliceheads] {self.__class__.__name__} fitted: "
              f"{len(y_train)} samples ({n_pos}+/{len(y_train)-n_pos}-).")
        return self

    def fit_h5(self, h5_path: str):
        X, mask, y, _, _ = _load_split_h5(h5_path, "train")
        return self.fit(X, y, mask_train=mask)

    def predict_proba(self, X, mask=None) -> np.ndarray:
        super().predict_proba(X, mask)
        self._check_fitted()
        return self._pipeline.predict_proba(self._pool_batch(X, mask))

    def predict_proba_h5(self, h5_path: str) -> Dict[str, dict]:
        self._check_fitted()
        X, mask, y, splits, ids, pids = _load_all_h5(h5_path)
        proba = self.predict_proba(X, mask=mask)
        preds = proba.argmax(axis=1)
        return {
            sid: {
                "patient_id":   pids[i],
                "label":        int(y[i]),
                "prediction":   int(preds[i]),
                "prob_class_0": float(proba[i, 0]),
                "prob_class_1": float(proba[i, 1]),
                "split":        splits[i],
                "head_name":    self.__class__.__name__,
            }
            for i, sid in enumerate(ids)
        }

    def get_params(self):
        return {"C": self.C, "class_weight": self.class_weight,
                "max_iter": self.max_iter, "random_state": self.random_state}

    def _save_backend(self, path):
        self._check_fitted()
        joblib.dump(self._pipeline, os.path.join(path, "model.joblib"))

    def _load_backend(self, path):
        art = os.path.join(path, "model.joblib")
        if not os.path.exists(art):
            raise FileNotFoundError(f"model.joblib not found in {path}.")
        self._pipeline  = joblib.load(art)
        self._is_fitted = True

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} not fitted yet.")

    def __repr__(self):
        return (f"{self.__class__.__name__}(C={self.C}, "
                f"class_weight={self.class_weight!r})")


# ─────────────────────────────────────────────────────────────────────────────
# MeanPoolClassifier
# ─────────────────────────────────────────────────────────────────────────────

class MeanPoolClassifier(_BasePoolClassifier):
    """Mean-pooling baseline (design doc p.7). Input policy: native_mask."""
    def _pool_single(self, e): return e.mean(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# MaxPoolClassifier
# ─────────────────────────────────────────────────────────────────────────────

class MaxPoolClassifier(_BasePoolClassifier):
    """Max-pooling baseline (design doc p.7). Input policy: native_mask."""
    def _pool_single(self, e): return e.max(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# GeMPoolClassifier
# ─────────────────────────────────────────────────────────────────────────────

class GeMPoolClassifier(_BasePoolClassifier):
    """
    Generalized Mean Pooling (design doc p.7).
    pool(e) = mean(max(e, eps)^p)^(1/p)  over real slices.
    Input policy: native_mask. Backend: sklearn.
    """

    def __init__(self, p=3.0, eps=1e-6, C=1.0, class_weight="balanced",
                 max_iter=1000, random_state=42):
        super().__init__(C=C, class_weight=class_weight,
                         max_iter=max_iter, random_state=random_state)
        self.p   = p
        self.eps = eps

    def _pool_single(self, e: np.ndarray) -> np.ndarray:
        # GeM: (mean(max(x, eps)^p))^(1/p) — matches the reference implementation
        return np.power(
            np.mean(np.power(np.maximum(e, self.eps), self.p), axis=0),
            1.0 / self.p,
        )

    def get_params(self):
        d = super().get_params()
        d["p"]   = self.p
        d["eps"] = self.eps
        return d


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch head base
# ─────────────────────────────────────────────────────────────────────────────

class _CTDataset(Dataset):
    def __init__(self, X, y, mask):
        self.X    = torch.from_numpy(X)
        self.y    = torch.from_numpy(y).long()
        self.lens = torch.from_numpy(mask.sum(axis=1).astype(np.int64))

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.lens[idx]


def _make_collate():
    def collate(batch):
        seqs, labels, lens = zip(*batch)
        seqs   = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
        labels = torch.stack(labels)
        lens   = torch.stack(lens)
        return seqs, labels, lens
    return collate


class _BasePyTorchHead(BaseHead):
    """
    Shared base for PyTorch heads.
    Subclasses must implement _build_model(), forward() lives in the nn.Module.
    """

    input_policy: InputPolicy = "native_mask"

    def __init__(self, embedding_dim=768, num_classes=2,
                 lr=3e-4, weight_decay=1e-4,
                 max_epochs=100, patience=15,
                 batch_size=8, random_state=42,
                 dropout=0.25, device=None):
        self.embedding_dim = embedding_dim
        self.num_classes   = num_classes
        self.lr            = lr
        self.weight_decay  = weight_decay
        self.max_epochs    = max_epochs
        self.patience      = patience
        self.batch_size    = batch_size
        self.random_state  = random_state
        self.dropout       = dropout
        self._device       = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[nn.Module] = None
        self._is_fitted = False

    @abstractmethod
    def _build_model(self) -> nn.Module: ...

    def _make_loader(self, X, y, mask, shuffle=False, balanced=False):
        ds = _CTDataset(X, y, mask)
        sampler = None
        if balanced:
            counts  = np.bincount(y)
            weights = 1.0 / counts[y]
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            shuffle = False
        return DataLoader(ds, batch_size=self.batch_size,
                          shuffle=shuffle, sampler=sampler,
                          collate_fn=_make_collate())

    def _class_weights(self, y):
        counts  = np.bincount(y)
        weights = len(y) / (2.0 * counts)
        return torch.tensor(weights, dtype=torch.float32).to(self._device)

    def fit(self, X_train, y_train, mask_train=None,
            X_val=None, y_val=None, mask_val=None):
        super().fit(X_train, y_train, mask_train, X_val, y_val, mask_val)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._model = self._build_model().to(self._device)
        criterion   = nn.CrossEntropyLoss(weight=self._class_weights(y_train))
        optimizer   = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        train_loader = self._make_loader(X_train, y_train, mask_train, balanced=True)
        val_loader   = (self._make_loader(X_val, y_val, mask_val)
                        if X_val is not None else None)

        best_auc     = -1.0
        best_state   = None
        patience_cnt = 0

        for epoch in range(self.max_epochs):
            self._model.train()
            for bx, by, bl in train_loader:
                bx, by, bl = bx.to(self._device), by.to(self._device), bl.to(self._device)
                optimizer.zero_grad()
                loss = criterion(self._forward(bx, bl), by)
                loss.backward(); optimizer.step()

            if val_loader is not None:
                val_auc = self._eval_auc(val_loader)
                if val_auc > best_auc:
                    best_auc   = val_auc
                    best_state = copy.deepcopy(self._model.state_dict())
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                if patience_cnt >= self.patience:
                    print(f"[sliceheads] {self.__class__.__name__} "
                          f"early stop at epoch {epoch+1} (val AUC={best_auc:.4f})")
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._is_fitted = True
        print(f"[sliceheads] {self.__class__.__name__} fitted.")
        return self

    def fit_h5(self, h5_path: str):
        X_tr, m_tr, y_tr, _, _ = _load_split_h5(h5_path, "train")
        X_vl, m_vl, y_vl, _, _ = _load_split_h5(h5_path, "val")
        return self.fit(X_tr, y_tr, mask_train=m_tr,
                        X_val=X_vl, y_val=y_vl, mask_val=m_vl)

    def _forward(self, bx, bl):
        return self._model(bx, bl)

    @torch.no_grad()
    def _eval_auc(self, loader):
        from sklearn.metrics import roc_auc_score
        self._model.eval()
        all_labels, all_probs = [], []
        for bx, by, bl in loader:
            bx, bl = bx.to(self._device), bl.to(self._device)
            probs  = F.softmax(self._forward(bx, bl), dim=1)[:, 1]
            all_labels.extend(by.numpy()); all_probs.extend(probs.cpu().numpy())
        try:
            return roc_auc_score(all_labels, all_probs)
        except Exception:
            return 0.5

    @torch.no_grad()
    def predict_proba(self, X, mask=None) -> np.ndarray:
        super().predict_proba(X, mask)
        self._check_fitted()
        self._model.eval()
        loader = self._make_loader(X, np.zeros(X.shape[0], dtype=np.int64), mask)
        probs  = []
        for bx, _, bl in loader:
            bx, bl = bx.to(self._device), bl.to(self._device)
            probs.append(F.softmax(self._forward(bx, bl), dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0)

    def predict_proba_h5(self, h5_path: str) -> Dict[str, dict]:
        self._check_fitted()
        X, mask, y, splits, ids, pids = _load_all_h5(h5_path)
        proba = self.predict_proba(X, mask=mask)
        preds = proba.argmax(axis=1)
        return {
            sid: {
                "patient_id":   pids[i], "label": int(y[i]),
                "prediction":   int(preds[i]),
                "prob_class_0": float(proba[i,0]),
                "prob_class_1": float(proba[i,1]),
                "split":        splits[i],
                "head_name":    self.__class__.__name__,
            }
            for i, sid in enumerate(ids)
        }

    def get_params(self):
        return dict(embedding_dim=self.embedding_dim, num_classes=self.num_classes,
                    lr=self.lr, weight_decay=self.weight_decay,
                    max_epochs=self.max_epochs, patience=self.patience,
                    batch_size=self.batch_size, random_state=self.random_state,
                    dropout=self.dropout)

    def _save_backend(self, path):
        self._check_fitted()
        torch.save(self._model.state_dict(), os.path.join(path, "model_state.pt"))

    def _load_backend(self, path):
        art = os.path.join(path, "model_state.pt")
        if not os.path.exists(art):
            raise FileNotFoundError(f"model_state.pt not found in {path}.")
        self._model = self._build_model().to(self._device)
        self._model.load_state_dict(
            torch.load(art, map_location=self._device))
        self._model.eval()
        self._is_fitted = True

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} not fitted yet.")


# ─────────────────────────────────────────────────────────────────────────────
# InceptionTime
# ─────────────────────────────────────────────────────────────────────────────

class _InceptionBlock1D(nn.Module):
    def __init__(self, in_ch, n_filters, kernel_sizes, bottleneck=32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, bottleneck, 1, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck, n_filters, k, padding=k//2, bias=False)
            for k in kernel_sizes
        ])
        self.maxpool  = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_ch, n_filters, 1, bias=False)
        self.bn        = nn.BatchNorm1d(n_filters * 4)

    def forward(self, x):
        z    = self.bottleneck(x)
        outs = [c(z) for c in self.convs] + [self.conv_pool(self.maxpool(x))]
        return F.relu(self.bn(torch.cat(outs, dim=1)))


class _InceptionTimeNet(nn.Module):
    def __init__(self, embedding_dim, n_filters=32, depth=3,
                 kernel_sizes=(9,19,39), num_classes=2, dropout_rate=0.25):
        super().__init__()
        in_ch  = embedding_dim
        blocks = []
        for _ in range(depth):
            blocks.append(_InceptionBlock1D(in_ch, n_filters, kernel_sizes))
            in_ch = n_filters * 4
        self.blocks  = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc      = nn.Linear(in_ch, num_classes)

    def forward(self, x, lengths):
        x = x.transpose(1, 2)          # [B, D, N]
        for blk in self.blocks: x = blk(x)
        x = x.mean(dim=2)              # global avg pool
        return self.fc(self.dropout(x))


class InceptionTimeClassifier(_BasePyTorchHead):
    """
    Multi-scale 1D CNN (InceptionTime) — design doc p.7.
    Input policy : adapter_mask  (global avg pool handles padding).
    Backend      : PyTorch.
    """

    input_policy: InputPolicy = "adapter_mask"

    def __init__(self, embedding_dim=768, n_filters=32, depth=3,
                 kernel_sizes=(9,19,39), num_classes=2, dropout=0.25,
                 lr=3e-4, weight_decay=1e-4, max_epochs=100, patience=15,
                 batch_size=8, random_state=42, device=None):
        super().__init__(embedding_dim=embedding_dim, num_classes=num_classes,
                         lr=lr, weight_decay=weight_decay,
                         max_epochs=max_epochs, patience=patience,
                         batch_size=batch_size, random_state=random_state,
                         dropout=dropout, device=device)
        self.n_filters    = n_filters
        self.depth        = depth
        self.kernel_sizes = list(kernel_sizes)

    def _build_model(self):
        return _InceptionTimeNet(
            self.embedding_dim, self.n_filters, self.depth,
            self.kernel_sizes, self.num_classes, self.dropout)

    def get_params(self):
        d = super().get_params()
        d.update(n_filters=self.n_filters, depth=self.depth,
                 kernel_sizes=self.kernel_sizes)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# ALSTMFCN
# ─────────────────────────────────────────────────────────────────────────────

class _ALSTMFCNNet(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=128,
                 num_classes=2, dropout_rate=0.25):
        super().__init__()
        # FCN branch
        self.conv1 = nn.Conv1d(embedding_dim, 128, 8, padding="same")
        self.bn1   = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, 5, padding="same")
        self.bn2   = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, 3, padding="same")
        self.bn3   = nn.BatchNorm1d(128)
        # LSTM + attention branch
        self.lstm            = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,
                                       bidirectional=True)
        self.attn_lin        = nn.Linear(hidden_dim*2, 1)
        self.ch_squeeze      = nn.Linear(hidden_dim*2, (hidden_dim*2)//4)
        self.ch_excite       = nn.Linear((hidden_dim*2)//4, hidden_dim*2)
        self.dropout         = nn.Dropout(dropout_rate)
        self.fc              = nn.Linear(128 + hidden_dim*2, num_classes)

    def forward(self, x, lengths):
        # FCN
        f = F.relu(self.bn1(self.conv1(x.transpose(1,2))))
        f = F.relu(self.bn2(self.conv2(f)))
        f = F.relu(self.bn3(self.conv3(f)))
        fcn_feat = f.mean(dim=2)
        # LSTM
        packed   = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_p, _ = self.lstm(packed)
        out, _   = nn.utils.rnn.pad_packed_sequence(out_p, batch_first=True)
        # attention
        scores   = self.attn_lin(out)
        mask     = torch.arange(out.size(1), device=out.device)[None,:] < lengths[:,None]
        scores   = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        ctx      = (out * F.softmax(scores, dim=1)).sum(dim=1)
        # channel reweighting
        sq       = F.relu(self.ch_squeeze(ctx))
        lstm_feat = ctx * torch.sigmoid(self.ch_excite(sq))
        fused    = torch.cat([fcn_feat, lstm_feat], dim=1)
        return self.fc(self.dropout(fused))


class ALSTMFCNClassifier(_BasePyTorchHead):
    """
    LSTM + FCN hybrid (design doc p.7).
    Input policy : native_mask  (LSTM uses pack_padded_sequence).
    Backend      : PyTorch.
    """

    supports_native_attention: bool = True

    def __init__(self, embedding_dim=768, hidden_dim=128, num_classes=2,
                 dropout=0.25, lr=3e-4, weight_decay=1e-4,
                 max_epochs=100, patience=15, batch_size=8,
                 random_state=42, device=None):
        super().__init__(embedding_dim=embedding_dim, num_classes=num_classes,
                         lr=lr, weight_decay=weight_decay,
                         max_epochs=max_epochs, patience=patience,
                         batch_size=batch_size, random_state=random_state,
                         dropout=dropout, device=device)
        self.hidden_dim = hidden_dim

    def _build_model(self):
        return _ALSTMFCNNet(self.embedding_dim, self.hidden_dim,
                            self.num_classes, self.dropout)

    def get_params(self):
        d = super().get_params()
        d["hidden_dim"] = self.hidden_dim
        return d


# ─────────────────────────────────────────────────────────────────────────────
# MultiRocketHydra  (aeon adapter-mask head)
# ─────────────────────────────────────────────────────────────────────────────

class MultiRocketHydraClassifier(BaseHead):
    """
    MultiRocket feature transform + linear classifier (design doc p.7).
    Input policy : adapter_mask  (adapter crops real slices before transform).
    Backend      : aeon (sktime MultiRocket) + sklearn LogisticRegression.
    """

    input_policy: InputPolicy = "adapter_mask"

    def __init__(self, n_kernels=6250, C=1.0, class_weight="balanced",
                 max_iter=1000, random_state=42):
        self.n_kernels    = n_kernels
        self.C            = C
        self.class_weight = class_weight
        self.max_iter     = max_iter
        self.random_state = random_state
        self._rocket     = None
        self._clf        = None
        self._fixed_length = None
        self._is_fitted  = False

    def _import_rocket(self):
        # design doc (p.6-7): "MultiRocketHydra uses the aeon ROCKET/Hydra
        # feature transforms (numba-based)" — aeon is the canonical backend.
        try:
            from aeon.transformations.collection.convolution_based import MultiRocket
            return MultiRocket
        except ImportError:
            pass
        try:
            from sktime.transformations.panel.rocket import MultiRocket
            print("[sliceheads] WARNING: 'aeon' not found, falling back to "
                  "'sktime'. Design doc specifies aeon as canonical backend; "
                  "install with: pip install aeon")
            return MultiRocket
        except ImportError:
            raise ImportError(
                "MultiRocketHydraClassifier requires 'aeon' (design doc "
                "canonical backend) or 'sktime' as a fallback. "
                "Install with: pip install aeon"
            )

    def _to_sktime_format(self, X: np.ndarray, mask: np.ndarray,
                          fixed_length: Optional[int] = None) -> np.ndarray:
        """
        Convert [B, N_max, D] + mask to sktime/aeon panel format [B, D, N].

        Each sample is first cropped to its real (unpadded) slices via mask.
        MultiRocket requires EQUAL-LENGTH series across fit and predict, so
        all samples are then cropped/padded to `fixed_length`:
          - if fixed_length is None (i.e. during fit), it is set to the
            longest real sequence in this batch and learned for later use.
          - if fixed_length is given (i.e. during predict), every sample is
            cropped (if longer) or zero-padded (if shorter) to match it.
        """
        B   = X.shape[0]
        mb  = mask.astype(bool)
        seqs = [X[i][mb[i]].T for i in range(B)]     # each [D, N_real_i]
        D    = seqs[0].shape[0]

        N = fixed_length if fixed_length is not None else max(s.shape[1] for s in seqs)

        out = np.zeros((B, D, N), dtype=np.float32)
        for i, s in enumerate(seqs):
            n = min(s.shape[1], N)    # crop if longer than N, else copy all
            out[i, :, :n] = s[:, :n]
        return out

    def fit(self, X_train, y_train, mask_train=None,
            X_val=None, y_val=None, mask_val=None):
        # adapter_mask: no mask validation needed here
        MR = self._import_rocket()
        self._rocket = self._construct_rocket(MR)

        mask_used = (mask_train if mask_train is not None
                    else np.ones(X_train.shape[:2], dtype=np.int8))

        # Learn the fixed sequence length from the train set's real slices,
        # so predict() can crop/pad to the SAME length (MultiRocket requires
        # equal-length series across fit and predict).
        real_lengths = mask_used.astype(bool).sum(axis=1)
        self._fixed_length = int(real_lengths.max())

        panel = self._to_sktime_format(X_train, mask_used,
                                       fixed_length=self._fixed_length)
        feats = self._rocket.fit_transform(panel)
        self._clf = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                C=self.C, class_weight=self.class_weight,
                max_iter=self.max_iter, random_state=self.random_state,
                solver="lbfgs")),
        ])
        self._clf.fit(feats, y_train)
        self._is_fitted = True
        print(f"[sliceheads] MultiRocketHydraClassifier fitted "
              f"(fixed_length={self._fixed_length}).")
        return self

    def _construct_rocket(self, MR):
        """
        Both aeon and sktime expose a MultiRocket transformer, but the
        constructor kwarg differs: aeon uses 'n_kernels', sktime uses
        'num_kernels'. Try aeon's signature first (canonical backend).
        """
        try:
            return MR(n_kernels=self.n_kernels, random_state=self.random_state)
        except TypeError:
            return MR(num_kernels=self.n_kernels, random_state=self.random_state)

    def fit_h5(self, h5_path: str):
        X, mask, y, _, _ = _load_split_h5(h5_path, "train")
        return self.fit(X, y, mask_train=mask)

    def predict_proba(self, X, mask=None) -> np.ndarray:
        self._check_fitted()
        m     = mask if mask is not None else np.ones(X.shape[:2], dtype=np.int8)
        panel = self._to_sktime_format(X, m, fixed_length=self._fixed_length)
        feats = self._rocket.transform(panel)
        return self._clf.predict_proba(feats)

    def predict_proba_h5(self, h5_path: str) -> Dict[str, dict]:
        self._check_fitted()
        X, mask, y, splits, ids, pids = _load_all_h5(h5_path)
        proba = self.predict_proba(X, mask=mask)
        preds = proba.argmax(axis=1)
        return {
            sid: {
                "patient_id": pids[i], "label": int(y[i]),
                "prediction": int(preds[i]),
                "prob_class_0": float(proba[i,0]),
                "prob_class_1": float(proba[i,1]),
                "split": splits[i],
                "head_name": self.__class__.__name__,
            }
            for i, sid in enumerate(ids)
        }

    def get_params(self):
        return dict(n_kernels=self.n_kernels, C=self.C,
                    class_weight=self.class_weight,
                    max_iter=self.max_iter, random_state=self.random_state)

    def _save_backend(self, path):
        self._check_fitted()
        joblib.dump({"rocket": self._rocket, "clf": self._clf,
                    "fixed_length": self._fixed_length},
                    os.path.join(path, "model.joblib"))

    def _load_backend(self, path):
        art = os.path.join(path, "model.joblib")
        if not os.path.exists(art): raise FileNotFoundError(art)
        d = joblib.load(art)
        self._rocket       = d["rocket"]
        self._clf          = d["clf"]
        self._fixed_length = d["fixed_length"]
        self._is_fitted    = True

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("MultiRocketHydraClassifier not fitted yet.")


# ─────────────────────────────────────────────────────────────────────────────
# ABMIL
# ─────────────────────────────────────────────────────────────────────────────

class _ABMILNet(nn.Module):
    """
    Attention-Based MIL (Ilse et al. 2018).
    Attention: a_k = softmax( w^T tanh(V h_k) )
    Bag repr:  z   = sum_k a_k * h_k
    """

    def __init__(self, embedding_dim: int, hidden_dim: int,
                 num_classes: int, dropout_rate: float):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.attention_V = nn.Linear(hidden_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1, bias=False)
        self.classifier  = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        x       : [B, N, D]
        lengths : [B]  — number of real (non-padded) slices per sample
        returns : logits [B, num_classes]
                  attentions [B, N]  (softmax weights, padded positions ≈ 0)
        """
        B, N, _ = x.shape
        h = self.feature_proj(x)                          # [B, N, H]

        # Build boolean mask: True = real slice
        mask = torch.arange(N, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)  # [B, N]

        scores = self.attention_w(torch.tanh(self.attention_V(h)))  # [B, N, 1]
        scores = scores.squeeze(-1)                                  # [B, N]
        scores = scores.masked_fill(~mask, float("-inf"))
        attn   = F.softmax(scores, dim=1)                           # [B, N]
        attn   = torch.nan_to_num(attn, nan=0.0)

        z      = (attn.unsqueeze(-1) * h).sum(dim=1)               # [B, H]
        return self.classifier(z), attn

    def attention_only(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return only attention weights (for native_attention)."""
        _, attn = self.forward(x, lengths)
        return attn


class ABMILClassifier(_BasePyTorchHead):
    """
    Attention-Based MIL (design doc p.7).
    Input policy : native_mask (mask applied before attention softmax).
    Backend      : PyTorch.
    supports_native_attention = True
    """

    input_policy: InputPolicy              = "native_mask"
    supports_native_attention: bool        = True

    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 128,
                 num_classes: int = 2, dropout: float = 0.25,
                 lr: float = 3e-4, weight_decay: float = 1e-4,
                 max_epochs: int = 100, patience: int = 15,
                 batch_size: int = 8, random_state: int = 42,
                 device: Optional[str] = None):
        super().__init__(
            embedding_dim=embedding_dim, num_classes=num_classes,
            lr=lr, weight_decay=weight_decay,
            max_epochs=max_epochs, patience=patience,
            batch_size=batch_size, random_state=random_state,
            dropout=dropout, device=device,
        )
        self.hidden_dim = hidden_dim

    def _build_model(self) -> nn.Module:
        return _ABMILNet(self.embedding_dim, self.hidden_dim,
                         self.num_classes, self.dropout)

    def _forward(self, bx: torch.Tensor, bl: torch.Tensor) -> torch.Tensor:
        logits, _ = self._model(bx, bl)
        return logits

    @torch.no_grad()
    def native_attention(self, X: np.ndarray,
                         mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Return attention weights [B, N] (padded positions ≈ 0)."""
        super().native_attention(X, mask)
        self._check_fitted()
        self._model.eval()
        loader = self._make_loader(
            X, np.zeros(X.shape[0], dtype=np.int64), mask)
        attn_list = []
        for bx, _, bl in loader:
            bx, bl = bx.to(self._device), bl.to(self._device)
            _, attn = self._model(bx, bl)
            attn_list.append(attn.cpu().numpy())
        return np.concatenate(attn_list, axis=0)          # [B, N]

    def get_params(self) -> Dict:
        d = super().get_params()
        d["hidden_dim"] = self.hidden_dim
        return d


# ─────────────────────────────────────────────────────────────────────────────
# GatedABMIL
# ─────────────────────────────────────────────────────────────────────────────

class _GatedABMILNet(nn.Module):
    """
    Gated Attention-Based MIL (Ilse et al. 2018, gating variant).
    Attention: a_k = softmax( w^T (tanh(V h_k) ⊙ sigmoid(U h_k)) )
    """

    def __init__(self, embedding_dim: int, hidden_dim: int,
                 num_classes: int, dropout_rate: float):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.attention_V = nn.Linear(hidden_dim, hidden_dim)
        self.attention_U = nn.Linear(hidden_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1, bias=False)
        self.classifier  = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        B, N, _ = x.shape
        h = self.feature_proj(x)                            # [B, N, H]

        mask = torch.arange(N, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)

        gate   = torch.tanh(self.attention_V(h)) * torch.sigmoid(self.attention_U(h))
        scores = self.attention_w(gate).squeeze(-1)         # [B, N]
        scores = scores.masked_fill(~mask, float("-inf"))
        attn   = F.softmax(scores, dim=1)
        attn   = torch.nan_to_num(attn, nan=0.0)

        z = (attn.unsqueeze(-1) * h).sum(dim=1)            # [B, H]
        return self.classifier(z), attn


class GatedABMILClassifier(_BasePyTorchHead):
    """
    Gated Attention-Based MIL (design doc p.7).
    Input policy : native_mask.
    Backend      : PyTorch.
    supports_native_attention = True
    """

    input_policy: InputPolicy       = "native_mask"
    supports_native_attention: bool = True

    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 128,
                 num_classes: int = 2, dropout: float = 0.25,
                 lr: float = 3e-4, weight_decay: float = 1e-4,
                 max_epochs: int = 100, patience: int = 15,
                 batch_size: int = 8, random_state: int = 42,
                 device: Optional[str] = None):
        super().__init__(
            embedding_dim=embedding_dim, num_classes=num_classes,
            lr=lr, weight_decay=weight_decay,
            max_epochs=max_epochs, patience=patience,
            batch_size=batch_size, random_state=random_state,
            dropout=dropout, device=device,
        )
        self.hidden_dim = hidden_dim

    def _build_model(self) -> nn.Module:
        return _GatedABMILNet(self.embedding_dim, self.hidden_dim,
                              self.num_classes, self.dropout)

    def _forward(self, bx, bl):
        logits, _ = self._model(bx, bl)
        return logits

    @torch.no_grad()
    def native_attention(self, X, mask=None) -> np.ndarray:
        super().native_attention(X, mask)
        self._check_fitted()
        self._model.eval()
        loader = self._make_loader(X, np.zeros(X.shape[0], dtype=np.int64), mask)
        attn_list = []
        for bx, _, bl in loader:
            bx, bl = bx.to(self._device), bl.to(self._device)
            _, attn = self._model(bx, bl)
            attn_list.append(attn.cpu().numpy())
        return np.concatenate(attn_list, axis=0)

    def get_params(self):
        d = super().get_params()
        d["hidden_dim"] = self.hidden_dim
        return d


# ─────────────────────────────────────────────────────────────────────────────
# DSMIL
# ─────────────────────────────────────────────────────────────────────────────

class _DSMILNet(nn.Module):
    """
    Dual-Stream MIL (Li et al. 2021).
    Stream 1 — instance classifier selects the critical instance.
    Stream 2 — bag classifier aggregates all instances weighted by
               distance to the critical instance.
    Final loss = lambda_instance * L_instance + L_bag  (combined in head).
    """

    def __init__(self, embedding_dim: int, hidden_dim: int,
                 num_classes: int, dropout_rate: float):
        super().__init__()
        self.feature_proj       = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        # Instance classifier (stream 1)
        self.instance_clf       = nn.Linear(hidden_dim, num_classes)
        # Bag-level attention (stream 2)
        self.bag_attention_V    = nn.Linear(hidden_dim, hidden_dim)
        self.bag_attention_w    = nn.Linear(hidden_dim, 1, bias=False)
        self.bag_clf            = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        Returns
        -------
        bag_logits      : [B, C]
        instance_logits : [B, N, C]  (use max over real slices for instance loss)
        attn            : [B, N]
        """
        B, N, _ = x.shape
        h = self.feature_proj(x)                            # [B, N, H]

        mask = torch.arange(N, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)

        instance_logits = self.instance_clf(h)              # [B, N, C]

        # Critical instance = argmax positive-class instance score among real slices
        pos_scores  = instance_logits[..., 1].masked_fill(~mask, float("-inf"))  # [B, N]
        crit_idx    = pos_scores.argmax(dim=1)              # [B]
        crit_h      = h[torch.arange(B), crit_idx]         # [B, H]

        # Bag attention: distance of each instance to critical instance
        # a_k = softmax( w^T tanh(V(h_k - h_crit)) )
        diff   = h - crit_h.unsqueeze(1)                   # [B, N, H]
        scores = self.bag_attention_w(torch.tanh(self.bag_attention_V(diff))).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn   = F.softmax(scores, dim=1)
        attn   = torch.nan_to_num(attn, nan=0.0)

        z          = (attn.unsqueeze(-1) * h).sum(dim=1)   # [B, H]
        bag_logits = self.bag_clf(z)

        return bag_logits, instance_logits, attn


class DSMILClassifier(_BasePyTorchHead):
    """
    Dual-Stream MIL (design doc p.7).
    Input policy : native_mask.
    Backend      : PyTorch.
    supports_native_attention = True

    lambda_instance weights the instance-level auxiliary loss added to the
    bag-level loss during training (0 = bag only; 1 = equal weighting).
    """

    input_policy: InputPolicy       = "native_mask"
    supports_native_attention: bool = True

    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 128,
                 num_classes: int = 2, dropout: float = 0.25,
                 lambda_instance: float = 0.5,
                 lr: float = 3e-4, weight_decay: float = 1e-4,
                 max_epochs: int = 100, patience: int = 15,
                 batch_size: int = 8, random_state: int = 42,
                 device: Optional[str] = None):
        super().__init__(
            embedding_dim=embedding_dim, num_classes=num_classes,
            lr=lr, weight_decay=weight_decay,
            max_epochs=max_epochs, patience=patience,
            batch_size=batch_size, random_state=random_state,
            dropout=dropout, device=device,
        )
        self.hidden_dim       = hidden_dim
        self.lambda_instance  = lambda_instance

    def _build_model(self) -> nn.Module:
        return _DSMILNet(self.embedding_dim, self.hidden_dim,
                         self.num_classes, self.dropout)

    def _forward(self, bx: torch.Tensor, bl: torch.Tensor) -> torch.Tensor:
        bag_logits, _, _ = self._model(bx, bl)
        return bag_logits

    def _dsmil_loss(self, bx, bl, by, criterion):
        """Combined bag + instance loss used during training."""
        bag_logits, inst_logits, _ = self._model(bx, bl)
        bag_loss = criterion(bag_logits, by)

        if self.lambda_instance > 0.0:
            # Instance loss: max-pooled positive-class logit among real slices
            B, N, _ = inst_logits.shape
            mask = torch.arange(N, device=bx.device).unsqueeze(0) < bl.unsqueeze(1)
            # Assign each instance the bag label (weak supervision)
            inst_labels = by.unsqueeze(1).expand(-1, N)        # [B, N]
            # Flatten, keep only real instances
            inst_flat   = inst_logits.reshape(B * N, -1)
            label_flat  = inst_labels.reshape(B * N)
            mask_flat   = mask.reshape(B * N)
            inst_loss   = criterion(inst_flat[mask_flat], label_flat[mask_flat])
            return bag_loss + self.lambda_instance * inst_loss

        return bag_loss

    # Override the training loop's loss computation
    def fit(self, X_train, y_train, mask_train=None,
            X_val=None, y_val=None, mask_val=None):
        """Override to use dual-stream loss."""
        # NOTE: BaseHead is defined earlier in this same module — no import
        # needed (this previously referenced a non-existent 'sliceheads_heads'
        # module, which has been removed).
        BaseHead.fit(self, X_train, y_train, mask_train, X_val, y_val, mask_val)

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._model = self._build_model().to(self._device)
        criterion   = nn.CrossEntropyLoss(weight=self._class_weights(y_train))
        optimizer   = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        train_loader = self._make_loader(X_train, y_train, mask_train, balanced=True)
        val_loader   = (self._make_loader(X_val, y_val, mask_val)
                        if X_val is not None else None)

        best_auc, best_state, patience_cnt = -1.0, None, 0

        for epoch in range(self.max_epochs):
            self._model.train()
            for bx, by, bl in train_loader:
                bx = bx.to(self._device)
                by = by.to(self._device)
                bl = bl.to(self._device)
                optimizer.zero_grad()
                loss = self._dsmil_loss(bx, bl, by, criterion)
                loss.backward()
                optimizer.step()

            if val_loader is not None:
                val_auc = self._eval_auc(val_loader)
                if val_auc > best_auc:
                    best_auc, patience_cnt = val_auc, 0
                    best_state = copy.deepcopy(self._model.state_dict())
                else:
                    patience_cnt += 1
                if patience_cnt >= self.patience:
                    print(f"[sliceheads] DSMILClassifier early stop "
                          f"epoch {epoch+1} (val AUC={best_auc:.4f})")
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._is_fitted = True
        print("[sliceheads] DSMILClassifier fitted.")
        return self

    @torch.no_grad()
    def native_attention(self, X, mask=None) -> np.ndarray:
        super().native_attention(X, mask)
        self._check_fitted()
        self._model.eval()
        loader = self._make_loader(X, np.zeros(X.shape[0], dtype=np.int64), mask)
        attn_list = []
        for bx, _, bl in loader:
            bx, bl = bx.to(self._device), bl.to(self._device)
            _, _, attn = self._model(bx, bl)
            attn_list.append(attn.cpu().numpy())
        return np.concatenate(attn_list, axis=0)

    def get_params(self):
        d = super().get_params()
        d["hidden_dim"]      = self.hidden_dim
        d["lambda_instance"] = self.lambda_instance
        return d


# ─────────────────────────────────────────────────────────────────────────────
# TransformerMIL
# ─────────────────────────────────────────────────────────────────────────────

class _LearnedPositionalEncoding(nn.Module):
    """Learned positional embedding (design doc: positional_encoding_transformer_encoder)."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D]"""
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # [1, N]
        return x + self.pe(positions)


class _TransformerMILNet(nn.Module):
    """
    Transformer encoder over slice embeddings (design doc p.7).
    Supports three pooling modes: 'cls', 'mean', 'attention'.
    """

    def __init__(self, embedding_dim: int, d_model: int, num_layers: int,
                 n_heads: int, dropout_rate: float, num_classes: int,
                 pooling: str = "cls", max_len: int = 2048):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

        self.pooling      = pooling
        self.input_proj   = nn.Linear(embedding_dim, d_model)
        self.pos_enc      = _LearnedPositionalEncoding(d_model, max_len)
        self.dropout      = nn.Dropout(dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True,           # Pre-LN for training stability
        )
        self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # [CLS] token for cls pooling
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Learnable attention pooling head
        if pooling == "attention":
            self.pool_attn_V = nn.Linear(d_model, d_model)
            self.pool_attn_w = nn.Linear(d_model, 1, bias=False)

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        x       : [B, N, D]
        lengths : [B]
        returns : logits [B, C], pool_attn [B, N] (or None)
        """
        B, N, _ = x.shape

        x = self.dropout(self.pos_enc(self.input_proj(x)))   # [B, N, d_model]

        # key_padding_mask: True = padding (ignored by MultiheadAttention)
        real_mask = (torch.arange(N, device=x.device).unsqueeze(0)
                     < lengths.unsqueeze(1))                  # [B, N]  True = real

        pool_attn = None

        if self.pooling == "cls":
            cls     = self.cls_token.expand(B, -1, -1)       # [B, 1, d_model]
            x       = torch.cat([cls, x], dim=1)             # [B, N+1, d_model]
            # Prepend False (real) for the CLS position
            cls_col = torch.ones(B, 1, dtype=torch.bool, device=x.device)
            pad_mask = ~torch.cat([cls_col, real_mask], dim=1)  # True = pad
            x       = self.encoder(x, src_key_padding_mask=pad_mask)
            z       = x[:, 0]                                # [B, d_model]

        elif self.pooling == "mean":
            pad_mask = ~real_mask
            x        = self.encoder(x, src_key_padding_mask=pad_mask)
            # Masked mean over real positions
            lens_f   = lengths.float().clamp(min=1).unsqueeze(-1)
            z        = (x * real_mask.unsqueeze(-1).float()).sum(dim=1) / lens_f

        elif self.pooling == "attention":
            pad_mask = ~real_mask
            x        = self.encoder(x, src_key_padding_mask=pad_mask)
            scores   = self.pool_attn_w(
                torch.tanh(self.pool_attn_V(x))).squeeze(-1)  # [B, N]
            scores   = scores.masked_fill(~real_mask, float("-inf"))
            pool_attn = F.softmax(scores, dim=1)
            pool_attn = torch.nan_to_num(pool_attn, nan=0.0)
            z         = (pool_attn.unsqueeze(-1) * x).sum(dim=1)

        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling!r}")

        return self.classifier(z), pool_attn


class TransformerMILClassifier(_BasePyTorchHead):
    """
    Transformer / self-attention over slices (design doc p.7).
    Input policy : native_mask (key_padding_mask passed to TransformerEncoder).
    Backend      : PyTorch.
    supports_native_attention = True  when pooling == 'attention'.

    Parameters
    ----------
    d_model   : transformer hidden dimension (must be divisible by n_heads).
    num_layers: number of TransformerEncoderLayer blocks.
    n_heads   : number of attention heads.
    pooling   : 'cls' | 'mean' | 'attention'  — how the encoder output is pooled.
    """

    input_policy: InputPolicy = "native_mask"

    def __init__(self, embedding_dim: int = 768, d_model: int = 256,
                 num_layers: int = 2, n_heads: int = 4,
                 num_classes: int = 2, dropout: float = 0.25,
                 pooling: str = "cls",
                 lr: float = 3e-4, weight_decay: float = 1e-4,
                 max_epochs: int = 100, patience: int = 15,
                 batch_size: int = 8, random_state: int = 42,
                 device: Optional[str] = None):
        super().__init__(
            embedding_dim=embedding_dim, num_classes=num_classes,
            lr=lr, weight_decay=weight_decay,
            max_epochs=max_epochs, patience=patience,
            batch_size=batch_size, random_state=random_state,
            dropout=dropout, device=device,
        )
        self.d_model    = d_model
        self.num_layers = num_layers
        self.n_heads    = n_heads
        self.pooling    = pooling
        # Expose native attention only when using attention pooling
        self.supports_native_attention = (pooling == "attention")

    def _build_model(self) -> nn.Module:
        return _TransformerMILNet(
            embedding_dim=self.embedding_dim,
            d_model=self.d_model,
            num_layers=self.num_layers,
            n_heads=self.n_heads,
            dropout_rate=self.dropout,
            num_classes=self.num_classes,
            pooling=self.pooling,
        )

    def _forward(self, bx: torch.Tensor, bl: torch.Tensor) -> torch.Tensor:
        logits, _ = self._model(bx, bl)
        return logits

    @torch.no_grad()
    def native_attention(self, X, mask=None) -> Optional[np.ndarray]:
        """
        Returns pooling attention weights [B, N] when pooling='attention',
        else returns None (consistent with BaseHead contract).
        """
        if not self.supports_native_attention:
            return None
        super().native_attention(X, mask)
        self._check_fitted()
        self._model.eval()
        loader    = self._make_loader(X, np.zeros(X.shape[0], dtype=np.int64), mask)
        attn_list = []
        for bx, _, bl in loader:
            bx, bl = bx.to(self._device), bl.to(self._device)
            _, attn = self._model(bx, bl)
            attn_list.append(attn.cpu().numpy())
        return np.concatenate(attn_list, axis=0)

    def get_params(self):
        d = super().get_params()
        d.update(d_model=self.d_model, num_layers=self.num_layers,
                 n_heads=self.n_heads, pooling=self.pooling)
        return d
