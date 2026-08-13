from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def _clean_text(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    if not s or s.upper() == "NA":
        return ""
    # drop obvious NA-only summaries
    if re.fullmatch(r"(NA[;\s]*)+", s, flags=re.I):
        return ""
    return s


def _pick_text(row: pd.Series) -> str:
    """Prefer fact, else preds, else empty."""
    for col in ("fact", "preds", "text", "news", "report"):
        if col in row.index:
            t = _clean_text(row[col])
            if t:
                return t
    return ""


def load_timemd_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    # normalize date
    date_col = "Date" if "Date" in df.columns else ("date" if "date" in df.columns else None)
    if date_col is None:
        raise ValueError(f"No Date/date column in {path}")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    if "OT" not in df.columns:
        # fallback: first numeric non-date column
        num = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
        if not num:
            raise ValueError(f"No OT / numeric target in {path}")
        df = df.rename(columns={num[0]: "OT"})

    df["OT"] = pd.to_numeric(df["OT"], errors="coerce")
    df = df.dropna(subset=["OT"]).reset_index(drop=True)
    df["text"] = df.apply(_pick_text, axis=1)
    df["has_text"] = df["text"].str.len() > 0
    out = df[[date_col, "OT", "text", "has_text"]].copy()
    out = out.rename(columns={date_col: "date"})
    return out.reset_index(drop=True)


class TimeRAGDataset(Dataset):
    """Aligned windows: numerical history → future OT, plus paired text at last hist step."""

    def __init__(
        self,
        values: np.ndarray,
        texts: List[str],
        has_text: np.ndarray,
        seq_len: int,
        pred_len: int,
        mean: float,
        std: float,
    ):
        self.values = values.astype(np.float32)
        self.texts = texts
        self.has_text = has_text.astype(np.bool_)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mean = float(mean)
        self.std = float(std) if float(std) > 1e-8 else 1.0
        self.n = len(values) - seq_len - pred_len + 1
        if self.n <= 0:
            raise ValueError("Series too short for seq_len/pred_len")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | bool]:
        s = idx
        e = s + self.seq_len
        p = e + self.pred_len
        x = (self.values[s:e] - self.mean) / self.std
        y = (self.values[e:p] - self.mean) / self.std
        # text aligned to last observed step in the window
        text = self.texts[e - 1]
        has = bool(self.has_text[e - 1])
        return {
            "x": torch.from_numpy(x).unsqueeze(-1),          # [L, 1]
            "y": torch.from_numpy(y).unsqueeze(-1),          # [H, 1]
            "text": text,
            "has_text": has,
            "scale_mean": self.mean,
            "scale_std": self.std,
        }


def collate(batch: List[dict]) -> Dict:
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "y": torch.stack([b["y"] for b in batch], dim=0),
        "text": [b["text"] for b in batch],
        "has_text": torch.tensor([b["has_text"] for b in batch], dtype=torch.bool),
        "scale_mean": batch[0]["scale_mean"],
        "scale_std": batch[0]["scale_std"],
    }


def make_loaders(
    csv_path: str,
    seq_len: int,
    pred_len: int,
    batch_size: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    df = load_timemd_csv(csv_path)
    values = df["OT"].to_numpy(dtype=np.float32)
    texts = df["text"].tolist()
    has_text = df["has_text"].to_numpy()

    n = len(values)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # stats from train only
    mean = float(values[:n_train].mean())
    std = float(values[:n_train].std())

    def slice_ds(a: int, b: int) -> TimeRAGDataset:
        # need lookback into previous split for continuity — use full series indices
        return TimeRAGDataset(values, texts, has_text, seq_len, pred_len, mean, std)

    # index ranges for window starts
    full = TimeRAGDataset(values, texts, has_text, seq_len, pred_len, mean, std)
    # remap: restrict indices by end of prediction falling in split
    train_idx, val_idx, test_idx = [], [], []
    for i in range(len(full)):
        pred_end = i + seq_len + pred_len - 1
        if pred_end < n_train:
            train_idx.append(i)
        elif pred_end < n_train + n_val:
            val_idx.append(i)
        else:
            test_idx.append(i)

    class Sub(Dataset):
        def __init__(self, base, idxs):
            self.base, self.idxs = base, idxs
        def __len__(self):
            return len(self.idxs)
        def __getitem__(self, i):
            return self.base[self.idxs[i]]

    meta = {"mean": mean, "std": std, "n": n, "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx)}
    kw = dict(batch_size=batch_size, collate_fn=collate, num_workers=num_workers)
    train_loader = DataLoader(Sub(full, train_idx), shuffle=True, **kw)
    val_loader = DataLoader(Sub(full, val_idx), shuffle=False, **kw)
    test_loader = DataLoader(Sub(full, test_idx), shuffle=False, **kw)
    return train_loader, val_loader, test_loader, meta
