#!/usr/bin/env python
"""Train TimeRAG on a Time-MMD / TaTS CSV."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from timerag.config import Config
from timerag.data import make_loaders
from timerag.train import fit


def main():
    p = argparse.ArgumentParser(description="TimeRAG trainer")
    p.add_argument("--data", type=str, required=True, help="Path to Time-MMD/TaTS CSV (OT + text columns)")
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--pred_len", type=int, default=6)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_dir", type=str, default="outputs/run")
    p.add_argument("--tsfm_backend", type=str, default="fallback", choices=["fallback", "moirai", "auto"])
    p.add_argument("--llm_backend", type=str, default="fallback", choices=["fallback", "qwen", "auto"])
    p.add_argument("--text_encoder", type=str, default="hash", choices=["hash", "sentence-transformers"])
    p.add_argument("--seed", type=int, default=2025)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = Config(
        data_csv=args.data,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        out_dir=args.out_dir,
        tsfm_backend=args.tsfm_backend,
        llm_backend=args.llm_backend,
        text_encoder=args.text_encoder,
        seed=args.seed,
    )

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Data not found: {data_path}")

    train_loader, val_loader, test_loader, meta = make_loaders(
        str(data_path),
        cfg.seq_len,
        cfg.pred_len,
        cfg.batch_size,
        cfg.train_ratio,
        cfg.val_ratio,
        cfg.num_workers,
    )
    print(f"Data meta: {meta}")
    fit(cfg, train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()
