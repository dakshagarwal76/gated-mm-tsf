#!/usr/bin/env python
"""Run inference with a trained TimeRAG checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from timerag.config import Config
from timerag.data import make_loaders
from timerag.models import TimeRAG
from timerag.train import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = Config.from_dict(ckpt.get("cfg", {}))
    cfg.device = args.device
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")

    _, _, test_loader, meta = make_loaders(
        args.data, cfg.seq_len, cfg.pred_len, cfg.batch_size,
        cfg.train_ratio, cfg.val_ratio, cfg.num_workers,
    )
    model = TimeRAG(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    te = evaluate(model, test_loader, device)
    print(meta)
    print(te)


if __name__ == "__main__":
    main()
