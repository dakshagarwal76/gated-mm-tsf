#!/usr/bin/env python
"""Compare TimeRAG vs classical + neural baselines on the same split/normalization."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from timerag.config import Config
from timerag.data import make_loaders
from timerag.models import TimeRAG
from timerag.train import evaluate as eval_timerag


def batch_metrics(pred: torch.Tensor, y: torch.Tensor):
    return F.mse_loss(pred, y).item(), F.l1_loss(pred, y).item()


@torch.no_grad()
def eval_fn(predict_fn, loader, device):
    tot_mse = tot_mae = n = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred = predict_fn(x, batch)
        b = x.shape[0]
        m, a = batch_metrics(pred, y)
        tot_mse += m * b
        tot_mae += a * b
        n += b
    return {"mse": tot_mse / max(n, 1), "mae": tot_mae / max(n, 1), "n": n}


# -------------------- baselines --------------------

def naive_last(x, batch):
    # repeat last value for horizon
    last = x[:, -1:, :]
    return last.repeat(1, batch["y"].shape[1], 1)


def mean_hist(x, batch):
    m = x.mean(dim=1, keepdim=True)
    return m.repeat(1, batch["y"].shape[1], 1)


def drift(x, batch):
    # linear drift from first to last of window
    B, L, C = x.shape
    H = batch["y"].shape[1]
    slope = (x[:, -1, :] - x[:, 0, :]) / max(L - 1, 1)
    steps = torch.arange(1, H + 1, device=x.device, dtype=x.dtype).view(1, H, 1)
    return x[:, -1:, :] + slope.view(B, 1, C) * steps


class DLinear(nn.Module):
    """Simple DLinear-style: trend + seasonal linear maps."""

    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.seq_len = seq_len
        kernel = 25 if seq_len >= 25 else (seq_len // 2 * 2 + 1)
        self.avg = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=kernel // 2, count_include_pad=False)
        self.linear_season = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B,L,1]
        t = self.avg(x.transpose(1, 2)).transpose(1, 2)
        if t.shape[1] != x.shape[1]:
            t = F.interpolate(t.transpose(1, 2), size=x.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        s = x - t
        y = self.linear_season(s.squeeze(-1)) + self.linear_trend(t.squeeze(-1))
        return y.unsqueeze(-1)


class TinyTransformer(nn.Module):
    def __init__(self, seq_len, pred_len, d_model=64):
        super().__init__()
        self.enc = nn.Linear(1, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=128, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x):
        h = self.tr(self.enc(x)).mean(dim=1)
        return self.head(h).unsqueeze(-1)


def train_torch_model(model, train_loader, val_loader, device, epochs=20, lr=1e-3, patience=5):
    opt = AdamW(model.parameters(), lr=lr)
    best = float("inf")
    best_state = None
    bad = 0
    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        va = eval_fn(lambda x, b: model(x), val_loader, device)
        if va["mse"] < best - 1e-6:
            best = va["mse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="outputs/agri/best.pt")
    ap.add_argument("--seq_len", type=int, default=24)
    ap.add_argument("--pred_len", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/agri/comparison.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    train_loader, val_loader, test_loader, meta = make_loaders(
        args.data, args.seq_len, args.pred_len, args.batch_size
    )
    print("meta", meta)
    print("device", device)

    results = {}

    # Classical
    for name, fn in [
        ("Naive (last)", naive_last),
        ("Historical mean", mean_hist),
        ("Drift", drift),
    ]:
        te = eval_fn(fn, test_loader, device)
        results[name] = te
        print(f"{name:22s}  MSE={te['mse']:.4f}  MAE={te['mae']:.4f}")

    # Neural baselines
    for name, ctor in [
        ("DLinear", lambda: DLinear(args.seq_len, args.pred_len)),
        ("TinyTransformer", lambda: TinyTransformer(args.seq_len, args.pred_len)),
    ]:
        t0 = time.time()
        model = ctor().to(device)
        model = train_torch_model(model, train_loader, val_loader, device, epochs=args.epochs)
        te = eval_fn(lambda x, b: model(x), test_loader, device)
        te["train_seconds"] = time.time() - t0
        results[name] = te
        print(f"{name:22s}  MSE={te['mse']:.4f}  MAE={te['mae']:.4f}  ({te['train_seconds']:.1f}s)")

    # TimeRAG checkpoint
    ckpt_path = Path(args.ckpt)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = Config.from_dict(ckpt.get("cfg", {}))
        cfg.seq_len = args.seq_len
        cfg.pred_len = args.pred_len
        model = TimeRAG(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        te = eval_timerag(model, test_loader, device)
        results["Ours"] = {"mse": te["mse"], "mae": te["mae"], "invoke_rate": te["invoke_rate"]}
        results["Ours (base-only)"] = {"mse": te["mse_base"], "mae": te["mae"], "note": "numerical path only"}
        print(f"{'Ours':22s}  MSE={te['mse']:.4f}  MAE={te['mae']:.4f}  invoke={te['invoke_rate']:.2f}")
        print(f"{'Ours (base-only)':22s}  MSE={te['mse_base']:.4f}")
    else:
        print(f"No checkpoint at {ckpt_path}")

    # Paper reference (different protocol — context only)
    results["paper_context"] = {
        "note": "TaTS Table2 Agriculture averages over pred_lens {6,8,10,12}; not same split/scale as this run",
        "TaTS_iTransformer_MSE": 0.109,
        "TaTS_PatchTST_MSE": 0.114,
        "TaTS_DLinear_MSE": 0.214,
        "UniModal_iTransformer_MSE": 0.122,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "seq_len": args.seq_len, "pred_len": args.pred_len, "device": str(device), "results": results}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("saved", out)

    # ranking
    ranked = sorted(
        [(k, v["mse"]) for k, v in results.items() if isinstance(v, dict) and "mse" in v and k != "Ours (base-only)"],
        key=lambda t: t[1],
    )
    print("\n=== Test MSE ranking (same split, z-normalized) ===")
    for i, (k, m) in enumerate(ranked, 1):
        print(f"{i}. {k:22s} {m:.4f}")


if __name__ == "__main__":
    main()
