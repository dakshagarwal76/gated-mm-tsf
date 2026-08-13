from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .config import Config
from .models import TimeRAG


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(a, b)


def mae(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(a, b)


@torch.no_grad()
def evaluate(model: TimeRAG, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    tot_mse = tot_mae = tot_base = n = 0
    invoke_rate = 0.0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        out = model(x, batch["text"], batch["has_text"].to(device), deterministic_policy=True)
        b = x.shape[0]
        tot_mse += mse(out["y_hat"], y).item() * b
        tot_mae += mae(out["y_hat"], y).item() * b
        tot_base += mse(out["y_tsfm"], y).item() * b
        invoke_rate += out["action"].sum().item()
        n += b
    return {
        "mse": tot_mse / max(n, 1),
        "mae": tot_mae / max(n, 1),
        "mse_base": tot_base / max(n, 1),
        "invoke_rate": invoke_rate / max(n, 1),
        "n": n,
    }


def grpo_loss(
    model: TimeRAG,
    z: torch.Tensor,
    mean: torch.Tensor,
    unc: torch.Tensor,
    y: torch.Tensor,
    y_tsfm: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    """
    GRPO-style ranking over K residual candidates:
    reward_k = improvement over base forecast when candidate is applied.
    """
    # only for invoked samples
    mask = action > 0.5
    if mask.sum() == 0:
        return y.new_zeros(())

    z_m = z[mask]
    mean_m = mean[mask]
    unc_m = unc[mask]
    y_m = y[mask]
    y_base = y_tsfm[mask]

    cands = model.llm.candidates(z_m)  # [B', K, D]
    b, k, d = cands.shape
    losses = []
    rewards = []
    for i in range(k):
        zi = cands[:, i, :]
        g = model.gate(zi, mean_m, unc_m)
        corr = model.proj(zi)
        y_hat = y_base + g * corr
        err = ((y_hat - y_m) ** 2).mean(dim=(1, 2))
        base_err = ((y_base - y_m) ** 2).mean(dim=(1, 2))
        # higher reward if better than base
        rew = (base_err - err).detach()
        rewards.append(rew)
        losses.append(err)

    rewards_t = torch.stack(rewards, dim=1)  # [B', K]
    # advantage vs group mean
    adv = rewards_t - rewards_t.mean(dim=1, keepdim=True)
    # weight candidate reconstruction toward better ones (soft GRPO)
    # use MSE of y_hat as supervised term weighted by softmax(adv)
    w = torch.softmax(adv * 5.0, dim=1)
    loss_stack = torch.stack(losses, dim=1)
    return (w * loss_stack).sum(dim=1).mean()


def train_one_epoch(
    model: TimeRAG,
    loader,
    opt_main,
    opt_policy,
    opt_llm,
    cfg: Config,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    # TSFM may be frozen
    meters = {"loss": 0.0, "mse": 0.0, "policy": 0.0, "grpo": 0.0, "invoke": 0.0, "n": 0}

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        has = batch["has_text"].to(device)
        bsz = x.shape[0]

        out = model(x, batch["text"], has, deterministic_policy=False)
        y_hat = out["y_hat"]
        y_tsfm = out["y_tsfm"]

        # supervised forecast loss on final prediction
        loss_fore = mse(y_hat, y)

        # also lightly supervise base head toward y (numerical path)
        loss_base = mse(y_tsfm, y)

        # reward from forecast error improvement vs base-only
        with torch.no_grad():
            err_final = ((y_hat - y) ** 2).mean(dim=(1, 2))
            err_base = ((y_tsfm.detach() - y) ** 2).mean(dim=(1, 2))
            improvement = err_base - err_final
            # reward: improvement minus invoke cost
            reward = cfg.reward_scale * improvement - cfg.invoke_cost * out["action"]

        # policy gradient (REINFORCE) + entropy bonus for exploration
        probs = out["probs"].clamp(1e-6, 1 - 1e-6)
        entropy = -(probs * probs.log() + (1 - probs) * (1 - probs).log()).mean()
        loss_policy = -(out["logp"] * reward.detach()).mean() - 0.01 * entropy

        # GRPO on LLM residual generator
        loss_grpo = grpo_loss(
            model, out["z"].detach(), out["mean"].detach(), out["unc"].detach(),
            y, y_tsfm.detach(), out["action"].detach(),
        )

        # warm-start: also train gate/proj as if always invoked when text exists
        # (helps text path learn before policy explores)
        z = out["z"]
        z_res = model.llm(z)
        g_full = model.gate(z_res, out["mean"].detach(), out["unc"].detach())
        corr_full = model.proj(z_res)
        y_oracle = y_tsfm.detach() + g_full * corr_full
        has_f = has.float().view(-1, 1, 1)
        loss_text = (((y_oracle - y) ** 2) * has_f).sum() / has_f.sum().clamp_min(1.0)

        loss = loss_fore + 0.3 * loss_base + loss_policy + 0.5 * loss_grpo + 0.2 * loss_text

        opt_main.zero_grad(set_to_none=True)
        opt_policy.zero_grad(set_to_none=True)
        opt_llm.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_main.step()
        opt_policy.step()
        opt_llm.step()

        meters["loss"] += loss.item() * bsz
        meters["mse"] += loss_fore.item() * bsz
        meters["policy"] += loss_policy.item() * bsz
        meters["grpo"] += float(loss_grpo.item()) * bsz if torch.is_tensor(loss_grpo) else 0.0
        meters["invoke"] += out["action"].sum().item()
        meters["n"] += bsz

    n = max(meters["n"], 1)
    return {k: (v / n if k != "n" else v) for k, v in meters.items()}


def fit(cfg: Config, train_loader, val_loader, test_loader) -> dict:
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")
    model = TimeRAG(cfg).to(device)

    main_params = list(model.base.parameters()) + list(model.gate.parameters()) + list(model.proj.parameters())
    if not cfg.freeze_tsfm:
        main_params += list(model.tsfm.parameters())

    opt_main = AdamW(main_params, lr=cfg.lr)
    opt_policy = AdamW(model.policy.parameters(), lr=cfg.lr_policy)
    opt_llm = AdamW(model.llm.parameters(), lr=cfg.lr_llm)

    best_val = float("inf")
    best_state = None
    bad = 0
    history = []

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt_main, opt_policy, opt_llm, cfg, device)
        va = evaluate(model, val_loader, device)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train_mse={tr['mse']:.4f} val_mse={va['mse']:.4f} val_mae={va['mae']:.4f} "
            f"base_mse={va['mse_base']:.4f} invoke={va['invoke_rate']:.2f}"
        )

        if va["mse"] < best_val - 1e-6:
            best_val = va["mse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            torch.save({"model": best_state, "cfg": cfg.__dict__, "val": va}, out_dir / "best.pt")
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    te = evaluate(model, test_loader, device)
    summary = {
        "best_val_mse": best_val,
        "test": te,
        "seconds": time.time() - t0,
        "history": history,
        "device": str(device),
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Test MSE={te['mse']:.4f} MAE={te['mae']:.4f} base_MSE={te['mse_base']:.4f} invoke={te['invoke_rate']:.2f}")
    print(f"Saved -> {out_dir / 'best.pt'} , {out_dir / 'results.json'}")
    return summary
