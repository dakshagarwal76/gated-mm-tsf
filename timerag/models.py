from __future__ import annotations

import hashlib
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Frozen / fallback TSFM → mean + uncertainty
# ---------------------------------------------------------------------------

class FallbackTSFM(nn.Module):
    """Lightweight frozen-style backbone (can be frozen). Emits mean + uncertainty."""

    def __init__(self, seq_len: int, pred_len: int, d_model: int = 128):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 2,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head_mean = nn.Linear(d_model, pred_len)
        self.head_logvar = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L, 1]
        h = self.enc(x)
        h = self.transformer(h)
        pooled = h.mean(dim=1)
        mean = self.head_mean(pooled).unsqueeze(-1)       # [B, H, 1]
        logvar = self.head_logvar(pooled).unsqueeze(-1)
        unc = torch.exp(0.5 * logvar).clamp(1e-4, 10.0)   # std
        return mean, unc


class MoiraiTSFM(nn.Module):
    """Optional Moirai wrapper. Falls back if uni2ts is unavailable."""

    def __init__(self, seq_len: int, pred_len: int, model_id: str, d_model: int = 128):
        super().__init__()
        self.fallback = FallbackTSFM(seq_len, pred_len, d_model)
        self.backend = "fallback"
        try:
            # Soft optional import — real Moirai integration is environment-specific
            import importlib
            importlib.import_module("uni2ts")
            self.backend = "moirai-stub"
            # Keep fallback weights as stand-in until full uni2ts forecast API is wired
        except Exception:
            pass

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fallback(x)


def build_tsfm(backend: str, seq_len: int, pred_len: int, d_model: int, model_id: str) -> nn.Module:
    if backend in ("moirai", "auto"):
        return MoiraiTSFM(seq_len, pred_len, model_id, d_model)
    return FallbackTSFM(seq_len, pred_len, d_model)


# ---------------------------------------------------------------------------
# Base forecast head (trainable refinement of TSFM mean)
# ---------------------------------------------------------------------------

class BaseForecastHead(nn.Module):
    def __init__(self, pred_len: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pred_len * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, pred_len),
        )
        self.pred_len = pred_len

    def forward(self, mean: torch.Tensor, unc: torch.Tensor) -> torch.Tensor:
        # mean, unc: [B, H, 1]
        b = mean.shape[0]
        feats = torch.cat([mean.squeeze(-1), unc.squeeze(-1)], dim=-1)
        out = self.net(feats).view(b, self.pred_len, 1)
        return out


# ---------------------------------------------------------------------------
# Acquisition policy: invoke / skip
# ---------------------------------------------------------------------------

class AcquisitionPolicy(nn.Module):
    """Bernoulli invoke/skip from numerical stats (mean, unc summary)."""

    def __init__(self, pred_len: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pred_len * 2 + 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, mean: torch.Tensor, unc: torch.Tensor, has_text: torch.Tensor) -> torch.Tensor:
        # returns logit of invoke  [B]
        b = mean.shape[0]
        m = mean.squeeze(-1)
        u = unc.squeeze(-1)
        ht = has_text.float().view(b, 1)
        # also encode whether text exists as feature
        feats = torch.cat([m, u, ht, u.mean(dim=1, keepdim=True)], dim=-1)
        return self.net(feats).squeeze(-1)

    def sample(self, logits: torch.Tensor, deterministic: bool = False):
        probs = torch.sigmoid(logits)
        if deterministic:
            action = (probs >= 0.5).float()
        else:
            action = torch.bernoulli(probs)
        # log π(a|s)
        logp = action * torch.log(probs.clamp_min(1e-8)) + (1 - action) * torch.log((1 - probs).clamp_min(1e-8))
        return action, logp, probs


# ---------------------------------------------------------------------------
# Text encoder (frozen)
# ---------------------------------------------------------------------------

class HashTextEncoder(nn.Module):
    """Deterministic bag-of-hashes encoder (no downloads). Output dim = text_dim."""

    def __init__(self, text_dim: int = 384, n_buckets: int = 4096):
        super().__init__()
        self.text_dim = text_dim
        self.n_buckets = n_buckets
        self.proj = nn.Linear(n_buckets, text_dim)
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, texts: List[str], device: torch.device) -> torch.Tensor:
        rows = []
        for t in texts:
            v = torch.zeros(self.n_buckets, dtype=torch.float32)
            toks = (t or "").lower().split()
            if not toks:
                rows.append(v)
                continue
            for tok in toks[:256]:
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.n_buckets
                v[h] += 1.0
            if v.norm() > 0:
                v = v / v.norm()
            rows.append(v)
        bag = torch.stack(rows, dim=0).to(device)
        return self.proj(bag)


class STTextEncoder(nn.Module):
    def __init__(self, model_id: str, text_dim: int):
        super().__init__()
        self.fallback = HashTextEncoder(text_dim)
        self.model = None
        self.text_dim = text_dim
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_id)
            for p in self.model.parameters():
                p.requires_grad = False
        except Exception:
            pass

    @torch.no_grad()
    def encode(self, texts: List[str], device: torch.device) -> torch.Tensor:
        if self.model is None:
            return self.fallback.encode(texts, device)
        emb = self.model.encode(texts, convert_to_tensor=True, device=str(device))
        if emb.shape[-1] != self.text_dim:
            # pad / truncate
            if emb.shape[-1] > self.text_dim:
                emb = emb[..., : self.text_dim]
            else:
                pad = torch.zeros(*emb.shape[:-1], self.text_dim - emb.shape[-1], device=emb.device)
                emb = torch.cat([emb, pad], dim=-1)
        return emb.float()


def build_text_encoder(kind: str, text_dim: int, model_id: str) -> nn.Module:
    if kind == "sentence-transformers":
        return STTextEncoder(model_id, text_dim)
    return HashTextEncoder(text_dim)


# ---------------------------------------------------------------------------
# GRPO LLM generator (trainable residual text / soft prompt features)
# ---------------------------------------------------------------------------

class FallbackGRPOGenerator(nn.Module):
    """
    Stand-in for Qwen2.5-1.5B GRPO training.
    Maps raw text embedding → residual embedding z_res, with K candidates for GRPO ranking.
    """

    def __init__(self, text_dim: int, hidden: int = 256, group_size: int = 2):
        super().__init__()
        self.group_size = group_size
        self.shared = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, text_dim),
        )
        # small noise heads for diverse candidates
        self.heads = nn.ModuleList([nn.Linear(text_dim, text_dim) for _ in range(group_size)])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, D] → z_res [B, D] (greedy / first head)
        h = self.shared(z)
        return z + self.heads[0](h)

    def candidates(self, z: torch.Tensor) -> torch.Tensor:
        # returns [B, K, D]
        h = self.shared(z)
        outs = [z + head(h) for head in self.heads]
        return torch.stack(outs, dim=1)


class QwenGRPOGenerator(nn.Module):
    """Optional Qwen path; uses fallback if transformers/Qwen unavailable."""

    def __init__(self, text_dim: int, model_id: str, group_size: int = 2):
        super().__init__()
        self.fallback = FallbackGRPOGenerator(text_dim, group_size=group_size)
        self.backend = "fallback"
        try:
            import transformers  # noqa: F401
            self.backend = "qwen-stub"
        except Exception:
            pass

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fallback(z)

    def candidates(self, z: torch.Tensor) -> torch.Tensor:
        return self.fallback.candidates(z)


def build_llm_generator(backend: str, text_dim: int, model_id: str, group_size: int) -> nn.Module:
    if backend in ("qwen", "auto"):
        return QwenGRPOGenerator(text_dim, model_id, group_size)
    return FallbackGRPOGenerator(text_dim, group_size=group_size)


# ---------------------------------------------------------------------------
# Dynamic gate + text projection
# ---------------------------------------------------------------------------

class DynamicGate(nn.Module):
    """g = σ(W · [z, s, u])"""

    def __init__(self, text_dim: int, pred_len: int, hidden: int = 128):
        super().__init__()
        in_dim = text_dim + pred_len * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor, mean: torch.Tensor, unc: torch.Tensor) -> torch.Tensor:
        # returns g in (0,1) shaped [B, 1, 1]
        s = mean.squeeze(-1)
        u = unc.squeeze(-1)
        g = torch.sigmoid(self.net(torch.cat([z, s, u], dim=-1)))
        return g.view(-1, 1, 1)


class TextProjection(nn.Module):
    def __init__(self, text_dim: int, pred_len: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, pred_len),
        )
        self.pred_len = pred_len

    def forward(self, z_res: torch.Tensor) -> torch.Tensor:
        return self.net(z_res).unsqueeze(-1)  # [B, H, 1]


# ---------------------------------------------------------------------------
# Full TimeRAG model
# ---------------------------------------------------------------------------

class TimeRAG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tsfm = build_tsfm(cfg.tsfm_backend, cfg.seq_len, cfg.pred_len, cfg.d_model, cfg.moirai_model_id)
        self.base = BaseForecastHead(cfg.pred_len, cfg.hidden)
        self.policy = AcquisitionPolicy(cfg.pred_len)
        self.text_encoder = build_text_encoder(cfg.text_encoder, cfg.text_dim, cfg.st_model_id)
        self.llm = build_llm_generator(cfg.llm_backend, cfg.text_dim, cfg.qwen_model_id, cfg.grpo_group_size)
        self.gate = DynamicGate(cfg.text_dim, cfg.pred_len, cfg.hidden)
        self.proj = TextProjection(cfg.text_dim, cfg.pred_len, cfg.hidden)

        if cfg.freeze_tsfm:
            for p in self.tsfm.parameters():
                p.requires_grad = False

    def encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        return self.text_encoder.encode(texts, device)

    def forward(
        self,
        x: torch.Tensor,
        texts: List[str],
        has_text: torch.Tensor,
        deterministic_policy: bool = False,
        force_invoke: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Returns dict with y_hat, y_tsfm, g, action, logp, probs, unc, z_res, ...
        Final: ŷ = ŷ_TSFM + g · f_text(z_res)   (g=0 / action=0 → pure numerical)
        """
        device = x.device
        mean, unc = self.tsfm(x)
        y_tsfm = self.base(mean, unc)

        logits = self.policy(mean, unc, has_text)
        action, logp, probs = self.policy.sample(logits, deterministic=deterministic_policy)
        if force_invoke is not None:
            action = force_invoke.float()

        # mask: cannot invoke if no text
        action = action * has_text.float()

        z = self.encode_text(texts, device)
        z_res = self.llm(z)
        g = self.gate(z_res, mean, unc)
        text_corr = self.proj(z_res)

        # apply acquisition: if skip, zero the correction
        invoke = action.view(-1, 1, 1)
        gated = (g * text_corr) * invoke
        y_hat = y_tsfm + gated

        return {
            "y_hat": y_hat,
            "y_tsfm": y_tsfm,
            "mean": mean,
            "unc": unc,
            "g": g,
            "text_corr": text_corr,
            "action": action,
            "logp": logp,
            "probs": probs,
            "z": z,
            "z_res": z_res,
            "logits": logits,
        }
