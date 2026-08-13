from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Config:
    # data
    data_csv: str = ""
    seq_len: int = 24
    pred_len: int = 6
    train_ratio: float = 0.7
    val_ratio: float = 0.1

    # model dims
    d_model: int = 128
    text_dim: int = 384
    hidden: int = 256

    # backends: "auto" | "moirai" | "fallback"  /  "auto" | "qwen" | "fallback"
    tsfm_backend: str = "fallback"
    llm_backend: str = "fallback"
    text_encoder: str = "hash"  # "hash" | "sentence-transformers"

    # training
    epochs: int = 10
    batch_size: int = 16
    lr: float = 1e-3
    lr_policy: float = 1e-4
    lr_llm: float = 5e-5
    patience: int = 5
    seed: int = 2025
    device: str = "cuda"
    num_workers: int = 0

    # RL / GRPO
    invoke_cost: float = 0.01          # penalty for calling text path
    reward_scale: float = 10.0
    grpo_group_size: int = 2           # candidates per step for GRPO-style ranking
    freeze_tsfm: bool = True
    freeze_text_encoder: bool = True

    # IO
    out_dir: str = "outputs"
    save_every: int = 1

    # optional HF ids
    moirai_model_id: str = "Salesforce/moirai-1.0-R-small"
    qwen_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    st_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self):
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        return cls(**{k: v for k, v in d.items() if k in known})
