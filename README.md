# Ours — RL-Enhanced Multimodal Time Series Forecasting

Numerical forecast, text augmentation, and closed-loop reward training.

```
Numerical TS + Paired Text
  → Frozen TSFM (mean + uncertainty)
  → Base Forecast ŷ_TSFM
  → Acquisition Policy (RL invoke / skip)
  → GRPO LLM Generator
  → Frozen Text Encoder → z
  → Dynamic Gate g = σ(W · [z, s, u]) × Text Projection
  → ŷ = ŷ_TSFM + g · f_text(z_res)

Closed-loop: forecast error → reward → policy + GRPO generator
```

See `architecture.svg` / `TimeRAG_architecture.png`.

## Requirements

- Python **3.11** (PyTorch CUDA wheels do not support 3.14)
- Optional NVIDIA GPU

```bash
python -m venv .venv
# Windows Git Bash:
source .venv/Scripts/activate

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

CPU-only: skip the CUDA index and `pip install torch`.

## Train

Time-MMD / TaTS-style CSV with `Date`/`date`, `OT`, and optional `fact`/`preds` text:

```bash
python train.py --data path/to/Security.csv \
  --seq_len 24 --pred_len 6 --epochs 10 --batch_size 8 \
  --device cuda --out_dir outputs/security
```

## Evaluate / compare

```bash
python predict.py --data path/to/Security.csv --ckpt outputs/security/best.pt --device cuda

python compare_baselines.py --data path/to/Security.csv \
  --ckpt outputs/security/best.pt --seq_len 24 --pred_len 6 \
  --device cuda --out outputs/security/comparison.json
```

## Defaults (runs offline)

| Component | Default | Optional upgrade |
|-----------|---------|------------------|
| Frozen TSFM | Transformer fallback | `--tsfm_backend moirai` (`uni2ts`) |
| Text encoder | Hash bag-of-words | `--text_encoder sentence-transformers` |
| GRPO LLM | Trainable residual MLP group | `--llm_backend qwen` (`transformers`) |

Reward: `reward_scale * (MSE_base − MSE_final) − invoke_cost * action`.

## Results — Security

Time-MMD Security, `seq_len=24`, `pred_len=6`, z-normalized test split (61 windows).

| Model | Test MSE | Test MAE |
|-------|----------|----------|
| Historical mean | **59.36** | **3.43** |
| DLinear | 73.96 | 4.59 |
| Ours | 80.24 | 5.29 |
| TinyTransformer | 89.40 | 5.95 |
| Drift | 113.20 | 5.10 |

Ours beats Drift and TinyTransformer, but not historical mean or DLinear on this split. At evaluation the acquisition policy skipped text (`invoke_rate = 0`), so Ours matched its numerical path only.

## Layout

```
train.py  predict.py  compare_baselines.py
timerag/  config.py data.py models.py train.py
architecture.svg  TimeRAG_architecture.png
```
