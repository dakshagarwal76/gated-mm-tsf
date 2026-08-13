"""
TimeRAG — RL-Enhanced Multimodal Time Series Forecasting

Architecture (from project diagram):
  Numerical TS + Paired Text
    → Frozen TSFM (Moirai / fallback) → Base Forecast ŷ_TSFM
    → Acquisition Policy (invoke/skip)
    → GRPO LLM Generator (Qwen2.5 / fallback)
    → Frozen Text Encoder → Dynamic Gate g × Text Projection
    → ŷ = ŷ_TSFM + g · f_text(z_res)
  Closed-loop: forecast error → reward → policy + GRPO generator
"""

__version__ = "0.1.0"
