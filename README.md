# Deep-RegimeNet
### An Unsupervised Deep Clustering Framework for Adaptive Market Regime Identification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![WQU](https://img.shields.io/badge/WorldQuant%20University-MScFE%20690-navy)](https://wqu.edu)

> **MScFE Capstone Project — Group 16019 — WorldQuant University, 2026**  
> Ian Kihara Wangui · Grace Nmon Monanyun · **Immaculate Wanjiru Kimani**

---

## Overview

Financial markets shift dynamically between distinct behavioural regimes — bull, bear, and
high-volatility states — with direct implications for portfolio construction, risk management,
and algorithmic trading. Classical methods such as Hidden Markov Models rely on parametric
Gaussian assumptions and sequentially separate dimensionality reduction from cluster
assignment, causing a systematic loss of regime-relevant information.

**Deep-RegimeNet** proposes a joint optimisation framework that simultaneously learns
temporal latent representations and optimises market regime cluster assignments in a single
end-to-end trained neural architecture.

---

## Architecture at a Glance

```
Input: S&P 500 daily OHLCV → 20-day sliding windows (W × 4)
           │
           ▼
  ┌─────────────────────────────┐
  │  Temporal Autoencoder (TAE) │  ← 2-layer LSTM, 256 hidden
  │  + Temporal Attention Pool  │     LayerNorm between layers
  │  → Latent code z ∈ ℝ^64    │
  └──────────────┬──────────────┘
                 │
                 ▼
  ┌─────────────────────────────┐
  │  Deep Embedded Clustering   │  ← K=4 centroids, Student-t kernel
  │  Soft assignments Q via     │     KL(P‖Q) clustering loss
  │  t-distribution kernel      │
  └──────────────┬──────────────┘
                 │
    Joint Loss: L = L_rec + λ·L_clust   (λ=0.1, dynamically annealed)
```

---

## Training Protocol

| Phase | Description | Epochs |
|-------|-------------|--------|
| **Phase I** | TAE pre-training on reconstruction loss only | 80 |
| **Phase II** | K-Means centroid initialisation (20 restarts) on pre-trained latent codes | — |
| **Phase III** | Joint optimisation with dynamic λ annealing (0 → 0.1 over 75 epochs) | 150 |

---

## Key Results

Four baselines were implemented for rigorous comparison:

| Model | Annualised Sharpe | Total Return | Max Drawdown |
|-------|:-----------------:|:------------:|:------------:|
| **Gaussian HMM** | **0.850** | **1,817%** | −43.9% |
| Naïve Vol. Quantile | 0.425 | 579% | −62.8% |
| Sequential AE + KMeans | 0.433 | 506% | −56.3% |
| **Deep-RegimeNet** | **0.284** | **143%** | −45.9% |
| Buy-and-Hold | 0.454 | 750% | −64.3% |

> **Main finding:** Deep-RegimeNet does not outperform simpler baselines at this sample size
> (~400 effective independent observations). The Gaussian HMM dominates on all strategy
> metrics. This constitutes a rigorous negative result: DEC-based deep temporal clustering
> cannot overcome cluster collapse and temporal incoherence at financial daily-data sample
> sizes without explicit temporal regularisation.

**Historical concordance** (7 pre-specified market event windows):

| Model | Mean Purity | GFC | Covid Crash |
|-------|:-----------:|:---:|:-----------:|
| Deep-RegimeNet | **58.9%** | 60.1% | 50.0% |
| Gaussian HMM | 55.9% | 65.2% | 87.5% |
| Naïve Quantile | 54.7% | 75.8% | 83.3% |

---

## Repository Structure

```
deep_regimenet/
├── notebooks/
│   ├── EDA_GSPC_Capstone_Deep_RegimeNet.ipynb        # Exploratory data analysis
│   └── v3_1_draft_report_capstone_deep_regime_net.ipynb  # Full pipeline
├── config.py                  # All hyperparameters centralised here
├── requirements.txt           # Python dependencies
├── CONTRIBUTORS.md            # Team roles and acknowledgements
├── LICENSE                    # MIT License
└── README.md                  # This file
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/Iandavidk/mscfe_capstone_deep_regime_net.git
cd mscfe_capstone_deep_regime_net

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the main notebook
jupyter notebook notebooks/v3_1_draft_report_capstone_deep_regime_net.ipynb
```

> **Recommended:** Run on Google Colab with T4 GPU for full training speed.
> Runtime → Change runtime type → T4 GPU → Run all cells.

---

## Data

Downloaded automatically via `yfinance` — no manual setup needed.

| Feature | Description | Normalisation |
|---------|-------------|---------------|
| `log_return` | Daily log return | Rolling Z-score 252d |
| `intraday_range` | (High − Low) / Close | Rolling Z-score 252d |
| `volume_z` | Volume 20d Z-score | Rolling Z-score 20d |
| `log_realised_vol` | Log 20d annualised realised vol | Log-level (training fold stats) |

`log_realised_vol` is intentionally kept in log-level space — rolling Z-score would erase the
level information needed to distinguish high-volatility from low-volatility regimes.

---

## Reproducing Results

All seeds and hyperparameters are in `config.py`. Primary configuration:

```python
RANDOM_SEED     = 42
MULTI_SEEDS     = [42, 123, 456, 789, 2024]
WINDOW_SIZE     = 20      # trading days
BEST_K          = 4       # regimes
BEST_LAM        = 0.1     # clustering weight
BEST_HIDDEN_DIM = 256
BEST_LATENT_DIM = 64
```

Expected outputs match Table 5.9 in the draft report (Sharpe: DRN=0.284, HMM=0.850).

---

## Ablation Studies

| Study | Finding |
|-------|---------|
| **A — RNN cell type** | LSTM chosen; GRU collapsed to 1 cluster at λ=0.1 |
| **B — Effect of λ** | Valid range: 0.01–0.1; collapse at λ ≥ 0.2 |
| **C — Number of regimes K** | K=4 best composite score; collapse at K ≥ 5 |
| **D — Joint vs Sequential** | Joint training underperforms sequential by −16.5% Silhouette |

---

## Limitations & Future Work

**Current limitations:**
- ~400 effective independent observations — insufficient for a model of this complexity
- DEC has no temporal coherence mechanism → mean run lengths of 3–5 days (HMM: 14–27 days)
- Single asset class (S&P 500) limits generalisability

**Recommended next steps:**
1. Add a **temporal coherence loss** penalising rapid regime switching between successive windows
2. Implement **IDEC local structure preservation** to prevent latent distortion during joint training
3. Expand to **multi-asset dataset** (20+ S&P 500 IT stocks → ~130,000 windows)
4. Explore **Mixture-VAE** frameworks with temporal state-regularisation terms

---

## Citation

```bibtex
@misc{deepregimenet2026,
  author    = {Wangui, Ian Kihara and Monanyun, Grace Nmon and Kimani, Immaculate Wanjiru},
  title     = {Deep-RegimeNet: An Unsupervised Deep Clustering Framework
               for Adaptive Market Regime Identification},
  year      = {2026},
  school    = {WorldQuant University},
  note      = {MScFE Capstone Project, Group 16019}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Peer reviewed by Group 16029 (En Chong Lok & Kurt Keissinger McKenzie), whose rigorous
feedback materially improved the evaluation framework, baseline suite, and methodological
rigour of this project.*
