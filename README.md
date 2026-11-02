# RL-Guided DiGress for Targeted Molecular Generation
### CHE657 — Machine Learning for Chemical Engineering | Group Project

> **Project:** RL-Guided Discrete Graph Diffusion for Targeted and Constrained De Novo Molecular Generation
> **Course:** CHE657 | **Semester:** 5 | **Credits:** 9
> **Team:** 4 members | **Timeline:** 3 months

---

## 📋 Project Summary

This repository implements **RL fine-tuning of the DiGress discrete graph diffusion model** for targeted molecular generation. Starting from a pre-trained DiGress model (which generates valid but unconstrained molecules), we apply **REINFORCE policy gradient optimization** to steer the generation toward molecules with:

- High **QED** (Quantitative Estimate of Drug-likeness)
- Optimal **logP** (lipophilicity in the Lipinski drug-like range 2–5)

while maintaining high **chemical validity** and **molecular uniqueness**.

---

## 🗂️ Repository Structure

```
rl-digress-mol-gen/
├── CODEBASE_ANALYSIS.md        ← DETAILED code documentation (READ THIS FIRST)
├── README.md                   ← This file
├── requirements.txt            ← Python dependencies
├── setup.py                    ← Package installer
│
├── configs/                    ← Hydra configuration files
│   ├── config.yaml             ← Root config
│   ├── dataset/qm9.yaml        ← QM9 dataset settings
│   ├── experiment/
│   │   ├── debug.yaml          ← Quick debug run (no GPU needed)
│   │   ├── qm9_no_h.yaml       ← Full baseline training
│   │   └── rl_qm9.yaml         ← RL fine-tuning config  ← OUR CONFIG
│   ├── general/general_default.yaml
│   ├── model/discrete.yaml
│   └── train/train_default.yaml
│
├── src/
│   ├── main.py                 ← Entry point
│   ├── diffusion_model_discrete.py  ← Core: discrete diffusion model
│   ├── diffusion_model.py      ← Continuous variant (not used)
│   ├── utils.py                ← General utilities
│   ├── rl_finetuning.py        ← *** OUR RL CODE *** (stub — implement in Phase 2)
│   │
│   ├── models/
│   │   ├── transformer_model.py ← GraphTransformer (the denoising NN / RL policy)
│   │   └── layers.py           ← Attention layers
│   │
│   ├── diffusion/
│   │   ├── noise_schedule.py   ← Cosine noise schedule + transition matrices
│   │   ├── diffusion_utils.py  ← Math: sampling, posteriors, masking
│   │   ├── extra_features.py   ← Cycle counts + Laplacian eigenvectors
│   │   └── extra_features_molecular.py
│   │
│   ├── datasets/
│   │   ├── qm9_dataset.py      ← QM9 auto-download + preprocessing
│   │   └── ...
│   │
│   ├── metrics/
│   │   ├── molecular_metrics.py ← Validity, Uniqueness, Novelty, QED, logP
│   │   └── ...
│   │
│   └── analysis/
│       ├── rdkit_functions.py  ← Graph → molecule, SMILES, reward computation
│       └── ...
│
└── generated_samples/          ← Pre-generated samples from DiGress paper
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.9 or 3.10
- **Linux/WSL2** (required for `graph-tool`) or Google Colab
- CUDA GPU for full training (Colab T4 works)

### Step 1: Create Conda Environment

```bash
conda create -n digress python=3.9 -y
conda activate digress
```

### Step 2: Install PyTorch (with CUDA)

```bash
# For Colab / CUDA 11.8
pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install PyTorch Geometric

```bash
pip install torch_geometric==2.3.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Step 4: Install RDKit (via conda — pip often fails)

```bash
conda install -c conda-forge rdkit -y
```

### Step 5: Install graph-tool (Linux/WSL/Colab only)

```bash
conda install -c conda-forge graph-tool -y
```

### Step 6: Install remaining dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### Step 7: Verify

```bash
python -c "import torch; import torch_geometric; from rdkit import Chem; print('All good!')"
```

---

## 🚀 Quick Start

### Test your setup (no GPU needed, runs in ~1 minute)

```bash
cd src
python main.py +experiment=debug
```

### Train baseline DiGress on QM9 (requires GPU, ~10-15 hours on T4)

```bash
cd src
python main.py +experiment=qm9_no_h
```

### RL Fine-Tuning (Phase 2 — implement `src/rl_finetuning.py` first)

```bash
# Set checkpoint path in configs/experiment/rl_qm9.yaml first
cd src
python main.py +experiment=rl_qm9
```

---

## 📊 Benchmark Targets

| Metric | Baseline DiGress | Our RL Target |
|--------|-----------------|---------------|
| Validity | ~98.8% | ≥95% |
| Uniqueness | ~95.7% | ≥95% |
| Novelty | ~97.2% | ≥99% |
| **Mean QED** | ~0.47 | **≥0.65** ← key result |
| **% QED > 0.6** | ~35% | **≥60%** ← key result |
| **% logP in [2,5]** | ~45% | **≥65%** ← key result |

---

## 📅 Timeline

| Phase | Weeks | Goal |
|-------|-------|------|
| 1: Baseline | 1–4 | Reproduce DiGress; collect baseline metrics |
| 2: RL Environment | 5–8 | Implement `rl_finetuning.py`; test reward loop |
| 3: Training & Eval | 9–12 | Run RL; compare vs. baseline; write report |

See `CODEBASE_ANALYSIS.md` for the detailed week-by-week task breakdown.

---

## 📖 Key References

1. **DiGress** (Base model): Vignac et al., *Discrete Denoising Diffusion for Graph Generation*, ICLR 2023. [Paper](https://openreview.net/forum?id=UaAD-Nu86WX) | [Code](https://github.com/cvignac/DiGress)

2. **DiffMeta-RL** (Closest related work): *RL-Guided Graph Diffusion for Metabolically Stable Molecular Generation*. [GitHub](https://github.com/prokia/DiffMeta-RL)

3. **REINFORCE** (RL algorithm): Williams, *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*, Machine Learning, 1992.

4. **QM9 Dataset**: Ramakrishnan et al., *Quantum chemistry structures and properties of 134 kilo molecules*, Scientific Data, 2014.

5. **RDKit**: Open-source cheminformatics library. [rdkit.org](https://www.rdkit.org/)

---

## 👥 Team

| Name | Role |
|------|------|
| Member 1 | Environment setup, baseline training |
| Member 2 | Baseline training, RDKit reward |
| Member 3 | RL loop implementation (Phase 2) |
| Member 4 | Literature review, evaluation, report |

---

## 📝 Notes for the Team

> [!IMPORTANT]
> Before writing ANY new code, read `CODEBASE_ANALYSIS.md` completely.
> It explains every function in the codebase and tells you exactly which files
> to modify and which to leave untouched.

> [!TIP]
> Always test new code with the debug config first:
> `python src/main.py +experiment=debug`
> This runs with tiny batch sizes (4 molecules) and no GPU in under 2 minutes.

> [!WARNING]
> `graph-tool` is required by `src/main.py` but is only available on Linux.
> On Windows, use **WSL2** or **Google Colab** for all training runs.

---

*CHE657 Group Project | November 2026*
