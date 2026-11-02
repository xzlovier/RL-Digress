# DiGress Codebase Analysis & RL Fine-Tuning Roadmap
## CHE657 — RL-Guided Discrete Graph Diffusion for Targeted Molecular Generation

> **Group Project | 4 members | 3 months**
> Base repository: [cvignac/DiGress](https://github.com/cvignac/DiGress)
> Base paper: *DiGress: Discrete Denoising Diffusion for Graph Generation*, Vignac et al., ICLR 2023

---

## Table of Contents
1. [Repository Structure Overview](#1-repository-structure-overview)
2. [File-by-File Code Analysis](#2-file-by-file-code-analysis)
   - [Entry Point](#21-entry-point--srcmainpy)
   - [Core Diffusion Model (Discrete)](#22-core-diffusion-model--srcdiffusion_model_discretepy)
   - [Core Diffusion Model (Continuous)](#23-continuous-diffusion-model--srcdiffusion_modelpy)
   - [Graph Transformer (Neural Network)](#24-graph-transformer--srcmodelstransformer_modelpy)
   - [Transformer Layers](#25-transformer-layers--srcmodelslayerspy)
   - [Noise Schedule](#26-noise-schedule--srcdiffusionnoise_schedulepy)
   - [Diffusion Utilities](#27-diffusion-utilities--srcdiffusiondiffusion_utilspy)
   - [Extra Features](#28-extra-features--srcdiffusionextra_featurespy)
   - [QM9 Dataset](#29-qm9-dataset--srcdatasetsqm9_datasetpy)
   - [RDKit Functions](#210-rdkit-functions--srcanalysisrdkit_functionspy)
   - [Molecular Metrics](#211-molecular-metrics--srcmetricsmsamplingy)
   - [Training Loss](#212-training-loss--srcmetricstrain_metricspy)
   - [Utilities](#213-utilities--srcutilspy)
3. [Configuration System (Hydra YAML)](#3-configuration-system-hydra-yaml)
4. [Data Flow Diagram](#4-data-flow-diagram)
5. [Environment Requirements](#5-environment-requirements)
6. [Benchmarks to Evaluate Against](#6-benchmarks-to-evaluate-against)
7. [RL Innovation: What to Change and Where](#7-rl-innovation-what-to-change-and-where)
8. [Files to Modify vs. Files to Leave Alone](#8-files-to-modify-vs-files-to-leave-alone)
9. [Recommended Timeline](#9-recommended-timeline)

---

## 1. Repository Structure Overview

```
DiGress/
├── configs/                    ← Hydra YAML configuration files
│   ├── config.yaml             ← Root config (picks defaults)
│   ├── dataset/                ← Per-dataset configs (qm9, moses, guacamol, etc.)
│   ├── experiment/             ← Full experiment overrides (e.g. qm9_no_h.yaml, debug.yaml)
│   ├── general/                ← General settings (logging, validation freq, GPU)
│   ├── model/                  ← Model architecture configs (discrete.yaml, continuous.yaml)
│   └── train/                  ← Training hyperparameters (lr, batch_size, epochs)
│
├── src/
│   ├── main.py                 ← Entry point. Orchestrates everything.
│   ├── diffusion_model_discrete.py  ← ** THE CORE FILE ** Discrete diffusion Lightning module
│   ├── diffusion_model.py      ← Continuous diffusion variant (less relevant for our project)
│   ├── utils.py                ← Helpers: PlaceHolder, to_dense, normalize, setup_wandb
│   │
│   ├── models/
│   │   ├── transformer_model.py ← GraphTransformer (the denoising neural network / "brain")
│   │   └── layers.py           ← NodeEdgeBlock, XEyTransformerLayer, Xtoy, Etoy
│   │
│   ├── diffusion/
│   │   ├── noise_schedule.py   ← Noise schedule classes: cosine betas, transition matrices
│   │   ├── diffusion_utils.py  ← Math helpers: sampling, posteriors, KL, masking
│   │   ├── extra_features.py   ← Graph structural features fed to transformer (cycles, eigenvectors)
│   │   └── extra_features_molecular.py ← Chemistry-specific features for molecular graphs
│   │
│   ├── datasets/
│   │   ├── abstract_dataset.py ← Base DataModule class
│   │   ├── qm9_dataset.py      ← QM9 download, preprocessing, DataModule
│   │   ├── guacamol_dataset.py ← GuacaMol drug-like dataset
│   │   ├── moses_dataset.py    ← MOSES benchmark dataset
│   │   └── spectre_dataset.py  ← Graph topology datasets (planar, SBM, comm20)
│   │
│   ├── metrics/
│   │   ├── abstract_metrics.py ← Base metric classes (NLL, KL, CrossEntropy)
│   │   ├── molecular_metrics.py ← Validity/Uniqueness/Novelty + RDKit evaluation
│   │   ├── molecular_metrics_discrete.py ← Discrete-specific training metrics
│   │   └── train_metrics.py    ← TrainLoss, TrainLossDiscrete (Cross-Entropy for nodes/edges)
│   │
│   └── analysis/
│       ├── rdkit_functions.py  ← Build molecules from graphs, compute validity/SMILES
│       ├── spectre_utils.py    ← Graph topology metrics (for non-molecular)
│       └── visualization.py   ← Molecule/graph visualizations
│
├── generated_samples/          ← Pre-generated samples from the paper (reference)
├── requirements.txt            ← Python dependencies
└── setup.py                    ← Package install
```

---

## 2. File-by-File Code Analysis

### 2.1 Entry Point — `src/main.py`

**Purpose:** Top-level script. Decides which dataset and model type to use, builds all components, and hands control to PyTorch Lightning's `Trainer`.

**Key code blocks:**

| Block | Lines | What it does |
|-------|-------|--------------|
| `get_resume()` | 26–38 | Loads a checkpoint for **testing only** (config is frozen). |
| `get_resume_adaptive()` | 41–64 | Loads a checkpoint for **resuming training** (allows some config changes). |
| `@hydra.main` `main()` | 68–223 | **Main function.** Routes to the right dataset, builds `datamodule`, `dataset_infos`, `train_metrics`, `sampling_metrics`, model, and `Trainer`. |
| Dataset routing block | 72–151 | `if dataset == 'sbm/comm20/planar'` → uses `SpectreGraphDataModule`. `elif dataset == 'qm9/guacamol/moses'` → uses molecular data modules. **For our project: always the `qm9` branch.** |
| Model instantiation | 164–167 | Creates `DiscreteDenoisingDiffusion` (for `model.type == 'discrete'`). |
| `Trainer` setup | 190–200 | PyTorch Lightning Trainer. Sets GPU, gradient clipping, epochs, callbacks. |
| `trainer.fit()` | 203 | **Starts training.** Calls `training_step()` on the model repeatedly. |
| `trainer.test()` | 205 | After training, evaluates on the test set and generates final molecules. |

**What to modify for RL:** Add an RL fine-tuning phase *after* `trainer.fit()` completes (or load a pre-trained checkpoint). This is where you call your custom RL training loop.

---

### 2.2 Core Diffusion Model — `src/diffusion_model_discrete.py`

**The most important file.** This is the `DiscreteDenoisingDiffusion` class — a `pl.LightningModule` (PyTorch Lightning module). It contains the full training loop, validation, sampling, and the diffusion math.

#### Constructor `__init__()` (lines 18–101)

Sets up:
- `self.model = GraphTransformer(...)` — **the neural network denoiser** (the thing RL will fine-tune)
- `self.noise_schedule = PredefinedNoiseScheduleDiscrete(...)` — cosine β schedule
- `self.transition_model = MarginalUniformTransition(...)` — the categorical noise transition matrices Q_t
- `self.train_loss = TrainLossDiscrete(...)` — cross-entropy loss for nodes/edges
- `self.limit_dist` — the stationary distribution (pure noise, what we start sampling from)

#### `training_step()` (lines 103–120) ⬅ **CHANGE THIS for RL**

Standard diffusion training step:
1. Convert sparse graph to dense `(X, E)` tensors
2. Call `apply_noise()` → corrupt the clean graph at a random timestep
3. Call `forward()` → pass noisy graph through GraphTransformer to predict clean graph
4. Compute **cross-entropy loss** between predicted and true atoms/bonds
5. Return loss for backprop

> **RL Modification:** You will **add an RL training step** alongside this. Instead of computing cross-entropy, you will compute a **policy gradient loss** = `−R(G₀) × Σ log p_θ(G_{t-1} | G_t)`.

#### `configure_optimizers()` (lines 122–124)

Uses `AdamW` optimizer. **For RL phase**, you will likely freeze the main optimizer and add a separate optimizer with a lower learning rate (to avoid catastrophic forgetting).

#### `apply_noise()` (lines 407–442) — Forward diffusion

Takes a clean molecule `(X, E)` and corrupts it to a noisy version at random timestep `t`:
1. Sample random `t ∈ [0, T]`
2. Compute transition matrix `Qtb` (the cumulative noise at timestep t)
3. Multiply: `probX = X @ Qtb.X` — probability of each atom type at time t
4. Sample a noisy graph from these probabilities

**Do NOT change this.** The forward diffusion is fixed math.

#### `sample_batch()` (lines 491–595) ⬅ **CRITICAL for RL**

**The reverse diffusion sampler.** This generates new molecules.

1. Sample pure noise `z_T` from `self.limit_dist`
2. Loop backward from `t=T` to `t=0`:
   - Call `sample_p_zs_given_zt()` — use the GraphTransformer to predict `z_{t-1}` from `z_t`
3. Return `molecule_list` — list of `(atom_types, edge_types)` tuples

> **RL Modification:** This function is decorated `@torch.no_grad()`. **You must remove this decorator** during RL training so gradients flow through the sampling trajectory. You also need to collect `log_prob` at each denoising step.

#### `sample_p_zs_given_zt()` (lines 597–655) ⬅ **THE KEY FUNCTION FOR RL**

**Single denoising step** — samples `z_{t-1}` given `z_t`:
1. Get transition matrices `Qt`, `Qsb`, `Qtb`
2. Call `self.forward()` → get `pred_X, pred_E` (predicted clean graph probabilities)
3. Compute posterior `p(z_{t-1} | z_t, pred_X)` using `compute_batched_over0_posterior_distribution()`
4. **Sample from this posterior** using `multinomial()` sampling

> **RL Modification:** You need to record the **log probability** of each sample taken here: `log_prob += torch.log(prob_X).gather(-1, sampled_X)`. This builds the trajectory log-probability needed for REINFORCE.

#### `forward()` (lines 485–489)

Concatenates `(noisy_data, extra_features)` and passes to `self.model` (the GraphTransformer). Returns predicted `(X, E, y)` logits.

#### `compute_val_loss()` (lines 444–483)

Computes the ELBO (Evidence Lower BOund) for evaluation. **Not used during RL training.**

#### `kl_prior()`, `compute_Lt()`, `reconstruction_logp()` — Mathematical losses

Used in validation/test for ELBO computation. **Leave unchanged.**

---

### 2.3 Continuous Diffusion Model — `src/diffusion_model.py`

**Not needed for our project.** This is the continuous (Gaussian noise) version of the diffusion model — it works on coordinates, not discrete categories. We use the discrete model exclusively.

---

### 2.4 Graph Transformer — `src/models/transformer_model.py`

**The neural network.** This is what learns to denoise molecular graphs and what RL will fine-tune.

#### `GraphTransformer` class (lines 216–284)

Architecture:
```
Input (X, E, y)
     ↓
mlp_in_X: Linear(input_dims['X'], hidden_mlp_dims['X']) → Linear → hidden_dims['dx']=256
mlp_in_E: Linear(input_dims['E'], hidden_mlp_dims['E']) → Linear → hidden_dims['de']=64
mlp_in_y: Linear(input_dims['y'], hidden_mlp_dims['y']) → Linear → hidden_dims['dy']=64
     ↓
[XEyTransformerLayer × n_layers]   (n_layers=9 for QM9 no-H)
     ↓
mlp_out_X: Linear(256, 256) → Linear(256, num_atom_types)
mlp_out_E: Linear(64, 128) → Linear(128, num_edge_types)
mlp_out_y: Linear(64, 128) → Linear(128, ydim_output)
     ↓
Residual connection + symmetrize E
     ↓
Output (pred_X logits, pred_E logits, pred_y logits)
```

**This is the RL policy π_θ.** Updating its weights via policy gradients IS the RL fine-tuning.

#### `forward()` (lines 255–284)

1. Project inputs to hidden dimension
2. Pass through N transformer layers
3. Project back to output dimension
4. Add residual skip connection
5. Symmetrize edge matrix (molecules have undirected bonds)

---

### 2.5 Transformer Layers — `src/models/layers.py`

#### `XEyTransformerLayer` (lines 16–92)

A single transformer layer that jointly updates node features X, edge features E, and global features y.

Architecture within each layer:
```
NodeEdgeBlock (self-attention with edge conditioning)
    ↓
Layer Norm + Dropout (for X, E, y separately)
    ↓
Feed-Forward MLP (for X, E, y separately)
    ↓
Layer Norm + Dropout
```

#### `NodeEdgeBlock` (lines 95–213)

**The core attention mechanism.** Custom self-attention that uses edge features to modulate attention scores (FiLM conditioning):

```
Q, K, V computed from X (node features)
Edge features E modulate attention via FiLM: Y = Y * (E_mul + 1) + E_add
Global features y modulate E and X via FiLM
New y = f(X_pooled, E_pooled, y)
```

#### `Xtoy`, `Etoy` (lines 5–38)

Pool node/edge features to global vector: `[mean, min, max, std]` → linear projection.

#### `masked_softmax` (lines 41–46)

Softmax that sets masked entries to `-inf` before softmax, preventing padded nodes from affecting attention.

---

### 2.6 Noise Schedule — `src/diffusion/noise_schedule.py`

#### `PredefinedNoiseScheduleDiscrete` (lines 44–79)

Implements the **cosine noise schedule** for discrete diffusion:
- `betas[t]` = noise level at step t (probability of flipping a category)
- `alphas_bar[t]` = cumulative product of (1 - beta) — probability a token is still its original value at step t
- `forward(t_normalized)` → returns `beta_t`
- `get_alpha_bar(t_normalized)` → returns `ᾱ_t`

#### `DiscreteUniformTransition` (lines 82–135)

The **uniform transition model** (baseline):
- `get_Qt(beta_t)` → one-step transition matrix: `Qt = (1-β)I + β/K * 1`
- `get_Qt_bar(alpha_bar_t)` → cumulative transition matrix

#### `MarginalUniformTransition` (lines 138–187) ← Used in practice

The **marginal transition model** (better baseline, used in the paper):
- Transitions toward the marginal distribution of the training data (not uniform)
- e.g., if most atoms are Carbon, noise transitions preferentially toward Carbon
- `get_Qt(beta_t)` → `Qt = (1-β)I + β * marginal`

#### `AbsorbingStateTransition` (lines 190–223)

Experimental: absorbing state (MASK token) transitions. Not used in the main paper runs.

---

### 2.7 Diffusion Utilities — `src/diffusion/diffusion_utils.py`

Helper functions for the diffusion math:

| Function | What it does |
|----------|--------------|
| `cosine_beta_schedule_discrete()` | Computes cosine schedule betas array |
| `sample_discrete_features()` | Samples node/edge categories from probability distributions (multinomial) |
| `sample_discrete_feature_noise()` | Samples pure noise from the stationary/limit distribution (for sampling start) |
| `compute_posterior_distribution()` | Computes `q(z_{t-1} | z_t, x_0)` — the posterior used in sampling |
| `compute_batched_over0_posterior_distribution()` | Batched version of posterior over all possible x_0 values |
| `mask_distributions()` | Zeros out masked (padding) positions in distributions |
| `posterior_distributions()` | Combines node and edge posteriors |
| `assert_correctly_masked()` | Sanity check: ensures padding rows are zero |

**Key insight for RL:** `sample_discrete_features()` uses `.multinomial(1)` — this is where stochastic sampling happens and where you need to record log-probabilities.

---

### 2.8 Extra Features — `src/diffusion/extra_features.py`

Computes additional structural features of the noisy graph appended to the transformer input:

| Class/Function | Features computed |
|---------------|-------------------|
| `DummyExtraFeatures` | Returns empty tensors (no extra features) |
| `ExtraFeatures` | Dispatches to cycle and/or eigenvalue features |
| `NodeCycleFeatures` | k-cycle counts (3,4,5-cycles) per node and globally |
| `EigenFeatures` | Laplacian eigenvalues, eigenvectors, connected component indicators |
| `KNodeCycles` | Efficient cycle counting using matrix powers of adjacency |

For the `extra_features='all'` setting (used in QM9 no-H), the transformer input includes:
- Node features: atom type one-hot + 3-cycle, 4-cycle, 5-cycle counts + connectivity indicator + 2 Laplacian eigenvectors
- Global features: node count + cycle counts + # connected components + 5 Laplacian eigenvalues

**Do NOT change this.** These features are already tuned for QM9.

---

### 2.9 QM9 Dataset — `src/datasets/qm9_dataset.py`

#### `QM9Dataset` (lines 53–191)

PyTorch Geometric `InMemoryDataset` for QM9:
- **Download URLs:** Automatically downloads from DeepChem S3 + figshare
- **Atoms encoded:** H=0, C=1, N=2, O=3, F=4 (or C=0,N=1,O=2,F=3 if remove_h=True)
- **Bonds encoded:** SINGLE=1, DOUBLE=2, TRIPLE=3, AROMATIC=4, No bond=0
- **Split:** 100,000 train / ~13,000 val / ~13,000 test
- Each molecule stored as `Data(x, edge_index, edge_attr, y, idx)`

#### `QM9DataModule` (lines 194–218)

Standard PyTorch Lightning `DataModule`. Creates train/val/test datasets and dataloaders.

#### `QM9infos` (lines 221–281)

Dataset statistics hardcoded:
- `max_n_nodes`: 9 (no H) / 29 (with H)
- `atom_decoder`: maps index → atom symbol
- `node_types`: empirical frequency of each atom type
- `edge_types`: empirical frequency of each bond type

These statistics are used to set up the **marginal transition model** and the **limit distribution** for sampling.

#### `get_train_smiles()` (lines 284–320)

Converts training set to SMILES strings and caches them. Used for **novelty** computation.

---

### 2.10 RDKit Functions — `src/analysis/rdkit_functions.py`

**Critical file for our RL reward function.**

| Function | What it does |
|----------|--------------|
| `build_molecule(atom_types, edge_types, atom_decoder)` | Converts a `(atom_types, edge_types)` tuple to an RDKit `Mol` object |
| `build_molecule_with_partial_charges(...)` | Same, but also tries to fix valence errors by adding formal charges |
| `mol2smiles(mol)` | Sanitizes RDKit mol and returns SMILES string. Returns `None` if invalid. |
| `check_valency(mol)` | Checks if all atoms have valid valence. Returns `(True, None)` if valid. |
| `correct_mol(mol)` | Tries to fix invalid molecules by removing overfull bonds |
| `check_stability(atom_types, edge_types, dataset_info)` | Checks if each atom has the correct number of bonds |
| `BasicMolecularMetrics` | Class with `compute_validity()`, `compute_uniqueness()`, `compute_novelty()` |
| `compute_molecular_metrics()` | Master function: calls stability + validity + uniqueness + novelty |

**For our RL reward function, the key pipeline is:**
```python
from rdkit.Chem import QED, Descriptors
from src.analysis.rdkit_functions import build_molecule_with_partial_charges, mol2smiles

def compute_reward(atom_types, edge_types, atom_decoder):
    mol = build_molecule_with_partial_charges(atom_types, edge_types, atom_decoder)
    smiles = mol2smiles(mol)
    if smiles is None:
        return 0.0  # Invalid molecule → zero reward
    qed_score = QED.qed(mol)
    logp = Descriptors.MolLogP(mol)
    # Shape logP to reward values in [2, 5]
    logp_reward = np.exp(-0.5 * ((logp - 3.5) / 1.5)**2)
    return alpha * qed_score + beta * logp_reward
```

---

### 2.11 Molecular Metrics — `src/metrics/molecular_metrics.py`

#### `SamplingMolecularMetrics` (lines 57–161)

Called after every validation sampling run and final test. Computes:
- **Validity**: % of generated molecules that are chemically valid (pass RDKit sanitization)
- **Relaxed Validity**: % valid with partial charge correction
- **Uniqueness**: % of valid molecules that are unique SMILES
- **Novelty**: % of unique valid molecules not seen in training set
- **Distribution MAEs**: How well generated atom/bond/valency distributions match training data

These are the **benchmark metrics** you report in your paper.

---

### 2.12 Training Loss — `src/metrics/train_metrics.py`

#### `TrainLossDiscrete` (lines 62–123) ⬅ **WILL BE BYPASSED in RL**

Cross-entropy loss for the discrete diffusion training:
- `node_loss`: CE between predicted atom type logits and true atom types
- `edge_loss`: CE between predicted bond type logits and true bond types
- Weighted: `total_loss = loss_X + λ_E * loss_E + λ_y * loss_y`

For QM9 no-H: `lambda_train = [5, 0]` — edge loss gets 5× weight, y-loss is off.

During RL, you **replace** this loss with a **policy gradient loss**.

---

### 2.13 Utilities — `src/utils.py`

| Component | Purpose |
|-----------|---------|
| `PlaceHolder(X, E, y)` | Container for the triplet of node features, edge features, global features. Has `.mask()` and `.type_as()` methods. |
| `to_dense(x, edge_index, edge_attr, batch)` | Converts sparse PyG graph format to dense `(X, E)` matrices |
| `normalize()` / `unnormalize()` | Normalizes continuous features (not used in discrete model) |
| `encode_no_edge(E)` | Adds the "no bond" (class 0) indicator where no edge exists |
| `setup_wandb(cfg)` | Initializes Weights & Biases logging |
| `EMA` | Exponential Moving Average callback for model weights |

---

## 3. Configuration System (Hydra YAML)

The config system uses [Hydra](https://hydra.cc/) — configs are composed from multiple YAML files.

### Root: `configs/config.yaml`
```yaml
defaults:
    - general : general_default     # general/general_default.yaml
    - model : discrete              # model/discrete.yaml
    - train : train_default         # train/train_default.yaml
    - dataset : qm9                 # dataset/qm9.yaml
```

### Key config values for QM9 (from `configs/experiment/qm9_no_h.yaml`):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `model.n_layers` | 9 | Number of GraphTransformer layers |
| `model.diffusion_steps` (T) | 500 | Number of diffusion timesteps |
| `model.transition` | `marginal` | Use marginal distribution for noise |
| `model.diffusion_noise_schedule` | `cosine` | Cosine noise schedule |
| `model.hidden_dims.dx` | 256 | Node hidden dim |
| `model.hidden_dims.de` | 64 | Edge hidden dim |
| `model.lambda_train` | `[5, 0]` | Edge CE weight=5, y-loss off |
| `train.n_epochs` | 1000 | Total training epochs |
| `train.batch_size` | 1024 | Molecules per batch |
| `train.lr` | 0.0002 | AdamW learning rate |
| `general.gpus` | 1 | Number of GPUs |
| `dataset.remove_h` | True | Remove hydrogen atoms |
| `general.samples_to_generate` | 512 | Molecules sampled per validation |
| `general.final_model_samples_to_generate` | 10000 | Molecules for final evaluation |

### Debug config (`configs/experiment/debug.yaml`):

Run locally to test your code changes quickly:
```bash
python src/main.py +experiment=debug
```
This runs with `batch_size=4`, `n_layers=2`, `diffusion_steps=110` — fast to iterate on your laptop without GPU.

---

## 4. Data Flow Diagram

### Training Phase (Baseline DiGress)

```
QM9Dataset (SDF files) 
    → QM9DataModule (PyTorch Geometric DataLoader)
    → Batch of sparse graphs (x, edge_index, edge_attr)
    → to_dense() → (X [bs,n,5], E [bs,n,n,5])
    → apply_noise(X, E, t) → noisy (X_t, E_t) at random t
    → compute_extra_data(noisy) → cycle counts + eigenvectors
    → GraphTransformer(X_t ⊕ extra_X, E_t ⊕ extra_E, y_t ⊕ extra_y)
    → pred (X_logits, E_logits)
    → TrainLossDiscrete: CE(pred_X, true_X) + 5×CE(pred_E, true_E)
    → AdamW backward()
```

### Sampling Phase (Molecule Generation)

```
sample_discrete_feature_noise(limit_dist) → z_T (pure noise graph)
    → For t = T, T-1, ..., 1:
        sample_p_zs_given_zt(z_t, t) →
            compute_extra_data(z_t)
            GraphTransformer(z_t ⊕ extra) → pred_X, pred_E
            compute_batched_over0_posterior_distribution() → p(z_{t-1}|z_t, pred_X)
            multinomial(p(z_{t-1}|z_t)) → z_{t-1}   ← sample here
    → z_0 (generated molecular graph)
    → build_molecule_with_partial_charges(z_0)
    → mol2smiles(mol) → SMILES string
    → QED(mol), MolLogP(mol) → chemical properties
```

### RL Fine-Tuning Phase (Our Innovation)

```
Pre-trained GraphTransformer (θ, fixed weights as starting point)
    → RL Training Loop:
        For each episode:
            sample_batch_with_grad() [NO @torch.no_grad()] →
                collect log_probs = Σ_t log p_θ(z_{t-1} | z_t)
                generate z_0 (molecule)
            Compute reward R(z_0) via RDKit:
                if invalid → R = 0
                else → R = α·QED(mol) + β·Gaussian(logP, μ=3.5, σ=1.5)
            Compute baseline b = mean(R) over batch [variance reduction]
            Compute REINFORCE loss: L = -mean[(R - b) · log_probs]
            AdamW(lr=1e-5) backward(L) → update θ
```

---

## 5. Environment Requirements

### Python Version
- Python **3.9** or **3.10** (recommended; PyTorch Geometric requires specific versions)

### Core Dependencies (from `requirements.txt`)

```
torch==2.0.x              # PyTorch — the deep learning framework
torch-geometric==2.3.1    # PyTorch Geometric — for graph data structures
pytorch_lightning==2.0.4  # Training loop management
hydra-core==1.3.2         # Config management
omegaconf==2.3.0          # Config objects
```

### Scientific/Chemistry Dependencies
```
rdkit                     # NOT in requirements.txt! Must install separately:
                          # conda install -c conda-forge rdkit
                          # (conda is strongly recommended for rdkit)
numpy==1.23
scipy==1.11.0
pandas==1.4
networkx==2.8.7
```

### Visualization & Logging
```
wandb==0.15.4             # Weights & Biases logging
matplotlib==3.7.1
imageio==2.31.1
```

### Graph Metrics
```
pyemd==1.0.0              # Earth Mover Distance (for graph quality metrics)
PyGSP==0.5.1              # Graph signal processing
torchmetrics==0.11.4      # Metrics library
tqdm==4.65.0              # Progress bars
```

### Additional (Not in requirements.txt but needed)
```
rdkit                     # Chemical informatics — MUST install via conda
graph-tool                # Used in main.py import! Install via: conda install -c conda-forge graph-tool
                          # (Linux/Mac only; on Windows use WSL or Docker)
```

> [!WARNING]
> `graph-tool` is a C++ library that is **not available on Windows natively**. You will need to either:
> 1. Use WSL2 (Windows Subsystem for Linux) — recommended
> 2. Use Google Colab/Kaggle (Linux environment)
> 3. Use Docker with a Linux image

### Full Setup Commands

```bash
# 1. Create conda environment (STRONGLY recommended)
conda create -n digress python=3.9 -y
conda activate digress

# 2. Install PyTorch (CUDA 11.8 — for T4 GPU on Colab)
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# 3. Install PyTorch Geometric
pip install torch_geometric==2.3.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 4. Install rdkit (via conda — pip version often fails)
conda install -c conda-forge rdkit -y

# 5. Install graph-tool (Linux/WSL/Colab only)
conda install -c conda-forge graph-tool -y

# 6. Install remaining requirements
pip install -r requirements.txt

# 7. Install the package itself
pip install -e .

# 8. Verify
python -c "import torch; import torch_geometric; from rdkit import Chem; print('All good!')"
```

### Google Colab Setup (Recommended for GPU Training)

```python
# Run in Colab cell
!pip install torch==2.0.1+cu118 torch_geometric==2.3.1 pytorch_lightning==2.0.4
!pip install hydra-core==1.3.2 omegaconf wandb rdkit
!pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Clone the repo
!git clone https://github.com/YOUR_USERNAME/rl-digress-mol-gen
%cd rl-digress-mol-gen
!pip install -e .

# graph-tool (on Colab)
!apt-get install -y python3-graph-tool
```

---

## 6. Benchmarks to Evaluate Against

### Primary Dataset: QM9

| Property | Value |
|----------|-------|
| Name | QM9 (Quantum Machine 9) |
| Size | ~134,000 small organic molecules |
| Atoms | H, C, N, O, F |
| Max atoms | 9 (no H) / 29 (with H) |
| Source | Auto-downloaded by the code from DeepChem S3 |

**Standard split:** 100,000 train / ~13,000 val / ~13,000 test

### Benchmark Metrics (What to Report)

#### Generation Quality Metrics (from DiGress paper — Table 1)

| Metric | Definition | DiGress QM9 no-H result | Your target |
|--------|-----------|------------------------|-------------|
| **Validity** | % of generated mols that pass RDKit sanitization | **~98.8%** | ≥95% |
| **Relaxed Validity** | % valid with partial charge correction | ~99%+ | ≥97% |
| **Uniqueness** | % of valid mols that are unique SMILES | ~96%+ | ≥95% |
| **Novelty** | % of unique mols not in training set | ~100% | ≥99% |
| **V×U×N** | Product of the three (overall quality) | High | Should remain high after RL |

#### Property Optimization Metrics (RL Innovation Metrics)

These are the metrics that show **your RL contribution is working**:

| Metric | Definition | Baseline DiGress | RL Target |
|--------|-----------|-----------------|-----------|
| **Mean QED** | Average QED score of generated mols | ~0.47 | ≥0.65 |
| **% QED > 0.6** | Fraction with high drug-likeness | ~35% | ≥60% |
| **% QED > 0.8** | Fraction with very high drug-likeness | ~5% | ≥20% |
| **Mean logP** | Average partition coefficient | ~1.9 | Target: 2.0–5.0 range |
| **% logP in [2,5]** | Fraction in Lipinski window | ~45% | ≥65% |
| **Mean SA Score** | Synthetic accessibility (lower is easier to make) | ~2.8 | ≤2.8 (preserve) |

#### Distribution Metrics (from DiGress code — `SamplingMolecularMetrics`)

| Metric | Definition |
|--------|-----------|
| `n_dist_mae` | MAE between generated and real node count distributions |
| `node_dist_mae` | MAE between generated and real atom type distributions |
| `edge_dist_mae` | MAE between generated and real bond type distributions |
| `valency_dist_mae` | MAE between generated and real valency distributions |

#### Pre-existing Baseline Numbers from the Paper

From the DiGress paper (QM9, no hydrogens):

```
Validity: 98.8%
Uniqueness: 95.7%  
Novelty: 97.2%
```

Compare these against your RL model's validity/uniqueness/novelty to ensure RL fine-tuning does not collapse the generation quality.

### Secondary Datasets (Optional, if time allows)

| Dataset | Size | Use case |
|---------|------|----------|
| **MOSES** | 1.9M drug-like molecules | Larger-scale validation |
| **GuacaMol** | 1.6M drug-like molecules | GuacaMol benchmark suite |

---

## 7. RL Innovation: What to Change and Where

### Overview of Changes

The RL fine-tuning is a **post-training** step. It does NOT change the base diffusion training at all.

```
Phase 1: Run DiGress baseline training as-is (or use pre-trained checkpoint)
Phase 2: Load the pre-trained checkpoint, run RL fine-tuning loop
```

### New File to Create: `src/rl_finetuning.py`

This is the only **new file** you need to write. It implements the RL training loop.

**Pseudocode:**

```python
# src/rl_finetuning.py

import torch
import torch.nn.functional as F
from rdkit.Chem import QED, Descriptors
from src.analysis.rdkit_functions import build_molecule_with_partial_charges, mol2smiles

class RLFineTuner:
    def __init__(self, model, dataset_infos, cfg, lr=1e-5):
        self.model = model          # pre-trained DiscreteDenoisingDiffusion
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(),  # Only fine-tune the GraphTransformer
            lr=lr, weight_decay=1e-12
        )
        self.dataset_infos = dataset_infos
        self.atom_decoder = dataset_infos.atom_decoder
    
    def compute_reward(self, atom_types, edge_types):
        """Compute RDKit-based reward for a single molecule."""
        mol = build_molecule_with_partial_charges(atom_types, edge_types, self.atom_decoder)
        smiles = mol2smiles(mol)
        if smiles is None:
            return 0.0
        qed = QED.qed(mol)
        logp = Descriptors.MolLogP(mol)
        logp_reward = torch.exp(-0.5 * ((logp - 3.5) / 1.5) ** 2).item()
        return 0.7 * qed + 0.3 * logp_reward
    
    def sample_with_logprob(self, batch_size):
        """
        Modified sample_batch() that:
        1. Does NOT use @torch.no_grad()
        2. Collects log probabilities of each sampling decision
        """
        # ... (see Section 7.1 below for details)
    
    def rl_step(self, batch_size=32):
        """One RL training step (REINFORCE with baseline)."""
        molecule_list, log_probs = self.sample_with_logprob(batch_size)
        
        rewards = torch.tensor([
            self.compute_reward(at, et) 
            for at, et in molecule_list
        ], dtype=torch.float32)
        
        baseline = rewards.mean()          # Simple baseline: batch mean
        advantages = rewards - baseline    # Centered rewards
        
        # REINFORCE loss: maximize E[R * log_prob]
        loss = -(advantages.detach() * log_probs).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item(), rewards.mean().item()
```

### 7.1 Modifying `sample_p_zs_given_zt()` to Collect Log-Probs

In `diffusion_model_discrete.py`, line 597, the function `sample_p_zs_given_zt()` needs a version that returns log-probabilities:

```python
def sample_p_zs_given_zt_with_logprob(self, s, t, X_t, E_t, y_t, node_mask):
    """Same as sample_p_zs_given_zt but also returns log-probability of the sample."""
    # ... (existing code to get prob_X and prob_E) ...
    
    # Instead of pure sampling, also compute log-prob
    sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)
    
    # Compute log-prob of sampled choices
    log_prob_X = torch.log(prob_X + 1e-8).gather(-1, sampled_s.X.unsqueeze(-1)).squeeze(-1)
    log_prob_E = torch.log(prob_E + 1e-8).gather(-1, sampled_s.E.unsqueeze(-1)).squeeze(-1)
    
    # Sum log-probs over nodes and edges (for valid positions only)
    step_log_prob = (log_prob_X * node_mask).sum(dim=-1) + \
                    (log_prob_E * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)).sum(dim=(-1,-2))
    
    # ... (build X_s, E_s as before) ...
    return out_one_hot, out_discrete, step_log_prob
```

### 7.2 New Sampling Loop for RL (without `@torch.no_grad()`)

Add to `rl_finetuning.py`:

```python
def sample_batch_rl(self, batch_size):
    """Sampling loop that tracks gradients for RL."""
    model = self.model
    
    # Sample initial noise
    n_nodes = model.node_dist.sample_n(batch_size, model.device)
    n_max = torch.max(n_nodes).item()
    arange = torch.arange(n_max, device=model.device).unsqueeze(0).expand(batch_size, -1)
    node_mask = arange < n_nodes.unsqueeze(1)
    
    z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=model.limit_dist, node_mask=node_mask)
    X, E, y = z_T.X, z_T.E, z_T.y
    
    trajectory_log_prob = torch.zeros(batch_size, device=model.device)
    
    # Reverse diffusion with gradient tracking
    for s_int in reversed(range(0, model.T)):
        s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
        t_array = s_array + 1
        s_norm = s_array / model.T
        t_norm = t_array / model.T
        
        sampled_s, discrete_sampled_s, step_log_prob = \
            model.sample_p_zs_given_zt_with_logprob(s_norm, t_norm, X, E, y, node_mask)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
        trajectory_log_prob += step_log_prob
    
    # Collapse to discrete categories
    sampled_s = sampled_s.mask(node_mask, collapse=True)
    X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
    
    molecule_list = []
    for i in range(batch_size):
        n = n_nodes[i]
        molecule_list.append([X[i, :n].cpu(), E[i, :n, :n].cpu()])
    
    return molecule_list, trajectory_log_prob
```

### 7.3 New Config File for RL

Create `configs/experiment/rl_qm9.yaml`:

```yaml
# @package _global_
general:
    name: 'rl_qm9_no_h'
    gpus: 1
    wandb: 'online'
    test_only: null
train:
    batch_size: 32         # Smaller for RL (memory constrained)
    save_model: True
model:
    n_layers: 9
    lambda_train: [5, 0]
dataset:
    remove_h: True

# New RL-specific config
rl:
    n_rl_episodes: 2000    # Total RL training episodes
    rl_batch_size: 32      # Molecules per RL step
    rl_lr: 1e-5            # Very small LR (prevent catastrophic forgetting)
    alpha_qed: 0.7         # QED reward weight
    beta_logp: 0.3         # logP reward weight
    logp_target: 3.5       # Center of Gaussian logP reward
    logp_sigma: 1.5        # Width of Gaussian logP reward
    kl_penalty: 0.1        # Optional: KL divergence penalty vs. base model
    checkpoint_path: 'path/to/pretrained.ckpt'  # Path to Phase 1 checkpoint
```

---

## 8. Files to Modify vs. Files to Leave Alone

### 🔴 DO NOT MODIFY (Core DiGress baseline — keep for benchmarking)

| File | Reason |
|------|--------|
| `src/diffusion_model.py` | Continuous model, not used |
| `src/models/transformer_model.py` | Architecture stays the same for RL |
| `src/models/layers.py` | Leave unchanged |
| `src/diffusion/noise_schedule.py` | Fixed math |
| `src/diffusion/diffusion_utils.py` | Fixed math |
| `src/diffusion/extra_features.py` | Fixed input features |
| `src/diffusion/extra_features_molecular.py` | Fixed input features |
| `src/datasets/qm9_dataset.py` | Dataset loading stays the same |
| `src/datasets/abstract_dataset.py` | Base class |
| `src/metrics/abstract_metrics.py` | Base metrics |
| `src/utils.py` | General utilities |

### 🟡 MODIFY CAREFULLY (Add RL functionality without breaking baseline)

| File | What to add |
|------|------------|
| `src/diffusion_model_discrete.py` | Add `sample_p_zs_given_zt_with_logprob()` method alongside existing `sample_p_zs_given_zt()`. Add `sample_batch_rl()` method. Do NOT change `training_step()`. |
| `src/analysis/rdkit_functions.py` | Add QED + logP reward computation functions |
| `src/metrics/molecular_metrics.py` | Add property histogram reporting (QED, logP distributions) |
| `src/main.py` | Add optional RL fine-tuning phase that runs after `trainer.test()` |

### 🟢 CREATE NEW (RL infrastructure)

| File | What it contains |
|------|-----------------|
| `src/rl_finetuning.py` | `RLFineTuner` class with training loop, reward functions |
| `configs/experiment/rl_qm9.yaml` | RL experiment config |
| `notebooks/rl_analysis.ipynb` | Jupyter notebook for plotting QED/logP curves |
| `scripts/run_rl.py` | Command-line script to launch RL fine-tuning |

---

## 9. Recommended Timeline

### Phase 1: Baseline (Weeks 1–4)

**Goal:** Reproduce DiGress on QM9 (no-H). Use paper results directly if GPU time is limited.

| Week | Task | Person | Files |
|------|------|--------|-------|
| 1 | Environment setup (conda, rdkit, graph-tool in WSL/Colab) | All 4 | `requirements.txt`, `setup.py` |
| 1–2 | Run debug mode locally: `python src/main.py +experiment=debug` | Team A (2 people) | `src/main.py`, `configs/experiment/debug.yaml` |
| 2–3 | Launch full QM9 training on Colab (10–15 GPU hours) | Team A | `configs/experiment/qm9_no_h.yaml` |
| 2–4 | Literature review: Read DiGress paper, DiffMeta-RL paper | Team B (2 people) | — |
| 4 | Collect baseline metrics (Validity, Uniqueness, Novelty, QED, logP) | All 4 | `src/metrics/molecular_metrics.py`, `src/analysis/rdkit_functions.py` |

**Deliverable:** Table showing baseline DiGress results on QM9.

### Phase 2: RL Environment (Weeks 5–8)

**Goal:** Build and test the RL reward loop.

| Week | Task | Person | Files |
|------|------|--------|-------|
| 5 | Write `compute_reward()` function in `rdkit_functions.py`. Test it on the baseline generated samples. | Team B | `src/analysis/rdkit_functions.py` |
| 5–6 | Write `sample_p_zs_given_zt_with_logprob()` in `diffusion_model_discrete.py`. Unit test it. | Team A | `src/diffusion_model_discrete.py` |
| 6–7 | Write `sample_batch_rl()` and `RLFineTuner` class | Team A | `src/rl_finetuning.py` |
| 7–8 | Test full RL loop in debug mode (small batch, few steps) | All 4 | `src/rl_finetuning.py`, debug configs |

**Deliverable:** RL loop runs without errors in debug mode; rewards are computed correctly.

### Phase 3: Fine-Tuning & Evaluation (Weeks 9–12)

**Goal:** Train RL model and compare against baseline.

| Week | Task | Person | Files |
|------|------|--------|-------|
| 9–10 | Run full RL fine-tuning on Colab (5–10 GPU hours) | Team A | `configs/experiment/rl_qm9.yaml` |
| 10–11 | Generate evaluation metrics, plot QED/logP distributions | Team B | `notebooks/rl_analysis.ipynb` |
| 11–12 | Write final report, generate visualizations | All 4 | All |

**Deliverable:** Paper showing improved QED/logP while maintaining validity/uniqueness.

---

## Appendix: Key Mathematical Formulas

### Discrete Diffusion Forward Process

$$q(\mathbf{G}_t | \mathbf{G}_0) = \text{Cat}(\mathbf{X}\bar{Q}_t^X) \cdot \text{Cat}(\mathbf{E}\bar{Q}_t^E)$$

where $\bar{Q}_t = \bar{\alpha}_t \mathbf{I} + (1-\bar{\alpha}_t)\mathbf{m}^\top$ and **m** is the marginal distribution.

### Posterior (Used in Reverse Step)

$$q(\mathbf{G}_{t-1} | \mathbf{G}_t, \mathbf{G}_0) \propto [\mathbf{G}_t Q_t^\top] \odot [\mathbf{G}_0 \bar{Q}_{t-1}]$$

### RL Objective

$$J(\theta) = \mathbb{E}_{\mathbf{G}_0 \sim \pi_\theta} [R(\mathbf{G}_0)]$$

### REINFORCE Gradient Estimate

$$\nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N} (R(\mathbf{G}_0^{(i)}) - b) \sum_{t=1}^{T} \nabla_\theta \log p_\theta(\mathbf{G}_{t-1}^{(i)} | \mathbf{G}_t^{(i)})$$

where $b = \frac{1}{N}\sum_i R(\mathbf{G}_0^{(i)})$ is the baseline (batch mean reward).

### RDKit Reward Function

$$R(\mathbf{G}_0) = \begin{cases} 0 & \text{if } \mathbf{G}_0 \text{ is chemically invalid} \\ \alpha \cdot \text{QED}(\mathbf{G}_0) + \beta \cdot \exp\!\left(-\frac{(\log P(\mathbf{G}_0) - \mu)^2}{2\sigma^2}\right) & \text{otherwise} \end{cases}$$

Recommended: $\alpha=0.7$, $\beta=0.3$, $\mu=3.5$, $\sigma=1.5$.

---

*Document version: 1.0 | Created: November 2026 | CHE657 Group Project*
