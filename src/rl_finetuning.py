"""
rl_finetuning.py
================
RL Fine-Tuning module for CHE657 Group Project.

Project: RL-Guided Discrete Graph Diffusion for Targeted and Constrained
         De Novo Molecular Generation

This module wraps a pre-trained DiscreteDenoisingDiffusion model and fine-tunes
it using the REINFORCE policy gradient algorithm.

The reward signal is computed via RDKit:
  - QED  : Quantitative Estimate of Drug-likeness  [0, 1]  (higher = better)
  - logP : octanol-water partition coefficient — Gaussian reward centered at
           mu=3.5, sigma=1.5 (Lipinski rule of five favors logP in [2, 5])

Reward function:
  R(G_0) = 0                                  if G_0 is chemically invalid
  R(G_0) = alpha * QED + beta * f(logP)       otherwise

  where f(logP) = exp(-0.5 * ((logP - mu) / sigma)^2)

Usage
-----
This file is a scaffold — method bodies marked TODO need to be filled in
by the team in Phase 2 (Weeks 5–8). The mathematical formulas and pseudocode
are documented in CODEBASE_ANALYSIS.md Section 7.

Author: CHE657 Group | Date: November 2026
"""

import torch
import torch.nn as nn
import numpy as np
import wandb
from typing import List, Tuple, Optional

# RDKit imports (make sure rdkit is installed: conda install -c conda-forge rdkit)
try:
    from rdkit.Chem import QED, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("[WARNING] RDKit not found. Reward computation will fail.")

from src.analysis.rdkit_functions import (
    build_molecule_with_partial_charges,
    mol2smiles
)
from src.diffusion import diffusion_utils
from src import utils


# ---------------------------------------------------------------------------
# Reward Configuration (tune these hyperparameters)
# ---------------------------------------------------------------------------
ALPHA_QED    = 0.7    # Weight for QED reward
BETA_LOGP    = 0.3    # Weight for logP reward
LOGP_TARGET  = 3.5   # Target logP value (center of Gaussian reward)
LOGP_SIGMA   = 1.5   # Gaussian width for logP reward


def compute_reward_single(
    atom_types: torch.Tensor,
    edge_types: torch.Tensor,
    atom_decoder: list,
    alpha: float = ALPHA_QED,
    beta: float  = BETA_LOGP,
    logp_target: float = LOGP_TARGET,
    logp_sigma:  float = LOGP_SIGMA,
) -> float:
    """
    Compute the scalar reward for a single generated molecule.

    Parameters
    ----------
    atom_types : torch.Tensor of shape (n,)
        Integer atom type indices for each atom.
    edge_types : torch.Tensor of shape (n, n)
        Integer bond type indices. 0 = no bond, 1 = single, 2 = double,
        3 = triple, 4 = aromatic.
    atom_decoder : list of str
        Maps index → atom symbol, e.g. ['C', 'N', 'O', 'F']
    alpha, beta : float
        Reward weights for QED and logP terms.
    logp_target, logp_sigma : float
        Gaussian reward parameters for logP.

    Returns
    -------
    float
        Reward in [0, 1]. Returns 0.0 for invalid molecules.
    """
    assert RDKIT_AVAILABLE, "RDKit must be installed to compute rewards."

    # 1. Build RDKit molecule from the generated graph
    mol = build_molecule_with_partial_charges(atom_types, edge_types, atom_decoder)

    # 2. Check if the molecule is chemically valid (has a valid SMILES)
    smiles = mol2smiles(mol)
    if smiles is None:
        return 0.0  # Invalid molecule → zero reward

    # 3. Compute QED (Quantitative Estimate of Drug-likeness)
    #    Range: [0, 1] where 1 is ideal drug-like.
    qed_score = QED.qed(mol)

    # 4. Compute logP (octanol-water partition coefficient)
    logp_value = Descriptors.MolLogP(mol)

    # 5. Shape logP reward: Gaussian centered at logp_target
    #    This rewards molecules with logP near the target (2–5 range).
    logp_reward = float(np.exp(-0.5 * ((logp_value - logp_target) / logp_sigma) ** 2))

    # 6. Combine rewards
    reward = alpha * qed_score + beta * logp_reward
    return float(reward)


def compute_rewards_batch(
    molecule_list: List[Tuple[torch.Tensor, torch.Tensor]],
    atom_decoder: list,
    **reward_kwargs
) -> torch.Tensor:
    """
    Compute rewards for a batch of molecules.

    Parameters
    ----------
    molecule_list : list of (atom_types, edge_types) tuples
    atom_decoder  : list mapping index → atom symbol

    Returns
    -------
    torch.Tensor of shape (batch_size,)
    """
    rewards = []
    for atom_types, edge_types in molecule_list:
        r = compute_reward_single(atom_types, edge_types, atom_decoder, **reward_kwargs)
        rewards.append(r)
    return torch.tensor(rewards, dtype=torch.float32)


# ---------------------------------------------------------------------------
# RL Fine-Tuner
# ---------------------------------------------------------------------------

class RLFineTuner:
    """
    Fine-tunes a pre-trained DiscreteDenoisingDiffusion model using REINFORCE.

    The key modification is that during RL fine-tuning, we run the reverse
    diffusion sampler WITHOUT @torch.no_grad(), so gradients can flow back
    through the log-probabilities of each denoising step.

    Algorithm (REINFORCE with baseline):
      For each episode:
        1. Sample a batch of molecules using the current policy π_θ
        2. Collect trajectory log-probabilities: log_prob = Σ_t log p_θ(z_{t-1}|z_t)
        3. Compute RDKit rewards: R(G_0)
        4. Compute baseline b = mean(R) [reduces variance]
        5. Compute REINFORCE loss: L = -mean[(R - b) * log_prob]
        6. Backpropagate L and update θ with AdamW(lr=rl_lr)
    """

    def __init__(
        self,
        model,                     # DiscreteDenoisingDiffusion instance
        dataset_infos,             # QM9infos instance
        rl_lr: float = 1e-5,       # Very small LR to avoid forgetting
        rl_batch_size: int = 32,   # Molecules per RL step
        n_episodes: int = 2000,    # Total RL training episodes
        log_every: int = 10,       # Log to wandb every N episodes
        save_every: int = 100,     # Save checkpoint every N episodes
        device: str = 'cuda',
    ):
        self.model = model
        self.dataset_infos = dataset_infos
        self.atom_decoder = dataset_infos.atom_decoder
        self.rl_lr = rl_lr
        self.rl_batch_size = rl_batch_size
        self.n_episodes = n_episodes
        self.log_every = log_every
        self.save_every = save_every
        self.device = device

        # Fine-tune ONLY the GraphTransformer weights (model.model)
        # The noise schedule and transition matrices are not trainable.
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=rl_lr,
            weight_decay=1e-12
        )

    def sample_batch_with_logprob(
        self,
        batch_size: int
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Run the reverse diffusion sampler and collect log-probabilities.

        This is a gradient-enabled version of DiscreteDenoisingDiffusion.sample_batch().
        The @torch.no_grad() decorator is NOT used here so gradients flow.

        TODO (Phase 2, Week 6): Implement this method.
        See CODEBASE_ANALYSIS.md Section 7.2 for detailed pseudocode.

        Key steps:
          1. Sample initial noise z_T from limit distribution
          2. For t = T, T-1, ..., 1:
             - Call model.sample_p_zs_given_zt_with_logprob() [see below]
             - Accumulate step_log_prob into trajectory_log_prob
          3. Collapse final z_0 to discrete atom/bond categories
          4. Return (molecule_list, trajectory_log_prob)

        Returns
        -------
        molecule_list : list of (atom_types [n], edge_types [n,n]) tuples
        trajectory_log_prob : torch.Tensor of shape (batch_size,)
        """
        # TODO: Implement gradient-enabled sampling loop
        # Reference: src/diffusion_model_discrete.py sample_batch() lines 491–595
        raise NotImplementedError(
            "TODO (Week 6): Implement gradient-enabled sampling loop. "
            "See CODEBASE_ANALYSIS.md Section 7.2 for pseudocode."
        )

    def rl_step(self) -> Tuple[float, float, float]:
        """
        Execute one REINFORCE training step.

        Returns
        -------
        loss_val : float — REINFORCE loss
        mean_reward : float — average reward in this batch
        validity : float — fraction of valid molecules in this batch
        """
        self.model.model.train()

        # 1. Sample molecules with gradient tracking
        molecule_list, log_probs = self.sample_batch_with_logprob(self.rl_batch_size)

        # 2. Compute RDKit rewards for each molecule
        rewards = compute_rewards_batch(molecule_list, self.atom_decoder)
        rewards = rewards.to(self.device)

        # 3. Compute baseline (batch mean) for variance reduction
        baseline = rewards.mean()
        advantages = rewards - baseline  # Centered advantages

        # 4. REINFORCE loss: we want to MAXIMIZE E[R * log_prob]
        #    so we MINIMIZE the negative: L = -mean[(R - b) * log_prob]
        loss = -(advantages.detach() * log_probs).mean()

        # 5. Backpropagate and update
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents catastrophic forgetting
        torch.nn.utils.clip_grad_norm_(
            self.model.model.parameters(), max_norm=1.0
        )
        self.optimizer.step()

        # 6. Compute validity fraction
        validity = sum(1 for r in rewards if r > 0) / len(rewards)

        return loss.item(), rewards.mean().item(), validity

    def train(self, save_path: str = 'checkpoints/rl_finetuned.ckpt'):
        """
        Main RL training loop.

        Parameters
        ----------
        save_path : str — path to save checkpoints
        """
        print(f"\n{'='*60}")
        print(f"Starting RL Fine-Tuning")
        print(f"  Episodes:    {self.n_episodes}")
        print(f"  Batch size:  {self.rl_batch_size}")
        print(f"  LR:          {self.rl_lr}")
        print(f"  Reward:      {ALPHA_QED}*QED + {BETA_LOGP}*f(logP)")
        print(f"{'='*60}\n")

        best_reward = -float('inf')

        for episode in range(self.n_episodes):
            loss, mean_reward, validity = self.rl_step()

            if episode % self.log_every == 0:
                print(
                    f"Episode {episode:4d}/{self.n_episodes} | "
                    f"Loss: {loss:.4f} | "
                    f"Reward: {mean_reward:.4f} | "
                    f"Validity: {validity:.2%}"
                )
                if wandb.run:
                    wandb.log({
                        'rl/episode': episode,
                        'rl/loss': loss,
                        'rl/mean_reward': mean_reward,
                        'rl/validity': validity,
                    })

            if episode % self.save_every == 0 and mean_reward > best_reward:
                best_reward = mean_reward
                torch.save(self.model.state_dict(), save_path)
                print(f"  ✓ Checkpoint saved (best reward: {best_reward:.4f})")

        print(f"\nRL Fine-Tuning complete. Best mean reward: {best_reward:.4f}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_rl_model(
    model,
    dataset_infos,
    n_molecules: int = 1000,
    device: str = 'cuda',
) -> dict:
    """
    Generate molecules from the RL fine-tuned model and compute statistics.

    TODO (Phase 3, Week 10): Implement evaluation.

    Metrics computed:
      - validity, uniqueness, novelty (standard DiGress metrics)
      - mean QED, % QED > 0.6, % QED > 0.8
      - mean logP, % logP in [2, 5]
      - mean SA score (synthetic accessibility)

    Returns
    -------
    dict with all metric values
    """
    raise NotImplementedError(
        "TODO (Week 10): Implement evaluation of the RL fine-tuned model."
    )


# ---------------------------------------------------------------------------
# Entry point for standalone testing
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Testing reward computation on a simple molecule...")
    import torch
    # Caffeine-like: C, C, N, N  with some bonds
    atom_types = torch.tensor([1, 1, 2, 2])  # C, C, N, N  (indices for remove_h=True)
    edge_types = torch.zeros(4, 4, dtype=torch.long)
    edge_types[0, 1] = 1  # C-C single bond
    edge_types[1, 0] = 1
    atom_decoder = ['C', 'N', 'O', 'F']
    r = compute_reward_single(atom_types, edge_types, atom_decoder)
    print(f"Reward for test molecule: {r:.4f}")
