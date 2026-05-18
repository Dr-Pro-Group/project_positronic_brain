#!/usr/bin/env python3
"""
train.py - Training script for Project Positronic Brain v3

Unlike v2 (which froze the connectome and trained only a readout head), v3
learns the **synapses themselves**. Each edge in the sparse 3D graph carries a
trainable conductance magnitude; Dale's law is preserved automatically because
the sign of every synapse is fixed by its presynaptic neuron's polarity while
the learnable magnitude is constrained to be non-negative.

The synthetic task teaches the multi-modal brain to map combinations of zone
activations (Visual, Auditory, Somatosensory, Memory, Emotion, Association) to a
meaningful scalar "salience / integration" signal, including cross-modal logic.

Supports optional Metal (MPS) on Apple Silicon MacBook Pro via --device.

Usage:
    python train.py
    python train.py --epochs 80 --grid-size 4
    python train.py --device mps          # Apple Metal on MacBook Pro (M-series)
    python train.py --device cpu          # Force CPU (often fastest for tiny G=4)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from positronic_brain.model import PositronicBrain, BrainConfig
from positronic_brain.utils import get_device


def generate_synthetic_data(
    n_samples: int,
    num_zones: int,
    seed: int = 123,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic (zone_inputs, target) pairs for the 6-zone brain.

    The target is a continuous "salience" value in [0, 1] driven by soft,
    partly cross-modal rules:

    - Co-active Visual + Auditory  -> external event (strong drive)
    - Somatosensory + Emotion      -> embodied/affective salience
    - Memory with low sensory      -> internal replay (moderate)
    - Association amplifies any coherent multi-modal pattern
    - Conflicting / pure-noise patterns -> low

    Generalises to any zone count: the first up-to-6 channels carry the named
    semantics above; extra channels (if any) act as mild distractors.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n_samples, num_zones)).astype(np.float32)

    def col(i):
        return X[:, i] if i < num_zones else np.zeros(n_samples, dtype=np.float32)

    visual, auditory, touch, memory, emotion, assoc = (col(i) for i in range(6))

    sensory = 0.6 * visual + 0.5 * auditory
    embodied = 0.5 * touch + 0.45 * emotion
    internal = 0.45 * memory * (1.0 - 0.5 * sensory)
    cross = 0.4 * (visual * auditory) + 0.3 * (touch * emotion) + 0.2 * (memory * assoc)

    logit = 1.7 * sensory + 1.3 * embodied + 1.1 * internal + 1.8 * cross
    logit += 1.2 * assoc * (sensory + embodied)  # association gates integration
    logit -= 1.6
    logit += rng.normal(0.0, 0.3, size=n_samples)
    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -8, 8)))

    y = prob.astype(np.float32).reshape(-1, 1)
    return torch.from_numpy(X), torch.from_numpy(y)


def train(
    epochs: int = 70,
    batch_size: int = 32,
    lr: float = 0.01,
    grid_size: int = 4,
    recurrent_steps: int = 12,
    l1_synapse: float = 1e-4,
    seed: int = 42,
    save_path: str = "trained_models/positronic_brain_v2.pt",
    device: str = "auto",
) -> PositronicBrain:
    """
    Run the training loop and return the trained model.

    Args:
        l1_synapse: L1 penalty on synaptic magnitudes encouraging sparse,
            interpretable connectivity (set 0 to disable).
        device: "auto" (default), "cpu", "mps" (Apple Metal), or "cuda".
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    resolved_device = get_device(device)
    print(f"Using device: {resolved_device} (requested: {device})")

    cfg = BrainConfig(
        grid_size=grid_size,
        recurrent_steps=recurrent_steps,
        seed=seed,
        readout_hidden=18,
    )
    model = PositronicBrain(cfg, device=device)
    model.train()
    Z = cfg.num_zones

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    X_train, y_train = generate_synthetic_data(2400, Z, seed=seed + 1)
    X_val, y_val = generate_synthetic_data(500, Z, seed=seed + 999)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=128, shuffle=False)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training PositronicBrain v3 (G={grid_size}, N={cfg.num_neurons}, "
          f"E={model.num_edges}, steps={recurrent_steps})")
    print(f"  Zones: {cfg.zone_names}")
    print(f"  Trainable parameters: {n_trainable:,} "
          f"(of which {model.num_edges:,} are learnable synapses)")
    print(f"  Inhibitory neurons: {int(model.is_inhibitory.sum())}/{cfg.num_neurons}")
    print("-" * 64)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(resolved_device)
            yb = yb.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            if l1_synapse > 0:
                loss = loss + l1_synapse * model.edge_weight.abs().mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(resolved_device)
                yb = yb.to(resolved_device)
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | train: {train_loss:.4f} | val BCE: {val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"\n[ok] Best model saved to {save_path} (val BCE: {best_val:.4f})")

    # Sanity-check forward passes on hand-crafted multi-modal patterns.
    print("\nSanity-check forward passes (after training):")
    names = cfg.zone_names
    test_cases = {
        "Threat (visual+audio)":      {"Visual": 0.95, "Auditory": 0.9},
        "Calm memory recall":         {"Memory": 0.9},
        "Painful touch + emotion":    {"Somatosensory": 0.9, "Emotion": 0.85},
        "Cross-modal integration":    {"Visual": 0.6, "Auditory": 0.55, "Association": 0.8},
        "Low activity baseline":      {},
    }
    for label, active in test_cases.items():
        vec = np.zeros(Z, dtype=np.float32)
        for zname, val in active.items():
            if zname in names:
                vec[names.index(zname)] = val
        res = model.run_with_inputs(vec)
        print(f"  {label:28s} -> output={float(res['output'].ravel()[0]):.3f}  "
              f"mean rate={res['rates'].mean():.3f}")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Positronic Brain v3 (learnable synapses)")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--recurrent-steps", type=int, default=12)
    parser.add_argument("--l1-synapse", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="trained_models/positronic_brain_v2.pt")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"],
        help="auto (default) = MPS on Apple Silicon if available, else CPU.",
    )
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grid_size=args.grid_size,
        recurrent_steps=args.recurrent_steps,
        l1_synapse=args.l1_synapse,
        seed=args.seed,
        save_path=args.save_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
