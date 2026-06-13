"""
Quantitative measures of emergent functional specialization.

Given a trained :class:`~positronic_brain.multimodal.MultiStreamBrain` and a set of
per-stream sample embeddings, these tools answer: *have the zones specialized for the
stream that enters them?* They are the scientific instruments of the project:

* :func:`selectivity_index` — for each zone, which stream it most responds to, and how
  selectively (0 = unselective, →1 = responds to one stream only).
* :func:`zone_decoding_accuracy` — how well a linear probe recovers which stream drove
  the network from its activity (separability of the streams in neural space).
* :func:`lesion_effect` — the increase in each stream's reconstruction error when a zone
  is silenced (modality-specific deficits, as in cortical lesions).
* :func:`representational_dissimilarity` / :func:`rsa` — compare representational
  geometries (e.g. a zone's geometry vs the input-embedding geometry).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


@torch.no_grad()
def selectivity_index(model, stream_samples: Dict[str, torch.Tensor]) -> Dict[str, Tuple[str, float]]:
    """Per-zone preferred stream and selectivity.

    ``stream_samples`` maps each stream name to an ``(M, dim)`` batch of embeddings.
    Returns ``{zone_name: (preferred_stream, selectivity)}`` where selectivity is
    ``(r_max − r_others) / (r_max + r_others)`` of the zone's mean firing rate.
    """
    streams = list(stream_samples)
    rows = [model.zone_activity({s: stream_samples[s]}).mean(0) for s in streams]  # each (Z,)
    A = torch.stack(rows)                                   # (S, Z)
    pref = A.argmax(0)                                      # (Z,)
    r_max = A.max(0).values                                 # (Z,)
    r_oth = (A.sum(0) - r_max) / max(len(streams) - 1, 1)   # (Z,)
    sel = (r_max - r_oth) / (r_max + r_oth + 1e-8)
    zone_names = model.brain.config.zone_names
    return {zone_names[z]: (streams[int(pref[z])], float(sel[z])) for z in range(len(zone_names))}


def _linear_probe(X: torch.Tensor, y: torch.Tensor, n_classes: int,
                  epochs: int = 150, lr: float = 0.05, seed: int = 0) -> float:
    """Train a logistic-regression probe and return held-out accuracy."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(X.shape[0], generator=g)
    X, y = X[perm], y[perm]
    n_tr = int(0.7 * X.shape[0])
    Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    clf = torch.nn.Linear(X.shape[1], n_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(clf(Xtr), ytr)
        loss.backward(); opt.step()
    with torch.no_grad():
        acc = (clf(Xte).argmax(1) == yte).float().mean().item()
    return acc


@torch.no_grad()
def _collect(model, stream_samples, per_zone: bool):
    feats, labels, names = [], [], list(stream_samples)
    for i, s in enumerate(names):
        f = (model.zone_activity if per_zone else model.neuron_rates)({s: stream_samples[s]})
        feats.append(f.detach())
        labels.append(torch.full((f.shape[0],), i, dtype=torch.long))
    return torch.cat(feats), torch.cat(labels), names


def zone_decoding_accuracy(model, stream_samples: Dict[str, torch.Tensor],
                           per_zone: bool = False) -> float:
    """Held-out accuracy of a linear probe decoding *which stream* drove the network.

    Chance level is ``1 / n_streams``; above-chance accuracy means the streams are
    linearly separable in the brain's activity (the routing/specialization is real).
    Uses per-neuron rates by default, or per-zone means if ``per_zone=True``.
    """
    X, y, names = _collect(model, stream_samples, per_zone)
    return _linear_probe(X.float(), y, n_classes=len(names))


@torch.no_grad()
def lesion_effect(model, inputs: Dict[str, torch.Tensor], zone: str) -> Dict[str, float]:
    """Increase in each stream's reconstruction MSE when ``zone`` is silenced.

    A large, stream-specific increase is a modality-specific deficit — evidence that
    the zone carries that stream's representation.
    """
    brain = model.brain
    prepped = model._prep(inputs)
    recon, rates = model.perceive(prepped, return_rates=True)
    base = {s: float(((recon[s] - prepped[s]) ** 2).mean()) for s in prepped}
    zid = brain.config.zone_names.index(zone)
    mask = (brain.zones == zid)
    rates_l = rates.clone()
    rates_l[:, mask] = 0.0
    out = {}
    for s in prepped:
        rec_l = brain.sensory_decode[s](rates_l)
        out[s] = float(((rec_l - prepped[s]) ** 2).mean()) - base[s]
    return out


@torch.no_grad()
def representational_dissimilarity(activity: torch.Tensor) -> torch.Tensor:
    """RDM = 1 − Pearson correlation between every pair of stimulus activity vectors."""
    a = activity - activity.mean(1, keepdim=True)
    a = a / (a.norm(dim=1, keepdim=True) + 1e-8)
    corr = a @ a.t()
    return 1.0 - corr


@torch.no_grad()
def rsa(rdm_a: torch.Tensor, rdm_b: torch.Tensor) -> float:
    """Pearson correlation between the upper triangles of two RDMs (RSA score)."""
    n = rdm_a.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    a, b = rdm_a[iu[0], iu[1]], rdm_b[iu[0], iu[1]]
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-8))
