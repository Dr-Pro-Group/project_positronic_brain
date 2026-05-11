"""
Sparse 3D graph connectivity for the Positronic Brain.

Neurons live on a cubic lattice inside the unit cube. Instead of a dense
``N x N`` weight matrix (which costs O(N^2) memory and compute and does not
scale), connectivity is stored as a **sparse edge list**:

    edge_index : (2, E) int64   - row 0 = source/presynaptic neuron j
                                  row 1 = target/postsynaptic neuron i
    edge_dist  : (E,)  float32  - Euclidean distance between the two neurons

Edges are drawn with a **distance-biased, fan-in-capped** rule that mirrors real
cortical wiring: each neuron preferentially connects to nearby neurons (Gaussian
decay with distance), with a hard radius cut-off and a cap ``k_max`` on the
number of incoming connections. This keeps the graph sparse and O(E).

Synaptic weights are initialised separately by :func:`init_edge_weights`, which
also applies **Dale's law**: a fixed fraction of neurons are inhibitory and *all*
of their outgoing synapses are negative; the rest are excitatory (positive).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def neuron_positions(grid_size: int) -> np.ndarray:
    """
    Return (N, 3) float32 neuron coordinates on the cubic lattice.

    Index order is row-major: ``index = x*G*G + y*G + z``.
    """
    G = grid_size
    xs, ys, zs = np.meshgrid(np.arange(G), np.arange(G), np.arange(G), indexing="ij")
    return np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1).astype(np.float32)


def laminar_bands(grid_size: int, bands: int = 3) -> np.ndarray:
    """Assign each neuron a cortical-layer band from its depth (z) coordinate.

    The cube's z-axis is read as cortical depth, sliced into ``bands`` laminar
    bands. With 3 bands the convention is 0 = L2/3 (supragranular, high z),
    1 = L4 (granular, middle), 2 = L5/6 (infragranular, deep). Returns (N,) int.
    """
    G = int(grid_size)
    z = neuron_positions(grid_size)[:, 2]
    b = np.floor(z * bands / G).astype(np.int64)
    return np.clip(b, 0, bands - 1)


def canonical_laminar_motif(bands: int = 3) -> np.ndarray:
    """Canonical intracolumnar connection-strength multipliers M[src_band, dst_band].

    Encodes the stereotyped cortical flow L4 → L2/3 → L5/6 (Douglas & Martin 2004;
    Bastos et al. 2012): strong feedforward L4→L2/3, L2/3→L5/6, within-layer
    recurrence, and weaker feedback. Used to bias (not replace) the distance prior,
    so edge count and locality are preserved. Bands order: 0=L2/3, 1=L4, 2=L5/6.
    """
    if bands == 3:
        return np.array([
            # to:  L2/3  L4   L5/6     from:
            [1.0,  0.2,  1.5],   # L2/3 -> strong to L5/6 (infragranular output)
            [1.8,  1.0,  0.6],   # L4   -> strong to L2/3 (canonical feedforward)
            [0.4,  0.2,  1.0],   # L5/6 -> mostly recurrent
        ], dtype=np.float64)
    # Generic fallback: mild feedforward bias toward the next-shallower band.
    M = np.ones((bands, bands), dtype=np.float64)
    return M


def build_sparse_graph(
    grid_size: int = 4,
    connection_radius: float = 2.6,
    k_max: int = 16,
    decay_sigma: float = 1.75,
    seed: int = 42,
    self_connections: bool = False,
    laminar_motif: np.ndarray = None,
    laminar_n_bands: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a sparse, distance-biased connectivity graph.

    For every target neuron ``i`` we consider all neurons within
    ``connection_radius``, sample at most ``k_max`` of them as presynaptic
    sources with probability proportional to a Gaussian of their distance
    (``exp(-d^2 / (2 sigma^2))``), and record the resulting directed edges.

    Args:
        grid_size: side length of the cubic volume (N = grid_size**3 neurons).
        connection_radius: maximum Euclidean distance for a candidate edge.
        k_max: maximum number of incoming (presynaptic) connections per neuron.
        decay_sigma: standard deviation of the Gaussian distance bias.
        seed: RNG seed for reproducible sampling.
        self_connections: if False, neurons never connect to themselves.

    Returns:
        edge_index: (2, E) int64, row 0 = source j, row 1 = target i.
        edge_dist:  (E,) float32 distances.
        pos:        (N, 3) float32 neuron positions.
    """
    rng = np.random.default_rng(seed)
    G = int(grid_size)
    pos = neuron_positions(grid_size)
    N = pos.shape[0]

    # Because every neuron sits on a regular integer lattice and the connection
    # radius is small, a neuron's candidate presynaptic partners are exactly the
    # lattice points within a fixed set of (dx, dy, dz) offsets. We enumerate
    # those offsets ONCE and reuse them for every neuron, which makes graph
    # construction O(N * k) instead of the old O(N^2) full distance matrix
    # (the latter needs ~N^2 floats and does not scale beyond a few thousand
    # neurons). Sampling semantics — Gaussian distance bias and the k_max fan-in
    # cap — are identical to before.
    R = int(np.floor(connection_radius))
    axis = np.arange(-R, R + 1)
    ox, oy, oz = np.meshgrid(axis, axis, axis, indexing="ij")
    offs = np.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=1).astype(np.int64)
    odist = np.sqrt((offs ** 2).sum(axis=1)).astype(np.float32)
    keep = odist <= connection_radius
    if not self_connections:
        keep &= ~np.all(offs == 0, axis=1)
    offs = offs[keep]
    odist = odist[keep]
    # Gaussian distance bias is the same for every neuron (depends only on offset).
    offw = np.exp(-(odist ** 2) / (2.0 * decay_sigma ** 2)).astype(np.float64)

    coords = pos.astype(np.int64)        # integer lattice coordinates, (N, 3)
    GG = G * G

    # Optional laminar motif: bias each candidate's sampling probability by the
    # canonical layer→layer strength M[band(source), band(target)]. This reshapes
    # WHICH local neighbours are chosen (preserving locality, k_max and edge count)
    # to follow the cortical L4→L2/3→L5/6 flow, instead of pure distance.
    use_lam = laminar_motif is not None
    if use_lam:
        nbands = laminar_motif.shape[0]
        band = laminar_bands(grid_size, nbands)          # (N,) band per neuron

    src_list: list[int] = []
    dst_list: list[int] = []
    dst_dist: list[float] = []

    for i in range(N):
        nb = coords[i] + offs            # candidate neighbour coordinates (M, 3)
        inb = (
            (nb[:, 0] >= 0) & (nb[:, 0] < G)
            & (nb[:, 1] >= 0) & (nb[:, 1] < G)
            & (nb[:, 2] >= 0) & (nb[:, 2] < G)
        )
        if not np.any(inb):
            continue
        nb = nb[inb]
        d = odist[inb]
        w = offw[inb]
        idx = nb[:, 0] * GG + nb[:, 1] * G + nb[:, 2]   # row-major neuron index

        if use_lam:
            w = w * laminar_motif[band[idx], band[i]]    # canonical layer motif

        probs = w / w.sum()
        k = min(k_max, idx.size)
        chosen = rng.choice(idx.size, size=k, replace=False, p=probs)

        for c in chosen:
            src_list.append(int(idx[c]))   # presynaptic
            dst_list.append(i)             # postsynaptic
            dst_dist.append(float(d[c]))

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_dist = np.array(dst_dist, dtype=np.float32)
    return edge_index, edge_dist, pos


def init_edge_weights(
    edge_index: np.ndarray,
    edge_dist: np.ndarray,
    num_neurons: int,
    decay_sigma: float = 1.75,
    frac_inhibitory: float = 0.2,
    g_max: float = 0.4,
    inh_scale: float = 4.0,
    seed: int = 42,
    is_inhibitory: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialise per-edge synaptic weights obeying Dale's law.

    A neuron is either excitatory (positive outgoing weights) or inhibitory
    (negative outgoing weights); the sign is a property of the *source* neuron,
    not of the individual synapse. Magnitudes decay with distance (closer
    neurons form stronger synapses on average) plus a little jitter.

    Inhibitory synapses are initialised stronger than excitatory ones
    (``inh_scale``), reflecting the cortical "detailed balance" in which a
    smaller population of inhibitory neurons exerts proportionally stronger
    control to keep the network stable and out of runaway excitation.

    Args:
        edge_index: (2, E) int64 edge list from :func:`build_sparse_graph`.
        edge_dist:  (E,) float32 edge distances.
        num_neurons: total neuron count N.
        decay_sigma: Gaussian scale for the distance-dependent magnitude.
        frac_inhibitory: fraction of neurons that are inhibitory (~0.2 cortex).
        g_max: maximum excitatory synaptic conductance scale.
        inh_scale: multiplier applied to inhibitory synapse magnitudes.
        seed: RNG seed.

    Returns:
        edge_weight: (E,) float32 signed weights (sign = Dale's law).
        is_inhibitory: (N,) bool mask, True where a neuron is inhibitory.
    """
    rng = np.random.default_rng(seed + 1)
    N = num_neurons
    src = edge_index[0]

    # Decide neuron polarity (Dale's law) once per neuron, unless a placement mask
    # is supplied (e.g. the laminar microcircuit's spatially-even inhibition).
    if is_inhibitory is None:
        n_inh = int(round(frac_inhibitory * N))
        is_inhibitory = np.zeros(N, dtype=bool)
        if n_inh > 0:
            inh_ids = rng.choice(N, size=n_inh, replace=False)
            is_inhibitory[inh_ids] = True
    else:
        is_inhibitory = np.asarray(is_inhibitory, dtype=bool)

    # Distance-decayed magnitude with multiplicative jitter, scaled by g_max.
    mag = np.exp(-(edge_dist ** 2) / (2.0 * decay_sigma ** 2))
    jitter = 1.0 + 0.15 * rng.standard_normal(edge_dist.shape[0]).astype(np.float32)
    mag = np.clip(mag * jitter, 0.0, None) * g_max

    inh_edge = is_inhibitory[src]
    mag = mag.copy()
    mag[inh_edge] *= inh_scale

    sign = np.where(inh_edge, -1.0, 1.0).astype(np.float32)
    edge_weight = (sign * mag).astype(np.float32)
    return edge_weight, is_inhibitory


# -----------------------------------------------------------------------------
# Backwards-compatible dense builder (deprecated).
# -----------------------------------------------------------------------------
def build_distance_biased_connectivity(
    grid_size: int = 4,
    connection_radius: float = 2.6,
    decay_sigma: float = 1.75,
    sparsity: float = 0.12,
    seed: int = 42,
    **_ignored,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DEPRECATED dense connectivity builder kept for backwards compatibility.

    Returns a dense ``(N, N)`` weight matrix and distance matrix reconstructed
    from the sparse graph. New code should use :func:`build_sparse_graph`.
    """
    edge_index, edge_dist, pos = build_sparse_graph(
        grid_size=grid_size,
        connection_radius=connection_radius,
        decay_sigma=decay_sigma,
        seed=seed,
    )
    N = pos.shape[0]
    edge_weight, _ = init_edge_weights(
        edge_index, edge_dist, N, decay_sigma=decay_sigma, seed=seed
    )
    W = np.zeros((N, N), dtype=np.float32)
    src, dst = edge_index
    W[dst, src] = edge_weight  # W[i, j]: weight of edge j -> i
    diff = pos[:, None, :] - pos[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=2)).astype(np.float32)
    return W, D
