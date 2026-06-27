"""
Core Positronic Brain model: a sparse, learnable, conductance-based 3D brain.

Biological mapping (the design philosophy of v3)
------------------------------------------------
Two complementary "regression" views, exactly as discussed:

* **Neuron = leaky linear integrator** ("linear regression").
  Each neuron holds a membrane potential ``V`` that linearly integrates its
  inputs and decays toward a resting potential ``E_L``:

      tau_m * dV_i/dt = -(V_i - E_L) + I_syn_i + I_ext_i + bias_i

* **Synapse = logistic transfer + conductance** ("logistic regression").
  A presynaptic neuron's firing *rate* is a logistic function of its membrane
  potential, ``r_j = sigmoid(gamma * (V_j - V_thr))``. Synapses are
  conductance-based, so the current they inject depends on the postsynaptic
  voltage through reversal potentials (driving force):

      g^E_i = sum_{j: excitatory} w_ij * r_j
      g^I_i = sum_{j: inhibitory} |w_ij| * r_j
      I_syn_i = g^E_i * (E_E - V_i) + g^I_i * (E_I - V_i)

Connectivity is a **sparse 3D graph** (``edge_index``), per-edge weights are
**learnable** (``nn.Parameter``), and neuron polarity follows **Dale's law**.
This makes the network scalable (O(E) not O(N^2)) and genuinely trainable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from .connectivity import (
    build_sparse_graph, init_edge_weights, neuron_positions,
    laminar_bands, canonical_laminar_motif,
)
from .zones import DEFAULT_ZONES, ZoneSpec, assign_zones, num_zones
from .utils import get_device


@dataclass
class BrainConfig:
    """
    Configuration for a :class:`PositronicBrain`.

    Geometry / graph:
        grid_size:         cube side length; N = grid_size**3 neurons.
        connection_radius: max distance for a candidate synapse.
        k_max:             max incoming (presynaptic) connections per neuron.
        decay_sigma:       Gaussian scale for distance-biased wiring + weights.
        frac_inhibitory:   fraction of neurons that are inhibitory (Dale).
        g_max:             initial synaptic conductance scale.

    Membrane / synapse dynamics:
        recurrent_steps:   number of integration steps per forward pass.
        tau_m:             membrane time constant (steps).
        dt:                integration step size; alpha = dt / tau_m.
        E_L:               leak/resting reversal potential.
        E_E:               excitatory reversal potential.
        E_I:               inhibitory reversal potential.
        gamma:             logistic gain of the firing-rate transfer.
        v_thr:             firing threshold of the transfer.
        input_gain:        scaling of external zone inputs into I_ext.

    Readout:
        readout_hidden:    hidden width of the MLP readout head.

    Misc:
        seed:              RNG seed for graph + weights.
        zone_names:        names of the active zones (defaults to 6-zone brain).
    """

    # Geometry / graph
    grid_size: int = 4
    connection_radius: float = 2.6
    k_max: int = 16
    decay_sigma: float = 1.75
    frac_inhibitory: float = 0.2
    g_max: float = 0.4
    inh_scale: float = 4.0

    # Dynamics
    recurrent_steps: int = 12
    tau_m: float = 4.0
    dt: float = 1.0
    E_L: float = 0.0
    E_E: float = 1.0
    E_I: float = -0.2
    gamma: float = 6.0
    v_thr: float = 0.5
    input_gain: float = 2.8

    # Readout
    readout_hidden: int = 16

    # Ablation switches (all True = full biology). These are first-class and
    # config-driven so ablations are reproducible from a saved checkpoint rather
    # than relying on runtime monkeypatching.
    #   use_conductance: conductance driving force g*(E - V); if False, plain
    #                    current-based synapses I_syn = gE - gI.
    #   use_dale:        Dale's law (per-source fixed sign); if False, synapses
    #                    are freely signed (no E/I constraint).
    #   spatial_wiring:  distance-biased 3D graph; if False, a random graph with
    #                    the same neuron count, edge count and Dale signs.
    use_conductance: bool = True
    use_dale: bool = True
    spatial_wiring: bool = True

    # Divisive normalization (Carandini & Heeger 2012): a canonical cortical
    # computation in which a neuron's excitatory drive is divided by the pooled
    # activity of its neighbourhood. Here the pool is the mean squared firing rate
    # of the neuron's zone, so the Voronoi zones gain a real computational role and
    # the recurrent dynamics are kept bounded (an inhibition-stabilised gain
    # control on top of the subtractive Dale inhibition). Off by default; when off
    # the dynamics are byte-identical to the un-normalised model.
    #   shunt g_norm = zone_mean(r^2) / dn_sigma^2 added to the membrane leak, so
    #   steady-state V is divided by (1 + g_norm + gE + gI).
    use_divnorm: bool = False
    dn_sigma: float = 0.5

    # Short-term synaptic plasticity (Tsodyks-Markram): per-synapse facilitation
    # (u) and depression (x) give the network a SECOND, slower-than-membrane
    # timescale carried in the synapses, the substrate of activity-silent working
    # memory (Mongillo et al. 2008). Effective weight is w * u * x. Off by default.
    use_stp: bool = False
    stp_tau_facil: float = 12.0   # facilitation time constant (steps)
    stp_tau_rec: float = 4.0      # depression / recovery time constant (steps)
    stp_U: float = 0.2            # baseline release probability increment

    # Homeostatic intrinsic-gain control (Turrigiano): a slow per-neuron gain that
    # nudges each neuron's firing rate toward a target set-point, keeping the
    # network active-but-bounded without leaning on grad-clip. Off by default; the
    # gain is a non-learnable buffer updated under no_grad during training.
    use_homeostasis: bool = False
    homeo_target_rate: float = 0.02
    homeo_tau: float = 200.0      # EMA time constant (steps) for the rate estimate
    homeo_lr: float = 0.01        # multiplicative gain adjustment rate

    # Oscillatory pacemaker (theta-like): a rhythmic drive applied to the
    # inhibitory population, giving the network a temporal clock / phase code
    # (Buzsaki; Lisman & Jensen theta-gamma). The phase advances one step per
    # integration step. Off by default.
    use_oscillation: bool = False
    osc_period: float = 8.0       # steps per cycle
    osc_amp: float = 0.3          # amplitude of the rhythmic inhibitory drive

    # Dendritic computation: group each neuron's incoming excitatory synapses into
    # branches and apply an NMDA-like thresholded (supralinear-onset) nonlinearity
    # per branch before summing — adding genuine per-neuron nonlinear depth
    # (Poirazi 2003; Beniaguev, Segev & London 2021). Off by default.
    use_dendrites: bool = False
    dend_branches: int = 4
    dend_gain: float = 4.0        # steepness of the per-branch NMDA gate
    dend_thr: float = 0.05        # branch activation threshold

    # Laminar microcircuit: read the cube's z-axis as cortical depth (L2/3, L4,
    # L5/6) and bias connectivity toward the canonical L4→L2/3→L5/6 flow
    # (Douglas & Martin 2004; Bastos et al. 2012). Also places inhibitory neurons
    # on a spatially-even sublattice instead of fully at random, so every neuron
    # has local inhibition (fixing the ~2%-of-targets-with-zero-inhibition gap).
    # Off by default; keeps edge count, k_max and Dale fraction unchanged.
    use_laminar: bool = False
    laminar_bands: int = 3

    # Sensory input pathway (real multimodal data -> per-neuron drive)
    #   sensory_embedding_dims:  modality name -> encoder embedding dimension.
    #   modality_to_zone:        modality name -> zone name it projects into.
    #   sensory_gain:            scaling of projected sensory current (None -> input_gain).
    # Leaving these empty yields the classic zone-scalar-only brain (back-compat).
    sensory_embedding_dims: Dict[str, int] = field(default_factory=dict)
    modality_to_zone: Dict[str, str] = field(default_factory=dict)
    sensory_gain: Optional[float] = None

    # Misc
    seed: int = 42
    zone_names: List[str] = field(default_factory=lambda: [z.name for z in DEFAULT_ZONES])

    @property
    def num_neurons(self) -> int:
        return self.grid_size ** 3

    @property
    def num_zones(self) -> int:
        return len(self.zone_names)

    @property
    def modalities(self) -> List[str]:
        """Names of the configured sensory modalities (stable order)."""
        return list(self.sensory_embedding_dims.keys())

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "BrainConfig":
        """Build a config from a dict, ignoring unknown / legacy keys."""
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in valid})


def _zone_specs_from_names(names: Sequence[str]) -> List[ZoneSpec]:
    """Resolve zone names to ZoneSpec objects from the default catalogue."""
    by_name = {z.name: z for z in DEFAULT_ZONES}
    specs: List[ZoneSpec] = []
    for nm in names:
        if nm in by_name:
            specs.append(by_name[nm])
        else:
            # Unknown zone name: synthesise a neutral spec at the cube centre.
            specs.append(ZoneSpec(nm, "#9CA3AF", (0.5, 0.5, 0.5), modality=True))
    return specs


class PositronicBrain(nn.Module):
    """
    A sparse, learnable, conductance-based 3D recurrent brain model.

    Trainable parameters:
        * ``edge_weight`` (E,)  - per-synapse conductance magnitude (signed by
          Dale's law via a fixed sign buffer; the raw parameter is unsigned).
        * ``neuron_bias`` (N,)  - per-neuron excitability bias.
        * ``zone_gain``   (Z,)  - per-zone input gain.
        * readout MLP           - maps final firing rates to a scalar output.
    """

    def __init__(
        self,
        config: Optional[BrainConfig] = None,
        device: Union[str, torch.device] = "auto",
    ):
        super().__init__()
        self.config = config or BrainConfig()
        cfg = self.config
        self._device = get_device(device) if isinstance(device, str) else device

        # Seed torch so the readout MLP (and any torch-side init) is
        # reproducible for a given config seed, matching the numpy-seeded graph.
        torch.manual_seed(cfg.seed)

        N = cfg.num_neurons
        Z = cfg.num_zones
        zspecs = _zone_specs_from_names(cfg.zone_names)

        # --- Build sparse connectivity graph ---
        # Laminar microcircuit (optional): bias wiring toward the canonical
        # L4→L2/3→L5/6 flow and place inhibition on a spatially-even sublattice.
        lam_motif = canonical_laminar_motif(cfg.laminar_bands) if cfg.use_laminar else None
        edge_index, edge_dist, pos = build_sparse_graph(
            grid_size=cfg.grid_size,
            connection_radius=cfg.connection_radius,
            k_max=cfg.k_max,
            decay_sigma=cfg.decay_sigma,
            seed=cfg.seed,
            laminar_motif=lam_motif,
            laminar_n_bands=cfg.laminar_bands,
        )
        inh_mask = None
        if cfg.use_laminar:
            # Spatially-even inhibitory placement: a quasi-periodic 3D pattern so
            # every neuron has local inhibitory neighbours (no zero-inhibition
            # gaps), unlike fully-random selection. Density ≈ frac_inhibitory.
            m = max(2, int(round(1.0 / max(cfg.frac_inhibitory, 1e-3))))
            ipos = neuron_positions(cfg.grid_size).astype(np.int64)
            key = 7 * ipos[:, 0] + 13 * ipos[:, 1] + 23 * ipos[:, 2]
            inh_mask = (key % m == 0)
        edge_weight, is_inh = init_edge_weights(
            edge_index,
            edge_dist,
            num_neurons=N,
            decay_sigma=cfg.decay_sigma,
            frac_inhibitory=cfg.frac_inhibitory,
            g_max=cfg.g_max,
            inh_scale=cfg.inh_scale,
            seed=cfg.seed,
            is_inhibitory=inh_mask,
        )

        # Ablation: destroy distance-biased geometry by reshuffling the edge list
        # into a random graph with the SAME neuron count, edge count and (below)
        # Dale signs. Only the spatial structure is removed.
        if not cfg.spatial_wiring:
            E = edge_index.shape[1]
            rng_g = np.random.default_rng(cfg.seed + 7)
            rsrc = rng_g.integers(0, N, size=E)
            rdst = rng_g.integers(0, N, size=E)
            same = rsrc == rdst
            rdst[same] = (rdst[same] + 1) % N
            edge_index = np.stack([rsrc, rdst]).astype(np.int64)
            edge_dist = np.linalg.norm(pos[rsrc] - pos[rdst], axis=1).astype(np.float32)
            edge_weight, is_inh = init_edge_weights(
                edge_index, edge_dist, num_neurons=N, decay_sigma=cfg.decay_sigma,
                frac_inhibitory=cfg.frac_inhibitory, g_max=cfg.g_max,
                inh_scale=cfg.inh_scale, seed=cfg.seed,
            )

        # Sign per source neuron (Dale's law), applied at runtime to keep the
        # learnable magnitude unsigned and stable.
        src = edge_index[0]
        edge_sign = np.where(is_inh[src], -1.0, 1.0).astype(np.float32)

        self.register_buffer("edge_index", torch.as_tensor(edge_index, dtype=torch.long))
        self.register_buffer("edge_sign", torch.as_tensor(edge_sign, dtype=torch.float32))
        self.register_buffer("is_inhibitory", torch.as_tensor(is_inh, dtype=torch.bool))
        self.register_buffer("positions", torch.as_tensor(pos, dtype=torch.float32))
        if cfg.use_laminar:
            self.register_buffer(
                "laminar_band",
                torch.as_tensor(laminar_bands(cfg.grid_size, cfg.laminar_bands),
                                dtype=torch.long))

        # Zone assignment (Voronoi) for inputs and visualisation.
        zones = assign_zones(cfg.grid_size, zspecs)
        self.register_buffer("zones", torch.as_tensor(zones, dtype=torch.long))

        # --- Learnable parameters ---
        # Store unsigned magnitudes; the Dale sign is applied via edge_sign.
        self.edge_weight = nn.Parameter(torch.as_tensor(np.abs(edge_weight), dtype=torch.float32))
        self.neuron_bias = nn.Parameter(torch.zeros(N))
        self.zone_gain = nn.Parameter(torch.ones(Z))

        self.readout = nn.Sequential(
            nn.Linear(N, cfg.readout_hidden),
            nn.Tanh(),
            nn.Linear(cfg.readout_hidden, 1),
        )

        # --- Sensory pathway: real multimodal embeddings -> per-neuron drive ---
        # For each modality we learn (a) an encode projection from the encoder's
        # embedding into the neurons of its target zone, and (b) a decode head
        # that reconstructs that embedding from the final firing rates. The
        # decode head is what makes self-supervised, label-free online learning
        # (and cross-modal recall) possible.
        self._modalities: List[str] = list(cfg.sensory_embedding_dims.keys())
        self.sensory_encode = nn.ModuleDict()
        self.sensory_decode = nn.ModuleDict()
        for m in self._modalities:
            emb_dim = int(cfg.sensory_embedding_dims[m])
            zone_name = cfg.modality_to_zone.get(m, cfg.zone_names[0])
            zone_id = cfg.zone_names.index(zone_name) if zone_name in cfg.zone_names else 0
            zidx = torch.as_tensor(np.where(zones == zone_id)[0], dtype=torch.long)
            self.register_buffer(f"_zoneidx_{m}", zidx)
            self.sensory_encode[m] = nn.Linear(emb_dim, int(zidx.numel()))
            self.sensory_decode[m] = nn.Linear(N, emb_dim)

        # Cache integration constant.
        self.alpha = cfg.dt / cfg.tau_m

        # Transient short-term-plasticity state (set by stp_begin during a forward).
        self._stp_u = None
        self._stp_x = None
        self._osc_t = 0          # oscillatory phase counter (reset each forward)

        # Dendritic branch assignment: deterministically bucket each neuron's
        # incoming edges into `dend_branches` branches. nb maps an edge to its
        # (dst-neuron, branch) slot in a flattened (N * branches) layout.
        if cfg.use_dendrites:
            K = max(1, int(cfg.dend_branches))
            rng_d = np.random.default_rng(cfg.seed + 11)
            branch = rng_d.integers(0, K, size=edge_index.shape[1])
            nb = edge_index[1] * K + branch
            self.register_buffer("_dend_nb", torch.as_tensor(nb, dtype=torch.long))
            self._dend_K = K

        # Homeostatic intrinsic gain (registered only when enabled, so default and
        # legacy checkpoints keep an unchanged state_dict). homeo_gain multiplies
        # the firing rate; rate_ema tracks each neuron's recent mean activity.
        if cfg.use_homeostasis:
            self.register_buffer("homeo_gain", torch.ones(N))
            self.register_buffer("rate_ema", torch.full((N,), cfg.homeo_target_rate))

        self.to(self._device)

    # ------------------------------------------------------------------ utils
    @property
    def num_neurons(self) -> int:
        return self.config.num_neurons

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def device(self) -> torch.device:
        return self.positions.device

    @property
    def modalities(self) -> List[str]:
        """Names of configured sensory modalities (stable order)."""
        return list(self._modalities)

    def firing_rate(self, V: torch.Tensor) -> torch.Tensor:
        """Logistic firing-rate transfer r = sigmoid(gamma (V - v_thr)).

        With homeostasis enabled, the rate is scaled by a slow per-neuron
        intrinsic gain that drifts to hold each neuron near its target activity.
        """
        cfg = self.config
        r = torch.sigmoid(cfg.gamma * (V - cfg.v_thr))
        if cfg.use_homeostasis:
            r = r * self.homeo_gain
        return r

    @torch.no_grad()
    def _homeo_update(self, rates: torch.Tensor) -> None:
        """Slowly nudge each neuron's intrinsic gain toward its target rate.

        A multiplicative set-point controller (Turrigiano synaptic scaling in
        spirit): neurons firing below target have their gain raised, those above
        lowered, on a slow EMA so it does not fight the fast dynamics. Runs only in
        training mode; the gain is a non-learnable buffer.
        """
        cfg = self.config
        mean_r = rates.detach().mean(0)                    # (N,)
        a = 1.0 / max(cfg.homeo_tau, 1.0)
        # Out-of-place updates (reassign, do NOT mutate in place): homeo_gain is
        # multiplied into the firing rate in the forward graph, so an in-place
        # edit would corrupt the tensor saved for backward. Reassigning leaves the
        # old tensor (held by the graph) untouched.
        self.rate_ema = self.rate_ema * (1.0 - a) + a * mean_r
        err = (cfg.homeo_target_rate - self.rate_ema) / max(cfg.homeo_target_rate, 1e-6)
        self.homeo_gain = (self.homeo_gain * (1.0 + cfg.homeo_lr * err)).clamp(0.2, 5.0)

    def signed_weights(self) -> torch.Tensor:
        """Return synaptic weights.

        With Dale's law (default) the magnitude is constrained non-negative and
        the sign is fixed per presynaptic neuron. In the ``use_dale=False``
        ablation the parameter is used directly, so synapses are freely signed
        (no E/I constraint) — this is the unstable regime that diverges to NaN in
        the matched experiment, which is precisely the point of the ablation.
        """
        if self.config.use_dale:
            # Use abs() rather than clamp(min=0): clamp zeroes both the synapse
            # AND its gradient below 0, a one-way "dying-ReLU" trap that lets the
            # already-weak recurrent core slowly hollow out. abs() keeps the
            # non-negative magnitude Dale's law requires while leaving a non-zero
            # gradient everywhere, so an over-shrunk synapse can recover. It is
            # also checkpoint-safe: edge_weight is initialised to |w| (positive),
            # for which abs() is the identity, so loaded weights are unchanged.
            return self.edge_sign * self.edge_weight.abs()
        return self.edge_weight

    # ------------------------------------------------------------------ core
    def _scatter_inputs(self, external: torch.Tensor) -> torch.Tensor:
        """
        Map a (B, Z) zone-input tensor to a (B, N) per-neuron drive using the
        Voronoi zone assignment and learnable per-zone gains.
        """
        gained = external * self.zone_gain.unsqueeze(0)  # (B, Z)
        # Gather each neuron's zone column -> (B, N)
        return gained[:, self.zones] * self.config.input_gain

    def _scatter_sensory(
        self, sensory: Dict[str, torch.Tensor], batch: int
    ) -> torch.Tensor:
        """
        Project per-modality embeddings into per-neuron current and scatter them
        onto the neurons of each modality's target zone.

        Args:
            sensory: modality name -> (B, emb_dim) or (emb_dim,) tensor.
            batch:   batch size B.

        Returns:
            (B, N) additive external drive from the sensory pathway.
        """
        cfg = self.config
        gain = cfg.sensory_gain if cfg.sensory_gain is not None else cfg.input_gain
        acc = torch.zeros((batch, self.num_neurons), device=self.device, dtype=torch.float32)
        for m, emb in sensory.items():
            if m not in self.sensory_encode:
                continue
            e = torch.as_tensor(emb, device=self.device, dtype=torch.float32)
            if e.dim() == 1:
                e = e.unsqueeze(0)
            proj = self.sensory_encode[m](e) * gain          # (B, n_zone)
            idx = getattr(self, f"_zoneidx_{m}")             # (n_zone,)
            acc = acc.index_add(1, idx, proj)
        return acc

    def reconstruct(self, rates: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode each modality's embedding from firing rates (cross-modal recall)."""
        return {m: self.sensory_decode[m](rates) for m in self._modalities}

    # ----------------------------------------------------- short-term plasticity
    # STP state (release utilisation u, available resources x) is held as transient
    # per-(batch, edge) tensors on the module, begun at the start of a forward pass
    # and threaded automatically through every step() within it. Kept off the
    # public step()/integrate() signatures so existing callers and ablations are
    # untouched; it only activates when ``use_stp`` is set.
    def stp_begin(self, batch: int) -> None:
        """Begin a forward pass: reset the oscillation phase and (if enabled) STP.

        Called at the start of every forward/generate so transient per-sequence
        state (oscillator phase, release variables) starts fresh.
        """
        self._osc_t = 0
        if not self.config.use_stp:
            self._stp_u = self._stp_x = None
            return
        E = self.num_edges
        self._stp_u = torch.full((batch, E), self.config.stp_U, device=self.device)
        self._stp_x = torch.ones((batch, E), device=self.device)

    def stp_detach(self) -> None:
        """Detach STP state from the graph (for carrying it across windows)."""
        if getattr(self, "_stp_u", None) is not None:
            self._stp_u = self._stp_u.detach()
            self._stp_x = self._stp_x.detach()

    def stp_end(self) -> None:
        """Clear STP state after a forward pass."""
        self._stp_u = self._stp_x = None

    def _zone_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Per-neuron zone-mean of ``x`` (B, N) -> (B, N).

        Each neuron receives the mean of ``x`` over all neurons sharing its zone —
        the spatial pool used by divisive normalization. Cheap: one ``index_add``
        per call, O(N).
        """
        Z = self.config.num_zones
        pooled = torch.zeros((x.shape[0], Z), device=x.device, dtype=x.dtype)
        pooled = pooled.index_add(1, self.zones, x)
        counts = torch.bincount(self.zones, minlength=Z).clamp(min=1).to(x.dtype)
        pooled = pooled / counts.unsqueeze(0)
        return pooled[:, self.zones]

    def step(self, V: torch.Tensor, I_ext: torch.Tensor) -> torch.Tensor:
        """
        Advance the membrane potential one integration step (vectorised, batched).

        Args:
            V:     (B, N) membrane potentials.
            I_ext: (B, N) external drive.

        Returns:
            (B, N) updated membrane potentials.
        """
        cfg = self.config
        src, dst = self.edge_index[0], self.edge_index[1]
        N = self.num_neurons

        r = self.firing_rate(V)                      # (B, N) presyn firing rates
        if cfg.use_homeostasis and self.training:
            self._homeo_update(r)
        w = self.signed_weights()                    # (E,)
        r_pre = r[:, src]                            # (B, E) presyn rate per edge

        if cfg.use_stp and self._stp_u is not None:
            # Short-term plasticity (Tsodyks-Markram, rate form). The effective
            # synaptic efficacy is modulated by u*x (utilisation × resources),
            # normalised so it is 1 at rest (u=U, x=1) — facilitation transiently
            # boosts recently-active synapses while depression depletes heavily-used
            # ones, giving a second memory timescale of ~tau_facil steps.
            u, x = self._stp_u, self._stp_x
            factor = (u * x) / cfg.stp_U             # (B, E), == 1 at rest
            contrib = w.unsqueeze(0) * factor * r_pre
            u_next = u + (-(u - cfg.stp_U) / cfg.stp_tau_facil
                          + cfg.stp_U * (1.0 - u) * r_pre)
            x_next = x + ((1.0 - x) / cfg.stp_tau_rec - u * x * r_pre)
            self._stp_u = u_next.clamp(0.0, 1.0)
            self._stp_x = x_next.clamp(0.0, 1.0)
        else:
            contrib = w.unsqueeze(0) * r_pre         # (B, E) per-edge current

        exc = contrib.clamp(min=0.0)
        gI = torch.zeros_like(V).index_add_(1, dst, (-contrib).clamp(min=0.0))
        if cfg.use_dendrites:
            # Per-branch NMDA-like nonlinearity: sum excitation within each
            # dendritic branch, apply a thresholded supralinear gate, then sum
            # branches. Inhibition stays linear (somatic), as in biology.
            K = self._dend_K
            B = V.shape[0]
            branch = torch.zeros((B, self.num_neurons * K), device=V.device, dtype=V.dtype)
            branch = branch.index_add(1, self._dend_nb, exc)
            branch = branch * torch.sigmoid(cfg.dend_gain * (branch - cfg.dend_thr))
            gE = branch.view(B, self.num_neurons, K).sum(dim=2)
        else:
            gE = torch.zeros_like(V).index_add_(1, dst, exc)

        if cfg.use_conductance:
            # Conductance-based: current depends on postsynaptic voltage through
            # reversal potentials (driving force).
            I_syn = gE * (cfg.E_E - V) + gI * (cfg.E_I - V)
        else:
            # Ablation: plain current-based synapses, no reversal driving force.
            I_syn = gE - gI
        dV = -(V - cfg.E_L) + I_syn + I_ext + self.neuron_bias.unsqueeze(0)

        if cfg.use_divnorm:
            # Divisive normalization as a SHUNTING conductance (the conductance-
            # model form of Carandini & Heeger). A zone-pooled, activity-dependent
            # conductance g_norm is added to the effective membrane leak, so the
            # steady-state V is divided by (1 + g_norm + gE + gI) — genuinely
            # divisive gain control that bounds activity (unlike dividing gE, which
            # can weaken the gE·(E_E−V) restoring force and *raise* activity once V
            # overshoots E_E). Identity-like at low activity, suppressive when the
            # zone pool is active.
            g_norm = self._zone_pool(r * r) / (cfg.dn_sigma ** 2)
            dV = dV - g_norm * (V - cfg.E_L)

        if cfg.use_oscillation:
            # Rhythmic pacemaker drive on the inhibitory population: a global theta-
            # like clock that periodically raises inhibitory excitability, gating
            # the network's activity in time (a substrate for phase coding).
            phase = 2.0 * np.pi * (self._osc_t / cfg.osc_period)
            osc = cfg.osc_amp * float(np.sin(phase))
            dV = dV + osc * self.is_inhibitory.to(V.dtype).unsqueeze(0)
            self._osc_t += 1

        return V + self.alpha * dV

    def integrate(self, V, I_ext, steps, collect_trace: bool = False):
        """Run ``steps`` recurrent integration steps from state ``V``.

        This is the single shared reverberation loop used by BOTH the standalone
        :meth:`forward` (which uses ``config.recurrent_steps``) and the language
        model's per-character reverberation (which uses ``LMConfig.inner_steps``).
        Keeping one implementation means the integration depth is the only thing
        that differs between the two call sites, instead of two divergent loops.

        Returns the final ``V`` (and, if ``collect_trace``, the list of per-step
        detached firing-rate snapshots).
        """
        trace: List[torch.Tensor] = []
        for _ in range(int(steps)):
            V = self.step(V, I_ext)
            if collect_trace:
                trace.append(self.firing_rate(V).detach())
        return (V, trace) if collect_trace else V

    def forward(
        self,
        external: Optional[torch.Tensor] = None,
        sensory: Optional[Dict[str, torch.Tensor]] = None,
        return_trace: bool = False,
        return_recon: bool = False,
    ):
        """
        Run the recurrent dynamics and produce a scalar readout.

        Args:
            external: (B, Z) external zone inputs (or (Z,) for a single sample).
                      May be ``None`` when driving the brain purely by sensory data.
            sensory:  optional modality -> (B, emb_dim) embeddings from the
                      multimodal encoders. Projected into their target zones.
            return_trace: if True, also return the (steps+1, B, N) activity trace.
            return_recon: if True, also return decoded per-modality embeddings.

        Returns:
            out: (B, 1) readout probability. Extra tensors appended in the order
            (trace, final_rates, recon) depending on the flags requested.
        """
        cfg = self.config
        if external is None and sensory is None:
            raise ValueError("forward() requires `external` and/or `sensory` input.")

        if external is not None:
            if external.dim() == 1:
                external = external.unsqueeze(0)
            external = external.to(self.device)
            B = external.shape[0]
        else:
            sample = next(iter(sensory.values()))
            B = sample.shape[0] if torch.as_tensor(sample).dim() == 2 else 1

        N = self.num_neurons

        V = torch.full((B, N), cfg.E_L, device=self.device, dtype=torch.float32)
        I_ext = torch.zeros((B, N), device=self.device, dtype=torch.float32)
        if external is not None:
            I_ext = I_ext + self._scatter_inputs(external)
        if sensory:
            I_ext = I_ext + self._scatter_sensory(sensory, B)

        self.stp_begin(B)
        trace: List[torch.Tensor] = []
        if return_trace:
            trace.append(self.firing_rate(V).detach())
            V, steps_trace = self.integrate(V, I_ext, cfg.recurrent_steps,
                                            collect_trace=True)
            trace.extend(steps_trace)
        else:
            V = self.integrate(V, I_ext, cfg.recurrent_steps)
        self.stp_end()

        rates = self.firing_rate(V)                  # (B, N)
        out = torch.sigmoid(self.readout(rates))     # (B, 1)

        extras: List = []
        if return_trace:
            extras.append(torch.stack(trace, dim=0))  # (steps+1, B, N)
            extras.append(rates)
        if return_recon:
            extras.append(self.reconstruct(rates))
        if extras:
            return (out, *extras)
        return out

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def run_with_inputs(self, external: Union[np.ndarray, Sequence[float]]) -> Dict[str, np.ndarray]:
        """
        Convenience inference helper returning numpy arrays for visualisation.

        Args:
            external: (Z,) or (B, Z) external zone inputs.

        Returns:
            dict with keys:
                'output'      (B,) readout probabilities,
                'rates'       (B, N) final firing rates,
                'trace'       (steps+1, B, N) firing-rate trace,
                'positions'   (N, 3) neuron coordinates,
                'zones'       (N,) zone ids.
        """
        x = torch.as_tensor(np.asarray(external, dtype=np.float32))
        out, tr, rates = self.forward(x, return_trace=True)
        return {
            "output": out.squeeze(-1).cpu().numpy(),
            "rates": rates.cpu().numpy(),
            "trace": tr.cpu().numpy(),
            "positions": self.positions.cpu().numpy(),
            "zones": self.zones.cpu().numpy(),
        }

    @torch.no_grad()
    def run_multimodal(
        self,
        sensory: Dict[str, np.ndarray],
        external: Optional[Union[np.ndarray, Sequence[float]]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Drive the brain with real multimodal embeddings and return its state plus
        the decoded (reconstructed / cross-modally recalled) embeddings.

        Args:
            sensory:  modality -> (emb_dim,) or (B, emb_dim) embedding array.
            external: optional (Z,) / (B, Z) zone-scalar drive added on top.

        Returns:
            dict with 'output', 'rates', 'trace', 'positions', 'zones', and
            'recon' (modality -> decoded embedding).
        """
        sens = {
            m: torch.as_tensor(np.asarray(v, dtype=np.float32))
            for m, v in sensory.items()
        }
        ext = None
        if external is not None:
            ext = torch.as_tensor(np.asarray(external, dtype=np.float32))
        out, tr, rates, recon = self.forward(
            ext, sensory=sens, return_trace=True, return_recon=True
        )
        return {
            "output": out.squeeze(-1).cpu().numpy(),
            "rates": rates.cpu().numpy(),
            "trace": tr.cpu().numpy(),
            "positions": self.positions.cpu().numpy(),
            "zones": self.zones.cpu().numpy(),
            "recon": {m: v.cpu().numpy() for m, v in recon.items()},
        }

    def get_neuron_positions(self) -> np.ndarray:
        """Return (N, 3) neuron coordinates as numpy."""
        return self.positions.cpu().numpy()

    # ------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        """Save config + weights to a checkpoint file."""
        torch.save(
            {"config": self.config.to_dict(), "state_dict": self.state_dict()},
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        device: Union[str, torch.device] = "auto",
        strict: bool = True,
    ) -> "PositronicBrain":
        """
        Load a brain from a checkpoint.

        If the checkpoint is incompatible with the current architecture (e.g. an
        older v2 dense model), a fresh brain with the saved config is returned
        and a warning is printed instead of raising, unless ``strict`` is set.
        """
        # weights_only=True forbids arbitrary pickle execution; our checkpoints are
        # only a plain config dict + a tensor state_dict, so this loads safely and
        # closes the CodeQL "unsafe deserialization" path for untrusted files.
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        cfg = BrainConfig.from_dict(ckpt.get("config", {}))
        brain = cls(cfg, device=device)
        try:
            brain.load_state_dict(ckpt["state_dict"], strict=strict)
        except (RuntimeError, KeyError) as exc:
            if strict:
                raise
            print(
                f"[PositronicBrain.load] checkpoint incompatible with current "
                f"architecture ({exc}); returning freshly initialised brain."
            )
        return brain
