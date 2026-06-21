# Scientific Background

Project Positronic Brain v3 is a **biomimetic toy model**, not a detailed biophysical simulation. Its purpose is to let researchers, students, and engineers *feel* how a handful of biologically motivated ingredients — 3D space, sparse distance-dependent wiring, conductance-based membranes, Dale's law, and learnable synapses — already produce rich, interpretable collective dynamics, and how such a network can be *grown* and *trained* like a miniature brain.

## The Guiding Metaphor

The project's central intuition is a deliberate simplification:

- A **neuron acts like a linear integrator** — its membrane potential is a leaky linear sum of the currents flowing across it:
  $$\tau_m \frac{dV_i}{dt} = -(V_i - E_L) + I^{\text{syn}}_i + I^{\text{ext}}_i + b_i.$$
- A **synapse acts like a logistic unit** — the presynaptic firing rate it transmits is a saturating sigmoid of the source neuron's membrane potential:
  $$r_j = \sigma\big(\gamma (V_j - V_{\text{thr}})\big).$$

This is the "neuron = linear regression, synapse = logistic regression" idea made concrete. Stacking many such units on a 3D lattice and letting them interact recurrently is what produces the emergent behavior.

## Core Inspirations

### 1. Cortical Microcircuit Geometry
Real neocortex is a sheet with a pronounced depth axis. Connection probability between pyramidal cells drops roughly as a Gaussian with lateral distance (Hellwig 2000; Perin et al. 2011). v3 abstracts this to a 3D cubic lattice and samples synapses with a **Gaussian distance bias** and a **capped fan-in**, producing a sparse graph that is locally dense with rarer long-range fibers.

### 2. Conductance-Based Membranes
Biological synaptic input is not simple additive current — it opens ion channels with their own reversal potentials. v3 uses a conductance formulation:
$$I^{\text{syn}}_i = g^E_i\,(E_E - V_i) + g^I_i\,(E_I - V_i),$$
where excitatory conductance $g^E$ drives $V$ toward $E_E$ and inhibitory conductance $g^I$ drives it toward $E_I$. This makes inhibition *shunting* and naturally bounds activity, unlike a purely additive weight matrix.

### 3. Dale's Law (E/I Segregation)
A real neuron releases the same neurotransmitters at all its synapses: it is either excitatory or inhibitory, not both. v3 fixes the **sign of every outgoing synapse by the source neuron's type** (~20% of neurons are inhibitory). The learnable parameter only stores synaptic *magnitude*, so training adjusts strengths without ever violating Dale's law.

### 4. Learnable Synapses
The biggest change from v2: the connectome weights are **trained end-to-end**. The fixed priors are the geometry, the sparse topology, and each neuron's E/I identity; the plastic part is the set of synaptic magnitudes (plus biases, zone gains, and a small readout). This mirrors the view that structure constrains computation while synaptic strengths are learned.

### 5. Functional Specialization + Integration
The six zones (Visual, Auditory, Somatosensory, Memory, Emotion, Association) are a caricature of cortical division of labor:
- **Sensory zones** (Visual, Auditory, Somatosensory) receive focal external drive.
- **Memory / Emotion** participate in persistence, valence, and internally generated states.
- **Association** sits centrally and binds across modalities.

Zones are assigned by **Voronoi nearest-seed**, so the scheme generalizes to any number of regions and any grid size.

## Emergent Phenomena You Can Observe

- **Local ignition then spread**: a focal sensory drive first lights its own zone, then leaks into neighbors via short-range fibers.
- **E/I balance**: inhibitory neurons clamp runaway excitation; raising sensory drive does not blow activity up to saturation.
- **Cross-modal boosting**: co-driving two sensory zones produces super-additive activation in the Association region, which receives convergent input.
- **Reverberation**: the membrane time constant lets activity persist for several steps after input offset.

## Relation to Other Models

| Model family             | Spatial structure | Synapses        | E/I        | Typical size | Positronic v3 stance |
|--------------------------|-------------------|-----------------|------------|--------------|----------------------|
| Rate-based RNNs          | None / 1D ring    | Trained, dense  | usually no | 10²–10³      | we add geometry + sparsity + Dale |
| Reservoir computing      | Sometimes         | Fixed random    | rarely     | 10²–10³      | similar spirit, but ours is trainable |
| Neural mass / mean-field | 2D sheet          | Effective       | yes        | 10³–10⁵      | toy version of the same ideas |
| Detailed connectomics    | Real EM data      | Measured        | yes        | 10⁴–10⁶      | we are deliberately abstract |

## Training Philosophy

v3 trains the synaptic magnitudes together with per-neuron biases, per-zone input gains, and a tiny readout head, using Adam + BCE on a synthetic multi-modal saliency task (with an optional L1 penalty encouraging sparse, efficient synapses). The fixed wiring *topology* and the Dale's-law *signs* remain frozen — structure is the prior, strength is learned.

## Further Reading (starting points)

- Hellwig, B. (2000). A quantitative analysis of local connectivity between pyramidal neurons in layers 2/3 of rat visual cortex. *Biol Cybern*.
- Perin, R. et al. (2011). A synaptic organizing principle for cortical neuronal groups. *PNAS*.
- Vogels, T. & Abbott, L.F. (2005). Signal propagation and logic gating in networks of integrate-and-fire neurons. *J Neurosci*.
- Dayan, P. & Abbott, L.F. (2001). *Theoretical Neuroscience* — conductance-based models and rate dynamics.
- Any modern review on cortical motifs, small-world connectomics, or E/I balance.

The Positronic Brain is intentionally *not* trying to be any of the above at scale. It is a **pedagogical and exploratory instrument** that nonetheless takes a few biological constraints seriously.
