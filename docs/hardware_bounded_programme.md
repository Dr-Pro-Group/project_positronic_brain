# Hardware-bounded programme

**Status:** operational plan for results you can finish on Apple Silicon
(this MacBook Pro M1 Pro 16 GB, optionally a Mac Mini M4), without waiting for
a datacenter GPU. Larger GPUs appear only as a **scale opportunity**, not a
dependency for the paper.

**Aesthetic note (once):** the 3D lattice and zone labels are a convenient
inductive bias and a nice visual. They are **not** a scientific claim that the
model “is a brain.” The paper’s claims are about **which constraints and
scaling axes improve a trainable generator under matched budgets**. Do not
spend prose defending biomimicry as identity.

---

## 1. What we already know on this machine

Measured on **M1 Pro 16 GB / MPS**, char-level SODA, committed under `runs/`:

| Lever | Effect (order of magnitude) | Hardware cost |
|---|---|---|
| Training budget 100× | **−0.885 bpc** | wall-clock (cheap conceptually, long runs) |
| Neurons 27×, fixed readout | **−0.294 bpc** | memory + time (G=24 ~hours) |
| Init scale `g_max` 0.4→0.691 | **−0.113 bpc** | free |
| Best mechanism (`--stp`) | **−0.085 bpc** | +time (STP ~1.5×) |
| Density vs volume (matched edges) | **−0.074 bpc** | density arm slower (more steps on smaller N, more walls) |
| Dale’s law | stability + accuracy | free when left on |
| Other bio flags | ~0 or worse | often slower |

**Bottleneck is not graph build** (O(N·k), seconds). **Bottleneck is BPTT:**
each character does `inner_steps` sparse scatters (`index_add` over E edges)
and stores activations for backprop. Caps practical training near **~32k
neurons** on 16 GB at usable batch sizes.

---

## 2. Hardware map: stay local

| Machine | Role | Expected gain vs M1 Pro 16 GB |
|---|---|---|
| **This MBP M1 Pro 16 GB** | Source of truth for published numbers; do not invalidate series without re-running baselines | baseline |
| **Mac Mini M4 base (16 GB)** | Optional second always-on worker later; **same 16 GB wall** as M1 Pro; bandwidth/GPU often *not* better than M1 Pro | Use for overnight parallelism, not larger models. **Currently deferred — all runs on M1 Pro.** |
| Cloud / workstation GPU | Opportunity section only | 5–20× wall-clock depending on model; needed for multi-seed long-matched at G≥16 with many ablations |

### Should you move the next heavy job to the Mini M4?

**Yes, if** the Mini has **≥24 GB unified memory** (or at least the same 16 GB but you want the laptop free).  
**No strong reason if** it is 16 GB and already busy — same memory wall, only faster epochs.

**Protocol if you use both machines:**

1. Pin **software stack**: same torch major, same commit SHA in `*.meta.json` / JSON `args`.
2. One **cross-check**: re-run a single known config (e.g. G=12, 400 steps, seed 42) on M4; accept if |Δ bpc| < ~0.01 vs M1 (seed noise floor is ~0.02).
3. Publish tables with `device` + `chip` fields (already partially in meta sidecars).
4. Do **not** mix half a multi-seed table across machines without saying so.

### GPU opportunity (for discussion / outlook, not blockers)

| Claim you can make after laptop science | What a larger GPU buys |
|---|---|
| Density > volume at matched synapses (3 seeds) | Same contrast at G=32/48, multi-seed |
| Training budget dominates | 40k→400k steps, word-level corpus |
| 3000-step matched ablations (today 1 seed) | 3–5 seeds, full factorial |
| Mechanism table (1 seed) | 3 seeds × helpers/hurters |
| Modular multi-area (not built yet) | Real multi-area training |

Frame GPUs as **replication and scale**, not as a requirement for the existence of the result.

---

## 3. C / C++ / “go further on same silicon” — honest ROI

Hot path today (`PositronicBrain.step`):

```
r = σ(γ(V−θ))
r_pre = r[src]          # gather
contrib = w * r_pre
gE, gI = index_add(dst, …)   # scatter  ← most of the time
I_syn = conductance(V)
V ← V + α * dV
```

Repeated `inner_steps × seq_len × batch` forward and again in backward.

| Approach | Effort | Likely speedup on Apple Silicon | Autograd / paper value |
|---|---|---|---|
| **PyTorch hygiene** (batch tune, `torch.compile`, fuse flags off, AMP where stable) | days | 1.2–2× | High — ship numbers sooner |
| **MLX** port of the sparse step | 1–3 weeks | Often better than MPS PyTorch on M-series | Medium — second backend; scientific parity must be proven |
| **Custom Metal kernel** for gather–scatter | weeks–months | 1.5–3× on the scatter alone | Low for paper; high for product |
| **C++/CPU (Accelerate, Eigen, raw loops)** | weeks | Can beat **Python CPU**; usually **loses to MPS GPU** for this workload | Low for training; OK for graph build / offline probes |
| **libtorch C++ training loop** | weeks | Small (removes Python overhead only) | Low — bottleneck is kernels, not Python |
| **CUDA C++** | — | Only if/when you have NVIDIA | Opportunity section |

**Recommendation:** do **not** start a full C++ rewrite for the paper. Order of attack:

1. **Profile one step** on MPS (`torch.profiler`) — confirm `index_add` dominance.  
2. **Software knobs that preserve science:** larger batch until OOM, `tbptt_chunk`, grad checkpoint (already in tree), fewer `inner_steps` only if ablated and reported.  
3. **`torch.compile`** on `step` / LM forward — try; if MPS backend flaky, document and skip.  
4. **Optional MLX prototype** of *inference-only* or *forward step* as a tech-report appendix — only if Mini M4 is the long-term home.  
5. **C++ only for non-training utilities** if needed: graph builder, n-gram baseline, topology metrics (already Python-fast enough).

If you still want a systems claim: *“A fused Metal sparse conductance kernel is future work; current results use stock PyTorch MPS.”* That is honest and sufficient.

---

## 4. Science to finish **on this hardware** (priority order)

These are the experiments that still move the paper and **fit** laptop/Mini scale. Aesthetics and “more brain flags” are out of scope.

### Tier A — finish what is in flight (no new design)

| ID | Work | Machine | Est. wall |
|---|---|---|---|
| A1 | Let `replicate.py` complete; freeze density 3-seed + ladder endpoints | M1 (current) | remaining ~2–4 h |
| A2 | Integrate numbers into RESULTS + paper density table | CPU | 1 h |
| A3 | Commit `replication.json` once complete | — | 5 min |

### Tier B — highest scientific leverage per GPU-hour

| ID | Work | Why | Est. on M1 / M4 |
|---|---|---|---|
| B1 | **3-seed matched_long** (3000 steps, 6 systems) | Retires single-seed ablation reversals | ~3–5 h / seed ≈ **9–15 h** total; better on M4 |
| B2 | **3-seed** baseline vs `--stp` vs `g_max=0.691` vs density winner | Mechanism + init + density claims with error bars | ~4–8 h |
| B3 | **Wiring-cost λ** sweep (small G=12, 400–1500 steps) | Tests *organisation as objective*, not aesthetics | overnight |
| B4 | Fix default `g_max` or report both; one seed already enough to show magnitude | Removes confound | <1 h |

### Tier C — still laptop, optional for v1 paper

| ID | Work | Note |
|---|---|---|
| C1 | Word-level tiny LM or longer seq_len=128 on G=12 | Stresses memory beyond 4-gram; may OOM — use TBPTT |
| C2 | Wider delay spectrum (`delay_velocity` down) 3 seeds | Settles “geometry in time” without new hardware |
| C3 | n-gram matched-protocol script committed next to free-running | Closes G1/G2 provenance |

### Explicitly **not** on this hardware for v1

- Full multi-area modular brain at scale  
- 100× corpus + transformer baselines at GPT scale  
- Mechanism factorial 8 × 5 seeds × G=24  
- “C++ rewrite then re-run everything”

Those belong under **§ GPU opportunity**.

---

## 5. Paper framing (hardware-honest)

**Title/abstract tone:** measurement of constraints and scaling axes on a sparse recurrent generator; **not** “we built a digital brain.”

**Results you can defend on M1/M4 alone:**

1. Training budget ≫ architecture ≫ unit biophysics (on this task).  
2. Dale buys stability (multi-seed).  
3. Density beats volume at matched synapses (after A1 multi-seed).  
4. Fixed-readout neuron scaling still helps (75% class claim once ladder replicated).  
5. Zones / delays / decay_sigma: controls that killed naive bio stories.  
6. n-gram and matched RNN/LSTM: honesty baselines.

**Discussion paragraph (template):**

> All quantitative claims were obtained on consumer Apple Silicon (M1 Pro 16 GB MPS; [optional: M4 Mini]). Graph construction is cheap; training is limited by sequential BPTT memory and sparse-scatter throughput. A single workstation GPU would primarily multiply seed count and sequence length rather than change the qualitative ranking of levers we report. We therefore treat larger accelerators as a scale-up path, not a prerequisite for the conclusions.

**Do not claim:** neuromorphic deployment, biological equivalence, SOTA language modelling.

---

## 6. Suggested run queue (after current `replicate` ends)

```text
# 0. Freeze A1 artifacts; update RESULTS density table; git commit results only.

# 1. Cross-check on Mini M4 (optional, 20 min)
python experiments/matched_experiment.py --mode brain --grid-size 12 --steps 400 \
  --seeds 42 --hf-chat soda --json runs/m4_crosscheck_g12.json

# 2. Multi-seed long matched (queue overnight on free machine)
python experiments/matched_experiment.py --mode all --grid-size 12 --steps 3000 \
  --seq-len 48 --seeds 43,44 --hf-chat soda --json runs/matched_long_seeds43_44.json
# (seed 42 already in matched_long.json)

# 3. Init + density + stp error bars (one scripted loop)
#    baseline / g_max=0.691 / density G12 k38 / stp   × seeds 42,43,44 @ 1500–3000 steps

# 4. Wiring cost λ ∈ {0, 1e-5, 1e-4, 1e-3} @ G=12, 800 steps, seed 42 first
```

Do **not** start 2–4 while `replicate.py` still holds MPS.

---

## 7. Systems work that is worth a small PR (still hardware-bounded)

1. **Profiler script** `experiments/profile_step.py` — time `step` and report ms/edge.  
2. **JSON always records** `chip`, `torch`, `device`, `git_sha` (mostly done).  
3. **Batch auto-tune** once per (G, k, seq) and cache — saves OOM thrash.  
4. Optional: `torch.compile` flag behind `--compile` default off.  
5. Defer C++/Metal to a separate `systems/` note after the science freeze.

---

## 8. Success criteria for “laptop paper is enough”

Ship when:

- [ ] Density multi-seed complete and significant in the same direction  
- [ ] matched_long ≥3 seeds or clearly labelled single-seed with caveat  
- [ ] RESULTS.md ↔ LaTeX ↔ CLAIMS_LEDGER aligned  
- [ ] Abstract does not sell biomimicry-as-identity  
- [ ] Outlook names GPU multi-seed + modular multi-area as **opportunities**  
- [ ] Optional M4 cross-check within noise  

That is a complete research article on consumer hardware. Everything else is sequel work.
