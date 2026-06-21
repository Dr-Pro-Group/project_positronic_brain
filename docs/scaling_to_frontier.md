# Scaling Positronic Brain to Frontier Capability (Gemma‑4 / Qwen‑3.5 class)

**Status: technical plan / design document.** Nothing here is a reported result.
Every compute figure is an explicit order‑of‑magnitude estimate, labelled as
such. This document extends [`scaling_plan.md`](scaling_plan.md) (which stops at
~500k neurons / modular areas) up to **frontier dense‑equivalent capability**,
and is grounded line‑by‑line in the current code: `build_sparse_graph`
(`connectivity.py`), `PositronicBrain.step` / `signed_weights` (`model.py`),
`BrainLanguageModel.{loss_on,tbptt_step,step_token}` (`language.py`), the sensory
`encode/decode` pathway, `preference.dpo_finetune` / `BrainPolicy`, and
`repro.run_metadata`.

> **One‑paragraph honest summary.** Matching a frontier model's *parameter count*
> is mechanically easy for this architecture — synapses are `E ≈ N·k`, so capacity
> is a free dial. Matching its *capability* is not. Three things stand between the
> current 1.7k‑neuron char‑model and Gemma‑4‑class quality, in order of
> difficulty: (1) the **sequential BPTT recurrence** under‑utilises GPUs and makes
> long‑context credit assignment expensive — this, not memory, is the true ceiling;
> (2) several design choices that are fine at 1.7k neurons become pathological at
> scale and **must be relaxed or re‑engineered** (notably the `Linear(N, vocab)`
> readout, the Python‑loop graph builder, and pure char‑level tokenisation);
> (3) the data + compute bill is **frontier‑lab‑scale** (~10²³–10²⁴ FLOPs, low MFU,
> ≈$1–10M). The plan below keeps the biology that is cheap and load‑bearing (3D
> sparse wiring, Dale, modularity, persistent‑state context) and is explicit about
> the places where biophysical fidelity (full conductance driving force,
> strict per‑step sequentiality) is traded for trainability.

---

## 1. Honest feasibility assessment

### 1.1 What breaks at scale (bottlenecks, ranked)

| # | Bottleneck | Where in code | Severity at frontier | Why it bites |
|---|---|---|---|---|
| 1 | **Sequential recurrence / BPTT** | `step()` called `inner_steps × T` times in `forward`/`tbptt_step` | **Critical, structural** | Each token needs `inner_steps` sequential `index_add` scatters; backprop is serial in `T`. GPUs idle between steps; MFU is 2–5× worse than a transformer at equal FLOPs. No amount of memory tricks removes the *time* dependency. |
| 2 | **`Linear(N, vocab)` readout head** | `language.py:159` `self.head = nn.Linear(N, vocab)` | **Critical, but fixable** | Reads from *all* N neurons. Already the single largest block at G=12 (≈88k of 170k params). At N=10M, vocab=32k this is **3.2×10¹¹ params** in the head alone — larger than the whole target model. Biologically wrong too (whole cortex → tongue). |
| 3 | **Long‑range credit assignment** | implicit in BPTT depth | **High** | Persistent `V` is O(1)‑memory context, but gradients must still flow back through `inner_steps × T` nonlinear steps. Vanishing/exploding over thousands of tokens; conductance `g·(E−V)` is multiplicative in `V`, worsening conditioning. |
| 4 | **Graph construction** | `build_sparse_graph` Python `for i in range(N)` | **High at >1M neurons** | O(N·k) *work* but single‑threaded Python + per‑neuron `rng.choice`. ~5 s at 110k → minutes–hours and unparallelised at 10⁷. Must be vectorised / moved to GPU. |
| 5 | **Edge memory** | `edge_index` int64 (2,E), `edge_weight` fp32 (E) | **Medium** | int64 indices double the footprint. At E=10⁹: 16 GB indices + 4 GB weights *per area*. int32 indices + bf16 weights + per‑area sharding required. |
| 6 | **Optimisation under strong priors** | distance prior + Dale + conductance | **Medium, partly empirical** | Strong inductive biases can slow convergence and create a worse loss‑landscape than a free dense net (our matched study already shows the *full* brain loses to LSTM/RNN at tiny budget). Whether this inverts with scale is **the** open scaling‑law question (§7). |
| 7 | **Per‑step homeostasis / stability** | `inh_scale`, no explicit normalisation | **Medium** | At depth, E/I balance drift causes runaway or silence. Dale buys *some* stability (seed‑dependent — see multi‑seed run), but explicit homeostatic gain control is needed at scale. |

### 1.2 Which biological constraints to keep, relax, or approximate

| Constraint | Keep at frontier? | Rationale / required modification |
|---|---|---|
| **3D spatial embedding + distance‑biased sparse wiring** (`build_sparse_graph`) | **Keep — it is the cheap, load‑bearing prior.** | Gives O(E) compute, locality for parallelism, and the modular/small‑world structure. Only the *builder* needs re‑engineering (vectorise/GPU). The prior itself costs nothing at inference. |
| **Dale's law** (per‑source `edge_sign`) | **Keep.** | Free to enforce (a sign buffer), aids stability, and is a core scientific claim. Our multi‑seed data shows its removal is *sometimes* fine, so it is a stability *prior*, not a hard requirement — keep it but treat instability as a tunable, not a certainty. |
| **Modularity / zones / pathways** | **Keep & promote to first‑class.** | The only way sequential BPTT stays tractable is to cap each *backward graph* at one area; modularity is what makes per‑area model‑parallelism work. |
| **Persistent membrane `V` as context** | **Keep.** | O(1) state memory vs transformer KV cache is a genuine asset, especially for RL/long‑context. Accept its weakness (exact long‑range copy) and lean into associative/working memory. |
| **Conductance driving force `g·(E−V)`** | **Keep as a switch; relax by default for training‑efficiency runs.** | It is multiplicative in `V` (hurts conditioning) and our ablation shows it *costs* perplexity at small scale. Keep `use_conductance=True` for the science track; default large LM runs to the current‑based variant or a **mildly conductance‑clipped** form, and measure whether the cost inverts at scale. |
| **Strict per‑step sequential integration** | **Relax — this is the necessary approximation.** | The single biggest lever. Replace exact sequential BPTT with a **parallelisable recurrence** (chunked parallel scan / DEER‑style fixed‑point, or e‑prop forward credit) — §2.4. This is biophysically a *parallel solver for the same ODE*, not a different model. |
| **fp32 everywhere** | **Relax to bf16 compute, fp32 scatter accumulation.** | Standard mixed precision; keep `index_add` accumulation and the `g·(E−V)` term in fp32 for stability. |

### 1.3 Realistic parameter / neuron target

Capacity is synapse‑dominated: `params ≈ E = N·k` (plus head + pathways). There is
a **fundamental N‑vs‑k trade‑off** — biological cortex is N≈10¹⁰, k≈10³–10⁴, which
we cannot match on both axes. We pick moderate k and the largest N that BPTT can
still train *per area*.

| "Class" proxy | Target trainable params | Realisation as a modular brain | Per‑area cap (BPTT‑safe) |
|---|---:|---|---|
| Gemma‑4‑2B‑ish | ~2–3 B | 8–16 areas × ~1.5M neurons, k≈128 → ~1.5–3 B synapses | ≤ ~2M neurons / area |
| **Gemma‑4 / Qwen‑3.5 dense‑class** | **~7–15 B** | **16–48 areas × ~2M neurons, k≈128–256 → ~6–16 B synapses** | **≤ ~2–3M neurons / area** |
| 30B+ / strong‑MoE‑equivalent | ~25–40 B | 64–128 areas, gated routing (compute ∝ engaged areas, §2.5) | ≤ ~3M neurons / area |

So a credible **Gemma‑4‑class Positronic Brain ≈ 30–100M neurons total, k≈128–256,
~7–16B synapse‑parameters, organised as 16–48 BPTT‑trainable areas linked by
learned pathways**, trained on ~1–10T tokens. Note neuron count is ~100× below
biological for this param budget — the honest statement is "frontier‑param,
brain‑*inspired*, not brain‑*scale*."

---

## 2. Architectural scaling strategy

### 2.1 The growth path: 1.7k → 100k → 500k → frontier

Reuse `build_sparse_graph`/`step()` unchanged for a *single* area; scale by
**adding areas**, never by growing one cube past the BPTT ceiling.

```
 G=12  1.7k     single cube           (today, laptop)
 G=32  33k      single cube           (today, MPS ceiling)
 G=48  110k     single cube + AMP     (S0, 1×A100 — graph build already ~5s)
 ~150k 4–6 areas + static pathways    (S2, modular thesis validated)
 ~500k 10 areas + plastic pathways    (S4, scaling_plan.md endpoint)
 30–100M  16–48 areas + routing       (THIS doc: frontier-param regime)
```

The jump from `scaling_plan.md`'s 500k to 30–100M is **not** a new architecture —
it is (a) more areas, (b) higher per‑neuron fan‑in `k`, (c) the three
re‑engineering fixes from §1.1 (parallel recurrence, output head, graph builder),
and (d) cluster‑scale infra (§3, §6).

### 2.2 Modular areas, cortical columns, learned pathways

Adopt the `ModularBrain` design from `scaling_plan.md` §2 verbatim as the base
data structure (`AreaSpec` / `PathwaySpec`, one `PositronicBrain` per area, sparse
inter‑area `edge_index_path`). Frontier additions:

- **Columns as the checkpoint & parallel‑scan unit.** Tile each area cube into
  6×6×6 columns (dense intra‑column, sparse inter‑column — the canonical cortical
  motif). Columns become (i) the gradient‑checkpoint boundary, (ii) the chunk unit
  for the parallel recurrence (§2.4), and (iii) a natural tensor‑parallel shard.
- **Hierarchy of areas.** Group areas into a shallow hierarchy (sensory → 2–3
  association tiers → working‑memory → output/motor), mirroring cortical
  hierarchy. Lower tiers run *more* inner steps (fast local computation); higher
  tiers integrate over longer effective time constants (`tau_m` per area).
- **Output via a dedicated motor zone, not all N.** **Fix bottleneck #2:** replace
  `head = nn.Linear(N, vocab)` with a small **output area** (≤ a few ×10⁴ neurons)
  that the rest of the brain projects into via pathways; the readout is
  `Linear(N_motor, vocab)` with `N_motor ≪ N`. Equivalently, a **low‑rank readout**
  `Linear(N, r) → Linear(r, vocab)` with r≈512. Both keep the head at ≤10⁸ params
  and are more biologically faithful. This is the single most important code
  change for scale and is a drop‑in to `BrainLanguageModel`.

### 2.3 Activity‑dependent pathway formation (Hebbian + structural)

Keep `scaling_plan.md` §3's three‑tier plan (static distance prior → Oja‑normalised
Hebbian weight plasticity → structural grow/prune), implemented as auxiliary,
label‑free updates layered on BPTT. Frontier‑specific guidance:

- Ship **static pathways first** (the trainable control). Treat Hebbian/structural
  plasticity as *flagged research*, ablated against static — a negative result is
  still publishable.
- Run structural rewrites **only between TBPTT windows** (edge‑list rewrite is
  cheap because pathways are sparse) to avoid disturbing the active backward graph.
- Use the existing **online learner** machinery (`online.py` replay + consolidate)
  as the substrate for activity‑dependent updates — it already does periodic
  gradient replay; Hebbian pathway updates slot in beside it.

### 2.4 Efficient recurrent computation — the central unlock

The sequential dependency (bottleneck #1) is attacked on four fronts; the first
two exist in code, the last two are the high‑value research bets.

| Technique | Status | Effect |
|---|---|---|
| **Truncated BPTT** (`tbptt_step`, `--tbptt-chunk`) | ✅ in code | Decouples trainable length from activation memory; carries `V` across the full sequence, cuts the graph every `chunk`. |
| **Gradient checkpointing** (`grad_checkpoint`) | ✅ in code | Recomputes the `inner_steps` reverberation in backward; ~½–¼ activation memory, enables larger `inner_steps`/batch. |
| **Chunked parallel recurrence (DEER‑style)** | research bet ★ | Solve the within‑chunk recurrence `V_{t+1}=f(V_t,·)` as a *parallel fixed‑point* (Newton / quasi‑Newton over the whole chunk at once) instead of a serial loop. The dynamics are an ODE; a parallel solver reproduces the same trajectory while exposing chunk‑level GPU parallelism. Highest expected MFU win. |
| **e‑prop forward credit** (Bellec et al. 2020) | research bet ★ | Eligibility‑trace forward‑mode credit assignment designed for recurrent (spiking) nets: O(1) memory in `T`, biologically plausible, no BPTT unroll. Pairs naturally with the persistent‑state framing; accept some gradient bias for unbounded context. |
| **Mixture‑of‑recurrent‑experts routing** (§2.5) | planned | Compute ∝ engaged areas, not all areas. |

**Recommendation:** treat the recurrence as an ODE and invest in the
**DEER‑style parallel scan over columns/chunks** as the primary throughput lever,
with **e‑prop** as the fallback / long‑context credit mechanism. Both preserve the
conductance‑rate dynamics; neither requires abandoning the substrate.

### 2.5 Mixture‑of‑recurrent‑experts (sparse area routing)

Promote `scaling_plan.md` §5 to first‑class. A small learned router (conditioned on
the working‑memory area's state) emits `sigmoid(gate)` per pathway; ungated areas
do not advance this token. This is a sparse‑MoE economics applied to *physical
areas* — capacity grows with total areas while per‑token compute tracks engaged
areas. It also yields a clean substrate for studying **task‑dependent functional
connectivity** (which tracts light up for which inputs).

### 2.6 Keeping conductance + Dale stable at scale

- **Homeostatic gain control** (synaptic scaling): a per‑neuron multiplicative gain
  with a slow update driving mean firing rate toward a target (~2%, the rate the
  trained brain already self‑selects). Biologically real *and* a stabiliser. Add as
  a buffer updated between steps.
- **Conductance clipping / soft driving force.** Clamp `gE,gI` and optionally use a
  *bounded* driving force (e.g. `g·tanh(E−V)`) to tame the multiplicative term
  without losing the reversal‑potential semantics.
- **Spectral / E‑I balance monitors.** Track the effective Jacobian spectral radius
  per area during training; use `inh_scale` and gain control to hold it ≲1.
- **Keep `use_dale=True` but don't *rely* on it for stability** — the multi‑seed
  matched run shows divergence is seed‑dependent, so pair Dale with explicit
  homeostasis rather than treating Dale as sufficient.

---

## 3. Training infrastructure & optimisation

### 3.1 Large‑scale recipe (concrete starting point)

| Knob | Recommendation | Note |
|---|---|---|
| Data | byte‑level BPE, ~1–3T tokens, text + multimodal mix (§4) | longer sequences hurt sequential recurrence → favour a *modest* vocab to shorten them |
| Tokeniser | BPE **8k–16k** (compromise, §4) | shrinks `Linear(N_motor, vocab)` head and shortens sequences vs pure char |
| Optimiser | AdamW, β=(0.9,0.95), wd=0.1 | matches transformer practice; the brain's params are ordinary `nn.Parameter`s |
| LR schedule | warmup 2k steps → cosine to 10% | **lower peak LR than transformers** (strong priors + recurrence are LR‑sensitive; today's runs use 8e‑4 at tiny scale, expect ~1–3e‑4 at scale) |
| Grad clip | global‑norm 0.5–1.0 (already default 0.5) | essential for recurrent stability; keep tight |
| Precision | bf16 autocast; fp32 for `index_add` + `g·(E−V)` | mixed precision with stable accumulation |
| TBPTT chunk | 256–512 chars‑equiv, carry `V` across full doc | `tbptt_step` already implements this |
| Checkpointing | on, per column/area | `grad_checkpoint` already implements the inner loop |
| Parallelism | **per‑area FSDP/ZeRO‑3 + pipeline across areas** | areas are independent modules → natural shard boundary; pathways are the only cross‑shard traffic (a few thousand floats/step) |

### 3.2 Distributed strategy that fits sparse graphs

- **Per‑area model parallelism is the key unlock** (as in `scaling_plan.md` §4.1):
  put each area (or a few) on its own GPU; exchange only pathway‑endpoint firing
  rates. Tiny communication surface vs full‑state exchange.
- **FSDP/ZeRO‑3 within an area** for `edge_weight`/`neuron_bias` sharding. Sparse
  `index_add` scatters are local to a shard if columns are the shard unit.
- **Replace `index_add` with a fused segment‑reduction kernel** (`torch_scatter`
  segment‑sum or a Triton kernel) — the current `torch.zeros_like(V).index_add_`
  pattern in `step()` is correct but memory‑bound; a sorted‑segment kernel is the
  throughput fix for big `E`.
- **Tensor parallelism** maps onto columns (dense intra‑column blocks are matmul‑
  shaped). Avoid splitting an area's recurrence across pipeline stages — keep the
  recurrent loop on‑device to dodge per‑step comms.

### 3.3 Long credit assignment vs transformers

Transformers get exact gradients to every position in one parallel backward;
this substrate does not. The plan:
1. **TBPTT with state carry** (have) for in‑window exactness + cross‑window context.
2. **e‑prop eligibility traces** (§2.4) for unbounded context with O(1) memory,
   accepting bias. This is the biologically‑honest answer to "how does a brain
   assign credit over minutes without storing every activation."
3. **Auxiliary local objectives** (next‑step membrane prediction, reconstruction
   from the existing `sensory_decode` heads) to inject gradient locally and reduce
   reliance on deep backprop — a self‑supervised shortening of the credit path.

### 3.4 Staged curriculum

1. **Substrate dynamics pre‑training (self‑supervised).** Before any language,
   train the brain to have rich, stable dynamics: next‑step / reconstruction
   objectives using the `sensory_encode/decode` autoencoding path (already
   implemented for multimodal). Goal: a well‑conditioned recurrent substrate (good
   spectral radius, ~2% activity) — a better init than random for the LM phase.
2. **Language modelling (next‑token BPTT/TBPTT).** The current `train_language.py`
   objective, scaled.
3. **Multimodal alignment.** Re‑enable the masked multimodal autoencoding
   (modality dropout → cross‑modal recall) concurrently with LM, using frozen
   CLIP/Wav2Vec2 encoders projecting into sensory zones.
4. **Instruction tuning (SFT)** on `User:`/`Brain:` formatted data (the corpus
   pipeline already produces this format from `soda`/`ultrachat`/`hh-rlhf`).
5. **Preference / RL** (§5): DPO via `preference.dpo_finetune`, then GRPO/PPO via
   `BrainPolicy`.

Each stage writes a `repro.run_metadata` provenance sidecar (seed, git SHA, config)
so every checkpoint in the curriculum is auditable — the infra already does this.

---

## 4. Data & tokenizer strategy

### 4.1 Beyond char‑level

Pure char‑level (`CharTokenizer`) maximises sequence length, which is the worst
axis for a sequential recurrent model. Options, with the trade‑off that matters
here (sequence length ↔ head/vocab size ↔ "minimal prior" purity):

| Choice | Vocab | Seq‑len factor | Head cost `Linear(N_motor,V)` | Verdict |
|---|---:|---|---|---|
| Pure char (today) | ~50–256 | 1.0 (longest) | smallest | keep for science‑track ablations |
| **Byte‑level BPE 8k–16k** | 8–16k | ~0.25–0.3× | moderate | **recommended default** — shortens sequences ~3–4× (directly easing the recurrence bottleneck) while keeping the head ≤10⁸ and staying "trained‑from‑scratch" |
| BPE 32k–64k (Llama‑class) | 32–64k | ~0.2× | large | only with the low‑rank/motor‑zone head fix; fine at frontier |

**Recommendation:** byte‑level BPE at **8k–16k**. It is the sweet spot: meaningfully
shorter sequences (the lever that most helps the recurrent core), a tractable head,
and no pretrained‑tokenizer baggage that would undercut the "from scratch" framing.
A biological gloss: sub‑word units are a mild, defensible prior (cf. syllable/
morpheme chunking) — more honest than importing a 256k vocab.

### 4.2 Datasets / mixtures for strong reasoning + instruction following

- **Text backbone:** a high‑quality web mix (FineWeb‑Edu‑style filtered web),
  plus code (the StarCoder/StackV2 lineage) and math (OpenWebMath / proof‑pile).
  Reasoning capability tracks code+math fraction — budget ≥25%.
- **Long‑form / books** for long‑range dependency pressure (where persistent `V`
  is tested hardest).
- **Instruction / chat:** the existing streamed `soda` / `ultrachat` / `hh-rlhf`
  pipeline scales straight up (already reformatted to `User:`/`Brain:`); add a
  larger SFT mix (e.g. Tülu‑style) at the SFT stage.
- **Multimodal:** COCO/LAION‑style image–text and speech–text via the existing
  `datasets.py` streaming + `seed.py` flow, encoded by frozen CLIP/Wav2Vec2 into
  sensory zones.

### 4.3 Leverage the existing self‑supervised multimodal path during pretraining

The `sensory_encode`/`sensory_decode` heads + modality dropout already implement
**masked multimodal autoencoding** with cross‑modal recall. At scale this is not a
side feature — it is the **substrate pre‑training objective** (§3.4 stage 1) and a
continual source of local gradient (§3.3). Run it *alongside* LM so the brain forms
grounded cross‑modal associations rather than text‑only statistics.

---

## 5. Reinforcement learning / alignment path

### 5.1 The brain *is* a recurrent policy already

`preference.BrainPolicy` already exposes `reset()`/`act()` with the membrane `V` as
the recurrent policy state, and `preference.dpo_finetune` already implements DPO
over `(prompt, chosen, rejected)` triples with a frozen reference (the `V` warmed on
the prompt is the "memory"). The alignment ladder:

1. **DPO** (`dpo_finetune`) — works today; the cheapest alignment, no reward model,
   no rollouts. Validated as a smoke test in this very run.
2. **GRPO** — recommended next: group‑relative advantages from sampled rollouts, **no
   value network**. This suits the brain because (a) rollouts are cheap in *memory*
   (O(1) state, no growing KV cache), so long rollouts are affordable, and (b)
   removing the critic avoids training a second large recurrent net.
3. **Full PPO / actor‑critic** — add a value head reading from the working‑memory
   area (small `Linear(N_wm,1)`); the `BrainPolicy.act` contract already leaves the
   value slot open (`return logits, None`).

### 5.2 Where the brain may have a real RL edge

- **O(1)‑memory state → cheap long rollouts.** No KV cache growth; long‑horizon
  episodes cost constant memory per step. Directly lowers RL compute vs a
  transformer policy.
- **Online / continual learning** is native (`online.py` replay + consolidate) —
  the substrate is built to keep learning during interaction, a natural fit for
  streaming/continual RL and test‑time adaptation.
- **Stability from inductive bias.** Dale + homeostasis bound the dynamics, which
  can reduce policy collapse / reward hacking instabilities.
- **Sample efficiency (hypothesis).** Strong priors *may* pay off more in the
  low‑data RL regime than in the data‑rich pretraining regime where they currently
  cost perplexity — a concrete, falsifiable claim worth testing (§7).

These are **hypotheses to test**, not established advantages — stated as such.

---

## 6. Compute & hardware roadmap

### 6.1 Order‑of‑magnitude compute (clearly labelled estimates)

Using the transformer proxy `C ≈ 6·P·D` (params × tokens) as a floor, then a
recurrent penalty for `inner_steps` and low MFU:

- Target P ≈ 1×10¹⁰ params, D ≈ 2×10¹² tokens → floor `6PD ≈ 1.2×10²³` FLOPs.
- Recurrent overhead (`inner_steps≈3`, serial dependency, MFU ~15–30% vs ~50% for
  transformers) → **effective ~4–10× → ~5×10²³–10²⁴ FLOPs**.
- At H100 (~10¹⁵ bf16 FLOP/s) × 20% realised → ~2×10¹⁴ eff FLOP/s → ~3–5×10⁹
  GPU‑seconds ≈ **~1×10⁶ GPU‑hours**.
- ≈ **1,000 H100s for ~5–6 weeks**, or 2,000 for ~3 weeks. Cloud ≈ **$1–10M**
  depending on spot/committed pricing and how badly the recurrence under‑utilises.

These are ±1 order of magnitude. The recurrence penalty is the dominant uncertainty
and the main thing the §2.4 research bets aim to shrink.

### 6.2 Phased plan (extends `scaling_plan.md` S0–S5)

| Phase | Neurons / params | New capability | Hardware | Token budget | Acceptance |
|---|---|---|---|---|---|
| **F0** | ≤110k (1 cube) | AMP + Triton scatter + parallel‑scan prototype; **vectorised graph builder**; **motor‑zone head** | 1×A100/H100 | ~1–5B | G=48 trains 50k steps in hours; MFU ≥2× today; head/builder fixes landed |
| **F1** | ~1–3M (1–2 areas) | byte‑BPE, e‑prop or DEER scan validated on one area; substrate pretrain stage | 1–4×H100 | ~10–50B | matches a same‑param LSTM perplexity *trend*; recurrence MFU measured |
| **F2** | ~10–30M (8–16 areas) | full modular brain, routing, multimodal alignment | 8–32×H100 | ~200–500B | scaling‑law plot (loss vs params/tokens); ablations run |
| **F3** | ~30–100M / 7–15B params | frontier‑param pretrain + SFT + DPO/GRPO | 256–1024×H100 | ~1–3T | downstream evals vs Gemma‑4/Qwen‑3.5 on a fixed suite; honest gap reported |
| **F4** | research | neuromorphic mapping study (event readout → Loihi/SpiNNaker) | neuromorphic access | — | SynOps on hardware, not proxy |

### 6.3 Hardware features that help most

- **Large HBM** (per‑area edge tables + activations) and **fast scatter/gather**
  (the core `index_add` primitive) — favours H100/H200/MI300 with high memory
  bandwidth; the workload is memory‑bound, not matmul‑bound.
- **Good fused sparse kernels** (Triton/CUTLASS segment‑reduction) — bigger lever
  than raw FLOPs here.
- **Fast intra‑node interconnect** (NVLink) so per‑area pipeline comms (pathway
  messages) are cheap; the design deliberately keeps inter‑GPU traffic tiny.
- **Later: neuromorphic / event‑driven accelerators** (Loihi 2, SpiNNaker 2) for
  inference — the ~2% activity sparsity the brain self‑selects is exactly what
  event‑driven hardware exploits (cf. SpikingBrain's ~70% sparsity → energy wins).

---

## 7. Research & paper opportunities

### 7.1 Questions answerable only at scale

- **Does the conductance "cost" invert?** Our matched study shows conductance
  *costs* perplexity at 1.7k neurons. The headline scaling question: is the
  cost a small‑budget artefact that flips sign at 10⁷ neurons / 10¹² tokens? A
  scaling‑law sweep over `use_conductance` is a clean, high‑impact result either way.
- **Do strong biological priors help more in low‑data / RL regimes than in
  data‑rich pretraining?** (§5.2 hypothesis.) Measurable via matched RL sample‑
  efficiency curves.
- **Emergent topology at scale.** Does distance‑biased wiring + training reproduce
  cortical motifs (modularity, small‑worldness, rich‑club hubs, functional
  connectivity) at 10⁷ neurons? Directly comparable to seRNN (Achterberg et al.
  2023), now at a scale seRNN never reached.
- **Membrane‑field context vs attention.** What is the effective context length of
  a persistent‑`V` model, and where does it beat / lose to attention (associative
  recall vs exact copy)? A new memory‑mechanism scaling law.

### 7.2 Positioning vs frontier & brain‑inspired efforts

- **vs Gemma‑4 / Qwen‑3.5:** do **not** claim SOTA. Position as the first
  *frontier‑param, 3D‑spatial, conductance‑rate, Dale‑constrained generator*, and
  report the honest quality gap plus any efficiency/energy advantage — consistent
  with the field's "efficiency at competitive, not superior, quality" consensus.
- **vs SpikeGPT / SpikingBrain:** those are binary‑spike units inside
  transformer/linear‑attention blocks. Our differentiator stays **smooth
  conductance‑rate dynamics on a 3D distance‑biased graph with no attention** — a
  distinct, unoccupied point (see `literature_gap.md` §7 table). Borrow their
  energy methodology (SynOps, activation sparsity) for the efficiency story.
- **vs seRNN / EI‑RNN / DANN:** scale and the generator framing. We impose
  (not merely regularise) space + Dale, and run it as an LM at a scale those works
  do not target.

### 7.3 Ablations & scaling laws to run (priority order)

1. **`use_conductance` × scale** (the inversion question) — sweep N ∈ {10⁴…10⁷}.
2. **Recurrence method**: exact TBPTT vs DEER‑parallel‑scan vs e‑prop — quality &
   MFU at matched compute.
3. **Tokeniser**: char vs BPE‑8k vs BPE‑32k — quality per token vs per char vs
   per FLOP.
4. **Head design**: full `Linear(N,V)` vs motor‑zone vs low‑rank — quality &
   param efficiency.
5. **Pathways**: static vs Hebbian vs structural‑plastic — perplexity + retrieval@k
   on a long‑range memory task.
6. **`k` (fan‑in) vs N at fixed params** — the capacity allocation law for this
   substrate.
7. **Homeostasis on/off** — stability and final quality at depth.

---

## 8. Concrete next experiments (1–8× H100/A100, before any cluster)

These are runnable on the existing codebase with bounded effort and validate the
riskiest assumptions cheaply:

1. **Land the three scale‑blockers on the current model (1 GPU).**
   - Vectorise `build_sparse_graph` (shared offsets → batched neighbour gather +
     segmented Gumbel/top‑k sampling) and confirm bit‑comparable graphs; benchmark
     to 10⁶ neurons.
   - Add the **motor‑zone / low‑rank readout** option to `BrainLanguageModel`;
     verify param count drops and quality holds at G=12/32.
   - Add **bf16 autocast + Triton segment‑sum** for `step()`; measure MFU uplift.
   *Acceptance:* G=48 (110k) trains end‑to‑end on one A100 with ≥2× today's MFU.

2. **Parallel‑recurrence prototype on one area (1–2 GPUs).** Implement a DEER‑style
   chunked fixed‑point solver for the within‑chunk recurrence and validate it
   reproduces the serial `tbptt_step` trajectory to tolerance, then measure
   wall‑clock at chunk 256–512. *Acceptance:* same loss curve, ≥3× step throughput.

3. **e‑prop credit on the brain LM (1 GPU).** Add eligibility‑trace forward credit
   as an optional path; compare perplexity & memory vs TBPTT at matched steps on
   the built‑in corpus. *Acceptance:* within a small perplexity margin at O(1)
   memory in `T`.

4. **Two‑area `ModularBrain` + static pathway on a copy/recall task (1–2 GPUs).**
   The `scaling_plan.md` S1 experiment — proves long‑range pathways carry gradient
   and beat a single same‑param cube on a task that *requires* communication.
   *Acceptance:* modular > monolithic on held‑out recall@k.

5. **Byte‑BPE‑8k swap + small scaling sweep (1–4 GPUs).** Retrain G=16/24/32 with
   BPE‑8k; fit a mini scaling curve (loss vs params) and extrapolate. *Acceptance:*
   a clean loss‑vs‑N line and a first read on whether the brain's slope is
   competitive with an LSTM's.

6. **GRPO smoke on the trained LM (1 GPU).** Extend `BrainPolicy` with a value‑free
   GRPO loop on a tiny verifiable task (e.g. arithmetic format reward). *Acceptance:*
   reward increases; membrane‑state policy is stable over long rollouts.

Each experiment is independently informative and degrades gracefully — e.g. if the
parallel‑recurrence solver underperforms, e‑prop and per‑area parallelism still
carry the throughput story; if the conductance inversion never materialises, the
`use_conductance=False` default makes the brain *competitive now* and the result is
a clean negative for biophysical fidelity. The plan never has a single point of
failure.

---

## 9. Summary of recommended changes to the codebase

| Change | File | Priority | Type |
|---|---|---|---|
| Motor‑zone / low‑rank readout head | `language.py` | **P0** | re‑engineer (scale blocker) |
| Vectorised / GPU graph builder | `connectivity.py` | **P0** | re‑engineer (scale blocker) |
| bf16 autocast + Triton segment‑sum in `step()` | `model.py` | **P0** | efficiency |
| `ModularBrain` (areas + pathways) | new `modular.py` | **P1** | feature (from `scaling_plan.md`) |
| DEER‑style parallel chunk solver | `language.py`/new | **P1** | research bet (throughput) |
| e‑prop credit path | new `eprop.py` | **P1** | research bet (credit) |
| Homeostatic gain control + spectral monitor | `model.py` | **P1** | stability at scale |
| Byte‑BPE tokenizer option | `language.py`/new | **P1** | data |
| Routing gate for pathways (MoRE) | `modular.py` | **P2** | efficiency |
| GRPO loop on `BrainPolicy` | `preference.py` | **P2** | alignment |

P0 items are the prerequisites for *any* honest scale run and are all bounded,
single‑GPU changes to existing files. Everything biological that is cheap and
load‑bearing — 3D sparse wiring, Dale, modularity, persistent‑state context — is
preserved; the only constraint relaxed by necessity is **strict per‑step
sequentiality**, and even that is replaced by a *parallel solver for the same
dynamics*, not a different model.
