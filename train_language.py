#!/usr/bin/env python
"""
Train the biomimetic generative language model — the 3D brain that writes text.

The :class:`~positronic_brain.language.BrainLanguageModel` learns next-character
prediction by backpropagation-through-time over its recurrent membrane dynamics.
By default it trains fully offline on a built-in conversational corpus so you can
immediately chat with the result.

Examples
--------
    # Quick offline conversational model (no downloads)
    python train_language.py --steps 800 --grid-size 12 --out trained_models/brain_lm.pt

    # Full "grok" run: 32,768 neurons, ~90% of RAM, 3000 steps (MPS)
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python train_language.py \
        --grid-size 32 --steps 3000 --target-mem-frac 0.9 --device mps

    # Train on real public dialogue data (streamed, formatted as User:/Brain:)
    python train_language.py --hf-chat soda --hf-chat-limit 4000 --steps 3000

    # Train on your own text file (still keeps the chat structure)
    python train_language.py --text mydata.txt --steps 2000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from positronic_brain.corpus import load_corpus_splits
from positronic_brain.language import BrainLanguageModel, CharTokenizer, LMConfig
from positronic_brain.streaming import StreamingBatcher
from positronic_brain.utils import get_device
from positronic_brain.repro import seed_everything, run_metadata
from positronic_brain import diagnostics


def make_batch(data: torch.Tensor, seq_len: int, batch_size: int, device) -> torch.Tensor:
    """Sample `batch_size` random contiguous chunks of length `seq_len + 1`."""
    n = data.numel() - seq_len - 1
    ix = torch.randint(0, max(n, 1), (batch_size,))
    chunks = torch.stack([data[i : i + seq_len + 1] for i in ix])
    return chunks.to(device)


@torch.no_grad()
def evaluate(model, data: torch.Tensor, seq_len: int, batch_size: int, device,
             max_windows: int = 256):
    """Held-out evaluation over FIXED, non-overlapping windows.

    Returns a dict with mean per-character cross-entropy (nats), bits-per-char
    (= CE / ln 2, the standard comparable metric), perplexity, and the fraction of
    neurons saturated in the sigmoid tails on the final window. Deterministic — no
    random sampling — so the number is a stable held-out signal, not an in-
    distribution training loss.
    """
    if data.numel() < seq_len + 2:
        return {"ce": float("nan"), "bpc": float("nan"), "ppl": float("nan"),
                "saturation": float("nan"), "windows": 0}
    model.eval()
    step = seq_len  # non-overlapping windows
    starts = list(range(0, data.numel() - seq_len - 1, step))[:max_windows]
    total_ce, n_tok, sat = 0.0, 0, float("nan")
    for i in range(0, len(starts), batch_size):
        idx = starts[i : i + batch_size]
        batch = torch.stack([data[s : s + seq_len + 1] for s in idx]).to(device)
        logits, V = model(batch[:, :-1])
        target = batch[:, 1:]
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, model.vocab_size), target.reshape(-1),
            reduction="sum")
        total_ce += float(ce.item())
        n_tok += target.numel()
        sat = diagnostics.saturation_fraction(model.brain.firing_rate(V))
    mean_ce = total_ce / max(n_tok, 1)
    return {"ce": mean_ce, "bpc": mean_ce / math.log(2),
            "ppl": math.exp(min(mean_ce, 50)), "saturation": sat,
            "windows": len(starts)}


@torch.no_grad()
def evaluate_persistent(model, data: torch.Tensor, seq_len: int, device,
                        max_tokens: int = 16384):
    """Held-out eval that CARRIES the membrane state across the whole stream.

    This is the matched metric for ``--persistent-state`` training: the brain
    walks ``data`` contiguously in ``seq_len`` chunks, carrying ``V`` across them
    (single lane), so the reported bits-per-char reflects the full available
    context rather than a cold-started window. Lower than the cold-window number
    by exactly the amount the persistent state actually helps.
    """
    if data.numel() < seq_len + 2:
        return {"ce": float("nan"), "bpc": float("nan"), "ppl": float("nan"),
                "saturation": float("nan"), "windows": 0}
    model.eval()
    V = None
    total_ce, n_tok, chunks = 0.0, 0, 0
    limit = min(int(data.numel()), max_tokens)
    for start in range(0, limit - 1, seq_len):
        chunk = data[start : start + seq_len + 1]
        if chunk.numel() < 2:
            break
        logits, V = model(chunk[:-1].unsqueeze(0), state=V)
        target = chunk[1:].unsqueeze(0)
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, model.vocab_size), target.reshape(-1), reduction="sum")
        total_ce += float(ce.item())
        n_tok += target.numel()
        chunks += 1
        V = V.detach()
    mean_ce = total_ce / max(n_tok, 1)
    return {"ce": mean_ce, "bpc": mean_ce / math.log(2),
            "ppl": math.exp(min(mean_ce, 50)), "saturation": float("nan"),
            "windows": chunks}


def _total_memory_bytes() -> int:
    """Total physical RAM (unified memory on Apple Silicon), in bytes."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, AttributeError, OSError):
        return 16 * 1024 ** 3


def _allocated_bytes(device) -> int:
    """Best-effort current GPU/accelerator allocation in bytes."""
    if device.type == "mps":
        try:
            return int(torch.mps.driver_allocated_memory())
        except Exception:
            return 0
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated())
    return 0


def auto_tune_batch_size(model, data, seq_len, target_frac, device,
                         min_batch=2, max_batch=256) -> int:
    """
    Pick the largest batch size whose training step is projected to fit within
    ``target_frac`` of total machine memory.

    A single probe step measures the activation+gradient footprint of a small
    batch, then we scale linearly (activation memory grows ~linearly in batch).
    On Apple Silicon's unified memory this lets us deliberately fill ~90% of RAM.
    """
    target = int(_total_memory_bytes() * target_frac)
    if device.type == "mps":
        torch.mps.empty_cache()
    base = _allocated_bytes(device)

    probe = min_batch
    batch = make_batch(data, seq_len, probe, device)
    loss = model.loss_on(batch)          # builds the BPTT graph (peak activations)
    peak = _allocated_bytes(device)
    loss.backward()
    model.zero_grad(set_to_none=True)
    if device.type == "mps":
        torch.mps.empty_cache()

    per_sample = max((peak - base) / probe, 1.0)
    budget = max(target - base, per_sample)
    tuned = int(budget // per_sample)
    tuned = max(min_batch, min(tuned, max_batch))
    print(f"[train] auto batch-size: base={base/1e9:.2f}GB  "
          f"per-sample~{per_sample/1e6:.0f}MB  target={target/1e9:.2f}GB "
          f"({target_frac:.0%} of {_total_memory_bytes()/1e9:.0f}GB) -> batch={tuned}",
          flush=True)
    return tuned


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # corpus
    p.add_argument("--text", default=None, help="path to a UTF-8 text file to train on")
    p.add_argument("--hf", default=None, help="public PLAIN-TEXT dataset to stream (e.g. tinystories, wikitext)")
    p.add_argument("--hf-limit", type=int, default=2000)
    p.add_argument("--hf-chat", default=None,
                   help="public CONVERSATIONAL dataset to stream, reformatted as "
                        "User:/Brain: turns (e.g. soda, hh-rlhf, ultrachat)")
    p.add_argument("--hf-chat-limit", type=int, default=2000,
                   help="max conversations to pull from --hf-chat")
    p.add_argument("--no-builtin", action="store_true", help="do NOT append the built-in conversational corpus")
    p.add_argument("--repeats", type=int, default=60, help="how many times to repeat the conversational seed")
    # model
    p.add_argument("--grid-size", type=int, default=32,
                   help="cube side; neurons = grid_size**3. Default 32 -> 32,768 "
                        "neurons, the practical training ceiling on a 16 GB M1 Pro.")
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--inner-steps", type=int, default=3, help="brain integration steps per character")
    p.add_argument("--input-zone", default="Association")
    # training
    p.add_argument("--steps", type=int, default=800,
                   help="optimizer steps. Use ~3000 to fully 'grok' the corpus.")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16,
                   help="ignored when --target-mem-frac > 0 (auto-tuned instead)")
    p.add_argument("--target-mem-frac", type=float, default=0.0,
                   help="if > 0, auto-pick the batch size to use this fraction of "
                        "total RAM (e.g. 0.9 fills ~90%% of a 16 GB M1 Pro)")
    p.add_argument("--lr", type=float, default=8e-4,
                   help="learning rate. Large grids need <=1e-3 + tight grad-clip.")
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--sample-every", type=int, default=100)
    # ---- held-out evaluation (the honesty contract) -------------------------
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="fraction of the corpus held out for validation "
                        "(content-disjoint: no train material leaks into val)")
    p.add_argument("--test-frac", type=float, default=0.1,
                   help="fraction held out for a final test, scored ONCE at the end")
    p.add_argument("--eval-every", type=int, default=0,
                   help="run held-out val every N steps (0 -> same as --sample-every). "
                        "Reports bits-per-char and gates the best-val checkpoint.")
    p.add_argument("--no-best-checkpoint", action="store_true",
                   help="save the final model instead of the best-val one")
    p.add_argument("--diagnostics", action="store_true",
                   help="log per-group grad norms + saturation during training and "
                        "report settling residual + memory horizon at the end")
    p.add_argument("--frozen-reservoir", action="store_true",
                   help="freeze the recurrent core (edge_weight + neuron_bias) and "
                        "train only the read-in/read-out — isolates how much the "
                        "dynamics actually contribute vs. the Linear head")
    p.add_argument("--persistent-state", action="store_true",
                   help="carry the membrane state V across consecutive (contiguous) "
                        "windows instead of resetting to rest each window — makes the "
                        "'state IS the context/memory' claim actually trained. Uses a "
                        "stateful streaming batcher + truncated BPTT across batches.")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true",
                   help="request deterministic kernels (slower, more reproducible)")
    p.add_argument("--out", default="trained_models/brain_lm.pt")
    # ---- scaling / memory knobs (for grid_size 48-64+) ----------------------
    p.add_argument("--tbptt-chunk", type=int, default=0,
                   help="if > 0, use truncated BPTT with this chunk length so the "
                        "trainable sequence length decouples from activation memory")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="checkpoint the inner reverberation loop (less memory, "
                        "~1 extra forward) — recommended for grid_size >= 48")
    # ---- ablations (config-driven, saved into the checkpoint) ---------------
    p.add_argument("--no-spatial", action="store_true",
                   help="ablation: random wiring instead of distance-biased 3D graph")
    p.add_argument("--no-dale", action="store_true",
                   help="ablation: freely-signed synapses (no Dale's law) — unstable")
    p.add_argument("--additive-current", action="store_true",
                   help="ablation: current-based synapses (no conductance driving force)")
    # ---- biological mechanism flags (off by default, ablatable) --------------
    p.add_argument("--divnorm", action="store_true",
                   help="enable divisive normalization (Carandini & Heeger): divide "
                        "excitatory drive by zone-pooled activity for gain control")
    p.add_argument("--stp", action="store_true",
                   help="enable short-term synaptic plasticity (Tsodyks-Markram): "
                        "facilitation/depression gives a second, slower memory timescale")
    p.add_argument("--homeostasis", action="store_true",
                   help="enable homeostatic intrinsic-gain control toward a target "
                        "firing rate (stability; lets grad-clip loosen at scale)")
    p.add_argument("--oscillation", action="store_true",
                   help="enable a theta-like oscillatory pacemaker on inhibitory cells")
    p.add_argument("--dendrites", action="store_true",
                   help="enable per-branch dendritic NMDA nonlinearity")
    p.add_argument("--laminar", action="store_true",
                   help="enable laminar microcircuit: canonical L4->L2/3->L5/6 "
                        "connectivity bias + spatially-even inhibition")
    p.add_argument("--sparse-weight", type=float, default=0.0,
                   help="weight of a metabolic sparse-coding penalty that pulls the "
                        "mean firing rate toward homeo_target_rate (anti-dead-unit)")
    p.add_argument("--lr-schedule", choices=["none", "warmcos"], default="none",
                   help="warmcos = linear warmup (10%% of steps) then cosine decay")
    p.add_argument("--learning-rule", choices=["bptt", "eprop"], default="bptt",
                   help="bptt (default) or eprop: a forward-only, biologically-local "
                        "eligibility-trace rule (fidelity / online learning, not a "
                        "quality booster)")
    args = p.parse_args()

    seed_everything(args.seed, deterministic=args.deterministic)
    device = get_device(args.device)

    # ---- corpus + tokenizer (content-disjoint train/val/test) ---------------
    train_text, val_text, test_text = load_corpus_splits(
        text_path=args.text,
        hf=args.hf,
        hf_limit=args.hf_limit,
        hf_chat=args.hf_chat,
        hf_chat_limit=args.hf_chat_limit,
        builtin=not args.no_builtin,
        repeats=args.repeats,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )
    # Build the tokenizer from the TRAIN split only: held-out characters that never
    # appear in training are then scored as UNK exactly as a deployed model sees
    # them (no silent vocabulary leakage from val/test into the tokenizer).
    tokenizer = CharTokenizer.from_text(train_text)
    data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_data = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    test_data = torch.tensor(tokenizer.encode(test_text), dtype=torch.long)
    print(f"[train] corpus chars={len(train_text)+len(val_text)+len(test_text)}  "
          f"vocab={tokenizer.vocab_size}")
    print(f"[train] tokens  train={data.numel()}  val={val_data.numel()}  "
          f"test={test_data.numel()}")
    if val_text:
        print(f"[train] held-out UNK rate  val={tokenizer.unk_rate(val_text):.4%}  "
              f"test={tokenizer.unk_rate(test_text):.4%}")

    # ---- model --------------------------------------------------------------
    brain_overrides = {
        "use_conductance": not args.additive_current,
        "use_dale": not args.no_dale,
        "spatial_wiring": not args.no_spatial,
        "use_divnorm": args.divnorm,
        "use_stp": args.stp,
        "use_homeostasis": args.homeostasis,
        "use_oscillation": args.oscillation,
        "use_dendrites": args.dendrites,
        "use_laminar": args.laminar,
    }
    cfg = LMConfig(
        grid_size=args.grid_size,
        embed_dim=args.embed_dim,
        inner_steps=args.inner_steps,
        input_zone=args.input_zone,
        grad_checkpoint=args.grad_checkpoint,
        seed=args.seed,
        brain_overrides=brain_overrides,
    )
    # Classic biology ablations are the three that default ON (removing biology);
    # the additive mechanisms default OFF (adding biology) — report them apart so
    # an enabled mechanism is never mislabelled as an "ablation".
    classic = {"use_conductance", "use_dale", "spatial_wiring"}
    ablations = [k for k in classic if not brain_overrides.get(k, True)]
    enabled = [k for k, v in brain_overrides.items() if k not in classic and v]
    if ablations:
        print(f"[train] ABLATION active (biology removed): {', '.join(ablations)}")
    if enabled:
        print(f"[train] mechanisms enabled (biology added): {', '.join(enabled)}")
    model = BrainLanguageModel(tokenizer.vocab_size, cfg, device=device)

    # Diagnostic baseline: freeze the recurrent core so only the Linear read-in/
    # read-out learn. If this reaches ~the same loss as full training, the
    # dynamics are an (almost) fixed reservoir and the read-out is doing the
    # language modelling — the single most important thing to know about whether
    # the "brain learns language from its own dynamics" narrative holds.
    if args.frozen_reservoir:
        model.brain.edge_weight.requires_grad_(False)
        model.brain.neuron_bias.requires_grad_(False)
        print("[train] FROZEN RESERVOIR: edge_weight + neuron_bias frozen "
              "(training read-in/read-out only)")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] brain neurons={model.num_neurons}  synapses={model.brain.num_edges}  "
          f"trainable params={n_params}  device={device}")

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    seq_len = min(args.seq_len, max(2, data.numel() - 2))
    eval_every = args.eval_every or args.sample_every or 100

    # Optional metabolic sparse-coding penalty (toward the homeostatic set-point).
    sparse_target = model.brain.config.homeo_target_rate
    if args.sparse_weight > 0:
        model.track_activity = True
        print(f"[train] sparse-coding penalty: weight={args.sparse_weight} "
              f"target_rate={sparse_target}")

    # Optional LR schedule: linear warmup (10% of steps) then cosine decay.
    scheduler = None
    if args.lr_schedule == "warmcos":
        warmup = max(1, args.steps // 10)
        def lr_lambda(step):
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / max(1, args.steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        print(f"[train] LR schedule: warmup={warmup} then cosine to 0")

    # Choose the batch size: either fixed, or auto-tuned to fill ~target_mem_frac
    # of total machine memory (lets the brain "grok" with the largest batch the
    # 16 GB unified memory can hold).
    if args.target_mem_frac > 0:
        batch_size = auto_tune_batch_size(model, data, seq_len,
                                          args.target_mem_frac, device)
    else:
        batch_size = args.batch_size

    # ---- loop ---------------------------------------------------------------
    if args.persistent_state:
        stream = StreamingBatcher(data, seq_len, batch_size, device)
        batch_size = stream.batch_size  # may shrink for a tiny corpus
        carry_V = None
        print(f"[train] PERSISTENT STATE: {batch_size} contiguous lanes, V carried "
              f"across windows (TBPTT across batches)")
    elif args.tbptt_chunk > 0:
        print(f"[train] truncated BPTT: chunk={args.tbptt_chunk} (seq_len={seq_len})")

    model.train()
    t0 = time.time()
    running = 0.0
    best_val = float("inf")
    best_state = None
    grad_groups = None
    for step in range(1, args.steps + 1):
        if args.persistent_state:
            batch, reset = stream.next_batch()
            # Reset the carried state for lanes that wrapped to their start, and
            # detach so the graph does not extend across optimiser steps.
            if carry_V is not None:
                carry_V = carry_V.detach()
                carry_V[reset] = model.brain.config.E_L
            loss, carry_V = model.loss_with_state(batch, state=carry_V)
            if args.sparse_weight > 0:
                loss = loss + args.sparse_weight * model.sparse_penalty(sparse_target)
            opt.zero_grad()
            loss.backward()
            grad_groups = diagnostics.grad_norms_by_group(model) if args.diagnostics else None
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            loss_val = float(loss.item())
        elif args.learning_rule == "eprop":
            from positronic_brain.eprop import eprop_step
            batch = make_batch(data, seq_len, batch_size, device)
            loss_val = eprop_step(model, batch, opt, grad_clip=args.grad_clip)
        elif args.tbptt_chunk > 0:
            batch = make_batch(data, seq_len, batch_size, device)
            # TBPTT performs its own backward/step(s) per chunk and returns the
            # mean per-character loss over the sequence.
            loss_val = model.tbptt_step(batch, opt, chunk=args.tbptt_chunk,
                                        grad_clip=args.grad_clip)
        else:
            batch = make_batch(data, seq_len, batch_size, device)
            loss = model.loss_on(batch)
            if args.sparse_weight > 0:
                loss = loss + args.sparse_weight * model.sparse_penalty(sparse_target)
            opt.zero_grad()
            loss.backward()
            grad_groups = diagnostics.grad_norms_by_group(model) if args.diagnostics else None
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            loss_val = float(loss.item())

        if scheduler is not None:
            scheduler.step()
        running += loss_val
        if step % 20 == 0 or step == 1:
            avg = running / (20 if step % 20 == 0 else 1)
            running = 0.0 if step % 20 == 0 else running
            elapsed = time.time() - t0
            msg = (f"[train] step {step:>5}/{args.steps}  loss={avg:.4f}  "
                   f"({step / max(elapsed, 1e-6):.1f} it/s)")
            if args.diagnostics and grad_groups is not None:
                msg += ("  |g| head={head:.2e} token_in={token_in:.2e} "
                        "edge={edge_weight:.2e}".format(**grad_groups))
            print(msg, flush=True)

        # ---- held-out validation (the comparable signal) -------------------
        if val_data.numel() > seq_len + 1 and step % eval_every == 0:
            ev = (evaluate_persistent(model, val_data, seq_len, device)
                  if args.persistent_state
                  else evaluate(model, val_data, seq_len, batch_size, device))
            tag = ""
            if ev["ce"] < best_val:
                best_val = ev["ce"]
                if not args.no_best_checkpoint:
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                    tag = "  *best*"
            print(f"[val   ] step {step:>5}  val_bpc={ev['bpc']:.4f}  "
                  f"val_ppl={ev['ppl']:.2f}  sat={ev['saturation']:.1%}{tag}", flush=True)
            model.train()

        if args.sample_every and step % args.sample_every == 0:
            sample = model.generate(tokenizer, prompt="User: hello\nBrain:",
                                    max_new_tokens=120, temperature=0.8, top_k=30)
            print(f"[sample @ {step}] User: hello\\nBrain:{sample!r}\n", flush=True)
            model.train()

    # Restore the best-val checkpoint before final eval + save (early-stopping in
    # spirit: we keep the model that generalised best, not the last one).
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"[train] restored best-val checkpoint (val CE={best_val:.4f})")

    # ---- final held-out test (touched exactly once) -------------------------
    _final_eval = ((lambda d: evaluate_persistent(model, d, seq_len, device))
                   if args.persistent_state
                   else (lambda d: evaluate(model, d, seq_len, batch_size, device)))
    final = {}
    if val_data.numel() > seq_len + 1:
        final["val"] = _final_eval(val_data)
    if test_data.numel() > seq_len + 1:
        final["test"] = _final_eval(test_data)
    for split, ev in final.items():
        print(f"[final ] {split:<4} bpc={ev['bpc']:.4f}  ppl={ev['ppl']:.2f}  "
              f"ce={ev['ce']:.4f}  windows={ev['windows']}")

    # ---- diagnostics (settling + memory horizon) ----------------------------
    if args.diagnostics and val_data.numel() > 8:
        probe_ids = val_data[: min(64, val_data.numel())].tolist()
        sr = diagnostics.settling_residual(model, probe_ids[0])
        mh = diagnostics.memory_horizon(model, probe_ids, perturb_id=probe_ids[1])
        print(f"[diag  ] settling last-step |dV|={sr['last_step_dV']:.4f} "
              f"(relative={sr['relative']:.2%}; ~0 = settled, large = transient filter)")
        print(f"[diag  ] memory horizon (perturbation half-life) "
              f"~{mh['half_life_chars']:.0f} chars")

    # ---- save ---------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out, tokenizer)
    # Self-describing provenance sidecar: seed, git SHA, env, exact command, and
    # the run config — enough to reproduce or audit any reported number.
    meta = run_metadata(
        seed=args.seed, device=device,
        extra={
            "lm_config": cfg.to_dict(),
            "brain_overrides": brain_overrides,
            "steps": args.steps, "seq_len": seq_len, "batch_size": batch_size,
            "lr": args.lr, "grad_clip": args.grad_clip,
            "tbptt_chunk": args.tbptt_chunk, "grad_checkpoint": args.grad_checkpoint,
            "val_frac": args.val_frac, "test_frac": args.test_frac,
            "frozen_reservoir": args.frozen_reservoir,
            "persistent_state": args.persistent_state,
            "learning_rule": args.learning_rule,
            "held_out": {s: {"bpc": ev["bpc"], "ppl": ev["ppl"], "ce": ev["ce"]}
                         for s, ev in final.items()},
            "wall_clock_s": round(time.time() - t0, 1),
        },
    )
    with open(args.out + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[train] done in {time.time() - t0:.1f}s -> saved {args.out} "
          f"(+ {os.path.basename(args.out)}.meta.json)")


if __name__ == "__main__":
    main()
