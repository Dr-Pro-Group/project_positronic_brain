"""
Preference optimisation (DPO) for the generative brain.

This module turns the autoregressive :class:`~positronic_brain.language.
BrainLanguageModel` into a *policy* that can be aligned to preferences, with the
brain's **carried membrane potential acting as the recurrent policy memory**.
Because the brain already compresses all context into its evolving 3D membrane
state ``V`` (there is no attention window), scoring a continuation is simply a
matter of warming the state on the prompt and then reading the per-character
log-probabilities of the response — the state *is* the policy's memory.

We implement **Direct Preference Optimisation** (Rafailov et al., NeurIPS 2023)
because it is markedly simpler and more stable than PPO/RLHF for a small
recurrent model: no separate reward model, no rollouts, no value function — just
a classification-style loss over (prompt, chosen, rejected) triples against a
frozen reference copy of the brain. A PPO-style interface is sketched at the end
for the RL-on-POMDP track of the roadmap.

Honest scope
------------
DPO here is a *hook*: it is correct and runnable, but preference alignment only
pays off once supervised dialogue quality is already coherent (see the paper's
limitations). It is provided so the substrate demonstrably supports policy
learning, not as a tuned result.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .language import BrainLanguageModel, CharTokenizer


@dataclass
class DPOConfig:
    beta: float = 0.1          # KL strength: higher = stay closer to reference
    lr: float = 1e-4
    grad_clip: float = 0.5


def sequence_logprob(
    model: BrainLanguageModel,
    tokenizer: CharTokenizer,
    prompt: str,
    response: str,
) -> torch.Tensor:
    """
    Summed log-probability of ``response`` given ``prompt`` under the brain.

    The membrane state is warmed on the prompt (the recurrent "memory"), then we
    accumulate ``log p(response_t | everything so far)`` character by character.
    Returns a scalar tensor that is differentiable w.r.t. the model parameters.
    """
    device = model.device
    p_ids = tokenizer.encode(prompt)
    r_ids = tokenizer.encode(response)
    if not r_ids:
        return torch.zeros((), device=device)

    V = model.init_state(1)
    # Warm the state on every prompt character except the last (the last prompt
    # char is the input that *predicts* the first response character).
    for tid in p_ids[:-1]:
        V, _ = model.step_token(V, torch.tensor([tid], device=device))

    # Build the (input, target) stream that scores the response. With a prompt,
    # inputs = [last_prompt_char] + response[:-1] and targets = response. Without
    # a prompt we score response[1:] given response[0] (treat char 0 as a BOS).
    if p_ids:
        inputs = [p_ids[-1]] + r_ids[:-1]
        targets = r_ids
    else:
        inputs = r_ids[:-1]
        targets = r_ids[1:]

    total = torch.zeros((), device=device)
    for tok_in, tok_tgt in zip(inputs, targets):
        V, logits = model.step_token(V, torch.tensor([tok_in], device=device))
        logp = F.log_softmax(logits[0], dim=-1)
        total = total + logp[tok_tgt]
    return total


def dpo_loss(
    policy: BrainLanguageModel,
    reference: BrainLanguageModel,
    tokenizer: CharTokenizer,
    prompt: str,
    chosen: str,
    rejected: str,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    The DPO loss for a single (prompt, chosen, rejected) preference triple.

        L = -log σ( β [ (logπ_c - logπ_ref_c) - (logπ_r - logπ_ref_r) ] )

    where π is the policy brain and π_ref is a frozen reference brain. Minimising
    it raises the policy's relative likelihood of the preferred response while a
    β-weighted term keeps it near the reference (implicit KL control).
    """
    pc = sequence_logprob(policy, tokenizer, prompt, chosen)
    pr = sequence_logprob(policy, tokenizer, prompt, rejected)
    with torch.no_grad():
        rc = sequence_logprob(reference, tokenizer, prompt, chosen)
        rr = sequence_logprob(reference, tokenizer, prompt, rejected)
    logits = beta * ((pc - rc) - (pr - rr))
    return -F.logsigmoid(logits)


def make_reference(model: BrainLanguageModel) -> BrainLanguageModel:
    """A frozen deep copy of ``model`` to serve as the DPO reference policy."""
    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def dpo_finetune(
    model: BrainLanguageModel,
    tokenizer: CharTokenizer,
    pairs: Sequence[Tuple[str, str, str]],
    config: Optional[DPOConfig] = None,
    epochs: int = 1,
    log_every: int = 10,
) -> BrainLanguageModel:
    """
    Run DPO over a list of ``(prompt, chosen, rejected)`` triples.

    The reference policy is a frozen snapshot taken *before* fine-tuning, exactly
    as in the DPO paper. Returns the (in-place) updated model.
    """
    cfg = config or DPOConfig()
    reference = make_reference(model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    step = 0
    for _ in range(epochs):
        for prompt, chosen, rejected in pairs:
            loss = dpo_loss(model, reference, tokenizer, prompt, chosen, rejected,
                            beta=cfg.beta)
            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            step += 1
            if log_every and step % log_every == 0:
                print(f"[dpo] step {step}  loss={float(loss.item()):.4f}", flush=True)
    return model


# ---------------------------------------------------------------------------
# PPO interface sketch (roadmap Track A: brain as a recurrent policy on a POMDP).
# ---------------------------------------------------------------------------
class BrainPolicy:
    """
    Thin adapter exposing the brain as a recurrent policy for RL on memory tasks.

    The membrane potential ``V`` is the agent's recurrent state: ``reset`` clears
    it to rest, ``act`` advances one step and returns an action distribution. A
    full PPO loop (advantage estimation, clipped surrogate, value head) is left to
    the RL track of the roadmap; this class defines the contract that loop needs
    so the brain can drop into a standard actor-critic trainer.
    """

    def __init__(self, model: BrainLanguageModel):
        self.model = model
        self._V: Optional[torch.Tensor] = None

    def reset(self, batch: int = 1) -> None:
        self._V = self.model.init_state(batch)

    def act(self, token_ids: torch.Tensor):
        """Return (action_logits, value_placeholder); advances the recurrent state."""
        assert self._V is not None, "call reset() first"
        self._V, logits = self.model.step_token(self._V, token_ids)
        return logits, None  # plug a value head here for actor-critic
