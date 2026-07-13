"""Tests for gradient checkpointing and its interaction with stateful mechanisms.

Regression guard: ``step`` mutates transient module state (STP, oscillation phase,
homeostatic gain, adaptation) in place each call. Wrapping the reverberation in
``torch.utils.checkpoint`` re-executes the forward during backward against the
END-of-forward state, so the recomputed function differs from the forward and the
recurrent-core gradient is silently corrupted. The fix disables checkpointing when
any such mechanism is active, so the gradients must match the plain path.
"""

import warnings

import torch

from positronic_brain.model import BrainConfig, PositronicBrain
from positronic_brain.language import BrainLanguageModel, CharTokenizer, LMConfig


def _grads(model, batch):
    model.zero_grad()
    loss = model.loss_on(batch)
    loss.backward()
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def _batch(tok, n=48):
    text = "User: hi\nBrain: hello\n" * 20
    return torch.tensor(tok.encode(text))[:n].unsqueeze(0)


def test_step_mutates_state_reports_stateful_mechanisms():
    assert not PositronicBrain(BrainConfig(grid_size=4), device="cpu").step_mutates_state()
    for flag in ("use_stp", "use_oscillation", "use_homeostasis", "use_adaptation"):
        b = PositronicBrain(BrainConfig(grid_size=4, **{flag: True}), device="cpu")
        assert b.step_mutates_state(), flag


def test_grad_checkpoint_matches_plain_without_stateful():
    # With no stateful mechanism, checkpointing IS used and must be numerically
    # equivalent to the plain path (the whole point of gradient checkpointing).
    tok = CharTokenizer.from_text("User: hi\nBrain: hello\n" * 20)
    batch = _batch(tok)
    base = dict(grid_size=6, embed_dim=16, inner_steps=3, seed=1)
    m_plain = BrainLanguageModel(tok.vocab_size, LMConfig(grad_checkpoint=False, **base), device="cpu")
    m_ckpt = BrainLanguageModel(tok.vocab_size, LMConfig(grad_checkpoint=True, **base), device="cpu")
    m_ckpt.load_state_dict(m_plain.state_dict())
    g_plain = _grads(m_plain, batch)
    g_ckpt = _grads(m_ckpt, batch)
    for k in g_plain:
        assert torch.allclose(g_plain[k], g_ckpt[k], atol=1e-5), k


def test_grad_checkpoint_disabled_and_correct_under_stateful_mechanism():
    # The bug: STP + grad_checkpoint recomputed against end-of-forward STP state
    # (already cleared by stp_end) -> wrong gradients, no error. The fix falls back
    # to the plain path, so gradients must be EXACTLY the plain-path gradients, and
    # a warning must be emitted (no silent corruption).
    tok = CharTokenizer.from_text("User: hi\nBrain: hello\n" * 20)
    batch = _batch(tok)
    base = dict(grid_size=6, embed_dim=16, inner_steps=3, seed=2,
                brain_overrides={"use_stp": True})
    m_plain = BrainLanguageModel(tok.vocab_size, LMConfig(grad_checkpoint=False, **base), device="cpu")
    m_ckpt = BrainLanguageModel(tok.vocab_size, LMConfig(grad_checkpoint=True, **base), device="cpu")
    m_ckpt.load_state_dict(m_plain.state_dict())
    g_plain = _grads(m_plain, batch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g_ckpt = _grads(m_ckpt, batch)
    # Fell back to the plain path -> identical code path -> exactly equal gradients.
    for k in g_plain:
        assert torch.equal(g_plain[k], g_ckpt[k]), k
    assert any(issubclass(c.category, RuntimeWarning) and "disabled" in str(c.message)
               for c in caught)
