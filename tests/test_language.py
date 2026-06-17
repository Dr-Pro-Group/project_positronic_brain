"""Tests for the biomimetic generative language model."""

import torch

from positronic_brain.language import BrainLanguageModel, CharTokenizer, LMConfig
from positronic_brain.corpus import (
    build_conversational_corpus, load_corpus, load_corpus_splits,
)
from positronic_brain.streaming import StreamingBatcher


def _tiny_model(text):
    tok = CharTokenizer.from_text(text)
    cfg = LMConfig(grid_size=6, embed_dim=16, inner_steps=2)
    return BrainLanguageModel(tok.vocab_size, cfg, device="cpu"), tok


def test_tokenizer_roundtrip():
    text = "hello, brain! 123"
    tok = CharTokenizer.from_text(text)
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    # +1 for the reserved UNK token at index 0.
    assert tok.vocab_size == len(set(text)) + 1
    assert tok.itos[0] == CharTokenizer.UNK


def test_tokenizer_unk_handling():
    # Out-of-vocabulary characters map to UNK (index 0), not dropped.
    tok = CharTokenizer.from_text("abc")
    ids = tok.encode("axc")  # 'x' is OOV
    assert ids[1] == tok.unk_id == 0
    assert len(ids) == 3  # nothing dropped
    assert tok.unk_rate("axc") == 1 / 3


def test_tokenizer_serialization():
    tok = CharTokenizer.from_text("abcABC ")
    tok2 = CharTokenizer.from_dict(tok.to_dict())
    assert tok2.itos == tok.itos
    assert tok2.encode("cab") == tok.encode("cab")


def test_forward_shapes():
    model, tok = _tiny_model("User: hi\nBrain: hello\n")
    tokens = torch.tensor(tok.encode("User: hi\nBrain:"))[None, :]
    logits, state = model(tokens)
    assert logits.shape == (1, tokens.shape[1], tok.vocab_size)
    assert state.shape == (1, model.num_neurons)


def test_state_persists_across_tokens():
    # Two different single-token inputs should leave different membrane states.
    model, tok = _tiny_model("abcdef ")
    V = model.init_state(1)
    a = tok.encode("a")[0]
    b = tok.encode("b")[0]
    Va, _ = model.step_token(V.clone(), torch.tensor([a]))
    Vb, _ = model.step_token(V.clone(), torch.tensor([b]))
    assert not torch.allclose(Va, Vb)


def test_generation_runs():
    model, tok = _tiny_model("User: hi\nBrain: hello there\n")
    out = model.generate(tok, prompt="User: hi\nBrain:", max_new_tokens=15)
    assert isinstance(out, str)
    assert len(out) <= 15
    # every generated char must be in the vocabulary
    assert all(c in tok.stoi for c in out)


def test_training_reduces_loss():
    text = "User: hi\nBrain: hello\n" * 40
    model, tok = _tiny_model(text)
    data = torch.tensor(tok.encode(text))
    batch = data[: 60].unsqueeze(0)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    first = float(model.loss_on(batch).item())
    for _ in range(30):
        loss = model.loss_on(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    last = float(model.loss_on(batch).item())
    assert last < first


def test_save_load_roundtrip(tmp_path):
    model, tok = _tiny_model("User: hi\nBrain: hello\n")
    p = tmp_path / "lm.pt"
    model.save(str(p), tok)
    loaded, tok2 = BrainLanguageModel.load(str(p), device="cpu")
    assert tok2.itos == tok.itos
    assert loaded.vocab_size == model.vocab_size

    tokens = torch.tensor(tok.encode("User: hi"))[None, :]
    with torch.no_grad():
        l1, _ = model(tokens)
        l2, _ = loaded(tokens)
    assert torch.allclose(l1, l2, atol=1e-5)


def test_conversational_corpus_structure():
    text = build_conversational_corpus(repeats=2)
    assert "User:" in text and "Brain:" in text
    assert len(text) > 100


def test_load_corpus_builtin_default():
    text = load_corpus()  # no args -> built-in conversational
    assert "User:" in text


def test_corpus_splits_are_content_disjoint():
    # The built-in corpus repeats fixed dialogues; the split must still keep the
    # held-out material out of train (no leaked block appears in both).
    train, val, test = load_corpus_splits(builtin=True, repeats=4, seed=0,
                                          val_frac=0.2, test_frac=0.2)
    assert train and val and test
    train_blocks = {b for b in train.split("\n\n") if b.strip()}
    val_blocks = {b for b in val.split("\n\n") if b.strip()}
    test_blocks = {b for b in test.split("\n\n") if b.strip()}
    assert train_blocks.isdisjoint(val_blocks)
    assert train_blocks.isdisjoint(test_blocks)


def test_streaming_batcher_is_contiguous_and_wraps():
    data = torch.arange(100)
    batcher = StreamingBatcher(data, seq_len=8, batch_size=4, device="cpu")
    b1, reset1 = batcher.next_batch()
    assert b1.shape == (4, 9)
    assert not reset1.any()  # first batch never resets
    # lane 0 starts at 0; its second window must continue contiguously at +8.
    b2, _ = batcher.next_batch()
    assert int(b2[0, 0]) == int(b1[0, 0]) + 8


def test_persistent_state_lowers_loss_on_periodic_text():
    # On a strongly periodic stream, knowing the phase (warm carried state) must
    # predict the next char better than a cold reset — this is the behavioural
    # test backing the "membrane state carries context" claim.
    torch.manual_seed(0)
    text = ("abcdefgh" * 400)
    tok = CharTokenizer.from_text(text)
    cfg = LMConfig(grid_size=6, embed_dim=16, inner_steps=2)
    model = BrainLanguageModel(tok.vocab_size, cfg, device="cpu")
    data = torch.tensor(tok.encode(text))

    # Train with carried state so the model actually learns to use context.
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    seq_len = 16
    V = None
    for start in range(0, 16 * seq_len, seq_len):
        chunk = data[start : start + seq_len + 1].unsqueeze(0)
        if V is not None:
            V = V.detach()
        loss, V = model.loss_with_state(chunk, state=V)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        # warm: feed a prefix, carry V into the scored window.
        prefix = data[: seq_len].unsqueeze(0)
        _, warm_V = model(prefix)
        window = data[seq_len : 2 * seq_len + 1].unsqueeze(0)
        warm_loss, _ = model.loss_with_state(window, state=warm_V.detach())
        cold_loss, _ = model.loss_with_state(window, state=None)
    assert float(warm_loss) < float(cold_loss)
