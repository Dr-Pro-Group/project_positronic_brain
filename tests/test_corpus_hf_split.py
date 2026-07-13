"""Regression tests for the held-out split of streamed plain-text (--hf) corpora.

Bug: ``_stream_hf_text`` joined rows with a single newline while
``split_text_blocks`` splits on a blank line, so a dataset whose rows have no
internal blank line (e.g. tinystories) collapsed into ONE block — every character
went to train and the held-out val/test were empty, silently breaking the
content-disjoint-split honesty guarantee. Rows are now joined with a blank line so
each row is its own splittable block.

The HF dataset stream is faked (injected as the ``datasets`` module) so these run
fully offline.
"""

import sys
import types

from positronic_brain import corpus


def _fake_datasets(rows):
    mod = types.ModuleType("datasets")
    mod.load_dataset = lambda streaming=True, **kw: iter(rows)
    return mod


def test_hf_text_rows_join_as_separate_blocks(monkeypatch):
    rows = [{"text": f"story number {i} about a cat and a hat."} for i in range(20)]
    monkeypatch.setitem(sys.modules, "datasets", _fake_datasets(rows))
    text = corpus._stream_hf_text("fake", 100)
    # Each row is its own block (blank-line separated), so the held-out splitter
    # can partition rows instead of seeing one indivisible block.
    assert text.count("\n\n") == len(rows) - 1
    blocks = [b for b in text.split("\n\n") if b.strip()]
    assert len(blocks) == len(rows)


def test_hf_corpus_split_is_nonempty_and_disjoint(monkeypatch):
    rows = [{"text": f"unique story {i}: the quick brown fox jumps over it."}
            for i in range(20)]
    monkeypatch.setitem(sys.modules, "datasets", _fake_datasets(rows))
    tr, va, te = corpus.load_corpus_splits(
        hf="fake", builtin=False, val_frac=0.2, test_frac=0.2, seed=0)
    # No split may be empty — the whole point of the fix.
    assert tr.strip() and va.strip() and te.strip()
    tr_b = {b for b in tr.split("\n\n") if b.strip()}
    va_b = {b for b in va.split("\n\n") if b.strip()}
    te_b = {b for b in te.split("\n\n") if b.strip()}
    assert tr_b and va_b and te_b
    # Content-disjoint across all three splits.
    assert tr_b.isdisjoint(va_b)
    assert tr_b.isdisjoint(te_b)
    assert va_b.isdisjoint(te_b)
