"""
Subword tokenizer for public LLM-scale text (no extra deps).

Pure-Python byte-pair encoding trained on a sample of the corpus. Avoids the
``tokenizers`` / ``sentencepiece`` packages so laptop/Mini envs with only
``torch`` + ``datasets`` still work.

API mirrors :class:`~positronic_brain.language.CharTokenizer` enough that LM
code can call ``encode`` / ``decode`` / ``vocab_size`` / ``to_dict``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


_WORD_RE = re.compile(r"\S+|\s+")


class SubwordTokenizer:
    """Byte-pair subword tokenizer with special tokens.

    Specials: ``<unk>=0``, ``<pad>=1``, ``<bos>=2``, ``<eos>=3``.
    Base units are UTF-8 bytes decoded as latin-1 single chars (256 base ids
    after specials) so any Unicode text is encodable without OOV at the byte
    level; merges build multi-byte / multi-char tokens.
    """

    SPECIALS = ("<unk>", "<pad>", "<bos>", "<eos>")

    def __init__(
        self,
        merges: Optional[List[Tuple[str, str]]] = None,
        vocab: Optional[Dict[str, int]] = None,
    ):
        self.merges: List[Tuple[str, str]] = list(merges or [])
        if vocab is not None:
            self.token_to_id = dict(vocab)
        else:
            self.token_to_id = {s: i for i, s in enumerate(self.SPECIALS)}
            # 256 base bytes as latin-1 characters
            for b in range(256):
                ch = bytes([b]).decode("latin-1")
                if ch not in self.token_to_id:
                    self.token_to_id[ch] = len(self.token_to_id)
            for a, b in self.merges:
                tok = a + b
                if tok not in self.token_to_id:
                    self.token_to_id[tok] = len(self.token_to_id)
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        # ranked merge table for greedy encoding
        self._merge_rank = {pair: i for i, pair in enumerate(self.merges)}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def unk_id(self) -> int:
        return 0

    @property
    def pad_id(self) -> int:
        return 1

    @property
    def bos_id(self) -> int:
        return 2

    @property
    def eos_id(self) -> int:
        return 3

    # ------------------------------------------------------------------ train
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 4096,
        max_chars: int = 2_000_000,
    ) -> "SubwordTokenizer":
        """Train BPE merges on up to ``max_chars`` of sample text."""
        buf: List[str] = []
        n = 0
        for t in texts:
            if not t:
                continue
            buf.append(t)
            n += len(t)
            if n >= max_chars:
                break
        blob = "\n".join(buf)
        if not blob:
            return cls()

        # Represent each "word" as a tuple of base symbols (bytes as latin-1).
        words: Counter = Counter()
        for m in _WORD_RE.finditer(blob):
            w = m.group(0)
            # encode as latin-1 byte chars
            syms = tuple(w.encode("utf-8").decode("latin-1"))
            if syms:
                words[syms] += 1

        merges: List[Tuple[str, str]] = []
        # base vocab size: specials + 256
        target_merges = max(0, int(vocab_size) - 4 - 256)
        # Cap unique word types for speed (tail is long-tail noise for tiny BPE).
        if len(words) > 50_000:
            words = Counter(dict(words.most_common(50_000)))

        def pair_counts(ws: Counter) -> Counter:
            pc: Counter = Counter()
            for syms, c in ws.items():
                for i in range(len(syms) - 1):
                    pc[(syms[i], syms[i + 1])] += c
            return pc

        def apply_merge(ws: Counter, pair: Tuple[str, str]) -> Counter:
            a, b = pair
            merged = a + b
            out: Counter = Counter()
            for syms, c in ws.items():
                if a not in syms or b not in syms:
                    # fast path: pair cannot occur (approx; may miss rare split cases)
                    # still scan if either symbol present as substring of a multi-char piece
                    if all(a != s and b != s for s in syms):
                        out[syms] += c
                        continue
                new: List[str] = []
                i = 0
                L = len(syms)
                while i < L:
                    if i < L - 1 and syms[i] == a and syms[i + 1] == b:
                        new.append(merged)
                        i += 2
                    else:
                        new.append(syms[i])
                        i += 1
                out[tuple(new)] += c
            return out

        # Incremental-ish: recompute pair counts each merge (still O(merges·tokens)
        # but with capped word types this finishes in seconds for 1–2M chars).
        for _ in range(target_merges):
            pc = pair_counts(words)
            if not pc:
                break
            best, best_c = max(pc.items(), key=lambda kv: (kv[1], kv[0]))
            if best_c < 2:
                break
            merges.append(best)
            words = apply_merge(words, best)

        return cls(merges=merges)

    # ----------------------------------------------------------------- encode
    def _bpe(self, token: str) -> List[str]:
        if not token:
            return []
        # start from base symbols
        syms = list(token.encode("utf-8").decode("latin-1"))
        if len(syms) == 1:
            return syms
        while True:
            pairs = [(syms[i], syms[i + 1]) for i in range(len(syms) - 1)]
            ranked = [
                (self._merge_rank[p], i, p)
                for i, p in enumerate(pairs)
                if p in self._merge_rank
            ]
            if not ranked:
                break
            _, idx, (a, b) = min(ranked, key=lambda x: x[0])
            # merge at first occurrence of this rank (greedy left-to-right for that pair)
            # recompute using best global rank
            best_rank = min(self._merge_rank[p] for p in pairs if p in self._merge_rank)
            new: List[str] = []
            i = 0
            merged_once = False
            while i < len(syms):
                if (
                    not merged_once
                    and i < len(syms) - 1
                    and (syms[i], syms[i + 1]) in self._merge_rank
                    and self._merge_rank[(syms[i], syms[i + 1])] == best_rank
                ):
                    new.append(syms[i] + syms[i + 1])
                    i += 2
                    merged_once = True
                else:
                    new.append(syms[i])
                    i += 1
            syms = new
            if len(syms) == 1:
                break
        return syms

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for m in _WORD_RE.finditer(text or ""):
            for piece in self._bpe(m.group(0)):
                ids.append(self.token_to_id.get(piece, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        parts: List[str] = []
        for i in ids:
            t = self.id_to_token.get(int(i), "")
            if t in self.SPECIALS:
                continue
            parts.append(t)
        # pieces are latin-1 byte strings; recover utf-8
        raw = "".join(parts).encode("latin-1", errors="ignore")
        return raw.decode("utf-8", errors="replace")

    def unk_rate(self, text: str) -> float:
        ids = self.encode(text)
        if not ids:
            return 0.0
        return sum(1 for i in ids if i == self.unk_id) / len(ids)

    def to_dict(self) -> Dict:
        return {
            "type": "subword_bpe",
            "merges": [list(p) for p in self.merges],
            "vocab": self.token_to_id,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "SubwordTokenizer":
        merges = [tuple(p) for p in d.get("merges", [])]
        return cls(merges=merges, vocab=d.get("vocab"))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "SubwordTokenizer":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
