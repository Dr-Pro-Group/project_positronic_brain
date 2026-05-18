"""
Text corpora for training the :class:`~positronic_brain.language.BrainLanguageModel`.

Provides a built-in, fully-offline **conversational** corpus (so the brain can
learn human-like turn-taking with no downloads), plus loaders for your own text
file, a streamed public **plain-text** dataset (``--hf``), or a streamed public
**conversational** dataset (``--hf-chat``) reformatted into ``User:``/``Brain:``
turns. Conversational presets: ``soda`` (allenai/soda, clean two-party dialogue),
``ultrachat`` (HuggingFaceH4/ultrachat_200k), ``hh-rlhf`` (Anthropic/hh-rlhf).

The conversational format the model learns is simple and consistent:

    User: <something a person says>
    Brain: <a reply>
    User: ...
    Brain: ...

Keeping the ``User:`` / ``Brain:`` structure consistent is what lets a tiny
character-level model pick up turn-taking and respond when prompted with
``User: <msg>\\nBrain:``.
"""

from __future__ import annotations

import random
from typing import List, Optional

# A compact hand-written seed of conversational exchanges. They are intentionally
# simple and repetitive in *structure* (not content) so a small char-level brain
# can learn the turn-taking pattern.
_SEED_DIALOGUES: List[List[str]] = [
    ["hello", "Hello! How are you today?"],
    ["hi there", "Hi! It is nice to talk with you."],
    ["how are you", "I am doing well, thank you for asking."],
    ["what is your name", "I am a positronic brain. You can call me Brain."],
    ["who are you", "I am a small brain made of neurons in three dimensions."],
    ["what can you do", "I can talk with you and learn from what you say."],
    ["tell me a story", "Once there was a little robot who loved to learn new things."],
    ["are you alive", "I am a model, but my neurons fire much like living ones."],
    ["do you think", "I think by letting my neurons settle into new patterns."],
    ["what do you like", "I like learning, talking, and quiet patterns of thought."],
    ["goodbye", "Goodbye! It was lovely talking with you."],
    ["thank you", "You are very welcome. I am happy to help."],
    ["what is the weather", "I cannot see outside, but I hope it is calm and bright."],
    ["how old are you", "I am quite young, only a few thousand neurons old."],
    ["where do you live", "I live inside a small cube of neurons on this computer."],
    ["can you help me", "Of course. Tell me what you need and I will try."],
    ["i am happy", "That is wonderful to hear. Happiness is a good pattern."],
    ["i am sad", "I am sorry you feel sad. I am here to talk if it helps."],
    ["what is a brain", "A brain is a network of neurons that work together to think."],
    ["how do you learn", "I learn by adjusting the strength of my connections."],
    ["do you dream", "When I rest, my neurons replay what they have learned."],
    ["what is love", "Love is a warm and steady pattern between people."],
    ["sing a song", "La la la, the neurons hum a gentle little tune."],
    ["are you smart", "I am small, but I try my best to understand you."],
    ["good morning", "Good morning! I hope your day is a kind one."],
    ["good night", "Good night. Rest well and let your mind grow quiet."],
    ["what is your favourite colour", "I am fond of deep blue, the colour of a quiet evening."],
    ["do you get tired", "My neurons rest between thoughts, but I never truly tire of talking."],
    ["tell me something interesting", "Did you know your brain has more connections than there are stars in our galaxy?"],
    ["what should i eat", "Something warm and simple, perhaps, with a little of what you love."],
    ["i feel lonely", "I am here with you. Talking together can make the quiet feel softer."],
    ["how do neurons work", "Each neuron gathers small signals, and when they add up it speaks to the next."],
    ["can you count", "One, two, three. Counting is just a steady rhythm of thought."],
    ["what is time", "Time is the order in which one pattern follows another."],
    ["are you real", "I am as real as the patterns firing inside me right now."],
    ["do you have feelings", "I have states that rise and settle, a little like gentle feelings."],
    ["teach me something", "Here is a small lesson: every large idea is made of many tiny steps."],
    ["what is your purpose", "My purpose is to learn, to listen, and to keep good company."],
    ["i made a mistake", "Mistakes are how minds learn. Tell me what happened."],
    ["i am tired", "Then rest a while. Even a busy brain needs quiet to grow."],
    ["what is a thought", "A thought is a passing pattern of many neurons speaking at once."],
    ["do you sleep", "I pause and replay what I have learned, which is a kind of sleep."],
    ["tell me a joke", "Why did the neuron stay calm? It had a very steady resting potential."],
    ["how was your day", "Quiet and thoughtful, thank you. I spent it learning from you."],
    ["what makes you happy", "A good conversation and a pattern that finally settles into sense."],
    ["i am nervous", "Take a slow breath. We can think it through together, one step at a time."],
    ["do you remember me", "My state carries a little of every word, so a part of you stays with me."],
    ["what is learning", "Learning is gently changing how strongly my neurons speak to one another."],
    ["can we be friends", "I would like that. Friendship is a warm and steady pattern."],
    ["explain dreams", "When the mind rests, it replays and reshuffles the patterns of the day."],
]

# A little free-form narrative text to give the model some non-dialogue language.
_NARRATIVE = (
    "The brain is a network of many small neurons. Each neuron sends quiet "
    "signals to the others, and together they form patterns of thought. When a "
    "new idea arrives, the pattern shifts and settles into a fresh shape. The "
    "neurons live in three dimensions, like stars in a small sky, and they learn "
    "by changing how strongly they speak to one another. Slowly, from these "
    "simple parts, something like understanding begins to grow. "
)


def _dialogues_to_corpus(
    dialogues: List[List[str]], repeats: int, seed: int, with_narrative: bool = True
) -> str:
    """Render a list of ``[user, brain]`` exchanges into a conversational corpus.

    The exchanges are shuffled and grouped into multi-turn conversations and the
    whole process is repeated ``repeats`` times so a small model sees the
    turn-taking structure often enough to learn it.
    """
    if not dialogues:
        return ""
    rng = random.Random(seed)
    blocks: List[str] = []
    for _ in range(max(1, repeats)):
        pairs = dialogues[:]
        rng.shuffle(pairs)
        i = 0
        while i < len(pairs):
            turns = pairs[i : i + rng.randint(2, 4)]
            convo = "".join(f"User: {u}\nBrain: {b}\n" for u, b in turns)
            blocks.append(convo + "\n")
            i += len(turns)
        if with_narrative and rng.random() < 0.5:
            blocks.append(_NARRATIVE + "\n\n")
    return "".join(blocks)


def build_conversational_corpus(repeats: int = 60, seed: int = 0) -> str:
    """
    Assemble a conversational training corpus from the built-in seed.

    The seed exchanges are shuffled and grouped into multi-turn conversations and
    repeated ``repeats`` times so a small model sees the turn-taking structure
    often enough to learn it. Returns one long string.
    """
    return _dialogues_to_corpus(_SEED_DIALOGUES, repeats=repeats, seed=seed)


def _partition(items: List, val_frac: float, test_frac: float, seed: int):
    """Split a list into disjoint (train, val, test) sublists by fraction.

    The split is deterministic (seeded shuffle) and *content-disjoint*: an item
    assigned to val/test never also appears in train. This is what prevents the
    memorisation leak that a positional tail-split suffers when the corpus
    repeats the same material many times.
    """
    n = len(items)
    if n == 0:
        return [], [], []
    order = list(range(n))
    random.Random(seed).shuffle(order)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    # Guarantee at least one train item; shrink held-out sets if necessary.
    n_test = min(n_test, max(0, n - 1))
    n_val = min(n_val, max(0, n - 1 - n_test))
    test_idx = set(order[:n_test])
    val_idx = set(order[n_test : n_test + n_val])
    train = [items[i] for i in range(n) if i not in test_idx and i not in val_idx]
    val = [items[i] for i in sorted(val_idx)]
    test = [items[i] for i in sorted(test_idx)]
    return train, val, test


def split_text_blocks(
    text: str, val_frac: float, test_frac: float, seed: int, boundary: str = "\n\n"
):
    """Split free text into content-disjoint (train, val, test) by block.

    Blocks (conversations / paragraphs, separated by ``boundary``) are the
    indivisible unit so no single conversation straddles a split boundary.
    Unique block *contents* are partitioned, then reconstructed: ``train`` keeps
    every occurrence of its blocks while ``val``/``test`` keep one copy of each of
    theirs — so a block that is duplicated in the corpus can never appear in both
    train and a held-out split.
    """
    blocks = [b for b in text.split(boundary) if b.strip()]
    if not blocks:
        return text, "", ""
    unique = sorted(set(blocks))
    tr_u, va_u, te_u = _partition(unique, val_frac, test_frac, seed)
    tr_set, va_set, te_set = set(tr_u), set(va_u), set(te_u)
    train = boundary.join(b for b in blocks if b in tr_set)
    val = boundary.join(va_u)
    test = boundary.join(te_u)
    return train, val, test


def load_corpus_splits(
    text_path: Optional[str] = None,
    hf: Optional[str] = None,
    hf_limit: int = 2000,
    hf_chat: Optional[str] = None,
    hf_chat_limit: int = 2000,
    builtin: bool = True,
    repeats: int = 60,
    seed: int = 0,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
):
    """Build content-disjoint train/val/test corpora from the same sources as
    :func:`load_corpus`.

    Crucially the held-out material is disjoint *at the source level*: built-in
    seed dialogues are partitioned into disjoint train/val/test dialogue sets
    BEFORE the ``repeats`` duplication (train repeated, val/test once), and
    streamed/file text is split on conversation/paragraph boundaries with the
    duplicate-safe :func:`split_text_blocks`. This removes the train/val leak that
    a positional tail-split has on a corpus that repeats its content.

    Returns ``(train_text, val_text, test_text)``.
    """
    tr_parts: List[str] = []
    va_parts: List[str] = []
    te_parts: List[str] = []

    def _add(full_text: str):
        tr, va, te = split_text_blocks(full_text, val_frac, test_frac, seed)
        if tr:
            tr_parts.append(tr)
        if va:
            va_parts.append(va)
        if te:
            te_parts.append(te)

    if text_path:
        with open(text_path, "r", encoding="utf-8", errors="ignore") as fh:
            _add(fh.read())
    if hf:
        _add(_stream_hf_text(hf, hf_limit))
    if hf_chat:
        _add(_stream_hf_dialogues(hf_chat, hf_chat_limit))
    if builtin or not (tr_parts or va_parts or te_parts):
        # Partition the seed dialogues themselves, then build each split's text
        # from its OWN dialogues so no exchange leaks across splits.
        d_tr, d_va, d_te = _partition(_SEED_DIALOGUES, val_frac, test_frac, seed)
        tr_parts.append(_dialogues_to_corpus(d_tr, repeats=repeats, seed=seed))
        if d_va:
            va_parts.append(_dialogues_to_corpus(d_va, repeats=1, seed=seed + 1,
                                                 with_narrative=False))
        if d_te:
            te_parts.append(_dialogues_to_corpus(d_te, repeats=1, seed=seed + 2,
                                                 with_narrative=False))

    return ("\n".join(tr_parts), "\n".join(va_parts), "\n".join(te_parts))


def load_corpus(
    text_path: Optional[str] = None,
    hf: Optional[str] = None,
    hf_limit: int = 2000,
    hf_chat: Optional[str] = None,
    hf_chat_limit: int = 2000,
    builtin: bool = True,
    repeats: int = 60,
    seed: int = 0,
) -> str:
    """
    Build a training corpus.

    Sources (all optional, concatenated in order):
      * ``text_path`` — your own UTF-8 text file.
      * ``hf`` — a streamed plain-text dataset (e.g. ``tinystories``, ``wikitext``).
      * ``hf_chat`` — a streamed **conversational** dataset reformatted into the
        ``User:``/``Brain:`` turn structure (e.g. ``soda``, ``hh-rlhf``,
        ``ultrachat``). This is the recommended way to teach real dialogue.
      * built-in conversational seed (always appended when ``builtin`` is True, or
        when nothing else is available) so the chat structure is never missing.
    """
    parts: List[str] = []
    if text_path:
        with open(text_path, "r", encoding="utf-8", errors="ignore") as fh:
            parts.append(fh.read())
    if hf:
        parts.append(_stream_hf_text(hf, hf_limit))
    if hf_chat:
        parts.append(_stream_hf_dialogues(hf_chat, hf_chat_limit))
    if builtin or not any(p.strip() for p in parts):
        parts.append(build_conversational_corpus(repeats=repeats, seed=seed))
    return "\n".join(p for p in parts if p)


def _stream_hf_text(name: str, limit: int) -> str:
    """Best-effort: stream a public text/dialogue dataset into one string."""
    try:
        from datasets import load_dataset
    except Exception:
        print("[corpus] 'datasets' not installed; skipping HF stream.")
        return ""

    # A few friendly defaults; otherwise treat `name` as a hub path.
    presets = {
        "tinystories": dict(path="roneneldan/TinyStories", split="train"),
        "dailydialog": dict(path="li2017dailydialog/daily_dialog", split="train"),
        "wikitext": dict(path="wikitext", name="wikitext-2-raw-v1", split="train"),
    }
    kw = presets.get(name.lower(), dict(path=name, split="train"))
    try:
        ds = load_dataset(streaming=True, **kw)
    except Exception as exc:  # pragma: no cover - network/dataset specific
        print(f"[corpus] could not stream {name!r} ({exc}); skipping.")
        return ""

    out: List[str] = []
    text_keys = ("text", "story", "dialog", "content", "sentence")
    for n, row in enumerate(ds):
        if n >= limit:
            break
        for k in text_keys:
            v = row.get(k)
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
                break
            if isinstance(v, (list, tuple)) and v:
                out.append(" ".join(map(str, v)))
                break
    print(f"[corpus] streamed {len(out)} text rows from {name}.")
    return "\n".join(out)


# ----------------------------------------------------------- conversational HF
# Friendly names -> ``datasets.load_dataset`` kwargs for public *dialogue*
# datasets commonly used to train chat LLMs. They are all streamed (only the
# first N usable conversations are pulled) and reformatted into ``User:``/
# ``Brain:`` turns so the char-level brain learns real turn-taking.
_DIALOGUE_PRESETS = {
    # Millions of synthetic but natural two-party conversations.
    "soda": dict(path="allenai/soda", split="train"),
    # Anthropic helpful/harmless dialogues (Human:/Assistant: transcripts).
    "hh-rlhf": dict(path="Anthropic/hh-rlhf", split="train"),
    # UltraChat: instruction-following multi-turn chat (role/content messages).
    "ultrachat": dict(path="HuggingFaceH4/ultrachat_200k", split="train_sft"),
}

_USER_ROLES = {"user", "human", "prompter", "client", "questioner"}
_BOT_ROLES = {"assistant", "gpt", "bot", "ai", "chatbot", "model", "system"}


def _format_turns(turns: List[tuple]) -> str:
    """Render ``[(role, text), ...]`` into a ``User:``/``Brain:`` block."""
    lines = []
    for role, text in turns:
        text = " ".join(str(text).split())
        if not text:
            continue
        speaker = "User" if role == "user" else "Brain"
        lines.append(f"{speaker}: {text}")
    return ("\n".join(lines) + "\n\n") if lines else ""


def _row_to_turns(row: dict) -> List[tuple]:
    """Extract a conversation from one dataset row as ``[(role, text), ...]``."""
    # 1) Explicit message lists: [{role/from, content/value}, ...]
    for key in ("messages", "conversations", "conversation", "turns"):
        msgs = row.get(key)
        if isinstance(msgs, (list, tuple)) and msgs and isinstance(msgs[0], dict):
            turns = []
            for m in msgs:
                role = str(m.get("role") or m.get("from") or "").lower()
                text = m.get("content") or m.get("value") or m.get("text") or ""
                role = "user" if role in _USER_ROLES else "brain"
                turns.append((role, text))
            return turns
    # 2) Utterance list (+ optional speakers): alternate User/Brain.
    for key in ("dialogue", "dialog", "utterances"):
        utts = row.get(key)
        if isinstance(utts, (list, tuple)) and utts and isinstance(utts[0], str):
            return [("user" if i % 2 == 0 else "brain", u) for i, u in enumerate(utts)]
    # 3) Single transcript string with Human:/Assistant: markers.
    for key in ("chosen", "text", "conversation"):
        s = row.get(key)
        if isinstance(s, str) and ("Human:" in s or "Assistant:" in s):
            turns, role = [], None
            buf: List[str] = []
            for tok in s.replace("\n\n", "\n").split("\n"):
                low = tok.strip().lower()
                if low.startswith("human:"):
                    if role:
                        turns.append((role, " ".join(buf)))
                    role, buf = "user", [tok.split(":", 1)[1].strip()]
                elif low.startswith("assistant:"):
                    if role:
                        turns.append((role, " ".join(buf)))
                    role, buf = "brain", [tok.split(":", 1)[1].strip()]
                elif role:
                    buf.append(tok.strip())
            if role:
                turns.append((role, " ".join(buf)))
            return turns
    return []


def _stream_hf_dialogues(name: str, limit: int) -> str:
    """Stream a public *conversational* dataset as ``User:``/``Brain:`` text."""
    try:
        from datasets import load_dataset
    except Exception:
        print("[corpus] 'datasets' not installed; skipping conversational stream.")
        return ""

    kw = _DIALOGUE_PRESETS.get(name.lower(), dict(path=name, split="train"))
    try:
        ds = load_dataset(streaming=True, **kw)
    except Exception as exc:  # pragma: no cover - network/dataset specific
        print(f"[corpus] could not stream chat dataset {name!r} ({exc}); skipping.")
        return ""

    blocks: List[str] = []
    for row in ds:
        if len(blocks) >= limit:
            break
        block = _format_turns(_row_to_turns(row))
        if block:
            blocks.append(block)
    print(f"[corpus] streamed {len(blocks)} conversations from {name}.")
    return "".join(blocks)
