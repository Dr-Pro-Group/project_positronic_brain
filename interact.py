#!/usr/bin/env python
"""
Interactive live-learning session for the Positronic Brain.

Feed the brain real multimodal data (images, text, audio, video) and watch it
learn online via periodic gradient replay. The brain reconstructs the embeddings
it sees and forms cross-modal associations, so you can also ask it to *recall*
one modality from another.

Examples
--------
    # Start a live brain (uses real encoders if available, else offline fallback)
    python interact.py --modalities image text --grid-size 12

    # Force offline fallback encoders (no downloads)
    python interact.py --no-pretrained

REPL commands
-------------
    text: <some text>            perceive text
    image: <path>                perceive an image file
    audio: <path>                perceive an audio file
    video: <path>                perceive a video file
    pair image=<path> text=<...> perceive several modalities together
    recall image=<path>          drive with image, show recalled text/audio
    consolidate [N]              run N extra replay gradient steps (default 20)
    chat <message>               talk to the generative brain language model
    ask <message>                alias for chat
    stats                        show buffer size + recent loss
    save <path>                  save the brain checkpoint
    help                         show this help
    quit / exit                  leave

The `chat`/`ask` commands use a separately-trained generative language model
(trained_models/brain_lm.pt by default — see train_language.py). The first use
lazily loads it.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from typing import Dict, Optional

import numpy as np

from positronic_brain.online import LiveBrain, OnlineConfig


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def _parse_kv(tokens) -> Dict[str, str]:
    """Parse `key=value` tokens (value may be a quoted string)."""
    out: Dict[str, str] = {}
    for tok in tokens:
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _report(state: Dict) -> None:
    out = float(np.asarray(state["output"]).ravel()[0])
    loss = state.get("loss")
    mean_rate = float(np.asarray(state["rates"]).mean())
    print(f"  output={out:.4f}  mean_firing={mean_rate:.3f}  online_loss={_fmt(loss)}")


# Lazily-loaded generative language model (the talking brain).
_LM_CACHE: Dict[str, object] = {}


def _get_language_model(path: str = "trained_models/brain_lm.pt", device: str = "auto"):
    """Load the BrainLanguageModel + tokenizer once and cache it."""
    if "model" not in _LM_CACHE:
        from positronic_brain.language import BrainLanguageModel
        model, tok = BrainLanguageModel.load(path, device=device)
        model.eval()
        _LM_CACHE["model"] = model
        _LM_CACHE["tokenizer"] = tok
    return _LM_CACHE["model"], _LM_CACHE["tokenizer"]


def _chat(message: str, lm_path: str, device: str) -> None:
    """Generate a reply from the brain language model for one user message."""
    try:
        model, tok = _get_language_model(lm_path, device)
    except FileNotFoundError:
        print(f"  no language model at {lm_path!r}. Train one first:\n"
              f"    python train_language.py --steps 800 --out {lm_path}")
        return
    if tok is None:
        print("  language model has no tokenizer saved; retrain with train_language.py.")
        return
    prompt = f"User: {message}\nBrain:"
    reply = model.generate(tok, prompt=prompt, max_new_tokens=160,
                           temperature=0.8, top_k=30, stop="\nUser:")
    # keep only the brain's turn (cut at the next 'User:' if present)
    reply = reply.split("\nUser:")[0].strip()
    print(f"  Brain: {reply}")


def _recall(live: LiveBrain, **inputs) -> None:
    recon = live.recall(**inputs)
    driven = ", ".join(inputs.keys())
    print(f"  recalled from [{driven}]:")
    for m, vec in recon.items():
        v = np.asarray(vec).ravel()
        print(f"    {m:6s} -> embedding[{v.size}]  |v|={np.linalg.norm(v):.3f}  head5={np.round(v[:5], 3)}")


def run_repl(live: LiveBrain, lm_path: str = "trained_models/brain_lm.pt",
             device_name: str = "auto") -> None:
    print("\nPositronic Brain — live session. Type 'help' for commands, 'quit' to exit.")
    print(f"Brain: {live.brain.num_neurons} neurons, {live.brain.num_edges} synapses, "
          f"modalities={live.brain.modalities}, device={live.brain.device}\n")

    while True:
        try:
            line = input("brain> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return
        if not line:
            continue

        low = line.lower()
        if low in ("quit", "exit"):
            print("bye.")
            return
        if low == "help":
            print(__doc__)
            continue
        if low == "stats":
            lh = live.learner.loss_history
            print(f"  buffer={len(live.learner.buffer)}  observations={live.learner._obs_count}  "
                  f"last_loss={_fmt(lh[-1] if lh else None)}  rounds={len(lh)}")
            continue
        if low.startswith("consolidate"):
            parts = line.split()
            steps = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 20
            loss = live.learner.consolidate(steps=steps)
            print(f"  consolidated {steps} steps -> loss={_fmt(loss)}")
            continue
        if low.startswith(("chat", "ask")):
            msg = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else ""
            if not msg:
                print("  usage: chat <message>")
            else:
                _chat(msg, lm_path, device_name)
            continue
        if low.startswith("save "):
            path = line[5:].strip()
            live.save(path)
            print(f"  saved -> {path}")
            continue

        # modality commands
        try:
            if line.startswith(("text:", "image:", "audio:", "video:")):
                mod, val = line.split(":", 1)
                state = live.perceive(**{mod.strip(): val.strip()})
                _report(state)
            elif low.startswith("pair"):
                kv = _parse_kv(shlex.split(line)[1:])
                state = live.perceive(**kv)
                _report(state)
            elif low.startswith("recall"):
                kv = _parse_kv(shlex.split(line)[1:])
                _recall(live, **kv)
            else:
                print("  unknown command — type 'help'.")
        except FileNotFoundError as exc:
            print(f"  file not found: {exc}")
        except Exception as exc:  # keep the REPL alive
            print(f"  error: {type(exc).__name__}: {exc}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Interactive live-learning Positronic Brain.")
    ap.add_argument("--modalities", nargs="+", default=["image", "text", "audio"],
                    choices=["image", "text", "audio", "video"])
    ap.add_argument("--grid-size", type=int, default=12,
                    help="cube side; grid_size**3 neurons. 12->1728 (fast online "
                         "learning); up to 32->32,768 (max trainable on a 16 GB M1 Pro).")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="force offline deterministic fallback encoders.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--train-every", type=int, default=4)
    ap.add_argument("--train-steps", type=int, default=2)
    ap.add_argument("--load", default=None, help="optional checkpoint to start from.")
    ap.add_argument("--lm-path", default="trained_models/brain_lm.pt",
                    help="generative language model used by the 'chat'/'ask' command.")
    args = ap.parse_args(argv)

    online_cfg = OnlineConfig(lr=args.lr, train_every=args.train_every,
                              train_steps=args.train_steps)

    print("Building live brain (first real encode may download a model)...")
    live = LiveBrain.create(
        modalities=args.modalities,
        grid_size=args.grid_size,
        prefer_pretrained=not args.no_pretrained,
        online_config=online_cfg,
        device=args.device,
    )
    if args.load:
        from positronic_brain.model import PositronicBrain
        live.brain = PositronicBrain.load(args.load, device=args.device, strict=False)
        live.learner.brain = live.brain

    run_repl(live, lm_path=args.lm_path, device_name=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
