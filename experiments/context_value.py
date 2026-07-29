"""
How much is context actually worth on this task?

Every diagnostic in this investigation measures long-range temporal integration:
memory capacity, dimensional expansion, propagation reach, perturbation half-life.
They all say the same thing — the recurrent core retains almost nothing. But a
defect only counts if the task would have paid for the missing capability, and
next-character prediction may not. If knowing the previous twenty characters is
barely better than knowing the previous three, then a network with a one-character
memory is forgoing almost nothing, and the whole diagnostic frame is measuring a
property that carries no consequence here.

That question has an answer that does not require training anything. The empirical
conditional entropy H(c_t | c_{t-k..t-1}) is the information-theoretic floor on how
well ANY model can predict the next character given k of context, so its curve is
the most any amount of memory could possibly buy on this corpus.

Two cautions are built in. Estimating a conditional entropy from counts is biased
downward as the context grows, because longer contexts are rarer and eventually each
appears once, at which point the estimate reports zero uncertainty that is really
just memorisation of the sample. So each order is scored on HELD-OUT text with
backoff, and the count of unseen contexts is reported alongside — once that fraction
is large the estimate has stopped being trustworthy and the curve is annotated
rather than extrapolated.

    python experiments/context_value.py

Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from positronic_brain.corpus import load_corpus_splits


def conditional_bpc(train: str, test: str, order: int, alpha: float = 0.2,
                    vocab: int = None) -> Dict:
    """Held-out bits-per-character of an order-k context model with add-alpha backoff.

    Contexts never seen in training fall back to the unigram distribution, and the
    proportion that do is returned: it is the honest indicator of when a longer
    context has stopped being estimable rather than becoming more informative.
    """
    if vocab is None:
        vocab = len(set(train) | set(test))

    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: Dict[str, int] = defaultdict(int)
    uni: Dict[str, int] = defaultdict(int)

    for i in range(order, len(train)):
        ctx = train[i - order:i]
        ch = train[i]
        counts[ctx][ch] += 1
        totals[ctx] += 1
        uni[ch] += 1
    uni_total = sum(uni.values())

    total_bits, n, unseen = 0.0, 0, 0
    for i in range(order, len(test)):
        ctx = test[i - order:i]
        ch = test[i]
        if totals.get(ctx, 0) > 0:
            p = (counts[ctx].get(ch, 0) + alpha) / (totals[ctx] + alpha * vocab)
        else:
            unseen += 1
            p = (uni.get(ch, 0) + alpha) / (uni_total + alpha * vocab)
        total_bits += -math.log2(max(p, 1e-12))
        n += 1

    return {"order": order, "bpc": total_bits / max(n, 1),
            "unseen_context_frac": unseen / max(n, 1),
            "distinct_contexts": len(totals)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-chat", default="soda")
    p.add_argument("--hf-chat-limit", type=int, default=4000)
    p.add_argument("--orders", default="0,1,2,3,4,5,6,8,10,12,16,20")
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--max-train", type=int, default=2_000_000)
    p.add_argument("--max-test", type=int, default=200_000)
    p.add_argument("--json", default="runs/context_value.json")
    args = p.parse_args()

    tr, va, _ = load_corpus_splits(hf_chat=args.hf_chat, hf_chat_limit=args.hf_chat_limit,
                                   builtin=True, repeats=60, seed=42,
                                   val_frac=0.1, test_frac=0.0)
    train, test = tr[:args.max_train], va[:args.max_test]
    vocab = len(set(train) | set(test))
    print(f"train {len(train):,} chars · held-out {len(test):,} chars · vocab {vocab}\n")
    print(f"{'context':>8}{'held-out bpc':>15}{'gain vs prev':>14}"
          f"{'unseen ctx':>13}{'distinct ctx':>14}")
    print("-" * 66)

    rows: List[Dict] = []
    prev = None
    for order in [int(o) for o in args.orders.split(",")]:
        r = conditional_bpc(train, test, order, args.alpha, vocab)
        gain = "" if prev is None else f"{prev - r['bpc']:+.4f}"
        rows.append(r)
        print(f"{order:>8}{r['bpc']:>15.4f}{gain:>14}"
              f"{r['unseen_context_frac']:>12.1%}{r['distinct_contexts']:>14,}")
        prev = r["bpc"]

    best = min(rows, key=lambda r: r["bpc"])
    order0 = rows[0]["bpc"]
    # Where does the curve stop paying? The last order that buys more than 0.01 bpc.
    knee = None
    for a, b in zip(rows, rows[1:]):
        if (a["bpc"] - b["bpc"]) < 0.01 and knee is None:
            knee = a["order"]

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump({"args": vars(args), "vocab": vocab, "rows": rows,
                   "best_order": best["order"], "knee_order": knee}, fh, indent=2)

    print("\n" + "=" * 66)
    print(f"context-free (order 0): {order0:.4f} bpc")
    print(f"best estimable order:   {best['order']} at {best['bpc']:.4f} bpc "
          f"({order0 - best['bpc']:.4f} total gain)")
    if knee is not None:
        at_knee = next(r for r in rows if r["order"] == knee)
        print(f"gains fall below 0.01 bpc/char after order {knee} "
              f"({(order0 - at_knee['bpc']) / max(order0 - best['bpc'], 1e-9):.0%} of the "
              f"total gain already collected)")
    print("Rows where 'unseen ctx' is large are memorisation-limited, not evidence")
    print("that longer context stops helping.")
    print("=" * 66)
    print(f"[saved] {args.json}")


if __name__ == "__main__":
    main()
