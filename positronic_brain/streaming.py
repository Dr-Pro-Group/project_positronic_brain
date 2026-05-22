"""
Stateful streaming batcher for persistent cross-window training.

The default trainer samples random, unrelated windows and resets the membrane
state ``V`` to rest at the start of every window — so the brain is trained as a
fixed-`seq_len` RNN and the project's headline claim ("the living membrane state
*is* the context/memory") is never actually trained. :class:`StreamingBatcher`
fixes that: it lays ``batch_size`` parallel **lanes** that walk *contiguously*
through the corpus, so the caller can carry ``V`` across optimiser steps
(truncated-BPTT *across* batches, detaching between steps) and only reset a lane's
state when that lane wraps past the end of its contiguous span.

This is the standard "stateful RNN" data layout (cf. char-RNN / TBPTT), here used
to give the recurrent dynamics a context far longer than one window while keeping
the activation memory of a single window.
"""

from __future__ import annotations

import torch


class StreamingBatcher:
    """Yield contiguous, lane-aligned windows for persistent-state training.

    The corpus is divided into ``batch_size`` equal contiguous lanes; lane ``b``
    keeps a cursor that advances by ``seq_len`` each step. Each call to
    :meth:`next_batch` returns a ``(batch_size, seq_len + 1)`` tensor and a boolean
    ``reset`` mask flagging the lanes that wrapped to their lane start this step
    (whose carried state the caller should reset to rest).
    """

    def __init__(self, data: torch.Tensor, seq_len: int, batch_size: int, device):
        self.data = data
        self.seq_len = int(seq_len)
        self.device = device
        n = int(data.numel())
        # Need at least one full window per lane; shrink batch if the corpus is tiny.
        max_lanes = max(1, n // (self.seq_len + 1))
        self.batch_size = max(1, min(int(batch_size), max_lanes))
        self.lane_len = n // self.batch_size
        self.starts = [b * self.lane_len for b in range(self.batch_size)]
        self.cursors = list(self.starts)

    def next_batch(self):
        chunks = []
        reset = []
        for b in range(self.batch_size):
            c = self.cursors[b]
            lane_end = self.starts[b] + self.lane_len
            if c + self.seq_len + 1 > min(lane_end, int(self.data.numel())):
                c = self.starts[b]            # wrap to the lane's beginning
                reset.append(True)
            else:
                reset.append(False)
            chunks.append(self.data[c : c + self.seq_len + 1])
            self.cursors[b] = c + self.seq_len
        batch = torch.stack(chunks).to(self.device)
        return batch, torch.tensor(reset, device=self.device, dtype=torch.bool)
