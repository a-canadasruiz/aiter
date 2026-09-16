# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP4 paged MQA logits for DeepSeek-style sparse attention on gfx1250 (OPUS kernel).

Per query row ``r`` over a window ``[s, e)``:
``out[r, s:e] = sum_H( relu(Q[r] . K^T) * weight[r] ) * weight_scale``

PREFILL ONLY. Decode is not implemented; the ABI already carries its fields.

ALL FIVE INPUTS TAKE THEIR NATURAL LAYOUT, unlike the gfx950 sibling, which adopted FlyDSL's
preshuffled forms for three of them:

===============  ===========================================  ==================
tensor           shape                                        layout
===============  ===========================================  ==================
``q``            ``[total_q, H, D/2]`` uint8                   natural
``q_scale``      ``[total_q, H, 4]`` uint8                     natural
``kv_cache``     ``[num_blocks, PAGE, D/2]`` uint8             natural
``kv_scale``     ``[num_blocks, PAGE, 4]`` uint8               natural
``weights``      ``[total_q, H]`` bfloat16                     natural
===============  ===========================================  ==================

``q_scale[t, h, b]`` and ``kv_scale[blk, o, b]`` are the plain E8M0 byte for 32-element K block
``b`` of that row. ``q[t, h]`` and ``kv_cache[blk, o]`` are the 64 packed bytes of that row's
128 fp4 elements, low nibble first.

**Passing the gfx950 op's arrays here is SILENT.** Every fp4 scale layout has the same byte
count, so the C++ size checks cannot tell them apart and the result is plausible-looking wrong
logits. Validate against a dequantized reference on RANDOM data -- uniform data passes under
any permutation of K.

ONE CTA SERVES A GROUP OF ``Q_PER_BLOCK`` QUERY ROWS that share the KV window, so the HBM->LDS
traffic is paid once per group instead of once per row. It costs one extra array,
``group_starts``, and puts three conditions on the input that the kernel cannot check and does
not survive -- a CTA whose waves disagree about the trip count DEADLOCKS on the phase barrier
rather than returning a wrong answer:

1. a group is contiguous rows of one batch (:func:`compute_prefill_groups` guarantees it);
2. the window rule is NON-DECREASING in the row index within a group, which every causal and
   CSA-compressed rule is, and :func:`assert_qshare_windows` checks on demand;
3. the store is bounded by the WINDOW, so a ``local_ends`` entry past ``out.shape[1]`` writes
   past the row.

Both builders are device-side and the launch grid comes from the static shapes, so the path is
schedule-free and cudagraph-safe. Both are PER-FORWARD quantities while the kernel runs PER
LAYER: build them once per forward and pass them in.
"""

import torch

from ...jit.core import compile_ops
from ...jit.utils.chip_info import get_gfx_runtime

MD_NAME_MXFP4_GFX1250 = "module_pa_mqa_logits_mxfp4_gfx1250_opus"

DEFAULT_HEADS = 64
DEFAULT_HEAD_DIM = 128
DEFAULT_KV_BLOCK_SIZE = 64

# Query rows per CTA == waves per CTA; the group size the builders and the kernel agree on.
Q_PER_BLOCK = 4

# The KV tile in tokens. Not an argument, unlike the gfx950 op's ``block_k``: this target
# compiles one variant. Exported because ``block_tables`` must be sized for it -- a CTA rounds
# its window up to a whole tile and indexes the table there.
BLOCK_K = 128


# ── JIT stubs: signatures must match PA_MQA_LOGITS_MXFP4_GFX1250_PYBIND exactly ───────────────
@compile_ops(MD_NAME_MXFP4_GFX1250, develop=True)
def pa_mqa_logits_mxfp4_gfx1250_fwd_prefill(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    group_starts: torch.Tensor,
    out: torch.Tensor,
    num_rows: int,
    num_groups: int,
    weight_scale: float,
    kv_block_size: int,
    max_seq_len: int,
) -> None: ...


@compile_ops(MD_NAME_MXFP4_GFX1250, develop=True)
def pa_mqa_logits_mxfp4_gfx1250_prefill_windows(
    cu_seq_q: torch.Tensor,
    context_lens: torch.Tensor,
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    total_q: int,
) -> None: ...


@compile_ops(MD_NAME_MXFP4_GFX1250, develop=True)
def pa_mqa_logits_mxfp4_gfx1250_prefill_groups(
    cu_seq_q: torch.Tensor,
    group_starts: torch.Tensor,
    total_q: int,
    max_groups: int,
) -> None: ...


def max_groups_for(total_q: int, batch: int) -> int:
    """The grid width, from the static shapes alone -- which is what keeps the launch
    schedule-free.

    The real count is ``sum_b ceil(qlen_b / Q_PER_BLOCK)`` and depends on device data. This sums
    the per-batch roundings before the divide, so it is never short and is exact whenever they
    tile. The slack is at most ``batch - 1`` CTAs, each returning on its first instruction.
    """
    return (int(total_q) + int(batch) * (Q_PER_BLOCK - 1)) // Q_PER_BLOCK


def compute_prefill_windows(
    cu_seq_q: torch.Tensor,
    context_lens: torch.Tensor,
    total_q: int,
    out: tuple | None = None,
):
    """Build the per-row ``[local_start, local_end)`` window arrays, device-side.

    MTP tail-causal: batch ``b``'s ``n``-th row sees
    ``[0, context_lens[b] - (qlen - 1 - n))``, which reduces to plain causal when
    ``qlen == ctx``. That is the ONLY rule this expresses, and not the one a CSA-compressed
    cache follows -- row ``n`` there sees ``floor((pos + 1) / R)``, and
    ``floor((x - d) / R) != floor(x / R) - d``. Such a caller builds ``local_ends`` itself.
    """
    dev = cu_seq_q.device
    cu = cu_seq_q.to(torch.int32).contiguous()
    ctx = context_lens.to(torch.int32).contiguous()
    if out is None:
        row_to_batch = torch.empty(total_q, dtype=torch.int32, device=dev)
        local_starts = torch.empty(total_q, dtype=torch.int32, device=dev)
        local_ends = torch.empty(total_q, dtype=torch.int32, device=dev)
    else:
        row_to_batch, local_starts, local_ends = out
    pa_mqa_logits_mxfp4_gfx1250_prefill_windows(
        cu, ctx, row_to_batch, local_starts, local_ends, int(total_q)
    )
    return row_to_batch, local_starts, local_ends


def compute_prefill_groups(
    cu_seq_q: torch.Tensor,
    total_q: int,
    out: torch.Tensor | None = None,
):
    """Build the qshare group boundaries from the BATCH ``cu_seq_q``, device-side.

    Returns ``(group_starts, num_groups)``, where group ``g`` covers rows
    ``[group_starts[g], group_starts[g + 1])``. ``num_groups`` is :func:`max_groups_for`, an
    upper bound rather than the exact count, so it stays a host int and no device read is needed
    to launch; groups past the real count are written empty.
    """
    cu = cu_seq_q.to(torch.int32).contiguous()
    batch = int(cu.shape[0]) - 1
    num_groups = max_groups_for(total_q, batch)
    if out is None:
        group_starts = torch.empty(num_groups + 1, dtype=torch.int32, device=cu.device)
    else:
        group_starts = out
    pa_mqa_logits_mxfp4_gfx1250_prefill_groups(
        cu, group_starts, int(total_q), int(num_groups)
    )
    return group_starts, num_groups


def assert_qshare_windows(
    group_starts, num_groups, row_to_batch, local_starts, local_ends
):
    """Check conditions 1 and 2 of the module docstring: a group stays inside one batch and its
    window rule is non-decreasing.

    HOST-SIDE AND SYNCHRONISING -- a debug/test helper, not something for a hot path. It exists
    because breaking either condition DEADLOCKS the CTA rather than returning a wrong answer.
    """
    gs = group_starts[: num_groups + 1].tolist()
    rb = row_to_batch.tolist()
    ls = local_starts.tolist()
    le = local_ends.tolist()
    for g in range(num_groups):
        lo, hi = gs[g], gs[g + 1]
        if hi == lo:
            continue  # a padding group; its CTA returns before reading anything
        if not (0 < hi - lo <= Q_PER_BLOCK):
            raise AssertionError(
                f"qshare: group {g} spans rows [{lo},{hi}), which is not 1..{Q_PER_BLOCK} rows"
            )
        for r in range(lo + 1, hi):
            if rb[r] != rb[lo]:
                raise AssertionError(
                    f"qshare: group {g} rows [{lo},{hi}) straddles batches "
                    f"{rb[lo]} and {rb[r]}"
                )
            if ls[r] < ls[r - 1] or le[r] < le[r - 1]:
                raise AssertionError(
                    f"qshare: group {g} window is not non-decreasing (row {r - 1}: "
                    f"[{ls[r - 1]},{le[r - 1]}) then row {r}: [{ls[r]},{le[r]}))"
                )


def _require_gfx1250(name):
    gfx = get_gfx_runtime()
    if gfx != "gfx1250":
        raise RuntimeError(f"{name} requires gfx1250, got {gfx}")


def pa_mqa_logits_mxfp4_gfx1250_prefill(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    max_seq_len: int,
    *,
    cu_seq_q: torch.Tensor | None = None,
    groups: tuple | None = None,
    weight_scale: float = 1.0,
    kv_block_size: int = DEFAULT_KV_BLOCK_SIZE,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ragged-prefill paged MQA logits (gfx1250): one CTA per group of ``Q_PER_BLOCK`` query
    rows sharing a KV window, each covering its whole ``[local_start, local_end)``.

    See the module docstring for the input layouts -- in particular that all of them are NATURAL
    and that the gfx950 op's permuted scales are accepted silently -- and for the three
    conditions a qshare caller owes.

    ``groups`` is ``(group_starts, num_groups)`` from :func:`compute_prefill_groups`. Pass it:
    it is a per-forward quantity and the kernel runs per layer, so leaving it ``None`` rebuilds
    it on every call and a profiler that sums CUDA events in the region charges that builder to
    this kernel. ``cu_seq_q`` (the BATCH boundaries, length ``batch + 1``) is read only when
    ``groups`` is None.

    ``block_tables`` must be sized for ``BLOCK_K`` = 128, not for ``kv_block_size``. A reused
    ``out`` must be pre-filled with -inf, since the kernel only writes in-window cells.
    """
    _require_gfx1250("pa_mqa_logits_mxfp4_gfx1250")
    total_rows = int(q_fp4.shape[0])
    if groups is None:
        if cu_seq_q is None:
            raise ValueError(
                "pass either `groups` from compute_prefill_groups (preferred: it is a "
                "per-forward quantity and this kernel runs per layer) or `cu_seq_q`, the "
                "batch boundaries, to build them here"
            )
        groups = compute_prefill_groups(cu_seq_q, total_rows)
    group_starts, num_groups = groups

    if out is None:
        out = torch.full(
            (total_rows, max_seq_len),
            float("-inf"),
            dtype=torch.float32,
            device=q_fp4.device,
        )
    pa_mqa_logits_mxfp4_gfx1250_fwd_prefill(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        row_to_batch.to(torch.int32).contiguous(),
        local_starts.to(torch.int32).contiguous(),
        local_ends.to(torch.int32).contiguous(),
        group_starts,
        out,
        total_rows,
        int(num_groups),
        float(weight_scale),
        int(kv_block_size),
        int(max_seq_len),
    )
    return out


__all__ = [
    "BLOCK_K",
    "Q_PER_BLOCK",
    "assert_qshare_windows",
    "compute_prefill_groups",
    "compute_prefill_windows",
    "max_groups_for",
    "pa_mqa_logits_mxfp4_gfx1250_fwd_prefill",
    "pa_mqa_logits_mxfp4_gfx1250_prefill",
    "pa_mqa_logits_mxfp4_gfx1250_prefill_groups",
    "pa_mqa_logits_mxfp4_gfx1250_prefill_windows",
]
