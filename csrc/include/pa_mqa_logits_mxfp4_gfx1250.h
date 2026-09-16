// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// MXFP4 paged MQA logits on gfx1250 (OPUS, wave32 / WMMA 32x16x128 / TDM), PREFILL ONLY.
// Decode is not implemented; the ABI already carries its fields, so adding it needs no ABI
// change.
//
// Shares `opus_mqa_logits_kargs` with the gfx950 op byte for byte, and differs in the layouts
// it reads through it:
//
//     q         [total_q, H, D/2]       u8    natural
//     q_scale   [total_q, H, 4]         u8    natural   (gfx950: [T, 2, 32, 4])
//     kv_cache  [num_blocks, PAGE, 64]  u8    natural   (gfx950: [nb, 4, PAGE, 16])
//     kv_scale  [num_blocks, PAGE, 4]   u8    natural   (gfx950: [nb, 2, 32, 4])
//     weights   [total_q, H]            bf16  natural
//     out       [total_q, max_seq_len]  fp32
//
// HANDING THIS KERNEL THE gfx950 ARRAYS IS SILENT: every fp4 scale layout has the same byte
// count, so the shape checks in the launcher catch a wrong array and never a wrong permutation.
// Only a random-data comparison against a dequantized reference does -- see
// op_tests/test_pa_mqa_logits_gfx1250_opus.py.
//
// There is no `block_k` argument: the KV tile is fixed at 128 (two pages) and the CTA width
// comes from Q_PER_BLOCK, so the knob would have one legal value. `block_tables` must be sized
// for 128 all the same, because a CTA rounds its window up to a whole tile.
#pragma once
#include "aiter_tensor.h"
#include <cstdint>

// Prefill. ONE CTA PER GROUP of Q_PER_BLOCK query rows sharing a KV window, so the HBM->LDS
// traffic is paid once per group instead of once per row; grid.x == `num_groups`.
//
// THE THREE THINGS THE CALLER OWES, none of which the kernel can check. Broken, it does not
// give a wrong answer: the waves of a CTA disagree about the trip count and DEADLOCK on the
// phase barrier.
//   1. A GROUP IS CONTIGUOUS ROWS OF ONE BATCH, 1..Q_PER_BLOCK of them. Group g covers rows
//      [group_starts[g], group_starts[g+1]). Build it with
//      `pa_mqa_logits_mxfp4_gfx1250_prefill_groups` and this holds by construction.
//   2. THE WINDOW RULE IS NON-DECREASING IN THE ROW INDEX within a group. The loop bound is the
//      group's UNION window, taken as the first row's start and the last row's end, which is
//      only the union when the rule is monotone. Every causal and CSA-compressed rule is.
//   3. THE STORE IS BOUNDED BY THE WINDOW, NOT BY `max_seq_len`. A `local_ends` entry past
//      out.size(1) writes past the row; `max_seq_len` rides in kargs and is never read, only
//      `stride_out_row` is. The launcher's shape checks are the only guard.
//
// `num_groups` may exceed the real group count: a group whose start equals its end returns on a
// CTA-uniform condition before touching memory. That is what lets grid.x come from the static
// shapes, keeping the launch schedule-free and cudagraph-safe.
void pa_mqa_logits_mxfp4_gfx1250_fwd_prefill(aiter_tensor_t& q,
                                             aiter_tensor_t& q_scale,
                                             aiter_tensor_t& kv_cache,
                                             aiter_tensor_t& kv_scale,
                                             aiter_tensor_t& block_tables,
                                             aiter_tensor_t& weights,
                                             aiter_tensor_t& row_to_batch,
                                             aiter_tensor_t& local_starts,
                                             aiter_tensor_t& local_ends,
                                             aiter_tensor_t& group_starts,
                                             aiter_tensor_t& out,
                                             int num_rows,
                                             int num_groups,
                                             float weight_scale,
                                             int kv_block_size,
                                             int max_seq_len);

// Per-row `[local_start, local_end)` windows, MTP tail-causal: batch b's n-th query token sees
// `[0, context_lens[b] - (qlen - 1 - n))`. Device-side, no host sync.
//
// THE ONLY RULE THIS EXPRESSES, and not the one a CSA-compressed cache follows -- row n there
// sees `floor((pos + 1) / R)`, and `floor((x - d) / R) != floor(x / R) - d`. Such a caller
// builds `local_ends` itself; the arrays carry no constraint beyond `0 <= start <= end` and
// condition 2 above.
void pa_mqa_logits_mxfp4_gfx1250_prefill_windows(aiter_tensor_t& cu_seq_q,
                                                 aiter_tensor_t& context_lens,
                                                 aiter_tensor_t& row_to_batch,
                                                 aiter_tensor_t& local_starts,
                                                 aiter_tensor_t& local_ends,
                                                 int total_q);

// The qshare group boundaries, from the BATCH `cu_seq_q` alone: batch b's rows split into
// `ceil(qlen_b / Q_PER_BLOCK)` groups, which satisfies condition 1 by construction.
// Device-side, no host sync.
//
// `group_starts` holds `max_groups + 1` entries and `max_groups` must be at least the real
// count; entries past it are written equal so the extra CTAs return immediately. The bound the
// Python wrapper uses is `(total_q + batch * (QPB - 1)) / QPB`, which is never short because a
// sum of floors is at most the floor of the sum.
void pa_mqa_logits_mxfp4_gfx1250_prefill_groups(aiter_tensor_t& cu_seq_q,
                                                aiter_tensor_t& group_starts,
                                                int total_q,
                                                int max_groups);

#ifdef PA_MQA_LOGITS_MXFP4_GFX1250_IMPL

#include "pa_mqa_logits_mxfp4_gfx1250_traits.h" // pulls in the _defs ABI

// A check rather than a comment, because a JSON build config cannot carry the reason:
// `-ffast-math` implies `-ffinite-math-only`, which folds OPUS_LOGITS_RELU's IEEE maximum back
// to a NaN-swallowing select. See the _defs header for what that costs.
#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ && OPUS_LOGITS_IEEE_RELU
#error "build this TU with -fno-finite-math-only: -ffinite-math-only folds OPUS_LOGITS_RELU's \
IEEE maximum back to a compare-and-select, which SWALLOWS a NaN E8M0 scale instead of \
propagating it. See optCompilerConfig.json's flags_extra_hip for this module."
#endif

// The device pass on gfx1250 gets the real kernel; every other pass gets an empty stub so the
// launcher's `__device_stub__` reference still resolves.
//
// BOTH HALVES OF THE GUARD ARE LOAD-BEARING: the template names opus's gfx1250-only device API,
// and `opus::get_warp_size()` answers 64 in the host pass, which would silently build the
// wave64 fragment layout with every byte count still matching.
#if !defined(__HIP_DEVICE_COMPILE__) || !defined(__gfx1250__)
namespace opus_logits {
namespace qshare {
template <class T, mqa_logits_sched SCHED = mqa_logits_sched::Prefill>
__global__ void mqa_logits_mxfp4_32x16x128_qshare_kernel(opus_mqa_logits_kargs)
{
}
} // namespace qshare
} // namespace opus_logits
#else
#include "pa_mqa_logits_mxfp4_gfx1250_kernel.h"
#endif

#endif // PA_MQA_LOGITS_MXFP4_GFX1250_IMPL
