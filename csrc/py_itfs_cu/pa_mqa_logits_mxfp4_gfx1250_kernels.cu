// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// MXFP4 paged MQA logits (gfx1250) -- host launcher plus the two device-side builders the
// caller needs. The input ABI and the three conditions a qshare caller owes are documented in
// pa_mqa_logits_mxfp4_gfx1250.h.

#define PA_MQA_LOGITS_MXFP4_GFX1250_IMPL
#include "pa_mqa_logits_mxfp4_gfx1250.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"

// The single compiled config: 4 query rows per CTA (4 waves of 32), KV tile 128 = 2 pages,
// D = 128, H = 64, page 64.
using mqa_logits_fp4_gfx1250_traits = logits_fp4_qshare_traits_4q;

// aiter ships the NATURAL kv_cache only. The shuffled branch exists in the traits for an A/B
// against a FlyDSL-shaped cache, and asserting here means a stray -D cannot build a launcher
// whose shape checks describe the other layout.
static_assert(mqa_logits_fp4_gfx1250_traits::KV_LAYOUT == OPUS_LOGITS_KV_NATURAL,
              "the aiter launcher validates the NATURAL kv_cache layout");

// Every input is strided by a COMPILE-TIME constant in the kernel and no runtime stride is ever
// read, so a padded or permuted input is silently wrong rather than rejected -- hence the
// layout is required, not adapted to. `out` is the one exception: its row stride is passed
// through as stride_out_row.
//
// The two scale arrays get a byte-count check because they are the only buffers whose layout is
// specific to this matrix instruction. It catches a wrong ARRAY, never a wrong PERMUTATION:
// every fp4 scale layout has the same byte count, gfx950's included.
template <class Traits>
static void pa_mqa_logits_mxfp4_gfx1250_check_shapes(aiter_tensor_t& q,
                                                     aiter_tensor_t& q_scale,
                                                     aiter_tensor_t& kv_cache,
                                                     aiter_tensor_t& kv_scale,
                                                     aiter_tensor_t& block_tables,
                                                     aiter_tensor_t& weights,
                                                     aiter_tensor_t& out,
                                                     int kv_block_size,
                                                     int max_seq_len)
{
    AITER_CHECK(q.dim() == 3, "q must be 3-D [T, H, D/2], got ndim=", q.dim());
    AITER_CHECK(weights.dim() == 2, "weights must be 2-D [T, H], got ndim=", weights.dim());
    AITER_CHECK(block_tables.dim() == 2, "block_tables must be 2-D [batch, max_blocks_per_seq]");
    AITER_CHECK(out.dim() == 2, "out must be 2-D [T, max_seq_len], got ndim=", out.dim());
    AITER_CHECK(weights.size(0) >= q.size(0),
                "weights is per query row; need at least ",
                q.size(0),
                " rows, got ",
                weights.size(0));
    // The kernel bounds its store by the WINDOW, not by max_seq_len (header, condition 3), so
    // these two are the only place an undersized `out` can be caught at all.
    AITER_CHECK(out.size(0) >= q.size(0),
                "out is [T, max_seq_len]; need at least ",
                q.size(0),
                " rows, got ",
                out.size(0));
    AITER_CHECK(out.size(1) >= max_seq_len,
                "out is [T, max_seq_len]; need at least ",
                max_seq_len,
                " columns, got ",
                out.size(1));

    constexpr int HEAD_BYTES = Traits::HEAD_DIM * Traits::ELEM_BITS / 8; // 64: D/2 per head
    const int H              = static_cast<int>(q.size(1));
    const int D_BYTES        = static_cast<int>(q.size(2));
    AITER_CHECK(H == Traits::N_HEADS, "compiled for H=", (int)Traits::N_HEADS, ", got H=", H);
    AITER_CHECK(D_BYTES == HEAD_BYTES,
                "q last dim is D/2 packed bytes; compiled for ",
                HEAD_BYTES,
                " (D=",
                (int)Traits::HEAD_DIM,
                "), got ",
                D_BYTES);
    static_assert(Traits::N_HEADS * HEAD_BYTES == Traits::Q_ROW_BYTES,
                  "the kernel strides q rows by Q_ROW_BYTES; it must equal H * D/2");
    AITER_CHECK(kv_block_size == Traits::PAGE_SIZE,
                "compiled for kv_block_size=",
                (int)Traits::PAGE_SIZE,
                ", got ",
                kv_block_size);

    AITER_CHECK(q.dtype() == AITER_DTYPE_fp4x2 || q.dtype() == AITER_DTYPE_u8,
                "q must be fp4x2 (E2M1, 2/byte) or u8 bytes");
    AITER_CHECK(kv_cache.dtype() == AITER_DTYPE_fp4x2 || kv_cache.dtype() == AITER_DTYPE_u8,
                "kv_cache must be fp4x2 (E2M1, 2/byte) or u8 bytes");
    AITER_CHECK(q_scale.dtype() == AITER_DTYPE_u8 && kv_scale.dtype() == AITER_DTYPE_u8,
                "q_scale / kv_scale are E8M0 bytes and must be u8");
    AITER_CHECK(weights.dtype() == AITER_DTYPE_bf16, "weights must be bf16");
    AITER_CHECK(block_tables.dtype() == AITER_DTYPE_i32, "block_tables must be int32");
    AITER_CHECK(out.dtype() == AITER_DTYPE_fp32, "out must be fp32");

    AITER_CHECK(q.is_contiguous() && q_scale.is_contiguous() && kv_cache.is_contiguous() &&
                    kv_scale.is_contiguous() && block_tables.is_contiguous() &&
                    weights.is_contiguous(),
                "q / q_scale / kv_cache / kv_scale / block_tables / weights must be contiguous");
    AITER_CHECK(out.stride(1) == 1, "out must be contiguous along its last dim");

    AITER_CHECK(q_scale.numel() == (int64_t)q.size(0) * Traits::QS_ROW_BYTES,
                "q_scale is natural [T, H, ",
                (int)Traits::SCALE_BYTES_PER_DWORD,
                "] and must hold ",
                (int)Traits::QS_ROW_BYTES,
                " bytes per query row, got numel=",
                q_scale.numel(),
                " for T=",
                q.size(0));
    AITER_CHECK(kv_cache.numel() % Traits::KV_PAGE_BYTES == 0,
                "kv_cache must be a whole number of ",
                (int)Traits::KV_PAGE_BYTES,
                "-byte pages, got numel=",
                kv_cache.numel());
    const int64_t num_blocks = kv_cache.numel() / Traits::KV_PAGE_BYTES;
    AITER_CHECK(kv_scale.numel() == num_blocks * Traits::KVS_PAGE_BYTES,
                "kv_scale is natural [num_blocks, PAGE, ",
                (int)Traits::SCALE_BYTES_PER_DWORD,
                "] and must hold ",
                (int)Traits::KVS_PAGE_BYTES,
                " bytes per page, got numel=",
                kv_scale.numel(),
                " for num_blocks=",
                num_blocks);
}

template <class Traits>
static void pa_mqa_logits_mxfp4_gfx1250_launch_prefill(aiter_tensor_t& q,
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
                                                       int max_seq_len)
{
    pa_mqa_logits_mxfp4_gfx1250_check_shapes<Traits>(
        q, q_scale, kv_cache, kv_scale, block_tables, weights, out, kv_block_size, max_seq_len);
    AITER_CHECK(
        row_to_batch.dtype() == AITER_DTYPE_i32 && local_starts.dtype() == AITER_DTYPE_i32 &&
            local_ends.dtype() == AITER_DTYPE_i32 && group_starts.dtype() == AITER_DTYPE_i32,
        "row_to_batch / local_starts / local_ends / group_starts must be int32");
    AITER_CHECK(row_to_batch.is_contiguous() && local_starts.is_contiguous() &&
                    local_ends.is_contiguous() && group_starts.is_contiguous(),
                "row_to_batch / local_starts / local_ends / group_starts must be contiguous");
    AITER_CHECK(num_rows <= q.size(0),
                "num_rows exceeds the query rows in q: ",
                num_rows,
                " > ",
                q.size(0));
    AITER_CHECK(static_cast<int64_t>(row_to_batch.numel()) >= num_rows &&
                    static_cast<int64_t>(local_starts.numel()) >= num_rows &&
                    static_cast<int64_t>(local_ends.numel()) >= num_rows,
                "row_to_batch / local_starts / local_ends are per query row; need at least ",
                num_rows,
                " entries each, got ",
                row_to_batch.numel(),
                " / ",
                local_starts.numel(),
                " / ",
                local_ends.numel());
    // Group g reads BOTH group_starts[g] and group_starts[g + 1], so the array is one longer
    // than the grid. Under-sized, a CTA reads a neighbouring allocation and walks whatever row
    // range that dword implies -- in bounds for the hardware, wrong for the answer.
    AITER_CHECK(static_cast<int64_t>(group_starts.numel()) >= (int64_t)num_groups + 1,
                "group_starts holds one boundary per group PLUS a terminator; need at least ",
                num_groups + 1,
                " entries, got ",
                group_starts.numel());

    if(num_rows <= 0 || num_groups <= 0)
        return;

    opus_mqa_logits_kargs kargs{};
    kargs.ptr_q            = q.data_ptr();
    kargs.ptr_q_scale      = q_scale.data_ptr();
    kargs.ptr_kv           = kv_cache.data_ptr();
    kargs.ptr_kv_scale     = kv_scale.data_ptr();
    kargs.ptr_block_tables = reinterpret_cast<const int*>(block_tables.data_ptr());
    kargs.ptr_weights      = weights.data_ptr();
    kargs.ptr_out          = reinterpret_cast<float*>(out.data_ptr());
    kargs.ptr_row_to_batch = reinterpret_cast<const int*>(row_to_batch.data_ptr());
    kargs.ptr_local_starts = reinterpret_cast<const int*>(local_starts.data_ptr());
    kargs.ptr_local_ends   = reinterpret_cast<const int*>(local_ends.data_ptr());
    // The kernel's `cu_seq_q` is the qshare GROUP boundary array, not the batch one.
    kargs.ptr_cu_seq_q       = reinterpret_cast<const int*>(group_starts.data_ptr());
    kargs.num_rows           = num_rows;
    kargs.max_seq_len        = max_seq_len;
    kargs.stride_out_row     = static_cast<int>(out.stride(0));
    kargs.weight_scale       = weight_scale;
    kargs.block_k            = Traits::KV_TILE_SIZE;
    kargs.kv_block_size      = kv_block_size;
    kargs.max_blocks_per_seq = static_cast<int>(block_tables.size(1));

    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    dim3 grid(static_cast<unsigned>(num_groups)); // one CTA per qshare group
    dim3 block(Traits::BLOCK_SIZE);
    opus_logits::qshare::
        mqa_logits_mxfp4_32x16x128_qshare_kernel<Traits, opus_logits::mqa_logits_sched::Prefill>
        <<<grid, block, 0, stream>>>(kargs);
    HIP_CALL_LAUNCH(hipGetLastError());
}

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
                                             int max_seq_len)
{
    // pybind path: make the shape checks throw a Python RuntimeError instead of abort()ing the
    // interpreter. Same convention as opus_gemm.cu / gradlib.
    aiter_detail::g_aiter_can_throw = true;
    pa_mqa_logits_mxfp4_gfx1250_launch_prefill<mqa_logits_fp4_gfx1250_traits>(q,
                                                                              q_scale,
                                                                              kv_cache,
                                                                              kv_scale,
                                                                              block_tables,
                                                                              weights,
                                                                              row_to_batch,
                                                                              local_starts,
                                                                              local_ends,
                                                                              group_starts,
                                                                              out,
                                                                              num_rows,
                                                                              num_groups,
                                                                              weight_scale,
                                                                              kv_block_size,
                                                                              max_seq_len);
}

// Both builders take and return caller-allocated device buffers: no hipMalloc, no host<->device
// sync, and a grid that is a function of the static shapes. Both are PER-FORWARD quantities
// while the kernel runs PER LAYER.
namespace {

constexpr int WINDOW_BUILD_BLOCK = 256;

// MTP tail-causal: batch b's n-th query token (n in [0, qlen)) sees
// [0, context_len[b] - (qlen - 1 - n)); rows past cu[B] get an empty window.
__global__ void mqa_logits_fp4_gfx1250_prefill_windows_kernel(const int* __restrict__ cu,
                                                              const int* __restrict__ ctx,
                                                              int* __restrict__ row_to_batch,
                                                              int* __restrict__ local_starts,
                                                              int* __restrict__ local_ends,
                                                              int total_q,
                                                              int B)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if(r >= total_q)
        return;

    // searchsorted(cu[1:], r, right=True) = count(cu[1..B] <= r).
    int lo = 0, hi = B;
    while(lo < hi)
    {
        int mid    = (lo + hi) >> 1;
        int cu_mid = cu[1 + (mid < (B - 1) ? mid : (B - 1))];
        if(cu_mid <= r)
            lo = mid + 1;
        else
            hi = mid;
    }
    const int b = (lo < (B - 1)) ? lo : (B - 1);

    const int cu_b  = cu[b];
    const int cu_b1 = cu[b + 1];
    const int ctx_b = ctx[b];
    const int n     = r - cu_b;
    const int qlen  = cu_b1 - cu_b;
    int le          = ctx_b - qlen + n + 1;
    le              = le > 0 ? le : 0;
    // Rows beyond the real total are flat tail-padding -> empty window.
    const int real_total = cu[B];
    if(r >= real_total)
        le = 0;

    row_to_batch[r] = b;
    local_starts[r] = 0;
    local_ends[r]   = le;
}

// ONE workgroup, because the group offsets are a prefix sum over batches and a second kernel
// would need a second buffer for it. B is a batch count and this runs once per forward.
constexpr int GROUPS_BUILD_BLOCK     = 256;
constexpr int GROUPS_BUILD_MAX_BATCH = 2048; // 8 KB of LDS for the offsets

__global__ void mqa_logits_fp4_gfx1250_prefill_groups_kernel(
    const int* __restrict__ cu, int* __restrict__ group_starts, int B, int max_groups, int qpb)
{
    __shared__ int g_off[GROUPS_BUILD_MAX_BATCH + 1];

    for(int b = threadIdx.x; b < B; b += blockDim.x)
        g_off[b] = (cu[b + 1] - cu[b] + qpb - 1) / qpb;
    __syncthreads();

    // Exclusive scan, in place, serial on one thread: B <= 2048 and this is a per-forward call,
    // so the scan is not worth the correctness surface of a parallel one.
    if(threadIdx.x == 0)
    {
        int acc = 0;
        for(int b = 0; b < B; ++b)
        {
            const int c = g_off[b];
            g_off[b]    = acc;
            acc += c;
        }
        g_off[B] = acc;
    }
    __syncthreads();

    const int num_groups = g_off[B];
    const int end_row    = cu[B];

    // Group g's END needs no write of its own: the groups tile the rows in order, so it is
    // group_starts[g + 1] -- a batch's last group ends at cu[b + 1], which is where the next
    // batch's first group starts.
    //
    // Groups from num_groups to max_groups inclusive are written end_row, so each is empty and
    // its CTA returns on `group_end <= group_row`. That is what lets grid.x be a static shape.
    for(int g = threadIdx.x; g <= max_groups; g += blockDim.x)
    {
        if(g >= num_groups)
        {
            group_starts[g] = end_row;
            continue;
        }
        // The LARGEST b with g_off[b] <= g, which is what makes an EMPTY batch skip itself: it
        // has g_off[b] == g_off[b + 1] and loses the tie to b + 1.
        int lo = 0, hi = B - 1;
        while(lo < hi)
        {
            const int mid = (lo + hi + 1) >> 1;
            if(g_off[mid] <= g)
                lo = mid;
            else
                hi = mid - 1;
        }
        group_starts[g] = cu[lo] + (g - g_off[lo]) * qpb;
    }
}

} // namespace

void pa_mqa_logits_mxfp4_gfx1250_prefill_windows(aiter_tensor_t& cu_seq_q,
                                                 aiter_tensor_t& context_lens,
                                                 aiter_tensor_t& row_to_batch,
                                                 aiter_tensor_t& local_starts,
                                                 aiter_tensor_t& local_ends,
                                                 int total_q)
{
    aiter_detail::g_aiter_can_throw = true;
    const int B                     = static_cast<int>(context_lens.size(0));
    AITER_CHECK(cu_seq_q.dtype() == AITER_DTYPE_i32 && context_lens.dtype() == AITER_DTYPE_i32,
                "cu_seq_q / context_lens must be int32");
    AITER_CHECK(cu_seq_q.is_contiguous() && context_lens.is_contiguous(),
                "cu_seq_q / context_lens must be contiguous");
    AITER_CHECK(cu_seq_q.size(0) == B + 1, "cu_seq_q must have length B+1");
    AITER_CHECK(row_to_batch.dtype() == AITER_DTYPE_i32 &&
                    local_starts.dtype() == AITER_DTYPE_i32 &&
                    local_ends.dtype() == AITER_DTYPE_i32,
                "row_to_batch / local_starts / local_ends must be int32");
    AITER_CHECK(row_to_batch.is_contiguous() && local_starts.is_contiguous() &&
                    local_ends.is_contiguous(),
                "row_to_batch / local_starts / local_ends must be contiguous");
    // The kernel writes all three arrays at every r < total_q, unconditionally.
    AITER_CHECK(static_cast<int64_t>(row_to_batch.numel()) >= total_q &&
                    static_cast<int64_t>(local_starts.numel()) >= total_q &&
                    static_cast<int64_t>(local_ends.numel()) >= total_q,
                "the window arrays hold one entry per query row; need at least ",
                total_q,
                " each, got ",
                row_to_batch.numel(),
                " / ",
                local_starts.numel(),
                " / ",
                local_ends.numel());

    if(total_q <= 0 || B <= 0)
        return;

    HipDeviceGuard guard(context_lens.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    const int grid = (total_q + WINDOW_BUILD_BLOCK - 1) / WINDOW_BUILD_BLOCK;
    mqa_logits_fp4_gfx1250_prefill_windows_kernel<<<grid, WINDOW_BUILD_BLOCK, 0, stream>>>(
        reinterpret_cast<const int*>(cu_seq_q.data_ptr()),
        reinterpret_cast<const int*>(context_lens.data_ptr()),
        reinterpret_cast<int*>(row_to_batch.data_ptr()),
        reinterpret_cast<int*>(local_starts.data_ptr()),
        reinterpret_cast<int*>(local_ends.data_ptr()),
        total_q,
        B);
    HIP_CALL_LAUNCH(hipGetLastError());
}

void pa_mqa_logits_mxfp4_gfx1250_prefill_groups(aiter_tensor_t& cu_seq_q,
                                                aiter_tensor_t& group_starts,
                                                int total_q,
                                                int max_groups)
{
    aiter_detail::g_aiter_can_throw = true;
    constexpr int QPB               = mqa_logits_fp4_gfx1250_traits::Q_PER_BLOCK;
    const int B                     = static_cast<int>(cu_seq_q.size(0)) - 1;
    AITER_CHECK(cu_seq_q.dtype() == AITER_DTYPE_i32 && group_starts.dtype() == AITER_DTYPE_i32,
                "cu_seq_q / group_starts must be int32");
    AITER_CHECK(cu_seq_q.is_contiguous() && group_starts.is_contiguous(),
                "cu_seq_q / group_starts must be contiguous");
    AITER_CHECK(B >= 1, "cu_seq_q must have length batch+1 with batch >= 1, got ", B + 1);
    AITER_CHECK(B <= GROUPS_BUILD_MAX_BATCH,
                "the group builder scans the batch prefix in LDS and is capped at ",
                GROUPS_BUILD_MAX_BATCH,
                " batches, got ",
                B);
    // The kernel writes every g in [0, max_groups], so the array is max_groups + 1 long.
    AITER_CHECK(static_cast<int64_t>(group_starts.numel()) >= (int64_t)max_groups + 1,
                "group_starts needs max_groups + 1 = ",
                max_groups + 1,
                " entries, got ",
                group_starts.numel());
    // A correctness bound, not a tuning one: a max_groups below the real count silently DROPS
    // the tail groups, and their rows are then never written.
    AITER_CHECK((int64_t)max_groups >= ((int64_t)total_q + QPB - 1) / QPB,
                "max_groups must cover every row; at Q_PER_BLOCK=",
                (int)QPB,
                " and total_q=",
                total_q,
                " it must be at least ",
                (total_q + QPB - 1) / QPB,
                ", got ",
                max_groups);

    if(max_groups <= 0)
        return;

    HipDeviceGuard guard(cu_seq_q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    mqa_logits_fp4_gfx1250_prefill_groups_kernel<<<1, GROUPS_BUILD_BLOCK, 0, stream>>>(
        reinterpret_cast<const int*>(cu_seq_q.data_ptr()),
        reinterpret_cast<int*>(group_starts.data_ptr()),
        B,
        max_groups,
        QPB);
    HIP_CALL_LAUNCH(hipGetLastError());
}
