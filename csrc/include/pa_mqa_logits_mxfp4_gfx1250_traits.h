// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Shapes, layouts and LDS geometry for the gfx1250 MXFP4 paged MQA logits kernel: wave32,
// `v_wmma_scale_f32_32x16x128_f4`, TDM, one CTA per group of Q_PER_BLOCK query rows sharing a
// KV window.
//
// UNITS: every stride, extent and offset below is in BYTES, never elements. fp4 has no
// host-representable scalar type -- `sizeof(opus::fp4_t)` is 1 while it holds two values -- so
// an element count here would be ambiguous at exactly the places that matter.
//
// The layout facts (which register of a fragment holds which row and which 32-element K block,
// and what each `scale_sel` selects) were measured on hardware rather than read off a table;
// the `frag_*_is_bijection` checks below are that reading turned into a compile-time test.
#pragma once

#include "pa_mqa_logits_mxfp4_gfx1250_defs.h"   // opus_mqa_logits_kargs, mqa_logits_sched, bf16_t, OPUS_LOGITS_RELU

#define OPUS_LOGITS_KV_NATURAL  0
#define OPUS_LOGITS_KV_SHUFFLED 1
#ifndef OPUS_LOGITS_FP4_KV_LAYOUT
#define OPUS_LOGITS_FP4_KV_LAYOUT OPUS_LOGITS_KV_NATURAL
#endif

#ifndef OPUS_LOGITS_FP4_QSHARE_STAGES
#define OPUS_LOGITS_FP4_QSHARE_STAGES 2
#endif

#ifndef OPUS_LOGITS_FP4_QSHARE_TDM_ISSUE_WAVES
#define OPUS_LOGITS_FP4_QSHARE_TDM_ISSUE_WAVES 0
#endif

#ifndef OPUS_LOGITS_FP4_QSHARE_WIN_VSTORE
#define OPUS_LOGITS_FP4_QSHARE_WIN_VSTORE 1
#endif

#ifndef OPUS_LOGITS_FP4_QSHARE_TDM_SCOPE
#define OPUS_LOGITS_FP4_QSHARE_TDM_SCOPE 0
#endif

#ifndef OPUS_LOGITS_FP4_QSHARE_BREAK_BSCALE_SEL
#define OPUS_LOGITS_FP4_QSHARE_BREAK_BSCALE_SEL 0
#endif

template<int Q_PER_BLOCK_  = 4,
         int LDS_STAGES_   = OPUS_LOGITS_FP4_QSHARE_STAGES,
         int KV_TILE_SIZE_ = 128,
         int PAGE_SIZE_    = 64,
         int HEAD_DIM_     = 128,
         int N_HEADS_      = 64,
         int KV_LAYOUT_    = OPUS_LOGITS_FP4_KV_LAYOUT>
struct opus_mqa_logits_fp4_qshare_traits {
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;  // block_k
    static constexpr int PAGE_SIZE    = PAGE_SIZE_;     // kv_block_size
    static constexpr int HEAD_DIM     = HEAD_DIM_;
    static constexpr int N_HEADS      = N_HEADS_;
    static constexpr int KV_LAYOUT    = KV_LAYOUT_;

    // A CONSTANT, never `opus::get_warp_size()`: that returns 64 in the host pass, which would
    // build the wave64 fragment layout with every byte count still matching.
    static constexpr int WAVE_SIZE   = 32;
    static constexpr int Q_PER_BLOCK = Q_PER_BLOCK_;               // query rows per CTA == waves
    static constexpr int NUM_WAVES   = Q_PER_BLOCK;
    static constexpr int BLOCK_SIZE  = NUM_WAVES * WAVE_SIZE;      // 128

    using D_WEIGHT = bf16_t;   // per-head weights
    using D_ACC    = float;    // WMMA accumulator (C)
    using D_OUT    = float;    // output logits
    using D_SCALE  = int;      // E8M0 blockscale, packed as one int32 dword per lane (BX32)

    static constexpr int MMA_M = 32;   // heads
    static constexpr int MMA_N = 16;   // tokens
    static constexpr int MMA_K = 128;  // head_dim

    static constexpr int ELEM_BITS   = 4;    // fp4: half a byte per element
    static constexpr int SCALE_BLOCK = 32;   // E8M0 blockscale granularity

    static constexpr int M_TILES  = N_HEADS / MMA_M;      // 2  (head tiles along M)
    static constexpr int K_TILES  = HEAD_DIM / MMA_K;     // 1  (no outer K loop)
    static constexpr int K_CHUNKS = MMA_K / SCALE_BLOCK;  // 4  (32-K scale blocks per instruction)
    static constexpr int SCALE_BLOCKS_ROW = HEAD_DIM / SCALE_BLOCK;  // 4 == K_TILES * K_CHUNKS

    static constexpr int N_TILES         = KV_TILE_SIZE / MMA_N;   // 8
    static constexpr int TILES_PER_PAGE  = PAGE_SIZE / MMA_N;      // 4
    static constexpr int PAGES_PER_TILE  = KV_TILE_SIZE / PAGE_SIZE;  // 2

    static constexpr int C_FRAG = MMA_M * MMA_N / WAVE_SIZE;   // 16 floats per lane per m-tile
    static constexpr int GRPN_C = MMA_N;                       // 16: n == lane % 16
    static constexpr int GRPM_C = WAVE_SIZE / GRPN_C;          // 2:  M spread over 2 lane groups
    static constexpr int PACK_C = C_FRAG / 2;                  // 8:  c[0..7] is m = base + i
    static constexpr int REPT_C = C_FRAG / PACK_C;             // 2:  and c[8..15] is base + 16
    static constexpr int HEADS_PER_LANE = M_TILES * C_FRAG;    // 32
    static constexpr int SWAP_DISTANCE  = GRPN_C;              // 16 == permlane16_swap's distance

    static constexpr int KV_GRP_ELEMS = SCALE_BLOCK;                     // 32 fp4 per block
    static constexpr int KV_GRP_BYTES = KV_GRP_ELEMS * ELEM_BITS / 8;    // 16 B == one ds_load_b128
    static constexpr int A_BYTES_PER_LANE = MMA_M * MMA_K / WAVE_SIZE * ELEM_BITS / 8;  // 64
    static constexpr int B_BYTES_PER_LANE = MMA_N * MMA_K / WAVE_SIZE * ELEM_BITS / 8;  // 32
    static constexpr int A_VGPR_GROUPS = A_BYTES_PER_LANE / KV_GRP_BYTES;   // 4
    static constexpr int B_VGPR_GROUPS = B_BYTES_PER_LANE / KV_GRP_BYTES;   // 2

    static constexpr int SCALE_BYTES_PER_DWORD = K_CHUNKS;   // 4
    static_assert(SCALE_BYTES_PER_DWORD == 4,
                  "a BX32 scale operand is one dword; byte b must be block b with nothing left over");

    static constexpr int TOKEN_BYTES  = HEAD_DIM * ELEM_BITS / 8;     // 64 == one token's fp4 row
    static constexpr int Q_ROW_BYTES  = N_HEADS * TOKEN_BYTES;        // 4096 == H * D/2
    static constexpr int QS_ROW_BYTES = N_HEADS * SCALE_BYTES_PER_DWORD;   // 256
    static constexpr int KV_PAGE_BYTES  = PAGE_SIZE * TOKEN_BYTES;    // 4096
    static constexpr int KVS_PAGE_BYTES = PAGE_SIZE * SCALE_BYTES_PER_DWORD;  // 256
    static constexpr int W_ROW_ELEMS    = N_HEADS;                    // natural [T, H] bf16

    static constexpr int KV_CHUNK_BYTES = PAGE_SIZE * KV_GRP_BYTES;   // 1024

    // ── the ABI, as byte offsets within one row / one page ──
    // `block` is the 32-element E8M0 block index in [0, K_CHUNKS). aiter ships the NATURAL
    // kv_cache; the shuffled branch exists for an A/B against a FlyDSL-shaped cache and puts
    // the block index OUTSIDE the token, which is why LDS_BLOCK_BYTES below is not a constant.
    static constexpr int q_byte(int head, int block) {
        return head * TOKEN_BYTES + block * KV_GRP_BYTES;
    }
    static constexpr int q_scale_byte(int head) {
        return head * SCALE_BYTES_PER_DWORD;
    }
    static constexpr int kv_byte(int token, int block) {
        return (KV_LAYOUT == OPUS_LOGITS_KV_NATURAL)
                 ? token * TOKEN_BYTES + block * KV_GRP_BYTES
                 : block * KV_CHUNK_BYTES + token * KV_GRP_BYTES;
    }
    static constexpr int kv_scale_byte(int token) {
        return token * SCALE_BYTES_PER_DWORD;
    }

    // ── the measured fragment maps, shared by the kernel and the op test's reference ──
    static constexpr int lane_m0(int lane) { return lane % GRPN_C; }   // L % 16
    static constexpr int lane_g (int lane) { return lane / GRPN_C; }   // L / 16

    static constexpr int frag_a_row  (int lane, int v) { return lane_m0(lane) + (v >= 2 ? 16 : 0); }
    static constexpr int frag_a_block(int lane, int v) { return lane_g(lane) + (v % 2 ? 2 : 0); }
    static constexpr int frag_b_col  (int lane)        { return lane_m0(lane); }
    static constexpr int frag_b_block(int lane, int v) { return lane_g(lane) + (v ? 2 : 0); }
    static constexpr int frag_c_n    (int lane)        { return lane_m0(lane); }
    static constexpr int frag_c_m    (int lane, int reg) {
        return (reg < PACK_C ? 0 : 16) + lane_g(lane) * PACK_C + reg % PACK_C;
    }
    static constexpr bool frag_a_is_bijection() {
        bool seen[MMA_M][K_CHUNKS] = {};
        for (int lane = 0; lane < WAVE_SIZE; ++lane)
            for (int v = 0; v < A_VGPR_GROUPS; ++v) {
                const int r = frag_a_row(lane, v), b = frag_a_block(lane, v);
                if (r < 0 || r >= MMA_M || b < 0 || b >= K_CHUNKS || seen[r][b]) return false;
                seen[r][b] = true;
            }
        for (int r = 0; r < MMA_M; ++r)
            for (int b = 0; b < K_CHUNKS; ++b) if (!seen[r][b]) return false;
        return true;
    }
    static constexpr bool frag_b_is_bijection() {
        bool seen[MMA_N][K_CHUNKS] = {};
        for (int lane = 0; lane < WAVE_SIZE; ++lane)
            for (int v = 0; v < B_VGPR_GROUPS; ++v) {
                const int n = frag_b_col(lane), b = frag_b_block(lane, v);
                if (n < 0 || n >= MMA_N || b < 0 || b >= K_CHUNKS || seen[n][b]) return false;
                seen[n][b] = true;
            }
        for (int n = 0; n < MMA_N; ++n)
            for (int b = 0; b < K_CHUNKS; ++b) if (!seen[n][b]) return false;
        return true;
    }
    static constexpr bool frag_c_is_bijection() {
        bool seen[MMA_M][MMA_N] = {};
        for (int lane = 0; lane < WAVE_SIZE; ++lane)
            for (int reg = 0; reg < C_FRAG; ++reg) {
                const int m = frag_c_m(lane, reg), n = frag_c_n(lane);
                if (m < 0 || m >= MMA_M || n < 0 || n >= MMA_N || seen[m][n]) return false;
                seen[m][n] = true;
            }
        for (int m = 0; m < MMA_M; ++m)
            for (int n = 0; n < MMA_N; ++n) if (!seen[m][n]) return false;
        return true;
    }
    static_assert(frag_a_is_bijection(), "A's (lane, VGPR group) -> (row, block) map is not a bijection");
    static_assert(frag_b_is_bijection(), "B's (lane, VGPR group) -> (col, block) map is not a bijection");
    static_assert(frag_c_is_bijection(), "C's (lane, reg) -> (m, n) map is not a bijection");

    static constexpr int scale_a_row(int lane) { return lane; }
    static constexpr int scale_b_sel (int n_tile) { return n_tile % 2; }
    static constexpr int scale_b_lane(int n_tile, int n) { return scale_b_sel(n_tile) * GRPN_C + n; }

    // ── LDS: the tile image, and the one padding policy that breaks the bank conflict ──
    // TDM copies a page as it is, so the pad is the ONLY LDS-side freedom: WMMA fixes which
    // lane reads which token, so the lane mapping cannot be rearranged instead.
    static constexpr int LDS_PAD_INTERVAL = (KV_LAYOUT == OPUS_LOGITS_KV_NATURAL) ? 128 : 0;
    static constexpr int LDS_PAD_AMOUNT   = (KV_LAYOUT == OPUS_LOGITS_KV_NATURAL) ?  16 : 0;
    static constexpr int LDS_PAD_STRIDE = LDS_PAD_INTERVAL ? LDS_PAD_INTERVAL : 1;

    static constexpr int lds_expand(int page_byte) {
        return page_byte + (page_byte / LDS_PAD_STRIDE) * LDS_PAD_AMOUNT;
    }
    static constexpr int lds_byte(int token_in_page, int block) {
        return lds_expand(kv_byte(token_in_page, block));
    }
    static constexpr int LDS_PAGE_BYTES = lds_expand(KV_PAGE_BYTES);  // 4608 natural / 4096 shuffled
    static constexpr int LDS_TILE_BYTES  = PAGES_PER_TILE * LDS_PAGE_BYTES;   // 9216 / 8192
    static constexpr int LDS_SCALE_BYTES = KV_TILE_SIZE * SCALE_BYTES_PER_DWORD;   // 512
    static constexpr int LDS_STAGE_BYTES = LDS_TILE_BYTES + LDS_SCALE_BYTES;
    static constexpr int LDS_STAGES      = LDS_STAGES_;
    static constexpr int LDS_BYTES       = LDS_STAGES * LDS_STAGE_BYTES;
    static constexpr size_t smem_size_bytes() { return (size_t)LDS_BYTES; }
    static_assert((LDS_STAGES & (LDS_STAGES - 1)) == 0,
                  "LDS_STAGES must be a power of two: the stage index is a mask, so a non-power "
                  "turns one `and` into a division in the phase's address arithmetic");

    static constexpr int lds_stage_byte(int tile) { return (tile & (LDS_STAGES - 1)) * LDS_STAGE_BYTES; }
    static constexpr int lds_page_byte (int page) { return page * LDS_PAGE_BYTES; }
    static constexpr int lds_scale_byte(int token_in_tile) {
        return LDS_TILE_BYTES + token_in_tile * SCALE_BYTES_PER_DWORD;
    }

    static constexpr int LDS_NTILE_BYTES = lds_byte(MMA_N, 0) - lds_byte(0, 0);   // 1152 / 256
    static constexpr bool lds_ntile_step_is_uniform() {
        for (int t = 0; t + MMA_N <= PAGE_SIZE - MMA_N; ++t)
            for (int b = 0; b < K_CHUNKS; ++b)
                if (lds_byte(t + MMA_N, b) - lds_byte(t, b) != LDS_NTILE_BYTES) return false;
        return true;
    }
    static_assert(lds_ntile_step_is_uniform(),
                  "the n-tile step in LDS is not lane-independent, so the B reads cannot share an "
                  "address register. The pad interval must divide MMA_N * TOKEN_BYTES.");

    static constexpr int LDS_BLOCK_BYTES = lds_byte(0, 1) - lds_byte(0, 0);   // 16 / 1024
    static constexpr bool lds_block_step_is_uniform() {
        for (int t = 0; t < PAGE_SIZE; ++t)
            for (int b = 0; b + 1 < K_CHUNKS; ++b)
                if (lds_byte(t, b + 1) - lds_byte(t, b) != LDS_BLOCK_BYTES) return false;
        return true;
    }
    static_assert(lds_block_step_is_uniform(),
                  "the K-block step in LDS is not uniform -- under the natural layout that means "
                  "the pad is cutting a token in half -- so the B reads' block offset cannot be "
                  "an immediate");

    static constexpr int lds_bank_unit(int lane, int v) {
        return (lds_byte(frag_b_col(lane), frag_b_block(lane, v)) >> 4) % 8;
    }
    static constexpr bool lds_conflict_free() {
        for (int v = 0; v < B_VGPR_GROUPS; ++v)
            for (int base = 0; base < WAVE_SIZE; base += 8) {
                int seen = 0;
                for (int i = 0; i < 8; ++i) seen |= 1 << lds_bank_unit(base + i, v);
                if (seen != 0xFF) return false;
            }
        return true;
    }
    static constexpr bool lds_conflict_free_unpadded_control() {
        for (int base = 0; base < WAVE_SIZE; base += 8) {
            int seen = 0;
            for (int i = 0; i < 8; ++i) {
                const int lane = base + i;
                const int linear = frag_b_col(lane) * TOKEN_BYTES + frag_b_block(lane, 0) * KV_GRP_BYTES;
                seen |= 1 << ((linear >> 4) % 8);
            }
            if (seen != 0xFF) return false;
        }
        return true;
    }
    static_assert(!lds_conflict_free_unpadded_control(),
                  "the unpadded natural layout is supposed to CONFLICT; if it does not, this "
                  "checker cannot detect a conflict and the assert below proves nothing");
    static_assert(lds_conflict_free(),
                  "the LDS padding policy leaves a bank conflict. Unpadded, the natural layout's "
                  "64 B token stride puts every bank in {0-3} or {16-19} -- a 4-way conflict; the "
                  "pad of one read vector per 128 B is what makes the 8 lanes tile all 32 banks.");
    static_assert((KV_LAYOUT == OPUS_LOGITS_KV_SHUFFLED) == (LDS_PAD_INTERVAL == 0),
                  "the shuffled layout's lanes are 16 B apart and tile the banks unpadded; the "
                  "natural layout's are 64 B apart and do not. Each branch carries its own policy.");

    // ── TDM. Every wave issues, so the predicate is a compile-time constant and the steady
    // loop keeps no branch for it. A page splits by CONTIGUOUS BYTES -- the one thing both
    // kv_cache layouts agree on -- and wave w takes piece (w % SPLIT) of page (w / SPLIT). ──
    static constexpr int TDM_ISSUE_WAVES = OPUS_LOGITS_FP4_QSHARE_TDM_ISSUE_WAVES
                                             ? OPUS_LOGITS_FP4_QSHARE_TDM_ISSUE_WAVES : NUM_WAVES;
    static constexpr int TDM_PAGE_SPLIT  = TDM_ISSUE_WAVES / PAGES_PER_TILE;   // 2
    static constexpr int TDM_OPS_PER_TILE = 2 * TDM_ISSUE_WAVES;               // 8
    static constexpr int TDM_INFLIGHT_PER_WAVE = 3;                   // hardware, per opus.hpp
    static constexpr int TDM_OPS_PER_ISSUING_WAVE = TDM_OPS_PER_TILE / TDM_ISSUE_WAVES;   // 2
    static constexpr int TDM_INFLIGHT_CAP = TDM_INFLIGHT_PER_WAVE / TDM_OPS_PER_ISSUING_WAVE;

    static constexpr int TDM_ROWS        = PAGE_SIZE / TDM_PAGE_SPLIT;         // 32
    static constexpr int TDM_PIECE_BYTES = KV_PAGE_BYTES / TDM_PAGE_SPLIT;     // 2048, global side
    static constexpr int LDS_PIECE_BYTES = lds_expand(TDM_PIECE_BYTES);        // 2304 / 2048
    static constexpr int TDM_SCALE_PIECE_BYTES = TDM_ROWS * SCALE_BYTES_PER_DWORD;   // 128

    static_assert(TDM_ISSUE_WAVES <= NUM_WAVES,
                  "there are not enough waves to spread a tile's TDM issues over");
    static_assert(TDM_ISSUE_WAVES % PAGES_PER_TILE == 0,
                  "the issuing waves must divide into whole pages, or a wave straddles two pages "
                  "and needs two page ids");
    static_assert(PAGE_SIZE % TDM_PAGE_SPLIT == 0 && KV_PAGE_BYTES % TDM_PAGE_SPLIT == 0,
                  "a page must split into equal pieces");
    static_assert(LDS_PIECE_BYTES * TDM_PAGE_SPLIT == LDS_PAGE_BYTES,
                  "the pieces must tile the padded LDS page exactly. They do not when a piece's "
                  "byte count is not a whole number of pad intervals, and the pieces then overlap "
                  "by the rounding -- silently, because each DMA on its own is still in range.");
    static_assert(TDM_OPS_PER_ISSUING_WAVE <= TDM_INFLIGHT_PER_WAVE,
                  "one tile's share already exceeds a wave's TDM queue depth");

    // ── pipeline depth: it follows from the stage count, not the other way round ──
    static constexpr int TILES_IN_FLIGHT = LDS_STAGES - 1;
    static constexpr int ISSUE_LEAD      = TILES_IN_FLIGHT;   // phase t issues tile t + this
    static_assert(LDS_STAGES >= 2, "need at least a double buffer");
    static_assert(TILES_IN_FLIGHT <= TDM_INFLIGHT_CAP,
                  "LDS_STAGES asks for more tiles in flight than the per-wave TDM limit allows "
                  "under this issue policy. Raise OPUS_LOGITS_FP4_QSHARE_TDM_ISSUE_WAVES -- that "
                  "is what raises the cap -- or lower the stage count.");
    static constexpr int TENSORCNT_KEEP = (TILES_IN_FLIGHT - 1) * TDM_OPS_PER_ISSUING_WAVE;

    // ── the accumulator. TWO copies, ping-ponged, so tile t+1's WMMAs issue while tile t is
    // reduced; measured at 20-52% over one copy wherever the grid fills the machine, which is
    // why occupancy 1 is the deliberate operating point. ──
    static constexpr int ACC_SETS = 2;
    static constexpr int ACC_VGPR_PER_SET = M_TILES * N_TILES * C_FRAG;   // 256
    static constexpr int ACC_VGPR = ACC_SETS * ACC_VGPR_PER_SET;          // 512
    static constexpr int WMMA_PER_TILE = M_TILES * N_TILES;               // 16
    static constexpr int WAVES_PER_EU = 1;
    static constexpr int VGPR_ADDRESSABLE = 1024;                    // wave32, per lane
    static_assert(ACC_VGPR < VGPR_ADDRESSABLE,
                  "the accumulator alone must not fill the register file");

    static_assert(ELEM_BITS == 4, "this traits set is fp4-only");
    static_assert(HEAD_DIM == MMA_K,
                  "K = 128 is the whole head_dim in one instruction; a HEAD_DIM past MMA_K needs "
                  "the outer kt loop back, and with it the kt term in both scale byte indices");
    static_assert(N_HEADS % MMA_M == 0, "N_HEADS must be a multiple of MMA_M (32)");
    static_assert(KV_TILE_SIZE % MMA_N == 0, "KV_TILE must be a multiple of MMA_N");
    static_assert(KV_TILE_SIZE % PAGE_SIZE == 0, "KV_TILE must be a whole number of pages");
    static_assert(PAGE_SIZE % MMA_N == 0, "PAGE must be a multiple of MMA_N");
    static_assert(M_TILES == 2, "the head reduction assumes 64 heads over 2 m-tiles");
    static_assert(GRPM_C == 2, "the head reduction emits exactly one permlane16_swap");
    static_assert(HEADS_PER_LANE * GRPM_C == N_HEADS,
                  "a lane and its swap partner must together hold every head, or the reduction "
                  "drops heads without any count changing");
    static_assert(A_BYTES_PER_LANE == 64 && B_BYTES_PER_LANE == 32,
                  "the dedicated f4 instruction takes i32x16 / i32x8 and uses all of both");
    static_assert(KV_GRP_BYTES == 16, "a 32-element fp4 block must be exactly one ds_load_b128");
    static_assert(Q_ROW_BYTES == N_HEADS * HEAD_DIM * ELEM_BITS / 8, "q must be size-preserving");
    static_assert(QS_ROW_BYTES == N_HEADS * SCALE_BLOCKS_ROW, "q_scale must be size-preserving");
    static_assert(KV_PAGE_BYTES == PAGE_SIZE * HEAD_DIM * ELEM_BITS / 8, "kv must be size-preserving");
    static_assert(Q_PER_BLOCK >= 1, "Q_PER_BLOCK must be positive");
};

using logits_fp4_qshare_traits_4q = opus_mqa_logits_fp4_qshare_traits<4>;

#ifndef OPUS_LOGITS_FP4_QSHARE_Q_PER_BLOCK
#define OPUS_LOGITS_FP4_QSHARE_Q_PER_BLOCK (logits_fp4_qshare_traits_4q::Q_PER_BLOCK)
#endif
