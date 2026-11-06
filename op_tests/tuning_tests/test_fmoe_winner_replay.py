# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Winner replay for FmoeTuner.post_process.

Each case feeds post_process hand-made (info, us, err) results, shaped like
the ones mp_tuner returns, and checks the full winning row: kernels, block_m,
ksplit, run_1stage, flat, nt, xbf16, times and errors.

post_process times some extra work on the GPU to keep comparisons fair:
the fp4 quant + sort or fp8 cast between stages (per_1x32), the per_1x128
quant that xbf16 kernels fuse, and moe_sorting for FLAT rows. The replay
replaces run_perftest and the tensors built for it, records which function
would have been timed, and returns a fixed time, so post_process launches
nothing. gemm_moe_tune.py still needs torch and aiter to import.
profile_fmoe.csv, which post_process always appends to, is written under a
temporary AITER_ROOT_DIR.
"""

import contextlib
import importlib.util
import io
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Same order as the key list in gemm_moe_tune.py's __main__.
KEYS = [
    "gfx",
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
]
RESULT_LIST = [
    "block_m",
    "ksplit",
    "us1",
    "kernelName1",
    "err1",
    "us2",
    "kernelName2",
    "err2",
    "us",
    "run_1stage",
    "xbf16",
    "flat",
    "tflops",
    "bw",
]
CHECKED = [
    "block_m",
    "ksplit",
    "kernelName1",
    "kernelName2",
    "run_1stage",
    "flat",
    "nt",
    "xbf16",
    "us1",
    "us2",
    "us",
    "err1",
    "err2",
]

CK1 = "moe_ck2stages_gemm1"
CK2 = "moe_ck2stages_gemm2"


def two_stage(us1, us2, kernel1="s1", kernel2="s2", **row):
    """Expected winning row of a stage1 + stage2 pair."""
    return {
        "block_m": 32,
        "ksplit": 0,
        "kernelName1": kernel1,
        "kernelName2": kernel2,
        "run_1stage": 0,
        "flat": 0,
        "nt": 0,
        "xbf16": 0,
        "us1": us1,
        "us2": us2,
        "us": round(us1 + us2, 4),
        "err1": "0.0%",
        "err2": "0.0%",
        **row,
    }


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None, "gemm_moe_tune.py needs torch"
)
class TestFmoeWinnerReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        cls.gemm_moe_tune = gemm_moe_tune
        cls.shape = cls.make_shape(token=32)
        cls.other_shape = cls.make_shape(token=64)

    @classmethod
    def make_shape(cls, token, q_dtype_a=None, q_dtype_w=None, q_type=None):
        m = cls.gemm_moe_tune
        bf16 = m.torch.bfloat16
        return (
            "gfx950",
            256,
            token,
            4096,
            1024,
            8,
            2,
            m.ActivationType.Silu,
            bf16,
            q_dtype_a or bf16,
            q_dtype_w or bf16,
            q_type or m.QuantType.No,
            1,
            0,
        )

    def setUp(self):
        m = self.gemm_moe_tune
        self.tuner = m.FmoeTuner(
            "replayFmoe", KEYS, RESULT_LIST + ["nt"], "winner replay"
        )
        with mock.patch.object(sys, "argv", ["gemm_moe_tune.py"]):
            self.args = self.tuner.parse_args()
        # FmoeTuner's own default, not TunerCommon's 0.05.
        self.assertEqual(self.args.errRatio, 0.1)

    def cand(self, stage, kernel, us, err=0.0, bm=32, v2=0, nt=0, shape=None):
        """One mp_tuner result; the tag matches the generators' layout."""
        tag = (shape or self.shape, stage, kernel, bm, 0, v2, nt)
        return (tag, us, err)

    def asm(self, kernel, us, err=0.0, bm=32, shape=None, stage="asm_1stage", flat=0):
        return ((shape or self.shape, stage, kernel, bm, flat), us, err)

    def replay(self, results, timed_us=5.0):
        """timed_us: the time of every timed function, or a function of it."""
        self.timed = []

        def fake_perftest(func, *_args, **_kwargs):
            self.timed.append(func)
            return None, timed_us(func) if callable(timed_us) else timed_us

        m = self.gemm_moe_tune
        torch = m.torch
        with tempfile.TemporaryDirectory() as root:
            configs = Path(root, "aiter", "configs")
            configs.mkdir(parents=True)
            env = {
                k: v
                for k, v in os.environ.items()
                if k != "AITER_TUNE_CANDIDATE_DUMP_DIR"
            }
            with (
                mock.patch.object(self.gemm_moe_tune, "AITER_ROOT_DIR", root),
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch("aiter.test_common.run_perftest", fake_perftest),
                mock.patch.object(torch, "randn"),
                mock.patch.object(torch, "arange"),
                mock.patch.object(torch, "tensor"),
                mock.patch.object(torch, "randint"),
                mock.patch.object(torch, "rand"),
                # The FLAT head-to-head unpacks 19 inputs; its sequences only
                # reach fake_perftest, so the values are never used.
                mock.patch.object(
                    m.FmoeTuner, "generate_data_1stage", return_value=(None,) * 19
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                out = self.tuner.post_process(list(results), self.args)
            self.assertTrue((configs / "profile_fmoe.csv").exists())
        return out

    def assertTimed(self, expected):
        """expected: the functions timed, in order; a str matches __name__."""
        self.assertEqual(len(self.timed), len(expected), self.timed)
        for func, want in zip(self.timed, expected):
            if isinstance(want, str):
                self.assertEqual(func.__name__, want)
            else:
                self.assertIs(func, want)

    def assertRow(self, row, expected):
        actual = {}
        for col in CHECKED:
            value = row[col]
            if expected[col] is None and (value is None or _is_nan(value)):
                value = None
            actual[col] = value
        self.assertEqual(actual, expected)

    def assertWinner(self, results, expected, timed=(), timed_us=5.0):
        out = self.replay(results, timed_us)
        self.assertEqual(len(out), 1)
        self.assertRow(out.iloc[0], expected)
        self.assertTimed(timed)

    def test_a_invalid_times_and_errors_are_dropped(self):
        results = [
            self.cand("stage1", "s1_err_high", 0.5, err=0.11),
            self.cand("stage1", "s1_minus1", -1),
            self.cand("stage1", "s1_neg_inf", float("-inf")),
            self.cand("stage1", "s1_inf", float("inf")),
            self.cand("stage1", "s1_at_err_ratio", 2.0, err=0.1),
            self.cand("stage1", "s1_slow", 3.0),
            self.cand("stage2", "s2", 1.0),
        ]
        self.assertWinner(
            results, two_stage(2.0, 1.0, kernel1="s1_at_err_ratio", err1="10.0%")
        )

    def test_b_pairs_share_block_m(self):
        # The best stage1 (bm32) and best stage2 (bm64) never pair: 1 + 1 = 2.
        results = [
            self.cand("stage1", "s1_bm32", 1.0, bm=32),
            self.cand("stage1", "s1_bm64_sk2", 4.0, bm=64),
            self.cand("stage2", "s2_bm32", 5.0, bm=32),
            self.cand("stage2", "s2_bm64", 1.0, bm=64),
        ]
        self.assertWinner(
            results,
            two_stage(
                4.0, 1.0, kernel1="s1_bm64", kernel2="s2_bm64", block_m=64, ksplit=2
            ),
        )

    def test_c_faster_asm_1stage_beats_every_pair(self):
        results = [
            self.cand("stage1", "s1", 2.0),
            self.cand("stage2", "s2", 2.0),
            self.asm("asm_fmoe", 3.0),
        ]
        self.assertWinner(
            results,
            two_stage(3.0, 0, kernel1="asm_fmoe", kernel2=None, run_1stage=1),
        )

    def test_d_ties_keep_the_first(self):
        with self.subTest("same stage, same us: first result wins"):
            results = [
                self.cand("stage1", "s1_first", 2.0),
                self.cand("stage1", "s1_second", 2.0),
                self.cand("stage2", "s2", 1.0),
            ]
            self.assertWinner(results, two_stage(2.0, 1.0, kernel1="s1_first"))
        with self.subTest("same pair total: the faster stage1 sorts first"):
            results = [
                self.cand("stage1", "s1_bm64", 3.0, bm=64),
                self.cand("stage2", "s2_bm64", 2.0, bm=64),
                self.cand("stage1", "s1_bm32", 2.0, bm=32),
                self.cand("stage2", "s2_bm32", 3.0, bm=32),
            ]
            self.assertWinner(
                results, two_stage(2.0, 3.0, kernel1="s1_bm32", kernel2="s2_bm32")
            )

    def test_e_v2_pairs_only_with_v2(self):
        # Crossing v2 and non-v2 would give 1 + 1 = 2.
        results = [
            self.cand("stage1", "s1", 5.0),
            self.cand("stage1", "s1_v2", 1.0, v2=1),
            self.cand("stage2", "s2", 1.0),
            self.cand("stage2", "s2_v2", 4.0, v2=1),
        ]
        self.assertWinner(
            results, two_stage(1.0, 4.0, kernel1="s1_v2", kernel2="s2_v2")
        )

    def test_nt_hint_pairs_ck_with_nt_agnostic_kernels(self):
        # The nt sweep measures CK on both sides; other kernels only at nt=0.
        with self.subTest("a non-CK stage2 also serves the nt=1 CK stage1"):
            results = [
                self.cand("stage1", f"{CK1}_a", 10.0, nt=0),
                self.cand("stage1", f"{CK1}_b", 1.0, nt=1),
                self.cand("stage2", "flydsl_moe2_x", 1.0, nt=0),
                self.cand("stage2", CK2, 8.0, nt=1),
            ]
            self.assertWinner(
                results,
                two_stage(1.0, 1.0, kernel1=f"{CK1}_b", kernel2="flydsl_moe2_x", nt=1),
            )
        with self.subTest("a CK stage2 measured at nt=0 serves only nt=0"):
            # Crossing nt would give 1 + 1 = 2.
            results = [
                self.cand("stage1", f"{CK1}_a", 10.0, nt=0),
                self.cand("stage1", f"{CK1}_b", 1.0, nt=1),
                self.cand("stage2", CK2, 1.0, nt=0),
                self.cand("stage2", f"{CK2}_nt1", 8.0, nt=1),
            ]
            self.assertWinner(
                results,
                two_stage(1.0, 8.0, kernel1=f"{CK1}_b", kernel2=f"{CK2}_nt1", nt=1),
            )

    def test_flydsl_bf16_layout_pairs_only_with_itself(self):
        # The a16w16 stage1 writes a layout CK stage2 cannot read: 1 + 1 = 2
        # would cross them.
        fly1 = "flydsl_moe1_abf16_wbf16_k"
        fly2 = "flydsl_moe2_abf16_wbf16_k"
        results = [
            self.cand("stage1", fly1, 1.0),
            self.cand("stage1", CK1, 5.0),
            self.cand("stage2", CK2, 1.0),
            self.cand("stage2", fly2, 3.0),
        ]
        self.assertWinner(results, two_stage(1.0, 3.0, kernel1=fly1, kernel2=fly2))

    def test_f_fp4_activation_charges_quant_sort_to_unfused_stage1(self):
        from aiter.ops.quant import fused_dynamic_mxfp4_quant_moe_sort

        m = self.gemm_moe_tune
        fp4 = m.dtypes.fp4x2
        shape = self.make_shape(32, fp4, fp4, m.QuantType.per_1x32)
        timed = [fused_dynamic_mxfp4_quant_moe_sort]
        with self.subTest("the _fp4 stage1 fuses it: 4 + 1 beats 1 + 5 + 1"):
            results = [
                self.cand("stage1", "s1", 1.0, shape=shape),
                self.cand("stage1", "s1_fp4", 4.0, shape=shape),
                self.cand("stage2", "s2", 1.0, shape=shape),
            ]
            self.assertWinner(results, two_stage(4.0, 1.0, kernel1="s1_fp4"), timed)
        with self.subTest("a v2 pair fuses it too: 2 + 1 beats 1 + 5 + 1"):
            results = [
                self.cand("stage1", "s1", 1.0, shape=shape),
                self.cand("stage1", "s1_v2", 2.0, v2=1, shape=shape),
                self.cand("stage2", "s2", 1.0, shape=shape),
                self.cand("stage2", "s2_v2", 1.0, v2=1, shape=shape),
            ]
            self.assertWinner(
                results, two_stage(2.0, 1.0, kernel1="s1_v2", kernel2="s2_v2"), timed
            )

    def test_f_fp8_activation_charges_cast_to_unfused_stage1(self):
        m = self.gemm_moe_tune
        shape = self.make_shape(32, m.dtypes.fp8, m.dtypes.fp4x2, m.QuantType.per_1x32)
        # The _fp8 stage1 fuses the cast: 4 + 1 beats 1 + 5 + 1.
        results = [
            self.cand("stage1", "s1", 1.0, shape=shape),
            self.cand("stage1", "s1_fp8", 4.0, shape=shape),
            self.cand("stage2", "s2", 1.0, shape=shape),
        ]
        self.assertWinner(
            results, two_stage(4.0, 1.0, kernel1="s1_fp8"), timed=["_act_to_fp8"]
        )

    def test_f_per_1x128_charges_quant_to_non_xbf16_kernels(self):
        from aiter.ops.quant import per_group_quant_hip

        m = self.gemm_moe_tune
        fp8 = m.dtypes.fp8
        shape = self.make_shape(32, fp8, fp8, m.QuantType.per_1x128)
        # The xbf16 kernel quantizes its own input: 4 beats 1 + 5.
        results = [
            self.asm("asm_plain", 1.0, shape=shape),
            self.asm("asm_xbf16", 4.0, shape=shape, stage="asm_1stage_xbf16"),
        ]
        self.assertWinner(
            results,
            two_stage(4.0, 0, kernel1="asm_xbf16", kernel2=None, run_1stage=1, xbf16=1),
            timed=[per_group_quant_hip],
        )

    def flat_race(self, shape):
        no_sort = self.gemm_moe_tune.AsmMoeMode.NO_SORT.value
        return [
            self.asm("asm_flat", 3.0, shape=shape, flat=no_sort),
            self.asm("asm_sorted", 1.0, shape=shape),
        ]

    def test_g_sorted_kernels_pay_moe_sorting(self):
        from aiter.fused_moe import moe_sorting

        no_sort = self.gemm_moe_tune.AsmMoeMode.NO_SORT.value
        # The sorted kernel pays the 5 us sort: FLAT 3 beats 1 + 5. The
        # head-to-head only runs for per_1x128, so nothing else is timed.
        self.assertWinner(
            self.flat_race(self.shape),
            two_stage(
                3.0, 0, kernel1="asm_flat", kernel2=None, run_1stage=1, flat=no_sort
            ),
            timed=[moe_sorting],
        )

    def test_g_flat_head_to_head_on_per_1x128(self):
        from aiter.fused_moe import moe_sorting

        m = self.gemm_moe_tune
        fp8 = m.dtypes.fp8
        no_sort = m.AsmMoeMode.NO_SORT.value
        shape = self.make_shape(32, fp8, fp8, m.QuantType.per_1x128)
        timed = [moe_sorting, "_flat_seq", "_nf_seq"]

        def e2e(flat_us, sorted_us):
            times = {"_flat_seq": flat_us, "_nf_seq": sorted_us}
            return lambda func: times.get(func.__name__, 5.0)

        with self.subTest("non-FLAT faster by more than 0.5 us: FLAT dropped"):
            self.assertWinner(
                self.flat_race(shape),
                two_stage(6.0, 0, kernel1="asm_sorted", kernel2=None, run_1stage=1),
                timed,
                timed_us=e2e(10.0, 8.0),
            )
        with self.subTest("within 0.5 us: FLAT kept"):
            self.assertWinner(
                self.flat_race(shape),
                two_stage(
                    3.0, 0, kernel1="asm_flat", kernel2=None, run_1stage=1, flat=no_sort
                ),
                timed,
                timed_us=e2e(10.0, 9.8),
            )

    def test_two_shapes_get_one_winner_each(self):
        other = self.other_shape
        results = [
            self.cand("stage1", "a_s1", 1.0),
            self.cand("stage1", "b_s1", 3.0, shape=other),
            self.cand("stage2", "b_s2", 3.0, shape=other),
            self.cand("stage1", "a_s1_slow", 2.0),
            self.cand("stage2", "a_s2", 1.0),
            self.cand("stage1", "b_s1_slow", 4.0, shape=other),
        ]
        out = self.replay(results)
        self.assertEqual([int(t) for t in out["token"]], [32, 64])
        self.assertRow(out.iloc[0], two_stage(1.0, 1.0, "a_s1", "a_s2"))
        self.assertRow(out.iloc[1], two_stage(3.0, 3.0, "b_s1", "b_s2"))
        self.assertTimed([])

    def test_h_no_valid_candidate_records_a_failure(self):
        results = [
            self.cand("stage1", "s1", 1.0, err=0.5),
            self.cand("stage2", "s2", -1),
        ]
        out = self.replay(results)
        self.assertTimed([])
        self.assertEqual(len(out), 0)
        self.assertEqual(len(self.tuner.failed), 1)
        failed = self.tuner.failed.iloc[0]
        self.assertEqual(int(failed["token"]), 32)
        self.assertEqual(failed["us"], self.tuner.INVALID_TIME)
        self.assertTrue(failed["kernelName1"] is None or _is_nan(failed["kernelName1"]))


def _is_nan(value):
    return isinstance(value, float) and math.isnan(value)


if __name__ == "__main__":
    unittest.main()
