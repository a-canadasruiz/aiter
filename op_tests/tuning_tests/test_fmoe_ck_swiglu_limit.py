# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Where --swiglu_limit reaches the CK two-stage candidates.

tuner.gen_2stages_task reads the tuner's limit. On an FP8 per_1x128 shape every
CK stage-1 task passes swiglu_limit to the kernel and to the torch reference,
and both stages compare with limit_cosine_diff_compare. Without a limit, or on
any other shape, the same tasks carry neither. Tasks are only built: nothing
runs on the GPU.
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

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
RESULTS = [
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
    "nt",
]
BLOCK_MS = [16, 32, 64, 128]
# Task layout: (tag, gen, gen_args, func, func_args, func_kwargs,
#               ref, ref_args, ref_kwargs, ref_out, rtol, atol, compare)
TAG, FUNC_KWARGS, REF_KWARGS, COMPARE = 0, 5, 8, 12


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None, "the fmoe tuner needs torch"
)
class TestFmoeCkSwigluLimit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from aiter import ActivationType, QuantType, dtypes
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        cls.FmoeTuner = gemm_moe_tune.FmoeTuner
        cls.ActivationType = ActivationType
        cls.QuantType = QuantType
        cls.dtypes = dtypes

    def tuner(self, *extra):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = os.path.join(tmp.name, "tuned.csv")
        tuner = self.FmoeTuner("fmoeTuner", KEYS, list(RESULTS), "fmoe tuner")
        with mock.patch.object(sys, "argv", ["gemm_moe_tune.py", "-o", out, *extra]):
            tuner.parse_args()
        return tuner

    def info(self, q_type):
        d = self.dtypes
        return (
            "gfx950",
            256,
            16,
            4096,
            384,
            128,
            8,
            self.ActivationType.Silu,
            d.bf16,
            d.fp8,
            d.fp8,
            q_type,
            1,
            0,
        )

    def stages(self, tuner, q_type):
        tasks = tuner.gen_2stages_task(self.info(q_type), BLOCK_MS)
        s1 = [t for t in tasks if t[TAG][1] == "stage1"]
        s2 = [t for t in tasks if t[TAG][1] == "stage2"]
        self.assertTrue(s1 and s2, "expected CK tasks for both stages")
        return s1, s2

    def compare_name(self, task):
        fn = task[COMPARE]
        return None if fn is None else fn.__name__

    def test_limit_reaches_stage1_kwargs_and_both_compares(self):
        s1, s2 = self.stages(
            self.tuner("--swiglu_limit", "7"), self.QuantType.per_1x128
        )
        for t in s1:
            self.assertEqual(t[FUNC_KWARGS].get("swiglu_limit"), 7.0, t[TAG])
            self.assertEqual(t[REF_KWARGS], {"swiglu_limit": 7.0}, t[TAG])
            self.assertEqual(self.compare_name(t), "limit_cosine_diff_compare")
        for t in s2:
            self.assertNotIn("swiglu_limit", t[FUNC_KWARGS], t[TAG])
            self.assertEqual(t[REF_KWARGS], {}, t[TAG])
            self.assertEqual(self.compare_name(t), "limit_cosine_diff_compare")

    def test_without_a_limit_the_same_tasks_carry_no_limit(self):
        limited = self.stages(
            self.tuner("--swiglu_limit", "7"), self.QuantType.per_1x128
        )
        plain = self.stages(self.tuner(), self.QuantType.per_1x128)
        self.assertEqual(
            [t[TAG] for s in plain for t in s], [t[TAG] for s in limited for t in s]
        )
        for t in plain[0] + plain[1]:
            self.assertNotIn("swiglu_limit", t[FUNC_KWARGS], t[TAG])
            self.assertEqual(t[REF_KWARGS], {}, t[TAG])
            self.assertIsNone(t[COMPARE], t[TAG])

    def test_limit_is_ignored_outside_fp8_blockscale(self):
        s1, s2 = self.stages(
            self.tuner("--swiglu_limit", "7"), self.QuantType.per_Token
        )
        for t in s1 + s2:
            self.assertNotIn("swiglu_limit", t[FUNC_KWARGS], t[TAG])
            self.assertEqual(t[REF_KWARGS], {}, t[TAG])
            self.assertIsNone(t[COMPARE], t[TAG])


if __name__ == "__main__":
    unittest.main(verbosity=2)
