# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Level 1: the a8w8 bpreshuffle tuner builds tasks from the args it is given.

The task builders used to read a module-global ``args`` that only exists when
the file runs as ``__main__``; importing the tuner and calling them raised
NameError. These tests import the file as a plain module.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

TUNER_DIR = Path(__file__).resolve().parents[2] / "csrc" / "ck_gemm_a8w8_bpreshuffle"
TUNER_FILE = TUNER_DIR / "gemm_a8w8_bpreshuffle_tune.py"

KEY = ["gfx", "cu_num", "M", "N", "K", "q_dtype_w"]
RESULT_LIST = [
    "libtype",
    "kernelId",
    "splitK",
    "us",
    "kernelName",
    "tflops",
    "bw",
    "errRatio",
]
# Index of the bench kwargs dict ({"num_warmup", "num_iters"}) in a task tuple.
BENCH_KWARGS = 5


def load_tuner_module():
    spec = importlib.util.spec_from_file_location(
        "gemm_a8w8_bpreshuffle_tune_under_test", TUNER_FILE
    )
    module = importlib.util.module_from_spec(spec)
    saved_path = list(sys.path)
    sys.path.insert(0, str(TUNER_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        # The tuner inserts its cktile directory at sys.path[0] on import.
        sys.path[:] = saved_path
    return module


class TestBpreshuffleTaskBuildersUseGivenArgs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mod = load_tuner_module()
        cls.tuner = cls.mod.GemmA8W8BpreShuffleTuner(
            "GemmA8W8BpreShuffleTuner", key=KEY, resultList=RESULT_LIST
        )
        cls.args = cls.tuner.parser.parse_args(
            ["--warmup", "7", "--iters", "13", "--blockPerCu", "1"]
        )
        from aiter import dtypes

        cls.info_keys = ("gfx950", 256, 128, 1024, 512, str(dtypes.fp8))

    def assert_bench_kwargs(self, tasks):
        for task in tasks:
            self.assertEqual(task[BENCH_KWARGS], {"num_warmup": 7, "num_iters": 13})

    def test_module_has_no_global_args(self):
        self.assertFalse(hasattr(self.mod, "args"))

    def test_ck_tasks(self):
        tasks = self.tuner.get_ck_gemm_a8w8_bpreshuffle_tune_task(
            self.info_keys, False, 1, self.args
        )
        self.assertGreater(len(tasks), 0)
        self.assert_bench_kwargs(tasks)

    def test_cktile_tasks_follow_block_per_cu(self):
        tasks = self.tuner.get_cktile_gemm_a8w8_bpreshuffle_tune_task(
            self.info_keys, False, 1, self.args
        )
        self.assertGreater(len(tasks), 0)
        self.assert_bench_kwargs(tasks)
        for task in tasks:
            kernel_id = task[0][1]
            self.assertEqual(self.mod.kernels_list_cktile[kernel_id].BlockPerCu, 1)

    def test_flydsl_tasks(self):
        tasks = self.tuner.get_flydsl_gemm_a8w8_bpreshuffle_tune_task(
            self.info_keys, False, 1, self.args
        )
        self.assert_bench_kwargs(tasks)


if __name__ == "__main__":
    unittest.main(verbosity=2)
