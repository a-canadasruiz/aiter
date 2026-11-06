# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU locks for FmoeTuner.tune() dispatch, generator filters, and CSV write.

The winner replay feeds post_process hand-made times. This file covers the
paths that replay never calls: which generators tune() runs, the opus and
FlyDSL shape filters, GroupedFmoeTuner and Mxfp4FlydslTuner candidate
selection, and result_to_csv. Generators are called for real where the check
is the task list. The GPU timing entry, mp_tuner, is replaced.
"""

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

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
GENERATORS = (
    "gen_2stages_asm1_task",
    "gen_2stages_task",
    "gen_flydsl_2stages_task",
    "gen_flydsl_v2_2stages_task",
    "gen_flydsl_i4_2stages_task",
    "gen_opus_2stages_task",
    "gen_1stage_asm_task",
)
_TUNER_ENV = (
    "TUNE_ONLY",
    "OPUS_ONLY",
    "OPUS_SKIP_CKTILE",
    "TUNE_MOE_KERNEL_REGEX",
    "AITER_FMOE_TUNE_EXCLUDE_KERNELS",
)
BLOCK_MS = [16, 32, 64, 128]


def _task(name):
    return (("info", "stage1", name, 32), None)


def _kernel_names(tasks):
    return [task[0][2] for task in tasks]


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None, "gemm_moe_tune.py needs torch"
)
class TestFmoeTuneDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        cls.gemm_moe_tune = gemm_moe_tune

    def setUp(self):
        m = self.gemm_moe_tune
        self.tuner = m.FmoeTuner(
            "dispatchFmoe", KEYS, RESULT_LIST + ["nt"], "tune dispatch"
        )
        with mock.patch.object(sys, "argv", ["gemm_moe_tune.py"]):
            self.args = self.tuner.parse_args()

    def untuned(self, **overrides):
        row = {
            "gfx": "gfx950",
            "cu_num": 256,
            "token": 32,
            "model_dim": 4096,
            "inter_dim": 1024,
            "expert": 8,
            "topk": 2,
            "act_type": "ActivationType.Silu",
            "dtype": "torch.bfloat16",
            "q_dtype_a": "torch.bfloat16",
            "q_dtype_w": "torch.bfloat16",
            "q_type": "QuantType.No",
            "use_g1u1": 1,
            "doweight_stage1": 0,
        }
        row.update(overrides)
        return pd.DataFrame([row])[KEYS]

    def run_tune(self, frame, env=None, gfx="gfx950", returns=None):
        """Call tune() with every generator replaced. returns maps a generator
        name to the tasks it yields; the rest yield nothing."""
        returns = {} if returns is None else returns
        calls = []
        got = {}

        def make(name):
            def fn(*args, **kwargs):
                calls.append(name)
                return list(returns.get(name, []))

            return fn

        def fake_mp(tasks, *args, **kwargs):
            got["tasks"] = list(tasks)
            return []

        cleaned = {k: v for k, v in os.environ.items() if k not in _TUNER_ENV}
        cleaned.update(env or {})
        with (
            mock.patch.object(self.gemm_moe_tune, "get_gfx", return_value=gfx),
            mock.patch.object(self.gemm_moe_tune, "mp_tuner", fake_mp),
            mock.patch.dict(os.environ, cleaned, clear=True),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.ExitStack() as stack,
        ):
            for name in GENERATORS:
                stack.enter_context(mock.patch.object(self.tuner, name, make(name)))
            out = self.tuner.tune(frame, None, self.args)
        return out, calls, got

    def info(
        self,
        *,
        token=1,
        model_dim=5120,
        inter_dim=1152,
        q_dtype_a=None,
        q_dtype_w=None,
        q_type=None,
        act_type=None,
        doweight_stage1=0,
        dtype=None,
    ):
        m = self.gemm_moe_tune
        return (
            "gfx950",
            256,
            token,
            model_dim,
            inter_dim,
            8,
            2,
            m.ActivationType.Silu if act_type is None else act_type,
            m.torch.bfloat16 if dtype is None else dtype,
            m.dtypes.fp8 if q_dtype_a is None else q_dtype_a,
            m.dtypes.fp4x2 if q_dtype_w is None else q_dtype_w,
            m.QuantType.per_1x32 if q_type is None else q_type,
            1,
            doweight_stage1,
        )

    def test_per_1x32_off_gfx950_returns_no_tasks(self):
        frame = self.untuned(q_type="QuantType.per_1x32")
        out, calls, got = self.run_tune(frame, gfx="gfx942")
        self.assertEqual(out, [])
        self.assertEqual(calls, [])
        self.assertNotIn("tasks", got)
        # The same shape on gfx950 still reaches the generators.
        out, calls, got = self.run_tune(frame, gfx="gfx950")
        self.assertEqual(out, [])
        self.assertIn("gen_flydsl_v2_2stages_task", calls)

    def test_g1u0_is_skipped(self):
        out, calls, got = self.run_tune(self.untuned(use_g1u1=0))
        self.assertEqual(out, [])
        self.assertEqual(calls, [])
        self.assertNotIn("tasks", got)

    def test_default_tune_only_stops_at_flydsl_v2_when_it_has_tasks(self):
        returns = {"gen_flydsl_v2_2stages_task": [_task("v2_kernel")]}
        out, calls, got = self.run_tune(self.untuned(), returns=returns)
        self.assertEqual(out, [])
        self.assertEqual(calls, ["gen_flydsl_v2_2stages_task"])
        self.assertEqual(_kernel_names(got["tasks"]), ["v2_kernel"])

    def test_default_tune_only_runs_every_generator_when_v2_is_empty(self):
        returns = {name: [_task(name)] for name in GENERATORS}
        returns["gen_flydsl_v2_2stages_task"] = []
        out, calls, got = self.run_tune(self.untuned(), returns=returns)
        self.assertEqual(out, [])
        # The v2 probe is the first call. An empty result is reused, not
        # generated a second time, and the other six generators follow.
        self.assertEqual(calls[0], "gen_flydsl_v2_2stages_task")
        self.assertEqual(calls.count("gen_flydsl_v2_2stages_task"), 1)
        self.assertEqual(set(calls), set(GENERATORS))
        self.assertEqual(
            _kernel_names(got["tasks"]),
            [name for name in GENERATORS if name != "gen_flydsl_v2_2stages_task"],
        )

    def test_opus_only_and_skip_cktile_select_generators(self):
        returns = {name: [_task(name)] for name in GENERATORS}
        with self.subTest("OPUS_ONLY"):
            out, calls, got = self.run_tune(
                self.untuned(),
                env={"TUNE_ONLY": "", "OPUS_ONLY": "1"},
                returns=returns,
            )
            self.assertEqual(calls, ["gen_opus_2stages_task"])
            self.assertEqual(_kernel_names(got["tasks"]), ["gen_opus_2stages_task"])
            self.assertEqual(out, [])
        with self.subTest("OPUS_SKIP_CKTILE skips gen_2stages_task"):
            out, calls, got = self.run_tune(
                self.untuned(),
                env={"TUNE_ONLY": "", "OPUS_SKIP_CKTILE": "1"},
                returns=returns,
            )
            self.assertNotIn("gen_2stages_task", calls)
            self.assertIn("gen_opus_2stages_task", calls)
            self.assertNotIn("gen_2stages_task", _kernel_names(got["tasks"]))

    def test_kernel_regex_filters_dispatched_tasks(self):
        returns = {name: [_task("drop_me")] for name in GENERATORS}
        returns["gen_opus_2stages_task"] = [_task("keep_me")]
        out, calls, got = self.run_tune(
            self.untuned(),
            env={"TUNE_ONLY": "", "TUNE_MOE_KERNEL_REGEX": "keep_me"},
            returns=returns,
        )
        self.assertEqual(out, [])
        self.assertEqual(set(calls), set(GENERATORS))
        self.assertEqual(_kernel_names(got["tasks"]), ["keep_me"])

    def test_opus_stage1_rejects_a_model_dim_that_misses_the_mfma_tile(self):
        # Stage1 needs model_dim % 128 == 0. 5184 % 128 == 64, and 1152 still
        # passes the inter_dim gate, so only stage1 disappears.
        m = self.gemm_moe_tune
        bad = self.tuner.gen_opus_2stages_task(
            self.info(model_dim=5184, inter_dim=1152), BLOCK_MS
        )
        good = self.tuner.gen_opus_2stages_task(
            self.info(model_dim=5120, inter_dim=1152, token=1), BLOCK_MS
        )
        self.assertEqual([t[0][1] for t in bad if t[0][1] == "stage1"], [])
        self.assertTrue(any(t[0][1] == "stage1" for t in good))
        # inter_dim % 32 == 0 lives in the stage1 helper. 1168 % 32 == 16.
        # The tuner also rejects it earlier (inter_dim % 128), so both ends
        # of that chain have to stay.
        self.assertEqual(
            self.tuner.gen_opus_2stages_task(self.info(inter_dim=1168), BLOCK_MS),
            [],
        )
        self.assertEqual(
            m.opus_a8w4_stage1_instances_for_shape(
                model_dim=5120, logical_inter_dim=1168, inter_dim_pad=0
            ),
            (),
        )
        self.assertTrue(
            m.opus_a8w4_stage1_instances_for_shape(
                model_dim=5120, logical_inter_dim=1152, inter_dim_pad=0
            )
        )

    def test_opus_stage1_gates_leave_stage2_in_place(self):
        # stage1_supported only guards the stage1 loop. An unsupported
        # activation or doweight_stage1 drops stage1 and keeps every stage2.
        m = self.gemm_moe_tune

        def by_stage(tasks):
            return {
                stage: sorted(t[0][2:4] for t in tasks if t[0][1] == stage)
                for stage in ("stage1", "stage2")
            }

        base = by_stage(self.tuner.gen_opus_2stages_task(self.info(), BLOCK_MS))
        self.assertTrue(base["stage1"])
        self.assertTrue(base["stage2"])
        cases = {
            "Gelu": self.info(act_type=m.ActivationType.Gelu),
            "doweight_stage1": self.info(doweight_stage1=1),
        }
        for name, info in cases.items():
            with self.subTest(name):
                got = by_stage(self.tuner.gen_opus_2stages_task(info, BLOCK_MS))
                self.assertEqual(got["stage1"], [])
                self.assertEqual(got["stage2"], base["stage2"])

    def test_opus_stage2_drops_a_block_n_that_misses_model_dim(self):
        # 5248 = 41 * 128: a multiple of block_n 64 and 128, not of 256.
        m = self.gemm_moe_tune
        catalog = m.get_opus_a8w4_stage2_kernels(token=1)
        by_block_n = {}
        for name, kparams in catalog.items():
            by_block_n.setdefault(int(kparams["kernel_block_n"]), set()).add(name)
        self.assertEqual(set(by_block_n), {64, 128, 256})

        def stage2(model_dim):
            tasks = self.tuner.gen_opus_2stages_task(
                self.info(model_dim=model_dim), BLOCK_MS
            )
            return {t[0][2] for t in tasks if t[0][1] == "stage2" and len(t[0]) == 4}

        self.assertEqual(stage2(5120), set(catalog))
        self.assertEqual(stage2(5248), by_block_n[64] | by_block_n[128])

    def test_opus_stage2_kernels_depend_on_token(self):
        small = self.tuner.gen_opus_2stages_task(self.info(token=1), BLOCK_MS)
        large = self.tuner.gen_opus_2stages_task(self.info(token=2048), BLOCK_MS)

        def stage2(tasks):
            return {t[0][2] for t in tasks if t[0][1] == "stage2"}

        self.assertTrue(stage2(small))
        self.assertTrue(stage2(large))
        self.assertNotEqual(stage2(small), stage2(large))

    def test_flydsl_stage1_drops_kernels_whose_k_does_not_divide(self):
        m = self.gemm_moe_tune
        kernels = m.get_flydsl_stage1_kernels("fp4", "fp4", "bf16")

        def divides(kparams, model_dim):
            k_batch = kparams.get("k_batch", 1) or 1
            tile_k = kparams["tile_k"]
            k_wave = kparams.get("k_wave", 1) or 1
            if model_dim % k_batch != 0:
                return False
            per_batch = model_dim // k_batch
            if per_batch % tile_k != 0:
                return False
            return not (
                k_wave > 1
                and (per_batch % k_wave != 0 or (per_batch // k_wave) % tile_k != 0)
            )

        model_dim = None
        rejected = []
        for candidate in (100, 96, 80, 48, 7):
            rejected = [
                name
                for name, kparams in kernels.items()
                if kparams.get("tile_m") in (16, 32, 64, 128)
                and not divides(kparams, candidate)
            ]
            if rejected:
                model_dim = candidate
                break
        self.assertIsNotNone(
            model_dim, "no catalog kernel misses these model_dim values"
        )
        tasks = self.tuner.gen_flydsl_2stages_task(
            self.info(
                model_dim=model_dim,
                inter_dim=256,
                q_dtype_a=m.dtypes.fp4x2,
                q_dtype_w=m.dtypes.fp4x2,
            ),
            BLOCK_MS,
        )
        emitted = {t[0][2] for t in tasks if t[0][1] == "stage1"}
        leaked = [
            name
            for name in rejected
            if name in emitted or f"{name}_fp4" in emitted or f"{name}_fp8" in emitted
        ]
        self.assertEqual(leaked, [])

    def test_flydsl_v2_skips_block_m_16(self):
        m = self.gemm_moe_tune
        tasks = self.tuner.gen_flydsl_v2_2stages_task(
            self.info(
                model_dim=4096,
                inter_dim=1024,
                q_dtype_a=m.dtypes.fp4x2,
                q_dtype_w=m.dtypes.fp4x2,
            ),
            BLOCK_MS,
        )
        self.assertTrue(tasks)
        self.assertNotIn(16, {t[0][3] for t in tasks})

    def test_flydsl_v2_rejects_doweight_and_non_bf16_output(self):
        m = self.gemm_moe_tune
        gen = self.tuner.gen_flydsl_v2_2stages_task
        self.assertTrue(gen(self.info(), BLOCK_MS))
        self.assertEqual(gen(self.info(doweight_stage1=1), BLOCK_MS), [])
        self.assertEqual(gen(self.info(dtype=m.torch.float16), BLOCK_MS), [])

    def test_flydsl_v2_stage2_tile_n_must_divide_model_dim(self):
        # 5248 = 41 * 128: tile_n 128 divides it, tile_n 256 does not.
        def tile_ns(model_dim):
            tasks = self.tuner.gen_flydsl_v2_2stages_task(
                self.info(model_dim=model_dim), BLOCK_MS
            )
            # Stage2 run args end with the kernel params.
            return {t[4][-1]["tile_n"] for t in tasks if t[0][1] == "stage2"}

        self.assertEqual(tile_ns(5120), {128, 256})
        self.assertEqual(tile_ns(5248), {128})

    def test_default_tune_only_keeps_v2_without_stage2(self):
        # Current behavior, not a requirement: v2 stage1 has no inter_dim
        # filter, so with inter_dim % 128 != 0 the default mode keeps v2 alone
        # and submits stage1 with no stage2 to pair. TUNE_ONLY="" still
        # pairs other backends.
        m = self.gemm_moe_tune
        frame = self.untuned(
            token=32,
            model_dim=5120,
            inter_dim=1088,
            q_dtype_a=str(m.dtypes.fp8),
            q_dtype_w=str(m.dtypes.fp4x2),
            q_type="QuantType.per_1x32",
        )

        def submitted(env):
            got = {}

            def fake_mp(tasks, *args, **kwargs):
                got["tasks"] = list(tasks)
                return []

            cleaned = {k: v for k, v in os.environ.items() if k not in _TUNER_ENV}
            cleaned.update(env)
            with (
                mock.patch.object(self.gemm_moe_tune, "get_gfx", return_value="gfx950"),
                mock.patch.object(self.gemm_moe_tune, "mp_tuner", fake_mp),
                mock.patch.dict(os.environ, cleaned, clear=True),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.tuner.tune(frame, None, self.args)
            return got.get("tasks", [])

        def block_ms(tasks, stage):
            return {t[0][3] for t in tasks if t[0][1] == stage}

        default = submitted({})
        self.assertTrue(default)
        self.assertEqual({t[0][1] for t in default}, {"stage1"})
        self.assertTrue(all(t[0][5] == 1 for t in default))
        every = submitted({"TUNE_ONLY": ""})
        self.assertTrue(block_ms(every, "stage1") & block_ms(every, "stage2"))

    def test_grouped_candidate_names_follow_gate_and_tile(self):
        tuner = self.gemm_moe_tune.GroupedFmoeTuner(
            "groupedFmoe", KEYS, ["kernelName1", "us"], "grouped"
        )
        row = {
            "token": 100,
            "expert": 4,
            "topk": 8,
            "q_dtype_a": "torch.float4_e2m1fn_x2",
            "gate_mode": "GateMode.INTERLEAVE",
        }
        rows = tuner._candidate_rows(row)
        self.assertEqual([item["tile_m"] for item in rows], [16, 32, 64, 128])
        # max_m_raw = (100 * 8 + 4 - 1) // 4 = 200; tile 64 rounds it to 256.
        self.assertEqual(rows[2]["max_m"], 256)
        self.assertEqual(rows[2]["kernelName1"], "grouped_gemm1_fp4_gugu")
        self.assertEqual(rows[2]["kernelName2"], "grouped_gemm2_fp4")
        separated = tuner._candidate_row(
            {
                **row,
                "q_dtype_a": "torch.float8_e4m3fn",
                "gate_mode": "GateMode.SEPARATED",
            },
            16,
        )
        self.assertEqual(separated["kernelName1"], "grouped_gemm1_a8w4_gguu")

    def test_grouped_and_mxfp4_post_process_keep_watts(self):
        m = self.gemm_moe_tune
        cases = (
            m.GroupedFmoeTuner("groupedFmoe", KEYS, RESULT_LIST, "grouped"),
            m.Mxfp4FlydslTuner("mxfp4Fmoe", KEYS, RESULT_LIST, "mxfp4"),
        )
        for tuner in cases:
            with self.subTest(tuner.name):
                kept = tuner.post_process(
                    [{"us": 1.0, "kernelName1": "k", "watts": 3.5}], None
                )
                self.assertAlmostEqual(float(kept.iloc[0]["watts"]), 3.5)
                plain = tuner.post_process([{"us": 1.0, "kernelName1": "k"}], None)
                self.assertNotIn("watts", plain.columns)

    def test_mxfp4_finalist_stays_inside_the_latency_band(self):
        m = self.gemm_moe_tune

        def cand(name, us, err):
            return {
                "kernelName1": name,
                "kernelName2": name + "_s2",
                "us": us,
                "_err": err,
            }

        fast = cand("fast", 100.0, 0.05)
        close = cand("close", 100.5, 0.01)  # 0.5% from the fastest, inside 1%
        far = cand("far", 102.0, 0.0)  # 2% out, but the most accurate
        self.assertEqual(
            m._select_mxfp4_finalist([far, fast, close])["kernelName1"], "close"
        )
        percent = {
            "kernelName1": "pct",
            "kernelName2": "pct_s2",
            "us": 100.0,
            "err1": "1.0%",
        }
        raw = cand("raw", 100.0, 0.02)
        self.assertEqual(m._select_mxfp4_finalist([raw, percent])["kernelName1"], "pct")
        finalists = m._select_mxfp4_performance_finalists([far, fast, close], 1)
        self.assertEqual(finalists[0]["kernelName1"], "fast")
        # far is outside the band, so promotion is what keeps its accuracy.
        promoted = m._select_mxfp4_accuracy_promotion([fast, close, far], finalists)
        self.assertEqual(promoted["kernelName1"], "far")
        self.assertIsNone(m._select_mxfp4_accuracy_promotion([fast], finalists))

    def test_result_to_csv_migrates_gfx_drops_tags_and_keeps_the_last_row(self):
        columns = self.tuner.columns
        identity = {
            "cu_num": 304,
            "model_dim": 4096,
            "inter_dim": 1024,
            "expert": 8,
            "topk": 2,
            "act_type": "ActivationType.Silu",
            "dtype": "torch.bfloat16",
            "q_dtype_a": "torch.bfloat16",
            "q_dtype_w": "torch.bfloat16",
            "q_type": "QuantType.No",
            "use_g1u1": 1,
            "doweight_stage1": 0,
            "block_m": 32,
            "ksplit": 0,
            "us1": 1,
            "err1": "0.0%",
            "us2": 1,
            "kernelName2": "s2",
            "err2": "0.0%",
            "us": 2,
            "run_1stage": 0,
            "xbf16": 0,
            "tflops": 0,
            "bw": 0,
        }

        def legacy(token, kernel, tag):
            row = dict(identity)
            row.update(token=token, kernelName1=kernel, _tag=tag)
            return row

        updates = pd.DataFrame(
            [
                {
                    **identity,
                    "gfx": "gfx942",
                    "token": 32,
                    "kernelName1": "new_kernel",
                    "flat": 1,
                    "nt": 1,
                },
                {
                    **identity,
                    "gfx": "gfx950",
                    "cu_num": 256,
                    "token": 96,
                    "kernelName1": "added_first",
                    "flat": 0,
                    "nt": 0,
                },
                {
                    **identity,
                    "gfx": "gfx950",
                    "cu_num": 256,
                    "token": 96,
                    "kernelName1": "added_last",
                    "flat": 0,
                    "nt": 0,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / "tuned.csv")
            pd.DataFrame(
                [
                    legacy(32, "old_kernel", ""),
                    legacy(64, "tagged_kernel", "stale"),
                    legacy(7, "untouched", ""),
                ]
            ).to_csv(path, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                self.tuner.result_to_csv(updates[columns], path)
            saved = pd.read_csv(path, dtype=str)
        self.assertEqual(next(iter(saved.columns)), "gfx")
        self.assertNotIn("_tag", saved.columns)
        by_token = {row["token"]: row for _, row in saved.iterrows()}
        self.assertEqual(set(by_token), {"32", "7", "96"})
        self.assertEqual(by_token["32"]["gfx"], "gfx942")
        self.assertEqual(by_token["32"]["kernelName1"], "new_kernel")
        self.assertEqual(by_token["32"]["flat"], "1")
        self.assertEqual(by_token["7"]["gfx"], "gfx942")
        self.assertEqual(by_token["7"]["kernelName1"], "untouched")
        self.assertEqual(by_token["7"]["nt"], "0")
        self.assertEqual(by_token["7"]["flat"], "0")
        self.assertEqual(by_token["96"]["kernelName1"], "added_last")
        self.assertEqual(by_token["96"]["gfx"], "gfx950")
