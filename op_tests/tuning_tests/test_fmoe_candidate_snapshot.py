# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Snapshot of the candidates FmoeTuner.tune() submits, per shape and arch.

tune() runs for real, but mp_tuner is replaced, so no candidate is built or
timed. The task tags it would have timed are compared, in order, with
fmoe_candidate_snapshot.json.

Each arch is collected in its own process: get_gfx() reads GPU_ARCHS once and
caches it, and the CK kernel lists read it through a second copy of chip_info.
gemm_moe_tune.py needs torch and aiter to import, so run this on a ROCm node.

After an intended candidate change, regenerate the snapshot and review its diff:
    python3 op_tests/tuning_tests/test_fmoe_candidate_snapshot.py --regenerate
"""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

SNAPSHOT = Path(__file__).with_name("fmoe_candidate_snapshot.json")

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

ARCH_ENV = {
    "gfx950": {"GPU_ARCHS": "gfx950", "CU_NUM": "256"},
    "gfx942": {"GPU_ARCHS": "gfx942", "CU_NUM": "304"},
}

# Unset runs FlyDSL v2 alone when it has candidates; empty runs every backend.
TUNE_ONLY_MODES = {"default": None, "all": ""}
_TUNER_ENV = (
    "TUNE_ONLY",
    "OPUS_ONLY",
    "OPUS_SKIP_CKTILE",
    "TUNE_MOE_KERNEL_REGEX",
    "AITER_FMOE_TUNE_EXCLUDE_KERNELS",
)

GENERATORS = {
    "asm_1stage": "gen_1stage_asm_task",
    "asm_stage1": "gen_2stages_asm1_task",
    "ck": "gen_2stages_task",
    "flydsl": "gen_flydsl_2stages_task",
    "flydslv2": "gen_flydsl_v2_2stages_task",
    "flydsli4": "gen_flydsl_i4_2stages_task",
    "opus": "gen_opus_2stages_task",
}
BACKENDS = (*GENERATORS, "cktile")


def _shape(gfx, token, dims, act, q_type, q_dtype_a, q_dtype_w, doweight=0):
    model_dim, inter_dim, expert, topk = dims
    return (
        gfx,
        int(ARCH_ENV[gfx]["CU_NUM"]),
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        f"ActivationType.{act}",
        "torch.bfloat16",
        f"torch.{q_dtype_a}",
        f"torch.{q_dtype_w}",
        f"QuantType.{q_type}",
        1,
        doweight,
    )


BF16 = "bfloat16"
FP8 = "float8_e4m3fn"
FNUZ = "float8_e4m3fnuz"
FP4 = "float4_e2m1fn_x2"

# One shipped row per dtype / quant / activation group of the tuned fmoe CSVs,
# Coverage is checked after collection; these rows are not exhaustive.
# The opus a8w4 stage2 list is filtered by token, hence the token 1 and 2048 pair.
SHAPES = [
    # gfx950
    _shape("gfx950", 1, (6144, 512, 192, 8), "Silu", "No", BF16, BF16),
    _shape("gfx950", 1, (5120, 2560, 32, 8), "Silu", "per_Token", FP8, FP8),
    _shape("gfx950", 1, (5120, 1024, 128, 1), "Silu", "per_Token", FP8, FP8, 1),
    _shape("gfx950", 4096, (4096, 384, 128, 8), "Silu", "per_1x128", FP8, FP8),
    _shape("gfx950", 1, (2048, 512, 512, 10), "Gelu", "per_1x128", FP8, FP8),
    _shape("gfx950", 1, (7168, 256, 257, 9), "Silu", "per_1x32", FP4, FP4),
    _shape("gfx950", 256, (3072, 3072, 128, 4), "Swiglu", "per_1x32", FP4, FP4),
    _shape("gfx950", 1, (5120, 1152, 385, 7), "Silu", "per_1x32", FP8, FP4),
    _shape("gfx950", 2048, (5120, 1152, 385, 7), "Silu", "per_1x32", FP8, FP4),
    _shape("gfx950", 512, (3072, 3072, 128, 4), "Swiglu", "per_1x32", FP8, FP4),
    _shape("gfx950", 1, (3584, 512, 896, 16), "Situv2", "per_1x32", FP8, FP4),
    _shape("gfx950", 1, (6144, 512, 257, 9), "Silu", "per_1x32", FP8, FP8),
    _shape("gfx950", 1, (3072, 3072, 128, 4), "Swiglu", "per_1x32", BF16, FP4),
    _shape("gfx950", 1, (7168, 256, 384, 8), "Silu", "per_1x32", BF16, "int4"),
    # gfx942
    _shape("gfx942", 1, (4096, 384, 128, 8), "Silu", "No", BF16, BF16),
    _shape("gfx942", 2048, (2048, 192, 128, 8), "Gelu", "No", BF16, BF16, 1),
    _shape("gfx942", 1, (4096, 192, 128, 8), "Silu", "per_Token", FNUZ, FNUZ),
    _shape("gfx942", 1, (6144, 384, 128, 4), "Swiglu", "per_Token", FNUZ, FNUZ),
    _shape("gfx942", 1, (7168, 2048, 33, 9), "Silu", "per_1x128", FNUZ, FNUZ),
    _shape("gfx942", 2048, (4096, 384, 128, 8), "Silu", "per_Tensor", FNUZ, FNUZ),
    _shape("gfx942", 512, (6144, 4096, 8, 2), "Silu", "per_Tensor", "int8", "int8"),
    # tune() returns [] for per_1x32 outside gfx950.
    _shape("gfx942", 1, (3584, 384, 896, 16), "Situv2", "per_1x32", BF16, FP4),
]


def _shape_id(shape):
    return ",".join(str(v) for v in shape)


def _plain(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _tag(task):
    """The task tag without its info tuple: stage, kernel, block_m, extras."""
    _info, *rest = task[0]
    return [_plain(v) for v in rest]


def collect(gfx):
    """Return ordered task tags and per-backend counts for every shape of gfx."""
    import pandas as pd

    from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

    tuner = gemm_moe_tune.FmoeTuner("snapshotFmoe", KEYS, [], "candidate snapshot")
    with mock.patch.object(sys, "argv", ["gemm_moe_tune.py"]):
        args = tuner.parse_args()

    captured = []

    def fake_mp_tuner(tasks, *_args, **_kwargs):
        captured.extend(_tag(task) for task in tasks)
        return []

    base_env = {k: v for k, v in os.environ.items() if k not in _TUNER_ENV}
    snapshot = {}
    coverage = {}
    counts = {}

    def observe(backend, generate):
        def wrapped(*args, **kwargs):
            tasks = generate(*args, **kwargs)
            if backend == "ck":
                # gen_2stages_task delegates a8w4 to CK-tile. Attribute tasks
                # by their runner, rather than counting CK-tile as CK coverage.
                for task in tasks:
                    family = (
                        "cktile" if task[3].__name__.startswith("cktile_") else "ck"
                    )
                    counts[family] += 1
            else:
                counts[backend] += len(tasks)
            return tasks

        return wrapped

    for shape in SHAPES:
        if shape[0] != gfx:
            continue
        untunedf = pd.DataFrame([shape], columns=KEYS)
        for mode, tune_only in TUNE_ONLY_MODES.items():
            env = dict(base_env)
            if tune_only is not None:
                env["TUNE_ONLY"] = tune_only
            captured.clear()
            counts = dict.fromkeys(BACKENDS, 0)
            with (
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch.object(gemm_moe_tune, "mp_tuner", fake_mp_tuner),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.ExitStack() as patches,
            ):
                if mode == "all":
                    for backend, method in GENERATORS.items():
                        patches.enter_context(
                            mock.patch.object(
                                tuner, method, observe(backend, getattr(tuner, method))
                            )
                        )
                tuner.tune(untunedf, pd.DataFrame(columns=KEYS), args)
            snapshot[f"{_shape_id(shape)} TUNE_ONLY={mode}"] = list(captured)
            if mode == "all":
                coverage[_shape_id(shape)] = dict(counts)
    return {"candidates": snapshot, "coverage": coverage}


def collect_in_subprocess(gfx):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, f"{gfx}.json")
        # Exclusions are cached at module import, so clean the child environment
        # before importing the tuner, not only around tune().
        env = {k: v for k, v in os.environ.items() if k not in _TUNER_ENV}
        env.update(ARCH_ENV[gfx])
        result = subprocess.run(
            [sys.executable, __file__, "--collect", gfx, out],
            env=env,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"collecting {gfx} candidates exited {result.returncode}:\n"
                f"{result.stderr[-4000:]}"
            )
        with open(out) as f:
            return json.load(f)


def validate_coverage(snapshot):
    """Require emitting and empty cases per backend across the arch fixture.

    An empty case may be rejected by tune() before reaching a generator.
    This is a minimum coverage check, not proof of every internal branch.
    """
    rows = [
        counts for arch in snapshot.values() for counts in arch["coverage"].values()
    ]
    missing = []
    for backend in BACKENDS:
        if not any(row[backend] > 0 for row in rows):
            missing.append(f"{backend}: no case emits candidates")
        if not any(row[backend] == 0 for row in rows):
            missing.append(f"{backend}: no empty case")
    if missing:
        raise AssertionError("Incomplete backend coverage:\n" + "\n".join(missing))


def _diff(expected, actual):
    exp = Counter(json.dumps(t) for t in expected)
    act = Counter(json.dumps(t) for t in actual)
    missing = sorted((exp - act).elements())
    extra = sorted((act - exp).elements())
    if not missing and not extra:
        return "same candidates in a different order"
    return (
        f"{len(missing)} missing, {len(extra)} new\n"
        f"missing: {missing[:20]}\nnew: {extra[:20]}"
    )


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None, "gemm_moe_tune.py needs torch"
)
class TestFmoeCandidateSnapshot(unittest.TestCase):
    def _check(self, gfx):
        baseline = json.loads(SNAPSHOT.read_text())
        validate_coverage(baseline)
        collected = collect_in_subprocess(gfx)
        self.assertEqual(collected["coverage"], baseline[gfx]["coverage"])
        expected = baseline[gfx]["candidates"]
        actual = collected["candidates"]
        self.assertEqual(sorted(actual), sorted(expected), "snapshot keys changed")
        for key, tags in expected.items():
            with self.subTest(shape=key):
                self.assertEqual(actual[key], tags, _diff(tags, actual[key]))

    def test_gfx950(self):
        self._check("gfx950")

    def test_gfx942(self):
        self._check("gfx942")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--collect"]:
        gfx, out = sys.argv[2], sys.argv[3]
        with open(out, "w") as f:
            json.dump(collect(gfx), f)
    elif sys.argv[1:] == ["--regenerate"]:
        snapshot = {gfx: collect_in_subprocess(gfx) for gfx in ARCH_ENV}
        validate_coverage(snapshot)
        with open(SNAPSHOT, "w") as f:
            json.dump(snapshot, f, indent=1)
            f.write("\n")
        for gfx, data in snapshot.items():
            print(gfx, {k: len(v) for k, v in data["candidates"].items()})
            for backend in BACKENDS:
                rows = data["coverage"].values()
                emitting = sum(row[backend] > 0 for row in rows)
                print(
                    f"  {backend}: {emitting} emitting, {len(data['coverage']) - emitting} empty"
                )
    else:
        unittest.main()
