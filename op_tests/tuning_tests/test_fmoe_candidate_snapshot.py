# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Snapshot of the candidates FmoeTuner.tune() submits, per shape and arch.

tune() runs for real, but mp_tuner is replaced, so no candidate is built or
timed. The task tags it would have timed are compared, in order, with
fmoe_candidate_snapshot.json.

Only the arch of the GPU in the machine is collected. GPU_ARCHS cannot stand
in for another arch: dtypes.fp8 and many other checks read the real GPU
through get_gfx_runtime(), so a gfx942 run on gfx950 mixes both. Each arch's
entry is therefore generated on its own node; the test skips an arch the
snapshot does not have yet.

After an intended candidate change, regenerate on each arch and review the diff:
    python3 op_tests/tuning_tests/test_fmoe_candidate_snapshot.py --regenerate
--regenerate rewrites only the entry of the current arch.
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

CU_NUM = {"gfx950": 256, "gfx942": 304}
# Removed from the child so the arch and CU count come from the real GPU.
_ARCH_ENV = ("GPU_ARCHS", "CU_NUM")

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
        CU_NUM[gfx],
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


def collect():
    """Return the GPU arch, and ordered task tags and per-backend counts for
    every shape of that arch."""
    import pandas as pd

    from aiter.jit.utils.chip_info import get_gfx, get_gfx_runtime
    from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

    gfx = get_gfx_runtime()
    if gfx not in CU_NUM:
        raise RuntimeError(f"no snapshot shapes for {gfx}; known: {list(CU_NUM)}")
    if get_gfx() != gfx:
        raise RuntimeError(f"get_gfx() is {get_gfx()} but the GPU is {gfx}")

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
    return {"gfx": gfx, "candidates": snapshot, "coverage": coverage}


def collect_in_subprocess():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "collected.json")
        # Exclusions are cached at module import, so clean the child environment
        # before importing the tuner, not only around tune().
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in _TUNER_ENV and k not in _ARCH_ENV
        }
        result = subprocess.run(
            [sys.executable, __file__, "--collect", out],
            env=env,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"collecting candidates exited {result.returncode}:\n"
                f"{result.stderr[-4000:]}"
            )
        with open(out) as f:
            data = json.load(f)
    return data.pop("gfx"), data


def validate_coverage(snapshot):
    """Require emitting and empty cases per backend across every arch in CU_NUM.

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
    def _baseline(self):
        if not SNAPSHOT.exists():
            self.skipTest(f"{SNAPSHOT.name} missing; run --regenerate on each arch")
        return json.loads(SNAPSHOT.read_text())

    def test_snapshot_covers_every_backend(self):
        baseline = self._baseline()
        missing = [gfx for gfx in CU_NUM if gfx not in baseline]
        if missing:
            self.skipTest(f"no snapshot entry yet for {missing}")
        validate_coverage(baseline)

    def test_candidates_match_snapshot_on_this_gpu(self):
        baseline = self._baseline()
        gfx, collected = collect_in_subprocess()
        if gfx not in baseline:
            self.skipTest(f"no {gfx} entry; run --regenerate on a {gfx} node")
        self.assertEqual(collected["coverage"], baseline[gfx]["coverage"])
        expected = baseline[gfx]["candidates"]
        actual = collected["candidates"]
        self.assertEqual(sorted(actual), sorted(expected), "snapshot keys changed")
        for key, tags in expected.items():
            with self.subTest(shape=key):
                self.assertEqual(actual[key], tags, _diff(tags, actual[key]))


def _regenerate():
    snapshot = json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else {}
    gfx, data = collect_in_subprocess()
    snapshot[gfx] = data
    missing = [arch for arch in CU_NUM if arch not in snapshot]
    if missing:
        print(f"backend coverage not checked: no entry yet for {missing}")
    else:
        validate_coverage(snapshot)
    with open(SNAPSHOT, "w") as f:
        json.dump(
            {arch: snapshot[arch] for arch in CU_NUM if arch in snapshot}, f, indent=1
        )
        f.write("\n")
    print(gfx, {k: len(v) for k, v in data["candidates"].items()})
    for backend in BACKENDS:
        emitting = sum(row[backend] > 0 for row in data["coverage"].values())
        empty = len(data["coverage"]) - emitting
        print(f"  {backend}: {emitting} emitting, {empty} empty")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--collect"]:
        with open(sys.argv[2], "w") as f:
            json.dump(collect(), f)
    elif sys.argv[1:] == ["--regenerate"]:
        _regenerate()
    else:
        unittest.main()
