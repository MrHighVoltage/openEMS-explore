# -*- coding: utf-8 -*-
"""GPU-vs-CPU regression tests for the Vulkan FDTD engine.

Each test builds the *same* small geometry twice -- once run with
engine="gpu" and once with engine="avx2-multithreaded" -- and asserts that a
recorded probe time series agrees within a relative-error tolerance chosen to
absorb floating-point accumulation-order differences (not bit-identical
across engines).

These target the specific extension/feature paths a prior audit found had no
GPU test coverage anywhere in the repo: UPML (fused Yee+PML kernel), lumped
RLC, TF/SF excitation, GPU-side steady-state detection, dispersive
conducting-sheet material, and non-voltage probe types (current, E-field,
H-field -- previously only p_type=0 voltage probes were ever exercised on
GPU).

Requires a local build with -DWITH_GPU=ON and a usable Vulkan device; the
whole module is skipped otherwise (see gpu_test_utils.gpu_engine_available).
"""

import os
import sys
import re
import tempfile
import unittest

import numpy as np


def _bootstrap_local_openems_runtime():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    python_dir = os.path.join(repo_root, "python")
    build_dir = os.path.join(repo_root, "build")

    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)

    libopenems = os.path.join(build_dir, "libopenEMS.so")
    if not os.path.exists(libopenems):
        return

    paths = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    if paths and os.path.abspath(paths[0]) == os.path.abspath(build_dir):
        return

    paths = [p for p in paths if os.path.abspath(p) != os.path.abspath(build_dir)]
    os.environ["LD_LIBRARY_PATH"] = ":".join([build_dir] + paths)
    os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)


_bootstrap_local_openems_runtime()

sys.path.insert(0, os.path.dirname(__file__))

from CSXCAD import ContinuousStructure  # noqa: E402

from openEMS import openEMS  # noqa: E402

import gpu_test_utils as gtu  # noqa: E402

CPU_ENGINE = "avx2-multithreaded"

# Small cube grid shared by every scenario; kept tiny so gpu+cpu pairs run in
# well under a minute each.
N = 20
UNIT = 1e-3  # mm


def _new_domain(NrTS, boundary, excite_freq=None, sinus_freq=None):
    FDTD = openEMS(NrTS=NrTS, EndCriteria=1e-300)
    if sinus_freq is not None:
        FDTD.SetSinusExcite(sinus_freq)
    else:
        f0, fc = excite_freq
        FDTD.SetGaussExcite(f0, fc)
    FDTD.SetBoundaryCond(boundary)

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(UNIT)
    mesh.AddLine("x", list(range(N + 1)))
    mesh.AddLine("y", list(range(N + 1)))
    mesh.AddLine("z", list(range(N + 1)))
    return FDTD, CSX, mesh


def _new_graded_domain(NrTS, boundary, excite_freq, ratio=1.02):
    """Same cube, but with geometrically graded cell sizes.

    Each cell is `ratio` times its predecessor, which makes the per-cell
    operator coefficients (they scale with the local edge length and area)
    almost all distinct. That is what drives Operator_Vulkan::CompressOperator
    into its one-entry-per-cell mode, so these are the meshes that exercise the
    OPIDX_BITS == 0 shader path. The grading is geometric rather than a power
    law so no cell is degenerately thin -- that would shrink the timestep to
    the point where nothing propagates in the timesteps we run.
    """
    FDTD = openEMS(NrTS=NrTS, EndCriteria=1e-300)
    f0, fc = excite_freq
    FDTD.SetGaussExcite(f0, fc)
    FDTD.SetBoundaryCond(boundary)

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(UNIT)
    steps = np.cumsum(np.concatenate([[0.0], ratio ** np.arange(N)]))
    for axis in ("x", "y", "z"):
        mesh.AddLine(axis, list(steps))
    return FDTD, CSX, mesh, steps


def _exc_box(steps):
    """Source spanning one cell in z, centred in the grid.

    It has to span a cell rather than being a single point: on a graded mesh
    the coordinates are floats, and a zero-volume box does not reliably snap to
    a mesh line -- openEMS then reports "Unused primitive" and excites nothing,
    which would make every comparison below pass vacuously.
    """
    mid = N // 2
    return ([steps[mid]] * 3, [steps[mid], steps[mid], steps[mid + 1]])


def _probe_box(steps):
    """Voltage probe two cells from the source, where the signal is strong."""
    mid = N // 2
    return ([steps[mid], steps[mid], steps[mid + 1]],
            [steps[mid], steps[mid], steps[mid + 3]])


def _run_and_read_probe(engine, sim_tag, build_fn, probe_name, extra_opts=None):
    """build_fn(FDTD, CSX, mesh) -> None; adds excitation/extension/probe.

    Returns the probe's recorded columns (time excluded) as a flat float array.
    """
    sim_path = os.path.join(tempfile.gettempdir(), "gpu_regress_{}_{}".format(sim_tag, engine))
    FDTD, CSX, mesh = build_fn()
    run_opts = {"cleanup": True, "engine": engine}
    if extra_opts:
        run_opts.update(extra_opts)
    FDTD.Run(sim_path, **run_opts)

    _, data = gtu.load_probe_file(os.path.join(sim_path, probe_name))
    return data[:, 1:].astype(np.float64).ravel()


@unittest.skipUnless(gtu.gpu_engine_available(), "GPU engine not available in this build")
class Test_GPU_vs_CPU(unittest.TestCase):
    """Compare Engine_Vulkan against the AVX2 multithreaded CPU engine."""

    def _compare(self, sim_tag, build_fn, probe_name, *, rtol=2e-2):
        gpu_vals = _run_and_read_probe("gpu", sim_tag, build_fn, probe_name)
        cpu_vals = _run_and_read_probe(CPU_ENGINE, sim_tag, build_fn, probe_name)
        gtu.assert_engines_agree(gpu_vals, cpu_vals, rtol=rtol, label=sim_tag)

    def test_pml_waveguide_probe(self):
        """Fused Yee+UPML kernel: PML on all sides, voltage probe near a boundary."""
        def build():
            FDTD, CSX, mesh = _new_domain(NrTS=250, boundary=["PML_8"] * 6,
                                           excite_freq=(2e9, 1e9))
            exc = CSX.AddExcitation("exc", exc_type=0, exc_val=[0, 0, 1])
            exc.AddBox([N // 2, N // 2, N // 2], [N // 2, N // 2, N // 2])
            probe = CSX.AddProbe("probe_v.dat", p_type=0)
            probe.AddBox([N // 2, N // 2, 2], [N // 2, N // 2, 4])
            return FDTD, CSX, mesh
        self._compare("pml", build, "probe_v.dat")

    def test_lumped_rlc(self):
        """Series RLC lumped element path."""
        def build():
            FDTD, CSX, mesh = _new_domain(NrTS=300, boundary=["MUR"] * 6,
                                           excite_freq=(1e9, 0.5e9))
            exc = CSX.AddExcitation("exc", exc_type=0, exc_val=[0, 0, 1])
            exc.AddBox([N // 2, N // 2, 3], [N // 2, N // 2, 3])
            rlc = CSX.AddLumpedElement("ser_rlc", ny="z", caps=False,
                                        R=50.0, L=10e-9, C=2e-12, LEtype=1)
            rlc.AddBox([N // 2, N // 2, N // 2 - 1], [N // 2, N // 2, N // 2 + 1], priority=20)
            probe = CSX.AddProbe("probe_v.dat", p_type=0)
            probe.AddBox([N // 2, N // 2, N // 2 - 1], [N // 2, N // 2, N // 2 + 1])
            return FDTD, CSX, mesh
        self._compare("rlc", build, "probe_v.dat")

    def test_tfsf_field_probes(self):
        """TF/SF plane-wave injection, checked with both E- and H-field probes."""
        def build():
            FDTD, CSX, mesh = _new_domain(NrTS=250, boundary=["MUR"] * 6,
                                           excite_freq=(2e9, 1e9))
            exc = CSX.AddExcitation("tfsf_exc", exc_type=10, exc_val=[0, 0, 1])
            exc.SetPropagationDir([1, 0, 0])
            exc.AddBox([5, 5, 5], [N - 5, N - 5, N - 5])
            eprobe = CSX.AddProbe("probe_e.dat", p_type=2)
            eprobe.AddBox([N // 2, N // 2, N // 2], [N // 2, N // 2, N // 2])
            hprobe = CSX.AddProbe("probe_h.dat", p_type=3)
            hprobe.AddBox([N // 2, N // 2, N // 2], [N // 2, N // 2, N // 2])
            return FDTD, CSX, mesh

        # The excitation pulse is still on its rising edge at the end of this
        # short window (only enough timesteps for a quick smoke-level run),
        # so the last sample is where GPU/CPU injection-timing differences of
        # a few sub-timesteps show up largest as relative error -- loosen the
        # tolerance accordingly rather than lengthening the run.
        gpu_e = _run_and_read_probe("gpu", "tfsf", build, "probe_e.dat")
        cpu_e = _run_and_read_probe(CPU_ENGINE, "tfsf", build, "probe_e.dat")
        gtu.assert_engines_agree(gpu_e, cpu_e, rtol=2e-2, atol_frac=0.15, label="tfsf E-field")

        gpu_h = _run_and_read_probe("gpu", "tfsf", build, "probe_h.dat")
        cpu_h = _run_and_read_probe(CPU_ENGINE, "tfsf", build, "probe_h.dat")
        gtu.assert_engines_agree(gpu_h, cpu_h, rtol=2e-2, atol_frac=0.15, label="tfsf H-field")

    def test_steady_state_sinusoidal(self):
        """Periodic excitation auto-enables GPU-side steady-state sampling."""
        def build():
            FDTD, CSX, mesh = _new_domain(NrTS=600, boundary=["MUR"] * 6,
                                           sinus_freq=1.5e9)
            exc = CSX.AddExcitation("exc", exc_type=0, exc_val=[0, 0, 1])
            exc.AddBox([N // 2, N // 2, 3], [N // 2, N // 2, 3])
            probe = CSX.AddProbe("probe_v.dat", p_type=0)
            probe.AddBox([N // 2, N // 2, N // 2 - 1], [N // 2, N // 2, N // 2 + 1])
            return FDTD, CSX, mesh
        self._compare("steadystate", build, "probe_v.dat", rtol=3e-2)

    def test_conducting_sheet_material(self):
        """Dispersive conducting-sheet material, checked with a current probe."""
        def build():
            FDTD, CSX, mesh = _new_domain(NrTS=250, boundary=["PML_8"] * 6,
                                           excite_freq=(2e9, 1e9))
            exc = CSX.AddExcitation("exc", exc_type=0, exc_val=[0, 0, 1])
            exc.AddBox([N // 2, N // 2, 3], [N // 2, N // 2, 3])
            sheet = CSX.AddConductingSheet("sheet", conductivity=3.5e7, thickness=35e-6)
            sheet.AddBox([2, 2, N // 2], [N - 2, N - 2, N // 2])
            probe = CSX.AddProbe("probe_i.dat", p_type=1, norm_dir=2)
            probe.AddBox([N // 2 - 1, N // 2, N // 2], [N // 2 + 1, N // 2, N // 2])
            return FDTD, CSX, mesh
        self._compare("condsheet", build, "probe_i.dat")


    def test_graded_mesh_pml(self):
        """Graded mesh + PML: the one-entry-per-cell operator path, vs CPU."""
        def build():
            FDTD, CSX, mesh, steps = _new_graded_domain(
                NrTS=250, boundary=["PML_8"] * 6, excite_freq=(2e9, 1e9))
            exc = CSX.AddExcitation("exc", exc_type=0, exc_val=[0, 0, 1])
            exc.AddBox(_exc_box(steps)[0], _exc_box(steps)[1])
            probe = CSX.AddProbe("probe_v.dat", p_type=0)
            probe.AddBox(_probe_box(steps)[0], _probe_box(steps)[1])
            return FDTD, CSX, mesh
        self._compare("graded_pml", build, "probe_v.dat")

    def test_graded_mesh_pec(self):
        """Graded mesh in a closed PEC box, vs CPU."""
        def build():
            FDTD, CSX, mesh, steps = _new_graded_domain(
                NrTS=250, boundary=[0] * 6, excite_freq=(2e9, 1e9))
            exc = CSX.AddExcitation("exc", exc_type=0, exc_val=[0, 0, 1])
            exc.AddBox(_exc_box(steps)[0], _exc_box(steps)[1])
            probe = CSX.AddProbe("probe_v.dat", p_type=0)
            probe.AddBox(_probe_box(steps)[0], _probe_box(steps)[1])
            return FDTD, CSX, mesh
        self._compare("graded_pec", build, "probe_v.dat")

@unittest.skipUnless(gtu.gpu_engine_available(), "GPU engine not available in this build")
class Test_GPU_OperatorCompression(unittest.TestCase):
    """Operator_Vulkan::CompressOperator picks between two storage modes.

    Deduplicating the per-cell coefficients is a large win while the unique
    table stays cache-resident, and a large loss once it does not: the shaders
    reach the coefficients through six independent gathers per cell, so an
    oversized table turns each cell into six scattered cache lines. Graded
    meshes make nearly every cell unique and used to hit exactly that case.
    The operator therefore falls back to storing one entry per cell, with the
    per-cell index dropped entirely (OPIDX_BITS == 0).

    Which mode is chosen must never change the numbers -- the two hold the same
    coefficients in different order -- so these tests force each mode via
    OPENEMS_GPU_COMPRESS_MAX_TABLE_KB and require bit-identical results.
    """

    def _build_graded(self, boundary):
        FDTD, CSX, mesh, steps = _new_graded_domain(
            NrTS=250, boundary=boundary, excite_freq=(2e9, 1e9))
        exc = CSX.AddExcitation("exc", exc_type=0, exc_val=[0, 0, 1])
        exc.AddBox(_exc_box(steps)[0], _exc_box(steps)[1])
        probe = CSX.AddProbe("probe_v.dat", p_type=0)
        probe.AddBox(_probe_box(steps)[0], _probe_box(steps)[1])
        return FDTD, CSX, mesh

    def _run(self, tag, boundary, max_table_kb):
        """Run on the GPU with the table budget forced, returning (values, log)."""
        prev = os.environ.get("OPENEMS_GPU_COMPRESS_MAX_TABLE_KB")
        os.environ["OPENEMS_GPU_COMPRESS_MAX_TABLE_KB"] = str(max_table_kb)
        try:
            with gtu.capture_c_stdout() as captured:
                vals = _run_and_read_probe(
                    "gpu", tag, lambda: self._build_graded(boundary), "probe_v.dat")
            return vals, captured.read().decode("utf-8", errors="replace")
        finally:
            if prev is None:
                os.environ.pop("OPENEMS_GPU_COMPRESS_MAX_TABLE_KB", None)
            else:
                os.environ["OPENEMS_GPU_COMPRESS_MAX_TABLE_KB"] = prev

    def _assert_modes_agree(self, tag, boundary):
        # 0 KB forces per-cell storage; a huge budget forces deduplication.
        percell, percell_log = self._run(tag + "_percell", boundary, 0)
        dedup, dedup_log = self._run(tag + "_dedup", boundary, 1 << 30)

        # Guard the guard: if the plumbing ever stopped honouring the budget,
        # both runs would take the same path and agree for the wrong reason.
        self.assertIn("Uncompressed operator", percell_log,
                      "expected the per-cell operator path to be selected")
        self.assertIn("Compressed operator", dedup_log,
                      "expected the deduplicated operator path to be selected")

        self.assertEqual(percell.shape, dedup.shape)
        self.assertGreater(float(np.abs(dedup).max()), 0.0,
                           "probe recorded nothing; the comparison would be vacuous")
        # Same coefficients, only a different storage order -> bit-identical.
        self.assertTrue(
            np.array_equal(percell, dedup),
            "{}: per-cell and deduplicated operators disagree (max abs diff {:.3g})".format(
                tag, float(np.abs(percell - dedup).max())))

    def test_modes_agree_pml(self):
        self._assert_modes_agree("cmp_pml", ["PML_8"] * 6)

    def test_modes_agree_pec(self):
        self._assert_modes_agree("cmp_pec", [0] * 6)

    def test_graded_mesh_defeats_deduplication(self):
        """The premise of the fallback: grading makes coefficients near-unique.

        This is the property that decides the storage mode, and unlike the mode
        itself it does not depend on grid size -- the 20^3 grid used here is far
        too small for any table to exceed the budget, while the production grids
        that motivated the fallback are not.
        """
        def uniques(build_fn, tag):
            prev = os.environ.pop("OPENEMS_GPU_COMPRESS_MAX_TABLE_KB", None)
            try:
                with gtu.capture_c_stdout() as captured:
                    _run_and_read_probe("gpu", tag, build_fn, "probe_v.dat")
                log = captured.read().decode("utf-8", errors="replace")
            finally:
                if prev is not None:
                    os.environ["OPENEMS_GPU_COMPRESS_MAX_TABLE_KB"] = prev
            match = re.search(r"Compressed operator: (\d+) unique", log)
            self.assertIsNotNone(match, "no compression line in output for " + tag)
            return int(match.group(1))

        def uniform_build():
            FDTD, CSX, mesh = _new_domain(NrTS=100, boundary=[0] * 6,
                                          excite_freq=(2e9, 1e9))
            exc = CSX.AddExcitation("exc", exc_type=0, exc_val=[0, 0, 1])
            exc.AddBox([N // 2, N // 2, N // 2], [N // 2, N // 2, N // 2 + 1])
            probe = CSX.AddProbe("probe_v.dat", p_type=0)
            probe.AddBox([N // 2, N // 2, N // 2 + 1], [N // 2, N // 2, N // 2 + 3])
            return FDTD, CSX, mesh

        cells = (N + 1) ** 3
        graded = uniques(lambda: self._build_graded([0] * 6), "uniq_graded")
        uniform = uniques(uniform_build, "uniq_uniform")

        self.assertGreater(graded, 0.5 * cells,
                           "graded mesh deduplicated to {} of {} cells; the "
                           "per-cell fallback would never be needed".format(graded, cells))
        self.assertLess(uniform, 64,
                        "uniform mesh produced {} unique sets".format(uniform))

    def test_uniform_mesh_keeps_deduplication(self):
        """A uniform mesh collapses to a handful of sets and must stay compressed."""
        def build():
            FDTD, CSX, mesh = _new_domain(NrTS=100, boundary=[0] * 6,
                                          excite_freq=(2e9, 1e9))
            exc = CSX.AddExcitation("exc", exc_type=0, exc_val=[0, 0, 1])
            exc.AddBox([N // 2, N // 2, N // 2], [N // 2, N // 2, N // 2])
            probe = CSX.AddProbe("probe_v.dat", p_type=0)
            probe.AddBox([N // 2, N // 2, 2], [N // 2, N // 2, 4])
            return FDTD, CSX, mesh

        prev = os.environ.pop("OPENEMS_GPU_COMPRESS_MAX_TABLE_KB", None)
        try:
            with gtu.capture_c_stdout() as captured:
                _run_and_read_probe("gpu", "cmp_uniform", build, "probe_v.dat")
            log = captured.read().decode("utf-8", errors="replace")
        finally:
            if prev is not None:
                os.environ["OPENEMS_GPU_COMPRESS_MAX_TABLE_KB"] = prev
        self.assertIn("Compressed operator", log)


@unittest.skipUnless(gtu.gpu_engine_available(), "GPU engine not available in this build")
class Test_GPU_UPMLCoefficientTables(unittest.TestCase):
    """The UPML coefficients are deduplicated into shared tables + a narrow index.

    They follow the grading profile (a function of depth into the PML) and the
    local material, so they hardly vary across the transverse plane: an 80^3
    PML_8 grid has 821016 coefficient entries and a few hundred distinct
    tuples, and mesh grading barely changes that -- unlike the Yee operator
    table (see Test_GPU_OperatorCompression). Deduplicating them turns ~19 MB
    of per-timestep coefficient traffic into ~1.6 MB of index.

    The same tables and index feed both the fused Yee+UPML kernels and the
    separate-dispatch fallback, so the two must still agree exactly.
    """

    def _build(self, boundary, graded):
        if graded:
            FDTD, CSX, mesh, steps = _new_graded_domain(
                NrTS=250, boundary=boundary, excite_freq=(2e9, 1e9))
            exc_lo, exc_hi = _exc_box(steps)
            pr_lo, pr_hi = _probe_box(steps)
        else:
            FDTD, CSX, mesh = _new_domain(NrTS=250, boundary=boundary,
                                          excite_freq=(2e9, 1e9))
            mid = N // 2
            exc_lo, exc_hi = [mid] * 3, [mid, mid, mid + 1]
            pr_lo, pr_hi = [mid, mid, mid + 1], [mid, mid, mid + 3]
        exc = CSX.AddExcitation("exc", exc_type=0, exc_val=[0, 0, 1])
        exc.AddBox(exc_lo, exc_hi)
        probe = CSX.AddProbe("probe_v.dat", p_type=0)
        probe.AddBox(pr_lo, pr_hi)
        return FDTD, CSX, mesh

    def _run(self, tag, boundary, graded, fused, extra_opts=None):
        prev = os.environ.get("OPENEMS_GPU_DISABLE_FUSED_UPML")
        if fused:
            os.environ.pop("OPENEMS_GPU_DISABLE_FUSED_UPML", None)
        else:
            os.environ["OPENEMS_GPU_DISABLE_FUSED_UPML"] = "1"
        try:
            with gtu.capture_c_stdout() as captured:
                vals = _run_and_read_probe(
                    "gpu", tag, lambda: self._build(boundary, graded),
                    "probe_v.dat", extra_opts=extra_opts)
            return vals, captured.read().decode("utf-8", errors="replace")
        finally:
            os.environ.pop("OPENEMS_GPU_DISABLE_FUSED_UPML", None)
            if prev is not None:
                os.environ["OPENEMS_GPU_DISABLE_FUSED_UPML"] = prev

    def _assert_paths_agree(self, tag, graded):
        fused, fused_log = self._run(tag + "_fused", ["PML_8"] * 6, graded, fused=True)
        sep, sep_log = self._run(tag + "_sep", ["PML_8"] * 6, graded, fused=False)

        # Guard the guard: without this, a change that silently stopped
        # honouring the switch would make both runs the same run.
        self.assertIn("Fused Yee+UPML kernels active", fused_log)
        self.assertIn("fused UPML disabled by environment", sep_log)

        self.assertEqual(fused.shape, sep.shape)
        self.assertGreater(float(np.abs(fused).max()), 0.0,
                           "probe recorded nothing; the comparison would be vacuous")
        # Same tables, same index, same arithmetic -> bit-identical.
        self.assertTrue(
            np.array_equal(fused, sep),
            "{}: fused and separate-dispatch UPML disagree (max abs diff {:.3g})".format(
                tag, float(np.abs(fused - sep).max())))

    def test_fused_matches_separate_dispatch_uniform(self):
        self._assert_paths_agree("upmltab_uniform", graded=False)

    def test_fused_matches_separate_dispatch_graded(self):
        self._assert_paths_agree("upmltab_graded", graded=True)

    def test_coefficients_are_deduplicated(self):
        """The premise: PML coefficients collapse to a small shared table."""
        _, log = self._run("upmltab_stats", ["PML_8"] * 6, graded=True,
                           fused=True, extra_opts={"verbose": 1})
        match = re.search(r"UPML coefficients: (\d+) entries -> (\d+) unique", log)
        self.assertIsNotNone(match, "no UPML coefficient table line in output")
        entries, unique = int(match.group(1)), int(match.group(2))
        self.assertGreater(entries, 0)
        # Even this 20^3 grid should collapse by well over an order of
        # magnitude; production grids reach several hundred times.
        self.assertLess(unique * 10, entries,
                        "UPML coefficients barely deduplicated: {} entries -> {} unique".format(
                            entries, unique))


if __name__ == "__main__":
    unittest.main()
