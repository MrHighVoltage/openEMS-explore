# -*- coding: utf-8 -*-
"""Regression tests for Engine_AVX2_Multithread's trapezoidal temporal blocking.

The schedule in FDTD/engine_avx2_multithread.cpp (OPENEMS_AVX2_TEMPORAL_BLOCK)
advances a cache-resident tile of x-lines through k timesteps before moving
on, instead of streaming the whole grid once per timestep. It is claimed to
be bit-identical to the flat sweep -- but only for extensions that implement
the slab hooks in Engine_Extension (SupportsSlabApply() / *Slab()); anything
else makes ConfigureTemporalBlocking() refuse to engage and fall back to the
flat sweep, printing:

    AVX2 temporal blocking: disabled, extension '<name>' cannot be applied
    per x-range.

instead of, when it does engage:

    AVX2 temporal blocking ACTIVE (prototype): k=..., tile width ... x-lines,
    ... MB per tile

A comparison test that only checks "flat run == blocked run" would pass
*vacuously* whenever blocking silently declines to engage -- of course a
sweep is identical to itself. Every case here therefore greps the blocked
run's stdout for the ACTIVE line before trusting the field comparison; if
blocking did not engage, the test fails with a message naming whichever
extension the engine refused (or reports some other reason, e.g. the grid
still fitting cache). That failure is itself the useful signal while slab
support is still being added extension-by-extension: it tells you which one
is still missing, rather than reporting a green, meaningless test.

Each model is written to XML by a Python subprocess and simulated by the
openEMS binary of the build (OPENEMS_TEST_BIN, default build/openEMS), so the
engine's stdout -- a C++-level side effect of ConfigureTemporalBlocking(), not
something the Python bindings surface -- can be captured cleanly, and so
OPENEMS_AVX2_TEMPORAL_BLOCK can be set for one run and not the other without
disturbing this process's own environment. The Python modules only write the
XML, with whatever library they were built against (OPENEMS_TEST_PYTHON picks
another interpreter): running the build's library under modules built for
another one fails, or worse, works with a different class layout.

A sine excitation adds steady-state detection, which the blocked schedule
carries since the energy of a check is summed per slab (plans/STEADY_STATE_
TEMPORAL_BLOCKING.md). The steady-state cases (Test10-12) run a lossy model
until it stops on its own and compare more than the fields: the trace of
every check (OPENEMS_STEADYSTATE_TRACE=1: timestep, energy, verdict, all
digits) and the timestep the run stopped at must be identical too.

The grid is 64 x-lines split into four 16-line tiles (OPENEMS_AVX2_TEMPORAL_
BLOCK=4:16, i.e. k=4, W=16, satisfying the engine's hard W>=2k constraint),
small enough to finish in well under a second. Left alone, the two
performance guards in ConfigureTemporalBlocking (grid too small relative to
cache, not enough x-lines for two tiles) would refuse to engage on a grid
this size -- that is the right call for real runs, but it would make this
test file impossible to run in CI in a reasonable time. OPENEMS_AVX2_
TEMPORAL_BLOCK_FORCE=1 lifts exactly those two guards (not the extension
-support guard, which is what is actually under test here), so the schedule
can be exercised, correctly, on a CI-sized grid. Every object placed in the
grid straddles the tile boundary at x=48 on purpose, so a wedge actually has
to reconcile something other than vacuum.
"""

import os
import re
import sys
import shutil
import subprocess
import tempfile
import textwrap
import unittest

import h5py
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OPENEMS_BIN = os.environ.get("OPENEMS_TEST_BIN", os.path.join(REPO_ROOT, "build", "openEMS"))
XML_PYTHON = os.environ.get("OPENEMS_TEST_PYTHON", sys.executable)

# ---------------------------------------------------------------------------
# Grid shared by every case: 64 x-lines, four 16-line tiles.
# ---------------------------------------------------------------------------
NX, NY, NZ = 64, 32, 32
TIMESTEPS = 120
BLOCK_ENV = "4:16"          # k=4, W=16 -> 4 tiles across the grid
THREADS = 2                 # fixed, so a comparison is not also a thread-count change
EXC_X = 20                  # excitation plane, inside tile 1 ([16,32))

# Every extra object straddles the tile boundary at x=48 so that a wedge has
# to reconcile it, not just vacuum either side.
OBJ_X_LO, OBJ_X_HI = 38, 54
LUMPED_X_LO, LUMPED_X_HI = 46, 50


# The excitation and end criteria of the cases: a Gaussian pulse for a fixed
# number of timesteps, or a sine until the steady state is reached.
GAUSS_SETUP = "FDTD = openEMS(NrTS=%d, EndCriteria=0)\nFDTD.SetGaussExcite(5e9, 5e9)" % TIMESTEPS
SINE_SETUP = "FDTD = openEMS(NrTS=20000, EndCriteria=1e-3)\nFDTD.SetSinusExcite(5e9)"

# The steady-state cases fill the grid with a lossy dielectric, so that the
# sine settles within about a thousand timesteps (twelve checks). With 1 mm
# cells the period is 74 timesteps and the probe and dump interval 9, which
# does not divide it; with 0.925 mm cells they are 80 and 10, which does.
LOSSY_SETUP = """
lossy = CSX.AddMaterial("lossy", epsilon=2.0, kappa=0.1)
lossy.AddBox([xl[0], yl[0], zl[0]], [xl[-1], yl[-1], zl[-1]], priority=1)
"""
DELTA_P74, DELTA_P80 = 1e-3, 0.925e-3

_SCRIPT_TEMPLATE = """
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS

{fdtd_setup}
FDTD.SetBoundaryCond({boundary!r})

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit({delta})
mesh.SetLines("x", np.arange({nx}, dtype=float))
mesh.SetLines("y", np.arange({ny}, dtype=float))
mesh.SetLines("z", np.arange({nz}, dtype=float))
xl, yl, zl = mesh.GetLines("x"), mesh.GetLines("y"), mesh.GetLines("z")

{extra_setup}

exc = CSX.AddExcitation("exc", exc_type=0, exc_val=[0, 0, 1])
exc.AddBox([xl[{exc_x}], yl[10], zl[10]], [xl[{exc_x}] + 1, yl[22], zl[22]])

dumpE = CSX.AddDump("Et", dump_type=0, dump_mode=0, file_type=1)
dumpE.AddBox([xl[0], yl[0], zl[0]], [xl[-1], yl[-1], zl[-1]])
dumpH = CSX.AddDump("Ht", dump_type=1, dump_mode=0, file_type=1)
dumpH.AddBox([xl[0], yl[0], zl[0]], [xl[-1], yl[-1], zl[-1]])

FDTD.Write2XML({xml!r})
"""


def _run(tag, boundary, extra_setup, blocked, fdtd_setup=GAUSS_SETUP, delta=1e-3):
    """Write one small model to XML and run it on the avx2-multithreaded engine.

    Returns (stdout, E_time_series, H_time_series). `blocked` selects whether
    OPENEMS_AVX2_TEMPORAL_BLOCK[_FORCE] are set for this particular run --
    the two runs being compared must differ in *only* that.
    """
    sim_path = os.path.join(tempfile.gettempdir(), "tblk_%s" % tag)
    shutil.rmtree(sim_path, ignore_errors=True)
    os.makedirs(sim_path)
    xml = os.path.join(sim_path, "model.xml")

    script = _SCRIPT_TEMPLATE.format(
        fdtd_setup=fdtd_setup,
        delta=delta,
        boundary=boundary,
        nx=NX, ny=NY, nz=NZ,
        extra_setup=textwrap.dedent(extra_setup).strip(),
        exc_x=EXC_X,
        xml=xml,
    )
    script_path = os.path.join(sim_path, "model.py")
    with open(script_path, "w") as f:
        f.write(script)
    proc = subprocess.run([XML_PYTHON, script_path], cwd=sim_path, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0 or not os.path.exists(xml):
        shutil.rmtree(sim_path, ignore_errors=True)
        raise RuntimeError("writing the model failed (tag=%s, rc=%s):\n%s\n%s"
                           % (tag, proc.returncode, proc.stdout[-4000:], proc.stderr[-4000:]))

    env = dict(os.environ)
    env.pop("OPENEMS_AVX2_TEMPORAL_BLOCK", None)
    env.pop("OPENEMS_AVX2_TEMPORAL_BLOCK_FORCE", None)
    env["OPENEMS_STEADYSTATE_TRACE"] = "1"
    if blocked:
        env["OPENEMS_AVX2_TEMPORAL_BLOCK"] = BLOCK_ENV
        env["OPENEMS_AVX2_TEMPORAL_BLOCK_FORCE"] = "1"

    proc = subprocess.run(
        [OPENEMS_BIN, xml, "--engine=avx2-multithreaded", "--numThreads=%d" % THREADS],
        cwd=sim_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    try:
        if proc.returncode != 0:
            raise RuntimeError(
                "simulation failed (tag=%s, rc=%s):\n%s\n%s"
                % (tag, proc.returncode, proc.stdout[-4000:], proc.stderr[-4000:])
            )

        def _load(name):
            with h5py.File(os.path.join(sim_path, name), "r") as f:
                td = f["FieldData/TD"]
                return np.array([np.array(td[k]) for k in sorted(td.keys())])

        E = _load("Et.h5")
        H = _load("Ht.h5")
    finally:
        shutil.rmtree(sim_path, ignore_errors=True)

    return proc.stdout + proc.stderr, E, H


def _relevant_lines(stdout):
    """The one or two lines out of the whole run's stdout that actually say
    what ConfigureTemporalBlocking() decided -- for failure messages."""
    lines = [l for l in stdout.splitlines() if "temporal blocking" in l]
    return "\n".join(lines) if lines else "(no 'temporal blocking' line in stdout at all)"


def _steady_trace(stdout):
    """Every steady-state check of a run, as OPENEMS_STEADYSTATE_TRACE prints it."""
    return [l for l in stdout.splitlines() if l.startswith("SteadyState check:")]


def _stop_timestep(stdout):
    """The timestep a run stopped at on its steady-state criterion, or None."""
    m = re.search(r"steady-state criterion of .* reached after (\d+) timesteps", stdout)
    return int(m.group(1)) if m else None


class TemporalBlockingCaseMixin:
    """Shared machinery: run flat vs. blocked, assert ACTIVE, compare fields.

    Not a TestCase itself; each concrete extension's test below supplies
    `boundary` and `extra_setup` and calls `_compare()`.
    """

    def _compare(self, tag, boundary, extra_setup):
        flat_out, flat_E, flat_H = _run(tag + "_flat", boundary, extra_setup, blocked=False)
        blk_out, blk_E, blk_H = _run(tag + "_blk", boundary, extra_setup, blocked=True)

        self.assertNotIn(
            "AVX2 temporal blocking ACTIVE", flat_out,
            "the control run (no OPENEMS_AVX2_TEMPORAL_BLOCK set) engaged "
            "blocking anyway -- the env var is not gating it",
        )
        self.assertGreater(
            np.max(np.abs(flat_E)), 0.0,
            "flat run produced an all-zero field -- excitation never reached the dump",
        )

        # This is the assertion that keeps the test from passing vacuously:
        # if the extension under test cannot be slabbed, blocking silently
        # falls back to the flat sweep and the field comparison below would
        # trivially succeed without having exercised the blocked schedule at
        # all. Fail loudly instead, naming whatever the engine actually said.
        self.assertIn(
            "AVX2 temporal blocking ACTIVE", blk_out,
            "temporal blocking did not engage for '%s'; engine said:\n%s"
            % (tag, _relevant_lines(blk_out)),
        )

        self.assertEqual(
            flat_E.shape, blk_E.shape,
            "flat and blocked E dumps have different shapes (%s vs %s) -- "
            "one of the runs went unstable or dumped a different number of steps"
            % (flat_E.shape, blk_E.shape),
        )
        self.assertEqual(flat_H.shape, blk_H.shape, "flat and blocked H dumps have different shapes")

        self.assertTrue(
            np.array_equal(flat_E, blk_E),
            "AVX2 temporal blocking is not bit-identical to the flat sweep "
            "for E in '%s': max|d| = %.3e"
            % (tag, float(np.max(np.abs(blk_E.astype(np.float64) - flat_E.astype(np.float64))))),
        )
        self.assertTrue(
            np.array_equal(flat_H, blk_H),
            "AVX2 temporal blocking is not bit-identical to the flat sweep "
            "for H in '%s': max|d| = %.3e"
            % (tag, float(np.max(np.abs(blk_H.astype(np.float64) - flat_H.astype(np.float64))))),
        )


class Test1_PEC_Control(TemporalBlockingCaseMixin, unittest.TestCase):
    """Control: PEC boundaries only, no extension beyond the excitation.

    Only the excitation extension is present, and it already supports slab
    apply, so this is the one case expected to engage and match today. If
    this one fails, something is wrong with the harness or the schedule
    itself, not with a specific extension's slab support.
    """

    def test_pec_only(self):
        self._compare(
            "pec",
            [0, 0, 0, 0, 0, 0],
            """
            metal = CSX.AddMetal("block")
            metal.AddBox([xl[%d], yl[10], zl[10]], [xl[%d], yl[22], zl[22]], priority=10)
            """ % (OBJ_X_LO, OBJ_X_HI),
        )


class Test2_UPML(TemporalBlockingCaseMixin, unittest.TestCase):
    """Uniaxial PML absorbing boundary on all six sides (PML_8)."""

    def test_pml(self):
        self._compare(
            "pml",
            ["PML_8"] * 6,
            """
            sub = CSX.AddMaterial("sub", epsilon=2.5)
            sub.AddBox([xl[%d], yl[10], zl[10]], [xl[%d], yl[22], zl[22]], priority=5)
            """ % (OBJ_X_LO, OBJ_X_HI),
        )


class Test3_MurABC(TemporalBlockingCaseMixin, unittest.TestCase):
    """Simple Mur absorbing boundary condition on all six sides."""

    def test_mur(self):
        self._compare(
            "mur",
            ["MUR"] * 6,
            """
            sub = CSX.AddMaterial("sub", epsilon=2.5)
            sub.AddBox([xl[%d], yl[10], zl[10]], [xl[%d], yl[22], zl[22]], priority=5)
            """ % (OBJ_X_LO, OBJ_X_HI),
        )


class Test4_DispersiveDrude(TemporalBlockingCaseMixin, unittest.TestCase):
    """A Drude/plasma dispersive material (CSPropLorentzMaterial, order 1,
    pole frequency left at zero so only the plasma/damping terms act)."""

    def test_drude(self):
        self._compare(
            "drude",
            [0, 0, 0, 0, 0, 0],
            """
            from CSXCAD.CSProperties import CSPropLorentzMaterial
            mat = CSPropLorentzMaterial(CSX.GetParameterSet(), epsilon=1.0, order=1)
            CSX.AddProperty(mat)
            mat.SetName("drude")
            mat.SetDispersiveMaterialProperty(0, eps_plasma=2 * np.pi * 5e9, eps_relax=2e-11)
            mat.AddBox([xl[%d], yl[10], zl[10]], [xl[%d], yl[22], zl[22]], priority=5)
            """ % (OBJ_X_LO, OBJ_X_HI),
        )


class Test5_LumpedRLC(TemporalBlockingCaseMixin, unittest.TestCase):
    """A single parallel-RLC lumped element (see LumpedRLC.py for the full
    port-based validation of this extension; here it is only load-bearing
    geometry, not something whose impedance is checked)."""

    def test_lumped_rlc(self):
        self._compare(
            "lumped",
            [0, 0, 0, 0, 0, 0],
            """
            lrc = CSX.AddLumpedElement("rlc", ny="z", caps=True, R=75.0, L=5e-9, C=2e-12, LEtype=0)
            lrc.AddBox([xl[%d], yl[14], zl[10]], [xl[%d], yl[18], zl[14]], priority=10)
            """ % (LUMPED_X_LO, LUMPED_X_HI),
        )


class Test6_Combined(TemporalBlockingCaseMixin, unittest.TestCase):
    """PML boundary + a lumped element + an E-field probe together.

    ConfigureTemporalBlocking() checks every active extension and stops at
    the first one that refuses, so this also exercises that the loop over
    m_Eng_exts does not, say, stop checking after the first extension that
    *does* support slab apply.
    """

    def test_pml_plus_lumped_plus_probe(self):
        self._compare(
            "combo",
            ["PML_8"] * 6,
            """
            lrc = CSX.AddLumpedElement("rlc", ny="z", caps=True, R=75.0, L=5e-9, C=2e-12, LEtype=0)
            lrc.AddBox([xl[%d], yl[14], zl[10]], [xl[%d], yl[18], zl[14]], priority=10)
            probe = CSX.AddProbe("p1", p_type=2)
            probe.AddBox([xl[49], yl[16], zl[12]], [xl[49], yl[16], zl[12]])
            """ % (LUMPED_X_LO, LUMPED_X_HI),
        )


class Test7_AbsorbingBC(TemporalBlockingCaseMixin, unittest.TestCase):
    """A local absorbing-BC sheet (Engine_Ext_Absorbing_BC), as used by
    MSL_With_Local_Absorbers.py.

    Unlike the global MUR boundary this sits on an interior plane, so it is
    the case where a boundary-shaped extension is *not* protected by the
    trapezoid holding the domain edge. The sheet is normal to z and spans the
    full x range, so it is the pattern-C path -- the axis restriction, not
    the "is the plane in this slab" test -- that has to be right.
    """

    def test_local_absorber(self):
        self._compare(
            "abc",
            [0, 0, 0, 0, 0, 0],
            """
            from CSXCAD.CSProperties import ABCtype
            ab = CSX.AddAbsorbingBC("abs1", NormalSignPositive=False,
                                    AbsorbingBoundaryType=ABCtype.MUR_1ST)
            ab.AddBox([xl[2], yl[2], zl[26]], [xl[-3], yl[-3], zl[26]], priority=20)
            """,
        )


class Test8_TFSF(TemporalBlockingCaseMixin, unittest.TestCase):
    """A total-field/scattered-field plane-wave box (Engine_Ext_TFSF).

    The box straddles the tile boundary at x=48, so its two x-normal faces
    land in different tiles and its four x-spanning faces are cut by every
    tile. openems.cpp adds Operator_Ext_TFSF to every Cartesian run, but it
    only builds an engine extension when an exc_type=10 excitation with
    exactly one box primitive is present -- which is what this sets up.
    """

    def test_plane_wave(self):
        self._compare(
            "tfsf",
            [0, 0, 0, 0, 0, 0],
            """
            pw = CSX.AddExcitation("planewave", exc_type=10, exc_val=[0, 0, 1])
            pw.SetPropagationDir([1, 0, 0])
            pw.AddBox([xl[%d], yl[6], zl[6]], [xl[%d], yl[26], zl[26]])
            """ % (OBJ_X_LO, OBJ_X_HI),
        )


class Test9_ScheduleSweep(TemporalBlockingCaseMixin, unittest.TestCase):
    """The same simulation over several (k, W) and thread counts.

    One (k, W) proves the hooks are wired up; it does not prove the geometry
    is right. Different tile widths put the tile boundaries -- and so the
    wedges -- in different places relative to the PML, the lumped element and
    the absorber, and different thread counts change how each slab is split
    across workers. A slab hook that writes one cell outside its range, or
    partitions its work so two threads share a cell, shows up here and in
    nothing above.
    """

    SETUP = """
    from CSXCAD.CSProperties import ABCtype
    lrc = CSX.AddLumpedElement("rlc", ny="z", caps=True, R=75.0, L=5e-9, C=2e-12, LEtype=0)
    lrc.AddBox([xl[46], yl[14], zl[10]], [xl[50], yl[18], zl[14]], priority=10)
    ab = CSX.AddAbsorbingBC("abs1", NormalSignPositive=False,
                            AbsorbingBoundaryType=ABCtype.MUR_1ST)
    ab.AddBox([xl[2], yl[2], zl[26]], [xl[-3], yl[-3], zl[26]], priority=20)
    """

    def test_sweep(self):
        global BLOCK_ENV, THREADS
        flat_out, flat_E, flat_H = _run("sweep_flat", ["PML_8"] * 6, self.SETUP, blocked=False)
        self.assertGreater(np.max(np.abs(flat_E)), 0.0, "flat run produced an all-zero field")

        saved = (BLOCK_ENV, THREADS)
        try:
            for kw in ("2:8", "4:16", "3:13", "6:20"):
                for threads in (1, 3, 4):
                    BLOCK_ENV, THREADS = kw, threads
                    tag = "sweep_%s_%d" % (kw.replace(":", "_"), threads)
                    out, E, H = _run(tag, ["PML_8"] * 6, self.SETUP, blocked=True)
                    self.assertIn(
                        "AVX2 temporal blocking ACTIVE", out,
                        "blocking did not engage at k:W=%s, %d threads; engine said:\n%s"
                        % (kw, threads, _relevant_lines(out)),
                    )
                    self.assertTrue(
                        np.array_equal(flat_E, E) and np.array_equal(flat_H, H),
                        "k:W=%s with %d threads is not bit-identical to the flat sweep"
                        % (kw, threads),
                    )
        finally:
            BLOCK_ENV, THREADS = saved


class SteadyStateCaseMixin:
    """A sine model run until it stops on its own, flat and blocked.

    Beyond the fields, the steady-state detection has to reach the same
    verdict at every check and stop the run at the same timestep: the
    blocked schedule sums the energy of a check per slab and evaluates it at
    the end of the block, the flat sweep all at once and at once.
    """

    def _compare_steady(self, tag, boundary, extra_setup, delta, schedules):
        global BLOCK_ENV, THREADS
        setup = LOSSY_SETUP + textwrap.dedent(extra_setup)
        flat_out, flat_E, flat_H = _run(tag + "_flat", boundary, setup, blocked=False,
                                        fdtd_setup=SINE_SETUP, delta=delta)
        flat_trace, flat_stop = _steady_trace(flat_out), _stop_timestep(flat_out)
        self.assertGreater(np.max(np.abs(flat_E)), 0.0, "flat run produced an all-zero field")
        self.assertNotIn("AVX2 temporal blocking ACTIVE", flat_out)
        self.assertIsNotNone(flat_stop, "the flat run did not stop on its steady-state criterion")
        self.assertGreaterEqual(len(flat_trace), 4, "too few steady-state checks to compare:\n%s"
                                % "\n".join(flat_trace))

        saved = (BLOCK_ENV, THREADS)
        try:
            for kw, threads in schedules:
                BLOCK_ENV, THREADS = kw, threads
                name = "%s_%s_%d" % (tag, kw.replace(":", "_"), threads)
                out, E, H = _run(name, boundary, setup, blocked=True, fdtd_setup=SINE_SETUP, delta=delta)
                self.assertIn("AVX2 temporal blocking ACTIVE", out,
                              "blocking did not engage with steady-state detection at k:W=%s, "
                              "%d threads; engine said:\n%s" % (kw, threads, _relevant_lines(out)))
                self.assertEqual(flat_trace, _steady_trace(out),
                                 "k:W=%s with %d threads: the steady-state checks differ from the flat sweep's"
                                 % (kw, threads))
                self.assertEqual(flat_stop, _stop_timestep(out),
                                 "k:W=%s with %d threads stopped at another timestep" % (kw, threads))
                self.assertTrue(np.array_equal(flat_E, E) and np.array_equal(flat_H, H),
                                "k:W=%s with %d threads is not bit-identical to the flat sweep"
                                % (kw, threads))
        finally:
            BLOCK_ENV, THREADS = saved


class Test10_SteadyStateSweep(SteadyStateCaseMixin, unittest.TestCase):
    """A lossy cavity driven by a sine, over several (k, W) and thread counts.

    The period (74 timesteps) is not a multiple of the processing interval
    (9), so the checks fall anywhere in a block, and 6:20 folds a short tail
    tile into its neighbour.
    """

    def test_sweep(self):
        schedules = [(kw, threads) for kw in ("2:8", "4:16", "3:13", "6:20") for threads in (1, 3, 4)]
        self._compare_steady("ss_sweep", [0, 0, 0, 0, 0, 0], "", DELTA_P74, schedules)


class Test11_SteadyStatePeriodMultiple(SteadyStateCaseMixin, unittest.TestCase):
    """The same with a period of 80 timesteps and a processing interval of
    10, which divides it: the checks fall on block boundaries or inside, as
    k divides the interval or not."""

    def test_period_multiple(self):
        self._compare_steady("ss_p80", [0, 0, 0, 0, 0, 0], "", DELTA_P80, [("5:16", 2), ("3:13", 3)])


class Test12_SteadyStateExtensions(SteadyStateCaseMixin, unittest.TestCase):
    """Steady-state detection next to the extensions that write the fields
    in the same Apply2Voltages phase: UPML or Mur, a lumped RLC element and
    a probe. Its hook runs first there and reads before they write."""

    SETUP = """
    lrc = CSX.AddLumpedElement("rlc", ny="z", caps=True, R=75.0, L=5e-9, C=2e-12, LEtype=0)
    lrc.AddBox([xl[%d], yl[14], zl[10]], [xl[%d], yl[18], zl[14]], priority=10)
    probe = CSX.AddProbe("p1", p_type=2)
    probe.AddBox([xl[49], yl[16], zl[12]], [xl[49], yl[16], zl[12]])
    """ % (LUMPED_X_LO, LUMPED_X_HI)

    def test_pml(self):
        self._compare_steady("ss_pml", ["PML_8"] * 6, self.SETUP, DELTA_P74, [("4:16", 2), ("6:20", 3)])

    def test_mur(self):
        self._compare_steady("ss_mur", ["MUR"] * 6, self.SETUP, DELTA_P74, [("4:16", 2), ("3:13", 4)])


if __name__ == "__main__":
    unittest.main()
