# -*- coding: utf-8 -*-
"""Regression tests for Engine_Vulkan's trapezoidal temporal blocking.

The schedule in FDTD/engine_vulkan.cpp (OPENEMS_GPU_TEMPORAL_BLOCK) advances
one x-slab through k timesteps before moving on, so that the slab -- not the
whole grid -- is the working set the GPU's last-level cache has to hold. It is
a transcription of the geometry Engine_AVX2_Multithread already runs, and the
claim under test is the same one: bit-identical to the flat sweep.

The trap this file exists to avoid is the same as its AVX2 counterpart's. A
test that only checks "flat run == blocked run" passes *vacuously* whenever
blocking declines to engage -- a sweep is of course identical to itself, and
ConfigureTemporalBlocking() declines for several good reasons (an extension
without a slab path, a grid that already fits the tile target, too few
x-lines). Every case here therefore greps the blocked run's output for the
line that says blocking actually engaged, before trusting any comparison.

The grid is 64 x-lines in four 16-line tiles (k=4, W=16, satisfying the hard
W>=2k constraint). OPENEMS_GPU_TEMPORAL_BLOCK_MB=1 shrinks the tile target so
the "grid already fits cache" guard -- the right call for real runs -- does
not refuse a CI-sized grid.

Each model is written to XML by a Python subprocess and run by the build's
openEMS binary (OPENEMS_TEST_BIN, default build/openEMS; OPENEMS_TEST_PYTHON
picks the interpreter that writes the XML), so the Python modules never run
a library they were not built against.

A sine excitation adds steady-state detection. The blocked schedule carries
it (plans/STEADY_STATE_TEMPORAL_BLOCKING.md), and its cases compare the trace
of every check (OPENEMS_STEADYSTATE_TRACE=1) and the timestep the run stops
at, besides the fields.
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

NX, NY, NZ = 64, 32, 32
TIMESTEPS = 120
BLOCK_ENV = "4:16"          # k=4, W=16 -> four tiles across the grid
EXC_X = 20                  # excitation plane, inside tile 1 ([16,32))
GPU_INDEX = os.environ.get("OPENEMS_TEST_GPU_INDEX", "0")

# Objects straddle the tile boundary at x=48 on purpose, so a wedge has to
# reconcile something other than vacuum.
OBJ_X_LO, OBJ_X_HI = 38, 54

ACTIVE_MARK = "trapezoidal temporal blocking active"

# A Gaussian pulse for a fixed number of timesteps, or a sine in a lossy
# dielectric until the steady state is reached (about a thousand timesteps).
# With 1 mm cells the period is 74 timesteps and the processing interval 9,
# which does not divide it; with 0.925 mm cells 80 and 10, which does.
GAUSS_SETUP = "FDTD = openEMS(NrTS=%d, EndCriteria=0)\nFDTD.SetGaussExcite(5e9, 5e9)" % TIMESTEPS
SINE_SETUP = "FDTD = openEMS(NrTS=20000, EndCriteria=1e-3)\nFDTD.SetSinusExcite(5e9)"
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


def _run(tag, boundary, block_env, extra_env=None, extra_setup="", fdtd_setup=GAUSS_SETUP, delta=1e-3):
    """Write one small model to XML and run it on the GPU engine.

    Returns (output, E_time_series, H_time_series). `block_env` is the value
    of OPENEMS_GPU_TEMPORAL_BLOCK for this run, or None for the flat sweep
    (blocking explicitly off: it is on by default for large grids); the two
    runs being compared must differ in *only* that.
    """
    sim_path = os.path.join(tempfile.gettempdir(), "gputblk_%s" % tag)
    shutil.rmtree(sim_path, ignore_errors=True)
    os.makedirs(sim_path)
    xml = os.path.join(sim_path, "model.xml")

    script = _SCRIPT_TEMPLATE.format(
        fdtd_setup=fdtd_setup, delta=delta, boundary=boundary, nx=NX, ny=NY, nz=NZ,
        exc_x=EXC_X, xml=xml,
        extra_setup=textwrap.dedent(extra_setup).strip(),
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
    env["OPENEMS_STEADYSTATE_TRACE"] = "1"
    env["OPENEMS_GPU_INDEX"] = GPU_INDEX
    env.pop("OPENEMS_GPU_TEMPORAL_BLOCK", None)
    env.pop("OPENEMS_GPU_TEMPORAL_BLOCK_MB", None)
    if block_env:
        env["OPENEMS_GPU_TEMPORAL_BLOCK"] = block_env
        env["OPENEMS_GPU_TEMPORAL_BLOCK_MB"] = "1"
    else:
        env["OPENEMS_GPU_TEMPORAL_BLOCK"] = "0"
    env.update(extra_env or {})

    proc = subprocess.run([OPENEMS_BIN, xml, "--engine=gpu"], cwd=sim_path, env=env,
                          capture_output=True, text=True, timeout=300)
    try:
        if proc.returncode != 0:
            raise RuntimeError("simulation failed (tag=%s, rc=%s):\n%s\n%s"
                               % (tag, proc.returncode, proc.stdout[-4000:], proc.stderr[-4000:]))

        def _load(name):
            with h5py.File(os.path.join(sim_path, name), "r") as f:
                td = f["FieldData/TD"]
                return np.array([np.array(td[k]) for k in sorted(td.keys())])

        E, H = _load("Et.h5"), _load("Ht.h5")
    finally:
        shutil.rmtree(sim_path, ignore_errors=True)

    return proc.stdout + proc.stderr, E, H


def _relevant_lines(output):
    """The lines that say what ConfigureTemporalBlocking() decided."""
    lines = [l for l in output.splitlines() if "temporal blocking" in l]
    return "\n".join(lines) if lines else "(no 'temporal blocking' line in the output at all)"


def _steady_trace(output):
    """Every steady-state check of a run, as OPENEMS_STEADYSTATE_TRACE prints it."""
    return [l for l in output.splitlines() if l.startswith("SteadyState check:")]


def _stop_timestep(output):
    """The timestep a run stopped at on its steady-state criterion, or None."""
    m = re.search(r"steady-state criterion of .* reached after (\d+) timesteps", output)
    return int(m.group(1)) if m else None


class GPUTemporalBlockingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flat_out, cls.flat_E, cls.flat_H = _run("flat", ["PEC"] * 6, None)

    def test_flat_reference_is_not_vacuous(self):
        """A comparison against an all-zero field would pass no matter what."""
        self.assertGreater(
            float(np.max(np.abs(self.flat_E))), 0.0,
            "flat run produced an all-zero E field -- the excitation never reached the dump")

    def test_blocking_engages(self):
        out, _, _ = _run("engage", ["PEC"] * 6, BLOCK_ENV)
        self.assertIn(ACTIVE_MARK, out,
                      "temporal blocking did not engage, so any field comparison below "
                      "would pass vacuously. The engine said:\n%s" % _relevant_lines(out))

    def test_pec_bit_identical(self):
        out, E, H = _run("pec", ["PEC"] * 6, BLOCK_ENV)
        self.assertIn(ACTIVE_MARK, out, _relevant_lines(out))
        self.assertEqual(self.flat_E.shape, E.shape,
                         "flat and blocked E dumps have different shapes -- one run went "
                         "unstable or dumped a different number of steps")
        self.assertTrue(np.array_equal(self.flat_E, E),
                        "blocked E differs from flat: max |delta| = %g"
                        % float(np.max(np.abs(self.flat_E - E))))
        self.assertTrue(np.array_equal(self.flat_H, H),
                        "blocked H differs from flat: max |delta| = %g"
                        % float(np.max(np.abs(self.flat_H - H))))

    def test_schedule_sweep(self):
        """Several (k, W) pairs, including one where W does not divide NX.

        A tile width that divides the domain evenly is the easy case; the
        wedges only have to reconcile a boundary the tiling put there. 4:13
        leaves a short tail, which is what exercises the fold in
        ConfigureTemporalBlocking().
        """
        for env in ("2:8", "4:16", "3:12", "4:13", "6:20"):
            with self.subTest(schedule=env):
                out, E, H = _run("sweep_%s" % env.replace(":", "_"), ["PEC"] * 6, env)
                self.assertIn(ACTIVE_MARK, out, _relevant_lines(out))
                self.assertTrue(np.array_equal(self.flat_E, E) and np.array_equal(self.flat_H, H),
                                "schedule %s is not bit-identical to the flat sweep" % env)

    def test_pml_fused_bit_identical(self):
        """UPML through the fused kernel, which is the default path.

        The fused shader derives PML membership per cell from the region
        metadata, so it needs no slab logic of its own -- but its current pass
        does real work on the last x-line (flux recursion with a zero curl)
        where the Yee sweep does none, which is why the last tile extends the
        current range to NX. This test is what would catch getting that wrong.
        """
        flat_out, flat_E, flat_H = _run("pmlf_flat", ["PML_8"] * 6, None)
        self.assertGreater(float(np.max(np.abs(flat_E))), 0.0,
                           "flat PML run produced an all-zero field")
        out, E, H = _run("pmlf_blocked", ["PML_8"] * 6, BLOCK_ENV)
        self.assertIn(ACTIVE_MARK, out, _relevant_lines(out))
        self.assertTrue(np.array_equal(flat_E, E),
                        "blocked E differs from flat under fused UPML: max |delta| = %g"
                        % float(np.max(np.abs(flat_E - E))))
        self.assertTrue(np.array_equal(flat_H, H),
                        "blocked H differs from flat under fused UPML: max |delta| = %g"
                        % float(np.max(np.abs(flat_H - H))))

    def test_pml_unfused_bit_identical(self):
        """The same, through the separate upml_pre/post dispatches.

        OPENEMS_GPU_DISABLE_FUSED_UPML=1 selects the fallback path, where the
        PML pre/post passes are their own per-region dispatches indexed in
        region-local coordinates. Both arms set it, so this compares the
        unfused flat sweep against the unfused blocked one.
        """
        unfused = {"OPENEMS_GPU_DISABLE_FUSED_UPML": "1"}
        flat_out, flat_E, flat_H = _run("pmlu_flat", ["PML_8"] * 6, None, unfused)
        self.assertGreater(float(np.max(np.abs(flat_E))), 0.0,
                           "unfused flat PML run produced an all-zero field")
        out, E, H = _run("pmlu_blocked", ["PML_8"] * 6, BLOCK_ENV, unfused)
        self.assertIn(ACTIVE_MARK, out, _relevant_lines(out))
        self.assertTrue(np.array_equal(flat_E, E) and np.array_equal(flat_H, H),
                        "blocked run differs from flat under unfused UPML")

    def test_fused_and_unfused_pml_agree(self):
        """Guards against both PML arms being wrong in the same way.

        The two tests above each compare a path against itself. If the slab
        range were wrong in a way both the fused and unfused blocked paths
        shared, they would both still pass. This compares across paths.
        """
        _, fused_E, _ = _run("pmlx_fused", ["PML_8"] * 6, BLOCK_ENV)
        _, unfused_E, _ = _run("pmlx_unfused", ["PML_8"] * 6, BLOCK_ENV,
                               {"OPENEMS_GPU_DISABLE_FUSED_UPML": "1"})
        self.assertTrue(np.allclose(fused_E, unfused_E, rtol=0, atol=1e-9),
                        "fused and unfused blocked PML disagree: max |delta| = %g"
                        % float(np.max(np.abs(fused_E - unfused_E))))

    def _compare(self, tag, boundary, extra_setup):
        """Run flat vs blocked with the same geometry and compare."""
        _, flat_E, flat_H = _run(tag + "_flat", boundary, None, extra_setup=extra_setup)
        self.assertGreater(float(np.max(np.abs(flat_E))), 0.0,
                           "flat run for %s produced an all-zero field" % tag)
        out, E, H = _run(tag + "_blocked", boundary, BLOCK_ENV, extra_setup=extra_setup)
        self.assertIn(ACTIVE_MARK, out, _relevant_lines(out))
        self.assertTrue(np.array_equal(flat_E, E),
                        "%s: blocked E differs from flat, max |delta| = %g"
                        % (tag, float(np.max(np.abs(flat_E - E)))))
        self.assertTrue(np.array_equal(flat_H, H),
                        "%s: blocked H differs from flat, max |delta| = %g"
                        % (tag, float(np.max(np.abs(flat_H - H)))))

    def test_drude(self):
        """A dispersive (Drude) block straddling a tile boundary."""
        self._compare("drude", ["PEC"] * 6, """
            from CSXCAD.CSProperties import CSPropLorentzMaterial
            mat = CSPropLorentzMaterial(CSX.GetParameterSet(), epsilon=1.0, order=1)
            CSX.AddProperty(mat)
            mat.SetName("drude")
            mat.SetDispersiveMaterialProperty(0, eps_plasma=2 * np.pi * 5e9, eps_relax=2e-11)
            mat.AddBox([xl[%d], yl[10], zl[10]], [xl[%d], yl[22], zl[22]], priority=5)
            """ % (OBJ_X_LO, OBJ_X_HI))

    def test_conducting_sheet(self):
        """The two-order case.

        Operator_Ext_ConductingSheet pushes the *identical* position list into
        both of its orders, so one engine cell is written once per order. The
        x-sort has to give both orders the same permutation, or a cell sees the
        two subtractions in the wrong order and the result drifts.
        """
        self._compare("sheet", ["PEC"] * 6, """
            sheet = CSX.AddConductingSheet("sheet", conductivity=5.8e7, thickness=1e-6)
            sheet.AddBox([xl[%d], yl[10], zl[10]], [xl[%d], yl[22], zl[22]], priority=5)
            """ % (OBJ_X_LO, OBJ_X_HI))

    def test_drude_with_pml(self):
        """Dispersive material and UPML together, since they share the sweep."""
        self._compare("drude_pml", ["PML_8"] * 6, """
            from CSXCAD.CSProperties import CSPropLorentzMaterial
            mat = CSPropLorentzMaterial(CSX.GetParameterSet(), epsilon=1.0, order=1)
            CSX.AddProperty(mat)
            mat.SetName("drude")
            mat.SetDispersiveMaterialProperty(0, eps_plasma=2 * np.pi * 5e9, eps_relax=2e-11)
            mat.AddBox([xl[%d], yl[10], zl[10]], [xl[%d], yl[22], zl[22]], priority=5)
            """ % (OBJ_X_LO, OBJ_X_HI))

    def test_mur(self):
        """Mur on all six faces.

        A face normal to x writes its boundary line and reads the next one at
        the same timestep, so it runs only in the slab that holds both; the
        y- and z-normal faces are clipped to every slab.
        """
        self._compare("mur", ["MUR"] * 6, "")

    def test_mur_with_pml(self):
        """Mur on the x faces, UPML on the others: the Mur faces reach into
        the PML regions, where the order of the two extensions matters."""
        self._compare("mur_pml", ["MUR", "MUR", "PML_8", "PML_8", "PML_8", "PML_8"], "")

    def test_lumped_rlc_and_port(self):
        """A series RLC straddling a tile boundary, and a resistor on the
        excitation's own edges (a lumped port), so the RLC result and the
        excitation are applied to the same cells."""
        self._compare("rlc", ["MUR"] * 6, """
            rlc = CSX.AddLumpedElement("rlc", ny="z", caps=True, R=75.0, L=5e-9, C=2e-12, LEtype=1)
            rlc.AddBox([xl[%d], yl[14], zl[10]], [xl[%d], yl[18], zl[14]], priority=10)
            port = CSX.AddLumpedElement("port_R", ny="z", caps=False, R=50.0)
            port.AddBox([xl[%d], yl[10], zl[10]], [xl[%d] + 1, yl[22], zl[22]], priority=10)
            """ % (OBJ_X_LO, OBJ_X_HI, EXC_X, EXC_X))

    def test_tfsf(self):
        """A TF/SF plane-wave box straddling a tile boundary: its x-normal
        faces land in different tiles, its other faces are cut by every one."""
        self._compare("tfsf", ["MUR"] * 6, """
            pw = CSX.AddExcitation("planewave", exc_type=10, exc_val=[0, 0, 1])
            pw.SetPropagationDir([1, 0, 0])
            pw.AddBox([xl[%d], yl[6], zl[6]], [xl[%d], yl[26], zl[26]])
            """ % (OBJ_X_LO, OBJ_X_HI))

    def test_local_absorbing_sheet(self):
        """A local absorbing sheet normal to z, cut by every tile edge."""
        self._compare("abs", ["PEC"] * 6, """
            from CSXCAD.CSProperties import ABCtype
            ab = CSX.AddAbsorbingBC("abs1", NormalSignPositive=False,
                                    AbsorbingBoundaryType=ABCtype.MUR_1ST)
            ab.AddBox([xl[2], yl[2], zl[26]], [xl[-3], yl[-3], zl[26]], priority=20)
            """)

    def _compare_steady(self, tag, boundary, extra_setup, delta, schedules):
        """A sine model run until it stops on its own, flat and blocked.

        The blocked schedule samples and sums each slab as it passes a
        check's timestep, and the host evaluates the check after the batch:
        the verdict of every check and the timestep the run stops at must be
        those of the flat sweep, besides the fields.
        """
        setup = LOSSY_SETUP + textwrap.dedent(extra_setup)
        flat_out, flat_E, flat_H = _run(tag + "_flat", boundary, None, extra_setup=setup,
                                        fdtd_setup=SINE_SETUP, delta=delta)
        flat_trace, flat_stop = _steady_trace(flat_out), _stop_timestep(flat_out)
        self.assertGreater(float(np.max(np.abs(flat_E))), 0.0, "flat run produced an all-zero field")
        self.assertIsNotNone(flat_stop, "the flat run did not stop on its steady-state criterion")
        self.assertGreaterEqual(len(flat_trace), 4, "too few steady-state checks to compare")
        for env in schedules:
            with self.subTest(schedule=env):
                out, E, H = _run("%s_%s" % (tag, env.replace(":", "_")), boundary, env,
                                 extra_setup=setup, fdtd_setup=SINE_SETUP, delta=delta)
                self.assertIn(ACTIVE_MARK, out, _relevant_lines(out))
                self.assertEqual(flat_trace, _steady_trace(out),
                                 "schedule %s: the steady-state checks differ from the flat sweep's" % env)
                self.assertEqual(flat_stop, _stop_timestep(out),
                                 "schedule %s stopped at another timestep" % env)
                self.assertTrue(np.array_equal(flat_E, E) and np.array_equal(flat_H, H),
                                "schedule %s is not bit-identical to the flat sweep" % env)

    def test_steady_state_schedule_sweep(self):
        """Steady-state detection under several (k, W), 4:13 and 6:20 with a
        folded tail, on a period (74) the processing interval (9) does not
        divide, so checks fall anywhere in a block."""
        self._compare_steady("ss_sweep", ["PEC"] * 6, "", DELTA_P74, ("2:8", "4:16", "3:12", "4:13", "6:20"))

    def test_steady_state_period_multiple(self):
        """A period of 80 and a processing interval of 10, which divides it."""
        self._compare_steady("ss_p80", ["PEC"] * 6, "", DELTA_P80, ("5:16", "3:12"))

    def test_steady_state_with_extensions(self):
        """Mur, a lumped port and an RLC element, which write the fields in
        the voltage phase the detection reads, and UPML. (A parallel RLC: the
        series one of test_lumped_rlc_and_port goes NaN when read from XML.)"""
        setup = """
            rlc = CSX.AddLumpedElement("rlc", ny="z", caps=True, R=75.0, L=5e-9, C=2e-12, LEtype=0)
            rlc.AddBox([xl[%d], yl[14], zl[10]], [xl[%d], yl[18], zl[14]], priority=10)
            port = CSX.AddLumpedElement("port_R", ny="z", caps=False, R=50.0)
            port.AddBox([xl[%d], yl[10], zl[10]], [xl[%d] + 1, yl[22], zl[22]], priority=10)
            """ % (OBJ_X_LO, OBJ_X_HI, EXC_X, EXC_X)
        self._compare_steady("ss_mur", ["MUR"] * 6, setup, DELTA_P74, ("4:16", "3:12"))
        self._compare_steady("ss_pml", ["PML_8"] * 6, "", DELTA_P74, ("4:16",))


if __name__ == "__main__":
    unittest.main(verbosity=2)
