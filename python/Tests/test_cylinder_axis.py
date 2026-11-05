# -*- coding: utf-8 -*-
"""Regression test for E_z on the axis of a closed cylindrical mesh.

On a closed alpha mesh including r=0, Engine_Ext_Cylinder updates E_z on the
axis from the alpha currents on the ring around it. The mesh carries two alpha
lines past the seam, copies of the lines 0 and 1; the sum took the copy of
line 0 as well, so that current entered twice, while the capacitance of the
update (Operator_Ext_Cylinder) counts every line once.

Two opposite z-line sources at alpha and alpha+pi make a field that is odd
under a rotation by pi, and the mesh is symmetric under it, so E_z on the axis
must stay zero. The doubled current broke the symmetry at the seam and put
0.5 % of the source voltage on the axis here; without it, float rounding is
left (8e-8).

As test_temporal_blocking.py, the model is written to XML by a Python
subprocess (OPENEMS_TEST_PYTHON) and run by the openEMS binary of the build
(OPENEMS_TEST_BIN, default build/openEMS), so the Python modules never run
the build's library. On a cylindrical mesh every CPU engine is the cylinder
engine; the test runs two of them.
"""

import os
import sys
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OPENEMS_BIN = os.environ.get("OPENEMS_TEST_BIN", os.path.join(REPO_ROOT, "build", "openEMS"))
XML_PYTHON = os.environ.get("OPENEMS_TEST_PYTHON", sys.executable)

# A closed PEC cylinder including r=0, two opposite sources at r=3 mm on the
# alpha lines pi/4 and -3pi/4. The source boxes reach 1e-4 rad to either side,
# so they snap to their one line also where the XML rounds the coordinate (a
# voltage probe has to be a line, it snaps to the nearest one).
_SCRIPT = """
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS

FDTD = openEMS(NrTS=3000, EndCriteria=0, CoordSystem=1)
FDTD.SetGaussExcite(5e9, 4e9)
FDTD.SetBoundaryCond(["PEC"] * 6)

CSX = ContinuousStructure(CoordSystem=1)
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(1e-3)
mesh.SetLines("r", np.arange(12, dtype=float))
mesh.SetLines("a", np.linspace(-np.pi, np.pi, 17))
mesh.SetLines("z", np.arange(12, dtype=float) - 5.5)

e = 1e-4
for name, a, sign in (("p", np.pi / 4, 1), ("m", -3 * np.pi / 4, -1)):
    CSX.AddExcitation(name, exc_type=0, exc_val=[0, 0, sign]).AddBox([3, a - e, -1.5], [3, a + e, 1.5])
CSX.AddProbe("v_axis", p_type=0).AddBox([0, 0, -0.5], [0, 0, 0.5])
CSX.AddProbe("v_src", p_type=0).AddBox([3, np.pi / 4, -0.5], [3, np.pi / 4, 0.5])

FDTD.Write2XML({xml!r})
"""


def _run(engine):
    """Returns the voltages on the axis and on the positive source over time."""
    sim_path = os.path.join(tempfile.gettempdir(), "cyl_axis_%s" % engine)
    shutil.rmtree(sim_path, ignore_errors=True)
    os.makedirs(sim_path)
    xml = os.path.join(sim_path, "model.xml")
    script_path = os.path.join(sim_path, "model.py")
    with open(script_path, "w") as f:
        f.write(_SCRIPT.format(xml=xml))
    proc = subprocess.run([XML_PYTHON, script_path], cwd=sim_path, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError("writing the model failed:\n" + proc.stdout + proc.stderr)

    proc = subprocess.run([OPENEMS_BIN, xml, "--engine=" + engine, "--numThreads=2"], cwd=sim_path,
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError("openEMS failed:\n" + proc.stdout + proc.stderr)
    if "Unused primitive" in proc.stdout + proc.stderr:
        raise RuntimeError("a source or probe did not snap to the mesh:\n" + proc.stdout + proc.stderr)
    axis = np.loadtxt(os.path.join(sim_path, "v_axis"), comments="%")[:, 1]
    src = np.loadtxt(os.path.join(sim_path, "v_src"), comments="%")[:, 1]
    shutil.rmtree(sim_path, ignore_errors=True)
    return axis, src


@unittest.skipUnless(os.path.exists(OPENEMS_BIN), "no openEMS binary at %s (set OPENEMS_TEST_BIN)" % OPENEMS_BIN)
class TestCylinderAxis(unittest.TestCase):
    """E_z on the axis must keep the symmetry of the field around it."""

    def _check(self, engine):
        axis, src = _run(engine)
        peak = float(np.max(np.abs(src)))
        self.assertGreater(peak, 0.0, "the sources produced no field")
        ratio = float(np.max(np.abs(axis))) / peak
        self.assertLess(
            ratio, 1e-5,
            "%s: E_z on the axis reaches %.3e of the source voltage, it must stay zero "
            "for a field odd under a rotation by pi" % (engine, ratio),
        )

    def test_basic(self):
        self._check("basic")

    def test_multithreaded(self):
        self._check("multithreaded")


if __name__ == "__main__":
    unittest.main()
