# -*- coding: utf-8 -*-
"""Regression tests for in-batch sampling + replay: temporal blocking for every excitation.

Temporal blocking (TB) used to be capped by the probe interval: the main loop
in openems.cpp calls IterateTS(n) with the number of timesteps until the next
probe, dump or end-criterion check is due. A Dirac or step excitation has a
Nyquist interval of one timestep, so its probes are due every timestep and TB
never ran a single block. In-batch sampling lets the engine run long batches
and copy what the probes read (and the energy the end criterion needs) into a
ring buffer at every due timestep; the main loop replays the processing from
it afterwards. Field dumps still end a batch.

This file tests that feature against its contract (CONTRACT.md of the project,
quoted where used below), not against its implementation:

  * OPENEMS_BATCH_SAMPLING=0 runs the old loop; unset or 1 samples in-batch
    where the engine supports it. Every run prints exactly one of
        In-batch sampling ON: <c> probe cells, ring <r> timesteps
        In-batch sampling OFF: <reason>
    and OPENEMS_BATCH_SAMPLING_RING=<n> caps a batch at n sampled timesteps.
  * Every run with a TB request ends with
        Temporal blocking: <b> of <n> timesteps ran blocked
    where b counts the timesteps run inside blocks of depth >= 2.
  * OPENEMS_ENDCRIT_TRACE=1 prints one line per evaluation of the energy end
    criterion, OPENEMS_STEADYSTATE_TRACE=1 one per steady-state check, and
    OPENEMS_PROBE_PRECISION=17 writes the probe files with 17 digits.
  * Identity: with in-batch sampling, every probe file, dump, stop timestep
    and end-criterion trace is bit-identical to the old loop's on the same
    engine, for flat and blocked schedules alike.

Every case writes one model and runs it on one engine in these variants:

    ref          the old loop, flat (OPENEMS_BATCH_SAMPLING=0, no TB): the reference
    ref_again    the same again: the engine has to be deterministic to begin with
    ref_blocked  the old loop, blocked: as before this work, TB capped by the
                 probe interval -- so it may block only where that is >= 2
    new_flat     in-batch sampling (OPENEMS_BATCH_SAMPLING unset), flat
    new_<k>_<W>  in-batch sampling, blocked, over three (k, W) shapes and, on
                 AVX2, 1, 3 and 4 threads (ref and new_flat run 2)
    new_ring5    some cases: blocked 4:16 with OPENEMS_BATCH_SAMPLING=1 set and
                 OPENEMS_BATCH_SAMPLING_RING=5, so batches end every 5 samples

and asserts, in this order of test methods:

    test_a_reference      the reference ran, did not diverge, wrote non-zero
                          probes (and dumps), and stopped where the case
                          needs it to (energy, steady state or NrTS)
    test_b_old_loop       ref_again and ref_blocked are identical to ref, and
                          blocking engaged in ref_blocked (the engine's own
                          ACTIVE line): the machinery compares what it claims
    test_c_sampling_line  one "In-batch sampling" line per run, OFF in the old
                          loop, ON with probe cells > 0 in the new one
    test_d_ran_blocked    the "Temporal blocking" line: b > 0 in every new
                          blocked variant, b == 0 in the old loop where every
                          timestep is due (Dirac, step), b > 0 in the old loop
                          elsewhere, b <= n, n >= the iterations
    test_e_traces         OPENEMS_PROBE_PRECISION took effect (> 12 digits in
                          the probe files) and the end-criterion trace is
                          there (the steady-state trace for a sine)
    test_f_identity       every new variant against ref: the same files,
                          probe files byte for byte (the header timestamp
                          masked), every HDF5 dataset and attribute bit for
                          bit, the end-criterion and steady-state traces, the
                          stop message and "Time for <n> iterations". A
                          variant whose sampling or blocking did not engage
                          fails here as well, so no comparison passes
                          vacuously.

The cases (each one class per engine, AVX2_<case> and GPU_<case>):

    Dirac         a Dirac pulse into a lumped port: voltage and current probes,
                  E and H field probes on a tile edge, a voltage probe with
                  frequency samples, an E probe with a start/stop time; stops
                  on the energy criterion between two blocks
    Step          the same with a step excitation; ends by NrTS
    GaussOverSampled  a Gaussian with OverSampling=12 (probes every 2
                  timesteps, below k) and an E and an H mode-match probe
    Custom        a custom excitation, probes every 3 timesteps
    SineSteady    a sine in a lossy model, until steady state stops it
    DumpInterval  TD (E, H with node interpolation) and FD field dumps every
                  25 timesteps, a probe due every 6: dumps end the batches
    DumpWindow    a Dirac pulse with E and H dumps in two start/stop windows
    ExactEndcrit  --exact-endcriteria, a Gaussian stopping on the energy
                  criterion at a timestep no k divides
    NrTSMidBlock  a Gaussian ending by NrTS = 233, inside a block of every k
    Objects       a Dirac pulse among metal, a lossy dielectric and a series
                  RLC straddling the tile edges, Mur on x, PML_8 on y and z

The grid is 64 x 32 x 32 cells, so every run takes well under a second.
OPENEMS_AVX2_TEMPORAL_BLOCK_FORCE=1 and OPENEMS_GPU_TEMPORAL_BLOCK_MB=1 lift
the engines' performance guards that refuse a grid this small.

Each model is written to XML by a Python subprocess (OPENEMS_TEST_PYTHON, by
default this interpreter; it needs the CSXCAD and openEMS modules, any build
of them: they only write the XML) and run by the binary under test
(OPENEMS_TEST_BIN, default build/openEMS), whose stdout is what the asserts
read. Outputs go to OPENEMS_TEST_OUT (default <tmp>/openems_batch_sampling)
and are removed after each class unless OPENEMS_TEST_KEEP=1.
OPENEMS_TEST_ENGINES=avx2 or =gpu runs one engine only; the GPU classes skip
when the binary cannot start its Vulkan engine (OPENEMS_TEST_GPU_INDEX picks
the device). Run it as

    OPENEMS_TEST_BIN=/path/to/openEMS OPENEMS_TEST_PYTHON=/usr/bin/python3 \\
        python3 python/Tests/test_batch_sampling.py -v [AVX2_Dirac ...]
"""

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

import h5py
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OPENEMS_BIN = os.environ.get("OPENEMS_TEST_BIN", os.path.join(REPO_ROOT, "build", "openEMS"))
XML_PYTHON = os.environ.get("OPENEMS_TEST_PYTHON", sys.executable)
OUT_ROOT = os.environ.get("OPENEMS_TEST_OUT", os.path.join(tempfile.gettempdir(), "openems_batch_sampling"))
KEEP = os.environ.get("OPENEMS_TEST_KEEP", "") == "1"
ENGINES = [e.strip() for e in os.environ.get("OPENEMS_TEST_ENGINES", "avx2,gpu").split(",") if e.strip()]
GPU_INDEX = os.environ.get("OPENEMS_TEST_GPU_INDEX")
RUN_TIMEOUT = 120               # s per run; a run here takes well under one

NX, NY, NZ = 64, 32, 32
PROBE_PRECISION = 17
DEFAULT_PRECISION = 12          # the probe files' digits without OPENEMS_PROBE_PRECISION
RING = 5                        # OPENEMS_BATCH_SAMPLING_RING of the new_ring5 variant
EXCITATION_FILES = ("et", "ht") # the excitation signal openEMS writes next to the probes

# Every environment variable a variant sets is cleared first, so a run sees
# exactly its own settings and nothing inherited from the caller's shell.
_KNOBS = (
    "OPENEMS_BATCH_SAMPLING", "OPENEMS_BATCH_SAMPLING_RING", "OPENEMS_PROBE_PRECISION",
    "OPENEMS_ENDCRIT_TRACE", "OPENEMS_STEADYSTATE_TRACE",
    "OPENEMS_AVX2_TEMPORAL_BLOCK", "OPENEMS_AVX2_TEMPORAL_BLOCK_FORCE",
    "OPENEMS_GPU_TEMPORAL_BLOCK", "OPENEMS_GPU_TEMPORAL_BLOCK_MB",
    "OPENEMS_GPU_PIPELINE", "OPENEMS_GPU_SPEC_DEPTH", "OPENEMS_GPU_SPEC_MAX_DEPTH",
)

ENGINE_BANNER = {
    "avx2": "openEMS - enabled AVX2+FMA multithreaded engine",
    "gpu": "openEMS - enabled Vulkan GPU engine",
}
# What each engine prints when its blocked schedule is configured (not part
# of the contract; it predates this work and says blocking was set up, not
# that any timestep ran blocked).
ENGAGED_MARK = {
    "avx2": "AVX2 temporal blocking ACTIVE",
    "gpu": "trapezoidal temporal blocking active",
}


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

class Variant:
    """One way of running a model: old or new loop, flat or blocked."""

    def __init__(self, name, sampling, block=None, threads=2, ring=None, explicit=False):
        self.name = name
        self.sampling = sampling        # False: OPENEMS_BATCH_SAMPLING=0
        self.block = block              # "k:W" or None for the flat schedule
        self.threads = threads          # AVX2 only
        self.ring = ring                # OPENEMS_BATCH_SAMPLING_RING
        self.explicit = explicit        # OPENEMS_BATCH_SAMPLING=1 rather than unset

    @property
    def k(self):
        return int(self.block.split(":")[0]) if self.block else 1


def _variants(engine, ref_again=False, ring=False):
    gpu = engine == "gpu"
    v = [Variant("ref", False)]
    if ref_again:
        v.append(Variant("ref_again", False))
    v += [
        Variant("ref_blocked", False, "4:16", threads=3),
        Variant("new_flat", True),
        Variant("new_4_16" if gpu else "new_4_16_t1", True, "4:16", threads=1),
        Variant("new_3_13" if gpu else "new_3_13_t3", True, "3:13", threads=3),
        Variant("new_6_20" if gpu else "new_6_20_t4", True, "6:20", threads=4),
    ]
    if ring:
        v.append(Variant("new_ring%d" % RING, True, "4:16", threads=2, ring=RING, explicit=True))
    return v


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

_SCRIPT_TEMPLATE = """
import sys
import xml.etree.ElementTree as ET
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS

FDTD = openEMS(NrTS={nrts}, EndCriteria={endcrit!r}, OverSampling={oversampling})
{excitation}
FDTD.SetBoundaryCond({boundary!r})

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit({delta!r})
mesh.SetLines("x", np.arange({nx}, dtype=float))
mesh.SetLines("y", np.arange({ny}, dtype=float))
mesh.SetLines("z", np.arange({nz}, dtype=float))
xl, yl, zl = mesh.GetLines("x"), mesh.GetLines("y"), mesh.GetLines("z")

{setup}

FDTD.Write2XML({xml!r})

# The Python modules have no setter for a probe's or dump's start/stop time;
# CSXCAD reads them from these attributes.
windows = {windows!r}
if windows:
    tree = ET.parse({xml!r})
    found = set()
    for el in tree.getroot().iter():
        if el.tag in ("ProbeBox", "DumpBox") and el.get("Name") in windows:
            start, stop = windows[el.get("Name")]
            el.set("StartTime", repr(start))
            el.set("StopTime", repr(stop))
            found.add(el.get("Name"))
    if found != set(windows):
        sys.exit("no probe or dump named %s in the XML" % sorted(set(windows) - found))
    tree.write({xml!r}, encoding="UTF-8", xml_declaration=True)
"""

# A lumped port (50 ohm, excited; it writes port_ut_1 and port_it_1) in tile 1
# of the 4:16 schedule, E and H field probes beside the tile edge at x = 48, a
# voltage probe with frequency samples (ut_fd and ut_fd_FD), and an E probe
# that only records between 0.2 and 0.5 ns (timesteps 103 to 260 at the
# 1.926 ps timestep of the 1 mm grid).
PORT_AND_PROBES = """
port = FDTD.AddLumpedPort(1, 50.0, [xl[20], yl[14], zl[10]], [xl[20], yl[18], zl[22]], "z", excite=1.0, priority=5)
et = CSX.AddProbe("et_edge", p_type=2)
et.AddBox([xl[48], yl[16], zl[12]], [xl[48], yl[16], zl[12]])
ht = CSX.AddProbe("ht_edge", p_type=3)
ht.AddBox([xl[47], yl[15], zl[14]], [xl[47], yl[15], zl[14]])
ufd = CSX.AddProbe("ut_fd", p_type=0, frequency=[3e9, 6e9, 9e9])
ufd.AddBox([xl[44], yl[16], zl[10]], [xl[44], yl[16], zl[20]])
ewin = CSX.AddProbe("et_win", p_type=2)
ewin.AddBox([xl[33], yl[16], zl[16]], [xl[33], yl[16], zl[16]])
"""
PROBE_WINDOWS = {"et_win": (2.0e-10, 5.0e-10)}

# The steady-state cases of test_temporal_blocking.py fill the grid with a
# lossy dielectric so that a sine settles within about a thousand timesteps.
LOSSY = """
lossy = CSX.AddMaterial("lossy", epsilon=2.0, kappa=0.1)
lossy.AddBox([xl[0], yl[0], zl[0]], [xl[-1], yl[-1], zl[-1]], priority=1)
"""


class CaseBase:
    """A model and what its reference run has to show. Subclasses set these."""

    CASE = None
    EXCITATION = None
    NRTS = 600
    ENDCRIT = 1e-9              # small enough not to stop, but evaluated (traced)
    OVERSAMPLING = 4
    BOUNDARY = ["MUR"] * 6
    DELTA = 1e-3
    SETUP = PORT_AND_PROBES
    WINDOWS = PROBE_WINDOWS
    EXTRA_ARGS = []
    STOPS_ON = None             # "energy", "steady" or None (NrTS)
    INTERVAL_ONE = False        # something is due every timestep: the old loop cannot block
    REF_AGAIN = True
    RING_VARIANT = False
    HAS_DUMPS = False


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

_xml_cache = {}


def _write_model(case):
    """Write the case's XML once per test process; both engines share it."""
    if case.CASE in _xml_cache:
        return _xml_cache[case.CASE]
    model_dir = os.path.join(OUT_ROOT, "models", case.CASE)
    shutil.rmtree(model_dir, ignore_errors=True)
    os.makedirs(model_dir)
    xml = os.path.join(model_dir, "model.xml")
    script = _SCRIPT_TEMPLATE.format(
        nrts=case.NRTS, endcrit=case.ENDCRIT, oversampling=case.OVERSAMPLING,
        excitation=case.EXCITATION, boundary=case.BOUNDARY, delta=case.DELTA,
        nx=NX, ny=NY, nz=NZ, setup=textwrap.dedent(case.SETUP).strip(),
        xml=xml, windows=case.WINDOWS or {},
    )
    script_path = os.path.join(model_dir, "model.py")
    with open(script_path, "w") as f:
        f.write(script)
    proc = subprocess.run([XML_PYTHON, script_path], cwd=model_dir, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0 or not os.path.exists(xml):
        raise RuntimeError("writing the model failed (case=%s, rc=%s, python=%s):\n%s\n%s"
                           % (case.CASE, proc.returncode, XML_PYTHON, proc.stdout[-4000:], proc.stderr[-4000:]))
    _xml_cache[case.CASE] = xml
    return xml


class Run:
    """The outcome of one variant: its output and the files it wrote."""

    def __init__(self, variant, path, rc, stdout, stderr):
        self.variant = variant
        self.path = path
        self.rc = rc
        self.stdout = stdout
        self.output = stdout + stderr
        self.files = sorted(f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))

    def text_files(self):
        """Every file but the dumps: the probes, their _FD files, and the
        excitation signal ('et', 'ht'). All of them are compared."""
        return [f for f in self.files if not f.endswith(".h5")]

    def probe_files(self):
        """The time-domain probe files, which OPENEMS_PROBE_PRECISION governs."""
        return [f for f in self.text_files() if f not in EXCITATION_FILES and not f.endswith("_FD")]

    def h5_files(self):
        return [f for f in self.files if f.endswith(".h5")]

    def read(self, name):
        with open(os.path.join(self.path, name), "rb") as f:
            return f.read()


def _env_for(engine, variant):
    env = dict(os.environ)
    for k in _KNOBS:
        env.pop(k, None)
    env["OPENEMS_PROBE_PRECISION"] = str(PROBE_PRECISION)
    env["OPENEMS_ENDCRIT_TRACE"] = "1"
    env["OPENEMS_STEADYSTATE_TRACE"] = "1"
    if not variant.sampling:
        env["OPENEMS_BATCH_SAMPLING"] = "0"
    elif variant.explicit:
        env["OPENEMS_BATCH_SAMPLING"] = "1"
    if variant.ring:
        env["OPENEMS_BATCH_SAMPLING_RING"] = str(variant.ring)
    if engine == "avx2":
        if variant.block:
            env["OPENEMS_AVX2_TEMPORAL_BLOCK_FORCE"] = "1"
    else:
        if GPU_INDEX is not None:
            env["OPENEMS_GPU_INDEX"] = GPU_INDEX
        # Blocking is on by default for large grids; the flat variants turn it off.
        env["OPENEMS_GPU_TEMPORAL_BLOCK"] = variant.block or "0"
        if variant.block:
            env["OPENEMS_GPU_TEMPORAL_BLOCK_MB"] = "1"
    return env


def _command(engine, xml, variant, extra_args):
    if engine == "avx2":
        cmd = [OPENEMS_BIN, xml, "--engine=avx2-multithreaded", "--numThreads=%d" % variant.threads]
        if variant.block:
            cmd.append("--avx2-temporal-block=%s" % variant.block)
    else:
        cmd = [OPENEMS_BIN, xml, "--engine=gpu"]
    return cmd + list(extra_args)


def _run_variant(engine, case, xml, variant, root):
    path = os.path.join(root, variant.name)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    cmd = _command(engine, xml, variant, case.EXTRA_ARGS)
    env = _env_for(engine, variant)
    proc = subprocess.run(cmd, cwd=path, env=env, capture_output=True, text=True, timeout=RUN_TIMEOUT)
    # Next to the run's directory, not in it: everything in it is compared.
    with open(path + ".log", "w") as f:
        f.write("# %s\n# %s\n%s\n# stderr:\n%s" % (
            " ".join("%s=%s" % (k, env[k]) for k in sorted(env) if k.startswith("OPENEMS_")),
            " ".join(cmd), proc.stdout, proc.stderr))
    return Run(variant, path, proc.returncode, proc.stdout, proc.stderr)


_gpu_available = None


def _gpu_check():
    """(available, why): can the binary start its Vulkan engine at all?"""
    global _gpu_available
    if _gpu_available is not None:
        return _gpu_available
    path = os.path.join(OUT_ROOT, "gpu_probe")
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    xml = os.path.join(path, "tiny.xml")
    with open(xml, "w") as f:
        f.write(_TINY_XML)
    env = dict(os.environ)
    if GPU_INDEX is not None:
        env["OPENEMS_GPU_INDEX"] = GPU_INDEX
    try:
        proc = subprocess.run([OPENEMS_BIN, xml, "--engine=gpu"], cwd=path, env=env,
                              capture_output=True, text=True, timeout=120)
        out = proc.stdout + proc.stderr
        if proc.returncode == 0 and ENGINE_BANNER["gpu"] in out:
            _gpu_available = (True, "")
        else:
            tail = "\n".join(out.splitlines()[-5:])
            _gpu_available = (False, "the binary did not start its Vulkan engine (rc=%s):\n%s"
                              % (proc.returncode, tail))
    except (OSError, subprocess.SubprocessError) as e:
        _gpu_available = (False, "running the binary failed: %s" % e)
    shutil.rmtree(path, ignore_errors=True)
    return _gpu_available


_TINY_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<openEMS>
  <FDTD NumberOfTimesteps="10" endCriteria="1e-30" f_max="2e9">
    <Excitation Type="0" f0="1e9" fc="1e9"/>
    <BoundaryCond xmin="0" xmax="0" ymin="0" ymax="0" zmin="0" zmax="0"/>
  </FDTD>
  <ContinuousStructure CoordSystem="0">
    <Properties>
      <Excitation ID="0" Name="exc" Type="0" Excite="0,0,1">
        <Primitives><Box Priority="0"><P1 X="3" Y="3" Z="3"/><P2 X="4" Y="4" Z="4"/></Box></Primitives>
      </Excitation>
    </Properties>
    <RectilinearGrid DeltaUnit="0.001" CoordSystem="0">
      <XLines>0,1,2,3,4,5,6,7</XLines>
      <YLines>0,1,2,3,4,5,6,7</YLines>
      <ZLines>0,1,2,3,4,5,6,7</ZLines>
    </RectilinearGrid>
  </ContinuousStructure>
</openEMS>
"""


# ---------------------------------------------------------------------------
# Reading a run's output
# ---------------------------------------------------------------------------

_SAMPLING_RE = re.compile(r"^In-batch sampling (ON|OFF)\b.*$", re.M)
_SAMPLING_ON_RE = re.compile(r"^In-batch sampling ON: (\d+) probe cells, ring (\d+) timesteps\s*$")
_TB_RE = re.compile(r"^Temporal blocking: (\d+) of (\d+) timesteps ran blocked\s*$", re.M)
_ENDCRIT_RE = re.compile(r"^endcrit trace: ts=(\d+) energy=(\S+) maxE=(\S+) change=(\S+)\s*$")
_ITER_RE = re.compile(r"Time for (\d+) iterations")
_STOP_PREFIXES = ("RunFDTD: residual field energy reached", "RunFDTD: steady-state criterion of")


def sampling_lines(run):
    return [m.group(0) for m in _SAMPLING_RE.finditer(run.output)]


def tb_lines(run):
    return [(int(m.group(1)), int(m.group(2))) for m in _TB_RE.finditer(run.output)]


def endcrit_trace(run):
    return [l for l in run.stdout.splitlines() if l.startswith("endcrit trace:")]


def steady_trace(run):
    return [l for l in run.stdout.splitlines() if l.startswith("SteadyState check:")]


def stop_lines(run):
    return [l for l in run.stdout.splitlines() if l.startswith(_STOP_PREFIXES)]


def stop_timestep(run):
    for l in stop_lines(run):
        m = re.search(r"after (\d+) timesteps", l)
        if m:
            return int(m.group(1))
    return None


def iterations(run):
    m = _ITER_RE.findall(run.stdout)
    return int(m[-1]) if m else None


def timestep(run):
    m = re.search(r"FDTD timestep is: (\S+) s", run.stdout)
    return float(m.group(1)) if m else None


def diverged(run):
    return [l for l in run.output.splitlines() if "cancelling simulation" in l]


def _mask_header(data):
    """A probe file with the creation time of its header line removed.

    The first line reads '% time-domain <probe> by openEMS <version> @<ctime>';
    the time is the only part two identical runs may differ in."""
    out = []
    for line in data.split(b"\n"):
        if line.startswith(b"%") and b" by openEMS " in line and b" @" in line:
            line = line[:line.rindex(b" @")] + b" @<time>"
        out.append(line)
    return b"\n".join(out)


def _data_rows(data):
    return [l.split() for l in data.decode("utf-8", "replace").splitlines()
            if l.strip() and not l.startswith("%")]


def _significant_digits(token):
    mant = token.lstrip("+-").lower().split("e")[0].replace(".", "").lstrip("0")
    return len(mant) if mant.isdigit() else 0


def _first_difference(a, b):
    la, lb = a.split(b"\n"), b.split(b"\n")
    for i, (x, y) in enumerate(zip(la, lb)):
        if x != y:
            return "line %d: %r vs %r" % (i + 1, x[:160], y[:160])
    return "lengths differ: %d vs %d lines" % (len(la), len(lb))


def _h5_contents(path):
    """Every dataset and attribute of an HDF5 file, as name -> (dtype, shape, digest)."""
    items = {}

    def describe(value):
        a = np.asarray(value)
        return (a.dtype.str, a.shape, hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())

    def visit(name, obj):
        for k, v in obj.attrs.items():
            items["attr %s@%s" % (name, k)] = describe(v)
        if isinstance(obj, h5py.Dataset):
            items["data %s" % name] = describe(obj[()])

    with h5py.File(path, "r") as f:
        visit("/", f)
        f.visititems(visit)
    return items


def _h5_dataset_delta(path_a, path_b, name):
    with h5py.File(path_a, "r") as fa, h5py.File(path_b, "r") as fb:
        a, b = np.asarray(fa[name][()]), np.asarray(fb[name][()])
    if a.shape != b.shape or a.dtype.kind not in "fc":
        return ""
    d = np.abs(a.astype(np.complex128) - b.astype(np.complex128))
    return ", max|d| = %.3e of max|ref| = %.3e" % (float(d.max()), float(np.abs(a).max()))


def compare_runs(ref, other):
    """Everything in which `other` differs from `ref`, as a list of strings."""
    problems = []
    if ref.files != other.files:
        problems.append("different output files: only in ref %s, only in %s %s"
                        % (sorted(set(ref.files) - set(other.files)), other.variant.name,
                           sorted(set(other.files) - set(ref.files))))
    for name in sorted(set(ref.text_files()) & set(other.text_files())):
        a, b = _mask_header(ref.read(name)), _mask_header(other.read(name))
        if a != b:
            problems.append("probe file %s differs, %s" % (name, _first_difference(a, b)))
    for name in sorted(set(ref.h5_files()) & set(other.h5_files())):
        pa, pb = os.path.join(ref.path, name), os.path.join(other.path, name)
        ca, cb = _h5_contents(pa), _h5_contents(pb)
        if set(ca) != set(cb):
            problems.append("dump %s: different datasets/attributes: only in ref %s, only in %s %s"
                            % (name, sorted(set(ca) - set(cb))[:5], other.variant.name,
                               sorted(set(cb) - set(ca))[:5]))
        bad = [k for k in sorted(set(ca) & set(cb)) if ca[k] != cb[k]]
        if bad:
            detail = ""
            if bad[0].startswith("data "):
                detail = _h5_dataset_delta(pa, pb, bad[0][len("data "):])
            problems.append("dump %s: %d of %d datasets/attributes differ, first %s%s"
                            % (name, len(bad), len(ca), bad[0], detail))
    for what, get in (("end-criterion trace", endcrit_trace), ("steady-state trace", steady_trace)):
        ta, tb = get(ref), get(other)
        if ta != tb:
            i = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
            problems.append("%s differs (%d vs %d lines), first at line %d: %r vs %r"
                            % (what, len(ta), len(tb), i + 1, ta[i] if i < len(ta) else None,
                               tb[i] if i < len(tb) else None))
    if stop_lines(ref) != stop_lines(other):
        problems.append("stop message differs: %r vs %r" % (stop_lines(ref), stop_lines(other)))
    if iterations(ref) != iterations(other):
        problems.append("Time for <n> iterations: %s vs %s" % (iterations(ref), iterations(other)))
    return problems


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------

class BatchSamplingTests:
    """The asserts every case makes; mixed into one TestCase per case and engine."""

    ENGINE = None

    @classmethod
    def setUpClass(cls):
        if cls.ENGINE not in ENGINES:
            raise unittest.SkipTest("engine %s not selected (OPENEMS_TEST_ENGINES)" % cls.ENGINE)
        if cls.ENGINE == "gpu":
            ok, why = _gpu_check()
            if not ok:
                raise unittest.SkipTest("no usable Vulkan device: " + why)
        cls.variants = _variants(cls.ENGINE, ref_again=cls.REF_AGAIN, ring=cls.RING_VARIANT)
        cls.root = os.path.join(OUT_ROOT, "%s_%s" % (cls.ENGINE, cls.CASE))
        shutil.rmtree(cls.root, ignore_errors=True)
        os.makedirs(cls.root)
        xml = _write_model(cls)
        cls.runs, cls.errors = {}, {}
        for v in cls.variants:
            try:
                cls.runs[v.name] = _run_variant(cls.ENGINE, cls, xml, v, cls.root)
            except subprocess.TimeoutExpired as e:
                cls.errors[v.name] = "timed out after %s s" % e.timeout

    @classmethod
    def tearDownClass(cls):
        if not KEEP and getattr(cls, "root", None):
            shutil.rmtree(cls.root, ignore_errors=True)

    # -- helpers ------------------------------------------------------------

    def run_of(self, name):
        """The run of variant `name`; fails the test if it did not finish cleanly."""
        if name in self.errors:
            self.fail("%s/%s: %s" % (self.ENGINE, name, self.errors[name]))
        run = self.runs[name]
        tail = "\n".join(run.output.splitlines()[-15:])
        self.assertEqual(run.rc, 0, "%s/%s exited with %s:\n%s" % (self.ENGINE, name, run.rc, tail))
        self.assertIn(ENGINE_BANNER[self.ENGINE], run.output,
                      "%s/%s did not run on the %s engine:\n%s" % (self.ENGINE, name, self.ENGINE, tail))
        if diverged(run):
            self.fail("%s/%s diverged: %s" % (self.ENGINE, name, diverged(run)))
        return run

    def new_variants(self):
        return [v for v in self.variants if v.sampling]

    def old_variants(self):
        return [v for v in self.variants if not v.sampling]

    def _sampling_problem(self, run):
        lines = sampling_lines(run)
        if len(lines) != 1:
            return "expected exactly one 'In-batch sampling' line, found %d%s" % (
                len(lines), (": %r" % lines) if lines else "")
        if not run.variant.sampling:
            if not lines[0].startswith("In-batch sampling OFF:"):
                return "OPENEMS_BATCH_SAMPLING=0 but the run says %r" % lines[0]
            return None
        m = _SAMPLING_ON_RE.match(lines[0])
        if not m:
            return "in-batch sampling did not engage: %r" % lines[0]
        cells, ring = int(m.group(1)), int(m.group(2))
        if cells <= 0:
            return "the model has probes, but the run samples %d probe cells: %r" % (cells, lines[0])
        if ring < 1:
            return "ring of %d timesteps: %r" % (ring, lines[0])
        if run.variant.ring and ring != run.variant.ring:
            return "OPENEMS_BATCH_SAMPLING_RING=%d, but the run says %r" % (run.variant.ring, lines[0])
        return None

    def _blocking_problem(self, run):
        """None if the TB line says what it must for this variant, else why not."""
        problem = self._blocking_problem_bare(run)
        if problem and run.variant.block:
            status = [l.strip() for l in run.output.splitlines() if "temporal blocking" in l.lower()]
            problem += " [engine: %s]" % (" | ".join(status) or "no temporal-blocking line at all")
        return problem

    def _blocking_problem_bare(self, run):
        v, lines = run.variant, tb_lines(run)
        if not v.block:
            if any(b != 0 for b, _ in lines):
                return "a flat run reports blocked timesteps: %r" % lines
            return None
        if len(lines) != 1:
            return "expected one 'Temporal blocking: <b> of <n> timesteps ran blocked' line, found %d" % len(lines)
        b, n = lines[0]
        # n may exceed the iterations the run reports if the engine computed
        # past a stop it only found on replay; it may not fall short of them.
        if iterations(run) is None or n < iterations(run):
            return "the TB line counts %d timesteps, the run %s" % (n, iterations(run))
        if b > n:
            return "%d of %d timesteps blocked" % (b, n)
        if v.sampling and b == 0:
            return "no timestep ran blocked (0 of %d) although in-batch sampling should lift the " \
                   "probe-interval cap" % n
        if not v.sampling and self.INTERVAL_ONE and b != 0:
            return "the old loop ran %d of %d timesteps blocked, but something is due every timestep " \
                   "here, so every batch of the old loop is one timestep" % (b, n)
        if not v.sampling and not self.INTERVAL_ONE and b == 0:
            return "the old loop ran no timestep blocked (0 of %d), although nothing is due every timestep" % n
        return None

    # -- the tests ------------------------------------------------------------

    def test_a_reference(self):
        """The reference ran, is not vacuous and stopped as the case needs."""
        ref = self.run_of("ref")
        probes = ref.probe_files()
        self.assertTrue(probes, "the reference wrote no probe file")
        for name in probes:
            rows = _data_rows(ref.read(name))
            self.assertTrue(rows, "probe file %s has no data rows" % name)
            values = np.array([[float(x) for x in r[1:]] for r in rows])
            self.assertGreater(float(np.max(np.abs(values))), 0.0,
                               "probe %s recorded only zeros -- the wave never reached it" % name)
        for name, (start, stop) in (self.WINDOWS or {}).items():
            if name in ref.text_files():
                times = [float(r[0]) for r in _data_rows(ref.read(name))]
                self.assertGreaterEqual(min(times), start - 2.0e-12, "%s recorded before its start time" % name)
                self.assertLessEqual(max(times), stop + 2.0e-12, "%s recorded after its stop time" % name)
        if self.HAS_DUMPS:
            self.assertTrue(ref.h5_files(), "the reference wrote no HDF5 dump")
            for name in ref.h5_files():
                with h5py.File(os.path.join(ref.path, name), "r") as f:
                    grp = f["FieldData"]
                    data = [np.asarray(grp[g][k][()]) for g in grp for k in grp[g]]
                self.assertTrue(data, "dump %s holds no field data" % name)
                self.assertTrue(all(np.all(np.isfinite(d)) for d in data), "dump %s is not finite" % name)
                self.assertGreater(max(float(np.max(np.abs(d))) for d in data), 0.0,
                                   "dump %s holds only zeros" % name)

        n, stop = iterations(ref), stop_timestep(ref)
        if self.STOPS_ON is None:
            self.assertEqual(n, self.NRTS, "the reference did not run to NrTS: %s" % stop_lines(ref))
            self.assertIsNone(stop, "the reference stopped early: %s" % stop_lines(ref))
        elif self.STOPS_ON == "energy":
            self.assertIsNotNone(stop, "the reference did not stop on the energy criterion")
            self.assertIn("residual field energy", stop_lines(ref)[0])
            self.assertLess(stop, self.NRTS)
        else:
            self.assertIsNotNone(stop, "the reference did not stop on the steady-state criterion")
            self.assertIn("steady-state criterion", stop_lines(ref)[0])
            self.assertGreaterEqual(len(steady_trace(ref)), 4,
                                    "too few steady-state checks to compare:\n%s" % "\n".join(steady_trace(ref)))
        self.check_case(ref)

    def check_case(self, ref):
        """Case-specific preconditions of the reference run."""

    def test_b_old_loop(self):
        """The old loop agrees with itself: run again, and blocked."""
        ref = self.run_of("ref")
        for v in self.old_variants():
            if v.name == "ref":
                continue
            with self.subTest(variant=v.name):
                run = self.run_of(v.name)
                if v.block:
                    self.assertIn(ENGAGED_MARK[self.ENGINE], run.output,
                                  "%s: the engine did not set up blocking, so this comparison would be "
                                  "vacuous:\n%s" % (v.name, "\n".join(l for l in run.output.splitlines()
                                                                     if "temporal blocking" in l.lower())))
                problems = compare_runs(ref, run)
                if problems:
                    self.fail("%s differs from ref:\n  %s" % (v.name, "\n  ".join(problems)))

    def test_c_sampling_line(self):
        """One 'In-batch sampling ON/OFF' line per run, as the variant asks."""
        failures = []
        for v in self.variants:
            problem = self._sampling_problem(self.run_of(v.name))
            if problem:
                failures.append("%s: %s" % (v.name, problem))
        if failures:
            self.fail("\n".join(failures))

    def test_d_ran_blocked(self):
        """The 'Temporal blocking: <b> of <n> timesteps ran blocked' line."""
        failures = []
        for v in self.variants:
            problem = self._blocking_problem(self.run_of(v.name))
            if problem:
                failures.append("%s: %s" % (v.name, problem))
        if failures:
            self.fail("\n".join(failures))

    def test_e_traces(self):
        """OPENEMS_PROBE_PRECISION and the end-criterion trace took effect."""
        ref = self.run_of("ref")
        failures = []
        digits = {name: max((_significant_digits(t) for r in _data_rows(ref.read(name)) for t in r), default=0)
                  for name in ref.probe_files()}
        short = {name: d for name, d in digits.items() if d <= DEFAULT_PRECISION}
        if not digits or short:
            failures.append("OPENEMS_PROBE_PRECISION=%d had no effect: these probe files carry at most this "
                            "many significant digits: %s" % (PROBE_PRECISION, short or digits))
        if self.STOPS_ON == "steady":
            # The energy criterion is not evaluated next to the steady-state one.
            if len(steady_trace(ref)) < 4:
                failures.append("OPENEMS_STEADYSTATE_TRACE=1 printed %d check lines" % len(steady_trace(ref)))
        else:
            trace = endcrit_trace(ref)
            bad = [l for l in trace if not _ENDCRIT_RE.match(l)]
            ts = [int(_ENDCRIT_RE.match(l).group(1)) for l in trace if _ENDCRIT_RE.match(l)]
            if not trace:
                failures.append("OPENEMS_ENDCRIT_TRACE=1 printed no 'endcrit trace:' line")
            elif bad:
                failures.append("malformed end-criterion trace lines: %r" % bad[:3])
            elif ts != sorted(set(ts)):
                failures.append("the end-criterion trace is not one line per evaluated timestep, in order")
            elif self.STOPS_ON == "energy" and ts[-1] != stop_timestep(ref):
                failures.append("the last end-criterion evaluation (ts=%d) is not the timestep the run "
                                "stopped at (%s)" % (ts[-1], stop_timestep(ref)))
        if failures:
            self.fail("\n".join(failures))

    def test_f_identity(self):
        """Every in-batch-sampling variant is bit-identical to the old loop's reference."""
        ref = self.run_of("ref")
        for v in self.new_variants():
            with self.subTest(variant=v.name):
                run = self.run_of(v.name)
                problems = compare_runs(ref, run)
                engaged = [p for p in (self._sampling_problem(run), self._blocking_problem(run)) if p]
                msg = []
                if problems:
                    msg.append("not identical to ref:\n  " + "\n  ".join(problems))
                if engaged:
                    msg.append("%s (identity with ref: %s)" % ("; ".join(engaged), "FAILED" if problems else "OK"))
                if msg:
                    self.fail("%s/%s: %s" % (self.ENGINE, v.name, "\n".join(msg)))


def tearDownModule():
    if not KEEP:
        shutil.rmtree(os.path.join(OUT_ROOT, "models"), ignore_errors=True)


class AVX2:
    ENGINE = "avx2"


class GPU:
    ENGINE = "gpu"


# ---------------------------------------------------------------------------
# The cases
# ---------------------------------------------------------------------------

class DiracCase(CaseBase):
    """A Dirac pulse: the Nyquist interval, and with it the probe and the
    end-criterion interval, is one timestep -- the case this work is for. The
    energy criterion stops the run at a timestep no k of the schedules divides,
    so a run that only checks it per batch or per block would stop elsewhere."""

    CASE = "dirac"
    EXCITATION = "FDTD.SetDiracExcite(10e9)"
    NRTS = 1200
    ENDCRIT = 10 ** (-23 / 10.0)        # stops at timestep 179 on both engines
    STOPS_ON = "energy"
    INTERVAL_ONE = True
    RING_VARIANT = True

    def check_case(self, ref):
        stop = stop_timestep(ref)
        self.assertTrue(all(stop % v.k for v in self.variants if v.block),
                        "the Dirac case stops at %d, a multiple of a block depth: pick another EndCriteria" % stop)


class StepCase(CaseBase):
    """A step excitation (Nyquist interval one), which does not decay: the
    run ends by NrTS, at a timestep in the middle of a block of every k."""

    CASE = "step"
    EXCITATION = "FDTD.SetStepExcite(10e9)"
    NRTS = 397
    INTERVAL_ONE = True


class GaussOverSampledCase(CaseBase):
    """A Gaussian pulse with OverSampling=12: the probes are due every 2
    timesteps (Nyquist 25 / 12), below every k, so the old loop's blocks are
    two deep at most. An E and an H mode-match probe over a plane at x = 44
    read many cells per sample."""

    CASE = "gauss_os12"
    EXCITATION = "FDTD.SetGaussExcite(5e9, 5e9)"
    OVERSAMPLING = 12
    NRTS = 1200
    ENDCRIT = 10 ** (-33 / 10.0)
    STOPS_ON = "energy"
    SETUP = PORT_AND_PROBES + """
mme = CSX.AddProbe("mm_e", p_type=10, mode_function=["0", "0", "cos(pi*(y-16)/28)*cos(pi*(z-16)/28)"])
mme.AddBox([xl[44], yl[2], zl[2]], [xl[44], yl[30], zl[30]])
mmh = CSX.AddProbe("mm_h", p_type=11, mode_function=["0", "cos(pi*(y-16)/28)*cos(pi*(z-16)/28)", "0"])
mmh.AddBox([xl[44], yl[2], zl[2]], [xl[44], yl[30], zl[30]])
"""

    def check_case(self, ref):
        rows = _data_rows(ref.read("et_edge"))
        steps = np.round(np.diff([float(r[0]) for r in rows]) / timestep(ref)).astype(int)
        self.assertEqual(int(np.median(steps)), 2, "the probes are not due every 2 timesteps")


class CustomCase(CaseBase):
    """A custom excitation (a modulated Gaussian written as a function of t).
    Its f0 of 20 GHz gives a Nyquist interval of 12 timesteps and probes every
    3, so k = 4 and 6 reach past a probe and k = 3 lands on every one."""

    CASE = "custom"
    EXCITATION = 'FDTD.SetCustomExcite("sin(2*pi*6e9*t)*exp(-((t-3e-10)/1e-10)^2)", 20e9, 20e9)'
    NRTS = 1500
    ENDCRIT = 1e-4
    STOPS_ON = "energy"


class SineSteadyCase(CaseBase):
    """A sine in a lossy model with PEC walls, until the steady-state
    detection stops it."""

    CASE = "sine_steady"
    EXCITATION = "FDTD.SetSinusExcite(5e9)"
    NRTS = 20000
    ENDCRIT = 1e-3
    BOUNDARY = ["PEC"] * 6
    SETUP = LOSSY + PORT_AND_PROBES
    STOPS_ON = "steady"


class DumpIntervalCase(CaseBase):
    """Field dumps, which end a batch. With OverSampling=1 the dumps and the
    time-domain probes are due every 25 timesteps, but a voltage probe with a
    40 GHz frequency sample is due every 6, so a batch passes probe timesteps
    and has to end at the dumps (25 is no multiple of any k). E in full (no
    interpolation, every second line), H on a plane with node interpolation,
    and an E frequency-domain dump."""

    CASE = "dump_interval"
    EXCITATION = "FDTD.SetGaussExcite(5e9, 5e9)"
    OVERSAMPLING = 1
    NRTS = 420
    HAS_DUMPS = True
    SETUP = PORT_AND_PROBES + """
fast = CSX.AddProbe("ut_fast", p_type=0, frequency=[40e9])
fast.AddBox([xl[52], yl[16], zl[10]], [xl[52], yl[16], zl[20]])
de = CSX.AddDump("Et", dump_type=0, dump_mode=0, file_type=1, sub_sampling=[2, 2, 2])
de.AddBox([xl[0], yl[0], zl[0]], [xl[-1], yl[-1], zl[-1]])
dh = CSX.AddDump("Ht_plane", dump_type=1, dump_mode=1, file_type=1)
dh.AddBox([xl[0], yl[0], zl[16]], [xl[-1], yl[-1], zl[16]])
df = CSX.AddDump("Ef", dump_type=10, dump_mode=0, file_type=1, frequency=[5e9], sub_sampling=[2, 2, 2])
df.AddBox([xl[0], yl[0], zl[0]], [xl[-1], yl[-1], zl[-1]])
"""

    def check_case(self, ref):
        with h5py.File(os.path.join(ref.path, "Et.h5"), "r") as f:
            frames = len(f["FieldData/TD"])
        self.assertGreaterEqual(frames, 10, "the E dump has only %d frames" % frames)


class DumpWindowCase(CaseBase):
    """A Dirac pulse with an E dump on the plane x = 40 between 0.1 and 0.2 ns
    and an H dump on z = 16 between 0.4 and 0.45 ns: every timestep inside a
    window ends a batch, outside them the batches may run long."""

    CASE = "dump_window"
    EXCITATION = "FDTD.SetDiracExcite(10e9)"
    NRTS = 400
    INTERVAL_ONE = True
    HAS_DUMPS = True
    SETUP = PORT_AND_PROBES + """
de = CSX.AddDump("Et_win", dump_type=0, dump_mode=0, file_type=1)
de.AddBox([xl[40], yl[0], zl[0]], [xl[40], yl[-1], zl[-1]])
dh = CSX.AddDump("Ht_win", dump_type=1, dump_mode=0, file_type=1)
dh.AddBox([xl[0], yl[0], zl[16]], [xl[-1], yl[-1], zl[16]])
"""
    WINDOWS = dict(PROBE_WINDOWS, Et_win=(1.0e-10, 2.0e-10), Ht_win=(4.0e-10, 4.5e-10))

    def check_case(self, ref):
        with h5py.File(os.path.join(ref.path, "Et_win.h5"), "r") as f:
            frames = len(f["FieldData/TD"])
        self.assertTrue(40 <= frames <= 60, "the windowed E dump has %d frames, not about 52" % frames)


class ExactEndcritCase(CaseBase):
    """--exact-endcriteria: the energy criterion every Nyquist interval (25
    timesteps) for the whole run, probes every 3 (OverSampling=8). The run
    stops on the criterion at a multiple of 25 that no k divides, so stopping
    at the end of the batch or block that holds it would give another
    timestep, other probe files and another trace."""

    CASE = "exact_endcrit"
    EXCITATION = "FDTD.SetGaussExcite(5e9, 5e9)"
    OVERSAMPLING = 8
    NRTS = 3000
    ENDCRIT = 10 ** (-33 / 10.0)        # stops at timestep 350 on both engines
    EXTRA_ARGS = ["--exact-endcriteria"]
    STOPS_ON = "energy"
    RING_VARIANT = True

    def check_case(self, ref):
        self.assertIn("Exact-endcriteria: evaluating the end criteria every 25 timestep(s)", ref.stdout)
        stop = stop_timestep(ref)
        self.assertEqual(stop % 25, 0)
        ks = sorted({v.k for v in self.variants if v.block})
        self.assertTrue(all(stop % k for k in ks),
                        "the run stops at %d, which one of %s divides: pick another EndCriteria" % (stop, ks))


class NrTSMidBlockCase(CaseBase):
    """A Gaussian with probes every 5 timesteps (OverSampling=5) that ends by
    NrTS = 233: not a multiple of any k, nor of the probe interval, so the
    last batch has to be cut short of a full block."""

    CASE = "nrts_mid_block"
    EXCITATION = "FDTD.SetGaussExcite(5e9, 5e9)"
    OVERSAMPLING = 5
    NRTS = 233

    def check_case(self, ref):
        self.assertTrue(all(self.NRTS % k for k in (3, 4, 5, 6, RING)))


class ObjectsCase(CaseBase):
    """A Dirac pulse among objects that straddle the tile edges of every
    schedule (x = 39, 40, 48, 52): a metal block, a lossy dielectric and a
    series RLC element, with Mur on the x faces and PML_8 on the others."""

    CASE = "objects"
    EXCITATION = "FDTD.SetDiracExcite(10e9)"
    NRTS = 500
    BOUNDARY = ["MUR", "MUR", "PML_8", "PML_8", "PML_8", "PML_8"]
    INTERVAL_ONE = True
    SETUP = PORT_AND_PROBES + """
metal = CSX.AddMetal("block")
metal.AddBox([xl[38], yl[8], zl[8]], [xl[54], yl[12], zl[11]], priority=10)
diel = CSX.AddMaterial("diel", epsilon=3.5, kappa=0.02)
diel.AddBox([xl[38], yl[18], zl[16]], [xl[54], yl[24], zl[24]], priority=5)
rlc = CSX.AddLumpedElement("rlc", ny="z", caps=True, R=75.0, L=5e-9, C=2e-12, LEtype=1)
rlc.AddBox([xl[46], yl[14], zl[12]], [xl[50], yl[17], zl[15]], priority=10)
"""


class AVX2_Dirac(DiracCase, AVX2, BatchSamplingTests, unittest.TestCase):
    pass


class GPU_Dirac(DiracCase, GPU, BatchSamplingTests, unittest.TestCase):
    pass


class AVX2_Step(StepCase, AVX2, BatchSamplingTests, unittest.TestCase):
    pass


class GPU_Step(StepCase, GPU, BatchSamplingTests, unittest.TestCase):
    pass


class AVX2_GaussOverSampled(GaussOverSampledCase, AVX2, BatchSamplingTests, unittest.TestCase):
    pass


class GPU_GaussOverSampled(GaussOverSampledCase, GPU, BatchSamplingTests, unittest.TestCase):
    pass


class AVX2_Custom(CustomCase, AVX2, BatchSamplingTests, unittest.TestCase):
    pass


class GPU_Custom(CustomCase, GPU, BatchSamplingTests, unittest.TestCase):
    pass


class AVX2_SineSteady(SineSteadyCase, AVX2, BatchSamplingTests, unittest.TestCase):
    pass


class GPU_SineSteady(SineSteadyCase, GPU, BatchSamplingTests, unittest.TestCase):
    pass


class AVX2_DumpInterval(DumpIntervalCase, AVX2, BatchSamplingTests, unittest.TestCase):
    pass


class GPU_DumpInterval(DumpIntervalCase, GPU, BatchSamplingTests, unittest.TestCase):
    pass


class AVX2_DumpWindow(DumpWindowCase, AVX2, BatchSamplingTests, unittest.TestCase):
    pass


class GPU_DumpWindow(DumpWindowCase, GPU, BatchSamplingTests, unittest.TestCase):
    pass


class AVX2_ExactEndcrit(ExactEndcritCase, AVX2, BatchSamplingTests, unittest.TestCase):
    pass


class GPU_ExactEndcrit(ExactEndcritCase, GPU, BatchSamplingTests, unittest.TestCase):
    pass


class AVX2_NrTSMidBlock(NrTSMidBlockCase, AVX2, BatchSamplingTests, unittest.TestCase):
    pass


class GPU_NrTSMidBlock(NrTSMidBlockCase, GPU, BatchSamplingTests, unittest.TestCase):
    pass


class AVX2_Objects(ObjectsCase, AVX2, BatchSamplingTests, unittest.TestCase):
    pass


class GPU_Objects(ObjectsCase, GPU, BatchSamplingTests, unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
