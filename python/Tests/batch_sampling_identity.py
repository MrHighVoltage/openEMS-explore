#!/usr/bin/env python3
"""Check that in-batch sampling gives what the old main loop gave, bit for bit.

    batch_sampling_identity.py [--bin build/openEMS] [--engine avx2-multithreaded]
                               [--threads 1,3] [--schedules flat,4:16]
                               [--loops pipelined,standard] [--rings 0,1,5]
                               [--cases all] [--work DIR] [--keep]

For every case, engine schedule and thread count (with --engine=gpu: main
loop), the same model runs on the same binary with OPENEMS_BATCH_SAMPLING=0
(the old loop: every IterateTS() ends at the next timestep a processing is due
at) and with in-batch sampling (plans/BATCH_SAMPLING.md), once per ring size
(--rings; 0 is the default ring). Both runs print the probe files
with 17 digits (OPENEMS_PROBE_PRECISION) and trace every evaluation of the
energy end criterion and every steady-state check. They must agree in:

- every probe file, time and frequency domain, byte for byte (the header line
  with the date aside);
- every HDF5 dump, dataset and attribute;
- the end-criterion and steady-state traces, line for line;
- the stop line (end criterion reached after <n> timesteps) and the
  "Time for <n> iterations" line.

The new run must say "In-batch sampling ON". A schedule other than flat is
asked for with --avx2-temporal-block (OPENEMS_AVX2_TEMPORAL_BLOCK_FORCE=1 lifts
the size guards on these small grids) or, with --engine=gpu,
OPENEMS_GPU_TEMPORAL_BLOCK (OPENEMS_GPU_TEMPORAL_BLOCK_MB=1 lifts them); the
"Temporal blocking: <b> of <n>" lines of both runs are reported. On the GPU
--loops picks the main loops: "pipelined" (the default one) and "standard"
(OPENEMS_GPU_PIPELINE=0, for the old run and the new alike); --gpu-index sets
OPENEMS_GPU_INDEX. The models write their XML with the CSXCAD/openEMS Python
modules of this interpreter; only the binary under test simulates.

Every openEMS run is wrapped in `flock <--lock>` when --lock is given.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

NX, NY, NZ = 64, 32, 32
EXC_X = 20


def write_model(path, case):
	"""Write the XML of one case into path/model.xml."""
	from CSXCAD import ContinuousStructure
	from openEMS import openEMS

	kw = dict(NrTS=case.get("nrts", 1000), EndCriteria=case.get("endcrit", 1e-5))
	if "oversampling" in case:
		kw["OverSampling"] = case["oversampling"]
	FDTD = openEMS(**kw)
	exc = case["exc"]
	if exc == "gauss":
		FDTD.SetGaussExcite(6e9, 4e9)
	elif exc == "dirac":
		FDTD.SetDiracExcite(10e9)
	elif exc == "step":
		FDTD.SetStepExcite(10e9)
	elif exc == "custom":
		FDTD.SetCustomExcite("sin(2*pi*5e9*t)*exp(-((t-0.6e-9)/0.3e-9)^2)", 5e9, 10e9)
	elif exc == "sine":
		FDTD.SetSinusExcite(5e9)
	else:
		raise ValueError(exc)
	FDTD.SetBoundaryCond(case.get("bc", ["MUR"] * 6))

	CSX = ContinuousStructure()
	FDTD.SetCSX(CSX)
	mesh = CSX.GetGrid()
	mesh.SetDeltaUnit(case.get("delta", 1e-3))
	mesh.SetLines("x", np.arange(NX, dtype=float))
	mesh.SetLines("y", np.arange(NY, dtype=float))
	mesh.SetLines("z", np.arange(NZ, dtype=float))

	if case.get("lossy"):
		lossy = CSX.AddMaterial("lossy", epsilon=2.0, kappa=0.1)
		lossy.AddBox([0, 0, 0], [NX - 1, NY - 1, NZ - 1], priority=1)
	if case.get("dielectric"):
		diel = CSX.AddMaterial("diel", epsilon=3.0)
		diel.AddBox([38, 4, 4], [54, 28, 28], priority=2)

	e = CSX.AddExcitation("exc", exc_type=0, exc_val=[0, 0, 1])
	e.AddBox([EXC_X, 10, 10], [EXC_X + 1, 22, 22])

	fd = [2e9, 5e9, 8e9]
	ut = CSX.AddProbe("ut", p_type=0, frequency=fd)
	ut.AddBox([40, 16, 12], [40, 16, 20])
	it = CSX.AddProbe("it", p_type=1, frequency=fd, norm_dir=0)
	it.AddBox([42, 12, 12], [42, 20, 20])
	ef = CSX.AddProbe("Ef", p_type=2)
	ef.AddBox([45, 16, 16], [45, 16, 16])
	hf = CSX.AddProbe("Hf", p_type=3, frequency=fd)
	hf.AddBox([46.5, 15.5, 16.5], [46.5, 15.5, 16.5])
	mm = CSX.AddProbe("mmE", p_type=10, mode_function=["0", "1", "y/30"])
	mm.AddBox([50, 8, 8], [50, 24, 24])
	mh = CSX.AddProbe("mmH", p_type=11, mode_function=["0", "z/30", "1"])
	mh.AddBox([51, 6, 10], [51, 26, 22])
	late = CSX.AddProbe("ut_late", p_type=0)
	late.AddBox([30, 20, 10], [30, 20, 22])

	if case.get("dumps"):
		td = CSX.AddDump("Et_win", dump_type=0, dump_mode=0, file_type=1)
		td.AddBox([0, 0, 0], [NX - 1, NY - 1, NZ - 1])
		fdd = CSX.AddDump("Hf_fd", dump_type=11, dump_mode=0, file_type=1, frequency=[5e9])
		fdd.AddBox([10, 0, 0], [50, NY - 1, NZ - 1])

	xml = os.path.join(path, "model.xml")
	FDTD.Write2XML(xml)

	# What the Python modules cannot set: the start and stop times of a probe or dump.
	tree = ET.parse(xml)
	for pb in tree.iter("ProbeBox"):
		if pb.get("Name") == "ut_late":
			pb.set("StartTime", repr(case.get("late_start", 0.4e-9)))
			pb.set("StopTime", repr(case.get("late_stop", 0.9e-9)))
	for db in tree.iter("DumpBox"):
		if db.get("Name") == "Et_win":
			db.set("StartTime", repr(case.get("dump_start", 0.3e-9)))
			db.set("StopTime", repr(case.get("dump_stop", 0.5e-9)))
			db.set("SubSampling", "2,2,2")
	tree.write(xml)
	return xml


CASES = {
	# a Gaussian pulse sampled twice as often as by default: the probe interval
	# (Nyquist/8) is below the blocked engines' best depth; default end criterion
	"gauss_os8": dict(exc="gauss", oversampling=8, nrts=4000, endcrit=5e-3),
	# the same with --exact-endcriteria, and a stop mid-run
	"gauss_exact": dict(exc="gauss", oversampling=8, nrts=4000, endcrit=5e-3, args=["--exact-endcriteria"], dielectric=True),
	# runs to NrTS (no end criterion), with a time-domain dump in a window
	# (it ends the batches inside it) and a frequency-domain dump
	"gauss_dumps": dict(exc="gauss", nrts=700, endcrit=0, dumps=True, dielectric=True),
	# Dirac and step: Nyquist interval 1, every timestep a probe timestep
	"dirac": dict(exc="dirac", nrts=500, endcrit=1e-6),
	"step": dict(exc="step", nrts=400, endcrit=1e-6),
	# a custom pulse
	"custom": dict(exc="custom", nrts=1500, endcrit=5e-3, oversampling=6),
	# a sine until the steady state (lossy filling, PEC box): steady-state checks
	"sine_ssd": dict(exc="sine", nrts=20000, endcrit=1e-3, lossy=True, bc=["PEC"] * 6),
	# a sine to NrTS with the dumps
	"sine_dumps": dict(exc="sine", nrts=900, endcrit=1e-9, lossy=True, dumps=True),
}


def header_free(path):
	"""A probe file without its header lines, which hold the date."""
	with open(path, "rb") as f:
		return [l for l in f.read().split(b"\n") if not (l.startswith(b"%") and b"@" in l)]


def h5_items(path):
	out = {}
	with h5py.File(path, "r") as f:
		def visit(name, obj):
			attrs = {k: np.array(v).tobytes() for k, v in obj.attrs.items()}
			if isinstance(obj, h5py.Dataset):
				out[name] = (np.array(obj).tobytes(), attrs)
			else:
				out[name] = (None, attrs)
		f.visititems(visit)
	return out


def outputs(sim):
	res = {}
	for name in sorted(os.listdir(sim)):
		p = os.path.join(sim, name)
		if name in ("model.xml", "log.txt") or os.path.isdir(p):
			continue
		if name.endswith(".h5"):
			res[name] = h5_items(p)
		else:
			res[name] = header_free(p)
	return res


TRACE = re.compile(r"^(endcrit trace:|SteadyState check:)")
STOP = re.compile(r"(after \d+ timesteps|Time for \d+ iterations)")
TBLOCK = re.compile(r"^Temporal blocking: (\d+) of (\d+) timesteps ran blocked")


def run(args, case_name, case, schedule, threads, batch, loop="", ring=0):
	tag = "%s_%s_t%d%s_%s" % (case_name, schedule.replace(":", "x"), threads, ("_" + loop) if loop else "",
	                          ("new_r%d" % ring) if batch else "old")
	sim = os.path.join(args.work, tag)
	shutil.rmtree(sim, ignore_errors=True)
	os.makedirs(sim)
	xml = write_model(sim, case)
	env = dict(os.environ)
	for k in ("OPENEMS_AVX2_TEMPORAL_BLOCK", "OPENEMS_AVX2_TEMPORAL_BLOCK_FORCE", "OPENEMS_GPU_TEMPORAL_BLOCK",
	          "OPENEMS_GPU_TEMPORAL_BLOCK_MB", "OPENEMS_GPU_PIPELINE",
	          "OPENEMS_BATCH_SAMPLING", "OPENEMS_BATCH_SAMPLING_RING"):
		env.pop(k, None)
	if loop == "standard":
		env["OPENEMS_GPU_PIPELINE"] = "0"
	if args.gpu_index is not None:
		env["OPENEMS_GPU_INDEX"] = args.gpu_index
	env["OPENEMS_PROBE_PRECISION"] = "17"
	env["OPENEMS_ENDCRIT_TRACE"] = "1"
	env["OPENEMS_STEADYSTATE_TRACE"] = "1"
	if not batch:
		env["OPENEMS_BATCH_SAMPLING"] = "0"
	elif ring:
		env["OPENEMS_BATCH_SAMPLING_RING"] = str(ring)
	cmd = [args.bin, xml, "--engine=" + args.engine] + case.get("args", [])
	if args.engine != "gpu":
		cmd.append("--numThreads=%d" % threads)
	if schedule != "flat":
		if args.engine == "gpu":
			env["OPENEMS_GPU_TEMPORAL_BLOCK"] = schedule
			env["OPENEMS_GPU_TEMPORAL_BLOCK_MB"] = "1"
		else:
			cmd.append("--avx2-temporal-block=" + schedule)
			env["OPENEMS_AVX2_TEMPORAL_BLOCK_FORCE"] = "1"
	if args.lock:
		cmd = ["flock", args.lock] + cmd
	proc = subprocess.run(cmd, cwd=sim, env=env, capture_output=True, text=True, timeout=600)
	log = proc.stdout + proc.stderr
	with open(os.path.join(sim, "log.txt"), "w") as f:
		f.write(log)
	if proc.returncode != 0:
		raise RuntimeError("%s: exit %d\n%s" % (tag, proc.returncode, log[-3000:]))
	lines = proc.stdout.splitlines()
	r = dict(
		sim=sim, log=log,
		out=outputs(sim),
		trace=[l for l in lines if TRACE.match(l)],
		stop=[m.group(1) for l in lines for m in [STOP.search(l)] if m],
		tblock=[(int(m.group(1)), int(m.group(2))) for l in lines for m in [TBLOCK.match(l)] if m],
		on="In-batch sampling ON" in log,
		failed="cancelling simulation" in log,
	)
	return r


def compare(old, new):
	"""The differences between two runs, as a list of strings."""
	diffs = []
	if not new["on"]:
		diffs.append("in-batch sampling did not engage")
	if old["on"]:
		diffs.append("the old loop ran with in-batch sampling")
	if old["failed"] or new["failed"]:
		diffs.append("a run was cancelled (old %s, new %s)" % (old["failed"], new["failed"]))
	if sorted(old["out"]) != sorted(new["out"]):
		diffs.append("files differ: %s vs %s" % (sorted(old["out"]), sorted(new["out"])))
	for name in sorted(set(old["out"]) & set(new["out"])):
		a, b = old["out"][name], new["out"][name]
		if a == b:
			continue
		if isinstance(a, list):
			n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
			diffs.append("%s differs from line %d (%d vs %d lines): %r vs %r"
			             % (name, n, len(a), len(b), a[n] if n < len(a) else None, b[n] if n < len(b) else None))
		else:
			keys = sorted(set(a) | set(b))
			bad = [k for k in keys if a.get(k) != b.get(k)]
			diffs.append("%s: %d of %d items differ, e.g. %s" % (name, len(bad), len(keys), bad[:3]))
	if old["trace"] != new["trace"]:
		n = next((i for i, (x, y) in enumerate(zip(old["trace"], new["trace"])) if x != y),
		         min(len(old["trace"]), len(new["trace"])))
		diffs.append("traces differ at line %d of %d/%d: %r vs %r"
		             % (n, len(old["trace"]), len(new["trace"]),
		                old["trace"][n] if n < len(old["trace"]) else None,
		                new["trace"][n] if n < len(new["trace"]) else None))
	if old["stop"] != new["stop"]:
		diffs.append("stop lines differ: %s vs %s" % (old["stop"], new["stop"]))
	return diffs


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--bin", default=os.environ.get("OPENEMS_TEST_BIN", os.path.join(REPO_ROOT, "build", "openEMS")))
	ap.add_argument("--engine", default="avx2-multithreaded")
	ap.add_argument("--threads", default="1,3")
	ap.add_argument("--schedules", default="flat,4:16", help="flat and/or <k>:<W> schedules")
	ap.add_argument("--cases", default="all")
	ap.add_argument("--rings", default="0", help="OPENEMS_BATCH_SAMPLING_RING values for the new runs, 0: the default")
	ap.add_argument("--loops", default="pipelined", help="GPU main loops: pipelined and/or standard")
	ap.add_argument("--gpu-index", default=None, help="OPENEMS_GPU_INDEX for every run")
	ap.add_argument("--work", default="/dev/shm/openems_batch_sampling_identity")
	ap.add_argument("--lock", default="", help="flock this file around every run")
	ap.add_argument("--keep", action="store_true", help="keep the simulation directories")
	args = ap.parse_args()
	args.bin = os.path.abspath(args.bin)  # the runs start in their own directories

	names = list(CASES) if args.cases == "all" else args.cases.split(",")
	threads = [int(t) for t in args.threads.split(",")] if args.engine != "gpu" else [0]
	schedules = args.schedules.split(",")
	loops = args.loops.split(",") if args.engine == "gpu" else [""]
	rings = [int(r) for r in args.rings.split(",")]
	os.makedirs(args.work, exist_ok=True)

	failures = 0
	print("%-12s %-6s %3s %-9s %4s  %-6s %-14s %-14s %s" % ("case", "sched", "thr", "loop", "ring", "result",
	      "blocked old", "blocked new", "stop"))
	fmt = lambda r: ("%d/%d" % r["tblock"][0]) if (r and r["tblock"]) else "-"
	for name in names:
		case = CASES[name]
		for schedule in schedules:
			for t in threads:
				for loop in loops:
					try:
						old = run(args, name, case, schedule, t, False, loop)
						olderr = None
					except Exception as exc:  # report and go on
						old, olderr = None, "error: %s" % exc
					kept = False
					for ring in rings:
						new = None
						if olderr:
							diffs = [olderr]
						else:
							try:
								new = run(args, name, case, schedule, t, True, loop, ring)
								diffs = compare(old, new)
							except Exception as exc:
								diffs = ["error: %s" % exc]
						failures += bool(diffs)
						print("%-12s %-6s %3d %-9s %4s  %-6s %-14s %-14s %s" % (name, schedule, t, loop or "-",
						      ring or "def", "FAIL" if diffs else "pass", fmt(old), fmt(new),
						      (new or {}).get("stop", "")), flush=True)
						for d in diffs:
							print("    " + d)
						kept = kept or bool(diffs)
						if not args.keep and not diffs and new:
							shutil.rmtree(new["sim"], ignore_errors=True)
					if not args.keep and not kept and old:
						shutil.rmtree(old["sim"], ignore_errors=True)
	print("%d failure(s)" % failures)
	return 1 if failures else 0


if __name__ == "__main__":
	sys.exit(main())
