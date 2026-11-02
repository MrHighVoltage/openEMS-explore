#!/usr/bin/env python3
"""Check the Vulkan engine's field dumps against a CPU engine's.

    gpu_dump_check.py --bin BUNDLE/fork/openEMS [--gpu 0000:03:00.0]
                      [--nts 1500] [--ref OLD/openEMS] [--only NAME,...]

One model, run one simulation at a time: a graded mesh with a lossy, dense
dielectric block (J, D and SAR), a magnetic block (B), a Mur and a PML face,
an E-field source, a voltage probe, and a voltage and a current probe that
start late.  It dumps every time-domain type
(E, H, J, rotH, D, B) in every interpolation (none, node, cell) on volumes,
planes and sub-boxes, sub-sampled, inside a start/stop window and as VTK;
frequency-domain E, H, J and rotH on planes and boxes; an NF2FF-style box of
six E and six H faces; local and raw SAR.  A second model adds a local
absorbing sheet, for the GPU-native sheet and for the hybrid loop it forces
with OPENEMS_GPU_CPU_LOCAL_ABC=1.

--engine=sse is the reference.  Every GPU variant must produce the same files,
the same datasets (timesteps, frequencies, mesh) and attributes, and values
within --tol (default 1e-4) of the largest magnitude in each dataset's group
(a time series, a set of frequencies).  Every dataset holding at least 1 % of
that maximum must also be within 10*tol of its own maximum, so that a dump of
the wrong timestep (1e-2 and more here) cannot hide behind a large
neighbour.  The probes' time series must be within --tol of their largest
magnitude, and no probe read may miss the GPU's probe cache.  For scale:
--engine=basic differs from sse by about 3e-5.

The variants cover the GPU paths a dump can take: the default, speculative
depths auto and 4, the standard loop (OPENEMS_GPU_PIPELINE=0),
frequency-domain dumps summed on the host (OPENEMS_GPU_DUMP_DFT=0),
whole-field dumps (OPENEMS_GPU_DUMP_GATHER=0), forced temporal blocking with
short submits, a model without probes, and the hybrid loop.

--ref adds the default GPU run of another build, reported but not counted.

Run it on an otherwise idle machine. The hybrid variant has gone unstable
(field energy to infinity) whenever another process was using the GPU, with
builds from before the dump gather as well; on an idle machine it matches.
"""
import argparse
import base64
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path

try:
	import h5py
	import numpy as np
except ImportError:
	sys.exit("gpu_dump_check.py needs h5py and numpy")


# --------------------------------------------------------------------------
# The model

def graded(total, n, lo, hi):
	w = [lo + (hi - lo) * (0.5 - 0.5 * math.cos(2 * math.pi * i / (n - 1))) for i in range(n - 1)]
	s = total / sum(w)
	out, x = [0.0], 0.0
	for d in w:
		x += d * s
		out.append(round(x, 6))
	return out


X = graded(60.0, 40, 1.0, 2.2)
Y = graded(50.0, 34, 1.0, 2.0)
Z = graded(40.0, 30, 1.0, 1.8)


def box(p1, p2, prio=0):
	return ('<Box Priority="%d"><P1 X="%r" Y="%r" Z="%r" /><P2 X="%r" Y="%r" Z="%r" /></Box>'
	        % ((prio,) + tuple(p1) + tuple(p2)))


def dump(i, name, dtype, mode, prims, ftype=1, extra="", fd=None):
	s = '<DumpBox ID="%d" Name="%s" DumpType="%d" DumpMode="%d" FileType="%d"%s>' % (i, name, dtype, mode, ftype, extra)
	if fd:
		s += "<FD_Samples>%s</FD_Samples>" % fd
	return s + "<Primitives>" + "".join(prims) + "</Primitives></DumpBox>"


def model(nts, probe=True, sheet=False):
	full = box((0, 0, 0), (60, 50, 40))
	zplane = box((0, 0, 20), (60, 50, 20))
	xplane = box((31, 0, 0), (31, 50, 40))
	yplane = box((0, 24, 0), (60, 24, 40))
	sub = box((15, 10, 8), (45, 40, 30))
	sar = box((18, 13, 8), (42, 37, 27))
	fd = "2e9,3e9,4.5e9"
	nf = [box((8, 8, 6), (52, 8, 32)), box((8, 42, 6), (52, 42, 32)),
	      box((8, 8, 6), (8, 42, 32)), box((52, 8, 6), (52, 42, 32)),
	      box((8, 8, 6), (52, 42, 6)), box((8, 8, 32), (52, 42, 32))]
	# time domain: type 0..5 = E, H, J, rotH, D, B; mode 0 none, 1 node, 2 cell
	d = [dump(100, "Et_vol_none", 0, 0, [full]),
	     dump(101, "Et_zplane_node", 0, 1, [zplane]),
	     dump(102, "Et_sub_cell", 0, 2, [sub]),
	     dump(103, "Ht_sub_cell", 1, 2, [sub]),
	     dump(104, "Ht_yplane_none", 1, 0, [yplane]),
	     dump(105, "Ht_vol_node", 1, 1, [full]),
	     dump(106, "Jt_sub_cell", 2, 2, [sub]),
	     dump(107, "Jt_xplane_node", 2, 1, [xplane]),
	     dump(108, "rotHt_vol_node_ss", 3, 1, [full], extra=' SubSampling="2,3,2"'),
	     dump(109, "rotHt_sub_cell", 3, 2, [sub]),
	     dump(110, "rotHt_zplane_none", 3, 0, [zplane]),
	     dump(111, "Dt_xplane_none", 4, 0, [xplane]),
	     dump(112, "Dt_sub_node", 4, 1, [sub]),
	     dump(113, "Bt_sub_cell", 5, 2, [sub]),
	     dump(114, "Bt_vol_none_ss", 5, 0, [full], extra=' SubSampling="3,2,4"'),
	     dump(115, "Et_window", 0, 2, [sub], extra=' StartTime="5e-10" StopTime="1.5e-9"'),
	     dump(116, "Et_zplane_vtk", 0, 1, [zplane], ftype=0),
	     # frequency domain: type 10..15
	     dump(120, "Ef_zplane_node", 10, 1, [zplane], fd=fd),
	     dump(121, "Hf_sub_cell", 11, 2, [sub], fd=fd),
	     dump(122, "Jf_sub_none", 12, 0, [sub], fd="3e9"),
	     dump(123, "rotHf_xplane_cell", 13, 2, [xplane], fd=fd),
	     dump(124, "Ef_vol_cell_vtk", 10, 2, [full], ftype=0, fd="3e9"),
	     dump(125, "nf2ff_E", 10, 1, nf, fd=fd),
	     dump(126, "nf2ff_H", 11, 1, nf, fd=fd),
	     # SAR, local and raw
	     dump(130, "SAR_local", 20, 2, [sar], fd="3e9,4e9"),
	     dump(131, "SAR_raw", 29, 2, [sar], fd="3e9")]
	props = [
		'<Material ID="1" Name="lossy" Isotropy="1"><Property Epsilon="4" Mue="1" Kappa="0.05" Sigma="0" Density="1000" />'
		'<Primitives>' + box((20, 15, 10), (40, 35, 25), 10) + '</Primitives></Material>',
		'<Material ID="2" Name="magnetic" Isotropy="1"><Property Epsilon="1" Mue="2" Kappa="0" Sigma="0" />'
		'<Primitives>' + box((42, 10, 25), (52, 20, 32), 10) + '</Primitives></Material>',
		# on mesh lines, or the source would snap to nothing
		'<Excitation ID="3" Name="src" Number="0" Enabled="1" Frequency="0" Delay="0" Type="0" Excite="0,0,1" PropDir="0,0,0">'
		'<Primitives>' + box((X[7], Y[17], Z[10]), (X[7], Y[17], Z[18]), 20) + '</Primitives><Weight X="1" Y="1" Z="1" /></Excitation>']
	if probe:
		props.append('<ProbeBox ID="4" Name="ut1" Number="0" Type="0" Weight="1" NormDir="-1"><Primitives>' +
		             box((30, 25, 12), (30, 25, 22)) + '</Primitives></ProbeBox>')
		# not due at timestep 0, so the recording there does not see them
		props.append('<ProbeBox ID="6" Name="ut_late" Number="0" Type="0" Weight="1" NormDir="-1" StartTime="3e-10">'
		             '<Primitives>' + box((24, 30, 14), (24, 30, 26)) + '</Primitives></ProbeBox>')
		props.append('<ProbeBox ID="7" Name="it_late" Number="0" Type="1" Weight="1" NormDir="2" StartTime="3e-10">'
		             '<Primitives>' + box((X[4], Y[14], Z[14]), (X[10], Y[20], Z[14])) + '</Primitives></ProbeBox>')
	if sheet:
		props.append('<AbsorbingBC ID="5" Name="abs" NormalSignPositive="1" AbsorbingBoundaryType="0" PhaseVelocity="299792458">'
		             '<Primitives>' + box((X[2], Y[30], Z[2]), (X[37], Y[30], Z[27]), 30) + '</Primitives></AbsorbingBC>')
	return ('<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>\n<openEMS>\n'
	        '<FDTD NumberOfTimesteps="%d" endCriteria="1e-300" OverSampling="4"><Excitation Type="0" f0="3e9" fc="3e9" />'
	        '<BoundaryCond xmin="PEC" xmax="MUR" ymin="PEC" ymax="PEC" zmin="PEC" zmax="PML_8" /></FDTD>\n'
	        '<ContinuousStructure CoordSystem="0"><RectilinearGrid DeltaUnit="0.001" CoordSystem="0">'
	        '<XLines>%s</XLines><YLines>%s</YLines><ZLines>%s</ZLines></RectilinearGrid>'
	        '<BackgroundMaterial Epsilon="1" Mue="1" Kappa="0" Sigma="0" /><ParameterSet />'
	        '<Properties>\n%s\n</Properties></ContinuousStructure>\n</openEMS>\n'
	        % (nts, ",".join(map(repr, X)), ",".join(map(repr, Y)), ",".join(map(repr, Z)), "\n".join(props + d)))


# --------------------------------------------------------------------------
# Runs

def run(binary, xml_text, outdir, env_extra, args, gpu):
	if outdir.exists():
		shutil.rmtree(str(outdir))
	outdir.mkdir(parents=True)
	(outdir / "sim.xml").write_text(xml_text)
	env = {k: v for k, v in os.environ.items() if not k.startswith("OPENEMS_GPU_")}
	env.update(env_extra)
	if gpu:
		env["DRI_PRIME"] = "pci-" + re.sub(r"[:.]", "_", gpu)
		env["MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE"] = "1"
	p = subprocess.run([str(binary), "sim.xml"] + args, cwd=str(outdir), env=env,
	                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
	(outdir / "openEMS.log").write_text(p.stdout)
	return p.returncode, p.stdout


# --------------------------------------------------------------------------
# Comparison

def h5_datasets(path):
	out = {}
	with h5py.File(str(path), "r") as f:
		def visit(name, obj):
			if isinstance(obj, h5py.Dataset):
				a = obj[()]
				if a.dtype.names:  # complex as a compound
					a = a[a.dtype.names[0]] + 1j * a[a.dtype.names[1]]
				out[name] = (np.asarray(a), {k: np.asarray(v) for k, v in obj.attrs.items()})
		f.visititems(visit)
	return out


def vtr_datasets(path):
	"""The DataArrays of a zlib-compressed binary VTK XML file."""
	out = {}
	text = path.read_bytes().decode("latin-1")
	for m in re.finditer(r"<DataArray([^>]*)>\s*([^<]*)</DataArray>", text):
		hdr, data = m.group(1), m.group(2).strip()
		name = re.search(r'Name="([^"]*)"', hdr)
		name = name.group(1) if name else ""
		if not name or name.startswith("Array 0x"):  # coordinates, named by address
			name = "coords%d" % len(out)
		typ = re.search(r'type="([^"]*)"', hdr).group(1)
		# a base64 header [blocks, block size, last size, compressed sizes...], then the base64 blocks
		nb = struct.unpack("<I", base64.b64decode(data[:8])[:4])[0]
		hlen = 4 * (3 + nb)
		h64 = ((hlen + 2) // 3) * 4
		head = struct.unpack("<%dI" % (3 + nb), base64.b64decode(data[:h64])[:hlen])
		raw, buf, pos = base64.b64decode(data[h64:]), b"", 0
		for cs in head[3:]:
			buf += zlib.decompress(raw[pos:pos + cs])
			pos += cs
		out[name] = (np.frombuffer(buf, dtype={"Float32": "<f4", "Float64": "<f8"}[typ]), {})
	return out


def compare_file(ref, run, tol):
	"""The worst relative error, and what failed."""
	if set(ref) != set(run):
		return math.inf, "datasets differ: only in the reference %s, only in the run %s" % (
			sorted(set(ref) - set(run))[:3], sorted(set(run) - set(ref))[:3])
	group = {}
	for k, (a, _) in ref.items():
		g = k.rpartition("/")[0]
		group[g] = max(group.get(g, 0.0), float(np.abs(a).max()) if a.size else 0.0)
	worst, own_worst = 0.0, 0.0
	for k, (a, attrs) in ref.items():
		b, battrs = run[k]
		if a.shape != b.shape:
			return math.inf, "%s: shape %s, not %s" % (k, b.shape, a.shape)
		for name, v in attrs.items():
			w = battrs.get(name)
			if w is None or w.shape != v.shape or not (
					np.allclose(v, w, rtol=tol, atol=0) if np.issubdtype(v.dtype, np.floating) else np.array_equal(v, w)):
				return math.inf, "%s: attribute %s differs" % (k, name)
		if not a.size:
			continue
		norm = group[k.rpartition("/")[0]]
		d = float(np.abs(a.astype(np.complex128) - b.astype(np.complex128)).max())
		if norm == 0:
			if d > 0:
				return math.inf, "%s: not zero" % k
			continue
		worst = max(worst, d / norm)
		own = float(np.abs(a).max())
		if own >= 0.01 * norm:
			own_worst = max(own_worst, d / own)
	if own_worst > 10 * tol:
		return max(worst, own_worst / 10), "a dataset differs by %.1e of its own maximum" % own_worst
	return worst, ""


def compare_probe(ref, run, tol):
	"""The worst error of a probe's time series, relative to its largest magnitude."""
	a = np.loadtxt(str(ref), comments="%", ndmin=2)
	b = np.loadtxt(str(run), comments="%", ndmin=2)
	if a.shape != b.shape:
		return math.inf, "%d samples, not %d" % (len(b), len(a))
	if not np.allclose(a[:, 0], b[:, 0], rtol=1e-6, atol=0):
		return math.inf, "sampled at other times"
	norm = float(np.abs(a[:, 1:]).max())
	if norm == 0:
		return math.inf, "the reference is zero"
	return float(np.abs(a[:, 1:] - b[:, 1:]).max()) / norm, ""


PROBES = ("ut1", "ut_late", "it_late")


def compare_runs(ref_dir, run_dir, tol):
	"""(files, worst error, failures)"""
	names = sorted(p.name for p in ref_dir.iterdir() if p.suffix in (".h5", ".vtr") or p.name in PROBES)
	got = sorted(p.name for p in run_dir.iterdir() if p.suffix in (".h5", ".vtr") or p.name in PROBES)
	fails = ["extra files %s" % sorted(set(got) - set(names))[:3]] if set(got) - set(names) else []
	worst = 0.0
	for name in names:
		if not (run_dir / name).exists():
			fails.append("%s missing" % name)
			continue
		if name in PROBES:
			w, why = compare_probe(ref_dir / name, run_dir / name, tol)
		else:
			load = h5_datasets if name.endswith(".h5") else vtr_datasets
			w, why = compare_file(load(ref_dir / name), load(run_dir / name), tol)
		worst = max(worst, w)
		if w > tol:
			fails.append("%s: %.1e %s" % (name, w, why))
	return len(names), worst, fails


def nonzero(d):
	files = [p for p in d.iterdir() if p.suffix == ".h5"]
	return sum(any(np.abs(a).max() > 0 for a, _ in h5_datasets(p).values() if a.size) for p in files), len(files)


# --------------------------------------------------------------------------

VARIANTS = [
	# name, model, environment, options
	("default", "dumps", {}, []),
	("spec auto", "dumps", {"OPENEMS_GPU_SPEC_DEPTH": "auto"}, []),
	("spec 4", "dumps", {"OPENEMS_GPU_SPEC_DEPTH": "4"}, []),
	("standard loop", "dumps", {"OPENEMS_GPU_PIPELINE": "0"}, []),
	("host DFT", "dumps", {"OPENEMS_GPU_DUMP_DFT": "0"}, []),
	("whole fields", "dumps", {"OPENEMS_GPU_DUMP_GATHER": "0"}, []),
	("tblock 3:8 chunk 7", "dumps", {"OPENEMS_GPU_MAX_TS_PER_SUBMIT": "7"}, ["--gpu-temporal-block=3:8"]),
	("no probe", "noprobe", {}, []),
	("sheet", "sheet", {}, []),
	("sheet hybrid", "sheet", {"OPENEMS_GPU_CPU_LOCAL_ABC": "1"}, []),
]


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--bin", type=Path, required=True, help="openEMS binary with the Vulkan engine")
	ap.add_argument("--ref", type=Path, help="default GPU run of another build, for information")
	ap.add_argument("--gpu", help="PCI address of the card to pin, e.g. 0000:03:00.0")
	ap.add_argument("--nts", type=int, default=1500, help="timesteps per run (default: %(default)s)")
	ap.add_argument("--tol", type=float, default=1e-4, help="relative tolerance (default: %(default)s)")
	ap.add_argument("--work", type=Path, default=Path("gpu_dump_check_work"),
	                help="scratch directory (default: %(default)s)")
	ap.add_argument("--keep", action="store_true", help="keep the dumps")
	ap.add_argument("--only", help="comma-separated variant names")
	args = ap.parse_args()
	binary = args.bin.resolve()
	models = {"dumps": model(args.nts), "noprobe": model(args.nts, probe=False),
	          "sheet": model(args.nts, sheet=True)}

	variants = [v for v in VARIANTS if not args.only or v[0] in args.only.split(",")]
	refs = {}
	for m in sorted({v[1] for v in variants} | ({"dumps"} if args.ref else set())):
		d = args.work / ("cpu-" + m)
		rc, log = run(binary, models[m], d, {}, ["--engine=sse"], args.gpu)
		if rc != 0:
			sys.exit("the CPU reference failed (rc=%d), see %s" % (rc, d / "openEMS.log"))
		nz, n = nonzero(d)
		print("  %-20s reference, %d of %d HDF5 files non-zero" % ("cpu " + m, nz, n), flush=True)
		refs[m] = d
	if args.ref:
		variants.append(("ref build", "dumps", {}, []))

	failures = 0
	for name, m, env, opts in variants:
		binary_v = args.ref.resolve() if name == "ref build" else binary
		d = args.work / re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
		rc, log = run(binary_v, models[m], d, env, ["--engine=gpu"] + opts, args.gpu)
		if rc != 0:
			verdict, fails = "FAIL (rc=%d, see %s)" % (rc, d / "openEMS.log"), []
		else:
			n, worst, fails = compare_runs(refs[m], d, args.tol)
			verdict = "%s  %d files, worst %.1e" % ("OK  " if not fails else "FAIL", n, worst)
		if "OPENEMS_GPU_CPU_LOCAL_ABC" in env and "hybrid mode" not in log:
			verdict, fails = "FAIL (the hybrid loop did not run)", []
		# a miss reads the live fields, which the pipelined loop has moved on
		misses = re.search(r"probe cache stats: hits=\d+, misses=(\d+)", log)
		if misses and int(misses.group(1)):
			verdict, fails = "FAIL (%s probe reads missed the probe cache)" % misses.group(1), []
		if name.startswith("tblock") and "temporal blocking active" not in log:
			verdict, fails = "FAIL (temporal blocking did not engage)", []
		if name == "ref build" and verdict.startswith("FAIL"):
			verdict = "differs (not counted): " + verdict
		else:
			failures += verdict.startswith("FAIL")
		print("  %-20s %s" % (name, verdict), flush=True)
		for f in fails[:6]:
			print("      " + f)
		if not args.keep:
			for p in d.iterdir():
				if p.suffix in (".h5", ".vtr"):
					p.unlink()
	print("%d variant(s) failed" % failures if failures else "all variants match the CPU")
	return 1 if failures else 0


if __name__ == "__main__":
	sys.exit(main())
