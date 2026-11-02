"""Controlled openEMS GPU-engine benchmark.

The benchmark intentionally omits field dumps and post-processing. It measures
operator setup and FDTD stepping on a fixed Cartesian grid, with optional PML.
"""

import argparse
import os
import shutil
import sys
import tempfile


def _bootstrap_local_openems_runtime():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    python_dir = os.path.join(repo_root, "python")
    build_dir = os.path.join(repo_root, "build")

    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)

    libopenems = os.path.join(build_dir, "libopenEMS.so")
    if not os.path.exists(libopenems):
        raise RuntimeError(f"Local openEMS library not found: {libopenems}")

    paths = [path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path]
    if paths and os.path.abspath(paths[0]) == os.path.abspath(build_dir):
        return

    paths = [path for path in paths if os.path.abspath(path) != os.path.abspath(build_dir)]
    os.environ["LD_LIBRARY_PATH"] = ":".join([build_dir] + paths)
    os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)


_bootstrap_local_openems_runtime()

from CSXCAD import ContinuousStructure
from CSXCAD.CSProperties import ABCtype
from openEMS import openEMS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", nargs=3, type=int, metavar=("NX", "NY", "NZ"),
                        default=(160, 128, 192))
    parser.add_argument("--timesteps", type=int, default=2000)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--engine", choices=("gpu", "multithreaded"), default="gpu")
    parser.add_argument("--boundary", choices=("pec", "pml"), default="pec")
    parser.add_argument("--excitation", choices=("gaussian", "sinusoidal"), default="gaussian")
    parser.add_argument("--local-abc", choices=("none", "mur", "mur-sa"), default="none")
    parser.add_argument("--material", choices=("none", "conducting-sheet"), default="none")
    parser.add_argument("--source-z", type=int, default=-1,
                        help="place the benchmark excitation on one z plane")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--probe-x", type=int, default=-1)
    parser.add_argument("--probe-y", type=int, default=-1)
    parser.add_argument("--probe-z", type=int, default=-1)
    parser.add_argument("--keep-output", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    nx, ny, nz = args.cells
    if min(nx, ny, nz) < 20:
        raise ValueError("Each grid dimension must contain at least 20 lines")
    if args.timesteps < 1:
        raise ValueError("--timesteps must be positive")

    os.environ["OPENEMS_GPU_INDEX"] = str(args.gpu_index)

    boundary = [0] * 6
    if args.boundary == "pml":
        boundary = ["PML_8"] * 6

    fdtd = openEMS(NrTS=args.timesteps, EndCriteria=1.0e-300)
    if args.excitation == "sinusoidal":
        fdtd.SetSinusExcite(1.0e9)
    else:
        fdtd.SetGaussExcite(1.0e9, 0.5e9)
    fdtd.SetBoundaryCond(boundary)

    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(1.0e-3)
    mesh.SetLines("x", list(range(nx)))
    mesh.SetLines("y", list(range(ny)))
    mesh.SetLines("z", list(range(nz)))

    excitation = csx.AddExcitation("benchmark_excitation", exc_type=0, exc_val=[1, 0, 0])
    source_z = args.source_z
    if source_z >= nz:
        raise ValueError("--source-z must be inside the z grid")
    excitation.AddBox(
        [nx // 2, 9, source_z if source_z >= 0 else 9],
        [nx // 2 + 1, ny - 10, source_z if source_z >= 0 else nz - 10],
        priority=10,
    )

    if args.local_abc != "none":
        abc_type = ABCtype.MUR_1ST if args.local_abc == "mur" else ABCtype.MUR_1ST_SA
        absorber = csx.AddAbsorbingBC(
            "benchmark_local_absorber",
            NormalSignPositive=False,
            AbsorbingBoundaryType=abc_type,
            PhaseVelocity=299792458.0,
        )
        absorber.AddBox([5, 5, nz - 10], [nx - 6, ny - 6, nz - 10], priority=20)

    if args.material == "conducting-sheet":
        sheet = csx.AddConductingSheet(
            "benchmark_conducting_sheet",
            conductivity=5.8e7,
            thickness=1.0e-6,
        )
        sheet.AddBox([5, 5, nz // 2 + 5], [nx - 6, ny - 6, nz // 2 + 5], priority=15)

    if args.probe:
        probe_x = args.probe_x if args.probe_x >= 0 else nx // 2
        probe_y = args.probe_y if args.probe_y >= 0 else ny // 2
        probe_z = args.probe_z if args.probe_z >= 0 else nz // 2
        probe = csx.AddProbe("benchmark_voltage", p_type=0)
        probe.AddBox(
            [probe_x, probe_y, probe_z],
            [probe_x + 1, probe_y, probe_z],
        )


    sim_path = os.path.join(tempfile.gettempdir(), "openems_gpu_benchmark")
    print(
        "BENCHMARK_CONFIG"
        f" cells={nx}x{ny}x{nz}"
        f" timesteps={args.timesteps}"
        f" engine={args.engine}"
        f" gpu_index={args.gpu_index}"
        f" boundary={args.boundary}"
        f" excitation={args.excitation}"
        f" local_abc={args.local_abc}"
        f" material={args.material}"
        f" profile={int(args.profile)}"
        f" probe={int(args.probe)}",
        flush=True,
    )

    run_options = {"cleanup": True, "engine": args.engine}
    if args.engine == "gpu":
        run_options.update(
            gpu_profile=args.profile,
        )
    fdtd.Run(sim_path, **run_options)

    if not args.keep_output:
        shutil.rmtree(sim_path, ignore_errors=True)


if __name__ == "__main__":
    main()
