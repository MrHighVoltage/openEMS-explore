"""Replay an existing openEMS XML setup with a selected engine.

This is intended for engine comparisons after setup generation.  It keeps the
mesh, materials, boundaries, sources, and processing boxes identical between
runs without rebuilding a Python/GDS model.
"""

import argparse
import os
import sys


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--engine", choices=("basic", "multithreaded", "avx2", "avx2-multithreaded", "gpu"), default="gpu")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    os.environ["OPENEMS_GPU_INDEX"] = str(args.gpu_index)

    # Import CSXCAD first, matching the working benchmark runtime.  This
    # also ensures the locally built CSXCAD/OpenEMS loader pair is active.
    from CSXCAD import ContinuousStructure  # noqa: F401
    from openEMS import openEMS

    fdtd = openEMS()
    if not fdtd.ReadFromXML(args.xml):
        raise RuntimeError(f"Unable to read openEMS XML setup: {args.xml}")

    options = {
        "cleanup": True,
        "engine": args.engine,
    }
    if args.threads > 0:
        options["numThreads"] = args.threads

    fdtd.Run(args.output, **options)


if __name__ == "__main__":
    _bootstrap_local_openems_runtime()
    main()
