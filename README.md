# openEMS

**openEMS** is a free, open-source FDTD (Finite-Difference Time-Domain) electromagnetic field solver.
It is the core simulation engine of the [openEMS-Project](https://github.com/thliebig/openEMS-Project)
and is used together with [CSXCAD](https://github.com/thliebig/CSXCAD) for geometry and material description.

- **License:** GPLv3
- **Website:** https://openEMS.de
- **Docs:** https://docs.openems.de
- **Discussions:** https://github.com/thliebig/openEMS-Project/discussions

---

## Features

- 3-D FDTD solver for electromagnetic wave propagation
- Cartesian and cylindrical (including multi-grid) coordinate systems
- SIMD-accelerated engines (SSE2, **AVX2/FMA**, multi-threaded, optional MPI)
- **Vulkan GPU engine** — see [PERFORMANCE.md](PERFORMANCE.md)
- Uniaxial PML and Mur ABC absorbing boundary conditions
- Total-field / scattered-field (TFSF) excitation
- Lumped RLC elements
- Dispersive materials: Drude, Lorentz, Debye (full multi-polar)
- Near-field to far-field (NF2FF) transformation (standalone tool + library)
- SAR calculations (1g / 10g averaging)
- HDF5 and VTK field dump output
- Scripting via **Octave / Matlab** and **Python** (Cython bindings)

---

## Repository Layout

```
openEMS/
├── openems.h / openems.cpp   # Top-level openEMS class (orchestrator)
├── main.cpp                  # openEMS binary entry point
├── FDTD/                     # Operator + Engine (Yee-mesh EC coefficients, time-stepping)
│   └── extensions/           # Physics add-ons (PML, excitation, lumped elements, …)
├── Common/                   # Processing classes (field dumps, ports, NF2FF, SAR)
├── tools/                    # Utility helpers (array types, constants, …)
├── nf2ff/                    # Near-field to far-field tool (libnf2ff + nf2ff binary)
├── matlab/                   # Octave/Matlab scripting interface
└── python/openEMS/           # Cython bindings + pure-Python helpers (ports, automesh, …)
```

---

## Building

openEMS is normally built as part of the **openEMS-Project** meta-repo using the
`update_openEMS.sh` script, which handles all dependencies automatically.
See the [openEMS-Project README](https://github.com/thliebig/openEMS-Project) for the
recommended full-stack build.

### Stand-alone CMake build

Required dependencies: CSXCAD, fparser, TinyXML, HDF5, VTK, Boost (≥ 1.46,
components: thread, date\_time, serialization, chrono, program\_options).

```bash
mkdir build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=~/opt/openEMS \
  -DFPARSER_ROOT_DIR=~/opt/openEMS \
  -DCSXCAD_ROOT_DIR=~/opt/openEMS
make -j$(nproc)
make install
```

Developer-local path overrides go in `localConfig.cmake` (gitignored) at the repo
root — this avoids having to pass `-D` flags every time:

```cmake
# localConfig.cmake
SET(CMAKE_INSTALL_PREFIX  "$ENV{HOME}/opt/openEMS")
SET(FPARSER_ROOT_DIR      "$ENV{HOME}/opt/openEMS")
SET(CSXCAD_ROOT_DIR       "$ENV{HOME}/opt/openEMS")
```

The build produces `libopenEMS.so` and the `openEMS` binary, both installed under
`<prefix>/lib` and `<prefix>/bin`.

### Octave post-install

After `make install`, run the Octave setup once to compile the HDF5 helper:
```bash
octave --no-gui --eval "setup()"
```

### Python bindings

The CSXCAD Python module must be installed first (from the CSXCAD repo):
```bash
python3 -m venv ~/.venv/openEMS
source ~/.venv/openEMS/bin/activate

# 1. Install CSXCAD Python module
export CSXCAD_INSTALL_PATH=~/opt/openEMS
cd /path/to/CSXCAD/python && pip install .

# 2. Install openEMS Python module
export OPENEMS_INSTALL_PATH=~/opt/openEMS
cd /path/to/openEMS/python && pip install .
```

Verify from outside the source tree:
```bash
python3 -c "import CSXCAD; print(CSXCAD.__version__)"
python3 -c "import openEMS; print(openEMS.__version__)"
```

---

## Performance

This fork adds AVX2/FMA CPU engines, a Vulkan GPU engine, and an opt-in
trapezoidal temporal-blocking schedule that runs on both. Measured against a
from-scratch build of upstream on the same machine (i9-13900K, Radeon RX 6800),
on models from 0.26 M to 11.2 M cells:

| | compared against | speedup |
|---|---|---|
| AVX2, single-threaded | upstream SSE, single-threaded | 1.31–2.04&times; |
| AVX2 + temporal blocking (opt-in), 8 threads | upstream's best CPU config | 2.9–4.5&times; *(grids larger than L3)* |
| Vulkan, Radeon RX 6800 | upstream's best CPU config | 6–39&times; |
| Vulkan + temporal blocking (opt-in) | the same GPU, flat sweep | 2.2&times; *(grids larger than the GPU's cache)* |

Cross-checked on a second machine (i7-6700K, integrated HD 530), where the
gains are smaller and one is a regression — an 8 MB L3 is too small for the
blocked CPU schedule to pay on every model. Which results are properties of the
code and which are properties of a 36 MB cache is exactly what the second host
is there to separate.

Full tables, the method, and the cases where this fork is *not* faster are in
**[PERFORMANCE.md](PERFORMANCE.md)**. The engineering record behind the numbers
— including approaches that were tried and rejected — is in
[OPTIMIZATIONS.md](OPTIMIZATIONS.md).

Engine selection is via `--engine=<name>`; this fork adds `avx2`,
`avx2-multithreaded` and `gpu` to the upstream set. The GPU engine requires
`-DWITH_GPU=ON` at configure time (Vulkan SDK with `glslc`).
`--gpu-device=<n>` selects the device when more than one is present, a number
of the list `--verbose=1` prints (as upstream; `OPENEMS_GPU_INDEX` still
works). A cylindrical mesh runs on the multithreaded engine only, and an
explicit `--engine=gpu` on one is an error.

---

## Getting Started

After installation, follow the official tutorials to run your first simulation:

- **Octave/Matlab tutorials:** https://docs.openems.de — also available locally in `matlab/Tutorials/`
- **Python tutorials:** https://docs.openems.de/python/openEMS/Tutorials/ — also available locally in `python/Tutorials/`

---

## Testing

**Octave smoke-test** (fastest sanity check):
```bash
cd .github/smoketests/octave
octave MSL_NotchFilter.m
```

**Python smoke-test:**
```bash
cd .github/smoketests/python
python3 MSL_NotchFilter.py
```

**Full Octave test suite:**
```bash
cd TESTSUITE
octave --no-gui run_testsuite.m
```

**Python unit tests:**
```bash
python3 -m unittest discover -s python/Tests -p "test_*.py" -v
```

---

## Contributing

Pull requests are welcome. Please:

1. Follow the existing code style (C++17, no trailing whitespace — enforced by CI).
2. Add or update tests in `TESTSUITE/` for non-trivial changes.
3. Disclose AI tool usage per [AI_POLICY.md](AI_POLICY.md).
4. Include `Signed-off-by: Your Name <email>` in commit messages (DCO).

---

## License

openEMS is free software licensed under the **GNU General Public License v3**.
See [COPYING](COPYING) for the full text.

Copyright © 2010-2026 Thorsten Liebig and contributors.
