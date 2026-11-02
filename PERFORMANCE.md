# Performance

This fork adds two engine families to openEMS — **AVX2/FMA CPU engines** and a
**Vulkan GPU engine** — plus a trapezoidal temporal-blocking schedule that runs
on **both** of them. This document reports what they are worth, measured against
from-scratch builds of upstream openEMS on the same machine, with the same
compiler and the same models.

Machines are identified by their hardware, not by a host name or letter. §2 lists
every machine with its full specification and the short label used in tables.

Every number below is the binary's own `Speed: <x> MCells/s` line — the
time-stepping loop alone, with no probes, no field dumps and no post-processing
(§2). The engineering record behind these numbers, including what was tried and
rejected, is in [OPTIMIZATIONS.md](OPTIMIZATIONS.md). This document is the
summary; that one is the evidence.

§11 is the one exception to that method: a real model with probes, run to
completion with every engine on three machines, to check that the engines all
compute the same result.

---

## 1. Where the fork helps, and where it does not

This is the one-page answer. §2 onward is the detail and the method.

### 1.1 Where it helps

- **Vector width, when the run is not yet memory-bound.** Single-threaded, the
  AVX2 engine is **1.31×–2.04×** faster than upstream's SSE engine on the
  i9-13900K, **1.65×–2.59×** on the i7-6700K and **1.42×–2.16×** on the Ryzen 5
  5600G, best on PEC grids and lowest on PML, where the passes vectorize least
  well (§3.1).
- **The discrete GPU, in absolute terms.** A Radeon RX 6800 is worth
  **6.1×–39.0×** over upstream's best CPU configuration, depending on model and
  boundary (§4.1). This is the largest single result in the document.
- **Trapezoidal temporal blocking on the CPU** — the largest *CPU* win, and the
  reason the CPU numbers move at all on large grids. On the i9-13900K it is
  **3.0×–4.1×** over the same engine's flat sweep, on every grid that does not
  fit last-level cache, and **2.0×–3.4×** on the Ryzen 5 5600G. It stays
  worthwhile on a large undivided cache on weaker parts (≈1.4× on the i5-8500),
  and recovers to roughly break-even even on a dual-socket server once its tile
  width is set by hand (§5.2–§5.3).
- **Cheaper synchronisation in that CPU schedule** (2026-10): fewer barriers, a
  per-row work split, an optional spinning barrier, and tiles that leave half
  the cache free where the model's coefficient tables need it. On a 5 M-cell
  transmission line with Mur and lumped ports that is **1.6×–2.0×** over the
  blocked schedule's previous default on three machines, bit-identical to it
  (§5.7). The other figures in this list predate it.
- **The largest CPU-only result comes from a modest part.** The Ryzen 5 5600G
  reaches **1.96×–3.79×** upstream's best on every model with no discrete GPU
  involved. Along with the i9-13900K it is one of only two hosts whose blocked
  schedule wins on every non-cache-resident grid with no regression anywhere —
  six Zen 3 cores behind one undivided 16 MB L3 is the shape the blocking
  heuristic was written for (§5.2).
- **Trapezoidal temporal blocking on the discrete GPU** is worth **2.2×** on the
  grids that exceed the card's Infinity Cache, and is what moves the 224³ PML_8
  case from 17.7× to **39.0×** over upstream (§5.4). It erases the throughput
  cliff at the cache boundary: a grid that does not fit runs within 6% of one
  that does.
  On the 5.03 M-cell transmission-line model of §5.7, with Mur on six faces
  and two lumped ports, it is worth **2.0×** on the RX 6800 and **2.8–3.0×**
  on an RX 7900 XTX, bit-identical to the flat sweep. A full 60 ps run of it on
  the XTX drops from 3.9 h to **1.5 h**, with port files identical to the flat
  run's. It is on by default for discrete GPUs on grids well beyond their cache
  (§5.8). Since 2026-10-20 it is on for integrated GPUs too, with a tile sized
  from the CPU's L3 (§5.9). On the UHD 770, which shares the i9-13900K's 36 MB
  L3, that is worth 1.37× and 29 % less energy per cell. The HD 530 keeps its
  speed and saves 6 %. The Vega 7, which cannot use the L3, runs 1 % slower.
  `--benchmark` measures the schedules on a given model and prints the options
  for the fastest.
- **`--benchmark` on the CPU** (2026-10-20) measures the thread count and the
  kind of core as well as the schedule, on Linux, Windows and macOS without a
  library. On the TL model its pick runs **4.1×** faster than a run with no
  options on the i9-13900K: 8 P-cores, blocked (§5.10).
- **Thread auto-tune on many-core machines.** Upstream's search leaves
  **15–53%** on the table on the 32-thread host; the fork's default lands within
  3–5% of its own pinned optimum there, and 99–100% on the four-core host (§3.3).
- **Host-side paths that are not kernels**: an asynchronous field-dump writer,
  parallel operator construction, GPU energy reduction and probe gather. These
  are wall-clock wins on the models that exercise them, not throughput wins on
  the bare stepping loop (§4.3, §7).

### 1.2 Where it does not help, or actively hurts

- **The AVX2 multithreaded engine is 5–10% slower than upstream's SSE engine on
  PML models once the cores can saturate memory** — the i9-13900K at 8 threads
  and the Ryzen 5 5600G at 4, both without blocking. Both engines are
  bandwidth-bound there and the wider footprint buys nothing. On the four-core
  i7-6700K the comparison never inverts — AVX2 stays 9–13% ahead — so this is a
  memory-system effect, not a kernel defect (§3.4). Practical guidance: on a
  PML-dominated model with no blocking, prefer `--engine=multithreaded`.
- **CPU temporal blocking needs a large, undivided last-level cache.** On the
  i7-6700K's 8 MB L3 the gains collapse to **1.05×–1.42×** and the Mur model
  **regresses 13%** (0.87×), because the tile shrinks to what the cache allows
  (§5.3). It is opt-in, so this costs nothing unless asked for. These figures
  predate §5.7, which measured the i7-6700K's blocked schedule 1.6× faster
  than before on a transmission-line model; the matrix has not been re-run.
- **The CPU blocking tile width is derived from one cache instance**, not from
  the cache the running threads collectively have. On a partitioned LLC this
  makes the shipped schedule a **30–50% regression** on a dual-socket server
  (0.50× and 0.71× on the 224³ PEC model); set correctly it is roughly
  break-even. This is a defect, not a mis-tuning, and it is worth up to **2×**
  where it bites (§5.3).
- **GPU temporal blocking is switched on by a constant, not by the device.**
  On the RX 6800 a grid between the default 64 MB target and the card's real
  128 MB cache is already resident but still looks too large, so the schedule
  engages and costs ~1% in the matrix and ~5% in a direct single-run A/B (§5.5).
  One environment variable recovers it per host.
- **GPU temporal blocking does nothing on either kind of integrated GPU.**
  On the HD 530 it is flat at 1.00×–1.04× because that part is limited by its
  24 execution units, not by DRAM traffic. On the Vega 7 it is flat for the
  opposite reason — that part *is* bandwidth-bound, but it has no cache worth
  blocking for, and the tile geometry clamps the depth to k=6 (§5.5).
- **Integrated GPUs are not general-purpose wins.** The UHD 770 never beats the
  CPU on the i9-13900K (0.31×–1.33× upstream's best). The HD 530 is the fastest
  engine on the i7-6700K on exactly two models — 224³ PML_8 and 224³ Mur — and
  only because that CPU is weak and those models are large and PML-dominated.
  The Vega 7 repeats that pattern exactly: **0.59×–2.85×** upstream's best, but
  it beats its own machine's blocked CPU engine only on the two PML models, and
  loses to it by up to 3.3× everywhere else (§4.2).
- **A pre-AVX2 host does not run slowly — it crashes.** AVX2 selection is
  compile-time only, so an x86-64 binary built on a modern GCC selects the AVX2
  engine on a Sandy Bridge host and dies with `SIGILL`, including through
  `--engine=fastest` (the default). Every fork AVX2 row on the 2× Xeon E5-2630
  host is absent for this reason. The capability is not missing, only the
  dispatch: `--engine=multithreaded` runs fine there and is a small win (§6).
  A runtime CPU check now falls back to that engine with a warning instead;
  the matrix below predates it.
- **Thread auto-tune on dual-socket hosts is unreliable in both directions.**
  It scores thread counts without any notion of socket placement, so it can land
  below the pinned optimum (the 2× Xeon Gold 6130's default reaches 992 MC/s
  against 1282 pinned on 224³ PEC) or above a badly-chosen pinned count (§3.3).

### 1.3 The spread in one table

Best fork configuration against best upstream configuration, both defined as in
§2.3, across all eight machines. `SIGILL` marks the AVX2 crash; the fork's
portable engine is measured separately for that host in §6. The bold rows are
where the discrete GPU is in play; the two daggered cells are the only ones in
the table where an *integrated* GPU is the fork's best configuration.

| Model | i9-13900K<br>+ RX 6800 | i7-6700K<br>+ HD 530 | Ryzen 5<br>5600G<br>+ Vega 7 | Ryzen 9<br>3950X | 2× Xeon<br>Gold 6130 | 2× Xeon<br>E5-2630 | Core<br>i5-8500 | 2× Xeon<br>E5-2620 v3 |
|---|---|---|---|---|---|---|---|---|
| 64³ PEC | **6.10×** | 2.45× | 1.96× | 2.52× | 2.35× | AVX2 crash | 2.16× | 1.89× |
| 160×128×192 PEC | **16.57×** | 1.93× | 3.79× | 2.68× | 1.35× | AVX2 crash | 2.53× | 1.36× |
| 224³ PEC | **21.01×** | 1.37× | 2.48× | 1.57× | 1.18× | AVX2 crash | 1.59× | 1.15× |
| 160×128×192 PML_8 | **35.15×** | 1.72× | 2.85×† | 2.16× | 1.32× | AVX2 crash | 1.85× | 1.16× |
| 224³ PML_8 | **38.98×** | 1.61× | 2.46×† | 1.68× | 1.09× | AVX2 crash | 1.60× | 1.08× |
| 224³ Mur | **9.66×** | 1.40× | 1.99× | 1.19× | 1.37× | AVX2 crash | 1.46× | 1.10× |

Strip out the discrete GPU and the picture is the CPU engines alone: on the
i9-13900K the best CPU configuration is **1.70×–4.45×** upstream's best, and on
the i7-6700K it is **1.13×–2.43×**. The GPU is what turns the i9 numbers into
double digits.

Four things to read out of the table, in descending order of how much they
should change what anyone does:

- **The GPU is not where the fork's CPU story lives.** A user with no Vulkan
  device still gets the AVX2 width and, on a big undivided cache, temporal
  blocking. The Ryzen 5 5600G is the clearest case: **1.96×–3.79×** with the
  discrete-GPU column empty, and its own integrated GPU beaten by its own CPU
  on four of six models.
- **The two hosts that gain least on the CPU, the dual-socket Xeons, are the two
  whose last-level cache is split across sockets** — and the loss is not
  inherent to them. Told the cache its threads actually have, the 2× Xeon Gold
  6130's blocked schedule goes from 636 to 1280 MC/s and the 2× E5-2620 v3's
  from 356 to 569 on the 224³ PEC model. The table reports the defaults, so it
  reports the defect (§5.3).
- **What predicts the CPU gain is cache topology, not machine class.** The two
  single-socket, six-core parts with one undivided L3 — the Ryzen 5 5600G
  (16 MB) and the Core i5-8500 (9 MB) — gain **1.96×–3.79×** and 1.46×–2.53×,
  more consistently than machines several times their price. Cache *size* then
  orders them: at 16 MB the 5600G's tile is 13 x-lines against the i5-8500's 7,
  and its blocked schedule is worth 2.5× on 224³ PEC against 1.35× (§5.2).
- **The two PML cells marked †** are the only place in this document where an
  integrated GPU is the best thing on its machine *and* the machine has a
  competent CPU. It is a narrow win: on the same host the blocked CPU engine is
  ahead on the other four models (§4.2).

### 1.4 And it computes the same answer

A speed-up is only worth reporting if the results do not move. On a
production-size model — a 2.27 M-cell on-chip transmission line, run for
2.36 M timesteps with every engine on three of the machines (§11) — the fork's
`sse`, `sse-compressed` and `multithreaded` engines are **bit-identical to
upstream's**, on every machine and at every thread count. The fork's AVX2
engines and its GPU engine form two more bit-identical groups. They differ from
upstream by **float32 rounding and nothing else**: at most 1.5e-6 of a probe's
peak, and ≤ 3e-5 dB and ≤ 5e-4° in S11 and S21. That is about 50 dB below the
change from one step of mesh refinement. CPU temporal blocking is bit-identical
to the flat sweep it replaces.

---

## 2. Machines, builds and method

### 2.1 The machines

All eight machines and their short table labels. `LLC` is the last-level cache;
the `getconf LEVEL3_CACHE_SIZE` column is the value the blocking heuristic
reads, and §5.3 is about the difference between the two on a partitioned part.

| Label | Machine | Cores | LLC | `getconf` LLC | GPUs | OS |
|---|---|---|---|---|---|---|
| **i9-13900K** | Intel Core i9-13900K | 8 P + 16 E / 32T, 1 socket | 36 MB, one cache | 36 MB | Radeon RX 6800 (RADV, 128 MB Infinity Cache, 256 MB BAR, no ReBAR) + integrated UHD 770 (ANV) | Mesa 26.2.1, Vulkan 1.4 |
| **i7-6700K** | Intel Core i7-6700K | 4C / 8T, 1 socket | 8 MB, one cache | 8 MB | integrated Intel HD Graphics 530 (Skylake GT2, ANV) | Mesa 26.2.1, Vulkan 1.4 |
| **Ryzen 9 3950X** | AMD Ryzen 9 3950X | 16C / 32T, 1 socket | 4 × 16 MB (one per CCX), **64 MB total** | 16 MB | none | Rocky 8.10, glibc 2.28 |
| **2× Xeon Gold 6130** | 2× Intel Xeon Gold 6130 | 32C / 64T, 2 sockets | 2 × 22 MB, non-inclusive, **+ 1 MB L2 per core** | 22 MB | none | Rocky 8.10, glibc 2.28 |
| **2× Xeon E5-2630** | 2× Intel Xeon E5-2630 | 12C / 24T, 2 sockets | 2 × 15 MB | 15 MB | none | Rocky 8.10, glibc 2.28 |
| **Core i5-8500** | Intel Core i5-8500 | 6C / 6T, 1 socket | 9 MB, undivided | 9 MB | none | CentOS 7.9, glibc 2.17, kernel 3.10 |
| **Ryzen 5 5600G** | AMD Ryzen 5 5600G (Cezanne) | 6C / 12T, 1 socket | 16 MB, one CCX, undivided | 16 MB | integrated Radeon Vega 7 (RADV RENOIR), no discrete card | Arch, glibc 2.44, Mesa/RADV, Vulkan 1.4 |
| **2× Xeon E5-2620 v3** | 2× Intel Xeon E5-2620 v3 | 12C / 24T, 2 sockets | 2 × 15 MB | 15 MB | none | Rocky 8.10, glibc 2.28 |

The i9-13900K has 32 GB dual-channel DDR5 and a `powersave` governor with turbo
active; its practical STREAM-Triad ceiling is **~49 GB/s** at 6–8 threads. The
i7-6700K has 64 GB dual-channel DDR4 and a `powersave` governor. The five
servers have no Vulkan ICD and no openEMS dependencies; none is used for any
tuning decision, only to test whether conclusions survive a change of cache and
socket topology.

The Ryzen 5 5600G is a sixth remote machine, added later and treated the same
way — nothing was installed on it either — but it differs from the five in the
one respect that earns it GPU columns: it has a working Vulkan driver. It is
therefore the only host here where a competent single-socket CPU and an
integrated GPU are measured on the *same* memory system, which is what makes
its §4.2 rows worth more than a portability check. It has 16 GB dual-channel
DDR4 and an `amd-pstate-epp` `powersave` governor with boost enabled; under
load it holds a steady 4.20 GHz all-core, varying 0.2% across a 90-second
sample, so `powersave` is not a clock cap on this driver (§10).

The i7-6700K is a **shared** machine and was not quiesced for its runs:
`rclone`, Jellyfin and an idle BOINC client were resident throughout, at a load
average of 0.6–2.8. See §10 for what that does and does not affect. It is the
machine [OPTIMIZATIONS.md](OPTIMIZATIONS.md) once described as "gone" — it is
not; it is a remote server.

The 2× Xeon E5-2630 is **pre-Haswell** — AVX and SSE4.2 but no AVX2 and no FMA.
It is included precisely for that: it is the only machine here that can answer
what a binary carrying an AVX2 engine does on hardware that cannot execute one
(§6).

### 2.2 Builds

For the i9-13900K and i7-6700K, all four builds were configured from scratch and
compiled with the same toolchain (GCC 16.2.1, CMake Release, `-O3 -DNDEBUG`)
against each machine's own CSXCAD, HDF5, VTK and Boost:

| | commit | engines available |
|---|---|---|
| upstream | `d3d2a49` (2026-08-15) | `basic`, `sse`, `sse-compressed`, `multithreaded` |
| this fork | `6c795de` (2026-09-07) | the above **+** `avx2`, `avx2-multithreaded`, `gpu` |

The fork's merge-base with upstream *is* upstream's head, so this is exactly
upstream plus 99 commits, with no divergence to account for.

The five servers **could not be built on**, and that constraint shaped the
method. They are shared machines that must not be modified, none has CSXCAD,
HDF5, VTK, CGAL or Boost installed, and one is a CentOS 7 box — glibc 2.17,
kernel 3.10 — which no binary from a current distribution will even start on. So
both sides were built *once*, on the workstation, inside a glibc-2.17 container
(`quay.io/pypa/manylinux2014_x86_64`, GCC 10.2.1) against a dependency stack
compiled from source in the same container, and linked statically against
everything except libc. The resulting bundle needs nothing on the target but a
glibc ≥ 2.17 x86-64 system: its highest versioned symbol requirement is exactly
`GLIBC_2.17`, and its only dynamic dependencies are `libc`, `libm`, `libdl`,
`libpthread` and the three project libraries shipped beside it.
`python/Tests/build_portable_bench.sh` is that build, end to end.

Nothing was installed on any of the five hosts. The bundle was copied to the NFS
home they share and staged to node-local `/tmp` for the runs, so no NFS round
trip is anywhere near a measurement.

The Ryzen 5 5600G runs from the same bundle, and from a second one built for
its GPU. `WITH_GPU=ON` produces a *different* fork binary — it links the Vulkan
loader and embeds the SPIR-V shaders — so it gets its own builder,
`python/Tests/build_portable_bench_gpu.sh`, and its own bundle rather than a
flag on the CPU one; the script that produced every CPU number in this document
is unchanged. The GPU bundle reuses the CPU bundle's dependency stack and its
*upstream* binary byte for byte, since upstream has no GPU engine and the
comparison would otherwise vary two things at once. Its fork binary is built
with the same flags plus `-DWITH_GPU`, which makes its CPU rows a free control:
they reproduce the CPU bundle's to **2.99%** across all 48, against a 2.88%
run-to-run spread between two runs of the same binary (§10). Unlike the CPU
bundle it is not fully self-contained — it deliberately carries no
`libvulkan.so.1`, because the loader must match the target's driver.

**The older compiler is not a variable.** GCC 10 against the GCC 16 used for the
i9-13900K and i7-6700K is a real difference, and since it applies to both sides
of every comparison it cannot manufacture a fork-vs-upstream result — but it
could still distort the *absolute* numbers this document quotes. It does not.
The two builds were compared on the i9-13900K in a paired test, the arms
alternating run by run on the same model so that machine state cancels:

| i9-13900K, 224³ PEC | GCC 16 | GCC 10 | |
|---|---|---|---|
| `sse` 1 thread | 235 | 234 | 1.00× |
| `avx2` 1 thread | 439 | 438 | 1.00× |
| `avx2-multithreaded` 8t | 612 | 619 | 1.01× |
| `avx2-multithreaded` +tblock 8t | 2639 | 2703 | 1.02× |

Across all sixteen model/engine combinations tested the ratio stays within
**0.98–1.04**, with no systematic direction. Full output in
`python/Tests/results/toolchain_gcc16_vs_gcc10_2026-09-11.txt`. Note the form of
that control: two *matrix runs* on different days would have conflated the
toolchain with the machine's state, and an attempt at exactly that comparison
came out 15–29% low on the PEC rows purely because the workstation was busy that
afternoon. Alternating arms is what makes it a measurement.

### 2.3 What was measured

Both builds' `openEMS` **binaries** are driven over byte-identical XML models,
written once with `Write2XML()` and then copied to the other hosts, so every
machine runs the same bytes. The models are committed as
`python/Tests/models/*.xml` for that reason — on the five servers nothing could
regenerate them, since none has the Python bindings. Nothing depends on which
Python bindings happen to be installed, and the models contain nothing a stock
upstream build cannot parse — a uniform Cartesian grid, one excitation box, a
boundary condition. There are **no probes, no field dumps and no
post-processing**, so what is reported is the time-stepping loop alone: the
binary's own final `Speed: <x> MCells/s` line. Operator build time is reported
separately in §7.

Each binary was checked to load its own `libopenEMS.so.0` rather than the
installed one — separately *configured* build directories, because RPATH is
absolute and a copied build tree is not an independent binary.

The harness is in the repo: `python/Tests/benchmark_fork_vs_upstream.py`.
Everything host-specific in it is a command-line option, which is what let the
same matrix run unchanged on all eight machines. See §9 to reproduce.

**Upstream's "best"** is the fastest of the three upstream configurations
measured, which is almost always its SSE multithreaded engine at a hand-picked
thread count — not its slower out-of-the-box default. The fork's "best" is
likewise the fastest fork configuration for that machine and model.

### 2.4 Why some rows are pinned, and at what thread count

Rows that compare engines to each other pin both builds to one hardware thread
per physical core and pass the same `--numThreads` on both sides. This is **not**
noise reduction — the one genuinely noisy configuration stays noisy when pinned
(§10) — it is to make the comparison controlled, so that both builds run the
same number of threads on the same cores and a column difference is the engine
and nothing else.

The pinned count is where **both builds peak**, from a separate calibration
sweep on the 224³ PEC model (600–1200 timesteps, unpinned):

| Machine | pinning | upstream SSE-MT | fork AVX2-MT | fork +tblock |
|---|---|---|---|---|
| i9-13900K | `-c 0,2,4,6,8,10,12,14`, 8 threads | 543 / 570 / **572** / 499 at 4/6/8/12 | 600 / 621 / **622** / 552 | 1765 / **2305** / 2109 at 4/8/12 |
| i7-6700K | `-c 0,2,4,6`, 4 threads | 384 / **441** / 394 / 377 at 2/4/6/8 | **487** / 480 / 431 / 391 | 484 / 554 / **592** / 575 |
| Ryzen 9 3950X | `-c 0-15`, 16 threads | 423 / 410 / **424** / 422 / 372 at 4/8/12/16/32 | 436 / 424 / **441** / 439 / 392 | **1132** / 584 / 570 / 520 / 447 |
| 2× Xeon Gold 6130 | `-c 0-31`, 32 threads | 559 / 856 / 856 / 929 / **1006** / 614 at 4/8/16 s0/16 spread/32/64 | 707 / 952 / 935 / 1048 / **1144** / 912 | 769 / **989** / 986 / 742 / 620 / 425 |
| 2× Xeon E5-2630 | `-c 0-11`, 12 threads | 332 / 351 / 439 / **538** / 421 at 4/6 s0/6 spread/12/24 | *SIGILL* | *SIGILL* |
| Core i5-8500 | `-c 0-1`, 2 threads | **207** / 201 / 180 at 2/4/6 | **225** / 211 / 190 | 304 / 316 / **337** |
| Ryzen 5 5600G | `-c 0-3`, 4 threads | 271 / 406 / **424** / 373 / 351 / 307 at 1/2/4/6/8/12 | 416 / **474** / 427 / 388 / 367 / 327 | 532 / 798 / 1056 / **1151** / 1080 / 1085 |
| 2× Xeon E5-2620 v3 | `-c 0-11`, 12 threads | 206 / 251 / 303 / **433** / 370 at 4/6 s0/6 spread/12/24 | 285 / 291 / 455 / **471** / 433 | 308 / **361** / 324 / 358 / 284 |

The i9-13900K's two engines agree on 8, and both roll off past it for the same
reason: a plain OpenMP STREAM-Triad on that host peaks at 6–8 threads and loses
~20% by 16 ([OPTIMIZATIONS.md](OPTIMIZATIONS.md) §4.3). The memory controller
sets that curve, not the engines.

The dual-socket rows carry an extra column the single-socket machines do not: on
the 2× E5-2630 and the 2× E5-2620 v3, six threads *spread across both sockets*
(`-c 0-2,6-8`) beat six threads on one (`-c 0-5`) by **25%** and **21%** — two
memory controllers rather than one, for the same core count. That is why the
pinned sets for those machines span both sockets.

**Three of the five servers' blocked schedules peak somewhere other than their
flat ones, and by a lot**: the Ryzen 9 3950X at 4 threads against 16, the 2×
Xeon Gold 6130 at 8 against 32, the Core i5-8500 at 6 against 2. On the
i7-6700K the same disagreement was read as SMT helping a less bandwidth-bound
kernel. That reading does not survive these machines: the blocked schedule
prefers *fewer* threads here, and §5.3 shows why — at low thread counts the run
fits inside one cache instance, which is the only case in which the derived tile
width is the right one. The pinned `+tblock` columns in §8 therefore understate
the schedule on the Ryzen 9 3950X, the 2× Xeon Gold 6130 and the 2× Xeon E5-2620
v3, and §5.3 is where the corrected figures are.

**The Ryzen 5 5600G's engines do not agree either**, and it is the same shape
of disagreement: upstream's SSE engine peaks at **4** threads, the flat AVX2
engine at **2**, and the blocked schedule at **6** — one thread per physical
core. Four was chosen for the same reason as on the i7-6700K, because it is
where the two *flat* engines are closest to their peaks, and it is the
conservative choice for the fork: the flat AVX2 engine gives up 10% at 4
threads against its own 2-thread peak, and the blocked schedule 8% against its
6-thread one. Both `default` columns in §8.4 are unpinned and land at or above
the pinned ones, so the summary in §1.3 — which quotes the best fork
configuration — is unaffected. Note also that this machine's blocked schedule
never collapses past its peak the way the partitioned-cache hosts' do: it holds
above 1080 MC/s from 4 threads all the way to full SMT, which is what a single
undivided cache looks like in this sweep.

**The i7-6700K's engines do not agree**, and that is itself a result. Its flat
AVX2 engine peaks at **2** threads, upstream's SSE engine at **4**, and the
temporally blocked schedule at **6** — past the physical core count, into SMT.
Blocking makes the engine less bandwidth-bound, and a less bandwidth-bound
kernel is exactly the kind that SMT siblings help. Pinning to 4 was chosen
because it is where the two *flat* engines are closest to their peaks, so the
engine-vs-engine columns stay controlled; the consequence is that the i7-6700K's
`tblock 4t` column is **16% below** that schedule's own optimum, and its
`tblock default` column (unpinned, auto-tuned, free to use all 8 logical CPUs)
is the more representative one. The summary table in §1.3 therefore quotes the
best fork configuration, and both are in §8.

Rows labelled **default** run exactly as a user gets them: no pinning, no
`--numThreads`, each build using its own thread auto-tune. Both families are
reported; §3.3 is specifically about the difference between them. GPU rows are
neither pinned nor thread-limited: the CPU is not the resource under test there,
so constraining it would only add a variable.

Every figure is the best of 3 repeats, or 6 on the 64³ grid, for the reason in
§10.

---

## 3. The AVX2 CPU engines

`Engine_AVX2` and `Engine_AVX2_Multithread` replace the 4-wide SSE Yee kernels
with 8-wide AVX2+FMA ones, keeping the compressed-operator storage of the SSE
engines.

### 3.1 Single-threaded: the width shows directly

Where the engine is not yet fighting the memory system, doubling the vector
width is visible, and it is visible on every machine that can run the engine:

| Model | i9-13900K<br>SSE → AVX2 | | i7-6700K<br>SSE → AVX2 | | Ryzen 5 5600G<br>SSE → AVX2 | |
|---|---|---|---|---|---|---|
| 64³ PEC | 476 → 972 | 2.04× | 198 → 513 | 2.59× | 290 → 628 | 2.16× |
| 160×128×192 PEC | 267 → 494 | 1.85× | 199 → 374 | 1.88× | 275 → 415 | 1.51× |
| 224³ PEC | 247 → 449 | 1.82× | 200 → 376 | 1.88× | 271 → 419 | 1.55× |
| 160×128×192 PML_8 | 111 → 148 | 1.34× | 70 → 124 | 1.77× | 66 → 105 | 1.58× |
| 224³ PML_8 | 112 → 147 | 1.31× | 83 → 137 | 1.65× | 79 → 111 | 1.42× |
| 224³ Mur | 219 → 364 | 1.66× | 173 → 286 | 1.65× | 238 → 339 | 1.42× |

On top of the raw width, three changes mattered:

- **Redundant field loads eliminated** in the update kernels (`e3932e8`) —
  +7–8% single-threaded, flat at 8 threads, which was the first clear sign the
  multithreaded engine was already bandwidth-bound.
- **The missing AVX2 fast field-extraction path** (`586a8ac`). Field extraction
  had an SSE fast path and no AVX2 one, so every dump fell back to the generic
  scalar path — an AVX2-specific tax SSE never paid. Dumping the full domain
  every timestep went 336 → 426 MC/s.
- **Vectorized UPML passes** (`08fa968`).

The 1.51–2.59× on PEC is close to what doubling the vector width can give. On
the i9-13900K the PML rows drop to 1.31–1.34× because the PML passes are a
larger share of the work and vectorize less well — see §3.4, where this becomes
a problem at high thread counts. On the i7-6700K the same rows hold
1.65–1.77×, which is the clearest single sign that §3.4 is a memory-system
effect and not a property of the kernel: the slower machine, with proportionally
more compute per byte of bandwidth, keeps the vector win the faster machine
loses. The Ryzen 5 5600G sits between them at 1.42–1.58× on PML — and note that
its *single-threaded* ratios are the lowest of the three on the large grids
(1.51–1.55× on PEC) even though it goes on to post the best multithreaded
numbers in this document. One core cannot use that machine's memory system;
what §5.2 shows is what happens when the schedule stops needing it.

### 3.2 Multithreaded, flat

Against upstream's best (usually its SSE multithreaded engine at a hand-picked
thread count), the flat AVX2 engine's margin depends on how bandwidth-bound the
run is. It is comfortably ahead on PEC and on the smaller machines, and it loses
on PML on the many-core i9-13900K. The full matrix is in §8; §3.4 is the one
place it inverts.

### 3.3 Thread auto-tune: fixed here, still worth knowing

Upstream's thread search hill-climbs inside `NextInterval()`, which is called
only after four seconds of wall time, adding one thread per call starting from
one. On a 4-core machine it converges quickly; on a 32-thread machine reaching
the optimum takes ~44 s, during which the run is nowhere near peak
([OPTIMIZATIONS.md](OPTIMIZATIONS.md) §4.2). `fd555ee` replaces it with a search
that scores candidates on real timesteps inside `IterateTS()` — legitimate
because thread count cannot change results — settling in ~250 timesteps instead.
One detail mattered more than expected: the first batch after each thread
respawn must be discarded, or cold private caches penalise exactly the wide
configurations under judgement.

How close each build's default gets to its own pinned optimum:

| Model | i9-13900K<br>upstream | i9-13900K<br>fork | i7-6700K<br>upstream | i7-6700K<br>fork |
|---|---|---|---|---|
| 64³ PEC | 47% | 83% | 100% | 100% |
| 160×128×192 PEC | 72% | 97% | 88% | 99% |
| 224³ PEC | 85% | 97% | 91% | 100% |
| 160×128×192 PML_8 | 74% | 96% | 97% | 100% |
| 224³ PML_8 | 69% | 95% | 79% | 100% |
| 224³ Mur | 81% | 95% | 90% | 100% |

The two machines together are the whole argument. On the i9-13900K upstream's
default leaves 15–53% on the table and the fork's lands within 3–5%. On the
i7-6700K — 4 cores, exactly the shape of machine upstream's search was written
for — upstream loses at most 21% and the fork reaches 99–100%. The fix is worth
most where the old search was worst, which is the many-core case, and costs
nothing where it was already fine.

Note what this means for reading §8: **comparing the two `default` columns
measures the auto-tune as much as the engines.** That is why the pinned columns
exist.

On the Ryzen 5 5600G the fork's search behaves as designed: its default lands
within 0–3% of its own pinned optimum on every model, and on four of six it is
*above* the pinned column, because the pinned count of 4 is conservative for
this host (§2.4). Upstream's default gives up 7–34% there, worst on the
cache-resident 64³ grid (66%) and on 224³ PML_8 (70%).

On the five servers the auto-tune is no longer uniformly the safer choice, and
on the two biggest machines it is the *worse* one: the 2× Xeon Gold 6130's
default reaches 992 MC/s on 224³ PEC against 1282 pinned, and the 2× Xeon E5-2620
v3's 422 against 514. Both are dual-socket, and the search scores thread counts
without any notion of socket placement, so it settles on a count that a NUMA-aware
placement would beat. In the other direction the 64³ row on the 2× Xeon Gold 6130
has the default at 1002 against 370 pinned, because there the pinned count is the
bad choice. The honest summary is that on a dual-socket host neither column is
reliably the one a user should expect; on the single-socket hosts the default
lands within 0–5% of pinned on the large models, though see §10 for the spread
on the cache-resident 64³ grid.

### 3.4 The inversion: AVX2 multithreaded loses to SSE on PML, on two hosts

On a PML model at the pinned thread count, the AVX2 engine is **5–10% slower**
than the SSE engine it was meant to replace — despite being faster
single-threaded on the same model.

| Model | | upstream SSE-MT | fork AVX2-MT | fork AVX2-MT<br>+ temporal blocking |
|---|---|---|---|---|
| 160×128×192 PML_8 | i9-13900K, 8t | 280 | 266 | 800 |
| 224³ PML_8 | i9-13900K, 8t | 262 | 237 | 828 |
| 224³ PEC | i9-13900K, 8t | 586 | 632 | 2607 |
| 160×128×192 PML_8 | Ryzen 5 5600G, 4t | 145 | 135 | 340 |
| 224³ PML_8 | Ryzen 5 5600G, 4t | 164 | 152 | 354 |
| 224³ PEC | Ryzen 5 5600G, 4t | 429 | 432 | 1062 |

Both engines are bandwidth-bound there, and the AVX2 kernel's wider per-lane
footprint buys nothing once DRAM sets the pace, while its PML passes cost
slightly more. The Ryzen 5 5600G reproduces the i9-13900K's inversion almost
exactly — **7% down** on both PML models, level on PEC — which is the useful
part: it is not a many-core effect. Four Zen 3 cores on dual-channel DDR4
saturate their memory system as thoroughly as eight P-cores saturate DDR5, and
the inversion follows the saturation, not the core count.

That this is a memory-system effect and not a kernel defect is confirmed from
the other side by the **i7-6700K, where the same comparison never inverts** —
164 vs 146 and 183 vs 168 MC/s, the AVX2 engine ahead by 9–13% on exactly the
models where it falls behind on the other two hosts. Four Skylake cores cannot
saturate DDR4 the way four Zen 3 cores can, so the crossover never happens.

The right fix is not more vectorization — temporal blocking removes the
constraint entirely and takes the same cases to 800–828 and 340–354 MC/s.

**Practical guidance:** on a PML-dominated model, without temporal blocking, on
any host whose cores can saturate its memory system, `--engine=multithreaded`
is the better choice.

---

## 4. The Vulkan GPU engine

A full Vulkan 1.3 compute implementation — Yee update, UPML, Mur, excitation,
TF/SF, dispersive materials, lumped RLC, energy reduction and probe gather all
run as compute shaders, with the operator compressed on the device.

### 4.1 Discrete GPUs

| Model | RX 6800<br>best | vs upstream best | UHD 770<br>best | vs upstream best | HD 530<br>best | vs i7-6700K<br>upstream best | Vega 7<br>best | vs 5600G<br>upstream best |
|---|---|---|---|---|---|---|---|---|
| 64³ PEC | 12433 | 6.10× | 633 | 0.31× | 505 | 0.93× | 587 | 0.59× |
| 160×128×192 PEC | 13070 | 16.57× | 611 | 0.78× | 381 | 0.85× | 557 | 1.28× |
| 224³ PEC | 12319 | 21.01× | 601 | 1.03× | 484 | 1.14× | 485 | 1.13× |
| 160×128×192 PML_8 | 9844 | 35.15× | 344 | 1.23× | 225 | 1.54× | 413 | 2.85× |
| 224³ PML_8 | 10212 | 38.98× | 347 | 1.33× | 270 | 1.61× | 402 | 2.46× |
| 224³ Mur | 5125 | 9.66× | 518 | 0.98× | 503 | 1.40× | 444 | 1.15× |

The changes that produced those numbers, in order of what they were worth:

- **Trapezoidal temporal blocking** (`99ac18a`, `d9a6431`, `88032e4`) —
  **2.2×** on grids exceeding the card's cache. §5.4.
- **Don't put field buffers in an undersized BAR aperture** (`fbd95a6`) —
  **6.4×** on grids exceeding the 256 MB aperture. This card has no resizable
  BAR, so above that size the buffers were being serviced across PCIe rather
  than from VRAM.
- **Per-cell operator storage when deduplication stops paying** (`6b571da`) —
  **1.93–2.76×** on graded meshes. Found by watching throughput collapse 4×
  when 493,040 unique coefficient sets in a 23 MB table turned six per-cell
  gathers into six scattered cache lines. What matters is table size *relative
  to cache*, not the compression ratio.
- **Field dumps gathered on the GPU** (`76fa8f6`, `d38071b`; replacing the
  async download of `ea82f59`, which helped only without ReBAR) — each dump's cells are
  gathered at its timestep in the batch that ends there, and FD/NF2FF/SAR
  dumps keep their DFT sums on the GPU. On 160³ with ReBAR: plane dumps
  **221×** (298 → 1.34 s), an NF2FF box **115×**, whole-volume dumps **97×**;
  plane and NF2FF dumps now cost nothing measurable, where the old default ran
  below the CPU engine. Not visible in this document's benchmarks, which
  deliberately dump nothing, but it is the difference that matters most in
  practice on dump-heavy models (FDTD/GPU_PIPELINE_OPTIMIZATION.md §4).
- **Field buffers always device-local** (`0d92e77`) — nothing reads them through
  the BAR any more, and on a card with a legacy 256 MB BAR, runs sharing the
  GPU no longer spill each other's fields into system RAM: **~9×** per run for
  three runs on the RX 6800 (175–210 → 1680–1740 MC/s). Single runs are
  unchanged.
- **Narrowed the compressed-operator index to 8/16/32 bits** (`49c7efc`) —
  +3.5–6.2%.
- **Deduplicated UPML coefficients into shared tables** (`00ba1c6`) —
  1.08–1.17×, and 19.2 MB → 1.6 MB of tables.
- **Stopped allocating the async dump ring for runs that never dump**
  (`05ce5f0`) — **2.8× less GPU memory** (224³: 803 → 288 MB VRAM). Throughput
  unchanged; this sets the largest grid the card can hold, which is the GPU's
  hard limit.

**The flat GPU Yee kernel is at roofline.** Counting traffic from the shader
source rather than assuming it gives 74 B/cell/timestep
([OPTIMIZATIONS.md](OPTIMIZATIONS.md) §4.7/C4), and the flat 5674 MC/s × 74 B =
**420 GB/s** against a `clpeak` ceiling of 477 GB/s on this card — **88% of
achievable bandwidth**. That is precisely why §5.4 was worth doing and why it
works: the only way past a roofline is to stop needing the resource it measures.

### 4.2 Integrated GPUs are not compute devices — but the comparison is relative

On the i9-13900K the UHD 770 measures **317–633 MC/s on every grid**, essentially
flat with grid size, i.e. **0.31×–1.33×** upstream's best CPU configuration. That
flatness has a cause: `clpeak` reports 48.7 GB/s for it against the CPU's ~49 GB/s
STREAM-Triad ceiling, because the iGPU and the CPU *are the same memory system*,
so there is no cache hierarchy of its own for grid size to interact with.

The i7-6700K result is the same device class against a much weaker CPU, and it
lands differently. The HD 530 is **slower in absolute terms** than the UHD 770 on
every one of the six models — and yet on two of them, 224³ PML_8 and 224³ Mur,
it is the **fastest engine on its machine**, beating every CPU configuration
including the blocked one. Against upstream's best it reaches 1.54×–1.61× on the
PML models and 1.40× on Mur. An iGPU is not fast; it is simply not competing with
much on a four-core desktop part, and it does not slow down when a PML is added
the way the CPU does (224³: 478 → 270 MC/s on the GPU, 423 → 168 on the CPU).

The qualifier matters. On the smaller 160×128×192 PML_8 model the blocked CPU
schedule reaches 251 MC/s against the GPU's 225, so the GPU is *not* simply the
right answer for PML on this machine — it wins where the grid is large enough
that the CPU's 8 MB L3 has stopped helping.

The Ryzen 5 5600G's Vega 7 is the third instance and the most informative,
because it is the only integrated GPU here facing a CPU that is *not* weak. It
is the fastest device on its machine on the two PML models — 413 and 402 MC/s
against the blocked CPU schedule's 354 — and it loses everywhere else, by 1.1×
on Mur, 2.2× on 224³ PEC and 3.3× on 64³ PEC. Against upstream alone it reads
0.59×–2.85×, but that flatters it: the right comparison on this host is against
the fork's own CPU engine on the same memory system, and there it wins two of
six.

That is the pattern all three integrated parts share, and the 5600G shows it is
not about CPU weakness. What decides it is arithmetic intensity: a PML pass does
enough work per byte that a GPU with no bandwidth advantage can still be ahead
on ALU throughput, while a plain Yee sweep on a large PEC grid is pure traffic,
and there a blocked CPU schedule that stops making the traffic beats a GPU that
still must.

So the honest statement is not "integrated GPUs are useless" but: **an
integrated GPU is worth trying when the model is large and PML-dominated, and
is a poor bet otherwise — regardless of how good the CPU beside it is.** On the
i9-13900K it also remains a portability check on a second Vulkan driver — ANV
rather than RADV, which is how driver-specific assumptions get caught. Device
selection already prefers the discrete GPU where one exists.

### 4.3 Correctness work that came with the GPU engine

Two fixes worth naming, because performance work that breaks results is worth
nothing:

- **A TF/SF write race** (`b87a1f7`): a non-atomic `+=` on field elements shared
  by two TF/SF box faces made the Vulkan engine **nondeterministic** on any TF/SF
  model — six runs gave six different answers. Found by hashing probe traces
  across repeated runs of *one* engine, which separates "these two engines
  disagree" from "this one disagrees with itself".
- **An operator indexing regression**, fixed in `81005f8` by restoring the
  x-fastest EC index in `Calc_ECOperatorPos`. Every engine routes through that
  function, so cross-engine comparison is structurally blind to this class of
  bug — all engines agree, and all are wrong together. Only absolute-value tests
  on inhomogeneous models catch it.

§11 is the end-to-end check that followed, and it is not blind in that way,
because it compares the fork against upstream rather than only against itself.
On a real model, every fork engine matches upstream to float32 rounding, and
the engines the two builds share match bit for bit.

---

## 5. Trapezoidal temporal blocking

### 5.1 How it works and why it is safe

This is the largest CPU result in this document, and the reasoning behind it is
worth stating because it is not "make the arithmetic faster".

The multithreaded AVX2 engine **is already at the DRAM roofline.** Traffic
measured with the core PMU on the i9-13900K is 31.4 bytes per cell per timestep
against a 24 B/cell ideal, which at its optimum is ~48 GB/s against a ~49 GB/s
STREAM-Triad ceiling. There is no arithmetic left to win and no scattered access
left to tile. (Traffic and roofline figures in this section are from
[OPTIMIZATIONS.md](OPTIMIZATIONS.md) §4.5 and §4.7, measured separately on that
same host; the throughput tables below are from this run.)

What is left is to *stop going to DRAM*. A trapezoidal schedule advances a
cache-resident tile of the grid through *k* consecutive timesteps before moving
on, which divides DRAM traffic by roughly *k*. Measured on a proxy benchmark:
30.95 → 2.40 B/cell, **12.9× less traffic.**

Three properties make this usable rather than merely fast:

- **It is bit-identical to the flat sweep.** Not "within tolerance" —
  element-for-element identical. `python/Tests/test_temporal_blocking.py` checks
  this for PEC, PML, Mur, Drude, lumped RLC, a local absorbing sheet, a TF/SF
  plane wave, a combined case, and a sweep over (k,W) ∈ {2:8, 4:16, 3:13, 6:20}
  × {1,3,4} threads; all 9 cases pass on the build measured here. Each case
  asserts that blocking actually *engaged* before comparing fields — otherwise
  the comparison passes vacuously every time an extension declines and the run
  silently falls back.
- **It refuses rather than risks.** Blocking engages only when *every* active
  engine extension implements the per-slab hooks. An extension that has not
  returns `false` by default and sends the whole run back to the flat sweep, so
  a newly added extension is safe by omission rather than by enumeration. The
  startup line names every extension that declined.
- **It carries the extensions**, not just the bare PEC kernel: UPML, Mur and
  local absorbing BCs, dispersive materials, lumped RLC and TF/SF plane-wave
  sources all have slab paths. Steady-state detection is the one extension that
  genuinely cannot be blocked — it integrates energy over the whole grid every
  period, and a blocked schedule has no moment at which the whole grid is at one
  timestep.

### 5.2 CPU results

Flat → blocked at each machine's pinned thread count:

| Model | field state | i9-13900K<br>flat → blocked | | i7-6700K<br>flat → blocked | | Ryzen 5 5600G<br>flat → blocked | |
|---|---|---|---|---|---|---|---|
| 64³ PEC | 6 MB | 3507 → 3458 | 0.99× | 937 → 938 | 1.00× | 1708 → 1711 | 1.00× |
| 160×128×192 PEC | 94 MB | 889 → 3249 | 3.66× | 497 → 706 | 1.42× | 443 → 1518 | 3.42× |
| 224³ PEC | 270 MB | 632 → 2607 | 4.13× | 475 → 497 | 1.05× | 432 → 1062 | 2.46× |
| 160×128×192 PML_8 | 94 MB | 266 → 800 | 3.00× | 164 → 221 | 1.34× | 135 → 340 | 2.52× |
| 224³ PML_8 | 270 MB | 237 → 828 | 3.50× | 183 → 205 | 1.12× | 152 → 354 | 2.32× |
| 224³ Mur | 270 MB | 569 → 1848 | 3.25× | 388 → 339 | **0.87×** | 384 → 771 | 2.01× |

(Both columns at each machine's pinned thread count, so the comparison is
controlled. The i7-6700K's blocked schedule prefers 6 threads and reaches
1.18×–1.75× when allowed them on the four models where it helps at all; on Mur
it still loses, 406 → 382. The Ryzen 5 5600G's prefers 6 as well and is
understated here by about 8%. See §2.4 and the `+tblock default` column in §8.)

The Ryzen 5 5600G is the second-best result in the document after the
i9-13900K, and it comes from a part with a fifth of the i9's core count and no
discrete GPU. Like the i9-13900K it regresses on nothing — 2.0×–3.4× on all
five non-resident grids — and unlike every other host it does not collapse at
high thread counts either. The
reason is the one §5.3 develops — 16 MB of undivided L3 behind six cores gives
a 13-line tile, wide enough to pay everywhere, where the i5-8500's 9 MB gives 7
and the i7-6700K's 8 MB gives 6.

The five servers put the same comparison on a wider range of cache topologies,
and split it in two. Flat → blocked at each machine's pinned thread count, 224³
PEC:

| Machine | last-level cache | flat → blocked | | |
|---|---|---|---|---|
| i9-13900K | 36 MB, one cache | 632 → 2607 | **4.13×** | |
| Ryzen 5 5600G | 16 MB, one cache | 432 → 1062 | **2.46×** | |
| Core i5-8500 | 9 MB, one cache | 225 → 303 | 1.35× | |
| i7-6700K | 8 MB, one cache | 475 → 497 | 1.05× | |
| Ryzen 9 3950X | 4 × 16 MB | 440 → 513 | 1.17× | → **2.27×** corrected |
| 2× Xeon E5-2620 v3 | 2 × 15 MB | 514 → 365 | **0.71×** | → 1.11× corrected |
| 2× Xeon Gold 6130 | 2 × 22 MB | 1282 → 637 | **0.50×** | → 1.00× corrected |

The first four rows are the story §5.3 tells: gains scale with how big a tile
the cache allows, monotonically in cache size across 8, 9, 16 and 36 MB, and
8–9 MB is near the bottom of what pays. The last three are
a different story, and the "corrected" column is the point — those are the same
machines with the tile width set by hand to the cache their threads actually
have. Taken at face value the bottom two rows say temporal blocking is a 30–50%
*regression* on a dual-socket server; corrected, they say it is roughly
break-even there. Neither is the 3–4× of the i9-13900K, but the difference
between "loses half your throughput" and "does nothing" matters, and only one of
them is a property of the schedule.

The 64³ row is the schedule declining to engage, and saying so:

```
AVX2 temporal blocking: disabled, only 64 x-lines for a tile width of 384.
```

The tile width is derived from the last-level cache, and at this grid's small
x-planes that comes out wider than the grid itself. There is nothing to gain
here anyway — 6 MB of field state is already L3-resident, and blocking a
cache-resident grid was measured to *lose* 32% — but the guard that fires first
is the geometric one. Either way the run falls back to the flat sweep, which is
why the row reads ~1.00× rather than a regression.

### 5.3 CPU limits: a large, undivided cache — and the tile-width defect

> **Since 2026-10 (§5.7)** the derived tile width starts at half the cache the
> threads span. It widens back towards all of it when that keeps the block
> depth on a model with small coefficient tables. The blocked schedule's
> synchronisation is also much cheaper. The tables in this section were
> measured before both changes.

On the i7-6700K's 8 MB L3 the schedule's gains collapse and one model regresses:

| Model | tile geometry chosen | flat → blocked (4t) | |
|---|---|---|---|
| 160×128×192 PEC | k=7, W=14, 11 tiles, 7.9 MB/tile | 497 → 706 | 1.42× |
| 224³ PEC | k=3, W=6, 37 tiles, 6.9 MB/tile | 475 → 497 | 1.05× |
| 224³ PML_8 | k=3, W=6, 37 tiles, 6.9 MB/tile | 183 → 205 | 1.12× |
| 224³ Mur | k=3, W=6, 37 tiles, 6.9 MB/tile | 388 → 339 | **0.87×** |

Compare the i9-13900K, where the same models get k=15–16 and 31–64-line tiles.
The tile width is derived from the L3, so a 4.5× smaller cache buys a 4.5×
narrower tile, and the depth *k* has to shrink with it to satisfy the geometric
`W >= 2k` constraint. At k=3 the schedule is paying the trapezoid's full wedge
and re-traversal overhead to divide DRAM traffic by three, and on the Mur model —
whose boundary work is per-face and does not shrink with the tile — that trade is
a net loss.

The Core i5-8500 is the same story on a 9 MB L3 and confirms it — k=3, W=7,
8.0 MB per tile, and 1.35× on 224³ PEC. What it adds is that the floor is not as
low as the i7-6700K suggested: at the same tile geometry the i5-8500 gains where
the i7-6700K barely did, so "small cache" is not on its own a prediction.

The Ryzen 5 5600G fills in the middle of that range and is the cleanest
confirmation of the mechanism, because it is the same six-core single-socket
shape as the i5-8500 with roughly double the cache. `getconf` reports its full
16 MB — one CCX, one undivided L3, so the derived width is the right one and
none of the defect below applies — and the engine announces:

```
AVX2 temporal blocking ACTIVE (prototype): k=6, tile width 13 x-lines, 14.9297 MB per tile, 17 tiles
```

k=6 and a 13-line tile against the i5-8500's k=3 and 7 lines, for 2.46× against
1.35× on the identical model. Cache size sets tile width, tile width sets the
depth the `W >= 2k` constraint allows, and depth is what divides the DRAM
traffic. That is the whole causal chain in one controlled pair.

The variable this section names — *cache size* — is nonetheless incomplete. The
other half is a defect:

**The tile width comes from `sysconf(_SC_LEVEL3_CACHE_SIZE)`
(`FDTD/engine_avx2_multithread.cpp`), which reports the size of one
last-level-cache instance.** On a machine whose LLC is a single shared cache —
the i9-13900K, i7-6700K and i5-8500 — that is also the cache the whole run has,
and the derived width is right. On a machine whose LLC is partitioned it is off
by the number of partitions the threads span: a quarter of the truth on a Ryzen
9 3950X (4 CCXs), a half on any dual-socket server.

The tile is divided across cores in x, so each core's slab lands in its own
cache instance; the capacity a run has is the *sum* over the instances its
threads occupy, not one of them. Sweeping the width by hand
(`OPENEMS_AVX2_TEMPORAL_BLOCK=k:W`) against thread count says so directly —
224³ PEC, MC/s, single runs:

| Machine | LLC the threads span | auto (derived) | best hand-set | |
|---|---|---|---|---|
| i9-13900K, 8t | 36 MB × 1 | **2544** (35.6 MB) | 2160 (52.8 MB) | auto is best |
| Core i5-8500, 2t | 9 MB × 1 | **306** (8.0 MB) | 274 (18.4 MB) | auto is best |
| Core i5-8500, 6t | 9 MB × 1 | **335** (8.0 MB) | 252 (18.4 MB) | auto is best |
| 2× Xeon E5-2620 v3, 6t — one socket | 15 MB × 1 | **363** (14.9 MB) | 326 (29.9 MB) | auto is best |
| 2× Xeon E5-2620 v3, 12t — two sockets | 15 MB × 2 | 356 (14.9 MB) | **569** (44.8 MB) | **1.60×** |
| Ryzen 9 3950X, 4t — one CCX | 16 MB × 1 | **1133** (14.9 MB) | 600 (29.9 MB) | auto is best |
| Ryzen 9 3950X, 16t — four CCXs | 16 MB × 4 | 504 (14.9 MB) | **1000** (59.7 MB) | **1.99×** |
| 2× Xeon Gold 6130, 16t — one socket | 22 MB × 1 | 983 (21.8 MB) | **1430** (43.6 MB) | **1.46×** |
| 2× Xeon Gold 6130, 32t — two sockets | 22 MB × 2 | 636 (21.8 MB) | **1280** (59.7 MB) | **2.01×** |

Every row where the thread set fits inside one cache instance picks the derived
width, and every row where it spans several wants a wider one. The 2× Xeon E5-2620
v3 makes the point without leaving the machine: the *same* binary on the *same*
model has the derived width optimal at 6 threads on one socket and 60% short at
12 threads across two.

Two details keep this from being a one-line fix. The optimum tracks the
aggregate capacity but not exactly — the 2× Xeon Gold 6130 at 16 threads on a
single 22 MB socket prefers a 43.6 MB tile, which is explained by Skylake-SP's
**non-inclusive** L3 and 1 MB private L2 per core (16 × 1 + 22 ≈ 38 MB of real
capacity, not 22), so the right quantity is aggregate capacity across the whole
hierarchy the threads own, not the L3 line from `sysconf`. And the trapezoid's
live set is smaller than *W* planes — it shrinks from *W* to *W* − 2*k* as the
tile advances — so the mapping from capacity to *W* already carries a fudge
factor that these hosts show is not universal.

**Until it is fixed, one environment variable recovers it per host.** Set the
width explicitly to about (number of LLC instances the run spans) × the derived
width:

```bash
OPENEMS_AVX2_TEMPORAL_BLOCK=16:52 openEMS model.xml --engine=avx2-multithreaded
```

Raw sweep output is `python/Tests/results/tile_width_sweep_2026-09-11.txt`. This
is the CPU twin of §5.5, which is the same mistake on the GPU: a tile target
that is a constant where it should be a property of the device and of how much
of the device the run is using.

### 5.4 GPU results

The same schedule runs in `Engine_Vulkan`, enabled by
`OPENEMS_GPU_TEMPORAL_BLOCK=<k>`. §4.1 concluded that the GPU Yee kernel is at
roofline and that further gains have to come from *reducing traffic* rather than
from tightening arithmetic. This is that reduction, and it is the largest new
GPU result:

| Model | field state | RX 6800<br>flat → blocked | | UHD 770<br>flat → blocked | | HD 530<br>flat → blocked | |
|---|---|---|---|---|---|---|---|
| 64³ PEC | 6 MB | 12433 → 12385 | 1.00× | 633 → 633 | 1.00× | 505 → 504 | 1.00× |
| 160×128×192 PEC | 94 MB | 13070 → 12950 | 0.99× | 524 → 611 | 1.17× | 368 → 381 | 1.04× |
| 224³ PEC | 270 MB | 5674 → **12319** | **2.17×** | 520 → 601 | 1.16× | 478 → 484 | 1.01× |
| 160×128×192 PML_8 | 94 MB | 9462 → 9844 | 1.04× | 317 → 344 | 1.08× | 225 → 224 | 1.00× |
| 224³ PML_8 | 270 MB | 4619 → **10212** | **2.21×** | 323 → 347 | 1.08× | 270 → 269 | 1.00× |
| 224³ Mur | 270 MB | 5125 → 5125 | 1.00× | 518 → 518 | 1.00× | 503 → 502 | 1.00× |

Read the RX 6800 column top to bottom and the mechanism is visible directly. The
card has a 128 MB Infinity Cache. The two small grids are already inside it and
there is nothing to save — and on the 94 MB PEC row the schedule engages anyway
and loses 1%, which is a defaulting bug and is §5.5. The two 270 MB grids are
outside the cache, and those are exactly the two rows that gain **2.2×** — which
erases the throughput cliff at the cache boundary: 224³ PEC went from 5674 (below
it) to 12319, which is within 6% of the 13070 the 94 MB grid reaches *inside* the
cache. The blocked schedule makes a grid that does not fit run at close to the
speed of one that does. That is the whole claim of the method, and this is the
cleanest demonstration of it in this document.

Three device classes, three outcomes, all consistent with the same explanation:

- **RX 6800 (discrete, bandwidth-bound, 128 MB cache)** — 2.2× where the grid
  exceeds the cache, nothing where it does not.
- **UHD 770 (integrated, shares 36 MB of CPU L3)** — a flat 1.08–1.17×
  everywhere above 6 MB. Its cache is small enough that every real grid is
  outside it, so the gain does not switch on and off with grid size; but the
  device is partly latency-bound, so the gain is small.
- **HD 530 (integrated, Gen9)** — nothing, anywhere. `clpeak`-class bandwidth is
  not what limits this part; it is 24 execution units. Removing DRAM traffic from
  a compute-bound kernel removes nothing that was in the way.

The correctness apparatus is the same as the CPU's, and for the same reason.
`python/Tests/test_gpu_temporal_blocking.py` checks the blocked sweep
**bit-identical** to the flat one for PEC, fused UPML, unfused UPML, a dispersive
material with UPML, and a sweep of (k,W) schedules including one where W does
not divide NX — 11 tests, all passing on the build measured here. Two of them
exist specifically to close gaps the others cannot see: one asserts the flat
reference is not all zeros (a comparison against zero passes no matter what),
and one checks the fused and unfused PML paths against *each other*, because
each of the first two compares a path against itself and both would pass if the
slab range were wrong in a way they shared.

The Mur row was the refusal mechanism working, at the time:

```
Engine_Vulkan: temporal blocking disabled — Mur ABC has no slab path yet
```

Since 2026-10-19 Mur, TF/SF, lumped RLC, GPU probes and local absorbing sheets
all have slab paths (§5.8), so that row would now block. It has not been
re-measured.

### 5.5 GPU limits: a guessed tile target, and compute-bound devices

The tile target is a **tunable with a per-device-class default** — 64 MB
discrete, and 16 MB integrated at the time (80 % of the CPU's L3 since §5.9) —
because Vulkan exposes no last-level-cache size to query. On the RX 6800 the real cache is 128 MB, so a grid between 64 and 128 MB
is already cache-resident but still looks too large to the guard. It engages,
adds the trapezoid's wedge overhead, and saves traffic that was not being spent:

| 160×128×192 PEC, RX 6800 | result |
|---|---|
| flat | 13070 |
| blocked, default 64 MB target | 12950 |
| blocked, `OPENEMS_GPU_TEMPORAL_BLOCK_MB=64` (single run) | 12508 |
| blocked, `OPENEMS_GPU_TEMPORAL_BLOCK_MB=128` (single run) | **13143** — *refuses, correctly* |

Told the card's actual cache size, the guard declines and full flat throughput
comes back. Told the default, it costs about 1% in the best-of-3 matrix row and
about 5% in the direct single-run A/B above. Either way it is a loss, and it is
avoidable per host with one environment variable.

This is a defaulting problem, not a schedule problem, and the fix is known: a
device-ID table or a startup micro-probe instead of a constant. §5.9 took a
third way: `--benchmark` measures the schedules on the model itself. §5.8 found a
second defect in the same guess: the target counts only field bytes, so with an
uncompressed operator a tile is three times larger than the target, and the
default reaches 1.22× where 3.2× was available. Until then, on a
card with an unusually large last-level cache, set
`OPENEMS_GPU_TEMPORAL_BLOCK_MB` (now also `--gpu-temporal-block-mb`) to it.

On the HD 530 the schedule does not help speed: it is flat at 1.00×–1.04× on
every row. This section first put that down to a compute limit, its 24
execution units. That was wrong. §5.9 measured the DRAM traffic: the HD 530
runs a few percent of its FLOP rate, and an on-chip ceiling happens to
coincide with its DRAM limit. Blocking removes a third of the DRAM traffic,
which saves energy, and the other ceiling keeps the speed where it was.

The Vega 7 is flat too — 482.5 flat against 485.1 blocked on 224³ PEC, and a
depth sweep that does not move at all:

| 224³ PEC, Vega 7 | k=0 | k=4 | k=8 | k=16 | k=32 |
|---|---|---|---|---|---|
| MC/s | 482.5 | 485.0 | 485.1 | 485.1 | 485.1 |

but for a different reason, and the difference matters because it is the one
integrated part here that is *not* compute-bound. It accepts the schedule and
says so:

```
temporal blocking — k reduced from 16 to 6 to fit a 13-line tile (W >= 2k)
trapezoidal temporal blocking active — k=6 W=13 (17 tiles, 14 MB/tile, 257 MB of field state)
```

The depth is clamped to 6 by the same `W >= 2k` geometry as on the CPU, and at
that depth there is nothing to keep the reused planes in: this part has no
last-level cache worth the name, and its "tile" is backed by the same DDR4 the
flat sweep was already streaming. So the sweep is flat from k=4 upward — every
value of *k* is doing the same thing. Blocking needs a cache to block into,
which is a third precondition alongside "not compute-bound" and "grid exceeds
the cache", and it is the one an integrated GPU fails by construction.

### 5.6 Limits common to both: opt-in, and capped by probe cadence

Both schedules were prototypes behind environment variables —
`OPENEMS_AVX2_TEMPORAL_BLOCK=<k>` and `OPENEMS_GPU_TEMPORAL_BLOCK=<k>` (here,
k=16 on the CPU and on the RX 6800, k=6 on the integrated devices) — and did
nothing unless set. The CPU schedule still is. Since 2026-10-19 the GPU one is
on by default on a discrete card for grids beyond twice its tile target
(§5.8), and since 2026-10-20 on an integrated one (§5.9). Both now also take
command-line options, `--gpu-temporal-block` and `--avx2-temporal-block`. Two limits are worth knowing before relying on the numbers
in §5.2 and §5.4:

- **Block depth is capped by the probe interval.** `openems.cpp` calls
  `IterateTS()` with the number of timesteps until the next probe or dump, and
  both schedules clamp *k* to it — which is what keeps probes and dumps correct
  with no work on their side. That interval is typically 5–25 timesteps, which
  happens to bracket the measured optimum (*k* = 6–24). A run oversampled far
  past Nyquist will not reach these figures.
- **These models have no probes or dumps at all**, so they see the ceiling of
  what blocking can do. A probe-dense model will land lower.

### 5.7 The CPU schedule was synchronisation-bound, and its tile must leave room for the coefficients (2026-10)

A production run made the problem visible. On the δ/10 skin-depth mesh of an
on-chip line (247×153×133 = 5.03 M cells, 0.48 MB per x-plane, Mur on all six
faces, two lumped ports), the GPU engine kept its cards at 100 %, while the
blocked CPU workers sat at **79–89 % busy**. Each worker went to sleep about
**30 000 times a second**, roughly 185 times per timestep. Memory stalls count
as busy time, so the missing time was sleep, at barriers. Three costs added up:

- **Barrier count.** The blocked path barriers after every extension hook, once
  per timestep for every tile and every wedge. On this model that is 24
  barriers per sweep step: six Mur faces with three hooks each, a lumped port
  and the excitation with two each, and two field updates. Per timestep that
  came to 264 barriers on the i9-13900K at 16:44 tiles, 552 on the Ryzen 5
  5600G at 10:20 and 696 on the i7-6700K at 8:16.
- **Waking up.** `Barrier` is a mutex plus a condition variable. A thread that
  arrives early sleeps in the kernel and takes 3–4 µs to run again after the
  release. With the old schedule that was **18–28 % of all thread time**.
- **Splitting by x-line.** Each step's slab was divided over the threads in
  whole x-lines, but slabs are often only a few lines wide and wedges start at
  one. Replaying the schedule gives a work balance of 0.81–0.91 at 6–8 threads
  on this model, and 0.59 at 32.

Four changes address these. All are switchable, so each can be measured on its
own against the old behaviour in the same binary:

| Change | Switch | Default |
|---|---|---|
| Skip a hook and its barrier for a slab the extension cannot reach (`SlabTouches()`: the Mur faces normal to x, lumped elements, excitation, UPML regions). Run consecutive hooks that only read the fields without a barrier between them (`SlabHookWritesFields()`: the Mur pre/post hooks, the lumped-element pre hook) | `OPENEMS_AVX2_TBLOCK_ELIDE=0` restores the old behaviour | on |
| Split each slab's update by y-row, a contiguous range of memory, instead of by x-line | `OPENEMS_AVX2_TBLOCK_SPLIT=lines` restores | on |
| `HybridBarrier`: atomic arrival, and an optional spin before sleeping | `OPENEMS_AVX2_BARRIER_SPIN_US=<µs>` | 0 (no spin) |
| Per-thread barrier counts, time spent waiting for the slowest thread, and time spent waking up, printed at the end of the run | `OPENEMS_AVX2_SYNC_STATS=1` | off |

**Results are bit-identical, not merely close.**
`python/Tests/tblock_identity.py` dumps E and H over the whole domain for
every variant and compares them bit for bit with the previous build's flat
sweep: both builds, flat and blocked, both splits, elision on and off,
spinning, and sync statistics on. Six models (the TL case; 224³ Mur, PML_8
and PEC; 160×128×192 PML_8 and PEC), each at 8 and 7 threads, came out
identical. The previous build's own blocked schedule matches as well.

**Each change on its own**, on the TL model with the tile width fixed (MC/s;
"waiting" is waiting for the slowest thread, "waking" is waking up after the
release; both are shares of thread time):

| Ryzen 5 5600G, 6 threads, 10:20 | MC/s | barriers per timestep | waiting | waking |
|---|---|---|---|---|
| previous build | 680 | – | – | – |
| new build, old schedule | 671 | 552 | 20.9 % | 27.2 % |
| + split by y-row | 698 | 552 | 16.2 % | 28.3 % |
| elision only (split by x-line) | 884 | 189 | 18.5 % | 13.3 % |
| y-row split + elision | 929 | 189 | 12.4 % | 14.0 % |
| + spin 20 µs | 1069 | 189 | 12.1 % | 2.1 % |
| + spin 100 µs | **1103** | 189 | 11.4 % | 0.9 % |

| i9-13900K, 8 P-cores, 16:44 (two rounds) | MC/s | barriers per timestep | waiting | waking |
|---|---|---|---|---|
| previous build | 1220 / 1209 | – | – | – |
| new build, old schedule | 1262 / 1207 | 264 | 22–24 % | 18 % |
| y-row split + elision | 1499 / 1553 | 92 | 16–18 % | 8–9 % |
| + spin 100 µs | **1680 / 1679** | 92 | 16–17 % | 1.5 % |

On the i7-6700K (4 threads, its auto tile of 8:16), the previous build ran at
293 MC/s, y-row split plus elision at 354 and spin 100 µs at 385. Elision is
the largest single step: it removes two thirds of the barriers. Spinning
removes almost all of the waking time. The y-row split is worth 3–7 % at these
thread counts and should matter more at 16–32 (see the balance model above).
DRAM traffic per cell update does not change: the gain is synchronisation
alone.

**The tile-width optimum moved, on this model to about half the cache.** With
all three changes on, the width was swept by hand (spin 20 µs / 100 µs, k=16 on
the i9-13900K). "Previous build" is the same sweep done the day before on the
old schedule. Bytes per cell update are DRAM traffic divided by cell updates,
from the memory-controller counters at spin 100 µs:

| i9-13900K, 36 MB L3 — tile (share of L3) | previous build | spin 20 | spin 100 | B/cell |
|---|---|---|---|---|
| 75 lines, auto at the time (99 %) | 865 | 972 | 944 | 39 |
| 64 (85 %) | 954 | 1141 | 1202 | 30 |
| 52 (69 %) | 1186 | 1548 | 1573 | 18 |
| 44 (58 %) | 1208 | 1688 | 1722 | 15 |
| 36 (48 %) | 1165 | 1726 | 1751 | 14 |
| 32 (42 %) | – | 1704 | **1758** | 14 |
| 12:24 (32 %) | – | 1503 | 1614 | 16 |
| 8:16 (21 %) | – | 1318 | 1373 | 22 |
| flat sweep | 387 | 401 (no spin) | | 107 |

| Ryzen 5 5600G, 16 MB L3 — tile (share of L3) | previous build | spin 20 | spin 100 | B/cell |
|---|---|---|---|---|
| 16:33, auto at the time (98 %) | 552 | 674 | 690 | 50 |
| 12:28 (83 %) | – | 924 | 950 | 34 |
| 12:24 (71 %) | 679 | 960 | 986 | 32 |
| 10:20 (60 %) | 681 | 1071 | **1088** | 26 |
| 8:16 (48 %) | 618 | 1053 | 1073 | 24 |
| 6:12 (36 %) | – | 931 | 943 | 29 |

| i7-6700K, 8 MB L3 — tile (share of L3) | previous build | spin 20 | spin 100 |
|---|---|---|---|
| 8:16, auto at the time (95 %) | 293 | 377 | 382 |
| 10:20 (119 %) | – | 338 | 345 |
| 6:12 (71 %) | – | 436 | 450 |
| 5:10 (60 %) | – | 465 | 464 |
| 4:8 (48 %) | – | 457 | **467** |

For this model, on all three machines, the best tile fills 40–60 % of the L3,
across a broad plateau, and a full-cache tile is 1.2–1.9× slower. That is a
property of the model, not of the cache:
- **The TL model's tables are as large as its fields.** Its strongly graded
  mesh gives the compressed coefficient tables 317 192 unique entries, 122 MB,
  as large as the field arrays. The tile accounting counts only the fields, so
  half the cache by that measure is about all of it in practice.
- **The uniform 224³ grids are the opposite.** With 12 unique entries, a
  half-cache tile only forces k down, and a full-cache tile that keeps k wins:

| spin 100 µs | half cache, k lowered | full cache, k kept |
|---|---|---|
| i9-13900K, 224³ PEC | 2323 (7:15) | **2971** (15:31) |
| i9-13900K, 224³ Mur | 1868 (7:15) | **2258** (15:31) |
| i9-13900K, 224³ PML_8 | 815 (7:15) | **831** (15:31) |
| Ryzen 5 5600G, 224³ PEC | 1096 (3:6) | 1644 (6:13); **1762** at 8:16 |
| Ryzen 5 5600G, 224³ Mur | 955 (3:6) | 1402 (6:13); **1410** at 8:16 |
| i9-13900K, 160×128×192 PML_8 | **1226** (16:32) | 868 (16:64) |

The last row is the TL case again, from another direction. Half the cache
still allows k=16 there, and the UPML regions' per-cell state takes the other
half.

So `ConfigureTemporalBlocking()` now works like this:
1. **Start at half the L3 the threads span.**
2. **Widen to keep k where tables are small.** If that tile is narrower than
   2k and the coefficient tables are under a quarter of the field bytes, widen
   it up to the full L3 to keep k.
3. **Otherwise lower k.** Beyond that, k drops to W/2. A shallow block that
   stays resident beats a deep one that does not (4:8 over 8:16 on the
   i7-6700K).

On the nine model/machine pairs measured, this lands within 7 % of the best
hand-set tile; the worst is 224³ PEC on the Ryzen, at 6:13 against 8:16.
Checked on the final build on the i9-13900K (auto tile, spin 100 µs):

| Model | Tile chosen | MC/s |
|---|---|---|
| TL model | 16:37 | 1760 |
| 224³ PEC | 15:31 | 2943 |
| 224³ Mur | 15:31 | 2191 |
| 160×128×192 PML_8 | 16:32 | 1214 | The
rule is fitted to those nine cases, and the tile campaign in OPTIMIZATIONS.md
§4.9 (D4) is what will show whether it holds beyond them.

On the TL model, against each machine's best tile on the previous build, the
new schedule is **+46 %** on the i9-13900K and **+60 %** on the Ryzen 5 5600G.
Against the previous auto tile it is **2.0×** on both and **1.6×** on the
i7-6700K. A full 60 ps run of this model on the i9-13900K drops from 26 h
(previous auto tile) to about 12.6 h.

Spin 100 µs beat 20 µs in 21 of 23 pairs, by 1–7 %, and the two exceptions
are within 1 %. It is not the default yet: a spinning thread shows as 100 %
CPU and costs power, and on a host with more runnable threads than cores it
takes time from the thread being waited for. About **12–20 % of thread time
is still spent waiting for the slowest thread**, more as tiles narrow. What
remains to try is in OPTIMIZATIONS.md §4.9.

What these numbers are not:
- **Few models, and short runs.** The full tile sweeps are single runs of
  8000–12000 timesteps on the TL model. The other models only have the
  half/full comparison above, and only the i9-13900K ablation has a second
  round.
- **Not quiet machines.** The i9-13900K runs a desktop and the i7-6700K serves
  applications; only the Ryzen 5 5600G was idle.
- **Not the §5.2 matrix.** The §5.2/§5.3 matrix and §8 predate this and have
  not been re-run. The new tile rule is also untested on the multi-socket
  hosts.

Raw output: `python/Tests/results/tblock_sync_2026-10-18.txt`. The driver is
`python/Tests/bench_tblock_sync.py`; it reads the memory controllers through
`perf` when `kernel.perf_event_paranoid` ≤ 0.

### 5.8 The GPU on the same model: limited by cache, not synchronisation (2026-10-19)

This is the question §5.7 answered for the CPU, asked of `Engine_Vulkan` on the
same TL model. Two cards:

- the RX 6800 in the i9-13900K host: 128 MB Infinity Cache, 512 GB/s;
- an RX 7900 XTX in a shared compute server: 96 MB Infinity Cache, 960 GB/s,
  run on an idle card.

Two profiling switches came with the work. Both need `--gpu-profile`:

- `OPENEMS_GPU_PROFILE_STAGES=1` writes seven timestamps into every flat
  timestep, so each stage gets its own GPU time. A stage is an extension phase
  or a Yee half-step, with its trailing barrier.
- `OPENEMS_GPU_PROFILE_NO_BARRIERS=1` drops the compute barriers inside the
  timestep loop. This is for timing only, because the fields come out wrong.
  The plain run's time minus this run's time bounds what the barriers cost.

The profile summary now also reports the idle time between batches and the
number of barriers per timestep.

**Synchronisation costs at most 1.4 % on large grids.**

| TL model, flat, 20000 TS | RX 6800 | RX 7900 XTX |
|---|---|---|
| MC/s | 3501 | 5690 |
| GPU time per timestep | 1436 µs | 882 µs |
| Yee V + Yee I | 94.1 % | 94.4 % |
| Mur, lumped ports, excitation | 5.9 % | 5.6 % |
| idle between batches | ≤ 0.07 % | ≤ 0.04 % |
| gain without the 5 barriers per timestep | +1.3 % | +0.8 % |
| 16, 64 or 256 timesteps per batch | within 0.2 % | within 0.2 % |

The 224³ Mur and PML grids behave the same way: removing the barriers gains at
most 1.4 % on either card. On the 224³ Mur grid the six Mur faces take 9–11 %
of the timestep, even though the face cells are a few percent of the grid.

The exception is a small grid on the XTX. 64³ PEC runs 1.92× faster without
its three barriers per timestep (13.3 → 6.4 µs), which is about 2 µs per
barrier. The RX 6800 gains only 4.5 % on the same grid.

**The Yee kernels run at the memory bandwidth.** Each half-step moves 60 B per
cell: the fields read and written, plus 24 B of coefficients. The coefficients
cannot be shared between cells. The TL model's per-cell coefficient sets are
nearly all unique, so its 235 MB operator stays uncompressed even with a 64 MB
table budget. At 60 B per cell the RX 6800 sustains 447 GB/s (87 % of 512) and
the XTX 725 GB/s (76 % of 960). That answers the question OPTIMIZATIONS.md
§4.9 (D6) left open: the XTX is only 1.6× the RX 6800 on this model because it
delivers 1.62× the bandwidth, a smaller share of its peak.

The model's state is 362 MB (121 MB of fields, 241 MB of coefficients), against
a 96 or 128 MB cache, so the only remaining lever is less memory traffic, which
means temporal blocking.

**Temporal blocking is worth 2.1× on the RX 6800 and 3.2× on the RX 7900 XTX on
this grid.** At first the TL model itself could not block, because Mur, lumped
RLC and GPU probes had no slab path. To measure the ceiling, a variant of it uses PEC
instead of Mur and drops the lumped resistors and the probes. It keeps the
port's soft excitation, the grid and the uncompressed operator. Run flat, it is
4–5 % faster than the real model.

Each run is 20000 timesteps, two rounds per row, values in MC/s. The real tile
counts 72 B per cell (fields plus coefficients). The engine's own log line
reports one third of that, because it counts only the fields.

| Schedule | Real tile | RX 6800 | RX 7900 XTX |
|---|---|---|---|
| flat | – | 3640 | 5998 |
| k=16, default tile (137 lines) | 191 MB | 6092 (1.67×) | 7311 (1.22×) |
| k=8, 16 lines | 22 MB | 6792 (1.87×) | 12684 (2.11×) |
| k=16, 32 lines | 45 MB | 7537 (2.07×) | 16622 (2.77×) |
| k=16, 45 lines | 63 MB | 7512 (2.06×) | 17539 (2.92×) |
| k=16, 60 lines | 84 MB | **7673 (2.11×)** | 17487 (2.92×) |
| k=24, 64 lines | 89 MB | 7436 (2.04×) | 18632 (3.11×) |
| k=32, 64 lines | 89 MB | 7540 (2.07×) | **19169 (3.20×)** |
| k=16, 80 lines | 112 MB | 7020 (1.93×) | 12897 (2.15×) |
| k=16, 100 lines | 140 MB | 6827 (1.88×) | 10952 (1.83×) |

- **The default tile repeats the mistake §5.7 found on the CPU.** The tile
  target counts field bytes only, 24 B per cell. With an uncompressed operator a
  cell carries 72 B. The 64 MB default therefore picks two 137-line tiles of
  191 MB each and reaches 1.67× and 1.22×. Counting the coefficients, the same
  64 MB target gives about 45 lines and 2.06× and 2.92×.
- **The XTX falls off at its cache size.** An 89 MB tile reaches 3.20×, a
  112 MB tile 2.15–2.61×. Depth matters there too: k=32 is 10 % faster than
  k=16 at the same tile size.
- **Once blocked, the RX 6800 is not limited by DRAM.** It stays at about
  7500 MC/s for any tile from 45 to 89 MB and any k, and falls off only gradually
  above its cache size. The likely limit is the Infinity Cache's own bandwidth,
  but that has not been measured. Blocked, the XTX is 2.5× the RX 6800,
  against 1.65× flat.
- **Bit-identical.** Full-volume E/H dumps at timesteps 0, 999 and 1998 are
  identical to the flat run for four schedules: k=16 with 45 lines, the default
  k=16 with 137 lines, k=8 with 16 lines and k=32 with 64 lines. That holds on
  both cards. The field at timestep 1998 spans x-lines 16–113, so it crosses
  every tile edge except the 137-line one.

**Implemented the same day: the real model now blocks.** What it took:

- **A tile that counts the coefficients.** The engine sizes the tile from the
  fields plus the operator's bytes per cell: 72 B uncompressed, or 24 B plus
  the index when compressed, whose shared tables come off the target once.
  The 64 MB default now picks 45 lines on this model, not 137.
- **Slab paths for every GPU extension the model uses**, in the flat path's
  order:
  - Mur: a face normal to x runs whole, in the one slab that holds both its
    boundary line and the line it reads; faces normal to y or z are clipped to
    the slab.
  - Lumped RLC and TF/SF: their entry lists are sorted by x, so a slab is a
    contiguous range, as the excitation's already was.
  - Local absorbing sheets: clipped like Mur faces. A sheet normal to x reads
    the line next to the one it writes. During a block of k steps a slab
    boundary only ever sweeps the k lines on either side of a tile edge, so
    the engine moves any tile edge that would come within k lines of such a
    sheet. The sheet then stays inside one core tile.
  - GPU probes needed nothing: the gather is recorded once at the end of a
    batch, when the whole grid is at one timestep.
- **On by default.** Unless `OPENEMS_GPU_TEMPORAL_BLOCK` is set, a discrete
  GPU blocks when the grid's fields and coefficients exceed twice the tile
  target, with the deepest block the tile allows (k = W/2, up to 32).
  `OPENEMS_GPU_TEMPORAL_BLOCK=0` turns it off. Only steady-state detection and
  extensions that run on the CPU still refuse.

Four races in the *flat* path turned up on the way, and are fixed:

- A lumped port's resistor and excitation sit on the same edges, and their
  dispatches had no barrier between them.
- Mur faces with different normals both write the field component along their
  shared edge. The earlier face now wins, as on the CPU.
- With the unfused UPML kernel, UPML's swap between field and flux was
  unordered against Mur and dispersive passes that read the same cells. On a
  test model with Mur faces reaching into PML regions the old flat path was
  not even identical to itself, and E grew to 4e19.
- Crossing absorbing sheets both wrote their common line.

Each fix only changes a result where the old code raced. On the TL model and
on four of the five synthetic cases, the old and new flat paths are
bit-identical.

**Bit-identical to the flat sweep.** `python/Tests/gpu_tblock_identity.py`
runs every case flat (twice), with explicit schedules, and with the default.
Every full-volume E and H dump and every probe file must match the first flat
run bit for bit. The cases:

- Mur with lumped ports and RLC elements;
- a TF/SF box;
- PML with Mur and a Lorentz material, fused and unfused;
- absorbing sheets: first-order normal to y, and super-absorbing, one of them
  normal to x on the tile edges;
- the TL model itself, with Mur on six faces, two lumped ports and four probes.

All pass on the RX 6800 and on the RX 7900 XTX. The XTX host has no h5py, so
its dumps were copied back and compared here (`--compare`). A sine excitation
keeps the fields non-zero from the first step, and a custom function avoids the
steady-state detection that a periodic signal switches on.

**Speed on the TL model itself**, 20000 timesteps, two rounds per row, values
in MC/s:

| Schedule | RX 6800 | RX 7900 XTX |
|---|---|---|
| flat | 3484 | 5697 |
| default (k=22, 45 lines, 62 MB) | 6790–7210 (≈2.0×) | 15448 (**2.71×**) |
| k=24, 48 lines | 6851–7255 | 15982 (2.81×) |
| k=32, 64 lines | 6856–7002 | 16694 (**2.93×**) |

The RX 6800 rows are ranges because its two rounds differ by up to 6 % on
this busy desktop. Every Mur phase is a dispatch and a barrier in every slab
step, and on the XTX a barrier costs about 1.8 µs:

- Running the next step's Mur and RLC pre-voltage alongside the current update
  cut the barriers from about 46 to 38 per timestep, worth 2–3 % on the XTX.
- The rest still costs the XTX 72 of its 323 µs per timestep (22 %). Without
  barriers it would run 251 µs, about 20 000 MC/s, which is 3.5× flat.
- The RX 6800 loses only 6 % to barriers.

A larger tile means fewer trapezoids and so fewer barriers, which is why the
XTX prefers 64 lines at 89 MB, just under its 96 MB cache.

**Then Mur's post and apply phases went into the voltage update.** The thread
that updates a shift cell holds the new V(shift), so it writes the face's
boundary value itself (`update_voltages_mur.comp`). The boundary cell's own
thread skips the components Mur overwrites, so every element has one writer.
The value is computed from V(shift) as the pre phase saved it, because a cell
can be one face's boundary and another's shift cell. The result is
bit-identical to the three-phase path. The kernel is used only when nothing
runs between Mur's post and apply in the flat order: no UPML, dispersive
material, TF/SF box or absorbing sheet. `OPENEMS_GPU_MUR_FUSION=0` turns it
off.

| TL model, 20000 TS, MC/s | Mur separate | Mur fused | |
|---|---|---|---|
| XTX flat | 5670 | 5741 | +1.3 % |
| XTX default (k=22, 45 lines) | 15368 | **16158** | +5.1 % |
| XTX k=32, 64 lines | 16612 | **17264** | +3.9 % |
| RX 6800 flat | 3461 | 3450 | −0.3 % |
| RX 6800 default | 6882–7216 | 7165–7238 | within noise |

On the XTX the blocked default is now **2.84×** flat and k=32 with 64-line tiles
**3.03×**. The barriers drop from 38 to 20 per timestep and cost 51 of 310 µs.
The fused kernel itself is about 3.5 % slower than the plain voltage update,
which the RX 6800, losing little to barriers, does not win back.

**The whole production run, blocked by default, gives the same answer
2.6× faster.** The 60 ps run with port 1 driven (sub-1: 15.8 M timesteps, Mur, both
ports, the port probes and the initial field dumps) ran on the RX 7900 XTX with
nothing set. The engine chose k=22 on 45-line tiles.

| | flat (2026-10-17, 4bfcd74) | blocked (2026-10-19) |
|---|---|---|
| wall time | 14116 s (3.92 h) | **5386 s (1.50 h)** |
| MC/s | 5627 | **14747** (2.62×) |

All four port files (145 samples each) and both field dumps are bit-identical
to the flat run of two days earlier. That holds with the probe cadence (every
~109k timesteps), the speculative single-timestep submissions of the pipelined
probe loop, and the 64-timestep batch limit all in play.

**On the integrated GPUs the default stays flat, and should for two of the
three.** (Superseded the next day by §5.9, which measured what the speed did
not show and turned it on.) The same TL model, 5000 timesteps, forced schedules
(`OPENEMS_GPU_TEMPORAL_BLOCK=k`, tile from `OPENEMS_GPU_TEMPORAL_BLOCK_MB`;
the integrated default target is 16 MB), MC/s. Rows with all three devices
are two rounds; the UHD 770-only rows are one:

| Schedule (tile incl. coefficients) | UHD 770 (i9-13900K) | HD 530 (i7-6700K) | Vega 7 (Ryzen 5 5600G) |
|---|---|---|---|
| flat (the default) | 315 | 251 | 298 |
| k=2, 5 lines, 6 MB | 352–354 | 253 | 291 |
| k=4, 11 lines, 15 MB | 402 | 255 | 296 |
| k=6, 17 lines, 23 MB | 422 | | |
| k=8, 16 lines, 22 MB | 428 | | |
| k=8, 22 lines, 30 MB | 429–430 | 251 | 297 |
| k=11, 22 lines, 30 MB | **435 (1.38×)** | | |
| k=16, 34 lines, 47 MB | 423 | | |
| k=16, 45 lines, 62 MB | 412–414 | 251 | 298 |

The three devices repeat §5.4 and §5.5 more sharply:

- **UHD 770**: up to 1.38× with a 30 MB tile, just under the 36 MB L3 it
  shares with the CPU, and 1.28× at the 16 MB default target. That is more
  than the 1.08–1.17× of §5.4's uniform grids, probably because this operator
  is uncompressed: each cell streams 72 B instead of about 25, so more of the
  time is DRAM traffic that a tile can save.
- **HD 530**: within 2 % of flat at every depth. This bullet said it was
  limited by its 24 execution units, not by memory. §5.9 found two
  coincident ceilings instead, one of them the DRAM.
- **Vega 7**: up to 2 % slower than flat. It has no last-level cache to hold a
  tile.

Enabling it by default on integrated GPUs would need to tell a UHD 770 from
the other two, which Vulkan's device type does not. §5.9 enables it anyway:
on a device that cannot hold the tile, it costs about 1 %.

One HD 530 flat run failed after 2048 timesteps with `VK_ERROR_DEVICE_LOST`. The
kernel log shows `Resetting rcs0 for preemption time out`: a 64-timestep batch
takes 1.3 s on that card, and i915 resets a context that holds the engine
longer than its preemption timeout when something else wants it. The repeat
ran clean. A smaller `OPENEMS_GPU_MAX_TS_PER_SUBMIT` shortens the batches.

What these numbers are not:

- **Short runs.** Each is 20000 timesteps.
- **One quiet machine.** The i9-13900K runs a desktop. The XTX's server is
  shared, but the card was idle, and its rounds agree within 1 %.
- **One mesh.** The tile target and depth defaults are fitted to this model
  and the PEC variant on two cards.

Raw output: `python/Tests/results/gpu_sync_tblock_2026-10-19.txt`. The drivers
are `python/Tests/bench_gpu_sync.py`, whose `tbK[wW]` and `tbmbN` options set
the schedule (`tb0` is flat), and `python/Tests/gpu_tblock_identity.py`. The
variant model, the sweep scripts and the dump comparison are in the
`tests/gpuprof` kit.

### 5.9 Integrated GPUs, and a benchmark mode in place of a cache table (2026-10-20)

§5.8 left blocking off on integrated GPUs: 1.38× on the UHD 770, nothing on
the HD 530 and the Vega 7, and no way to tell them apart through Vulkan. Two
more measurements on the same TL model changed that: the DRAM traffic, from
the memory controllers, and the energy, from RAPL. Both come from `perf`
system-wide (`bench_gpu_sync.py --counters`), with an idle reading of the
same counters subtracted.

**DRAM traffic per cell update**, 5000 timesteps, two rounds, MC/s and bytes
per cell per timestep:

| Schedule (tile) | UHD 770 MC/s | B/cell | HD 530 MC/s | B/cell | Vega 7 MC/s | B/cell |
|---|---|---|---|---|---|---|
| flat | 316 | 123 | 251 | 109 | 298 | 141 |
| k=2, 5 lines (6 MB) | 354 | 73 | 253 | 70 | 291 | 144 |
| k=4, 11 lines (15 MB) | 403 | 39 | 255 | 75 | 296 | 142 |
| k=8, 22 lines (30 MB) | 429 | 24 | 251 | 100 | 297 | 141 |
| k=11, 22 lines (30 MB) | **434** | **20** | 251 | 100 | 298 | 141 |
| k=16, 45 lines (62 MB) | 412 | 46 | 251 | 106 | 298 | 141 |

The ideal flat sweep moves about 120 B per cell and timestep: per half-step
it reads and writes the 12 B of the field it updates, and reads the 12 B of
the other field and 24 B of coefficients. Per device:

- **UHD 770.** Flat it draws 39 GB/s, about 80 % of what the memory delivers,
  so it is DRAM-bound. Blocking cuts the traffic sixfold, roughly as
  120/k + 9 B, until the next ceiling at about 430 MC/s. A tile larger than
  the 36 MB L3 gives half of the saving back.
- **HD 530.** It is not compute-bound, as §5.5 and §5.8 had it: it runs at
  a few percent of its FLOP rate. It has two ceilings that happen to
  coincide. One is the DRAM: flat draws 27.5 GB/s, the same as a run with
  16-timestep batches. The other is on chip, at about 255 MC/s, probably the
  GPU's path into the LLC. Blocking cuts the traffic by a third with a tile
  that fits the 8 MB L3, and the second ceiling keeps the speed where it was.
  On this Skylake the memory controller counts the GPU's reads apart from the
  cores' (`gt_requests`), but not its writes, which reach DRAM as LLC
  write-backs.
- **Vega 7.** The traffic is the same on every schedule. One x-plane of this
  model is 1.5 MB, its L2 is 1 MB, and it cannot use the CPU's L3, so an
  x-slab tile can never stay resident. Only tiling in y and z could help it.

**Energy per cell update**, package plus DRAM where RAPL reports it,
5000 timesteps, three rounds, against flat:

| | flat | k=2 (6 MB) | k=4 (15 MB) | k=11, 22 lines (30 MB) |
|---|---|---|---|---|
| UHD 770 | 66.5 nJ | −10 % | −20 % | **−26 %** |
| HD 530 | 95.7 nJ | **−6 %** | −5 % | −1 % |
| Vega 7 | 92.7 nJ | +0.3 % | 0 % | +0.1 % |

These are gross values, busy power over speed. On srv002 one round's idle
readings ran 2–3 W high, which makes the idle-subtracted values unreliable,
and one flat run is left out: its package drew 1.5 W less than every other
run. With it, the HD 530 row reads −4 / −3 / +1 %.

- On the UHD 770 the package draws about 17 W on every schedule, so the
  saving is the speed.
- On the HD 530 speed and package power (19.6–19.9 W) do not move. The DRAM
  modules draw 3.1 W instead of 4.4 W when the tile fits the L3, which is the
  whole saving. The 30 MB tile does not fit and saves nothing.

All three are bit-identical to the flat sweep on every schedule: the built-in
cases at 4:12 to 24:48 and the TL model at 2:5 to 16:45, probes included.

**The default since 0107f8e.** An integrated GPU blocks too, with a tile of
80 % of one instance of the CPU's L3. The size comes from the same query the
AVX2 engine uses (§5.10), and is 16 MB where it cannot be read or the CPU's
last level is not an L3. Neither the size nor the sharing is something Vulkan reports:

- No cache size is exposed, in core 1.4 or in any vendor extension. OpenCL's
  "global memory cache size" is no substitute: on the RX 6800 it reports
  16 KiB, the per-unit L0.
- Whether an integrated GPU can use the CPU's L3 depends on the SoC. Intel
  parts up to Raptor Lake can; from Meteor Lake on the GPU sits off the CPU's
  ring. The Vega 7 cannot.

Turning it on regardless is safe because a tile the cache cannot hold costs
only the wedges. With the default, 5000 timesteps, two rounds:

| | flat MC/s | default | MC/s | B/cell | nJ/cell |
|---|---|---|---|---|---|
| UHD 770 | 315 | k=10, 20 lines (27 MB) | **431** (1.37×) | 123 → 23 | 54.8 → 39.1 (−29 %) |
| HD 530 | 251 | k=2, 4 lines (5 MB) | 252 | 110 → 72 | 81.8 → 76.9 (−6 %) |
| Vega 7 | 298 | k=4, 9 lines (12 MB) | 295 (−1.1 %) | 141 → 142 | 79.8 → 80.0 (+0.3 %) |

Here nJ/cell is idle-subtracted, package only on the UHD 770 and the Vega 7.
The default is bit-identical on all three, and small built-in grids stay flat
because they fall below the threshold.

**Command-line options.** The schedule no longer needs the environment:

- `--gpu-temporal-block=auto|off|<k>|<k>:<W>`
- `--gpu-temporal-block-mb=<MB>`
- `--avx2-temporal-block=off|<k>|<k>:<W>`

Each replaces the environment variable of the same name, with a note where
both are set. From Python they are `Run()` keywords:
`FDTD.Run(sim_path, gpu_temporal_block='11:22')`.

**`--benchmark`** answers the cache question per model instead of per device
class. The simulation is set up as usual. Then the engine's schedules run one
after another on the real model, excitation and probes included. Every
schedule computes the same fields, so the schedule simply changes between two
batches. The candidates are:

- the flat sweep;
- on the GPU, tile targets from 4 to 256 MB at k = W/2;
- on the AVX2 engine, 25 to 140 % of the L3;
- the configured schedule.

A probe interval below 32 timesteps caps their depth. A 2 s warm-up on the
flat sweep, which also covers the AVX2 engine's thread calibration, sets the
run length: about 3 s of the flat sweep, unless `--benchmark-timesteps` sets
it. Each candidate runs twice, the second round in reverse order. Nothing is
written into the simulation directory: field dumps are off, and the probes
write into a scratch directory that is removed afterwards. The output, on the
UHD 770:

```
  schedule         tile      MC/s   round 1   round 2    vs flat
  flat                 -     315.5     315.0     316.0          -
  k=2 W=5           7 MB     350.7     350.1     351.4    +11.2 %
  k=5 W=11         15 MB     404.5     403.8     405.2    +28.2 %
  k=10 W=20        28 MB     428.1     427.7     428.5    +35.7 %   as configured
  k=11 W=22        31 MB     430.9     431.5     430.3    +36.6 %
  k=22 W=45        63 MB     385.0     385.4     384.5    +22.0 %
  ...
  Fastest: k=11 W=22, 430.9 MC/s, 1.37x the flat sweep.
  As configured (k=10 W=20): 428.1 MC/s, 0.6 % below the recommendation.
  To run this model with the recommendation on this machine:
    openEMS <model.xml> --engine=gpu --gpu-temporal-block=11:22
    FDTD.Run(sim_path, engine='gpu', gpu_temporal_block='11:22')
```

On the TL model it takes 1–1.5 minutes. Its picks:

| Device | flat | Recommended | Default | Default behind by |
|---|---|---|---|---|
| UHD 770 | 315 | k=11 W=22, 431 (1.37×) | k=10 W=20 | 0.6–1.5 % |
| RX 6800 | 3450 | k=32 W=68, 7040 (2.04×) | k=22 W=45 | 0.5 % |
| Vega 7 | 298 | flat: every schedule within 0.1 % | k=4 W=9 | 1.0 % |
| HD 530 | 221* | k=4 W=8, 252 | k=2 W=4 | 8.9 %* |
| AVX2, i9-13900K | 414 | k=18 W=37, 1407 (3.39×) | flat (opt-in) | 70 % |

The two rounds agree within 0.5 % on the UHD 770 and the Vega 7 and within
2 % on the HD 530. On the RX 6800 they differ by up to 3.9 % and on the AVX2
engine by up to 7 %: that host is a desktop.

\* **The HD 530 pays for short batches between processing steps.** Its
benchmark runs are 140 timesteps long, and each ends the way a probe step
does. A normal flat run of this model with probes every 140 or 280 timesteps
manages 221 MC/s, against 249 every 1000 and 256 with this model's real
interval of about 110 000. The benchmark therefore measures that GPU as a
probe-dense model would run on it: flat 12 % low, shallow blocks less so.
Its pick is still the fastest in a normal run: k=4 W=8 at 256.5 MC/s,
against 252.1 for the default and 251.2 for flat. The GPU clock sits at its
maximum in both cases; why short batches cost this much is open. No other
device showed it. The benchmark's flat speeds on the UHD 770 (315) and the
RX 6800 (3450) match their normal runs (315, 3484).

Two rules in the recommendation:

- **The flat sweep, unless a blocked schedule beats it by more than 1 %.** On
  the Vega 7 the "fastest" was k=32 W=183, 0.1 MC/s ahead of flat, which is
  noise.
- **No tie-break toward small tiles.** The idea was to prefer, within 2 % of
  the fastest, the schedule with the smallest tile, for the DRAM energy a
  resident tile saves at equal speed. It was tried and dropped. While both
  tiles fit, the deeper block of the larger one moves less data, and on the
  UHD 770 the rule chose k=8 W=17, 1.6 % slower than the fastest. The HD 530
  case it was meant for needs no rule. There the fastest schedule, at k=4,
  saves nearly as much DRAM traffic as the smallest tile: 75 B per cell
  against 70 at k=2 and 109 flat.

What the benchmark does not see:

- **Energy.** RAPL is neither portable nor readable without privileges.
- **Thread count.** The AVX2 engine benchmarked at the thread count it
  calibrated on the flat sweep: 6 of 32 on the i9-13900K. Blocking removes the
  DRAM contention that limits that count (§5.7 ran 8 P-cores), so more threads
  may be faster blocked. Measured since §5.10.
- **Field dumps.** They are off. A dump interval shorter than the probe
  interval caps the depth in a real run, the same way probes do.
- **Other meshes.** The options it prints are in x-lines, for this model.

Raw output: `python/Tests/results/gpu_igpu_tblock_2026-10-20.txt`. The scripts
(`sweep_igpu_dram.sh`, `split_skl_imc.sh`, `run_igpu_checks.sh`,
`energy_summary.py`, `verify_igpu_default.sh`, `run_benchmark_mode.sh`) are in
the `tests/gpuprof` kit.

### 5.10 Threads and cores in the benchmark, on any operating system (2026-10-20)

§5.9's benchmark ran the AVX2 engine on the thread count the engine had
calibrated on the flat sweep, 6 of 32 on the i9-13900K. That count does not
suit a blocked run. The flat sweep is DRAM-bound: 8 P-cores reach 1.6× one
of them (432 against 273 MC/s), and the calibration lands anywhere from 6 to
8 threads. Blocked, the run scales 3.5× on the same 8 P-cores (§5.7). The
benchmark now measures the threads and the cores they run on, before the
schedules.

**Sets of cores.** `--cpu-cores` keeps the worker threads to one of them:

- `all`, the default;
- `physical`, one thread per core;
- `performance` or `efficiency`, one kind of core of a hybrid CPU, one thread
  per core; `performance+smt` and `efficiency+smt` use every thread of them.

They are named by kind, not by CPU number, so a value means the same on
another machine and on every operating system. On a CPU with one kind,
`performance` is every core.

**What it measures**, for an engine with worker threads:

1. **Threads.** For each set that differs on the machine, the thread count
   doubles from 1 until a doubling gains no more than 1 %. One count halfway
   to each neighbour of the best refines it. A set that contains a measured
   one starts above that one's best: the P-cores with SMT start above the
   P-cores' best. Every point runs the engine's default tile, as deep as the
   probes allow, because blocking is what lets the threads scale. After each
   change of threads 0.3 s is discarded (new threads, cold caches), then
   about 2 s are measured. On a hybrid CPU, sets that mix kinds are skipped,
   except `all`, which is what a run without `--cpu-cores` gets.
2. **Schedules,** on the best threads and cores: the tile ladder, the flat
   sweep, the configured schedule, and the runner-up of step 1. Each runs
   about 2 s of the fastest setting, twice, the second round in reverse
   order.

The recommendation extends §5.9's rule. Anything more, a blocked schedule or
more threads, has to be more than 1 % faster than what it replaces.

On the TL model, at the default barrier spin of 0:

| Host | Sets | Best of each set (MC/s) | Recommended | MC/s | vs flat | Time |
|---|---|---|---|---|---|---|
| i9-13900K | P, P+SMT, E, all | P 8: 1695, P+SMT 16: 1544, E 16: 1037, all 12: 1338 | 8 performance, 18:37 | **1706** | 4.06× | 95 s |
| i7-6700K | physical, all | physical 4: 410, all 6: 368 | 4 physical, 4:8 | **405** | 1.58× | 59 s |
| Ryzen 5 5600G | physical, all | physical 6: 760, all 9: 651 | 6 physical, 11:23 | **830** | 2.69× | 65 s |

"vs flat" is the flat sweep on the same threads; "Time" is the benchmark
without the setup. srv002's rounds differ by up to 12 %: it serves media in
the background.

- **SMT makes the blocked run slower** on all three at spin 0. At §5.7's
  100 µs spin the 13900K's best was 8 P-cores with HT, 1857 MC/s. The
  benchmark measures the spin the run uses, so settling D3 would move the
  pick.
- **E-cores alone** reach 1037 MC/s on 16 threads, 61 % of the 8 P-cores.
- **A normal run with the recommendation** on the 13900K: 1657 MC/s over
  2000 timesteps, against 404 with no options (flat, threads calibrated over
  all 32), so **4.1×**. With only `--avx2-temporal-block=16` it is 1420.

**The startup calibration runs on the blocked schedule again.** aae6b2e made
the thread calibration measure the schedule the run uses. 4bfcd74 then moved
the tile setup behind the calibration, because the tile depends on the
threads, and that quietly put the calibration back on the flat sweep. Now
every candidate count gets its own tile, and a calibration batch is at least
one block deep. With `--avx2-temporal-block=16` on the 13900K it picked 8
threads in both runs.

**The `multithreaded` engine,** the default where the AVX2 engine is not
built (ARM64, for one), has worker threads but no blocking. Its benchmark
measures threads and cores on the flat sweep. On the 13900K: 6 P-cores,
410 MC/s, with 8 within 0.4 %.

**A warning when the settings are defaults.** A run that leaves to its
defaults what `--benchmark` would set prints:

```
openEMS - Warning: running on default performance settings, not on ones tuned for this model and machine (--numThreads, --cpu-cores, --avx2-temporal-block not set).
	More speed may be possible: --benchmark measures the settings on this model and prints the options to use.
```

It lists only what applies. `--cpu-cores` appears only where the machine
offers more than one set, a schedule option only for an engine with
blocking. An option set to its default (`--numThreads=0`, `--cpu-cores=all`,
`--avx2-temporal-block=off`) counts as set, and the environment variables
count too.

**The CPU topology, without hwloc** (`tools/cpu_topology`). hwloc reports the
kind of core only from 2.4 on, and the portable bundle builds against 1.11.
On Linux it reads the same files as the code below, and it was optional, so
builds without it had only `sysconf`'s one L3, which only glibc on x86 knows.
It is replaced by the operating systems' own interfaces:

- **Linux**, x86 and ARM alike, from sysfs:
  - cores from `core_cpus_list`;
  - kinds from the Intel hybrid PMUs (`cpu_core`, `cpu_atom`), else
    `cpu_capacity` (ARM), else the highest clock, the first that splits the
    CPUs. A kind starts where a CPU is more than 15 % slower than the next
    faster one, so the 13900K's two favoured P-cores (5.8 GHz instead of 5.5)
    stay P-cores;
  - the last-level cache from the highest data or unified level: an L3 on
    x86, often an L2 on ARM;
  - the process's affinity.

  Threads are restricted with `pthread_setaffinity_np`.
- **Windows:** `GetLogicalProcessorInformationEx` gives cores, the efficiency
  class and caches; `SetThreadGroupAffinity` restricts threads, spreading the
  workers over the processor groups of a machine with more than 64 CPUs.
- **macOS:** `sysctl` `hw.perflevelN.*` gives kinds, cores and caches. There
  is no affinity. A thread's QoS class steers it instead: background confines
  it to the efficiency cores, user-interactive prefers the performance cores.
  So only `performance`, `efficiency` and `all` are offered there.
- **Elsewhere:** the thread count only, and `all`.

The integrated-GPU tile of §5.9 now takes only an L3. Where the CPU's last
level is an L2 (Apple, most ARM SoCs) that cache belongs to a cluster of
cores, not to the GPU, so the tile is 16 MB.

Tested:

- the 13900K itself, also under `taskset`;
- seven fake sysfs trees: the 13900K with and without its PMU directories, an
  RK3588, an ARM with three kinds, a 7950X3D (96 and 32 MB L3), a Ryzen AI 9
  HX 370 (Zen 5 + Zen 5c) and an ARM server without cache information;
- the Windows build (MinGW-w64) under Wine, which reports the 13900K's
  efficiency classes and honours the group affinity;
- the ARM64 Linux build under QEMU.

macOS is untested: there is no SDK here.

**Bit-identical**, with full E/H dumps against the previous build's flat run:
8, 12 and 16 threads, calibrated counts, and the `performance`,
`performance+smt`, `efficiency` and `physical` sets, flat and blocked. The
`multithreaded` engine was checked on `efficiency` and `performance+smt`.

**An ARM64 engine** needs nothing in the benchmark or the topology, which have
no x86 code. An engine with worker threads implements
`Engine::GetNumThreads()`, `GetCoreSet()` and `SetThreads()`, takes its
cores from `Operator::GetCoreSet()`, and calls `CpuTopology::Restrict()` at
the start of each worker. With temporal blocking it also implements the four
blocking methods. `--benchmark`, `--cpu-cores` and the warning then work
for it unchanged.

Limits:

- **Kinds are not mixed.** Every worker gets an equal share of the grid, so a
  set of P- and E-cores runs at the pace of its E-cores (§5.7: 35 % of the
  time waiting). Weighted shares, D2, would change that.
- **macOS's QoS only steers.** `performance` there is a preference.
- **No set per cache.** A 7950X3D's CCD with the large L3 is not offered on
  its own.
- **AMD Zen 5 + Zen 5c on Linux** falls back to the clock to find the kinds.
  That is tested on a fake tree only.
- **Windows' process affinity** is honoured only on machines with one
  processor group.

Raw output: `python/Tests/results/cpu_threads_cores_2026-10-20.txt`. The test
programs are in `tests/topology` and `tests/gpuprof` (`identity_threads.py`,
`verify_cpu_settings.sh`).

---

## 6. The AVX2 engine has no runtime ISA guard — it does not fall back, it crashes

The 2× Xeon E5-2630 is a Sandy Bridge: SSE4.2 and AVX, no AVX2 and no FMA. On
it, **every** fork configuration in the matrix terminates with SIGILL — exit
status 132 — including `--engine=fastest`, which is what a user gets by typing
nothing:

```
$ openEMS model.xml
...
Create FDTD operator (AVX2 + FMA + multi-threading)
Illegal instruction (core dumped)        # exit 132
```

It does not even reach the stepping loop: the default path dies in operator
construction. Asked for `--engine=avx2` explicitly it gets one line further,
printing `Create FDTD engine (AVX2 + FMA, compressed flat arrays)` and
`Running FDTD engine...` before the same signal.

The guard that exists (`987fcb7`) is `OPENEMS_ENABLE_AVX2`, set by CMake from
`CMAKE_SYSTEM_PROCESSOR` and a `check_cxx_compiler_flag("-mavx2 -mfma")` probe.
Both questions are about the **build** machine and its compiler, not about the
machine that will run the binary. So the guard does what it was written for — it
keeps `<immintrin.h>` out of an ARM build — and does nothing at all for the case
that actually bites: an x86-64 binary built anywhere with a modern GCC and run on
a pre-Haswell host. `openems.cpp` then makes AVX2-multithreaded the default
engine unconditionally, and the `--engine=avx2` fallback message (`"AVX2+FMA
engine is unavailable on this platform; using multithreaded engine"`) is inside
`#if OPENEMS_ENABLE_AVX2 … #else`, so on this binary it is compiled out.

This matters beyond one old server: it is every distribution package, every
container image and every shared cluster filesystem — anywhere the build host and
the run host are not the same machine.

**The capability is not missing, only the dispatch.** Asking the 2× Xeon E5-2630
for the portable engine explicitly, on both builds, at 12 threads:

| 224³ model | upstream `multithreaded` | fork `multithreaded` | |
|---|---|---|---|
| PEC | 556 | 590 | 1.06× |
| PML_8 | 228 | 226 | 0.99× |
| Mur | 381 | 401 | 1.05× |

So the fork is a small gain on this host, from the host-side work in
[OPTIMIZATIONS.md](OPTIMIZATIONS.md) §1.1 and the thread auto-tune — it just
cannot be reached without `--engine=multithreaded` on the command line. The fix
is a runtime check (`__builtin_cpu_supports("avx2") && …("fma")`) at engine
selection, falling back the way the compile-time path already does. That check
now exists in `openEMS::SetupOperator()` (`OPENEMS_FORCE_NO_AVX2=1` simulates a
pre-AVX2 CPU); the measurements above predate it and have not been re-run on
this host. Raw output:
`python/Tests/results/e5-2630_non_avx2_2026-09-11.txt`.

**Practical guidance until then:** on any pre-Haswell x86 host, pass
`--engine=multithreaded` explicitly, or build on that host so CMake's probe
matches it.

---

## 7. Operator build time

Time-stepping throughput is not the whole cost. Operator construction, measured
as wall clock outside the stepping loop:

| Model | | upstream SSE-MT | fork AVX2-MT | fork Vulkan |
|---|---|---|---|---|
| 224³ PEC | i9-13900K | 6.0 s | 5.1 s | 7.0 s |
| 224³ PML_8 | i9-13900K | 8.2 s | 5.6 s | 6.8 s |
| 224³ Mur | i9-13900K | 6.0 s | 5.1 s | 7.0 s |
| 224³ PEC | i7-6700K | 13.9 s | 12.9 s | 12.2 s |
| 224³ PML_8 | i7-6700K | 17.3 s | 14.3 s | 13.1 s |
| 224³ Mur | i7-6700K | 13.9 s | 12.6 s | 12.1 s |
| 224³ PEC | Ryzen 5 5600G | 9.8 s | 9.2 s | 10.0 s |
| 224³ PML_8 | Ryzen 5 5600G | 12.9 s | 10.1 s | 10.4 s |
| 224³ Mur | Ryzen 5 5600G | 9.9 s | 9.2 s | 10.1 s |

Operator setup is parallelized in this fork, which is where the PML case's
8.2 s → 5.6 s comes from; on the i7-6700K's four cores the same change is worth
17.3 s → 14.3 s and on the Ryzen 5 5600G's six 12.9 s → 10.1 s. The saving is
confined to the PML case on every host — the PEC and Mur rows move only 3–7% —
because the UPML coefficient construction is the part that was serialized.

For the GPU, setup is the dominant fixed cost of a short run: on the i9-13900K
at 224³ the operator takes ~7.0 s while all 1200 timesteps take only ~1.1 s with
blocking on. Since the CPU pays ~6.0 s of setup too, the crossover is early —
the blocked GPU is ahead of upstream's best CPU configuration after about **55
timesteps**, and ahead of the fork's own blocked CPU engine after about **530**.

The integrated case inverts that arithmetic. On the Ryzen 5 5600G the Vega 7
costs about the same setup as the CPU (10.0 s against 9.2 s) but steps 224³ PEC
at 485 MC/s against the blocked CPU engine's 1062, so on that model it never
catches up — there is no crossover at any run length. On 224³ PML_8, where the
GPU is the faster engine, the 0.3 s of extra setup is repaid within about 60
timesteps. Setup cost is only decisive when the device wins the steady state.

---

## 8. Full results

### 8.1 i9-13900K — RX 6800, UHD 770

| Model | TS | upstream<br>SSE 1t | upstream<br>SSE-MT 8t | upstream<br>default | fork<br>AVX2 1t | fork<br>AVX2-MT 8t | fork<br>default | fork<br>+tblock 8t | fork<br>+tblock default | GPU<br>RX 6800 | GPU RX 6800<br>+tblock | GPU<br>UHD 770 | GPU UHD 770<br>+tblock |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64³ PEC | 40000 | 476 | 2039 | 956 | 972 | 3507 | 2928 | 3458 | 2846 | 12433 | 12385 | 633 | 633 |
| 160×128×192 PEC | 3000 | 267 | 789 | 569 | 494 | 889 | 865 | 3249 | 2723 | 13070 | 12950 | 524 | 611 |
| 224³ PEC | 1200 | 247 | 586 | 496 | 449 | 632 | 612 | 2607 | 1960 | 5674 | 12319 | 520 | 601 |
| 160×128×192 PML_8 | 1500 | 111 | 280 | 207 | 148 | 266 | 256 | 800 | 685 | 9462 | 9844 | 317 | 344 |
| 224³ PML_8 | 600 | 112 | 262 | 180 | 147 | 237 | 224 | 828 | 640 | 4619 | 10212 | 323 | 347 |
| 224³ Mur | 1000 | 219 | 531 | 431 | 364 | 569 | 543 | 1848 | 1466 | 5125 | 5125 | 518 | 518 |

### 8.2 i7-6700K — HD 530

| Model | TS | upstream<br>SSE 1t | upstream<br>SSE-MT 4t | upstream<br>default | fork<br>AVX2 1t | fork<br>AVX2-MT 4t | fork<br>default | fork<br>+tblock 4t | fork<br>+tblock default | GPU<br>HD 530 | GPU HD 530<br>+tblock |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64³ PEC | 40000 | 198 | 512 | 544 | 513 | 937 | 1335 | 938 | 1322 | 505 | 504 |
| 160×128×192 PEC | 3000 | 199 | 448 | 393 | 374 | 497 | 492 | 706 | 862 | 368 | 381 |
| 224³ PEC | 1200 | 200 | 423 | 387 | 376 | 475 | 489 | 497 | 577 | 478 | 484 |
| 160×128×192 PML_8 | 1500 | 70 | 146 | 141 | 124 | 164 | 169 | 221 | 251 | 225 | 224 |
| 224³ PML_8 | 600 | 83 | 168 | 133 | 137 | 183 | 186 | 205 | 227 | 270 | 269 |
| 224³ Mur | 1000 | 173 | 358 | 321 | 286 | 388 | 406 | 339 | 382 | 503 | 502 |

### 8.3 The five servers — CPU only

Same matrix, same models, same binaries; no GPU columns because none of these
machines has a Vulkan driver. The pinned thread count per machine is the one
§2.4 derives.

**Ryzen 9 3950X, 16C/32T, 4×16 MB L3**

| Model | TS | upstream<br>SSE 1t | upstream<br>SSE-MT 16t | upstream<br>default | fork<br>AVX2 1t | fork<br>AVX2-MT 16t | fork<br>default | fork<br>+tblock 16t | fork<br>+tblock default |
|---|---|---|---|---|---|---|---|---|---|
| 64³ PEC | 40000 | 273 | 586 | 631 | 588 | 990 | 1588 | 964 | 1587 |
| 160×128×192 PEC | 3000 | 207 | 430 | 403 | 338 | 473 | 616 | 705 | 1153 |
| 224³ PEC | 1200 | 212 | 422 | 390 | 348 | 440 | 460 | 513 | 661 |
| 160×128×192 PML_8 | 1500 | 73 | 184 | 153 | 96 | 175 | 177 | 292 | 397 |
| 224³ PML_8 | 600 | 82 | 201 | 128 | 109 | 174 | 171 | 270 | 337 |
| 224³ Mur | 1000 | 193 | 376 | 335 | 289 | 388 | 400 | 319 | 446 |

**2× Xeon Gold 6130, 32C/64T, 2×22 MB L3**

| Model | TS | upstream<br>SSE 1t | upstream<br>SSE-MT 32t | upstream<br>default | fork<br>AVX2 1t | fork<br>AVX2-MT 32t | fork<br>default | fork<br>+tblock 32t | fork<br>+tblock default |
|---|---|---|---|---|---|---|---|---|---|
| 64³ PEC | 40000 | 181 | 184 | 488 | 335 | 370 | 1002 | 374 | 1148 |
| 160×128×192 PEC | 3000 | 128 | 1057 | 473 | 221 | 1426 | 1242 | 909 | 1324 |
| 224³ PEC | 1200 | 128 | 1083 | 371 | 200 | 1282 | 992 | 637 | 807 |
| 160×128×192 PML_8 | 1500 | 49 | 275 | 139 | 74 | 364 | 344 | 190 | 294 |
| 224³ PML_8 | 600 | 55 | 396 | 100 | 77 | 432 | 309 | 160 | 261 |
| 224³ Mur | 1000 | 108 | 608 | 271 | 153 | 835 | 696 | 221 | 455 |

**2× Xeon E5-2630, 12C/24T, no AVX2**

| Model | TS | upstream<br>SSE 1t | upstream<br>SSE-MT 12t | upstream<br>default | every fork AVX2 row |
|---|---|---|---|---|---|
| 64³ PEC | 40000 | 100 | 192 | 222 | **SIGILL** |
| 160×128×192 PEC | 3000 | 99 | 549 | 283 | **SIGILL** |
| 224³ PEC | 1200 | 98 | 566 | 238 | **SIGILL** |
| 160×128×192 PML_8 | 1500 | 33 | 191 | 75 | **SIGILL** |
| 224³ PML_8 | 600 | 37 | 231 | 59 | **SIGILL** |
| 224³ Mur | 1000 | 77 | 407 | 157 | **SIGILL** |

All 30 fork AVX2 rows on this host exit with status 132 — 128 + SIGILL —
including the two `default` ones. §6. What the fork's *portable* engine does
there, which is the question the crash hides, is in §6's second table.

**Core i5-8500, 6C/6T, 9 MB L3**

| Model | TS | upstream<br>SSE 1t | upstream<br>SSE-MT 2t | upstream<br>default | fork<br>AVX2 1t | fork<br>AVX2-MT 2t | fork<br>default | fork<br>+tblock 2t | fork<br>+tblock default |
|---|---|---|---|---|---|---|---|---|---|
| 64³ PEC | 40000 | 166 | 430 | 530 | 436 | 772 | 1144 | 782 | 1128 |
| 160×128×192 PEC | 3000 | 124 | 210 | 206 | 226 | 230 | 230 | 474 | 531 |
| 224³ PEC | 1200 | 123 | 208 | 201 | 221 | 225 | 225 | 303 | 332 |
| 160×128×192 PML_8 | 1500 | 53 | 78 | 78 | 71 | 74 | 75 | 140 | 144 |
| 224³ PML_8 | 600 | 59 | 88 | 83 | 79 | 83 | 84 | 142 | 141 |
| 224³ Mur | 1000 | 110 | 182 | 175 | 179 | 192 | 190 | 265 | 259 |

**2× Xeon E5-2620 v3, 12C/24T, 2×15 MB L3**

| Model | TS | upstream<br>SSE 1t | upstream<br>SSE-MT 12t | upstream<br>default | fork<br>AVX2 1t | fork<br>AVX2-MT 12t | fork<br>default | fork<br>+tblock 12t | fork<br>+tblock default |
|---|---|---|---|---|---|---|---|---|---|
| 64³ PEC | 40000 | 143 | 238 | 175 | 338 | 449 | 364 | 446 | 358 |
| 160×128×192 PEC | 3000 | 114 | 430 | 244 | 194 | 519 | 440 | 583 | 461 |
| 224³ PEC | 1200 | 114 | 447 | 201 | 187 | 514 | 422 | 365 | 320 |
| 160×128×192 PML_8 | 1500 | 41 | 145 | 61 | 62 | 168 | 150 | 148 | 133 |
| 224³ PML_8 | 600 | 46 | 175 | 67 | 66 | 190 | 156 | 118 | 112 |
| 224³ Mur | 1000 | 87 | 349 | 136 | 125 | 383 | 332 | 204 | 219 |

Two rows in there are worth pausing on because they are not engine results:

- **The 2× Xeon Gold 6130's 64³ row collapses when pinned** — 184 MC/s at 32
  threads against 488 for the same build left to auto-tune. A 6 MB grid spread
  over 32 cores on two sockets is barrier traffic and cross-socket coherence,
  nothing else; both builds suffer it equally, so the column is still a
  controlled comparison, but the absolute number says more about the pinning
  than the engine. The `default` columns are the ones to read on that row.
- **The 2× Xeon E5-2620 v3's and 2× Xeon Gold 6130's `+tblock` columns are
  below their flat ones** on the large grids — 365 against 514, 637 against
  1282. That is the §5.3 defect, not a property of the schedule; corrected, the
  same configurations reach 569 and 1280.

### 8.4 Ryzen 5 5600G — Vega 7

Pinned at 4 threads on cores 0–3 (§2.4). The CPU columns are from the CPU
bundle; the GPU columns are from the GPU bundle, whose own CPU rows reproduce
these to 2.99% (§2.2).

| Model | TS | upstream<br>SSE 1t | upstream<br>SSE-MT 4t | upstream<br>default | fork<br>AVX2 1t | fork<br>AVX2-MT 4t | fork<br>default | fork<br>+tblock 4t | fork<br>+tblock default | GPU<br>Vega 7 | GPU Vega 7<br>+tblock |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64³ PEC | 40000 | 290 | 997 | 660 | 628 | 1708 | 1957 | 1711 | 1930 | 587 | 587 |
| 160×128×192 PEC | 3000 | 275 | 433 | 403 | 415 | 443 | 482 | 1518 | 1640 | 557 | 554 |
| 224³ PEC | 1200 | 271 | 429 | 400 | 419 | 432 | 469 | 1062 | 1047 | 483 | 485 |
| 160×128×192 PML_8 | 1500 | 66 | 145 | 123 | 105 | 135 | 135 | 340 | 354 | 413 | 412 |
| 224³ PML_8 | 600 | 79 | 164 | 115 | 111 | 152 | 149 | 354 | 344 | 402 | 402 |
| 224³ Mur | 1000 | 238 | 387 | 350 | 339 | 384 | 405 | 771 | 754 | 444 | 444 |

Three things in that table are worth naming:

- **The `+tblock` columns beat every other CPU column on the five non-resident
  grids**, by 2.0×–3.4× over the flat AVX2 engine beside them, with no
  regression anywhere — the only host in this document of which that is true.
- **The two PML rows are the only ones where the GPU column wins**, and it wins
  against the fork's own best CPU configuration, not merely against upstream
  (§4.2).
- **The `default` columns are at or above the pinned ones** on most rows,
  because 4 threads is the conservative pinned choice for this host (§2.4). The
  64³ row's 1957 vs 1708 is the largest such gap and is the auto-tune finding
  more threads for a cache-resident grid, not an engine difference.

Raw output for all eight machines, including every individual repeat and the
engine's own "temporal blocking active / disabled — …" line for each blocked
row, is committed under `python/Tests/results/`:
`fork_vs_upstream_i9-13900k_2026-09-07.json`,
`fork_vs_upstream_i7-6700k_2026-09-07.json`,
`fork_vs_upstream_i7-6700k_2026-09-08.json`, the five
`fork_vs_upstream_{3950x,gold6130,e5-2630,i5-8500,e5-2620v3}_2026-09-11.json`,
and for the Ryzen 5 5600G both
`fork_vs_upstream_ryzen5-5600g_2026-09-13.json` (CPU bundle) and
`fork_vs_upstream_gpu_ryzen5-5600g_2026-09-13.json` (GPU bundle, which also
carries the duplicate CPU rows behind the §2.2 cross-check). The thread-count
sweeps behind §2.4 are beside them as `calibration_<machine>_2026-09-11.txt`
and `calibration_ryzen5-5600g_2026-09-13.txt`.

---

## 9. Reproducing

```bash
# 1. build upstream from scratch, out of tree
git worktree add --detach ../openEMS-upstream-bench upstream/master
cmake -S ../openEMS-upstream-bench -B ../openEMS-upstream-bench/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCSXCAD_ROOT_DIR=$HOME/opt -DFPARSER_ROOT_DIR=$HOME/opt
make -C ../openEMS-upstream-bench/build -j$(nproc)

# 2. build this fork from scratch, with the GPU engine
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DWITH_GPU=ON \
      -DCSXCAD_ROOT_DIR=$HOME/opt -DFPARSER_ROOT_DIR=$HOME/opt
make -C build-bench -j$(nproc)

# 3. run the matrix (~2 h) -- i9-13900K
python python/Tests/benchmark_fork_vs_upstream.py --out /tmp/bench.json

# 3'. the same matrix on a different machine: everything host-specific is a flag
python python/Tests/benchmark_fork_vs_upstream.py \
    --tuned-threads 4 --pin-single 0 --pin-tuned 0,2,4,6 \
    --gpu hd530:0:6 --csxcad-lib ~/openEMS/build-deps/install/lib \
    --out /tmp/bench-i7-6700k.json
```

`--gpu LABEL:INDEX[:K]` is repeatable and adds both a flat and a `+tblock` row
for each device; `--gpu none` gives a CPU-only matrix. The script writes
incrementally and skips configurations already present in the output file, so it
can be interrupted and resumed. `--only <substring>` runs a single cell of the
matrix.

### 9.1 On a machine you cannot build on

The five servers had no openEMS dependencies and could not have any installed,
and one of them is old enough (CentOS 7, glibc 2.17) that a binary from a current
distribution will not start. Both builds therefore come out of one container and
travel as a self-contained bundle:

```bash
# 1. build fork + upstream + every dependency inside a glibc-2.17 container.
#    ~40 min from cold; stages are stamped, so a rerun resumes.
python/Tests/build_portable_bench.sh            # -> ~/openems-portable/bundle

# 2. copy the bundle to the target and run it there. Nothing is installed;
#    the bundle only needs a glibc >= 2.17 x86-64 host.
rsync -a ~/openems-portable/bundle/ target:openems-bench/
ssh target 'cp -a ~/openems-bench /tmp/oebench && cd /tmp/oebench &&
            ./run_portable_bench.sh calib 4:4:0-3 8:8:0-7 12:12:0-11 &&
            ./run_portable_bench.sh matrix 12 0-11 0'
```

`run_portable_bench.sh calib <label>:<threads>:<cpulist> …` is the thread sweep
behind §2.4 and prints upstream-MT, fork-AVX2 and fork-+tblock at each point;
`matrix <threads> <pin_tuned> [pin_single]` then runs the full matrix with
`--gpu none` and writes `results-<hostname>.json`. Staging to node-local disk
matters if `$HOME` is on NFS, as it was here.

**If the target has a Vulkan driver**, as the Ryzen 5 5600G does, a second
bundle adds the GPU rows. It is a separate pair of scripts on purpose: the CPU
pair above produced every CPU number in this document and is frozen.

```bash
# 1. reuses the CPU build's dependency stack and its upstream binary; adds
#    Vulkan-Headers, the loader and shaderc's glslc, and builds the fork
#    with -DWITH_GPU=ON.
python/Tests/build_portable_bench_gpu.sh        # -> ~/openems-portable/bundle-gpu

# 2. probe first, then pick the block depth, then run the matrix.
tar cf - -C ~/openems-portable bundle-gpu | ssh target 'tar xf - -C openems-bench'
ssh target 'cd openems-bench/bundle-gpu &&
            ./run_portable_bench_gpu.sh probe &&
            ./run_portable_bench_gpu.sh ksweep vega7:0 0 4 8 16 32 &&
            ./run_portable_bench_gpu.sh matrix vega7:0:6 4 0-3 0'
```

`probe` lists the ICDs, the loader and the devices the engine enumerates;
`ksweep` is the GPU depth sweep behind §5.5, with `k=0` as the flat control;
`matrix` writes `results-gpu-<hostname>.json`, a different filename from the CPU
runner's so a GPU run cannot overwrite a CPU record. Two things differ from the
CPU bundle: it is **not** self-contained — it carries no `libvulkan.so.1`,
because the loader has to match the target's driver, so the target needs a
system loader and an ICD — and its matrix re-runs the CPU rows as well, since
`benchmark_fork_vs_upstream.py` has no GPU-only mode. That redundancy is what
supplies the §2.2 cross-check, at the cost of most of the run time.

Note also that the run must survive a dropped connection: the driver dies on
SIGHUP while the `openEMS` child it launched keeps going, which produces a
truncated result file that looks complete. Launch it detached
(`setsid nohup … &`) and poll.

The bundle is deliberately boring to verify: `ldd` on either binary should show
only `libc`, `libm`, `libdl`, `libpthread` and the three project libraries, and

```bash
objdump -T bundle/fork/openEMS | grep -o 'GLIBC_[0-9.]*' | sort -V | tail -1
```

should print no more than the oldest target's glibc. Everything else — VTK,
HDF5, CGAL, Boost, TinyXML, fparser, GMP, MPFR, libstdc++ — is linked in
statically. GMP is built `--disable-assembly` on purpose: its configure script
otherwise selects assembly for the *build* host's microarchitecture, which is
exactly the portability bug this bundle exists to avoid.

**The toolchain differs from the i9-13900K and i7-6700K** (GCC 10.2.1 against
16.2.1), because glibc 2.17 constrains it. §2.2 shows that costs 0.98–1.04× in a
paired test, with no systematic direction. If you repeat that control, pair the
arms run by run — two matrix runs on different days measure the machine's
afternoon, not the compiler.

The raw output behind every table in this document is committed as
`python/Tests/results/fork_vs_upstream_i9-13900k_2026-09-07.json` and
`fork_vs_upstream_i7-6700k_2026-09-08.json`, including the individual repeats
and each blocked row's schedule announcement, so the best-of-N choices and the
engage/refuse decisions can be checked rather than taken on trust. The
i7-6700K's first, independently built run is kept beside it as
`fork_vs_upstream_i7-6700k_2026-09-07.json` — the two are a full-matrix
reproducibility check, not a duplicate. The previous single-host revision is
preserved as `fork_vs_upstream_2026-09-05.json`.

The five servers are
`fork_vs_upstream_{3950x,gold6130,e5-2630,i5-8500,e5-2620v3}_2026-09-11.json`,
with `calibration_<machine>_2026-09-11.txt` for the thread sweeps,
`tile_width_sweep_2026-09-11.txt` for §5.3, `e5-2630_non_avx2_2026-09-11.txt`
for §6 and `toolchain_gcc16_vs_gcc10_2026-09-11.txt` for the toolchain control.
The Ryzen 5 5600G is `fork_vs_upstream_ryzen5-5600g_2026-09-13.json`,
`fork_vs_upstream_gpu_ryzen5-5600g_2026-09-13.json` and
`calibration_ryzen5-5600g_2026-09-13.txt`. The models themselves are committed
as `python/Tests/models/*.xml`, so all eight machines demonstrably ran the same
bytes.

Related drivers: `python/Tests/benchmark_avx2_engine.py`,
`python/Tests/benchmark_gpu_engine.py` (`--gpu-index` selects the device),
`python/Tests/bench_temporal_blocking.c`. Correctness:
`python/Tests/test_temporal_blocking.py` and
`python/Tests/test_gpu_temporal_blocking.py`.

---

## 10. Measurement notes

Three hazards produced wrong numbers before they were understood, and all three
are worth repeating for anyone re-running this.

**Cache-resident grids are bimodal, for a reason not identified.** The 64³ model
returns either ~480 or ~150–310 MC/s from run to run — a 3× swing on identical
input. What it is *not*, each ruled out by direct measurement rather than by
argument:

| candidate | test | result |
|---|---|---|
| competing load | sample `ps`/loadavg through the run | machine otherwise idle |
| P-core vs E-core placement | `taskset -c 0` | still bimodal |
| ASLR-driven cache aliasing | `setarch -R` | still bimodal |
| CPU frequency / thermal throttle | sample `cpu MHz` through the run | constant 5.3 GHz in fast *and* slow runs |
| hugepage backing | `smaps_rollup` per run | identical `AnonHugePages` and RSS |

The leading remaining candidate is the **uncore/ring clock**, which ranges
800 MHz–5 GHz on the i9-13900K and would affect only L3-resident work — but this
is a hypothesis, not a finding: `current_freq_khz` is not readable unprivileged
here, so it was never confirmed, and the effect shows up at 8 threads as well as
at 1, which a simple "one core does not raise the uncore" story does not explain.
It reproduces on the i7-6700K's 64³ rows as well (198 vs 162 MC/s single-threaded
in the run tabulated here, and 199 vs 146 in the repeat), on a completely
different microarchitecture, which weakens any explanation specific to Raptor
Lake.

The five servers weaken it further: the same bimodality appears on the 64³
single-threaded rows of the **2× Xeon E5-2630** (100 / 100 / 100 / 78.5 / 99.6 /
81.7 MC/s), the **2× Xeon E5-2620 v3** (142 / 142 / 143 / 106 / 143 / 97.7) and
the **Core i5-8500** (128 / 166 / 160 / 166 / 160 / 142), and on the Core
i5-8500's `upstream default` row (266 among five repeats at 521–530). That is now
five microarchitectures — Sandy Bridge, Haswell, Skylake, Coffee Lake, Raptor
Lake — and it never appears on a DRAM-bound row on any of them. Whatever it is,
it is a property of cache-resident FDTD work on x86 generally and not of any one
part, which makes the uncore-clock hypothesis more plausible without confirming
it. The two AMD parts are the exceptions, and now they make a pattern: the Ryzen
9 3950X's 64³ repeats are tight throughout, and so are the Ryzen 5 5600G's —
every engine on that row repeats within 2–5%, the widest being upstream's
auto-tuned default at 1852–1957 MC/s. Five Intel microarchitectures show the
bimodality and neither Zen part does, which is the strongest evidence so far
that whatever it is, it is Intel-specific.

**A second, unrelated source of spread on the 64³ row** shows up on the
`default` columns of the dual-socket hosts, and should not be mistaken for the
above: upstream's thread auto-tune lands somewhere different on each run. The 2×
Xeon Gold 6130's six repeats of `upstream default` are 225 / 488 / 483 / 234 /
172 / 176 MC/s and the 2× Xeon E5-2630's are 204 / 222 / 156 / 124 / 112 / 66 — a
2.2× and 3.4× spread, on the pinned-thread-count *sibling* of a row that is
stable to a percent. That is a search converging to different thread counts on a
cache-resident grid where the right count is small and the penalty for
overshooting is large, and it is the same weakness §3.3 describes; it is visible
here rather than there because the 64³ grid punishes it hardest. The fork's
search is not immune on these hosts either.

**Repeat spread is not confined to the 64³ grid.** Every figure in this document
is a best-of-N. On the two original machines the DRAM-bound rows repeat to
0.5–4%, but the committed JSON for the five servers shows that does not
generalize: a number of large-grid configurations spread more than 4% between
their three repeats, and the worst are much worse — the 2× Xeon E5-2620 v3's
`upstream default` on 160×128×192 PEC ranges 140–244 MC/s (74%), the Ryzen 9
3950X's `fork avx2-mt default` on the same model ranges 470–616 (31%), and even
a *pinned* row, the 2× Xeon E5-2630's `upstream mt 12t` on 224³ Mur, ranges
311–407 (31%). Those are the auto-tune landing differently, or NUMA placement
varying between runs, not engine noise — but the effect on the table is the
same: a best-of-3 pick is biased upward by the spread, so the `default` columns
of the five servers should be read as optimistic. The pinned flat columns on the
two original machines, which is where the headline CPU ratios come from, are not
affected.

**The i7-6700K was not quiesced.** It is a shared server and ran `rclone`,
Jellyfin and an idle BOINC client throughout, at a load average of 0.6–2.8. This
is a real caveat and it is not hidden: it means the i7-6700K's absolute numbers
could be a few percent low. It does **not** undermine what that machine is used
for here, which is entirely *ratios measured within the host* — AVX2 vs SSE,
blocked vs flat, GPU vs CPU — all of which are best-of-3 over interleaved runs
that shared the same background load. The one place to be careful is the 64³
row, where that noise compounds with the bimodality above: its 8-thread repeats
spread 491–512 MC/s in the run tabulated here and 380–512 in the repeat, which is
why that row moved 1.4% between the two while every other row moved under 0.8%.
Nothing in §1 or §5 turns on it.

**The Ryzen 5 5600G is the best-controlled host in the set**, and it is worth
saying what that took, because none of it is exotic. It was idle and unshared;
its clock was sampled through a run and held 4.20 GHz all-core with 0.2% spread
across 90 seconds, so the `amd-pstate-epp` `powersave` governor was left alone
rather than forced to `performance`. Two independent full-matrix runs of the
same binary — an aborted one and its replacement — share 44 rows and agree to
**2.88%** worst case. A third run, of the *differently built* `WITH_GPU=ON`
binary, reproduces all 48 CPU rows to **2.99%**. Those two figures bound each
other usefully: the build difference costs no more than the run-to-run noise
already does, so the GPU bundle's CPU rows and the CPU bundle's are
interchangeable within the precision of this document.

**The five servers were idle**, which is the one thing that makes them easier to
read than the two workstations. All five are servers with no desktop session;
load average was 0.00–0.19 before each run started, they ran one matrix at a
time, and the binaries were staged to node-local disk so that no NFS traffic
could land in a measurement. The 2× Xeon Gold 6130 had other users logged in but
none of them running anything.

**Do not compare a matrix run on one day against one on another to answer a
build question.** That was tried here — the portable bundle against the
2026-09-07 i9-13900K run, to check the compiler — and it produced a confident,
entirely spurious 15–29% deficit on the PEC rows, because the workstation was
busy that afternoon and the PEC rows are the most bandwidth-sensitive in the
matrix. The PML and Mur rows of the same comparison agreed to 2–5%, which is
exactly how such an artefact looks when it is not recognised: partly consistent.
The paired alternating test in §2.2 put the same two builds within 0.98–1.04×.
The general lesson is the one already implicit in §2.4's pinning: when the
question is about the binary, vary only the binary.

**Never A/B a GPU change by disabling the excitation.** With no source the
fields stay exactly zero, and an all-zero grid runs ~28% faster than the
identical kernel on real data — zero operands flip almost no bits. This nearly
produced a fake 26% win once. Both arms of any comparison must carry real field
data.

**A harness bug worth naming**, because it silently produced missing data rather
than wrong data: the regex that reads the loop time out of openEMS's output
accepted `1200` and `262144.00` but not `1.12394e+07`, and the cell count prints
in scientific notation exactly when the run is too fast to emit an interval line
first — that is, on every GPU run on a large grid. The effect was that `setup_s`
came back `None` on precisely the rows with the most setup to report. Fixed in
this revision; §7's GPU column is the first that did not have to be measured by
hand.

---

## 11. A long run on a real model: the same answer from every engine

§1–§10 time the bare stepping loop on synthetic models. This section asks what
they cannot: whether the engines compute the same *result*. One real model was
run to completion with every engine on three of the machines, and the outputs
were compared fork against upstream and engine against engine.

### 11.1 Model and method

The model is the most finely meshed of a set of on-chip transmission-line test
cases: a 200 µm line, 0.20 µm wide, in the M2/M1 layers of an IHP back-end
stack. It is meshed at 0.0133 µm, which gives **2.27 M cells**. The setup is:

- a Gaussian excitation covering DC–300 GHz;
- Mur absorbing boundaries on all six faces;
- two lumped ports, each with a voltage and a current probe.

The field energy settles onto a static plateau, so the −50 dB end criterion
never fires. Each run therefore stops at its 60 ps cap on simulated time, which
is **2,359,786 timesteps**. All 30 runs ended at that cap, so every run computed
exactly the same number of steps.

Every run used the GPU portable bundle (§2.2, §9.1): fork `4bfcd74` and upstream
`d3d2a49`, built once with GCC 10.2.1. All three hosts ran the same binaries,
with the fork's GPU engine using each host's own Mesa (26.2.4 on the i9-13900K,
26.2.2 on the i7-6700K and the Ryzen 5 5600G). Each machine ran:

- `sse`, `sse-compressed` and `multithreaded`, from both upstream and the fork;
- the fork's `avx2` and `avx2-multithreaded`;
- `avx2-multithreaded` once more with temporal blocking
  (`OPENEMS_AVX2_TEMPORAL_BLOCK=16`), wherever it engages. On the i9-13900K it
  does not: the grid has only 191 x-lines, and the schedule needs at least two
  of that machine's 128-line tiles;
- the fork's GPU engine on every Vulkan device: the RX 6800 and UHD 770, the
  HD 530 and the Vega 7.

Each machine ran one simulation at a time, never a CPU and a GPU run together.
The multithreaded engines chose their own thread counts. `basic` was dropped:
at ~36 MC/s on the i9-13900K, a single run would take about 42 hours. Before the
full runs, a 2000-timestep smoke test on the coarsest case of the set ran every
engine. In it, fork and upstream `basic` gave bit-identical probes.

Results are compared on the probe values, which the files hold to 12
significant digits. The files themselves cannot be compared, because their
headers contain a version and a start time. S11 and S21 come from a DFT of the
port probes over 1–300 GHz, the same way openEMS' lumped port computes them.
Each run also hashed its excitation signal, and the hash is identical in all 30.

### 11.2 Three answers, bit for bit, differing only by rounding

The 30 runs produce exactly three distinct sets of probe values:

| Group | Runs, on all three machines unless noted | Probes vs A:<br>worst difference,<br>relative to the peak | S11 vs A | S21 vs A |
|---|---|---|---|---|
| **A** | upstream `sse`, `sse-compressed`, `multithreaded`; the same three fork engines; fork GPU on the **Vega 7** | — | — | — |
| **B** | fork `avx2` and `avx2-multithreaded`; `avx2-multithreaded` + temporal blocking (i7-6700K, Ryzen 5 5600G) | −116.3 dB | −117.5 dB<br>1.8e-5 dB, 4.4e-4° | −121.6 dB<br>1.5e-5 dB, 8.7e-5° |
| **C** | fork GPU on the **RX 6800**, **UHD 770** and **HD 530** | −116.8 dB | −119.8 dB<br>2.2e-5 dB, 3.0e-4° | −117.6 dB<br>1.3e-5 dB, 1.4e-4° |

Each S cell gives the worst complex difference |S − S<sub>A</sub>| over the band,
then the worst magnitude and phase differences. Groups B and C differ from each
other by −125.8 dB. For scale, this line's |S21| runs from −5.6 to −6.5 dB and
its |S11| from −6.4 to −15.3 dB.

The groups show three things:

- **The fork leaves upstream's engines unchanged, bit for bit.** For `sse`,
  `sse-compressed` and `multithreaded`, fork and upstream agree to the last bit
  on Raptor Lake, Skylake and Zen 3. That covers the code the fork changed
  underneath those engines: operator construction is parallelized (§7), and the
  Mur, excitation and probe paths are shared. This check is stronger than the
  cross-engine comparison in §4.3. Upstream's operator is independent code, so
  a fork defect common to all of the fork's engines would show up here.
- **Nothing that varies within a group changes a bit.** That includes:
  - thread count (3, 4, 7 and 8 threads);
  - compressed against plain operator storage;
  - temporal blocking against the flat sweep, at k=16 on 57-line tiles on the
    Ryzen 5 5600G and k=14 on 28-line tiles on the i7-6700K. The blocked
    schedule reorders the work; it does not approximate it.
- **Neither does the compiler nor the GPU vendor.** Groups B and C each also
  contain an earlier run of the same fork commit, built natively with GCC 16.2.1
  instead of the bundle's GCC 10.2.1, and those runs are bit-identical too.
  Group C spans three GPUs from two vendors and three generations, on two
  drivers (RADV and ANV).

The only thing that separates the groups is how a multiply-add is rounded. The
fields are float32 (`FDTD_FLOAT` in `tools/constants.h`):

- The SSE kernels round the product and the following sum separately.
- The AVX2 kernels use fused multiply-add, which rounds once.
- The GPU engine runs the same shader source on every device, so its split
  between A and C must come from the driver's shader compiler. Presumably that
  is whether the compiler fuses multiply-adds on that architecture. The Vega 7
  matches the CPU's unfused arithmetic bit for bit; the RX 6800, on the same
  RADV driver, does not.

The differences are also the size rounding would produce:

- **In the time domain**, the largest difference anywhere is 1.5e-6 of a probe's
  peak, **13 times float32's machine epsilon** (1.19e-7). It occurs at the pulse
  maximum, around 10 ps, and falls afterwards instead of accumulating: by 59 ps,
  2.3 M timesteps later, it is below half an epsilon of the peak on every probe.
  A real defect, such as a wrong coefficient, a missing boundary cell or an
  off-by-one at a tile edge, would leave a difference that persists or grows. A
  rounding difference follows the signal down.
- **In S11 and S21**, the quantities this model exists to produce, the groups
  agree to **≤ 3e-5 dB in magnitude and ≤ 5e-4° in phase** across 1–300 GHz.
  Refining the same line's mesh by one step, from 0.02 µm to 0.0133 µm, moves
  S21 by −63.6 dB (0.012 dB, 0.024°) and S11 by −68.9 dB (0.005 dB, 0.052°). The
  spread between engines is therefore about 50 dB below the discretization
  error of the finest mesh in the set.
- **Across all 15 cases**, an earlier comparison of the fork's GPU engine with
  its AVX2 engine (groups C and B) agrees to between −122.7 and −134.5 dB of the
  probe peak.

**So the fork's engines produce the same answer as upstream's.** Where the
arithmetic is the same, they match bit for bit. Where it is not, they agree to
within float32 rounding.

### 11.3 Throughput on this model

MC/s from each run's `Speed:` line; auto-tuned thread counts in parentheses.

| Engine | i9-13900K | i7-6700K | Ryzen 5 5600G |
|---|---|---|---|
| upstream `sse` | 253 | 147 | 205 |
| upstream `sse-compressed` | 268 | 138 | 222 |
| upstream `multithreaded` | 479 (7t) | 215 (3t) | 296 (3t) |
| fork `sse` | 248¹ | 148 | 212 |
| fork `sse-compressed` | 268 | 137 | 222 |
| fork `multithreaded` | 489 (7t) | 220 (4t) | 296 (3t) |
| fork `avx2` | 289 | 187 | 246 |
| fork `avx2-multithreaded` | 524 (8t) | 255 (4t) | 302 (4t) |
| fork `avx2-multithreaded` + tblock | does not engage | 333 (4t) | 556 (3t) |
| fork GPU | **5445** RX 6800<br>311 UHD 770 | 215 HD 530 | 322 Vega 7 |
| **best fork vs upstream `multithreaded`** | **11.4×** GPU<br>1.10× CPU | **1.55×** | **1.88×** |
| run time, best fork vs upstream `multithreaded` | 16 min vs 3.1 h | 4.5 h vs 6.9 h | 2.7 h vs 5.0 h |

These are single runs, not best-of-N. Fork and upstream run the same SSE
kernels, and those pairs agree to within about 3% (`sse-compressed` to within
0.7%). That is the noise floor of this table.

¹ The i9-13900K went to sleep for 9.2 h during this run. openEMS times its loop
by the wall clock, so its own `Speed:` line reads 97.6 MC/s. The 248 is computed
from the job's monotonic time, which does not advance during suspend. Expect the
same on any machine that can sleep during a multi-hour run.

The picture matches the synthetic matrix:

- **The RX 6800 is the only device that changes the order of magnitude.** At
  5445 MC/s it is in line with its 224³ Mur row (5125, §8.1), and it turns a
  3-hour job into a 16-minute one.
- **Temporal blocking is again the CPU result.** It is worth 1.84× over the flat
  AVX2 engine on the Ryzen 5 5600G's undivided 16 MB L3, and 1.31× on the
  i7-6700K's 8 MB. That 1.31× is on a Mur model, where the matrix's 224³ Mur row
  regresses on the same machine (0.87×, §1.2). Without blocking, as on the
  i9-13900K, AVX2 multithreaded reaches only 1.10× upstream, the
  bandwidth-limited margin described in §3.2.
- **The integrated GPUs do what §4.2 predicts for a model without PML.** The UHD
  770 reaches 0.65× upstream's best, the HD 530 ties it (1.00×) and the Vega 7
  reaches 1.09×. On both machines where the blocked CPU schedule engages, it
  beats the integrated GPU: by 1.55× on the i7-6700K and 1.73× on the Ryzen 5
  5600G.

**A GPU monitor is not a stall detector.** During the HD 530 run, a GPU monitor
on the i7-6700K showed the load flipping between 100% and 0%, with almost no CPU
load. This was not a stall:

- the kernel's per-process engine counter showed the GPU 99% busy over three
  minutes, with no idle second;
- the GPU held its 1150 MHz maximum clock, with no RC6 (idle) residency;
- the run finished at 215 MC/s, against 213 in a 20,000-timestep smoke test.

The near-zero CPU load is normal for the GPU engine, whose host thread spends
the run waiting on a fence. Neither Radeon showed the flicker.

### 11.4 Reproducing

The harness is in `tests/openems-verify/`:

- `run_verify.py` is standard-library Python, with the bundle's binaries beside
  it. It runs every engine on every device of the machine it starts on, one
  simulation at a time under a host-wide lock, and it can resume after an
  interruption. `--smoke` estimates the run time, and `--only 'cs0p013333um$'`
  restricts the run to this case.
- `compare_verify.py` builds the tables above from the collected
  `results/<host>/` directories.

Its full output, including every run's group, thread count, job time and
temporal-blocking announcement, is
`python/Tests/results/verify_preflight_tl_2026-10-15.txt`.
