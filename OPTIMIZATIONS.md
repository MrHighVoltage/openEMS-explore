# openEMS performance work — GPU and AVX2 engines

Running record of optimization work on the Vulkan GPU engine and the AVX2
engines: what was done, what it was worth, and — just as important — what was
tried and rejected, so it is not rediscovered blind.

Scope is the two fast engines — `Engine_Vulkan` / `Operator_Vulkan` and
`Engine_AVX2` / `Engine_AVX2_Multithread` — plus the host-side paths they
share with every other engine: field dumping, probe readback and operator
construction. The SSE and basic engines are out of scope as *engines*, but
they benefit from everything in §1.1.

Measurements come from two workstations here — plus five more used only for the
fork-vs-upstream matrix, listed at the end — and are not comparable across
them:

- **i7-6700K** — 4 cores / 8 threads (SMT2), 8 MB L3, dual-channel DDR4.
  Practical STREAM-Triad ceiling ~29 GB/s. Everything in §1's AVX2 table was
  measured here. It was believed lost when §4 was written; it is a remote
  server, it is still available, and it also carries an integrated **Intel HD
  Graphics 530** (Skylake GT2, ANV). Re-measured 2026-09-08 — see
  [PERFORMANCE.md](PERFORMANCE.md), where its small L3 is what shows which §4
  conclusions are properties of the code and which are properties of a 36 MB
  cache.
- **i9-13900K** (2026-09-04) — 8 P-cores + 16 E-cores, 32 threads, 32 GB
  dual-channel DDR5, STREAM-Triad ceiling **~49 GB/s** at 6–8 threads. Two
  GPUs: **Radeon RX 6800**, 512 GB/s peak, 128 MB Infinity Cache, **256 MB
  BAR** (no resizable BAR, so `fbd95a6` is load-bearing here), and an integrated
  **Intel UHD 770**. Mesa 26.2.1, RADV and ANV. Everything in §1's GPU table
  and all of §4 were measured here.
- **The five servers** (2026-09-11, CPU only) — Ryzen 9 3950X (16C, 4×16 MB
  L3), 2× Xeon Gold 6130 (32C, 2×22 MB), 2× Xeon E5-2630 (Sandy Bridge, 12C,
  **no AVX2**), Core i5-8500 (6C, 9 MB, CentOS 7), 2× Xeon E5-2620 v3 (12C,
  2×15 MB). Servers with no Vulkan driver and no openEMS dependencies,
  benchmarked from a self-contained bundle built in a glibc-2.17 container
  (`python/Tests/build_portable_bench.sh`). They are not used for any tuning
  decision in this document, but two of its assumptions do not survive them —
  see §4.8 and [PERFORMANCE.md](PERFORMANCE.md) §5.3 and §6.

Benchmark drivers: `python/Tests/benchmark_avx2_engine.py`,
`python/Tests/benchmark_field_dump.py` (dump-path A/B, see §1.0),
`python/Tests/bench_temporal_blocking.c` (standalone, see its header),
`python/Tests/benchmark_gpu_engine.py` (`--gpu-index` selects the device),
`python/Tests/test_gpu_engine.py`, `python/Tests/test_upml_engines.py`,
`OPENEMS_RUN_STABILITY_TESTS=1 python/Tests/test_stability_testsuite.py`.

---

## 1. Completed work

Reconciled against the complete diff from the upstream merge base
(`d3d2a49..HEAD`, 159 files): every source file that differs from upstream is
either accounted for below, or is a build, packaging or compatibility change —
the boost removal (`28f65db`), CI and CMake, vcpkg, the CSXCAD API updates
(`c5760e0`, `284f532`, `7b417f4`), and legacy-compiler fixes in `tools/sar_calc.cpp`
(`af06ecf`).

### 1.0 The engines

The two engines this fork adds are the largest optimization in it. Everything
else in this document tunes them; this entry is what they are worth against a
from-scratch build of upstream on the same machine, with the same compiler and
the same models. Full matrix and method in [PERFORMANCE.md](PERFORMANCE.md)
§1–§8; "upstream best" there is the fastest of three upstream configurations,
usually its SSE multithreaded engine at a hand-picked thread count.

| Commit | Change | Measured against upstream's best |
|---|---|---|
| `7effdfb`, `4e6530c` | **`Operator_AVX2` / `Engine_AVX2`** and `_Multithread`: 256-bit AVX2+FMA with an interleaved z-layout (8 z-cells per vector) and `std::map` operator deduplication | Single-threaded against upstream's SSE engine, where the width shows directly: **1.31×–2.04×** on the i9-13900K, **1.65×–2.59×** on the i7-6700K, lowest on PML where the passes vectorize least well. Multithreaded against upstream's best: **0.90×–1.72×** flat on the i9-13900K — it loses on PML, PERFORMANCE.md §3.4 — and **1.70×–4.45×** with temporal blocking |
| `63b2e49`, `4e6530c` | **`Operator_Vulkan` / `Engine_Vulkan`**: coefficient compression into compact `opIdx` + `vv`/`vi`/`ii`/`iv` tables, double-buffered command submission, ReBAR support, pre-recorded multi-timestep command buffers | **6.1×–39.0×** on an RX 6800 with temporal blocking (i9-13900K). On the i7-6700K's integrated HD 530, **0.85×–1.61×** — below upstream on cache-resident PEC grids, above it on PML and Mur, where it is the fastest engine in that machine |
| `28f65db` | boost → C++17: `std::thread`/`std::jthread`, a `std::mutex`/`condition_variable` barrier (`tools/barrier.h`), a lightweight option parser. Drops the boost dependency | Not a throughput change; it is what the AVX2 multithread engine's barrier and thread control are built on |
| `84551fe` | The regression `28f65db` caused: the worker loop tested `m_stopThreads` at the top, so a worker could wake from `m_stopBarrier`, observe the flag set by `Reset()`/`changeNumThreads()`, and exit without ever arriving at `m_startBarrier` — leaving the controller waiting forever for `m_numThreads+1` arrivals. `boost::thread_group::interrupt_all()` had unblocked workers wherever they were, so the barrier was never a shutdown path before | Hit in CI whenever thread auto-tuning called `changeNumThreads()`; reproduced 5/8 runs pinned to 2 cores, 0/8 after the fix |
| `987fcb7` | AVX2 dispatch made architecture-aware **at compile time**, so a non-x86 build never parses `<immintrin.h>` and never selects the AVX2 engine. It does **not** guard the run host: the test is on `CMAKE_SYSTEM_PROCESSOR` and the compiler, so an x86-64 binary still selects AVX2 on a pre-AVX2 CPU and dies with SIGILL. Measured on the 2× Xeon E5-2630 — §4.8, PERFORMANCE.md §6. A runtime CPU check added later closes this | Correctness of engine selection for cross-*architecture* builds only |

**GPU extension coverage is a performance property, not a feature checkbox.**
An extension without a shader has to run on the CPU, which means draining the
queue, downloading the fields it touches, running it, and uploading again —
every timestep — and it disqualifies the pipelined main loop (§1.1). So each
extension ported to Vulkan is an optimization of the models that use it. The
engine currently covers UPML, Mur ABC, local absorbing sheets, TF/SF,
dispersive materials, lumped RLC, excitation, steady-state detection, probe
gather and energy reduction (`FDTD/gpu/shaders/`).

| Commit | Change | Measured |
|---|---|---|
| `fb2049c` | **Local absorbing sheets on the GPU.** All localized first-order Mur phases as sparse surface dispatches with device-resident coefficients and E/H history, covering both the voltage-only `MUR_1ST` form and the current-side super-absorption of `MUR_1ST_SA`, including on the fused UPML path. `OPENEMS_GPU_CPU_LOCAL_ABC` keeps the CPU path for native-vs-hybrid validation | **7165 MC/s native vs 0.34 MC/s hybrid** — see below |
| `ed6f590` | **Steady-state detection as a GPU observer** rather than a CPU extension — electric probes sampled into a three-period coherent history ring after each voltage phase, convergence evaluated only when the control path asks for it | Enablement: as a CPU extension it defeats GPU batching entirely |
| `50c6817` | **The steady-state check with the energy of its own timestep**, summed by x-plane inside the batch, flat or blocked, and evaluated once the batch is done | Fixes a phase error that kept GPU sine runs from stopping; lifts the refusal of temporal blocking (§4.12) |

**What a missing shader costs** (the i9-13900K, RX 6800, 2026-09-08).
`python/Tests/benchmark_gpu_engine.py --local-abc mur`, A/B through
`OPENEMS_GPU_CPU_LOCAL_ABC=1`, which is the only extension with a switchable
CPU path:

| Grid | native GPU | CPU fallback | |
|---|---|---|---|
| 64³, 200 TS | 7165 MC/s | 0.34 MC/s | **~21,000×** |
| 64³, 400 TS | — | 0.34 MC/s | rate is flat in run length, so it is not a startup artifact |
| 160×128×192, 200 TS | 15519 MC/s | did not finish in 9 minutes (<1.5 MC/s) | — |

That is far more than the bulk transfer explains: at 64³ the fields are 6.3 MB,
which is milliseconds per timestep, while a hybrid timestep takes ~0.77 s. The
cost is in the per-cell host access the CPU extension makes against
GPU-resident memory, not in the copy — the same class of problem as the
`HOST_CACHED` staging bug in §1.1. Root cause not isolated further, because
the conclusion does not depend on it: **an extension without a shader takes
the Vulkan engine out of contention entirely**, so shader coverage is not a
convenience, it is the difference between the GPU engine being usable on a
model and not.

The per-cell access was the BAR mapping again (found 2026-11-01, §1.2):
`SyncFieldsToHost()` then `memcpy`'d both fields out of it, two 6.3 MB reads
per hybrid timestep at the ~25 MB/s measured for that mapping, about 0.5 s of
the 0.77 s.  The fallback and the switch are gone since `3a7633a`: a model with an
extension that has no shader stops with an error and needs a CPU engine.

### 1.1 Host paths: dumping, GPU interaction, startup

None of these touch an update kernel; all of them show up in wall clock.

#### Host-side field dumping

Every engine dumps fields through `Common/processfields*`, so this path is
shared with SSE and basic. Before this work it was fully synchronous: compute
the field, open the HDF5 file, write, close, resume stepping.

| Commit | Change | Measured |
|---|---|---|
| `d17da2c` | **Asynchronous ring-buffer writer.** `ProcessFieldsTD::Process()` hands the freshly extracted field plus a write-lambda to `AsyncFieldWriter::Submit()` and returns to the engine immediately; a background thread performs the HDF5/VTK write. Four ring slots by default, `Submit()` blocking when they are full, so memory is bounded to `(slots+1)` snapshots | **1.7–1.9×** wall clock on a dump-every-timestep run — see below |
| `92de381` | **Persistent HDF5 handle.** `OpenFile()`/`CloseFile()`/`FlushFile()` plus an `AcquireFile()`/`ReleaseFile()` access pattern; without a persistent handle the old open-use-close behaviour is preserved for every other caller. This is what lets the writer thread write without re-opening the file per timestep | Not isolated (no toggle); it is a precondition for `d17da2c`, not an independent lever |
| `d39cda0`, `7a9343d` | **Template-based `CalcField` with threaded extraction.** A compile-time switch on the concrete engine type replaces per-cell virtual `GetVolt`/`GetCurr` dispatch — 12N–24N virtual calls per dump — and the i-dimension is split across `std::thread` workers. Falls back to the virtual-dispatch loop for engine types it does not know | Not isolated (no toggle) |
| `586a8ac` | The AVX2 engine's missing fast extraction path, so AVX2 reaches the template path above rather than the fallback | 336 → 426 MC/s, dump every timestep — also in the AVX2 table below |
| `7364d69` | **Global HDF5 mutex.** Not an optimization but the price of one: libhdf5 as distributions ship it is not thread-safe, so the writer thread racing the main thread could corrupt file state. Every HDF5 C API call now goes through one process-wide recursive mutex | Correctness. It also caps what the async writer can ever be worth: two HDF5 users never overlap |

**The async writer measurement** (the i9-13900K, 2026-09-08). Fixture:
`python/Tests/benchmark_field_dump.py`, 128³ = 2.1 M cells,
`avx2-multithreaded`, 200 timesteps, full-domain E dump on *every* timestep
(25 MB per dump, 5.06 GB per run) to ext4 on NVMe. A/B through the
`OPENEMS_ASYNC_DUMP_BUFFERS` gate added for this measurement, three
alternating repeats each:

| | engine-loop speed | wall clock |
|---|---|---|
| `OPENEMS_ASYNC_DUMP_BUFFERS=0` (synchronous) | 109 / 123 / 115 MC/s | 4.70 / 4.30 / 4.53 s |
| default, 4 ring slots | 231 / 214 / 237 MC/s | 2.69 / 2.82 / 2.62 s |

**1.9× on the engine loop, 1.7× on wall clock**, or ~9.6 ms of stepping time
returned per dump. Two caveats on reading that number. It is buffered I/O —
the writes land in page cache, so this measures HDF5 serialization and
`memcpy`, not storage; on a filesystem slow enough to make the ring
back-pressure, the win is larger but bounded by the disk. And it scales with
dump *cadence*: this fixture forces `OverSampling` above the Nyquist rate to
dump every timestep, which is the worst case. At openEMS's normal cadence
(`Nyquist/OverSampling`, 173 timesteps on this fixture) the same absolute
saving is amortized over 173 steps and is not measurable.

#### GPU host interaction

The Vulkan engine's early problem was not its kernels but everything around
them: a sequential main loop that left the GPU idle at ~2.5% utilization while
the CPU processed the previous timestep, and two operations that forced a full
drain and a PCIe download every few seconds.

| Commit | Change | Measured |
|---|---|---|
| `41afbf5`, `3df17eb` | **Energy reduction on the GPU** (`reduce_energy.comp`). `CalcTotalEnergyEstimate()` used to scan all `3·Nx·Ny·Nz` floats — ~650 MB at 27 M cells — over the BAR aperture with random access. Now 256 workgroups reduce in shared memory and the CPU sums 256 partial pairs | **~300 ms → ~1 ms** per energy evaluation (i9-13900K) |
| `3df17eb` | **Probe cache and gather shader.** Probe cell accesses are recorded during the first `PA->Process()`, deduplicated, and thereafter gathered on-GPU into a small buffer, so a line integral no longer drains the queue and downloads the field | Structural; the end-to-end effect is in the pipelined loop below |
| `3df17eb` | **Fused UPML + Yee dispatches**, per-region PML buffers concatenated into shared allocations. 28 dispatches + 7 barriers per timestep become 2 + 2 | **1.53×–1.65×** on PML models, see below. The design estimate at the time was 1.3–1.5×, so it was, unusually, an underestimate |
| `3df17eb` | **Pipelined main loop with speculative submission** in `RunFDTD()`: the GPU computes timestep N+1 while the CPU processes N out of the probe cache. Falls back to the sequential loop when the preconditions do not hold | Design estimate ~40× at the 2.5%-utilization starting point; the honest end-to-end number is the fork-vs-upstream matrix in [PERFORMANCE.md](PERFORMANCE.md) |
| `e05bc8e` | **Energy reduction only when asked for.** The reduction had been dispatched after every submitted batch, scanning all six field components whether or not anything read the result; it now runs when convergence checking or a status code actually requests it | Removes one full-grid pass per batch from every run that is not energy-limited |
| `e05bc8e` | Tiny probe and energy outputs kept in coherent host-visible memory, avoiding dedicated transfer-queue ownership transfers for a few hundred bytes; sorted probe keys and direct mapped snapshots on the CPU side; missing compute memory dependencies added between excitation, post/apply extensions, current updates and the next timestep; ReBAR allocation made recoverable, so a compatible-but-undersized host-visible aperture falls back to device-local fields instead of failing | Mixed correctness and overhead; not separately quantified. The ReBAR fallback is the precursor to `fbd95a6`, which found the aperture was the dominant cost |
| `4eec325` | Persistently mapped field allocations for hybrid host sync (no staging copies or utility submissions on unified-memory/host-visible devices); per-region UPML coefficient uploads batched into one startup command; duplicate per-region flux/coefficient allocations released once fused resources exist | Startup time and footprint; not separately quantified |
| `66aca92` | Command-ring handling, a dedicated transfer queue for probe readback with per-slot sync, compute barriers tightened to the extensions actually dispatched, optional phase-level GPU timestamp profiling | Not separately quantified; the profiling hooks are what later measurements used |
| `55c9375`, `52730d3` | Device-side finite-value validation with stage/trace reporting; field sync before CPU extraction; UPML region packing that accounts for all three components; validation of compressed/UPML/excitation data at upload | Correctness. `validate_fields.comp` is why numerical failures stop the run instead of producing plausible garbage |

**The fused-UPML measurement** (the i9-13900K, RX 6800, 2026-09-08).
`python/Tests/benchmark_gpu_engine.py --boundary pml`,
A/B through `OPENEMS_GPU_DISABLE_FUSED_UPML=1`, no probes and no temporal
blocking, so the sequential main loop is on both arms and only the dispatch
structure differs:

| Grid | separate dispatches | fused | |
|---|---|---|---|
| 160×128×192, 3.9 M cells, 2000 TS | 5673 / 5516 MC/s | 9104 / 9386 MC/s | **1.65×** |
| 256×224×224, 12.8 M cells, 1000 TS | 3008 MC/s | 4609 MC/s | **1.53×** |

**Most of that is not dispatch overhead.** Removing ~26 dispatches at
§4.7/C3's measured ~5.6 µs each is ~145 µs per timestep, which is a fixed
cost. Against the measured per-timestep differences it accounts for about half
the gain on the small grid (419 → 693 µs/TS, 274 µs saved) and about a tenth
on the large one (2787 → 4270 µs/TS, 1483 µs saved). The rest is memory
traffic: the separate path makes extra full passes over `volt` and
`volt_flux` — the pre-pass stores the flux to VRAM, the Yee and post passes
load it back — while the fused kernel carries those intermediates in
registers. G3 later closed the *remaining* `volt_flux` traffic as a floor of
the UPML formulation; what fusion removed was the traffic above that floor.
That also explains the direction of the two numbers: the smaller grid has
proportionally more PML cells (31% vs 19% of the domain) and its 94 MB
working set fits the 128 MB Infinity Cache, so it gains more in ratio, while
the larger grid gains more in absolute time.

`ea82f59` (async field-dump download, ~18× on a dump-heavy GPU run) belongs to
this group chronologically but is already in the GPU table below.  It only
ever helped with `--gpu-no-rebar-fields`; the default ReBAR path kept reading
mapped VRAM at about 25 MB/s, and every GPU dump held a later timestep than
its label.  The fields have not been mapped at all since `0d92e77`.  The GPU-side gather that replaced it (`76fa8f6`, `d38071b`), and
the measurements, are in
[FDTD/GPU_PIPELINE_OPTIMIZATION.md](FDTD/GPU_PIPELINE_OPTIMIZATION.md) §4.
The fusion and pipelining design is in
[FDTD/GPU_FUSION_AND_PIPELINING.md](FDTD/GPU_FUSION_AND_PIPELINING.md).

#### Startup: operator construction

| Commit | Change | Measured |
|---|---|---|
| `d9b00bb` | **Parallel operator coefficient preparation.** Coefficient generation is split into x-ranges and run on the SSE/AVX2 operators' existing worker threads, with flat coefficient indices computed directly so workers never race through the stateful `AdrOp` cursor. The Vulkan operator gets the same range parallelism, and its serial ordered-map compression pass is replaced with exact-bit sharded hash tables plus parallel table packing — same coefficient identity, far less startup contention | **8.2 s → 5.6 s** (i9-13900K, 224³ PML_8) and **17.3 s → 14.3 s** (i7-6700K) versus upstream — see [PERFORMANCE.md](PERFORMANCE.md) §7 |
| `55c9375` | Vulkan material, PEC, coefficient and UPML construction parallelized with worker-local grading parsers | Included in the figures above |
| — | Controls: `OPENEMS_OPERATOR_THREADS` (`OPENEMS_GPU_OPERATOR_THREADS`), `OPENEMS_GPU_COMPRESS_THREADS`, `OPENEMS_OPERATOR_PROFILE`, `OPENEMS_GPU_STARTUP_TRACE` | — |

Operator build is not a footnote on the GPU: at 224³ on the i9-13900K it costs ~7.0 s
against ~1.1 s for 1200 blocked timesteps. Everything above is what keeps the
crossover against upstream's best CPU configuration at ~55 timesteps.

#### Not measured

- The **persistent HDF5 handle** and the **template/threaded field
  extraction** have no A/B number. Neither has a runtime toggle, so isolating
  them means a rebuild per arm. Note that the 1.7× above does *not* bound
  them: both were already present in both arms of that A/B, so the async
  writer's win is measured on top of them, not inclusive of them.
- The **pipelined main loop with speculative submission** carries only its
  design estimate (~40× from a 2.5%-utilization starting point). Unlike
  fused UPML it has no disable switch — it engages whenever probe gather is
  set up and no CPU-side extension forces a drain — so an A/B needs a code
  change, not an environment variable. The end-to-end evidence that it works
  is the fork-vs-upstream matrix in [PERFORMANCE.md](PERFORMANCE.md).
- `4eec325`, `66aca92` and the second `e05bc8e` row are unquantified entirely.
- **GPU extension coverage** is measured only for local absorbers (§1.0),
  the one extension with a switchable CPU path. The other extensions have no
  fallback to compare against, so their coverage is argued from that one
  measurement rather than measured.
- Everything in §1.1's dump and startup tables was measured on the i9-13900K only.

### 1.2 GPU

| Commit | Change | Measured |
|---|---|---|
| `fbd95a6` | Don't place field buffers in an undersized BAR aperture | **6.4–6.5×** on grids exceeding the 256 MB aperture (803 → 5115 MC/s at 256×224×224) |
| `ea82f59` | Async field-dump download; request `HOST_CACHED` staging | **~18×** wall clock on a dump-heavy 291³ run (74 s → 16 s), with `--gpu-no-rebar-fields` only; replaced by the row below |
| `76fa8f6`, `d38071b` | Gather each dump's cells on the GPU at its timestep; keep FD/NF2FF/SAR DFT sums on the GPU; no CPU reads of mapped VRAM | **221× / 115× / 97×** on 160³ with ReBAR (plane / NF2FF / whole-volume dumps: 298 → 1.34 s, 150 → 1.30 s, 299 → 3.08 s); plane and NF2FF dumps now free. Fixes dumps holding a later timestep than their label |
| `6900f90` | Probes with a start time read their cells during the probe-cache recording | Correctness: such a probe missed the cache once due and read the live fields, which the pipelined loop had already advanced |
| `0d92e77` | Field buffers always device-local; the BAR mapping, its readers and the GPU line integrals removed; `--gpu-no-rebar-fields` accepted and ignored | **~9×** per run with three runs sharing the RX 6800: 175–210 → 1680–1740 MC/s (6.6–10.9 M cells, measured with `--gpu-no-rebar-fields`). Each run's fields fit the 256 MB BAR, together they did not, and the driver put the rest in system RAM (`amd-requested-vram` above `drm-memory-vram` in `/proc/PID/fdinfo`). A single run is unchanged (160³: 1.39 s median either way); raw data in `python/Tests/results/gpu_cleanup_2026-11-02.txt` |
| `3a7633a` | No hybrid loop: a model with an extension that has no GPU implementation stops with an error | Simplification; every extension that reaches the GPU engine has a shader, and the hybrid loop was slower than a CPU engine (§1.0) |
| `b5b9c07` | Remove the never-enabled transfer-queue probe copy, its readback ring and staging fallback, and other dead probe code | Simplification, no behaviour change |
| `c66193d` | Speculative depth auto by default: while the host processes a timestep, the GPU computes up to the next one a processing is due at (≤ 64), not just one | **2.1×** on 160³ with 63 whole-volume dumps (2.97–3.00 → 1.44 s); dump-free, plane and NF2FF runs unchanged. `OPENEMS_GPU_SPEC_DEPTH=1` restores depth 1 |
| `2ed6523` | Reuse shared curl taps in the update shaders | Neutral on RADV (its NIR already CSE'd them); kernels shrink 11–14%. Portability insurance |
| `49c7efc` | Narrow the compressed-operator index to 8/16/32 bits | **+3.5% to +6.2%**; Coax index buffer 2793 → 698 KB |
| `fabcaa3` | Single Vulkan 1.3 baseline, negotiate device features | Enablement, not throughput |
| `6b571da` | Per-cell operator storage when deduplication stops paying | **1.93×–2.76×** on graded meshes; uniform meshes unchanged |
| `00ba1c6` | Deduplicate UPML coefficients into shared tables | **1.08×–1.17×**; 19.2 MB → 1.6 MB, 53.5 MB → 4.5 MB |
| `05ce5f0` | Stop allocating the async dump ring when the model has no dump boxes | **2.8× less GPU memory** on dump-free runs (224³: 803 → 288 MB VRAM, 657 → 142 MB host). Throughput unchanged — the buffers were never used |
| `b87a1f7` | Give each TF/SF field index to one thread instead of racing on `+=` | Correctness, not speed: the Vulkan engine was **nondeterministic** on any TF/SF model. GPU–CPU H-field disagreement 5.62e-06 → 4.64e-07 |

Two structural results worth carrying forward:

- **The Yee kernel is at roofline.** After `fbd95a6`, ~5.2 GCells/s × ~80
  B/cell/timestep ≈ 415 GB/s against a 512 GB/s peak. Grids that fit the
  128 MB Infinity Cache run 12–14 GCells/s. There is no headroom left in the
  plain PEC update path; remaining GPU wins must come from *reducing traffic
  or passes*, not from tightening arithmetic.
- **Operator-table locality dominates on graded meshes.** `6b571da` came from
  observing a 4× collapse (383 → 95 MC/s) when 493,040 unique coefficient
  sets in a 23 MB table turned six per-cell gathers into six scattered cache
  lines. Table *size relative to cache* is the thing that matters, not the
  compression ratio.

### 1.3 AVX2

| Commit | Change | Measured |
|---|---|---|
| `e3932e8` | Eliminate redundant field loads in the update kernels | **+7–8%** single-threaded (338 → 365 MC/s); flat at 8 threads (already bandwidth-bound) |
| `586a8ac` | Add the missing AVX2 fast field-extraction path | Full-domain dump every timestep: **336 → 426 MC/s**; removes an AVX2-specific tax SSE never paid |
| `574f714` | Document the MT bandwidth ceiling (comment only) | — |
| `08fa968` | Vectorize the UPML passes | See below |
| `fd555ee` | Calibrate the thread count on real timesteps instead of on 4 s `NextInterval` callbacks | **+50%** on a 400-timestep run (507 → 759 MC/s), **+6.4%** at 20000 timesteps. The i9-13900K only; the i7-6700K was never hurt by the old search |
| *(prototype, opt-in)* | Trapezoidal temporal blocking of the update sweep, `OPENEMS_AVX2_TEMPORAL_BLOCK=<k>` | **3.2×–3.6×** on grids that exceed L3 (873 → 2805, 613 → 2171, 610 → 2194 MC/s). Bit-identical to the flat sweep. Carries every engine extension, since 2026-11-04 steady-state detection too (§4.12): **3.13×** PEC, **2.96×** UPML, **2.60×** Mur at 224³ — see §4.7/C2 |
| `b39bd36` | Cheaper synchronisation in the blocked schedule (barrier elision, y-row split, optional spinning barrier, sync statistics), and a tile that leaves room for the coefficient tables | **1.6×–2.0×** over the previous blocked default on a 5 M-cell TL model, bit-identical — see §4.9 |

**The multithreaded AVX2 engine is memory-bound and already at the host
ceiling.** Measured DRAM traffic at its optimal 4-thread point is 27–38 GB/s
against a ~29 GB/s STREAM ceiling; IPC ~1.06. Thread count peaks at 4 and
*drops* 6% at 8 — SMT siblings add no bandwidth. **Do not pursue compute-side
optimizations for the multithreaded AVX2 engine on this class of host**;
single-threaded and low-thread-count configurations still respond.

> **Amended on the i9-13900K.** The bandwidth-bound conclusion survives — the engine
> still tracks the STREAM curve, including its roll-off past 8 threads. The
> clause *"the runtime auto-tune already converges to 3–4 threads unassisted"*
> did not: it was true only because the i7-6700K's optimum is near the search's
> starting point. See §4.2.

---

## 2. Tried and rejected — do not retry without new information

- **Non-temporal stores (`_mm256_stream_ps`) in the AVX2 update kernels.**
  Measured *worse* everywhere: −15% single-threaded, −2% at 8 threads. Root
  cause is structural: the per-polarisation stride is 3 `f8vector`s = **96
  bytes**, which does not align to 64-byte cache lines, so write-combining
  buffers cannot accumulate full lines. Revisiting this requires a
  cache-line-aligned field layout first (see candidate A3).
- **Subgroup-uniform coefficient fetch on the GPU.** Looked worth 26% on a
  uniform synthetic mesh but only 2.6% on Coax's graded mesh — the 26% was a
  fixture artifact, not a lever.
- **Shrinking the GPU coefficient tables themselves** (as opposed to the
  index): they are a few KB and already cache-resident.
- **A background CPU thread for `PA->Process()` on the GPU engine.** An
  earlier design added a `std::thread` with mutex/condvar handshaking to run
  post-processing concurrently with the next GPU batch. It was built but never
  wired into `RunFDTD()`, and was superseded by the pipelined main loop with
  speculative submission (§1.1), which achieves the same overlap with no
  second OS thread and no thread-safety surface. The scaffolding
  (`StartAsyncProcessing()`, `AsyncWorkerLoop()`, `TriggerAsyncProcessing()`,
  `WaitAsyncProcessing()`) was removed. Note this is *not* the same as the
  field-dump writer thread (`d17da2c`), which is alive, on the CPU dump path,
  and measured at 1.7× in §1.1.
- **Transferring the AVX2 UPML vectorization findings to the GPU.**
  Investigated and rejected on inspection: the GPU is already coalesced and
  already fuses UPML, and its PML penalty is 2.0× versus the CPU's 4.8×. The
  CPU problem was specifically the `f8vector` z-lane layout, which has no GPU
  analogue.

---

## 3. Open candidates

Ordered by expected value. Each states how to *falsify* it cheaply before
committing to an implementation.

### A1 — AVX2 dispersive extension  ✅ **done (`2bfb77d`)**

Priced, implemented and closed. No dispersive fixture existed, so one was
built: 96³ grid, Drude box (`eps_plasma = 3e10`, order 1) over the central
half of the domain — **12.5% of cells** — against an identical model with a
plain dielectric box.

The extension cost **33–34% of runtime**. The cause was not the arithmetic
but the addressing: the extension reaches the field only through
`GetVolt`/`SetVolt`, and on `Engine_AVX2` those carry an integer div and mod
against the runtime member `numVectors` — **18 divisions and 18 moduli per
cell per timestep** across the four passes. Precomputing the vector offset
and lane per list entry (the list is fixed after operator build) removed all
of them:

| Engine | before | after | |
|---|---|---|---|
| `avx2` | 239.95 MC/s | 286.54 MC/s | **+19.4%** |
| `avx2-multithreaded` | 259.45 MC/s | 311.26 MC/s | **+20.0%** |

Bit-identical on both engines; divergence from the untouched SSE engine is
unchanged at 9.235e-03.

**A2 (sort the cell list for locality) is closed out as not worth doing.**
93.7% of consecutive entries already land within 3 f8vector units — median
delta is exactly 3, i.e. one cell's three polarisations — so the list is
already effectively sequential in engine address space. Sorting would take
that to 99.96%.

**The threading claim is withdrawn.** The extension genuinely has no
threading, but on this grid threading adds little to *either* model (plain
+2.5%, dispersive +8.6% ST→MT), so a serial-section cost is not demonstrated
and would need a larger grid to assess. It was speculation and is not
supported by the measurement.

### A1b — is anything left in the dispersive extension?

After the addressing fix the extension still costs ~22% of runtime on 12.5%
of cells. A traffic estimate suggests this is now close to inherent: per cell
per timestep the ADE passes stream three operator coefficient arrays and
read-modify-write two state arrays across three components (~200 B), against
the Yee update's ~80 B/cell. A dispersive cell is simply ~3× a normal cell in
memory traffic, and those arrays are already dense and contiguous.

**Confirmed — closed.** Stubbing the scattered field read out entirely
(wrong physics, timing only) recovers only 5 of the 74 MC/s gap, **+1.8%**:
277.22 → 282.21 MC/s against a 351.41 MC/s non-dispersive baseline. The
addressing is no longer the cost; ~93% of what remains is dense streaming of
the ADE arrays. Supporting evidence: cost is linear in dispersive volume
above ~10% of cells, and order-2 dispersion costs exactly **2.00×** order-1's
overhead — the loop scales with the number of ADE arrays and nothing else.

This is a traffic floor, matching the two structural results in §1. The only
remaining levers change the physics (fewer ADE arrays, lower precision) and
are out of scope. **The AVX2 side is done.**

### A3 — Cache-line-aligned AVX2 field layout

The 96-byte per-polarisation stride is implicated in the NT-store failure and
also fragments prefetch. Padding 3 `f8vector`s to 4 (128 B) costs 25% more
memory on a bandwidth-bound kernel — plausibly a net loss on its own, and only
interesting *combined with* NT stores.

High risk, wide blast radius (touches every engine and extension indexing
scheme). Listed for completeness; **do not start here.**

### Hardware caveat — historic, and now lifted

> **Superseded on 2026-09-04.** The RX 6800's measurements were repeated on the
> same machine with its CPU and both GPUs together, and §4.1 reproduces the
> headline §1 figure to within 0.2%. The caveat below is kept because G1
> and G2 were written under it: those two were deliberately argued from
> dispatch and barrier *counts* rather than from throughput, and that
> reasoning is still sound. §4.7/C3 has since measured what the minimal
> pipeline actually costs on the real hardware — nothing — so the
> count-based argument turns out to have been conservative rather than
> merely careful.

When G1–G3 were written, the RX 6800 that produced every GPU measurement in §1
was not on the machine. The only GPU available was an Intel HD Graphics 530
(Skylake GT2, integrated, sharing the CPU's DDR4, no large last-level cache).
It ran the GPU engine correctly but in a completely different performance
regime — 386 MC/s on a 96³ PEC model, i.e. about the same as the AVX2 CPU
engine on that host, because it was the same memory. Throughput conclusions
drawn there would not have transferred, so the findings below were restricted
to *architecture-independent* facts.

### G1 — Fuse the dispersive passes into the main GPU kernels  ❌ closed

Measured dispatch counts (probe build, batch of 28 timesteps):

| Model | dispatches/timestep | barriers/timestep |
|---|---|---|
| PEC, no dispersion | 3 | 3 |
| PEC + dispersion | 5 | 5 |
| PML + dispersion | 5 | 6 |

Dispersion costs **2 extra dispatches per timestep**, not the four the shader
list suggests, and PML adds no dispatches at all (it is already fused) — only
barriers.

Fusing those two away cannot pay. The dispersive dispatches cover only the
sparse cell list, so they are small; what fusion would save is 2 barriers and
2 launches per timestep. What it would cost is a per-cell membership test in
the main kernel, which runs over the *entire* grid — either a full-size lookup
array (extra traffic on every cell, including the ~87% that are not
dispersive) or a hash. Adding whole-grid traffic to remove two barriers is
the wrong trade on hardware that is already at its bandwidth roofline.

### G2 — Dispatch and barrier count on small grids  ❌ closed

Answered by the same measurement: the per-timestep pipeline is **3 dispatches
and 3 barriers** for a plain model, rising to 5 and 6 with dispersion and PML.
That is already minimal — there is no dispatch bloat to remove. The earlier
suspicion that several active extension families would multiply dispatches is
wrong, because UPML is fused and the rest are sparse-list kernels.

### G3 — `volt_flux` traffic in the PML path  ❌ closed

Structural, not measurable away. A UPML cell's fused voltage update reads and
writes both `volt` and `volt_flux` (3 components each), roughly doubling
per-cell field traffic versus a Yee cell. The flux array is inherent to the
UPML formulation and the flux-swap needs the previous value, so the traffic
cannot be removed without changing the algorithm. After `00ba1c6` made the
coefficients cheap, this *is* the PML cost — it is a floor, not an
inefficiency.

---

## 3b. Status of the candidates above

Every candidate in §3 was either implemented or closed on evidence, and none of
those closures is reopened by the i9-13900K re-verification:

- AVX2 — at the traffic floor. The update kernels (`574f714`), and the
  dispersive extension (§A1b), are both bandwidth-bound with no addressing
  or arithmetic overhead left to remove.
- GPU — the Yee kernel is at roofline (§1), the per-timestep dispatch *count*
  is already minimal (G2), and the two remaining structural costs (PML flux
  traffic, dispersive passes) are inherent to their algorithms (G1, G3).

Both GPU closures were re-tested directly on the i9-13900K rather than taken on
trust, and both hold: the Yee kernel runs at 89% of a *measured* bandwidth
ceiling (§4.7/C4), and an extra dispatch plus barrier is free to within noise,
so dispatch count is not merely minimal but not a lever at all (§4.7/C3).

The one claim that did not survive was not a candidate at all but an aside in
§1 — that the runtime thread auto-tune needs no help. It needed a lot; see
§4.2. **New candidates opened by the i9-13900K measurements are in §4.7. All are
now resolved; the one that pays is C2 — temporal blocking of the AVX2 sweep,
validated at 3–4× with a bit-identical reference schedule, and awaiting
its engine port.**

---

## 4. Re-verification on the i9-13900K (2026-09-04)

The i7-6700K was believed gone when this section was written; it was not
(see the host list above, and PERFORMANCE.md, which re-measures it on
2026-09-07). This section's machine is the same i9-13900K the earlier GPU-only
measurements used, now measured with its CPU and integrated UHD 770 as well, so
the GPU and CPU numbers come from one machine and the two GPUs are directly
comparable.

Every number below is a fresh measurement on the i9-13900K. Benchmark drivers are
unchanged: `python/Tests/benchmark_avx2_engine.py` and
`python/Tests/benchmark_gpu_engine.py`, 300–400 timesteps, PEC unless stated.

### 4.1 The GPU results reproduce exactly

| Grid | cells | doc value | this run | |
|---|---|---|---|---|
| 256×224×224 | 12.8 M | 5115 MC/s | **5126 MC/s** | reproduced |

That is 0.2% apart on the headline `fbd95a6` figure, which retires any doubt
that §1's GPU table describes this hardware. The roofline argument holds
unchanged: 5126 MC/s × ~80 B/cell ≈ 410 GB/s against a 512 GB/s peak.

Full sweep, both GPUs, same fixture:

| cells | RX 6800 | UHD 770 | AVX2 MT (8 threads) |
|---|---|---|---|
| 64³ = 0.26 M | 9605 MC/s | 497 MC/s | 1851 MC/s |
| 96³ = 0.88 M | 14397 MC/s | 591 MC/s | 3449 MC/s |
| 128³ = 2.1 M | 16165 MC/s | 575 MC/s | 1881 MC/s |
| 160×128×192 = 3.9 M | 16888 MC/s | 510 MC/s | 887 MC/s |
| 224³ = 11.2 M | 5284 MC/s | 506 MC/s | 619 MC/s |
| 256×224×224 = 12.8 M | 5181 MC/s | 506 MC/s | — |
| 320×288×288 = 26.5 M | 5368 MC/s | 501 MC/s | — |

Three things to read off it:

- **The Infinity Cache cliff is between 3.9 M and 11.2 M cells**, exactly where
  the ~24 B/cell of field state crosses 128 MB. Above it the GPU sits flat at
  ~5.2 GC/s out to 26.5 M cells — the roofline regime, and stable there.
- **The UHD 770 is flat at ~0.5 GC/s regardless of grid size**, i.e. it is
  bounded by the same DDR5 the CPU uses and is beaten by the AVX2 engine on
  every grid up to 3.9 M cells. It is useful as a portability check on a
  second Vulkan driver (ANV rather than RADV) and for nothing else. Device
  selection already prefers the discrete GPU, which was confirmed here.
- **PML costs 15–20% on the RX 6800**: 4334 vs 5126 MC/s at 256×224×224, 9718
  vs 16888 MC/s at 160×128×192. Consistent with the 2.0× per-PML-cell figure
  in §2 once the PML's share of the volume is accounted for.

### 4.2 The AVX2 thread auto-tune was badly wrong on a many-core host

This is the one place the document was actively misleading. §1 asserted that
"the runtime auto-tune already converges to 3–4 threads unassisted." On the i7-6700K
that was true. On the i9-13900K the same code cost **43%** of throughput.

The old search hill-climbed inside `NextInterval()`, which `openems.cpp` calls
only when four seconds of wall time have elapsed, adding one thread per call
from a start of one. Reaching the i7-6700K's optimum of 3–4 takes a few intervals.
Reaching the i9-13900K's takes 44 s, during which the run is nowhere near peak:

| run length | old auto-tune | hand-pinned best | new calibration |
|---|---|---|---|
| 400 TS (~8 s) | 507 MC/s | 884 MC/s | **759 MC/s** |
| 20000 TS (~90 s) | 843 MC/s | 875 MC/s (8 thr) | **896 MC/s** |

`fd555ee` replaces it with a search that scores candidates on real timesteps
inside `IterateTS()` — legitimate because thread count cannot change results —
walking a geometric ladder and then bisecting around the winner. It settles in
about 250 timesteps rather than 44 s. The rewrite is worth roughly nothing on
the i7-6700K and 1.5× on the i9-13900K, which is the whole point.

One implementation detail that mattered more than expected: the first batch
after each thread respawn must be discarded. Fresh threads start with cold
private caches, which penalises precisely the wide configurations under
judgement; without the warm-up the search chose anywhere between 4 and 12
threads run to run, and with it, 8 every time.

### 4.3 Thread scaling, and why it rolls off

| threads | 1 | 2 | 4 | 6 | 8 | 12 | 16 | 20 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|
| MC/s | 486 | 706 | 814 | 865 | 884 | 884 | 707 | 673 | 677 | 702 |

Single-threaded `avx2` (not `avx2-multithreaded` at width 1) reaches
**535 MC/s**, up from the i7-6700K's 365.

The 20% collapse past 12 threads is *not* an engine defect and not a P-core /
E-core scheduling artifact. A plain OpenMP STREAM-Triad on the same host has
the same shape:

| threads | 1 | 2 | 4 | 6 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| GB/s | 28.8 | 33.2 | 44.8 | 48.2 | **48.9** | 45.0 | 41.8 | 39.3 | 39.3 |

Both peak at 6–8 and both lose ~20% by 16. The memory controller, not the
engine, sets this curve. There is nothing to fix — only a search that must not
walk into the far side of it, which §4.2 now provides.

### 4.4 How much a 36 MB L3 is worth

At a fixed 8 threads, throughput against grid size:

| cells | 0.26 M | 0.88 M | 2.1 M | 3.9 M | 4.1 M | 11.2 M |
|---|---|---|---|---|---|---|
| MC/s | 1851 | **3449** | 1881 | 887 | 862 | 619 |

A 96³ grid runs **3.9× faster per cell** than a 160×128×192 one. The field
state at 96³ is a few tens of MB and lives in the 36 MB L3; past that
everything streams from DDR5. The i7-6700K's 8 MB L3 could not show this — its
in-cache regime was too small to be interesting — which is why §1 concluded
"memory-bound" full stop and stopped there.

Read this as an upper bound on what removing DRAM traffic would buy, not as
evidence that traffic is currently being wasted. §4.5 measures the traffic and
shows it is not: the out-of-cache regime really is at the DRAM ceiling, so the
3.9× is only reachable by eliminating traffic (§4.7/C2), not by rearranging
it (§4.7/C1).

### 4.5 Measured DRAM traffic — §1's 80 B/cell was too high, its conclusion was not

§1 and §A1b both rest on an assumed ~80 B/cell/timestep that had never been
measured. `perf_event_paranoid` is 2 here so the `uncore_imc` CAS counters are
out of reach, but the *core* PMU is not, and `longest_lat_cache.miss`
differenced between two run lengths isolates the stepping phase cleanly
(500 vs 2500 timesteps, 8 threads pinned to the P-cores, 160×128×192):

| | 500 TS | 2500 TS | difference |
|---|---|---|---|
| L3 misses | 1.041e9 | 4.902e9 | 3.861e9 over 2000 TS |
| cell-timesteps | | | 7.864e9 |

That is **0.491 L3 misses per cell-timestep = 31.4 B/cell of DRAM fills.** The
ideal is 24 B/cell — read each cell's `volt` and `curr` once — so the sweep is
already within 31% of perfect read locality, with the neighbour accesses
absorbed by cache as intended. Add the ~24 B/cell of dirty writeback that a
94 MB working set must eventually push to DRAM and the total is **~55 B/cell**,
which at 873 MC/s is **~48 GB/s against the 49 GB/s STREAM ceiling.**

So: the constant in §1 was wrong by ~45%, and the conclusion it was used to
support is right anyway. The multithreaded AVX2 engine is at the DRAM roofline
on the i9-13900K, not merely near it.

### 4.6 GPU memory footprint — 2.8× more than the model needs

Not a throughput finding, but it sets the largest grid the card can hold, which
is the GPU's hard limit. Sampling `mem_info_vram_used` / `mem_info_gtt_used`
through a run with no dump boxes at all:

| grid | field buffer | VRAM before | VRAM after | host before | host after |
|---|---|---|---|---|---|
| 224³ | 128 MB | 803 MB | **288 MB** | 657 MB | **142 MB** |
| 320×288×288 | 303 MB | 1869 MB | **653 MB** | 1533 MB | **318 MB** |

The gap was exactly `DUMP_RING × 2 × m_fieldBufSize` in each of VRAM and host
memory — the async field-download ring from `ea82f59`, allocated for
simulations that never download a field. `RunFDTD()` decides whether to arm
that pipeline by scanning the `ProcessingArray` for a `ProcessFields`, and the
scan also matched `ProcField`, the bare `ProcessFields` registered a hundred
lines earlier purely to give the energy estimate a processing cadence. Fixed in
`05ce5f0`.

Two details worth keeping:

- **It only bit grids above the BAR aperture.** Below it the field buffers are
  host-mapped directly and `EnsureDumpBuffers()` bails out at its
  `m_voltMapped && m_currMapped` check. So the waste scaled precisely with the
  models where VRAM is the binding constraint, and never showed up on the
  small fixtures.
- **Throughput was untouched** (5633 vs 5663 MC/s at 224³). The buffers were
  allocated and then sat idle, which is why nothing in §1's timing work ever
  pointed at it. Footprint needs its own measurement; speed benchmarks will
  not find this class of bug.

### 4.7 Open candidates (new, from the i9-13900K)

#### C1 — Spatial blocking of the AVX2 sweep  ❌ closed on measurement

This was the obvious reading of §4.4: a 3.9× in-cache/out-of-cache gap looks
like scattered access waiting to be tiled. §4.5 closes it. Read traffic is
already 31.4 B/cell against a 24 B/cell floor — there is no scattered-access
waste to recover, because the x-line decomposition and the y–z plane sweep
already give the stencil the locality it needs. Tiling would rearrange traffic
that is not being wasted.

The 3.9× is simply what disappearing off DRAM buys, and reaching it needs
traffic *elimination*, not rearrangement. That is C2.

#### C2 — Temporal blocking of the AVX2 sweep  ✅ **validated, in the engine, and carrying every extension**

The only lever left on the CPU side, and the largest number in this document.
A trapezoidal scheme that advances a cache-resident tile through *k*
consecutive timesteps before moving on divides DRAM traffic by roughly *k*.
§4.4 bounded the prize at 2×–3.5×; measurement beats the bound.

**The vehicle.** `python/Tests/bench_temporal_blocking.c` reproduces the
*access pattern* of `Engine_AVX2_Multithread` — two arrays of three components
in N-I-J-K layout with z contiguous, the E pass reading H at
(x,y,z),(x−1,y,z),(x,y−1,z),(x,y,z−1) and the H pass the mirror image — but not
its physics. Values are meaningless; only the loads, stores and their order
matter, and those set cache behaviour. Two things say it is faithful: its flat
baseline is **920 MC/s** against the engine's 887, and its DRAM traffic is
**31.0 B/cell** against the engine's measured 31.4 (§4.5). Both within ~4%.

**Traffic — the falsification criterion asked for ~4×:**

| schedule | B/cell/timestep | |
|---|---|---|
| flat (what the engine does today) | 30.95 | |
| trapezoidal, W=72, k=24 | **2.40** | **12.9× less** |

**Throughput, 8 threads, best (W,k) per grid:**

| grid | field state | flat | blocked | | best (W,k) |
|---|---|---|---|---|---|
| 96³ | 21 MB | 2915 MC/s | — | **skip: already fits L3** | — |
| 160×128×192 | 94 MB | 864 MC/s | **3675 MC/s** | 4.25× | (72,24) |
| 224³ | 270 MB | 652 MC/s | **2270 MC/s** | 3.48× | (32,8) |
| 320×288×288 | 637 MB | 586 MC/s | **1679 MC/s** | 2.86× | (18,8) |

It helps at every thread count, and it removes the roll-off past 8 threads that
§4.3 traced to the memory controller — once the sweep is not DRAM-bound, extra
threads stop fighting each other:

| threads | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| flat | 562 | 753 | 846 | 925 | 756 | 687 |
| trapezoidal (72,24) | 824 | 1402 | 2410 | **3642** | 2866 | 2812 |
| gain | 1.47× | 1.86× | 2.85× | 3.94× | 3.79× | 4.09× |

**The schedule is verified, not just plausible.** `bench_tb verify` runs the
trapezoid and the flat sweep from the same pseudorandom initial state and
compares every element: **bit-identical** across W ∈ {24,26,32,72},
k ∈ {4,8,12,24}, threads ∈ {1,4,8}, and tile widths that do not divide the
domain. That property is what the engine port has to preserve, and the
benchmark is the reference implementation for it.

Two geometry details cost real debugging time and are worth carrying over:

- **Domain edges must be held, not sloped.** The trapezoid narrows because data
  outside the tile is at the wrong time; at x=0 and x=NX−1 there is no outside
  (the update clamps), so sloping there leaves edge cells un-advanced.
- **The wedge's E range is not symmetric with the core's.** The two passes must
  partition the domain — every cell updated exactly once per timestep. A
  narrowing core leaves gaps of width 2t+1 in E but 2t+2 in H, so the widening
  wedge sweeps E on `[An+1, Bn)`, one cell narrower per side than the
  natural-looking `[An, Bn+1)`. The symmetric choice double-updates those cells
  and was the first version's bug.

**Guards any implementation needs, both measured:**

- **Turn it off when the grid already fits L3.** At 96³ there is no traffic to
  remove; the overlapped-tiling prototype lost 32% there. The crossover is
  where field state (24 B/cell) exceeds ~36 MB.
- **Derive the tile width from the cross-section.** What must fit cache is
  `W × Ny × Nz × 24 B`; the optimum lands at L3/4–L3/2, so a fixed W is wrong.
  W=32 is right at 224³ and 27% off at 320×288×288, where W=18 wins. `k` also
  has an optimum (k=24 at 160×128×192, k=32 already worse), and `W ≥ 2k` is a
  hard constraint of the geometry.

**The engine prototype.** `Engine_AVX2_Multithread` now carries this schedule,
opt-in via `OPENEMS_AVX2_TEMPORAL_BLOCK=<k>` (or `<k>:<W>` to set the tile
width by hand). Measured on the real engine, 8 threads, 1200 timesteps:

| grid | flat | blocked | |
|---|---|---|---|
| 160×128×192 | 873 MC/s | **2805 MC/s** | 3.21× |
| 224³ | 613 MC/s | **2171 MC/s** | 3.54× |
| 256×224×224 | 610 MC/s | **2194 MC/s** | 3.59× |

Short of the microbenchmark's 4.25×, as expected: the real kernel also streams
a compressed operator index and is therefore not quite as purely
bandwidth-bound as the proxy.

**Bit-identical to the flat engine**, checked by dumping fields on a full-x
slab (the axis the trapezoid tiles, so a boundary error shows there) and
comparing every element: identical across grids 160×128×192, 224³, 256×224×224
and a deliberately non-round 173×131×197; k ∈ {6,8,16}; 1, 4, 8 and 12 threads;
and both auto-derived and hand-set tile widths.

**The tile width is derived from the last-level cache**, not fixed. One tile of
both field arrays should be about one L3: the measured optimum was W=64 on a
grid with 0.56 MB x-planes and W=32 on one with 1.20 MB — 36 and 38 MB against
this host's 36 MB. Getting this wrong is expensive rather than fatal: at 224³ a
tile sized for the *other* grid (W=64, 77 MB) gives 1267 MC/s where W=32 gives
2249.

> **Superseded on multi-cache hosts.** "One L3" is read from
> `sysconf(_SC_LEVEL3_CACHE_SIZE)`, which is one cache *instance*. On a
> multi-CCX or multi-socket machine a full-machine run has several, and the
> derived width is short by that factor — worth up to 2× where it bites. §4.8.

> **Refined (2026-10-18).** "One L3" holds for these uniform grids, whose
> coefficient tables are a few KB. On a graded mesh the tables can be as large
> as the fields (122 MB on a δ/10 transmission line), and there half the L3 is
> the optimum. The engine now starts at half and widens back to the full cache
> to keep k where the tables are small. §4.9.

**It refuses rather than risks.** Blocking engages only when every active
engine extension reports `SupportsSlabApply()`. An extension that has not
implemented the slab hooks returns false — the default — and sends the whole run
back to the flat sweep, so a new extension is safe by omission rather than by
enumeration, and the startup line names every extension that refused rather than
just the first. It also declines when the grid already fits cache (measured a
32% loss there) or when there are too few x-lines to tile. Verified: a 96³ run
reports *"disabled, grid fits cache (20.25 MB of field state)"*, and a run with
steady-state detection named that extension (until 2026-11-04, §4.12).

The paragraphs above describe the prototype as first landed, when the excitation
was the only slab-capable extension. That is no longer the state; see below.

##### Full adoption: every extension, and what it cost

The prototype above engaged only when the excitation was the sole active
extension, which is almost no real simulation. All of them now carry the
schedule. Measured on the same host, 224³, 8 threads, 800 timesteps, k=8:

| boundary | extensions | flat | blocked | |
|---|---|---|---|---|
| PEC | excitation | 618 MC/s | **1933 MC/s** | 3.13× |
| Mur ABC | 6 × Mur + excitation | 561 MC/s | **1456 MC/s** | 2.60× |
| UPML | 6 × UPML + excitation | 232 MC/s | **687 MC/s** | 2.96× |

**The interface was missing two thirds of itself.** The blocked path called
`Apply2VoltagesSlab` and `Apply2CurrentSlab` only, and only on thread 0. The
flat path runs *six* hooks per timestep — `DoPre*`, `DoPost*`, `Apply2*` for
voltages and currents — barrier-separated, on every thread. So UPML, Mur, the
absorbing BC and Lorentz were not merely un-migrated: the hooks they use had no
slab counterpart at all. `Engine_Extension` now declares all six, with the
contract in the header, and `TrapezoidSweep` calls them in the flat path's
order.

**Three extensions carried state that only works if one call is one timestep.**
That assumption is true of the flat sweep and false of a blocked one, which
visits the same timestep once per tile:

- `Engine_Ext_LumpedRLC` rotated its `v_Vdn`/`v_Jn` history rings by pointer
  swap per call. Now indexed by absolute timestep.
- `Engine_Ext_TFSF` rebuilt a shared `m_DelayLookup` scratch buffer per call —
  a race between threads and the wrong timestep inside a block. Now private to
  the call. It also read one table entry its own loop never filled, working
  only because the voltage hook had filled it earlier in the same timestep.
- Mur and the absorbing BC gated on `m_Eng->GetNumberOfTimesteps()`, which only
  advances at block boundaries. They take the timestep from the argument now.

**The schedule had to change for boundary extensions, and this is the part that
was not obvious.** A core tile narrows from its *left*, so the tail tile
`[xL,NX)` has slab `[xL+t+1,NX)` at step t. If the tail is short that slab
reaches x=NX-1 after x=NX-2 has dropped out of it. Harmless for the Yee update,
whose stencil wants the neighbour one timestep behind — and fatal for Mur, which
reads the line one cell inward at the *same* timestep it writes the edge. With
NX=100, W=30, k=15 the tail tile is `[90,100)` and at t=8 the slab owning x=99
is `[99,100)` alone. `ConfigureTemporalBlocking` therefore folds a short tail
into its neighbour so the last tile is at least k+3 lines, which is what makes
the contract's guarantee — *a slab containing a domain edge also contains the
two cells inward from it* — true at both ends. Without it, Mur x-boundaries
could not be blocked at all, and Mur is the most common absorbing boundary
there is.

**Two extensions cannot be split across threads, for the same reason.** TF/SF
adds into the cells along the edges where two box faces meet, once per face, so
any partition that sends two faces to two threads has them read-modify-writing
one cell — a lost update, and a summation order that varies. Lumped elements
number in the tens of cells and are not worth the question. Both run their slab
hooks on thread 0, which is parity rather than a concession: neither ever
overrode the threadID-taking hooks, so the flat sweep already ran them there.

**The dispersive cell list had to be split by x, not by index.**
`Operator_Ext_ConductingSheet` pushes the *identical* position list into both of
its two orders, so `Apply2*` subtracts from one engine cell once per order.
Splitting each order's indices independently would put those two subtractions on
two threads — a race, and `v-a-b ≠ v-b-a`. Splitting x and deriving each order's
range from that keeps every cell on one thread in order 0,1,…, as the flat sweep
has it. The list is indexed through an x-sorted permutation built in the engine
extension, so the operator's parallel arrays are never reordered.

**Barriers turned out to be the thing worth optimising.** The blocked path pays
the flat path's per-extension barrier once per timestep *per tile* — 13
trapezoids here, so a 13× multiplier. With six boundary faces plus the
excitation that was 42 barriers per timestep per tile, and Mur was capped at
2.28× against PEC's 3.05× purely by that. Most of them were for hooks the
extension does not implement: a no-op call is free, the barrier after it is not.
`SlabHookMask()` declares which of the six an extension overrides and the engine
skips call and barrier together, cutting 42 to about 20 — worth **+14% on Mur,
+7% on UPML**, and the numbers in the table above.

**Steady-state detection was taken to be the one extension that cannot be
blocked**, and said so rather than falling back silently: every period it
integrates total energy over the whole grid, and a blocked schedule has no
moment mid-block at which the whole grid is at one timestep. That was too
strong. Each slab can sum its share of that energy as it passes the timestep,
and the check can wait for the end of the block; it does since 2026-11-04
(§4.12).

**The block depth is capped by the probe interval.** `openems.cpp` calls
`IterateTS(step)` with the number of timesteps until the next probe or dump, and
`BlockedWorker` clamps `k` to it, so probes and field dumps are correct with no
work on their side — the grid is globally coherent whenever one reads it. But
that ceiling is real: `openems.cpp:639` sets the probe interval to
`Nyquist/oversampling`, typically 5–25 timesteps. It happens to bracket the
measured optimum (k=6…24), which is why this is usable at all; a run oversampled
far past Nyquist will not reach the numbers above.

**Verified by `python/Tests/test_temporal_blocking.py`**, which runs each
configuration twice — flat and blocked — and compares E and H dumps
element-for-element. It asserts blocking actually *engaged* first: without that
the comparison passes vacuously every time an extension declines and the engine
falls back to the flat sweep. Covered: PEC, UPML, Mur, Drude, lumped RLC, local
absorbing BC, TF/SF plane wave, a combined case, and a sweep over
(k,W) ∈ {2:8, 4:16, 3:13, 6:20} × {1,3,4} threads. `OPENEMS_AVX2_TEMPORAL_BLOCK_FORCE=1`
lifts the two performance guards so this runs on a CI-sized grid.

**What is still not blocked**, and why:

- ~~**Steady-state detection**, as above — a property of the method, not a
  gap.~~ It was a gap, closed in §4.12.
- **An absorbing sheet in the interior, normal to x.** The partition boundary
  between a core and its wedge sweeps one x-line per timestep, so it eventually
  falls between the line the sheet writes and the line it reads. No tile sizing
  fixes that. A sheet on a domain face is fine.
- **The cylindrical engines**, which never reach this code path.

##### Remaining notes on the prototype

- **Thread calibration now runs on whichever schedule the run will use.** It
  had been calibrating on the flat path, which is the wrong constraint:
  §4.3's roll-off past 8 threads is DRAM contention, and blocking removes it.
  Calibrating blocked, the search picks **16 threads instead of 8** and auto
  thread selection reaches 2412 MC/s against the flat path's 876 — **2.75×**
  end to end, up from 2.44×. Still short of the 3.21× a hand-pinned thread
  count gets, because the calibration timesteps are themselves spent
  measuring.
- **It is opt-in and prints what it decided.** Nothing changes unless the
  environment variable is set.


#### C3 — A fixed per-timestep GPU cost  ❌ closed, the candidate was an artifact

**Withdrawn.** The claim was a marginal rate of 17.9 GC/s and a 12.6 µs
intercept, fitted from the §4.1 sweep. Both the fit and the data were bad.

The data: that sweep used 300-timestep runs, short enough that startup is a
visible share of the total. Re-measured at 4000 timesteps the 64³ point moves
from 9605 to 13380 MC/s — a 39% difference from run length alone.

The fit: a single line was fitted across what turn out to be **three** distinct
cache regimes, so the "consistent marginal rate" was an average over regimes
that do not share one. Clean sweep, 4000 timesteps, GPU timestamps:

| grid | cells | µs/TS | throughput | marginal rate vs previous |
|---|---|---|---|---|
| 48³ | 0.11 M | 12.19 | 9.1 GC/s | — |
| 64³ | 0.26 M | 19.59 | 13.4 GC/s | 20.5 GC/s |
| 80³ | 0.51 M | 32.65 | 15.7 GC/s | 19.1 GC/s |
| 96³ | 0.88 M | 60.36 | 14.7 GC/s | 13.4 GC/s |
| 128³ | 2.1 M | 162.89 | 12.9 GC/s | 11.8 GC/s |
| 160×128×192 | 3.9 M | 309.71 | 12.7 GC/s | 12.5 GC/s |

The marginal rate falls from 20.5 to 12.5 GC/s — it is not one line. Below
~0.5 M cells the 24 B/cell of field state fits the 4 MB L2; from there to
~4 M it is Infinity-Cache-resident; past ~5 M it is DRAM (§4.1). Each regime is
bandwidth-limited by its own cache level. Extrapolating a line through all
three back to zero produces an "intercept" that is an artifact of the regime
change, and fitting only the Infinity-Cache points gives an intercept of
**−10 µs** — small grids are faster than the asymptote, not slower.

**Measured directly instead.** An extra pipeline barrier plus an extra dispatch
were recorded into the per-timestep command buffer, using the excitation
pipeline with `excCount=0` so every invocation returns immediately and writes
nothing. Field evolution is bit-identical to the baseline, so the delta is the
pure marginal cost of a dispatch boundary:

| grid | baseline | +1 dispatch +1 barrier | delta |
|---|---|---|---|
| 64³ | 19.53 µs/TS | 19.51 µs/TS | −0.1% |
| 96³ | 60.58 µs/TS | 59.73 µs/TS | −1.4% |
| 160×128×192 | 311.73 µs/TS | 317.37 µs/TS | +1.8% |

A whole extra dispatch and barrier is **free to within run-to-run noise**.
There is no fixed per-timestep overhead to remove at any grid size, and G2's
conclusion is strengthened rather than qualified: not only is the dispatch
count minimal, dispatch count is not a lever on this hardware at all.

#### C4 — Is the GPU Yee kernel at roofline?  ✅ closed, yes

§1 is right; the doubt raised here was mine and it was wrong. It came from
importing the CPU's measured ~55 B/cell (§4.5) to the GPU, where it does not
apply: the CPU's two passes share a 36 MB L3 and get cross-pass reuse, the
GPU's do not at these grid sizes.

Counting the traffic from `update_voltages.comp` instead of assuming it — one
thread per cell, all three components:

| per cell, per pass | bytes |
|---|---|
| read `curr` (3 components; the six neighbour taps hit cache) | 12 |
| read `volt` (read-modify-write) | 12 |
| write `volt` | 12 |
| `opIdx` (8-bit specialization constant on this fixture) | 1 |
| compressed coefficients (6 unique sets, L2-resident) | ~0 |
| **per pass** | **37** |

The current pass is symmetric, so **74 B/cell/timestep**. Against a measured
ceiling — `clpeak --bandwidth` on this RX 6800 reports **477 GB/s** global
memory (float4; 459 GB/s for float) — the out-of-cache figure is

    5729 MC/s × 74 B = 424 GB/s = **89% of achievable bandwidth**

which is at roofline for a memory-bound kernel. Independent corroboration: the
comment in `update_voltages.comp` records that a 32-bit `opIdx` was measured at
"~10% of all memory traffic", and 2 passes × 4 B = 8 B against ~80 B/cell is
exactly 10%.

So §1's 80 B/cell was right *for the GPU* and wrong *for the CPU* — the
document used one constant for two engines whose cache behaviour differs. §4.5
stands for the CPU, §1 stands for the GPU, and they are not in conflict.

Corollary: `clpeak` also reports **48.7 GB/s** for the UHD 770, against the
CPU's 49 GB/s STREAM-Triad ceiling (§4.3). The iGPU and the CPU are the same
memory system to within measurement error, which is why §4.1's iGPU column is
flat at ~0.5 GC/s regardless of grid size.

#### C5 — Temporal blocking of the GPU sweep  ✅ **validated, 2.2× on out-of-cache grids**

The direct consequence of C4. If the kernel is at 88–89% of achievable
bandwidth, the only remaining move is to stop needing the bandwidth — which is
what C2 did on the CPU. `RecordSlabSweep()` transcribes
`Engine_AVX2_Multithread::TrapezoidSweep`; an x-slab is a contiguous linear
cell index on the GPU (x is the slowest axis), so the shader change is a
`gidBase`/`gidEnd` push-constant pair and a smaller dispatch, with no change to
the kernel body. Enabled by `OPENEMS_GPU_TEMPORAL_BLOCK=<k>`.

Measured 2026-09-07 (i9-13900K) and 2026-09-08 (i7-6700K), 224³, flat → blocked (k=16):

| device | PEC | PML_8 | |
|---|---|---|---|
| RX 6800 (128 MB Infinity Cache) | 5674 → **12319** | 4619 → **10212** | **2.2×** |
| UHD 770 (share of 36 MB CPU L3) | 520 → 601 | 323 → 347 | 1.08–1.16× |
| HD 530 (Gen9; i7-6700K) | 478 → 484 | 270 → 269 | 1.00× |

The RX 6800 result erases the cache cliff §4.1 documents: a 270 MB grid now
runs within 6% of what a 94 MB grid reaches *inside* the Infinity Cache. The
HD 530 gains nothing because it is compute-bound, not bandwidth-bound — which
is a useful negative control for the whole premise, since it is the one device
here where the roofline C4 established was never the binding constraint.

Bit-identical to the flat sweep on both drivers
(`python/Tests/test_gpu_temporal_blocking.py`, 11 tests). Mur, TF/SF, lumped
RLC and GPU probes had no slab path then and made the schedule refuse; they
have one since 2026-10-19 (§4.10), and the schedule is on by default.

Two things this opened, both recorded in [PERFORMANCE.md](PERFORMANCE.md) §5.5:
Vulkan exposes no last-level-cache size, so the tile target is a tunable
(`OPENEMS_GPU_TEMPORAL_BLOCK_MB`, default 64 MB discrete / 16 MB integrated),
and on a card whose real cache is larger than the default the guard engages on
a grid that was already cache-resident and **costs ~5%**. A device-ID table or
a startup micro-probe would fix it.

### 4.8 What five more hosts changed (2026-09-11)

The five servers were added to [PERFORMANCE.md](PERFORMANCE.md)
as a portability and generality check on §4's conclusions, drawn on the i7-6700K
and i9-13900K —
both single-socket parts with one monolithic last-level cache. Two of those
conclusions do not survive a machine shaped differently, and both are recorded
here because they are cheap to rediscover the hard way.

**The C2 tile width is derived from the wrong quantity.** §4.7/C2 sizes one
tile to one last-level cache, taking the size from
`sysconf(_SC_LEVEL3_CACHE_SIZE)`, and §4.4 above is the measurement that
justified it. Both are correct on a chip with one shared L3 and wrong on a chip
with several: `sysconf` reports one instance, while a run spanning *n* of them
has roughly *n* times that capacity, because the tile is split across cores in
x and each core's slab lands in its own cache. Measured on 224³ PEC, setting
the width by hand to the aggregate is worth **1.99×** on the Ryzen 9 3950X at
16 threads (4 CCXs), **2.01×** on the 2× Xeon Gold 6130 at 32 (2 sockets) and
**1.60×** on the 2× Xeon E5-2620 v3 at 12 (2 sockets) — and is a *loss* on every configuration whose threads fit inside a
single instance, which is why the defect hides at low thread counts. The
corrected widths also show the right quantity is aggregate capacity across the
hierarchy, not the L3 line alone: the 2× Xeon Gold 6130's non-inclusive L3 plus
1 MB of L2 per
core wants a 43.6 MB tile on a 22 MB socket. Full sweep in
`python/Tests/results/tile_width_sweep_2026-09-11.txt`; consequences in
PERFORMANCE.md §5.3. This is the same class of mistake as §4.7/C5's GPU tile
target above — a cache size treated as a constant rather than as a property of
the device and of how much of it the run is using.

**`987fcb7` does not do what §1.0 says it does.** Its guard is a CMake-time
test of the *build* machine's `CMAKE_SYSTEM_PROCESSOR` and compiler, so an
x86-64 binary built with any modern GCC always has `OPENEMS_ENABLE_AVX2=1` —
including the copy that then runs on a pre-AVX2 host, where the default engine
and every `--engine=avx2*` selection die with SIGILL before the first timestep.
The 2× Xeon E5-2630 is that host and all 30 fork rows of its matrix crash. The `#else`
branch carrying the "AVX2+FMA engine is unavailable on this platform; using
multithreaded engine" fallback is compiled out in exactly the builds that need
it. A runtime `__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")`
at engine selection is the fix, and `openEMS::SetupOperator()` now makes it: an
AVX2 engine, default or explicit, falls back to the multithreaded engine with a
one-line warning (`OPENEMS_FORCE_NO_AVX2=1` exercises the path on AVX2 hosts).
Not yet re-run on the E5-2630. PERFORMANCE.md §6.

### 4.9 Synchronisation of the blocked AVX2 schedule (2026-10-18)

C2's barrier note above ("barriers turned out to be the thing worth
optimising") was right, and it did not go far enough. On a production model, a
5.03 M-cell δ/10 transmission line with Mur on six faces and two lumped ports,
the blocked workers were 79–89 % busy, and about 185 times per timestep each
one went to sleep at a barrier. `OPENEMS_AVX2_SYNC_STATS=1` now measures this
directly. On the Ryzen 5 5600G at 10:20 tiles the old schedule ran **552
barriers per timestep**. Its threads spent **21 %** of their time waiting for
the slowest thread and another **27 %** just getting back onto a CPU after a
release, because `Barrier` parks every early thread in the kernel.

**Done** in `b39bd36`, all bit-identical to the flat sweep (`python/Tests/tblock_identity.py`):

- **Fewer barriers.** Two new `Engine_Extension` queries:
  - `SlabTouches(hook, startX, stopX)` — false skips the hook and its barrier
    for that slab. Implemented for Mur (faces normal to x), LumpedRLC,
    Excitation and UPML (each region's x-range).
  - `SlabHookWritesFields(hook)` — false lets consecutive hooks of that kind
    share a single barrier. True for Mur's pre/post hooks and LumpedRLC's pre
    hook.

  Both default to the old behaviour, so an extension that does not answer
  loses nothing. On the TL model this cut 552 → 189 barriers per timestep (Ryzen) and
  264 → 92 (i9-13900K). It is the largest single step, about +30 %.
  `OPENEMS_AVX2_TBLOCK_ELIDE=0` turns it off.
- **Even work split.** Slab updates are divided over threads by y-row (one
  contiguous range of memory) instead of by x-line, through
  `Engine_AVX2::UpdateVoltageRows()` / `UpdateCurrentRows()`. Worth +3–7 % at
  4–8 threads. The schedule model says x-line splitting wastes up to 40 % at 32
  threads, which is still to be measured. `OPENEMS_AVX2_TBLOCK_SPLIT=lines`
  turns it off.
- **Spinning barrier.** `HybridBarrier` (`tools/barrier.h`) arrives with one
  atomic and can spin before it sleeps. With `OPENEMS_AVX2_BARRIER_SPIN_US=100`
  the waking-up share falls from 14 % to 1 %, a further +16–19 %. Off by
  default.
- **A tile that leaves room for the coefficients.** On the TL model the
  hand-swept optimum was 40–60 % of the L3 on all three hosts, with a
  full-cache tile 1.2–1.9× slower. The reason is that its compressed
  coefficient tables (317 192 unique entries, 122 MB) are as large as its
  fields, which are all the tile accounting counts. On uniform 224³ grids
  (12 entries), a half-cache tile only forces k down (to 3–7), and is 17–38 %
  slower than a full-cache one that keeps k. The rule is now:
  - start at half the L3 the threads span;
  - if that is narrower than 2k and the tables are under 25 % of the field
    bytes, widen up to the full L3 to keep k;
  - beyond that, lower k to W/2.

  That is within 7 % of the best hand-set tile on all nine model/host pairs
  measured. The rule is fitted to those nine; D4 is what will show whether it
  holds.

Together, on the TL model, against each host's previous default: **2.0×** on
the i9-13900K (944 → 1758 MC/s) and on the Ryzen 5 5600G (552 → 1088), and
**1.6×** on the i7-6700K (293 → 467). Tables: PERFORMANCE.md §5.7. Raw output:
`python/Tests/results/tblock_sync_2026-10-18.txt`.

#### Next steps

- **D1 — Fewer barriers still.**
  - Mur faces with the same normal write disjoint planes, so their field-writing
    Apply hooks could share a barrier (y-min with y-max, z-min with z-max). That
    is 4 → 2 per E half-step in interior slabs.
  - UPML regions are disjoint boxes, so their hooks could share barriers among
    themselves too. That needs a "commutes with" declaration beside
    `SlabHookWritesFields()`; a 160×128×192 PML model still has ~120 extension
    barriers per timestep.
  - About 12–20 % of thread time is still spent waiting for the slowest thread,
    and it grows as tiles narrow.
- **D2 — Balance what is left.** Mur faces normal to z still split a slab over
  threads by x-line, and LumpedRLC runs on thread 0 alone. Both are small, but
  each one holds up a barrier.
- **D3 — Defaults.** Spin 100 µs beat 20 µs in 21 of 23 pairs, by 1–7 %.
  Before it becomes the default, measure it on a host with more runnable
  threads than cores (where spinning steals time from the thread being waited
  for), and judge the power cost of a waiting thread that shows as 100 % CPU.
- **D4 — The tile campaign.** Check the tile rule across models and hosts,
  and decide whether its tables-vs-fields switch should become a working-set
  estimate: field bytes plus the coefficient entries a plane actually
  references, plus extension state such as UPML's. Models: the five §5 models, the TL model, a thin grid with small
  planes (e.g. 600×96×96) and one with large planes (e.g. 160×320×320, where
  W ≥ 2k forces k down). Hosts: i9-13900K, i7-6700K, Ryzen 5 5600G, 2× Xeon
  Gold 6130 and 2× Xeon E5-2620 v3, at one socket and at both. Only
  `avx2-multithreaded`. The multi-socket hosts are also where the y-row split
  and barrier count should matter most, at 16–32 threads.
  `python/Tests/bench_tblock_sync.py` is the driver.
- **D5 — Re-run the matrix.** PERFORMANCE.md §5.2/§5.3/§8 predate all of this.
- **D6 — The GPU engine.** Find out whether these findings hold for
  `Engine_Vulkan`'s blocked schedule, and where it can win:
  - per-extension, per-tile synchronisation: pipeline barriers or dispatches
    per hook per tile;
  - its tile target, still a constant (PERFORMANCE.md §5.5);
  - its refusal of Mur boundaries (the TL model above cannot block on the GPU
    at all);
  - on this model the RX 7900 XTX ran at ~5700 MC/s, only 1.6× the RX 6800's
    ~3500 despite ~1.9× the memory bandwidth. The RX 6800 itself drops from
    ~5450 MC/s on a 2.3 M-cell grid to ~3500 here. Find out whether that is
    bandwidth, cache or dispatch.

  Answered in §4.10: synchronisation is not the GPU's problem, the cache is,
  and blocking is worth 2.1–3.2× on this grid once the GPU can block it.
- **D7 — The flat path.** It still barriers after every extension in all six
  phases, implemented or not: about 50 barriers per timestep on the TL model,
  but only ~6 % waiting. `SlabHookMask()`-style skipping would carry over.

### 4.10 The GPU engine on the TL model (2026-10-19)

D6 asked whether the CPU's synchronisation findings hold for `Engine_Vulkan`.
They do not. Tables are in PERFORMANCE.md §5.8 and raw output in
`python/Tests/results/gpu_sync_tblock_2026-10-19.txt`.

**Tools.**
- `OPENEMS_GPU_PROFILE_STAGES=1` and `OPENEMS_GPU_PROFILE_NO_BARRIERS=1`, both
  with `--gpu-profile`, time each stage of a flat timestep and the timestep
  without its barriers.
- Every compute barrier inside the timestep loop goes through
  `Engine_Vulkan::CmdComputeBarrier()`, which counts the barriers and can drop
  them.
- `python/Tests/bench_gpu_sync.py` drives both switches. It also sets the batch
  size, the compression budget and the blocking schedule per run, and pins the
  card through `DRI_PRIME`.

**Findings.**
- **Synchronisation costs ≤ 1.4 % on large grids.** Idle time between
  batches is ≤ 0.1 %, and the batch size does not matter. Small grids are the
  exception on the RX 7900 XTX: about 2 µs per barrier, so 64³ PEC runs 1.92×
  faster without its barriers.
- **The six Mur faces take 5.6–5.9 % of the TL model's timestep**, and 9–11 %
  on 224³. That is kernel work, not waiting.
- **The Yee kernels are bandwidth-bound**: 87 % of the RX 6800's peak and 76 %
  of the XTX's. The TL model's coefficients are per cell and almost all
  unique, so compressing the operator cannot help here.
- **Temporal blocking on this grid** (PEC variant, no ports or probes) is
  **2.1×** on the RX 6800 and **3.2×** on the XTX. It is bit-identical to the
  flat sweep on both cards.
- **The default tile throws most of that away.** The tile target counts only
  field bytes, while an uncompressed operator triples a cell's footprint, so
  the default reaches 1.67× and 1.22×.

#### Full temporal blocking on the GPU — G1–G5 done the same day

Everything is bit-identical to the flat sweep:
`python/Tests/gpu_tblock_identity.py`, with full-volume dumps and probe files,
covering seven synthetic cases and the TL model; tables in PERFORMANCE.md §5.8.

- **G1 — A tile that counts the coefficients.** ✅ The tile is sized from
  24 B of fields plus the operator's bytes per cell, and a compressed
  operator's shared tables come off the target once.
- **G2 — Mur.** ✅ A face normal to x runs whole, in the slab that holds both
  of its lines; the y- and z-normal faces are clipped through `xLo/xHi` push
  constants.
- **G3 — Lumped RLC and TF/SF.** ✅ Both entry lists are sorted by x, with
  prefix tables. TF/SF sorts by x and then by field index, so the runs of
  points sharing an edge cell stay whole. Local absorbing sheets as well: an x-normal sheet
  keeps its footprint whole because the tile edges are moved at least k lines
  away from it. The CPU schedule still refuses interior x-normal sheets, which
  this placement would also fix there.
- **G4 — GPU probes.** ✅ No work needed: the gather is recorded at the end of
  each batch.
- **G5 — On by default.** ✅ On a discrete GPU, when fields and coefficients
  exceed twice the tile target, with k = min(32, W/2).
  `OPENEMS_GPU_TEMPORAL_BLOCK=0` turns it off.
- **G6 — Per-slab cost.** ✅ for Mur.
  - Running the next step's Mur and RLC pre-voltage with the current update cut
    46 → 38 barriers per timestep.
  - Folding Mur's post and apply into the voltage update
    (`update_voltages_mur.comp`, `SetupMurFusion()`) cut 38 → 20. It is
    bit-identical to the three-phase path, and on the RX 7900 XTX it is worth
    +5 % blocked (2.84× flat by default, 3.03× at k=32 with 64 lines).
  - The rest is the two barriers every slab step needs, between the E and H
    updates. Only fewer, wider trapezoids reduce them.
  - The fused kernel is off whenever UPML, a dispersive material, TF/SF or an
    absorbing sheet could put work between Mur's post and apply. Those models
    keep the separate phases.
- **G7 — The flat path's Mur.** ✅ The same kernel: +1.3 % on the XTX, a wash on
  the RX 6800.
- **Device cache table.** Replaced by measurement (2026-10-20). The XTX
  prefers a tile just under its 96 MB cache (+8 % over the 64 MB default),
  while the RX 6800 is flat from 45 to 89 MB. A per-device target would need
  a table of cache sizes, because Vulkan exposes none. Instead,
  `--benchmark` measures the schedules on the model itself and prints the
  options for the fastest. On the TL model it finds k=32 W=68 on the
  RX 6800, 0.5 % ahead of the default (PERFORMANCE.md §5.9).
- **Integrated GPUs.** ✅ On by default since 0107f8e, with a tile of 80 % of
  one instance of the CPU's L3. The speed alone did not justify it, but DRAM
  traffic and energy did (PERFORMANCE.md §5.9):
  - UHD 770: 1.37× and −29 % energy per cell.
  - HD 530: the same speed on −6 % energy, its DRAM traffic down by a third.
  - Vega 7: 1 % slower; its 1 MB L2 cannot hold a 1.5 MB x-plane.
- **Still open, from the same work:**
  - Why the HD 530 loses 12–14 % on a flat run whose probe interval is below
    about 300 timesteps. Its clock stays at the maximum. The benchmark's
    3 s runs see this as a probe-dense model would.
  - ~~The AVX2 engine calibrates its thread count on the flat sweep, also
    when it then runs blocked, and `--benchmark` inherits that count.~~
    Fixed in §4.11.
  - Tiling in y and z, for GPUs whose cache is smaller than one x-plane.

Four races in the flat path were found and fixed on the way (PERFORMANCE.md
§5.8):

- a lumped port's RLC apply and excitation;
- the shared edges of Mur faces;
- unfused UPML against the Mur and dispersive passes that read the same cells;
- crossing absorbing sheets.

### 4.11 Threads and cores in `--benchmark`, and a portable CPU topology (2026-10-20)

Details in PERFORMANCE.md §5.10; raw output in
`python/Tests/results/cpu_threads_cores_2026-10-20.txt`.

- **`--benchmark` measures threads and cores.** ✅ For each set of cores
  (P, P+SMT, E, all; or physical and all), the thread count doubles until it
  stops gaining 1 %, on the default tile. Then the schedules run on the best.
  On the TL model: 8 P-cores at 18:37 on the i9-13900K (1706 MC/s, 4.06× flat),
  4 cores at 4:8 on the i7-6700K, 6 cores at 11:23 on the Ryzen 5 5600G.
  SMT lost on all three at the default spin.
- **`--cpu-cores`.** ✅ `all`, `physical`, `performance[+smt]`,
  `efficiency[+smt]`, also for the `multithreaded` engine.
- **Calibration on the blocked schedule.** ✅ Undone by 4bfcd74 without
  notice and restored.
- **A warning** when a run leaves to defaults what `--benchmark` sets. ✅
- **`tools/cpu_topology`** replaces hwloc and `sysconf`: sysfs on Linux,
  `GetLogicalProcessorInformationEx` on Windows, `sysctl` and QoS classes on
  macOS. ✅ Tested on Linux x86, Linux ARM64 (QEMU), Windows (Wine) and fake
  sysfs trees. macOS is untested.
- **Still open:**
  - D2: weighted shares, so that P- and E-cores can work together.
  - D3: the barrier spin. It decides whether SMT pays off.
  - A set per L3 instance, e.g. for a 7950X3D's V-cache CCD.
  - macOS: a build and a run on a real Mac.

### 4.12 Steady-state detection inside the blocked schedules (2026-11-04)

C2 and C5 left one extension out, steady-state detection, and with it every
model driven by a sine: the whole benchmark ran flat. Details and the
analysis in `plans/STEADY_STATE_TEMPORAL_BLOCKING.md`, numbers in
PERFORMANCE.md §5.11.

- **The energy of a check is summed per slab.** Each slab passes the
  timestep of a check once, holding there the flat sweep's values, and the
  slabs of a timestep partition the grid. The AVX2 slab hook sums its rows
  (x,y), the Vulkan slab sweep its x-planes, and the check runs at the end of
  the block (AVX2, `Engine_Extension::SlabBlockEnd()`) or of the batch
  (Vulkan). The flat paths sum the same rows and planes in the same order:
  identical verdicts and stop timesteps, flat or blocked, any (k, W) and
  thread count (`test_temporal_blocking.py`, `test_gpu_temporal_blocking.py`).
- **The probes are recorded per slab**, into a ring of three periods indexed
  by the absolute timestep, so the check can wait while the block records on.
- **Cost**: one barrier per period on the AVX2 engine, and a hook with its
  barrier in the three slabs per timestep that hold a probe line; on the GPU
  one plane-sum dispatch and a barrier per slab at a check timestep.
- **Measured** against `4b88dc7`, which ran these flat (PERFORMANCE.md
  §5.11): AVX2 at k=13 antenna M 163 → 739, inhomogeneous M 228 → 882,
  ideal L 598 → 1727 MC/s; Vulkan by default 4110 → 7944, 4425 → 9457,
  5633 → 11392 MC/s.
- **On the way:**
  - the verdict is read after every batch instead of every 4 s (upstream
    `277ade0`/`eb1c5ee`, with `--exact-endcriteria`);
  - the probe positions are fixed (upstream `01efbdc`);
  - the excitation and the detection use one rounded period; with a period
    of 0.2 ns they disagree for 449 of the periods below 10000 timesteps, the
    benchmark's cylindrical case among them;
  - the GPU's energy of a check is that of its own timestep, which fixes a
    phase error that kept GPU sine runs from stopping;
  - the AVX2 fast energy sums rows in double (its single float accumulator
    was 2e-5 off on a 40³ grid);
  - a call of one timestep (Dirac, step) runs flat on both engines.
- **Found, not fixed:**
  - ~~The Vulkan blocked schedule is not deterministic with conducting sheets:
    on the benchmark's antenna case two blocked runs differ by 7e-4 and
    blocked from flat by 3e-3, also on `4b88dc7` with a Gaussian pulse. With
    the sheets as metal it is bit-identical. Before, sine runs were spared
    by the refusal; now the antenna case blocks by default at M and L.~~
    Fixed in `89f500d`: the slab path recorded the dispersive apply
    dispatches without the barriers the flat path has between them, and a
    sheet's two orders hold the same cells (so do a multi-pole material's,
    and a Lorentz and a Debye material's on their common face). Found with
    it, `9e22444`: the blocked separate UPML path still ran UPML's pre pass
    first, which 029a629 had moved last on the flat one only.
  - The fork goes NaN on a series lumped RLC read from XML (`4b88dc7` too;
    upstream's binary runs the file), so `test_lumped_rlc_and_port` fails
    under the XML harness the tests use now.
  - The benchmark's cylindrical case grows without bound over a long run
    (energy ×10³⁵ in 120 000 timesteps); `4b88dc7`'s verdict stays at 0 dB
    there too.

---

## 5. Method notes

Hard-won test-harness lessons; all three have produced false results in this
repo.

- **RPATH is absolute.** Copying a build tree does not give an independent
  binary — the copy still loads the original's `libopenEMS.so`. A/B builds
  need separately *configured* build directories.
- **`LD_LIBRARY_PATH` set from inside Python has no effect** on the running
  process's linker. The Cython extension resolves `libopenEMS.so` through its
  baked-in RUNPATH. Re-exec (see `_bootstrap_local_openems_runtime` in
  `test_gpu_engine.py`) or drive the `openEMS` CLI on a `Write2XML()` file.
- **The Python module links the *installed* library**, so `--engine gpu` is
  unavailable through it unless the installed build has `WITH_GPU=ON`.
- **Degenerate meshes make A/B tests pass vacuously.** Grading with `t**1.7`
  from zero produces near-zero first cells, collapsing the timestep so nothing
  propagates and every field dump is identically zero. Use geometric grading
  and assert a non-zero peak.
- **An intermittently failing tolerance check is a race until proven
  otherwise.** `test_tfsf_field_probes` had been failing about one run in
  three, which reads like a tolerance that wants loosening. It was not: six
  runs of the same TF/SF model gave six different GPU results, while every CPU
  engine gave one identical result across six runs each. Hashing whole probe
  traces across repeated runs of *one* engine — rather than comparing two
  engines to each other — is what separates "these two disagree" from "this one
  disagrees with itself", and it took three commands. The bug was a non-atomic
  `+=` on field elements shared by two TF/SF box faces (`b87a1f7`).

  The rest of the engine was then audited the same way rather than by
  inspection. Only three shaders accumulate into shared field memory at all
  (`apply_excitation`, `tfsf_voltage`, `tfsf_current`), and the excitation's
  index list has no duplicates. Repeating the hash-across-runs check on a plain
  model and on the dispersive, lumped-RLC, conducting-sheet and local-ABC paths
  found all of them reproducible, so TF/SF was the only one.
- **A GPU A/B that silences the source measures the source, not the change.**
  This one cost most of an afternoon and nearly produced a fake 26% win. The
  first C3 experiment removed the excitation dispatch and measured
  318 → 244 µs/TS, seemingly proving that one dispatch boundary was worth 26%.
  It proved nothing of the sort: with no excitation the fields stay *exactly
  zero*, and an all-zero grid runs ~28% faster than the identical kernel on
  real data. Controls, all on the same 160×128×192 fixture at 4000 timesteps:

  | configuration | µs/TS |
  |---|---|
  | dispatch present, real fields | 318.4 |
  | dispatch present, **amplitudes zeroed** | 241.3 |
  | dispatch absent, fields zero | 235.7 |

  The dispatch is worth 5.6 µs; being zero is worth 77 µs. Ruled out as
  explanations: iteration count (both ran exactly 4000), clock and power
  (2174 vs 2205 MHz, 202 vs 201 W sampled through the run), excitation
  waveform (gaussian and sinusoidal agree to 1%), excitation *size* (19,140
  points vs 110 changes nothing), and denormals — scaling all amplitudes by
  1e10, which pushes every value out of denormal range, leaves the time
  unchanged at 318 µs. What is left is datapath switching activity: zero
  operands flip almost no bits, so the same instruction stream costs far less
  energy and sustains more throughput inside the same power envelope.

  **Rule: never A/B a GPU change by disabling the excitation, and check that
  both arms carry real field data.** To isolate a dispatch's cost, add a no-op
  dispatch (the excitation pipeline with `excCount=0`) to the arm that lacks
  one, so field evolution stays bit-identical on both sides.
- **Fit within one cache regime or not at all.** The withdrawn C3 intercept
  came from a straight line through L2-resident, Infinity-Cache-resident and
  DRAM-resident grids at once. Check where the working set (24 B/cell) sits
  relative to L2 (4 MB), Infinity Cache (128 MB) and VRAM before fitting
  anything, and use run lengths long enough that startup is negligible —
  300-timestep runs put the 64³ point 39% off its 4000-timestep value.
- **`uncore_imc` is unavailable, the core PMU is not.** `perf_event_paranoid`
  is 2 on the i9-13900K and there is no passwordless sudo, so the memory-controller
  CAS counters that would give DRAM traffic directly cannot be read. The core
  PMU *does* work unprivileged for one's own process:
  `longest_lat_cache.miss` is the usable substitute (§4.5). On this hybrid CPU
  perf splits every event into `cpu_core/` and `cpu_atom/` variants — pin with
  `taskset -c 0-15` and count only `cpu_core/`, otherwise the atom counter is
  multiplexed down to a few percent enable time and silently scaled back up.
- **Difference two run lengths instead of trusting one.** A whole-process
  counter includes operator build, mesh setup and HDF5 writes, which on short
  runs are a large fraction of the total. Running 500 and 2500 timesteps and
  dividing the difference by 2000 isolates the stepping phase without needing
  to instrument anything.
- **Guard STREAM-style microbenchmarks against dead-code elimination.** The
  first cut of the bandwidth benchmark in §4.3 reported 14 *million* GB/s,
  because nothing read the output array and GCC deleted the loop. Consume the
  result and print it.
- **A dump-path A/B writes into page cache, not onto disk.** The §1.0
  async-writer measurement writes 5 GB per run and finishes in under three
  seconds, which no consumer NVMe sustains — the writes are buffered and the
  process exits before they are flushed. That is the right experiment for
  isolating HDF5 serialization and `memcpy` from the stepping thread, and the
  wrong one for claiming anything about storage. Do not run it on tmpfs
  (`/tmp` is tmpfs on the i9-13900K), where even the page-cache pressure disappears,
  and state which filesystem a dump number came from.
- **Speed benchmarks do not find footprint bugs.** §4.6 was 515 MB of VRAM
  allocated and never touched, invisible to every timing measurement in this
  document. Sample `mem_info_vram_used` and `mem_info_gtt_used` under
  `/sys/class/drm/card*/device` through a whole run and compare against
  `3 * N * sizeof(float)` per field buffer; anything unexplained is real.
  Sample continuously rather than at one fixed delay — allocation only reaches
  steady state after `CalcECOperator`, which takes 16 s on a 26 M-cell grid.
- **Cross-engine comparison is structurally blind to operator bugs.** Every
  engine routes through the same `Calc_ECOperatorPos`, so an indexing error
  there is invisible to all cross-engine tests (this is how the `81005f8`
  regression survived). Absolute-value tests on inhomogeneous models are the
  only guard.
