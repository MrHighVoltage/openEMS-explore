# Trapezoidal temporal blocking for the Vulkan engine

Brings to `Engine_Vulkan` the schedule that gave `Engine_AVX2_Multithread`
3.2–3.6× (`OPTIMIZATIONS.md` §4.7/C2): advance one x-slab through *k*
timesteps before moving on, so the working set the last-level cache must hold
is a slab rather than the whole grid.

## Why it applies to the GPU at all

§1 and §4.7/C4 concluded the Yee kernel is at roofline — 89% of a measured
477 GB/s on the RX 6800 — and that further gains must come from *reducing
traffic*, not from tightening arithmetic. That is exactly what this does. It
does not raise the roofline; it moves the kernel onto a different one.

The prize is bounded by the device's cache-vs-DRAM span, and that span is
device-specific. Measured on the i9-13900K:

| device | working set 2.7 MB → 270 MB | span |
|---|---|---|
| RX 6800 (128 MB Infinity Cache) | 13648 → 5664 MC/s | 2.41× |
| UHD 770 (share of 36 MB CPU L3) | 643 → 506 MC/s | 1.27× |
| AVX2 MT (36 MB L3) | 3449 → 619 MC/s | 5.6× |

The RDNA2 Infinity Cache is unusually large; a GPU with a conventional L2
falls out of cache at far smaller grids, so blocking applies to a *wider*
range of models there, not a narrower one. And the UHD 770's small span is
because it is compute/latency-bound rather than bandwidth-bound — a stronger
iGPU on the same DDR5 would be more bandwidth-bound and gain more. Neither
device is the general case; the mechanism is worth having on both.

## Status

**Done — the core sweep, PEC plus excitation, UPML and dispersive materials.**

- `gidBase`/`gidEnd` on `update_voltages.comp` and `update_currents.comp`. An
  x-slab is a *contiguous* linear cell index in both (x is the slowest axis),
  so this is a push-constant pair and a smaller dispatch, with no change to
  the kernel body.
- `ConfigureTemporalBlocking()` — tile edges, depth, the `W >= 2k` constraint,
  the short-tail fold, and the guards. It refuses by naming what refused.
- `RecordSlabSweep()` — a transcription of
  `Engine_AVX2_Multithread::TrapezoidSweep`, whose geometry is already
  verified. Keep the two in step.
- Excitation entries sorted by x at setup with a per-x prefix table, so a slab
  maps to a contiguous entry range (`entryBase`/`entryEnd` in `ExcPC`).
- `python/Tests/test_gpu_temporal_blocking.py` — bit-identical to the flat
  sweep across five (k,W) schedules including a folded tail, on both RADV and
  ANV. Asserts blocking *engaged* first, so it cannot pass vacuously.

Measured, 2000 timesteps:

| grid | device | flat | blocked | | energy |
|---|---|---|---|---|---|
| 224³ | RX 6800 | 5672 | **11956** (k=16) | 2.11× | — |
| 320×288×288 | RX 6800 | 5730 | **12203** (k=16) | 2.13× | 32.1 → **16.9** nJ/cell-update |
| 224³ | UHD 770 | 511 | **602** (k=6) | 1.18× | 42.8 → **35.9** nJ/cell-update |

The proxy that skipped the wedges reached 2.30×, so the wedges and the
narrowed slabs cost about 8% of the ideal — the correct schedule keeps most
of it.

Re-measured 2026-09-07 (i9-13900K) / 2026-09-08 (i7-6700K) in the full
fork-vs-upstream matrix (best of 3, no
probes or dumps), which is the version that reached PERFORMANCE.md §5.4:

| grid | RX 6800 | UHD 770 | HD 530 (i7-6700K) |
|---|---|---|---|
| 160×128×192 PEC (94 MB) | 13070 → 12950 (**0.99×**) | 524 → 611 (1.17×) | 368 → 381 (1.04×) |
| 224³ PEC (270 MB) | 5674 → **12319** (2.17×) | 520 → 601 (1.16×) | 478 → 484 (1.01×) |
| 160×128×192 PML_8 | 9462 → 9844 (1.04×) | 317 → 344 (1.08×) | 225 → 224 (1.00×) |
| 224³ PML_8 | 4619 → **10212** (2.21×) | 323 → 347 (1.08×) | 270 → 269 (1.00×) |

Three things that table says which the earlier one could not:

- The 224³ blocked figure is within 6% of what the 94 MB grid reaches *inside*
  the Infinity Cache, so the schedule very nearly erases the cache cliff.
- **The 94 MB row on the RX 6800 is a small loss**, and it is the tile-target
  default's fault, not the schedule's: the card's real cache is 128 MB, the
  discrete default target is 64 MB, so the guard engages on a grid that was
  already resident. With `OPENEMS_GPU_TEMPORAL_BLOCK_MB=128` it correctly
  refuses and flat throughput returns (13143 vs 12508 in a direct single-run
  A/B). This is the strongest argument yet for replacing the constant with a
  device-ID table or a startup micro-probe.
- A third device class was added: the **HD 530** gains nothing anywhere,
  because Gen9 is compute-bound rather than bandwidth-bound. It is a useful
  negative control — the one device here whose limit was never the roofline
  this work removes.

## Measured on a production grid (2026-10-19)

The 5.03 M-cell TL model of PERFORMANCE.md §5.7/§5.8 has an uncompressed
operator, 72 B per cell with its coefficients. With Mur, the lumped resistors
and the probes removed, it blocks at **2.1×** on the RX 6800 and **3.2×** on an
RX 7900 XTX, bit-identical to the flat sweep. The default tile reaches only
1.67× and 1.22×, because the tile target counts 24 B per cell. Hence item 0.

## Status 2026-10-19: done

Items 0 and 2–4 below are implemented, and so are local absorbing sheets. The
schedule is on by default for discrete GPUs (OPTIMIZATIONS.md §4.10,
PERFORMANCE.md §5.8). What remains is per-slab cost: barriers, which are 22 %
of the timestep on an RX 7900 XTX. The items are kept below as the record of
the plan.

## Remaining work, in the order that pays

### 0. Tile sizing that counts the coefficients

`ConfigureTemporalBlocking()` sizes `planeBytes` from volt+curr alone. Add the
operator's bytes per cell, and the per-cell state of any extension heavy enough
to matter. Measured optimum: real tiles of 45–89 MB on both cards, k=16–32.

### 1. ~~UPML~~ — done

It was the easy one here, unlike on the CPU. Fused into the main dispatch, one
thread per grid cell, PML membership derived per-cell from region metadata, so
it needed the same `gidBase`/`gidEnd` change and nothing else. The non-fused
`upml_pre_*`/`upml_post_*` fallback shaders are indexed in region-local
coordinates with lx slowest, so a global slab is a contiguous run there too
once clipped against each region's x-extent.

Bit-identical on both paths, and the two paths agree with each other — that
third test exists because the first two each compare a path against itself and
would both pass if the slab range were wrong in a way they shared.

### 2. The sparse cell lists — ~~dispersive~~, lumped RLC, TF/SF

Dispersive materials are done (`88032e4`). Lumped RLC and TF/SF are not.

All are flat index lists dispatched one thread per entry. Each needs the
same treatment the excitation got: sort by x at setup, build the prefix
table with `BuildExcitationSlabIndex()` (rename it), and add an entry range to
the push constants.

Two cautions carried from the CPU port:

- `Operator_Ext_ConductingSheet` pushes the *identical* position list into both
  of its two orders, so an index-space split puts two subtractions on one cell
  in an unpredictable order. Split by x and derive each order's range from
  that.
- The same holds across dispatches: two dispersive entries may hold one cell
  (a sheet's orders, a multi-pole material's, a Lorentz and a Debye material
  on their common face), and an apply pass rewrites all three components of
  each of its cells. Their apply dispatches need a barrier between them in
  every slab, as in the flat path; `88032e4` left it out, `89f500d` added it.
- TF/SF adds into cells where two box faces meet. `b87a1f7` already gave each
  field index to one thread; the slab ranges must preserve that, not reopen it.

### 3. Mur ABC

Per-face dispatches. The x-normal faces are a plane at fixed x (do nothing
unless the slab contains it); the y- and z-normal faces span x and need their
x loop restricted. Mur reads one cell inward at the *same* timestep it writes
the edge, which is what the short-tail fold in `ConfigureTemporalBlocking()`
already exists to guarantee — it is implemented but currently unexercised.

### 4. Probes and field dumps

`openems.cpp` calls `IterateTS(step)` with the number of timesteps to the next
probe or dump, and the block depth is clamped to it, so the grid is globally
coherent whenever anything reads it. Nothing to do for correctness — but the
ceiling is real: `openems.cpp:639` sets the probe interval to
`Nyquist/oversampling`, typically 5–25 timesteps, and k=16 is where the
measured optimum sits. A run oversampled far past Nyquist will not reach the
numbers above. GPU probes are gathered inside the batch, so they need slab
ranges before `m_hasGPU_Probes` can stop being a refusal.

### 5. Steady-state detection — ~~cannot be blocked~~ done 2026-11-04

Every period it integrates energy over the whole grid, and a blocked schedule
has no moment mid-block at which the whole grid is at one timestep. That was
read as a property of the method; it is not. `RecordSlabSweep()` samples the
probes of each slab and, at a check timestep, sums the energy of its x-planes
(`reduce_energy_planes.comp`) where the flat recorder does, and the host
evaluates the check once the batch is done. Flat and blocked give the same
verdicts and stop timesteps, and the old way of reading the energy, at
whichever batch boundary the verdict was asked for, had a phase error.
`plans/STEADY_STATE_TEMPORAL_BLOCKING.md`.

## Two things found while porting

- **A difference on the last x-line that turns out not to be observable —
  claimed too early, then falsified.** `TrapezoidSweep` passes the H range
  `[An, min(Bn, NX-1))` to the current-side extension hooks, while the flat
  path's `Apply2Current` runs the whole entry list (`stopX = UINT_MAX`). Read
  from the source that looks like a dropped update, and commit `99ac18a`'s
  message asserts it is one. It could not be reproduced. Two negative controls:

  - Clamping the Vulkan fused current dispatch to NX-1 leaves the UPML tests
    bit-identical. H on the last x-line is never written by the Yee update, so
    its flux recursion runs on zeros forever and produces zeros.
  - A soft magnetic excitation (`exc_type=2`) spanning x = NX-3..NX-1 gives a
    field of max |H| = 5.11, and the AVX2 flat and blocked sweeps are still
    bit-identical over it.

  So the range difference is real in the source and inert in behaviour, on
  every model that could be constructed for it. The Vulkan path still extends
  the last tile's current-side range to NX, which costs one x-plane in one tile
  and makes the union over tiles equal the flat dispatch by construction rather
  than by the last line happening to hold zeros — but it is insurance, not a
  bug fix, and `99ac18a`'s claim should be read as retracted.
- **Vulkan exposes no last-level-cache size.** The tile target is a tunable
  (`OPENEMS_GPU_TEMPORAL_BLOCK_MB`) with a per-device-class default: 64 MB
  discrete, 16 MB integrated. Both were measured optima on this host, and
  neither is portable knowledge. A device-ID table or a startup micro-probe
  would be better than a constant.
