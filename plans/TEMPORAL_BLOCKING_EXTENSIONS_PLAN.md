# Full adoption of trapezoidal temporal blocking

Closes the gap left open in `OPTIMIZATIONS.md` §4.7/C2: the blocked AVX2 sweep
is bit-identical and 3.2–3.6× faster, but engages only when *every* engine
extension reports `SupportsSlabApply()`, which today means the excitation and
nothing else. Every real simulation therefore falls back to the flat sweep.

## What the audit found

The engine's blocked path (`Engine_AVX2_Multithread::TrapezoidSweep`) calls
only `Apply2VoltagesSlab` / `Apply2CurrentSlab`, and only on thread 0. The flat
path runs **six** hooks per timestep — `DoPre*`, `DoPost*`, `Apply2*` for
voltages and currents — each barrier-separated, on all threads. So the slab
interface is not merely unimplemented by most extensions: it is missing four of
the six hooks they actually use.

| extension | hooks used | x-reach | slab pattern |
|---|---|---|---|
| Excitation | Apply2V, Apply2C | 0 | A (done, needs new signature) |
| UPML | DoPre/DoPost V+C | 0 (per-cell) | A |
| Dispersive (Drude) | Apply2V, Apply2C | 0 (cell list) | A, needs x-sorted index |
| Lorentz ⊃ ConductingSheet | + DoPreV, DoPreC | 0 | A, same index |
| LumpedRLC | DoPreV, Apply2V | 0 (cell list) | A + state-rotation fix |
| Mur ABC | DoPreV, DoPostV, Apply2V | 1, at a held edge | B or C |
| Absorbing BC | all six | 2, at a held edge | B or C |
| TF/SF | DoPostV, DoPostC | 0 (pure source) | C + scratch-buffer fix |
| SteadyState | Apply2V, Apply2C | global energy | ~~**cannot slab** — stays false~~ slab since 2026-11-04, see below |
| Cylinder, CylinderMultiGrid | — | — | unreachable from this engine |

Three slab patterns fall out, and every extension is one of them:

- **A — volume or cell list, x-decomposable.** Intersect the extension's own
  x-extent with `[startX,stopX)`, split the intersection across `threadID`.
- **B — plane anchored at a fixed x.** Do nothing unless the slab contains the
  plane's x-lines; otherwise keep the extension's existing thread split over
  its own axis.
- **C — plane or surface spanning x.** Restrict whichever loop axis maps to x,
  split the remaining axis across `threadID`.

Two extensions carry state that is *rotated once per timestep globally*, which
a blocked schedule breaks because different x-slabs reach a given timestep at
different moments:

- `Engine_Ext_LumpedRLC` rotates the `v_Vdn` / `v_Jn` pointer rings.
- `Engine_Ext_TFSF` rebuilds the shared `m_DelayLookup` scratch buffer.

Both must be re-expressed as functions of the *absolute timestep passed in*,
not of call order. That refactor is behaviour-preserving on the flat path and
is what makes the slab path possible at all.

Three extensions gate on `m_Eng->GetNumberOfTimesteps()` (`IsActive()`, and
TF/SF's signal lookup). Under blocking the engine's counter only advances at
block boundaries, so these must read the `numTS` argument instead.

## The other cap: the processing horizon

`openems.cpp` calls `IterateTS(step)` with `step = PA->Process()`, the number of
timesteps until the next probe or dump. `BlockedWorker` already clamps
`k = min(k, step)`, so probes and dumps stay correct without any change — the
grid is globally coherent at the end of every `IterateTS`. But it means the
achievable `k` is capped by the probe interval, which `openems.cpp:639` sets to
`Nyquist/m_OverSampling` — typically 5–25 timesteps. That happens to bracket the
measured optimum (k=6…24), so blocking is usable in real runs, but the ceiling
is real and must be documented rather than discovered.

## Plan

### Phase 0 — Make it testable before changing anything
1. `OPENEMS_AVX2_TEMPORAL_BLOCK_FORCE=1` bypasses the "grid fits cache" and
   "too few x-lines" guards, so bit-identity can be tested on grids small
   enough for CI instead of the ~48 MB the guard demands.
2. `python/Tests/test_temporal_blocking.py`, modelled on
   `test_upml_engines.py`: run each feature configuration twice — flat and
   blocked — and compare field dumps element-for-element. Configurations: PEC
   only; UPML; Mur ABC; absorbing BC; Drude; Lorentz; lumped RLC; TF/SF; and a
   combined one. This is the acceptance criterion for every later phase.

### Phase 1 — The interface (single-handed; everything else depends on it)
3. `Engine_Extension`: all six slab hooks, signature
   `(unsigned int startX, unsigned int stopX, int numTS, int threadID)`, with
   the contract documented in the header — writes confined to `[startX,stopX)`,
   reads confined to cells in that range at timestep `numTS` plus held domain
   edges, self-partitioning by `threadID`, no use of the engine's timestep
   counter.
4. `TrapezoidSweep`: call all six in the flat path's order (Pre backwards
   through `m_Eng_exts`, Post and Apply forwards — it is sorted descending by
   priority), on every thread, barrier-separated, exactly mirroring
   `Engine_AVX2_Multithread::DoPreVoltageUpdates` and friends.
5. Port the excitation to the new signature. Re-verify bit-identity: this
   alone must not regress the numbers already in `OPTIMIZATIONS.md`.

### Phase 2 — The extensions (parallelisable across disjoint files)
6. **UPML** (pattern A) — four hooks, scalar *and* the packed AVX2 path in
   `engine_ext_upml_avx2.cpp`. Highest risk: two implementations to keep in
   step, and the existing `test_upml_engines.py` must keep passing.
7. **Dispersive + Lorentz + ConductingSheet** (pattern A) — an x-sorted
   permutation over the cell list plus a per-x prefix table, built in the
   engine extension so the operator's parallel arrays are never reordered.
8. **LumpedRLC** (pattern A) — absolute-timestep indexing first, then hooks.
9. **Mur ABC + Absorbing BC** (patterns B/C) — `IsActive(numTS)`, and the
   loop-axis restriction that depends on which of the three axes is normal.
10. **TF/SF** (pattern C) — per-call delay lookup, plane restriction.
11. **SteadyState** — stays `false`, with a comment saying why (it needs total
    energy over a globally coherent grid, which a blocked schedule does not
    have mid-block) and a startup message naming it as the reason blocking is
    off, rather than a silent fallback. *Superseded 2026-11-04: each slab sums
    its rows of that energy as it passes the check's timestep, and
    `SlabBlockEnd()` runs the check at the end of the block
    (`plans/STEADY_STATE_TEMPORAL_BLOCKING.md`).*

### Phase 3 — Engine wiring
12. Replace the "the excitation and nothing else" comments with what is
    actually true, and print the slab-capable / not-capable split at startup.
13. Measure barrier overhead. The blocked path now runs `6 × nExtensions`
    barriers per timestep *per tile*, where the flat path runs them once per
    timestep for the whole grid — roughly a 4–10× multiplier on barrier count.
    If it costs more than a few percent, add a per-extension "no work in this
    slab" predicate so empty extensions cost neither the call nor the barrier.

### Phase 4 — Measure and document
14. Bit-identity across the full feature matrix, several `k`, several thread
    counts, and a non-round grid.
15. Throughput on real models (`MSL.py`, `Rect_Waveguide.py`, a dipole with
    PML) — the honest end-to-end number with PML and probes, not the PEC
    benchmark number.
16. Rewrite `OPTIMIZATIONS.md` §4.7/C2's "what the prototype does not yet do"
    into what it does, including the processing-horizon ceiling.

## Subagent split

Phase 1 is the contract everything else is written against, so it is done
first and alone. Phase 2 items 7, 9 and 10 touch disjoint files and are handed
to one subagent each with the contract text quoted verbatim. Items 6 and 8 —
UPML's dual implementation and LumpedRLC's state rotation — are the two with
real correctness risk and are done in-house. Phase 0's test harness runs as its
own subagent, concurrently with Phase 1.

---

## Status — complete

Every phase landed. `python/Tests/test_temporal_blocking.py`: 9/9, including a
sweep over (k,W) ∈ {2:8, 4:16, 3:13, 6:20} × {1,3,4} threads, all bit-identical
with blocking confirmed active. `test_upml_engines.py` 7/7 and `LumpedRLC.py`
6/6 still pass, so no refactor moved the flat path.

224³, 8 threads, 800 timesteps, k=8:

| boundary | flat | blocked | |
|---|---|---|---|
| PEC | 618 MC/s | 1933 MC/s | 3.13× |
| Mur ABC | 561 MC/s | 1456 MC/s | 2.60× |
| UPML | 232 MC/s | 687 MC/s | 2.96× |

### What the plan did not anticipate

- **The tail tile starves the high domain edge.** A core tile narrows from its
  left, so a short tail tile reaches x=NX-1 after x=NX-2 has left the slab —
  fine for the Yee stencil, fatal for Mur, which reads one line inward at the
  same timestep it writes the edge. Fixed in the schedule (fold a short tail
  into its neighbour, minimum k+3 lines) rather than by declining the face,
  because Mur x-boundaries are the common case. This was the single most
  valuable thing the audit turned up.
- **TF/SF cannot be split across threads at all.** Adjacent box faces share
  their edge cells and each face adds into them, so any face-wise partition has
  two threads read-modify-writing one cell. Runs on thread 0, which is parity
  with the flat path.
- **The dispersive cell list must be split by x, not by index.**
  `Operator_Ext_ConductingSheet` pushes the identical position list into both of
  its orders, so an index-wise split would put two subtractions on one cell onto
  two threads.
- **Barriers were the limiting cost, and the fix was not the one planned.** The
  planned "no work in this slab" predicate was unnecessary: most barriers were
  for hooks the extension does not implement at all. `SlabHookMask()` cut 42
  barriers per timestep per tile to ~20, worth +14% on Mur and +7% on UPML.

### Left undone, deliberately

- ~~Steady-state detection — integrates total energy over a globally coherent
  grid, which no blocked schedule has mid-block.~~ Done 2026-11-04, bit-identical
  verdicts and stop timesteps: `plans/STEADY_STATE_TEMPORAL_BLOCKING.md`.
- An absorbing sheet in the interior normal to x — the core/wedge partition
  boundary sweeps one x-line per timestep and eventually falls between the line
  written and the line read. Domain-face sheets work.
- The cylindrical engines, which never reach this code path.
