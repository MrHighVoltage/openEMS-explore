# Temporal blocking with periodic excitation and steady-state detection

A sine excitation adds steady-state detection to a run, and until 2026-11-04
both blocked schedules refused it, so temporal blocking was off in every case
of the benchmark (all of them use a sine). The analysis below, from before the
work, recommended (b) + (c); that is what was built. This section says what was
done, what turned out differently, and what is open. The analysis follows
unchanged, with its file:line references at `4b88dc7`.

## Status: done (2026-11-04)

Commits, on top of `4b88dc7`:

| stage | commit | what |
|---|---|---|
| 0 | `8efa17e` | probe positions, from upstream `01efbdc` |
| 0 | `8823cbf` | end criteria on a timestep schedule, from upstream `277ade0` and `eb1c5ee`, with `--exact-endcriteria` |
| 0 | `4f73f7b` | `CheckPeriod()`, from the steady-state part of upstream `c3a956b` |
| 0 | `a44e20e` | one rounded period in timesteps (§8.1) |
| 0 | `acee3af` | `OPENEMS_STEADYSTATE_TRACE=1` |
| 1 | `88e84dd` | probes recorded per slab, ring of three periods by absolute timestep; Vulkan sampler with an x-range |
| 2 | `11361c3` | AVX2: energy summed by rows inside the blocked schedule, deferred check, `SupportsSlabApply()` true, `iterTS < 2` flat |
| 2 | `50c6817` | Vulkan: energy by x-plane inside the batch, check after the batch, refusal gone, `iterTS < 2` flat |
| tests | `5ac4885`, `64619f3` | sine cases in `test_temporal_blocking.py` and `test_gpu_temporal_blocking.py` |

### Stage 0

- **Probe positions** (`01efbdc`): `N/4` and `3N/4` instead of 0.
- **Verdict read after every batch** (`277ade0`): `GetLastDiff()` after every
  `IterateTS()` in both main loops; the 4 s status block only prints it. The
  energy criterion of the other excitations runs every Nyquist period during
  the first 60 s (`eb1c5ee`), and for the whole run with `--exact-endcriteria`.
  The benchmark mode evaluates no end criteria. The stop message names the
  timestep and the level reached, which is what the tests parse.
  - Vulkan pipelined loop: the verdict is read after the drain that loop
    does every iteration anyway. Until stage 2 replaced it, a new period
    then cost one whole-grid reduction (`CalcFastEnergyGPU()`);
    the standard loop also drained once per period.
- **`CheckPeriod()`** (`c3a956b`, steady-state part only).
- **Rounded period** (§8.1, E3): `Excitation::GetSignalPeriodTS()` =
  `lround(T/dT)`, used by the excitation (CPU, TF/SF, Vulkan shaders), the
  steady-state operator and the log line. Changes results only where `int` and
  `round` disagreed, which includes the benchmark's cylindrical case at XS, S
  and M; upstream still truncates.

### Stage 1, (c)

- `Engine_Ext_SteadyState` keeps its records in one ring of three periods,
  `[probe][TS % 3p]`. `CheckPeriod(TS, energy, record)` reads it through
  `RingTimestep()` as the old ring of two periods, so the pairing quirk of §2.1
  is reproduced exactly; flat verdict traces are unchanged by this.
- AVX2: `Apply2VoltagesSlab()` records the probes of the slab (thread 0). A new
  `Engine_Extension::SlabTouchesAt()` (`SlabTouches()` with the timestep) lets
  the engine skip the hook and its barrier in slabs without a probe line, and
  call it in every slab at a check timestep.
- Vulkan: `steady_state_sample.comp` takes `[xLo, xHi)`, and `RecordSlabSweep()`
  samples every slab holding a probe line, after the excitation, where the
  flat recorder samples.

### Stage 2, (b)

AVX2:
- The fast energy is summed by rows (x,y): over z in an 8-lane float FMA
  accumulator as before, then the rows in order in double
  (`Engine_AVX2::FastEnergyRows()`, `FastEnergy()`).
  `Engine_Interface_AVX2_FDTD::CalcFastEnergy()` is that sum, for the flat
  check and for the energy criterion of other excitations. The old single
  float accumulator was already off by 2e-5 relative on a 40³ grid (§8.2).
- At a check timestep, the steady-state slab hook sums the rows of its slab,
  split over the threads, into a buffer of `(NX-1)(NY-1)` pairs of doubles.
- **Deviation from §4(b):** the currents are summed over the slab's E range,
  not over its H range `[An, hHi)`. The hook only gets the E range, and the
  currents there are those the flat sweep reads too: a narrowing core advanced
  its x-lines to TS-1 itself (its H range at TS-1 covers its E range at TS),
  and a widening wedge's extra lines were advanced by the cores. The E ranges
  of one timestep partition `[0, NX)`, so every row is summed once.
- `Engine_Extension::SlabBlockEnd()`: after every block, thread 0 runs the check
  of the block (energy = rows in order), the other threads wait at one barrier,
  once per period. `SlabBlockDepth()` caps a block at one period, so there is
  one check per block and the ring still holds its two periods.
- `iterTS < 2` runs flat (§6), also in the thread calibration.

Vulkan:
- `reduce_energy_planes.comp` sums the energy of x-planes, each plane in fixed
  chunks of 4096 cells (one workgroup each), into one of four slots of a
  host-visible buffer, at the point where the probes are sampled: the whole
  grid in the flat recorders, the slab's E range in `RecordSlabSweep()`, at
  check timesteps only, with one barrier. The host adds the chunks in plane
  order.
- `ScheduleSteadyStateChecks()` registers the checks of a batch with its
  command slot and serial; `GetLastDiff()` evaluates every check before the
  current timestep, waiting for the batch that computed it if need be (in the
  pipelined loop it is already done; the standard loop waits once per period).
  A batch holds at most one period, and a check that a new batch would overrun
  (ring or slot) is evaluated first.
- The drain and `CalcFastEnergyGPU()` of `UpdateSteadyStateResult()` and
  `UpdateGpuSamples()` are gone. That was the phase bug of §2.3 (E2).
- The refusal in `PlanTemporalBlocking()` is gone; blocking is on by default
  for sine runs as for any other.
- **Not done:** aligning the GPU sampling point with the CPU's (§8.3). The GPU
  still samples after the excitation, and its energy covers every cell, the
  CPU's x < NX-1 and y < NY-1. CPU and GPU verdicts agree to about three
  digits, not in the bits; flat and blocked agree in the bits on each engine.

(a′) was not needed.

### Experiments

- **E1** (antenna M, release `4b88dc7`, `--avx2-temporal-block=8`): the only
  refusal is `'Steady-State Detection Extension'`. Conducting sheets, UPML and
  the lumped port are slab-capable.
- **E2** (lossy cavity 200³, κ = 0.01 S/m, ε_r = 2, sine 5 GHz, p = 104,
  endCrit 1e-5, release `4b88dc7`): the CPU converged monotonically and stopped
  at 10478 (at a 4 s status). The GPU verdicts were -35, -4.5, -1.6, -21.6, 0,
  -44.6, -46.1, -55.0 dB and it stopped at 22503, only once the 4 s status
  cadence had locked onto 27 periods (2808 timesteps), i.e. onto one phase.
  After stage 2 both engines stop at **10413**, with the same trace to about
  three digits; the GPU blocked by default at 11350 MC/s (5615 flat before),
  the CPU at k=13 at 2450 MC/s (660 flat before).
- **E3**: in the baseline logs the cylindrical case has `int` periods 867,
  1729, 3455 at XS, S, M where the timestep implies 868, 1730, 3456; all other
  cases agree. That model does not settle: the energy grows ×10³⁵ over
  120 000 timesteps, and `4b88dc7`'s verdict stays at 0 dB there too, so a lossy cavity with 0.96 mm cells (TS = 109, `int` 108)
  shows the effect: `4b88dc7` stays at -8 to -10 dB for 40 000 timesteps, the
  rounded period stops at 2730 (-55 dB).

### Tests

- `test_temporal_blocking.py`: 13/13. New: Test10 (lossy cavity, p = 74,
  processing interval 9; k:W 2:8, 4:16, 3:13, 6:20 × 1, 3, 4 threads), Test11
  (p = 80, interval 10; 5:16, 3:13), Test12 (UPML or Mur with a lumped RLC and
  a probe). Each asserts blocking engaged, fields, check trace (timestep,
  energy, verdict, all digits) and stop timestep identical.
- `test_gpu_temporal_blocking.py`: 17/18. New: three sine cases (2:8, 4:16,
  3:12, 4:13, 6:20; p = 80 with 5:16, 3:12; Mur with a lumped port and an RLC,
  UPML). The old test that required the refusal is replaced.
  `test_lumped_rlc_and_port` fails: the fork goes NaN on its series RLC when it
  is read from XML, `4b88dc7` too, upstream's binary not (open below).
- Both harnesses now write XML and run the build's `openEMS`
  (`OPENEMS_TEST_BIN`, `OPENEMS_TEST_PYTHON`): the installed Python module no
  longer imports against the fork's library (undefined symbol).

### bench.py verify (XS, 2000 timesteps, all cases and dumps)

- **Against upstream** (`5fecbb5`, MT reference): fork AVX2 identical (ideal,
  dispersive) or up to 1.2e-5 on probes and 2.2e-5 on NF2FF dumps (pml,
  inhomogeneous, antenna); Vulkan 4e-6 to 2.9e-5; cylindrical 8.8e-3 to
  3.2e-2 on the CPU, which is the rounded period (868 against upstream's
  truncated 867); fork GPU cylindrical failed, by design.
- **Against `4b88dc7`** (its AVX2 as reference): the new AVX2 rows are
  identical in every Cartesian case, so the deviations above are older; only
  cylindrical changes, by the rounded period.
- **With blocking forced** (`--avx2-temporal-block=4:12` with
  `OPENEMS_AVX2_TEMPORAL_BLOCK_FORCE=1`; Vulkan `OPENEMS_GPU_TEMPORAL_BLOCK=4:12`,
  `_MB=1`; engaged in all five Cartesian cases on both engines): every AVX2
  row and every Vulkan row equals the unblocked run's deviation, except the
  Vulkan antenna rows, 3.0e-3 to 3.6e-3 instead of 4e-6 to 7e-6. That is the
  conducting-sheet race under Open.

### Performance

`bench.py run`, `4b88dc7` (A) against this work (B), interleaved, i9-13900K
with the RX 6800, 8 threads, median of 3 × 3 s, MC/s. A refused blocking in
every row, B blocked in every row; the CPU had `--avx2-temporal-block=13` on
both, the GPU its default. Probes are compared over the timesteps both did.

| case | engine | `4b88dc7` | new | | schedule (new) | probes |
|---|---|---|---|---|---|---|
| antenna M | AVX2, `--avx2-temporal-block=13` | 163.0 | **739.3** | 4.54× | k=13, W=26 | identical |
| antenna M | Vulkan, default | 4109.9 | **7943.5** | 1.93× | k=13, W=81 | 2.8e-3 (see below) |
| inhomogeneous M | AVX2, `--avx2-temporal-block=13` | 227.9 | **882.0** | 3.87× | k=13, W=26 | identical |
| inhomogeneous M | Vulkan, default | 4424.7 | **9456.6** | 2.14× | k=13, W=64 | identical |
| ideal L | AVX2, `--avx2-temporal-block=13` | 597.7 | **1727.3** | 2.89× | k=8, W=16 | identical |
| ideal L | Vulkan, default | 5632.9 | **11391.8** | 2.02× | k=13, W=27 | identical |

The antenna row's GPU mismatch is the conducting-sheet race below (since
fixed), not the
steady-state work. A Dirac excitation with blocking asked for (`iterTS = 1`)
runs flat, bit-identical to the flat run, at the flat speed (CPU 265 against
252 MC/s, GPU 682 against 655 on a 64×32×32 grid).

### Open

- ~~**The Vulkan blocked schedule is not deterministic with conducting
  sheets.** On the benchmark's antenna case (XS, 4:12), two blocked runs
  differ by 7e-4 and blocked from flat by 3e-3, also on `4b88dc7` with a
  Gaussian pulse; the flat path is reproducible, and with the sheets drawn as
  metal blocked is bit-identical to flat. `test_conducting_sheet` passes on its
  smaller model. Not caused by this work, but exposed by it: sine runs used to
  be refused, and the antenna case at M and L now blocks by default. Look at
  the two-order dispersive slab dispatches (`RecordDispersiveSlabPhase()`).~~
  Fixed in `89f500d`: that function recorded the apply dispatches of all
  dispersive entries without a barrier between them, and the two orders of a
  sheet hold the same cells. `test_conducting_sheet` passed because it drew
  the sheet as a box, which the operator turns into PEC.
- The fork goes NaN on a series lumped RLC read from XML (see Tests).
- The cylindrical benchmark case grows without bound in a long run (see
  E3); `4b88dc7` behaves the same as far as its verdict shows. Not looked
  into.
- The GPU samples the probes at another point than the CPU (§8.3), and its
  energy covers a slightly different set of cells.
- `eb1c5ee` evaluates the energy criterion of non-periodic runs every Nyquist
  period in the first minute. That is a processing boundary, and where the probe
  interval does not divide the Nyquist interval it cuts a block short.
- Vulkan announces `k = W/2` up to 32, but the processing interval caps it
  (13 in most benchmark cases); see `plans/GPU_TEMPORAL_BLOCKING_PLAN.md`.

---

## The analysis, before the work

Read-only analysis by a research subagent, 2026-11-04. Fork at `4b88dc7`; upstream at
`5fecbb5`. Nothing was built or run, and
the GPU was not used. All line numbers are in the fork unless marked `upstream:`. Three claims
were checked afterwards against the code: FTZ/DAZ, the probe positions, and the 4 s verdict read.

### 0. Summary and recommendation

**Recommendation: (b) + (c).**

- (c) Record the steady-state probes per slab. This is exact and cheap on both engines.
- (b) Accumulate the whole-grid energy for the check timestep `TS = c·p` inside the blocked schedule:
  - Each x-line contributes once, at the moment it passes through the point where the flat sweep reads it.
  - The partial sums are per row on the CPU and per x-plane on the GPU. They are reduced in a fixed order.
  - The flat path uses the same reduction, so flat and blocked runs give **bit-identical verdicts**.
  - The check is evaluated at the end of the block, from a 3-period probe ring.
  - Properties:
    - no extra sync point and no extra DRAM pass;
    - upstream's definition of the energy, and of when the verdict is read, is kept;
    - expected cost: < 0.5 %;
    - effort: about 4.5–6 days for both engines, including tests, staged as in §5.

Stop-gap if (b) is deferred, AVX2 only: **(a′)**.

- End a block at `c·p` and run that one timestep as a flat step inside `BlockedWorker`.
- It reuses the unchanged flat steady-state code (identical bits).
- Cost: about 3–7 %. Effort: about 1 day.

#### Corrections to the original brief

1. **FTZ/DAZ is already set.**
   - `tools/denormal.h:20-32` sets MXCSR `0x8040`.
   - It is called in:
     - `FDTD/engine_avx2.cpp:90` (constructor, main thread)
     - `FDTD/engine_avx2_multithread.cpp:1038` (every worker)
     - `FDTD/engine_sse.cpp:43`
     - `FDTD/engine_multithread.cpp:326`
   - Upstream does the same: `upstream:FDTD/engine_threadpool.cpp:46`.
   - Upstream also has an RAII guard, `upstream:openems.cpp:1559` (commit `367a63f`).
   - So Gaussian runs do not slow down from denormals on the CPU (§7).
2. **Conducting sheets do not refuse slabs in the source.**
   - `Operator_Ext_ConductingSheet` derives from `Operator_Ext_LorentzMaterial` (`operator_ext_conductingsheet.h:30`).
   - It therefore runs as `Engine_Ext_LorentzMaterial`, with `SupportsSlabApply()==true` (`engine_ext_lorentzmaterial.h:35-41`, `engine_ext_dispersive.h:52`).
   - The antenna case uses a sine, so its refusal is most likely the steady-state extension or a size guard.
   - The AVX2 startup line names every extension that refuses (`engine_avx2_multithread.cpp:380-397`); see E1.
3. **Only `Sinusoidal` has a period.**
   - `excitation.cpp:66-72` sets `m_SignalPeriod=1/f0`; Gaussian, Dirac, Step and Custom set 0 (`excitation.cpp:57-94`).
   - A custom `sin(...)` therefore never gets steady-state detection. `PERFORMANCE.md:1375-1376` uses this as a workaround.
4. **The fork's steady-state probes are misplaced.**
   - `openems.cpp:1574` has `pos[n] *= 1/4;` and `:1579` has `pos[n] *= 3/4;`. Both are integer division, so they give 0.
   - Result: 18 of the 21 probes sit at index 0 on the domain faces, as duplicate pairs.
   - Upstream fixed this in `01efbdc` (`N/4`, `3N/4`).
5. **The fork reads the steady-state verdict only every 4 s of wall time** (`openems.cpp:2160`, `:2246`, both inside `if (t_diff>4)`).
   - So the stop timestep of a sine run depends on the wall clock.
   - Upstream `277ade0` reads `GetLastDiff()` after every `IterateTS` (`upstream:openems.cpp:1617`).
   - Upstream `eb1c5ee` evaluates the energy criterion every Nyquist period for the first 60 s.
   - Neither commit is in the fork.
6. **The Vulkan energy term compares two different phases of the period.** This happens today, with or without blocking (§2.3).

### 1. Where blocking is refused today

**AVX2** (`--avx2-temporal-block=<k>`, off unless forced):

- `PlanTemporalBlocking` (`engine_avx2_multithread.cpp:311-468`) collects every extension with `SupportsSlabApply()==false` (`:380-397`).
- `Engine_Ext_SteadyState` keeps the default `false` (`engine_extension.h:111`). The reason is in `engine_ext_steadystate.h:39-46`.

**Vulkan** (blocking on by default):

- `Engine_Vulkan::PlanTemporalBlocking` refuses at `engine_vulkan.cpp:4591-4596` whenever `m_steadyPipeline != VK_NULL_HANDLE`.
- This is the only extension-based refusal on Vulkan.
- Interior absorbing sheets are handled by moving tile edges (`:4701-4720`).

**Where the steady-state extension comes from:**

- It is created whenever `m_Exc->GetSignalPeriod()>0` (`openems.cpp:1557-1584`), and nothing switches it off.
- The period in timesteps is `round(T/dT)` (`operator_ext_steadystate.cpp:50`).
- `dT` is shrunk so that `T/dT` is an integer (`operator.cpp:1037-1043`).

### 2. How steady-state detection works

#### 2.1 CPU flat path (AVX2 multithread)

Order of one flat-worker timestep (`engine_avx2_multithread.cpp:1061-1100`):

PreV → V update → barrier → PostV → **Apply2V** → PreC → C update → barrier → PostC → Apply2C → `++numTS`

- Every hook is followed by a barrier.
- The steady-state extension has priority `+2e6` (`engine_extension.h:24`), so it is the first Apply2V hook. It runs before Mur, dispersive, RLC and excitation (priority −1000).

`Engine_Ext_SteadyState::Apply2Voltages()` (`engine_ext_steadystate.cpp:106-164`):

- **Every timestep:** records the 21 E probes into a `2p` ring at `TS % 2p`.
- **When `TS%p==0 && TS>=2p`:**
  1. `CalcFastEnergy()` runs on thread 0 (`:117`).
  2. Energy term: `|E_now − E_prev|/E_prev`.
  3. Probe term: per probe, Σ(old−new)² / Σnew² over one period. Probes below 1 % of the peak probe power are ignored.
  4. The result is the maximum of both terms, clamped to 1.

What the check reads at that moment:

- voltages: after the V update and PostV of step `TS`, but before the other Apply2V hooks;
- currents: their final state of step `TS−1`.

Quirk in the probe comparison (`:127-132`):

- The pairing is always `new_pos=0, old_pos=p`, because `rel_pos ∈ {0,p}`.
- The half used for normalisation therefore alternates between the newer and the older period.
- This is harmless, but bit-identity requires reproducing it exactly.

`CalcFastEnergy` on AVX2 (`engine_interface_avx2_fdtd.cpp:44-111`):

- single-threaded;
- 8-lane float accumulators over x∈[0,NX−1), y∈[0,NY−1) and all z-vectors, then a horizontal sum;
- its rounding already differs from upstream's 4-lane SSE version.

How the verdict reaches the end criterion:

- `openems.cpp:2246` (standard loop) and `:2160` (GPU loop) read `GetLastDiff()`, but only inside the 4-s status block.
- The loop condition `change>endCrit` (`:2192`, `:2070`) then stops the run.
- While the steady-state extension exists, the energy criterion of non-periodic runs is skipped (`:2128`, `:2214`).

#### 2.2 How the energy criterion of non-periodic runs coexists with blocking (the template)

- `sampleEnergy()` (`openems.cpp:1902-1916`) leads to `ProcField->CalcTotalEnergyEstimate()`, which is `CalcFastEnergy()`.
- It runs on the host only, after `IterateTS` returns, so the grid is coherent.
- It is triggered:
  - once at `maxExcite`, through `ProcField->AddStep(maxExcite)` (`:1947`), which forces the block to end there;
  - and in the 4-s status block.
- Upstream since `eb1c5ee`/`277ade0` also sets `ProcField->SetProcessInterval(Nyquist)`, again a cadence that forces block ends.

**The template:** make the whole-grid read a block end, through the processing cadence. That is option (a).

How long one `IterateTS` can be:

- `PA = new ProcessingArray(Nyquist)` (`openems.cpp:768`) caps each `IterateTS` at Nyquist steps, about p/2 for a sine.
- Probes add `Nyquist/OverSampling` (default 4), about p/8.
- In the benchmark, p=104 gives a probe interval of 13. 13 divides 104, but in general the probe interval does not divide p.

#### 2.3 Vulkan path

Setup and sampling:

- `SetupGPU_SteadyState()` (`engine_vulkan.cpp:3205-3301`) creates a 3-period float history ring.
- The per-timestep sample is `RecordSteadyStateSample()` (`:3465-3481`).
  - It is recorded *after* the excitation and the late voltage writes (`:5735-5739` fused path, `:5975-5981` separate path).
  - That is a different sampling point from the CPU path.
- The extension is not a CPU hook (`:1577-1586`).

How the verdict is computed:

1. `GetLastDiff()` → `m_gpuUpdater` → `UpdateSteadyStateResult()` (`:3483-3494`), once per new completed period.
2. That does `DrainGPU()` and `CalcFastEnergyGPU()` (`:6911-6950`), a 256-workgroup float reduction.
3. Then `UpdateGpuSamples()` (`engine_ext_steadystate.cpp:57-104`):
   - compares periods `[c−2,c−1)` and `[c−1,c)` from the ring;
   - compares the energy with the previous call's energy.

**Bug: the two energies come from different phases of the period.**

- The current energy is taken at `numTS`, which is whatever batch boundary the 4-s status check lands on, not a multiple of `p`.
- In a driven steady state the unweighted energy ripples at 2f. εr and μr are ignored in the sum, which adds more ripple.
- So the energy term has a floor at roughly the ripple amplitude.
- Likely consequence: GPU sine runs with `endCrit` 1e-5..1e-6 never stop on steady state. This is unverified (E2).
- Upstream's HIP version (`upstream:FDTD/hip/hip_ext_steadystate.hip:78-105`) checks at `TS%p==0` with a sync each period.

**The refusal's stated reason is partly wrong.**

- The energy is already read at batch boundaries, which are coherent even when blocked.
- What actually breaks under blocking is the per-timestep probe sample: it is recorded only in the flat recorder.

### 3. Facts the options rest on

**F1. The steady-state extension only reads; it never writes fields.**

- All options therefore leave the field values bit-identical.
- What can differ: the energy value, the verdict sequence, and with them the stop timestep.

**F2. Inside the blocked schedule, the values at the check point are exact.**

- In `TrapezoidSweep` (`engine_avx2_multithread.cpp:623-687`), the passes' E ranges `[eS,eE)` partition `[0,NX)` at every `ts`. Their H ranges `[An,hHi)` partition `[0,NX−1)`.
- Consider pass P at timestep `ts`, between `SlabPostVoltage` (`:662`) and `SlabApply2Voltages` (`:663`):
  - The voltages on P's E range equal the flat path's values at the steady-state slot.
  - The currents on P's H range hold their final state of `ts−1`:
    - core passes narrow, so their H range at `ts` lies inside their range at `ts−1`;
    - wedge passes widen, and the cores before them have already advanced the extra lines.
- So summing over all passes covers exactly the flat path's set of values, each once.
- `RecordSlabSweep` (`engine_vulkan.cpp:5090-5300`) transcribes the same schedule. Its merged pre-voltage dispatches (`:5265-5285`) write only Mur and RLC state.

**F3. Floating-point sums depend on order.**

- Bit-identity needs one canonical order in both the flat and the blocked path: per (x,y) row on the CPU, or per x-plane on the GPU, summed in row order.

**F4. Probe values per slab are exact.**

- A slab hook placed first among the Apply2V hooks sees the flat values.
- Each probe's x-line falls into exactly one pass per `ts`.

**F5. A deferred check needs a larger probe ring.**

- When the check for `TS=c·p` runs at the end of the block, up to `k−1` newer samples have already been written.
- A ring of 2p would alias them, so use ≥ 2p+k_max = 3p, indexed by absolute timestep.
- Since `k ≤ Nyquist < p`, at most one check falls into any one `IterateTS`.

### 4. Options

Notation:

- `p` = period in timesteps (104 in the benchmark)
- `k` = block depth (8–13 under 13-step probes)
- `S` = speed-up from blocking (AVX2 2.6–3.1× at 224³, Vulkan 2.1–3.2×)
- `t_f` = time for one flat timestep; `t_b` = `t_f/S`

The costs below are modelled, not measured.

#### (c) Probe records per slab (needed by every option)

AVX2 changes to `Engine_Ext_SteadyState`:

- `SupportsSlabApply()=true`, `SlabHookMask()=SLAB_APPLY_VOLT`, `SlabHookWritesFields()=false`.
- `SlabTouches` returns true if a probe x lies in [x0,x1).
- `Apply2VoltagesSlab(x0,x1,numTS,tid)` writes `ring[numTS % 3p]`. It must use the `numTS` argument.

Behaviour and cost:

- The records are bit-identical (F4).
- Cost: one hook plus one barrier per pass that holds a probe line. That is about 3–6 barriers per timestep against about 20 per tile.

Vulkan:

- Dispatch `steady_state_sample.comp` in `RecordSlabSweep`, after the excitation block (`:5226-5241`) and before the current phase (`:5243`).
- Add an x-range to the push constants, or skip the dispatch when no probe lies in the slab.
- No barrier is needed after it.

Effort: about 0.5 d for AVX2 and 0.5 d for Vulkan. On its own, (c) does not lift the refusal.

#### (a) Period boundaries as sync points; energy on the coherent grid

Mechanism:

- A processing cadence of `p` (`SetProcessInterval(p)` / `AddStep`).
- It ends every `IterateTS` and every speculative GPU batch (`openems.cpp:2089-2091`) at `c·p`.
- The host then calls `CalcFastEnergy()` and hands the value to the extension.
- Host-side, the same for both engines, and it fixes the Vulkan phase bug.

Correctness:

- The values are those of the timestep boundary.
- The currents equal the flat values. The voltages are half a step earlier and already include the Apply2V writes.
- So this redefines the energy term. It is equally valid as a criterion, but the bits differ from upstream and the stop moves to `c·p`.

Cost and effort:

- Up to one extra block break per period, about 2 % when bandwidth-bound; none when the probe interval divides p.
- Plus one full-grid read: 3–4 % single-threaded, under 1 % multithreaded.
- In total 1–4 %; effort 1–1.5 d.

#### (a′) Run the check timestep flat inside the blocked worker

Mechanism:

- In `BlockedWorker` (`engine_avx2_multithread.cpp:689-719`), cap `kk` so that no block crosses a check timestep.
- At a check timestep, run one step with the flat loop body, factored out of `:1061-1100`.

Results and cost:

- Bit-identical energy, verdict and stop timestep to today's flat path.
- Cost: (S−1)·t_b ≈ 2 % for the flat step, ≤ 2 % for the block split, plus 3–4 % for the single-threaded `CalcFastEnergy`. That totals ≈ 3–7 %, or 2–4 % if the energy sum is parallelised.
- A poor fit for Vulkan, which would need the in-batch reduction of (b) anyway.
- Effort: about 1 d (AVX2).

#### (b) Accumulate the energy of `c·p` inside the blocked schedule

AVX2 mechanism:

1. In `TrapezoidSweep`, at `ts % p == 0 && ts >= 2p`, between `:662` and `:663`, each thread adds per-(x,y)-row partials:
   - Σ|V|² over `[eS, min(eE,NX−1))`;
   - Σ|I|² over `[An, hHi)`.
2. One barrier follows, only at check timesteps. Every thread must take the same branch.
3. After the block, thread 0 sums the rows in order.
4. The deferred check uses absolute-indexed ring access that reproduces upstream's `rel_pos` pairing.
5. Port the `CheckPeriod` refactor from upstream `c3a956b`.

Exactness and memory:

- Exactness follows from F2.
- Bit-identity needs `Engine_Interface_AVX2_FDTD::CalcFastEnergy` rewritten to the same per-row scheme, with rows summed in double.
- That changes the last bits of the flat energy, including the energy criterion of Gaussian runs.
- Memory: NX·NY·2 doubles.

Vulkan mechanism:

- At `ts%p==0`, record a per-x-plane reduction after the PostV group in both recorders: the flat one around `:5960-5975`, and `RecordSlabSweep` before `:5210`.
- One barrier follows, only at check timesteps.
- After the fence, the host sums the planes in x order.
- This replaces `DrainGPU()`+`CalcFastEnergyGPU()` in `UpdateSteadyStateResult`, which fixes the phase bug and removes the drain.

Cost:

- No extra DRAM traffic.
- AVX2: about 6 FMA per cell once per period.
- Vulkan: about 2 small dispatches plus a barrier per pass per period, ≈ 80 µs per ~50 ms.
- Total < 0.5 %.

Effort:

| part | days |
|---|---|
| AVX2 | 1.5–2 |
| Vulkan | 1.5–2 |
| tests | 1 |

#### (d) Simpler variants

- **(d1) (a) or (a′) every N periods.** Deterministic, cost divided by N, but up to N periods of extra latency.
- **(d2) A lazy, wall-clock check.** Not deterministic, against upstream's direction. Not recommended.
- **(d3) Drop the energy term.** Rejected: weaker criterion, and the verdicts differ between flat and blocked.
- **(d4) Workaround today: `SetCustomExcite('sin(2*pi*f0*t)', f0, f0)`.**
  - No period, so no steady-state extension, so blocking engages now.
  - Caveat: `CalcCustomExcitation` builds a table of `NrTS` samples (`excitation.cpp:222-251`). The benchmark's `NRTS = 1e9` would need 8 GB, so lower it to about 1e6 for such cases.
- **(d5) Probe-only checks at block ends.** Rejected.

#### Comparison

| option | correct | bit-identical flat↔blocked | upstream semantics | cost | AVX2 | Vulkan | effort |
|---|---|---|---|---|---|---|---|
| (c) alone | refusal stays | records yes | yes | ~0 | easy | easy | 1 d |
| (a) | yes (redefined) | only if flat redefined too | no | 1–4 % | host | host, fixes phase bug | 1–1.5 d |
| (a)+(d1) | yes (redefined) | same | no | (1–4 %)/N | host | host | 1.5 d |
| (a′) | yes | yes | yes | 3–7 % | easy | poor fit | 1 d |
| **(b)+(c)** | **yes** | **yes** | **yes** | **<0.5 %** | moderate | moderate, fixes phase bug | **4.5–6 d** |
| (d3) | weaker | no | no | 0 | trivial | trivial | — |

Blocking's ceiling here is the probe interval (13), not the steady-state check. Vulkan's default `k = W/2 ≤ 32` (`engine_vulkan.cpp:4675`) cannot be reached with 13-step probes; that is a separate item (`plans/GPU_TEMPORAL_BLOCKING_PLAN.md:152-161`).

### 5. Recommended plan

**Stage 0** (≈ 0.5 d):

- Port upstream `01efbdc` (probe positions).
- Port `277ade0` (read the verdict after every `IterateTS`).
- Optionally port `eb1c5ee`.
- Port the `CheckPeriod` refactor from `c3a956b`.
- Optionally use `lround` for every period computation.

**Stage 1, (c)** (≈ 1–1.5 d):

- Slab probe hook on AVX2, x-filtered sampler on Vulkan.
- 3p ring with absolute indexing, deferred check.
- Keep the AVX2 refusal until Stage 2, or gate it behind (a′).

**Stage 2, (b)** (≈ 2–3 d):

- AVX2 row partials and a canonical `CalcFastEnergy`.
- Vulkan per-plane reduction in both recorders.
- Remove the refusal (`engine_vulkan.cpp:4591-4596`) and the drain.
- Update the docs: `engine_ext_steadystate.h:39-46`, `OPTIMIZATIONS.md:879-883,905`, `PERFORMANCE.md:729-732,1342-1343`, and the plans.

**Tests** (≈ 1 d): extend `test_temporal_blocking.py` and `test_gpu_temporal_blocking.py` with a sine case. Each test must assert:

- blocking is engaged;
- the fields are bit-identical;
- the per-check verdict trace is identical (via a debug env var);
- the stop timestep is identical (a lossy model with `endCrit` 1e-3).

Cover several (k, W) and thread counts, a folded tail, and a p that is not a multiple of the probe interval.

**Stop-gap:** (a′) after Stage 1, about 1 d. It can later be replaced by (b) without changing results.

### 6. Other excitations and the remaining refusals

| feature | AVX2 | Vulkan | what support needs |
|---|---|---|---|
| Sinusoidal (the only signal with a period) | refused (steady-state extension) | refused (`engine_vulkan.cpp:4591`) | §5 |
| Gaussian; Custom (even a periodic custom function) | OK (`engine_ext_excitation.cpp:57-100`) | OK | — |
| Dirac, Step | engages but is useless: `SetNyquistNum(1)` caps every `IterateTS` at 1 step, so k=1 is slower than flat | same, `k=min(m_tbK, iterTS)=1` | treat `iterTS<2` as flat (`engine_avx2_multithread.cpp:244`, `engine_vulkan.cpp:5550`) |
| TF/SF, any signal | slab-capable (`engine_ext_tfsf.h:38`) | done | with a sine source, the steady-state extension again |
| Lumped ports (RLC + excitation) | slab | done | — |
| Conducting sheet | slab via Lorentz | GPU-native | nothing; check the antenna log (E1) |
| Mur | slab | done | — |
| Interior absorbing sheet normal to x | refused (`engine_ext_absorbing_bc.cpp:176-202`) | OK, tile edges moved (`engine_vulkan.cpp:4694-4724`) | port the edge moving to AVX2 `PlanTemporalBlocking` (`:441-449`), ≈ 0.5 d |
| Cylindrical | never reaches the AVX2 multithread engine | — | out of scope |
| Probes, dumps, NF2FF | cap k | cap k; gather at batch end | — |

### 7. Denormals: FTZ/DAZ

State:

- The fork sets FTZ/DAZ on every thread that does field work (see §0 item 1).
- Upstream does the same, and additionally:
  - restores the caller's MXCSR with an RAII guard (`367a63f`);
  - offers `-DENABLE_FLUSH_TO_ZERO=OFF`.

Effect on results:

- Only values below 1.18e-38 change, so the effect on probes and dumps is below 1e-38 absolute.
- Turning FTZ off would break bit-identity with upstream's default build.
- Vulkan shaders use the device default, so CPU-against-GPU comparisons in the subnormal range are meaningless either way.

**Recommendation: keep FTZ/DAZ, and port `367a63f`:**

- the RAII scope in `RunFDTD`;
- restoring MXCSR afterwards (today the `Engine_AVX2` constructor leaves the main thread, and in-process Python, in FTZ for good);
- the CMake option.

For Gaussian benchmark cases:

- The CPU does not slow down from denormals.
- Regions that have decayed to zero make GPUs about 28 % faster (`OPTIMIZATIONS.md:1360-1385`), which biases late-run GPU timings.
- A sustained sine, or a custom sine, is the better benchmark workload.

### 8. Side findings and risks

**8.1 Integer vs rounded period.**

- The excitation wraps at `int(T/dT)`:
  - `engine_ext_excitation.cpp:69,116`;
  - `engine_ext_tfsf.cpp:146,264`;
  - `engine_vulkan.cpp:1599,5079,5491`;
  - the same in upstream.
- The steady-state extension uses `round(T/dT)`.
- If `T/(T/TS)` evaluates to `TS−ε`, the excitation repeats every TS−1 steps while steady state compares every TS steps.
- The probe term then floors at about (2π/TS)², −24 dB at TS=104, and the run never converges.
- Check by comparing "Excitation signal period is" (`openems.cpp:1710`, int) with "Period TS" (`operator_ext_steadystate.cpp:103`, round) in the logs. Fix: `lround` everywhere.

**8.2 Float accumulation in `CalcFastEnergy`.**

- The relative noise floor is plausibly 1e-7..1e-5 on large grids, close to typical `endCrit` values.
- Row partials summed in double, as (b) needs, improve it.

**8.3 Different sampling points.**

- The GPU samples the probes after the excitation and late writes; the CPU samples before them.
- No effect on convergence, but the records differ for probes on excited, Mur or RLC cells. (b) is a natural point to align the two.

**8.4 Risks of (b).**

- It changes the hot loops. Every thread must take the same check decision, or the barrier counts diverge.
- Future schedule changes must preserve F2.
- The canonical `CalcFastEnergy` changes the fork's flat energy bits, including the energy criterion of Gaussian runs.
- If upstream changes the extension, the slab version has to follow.

### 9. Experiments worth running later (not run)

- **E1.** Antenna case on AVX2 with `--avx2-temporal-block=8`, about 50 steps. Read "AVX2 temporal blocking: disabled, …" to see which extensions refuse.
- **E2.** The GPU phase bug: a small lossy resonant sine model with `endCrit=1e-5`, flat CPU against Vulkan. Compare the "SteadyState: … dB" traces and the stop timesteps.
- **E3.** The period mismatch: compare the int and round periods in the logs of several sine cases (§8.1).
- **E4.** The acceptance test for (b), as in §5.
