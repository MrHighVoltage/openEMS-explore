# In-batch sampling: temporal blocking for every excitation

Temporal blocking (TB) ran only as deep as the probe interval, and not at all
for a Dirac or step excitation. Nothing refused those excitations; the main
loop ended every `IterateTS()` at the next timestep a probe was due at. In-batch
sampling lets a batch run past those timesteps: the engine samples, inside the
batch, what the probes and the energy end criterion read, and the main loop
replays the batch afterwards with the probes reading the samples. The probe
code runs unchanged on the same floats in the same order, so every output is
bit-identical to the old loop's.

This document is the design and the engine contract. The core (interface,
generic implementation, main loop) is done; the AVX2 in-block sampler and the
Vulkan engine are implemented against §4.

## 1. Status (2026-11-06)

Branch `tb-all-exc/core`, on top of `1922d00`:

| commit | what |
|---|---|
| `9420f28` | `Processing::ProcessAt()` and friends: `Process()` looked ahead without processing |
| `39662be` | `OPENEMS_PROBE_PRECISION` |
| `a577d02` | the engine interface (`engine.h`) and the generic implementation in `Engine` |
| `b41827a` | steady-state verdicts and trace as of the timestep the run has passed |
| `3cf209f` | `Engine_AVX2`, `Engine_AVX2_Multithread`: generic sampling; blocked-timestep counter |
| `a1a6264` | `RunFDTD()`: the sampled standard loop, `OPENEMS_ENDCRIT_TRACE`, the summary lines |
| `189ff16` | `planBatch()` continues from given positions; the replay checks the last end too |
| `53d827d` | `python/Tests/batch_sampling_identity.py` |
| `0ca88f5` | review: a replayed read is an index lookup; `Engine_AVX2::SampleFootprint()` from offsets |
| `c7a230b` | review: a batch holds at most 4094 steady-state periods (§3.8) |
| `90849d6` | review: the replay follows processings that change their schedule (§3.10) |

- **Done.** Everything in §3 and the generic implementation of §4, used by
  `Engine_AVX2` and `Engine_AVX2_Multithread`. Identity holds on every case
  tried (§6), flat and blocked, 1 to 4 threads, with small rings.
- **What it gains so far.** The generic implementation runs `IterateTS()`
  between consecutive sample timesteps, so its blocks still end at every probe
  timestep: the depth is the same as before. What changes is that only probe
  and energy timesteps split a batch, not the Nyquist cap or a probe that has
  not started yet, and that the work between two dumps is one batch. Dirac and
  step runs still block nowhere (every timestep is a probe timestep).
- **Not done.**
  - AVX2 in-block sampling: `Engine_AVX2_Multithread` overrides
    `IterateSampledTS()` and samples per slab inside the trapezoid schedule
    (§4.4). That is what lifts the depth cap, Dirac and step included.
  - Vulkan (§4.5): `SupportsBatchSampling()` false, it runs the old loops
    unchanged and prints `In-batch sampling OFF: the gpu engine does not
    support it`. Its blocked-timestep counter is not there either, so it
    prints no `Temporal blocking:` line.
  - The pipelined GPU loop (`openems.cpp`, "Pipelined GPU main loop") is
    unchanged.

## 2. Why blocking stopped at the probe interval

- No code refuses an excitation. Dirac and step set `SetNyquistNum(1)`
  (`FDTD/excitation.cpp`).
- Probes and dumps run every `max(1, Nyquist/OverSampling)` timesteps
  (`SetupProcessing()`). The energy end criterion runs every Nyquist interval
  during the first 60 s (`EXACT_ENDCRIT_TIME`), or always with
  `--exact-endcriteria`. `ProcessingArray(Nyquist)` caps every step at the
  Nyquist interval.
- The old loop:

  ```
  step = PA->Process()                    // at timestep 0
  while (numTS < NrTS && change > endCrit && !failure && !abort)
      [bench: Advance at its boundary; step = bench->Clamp(step, numTS)]
      IterateTS(step)
      step = PA->Process()                // min over the processings' next interval, <= Nyquist
      checkEndCriteria()                  // SSD: GetLastDiff(); else ProcField->CheckTimestep() -> energy
      step = min(step, NrTS - numTS)
      relaxEndCriteria(); every 4 s the status line (it samples the energy once relaxed)
  ```

  so an `IterateTS()` call never holds more than the probe interval, and both
  engines block `min(k, iterTS)` deep; `iterTS < 2` runs flat. With the default
  oversampling of 4 the interval is 4 to 25 timesteps, below the depth the
  engines want (AVX2 about 13, GPU 16 to 22), and 1 for Dirac and step.

## 3. Design

### 3.1 Batches and ends

A **batch** is one run of the engine. Its **ends** are the timesteps at which the
old loop would have ended a batch, in order; the last of them is the batch's
last timestep. After the batch, the loop replays each end: there it does what
the old loop did after an `IterateTS()`, with the engine in replay of that
timestep. So the replayed timesteps are exactly the old loop's batch ends, and
`GetLastDiff()`, the energy criterion, the probes and the stop are evaluated
where the old loop evaluated them.

A batch ends at the first end at which

- a processing the replay does not serve reads the fields (§3.6);
- one more sample would exceed the ring (`OPENEMS_BATCH_SAMPLING_RING`);
- `NrTS` or the benchmark's boundary is reached;
- the batch holds `bsMaxLen` timesteps or more (§3.7).

Its last end is processed on the live fields, as before.

### 3.2 Look-ahead: `planBatch()`

The ends come from simulating the old loop's step chain without processing.
`Processing` state that decides the schedule is only the position in the step
list (`m_PS_pos`, advanced by `CheckTimestep()` on an `AddStep()` timestep), so:

- `Processing::CheckTimestepAt(ts, psPos)` and `NextIntervalAt(ts, psPos)` are
  `CheckTimestep()` and `GetNextInterval()` at a given timestep and position
  (the originals call them now);
- `Processing::ProcessAt(ts, psPos, reads)` is what `Process()` returns at `ts`
  and whether it reads the fields there. `ProcessIntegral` and
  `ProcessFieldsTD` read whenever due, `ProcessFieldsFD` and
  `ProcessFieldsSAR` only at the samples of their DFT (`ts % m_FD_Interval ==
  0`), the base class (ProcField) never. A subclass that overrides `Process()`
  overrides `ProcessAt()` to match;
- `ProcessingArray::ProcessAt()` combines them as `Process()` does (minimum
  of the positive intervals, capped at Nyquist).

`planBatch(startTS, step[, positions])` (`openems.cpp`) walks the chain from
the processings' positions: at each end `ts` it calls `PA->ProcessAt()`, then,
as `checkEndCriteria()` does after `Process()`, `ProcField->CheckTimestepAt()`
when the energy criterion is on (no steady-state detection, no benchmark),
caps the step at `NrTS - ts` and clamps it to the benchmark's boundary, as the
old loop does. Probe reads at `ts` make it a footprint sample timestep, an
energy check an energy sample timestep. The cost is a few operations per
processing and end; the old loop called `PA->Process()` at each end anyway, so
nothing scans timestep by timestep that the old loop did not visit.

It also returns the step and positions after the last end (`nextStep`,
`nextPositions`), so that a batch can be planned before its predecessor is
replayed.

### 3.3 Replay: `replayBatch()`

For each end `ts`, in order: `BeginBatchReplay(ts)`, `step = PA->Process()`,
`checkEndCriteria()`, `EndBatchReplay()` (an RAII scope). It stops at the first
end where the run stops, an end criterion met or a numerical failure; that end
is `bsDoneTS`. Then the loop leaves (`bsStopped`) without the status block. A
stop at the batch's last end runs the status block first, as the old loop did.

The replay follows what the processings do, not the plan; where they leave
it, §3.10 applies. `replayBatch()` returns whether they did what the plan
looked ahead to, up to `nextStep` and `nextPositions` after the last end.

The post-loop lines report `bsDoneTS`: the stop message, `Time for <n>
iterations`, the speed and `--dumpstatistics`. The engine may have run on to
the end of the batch; `Temporal blocking: <b> of <n>` counts what it ran.

### 3.4 The footprint

Before the run (after `PA->PreProcess()`), the engine records the cells
`GetVolt()`/`GetCurr()` are asked for while the loop calls
`CalcMultipleIntegrals()` once on every enabled `ProcessIntegral`. That covers
probes with a start time too, which `Process()` at timestep 0 would not. What a
probe reads does not depend on the timestep: `ProcessVoltage`,
`ProcessCurrent`, `ProcessFieldProbe` and `ProcessModeMatch` read a fixed set
of cells through the engine interface (`Engine_Interface_FDTD::GetRawField()`
and friends, `CalcVoltageIntegral()`) or `Engine::GetCurr()` directly. The
footprint is sorted by `Engine::FootprintKey(field, n, x, y, z)`; a cell's
position is its index in every sample.

### 3.5 The energy

The energy criterion reads `ProcField->CalcTotalEnergyEstimate()`, which is the
engine interface's `CalcFastEnergy()`. In a replay of an earlier timestep each
interface's `CalcFastEnergy()` (`Engine_Interface_FDTD`, `_AVX2_`, `_SSE_`)
returns `Engine::GetBatchReplayEnergy()`, the sample of that timestep. The
generic implementation takes the sample with exactly that function: the loop
hands the engine `[ProcField] { return ProcField->CalcTotalEnergyEstimate(); }`
(`SetBatchEnergyFunction()`), called on the coherent grid outside a replay.
The 4 s status block samples the live energy at the batch's last end, which is
the only place it runs; that part was wall-clock-dependent anyway.

### 3.6 What the replay serves, and what it does not

Served: every `ProcessIntegral` (voltage, current, E/H field probes, mode
matching, with or without frequency samples, start and stop times), and
ProcField, which reads nothing itself.

Not served, ending a batch wherever they read the fields: time-domain field
dumps (`ProcessFieldsTD`), frequency-domain dumps (`ProcessFieldsFD`, at their
DFT samples only), SAR (`ProcessFieldsSAR`), and any processing that is not a
`ProcessIntegral`. A dump every probe interval therefore keeps the batches as
short as before; a dump with a start/stop window ends them only inside the
window.

### 3.7 Batch length and ring

- `bsMaxLen` starts at the old loop's cap, the Nyquist interval, and after each
  batch becomes about 0.5 s of work at the measured rate (batch plus replay),
  at most twice the previous, never below Nyquist (`adaptBatchLength()`). The
  status line and the abort check stay as frequent as they were; a batch may
  overshoot `bsMaxLen` by less than one old-loop step.
- The ring is the most sampled timesteps (footprint or energy) one batch may
  hold: by default 1024, fewer where the footprint is large (about 64 MB of
  samples, at least 16); `OPENEMS_BATCH_SAMPLING_RING=<n>` sets it. A batch
  that would need one more ends at that end instead.

### 3.8 Steady-state detection

The checks run inside the engine (`Engine_Ext_SteadyState`, see
`plans/STEADY_STATE_TEMPORAL_BLOCKING.md`) and need no samples. A check of
period end `TS` runs while the engine advances from `TS` to `TS+1`; the old
loop, at a batch end `ts`, had the verdicts of the checks `TS < ts`. The
extension now queues its verdicts, and `GetLastDiff()` hands on those of
`TS < ts` in a replay of `ts`, all of them outside a replay (as before). The
trace (`OPENEMS_STEADYSTATE_TRACE=1`) prints a check when `GetLastDiff()`
passes it, so a run prints the checks it stopped on and none it ran ahead to.
`SlabBlockDepth()` still caps a block at one period, which the batch length
does not affect.

The queue holds at most `Engine_Ext_SteadyState::MAX_QUEUED_VERDICTS` (4096)
checks; beyond that `CheckPeriod()` passes the oldest itself, which only a run
that never asks needs (`--benchmark`). `planBatch()` therefore ends a batch
after `GetMaxRunAhead()` = 4094 periods, so that a batch never queues more
checks than its replay finds (a 12³ box with an 11-timestep period ran batches
of 13 000 periods and stopped at 265213 instead of 167533 before this cap).

### 3.9 Old-loop equivalence, and what is not equivalent

Bit-identical by construction: the replayed timesteps, `PA->Process()` and
`checkEndCriteria()` at each, the values they read, hence the probe files, the
dumps (always live), the traces and the stop timestep.

Wall-clock-dependent, as before: the status line, `PA->FlushNext()`, the
relaxation of the energy criterion after 60 s, and the abort check. These run
at batch ends only. The intermediate `*_FD` flushes can therefore come at other
timesteps; the files written at the end are the same.

### 3.10 When a processing changes its own schedule

The plan assumes that a processing's schedule depends only on its step
position. A processing can still change it while processing: a dump that
fails to write disables itself (`ProcessFieldsTD`, e.g. `FileType` 2), and
`ProcessFieldsFD::Process()` returns -1 when `CalcField()` fails. Dumps only
process at a batch's last end, so this usually changes only what comes after
the batch, which the standard loop plans from the live state anyway.

`replayBatch()` therefore follows the processings, not the plan: after each
end it goes to the end their `step` asks for, also one the plan does not have
(the batch replays any timestep in it), and skips planned ends they no longer
ask for. It replays the batch's last timestep in any case, since the engine is
there. The real schedule of a processing that disabled itself is a subset of
the plan, and then

- every timestep at which a processing reads was planned, and sampled: a
  planned end is any timestep some processing's next interval pointed to;
- the replay visits the old loop's ends, except that it also replays the
  batch's last timestep, and that after it the Nyquist cap counts from there:
  where nothing is due, an end the old loop would not have had. Only the
  steady-state verdict is read there, so a sine run can stop up to one
  old-loop step earlier than the old loop in this case.

`replayBatch()` returns false where the processings left the plan (`-v`
counts these batches). A hard error remains only where they need what the
batch did not sample: a read at a replayed timestep without a probe sample or
outside the footprint (`BatchRead()`), or an energy estimate without a sample
(`GetBatchReplayEnergy()`), both `std::logic_error`, which `RunFDTD()` turns
into a cancelled run. For a batch planned ahead (the pipelined loop, §4.5),
the return value says that the plan of the next batch, made from this one's
`nextStep` and `nextPositions`, is stale: it is still a superset if the
change was a processing disabling itself, so it may run, but a new plan from
the live state is exact. The same holds for `relaxEndCriteria()`, which only
clears ProcField's interval: a batch planned before it and replayed after it
no longer throws.

## 4. The engine contract

### 4.1 Interface (`FDTD/engine.h`)

```cpp
enum BatchSamplingMode {BS_OFF = 0, BS_RECORD = 1, BS_REPLAY = 2};
virtual bool SupportsBatchSampling() const;    // default false
virtual bool SamplesInsideBlocks() const;      // default false
virtual long long GetBlockedTimesteps() const; // -1: not counted

void StartFootprintRecording();
void StopFootprintRecording();
const std::vector<uint64_t>& GetFootprint() const;
static uint64_t FootprintKey(field, n, x, y, z);   // field 0 voltages, 1 currents
static void DecodeFootprintKey(key, field, n, x, y, z);

void SetBatchEnergyFunction(std::function<double()> fn);
virtual void ScheduleBatchSamples(const std::vector<unsigned int>& probeTS,
                                  const std::vector<unsigned int>& energyTS);
virtual bool IterateSampledTS(unsigned int iterTS);
const std::vector<unsigned int>& GetBatchProbeTimesteps() const;
const std::vector<unsigned int>& GetBatchEnergyTimesteps() const;

void BeginBatchReplay(unsigned int ts);
void EndBatchReplay();
bool InBatchReplay() const;
double GetBatchReplayEnergy() const;

// protected, for the engines
bool BatchRead(field, n, x, y, z, FDTD_FLOAT& value) const;
FDTD_FLOAT* BatchProbeSample(size_t i);
void SetBatchEnergySample(size_t j, double e);
double CalcBatchEnergy() const;
void SampleFootprint(size_t i);
```

The schedule: absolute timesteps, ascending, each after the batch's first
timestep and before its last (`numTS < t < numTS + iterTS`). The last is never
sampled: the replay of the current timestep is a no-op and reads the live
fields.

### 4.2 What an engine must do

1. `SupportsBatchSampling()` returns true.
2. `GetVolt()`, both overloads, begin with
   `if (m_bsMode != BS_OFF) { FDTD_FLOAT v; if (BatchRead(0, n, x, y, z, v)) return v; }`,
   `GetCurr()` the same with field 1. `BatchRead()` records in `BS_RECORD`
   (and returns false: read live), serves the sample in `BS_REPLAY`, and throws
   for a cell outside the footprint or a timestep without a probe sample.
3. `GetNumberOfTimesteps()` is `Engine`'s (the replayed timestep in a replay),
   or calls it.
4. `IterateSampledTS(iterTS)` runs `iterTS` timesteps, leaving the fields
   `IterateTS(iterTS)` would leave, and fills
   - for the i-th probe timestep `t`: `BatchProbeSample(i)[c]` for every cell
     `c` of the footprint;
   - for the j-th energy timestep `t`: `SetBatchEnergySample(j, e)`;

   with the values defined in §4.3. It sets `m_bsBatchStart` to the timestep
   it starts at (the replay accepts timesteps from there to the current one).
5. Optionally `GetBlockedTimesteps()`: the timesteps run inside blocks of
   depth two or more, for the summary line.

The default `IterateSampledTS()` does item 4 with `IterateTS()` up to each
sample timestep, `SampleFootprint()` (the engine's own `GetVolt()`/`GetCurr()`)
and `CalcBatchEnergy()`. An engine that samples inside its blocks overrides it
and returns true from `SamplesInsideBlocks()`; the benchmark then takes the
batch length, not the longest gap between samples, as the depth the probes
allow.

### 4.3 What a sample is

**Probe sample of timestep `t`:** for each footprint cell, the value
`GetVolt()`/`GetCurr()` returns when the engine has done exactly `t` timesteps
(`numTS == t` after `IterateTS()`): the cell's value after the last write of
the iteration that brought `numTS` from `t-1` to `t` and before the first write
of the next. In the flat sweep's terms, for iteration `t-1`:
`DoPreVoltageUpdates, UpdateVoltages, DoPostVoltageUpdates, Apply2Voltages,
DoPreCurrentUpdates, UpdateCurrents, DoPostCurrentUpdates, Apply2Current`;
the sample is taken after `Apply2Current()` of `t-1`. A voltage is usually
final after `Apply2Voltages()` of `t-1`, a current after `Apply2Current()`, but
an extension hook of the current phase may write voltages; an in-block sampler
has to take a cell after the last hook of `t-1` that can write it.

**Energy sample of timestep `t`:** what the engine interface's
`CalcFastEnergy()` returns at `numTS == t`, bit for bit. That is
`Engine_AVX2::FastEnergy()` on AVX2 (rows `(x,y)`, x < NX-1, y < NY-1, each
row's |V|² and |I|² summed over z in float, the rows in order in double),
`Engine_Vulkan::CalcFastEnergyGPU()` on the GPU, the scalar loop of
`Engine_Interface_FDTD` elsewhere. Note that this is the state at the end of
the timestep, not the steady-state check's point (between `PostVoltage` and
`Apply2Voltages` of `TS`).

**Identity** then follows: the replay feeds the probes the same floats in the
same order, and `CalcFastEnergy()` the same double, as the old loop's live
reads at that timestep. Fields are unchanged by sampling (the engine only
reads), so dumps, which always read live, are unchanged too.

### 4.4 For the AVX2 in-block sampler

- Override `IterateSampledTS()`: one `m_startBarrier`/`m_stopBarrier` round for
  the whole batch, the workers sampling inside `TrapezoidSweep()`; return true
  from `SamplesInsideBlocks()`.
- Per `ts = t-1` and pass: the voltages of the pass's E range `[eS, eE)` after
  its `SlabApply2Voltages()`, the currents of its hook range `[hS, hE)` after
  its `SlabApply2Current()`. The E ranges of one timestep partition `[0, NX)`,
  the H ranges `[0, NX-1)` (the edge tile's hook range reaches NX), so every
  footprint cell is sampled once (F2 of `STEADY_STATE_TEMPORAL_BLOCKING.md`).
  Check per extension whether a current-phase hook writes voltages (`Slab*`
  hooks, `SlabHookWritesFields()`); if one does, take the voltages after it.
- The energy at the end of `t`: per row, |V|² after the V phase and |I|²
  after the I phase of `t-1`, into a rows buffer like the steady-state one
  (`FastEnergyRows()` sums both at once; it has to be split, or called once the
  row is complete), the rows added in order by `FastEnergy(rows)` after the
  block. With `SlabBlockEnd()` this already works for one check per block; a
  block may now hold several energy timesteps (Dirac: every timestep), so the
  rows need one buffer per energy timestep of the block, or a block may end at
  each.
- The footprint is sorted by (field, n, x, y, z); a per-x index of the
  footprint lets a slab find its cells. `BatchProbeSample(i)` is plain memory,
  `numTS` must not be touched by the samplers (the replay reads it).
- `GetBlockedTimesteps()` already counts in `BlockedWorker()` (thread 0, depth
  after `SlabBlockDepth()`).

### 4.5 For Vulkan

- `Engine_Vulkan::GetVolt()`/`GetCurr()` record and serve their own probe
  cache; `BatchRead()` has to come first, and its own recording has to stay
  out of `BS_RECORD` (the core records before `StartProbeRecording()`).
- `IterateSampledTS()` maps onto the gather of the probe cache at each sample
  timestep into a ring slot, and the energy onto a reduction whose result
  equals `CalcFastEnergyGPU()` bit for bit (or `Engine_Interface_FDTD::
  CalcFastEnergy()` changes for the GPU to the same reduction, flat and
  sampled).
- The pipelined loop can reuse `planBatch()` and `replayBatch()`, which are
  lambdas of `RunFDTD()` defined before both loops. Planning batch N+1 while
  batch N is replayed: `planBatch(endOfN, batchN.nextStep,
  &batchN.nextPositions)` (with the benchmark, clamp `nextStep` first). The
  samples of N must survive `ScheduleBatchSamples()` of N+1 (the base class
  keeps one batch: double-buffer, or schedule after the replay), and
  `BeginBatchReplay()` must accept the timesteps of N while the public `numTS`
  is N's last (`SubmitSpeculative()` keeps it there). `relaxEndCriteria()`
  changes ProcField's interval: a batch planned before it and replayed after
  it diverges and throws, so relax only with no plan outstanding.
- The standard loop's benchmark branch drains the GPU at a boundary; the
  sampled loop does not (no GPU engine takes it yet).
- `GetBlockedTimesteps()` is -1 on the GPU so far: no summary line.

## 5. Switches and lines

| | |
|---|---|
| `OPENEMS_BATCH_SAMPLING=0` | the old loop, as before this work; unset or 1: in-batch sampling where the engine supports it |
| `OPENEMS_BATCH_SAMPLING_RING=<n>` | the most sampled timesteps of one batch |
| `OPENEMS_ENDCRIT_TRACE=1` | `endcrit trace: ts=<n> energy=<%.17g> maxE=<%.17g> change=<%.17g>` at every evaluation of the energy criterion (in `checkEndCriteria()` and the status block) |
| `OPENEMS_STEADYSTATE_TRACE=1` | (existing) one line per steady-state check, now when the run passes it |
| `OPENEMS_PROBE_PRECISION=<n>` | significant digits of the probe files, time domain (default 12) and `*_FD` (default 6) |

Every run prints one of
`In-batch sampling ON: <c> probe cells, ring <r> timesteps` and
`In-batch sampling OFF: <reason>` (`OPENEMS_BATCH_SAMPLING=0`, or `the <engine>
engine does not support it`). A run whose engine counts blocked timesteps and
has a temporal-blocking request (forced, or resolving to active) ends with
`Temporal blocking: <b> of <n> timesteps ran blocked`, `n` the timesteps the
engine ran. `-v` adds `In-batch sampling: <batches> batches of <n> timesteps
(...), <e> timesteps replayed, <p> probe and <q> energy samples`.

## 6. Verification

`python/Tests/batch_sampling_identity.py` runs each case twice on one binary,
`OPENEMS_BATCH_SAMPLING=0` against in-batch sampling, with 17-digit probe files
and both traces, and compares every probe file (the dated header aside), every
HDF5 dataset and attribute, the traces, the stop line and `Time for <n>
iterations`. On a 64×32×32 grid with Mur (or PEC), probes of every kind (V, I
with frequency samples, E and H points, E and H mode matching, a windowed
voltage probe), `--engine=avx2-multithreaded`, flat and
`--avx2-temporal-block=4:16` forced, 1 and 3 threads:

| case | stop | result |
|---|---|---|
| Gaussian, OverSampling 8 (probes every 3), default end criterion | energy, 350 | identical |
| the same with `--exact-endcriteria`, a dielectric | energy, 425 | identical |
| Gaussian to NrTS 700, a windowed TD dump and an FD dump | NrTS | identical |
| Dirac to NrTS | NrTS | identical |
| step to NrTS | NrTS | identical |
| custom pulse, OverSampling 6 | energy, 612 | identical |
| sine, lossy, PEC box, steady state | steady state, 963 | identical |
| sine, lossy, dumps, to NrTS | NrTS | identical |

Also identical with `OPENEMS_BATCH_SAMPLING_RING` 1 and 2, on `--engine=avx2`,
and against the `1922d00` binary (old loop and in-batch sampling, default
precision). The blocked runs block exactly as deep as before (e.g. 341 of 350
timesteps for the old loop, 365 of 375 for the new one, which ran on to its
batch end). `test_temporal_blocking.py` passes (15). `--benchmark` runs and
reports "probes every 3 timesteps cap the block depth at 3" on the first case,
as before.

The independent suite (`python/Tests/test_batch_sampling.py`, branch
`tb-all-exc/tests`), AVX2 classes: 60 tests, 33 pass, 10 skip (GPU), 17 fail,
every one of them "no timestep ran blocked" in the four cases whose probes are
due every timestep (Dirac, step, and the Dirac-driven dump-window and objects
cases): the generic implementation cannot block there (§1). No identity
assertion fails.

## 7. Open

- The in-block samplers (§4.4, §4.5); until then Dirac and step block nowhere.
- A replayed probe read is an index lookup (`Engine::FootprintSlot()`, built
  once by `StopFootprintRecording()`: a dense box per field component where
  its cells fill at least 1/16 of their bounding box, a hash table for the
  rest), and `Engine_AVX2::SampleFootprint()` reads precomputed offsets. Until
  the review fix a read was a binary search behind an `std::ostringstream`,
  3.4 to 5.2 times slower than the old loop on mode-matching probes over whole
  planes; now 1.08 (50×50×200, 1000 timesteps) and 1.16 (120×120×40, 600
  timesteps) of the old loop's time, paired A/B of 3 runs each, 4 threads,
  medians, on the busy workstation. The rest is the sampling and the lookups.
- The ring holds `ring × footprint` floats; a footprint of millions of cells
  takes its 16 slots regardless of the 64 MB target.
- The intermediate flushes of the `*_FD` files and the 4 s status come at
  batch ends, i.e. less often than the old loop's ends could give them.
