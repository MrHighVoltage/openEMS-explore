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
generic implementation, main loop) and the AVX2 in-block sampler (§4.4) are
done; the Vulkan engine is implemented against §4.5.

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

Branch `tb-all-exc/avx2`, on top of the core and its tests (`2ad8c61`, the core with its review fixes):

| commit | what |
|---|---|
| `d9916ec` | `Engine_AVX2::FastEnergyFieldRows()`, `FastEnergy(eRows, hRows)`: a field's row sums, bit for bit those of `FastEnergy()` |
| `685951c` | `Engine::GetPreferredBatchLength()`; `RunFDTD()` keeps the batch length at least that |
| `fad8cb5` | `Engine_AVX2_Multithread`: in-batch sampling inside the blocked and the flat schedule (§4.4) |
| `2b32cca` | `--benchmark`: no probe cap where the engine samples inside its blocks |
| `00e148e` | `batch_sampling_identity.py`: `--env`, `--threads 0`, case `dirac_exact` |
| `e2b20f7` | `test_temporal_blocking.py`: Test13, Dirac and step blocked |

- **Done.** Everything in §3, the generic implementation of §4 (used by
  `Engine_AVX2`), and the AVX2 in-block sampler (§4.4): `Engine_AVX2_
  Multithread` samples the probes and the energy inside its blocks, so a
  batch runs blocked to its end. Dirac and step runs block (498 of 500 and
  398 of 400 timesteps on the identity models before the preferred batch
  length, 500 of 500 and 400 of 400 with it; the old loop 0). Identity holds on
  every case tried (§6 and below).
- **Shared code** the AVX2 work changed: `engine.h` (`GetPreferredBatchLength()`),
  `openems.cpp` (`bsMaxLen` at least that, two lines; the sampled loop's
  `NoteInterval()` call), `engine_benchmark.*`, and the slab contract in
  `engine_extension.h` (voltages only in voltage hooks, currents only in
  current hooks). The last sentence of §4.2 is replaced by §4.4's
  "`--benchmark`" item.
- **Verified** on the branch merged with the core's review fixes (`2ad8c61`):
  - `test_batch_sampling.py`, AVX2 classes: 60 tests, 50 pass, 10 skipped
    (GPU), none fail (on the core: 17 failed, "no timestep ran blocked").
  - `batch_sampling_identity.py`, every row passes (302): all twelve cases,
    flat and 4:16 × 1, 3 threads, and 8:16, 12:24, 16:32 × 1, 3, 8; all but
    `sine_long` with `OPENEMS_AVX2_TBLOCK_SPLIT=lines` (flat, 4:16, 8:16 × 3,
    8) and with `OPENEMS_BATCH_SAMPLING_RING` 1 and 2 (4:16, 8:16 × 3); all
    twelve with the thread calibration (`--threads 0`; flat, 4:16, 8:16).
  - `test_temporal_blocking.py`: 17 of 17, Test13 included.

  On the branch before that merge (same AVX2 code):
  - `tblock_identity.py` against the `bb3f974` binary: 160×128×192 PEC and
    PML (8 and 7 threads), 224³ Mur (8): every variant identical.
  - `--benchmark` on a 4.9 M-cell Dirac model: candidates up to k = 32, the
    report says the probes cap nothing; it recommended 6 threads, 15:30,
    2.4× the flat sweep.
  - Speed, indicative only (i9-13900K, not quiet; 192×160×160, Mur, a lumped
    port, three probes, 8 threads on the P-cores, `--avx2-temporal-block=12`,
    600 timesteps, median of 3 alternating rounds, MC/s): Dirac old loop flat
    395, old loop blocked 394 (runs flat), new loop blocked **1250**;
    Gaussian with probes every 32 timesteps old flat 560, old blocked 1302,
    new blocked 1387. The Dirac run evaluates the energy criterion at every
    timestep (first 60 s), the old loop single-threaded on the main thread.
Branch `tb-all-exc/vulkan`, on top of the core and its tests (`bb3f974`),
merged with the core's review fixes (`2ad8c61`) and the AVX2 work (`aef7da4`):

| commit | what |
|---|---|
| `5ff98ea` | `Engine_Vulkan`: in-batch sampling inside the flat and the blocked schedule, the energy by x-plane, the ring, the counter (§4.5) |
| `b3640ac` | `RunFDTD()`: the pipelined GPU loop with in-batch sampling; the old pipelined loop checks the end criteria after its first batch too |
| `14255b3` | `batch_sampling_identity.py`: `--loops`, `--rings`, `--gpu-index` |
| `a9f3a2e` | `Engine_Vulkan`: a sampled batch stashes the DFT dumps' samples |
| `98e1d79` | merge of the review fixes; the pipelined loop plans from the live state after a replay that left its plan |
| `08b444e` | `Processing::NextIntervalAt()`: a step-list entry at the timestep does not hide the interval (shared, see below) |
| `800919d` | merge of the AVX2 work; `Engine_Vulkan::GetPreferredBatchLength()` |
| `ca14991` | `test_gpu_temporal_blocking.py`: Dirac and step |

- **Done (Vulkan).** `Engine_Vulkan` samples the probes and the energy
  inside its blocks, flat and blocked, on discrete and integrated GPUs; the
  pipelined loop runs batch N+1 while the host replays batch N (§4.5). Dirac
  and step runs block: 500 of 500 and 400 of 400 timesteps on the identity
  models (the old loop 0).
- **Shared code** the Vulkan work changed: `openems.cpp` (the GPU loops;
  `planBatch()` reports what is read at a batch's last end, `lastProbe` and
  `lastEnergy`; `progress()` moved in front of the loops, with two defaulted
  parameters; the sampled standard loop drains the GPU at a benchmark
  boundary; the GPU's ring size and setup in the footprint block) and
  `Common/processing.cpp` (`08b444e`: `NextIntervalAt()` passes over a
  step-list entry at or before the timestep, so that ProcField's
  `AddStep(maxExcite)` no longer hides its Nyquist interval for one step in
  the standard loop; it changes the CPU loops only where nothing else is due
  in that step, e.g. `dump_fails`, where the old pipelined GPU loop, which
  checks before it processes, already checked at the interval).
- **Verified (Vulkan)** on the merged tree, RX 6800 unless noted:
  - `batch_sampling_identity.py --engine gpu`: 144 of 144 rows pass, all
    twelve cases × flat, 4:16 × pipelined, standard loop ×
    `OPENEMS_BATCH_SAMPLING_RING` default, 1, 5. On the UHD 770 (Intel
    RPL-S): `dirac`, `dirac_exact` × flat, 4:16 × both loops pass.
  - `test_batch_sampling.py`: GPU classes 60 of 60, AVX2 classes 60 of 60.
  - `batch_sampling_identity.py --engine avx2-multithreaded --threads 3`,
    with `08b444e`: 36 of 36 rows, all twelve cases × flat, 4:16, 8:16.
  - `test_gpu_temporal_blocking.py`: 23 of 23 (Dirac and step new);
    `gpu_tblock_identity.py`: all five cases identical.
  - `--benchmark` on a 160³ Dirac model: candidates up to k = 32, "probes
    sampled inside the blocks: they do not cap the block depth".
  - Speed, indicative only (not quiet; 224³ Mur, two probes, 3 alternating
    rounds, medians, MC/s, blocked = the default k=26 W=53): Dirac, 1500
    timesteps, the energy every timestep: old loop flat 3296, old blocked
    3283 (runs flat), new flat 3989, new blocked **8135**; Gaussian, probes
    every 6, 3000 timesteps: 5007, 9126, 5213, **10167**.
- **Not done.**
  - §6 below still describes the AVX2 state before its in-block sampler;
    §1 has the later results.
- **Open (AVX2).** A run still computes up to one batch past the timestep it
  stops at (a stop found in the replay), as §3.3 says: 775 timesteps for a
  stop at 425 on `gauss_exact`, about 33 000 past 167 533 on `sine_long` (half
  a second of a tiny grid). An energy sample at every timestep costs one more
  pass over each slab's rows from cache.

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
   `GetVoltRaw()`/`GetCurrRaw()` are the same reads without that test, and the
   extensions' updates (`eng->EngType::GetVoltRaw()` in the dispatcher's Impl
   functions) use them: they run only inside `IterateTS()`, where nothing is
   recorded or replayed. With the test inlined into their loops, the possible
   call to `BatchRead()` made the compiler reload every member and coefficient
   array per cell. On the AVX2 engine the Mur ABC ran twice the instructions,
   and flat runs with Mur were 6 % slower than at `1922d00`.
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
and returns true from `SamplesInsideBlocks()`; `RunFDTD()` then starts every
batch at `max(bsMinLen, GetPreferredBatchLength())` (`bsMaxLen` in openems.cpp),
so the gaps between samples no longer cap the batch depth anywhere — not only
in `--benchmark`.

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

### 4.4 The AVX2 in-block sampler

Built (`Engine_AVX2_Multithread`, `FDTD/engine_avx2_multithread.cpp`, the
comment block "In-batch sampling inside the schedules"):

- **One round per batch.** `IterateSampledTS()` checks the schedule as the
  generic one does, builds per-batch tables by offset `o = ts - start` (the
  probe sample of `ts+1`, and a prefix count of the energy samples), and calls
  `IterateTS()` once: one `m_startBarrier`/`m_stopBarrier` round, or the
  calibration's rounds (`CalibrateIterateTS()`), the workers sampling in all of
  them (`m_sp_on`). `SamplesInsideBlocks()` is true. A batch without samples
  is a plain `IterateTS()`.
- **Probe samples, blocked.** In `TrapezoidSweep()`, per pass and timestep
  `ts`, for the sample `t = ts+1`: the footprint voltages of the pass's E range
  `[eLo, eHi)` right after its `SlabApply2Voltages()` (`SampleVoltages()`), the
  currents of its hook range `[hS, hE)` right after its `SlabApply2Current()`
  (`SampleCurrents()`). No barrier of their own.
- **The partition.** At one `ts` the E ranges of all passes partition
  `[0, NX)` (core `[A0+t+1, B0-t)`, wedge `[e-t, e+t+1)`, the edge tiles held at
  0 and NX); the current updates `[An, hHi)` partition `[0, NX-1)` (core
  `[A0+t+1, B0-t-1)`, wedge `[e-t-1, e+t+1)`), and the hook ranges equal them
  except that the edge tile's reaches NX. So every footprint cell, x = NX-1
  included, is taken once per sample, by the pass that advances it through
  `ts`.
- **The level.** Every slab-capable extension (excitation, Mur, absorbing
  sheet, UPML, dispersive, Lorentz/conducting sheet, lumped RLC, TF/SF, steady
  state) writes voltages only in its voltage hooks and currents only in its
  current hooks; `engine_extension.h` now states this as part of the slab
  contract. So a voltage is final for `ts` after the pass's voltage phase and
  a current after its current phase, each phase ends at a barrier, and the
  next write of that field (a voltage hook of `ts+1`, or a later pass's first
  writes, which are voltages) comes after an update barrier that every thread
  reaches only after sampling. Reads of the other field meanwhile are reads.
- **The per-x index.** `IndexFootprint()` sorts the footprint per field by
  x-line (CSR: `m_fpX`, `m_fpCell` = the cell's index in a sample, `m_fpMem` =
  f8vector and lane), rebuilt only when the footprint changes. A pass takes
  the contiguous entries of its x-range, split evenly over the threads.
- **Energy samples.** A row (x, y) of `FastEnergy()` sums |V|² and |I|² with
  separate float accumulators, so the two can be summed at different times.
  `Engine_AVX2::FastEnergyFieldRows()` sums one field's rows exactly as
  `FastEnergy()` does (four rows interleaved, each with its own accumulator, so
  the FMA latency does not bound it); `FastEnergy(eRows, hRows)` adds the rows
  in row order in double as `FastEnergy()` does. After the voltage phase a
  pass sums |V|² of its E range's rows below NX-1, after the current phase
  |I|² of the rows of `[An, hHi)`, each thread over the rows it has just
  updated (`ForUpdateShare()`, the split of `UpdateVoltageShare()`/
  `UpdateCurrentShare()`, rows or lines). The rows go into one buffer per
  energy timestep of the block, `[slot][field][row]` floats, k slots, 8 bytes
  per row and slot (the row values are floats, so this is exact). At the block
  end `FinishBlockEnergy()` waits at one barrier, then thread `i` reduces the
  slots `i, i+n, ...` into `SetBatchEnergySample()`. No barrier after: the next
  block writes rows only past its first update barrier. A sample is therefore
  `CalcFastEnergy()` of its timestep bit for bit, for every thread count, tile
  and split. **The flat summation order is unchanged** (`FastEnergyRow()`
  keeps its one loop with two accumulators, each with the one-field helper's
  operations in its order; calling the helper twice was 5 % slower), and
  steady-state detection, which keeps `FastEnergyRows()`, is untouched.
- **Flat schedule.** The flat worker samples after `Apply2Current()` of each
  timestep, every thread the footprint cells and rows of its own x-lines, then
  one barrier (the next timestep's hooks may write any cell), and thread 0
  reduces the energy. Only at sample timesteps.
- **Steady state.** Sine runs have no energy samples (the energy criterion is
  off next to steady-state detection); `SlabBlockDepth()`, the probe records
  in `Apply2VoltagesSlab()` and `SlabBlockEnd()` run as before, beside the
  probe samples.
- **`numTS`.** The samplers never touch it; the workers compute `ts` from
  `numTS` read before their first barrier, thread 0 advances it at the end as
  before. `GetBlockedTimesteps()` counts in `BlockedWorker()` as before, now
  with long batches.
- **Batch length.** `Engine::GetPreferredBatchLength()` (new, default 0):
  this engine asks for `2k` when blocked, and `RunFDTD()` keeps `bsMaxLen` at
  least that (from the first batch on, and in `adaptBatchLength()`). A batch
  still ends at a dump, the ring, NrTS or a benchmark boundary. Without it a
  Dirac run's first batches were 1, 2, 4, ... timesteps, and on a grid where
  half a second is fewer than `k` timesteps the batch length capped the depth.
- **`--benchmark`.** The sampled loop calls `NoteInterval(longestRun,
  SamplesInsideBlocks())`; with in-block sampling the probes cap nothing
  (`MaxDepth()` unlimited) and the report prints "probes sampled inside the
  blocks (in-batch sampling): they do not cap the block depth" instead of
  "probes every N timesteps cap the block depth at N". This replaces the last
  sentence of §4.2 (the benchmark no longer takes the batch length as a
  depth). The old loop still notes the processing interval.
- **Cost.** Per pass and timestep two table lookups; at a probe timestep a
  gather of the pass's few footprint cells; at an energy timestep one more read
  of the slab's rows from cache, and one barrier plus a sequential reduction
  per block. With `--exact-endcriteria` or in the first 60 s a Dirac run has an
  energy sample at every timestep.

### 4.5 For Vulkan

Done on branch `tb-all-exc/vulkan` (§1). `Engine_Vulkan` samples inside its
blocks, flat and blocked, and the pipelined GPU loop runs the next batch
while the host replays the last one.

**Where a sample is taken.** `IterateSampledTS()` records the batch with,
at each sample timestep `t`, a gather of the footprint and the energy's
partial sums into a slot of a host-visible ring:

- flat sweep: at the top of iteration `t`, before its pre-voltage passes;
- blocked: at the top of each slab's step `t`, for the slab's E range
  `[eLo, eHi)` (`RecordSlabSweep()`). An x-line holds the end of `t-1` until
  that step: its voltages change first in the voltage update of `t`, and its
  currents are final for `t-1` (that update reads them) and change only in
  the current update of `t`, which comes after. The pre-voltage passes before
  the sample write only their own state (Mur, RLC, dispersive, local
  absorbing sheets, and Mur/RLC/sheets merged into the previous step's current
  update); unfused UPML's pre pass, which swaps field and flux, comes after
  it. No current-phase pass writes voltages on the GPU (§4.3). The E ranges of
  a timestep partition `[0, NX)`, so every cell and every plane is sampled
  once.

One barrier follows a sample (the next pass overwrites what it read); none
precedes it, since every stage ends with a barrier after its writes. A slab
whose E range holds no footprint cell dispatches no gather.

**The gather** (`batch_sample.comp`) reads a list of the footprint's cells
sorted by x, with a prefix table, so that the cells of an E range are one
contiguous run: each entry is the cell's index into its field buffer and
its footprint position, bit 31 set for a current. The slot is in footprint
order, so `CollectSampledBatch()` copies the slots into the base class's
`m_bsValues` as they are.

**The energy.** `CalcFastEnergyGPU()` is now the x-plane sum of
`reduce_energy_planes.comp` (one float pair (|V|², |I|²) per plane and chunk
of 4096 cells, which the host adds in plane order in double), the same the
steady-state checks use; `reduce_energy.comp` (256 workgroups over the whole
grid) is gone. A sampled energy is the same pipeline over the same planes,
so it is the live value bit for bit; the old loop's energies changed in
their last digits with it, both loops alike.

**The ring.** Two halves (the pipelined loop collects one while the next
batch writes the other), each `ring + 1` slots (the extra one for a batch's
last timestep), in host-visible memory, cached where the device has it and
so not in a small BAR. A slot holds the footprint and, where the energy
criterion is evaluated (no steady-state detection, no `--benchmark`), the
energy partials, `2 · NX · ceil(NY·NZ/4096)` floats. The default ring is the
most slots, at most 1024 and at least 1, that fit a byte budget of 64 MB
over both halves (`OPENEMS_GPU_BATCH_SAMPLING_MB`); `BatchSamplingRing()`
gives it to `RunFDTD()` before `OPENEMS_BATCH_SAMPLING_RING`, and
`SetupBatchSampling()` makes the ring. For 64×32×32 and a few hundred probe
cells that is 1024; a 10⁵-cell mode-matching probe gets about 80, and on
300³ with energy samples (13200 floats a slot) about 600.

**Probe cache and reads.** `GetVolt()`/`GetCurr()` call `BatchRead()` first;
the engine's own recording runs only outside `BS_RECORD`. With sampling the
probe cache is the core's footprint (`SetupProbeCache()`): the pipelined
reading serves the batch's last timestep from its end sample
(`ReadEndSample()`, `FootprintSlot()`), and `CalcFastEnergyGPU()` returns
the end sample's energy while a batch computes ahead; reading either without
a sample is a `std::logic_error`. Without a batch ahead, a missing end sample
reads the live fields.

**Standard loop** (`OPENEMS_GPU_PIPELINE=0`, host field readers):
`IterateSampledTS()` records, submits, waits and collects; the last end reads
the live fields, as before. Its benchmark branch drains the GPU at a
boundary.

**Pipelined loop** (`openems.cpp`, "Pipelined GPU main loop, with in-batch
sampling"):

```
plan and submit batch 0 (SubmitSampledSpeculative(): numTS stays)
while a batch is submitted:
    cur = it; CommitSampledSpeculative()      // wait, numTS = cur's last
    if it may run ahead: plan cur+1 from cur.nextStep, cur.nextPositions; submit
    CollectSampledBatch()                     // cur's samples to the replay
    replayBatch(cur) with the pipelined reading
    stop, failure: break
    adaptBatchLength(); progress()
    if nothing was submitted ahead: drain at a benchmark boundary, plan from the live state, submit
commit a batch computed past the stop (the timestep count includes it)
```

`planBatch()` also returns `lastProbe` and `lastEnergy`, whether a probe or
the energy criterion reads at the batch's last end: the GPU samples those too.
It does not run ahead where the host needs the GPU at the end of `cur` or the
plan may not hold: a frequency-domain dump due there that could not be
stashed (below), the relaxation of the end criterion after its first minute
(a batch planned before it would be replayed correctly, §3.10, but the one
after it would be planned from a stale ProcField interval),
a status that needs the live energy once relaxed (`progress()` puts it off to
the next batch, which ends drained), a replay that left its plan (§3.10: the
batch already ahead runs, the one after is planned from the live state), the
benchmark's boundary, NrTS, an abort.

Batches run about half a second (`adaptBatchLength()`), at most
`MaxSpeculativeBatch()`: what records without waiting for the GPU, 16 command
buffers of `OPENEMS_GPU_MAX_TS_PER_SUBMIT` (1024 timesteps), or one
steady-state period, beyond which `ScheduleSteadyStateChecks()` waits for a
check. `OPENEMS_GPU_SPEC_DEPTH=<n>` and `OPENEMS_GPU_SPEC_MAX_DEPTH=<n>`
cap the batch at n timesteps instead (the old loop keeps their old meaning).
The engine may run up to one batch past the timestep the replay stops at;
`Temporal blocking: <b> of <n>` counts it, the stop line and "Time for"
do not.

**Frequency-domain dumps.** A DFT sum cannot take a sample back, and a batch
may run past the stop, so a sampled batch adds no DFT sample. It gathers the
footprint of a DFT dump due at its end into a device-local stash (two halves),
and `AddDumpDFTSample()` adds the stash when the dump processes that timestep,
in a command of its own queued behind the batch running ahead, with the same
DFT pipeline (the stash read as a field through identity lines), hence the
same sums. A dump without a stash makes the loop wait there
(`DumpNeedsGPUAt()`).

**Time-domain dumps** gathered by the GPU (`RegisterDumpGather()`) still end
a batch, and are served at its end as before. Serving them inside a batch
would need a dump-ring slot per dump timestep in the batch, held until the
dump's thread has written it: a follow-up.

**The old pipelined loop** (`OPENEMS_BATCH_SAMPLING=0`) now evaluates the end
criteria at the end of its first batch too; it skipped that one, unlike the
standard loop, which showed in the end-criterion trace (Dirac: timestep 1).

With the review fix `08b444e` (§1) the two orders agree also where
ProcField's step-list timestep (maxExcite) is a batch end with nothing else
due before its next interval: the standard loop used to skip that interval
there, the old pipelined loop did not.

`GetBlockedTimesteps()` counts the timesteps recorded in blocks of depth >= 2
(`IterateTS()`), speculative batches included. `GetPreferredBatchLength()`
asks for two blocks of the blocked schedule (resolving the request if no
batch has been recorded yet). `SamplesInsideBlocks()` is true once the ring
is set up, so `--benchmark` offers blocked candidates as deep as the grid
allows, not as the probe interval does.

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
batch end). `test_temporal_blocking.py` passes (17), `test_gpu_temporal_blocking.py`
passes (23). `--benchmark` runs and reports "probes sampled inside the blocks"
where the engine has an in-block sampler, "probes every 3 timesteps cap the
block depth at 3" where it has not.

Current numbers on `tb-all-exc/integration` (incl. the review fixes, 6e3f90f),
binary of `6e3f90f`:

- `python/Tests/test_batch_sampling.py`: 120 tests, OK, engines avx2 and gpu
  (the Dirac/step cases that failed on the core with "no timestep ran blocked"
  pass now: the in-block samplers run them blocked, §4.4/§4.5).
- `batch_sampling_identity.py`: 0 failures on `--engine avx2-multithreaded`
  (all rows, flat and blocked, all thread counts and rings) and on
  `--engine gpu` (144 runs: flat and blocked, pipelined and standard, rings
  0/1/5, RX 6800).
- The two independent reviews (2026-11-06, /dev/shm/tball/review2 and
  review3): no correctness bug found. AVX2: 247 further old-loop vs in-batch
  comparisons, 0 failures. Vulkan: 144/144 stock identity on the RX 6800,
  24/24 on the UHD 770 iGPU, 100/100 custom odd shapes (W=2k, NX not divisible
  by W, probes at x=0 and x=NX-1, ring 1), extension-heavy models (fused and
  unfused UPML, TF/SF, sheets, Lorentz, Debye, RLC, TD+FD dumps) all identical,
  5× repeats of the barrier-stressful variants bit-identical. Their low
  findings were fixed in `6e3f90f` (a guard against NY==1 in
  `FastEnergyFieldRows()`, and a comment that the ring memory must stay
  HOST_COHERENT).

## 7. Open

Known limitations (documented, not bugs):

- A run computes up to one batch (~0.5 s) past the stop the replay finds; on
  the GPU that is a pipelined batch too (e.g. `custom` at 4:16: 1500 timesteps
  computed for a stop at 612).
- Computing the energy every timestep costs an extra pass while it is in cache:
  Dirac/step runs in the first 60 s, or any run with `--exact-endcriteria`.
- Time-domain dumps still end a batch on the GPU, and on the CPU so do FD
  dumps; the pipelined loop waits at a frequency-domain dump that could not
  be stashed. With steady-state detection, a pipelined GPU batch is one
  period long.
- No identity run covers the relaxed end-criterion paths after 60 s.
- Memory: the AVX2 energy-row buffer is sized for `k` slots per block whatever
  the ring or the actual number of energy samples (worst case with a forced
  `k:W` about 8·k·(NX-1)(NY-1) bytes); the GPU ring holds
  `2·(ring+1)·(footprint+energy)` floats within its 64 MB budget, and a
  footprint of millions of cells takes ring=1 (16 B/cell, which may exceed the
  budget) — allocation failure falls back to host sampling.
- Cross-version behaviour changes vs master `1922d00`, all intended: the GPU
  per-x-plane energy sum changes the last digits of every GPU energy in both
  loops; `08b444e` changes the CPU energy-check timesteps in the step-list edge
  case; the old pipelined GPU loop checks the end criterion after its first
  batch.
- The intermediate flushes of the `*_FD` files and the 4 s status come at
  batch ends, i.e. less often than the old loop's ends could give them.
