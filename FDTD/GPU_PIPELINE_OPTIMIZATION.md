# GPU Pipeline Optimization: Energy Reduction, Probe Gather, Field Dumps

## Problem Statement

The `RunFDTD()` main loop exhibited periodic stalls (~300 ms) caused by two operations
that forced full GPU↔CPU synchronization and data transfer every few seconds:

1. **`CalcTotalEnergyEstimate()` → `CalcFastEnergy()`**: Scans ALL grid cells
   (`3×Nx×Ny×Nz` floats ≈ 650 MB for a typical 27M-cell mesh) on the CPU side.
   Even with ReBAR, this means reading ~650 MB over PCIe via random-access patterns
   through memory-mapped VRAM — extremely slow.

2. **`PA->Process()` → `CalcVoltageIntegral()` → `GetVolt()` → `DrainGPU()` +
   `SyncFieldsToHost()`**: Voltage/current probes trigger a GPU drain and, without
   ReBAR, a full field download just to read a handful of cells along a line.

3. **No threading separation**: CPU processing (file I/O, FFT, data writing) runs
   sequentially with GPU submission — the GPU sits idle while CPUs process data.

## Solution Architecture

Three optimizations, implemented in priority order:

### 1. GPU Energy Reduction (highest impact)

**Shader**: `FDTD/gpu/shaders/reduce_energy.comp`

Computes total field energy entirely on-GPU using a parallel reduction:
- 256 workgroups × 256 threads = 65,536 threads
- Grid-stride loop: each thread accumulates `v0²+v1²+v2²` (E) and `c0²+c1²+c2²` (H)
  over its assigned cells
- Shared memory tree reduction within each workgroup (7 barrier-synchronized steps)
- Each workgroup writes one (E_partial, H_partial) float pair → 2 KB total output

The CPU sums 256 partial sums (trivially fast), applies `ε₀` and `μ₀` constants.

**Data flow**:
```
GPU field buffers (650 MB, already resident)
    ↓ reduce_energy.comp dispatch (256 workgroups)
GPU partial sums buffer (2 KB)
    ↓ ReBAR direct read or staging download
CPU: sum 256 + 256 floats → return ε₀·E + μ₀·H
```

**Performance**: Eliminates ~650 MB PCIe transfer. GPU reduction runs in ~100 μs.
Total CalcFastEnergy time drops from ~300 ms to ~1 ms.

### 2. GPU Voltage/Current Integral (moderate impact)

**Methods**: `CalcVoltageIntegralGPU()`, `CalcCurrentIntegralGPU()`

For voltage probes (line integrals), we bypass the generic `GetVolt()` virtual dispatch:
- **With ReBAR**: After `DrainGPU()`, read directly from `m_voltMapped` / `m_currMapped`.
  A line integral touches only a few dozen cells — just a few PCIe reads, negligible.
- **Without ReBAR**: Fall back to `SyncFieldsToHost()` + `Engine::GetVolt()` (cached
  after the first call per batch, so subsequent probes are free).

The key improvement is avoiding the per-cell virtual dispatch overhead and ensuring
the integration uses the optimal path for the memory configuration.

### 3. Async Processing Thread (removed)

An earlier design explored a background `std::thread` with mutex/condition_variable
synchronization to run `PA->Process()` concurrently with the next GPU batch. It was
never wired into `RunFDTD()` and was superseded before that integration happened by
the pipelined main loop with speculative submission (see `GPU_FUSION_AND_PIPELINING.md`),
which provides the same overlap without a second OS thread or thread-safety concerns.
The unused scaffolding (`StartAsyncProcessing()`, `AsyncWorkerLoop()`,
`TriggerAsyncProcessing()`, `WaitAsyncProcessing()`) has been removed.

### 4. Field Dumps Gathered on the GPU

A field dump (`ProcessFields` and its TD, FD and SAR subclasses) used to read
the whole fields: `CalcField()` called `SyncFieldsToHost()`, which drained the
GPU and copied **both** volt and curr to the host, whatever the dump's type
and extent, and the GPU stayed idle until the extraction was done.  Three
things made that the dominant cost of a dump-heavy GPU run:

1. **CPU reads of mapped VRAM.**  With ReBAR field buffers (the default when
   both fields fit the BAR), `SyncFieldsToHost()` `memcpy`'d straight out of
   the mapping.  Every load is an uncached PCIe read: about 25 MB/s on the
   RX 6800, 250 ms for one dump of a 64³ grid.  An earlier asynchronous
   download pipeline (a snapshot copy plus a transfer on a second queue,
   `ea82f59`) switched itself off for ReBAR on the premise that the memcpy
   was already fast, and its 18× win was measured with
   `--gpu-no-rebar-fields`, so the default path never had it.
2. **Wrong timesteps.**  The pipelined loop submits the next batch before
   `PA->Process()`, so a dump that read the live buffers saw a later
   timestep: the one after (depth 1, ReBAR or not, from the second dump on),
   or a whole interval later (`OPENEMS_GPU_SPEC_DEPTH=auto`, because
   `FinishAsyncFieldDownload()` cleared `m_hostDirty` behind the submitted
   batch's back and the next snapshot was skipped).  All earlier dump
   identity checks compared GPU runs with each other, so nothing caught it.
3. **Bytes.**  Even by DMA, every dump moved 24 bytes per cell, then copied
   them again into the host shadow, for a plane as for the whole volume.

**Design** (`76fa8f6` engine, `d38071b` dumps).  At `InitProcess()` every dump registers its *footprint* with the
engine (`RegisterDumpGather()`): all three components of the one field it
reads (volt for E, J, D; curr for H, B, rotH) on the grid lines it samples,
widened by the lines its interpolation reads (`GPUStencil()` in
`processfields.cpp`: none 0, node −1..0, cell 0..+1, rotH −1 more for the
curl), and a `due(ts)` predicate (`Processing::IsDueAt()`, the side-effect-free
twin of `CheckTimestep()`).  Every batch that ends on a timestep where a dump
is due records, at its end and in the same command buffer,
`dump_gather.comp` into a host-visible (cached) ring slot.  The dump waits
only for that batch's fence (the next batch is already running) and
interpolates from the slot with the unchanged templated extractor
(`FieldCalc::FootprintView` stands in for the engine), so its output is
bit-identical to extracting from the whole fields.  A slot is reused once its
timestep is older than the one the outermost `IterateTS()` started from.
Timestep 0 and the hybrid loop, where no batch ends on the dump, gather on
the spot (`AcquireDumpGather()`) when the GPU still holds the timestep.

Frequency-domain dumps (FD, NF2FF boxes, SAR) go further: `dump_dft.comp`
adds the footprint times the sample's DFT weight (computed on the host by
the same `ProcessFieldsFD::DFTWeight()` as on the CPU) to running sums kept
on the GPU, and `DumpFDData()` reads the sums once and interpolates their
real and imaginary parts.  Interpolation is linear, so this equals summing
the interpolated fields up to float rounding (≤ 6e-7 relative here).  The
sums may take a quarter of the device-local heap; beyond that, or with
`OPENEMS_GPU_DUMP_DFT=0`, a dump gathers every sample and sums on the host.

Whole-field reads remain for what the GPU cannot serve, and are now always
by DMA into the `HOST_CACHED` staging buffer.  `GetVolt()`/`GetCurr()` read the
host copy once it is current and serve at most `BarReadLimit()` scattered
values from the mapping before syncing by DMA.  A dump that cannot register
(or `OPENEMS_GPU_DUMP_GATHER=0`) makes `HasPipelinedProcessing()` false, so
it runs in the standard loop, where the live buffers are the timestep being
processed.  The second queue and the snapshot ring are gone.

**Speculation.**  The same audit found the pipelined loop sizing the
speculative batch from the interval that led to the current timestep, before
`PA->Process()` reported the next one: with `OPENEMS_GPU_SPEC_DEPTH=4` a
batch ran from 129 to 133 past dumps due at 130, with `auto` from 120 to 130
past an FD sample due at 129.  The loop now stops the batch at the next
timestep `PA->IsDueAt()` reports (`c3bba67`).  Depth 1, the default, could
not overshoot.

**Measured** (RX 6800, 160³ PEC, 4000 timesteps, 63 TD dumps or an NF2FF
box of 6+6 faces at 3 frequencies; stepping-loop seconds, raw data in
`python/Tests/results/gpu_dumps_2026-11-01.txt`):

| Dumps | Old, ReBAR (default) | Old, device-local | New | CPU AVX2, 32 threads |
|---|---|---|---|---|
| none | 1.32 | 1.31 | 1.32 | 19.1 |
| E, one plane | 297.8 | 2.28 | **1.34** | 19.3 |
| NF2FF box | 149.6 | 1.97 | **1.30** | 20.1 |
| E, whole volume | 298.6 | 3.29 | **3.08** | 21.0 |

Plane and NF2FF dumps now cost nothing measurable (12.2 and 12.6 GMC/s
against 12.4 without dumps); before, the default path ran at 55 and 110
MC/s, below the CPU engine.  A whole-volume dump still moves one field and
is interpolated and written on the host, about 28 ms per 49 MB dump here.
New results do not depend on the field-buffer placement (`new-norebar`:
1.39, 1.33, 3.10 s).

**Correctness.**  `python/Tests/gpu_dump_check.py` runs one model with every
TD dump type in every interpolation (volumes, planes, sub-boxes, sub-sampled,
a start/stop window, VTK), FD dumps, an NF2FF box and SAR against
`--engine=sse`, through eleven GPU variants: default, device-local fields,
speculative depths auto and 4, the standard loop, host-side DFT, whole-field
dumps, forced temporal blocking with 7-step submits, no probes, a local
absorbing sheet, and the hybrid loop.  All match to ≤ 4e-5 of the dataset
group's maximum, the same as `--engine=basic` against `--engine=sse` (3e-5).
TD dumps are bit-identical across all GPU variants, including the
whole-field path.  The previous build fails it in every TD file.

## Files Modified

These tables record the change that introduced §1–§3 and the asynchronous
dump download, which §4's gather has since replaced along with the
functions named below (`SetHasFieldDumps()`, `BeginAsyncFieldDownload()`,
the dump queue and the snapshot ring).

### New Files
| File | Purpose |
|------|---------|
| `FDTD/gpu/shaders/reduce_energy.comp` | GPU parallel energy reduction shader |
| `FDTD/GPU_PIPELINE_OPTIMIZATION.md` | This documentation |

### Modified Files

| File | Changes |
|------|---------|
| `CMakeLists.txt` | Added `reduce_energy` to `GPU_SHADER_NAMES` |
| `FDTD/engine_vulkan.h` | Added energy reduction members (`m_energyPartialBuf`, `m_energyMapped`, descriptor/pipeline objects, `EnergyPC` struct, `ENERGY_NUM_WG`), public methods (`CalcFastEnergyGPU`, `CalcVoltageIntegralGPU`, `CalcCurrentIntegralGPU`). Async dump download: `SetHasFieldDumps()`/`HasFieldDumps()`, `HasDumpTransferQueue()`, `BeginAsyncFieldDownload()`, `HasPipelinedProcessing()` now also true for dump-only sims, dump queue/buffer/sync members (`m_dumpQueue`, `m_dumpSnapVolt/Curr`, `m_dumpStageVolt/Curr`, fences/semaphores, `DUMP_RING`). |
| `FDTD/engine_vulkan.cpp` | Added `#include "tools/constants.h"`. Init: calls `SetupGPU_EnergyReduction()`. CleanupVulkan: calls `CleanupGPU_EnergyReduction()`. New method implementations: `SetupGPU_EnergyReduction()`, `CleanupGPU_EnergyReduction()`, `CalcFastEnergyGPU()`, `CalcVoltageIntegralGPU()`, `CalcCurrentIntegralGPU()`. Async dump download: independent queue family selection in `InitVulkan()` (`OPENEMS_GPU_DISABLE_DUMP_ASYNC` opt-out), `EnsureDumpBuffers()`, `BeginAsyncFieldDownload()`, `FinishAsyncFieldDownload()`, `CleanupDumpAsync()`; `SyncFieldsToHost()` checks for a pending async request first; `m_stagingBuf` and the new dump staging buffers now prefer a `HOST_CACHED` memory type. |
| `FDTD/engine_interface_fdtd.cpp` | Added `#include "engine_vulkan.h"` (under `WITH_GPU`). GPU fast-path dispatch at top of `CalcFastEnergy()` and `CalcVoltageIntegral()` |
| `openems.cpp` | `RunFDTD()`: detects registered `ProcessFields` dumps and calls `SetHasFieldDumps()`; pipelined loop calls `BeginAsyncFieldDownload()` once each committed state is stable (before the next `SubmitSpeculative()`); the sequential fallback loop does the same after `IterateTS()` as a defensive no-op path for `m_hasCPUExtensions` sims. |

## Resource Lifecycle

### Energy Reduction Resources
```
Init()
  └─ SetupGPU_EnergyReduction()
       ├─ CreateGpuBuf(m_energyPartialBuf, 2KB, ReBAR or device-local)
       ├─ vkMapMemory if ReBAR → m_energyMapped
       ├─ Create descriptor layout (3 bindings: volt, curr, partials)
       ├─ Create pipeline layout (EnergyPC push constant, 8 bytes)
       ├─ Create m_energyDescPool (maxSets=1, 3 storage buffers)
       ├─ Allocate + write m_energyDescSet
       └─ Create compute pipeline from reduce_energy SPIR-V

~Engine_Vulkan()
  └─ CleanupVulkan()
       └─ CleanupGPU_EnergyReduction()
            ├─ vkUnmapMemory(m_energyPartialBuf)
            ├─ DestroyGpuBuf(m_energyPartialBuf)
            ├─ vkDestroyPipeline(m_energyPipeline)
            ├─ vkDestroyPipelineLayout(m_energyPipeLayout)
            ├─ vkDestroyDescriptorSetLayout(m_energyDescLayout)
            └─ vkDestroyDescriptorPool(m_energyDescPool)
```

## Descriptor Layout

### Energy Reduction (set 0)
| Binding | Type | Content |
|---------|------|---------|
| 0 | STORAGE_BUFFER (readonly) | volt[] — 3×N×sizeof(float) |
| 1 | STORAGE_BUFFER (readonly) | curr[] — 3×N×sizeof(float) |
| 2 | STORAGE_BUFFER (writeonly) | partials[] — 2×256×sizeof(float) |

**Push constant** (8 bytes): `{ uint32_t N; uint32_t numWG; }`

## Future Work

- ~~**Async processing integration**~~ — Superseded by the pipelined main loop
  with speculative submission (see `GPU_FUSION_AND_PIPELINING.md`), which
  provides better overlap without thread-safety concerns.

- ~~**GPU probe gather shader**~~ — ✅ Implemented. `gather_probes.comp` is now
  wired up via `SetupProbeCache()` using a recording-based cell discovery
  approach.  See `GPU_FUSION_AND_PIPELINING.md` §3.

- **Batched energy reduction**: Instead of a separate `RunSingleCommand()` dispatch,
  the energy reduction could be appended to the main FDTD command buffer every N
  timesteps, eliminating the extra submit/fence overhead entirely.

- **Full kernel fusion**: Extend the fused Yee+UPML shaders to also inline
  dispersive, Mur, and RLC operations for further dispatch reduction.
  See `GPU_FUSION_AND_PIPELINING.md` §8.

- **Audit other `HOST_VISIBLE|HOST_COHERENT`-only allocations for the same
  uncached-read issue** (§4): the excitation/coefficient upload buffers are
  write-only from the CPU so they're unaffected, but any future buffer the
  CPU reads back from should request `HOST_CACHED` too.
