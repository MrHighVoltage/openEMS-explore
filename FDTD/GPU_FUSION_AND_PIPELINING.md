# GPU Kernel Fusion & Pipelined Processing

## Executive Summary

Two optimizations targeting discrete GPU performance, implemented together:

| Optimization | Root Cause | Mechanism | Expected Speedup |
|---|---|---|---|
| **Kernel Fusion** | 28 dispatches + 7 barriers per TS → 30–60% barrier overhead | Fuse UPML pre + Yee + UPML post into 1 dispatch per field | 1.3–1.5× on dGPU |
| **Pipelined Main Loop** | GPU idle during PA->Process() → ~2.5% utilization | GPU computes TS(N+1) while CPU processes TS(N) from probe cache | 5–40× on dGPU |

Both are backwards-compatible: they activate automatically when conditions are
met (UPML present, no CPU-only extensions) and fall back to the original code
path otherwise.

---

## 1. Problem: Discrete GPU Underperformance

On integrated GPUs (iGPU), the Vulkan engine performs well because CPU↔GPU
transfers use shared system memory with negligible latency.  On discrete GPUs
(dGPU), the engine runs at roughly half the speed of multi-threaded CPU engines
due to three compounding issues:

### 1.1 Barrier Overhead (30–60% of GPU time)

A typical UPML simulation records **28 dispatches** and **7 pipeline barriers**
per timestep into the command buffer:

```
 Pre-volt UPML×6  → barrier → Yee volt → barrier → V excite → barrier →
 Post-volt UPML×6 + Mur + TFSF → barrier →
 Apply2V (disp+Mur+RLC) →
 Pre-curr UPML×6 → barrier → Yee curr → barrier → I excite → barrier →
 Post-curr UPML×6 + TFSF + disp
```

Each `vkCmdPipelineBarrier` forces a full pipeline drain — all in-flight shader
invocations must complete before the next dispatch starts.  On dGPUs with deep
pipelines and high launch latency, these barriers cost 1–5 µs each.  With 7
barriers per timestep and ~50 µs of actual compute, barriers consume 14–35% of
wall-clock time.  The 28 dispatch descriptors + push constants represent another
~15% overhead from command processor serialization.

### 1.2 Sequential Main Loop (2.5% GPU utilization)

```
GPU:  ┌──50µs──┐ idle idle idle idle idle ┌──50µs──┐ ...
CPU:  IterateTS │←─── PA->Process (2ms) ──→│ IterateTS
```

The GPU completes a timestep in ~50 µs, then sits idle for ~2 ms while the CPU
runs `PA->Process()` (probe readback, frequency-domain processing, file I/O).
GPU utilization: `50µs / (50µs + 2000µs) ≈ 2.5%`.

### 1.3 PCIe BAR Read Scatter

Each `GetVolt(n,x,y,z)` reads 4 bytes over PCIe BAR at ~300 ns latency.
CPU L1/L2 caches miss on every VRAM read because the access pattern
(voltage/current line integrals, closed-loop current integrals) is sparse and
unpredictable from the CPU cache's perspective.  On iGPUs, these reads hit
system memory at ~5–10 ns.

---

## 2. Kernel Fusion Implementation

### 2.1 Concept

Replace the 3-phase UPML sequence (pre → Yee → post) with a single fused
shader that performs all three operations per cell in registers, eliminating
intermediate memory writes and inter-dispatch barriers:

```
BEFORE (per field, per timestep):
  6× UPML pre dispatches → barrier → 1× Yee → barrier → 6× UPML post → barrier
  = 13 dispatches + 3 barriers per field × 2 fields = 26+6

AFTER (per field, per timestep):
  1× Fused dispatch → barrier
  = 1 dispatch + 1 barrier per field × 2 fields = 2+2
```

### 2.2 Fused Shader Algorithm

Each thread processes one cell.  Instead of looking up whether the cell belongs
to a PML region via a per-cell index buffer, the shader uses **coordinate-based
PML detection** against a small array of PML region metadata (up to 6 regions):

```glsl
// Push constant includes numPmlRegions
// PML regions stored in an SSBO: { lo_x, lo_y, lo_z, hi_x, hi_y, hi_z, axis, dir }

// 1. PML pre-check: is this cell in any PML region?
for (uint r = 0; r < numPmlRegions; r++) {
    if (x >= region[r].lo_x && x < region[r].hi_x &&
        y >= region[r].lo_y && y < region[r].hi_y &&
        z >= region[r].lo_z && z < region[r].hi_z) {
        // PML pre-voltage: swap flux ↔ voltage (in registers)
        ...
    }
}

// 2. Yee voltage update (standard curl computation)
float curl = ...; // neighbor differences
volt[idx] = vv * volt[idx] + vi * curl;

// 3. PML post-voltage (same region loop)
for (uint r = 0; r < numPmlRegions; r++) {
    ...  // combine flux with Yee result
}
```

**Key properties:**
- PML cells execute all 3 phases in-register — no intermediate memory traffic
- Non-PML cells (majority) skip both PML loops at near-zero cost
- Compressed operator coefficients still work — opIdx lookup unchanged
- Each thread is fully independent — no workgroup barriers needed

### 2.3 Concatenated PML Buffers

The 6 UPML regions (±x, ±y, ±z faces) each have their own flux and coefficient
buffers.  For the fused shader, these are **concatenated** into single large
buffers with a per-region offset:

```
┌────────────────────────────────────────────────────────────┐
│  m_fusedVoltFlux                                           │
│  ┌──region0──┬──region1──┬──region2──┬──region3──┬──...──┐ │
│  │ 3×N₀ flt │ 3×N₁ flt │ 3×N₂ flt │ 3×N₃ flt │       │ │
│  └───────────┴───────────┴───────────┴───────────┴───────┘ │
└────────────────────────────────────────────────────────────┘

PML region metadata SSBO (per region, 9 uint32):
  { lo_x, lo_y, lo_z, size_x, size_y, size_z, pml_axis, pml_dir, offset }
```

The shader computes a region-local index from `(x - lo_x, y - lo_y, z - lo_z)`
and adds the region's `offset` to access the correct position in the
concatenated buffer.  GPU-side copies transfer data from the existing per-region
buffers during `SetupFusedUPML()`.

### 2.4 Descriptor Layout

Both fused shaders share a single descriptor set layout with 10 bindings:

| Binding | Content | Fused Voltage | Fused Current |
|---------|---------|---------------|---------------|
| 0 | field (volt or curr) | volt[] | curr[] |
| 1 | neighbor field (curr or volt) | curr[] | volt[] |
| 2 | opIdx[] | opIdx (per-cell) | opIdx (per-cell) |
| 3 | coeff_self (vv or ii) | vv_comp[] | ii_comp[] |
| 4 | coeff_curl (vi or iv) | vi_comp[] | iv_comp[] |
| 5 | PML region info SSBO | regions[] | regions[] |
| 6 | flux buffer | volt_flux[] | curr_flux[] |
| 7 | PML self-coeff | pml_vv[] | pml_ii[] |
| 8 | PML flux-old coeff | pml_vvfo[] | pml_iifo[] |
| 9 | PML flux-new coeff | pml_vvfn[] | pml_iifn[] |

**Push constant** (`FusedPC`, 20 bytes): `{ Nx, Ny, Nz, numComp, numPmlRegions }`

### 2.5 Activation Conditions

Fused dispatch activates when `m_hasFusedUPML == true`, which requires:
- At least one UPML region exists (`m_gpuUPML` non-empty)
- `SetupFusedUPML()` succeeded (buffer creation + shader compilation)

When active, the fused path replaces UPML-related dispatches in `IterateTS()`.
Non-UPML extensions (Mur, TF/SF, dispersive, RLC) continue to use their
separate dispatch slots before/after the fused kernel.  The `else` branch
preserves the entire original separate-dispatch path for non-UPML simulations.

### 2.6 Current Update Domain

The fused current shader dispatches over ALL `Nx×Ny×Nz` cells (same as
voltage), but includes a Yee domain check:

```glsl
bool inYeeDomain = (x < Nx-1) && (y < Ny-1) && (z < Nz-1);
if (inYeeDomain) {
    // Yee curl computation + coefficient application
}
```

PML boundary cells outside the Yee domain still get their PML pre/post
operations — this is correct because PML flux updates are valid for all cells
in the PML region, while the Yee curl is only valid for interior cells.

---

## 3. Pipelined Main Loop Implementation

### 3.1 Concept

Instead of the sequential `IterateTS → PA->Process → IterateTS → ...` loop,
overlap GPU compute with CPU probe processing:

```
BEFORE:
  GPU:  ┌─50µs─┐ idle idle idle ┌─50µs─┐ idle idle idle
  CPU:  submit  │← Process 2ms →│ submit │← Process 2ms →│

AFTER:
  GPU:  ┌─50µs─┐┌─50µs─┐┌─50µs─┐┌─50µs─┐
  CPU:  submit   │submit │submit │submit │
                 │←Proc──→│←Proc──→│←Proc──→│
                 (reads   (reads   (reads
                  cache)   cache)   cache)
```

### 3.2 Probe Gather Shader

The existing `gather_probes.comp` shader reads sparse field values and writes
them into a compact output buffer:

```
GPU field buffers (650 MB)
    ↓ gather_probes.comp (1 thread per probe cell)
GPU probe output buffer (few KB, host-visible)
    ↓ memcpy from the mapping
CPU probe cache (few KB)
```

This is appended to the end of each FDTD batch in the command buffer, after the
last timestep's current update completes.  A barrier ensures all field writes
are visible before the gather reads.

### 3.3 Probe Access Recording

The probe gather shader needs to know which cells `PA->Process()` will read.
Rather than requiring each `Processing` subclass to declare its cells (which
would need changes to every probe type), we use a **recording approach**:

1. Before the first `PA->Process()`, enable recording mode on the engine
2. During `PA->Process()`, every `GetVolt`/`GetCurr` call logs its
   `(fieldSel, linearIndex)` into a recording vector
3. After `PA->Process()`, deduplicate the recorded cells and build:
   - GPU index buffer (pre-linearized field indices)
   - GPU field selector buffer (0=volt, 1=curr)
   - GPU output buffer (host-visible, persistently mapped)
   - CPU cache map (`uint64_t key → uint32_t cacheIndex`)
   - Probe gather descriptor set and compute pipeline

This approach is completely transparent to the Processing framework — no changes
needed to `ProcessVoltage`, `ProcessCurrent`, `ProcessFieldProbe`, or any other
processing subclass.

### 3.4 CPU Probe Cache

```cpp
// Key encoding: fieldSel (0=volt, 1=curr) in bits 63-48, linearIndex in bits 47-0
uint64_t key = (uint64_t)fieldSel << 48 | linearIndex;

// During pipelined reading:
GetVolt(n, x, y, z) {
    auto it = m_probeCacheMap.find(key);
    if (it != end) return m_probeCacheCPU[it->second];
    // fallback: drain GPU (should not happen if recording was correct)
}
```

The cache is a flat `std::vector<float>` with the same layout as the GPU output
buffer.  `SnapshotProbeCache()` copies the mapped GPU output into this vector.

The recording sees only the probes that are due at the timestep it runs at.
Since `6900f90`, `RunFDTD()` also evaluates every other probe once while it records
(one with a start time), so that its cells are in the cache too; before, such
a probe missed the cache once due and read the live fields, which the
pipelined loop had already computed ahead.

### 3.5 Speculative Submission

To overlap GPU compute with CPU processing, we use a speculative pattern:

```cpp
// 1. Prime: run first batch normally
IterateTS(step);
DrainGPU();
SnapshotProbeCache();    // capture probe results for TS(N)

while (running) {
    // 2. Submit 1 speculative timestep (GPU starts, numTS NOT advanced)
    SubmitSpeculative(1);

    // 3. Process TS(N) from cache (CPU work, GPU concurrent)
    EnablePipelinedReading(true);
    step = PA->Process();   // reads from m_probeCacheCPU
    EnablePipelinedReading(false);

    // 4. Drain+commit: wait for speculative step, advance numTS
    DrainGPU();
    CommitSpeculative();
    SnapshotProbeCache();  // capture TS(N+1) results

    // 5. If step > 1, run remaining timesteps normally
    if (step > 1)
        IterateTS(step - 1);
        DrainGPU();
        SnapshotProbeCache();
}
```

`SubmitSpeculative()` internally calls `IterateTS()` but saves/restores `numTS`
so the public timestep counter doesn't advance until `CommitSpeculative()`.

### 3.6 Activation Conditions

The pipelined loop activates when `HasPipelinedProcessing()` returns true:
- `m_hasGPU_Probes == true` (probe gather pipeline was successfully created)
- no enabled field dump reads the whole fields (`SetHostFieldReaders()`)

The hybrid mode for CPU-only extensions that this list used to exclude was
removed in `3a7633a`; such a model now needs a CPU engine.

When conditions aren't met, `RunFDTD()` falls back to the original sequential
loop with no behavioral change.

---

## 4. Files Modified

### New Files

| File | Purpose |
|------|---------|
| `FDTD/gpu/shaders/fused_update_voltages.comp` | Fused Yee + UPML voltage shader (10 bindings) |
| `FDTD/gpu/shaders/fused_update_currents.comp` | Fused Yee + UPML current shader (10 bindings) |
| `FDTD/GPU_FUSION_AND_PIPELINING.md` | This documentation |

### Modified Files

| File | Changes |
|------|---------|
| `CMakeLists.txt` | Added `fused_update_voltages fused_update_currents` to `GPU_SHADER_NAMES` |
| `FDTD/engine_vulkan.h` | Added: `FusedPC` struct, fused pipeline members (`m_hasFusedUPML`, concatenated buffers, fused desc/pipeline objects, `SetupFusedUPML()`, `CleanupFusedUPML()`). Probe cache state (`m_pipelinedReading`, `m_recordingProbeAccess`, `m_recordedCells`, `m_probeCacheCPU`, `m_probeCacheMap`, staging buf, desc pool). Pipelined API (`StartProbeRecording()`, `SetupProbeCache()`, `SnapshotProbeCache()`, `EnablePipelinedReading()`, `HasPipelinedProcessing()`, `SubmitSpeculative()`, `CommitSpeculative()`). Modified `GetVolt()`/`GetCurr()` for recording and cache lookup. |
| `FDTD/engine_vulkan.cpp` | Added `#include "Common/processing.h"`. `SetupFusedUPML()` (~200 lines): concatenated buffer creation, descriptor set writes, fused pipeline creation. `CleanupFusedUPML()`. `SetupProbeCache()`: deduplicate recorded cells, create GPU index/selector/output buffers, gather pipeline. `SnapshotProbeCache()`: memcpy from ReBAR or staging download. `SubmitSpeculative()`/`CommitSpeculative()`: save/restore numTS for overlap. `IterateTS()`: conditional fused-vs-separate dispatch path, probe gather appended at end of batch. `CalcVoltageIntegralGPU()`/`CalcCurrentIntegralGPU()`: recording and pipelined reading support. Probe cache cleanup in `CleanupExtensions()`. |
| `openems.cpp` | `RunFDTD()`: probe recording during initial `PA->Process()`, `SetupProbeCache()` call, new pipelined main loop with speculative submission under `#ifdef WITH_GPU`, original loop preserved in `else` branch. |

---

## 5. Resource Lifecycle

### Fused UPML Resources

```
Init()
  └─ SetupFusedUPML()    (after SetupGPUExtensions)
       ├─ Compute total PML cells, build PMLRegionGPU metadata
       ├─ CreateGpuBuf: 6 concatenated buffers (flux×2, coeff×6)
       ├─ GPU→GPU copy from per-region buffers into concatenated buffers
       ├─ CreateGpuBuf: m_fusedPmlRegionInfo SSBO
       ├─ Create 10-binding descriptor layout
       ├─ Create pipeline layout (FusedPC push constant)
       ├─ Create descriptor pool (2 sets)
       ├─ Write voltage + current fused descriptor sets
       └─ Create 2 compute pipelines (fused volt, fused curr)

CleanupVulkan()
  └─ CleanupFusedUPML()
       ├─ Destroy 8 concatenated GpuBufs
       ├─ Destroy fused pipelines, pipe layout, desc layout
       └─ Destroy fused desc pool (frees both desc sets)
```

### Probe Cache Resources

```
RunFDTD()
  ├─ StartProbeRecording()       ← enable GetVolt/GetCurr cell logging
  ├─ PA->Process()               ← first process: records all cell accesses
  └─ SetupProbeCache(PA)
       ├─ Deduplicate recorded cells (sort + unique)
       ├─ Build m_probeCacheMap (unordered_map<uint64_t, uint32_t>)
       ├─ CreateGpuBuf: m_probeIdxBuf, m_probeSelBuf (device-local)
       ├─ Upload index + selector data
       ├─ CreateGpuBuf: m_probeOutBuf (host-visible)
       ├─ vkMapMemory → m_probeOutMapped
       ├─ Create desc pool (1 set, 5 storage buffers)
       ├─ Allocate + write probe gather desc set
       ├─ Create probe pipeline layout (if needed)
       └─ Create probe compute pipeline from gather_probes SPIR-V

CleanupExtensions()
  ├─ vkUnmapMemory(m_probeOutBuf) if mapped
  ├─ DestroyGpuBuf: m_probeIdxBuf, m_probeSelBuf, m_probeOutBuf, m_probeStagingBuf
  ├─ vkDestroyDescriptorPool(m_probeDescPool)
  └─ Clear cache map + cache vector
```

---

## 6. Performance Analysis

### 6.1 Kernel Fusion Dispatch Reduction

| Configuration | Dispatches/TS | Barriers/TS | Notes |
|---|---|---|---|
| Original (6 UPML regions) | 28 | 7 | 6+1+1+6+1+6+1+1+6+1 dispatches |
| Fused UPML only | 4–8 | 2–4 | 1 fused volt + exc + 1 fused curr + exc + non-UPML exts |
| Fused, no other extensions | 4 | 2 | Optimal: fused volt → barrier → exc → barrier → fused curr → barrier → exc |

### 6.2 Pipelined GPU Utilization

| Scenario | GPU Util. | Speedup vs Sequential |
|---|---|---|
| Sequential (current: 50µs compute, 2ms process) | 2.5% | 1× |
| Pipelined (50µs compute ∥ 2ms process) | ~97% | ~40× |
| Pipelined (50µs compute ∥ 50µs process) | ~50% | ~2× |
| Sequential with step=100 batching | ~71% | ~28× |

The pipelined loop is most impactful when `step == 1` (every timestep triggers
processing).  With large `step` values, the GPU already runs continuous batches
and the overlap benefit diminishes — but the overhead is negligible.

### 6.3 Probe Cache Memory

Typical probe counts and cache sizes:

| Probe Type | Cells Accessed | Cache Size |
|---|---|---|
| 1 voltage probe (line) | ~10 | 40 B |
| 1 current probe (2D loop) | ~200 | 800 B |
| 10 voltage + 10 current probes | ~2,100 | 8.4 KB |
| 1 field probe (E + H, interpolated) | ~24 | 96 B |
| 100 probes (mixed) | ~15,000 | 60 KB |

Even with hundreds of probes, the cache is tiny relative to GPU memory.  The
gather shader dispatch is correspondingly small (~60 workgroups for 15K cells).

---

## 7. Correctness Considerations

### 7.1 Fused PML Mathematical Equivalence

The fused shader produces bit-identical results to the separate 3-pass
algorithm.  For a cell in PML region `r`:

**Separate dispatches:**
1. Pre-voltage writes modified `volt[idx]` and `volt_flux[loc]` to memory
2. Yee reads `volt[idx]` (modified), computes curl, writes new `volt[idx]`
3. Post-voltage reads `volt[idx]` (Yee result), combines with flux

**Fused (in registers):**
1. Pre-voltage computes in registers: `f_help`, saves old volt
2. Yee uses modified volt value (same register), computes curl, updates
3. Post-voltage uses Yee result (same register), combines with flux
4. Single memory write at the end

The register computations follow the exact same algebraic sequence, just without
the intermediate global memory round-trips.

### 7.2 PML Boundary Cells Outside Yee Domain

For the current update, PML regions may include cells at `x=Nx-1`, `y=Ny-1`, or
`z=Nz-1` where the Yee curl is not defined (neighbor cells don't exist).  The
fused current shader handles this correctly:

```glsl
bool inYeeDomain = (x < Nx-1) && (y < Ny-1) && (z < Nz-1);
// PML pre runs for ALL cells in PML region (regardless of Yee domain)
if (inYeeDomain) { /* Yee curl */ }
// PML post runs for ALL cells in PML region
```

### 7.3 Probe Cache Staleness

In the pipelined loop, `PA->Process()` reads probe values from the **previous**
timestep's snapshot.  This is equivalent to the original sequential loop where
`PA->Process()` also reads the results of the just-completed batch — the
temporal relationship between compute and processing is preserved.

### 7.4 CalcTotalEnergyEstimate Bypass

Energy estimation (`CalcTotalEnergyEstimate`) does NOT use the probe cache — it
uses the full-field GPU energy reduction shader.  When
called during the pipelined loop, `EnablePipelinedReading(false)` has been
called, so `GetVolt`/`GetCurr` fall through to the normal drain+read path.

---

## 8. Relation to Previous Plans

This implementation realizes two of the four improvements described in
`GPU_FULLPIPELINE_PLAN.md` §2.3 and `GPU_ENGINE_ARCHITECTURE.md` §10:

| Plan Item | Status | Notes |
|---|---|---|
| Move PML to GPU | ✅ Done (prior work) | Separate dispatch path already existed |
| Fuse PML + Yee into single dispatch | ✅ Done (this PR) | `fused_update_voltages/currents.comp` |
| GPU probe gather for sparse readback | ✅ Done (this PR) | Recording-based cell discovery |
| Pipelined main loop | ✅ Done (this PR) | Speculative submission in `openems.cpp` |
| GPU-side DFT/FFT | ❌ Future | Would eliminate CPU processing entirely |
| Full kernel fusion (all extensions) | ❌ Future | Requires extension-specific fused shaders |
