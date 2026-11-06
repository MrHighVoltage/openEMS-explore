# Engine_Vulkan — GPU FDTD Architecture

## 1. Overview

`Engine_Vulkan` is a Vulkan compute-based FDTD engine that replaces the CPU
field-update loop with GPU dispatches.  Timesteps are batched into command
buffers with no CPU work in between.  The excitation and every extension
(PML, Mur, local absorbing sheets, dispersive materials, TF/SF, lumped RLC,
steady-state detection) run on the GPU.  A model with an extension that has no
GPU implementation stops with an error naming it; it needs a CPU engine.

Until 2026-11 such a model ran a hybrid loop instead, copying both fields to
the host and back twice per timestep (0.34 MC/s on 64³, against 7165 MC/s on
the GPU), and the field buffers could live in the PCI BAR.  Both are gone; see
`OPTIMIZATIONS.md` §1.2.

---

## 2. Memory Layout

```
┌─────────────────────────────────────────────────────────────┐
│                         GPU  VRAM                           │
│                                                             │
│  ┌──────────┐ ┌──────────┐   ┌────┐┌────┐┌────┐┌────┐     │
│  │ volt_buf │ │ curr_buf │   │ vv ││ vi ││ ii ││ iv │     │
│  │ 176 MB   │ │ 176 MB   │   │coef││coef││coef││coef│     │
│  │ RW+Xfer  │ │ RW+Xfer  │   │ RO │ │ RO ││ RO ││ RO │    │
│  └─┬────────┘ └─┬────────┘   └────┘└────┘└────┘└────┘     │
│    │ DMA        │ DMA       ┌────────────┐                  │
│    ▼            ▼           │ staging_buf│ (host-visible,   │
│  (copied on demand)         │  176 MB    │  host-cached)    │
│                             └────────────┘                  │
│                                                             │
│  ┌──────────────────────────────────────────────┐           │
│  │ Excitation buffers (idx, amp, delay, signal) │           │
│  │ Per volt + per curr                          │           │
│  └──────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
         │ PCIe (DMA through the staging buffer)
         ▼
┌─────────────────────────────────────────────────────────────┐
│                        CPU  RAM                             │
│                                                             │
│  volt_ptr (Engine base)    curr_ptr (Engine base)           │
│  Shadow copy, filled by    Shadow copy, filled by           │
│  SyncFieldsToHost() when   SyncFieldsToHost() when          │
│  something reads it        something reads it               │
└─────────────────────────────────────────────────────────────┘
```

### Field memory

The volt/curr buffers are plain `DEVICE_LOCAL`.  The host never reads them in
place: probes read the cells the GPU gathers at the end of each batch, field
dumps the cells it gathers at their timestep, and anything else copies both
fields by DMA through the staging buffer (`SyncFieldsToHost()`).

They used to be `DEVICE_LOCAL | HOST_VISIBLE` (the PCI BAR) whenever both fit.
A CPU read through that mapping is an uncached PCIe read, about 25 MB/s on an
RX 6800, and with a legacy 256 MB BAR the driver silently places what does not
fit in system RAM, where the update kernels then read it over PCIe — 10× slower
for three runs sharing one card.

---

## 3. Compute Shaders

Three GLSL compute shaders, compiled to SPIR-V at build time:

| Shader | Bindings | Push Constants | Workgroup | Dispatch |
|--------|----------|----------------|-----------|----------|
| `update_voltages.comp` | volt, curr, vv, vi | `{Nx, Ny, Nz}` | 256×1×1 | `⌈N/256⌉` |
| `update_currents.comp` | curr, volt, ii, iv | `{Nx, Ny, Nz}` | 256×1×1 | `⌈cN/256⌉` |
| `apply_excitation.comp` | field, amp, idx, delay, signal | `{count, ts, sigLen, period}` | 256×1×1 | `⌈count/256⌉` |

Where `N = Nx·Ny·Nz` (all cells) and `cN = (Nx-1)·(Ny-1)·(Nz-1)` (interior cells).

---

## 4. Command Buffer Architecture

Double-buffered command buffers with a utility buffer for one-shot operations:

```
┌────────────────────────────────────┐
│          VkCommandPool             │
│                                    │
│  m_cmdBufs[0]   m_cmdBufs[1]      │    m_utilCmdBuf
│  ┌───────────┐  ┌───────────┐     │    ┌───────────┐
│  │ FDTD work │  │ FDTD work │     │    │ Uploads,  │
│  │ slot 0    │  │ slot 1    │     │    │ downloads,│
│  └─────┬─────┘  └─────┬─────┘     │    │ one-offs  │
│        │              │           │    └─────┬─────┘
│  m_fences[0]    m_fences[1]       │    m_utilFence
│  (start signaled) (start signaled)│    (start unsignaled)
└────────────────────────────────────┘
         │              │
     m_cmdIdx alternates (0 → 1 → 0 → ...)
```

- `m_cmdBufs[m_cmdIdx]` — used by `IterateTS()` for FDTD compute
- `m_utilCmdBuf` — used by `RunSingleCommand()` for uploads, downloads, one-off commands
- Fences start **signaled** so the first `WaitForFences` is a no-op

---

## 5. Pure GPU Path — Main Loop Timeline

This is the critical path for performance.  The main loop in `RunFDTD()` is:

```cpp
while (running) {
    FDTD_Eng->IterateTS(step);   // GPU dispatch
    step = PA->Process();         // CPU probe processing
}
```

With in-batch sampling (the default since 2026-11-06,
`plans/BATCH_SAMPLING.md`) `step` is no longer the next probe timestep: a
batch runs about half a second, or to the next field dump, the GPU gathers
what the probes and the energy criterion read at each of their timesteps
inside it (inside the temporal blocks too), and the host replays the
processing of those timesteps afterwards, in the pipelined loop while the
GPU runs the next batch. So neither the probe interval nor a Dirac or step
excitation's interval of one timestep caps the batch or the block depth.
What follows describes one batch; `OPENEMS_BATCH_SAMPLING=0` runs the loop
as written above.

### What happens step by step:

```
 Call #    IterateTS(step)                         PA->Process()
 ──────   ─────────────────────────────────       ──────────────────────
   1      record cmdBufs[0], submit+fence[0]      DrainGPU: wait fence[0]
          flip → cmdIdx=1                         then fence[1] (no-op)
          [GPU starts FDTD batch 1]               read the probe cache
                                                  (probes, HDF5, etc.)

   2      wait fence[1] (no-op, signaled)         DrainGPU: wait fence[1]
          record cmdBufs[1], submit+fence[1]      then fence[0] (no-op)
          flip → cmdIdx=0                         read the probe cache
          [GPU starts FDTD batch 2]

   3      wait fence[0] (blocks if batch 1        ...
          not done during PA->Process #2)
          record cmdBufs[0], submit+fence[0]
          flip → cmdIdx=1
```

### Timeline diagram (ideal case where compute < process time):

```
         time ──────────────────────────────────────────────────────────►

GPU:     ┌──FDTD─1──┐         ┌──FDTD─2──┐         ┌──FDTD─3──┐
         │ Volt+Exc  │         │ Volt+Exc  │         │ Volt+Exc  │
         │ Curr+Exc  │         │ Curr+Exc  │         │ Curr+Exc  │
         └───────────┘         └───────────┘         └───────────┘

CPU:     ┌record┐              ┌record┐              ┌record┐
         │submit│              │submit│              │submit│
         └──┬───┘              └──┬───┘              └──┬───┘
            │  ┌──PA->Process──┐  │  ┌──PA->Process──┐  │
            │  │ DrainGPU:wait │  │  │ DrainGPU:wait │  │
            │  │ GetVolt       │  │  │ GetVolt       │  │
            │  │ HDF5 writes   │  │  │ HDF5 writes   │  │
            │  └───────────────┘  │  └───────────────┘  │
            ▼                     ▼                     ▼
Fence:   [0]=submit            [1]=submit            [0]=submit
```

### Timeline diagram (actual bottleneck: process < compute):

```
         time ──────────────────────────────────────────────────────────►

GPU:     ┌──FDTD─1──┐┌──FDTD─2──┐┌──FDTD─3──┐┌──FDTD─4──┐
         │ ~50 µs    ││ ~50 µs    ││ ~50 µs    ││ ~50 µs    │
         └───────────┘└───────────┘└───────────┘└───────────┘

CPU:     rec│wait│PA │rec│wait│PA │rec│wait│PA │rec│wait│PA │
                 │        │        │        │        │
                 ▼        ▼        ▼        ▼        ▼
             Drain:    Drain:   Drain:   Drain:   ...
             wait[0]   wait[1]  wait[0]  wait[1]
             ~50 µs    ~50 µs   ~50 µs   ~50 µs
             STALL!    STALL!   STALL!   STALL!
```

**This is the current bottleneck.** `PA->Process()` finishes in microseconds
for most probe types between sync points, but `DrainGPU()` blocks until the
GPU compute completes.  The GPU spends ~50 µs per timestep (for ~15M cells),
and the CPU can process probes faster than that, so the GPU is always the
bottleneck → `DrainGPU()` always stalls.

---

## 6. DrainGPU — Synchronization

```
DrainGPU()
┌─────────────────────────┐
│ for i in {0, 1}:        │
│   if gpuInFlight[i]:    │
│     vkWaitForFences(    │
│       fences[i])        │
│     gpuInFlight[i]=false│
└─────────────────────────┘
```

`DrainGPU()` is called from:
- **`GetVolt()` / `GetCurr()`** — a probe read that misses the probe cache
- **`SyncFieldsToHost()`** — full field download

Since probes call `GetVolt`/`GetCurr` hundreds of times per `PA->Process()`,
`DrainGPU()` is called hundreds of times — but only the first call actually
waits (subsequent calls see `gpuInFlight[i] == false`).

---

## 7. Field Access Paths

### GetVolt / GetCurr (Read)

```
GetVolt(n, x, y, z)
        │
        ├── recording (initial PA->Process())?  ── note the cell, read on
        │
        ├── pipelined loop and cell in the probe cache?
        │         │
        │    YES  ▼
        │   return the value the GPU gathered at the end of the batch
        │
        ▼
   DrainGPU()  ◄─── wait for both fences
        │
   m_hostDirty? ──YES──► SyncFieldsToHost()  ◄── 2 × 176 MB DMA download
        │
        ▼
   return Engine::GetVolt()  ◄── read from CPU shadow array
```

Nothing writes the shadow copy back: no CPU code changes the fields.

---

## 8. Extensions

Every extension the operator creates must have a GPU implementation.
`SetupGPUExcitation()` replaces the CPU excitation; the others are checked
against the GPU-native list (UPML, Lorentz/Drude and conducting sheets, TF/SF,
Mur, local absorbing sheets, lumped RLC, steady-state detection), and anything
else throws `"the extension … has no GPU implementation; use a CPU engine"`.
Cylindrical meshes never reach the GPU engine.

---

---

## 9. Initialization Sequence

```
Engine_Vulkan::New(op)
    │
    ▼
 Constructor
    │  Set all handles to VK_NULL_HANDLE
    │  Set all pointers to nullptr
    │
    ▼
 Init()
    ├── Engine::Init()         ◄── allocate CPU shadow arrays, create extensions
    ├── InitVulkan()           ◄── VkInstance → PhysDevice → Device → Queue
    │                               CmdPool → 3 CmdBufs → 3 Fences
    ├── CreateBuffers()        ◄── volt/curr, vv/vi/ii/iv (device-local),
    │                               staging (host-visible, host-cached)
    ├── CreateDescriptors()    ◄── descriptor sets for FDTD + excitation
    ├── CreatePipelines()      ◄── compile SPIR-V → 3 compute pipelines
    ├── UploadCoefficients()   ◄── CPU vv/vi/ii/iv → staging → GPU
    ├── SetupGPUExcitation()   ◄── convert CPU excitation → GPU buffers
    │                               remove CPU excitation extension,
    │                               reject extensions without a GPU path
    ├── Upload initial fields  ◄── via the staging buffer
    └── Print GPU info
```

---

## 10. Where the Performance Issues Arise

### Issue 1: Sequential Main Loop

```
┌──────────────────────────────────────────────────────────┐
│  while (running) {                                       │
│      FDTD_Eng->IterateTS(step);  // GPU: ~50 µs         │
│      step = PA->Process();        // CPU: ~1-5 ms        │
│  }                                    ▲                  │
│                                       │                  │
│  PA->Process calls GetVolt/GetCurr    │                  │
│  which calls DrainGPU() ──────────────┘                  │
│  GPU is IDLE during entire PA->Process                   │
└──────────────────────────────────────────────────────────┘
```

The GPU finishes in ~50 µs but the CPU takes 1-5 ms to process probes.
During that entire CPU-side processing, the GPU sits idle.  This means
**GPU utilization is roughly `50µs / (50µs + 2000µs) ≈ 2.5%`**.

### Issue 2: DrainGPU Blocks Before PA->Process Can Start

With the current double-buffered design, `IterateTS` submits fire-and-forget,
then `PA->Process()` immediately calls `GetVolt` → `DrainGPU()` → fence wait.
In practice, the GPU hasn't finished the just-submitted work yet, so the CPU
blocks for the full ~50 µs compute time.

### ~~Issue 3: PCIe BAR Read Latency~~

Resolved: probes read the probe cache, and the fields are no longer mapped
through the BAR at all (§2).

### Possible Future Improvements

1. ~~**Batch multiple timesteps**~~ — Already works via `IterateTS(step)`.

2. **Move probe processing to GPU** — compute integrals, DFTs, and field
   probes as additional compute shaders.  This would eliminate the
   GPU→CPU→GPU round-trip entirely.

3. ~~**Asynchronous probe download**~~ — Superseded by pipelined main loop
   (see `GPU_FUSION_AND_PIPELINING.md`).

4. ~~**Sparse field readback**~~ — ✅ Implemented via `gather_probes.comp` +
   probe cache in the pipelined main loop.  See `GPU_FUSION_AND_PIPELINING.md`.

5. ~~**Kernel fusion (UPML + Yee)**~~ — ✅ Implemented via
   `fused_update_voltages.comp` / `fused_update_currents.comp`.
   See `GPU_FUSION_AND_PIPELINING.md`.

6. ~~**Pipelined main loop**~~ — ✅ Implemented with speculative submission
   in `openems.cpp`.  See `GPU_FUSION_AND_PIPELINING.md`.
