/*
*	Copyright (C) 2026 openEMS contributors
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*/

#ifndef ENGINE_VULKAN_H
#define ENGINE_VULKAN_H

#ifdef WITH_GPU

#include "engine.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <deque>
#include <climits>
#include <complex>
#include <functional>
#include <algorithm>
#include <string>
#include <chrono>

class Operator_Ext_Excitation;
class Operator_Ext_UPML;
class Operator_Ext_LorentzMaterial;
class Operator_Ext_DebyeMaterial;
class Engine_Ext_SteadyState;
class Operator_Ext_TFSF;
class Operator_Ext_Mur_ABC;
class Operator_Ext_LumpedRLC;

/*!
 * \brief GPU-accelerated FDTD engine using Vulkan compute shaders.
 *
 * The voltage and current fields live on the GPU as storage buffers.
 * The CPU-side volt_ptr / curr_ptr (inherited from Engine) serve as a
 * shadow copy that is lazily synchronised on the first GetVolt / GetCurr
 * call after a GPU update.
 *
 * Timesteps are batched into Vulkan command buffers with no CPU work in
 * between.  The excitation and every extension run on the GPU; a model with
 * an extension that has no GPU implementation needs a CPU engine.
 */
class Engine_Vulkan : public Engine
{
public:
	//! Single Vulkan baseline for the whole engine -- instance creation,
	//! preflight probe, and device selection all negotiate against this one
	//! value.  1.3 is the lowest version that provides, as core, everything
	//! the engine's performance work needs: subgroup size control and
	//! synchronization2 (1.3), timeline semaphores and 8-bit storage (1.2),
	//! subgroup arithmetic and 16-bit storage (1.1).
	static constexpr uint32_t API_VERSION = VK_API_VERSION_1_3;

	//! Optional device capabilities negotiated at device creation.  Each flag
	//! is only true when the feature was reported supported *and* enabled.
	struct DeviceCaps
	{
		bool subgroupSizeControl   = false;
		bool computeFullSubgroups  = false;
		bool synchronization2      = false;
		bool timelineSemaphore     = false;
		bool storageBuffer16       = false;
		bool storageBuffer8        = false;
		bool shaderInt16           = false;
		bool shaderInt8            = false;
		uint32_t minSubgroupSize   = 0;
		uint32_t maxSubgroupSize   = 0;
	};
	const DeviceCaps& GetDeviceCaps() const { return m_caps; }
	//! Width actually chosen for the compressed-operator index (8/16/32 bits).
	uint32_t GetOpIdxBits() const { return m_opIdxBits; }

	static Engine_Vulkan* New(const Operator* op);
	//! Enable per-batch GPU timestamp profiling and fence-wait accounting.
	static void SetEnableProfiling(bool enable) { s_enableProfiling = enable; }
	static bool GetEnableProfiling() { return s_enableProfiling; }
	//! Fast-fail probe that allocates and frees representative GPU buffers
	//! for the given grid size before costly operator preparation begins.
	//! strictCoeffReserve=true also reserves a conservative coefficient/index budget.
	static bool PreflightAllocationForGrid(unsigned int Nx, unsigned int Ny, unsigned int Nz,
	                                      bool strictCoeffReserve, std::string* errMsg = nullptr);
	virtual ~Engine_Vulkan();

	virtual void Init();
	virtual void Reset();

	virtual bool IterateTS(unsigned int iterTS);

	virtual TemporalBlockingRequest GetTemporalBlocking() const {return m_tbRequest;}
	virtual bool SetTemporalBlocking(const TemporalBlockingRequest& req, bool announce = true);
	virtual TemporalBlockingPlan PlanTemporalBlocking(const TemporalBlockingRequest& req) const;
	virtual std::vector<TemporalBlockingRequest> TemporalBlockingCandidates(unsigned int maxDepth) const;

	//! GPU-side energy estimation — avoids downloading entire field arrays.
	double CalcFastEnergyGPU() const;

	// --- Pipelined processing support -------------------------------------
	//! Begin recording cell accesses. Call before PA->Process().
	void StartProbeRecording() { m_recordedCells.clear(); m_recordingProbeAccess = true; }
	//! Stop recording and build probe cache from recorded cells.
	//! Must be called after PA->Process(). Creates GPU gather resources.
	void SetupProbeCache(class ProcessingArray* PA);
	//! After DrainGPU, snapshot gathered probe values into CPU cache.
	void SnapshotProbeCache();
	//! Enable/disable pipelined read mode. When enabled, GetVolt/GetCurr
	//! return from the CPU-side probe cache instead of VRAM.
	void EnablePipelinedReading(bool enable) { m_pipelinedReading = enable; }
	//! Check if pipelined processing is active and usable.
	//! Probes read the probe cache and field dumps the cells the GPU gathered
	//! at their timestep (RegisterDumpGather()), so neither needs the GPU to
	//! still hold the timestep being processed, and the next batch can run
	//! meanwhile.  A dump the GPU cannot serve reads the whole fields, which
	//! only holds that timestep until the next batch is submitted.
	bool HasPipelinedProcessing() const
	{ return (m_hasGPU_Probes || !m_dumps.empty()) && !m_hostFieldReaders; }
	//! Submit a speculative batch that does NOT advance the public numTS.
	void SubmitSpeculative(unsigned int nTS);
	//! Drain the speculative batch and advance numTS.
	void CommitSpeculative();
	//! Wait for all in-flight GPU work to complete.
	void DrainGPU() const;

	// --- Field access (lazy sync from GPU) --------------------------------
	// Probes read the probe cache in the pipelined loop.  Any other read
	// brings both fields over by DMA, once per batch.
	inline virtual FDTD_FLOAT GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		uint64_t linIdx = (uint64_t)n * m_fieldN + x * m_strideYZ + y * numLines[2] + z;
		if (m_recordingProbeAccess)
			m_recordedCells.push_back((uint64_t)0 << 48 | linIdx);
		else if (m_pipelinedReading)
		{
			const uint64_t key = (uint64_t)0 << 48 | linIdx;
			auto it = std::lower_bound(m_probeCacheKeys.begin(), m_probeCacheKeys.end(), key);
			if (it != m_probeCacheKeys.end() && *it == key)
			{
				m_probeCacheHits++;
				return m_probeCacheCPU[(size_t)(it - m_probeCacheKeys.begin())];
			}
			m_probeCacheMisses++;
			// Fallback: not in cache — must drain GPU
		}
		if (!m_gpuDrained) DrainGPU();
		if (m_hostDirty) SyncFieldsToHost();
		return Engine::GetVolt(n,x,y,z);
	}
	inline virtual FDTD_FLOAT GetVolt(unsigned int n, const unsigned int pos[3]) const
	{ return GetVolt(n, pos[0], pos[1], pos[2]); }
	inline virtual FDTD_FLOAT GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		uint64_t linIdx = (uint64_t)n * m_fieldN + x * m_strideYZ + y * numLines[2] + z;
		if (m_recordingProbeAccess)
			m_recordedCells.push_back((uint64_t)1 << 48 | linIdx);
		else if (m_pipelinedReading)
		{
			const uint64_t key = (uint64_t)1 << 48 | linIdx;
			auto it = std::lower_bound(m_probeCacheKeys.begin(), m_probeCacheKeys.end(), key);
			if (it != m_probeCacheKeys.end() && *it == key)
			{
				m_probeCacheHits++;
				return m_probeCacheCPU[(size_t)(it - m_probeCacheKeys.begin())];
			}
			m_probeCacheMisses++;
		}
		if (!m_gpuDrained) DrainGPU();
		if (m_hostDirty) SyncFieldsToHost();
		return Engine::GetCurr(n,x,y,z);
	}
	inline virtual FDTD_FLOAT GetCurr(unsigned int n, const unsigned int pos[3]) const
	{ return GetCurr(n, pos[0], pos[1], pos[2]); }

	//! Copy GPU field data → CPU shadow.  Logically const (mutable internals).
	void SyncFieldsToHost() const;

	// --- Field dumps ------------------------------------------------------
	//! The raw cells a field dump reads: all three components of one field
	//! (volt or curr) on the tensor product of three sorted lists of grid lines.
	struct DumpFootprint
	{
		bool curr = false;
		std::vector<uint32_t> lines[3];
		uint64_t Count() const { return 3ull * lines[0].size() * lines[1].size() * lines[2].size(); }
	};
	//! Register a time-domain dump.  Every batch that ends on a timestep ts
	//! with due(ts) gathers the footprint into a host-visible ring slot, so the
	//! dump needs neither a drain nor the whole fields.  Returns the dump's id,
	//! or -1 if the engine cannot serve it (the dump then reads whole fields).
	//! Registration closes when the first batch is recorded.
	int RegisterDumpGather(const DumpFootprint& fp, std::function<bool(unsigned int)> due);
	//! Register a frequency-domain dump: at every timestep ts with due(ts) the
	//! GPU adds raw(ts) * w_k(ts) to a running sum for each of numFreq
	//! frequencies, weights(ts, w) supplying the w_k.  Returns -1 if the sums
	//! do not fit, or the engine cannot serve the dump.
	int RegisterDumpDFT(const DumpFootprint& fp, unsigned int numFreq,
	                    std::function<bool(unsigned int)> due,
	                    std::function<void(unsigned int, std::complex<float>*)> weights);
	//! The footprint of dump `id` at timestep ts, as [n][a][b][c] floats, or
	//! nullptr if the GPU no longer holds that timestep.
	const float* AcquireDumpGather(int id, unsigned int ts);
	//! Account for timestep ts in the DFT of dump `id`: the batch ending there
	//! normally added it already, otherwise it is added now.  False if the GPU
	//! no longer holds that timestep.
	bool AddDumpDFTSample(int id, unsigned int ts);
	//! The DFT sums of dump `id`, as [k][n][a][b][c] complex values, and the
	//! number of timesteps they hold.
	bool ReadDumpDFT(int id, std::vector<std::complex<float>>& sums, unsigned int& samples);
	//! True if the host copy of the fields is timestep ts.
	bool HostFieldsAt(unsigned int ts) const { return !m_hostDirty && m_liveTS == ts; }
	//! Whether any enabled field dump reads the whole fields, so that the run
	//! must not compute ahead of the processing (see HasPipelinedProcessing()).
	void SetHostFieldReaders(bool v) { m_hostFieldReaders = v; }

protected:
	Engine_Vulkan(const Operator* op);

private:
	static constexpr int CMD_RING_SIZE = 16;

	// ---- Vulkan core --------------------------------------------------
	VkInstance       m_instance;
	VkPhysicalDevice m_physDevice;
	VkDevice         m_device;
	VkQueue          m_computeQueue;
	uint32_t         m_computeQueueFamily;
	DeviceCaps       m_caps;

	// ---- GPU buffers --------------------------------------------------
	VkBuffer       m_voltBuf, m_currBuf;
	VkDeviceMemory m_voltMem, m_currMem;
	// Compressed operator coefficient buffers (much smaller than full-size)
	VkBuffer       m_opIndexBuf;    //!< Per-cell index into compressed tables [N]
	VkDeviceMemory m_opIndexMem;
	VkBuffer       m_vvCompBuf, m_viCompBuf, m_iiCompBuf, m_ivCompBuf;
	VkDeviceMemory m_vvCompMem, m_viCompMem, m_iiCompMem, m_ivCompMem;
	VkDeviceSize   m_opIndexBufSize; //!< packed size of the opIdx buffer, 4-byte aligned
	uint32_t       m_opIdxBits;      //!< 8, 16 or 32 -- width of one packed opIdx entry
	VkDeviceSize   m_coeffCompBufSize; //!< 3 * numCompressed * sizeof(float)
	VkBuffer       m_stagingBuf;
	VkDeviceMemory m_stagingMem;
	VkDeviceSize   m_fieldBufSize; // 3 * Nx * Ny * Nz * sizeof(float)
	uint32_t       m_fieldN;       //!< Nx * Ny * Nz (cached for fast index computation)
	uint32_t       m_strideYZ;     //!< Ny * Nz (cached for fast index computation)

	// ---- Excitation GPU data ------------------------------------------
	VkBuffer       m_excVoltIdxBuf, m_excVoltAmpBuf, m_excVoltDelayBuf;
	VkDeviceMemory m_excVoltIdxMem, m_excVoltAmpMem, m_excVoltDelayMem;
	VkBuffer       m_excCurrIdxBuf, m_excCurrAmpBuf, m_excCurrDelayBuf;
	VkDeviceMemory m_excCurrIdxMem, m_excCurrAmpMem, m_excCurrDelayMem;
	VkBuffer       m_excSignalVoltBuf, m_excSignalCurrBuf;
	VkDeviceMemory m_excSignalVoltMem, m_excSignalCurrMem;
	unsigned int   m_excVoltCount, m_excCurrCount;
	unsigned int   m_excSignalLen;
	int            m_excSignalPeriod;

	// ---- Descriptors --------------------------------------------------
	VkDescriptorSetLayout m_fdtdDescLayout;
	VkDescriptorSetLayout m_excDescLayout;
	VkDescriptorPool      m_descPool;
	VkDescriptorSet       m_updateVoltDescSet;
	VkDescriptorSet       m_updateCurrDescSet;
	VkDescriptorSet       m_excVoltDescSet;
	VkDescriptorSet       m_excCurrDescSet;

	// ---- Pipelines ----------------------------------------------------
	VkPipelineLayout m_fdtdPipeLayout;
	VkPipelineLayout m_excPipeLayout;
	VkPipeline       m_updateVoltPipeline;
	VkPipeline       m_updateCurrPipeline;
	VkPipeline       m_excPipeline;

	// ---- Command / sync -----------------------------------------------
	// Double-buffered command submission: IterateTS records pure FDTD
	// compute into m_cmdBufs[m_cmdIdx] and submits fire-and-forget.
	// The next call uses the other slot.
	VkCommandPool   m_cmdPool;
	VkCommandBuffer m_cmdBufs[CMD_RING_SIZE];   //!< Ring-buffered FDTD + copy command buffers
	VkCommandBuffer m_utilCmdBuf;   //!< Utility cmd buf (uploads, downloads, one-off commands)
	VkFence         m_fences[CMD_RING_SIZE];    //!< One per m_cmdBufs slot
	VkFence         m_utilFence;    //!< Fence for utility (upload/download) work
	int             m_cmdIdx;       //!< Index into m_cmdBufs/m_fences ring
	mutable bool    m_gpuInFlight[CMD_RING_SIZE]; //!< Whether m_cmdBufs[i] is submitted but not waited
	uint64_t        m_submitSerial[CMD_RING_SIZE] = {}; //!< m_submitCount at m_cmdBufs[i]'s last submit
	uint64_t        m_submitCount = 0;
	int             m_iterateDepth = 0; //!< IterateTS() nesting (it splits long batches)

	// ---- State --------------------------------------------------------
	//! The host copy differs from the device fields.
	mutable bool m_hostDirty;
	//! The timestep the device fields hold once the submitted batches are
	//! done; ahead of numTS while a speculative batch is uncommitted.
	unsigned int m_liveTS = 0;
	bool         m_hasGPUExcitation;
	bool         m_validateFields;
	bool         m_validateStages;
	bool         m_validateMagnitude;
	uint32_t     m_validateTraceIndex;
	mutable bool m_gpuDrained;       //!< true after DrainGPU(), cleared on submit
	unsigned int m_maxTSPerSubmit;   //!< Max timesteps per pure-GPU vkQueueSubmit

	// ---- Trapezoidal temporal blocking (OPENEMS_GPU_TEMPORAL_BLOCK) ----
	// Advances one x-slab through k timesteps before moving on, so that the
	// working set the last-level cache has to hold is a slab rather than the
	// whole grid; bit-identical to the flat sweep.  See
	// ConfigureTemporalBlocking() for the schedule, the guards and the
	// default (on for discrete GPUs when the grid is well beyond the cache).
	TemporalBlockingRequest m_tbRequest; //!< as asked for: option, environment or default
	unsigned int m_tbK;        //!< resolved timesteps per block
	unsigned int m_tbW;        //!< resolved tile width in x-lines
	bool         m_tbActive;   //!< blocking passed its guards for this run
	bool         m_tbAnnounced; //!< m_tbRequest has been resolved
	bool         m_tbQuiet;     //!< resolve without printing (--benchmark switching schedules)
	std::vector<int> m_tbEdges; //!< tile boundaries, [0, e1, .., NX]
	//! Excitation entries sorted by x, with a prefix table so an x-range maps
	//! to a contiguous entry range.  m_excVoltXStart[x] is the first entry with
	//! that x; it has numLines[0]+1 elements so [xStart[a], xStart[b]) is the
	//! entry range of x in [a,b).
	std::vector<uint32_t> m_excVoltXStart, m_excCurrXStart;
	//! The same prefix tables for the lumped RLC elements and the TF/SF
	//! injection points, whose lists are sorted by x at setup for this.
	std::vector<uint32_t> m_rlcXStart, m_tfsfVoltXStart, m_tfsfCurrXStart;
	double       m_chunkProgressIntervalSec; //!< Minimum interval between chunk-progress prints
	mutable bool m_chunkProgressHasLastPrint;
	mutable std::chrono::steady_clock::time_point m_chunkProgressLastPrint;

	// ---- Pipeline cache (faster shader load on restart) ---------------
	VkPipelineCache m_pipelineCache;

	// ---- Runtime configuration (static) ---------------------------------
	static bool s_enableProfiling;

	// ---- Optional profiling ---------------------------------------------
	bool m_profileEnabled = false;
	VkQueryPool m_tsQueryPool = VK_NULL_HANDLE;  //!< 2 queries per command slot
	float m_timestampPeriodNs = 0.0f;
	bool m_profilePhaseEnabled = false;
	uint32_t m_profileQueriesPerSlot = 2;
	bool m_profilePending[CMD_RING_SIZE] = {};
	unsigned int m_profileSubmittedTS[CMD_RING_SIZE] = {};
	mutable uint64_t m_profileBatchCount = 0;
	mutable uint64_t m_profileTSCount = 0;
	mutable double m_profileGpuMs = 0.0;
	mutable double m_profileWaitMs = 0.0;
	mutable double m_profileComputeMs = 0.0;
	mutable double m_profileProbeMs = 0.0;
	mutable double m_profileTailMs = 0.0;
	mutable uint64_t m_profileFirstTick = UINT64_MAX; //!< earliest batch start (GPU ticks)
	mutable uint64_t m_profileLastTick = 0;           //!< latest batch end
	mutable uint64_t m_profileBarriers = 0;           //!< compute barriers recorded in timesteps
	mutable uint64_t m_profileRecordedTS = 0;         //!< timesteps recorded while profiling

	//! OPENEMS_GPU_PROFILE_STAGES: timestamps between the stages of every
	//! flat-schedule timestep.  A timestep has PROFILE_STAGES stages; the slot's
	//! queries are one batch-start mark plus one mark per stage end.
	static constexpr int PROFILE_STAGES = 7;
	static const char* const s_profileStageNames[PROFILE_STAGES];
	bool m_profileStages = false;
	VkQueryPool m_stageQueryPool = VK_NULL_HANDLE;
	uint32_t m_stageQueriesPerSlot = 0;
	unsigned int m_stageSlotTS[CMD_RING_SIZE] = {}; //!< marked timesteps in each slot's batch
	mutable uint64_t m_stageTSCount = 0;
	mutable double m_stageMs[PROFILE_STAGES] = {};

	//! OPENEMS_GPU_PROFILE_NO_BARRIERS: skip the compute barriers inside the
	//! timestep loop.  Timing only -- the fields are wrong.
	bool m_profileNoBarriers = false;

	// ---- Internal helpers ---------------------------------------------
	void InitVulkan();
	void CreateBuffers();
	void CreateDescriptors();
	void CreatePipelines();
	void UploadCoefficients();
	void SetupGPUExcitation();
	void SetupGPU_SteadyState();
	void RecordSteadyStateSample(VkCommandBuffer cmd, uint32_t ts) const;
	void UpdateSteadyStateResult();
	void SetupGPU_LocalABC();
	void RecordLocalABCPhase(VkCommandBuffer cmd, uint32_t phase) const;

	//! Build the x-prefix table that maps an x-range to a contiguous run of a
	//! sparse extension's entries.  `sortedX` is the x index of each entry in
	//! the already-x-sorted list; `xStart` comes back with numLines[0]+1
	//! entries so [xStart[a], xStart[b]) is the entry range of x in [a,b).
	void BuildSlabEntryIndex(const std::vector<uint32_t>& sortedX,
	                         std::vector<uint32_t>& xStart) const;

	//! Decide tile edges and depth for trapezoidal temporal blocking, or leave
	//! m_tbActive false with a printed reason.  Called when the first batch
	//! after Init() or SetTemporalBlocking() is recorded.
	void ConfigureTemporalBlocking();
	//! PlanTemporalBlocking(), with the tile edges of an active plan.
	TemporalBlockingPlan PlanTemporalBlocking(const TemporalBlockingRequest& req,
	                                          std::vector<int>* edgesOut) const;
	//! Bytes one x-plane streams per sweep, the compressed operator's shared
	//! tables, and the whole grid's state.
	void TileGeometry(uint64_t& planeBytes, uint64_t& tableBytes, uint64_t& totalBytes) const;

	void CleanupVulkan();
	void SetupProfiling();
	void CollectProfileForSlot(int slot) const;
	void PrintProfileSummary() const;

	//! The compute -> compute dependency between two stages of a timestep;
	//! counted, or skipped, by the profiling switches above.
	void CmdComputeBarrier(VkCommandBuffer cmd, const VkMemoryBarrier& barrier) const;

	//! Upload CPU data to a device-local buffer via the staging buffer.
	void UploadToDeviceBuffer(VkBuffer dst, const void* data, VkDeviceSize size);
	//! Download device-local buffer to CPU via the staging buffer.
	void DownloadFromDeviceBuffer(VkBuffer src, void* data, VkDeviceSize size) const;

	//! Create a VkBuffer + allocate and bind VkDeviceMemory.
	void CreateBufferWithMemory(VkDeviceSize size, VkBufferUsageFlags usage,
	                            VkMemoryPropertyFlags memProps,
	                            VkBuffer& buf, VkDeviceMemory& mem);
	//! Like CreateBufferWithMemory, but returns false instead of throwing.
	bool TryCreateBufferWithMemory(VkDeviceSize size, VkBufferUsageFlags usage,
	                               VkMemoryPropertyFlags memProps,
	                               VkBuffer& buf, VkDeviceMemory& mem);

	VkShaderModule CreateShaderModule(const unsigned char* code, size_t codeSize);
	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
	//! Like FindMemoryType but returns UINT32_MAX on failure instead of throwing.
	uint32_t FindMemoryTypeSoft(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;

	//! Record and submit a one-shot command buffer, wait for completion.
	void RunSingleCommand(std::function<void(VkCommandBuffer)> func) const;

	//! Push constant structs matching the shader layouts.
	struct GridPC   { uint32_t Nx, Ny, Nz, numComp, gidBase, gidEnd; };
	struct FusedPC  { uint32_t Nx, Ny, Nz, numComp, numPmlRegions;
	                  uint32_t gidBase = 0; uint32_t gidEnd = 0xFFFFFFFFu; };
	//! Excitation push constants.  The last two fields default so that the
	//! flat path's brace-init `{count, ts, len, period}` keeps its old meaning
	//! (whole list); only the blocked path narrows them.
	struct ExcPC    { uint32_t count; int32_t numTS; uint32_t sigLen; int32_t period;
	                  uint32_t entryBase = 0; uint32_t entryEnd = 0xFFFFFFFFu; };
	//! Record one trapezoid of the blocked schedule: the x-interval [A0,B0)
	//! advanced through k timesteps starting at t0, narrowing (dir=+1, a core
	//! tile) or widening (dir=-1, a wedge over a tile boundary).  This is a
	//! transcription of Engine_AVX2_Multithread::TrapezoidSweep, whose schedule
	//! is verified bit-identical to the flat sweep; keep the two in step.
	void RecordSlabSweep(VkCommandBuffer cmd, int A0, int B0, int k, int dir,
	                     uint32_t t0, const GridPC& gridV, const GridPC& gridC,
	                     const FusedPC& fused, const VkMemoryBarrier& barrier,
	                     bool hasExcVolt, bool hasExcCurr) const;

	//! Record the UPML pre/post dispatches of one phase over the global x-range
	//! [x0,x1), clipping the range against each region's own x-extent.
	//! `phase` is 0=pre-volt, 1=post-volt, 2=pre-curr, 3=post-curr.
	void RecordUPMLSlabPhase(VkCommandBuffer cmd, int phase, int x0, int x1) const;

	//! Record the Mur dispatches of one phase over the E range [x0,x1) at
	//! timestep ts.  `phase` is 0=pre-volt, 1=post-volt, 2=apply.  A face normal
	//! to x runs whole or not at all: only the slab that holds both its boundary
	//! line and the line it reads one cell inward.  Faces normal to y or z are
	//! clipped to [x0,x1).  Returns true if anything was dispatched.
	bool RecordMurSlabPhase(VkCommandBuffer cmd, int phase, int x0, int x1, uint32_t ts) const;

	//! Record the lumped RLC dispatch of one phase (0=pre-volt, 1=apply) for
	//! the elements with x in [x0,x1).  Returns true if anything was dispatched.
	bool RecordRLCSlabPhase(VkCommandBuffer cmd, int phase, int x0, int x1) const;

	//! Record one local absorbing sheet phase (see RecordLocalABCPhase()) for
	//! the x-range [x0,x1): the E range for phases 0-2, the H range for 3-5.
	//! A sheet normal to x runs whole, in the one slab that holds its whole
	//! footprint (ConfigureTemporalBlocking() places the tile edges so that
	//! one always does); the others are clipped.  Returns true if anything
	//! was dispatched.  Records no barrier.
	bool RecordLocalABCSlabPhase(VkCommandBuffer cmd, uint32_t phase, int x0, int x1) const;

	//! Record the TF/SF injection (volt or curr) for the points with x in
	//! [x0,x1) at timestep ts.  Returns true if anything was dispatched.
	bool RecordTFSFSlabPhase(VkCommandBuffer cmd, bool volt, int x0, int x1, uint32_t ts) const;

	//! Record the dispersive dispatches of one phase over [x0,x1).
	//! `phase` is 0=pre-volt, 1=apply-volt, 2=pre-curr, 3=apply-curr.
	//! Returns true if anything was dispatched, so the caller can skip a
	//! barrier it does not need.
	bool RecordDispersiveSlabPhase(VkCommandBuffer cmd, int phase, int x0, int x1) const;
	//! The Debye entries of m_gpuDisp: the pre pass (\a apply false) or the post+apply
	//! pass, all entries (x0 < 0) or those of the slab [x0,x1); returns whether it dispatched.
	//! With \a barrierEach, a barrier follows every dispatch, as in the flat apply loops.
	bool RecordDebyeVoltPhase(VkCommandBuffer cmd, bool apply, int x0, int x1, bool barrierEach) const;

	//! Push constant struct for UPML shaders.  The trailing slab range
	//! defaults to "everything", so the flat path is unaffected.
	struct PmlPC    { uint32_t Nx, Ny, Nz, pNx, pNy, pNz, pStartX, pStartY, pStartZ;
	                  uint32_t gidBase = 0; uint32_t gidEnd = 0xFFFFFFFFu; };
	//! Push constant struct for dispersive material shaders.
	struct DispPC   { uint32_t count, Nx, Ny, Nz, hasLorADE;
	                  uint32_t entryBase = 0; uint32_t entryEnd = 0xFFFFFFFFu; };
	//! Push constant struct for TF/SF shaders (same as ExcPC).  The points are
	//! sorted by x, so a slab is the entry range [entryBase, entryEnd).
	struct TfsfPC   { uint32_t count; int32_t numTS; uint32_t sigLen; int32_t sigPeriod;
	                  uint32_t entryBase = 0; uint32_t entryEnd = 0xFFFFFFFFu; };
	//! Push constant struct for Mur ABC shaders.  [xLo, xHi) clips a face
	//! normal to y or z to an x-slab; the default covers the whole face.
	//! edgeSkip marks the edges of the face that an earlier face also writes
	//! (see SetupGPU_Mur); the apply shader leaves those cells to that face.
	//! faceOff is the face's first element in the concatenated Mur buffers.
	struct MurPC    { uint32_t Nx, Ny, Nz, ny, lineNr, lineNrShift, numLinesP, numLinesPP;
	                  uint32_t xLo = 0; uint32_t xHi = 0xFFFFFFFFu; uint32_t edgeSkip = 0;
	                  uint32_t faceOff = 0; };
	//! Push constant struct for LumpedRLC shaders.  The elements are sorted by
	//! x, so a slab is the entry range [entryBase, entryEnd).
	struct RlcPC    { uint32_t count; uint32_t entryBase = 0; uint32_t entryEnd = 0xFFFFFFFFu; };
	//! Push constant struct for probe gather shader.
	struct ProbePC  { uint32_t count; };
	struct SteadyPC { uint32_t count, ringSize, timestep; };
	struct LocalAbcPC {
		uint32_t Nx, Ny, Nz, ny, nyP, nyPP;
		uint32_t startX, startY, startZ;
		uint32_t voltBoundary, voltShift, currBoundary, currShift;
		uint32_t sizeP, sizePP, countVolt, countCurr, phase;
		uint32_t xLo = 0, xHi = 0xFFFFFFFFu;  //!< x-slab clip for sheets normal to y or z
	};
	//! The x-lines a sheet normal to x reads or writes in any phase.
	static void LocalABCFootprintX(const LocalAbcPC& pc, bool superAbsorption, int& lo, int& hi);

	//! Helper pairing a VkBuffer with its VkDeviceMemory.
	struct GpuBuf {
		VkBuffer buffer       = VK_NULL_HANDLE;
		VkDeviceMemory memory = VK_NULL_HANDLE;
	};
	void CreateGpuBuf(GpuBuf& gb, VkDeviceSize size, VkBufferUsageFlags usage,
	                  VkMemoryPropertyFlags memProps);
	void DestroyGpuBuf(GpuBuf& gb);
	void UploadGpuBuf(GpuBuf& gb, const void* data, VkDeviceSize size);

	// ---- Per-instance GPU data for each extension type ----------------

	//! Per-UPML-region GPU state.
	struct GpuUPMLData {
		GpuBuf voltFlux, currFlux;                       // auxiliary flux fields
		GpuBuf pmlIdx;                                   // narrow index into the shared coeff tables
		VkDeviceSize pmlIdxBufSize = 0;
		//! Table index per coefficient entry, [3][pNx][pNy][pNz], kept host-side
		//! so SetupFusedUPML can concatenate the regions without a readback.
		std::vector<uint32_t> coefIdx;
		VkDescriptorSet preVoltDesc  = VK_NULL_HANDLE;
		VkDescriptorSet postVoltDesc = VK_NULL_HANDLE;
		VkDescriptorSet preCurrDesc  = VK_NULL_HANDLE;
		VkDescriptorSet postCurrDesc = VK_NULL_HANDLE;
		PmlPC pc{};
		uint32_t totalCells = 0;                          // pNx*pNy*pNz for dispatch
	};

	//! Per-order dispersive material GPU state.
	struct GpuDispData {
		GpuBuf voltADE, voltLorADE;                       // voltage ADE state
		GpuBuf currADE, currLorADE;                       // current ADE state
		GpuBuf posIdx;                                    // [3][count] flat position indices
		GpuBuf vIntADE, vExtADE, vLorADE;                 // voltage coefficients
		GpuBuf iIntADE, iExtADE, iLorADE;                 // current coefficients
		VkDescriptorSet preVoltDesc   = VK_NULL_HANDLE;
		VkDescriptorSet preCurrDesc   = VK_NULL_HANDLE;
		VkDescriptorSet applyVoltDesc = VK_NULL_HANDLE;
		VkDescriptorSet applyCurrDesc = VK_NULL_HANDLE;
		DispPC voltPC{}, currPC{};
		//! Entries sorted by x, with the prefix table that maps an x-range to a
		//! contiguous run of them.
		std::vector<uint32_t> xStart;
		uint32_t count = 0;
		bool voltLorADEOn = false, currLorADEOn = false;
		bool voltADEOn = false, currADEOn = false;
		//! A Debye material (Operator_Ext_DebyeMaterial): voltage side only, its own
		//! pre pass and a post+apply pass instead of the Lorentz shaders, both on
		//! debyeDesc; voltPC carries the pole count in its hasLorADE slot.
		bool debye = false;
		uint32_t poles = 0;
		GpuBuf debyeW, debyePre;                          // pole states [poles][3][count], pre sum [3][count]
		GpuBuf debyeRelax, debyeDrive, debyeSolve;        // [poles][3][count], [poles][3][count], [3][count]
		VkDescriptorSet debyeDesc = VK_NULL_HANDLE;
	};

	//! TF/SF GPU state (all injection points flattened).
	struct GpuTFSFData {
		GpuBuf voltAmp, voltDelta, voltIdx, voltDelay, voltSignal;
		GpuBuf currAmp, currDelta, currIdx, currDelay, currSignal;
		VkDescriptorSet voltDesc = VK_NULL_HANDLE;
		VkDescriptorSet currDesc = VK_NULL_HANDLE;
		uint32_t voltCount = 0, currCount = 0;
	};

	//! Per-face Mur ABC GPU state.  The buffers are shared by all faces
	//! (m_murAll*), each face at pc.faceOff.
	struct GpuMurData {
		VkDescriptorSet preDesc   = VK_NULL_HANDLE;
		VkDescriptorSet postDesc  = VK_NULL_HANDLE;
		VkDescriptorSet applyDesc = VK_NULL_HANDLE;
		MurPC pc{};
		uint32_t totalCells = 0;
		unsigned int startTS = 0;                         // timestep after which Mur activates
	};

	//! Lumped RLC GPU state.
	struct GpuRLCData {
		GpuBuf il, vdn0, vdn1, vdn2, jn0, jn1, jn2;     // ADE state
		GpuBuf linIdx;                                    // pre-linearized positions
		GpuBuf i2v, ilv, vvd, vv2, vj1, vj2, ib0, b1, b2; // coefficients
		VkDescriptorSet preDesc   = VK_NULL_HANDLE;
		VkDescriptorSet applyDesc = VK_NULL_HANDLE;
		uint32_t count = 0;
	};

	// ---- Fused Yee+UPML pipeline (concatenated PML buffers) ----------
	bool m_hasFusedUPML = false;
	GpuBuf m_fusedVoltFlux;     //!< Concatenated volt flux (all PML regions)
	GpuBuf m_fusedCurrFlux;     //!< Concatenated curr flux (all PML regions)
	GpuBuf m_fusedPmlIdx;       //!< Concatenated per-entry PML coefficient index
	VkDeviceSize m_fusedPmlIdxBufSize = 0;
	GpuBuf m_fusedPmlRegionInfo; //!< PML region metadata SSBO
	uint32_t m_fusedTotalPmlCells = 0;  //!< Sum of all PML regions' cell counts
	VkDescriptorSetLayout m_fusedDescLayout = VK_NULL_HANDLE;
	VkPipelineLayout      m_fusedPipeLayout = VK_NULL_HANDLE;
	VkPipeline            m_fusedVoltPipeline = VK_NULL_HANDLE;
	VkPipeline            m_fusedCurrPipeline = VK_NULL_HANDLE;
	VkDescriptorPool      m_fusedDescPool = VK_NULL_HANDLE;
	VkDescriptorSet       m_fusedVoltDescSet = VK_NULL_HANDLE;
	VkDescriptorSet       m_fusedCurrDescSet = VK_NULL_HANDLE;
	void SetupFusedUPML();       //!< Create concatenated PML buffers + fused pipeline
	void CleanupFusedUPML();     //!< Destroy fused UPML resources

	// ---- Deduplicated UPML coefficient tables -------------------------
	// Shared by every region and by both the fused and the separate-dispatch
	// path, so an index fetched anywhere means the same table entry.
	GpuBuf m_pmlTabVv, m_pmlTabVvfo, m_pmlTabVvfn;
	GpuBuf m_pmlTabIi, m_pmlTabIifo, m_pmlTabIifn;
	VkDeviceSize m_pmlTabBufSize = 0;
	uint32_t m_pmlTabCount = 0;      //!< number of distinct coefficient tuples
	uint32_t m_pmlIdxBits  = 32;     //!< width of one packed table index
	void BuildUPMLCoeffTables();     //!< Deduplicate the UPML coefficients

	// ---- Extension GPU instances --------------------------------------
	VkDescriptorPool      m_extDescPool = VK_NULL_HANDLE;
	std::vector<GpuUPMLData>  m_gpuUPML;
	std::vector<GpuDispData>  m_gpuDisp;
	GpuTFSFData               m_gpuTFSF;
	std::vector<GpuMurData>   m_gpuMur;
	//! All Mur faces' state, coefficients and saved V(shift), concatenated.
	GpuBuf m_murAllNyP, m_murAllNyPP, m_murAllCoeffNyP, m_murAllCoeffNyPP, m_murAllVoldNyP, m_murAllVoldNyPP;
	VkDeviceSize m_murAllSize = 0;
	//! Mur post and apply folded into the Yee voltage update
	//! (update_voltages_mur.comp); see SetupMurFusion().
	bool m_murFused = false;
	GpuBuf m_murFaceBuf;
	VkDescriptorSetLayout m_murFusedDescLayout = VK_NULL_HANDLE;
	VkDescriptorPool      m_murFusedDescPool   = VK_NULL_HANDLE;
	VkDescriptorSet       m_murFusedDescSet    = VK_NULL_HANDLE;
	VkPipelineLayout      m_murFusedPipeLayout = VK_NULL_HANDLE;
	VkPipeline            m_murFusedPipeline   = VK_NULL_HANDLE;
	void SetupMurFusion();
	GpuRLCData                m_gpuRLC;

	// ---- GPU steady-state observer -----------------------------------
	Engine_Ext_SteadyState* m_gpuSteadyStateExt = nullptr; // owned by Engine::m_Eng_exts
	GpuBuf m_steadyProbeIdx, m_steadyHistory;
	float* m_steadyHistoryMapped = nullptr;
	uint32_t m_steadyProbeCount = 0;
	uint32_t m_steadyPeriod = 0;
	uint32_t m_steadyRingSize = 0;
	uint32_t m_steadyLastCompletedPeriods = 0;
	VkDescriptorSetLayout m_steadyDescLayout = VK_NULL_HANDLE;
	VkPipelineLayout m_steadyPipeLayout = VK_NULL_HANDLE;
	VkPipeline m_steadyPipeline = VK_NULL_HANDLE;
	VkDescriptorPool m_steadyDescPool = VK_NULL_HANDLE;
	VkDescriptorSet m_steadyDescSet = VK_NULL_HANDLE;

	struct GpuLocalABCData {
		GpuBuf voltState, currState, k1, k2;
		VkDescriptorSet voltDesc = VK_NULL_HANDLE;
		VkDescriptorSet currDesc = VK_NULL_HANDLE;
		LocalAbcPC pc{};
		bool superAbsorption = false;
	};
	std::vector<GpuLocalABCData> m_gpuLocalABC;
	VkDescriptorSetLayout m_localAbcDescLayout = VK_NULL_HANDLE;
	VkPipelineLayout m_localAbcPipeLayout = VK_NULL_HANDLE;
	VkPipeline m_localAbcPipeline = VK_NULL_HANDLE;
	VkDescriptorPool m_localAbcDescPool = VK_NULL_HANDLE;

	// ---- Extension descriptor set layouts -----------------------------
	VkDescriptorSetLayout m_upmlPreVoltDescLayout  = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_upmlPostVoltDescLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_upmlPreCurrDescLayout  = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_upmlPostCurrDescLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_dispPreDescLayout      = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_debyeDescLayout        = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_dispApplyDescLayout    = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_tfsfDescLayout         = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_murPreDescLayout       = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_murPostDescLayout      = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_murApplyDescLayout     = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_rlcPreDescLayout       = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_rlcApplyDescLayout     = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_probeDescLayout        = VK_NULL_HANDLE;

	// ---- Extension pipeline layouts -----------------------------------
	VkPipelineLayout m_upmlPrePipeLayout = VK_NULL_HANDLE;
	VkPipelineLayout m_upmlPostPipeLayout= VK_NULL_HANDLE;
	VkPipelineLayout m_dispPrePipeLayout = VK_NULL_HANDLE;
	VkPipelineLayout m_debyePipeLayout   = VK_NULL_HANDLE;
	VkPipelineLayout m_dispApplyPipeLayout = VK_NULL_HANDLE;
	VkPipelineLayout m_tfsfPipeLayout    = VK_NULL_HANDLE;
	VkPipelineLayout m_murUpdatePipeLayout = VK_NULL_HANDLE;
	VkPipelineLayout m_murApplyPipeLayout  = VK_NULL_HANDLE;
	VkPipelineLayout m_rlcPrePipeLayout  = VK_NULL_HANDLE;
	VkPipelineLayout m_rlcApplyPipeLayout= VK_NULL_HANDLE;
	VkPipelineLayout m_probePipeLayout   = VK_NULL_HANDLE;

	// ---- Extension compute pipelines ----------------------------------
	VkPipeline m_upmlPreVoltPipeline   = VK_NULL_HANDLE;
	VkPipeline m_upmlPostVoltPipeline  = VK_NULL_HANDLE;
	VkPipeline m_upmlPreCurrPipeline   = VK_NULL_HANDLE;
	VkPipeline m_upmlPostCurrPipeline  = VK_NULL_HANDLE;
	VkPipeline m_dispPreVoltPipeline   = VK_NULL_HANDLE;
	VkPipeline m_debyePreVoltPipeline   = VK_NULL_HANDLE;
	VkPipeline m_debyeApplyVoltPipeline = VK_NULL_HANDLE;
	VkPipeline m_dispPreCurrPipeline   = VK_NULL_HANDLE;
	VkPipeline m_dispApplyVoltPipeline = VK_NULL_HANDLE;
	VkPipeline m_dispApplyCurrPipeline = VK_NULL_HANDLE;
	VkPipeline m_tfsfVoltPipeline      = VK_NULL_HANDLE;
	VkPipeline m_tfsfCurrPipeline      = VK_NULL_HANDLE;
	VkPipeline m_murPreVoltPipeline    = VK_NULL_HANDLE;
	VkPipeline m_murPostVoltPipeline   = VK_NULL_HANDLE;
	VkPipeline m_murApplyVoltPipeline  = VK_NULL_HANDLE;
	VkPipeline m_rlcPreVoltPipeline    = VK_NULL_HANDLE;
	VkPipeline m_rlcApplyVoltPipeline  = VK_NULL_HANDLE;
	VkPipeline m_probePipeline         = VK_NULL_HANDLE;

	// ---- GPU extension state flags ------------------------------------
	bool m_hasGPU_UPML       = false;
	bool m_hasGPU_Dispersive = false;
	bool m_hasGPU_TFSF       = false;
	bool m_hasGPU_Mur        = false;
	bool m_hasGPU_RLC        = false;
	bool m_hasGPU_Probes     = false;

	// ---- Pipelined processing state -----------------------------------
	bool m_pipelinedReading = false;   //!< When true, GetVolt/GetCurr use probe cache
	mutable bool m_recordingProbeAccess = false; //!< When true, GetVolt/GetCurr record cell indices
	mutable std::vector<uint64_t> m_recordedCells; //!< Recorded (fieldSel<<48 | linearIdx) during dry run
	std::vector<float> m_probeCacheCPU;  //!< CPU-side snapshot of gathered probes
	std::vector<uint64_t> m_probeCacheKeys; //!< Sorted keys for binary search (fieldSel<<48 | linearIdx)
	uint32_t m_probeCacheCount = 0;     //!< Number of probe points
	mutable uint64_t m_probeCacheHits = 0;
	mutable uint64_t m_probeCacheMisses = 0;
	GpuBuf m_probeIdxBuf;              //!< GPU probe index buffer
	GpuBuf m_probeSelBuf;              //!< GPU field selector (0=volt, 1=curr)
	GpuBuf m_probeOutBuf;              //!< Coherent host-visible GPU probe output
	float* m_probeOutMapped = nullptr;  //!< Persistently mapped probe output
	VkDescriptorSet  m_probeGatherDescSet = VK_NULL_HANDLE;
	VkDescriptorPool m_probeDescPool     = VK_NULL_HANDLE;
	unsigned int m_speculativeTS = 0;   //!< Timesteps in speculative batch

	// ---- GPU energy reduction -----------------------------------------
	static constexpr uint32_t ENERGY_NUM_WG = 256; //!< workgroups for energy reduction
	GpuBuf       m_energyPartialBuf;   //!< [2 * ENERGY_NUM_WG] float partials (E, H)
	float*       m_energyMapped = nullptr; //!< m_energyPartialBuf, persistently mapped
	VkDescriptorSetLayout m_energyDescLayout = VK_NULL_HANDLE;
	VkPipelineLayout      m_energyPipeLayout = VK_NULL_HANDLE;
	VkPipeline            m_energyPipeline   = VK_NULL_HANDLE;
	VkDescriptorPool      m_energyDescPool   = VK_NULL_HANDLE;
	VkDescriptorSet       m_energyDescSet    = VK_NULL_HANDLE;
	//! Push constant struct for energy reduction shader.
	struct EnergyPC { uint32_t N; uint32_t numWG; };
	void SetupGPU_EnergyReduction();     //!< Create energy reduction resources
	void CleanupGPU_EnergyReduction();   //!< Destroy energy reduction resources

	// ---- Optional raw-field finite-value scan -------------------------
	GpuBuf       m_validateBadBuf;       //!< [V0,V1,V2,C0,C1,C2] first bad cell indices
	uint32_t*    m_validateBadMapped = nullptr;
	VkDescriptorSetLayout m_validateDescLayout = VK_NULL_HANDLE;
	VkPipelineLayout      m_validatePipeLayout = VK_NULL_HANDLE;
	VkPipeline            m_validatePipeline = VK_NULL_HANDLE;
	VkDescriptorPool      m_validateDescPool = VK_NULL_HANDLE;
	VkDescriptorSet       m_validateDescSet = VK_NULL_HANDLE;
	struct ValidatePC { uint32_t N, outputBase, traceIndex, numComp, Ny, Nz, checkMagnitude; };
	void SetupGPU_FieldValidation();
	void CleanupGPU_FieldValidation();
	void RecordGPUFieldValidation(VkCommandBuffer cmd, uint32_t outputBase) const;
	void ValidateGPUFields(unsigned int timestep) const; //!< Debug-only finite-value scan
	void ReportGPUFieldValidation(unsigned int timestep) const; //!< Report staged scans

	// ---- Extension setup methods --------------------------------------
	void SetupGPUExtensions();     //!< Detect and setup all GPU-native extensions
	void SetupGPU_UPML();          //!< Upload PML data and create descriptors/pipelines
	void SetupGPU_Dispersive();    //!< Upload dispersive ADE data
	void SetupGPU_Debye(const Operator_Ext_DebyeMaterial* deb);  //!< One Debye material, called by SetupGPU_Dispersive()
	void SetupGPU_TFSF();          //!< Flatten and upload TF/SF injection points
	void SetupGPU_Mur();           //!< Upload Mur ABC data
	void SetupGPU_RLC();           //!< Upload lumped RLC data
	void CreateExtensionDescriptorLayouts();   //!< Create all extension descriptor set layouts
	void CreateExtensionPipelines();           //!< Create all extension compute pipelines
	void CleanupExtensions();      //!< Destroy all extension GPU resources

	//! Record extension dispatches into a command buffer for the voltage half-step.
	void RecordVoltageExtensions(VkCommandBuffer cmd, uint32_t numTS) const;
	//! Record extension dispatches into a command buffer for the current half-step.
	void RecordCurrentExtensions(VkCommandBuffer cmd, uint32_t numTS) const;

	static void WriteDescriptorBuffers(VkDevice dev, VkDescriptorSet set,
	                                   const VkBuffer* bufs, const VkDeviceSize* sizes,
	                                   uint32_t count);

	// ---- Field dumps ------------------------------------------------------
	// A dump registers the raw cells it reads (its footprint).  Every batch
	// that ends on a timestep where a dump is due gathers those cells at its
	// end, in the same command buffer, into a host-visible ring slot, or adds
	// them to the dump's DFT sums.  Nothing waits for that: the dump reads the
	// slot after the batch's fence, while the next batch runs.  A full-field
	// dump moves one field instead of volt and curr, a plane dump a few planes.
	struct DumpReg
	{
		bool curr = false;
		bool dft = false;
		uint32_t n[3] = {0, 0, 0};
		uint32_t lineOff = 0;     //!< first of its n[0]+n[1]+n[2] lines in m_dumpLinesBuf
		uint32_t count = 0;       //!< 3*n[0]*n[1]*n[2]
		std::vector<uint32_t> lines;
		std::function<bool(unsigned int)> due;
		uint64_t slotOff = 0;     //!< gather: floats into a ring slot
		// DFT
		uint32_t numFreq = 0;
		uint32_t weightOff = 0;   //!< complex weights into one weight region
		std::function<void(unsigned int, std::complex<float>*)> weights;
		GpuBuf sums;              //!< numFreq*count complex floats, device-local
		VkDescriptorSet set = VK_NULL_HANDLE;
		std::deque<unsigned int> recorded; //!< timesteps added on the GPU, not yet accounted for
		unsigned int samples = 0; //!< timesteps accounted for
	};
	struct DumpSlot
	{
		GpuBuf buf;
		float* mapped = nullptr;
		VkDescriptorSet set = VK_NULL_HANDLE;
		unsigned int ts = UINT_MAX; //!< timestep held, UINT_MAX = none
		int cmdSlot = -1;           //!< m_cmdBufs slot whose batch writes it, -1 = written
		uint64_t serial = 0;        //!< that batch's m_submitSerial
		std::vector<char> has;      //!< per dump: gathered into this slot
	};
	static constexpr size_t MAX_DUMP_SLOTS = 8;
	std::vector<DumpReg>  m_dumps;
	std::vector<DumpSlot> m_dumpSlots;
	bool         m_dumpReady = false;    //!< buffers, pipelines and sets exist; registration closed
	bool         m_hostFieldReaders = false;
	unsigned int m_dumpKeepFrom = 0;     //!< ring slots from this timestep on may still be read
	uint64_t     m_dumpSlotFloats = 0;   //!< all gathered footprints, one ring slot
	uint32_t     m_dumpFreqs = 0;        //!< all DFT weights, one weight region
	VkDeviceSize m_dumpSumBytes = 0;     //!< all DFT sums
	GpuBuf       m_dumpLinesBuf;
	GpuBuf       m_dumpWeightBuf;        //!< CMD_RING_SIZE+1 weight regions, host-visible
	std::complex<float>* m_dumpWeightMapped = nullptr;
	VkDescriptorSetLayout m_dumpDescLayout = VK_NULL_HANDLE;
	VkPipelineLayout      m_dumpPipeLayout = VK_NULL_HANDLE;
	VkPipeline            m_dumpGatherPipeline = VK_NULL_HANDLE;
	VkPipeline            m_dumpDFTPipeline = VK_NULL_HANDLE;
	VkDescriptorPool      m_dumpDescPool = VK_NULL_HANDLE;
	mutable bool m_warnedStaleRead = false;
	bool         m_warnedDumpSlots = false;
	struct DumpPC { uint32_t n0, n1, n2, lineOff, curr, NyNz, Nz, N, outOff, numFreq, weightOff, count; };

	bool CheckDumpFootprint(const DumpFootprint& fp, const char* what) const;
	int  AddDump(const DumpFootprint& fp, std::function<bool(unsigned int)> due);
	//! Create the lines buffer, the pipelines and the DFT sums' sets, once.
	void EnsureDumpResources();
	VkDescriptorSet AllocDumpSet(VkBuffer out, VkDeviceSize outBytes);
	//! The ring slot for timestep ts: the one holding it already, else an
	//! empty one or one older than keepFrom.  Null if no slot can be made.
	DumpSlot* TakeDumpSlot(unsigned int ts, unsigned int keepFrom);
	void WaitDumpSlot(DumpSlot& slot) const;
	DumpPC DumpPushConstants(const DumpReg& d) const;
	//! Gather and DFT dispatches for the dumps due at ts, at the end of a batch.
	void RecordDumpSamples(VkCommandBuffer cmd, unsigned int ts);
	void RecordDumpDFT(VkCommandBuffer cmd, DumpReg& d, uint32_t region, unsigned int ts);
	void CleanupDumps();
};

#endif // WITH_GPU
#endif // ENGINE_VULKAN_H
