/*
*	Copyright (C) 2010 Thorsten Liebig (Thorsten.Liebig@gmx.de)
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ENGINE_H
#define ENGINE_H

#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <vector>
#include "operator.h"
#include "temporal_blocking.h"
#include "tools/cpu_topology.h"

#include "tools/arraylib/array_nijk.h"

//! The test of GetVolt()/GetCurr() for in-batch sampling, off the hot path (dumps and extensions read through them too).
#if defined(__GNUC__)
#define OPENEMS_BS_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define OPENEMS_BS_UNLIKELY(x) (x)
#endif

namespace NS_Engine_Multithread
{
class thread; // evil hack to access numTS from multithreading context
}

class Engine_Extension;

class Engine
{
public:
	enum EngineType
	{
		BASIC, SSE, AVX2,
#ifdef WITH_GPU
		GPU,
#endif
		UNKNOWN
	};

	static Engine* New(const Operator* op);
	virtual ~Engine();

	virtual void Init();
	virtual void Reset();

	//!Iterate a number of timesteps
	virtual bool IterateTS(unsigned int iterTS);

	//! The timesteps done; in a replay (BeginBatchReplay()) the replayed one.
	virtual unsigned int GetNumberOfTimesteps() {return (m_bsMode==BS_REPLAY) ? m_bsReplayTS : numTS;}

	virtual void NextInterval(float curr_speed) {UNUSED(curr_speed);};

	//! \name Trapezoidal temporal blocking
	//! Engine_Vulkan and Engine_AVX2_Multithread have it; the others run the
	//! flat sweep whatever they are asked. Every schedule computes the same
	//! fields bit for bit, so it may change between any two IterateTS() calls.
	//@{
	//! The schedule asked for.
	virtual TemporalBlockingRequest GetTemporalBlocking() const {return TemporalBlockingRequest::Off();}
	//! Ask for a schedule from the next IterateTS() on; false if this engine has none.
	//! \a announce prints what it resolves to, or why it does not engage.
	virtual bool SetTemporalBlocking(const TemporalBlockingRequest& req, bool announce = true) {UNUSED(req); UNUSED(announce); return false;}
	//! What \a req would resolve to on this grid, without applying it.
	virtual TemporalBlockingPlan PlanTemporalBlocking(const TemporalBlockingRequest& req) const {UNUSED(req); return TemporalBlockingPlan();}
	//! Blocked schedules worth measuring on this grid (--benchmark), none deeper than \a maxDepth.
	virtual std::vector<TemporalBlockingRequest> TemporalBlockingCandidates(unsigned int maxDepth) const {UNUSED(maxDepth); return {};}
	//! True while the engine still tunes itself on the timesteps it runs.
	virtual bool IsCalibrating() const {return false;}
	//! The timesteps run so far inside blocks at least two timesteps deep; -1 if the engine does not count them.
	virtual long long GetBlockedTimesteps() const {return -1;}
	//@}

	//! \name In-batch sampling (plans/BATCH_SAMPLING.md)
	/*!
	  The main loop (openEMS::RunFDTD()) used to end every IterateTS() at the
	  next timestep a probe or dump was due at, which capped a blocked
	  schedule's depth at the probe interval. With in-batch sampling it runs
	  one batch past the timesteps at which only probes and the energy end
	  criterion are due, and afterwards replays the batch: for each of those
	  timesteps t in order it calls the processings with the engine in
	  replay(t), where GetNumberOfTimesteps() is t and GetVolt()/GetCurr()
	  return what they would have returned at t. The probes run unchanged on
	  the same floats, so their files are bit-identical to the old loop's.

	  The cells the probes read, the footprint, are recorded once before the
	  run (StartFootprintRecording()): the main loop calls every probe once
	  while GetVolt()/GetCurr() record the cells they are asked for.

	  Contract for an engine that returns true from SupportsBatchSampling():
	  - Its GetVolt(), both overloads, begin with
	    \code
	    if (m_bsMode != BS_OFF) { FDTD_FLOAT v; if (BatchRead(0, n, x, y, z, v)) return v; }
	    \endcode
	    and its GetCurr() the same with field 1; its GetNumberOfTimesteps()
	    is this one.
	  - IterateSampledTS(iterTS) runs iterTS timesteps like IterateTS(), and
	    for every timestep t of the schedule (ScheduleBatchSamples()) stores
	    - in BatchProbeSample(i), i the index of t in the probe list, for
	      each cell c of GetFootprint(): the value GetVolt()/GetCurr() of
	      that cell returns after IterateTS() has brought the engine to
	      numTS == t, i.e. the value between the last write of timestep t-1
	      (Apply2Current() and anything after it) and the first write of
	      timestep t (DoPreVoltageUpdates() of t);
	    - with SetBatchEnergySample(j, e), j the index of t in the energy
	      list, the energy estimate the engine interface's CalcFastEnergy()
	      would return at numTS == t, computed the same way bit for bit
	      (CalcBatchEnergy() calls exactly that).
	    The default implementation runs IterateTS() up to each sample
	    timestep and reads the cells and the energy from the coherent grid,
	    so its blocks never cross a sample. An engine that samples inside its
	    blocks overrides IterateSampledTS() and SamplesInsideBlocks().
	  - The fields the batch leaves behind are those of IterateTS(iterTS).

	  The main loop never schedules a sample at the batch's last timestep:
	  the processings read the live fields there, as before. Nothing outside
	  the footprint may be read in a replay of an earlier timestep, and
	  BatchRead() throws std::logic_error if anything is, as it does for a
	  timestep without a sample: a replay never reads live fields silently.
	*/
	//@{
	enum BatchSamplingMode {BS_OFF = 0, BS_RECORD = 1, BS_REPLAY = 2};
	//! Whether IterateSampledTS() and the replay work on this engine. Default: false.
	virtual bool SupportsBatchSampling() const {return false;}
	//! True if IterateSampledTS() samples inside blocks, so that its depth is not capped by the gaps between samples.
	virtual bool SamplesInsideBlocks() const {return false;}
	//! The fewest timesteps a batch should hold, 0 for no preference. The main
	//! loop keeps its batch length at least this (a batch still ends at a dump,
	//! the ring's capacity, NrTS or a benchmark boundary). An engine that blocks
	//! inside a batch asks for a few blocks.
	virtual unsigned int GetPreferredBatchLength() const {return 0;}

	//! From now on GetVolt()/GetCurr() record the cells they read (and return the live values).
	void StartFootprintRecording();
	//! Stop recording; the cells recorded since StartFootprintRecording() are the footprint.
	void StopFootprintRecording();
	//! The footprint, sorted by key (FootprintKey()); a cell's position is its index in every probe sample.
	const std::vector<uint64_t>& GetFootprint() const {return m_bsFootprint;}
	//! A cell of the footprint as a key: field (0 voltages, 1 currents), component n, position; sorted by field, n, x, y, z.
	static uint64_t FootprintKey(unsigned int field, unsigned int n, unsigned int x, unsigned int y, unsigned int z)
	{
		return ((uint64_t)field<<62) | ((uint64_t)n<<60) | ((uint64_t)x<<40) | ((uint64_t)y<<20) | (uint64_t)z;
	}
	static void DecodeFootprintKey(uint64_t key, unsigned int& field, unsigned int& n, unsigned int& x, unsigned int& y, unsigned int& z)
	{
		field = (unsigned int)(key>>62);
		n = (unsigned int)((key>>60)&3);
		x = (unsigned int)((key>>40)&0xFFFFF);
		y = (unsigned int)((key>>20)&0xFFFFF);
		z = (unsigned int)(key&0xFFFFF);
	}

	//! The energy estimate of the main loop's end criterion, as its engine interface computes it.
	void SetBatchEnergyFunction(std::function<double()> fn) {m_bsEnergyFn = std::move(fn);}

	//! The samples the next IterateSampledTS() takes: absolute timesteps, ascending, each after the current timestep and before the batch's last.
	virtual void ScheduleBatchSamples(const std::vector<unsigned int>& probeTS, const std::vector<unsigned int>& energyTS);
	//! Run \a iterTS timesteps, taking the samples ScheduleBatchSamples() asked for.
	virtual bool IterateSampledTS(unsigned int iterTS);
	const std::vector<unsigned int>& GetBatchProbeTimesteps() const {return m_bsProbeTS;}
	const std::vector<unsigned int>& GetBatchEnergyTimesteps() const {return m_bsEnergyTS;}

	//! Replay timestep \a ts of the batch just run, up to and including the current timestep. At the current one, a no-op.
	void BeginBatchReplay(unsigned int ts);
	void EndBatchReplay();
	//! In a replay of a timestep before the current one.
	bool InBatchReplay() const {return m_bsMode==BS_REPLAY;}
	//! In a replay: the energy sample of the replayed timestep; throws std::logic_error if there is none.
	double GetBatchReplayEnergy() const;
	//@}

	//! \name Worker threads
	//! The multithreaded CPU engines run on worker threads, and how many and
	//! on which cores may change between any two IterateTS() calls too. The
	//! others have none: a GPU engine, the single-threaded ones.
	//@{
	//! Worker threads in use; 0 if the engine has none to set.
	virtual unsigned int GetNumThreads() const {return 0;}
	//! The cores they may run on.
	virtual CoreSet GetCoreSet() const {return CoreSet();}
	//! From the next IterateTS() on, run \a numThreads workers on \a cores; 0
	//! threads: calibrate the count there. False if this engine has no workers.
	virtual bool SetThreads(unsigned int numThreads, const CoreSet& cores) {UNUSED(numThreads); UNUSED(cores); return false;}
	//@}

	//this access functions muss be overloaded by any new engine using a different storage model
	inline virtual FDTD_FLOAT GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		ArrayLib::ArrayNIJK<FDTD_FLOAT>& volt = *volt_ptr;
		return volt(n,x,y,z);
	}

	inline virtual FDTD_FLOAT GetVolt(unsigned int n, const unsigned int pos[3]) const
	{
		ArrayLib::ArrayNIJK<FDTD_FLOAT>& volt = *volt_ptr;
		return volt(n, pos[0], pos[1], pos[2]);
	}

	inline virtual FDTD_FLOAT GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		ArrayLib::ArrayNIJK<FDTD_FLOAT>& curr = *curr_ptr;
		return curr(n,x,y,z);
	}

	inline virtual FDTD_FLOAT GetCurr(unsigned int n, const unsigned int pos[3]) const
	{
		ArrayLib::ArrayNIJK<FDTD_FLOAT>& curr = *curr_ptr;
		return curr(n, pos[0], pos[1], pos[2]);
	}

	inline virtual void SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)
	{
		ArrayLib::ArrayNIJK<FDTD_FLOAT>& volt = *volt_ptr;
		volt(n,x,y,z) = value;
	}

	inline virtual void SetVolt(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value)
	{
		ArrayLib::ArrayNIJK<FDTD_FLOAT>& volt = *volt_ptr;
		volt(n, pos[0], pos[1], pos[2]) = value;
	}

	inline virtual void SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)
	{
		ArrayLib::ArrayNIJK<FDTD_FLOAT>& curr = *curr_ptr;
		curr(n,x,y,z) = value;
	}

	inline virtual void SetCurr(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value )
	{
		ArrayLib::ArrayNIJK<FDTD_FLOAT>& curr = *curr_ptr;
		curr(n, pos[0], pos[1], pos[2]) = value;
	}

	//! Execute Pre-Voltage extension updates
	virtual void DoPreVoltageUpdates();
	//! Main FDTD engine voltage updates
	virtual void UpdateVoltages(unsigned int startX, unsigned int numX);
	//! Execute Post-Voltage extension updates
	virtual void DoPostVoltageUpdates();
	//! Apply extension voltage changes
	virtual void Apply2Voltages();

	//! Execute Pre-Current extension updates
	virtual void DoPreCurrentUpdates();
	//! Main FDTD engine current updates
	virtual void UpdateCurrents(unsigned int startX, unsigned int numX);
	//! Execute Post-Current extension updates
	virtual void DoPostCurrentUpdates();
	//! Apply extension current changes
	virtual void Apply2Current();

	//! Optional diagnostic trace for one native Yee cell, enabled by
	//! OPENEMS_CPU_TRACE_CELL=x,y,z.
	void TraceFieldCell(const char* stage) const;

	inline size_t GetExtensionCount() {return m_Eng_exts.size();}
	inline Engine_Extension* GetExtension(size_t nr) {return m_Eng_exts.at(nr);}
	virtual void SortExtensionByPriority();

	EngineType GetType() const {return m_type;}

protected:
	EngineType m_type;

	Engine(const Operator* op);
	const Operator* Op;

	unsigned int numLines[3];

	ArrayLib::ArrayNIJK<FDTD_FLOAT>* volt_ptr;
	ArrayLib::ArrayNIJK<FDTD_FLOAT>* curr_ptr;
	unsigned int numTS;

	bool m_traceFieldEnabled;
	unsigned int m_traceFieldCell[3];
	mutable std::ofstream m_traceFieldFile;

	virtual void InitExtensions();
	virtual void ClearExtensions();
	std::vector<Engine_Extension*> m_Eng_exts;

	//! \name In-batch sampling, for the engines (see the contract above)
	//@{
	//! Recording: record the cell, false. Replay: the cell's sample in \a value, true; std::logic_error if there is none.
	bool BatchRead(unsigned int field, unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT& value) const;
	//! BatchRead() of a cell the replay has no sample of, and of the recording.
	bool BatchReadSlow(unsigned int field, unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT& value) const;
	//! The position of a cell in the footprint, or -1 (built by StopFootprintRecording()).
	/*!
	  A component whose cells fill a box well (a mode-matching probe over a
	  plane) has a dense index of that box, which keeps neighbouring cells
	  together in the cache; the others are in a hash table.
	  */
	int64_t FootprintSlot(unsigned int field, unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		if ((field > 1) || (n > 2))
			return -1;
		const FootprintBox& b = m_bsBoxes[field*3+n];
		if (!b.slot.empty())
		{
			const unsigned int i = x-b.lo[0], j = y-b.lo[1], k = z-b.lo[2];
			if ((i < b.len[0]) && (j < b.len[1]) && (k < b.len[2]))
				return b.slot[((size_t)i*b.len[1] + j)*b.len[2] + k];
			return -1;
		}
		if (m_bsHashKeys.empty())
			return -1;
		const uint64_t key = FootprintKey(field, n, x, y, z);
		size_t h = (size_t)((key*0x9E3779B97F4A7C15ull) >> m_bsHashShift);
		for (;;)
		{
			const uint64_t k = m_bsHashKeys[h];
			if (k == key)
				return m_bsHashSlots[h];
			if (k == ~0ull)
				return -1;
			h = (h+1) & (m_bsHashKeys.size()-1);
		}
	}
	//! The footprint values of the \a i-th probe sample timestep, in footprint order.
	FDTD_FLOAT* BatchProbeSample(size_t i) {return m_bsValues.data() + i*m_bsFootprint.size();}
	//! Store the energy sample of the \a j-th energy sample timestep.
	void SetBatchEnergySample(size_t j, double e) {m_bsEnergy.at(j) = e;}
	//! The end criterion's energy estimate of the live fields (SetBatchEnergyFunction()).
	double CalcBatchEnergy() const;
	//! Read the footprint from the live fields into BatchProbeSample(\a i) with GetVolt()/GetCurr(); an engine may do it faster.
	virtual void SampleFootprint(size_t i);

	BatchSamplingMode m_bsMode;
	unsigned int m_bsReplayTS;    //!< BS_REPLAY: the replayed timestep
	int m_bsReplaySlot;           //!< BS_REPLAY: its probe sample, -1 = none
	int m_bsReplayEnergySlot;     //!< BS_REPLAY: its energy sample, -1 = none
	unsigned int m_bsBatchStart;  //!< numTS when the last IterateSampledTS() started
	std::vector<uint64_t> m_bsFootprint;
	std::vector<uint64_t> m_bsHashKeys;   //!< open addressing, ~0 = empty, a power of two of at least twice the footprint
	std::vector<int64_t> m_bsHashSlots;   //!< the footprint position of each key
	unsigned int m_bsHashShift;
	//! The dense index of one component's cells: footprint positions over [lo, lo+len), -1 = not in it
	struct FootprintBox {unsigned int lo[3] = {0, 0, 0}, len[3] = {0, 0, 0}; std::vector<int32_t> slot;};
	FootprintBox m_bsBoxes[6];            //!< [field*3+n]; empty slot: the hash table
	const FDTD_FLOAT* m_bsReplayValues;   //!< BS_REPLAY: the probe sample of the replayed timestep, or nullptr
	mutable std::vector<uint64_t> m_bsRecorded;
	std::vector<unsigned int> m_bsProbeTS, m_bsEnergyTS;
	std::vector<FDTD_FLOAT> m_bsValues;  //!< [probe sample][footprint cell]
	std::vector<double> m_bsEnergy;      //!< [energy sample]
	std::function<double()> m_bsEnergyFn;
	//@}

	friend class NS_Engine_Multithread::thread; // evil hack to access numTS from multithreading context
};

#endif // ENGINE_H
