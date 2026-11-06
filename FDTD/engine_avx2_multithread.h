/*
*	Copyright (C) 2026 openEMS contributors
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

#ifndef ENGINE_AVX2_MULTITHREAD_H
#define ENGINE_AVX2_MULTITHREAD_H

#include "operator_avx2_multithread.h"
#include "engine_avx2.h"

#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>
#include "tools/barrier.h"

#include "tools/useful.h"
#if defined(_WIN32) && !defined(__GNUC__)
#include <Winsock2.h>
#else
#include <sys/time.h>
#endif

class Engine_AVX2_Multithread;

namespace NS_Engine_AVX2_Multithread
{

class thread
{
public:
	thread(Engine_AVX2_Multithread* ptr, unsigned int start, unsigned int stop,
	       unsigned int stop_h, unsigned int threadID);
	void operator()();

protected:
	unsigned int m_start, m_stop, m_stop_h, m_threadID;
	Engine_AVX2_Multithread* m_enginePtr;
};

} // namespace NS_Engine_AVX2_Multithread


class Engine_AVX2_Multithread : public Engine_AVX2
{
	friend class NS_Engine_AVX2_Multithread::thread;
public:
	static Engine_AVX2_Multithread* New(const Operator_AVX2_Multithread* op,
	                                     unsigned int numThreads = 0);
	virtual ~Engine_AVX2_Multithread();

	virtual void setNumThreads(unsigned int numThreads);
	virtual void Init();
	virtual void Reset();
	virtual void NextInterval(float curr_speed);

	//! Iterate number of timesteps
	virtual bool IterateTS(unsigned int iterTS);

	virtual unsigned int GetNumThreads() const {return m_numThreads;}
	virtual CoreSet GetCoreSet() const {return m_cores;}
	virtual bool SetThreads(unsigned int numThreads, const CoreSet& cores);

	//! In-batch sampling inside the schedules (plans/BATCH_SAMPLING.md §4.4).
	/*!
	  One start/stop round for the whole batch, flat or blocked, the thread
	  calibration included. The workers take each probe and energy sample as
	  their slab completes the timestep: a blocked schedule's depth is not
	  capped by the samples, so Dirac and step excitations block too.
	*/
	virtual bool IterateSampledTS(unsigned int iterTS);
	virtual bool SamplesInsideBlocks() const {return true;}
	//! Two blocks: at least two thirds of a batch then run at full depth.
	virtual unsigned int GetPreferredBatchLength() const {return (m_blk_k > 1) ? 2 * m_blk_k : 0;}

	virtual void DoPreVoltageUpdates()  { throw std::runtime_error("Engine_AVX2_Multithread::DoPreVoltageUpdates without thread ID"); }
	virtual void DoPreVoltageUpdates(int threadID);
	virtual void DoPostVoltageUpdates() { throw std::runtime_error("Engine_AVX2_Multithread::DoPostVoltageUpdates without thread ID"); }
	virtual void DoPostVoltageUpdates(int threadID);
	virtual void Apply2Voltages()       { throw std::runtime_error("Engine_AVX2_Multithread::Apply2Voltages without thread ID"); }
	virtual void Apply2Voltages(int threadID);

	virtual void DoPreCurrentUpdates()  { throw std::runtime_error("Engine_AVX2_Multithread::DoPreCurrentUpdates without thread ID"); }
	virtual void DoPreCurrentUpdates(int threadID);
	virtual void DoPostCurrentUpdates() { throw std::runtime_error("Engine_AVX2_Multithread::DoPostCurrentUpdates without thread ID"); }
	virtual void DoPostCurrentUpdates(int threadID);
	virtual void Apply2Current()        { throw std::runtime_error("Engine_AVX2_Multithread::Apply2Current without thread ID"); }
	virtual void Apply2Current(int threadID);

protected:
	Engine_AVX2_Multithread(const Operator_AVX2_Multithread* op);
	void changeNumThreads(unsigned int numThreads);

	//! Run \a iterTS timesteps while measuring candidate thread counts.
	bool CalibrateIterateTS(unsigned int iterTS);
	//! Seed the candidate list with a geometric ladder over the available cores.
	void BuildCalibrationPlan();
	//! Score the current candidate and move on, refine, or settle.
	void AdvanceCalibration();
	//! Latch the fastest measured thread count and leave calibration mode.
	void FinishCalibration();

	// --- trapezoidal temporal blocking (prototype, opt-in) ---
	virtual TemporalBlockingRequest GetTemporalBlocking() const {return m_blk_request;}
	virtual bool SetTemporalBlocking(const TemporalBlockingRequest& req, bool announce = true);
	virtual TemporalBlockingPlan PlanTemporalBlocking(const TemporalBlockingRequest& req) const;
	virtual std::vector<TemporalBlockingRequest> TemporalBlockingCandidates(unsigned int maxDepth) const;
	virtual bool IsCalibrating() const {return m_opt_speed;}
	virtual long long GetBlockedTimesteps() const {return m_blk_blockedTS;}
	//! Resolve m_blk_request to k, the tile width and the tile edges, or leave blocking disabled.
	//! \a announce: print the result, unless m_blk_quiet.
	void ConfigureTemporalBlocking(bool announce = true);
	//! PlanTemporalBlocking(), with the tile edges of an active plan.
	TemporalBlockingPlan PlanTemporalBlocking(const TemporalBlockingRequest& req,
	                                          std::vector<int>* edgesOut) const;
	//! Static contiguous split of [lo,hi) over \a nThreads, part \a id.
	static void SplitRange(int lo, int hi, unsigned int nThreads, unsigned int id,
	                       unsigned int& start, unsigned int& stop);
	//! One trapezoid: k timesteps over a narrowing (dir=+1) or widening (dir=-1) x-range.
	void TrapezoidSweep(int A0, int B0, int k, int dir, int t0, unsigned int threadID);
	//! This thread's share of the voltage / current update of the x-lines [lo,hi).
	/*!
	  Split by y-rows, not whole x-lines (unless OPENEMS_AVX2_TBLOCK_SPLIT=lines):
	  a slab is often only a few x-lines wide, and a wedge starts at one, so a
	  split by x-line leaves most threads waiting on whichever got one more.
	*/
	void UpdateVoltageShare(int lo, int hi, unsigned int threadID);
	void UpdateCurrentShare(int lo, int hi, unsigned int threadID);

	//! The blocked counterparts of DoPreVoltageUpdates() and friends.
	void SlabPreVoltage(unsigned int startX, unsigned int stopX, int numTS, unsigned int threadID);
	void SlabPostVoltage(unsigned int startX, unsigned int stopX, int numTS, unsigned int threadID);
	void SlabApply2Voltages(unsigned int startX, unsigned int stopX, int numTS, unsigned int threadID);
	void SlabPreCurrent(unsigned int startX, unsigned int stopX, int numTS, unsigned int threadID);
	void SlabPostCurrent(unsigned int startX, unsigned int stopX, int numTS, unsigned int threadID);
	void SlabApply2Current(unsigned int startX, unsigned int stopX, int numTS, unsigned int threadID);
	//! Worker body for the blocked schedule.
	void BlockedWorker(unsigned int threadID);

	// --- in-batch sampling inside the schedules (IterateSampledTS()) ---
	//! Index the footprint by x-line (m_fpX, m_fpCell, m_fpMem), if it changed.
	void IndexFootprint();
	//! Before a start barrier of a sampled batch: room for the energy rows of one block.
	void PrepareSampledRound();
	//! The probe sample of timestep \a ts+1, or -1; the same for the energy sample.
	int ProbeSampleAt(int ts) const;
	int EnergySampleAt(int ts) const;
	//! Part \a part of \a parts of the footprint cells of \a field with x in [lo,hi), into probe sample \a slot.
	void GatherFootprint(unsigned int field, int lo, int hi, int slot, unsigned int part, unsigned int parts);
	//! Visit this thread's share of the rows [lo,hi) x [0,ny) as UpdateVoltageShare() (ny = NY) or
	//! UpdateCurrentShare() (ny = NY-1) splits them, as runs f(x, y0, y1).
	template <typename F>
	void ForUpdateShare(int lo, int hi, unsigned int ny, unsigned int threadID, F f) const;
	//! The rows buffer of the energy sample \a j for the block that starts at timestep \a t0.
	float* EnergyRowsFor(int j, int t0, bool currents);
	//! Blocked: the samples of \a ts+1 a pass takes once its E range [eLo,eHi) has passed Apply2Voltages of \a ts.
	void SampleVoltages(int eLo, int eHi, int ts, int t0, unsigned int threadID);
	//! Blocked: those it takes once its current update [An,hHi) and hook range [hS,hE) have passed Apply2Current of \a ts.
	void SampleCurrents(int An, int hHi, int hS, int hE, int ts, int t0, unsigned int threadID);
	//! Blocked, after the block [t0,t0+k): the energy samples of the block from their rows.
	void FinishBlockEnergy(int t0, unsigned int k, unsigned int threadID);
	//! Flat: the samples of \a ts+1, the thread's x-lines [xStart,xStop) once timestep \a ts is complete.
	void SampleFlat(unsigned int xStart, unsigned int xStop, int ts, unsigned int threadID);

	//! Wait at m_IterateBarrier, counting the wait if OPENEMS_AVX2_SYNC_STATS is set.
	enum SyncKind { SYNC_UPDATE = 0, SYNC_HOOK = 1 };
	void SyncWait(unsigned int threadID, SyncKind kind);
	//! Print and clear the synchronisation statistics gathered so far.
	void PrintSyncStats();

	const Operator_AVX2_Multithread* m_Op_MT;
	std::vector<std::thread> m_threads;
	Barrier* m_startBarrier;
	Barrier* m_stopBarrier;
	HybridBarrier* m_IterateBarrier;
	unsigned int m_barrierSpinUs;  //!< OPENEMS_AVX2_BARRIER_SPIN_US: spin before sleeping at m_IterateBarrier

	// --- synchronisation statistics (OPENEMS_AVX2_SYNC_STATS=1) ---
	//! Per worker, one cache line each so the counters do not share one.
	struct alignas(64) SyncStats
	{
		uint64_t rounds[2];   //!< barrier rounds, by SyncKind
		int64_t  waitNs[2];   //!< arrival until the last thread arrived
		int64_t  wakeNs[2];   //!< release until this thread was running again
		int64_t  activeNs;    //!< time between the start and stop barriers
		uint64_t timesteps;   //!< timesteps advanced while active
	};
	bool m_syncStats;
	std::vector<SyncStats> m_sync;
	volatile unsigned int m_iterTS;
	unsigned int m_numThreads;
	unsigned int m_max_numThreads; //!< hardware threads in m_cores
	CoreSet m_cores;               //!< where the workers may run (--cpu-cores)
	volatile bool m_stopThreads;
	bool m_opt_speed;

	// --- thread-count calibration (only live while m_opt_speed) ---
	std::vector<unsigned int> m_cal_plan;    //!< candidate thread counts, in probe order
	std::vector<double> m_cal_score;         //!< timesteps/second measured per candidate
	size_t m_cal_index;                      //!< candidate currently being measured
	double m_cal_time;                       //!< seconds accumulated for that candidate
	unsigned int m_cal_steps;                //!< timesteps accumulated for that candidate
	bool m_cal_refined;                      //!< refinement candidates already appended
	bool m_cal_warmup;                       //!< discard the next batch (threads just respawned)

	// --- temporal blocking state (read by every worker; fixed while a batch runs) ---
	TemporalBlockingRequest m_blk_request; //!< as asked for: option or environment; AUTO is off
	bool m_blk_quiet;        //!< resolve without printing (--benchmark switching schedules)
	unsigned int m_blk_k;    //!< timesteps advanced per block; 0 disables blocking
	unsigned int m_blk_W;    //!< tile width in x-planes
	//! Tile boundaries: cores are consecutive pairs, wedges the interior entries.
	/*!
	  Not simply multiples of m_blk_W -- a short tail tile is folded into its
	  neighbour so that the last tile is at least k+3 lines wide, which is what
	  lets a boundary-anchored extension reach two cells inward from x=NX-1.
	  See ConfigureTemporalBlocking().
	*/
	std::vector<int> m_blk_edges;
	bool m_blk_active;       //!< workers take the trapezoidal path this batch
	bool m_blk_rowSplit;     //!< split slab updates by y-row (OPENEMS_AVX2_TBLOCK_SPLIT)
	bool m_blk_elide;        //!< skip or merge extension barriers (OPENEMS_AVX2_TBLOCK_ELIDE)
	long long m_blk_blockedTS; //!< timesteps run in blocks of depth >= 2 (thread 0 counts)

	// --- in-batch sampling state (read by every worker; fixed while a batch runs) ---
	bool m_sp_on;                      //!< the workers sample this batch
	unsigned int m_sp_len;             //!< its timesteps
	//! By offset ts - m_bsBatchStart: the probe sample of timestep ts+1, -1 for none.
	std::vector<int> m_sp_probe;
	//! By offset: the energy samples of the timesteps before ts+1 (size m_sp_len+1);
	//! offset o holds one if m_sp_energyBefore[o+1] > m_sp_energyBefore[o].
	std::vector<unsigned int> m_sp_energyBefore;
	unsigned int m_sp_rowCount;        //!< FastEnergyRowCount()
	//! [slot][field][row]: the float row sums of the energy samples of one block, in
	//! the order of their timesteps (Engine_AVX2::FastEnergyFieldRows()); k slots.
	std::vector<float> m_sp_rows;
	//! The footprint by x-line, per field (0 voltages, 1 currents): the entries of
	//! x-line x are [m_fpX[f][x], m_fpX[f][x+1]) of m_fpCell[f] (the cell's index in a
	//! probe sample) and m_fpMem[f] (its float offset in m_volt or m_curr).
	std::vector<uint32_t> m_fpX[2];
	std::vector<uint32_t> m_fpCell[2];
	std::vector<uint64_t> m_fpMem[2];
	//! The footprint m_fpX was built for (StopFootprintRecording() swaps in a new vector).
	const uint64_t* m_fpIndexedData;
	size_t m_fpIndexedSize;
};

#endif // ENGINE_AVX2_MULTITHREAD_H
