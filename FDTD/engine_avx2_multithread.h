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
};

#endif // ENGINE_AVX2_MULTITHREAD_H
