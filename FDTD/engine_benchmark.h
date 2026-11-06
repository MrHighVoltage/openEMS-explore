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

#ifndef ENGINE_BENCHMARK_H
#define ENGINE_BENCHMARK_H

#include <chrono>
#include <ostream>
#include <string>
#include <vector>

#include "temporal_blocking.h"
#include "tools/cpu_topology.h"

class Engine;

//! --benchmark: the settings an engine runs a model fastest with, measured on that model.
/*!
  Every setting computes the same fields bit for bit -- the thread count, the
  cores the threads run on, the temporal-blocking schedule (PERFORMANCE.md
  §5.1) -- so the settings need no simulations of their own: the main loop
  keeps stepping the real model, excitation and probes included, and the
  setting changes at the timesteps this class asks the loop to stop at. One
  setup serves every measurement.

  A warm-up runs the flat sweep for at least two seconds, and until the engine
  has stopped calibrating itself. Then, for an engine with worker threads:

  - Threads. For each set of cores that differs on this machine (P-cores,
    with and without SMT, E-cores, all; or cores with and without SMT), the
    thread count doubles from 1 until a doubling gains no more than 1 %, and
    one count halfway to each neighbour of the best refines it. A set that
    contains a measured one starts above that one's best. All of it on the
    engine's own blocked schedule (the default tile), since blocking is what
    lets the threads scale past the memory bandwidth.
  - Schedules, at the best threads and cores: the flat sweep, the engine's
    ladder of tiles and the configured schedule, plus the runner-up of the
    threads at the default tile. Every one the same number of timesteps --
    about two seconds of the fastest thread setting -- twice, the second
    round in reverse order, so that a clock or thermal drift does not favour
    either end of the list.

  An engine without worker threads (the GPU) has only the schedules, with
  runs of about three seconds of the flat sweep.

  Speed is all it measures, and it recommends the fastest, or the flat sweep
  unless a blocked schedule beats it by more than 1 %. A tie-break
  towards smaller tiles, for the DRAM energy a tile that stays in the cache
  saves at equal speed (PERFORMANCE.md §5.5), was tried and dropped: while
  both tiles fit, the deeper block of the larger one moves less data, and on
  a UHD 770 it chose a schedule 1.6 % slower than the fastest.
*/
class EngineBenchmark
{
public:
	//! \a timestepsPerRun 0: from the measured speed. \a configuredThreads: what the run would have used, 0 = calibrated.
	EngineBenchmark(Engine* eng, double numCells, unsigned int timestepsPerRun, unsigned int configuredThreads);

	//! The timestep the main loop has to stop at next.
	unsigned int Boundary() const {return m_boundary;}
	//! \a step, shortened so that it does not run past Boundary().
	int Clamp(int step, unsigned int currTS) const;
	//! The main loop's next processing interval (PA->Process()); it caps the block depth.
	/*!
	  With in-batch sampling, \a step is the longest run between two samples
	  of a batch, which caps it the same way, unless \a sampledInsideBlocks:
	  the engine samples the probes inside its blocks
	  (Engine::SamplesInsideBlocks()), and the probes cap nothing.
	*/
	void NoteInterval(int step, bool sampledInsideBlocks = false);
	//! At Boundary(), with the engine idle: score the run that ended and start
	//! the next. False once every run is done.
	bool Advance(unsigned int currTS);
	bool Finished() const {return m_phase == DONE;}

	//! The tables, and the options that reproduce the recommended settings.
	/*!
	  \a engineName is the --engine value, \a optionName the engine's
	  temporal-blocking option, empty if it has none.
	*/
	void Report(std::ostream& os, const std::string& engineName, const std::string& optionName) const;

private:
	//! What one measurement runs with. threads 0: an engine without worker threads.
	struct Setting
	{
		unsigned int threads = 0;
		CoreSet cores;
		TemporalBlockingRequest req;
		bool SameThreads(const Setting& o) const {return threads == o.threads && cores == o.cores;}
	};
	//! One thread count on one set of cores, at the default tile.
	struct ThreadPoint
	{
		Setting s;
		TemporalBlockingPlan plan;
		double mcs = 0;
	};
	struct Candidate
	{
		Setting s;
		TemporalBlockingPlan plan;
		bool configured = false;  //!< the schedule the run would have used
		std::vector<double> mcs;  //!< one per round
		double Mean() const;
	};
	//! A set of cores the thread search covers.
	struct CoreSetPlan
	{
		CoreSet cores;
		unsigned int maxThreads = 0;
		std::vector<size_t> contains;    //!< earlier sets that are subsets of this one
		bool started = false;
		bool climbing = true;
		bool refined = false;
		unsigned int baseThreads = 0;    //!< the best point of those subsets, where the climb starts; 0: none
		double baseMcs = 0;
		std::vector<unsigned int> queue; //!< counts still to measure
	};
	enum Phase {WARMUP, THREADS, RUNS, DONE};

	void Start(unsigned int currTS, unsigned int length);
	//! Switch the engine to \a s; true if its threads changed, i.e. the next run should settle first.
	bool Apply(const Setting& s);
	//! The engine's blocked schedule for the thread search: its default tile, as deep as the probes allow.
	TemporalBlockingRequest SearchSchedule() const;
	void BuildThreadSearch();
	//! The next thread count to measure in m_sets[m_set], or false when that set is done.
	bool NextCount(unsigned int& threads);
	//! Start measuring the next thread point, or move on to the schedules.
	void StartNextPoint(unsigned int currTS);
	void BuildCandidates();
	void StartRun(unsigned int currTS);
	unsigned int MaxDepth() const;
	std::string Label(const TemporalBlockingPlan& plan) const;
	std::string ThreadsLabel(const Setting& s) const;
	//! Best of the thread points: the fastest, or the fewest threads within 1 % of it.
	int BestPoint(int exclude) const;
	//! Timesteps for about \a seconds at \a rate timesteps per second, and at least two blocks of \a k.
	static unsigned int SegmentTS(double rate, double seconds, unsigned int k);

	Engine* m_eng;
	double m_cells;
	unsigned int m_runTS;
	bool m_threaded;           //!< the engine has worker threads to set
	unsigned int m_configuredThreads;
	CoreSet m_configuredCores;
	Phase m_phase;
	unsigned int m_boundary;   //!< end of the current segment
	unsigned int m_startTS;    //!< its first timestep
	std::chrono::steady_clock::time_point m_startTime;
	double m_warmSeconds;      //!< warm-up so far
	double m_rate;             //!< timesteps per second of the last segment
	double m_warmRate;         //!< timesteps per second at the end of the warm-up
	unsigned int m_maxInterval; //!< longest processing interval seen; 0 = none
	bool m_sampledInside;      //!< the probes are sampled inside the blocks: no interval caps the depth
	TemporalBlockingRequest m_configured;
	std::string m_noBlocking;  //!< why no blocked schedule applies, if none does

	// thread search
	std::vector<CoreSetPlan> m_sets;
	size_t m_set;
	std::vector<ThreadPoint> m_points;
	ThreadPoint m_point;       //!< being measured
	bool m_settling;           //!< the segment running is discarded: threads just started
	double m_bestRate;         //!< timesteps per second of the fastest thread point

	// schedules
	Setting m_current;         //!< what the engine runs now
	bool m_settleNext;         //!< the threads changed before the first run: let it settle
	std::vector<Candidate> m_cand; //!< by thread setting, then flat first, then by tile size
	std::vector<size_t> m_order;   //!< indices into m_cand, all rounds in sequence
	size_t m_pos;
};

#endif // ENGINE_BENCHMARK_H
