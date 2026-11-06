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

#include "engine_benchmark.h"
#include "engine.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>

namespace
{
const double WARMUP_SECONDS = 2.0;  //!< GPU clocks ramp up, CPU turbo settles
const double WARMUP_STEP    = 0.25; //!< seconds per warm-up step
const double RUN_SECONDS    = 3.0;  //!< one run, in the flat sweep's time (no worker threads)
const double RUN_SECONDS_THREADED = 2.0; //!< one run, in the fastest thread setting's time
const double SETTLE_SECONDS = 0.3;  //!< after the threads changed: new threads, cold caches
const double POINT_SECONDS  = 2.0;  //!< one thread count
const size_t MAX_POINTS     = 40;   //!< thread counts at most, over all sets of cores
const unsigned int ROUNDS   = 2;
const double NOISE          = 0.02; //!< a spread between rounds worth a warning
const double GAIN           = 0.01; //!< what a blocked schedule has to beat flat by, or more threads fewer

//! \a x rounded up to two significant digits, so the timesteps per run read as chosen.
unsigned int RoundUp2(double x)
{
	if (x < 100)
		return (unsigned int)std::max(1.0, std::ceil(x));
	const double unit = std::pow(10.0, std::floor(std::log10(x)) - 1);
	return (unsigned int)(std::ceil(x / unit) * unit);
}

std::string Percent(double ratio)
{
	std::ostringstream os;
	os << std::showpos << std::fixed << std::setprecision(1) << 100.0 * (ratio - 1.0) << " %";
	return os.str();
}

bool IsSubset(const std::vector<unsigned int>& a, const std::vector<unsigned int>& b)
{
	return std::includes(b.begin(), b.end(), a.begin(), a.end());
}
}

double EngineBenchmark::Candidate::Mean() const
{
	if (mcs.empty())
		return 0;
	double sum = 0;
	for (double v : mcs)
		sum += v;
	return sum / mcs.size();
}

EngineBenchmark::EngineBenchmark(Engine* eng, double numCells, unsigned int timestepsPerRun,
                                 unsigned int configuredThreads)
	: m_eng(eng), m_cells(numCells), m_runTS(timestepsPerRun), m_threaded(eng->GetNumThreads() > 0),
	  m_configuredThreads(configuredThreads), m_phase(WARMUP), m_boundary(0), m_startTS(0),
	  m_warmSeconds(0), m_rate(0), m_warmRate(0), m_maxInterval(0), m_sampledInside(false), m_set(0), m_settling(false),
	  m_bestRate(0), m_settleNext(false), m_pos(0)
{
	m_configured = eng->GetTemporalBlocking();
	m_configuredCores = eng->GetCoreSet();
	m_current.req = TemporalBlockingRequest::Off();
	eng->SetTemporalBlocking(m_current.req, false);
	if (m_threaded)
	{
		// Every hardware thread for the warm-up; the search below replaces the
		// engine's own calibration, which would only delay it.
		m_current.threads = CpuTopology::Get().MaxThreads(m_current.cores);
		eng->SetThreads(m_current.threads, m_current.cores);
	}
	std::cout << "Benchmark: warming up on the flat sweep..." << std::endl;
	Start(eng->GetNumberOfTimesteps(), 4);
}

void EngineBenchmark::Start(unsigned int currTS, unsigned int length)
{
	m_startTS = currTS;
	m_boundary = currTS + std::max(1u, length);
	m_startTime = std::chrono::steady_clock::now();
}

int EngineBenchmark::Clamp(int step, unsigned int currTS) const
{
	if (m_phase == DONE || currTS >= m_boundary)
		return step;
	return std::min(step, (int)(m_boundary - currTS));
}

void EngineBenchmark::NoteInterval(int step, bool sampledInsideBlocks)
{
	// Only the warm-up's intervals shape the candidates. PA->Process() returns
	// the timesteps to the next probe or dump, and the old main loop hands the
	// engine at most that many at a time -- which is the deepest block it
	// can run. With in-batch sampling the loop runs past the probes; an engine
	// that samples between its blocks is still capped by the longest run
	// between two samples, one that samples inside them by nothing the probes
	// do (field dumps, which would still end the batches, are off here).
	if (m_phase != WARMUP)
		return;
	if (sampledInsideBlocks)
		m_sampledInside = true;
	else if (step > 0 && (unsigned int)step > m_maxInterval)
		m_maxInterval = (unsigned int)step;
}

unsigned int EngineBenchmark::MaxDepth() const
{
	// Beyond 32 no engine blocks anyway; a longer interval means "no cap".
	if (m_sampledInside)
		return UINT_MAX;
	return (m_maxInterval > 0 && m_maxInterval < 32) ? m_maxInterval : UINT_MAX;
}

unsigned int EngineBenchmark::SegmentTS(double rate, double seconds, unsigned int k)
{
	return std::max({8u, 2 * k, (unsigned int)std::ceil(rate * seconds)});
}

std::string EngineBenchmark::Label(const TemporalBlockingPlan& plan) const
{
	if (!plan.active)
		return "flat";
	std::ostringstream os;
	os << "k=" << std::min(plan.k, MaxDepth()) << " W=" << plan.W;
	return os.str();
}

std::string EngineBenchmark::ThreadsLabel(const Setting& s) const
{
	std::ostringstream os;
	os << std::setw(3) << s.threads << " on " << std::left << std::setw(15) << s.cores.ToString();
	return os.str();
}

bool EngineBenchmark::Apply(const Setting& s)
{
	const bool threadsChange = m_threaded && !s.SameThreads(m_current);
	m_eng->SetTemporalBlocking(s.req, false);
	// After the schedule: the engine sizes its tile for the threads it gets.
	if (threadsChange)
		m_eng->SetThreads(s.threads, s.cores);
	m_current = s;
	return threadsChange;
}

TemporalBlockingRequest EngineBenchmark::SearchSchedule() const
{
	// The default tile, as deep as it and the probes allow -- the ladder's
	// entry at the engine's own tile. An engine without blocking runs flat.
	return TemporalBlockingRequest::Forced(std::min(32u, MaxDepth()), 0);
}

// ----------------------------------------------------------------------------
// Threads
// ----------------------------------------------------------------------------

void EngineBenchmark::BuildThreadSearch()
{
	const CpuTopology& topo = CpuTopology::Get();
	std::vector<std::vector<unsigned int>> cpus;
	for (const CoreSet& c : topo.DistinctSets())
	{
		// A set mixing kinds of core runs at the pace of its slowest: every
		// worker gets an equal share of the grid. "all" stays, as what a run
		// without --cpu-cores gets.
		if (topo.NumKinds() > 1 && c.kinds == CoreSet::ALL_KINDS && !c.IsAll())
			continue;
		CoreSetPlan p;
		p.cores = c;
		p.maxThreads = topo.MaxThreads(c);
		const std::vector<unsigned int> mine = topo.Select(c);
		for (size_t i = 0; i < cpus.size(); ++i)
			if (IsSubset(cpus[i], mine))
				p.contains.push_back(i);
		cpus.push_back(mine);
		m_sets.push_back(p);
	}
}

bool EngineBenchmark::NextCount(unsigned int& threads)
{
	CoreSetPlan& p = m_sets[m_set];
	if (!p.started)
	{
		// A set containing measured ones starts above their best: below it,
		// the scheduler can place the threads as it did there.
		p.started = true;
		for (const ThreadPoint& t : m_points)
			for (size_t i : p.contains)
				if (t.s.cores == m_sets[i].cores && t.mcs > p.baseMcs)
				{
					p.baseThreads = t.s.threads;
					p.baseMcs = t.mcs;
				}
	}

	// This set's points, and the base as one of them.
	std::vector<std::pair<unsigned int, double>> pts;
	for (const ThreadPoint& t : m_points)
		if (t.s.cores == p.cores)
			pts.emplace_back(t.s.threads, t.mcs);
	const bool own = !pts.empty();
	if (p.baseThreads)
		pts.emplace_back(p.baseThreads, p.baseMcs);
	std::sort(pts.begin(), pts.end());

	if (p.climbing)
	{
		if (pts.empty())
		{
			threads = 1;
			return true;
		}
		// Doubling until the last doubling gained no more than GAIN.
		double before = 0;
		for (size_t i = 0; i + 1 < pts.size(); ++i)
			before = std::max(before, pts[i].second);
		const bool gained = !own || pts.back().second > (1.0 + GAIN) * before;
		if (gained && pts.back().first < p.maxThreads)
		{
			threads = std::min(2 * pts.back().first, p.maxThreads);
			return true;
		}
		p.climbing = false;
	}

	if (!p.refined && own)
	{
		// One count halfway to each neighbour of the best.
		p.refined = true;
		size_t best = 0;
		for (size_t i = 1; i < pts.size(); ++i)
			if (pts[i].second > pts[best].second)
				best = i;
		std::vector<unsigned int> mids;
		if (best > 0)
			mids.push_back((pts[best - 1].first + pts[best].first) / 2);
		if (best + 1 < pts.size())
			mids.push_back((pts[best].first + pts[best + 1].first) / 2);
		for (unsigned int m : mids)
		{
			const bool measured = std::any_of(pts.begin(), pts.end(),
				[m](const std::pair<unsigned int, double>& q) {return q.first == m;});
			if (m > 0 && !measured)
				p.queue.push_back(m);
		}
	}
	if (!p.queue.empty())
	{
		threads = p.queue.front();
		p.queue.erase(p.queue.begin());
		return true;
	}
	return false;
}

void EngineBenchmark::StartNextPoint(unsigned int currTS)
{
	while (m_set < m_sets.size())
	{
		unsigned int n = 0;
		if (m_points.size() < MAX_POINTS && NextCount(n))
		{
			// Expect the speed to scale with the threads, to size the segment
			// that lets the new threads settle; the measurement then uses the
			// speed it found.
			const double expected = m_rate * n / std::max(1u, m_current.threads);
			m_point = ThreadPoint();
			m_point.s.threads = n;
			m_point.s.cores = m_sets[m_set].cores;
			m_point.s.req = SearchSchedule();
			Apply(m_point.s);
			m_point.plan = m_eng->PlanTemporalBlocking(m_point.s.req);
			m_settling = true;
			Start(currTS, SegmentTS(expected, SETTLE_SECONDS, m_point.plan.active ? m_point.plan.k : 0));
			return;
		}
		++m_set;
	}

	BuildCandidates();
	std::cout << "Benchmark: " << m_cand.size() << " settings x " << ROUNDS << " rounds of "
	          << m_runTS << " timesteps" << std::endl;
	m_phase = RUNS;
	m_pos = 0;
	StartRun(currTS);
}

int EngineBenchmark::BestPoint(int exclude) const
{
	int best = -1;
	for (size_t i = 0; i < m_points.size(); ++i)
		if ((int)i != exclude && (best < 0 || m_points[i].mcs > m_points[best].mcs))
			best = (int)i;
	if (best < 0)
		return best;
	// More threads have to be worth it.
	int pick = best;
	for (size_t i = 0; i < m_points.size(); ++i)
		if ((int)i != exclude && m_points[i].mcs * (1.0 + GAIN) >= m_points[best].mcs &&
		    m_points[i].s.threads < m_points[pick].s.threads)
			pick = (int)i;
	return pick;
}

// ----------------------------------------------------------------------------
// Schedules
// ----------------------------------------------------------------------------

void EngineBenchmark::BuildCandidates()
{
	const unsigned int maxDepth = MaxDepth();

	Setting win = m_current;
	int runnerUp = -1;
	if (m_threaded && !m_points.empty())
	{
		const int best = BestPoint(-1);
		win = m_points[best].s;
		runnerUp = BestPoint(best);
	}
	win.req = TemporalBlockingRequest::Off();
	// The ladder is sized for the threads it runs on.
	m_settleNext = Apply(win);

	Candidate flat;
	flat.s = win;
	m_cand.push_back(flat);

	// The ladder first and the configured schedule last, so that where the
	// two coincide the row is marked rather than repeated.
	std::vector<TemporalBlockingRequest> reqs = m_eng->TemporalBlockingCandidates(maxDepth);
	reqs.push_back(m_configured);
	for (size_t i = 0; i < reqs.size(); ++i)
	{
		const bool configured = (i + 1 == reqs.size());
		const TemporalBlockingPlan plan = m_eng->PlanTemporalBlocking(reqs[i]);
		if (!plan.active)
		{
			if (configured)
				m_cand.front().configured = true;
			else if (m_noBlocking.empty())
				m_noBlocking = plan.why;
			continue;
		}
		const unsigned int k = std::min(plan.k, maxDepth);
		auto same = std::find_if(m_cand.begin(), m_cand.end(), [&](const Candidate& c)
		{
			return c.plan.active && std::min(c.plan.k, maxDepth) == k && c.plan.W == plan.W;
		});
		if (same != m_cand.end())
		{
			same->configured = same->configured || configured;
			continue;
		}
		Candidate c;
		c.s = win;
		c.s.req = TemporalBlockingRequest::Forced(k, plan.W);
		c.plan = plan;
		c.configured = configured;
		m_cand.push_back(c);
	}
	std::sort(m_cand.begin() + 1, m_cand.end(), [](const Candidate& a, const Candidate& b)
	{
		return a.plan.W != b.plan.W ? a.plan.W < b.plan.W : a.plan.k < b.plan.k;
	});

	// The runner-up of the thread search, so that the choice between the two
	// gets the same two rounds.
	if (runnerUp >= 0)
	{
		Candidate c;
		c.s = m_points[runnerUp].s;
		c.plan = m_points[runnerUp].plan;
		m_cand.push_back(c);
	}

	if (m_runTS == 0)
		m_runTS = (m_threaded && m_bestRate > 0) ? RoundUp2(m_bestRate * RUN_SECONDS_THREADED)
		                                         : RoundUp2(m_warmRate * RUN_SECONDS);

	// Forward, then backward: a drift in clock or temperature over the
	// benchmark then shows as a difference between the rounds instead of
	// favouring one end of the list.
	for (unsigned int r = 0; r < ROUNDS; ++r)
		for (size_t i = 0; i < m_cand.size(); ++i)
			m_order.push_back((r % 2 == 0) ? i : m_cand.size() - 1 - i);
}

void EngineBenchmark::StartRun(unsigned int currTS)
{
	const Candidate& c = m_cand[m_order[m_pos]];
	m_settling = Apply(c.s) || m_settleNext;
	m_settleNext = false;
	if (m_settling)
		Start(currTS, SegmentTS(m_bestRate, SETTLE_SECONDS, c.plan.active ? c.plan.k : 0));
	else
		Start(currTS, m_runTS);
}

// ----------------------------------------------------------------------------

bool EngineBenchmark::Advance(unsigned int currTS)
{
	const auto now = std::chrono::steady_clock::now();
	const double seconds = std::chrono::duration<double>(now - m_startTime).count();
	const unsigned int steps = currTS - m_startTS;
	if (steps > 0 && seconds > 0)
		m_rate = steps / seconds;

	switch (m_phase)
	{
	case DONE:
		return false;

	case WARMUP:
	{
		m_warmSeconds += seconds;
		const bool settled = m_warmSeconds >= WARMUP_SECONDS && seconds >= 0.5 * WARMUP_STEP &&
		                     !m_eng->IsCalibrating();
		if (!settled)
		{
			Start(currTS, (unsigned int)std::max(1.0, m_rate * WARMUP_STEP));
			return true;
		}
		m_warmRate = m_rate;
		std::cout << "Benchmark: warm-up " << std::fixed << std::setprecision(1) << m_warmSeconds
		          << " s, flat " << m_cells * m_warmRate * 1e-6 << " MC/s";
		if (m_threaded)
		{
			std::cout << " on " << m_current.threads << " threads" << std::endl;
			std::cout << "Benchmark: CPU " << CpuTopology::Get().Describe() << std::endl;
			BuildThreadSearch();
			m_phase = THREADS;
			StartNextPoint(currTS);
			return m_phase != DONE;
		}
		std::cout << std::endl;
		BuildCandidates();
		std::cout << "Benchmark: " << m_cand.size() << " schedules x " << ROUNDS << " rounds of "
		          << m_runTS << " timesteps" << std::endl;
		m_phase = RUNS;
		m_pos = 0;
		StartRun(currTS);
		return true;
	}

	case THREADS:
		if (m_settling)
		{
			m_settling = false;
			Start(currTS, SegmentTS(m_rate, POINT_SECONDS, m_point.plan.active ? m_point.plan.k : 0));
			return true;
		}
		m_point.mcs = (seconds > 0) ? m_cells * steps / seconds * 1e-6 : 0;
		m_bestRate = std::max(m_bestRate, m_rate);
		m_points.push_back(m_point);
		std::cout << "Benchmark threads " << ThreadsLabel(m_point.s) << "  " << std::left << std::setw(12)
		          << Label(m_point.plan) << std::right << std::fixed << std::setprecision(1) << std::setw(10)
		          << m_point.mcs << " MC/s" << std::endl;
		StartNextPoint(currTS);
		return true;

	case RUNS:
	{
		if (m_settling)
		{
			m_settling = false;
			Start(currTS, m_runTS);
			return true;
		}
		Candidate& c = m_cand[m_order[m_pos]];
		const double mcs = (seconds > 0) ? m_cells * steps / seconds * 1e-6 : 0;
		c.mcs.push_back(mcs);
		std::cout << "Benchmark " << std::setw(3) << m_pos + 1 << "/" << m_order.size() << "  ";
		if (m_threaded)
			std::cout << ThreadsLabel(c.s) << "  ";
		std::cout << std::left << std::setw(12) << Label(c.plan) << std::right << std::fixed
		          << std::setprecision(1) << std::setw(10) << mcs << " MC/s" << std::endl;
		if (++m_pos == m_order.size())
		{
			m_phase = DONE;
			return false;
		}
		StartRun(currTS);
		return true;
	}
	}
	return false;
}

// ----------------------------------------------------------------------------
// Report
// ----------------------------------------------------------------------------

void EngineBenchmark::Report(std::ostream& os, const std::string& engineName,
                             const std::string& optionName) const
{
	const std::ios::fmtflags flags = os.flags();
	const std::streamsize precision = os.precision();
	const unsigned int maxDepth = MaxDepth();
	const CpuTopology& topo = CpuTopology::Get();
	const auto restore = [&]()
	{
		os.flags(flags);
		os.precision(precision);
	};

	os << std::endl << "Benchmark results (--engine=" << engineName << "): " << std::fixed
	   << std::setprecision(2) << m_cells * 1e-6 << " M cells; probes kept, field dumps off" << std::endl;
	if (m_threaded)
		os << "  CPU: " << topo.Describe() << std::endl;
	if (m_phase == WARMUP)
	{
		os << "  interrupted during the warm-up" << std::endl;
		restore();
		return;
	}
	if (m_sampledInside)
		os << "  probes sampled inside the blocks (in-batch sampling): they do not cap the block depth"
		   << std::endl;
	else if (maxDepth != UINT_MAX)
		os << "  probes every " << m_maxInterval << " timesteps cap the block depth at "
		   << m_maxInterval << std::endl;

	// The thread search.
	if (!m_points.empty())
	{
		const int best = BestPoint(-1);
		// By set of cores, then by count; measured in the order of the search.
		std::vector<size_t> order(m_points.size());
		for (size_t i = 0; i < order.size(); ++i)
			order[i] = i;
		const auto setIndex = [this](const CoreSet& c)
		{
			for (size_t i = 0; i < m_sets.size(); ++i)
				if (m_sets[i].cores == c)
					return i;
			return m_sets.size();
		};
		std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b)
		{
			const size_t sa = setIndex(m_points[a].s.cores), sb = setIndex(m_points[b].s.cores);
			return sa != sb ? sa < sb : m_points[a].s.threads < m_points[b].s.threads;
		});
		os << std::endl << "  threads  cores              schedule          MC/s" << std::endl;
		for (size_t i : order)
		{
			const ThreadPoint& t = m_points[i];
			os << "  " << std::setw(7) << t.s.threads << "  " << std::left << std::setw(18)
			   << t.s.cores.ToString() << " " << std::setw(12) << Label(t.plan) << std::right
			   << std::setprecision(1) << std::setw(10) << t.mcs;
			if ((int)i == best)
				os << "   best";
			os << std::endl;
		}
		for (const CoreSetPlan& p : m_sets)
			if (p.started)
				os << "  " << std::left << std::setw(18) << p.cores.ToString() << std::right << " = "
				   << topo.DescribeSet(p.cores) << std::endl;
	}

	if (m_cand.empty())
	{
		if (!m_points.empty())
		{
			const ThreadPoint& t = m_points[BestPoint(-1)];
			os << std::endl << "  Interrupted during the thread search. Fastest so far: " << t.s.threads
			   << " threads on " << t.s.cores.ToString() << ", " << Label(t.plan) << ", "
			   << std::setprecision(1) << t.mcs << " MC/s." << std::endl;
		}
		restore();
		return;
	}

	// The schedules.
	double spread = 0;
	size_t best = 0;
	for (size_t i = 0; i < m_cand.size(); ++i)
	{
		const Candidate& c = m_cand[i];
		if (c.Mean() > m_cand[best].Mean())
			best = i;
		if (c.mcs.size() > 1)
		{
			const auto mm = std::minmax_element(c.mcs.begin(), c.mcs.end());
			spread = std::max(spread, (*mm.second - *mm.first) / c.Mean());
		}
	}
	// A run without --numThreads calibrates its count at the start, so the
	// table holds the configured schedule but not the configured run.
	const bool threadsConfigured = !m_threaded ||
		(m_configuredThreads == m_cand.front().s.threads && m_configuredCores == m_cand.front().s.cores);
	// The flat sweep on the same threads, or on the winner's where it was not measured.
	const auto flatFor = [&](const Candidate& c) -> const Candidate&
	{
		for (const Candidate& f : m_cand)
			if (!f.plan.active && f.s.SameThreads(c.s))
				return f;
		return m_cand.front();
	};

	os << std::endl << "  " << m_runTS << " timesteps per run, " << ROUNDS << " rounds:" << std::endl << "  ";
	if (m_threaded)
		os << "threads  cores              ";
	os << "schedule         tile      MC/s";
	for (unsigned int r = 0; r < ROUNDS; ++r)
		os << "   round " << r + 1;
	os << "    vs flat" << std::endl;
	for (const Candidate& c : m_cand)
	{
		os << "  ";
		if (m_threaded)
			os << std::setw(7) << c.s.threads << "  " << std::left << std::setw(18) << c.s.cores.ToString() << " ";
		os << std::left << std::setw(14) << Label(c.plan) << std::right;
		if (c.plan.active)
			os << std::setw(5) << std::setprecision(0) << c.plan.tileMB << " MB";
		else
			os << std::setw(8) << "-";
		os << std::setprecision(1);
		if (c.mcs.empty())
			os << std::setw(10) << "-";
		else
			os << std::setw(10) << c.Mean();
		for (unsigned int r = 0; r < ROUNDS; ++r)
		{
			if (r < c.mcs.size())
				os << std::setw(10) << c.mcs[r];
			else
				os << std::setw(10) << "-";
		}
		const Candidate& f = flatFor(c);
		if (c.plan.active && f.Mean() > 0 && c.Mean() > 0)
			os << std::setw(11) << Percent(c.Mean() / f.Mean());
		else
			os << std::setw(11) << "-";
		if (c.configured)
			os << (threadsConfigured ? "   as configured" : "   configured schedule");
		os << std::endl;
	}
	os << std::endl;

	if (!Finished())
		os << "  Interrupted: the means above cover only the runs that finished." << std::endl;
	if (m_cand.size() == 1 && !m_threaded)
	{
		if (optionName.empty())
			os << "  This engine has no temporal blocking; the flat sweep is all it runs." << std::endl;
		else if (!m_noBlocking.empty())
			os << "  No blocked schedule applies: " << m_noBlocking << std::endl;
		restore();
		return;
	}
	if (!m_noBlocking.empty() && std::none_of(m_cand.begin(), m_cand.end(),
	        [](const Candidate& c) {return c.plan.active;}))
		os << "  No blocked schedule applies: " << m_noBlocking << std::endl;

	// Anything more -- a blocked schedule, more threads -- has to be measurably
	// faster than what it replaces to be recommended: on a GPU that cannot keep
	// a tile in its cache every schedule runs at the flat speed (a Vega 7
	// within 0.1 %), and the fastest of them is the noise.
	const Candidate& f = m_cand[best];
	size_t rec = best;
	for (size_t i = 0; i < m_cand.size(); ++i)
	{
		const Candidate& c = m_cand[i];
		if (c.mcs.empty() || c.Mean() * (1.0 + GAIN) < f.Mean())
			continue;
		const Candidate& q = m_cand[rec];
		if (c.s.threads < q.s.threads || (c.s.threads == q.s.threads && !c.plan.active && q.plan.active))
			rec = i;
	}
	const Candidate& r = m_cand[rec];
	const Candidate& flat = flatFor(f);
	const auto describe = [&](const Candidate& c)
	{
		std::ostringstream d;
		if (m_threaded)
			d << c.s.threads << " threads on " << c.s.cores.ToString() << ", ";
		d << Label(c.plan) << ", " << std::fixed << std::setprecision(1) << c.Mean() << " MC/s";
		return d.str();
	};
	os << (Finished() ? "  Fastest: " : "  Fastest so far: ") << describe(f);
	if (f.plan.active && flat.Mean() > 0)
		os << ", " << std::setprecision(2) << f.Mean() / flat.Mean() << "x the flat sweep";
	os << "." << std::endl;
	if (&r != &f)
	{
		if (r.s.threads < f.s.threads)
			os << "  Recommended: " << describe(r) << ", within " << (int)(100 * GAIN)
			   << " % of the fastest on fewer threads." << std::endl;
		else
			os << "  No blocked schedule beats the flat sweep by more than " << (int)(100 * GAIN)
			   << " %: blocking gains nothing on this model and machine, so the recommendation is flat." << std::endl;
	}
	for (const Candidate& c : m_cand)
		if (c.configured && &c != &r && r.Mean() > 0 && !c.mcs.empty())
			os << "  " << (threadsConfigured ? "As configured (" : "The configured schedule (") << Label(c.plan)
			   << (m_threaded ? ", on these threads" : "") << "): " << std::setprecision(1) << c.Mean() << " MC/s, "
			   << 100.0 * (1.0 - c.Mean() / r.Mean()) << " % below the recommendation." << std::endl;
	if (spread > NOISE)
		os << "  The rounds differ by up to " << std::setprecision(1) << 100 * spread
		   << " % (other load, or clocks drifting): differences below that are noise." << std::endl;

	// The options, on the command line and for the Python interface.
	std::ostringstream cli, py;
	cli << "openEMS <model.xml> --engine=" << engineName;
	py << "FDTD.Run(sim_path, engine='" << engineName << "'";
	if (m_threaded)
	{
		cli << " --numThreads=" << r.s.threads;
		py << ", numThreads=" << r.s.threads;
		if (topo.Select(r.s.cores) != topo.Select(CoreSet()))
		{
			cli << " --cpu-cores=" << r.s.cores.ToString();
			py << ", cpu_cores='" << r.s.cores.ToString() << "'";
		}
	}
	if (!optionName.empty())
	{
		const std::string value = r.plan.active
			? TemporalBlockingRequest::Forced(std::min(r.plan.k, maxDepth), r.plan.W).ToString() : "off";
		std::string keyword = optionName;
		std::replace(keyword.begin(), keyword.end(), '-', '_');
		cli << " --" << optionName << "=" << value;
		py << ", " << keyword << "='" << value << "'";
	}
	py << ")";
	os << "  To run this model with the recommendation on this machine:" << std::endl
	   << "    " << cli.str() << std::endl
	   << "    " << py.str() << std::endl;
	if (r.configured && threadsConfigured && r.s.SameThreads(m_cand.front().s))
		os << "  (This is what the run uses already.)" << std::endl;
	restore();
}
