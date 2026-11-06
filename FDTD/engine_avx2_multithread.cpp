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

#include "engine_avx2_multithread.h"
#include "extensions/engine_extension.h"
#include "tools/denormal.h"
#include "tools/cpu_topology.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <iomanip>
#include <sstream>
#include <unistd.h>

using std::cout;
using std::endl;

namespace
{
// Calibration batch granularity. Small enough that a candidate is scored
// promptly even when the caller asks for many timesteps at once, large enough
// that the two barrier round-trips per timestep are not what is being timed.
const unsigned int CAL_BATCH_TS = 8;
// A candidate is only scored once it has run at least this many timesteps and
// this much wall time, so that a fast in-cache grid is not ranked on noise.
const unsigned int CAL_MIN_TS = 16;
const double CAL_MIN_SECONDS = 0.025;
}

// ============================================================================
// Engine_AVX2_Multithread
// ============================================================================

Engine_AVX2_Multithread* Engine_AVX2_Multithread::New(
	const Operator_AVX2_Multithread* op, unsigned int numThreads)
{
	cout << "Create FDTD engine (AVX2 + FMA + multi-threading)" << endl;
	Engine_AVX2_Multithread* e = new Engine_AVX2_Multithread(op);
	e->setNumThreads(numThreads);
	e->Init();
	return e;
}

Engine_AVX2_Multithread::Engine_AVX2_Multithread(const Operator_AVX2_Multithread* op)
	: Engine_AVX2(op)
{
	m_Op_MT = op;
	m_IterateBarrier = nullptr;
	m_startBarrier = nullptr;
	m_stopBarrier = nullptr;
	m_cores = op->GetCoreSet();
	m_max_numThreads = std::max(1u, CpuTopology::Get().MaxThreads(m_cores));
	m_numThreads = 0;
	m_opt_speed = false;
	m_stopThreads = true;
	m_cal_index = 0;
	m_cal_time = 0.0;
	m_cal_steps = 0;
	m_cal_refined = false;
	m_cal_warmup = true;
	m_blk_k = 0;
	m_blk_W = 0;
	m_blk_active = false;
	m_blk_blockedTS = 0;

	// Synchronisation knobs of the blocked schedule, kept switchable so each
	// can be measured against the old behaviour on the same binary.
	m_barrierSpinUs = 0;
	if (const char* env = std::getenv("OPENEMS_AVX2_BARRIER_SPIN_US"))
		m_barrierSpinUs = (unsigned int)std::strtoul(env, nullptr, 10);
	m_syncStats = false;
	if (const char* env = std::getenv("OPENEMS_AVX2_SYNC_STATS"))
		m_syncStats = (env[0] == '1');
	m_blk_rowSplit = true;
	if (const char* env = std::getenv("OPENEMS_AVX2_TBLOCK_SPLIT"))
		m_blk_rowSplit = (std::string(env) != "lines");
	m_blk_elide = true;
	if (const char* env = std::getenv("OPENEMS_AVX2_TBLOCK_ELIDE"))
		m_blk_elide = (env[0] != '0');

	// OPENEMS_AVX2_TEMPORAL_BLOCK=<k> derives the tile width from the last-level
	// cache; <k>:<W> sets both by hand. --avx2-temporal-block replaces it
	// (openEMS::SetupFDTD()).
	m_blk_quiet = false;
	if (const char* env = std::getenv("OPENEMS_AVX2_TEMPORAL_BLOCK"))
	{
		if (!TemporalBlockingRequest::Parse(env, m_blk_request))
			std::cerr << "AVX2 temporal blocking: cannot read OPENEMS_AVX2_TEMPORAL_BLOCK=" << env
			     << " (off, <k> or <k>:<W>); blocking off" << endl;
	}
}

Engine_AVX2_Multithread::~Engine_AVX2_Multithread()
{
	Reset();
}

void Engine_AVX2_Multithread::setNumThreads(unsigned int numThreads)
{
	m_numThreads = numThreads;
}

void Engine_AVX2_Multithread::Init()
{
	m_stopThreads = true;
	m_opt_speed = false;
	Engine_AVX2::Init();

	// Initialize threads
	m_stopThreads = false;
	if (m_numThreads == 0)
	{
		m_opt_speed = true;
		BuildCalibrationPlan();
		m_numThreads = m_cal_plan.front();
	}
	else if (m_numThreads > m_max_numThreads)
		m_numThreads = m_max_numThreads;

	this->changeNumThreads(m_numThreads);
	// The tile is sized from the L3 instances the threads occupy: while the
	// count is calibrated every candidate gets its own, and the run's is
	// announced once the count is settled (FinishCalibration()).
	ConfigureTemporalBlocking(!m_opt_speed);
}

bool Engine_AVX2_Multithread::SetThreads(unsigned int numThreads, const CoreSet& cores)
{
	const CpuTopology& topo = CpuTopology::Get();
	if (!topo.Offers(cores))
		return false;
	m_cores = cores;
	m_max_numThreads = std::max(1u, topo.MaxThreads(cores));
	m_opt_speed = false;
	if (numThreads == 0)
	{
		m_opt_speed = true;
		BuildCalibrationPlan();
		numThreads = m_cal_plan.front();
	}
	this->changeNumThreads(std::min(numThreads, m_max_numThreads));
	ConfigureTemporalBlocking(!m_opt_speed);
	return true;
}

void Engine_AVX2_Multithread::Reset()
{
	if (!m_threads.empty())
	{
		PrintSyncStats();
		ClearExtensions();

		m_stopThreads = true;
		m_startBarrier->wait();
		for (auto& t : m_threads) t.join();
		m_threads.clear();
		delete m_IterateBarrier;
		m_IterateBarrier = nullptr;
		delete m_startBarrier;
		m_startBarrier = nullptr;
		delete m_stopBarrier;
		m_stopBarrier = nullptr;
	}

	Engine_AVX2::Reset();
}

void Engine_AVX2_Multithread::changeNumThreads(unsigned int numThreads)
{
	if (!m_threads.empty())
	{
		PrintSyncStats();
		m_stopThreads = true;
		m_startBarrier->wait();
		for (auto& t : m_threads) t.join();
		m_threads.clear();
	}

	m_numThreads = numThreads;
	m_stopThreads = false;
	m_sync.assign(m_numThreads, SyncStats());

	if (g_settings.GetVerboseLevel() > 0)
		cout << "AVX2 multithreaded engine using " << m_numThreads << " threads. Utilization: (";

	std::vector<unsigned int> m_Start_Lines;
	std::vector<unsigned int> m_Stop_Lines;
	m_Op_MT->CalcStartStopLines(m_numThreads, m_Start_Lines, m_Stop_Lines);

	delete m_IterateBarrier;
	m_IterateBarrier = new HybridBarrier(m_numThreads, m_barrierSpinUs);

	delete m_startBarrier;
	m_startBarrier = new Barrier(m_numThreads + 1);

	delete m_stopBarrier;
	m_stopBarrier = new Barrier(m_numThreads + 1);

	for (unsigned int n = 0; n < m_numThreads; n++)
	{
		unsigned int start = m_Start_Lines.at(n);
		unsigned int stop = m_Stop_Lines.at(n);
		unsigned int stop_h = stop;
		if (n == m_numThreads - 1)
		{
			// Last thread processes H-field one row less
			stop_h = stop - 1;
			if (g_settings.GetVerboseLevel() > 0)
				cout << stop - start + 1 << ")" << endl;
		}
		else
			if (g_settings.GetVerboseLevel() > 0)
				cout << stop - start + 1 << ";";

		m_threads.emplace_back(
			NS_Engine_AVX2_Multithread::thread(this, start, stop, stop_h, n)
		);
	}

	for (size_t n = 0; n < m_Eng_exts.size(); ++n)
		m_Eng_exts.at(n)->SetNumberOfThreads(m_numThreads);
}

bool Engine_AVX2_Multithread::IterateTS(unsigned int iterTS)
{
	if (m_opt_speed)
		return CalibrateIterateTS(iterTS);

	m_iterTS = iterTS;
	// A call of one timestep runs flat: a block of depth one is the flat
	// sweep with the wedges' extra barriers. Dirac and step excitations, whose
	// Nyquist interval is one timestep, call nothing else.
	m_blk_active = (m_blk_k > 1) && (iterTS >= 2);

	m_startBarrier->wait(); // start threads
	m_stopBarrier->wait();  // wait for completion

	m_blk_active = false;
	return true;
}

// ----------------------------------------------------------------------------
// Trapezoidal temporal blocking  (prototype -- off unless asked for)
//
// The flat sweep streams the whole grid from DRAM once per timestep. Advancing
// a cache-resident tile through k timesteps before moving on divides that
// traffic by roughly k, which on this class of host is worth 3-4x
// (OPTIMIZATIONS.md §4.7/C2). The schedule below is the one verified
// bit-identical to the flat sweep in python/Tests/bench_temporal_blocking.c.
//
// Two passes per block. Pass 1 runs "cores": tiles of width W whose x-range
// narrows by one per side per timestep, because a cell can only be advanced
// while its neighbours are at the right time. That leaves wedges of
// un-advanced cells at every tile boundary, which pass 2 fills with trapezoids
// that widen instead.
//
// Enabled by OPENEMS_AVX2_TEMPORAL_BLOCK=<k>, and only when every active
// extension can be applied to an x-range at a chosen timestep
// (Engine_Extension::SupportsSlabApply). An extension that has not implemented
// the slab hooks returns false and sends the whole run back to the flat sweep,
// so this is safe by omission rather than by enumeration.
//
// The depth actually used is min(k, iterTS), and openems.cpp calls IterateTS()
// with the number of timesteps until the next probe or field dump. So the grid
// is globally coherent whenever a probe reads it, with no work from the
// processing side -- but the achievable depth is capped by the probe interval,
// which openems.cpp sets to Nyquist/oversampling, typically 5-25 timesteps.
// ----------------------------------------------------------------------------

void Engine_AVX2_Multithread::ConfigureTemporalBlocking(bool announce)
{
	m_blk_edges.clear();
	const TemporalBlockingPlan plan = PlanTemporalBlocking(m_blk_request, &m_blk_edges);
	m_blk_k = plan.active ? plan.k : 0;
	m_blk_W = plan.active ? plan.W : 0;
	if (m_blk_quiet || !announce)
		return;

	if (plan.active)
		cout << "AVX2 temporal blocking ACTIVE (prototype): k=" << m_blk_k
		     << ", tile width " << m_blk_W << " x-lines, " << plan.detail << endl;
	else if (!plan.why.empty())
		cout << "AVX2 temporal blocking: disabled, " << plan.why << endl;
}

bool Engine_AVX2_Multithread::SetTemporalBlocking(const TemporalBlockingRequest& req, bool announce)
{
	m_blk_request = req;
	m_blk_quiet = !announce;
	// While the thread count is calibrated, FinishCalibration() announces it.
	ConfigureTemporalBlocking(!m_opt_speed);
	return true;
}

TemporalBlockingPlan Engine_AVX2_Multithread::PlanTemporalBlocking(const TemporalBlockingRequest& req) const
{
	return PlanTemporalBlocking(req, nullptr);
}

TemporalBlockingPlan Engine_AVX2_Multithread::PlanTemporalBlocking(const TemporalBlockingRequest& req,
                                                                   std::vector<int>* edgesOut) const
{
	TemporalBlockingPlan plan;
	if (edgesOut)
		edgesOut->clear();
	// Off unless forced: "auto" is the flat sweep while this is a prototype.
	if (req.mode != TemporalBlockingRequest::FORCED || req.k < 2)
		return plan;

	// The two guards below refuse grids that blocking cannot help. Both are
	// performance guards, not correctness ones, and both keep the schedule out
	// of reach of a test small enough to run in CI -- "grid fits cache" alone
	// demands ~48 MB of field state. OPENEMS_AVX2_TEMPORAL_BLOCK_FORCE=1 lifts
	// them so bit-identity can be checked on a grid that takes seconds. It
	// makes small runs slower, which is the point of the guards.
	const char* forceEnv = std::getenv("OPENEMS_AVX2_TEMPORAL_BLOCK_FORCE");
	const bool force = forceEnv && forceEnv[0] == '1';
	long k = (long)req.k;

	// One tile of both field arrays should be about half the last-level cache
	// the run has -- unless that costs block depth on a model with little else
	// to cache. Measured with the cheaper synchronisation below (PERFORMANCE.md
	// §5.7): on a graded-mesh transmission line the optimum was 40-60 % of the
	// L3 on three hosts, and its compressed coefficient tables (317k unique
	// entries, 122 MB) were as large as its fields. The accounting here counts
	// only the fields, so there "half" is the whole cache in practice; extension
	// state (UPML) has the same effect. On uniform grids with a handful of
	// unique coefficients, a full-cache tile that kept k beat a half-cache one
	// that forced k down to 3-7 by 17-38 %. Hence: half the cache, and where
	// that is narrower than 2k and the coefficient tables are small next to the
	// fields, widen up to the full cache to keep k. Past that, k is lowered to
	// fit (below): a shallow block that stays resident beat a deep one that did
	// not (4:8 over 8:16 on an 8 MB L3). "The cache the run has" is the sum
	// over the L3 instances its threads occupy, not one of them: the tile is
	// split across cores, so each core's share lands in its own instance.
	// Taking one instance made tiles 2-4x too narrow on multi-socket servers
	// and multi-CCX Ryzens (§5.3).
	const double planeBytes = 2.0 * (double)m_vs_x * sizeof(f8vector);
	const double fieldBytes = planeBytes * (double)numLines[0];
	const LLC_Span llc = CpuTopology::Get().LastLevelCaches(m_cores, m_numThreads);
	long cacheBytes = (long)llc.totalBytes;
	if (cacheBytes <= 0)
		cacheBytes = 32L * 1024 * 1024;
	long W = (long)(0.5 * (double)cacheBytes / planeBytes);
	// vv, vi, iv and ii, three polarisations each; uncompressed, they are twice the fields.
	const double tableBytes = Op->m_Use_Compression
		? (double)Op->m_numCompressedEntries * 12.0 * sizeof(f8vector)
		: 2.0 * fieldBytes;
	if (W < 2 * k && tableBytes < 0.25 * fieldBytes)
		W = std::min((long)((double)cacheBytes / planeBytes), 2 * k);

	if (req.W)
		W = (long)req.W;

	// W >= 2k is a hard constraint of the geometry, not a tuning choice: a
	// wedge's dependencies reach k cells into each neighbouring core, so
	// narrower tiles cannot supply them. Prefer a shallower block over a tile
	// that will not stay resident.
	if (W < 2 * k)
		k = W / 2;
	if (k < 2)
	{
		std::ostringstream why;
		why << "x-planes too large (" << (planeBytes / (1024 * 1024)) << " MB) for a useful tile.";
		plan.why = why.str();
		return plan;
	}

	// Name every extension that refuses, not just the first: a user who has to
	// remove features one at a time to find out why blocking is off learns the
	// same thing in several runs that this tells them in one.
	std::string refused;
	for (size_t n = 0; n < m_Eng_exts.size(); ++n)
	{
		if (!m_Eng_exts.at(n)->SupportsSlabApply())
		{
			if (!refused.empty())
				refused += "', '";
			refused += m_Eng_exts.at(n)->GetExtensionName();
		}
	}
	if (!refused.empty())
	{
		plan.why = "extension '" + refused + "' cannot be applied per x-range.";
		return plan;
	}

	// W >= 2k is a hard constraint of the geometry, not a tuning choice: a
	// wedge's dependencies reach k cells into each neighbouring core, so
	// narrower tiles cannot supply them.
	// Room for at least two cores, so there is a wedge to fill.
	if ((long)numLines[0] < 2 * W && !force)
	{
		std::ostringstream why;
		why << "only " << numLines[0] << " x-lines for a tile width of " << W << ".";
		plan.why = why.str();
		return plan;
	}

	// Below roughly the run's L3 the flat sweep is already cache-resident and
	// there is no traffic to remove; blocking then costs the wedge work for
	// nothing (measured -32% on a 96^3 grid), so do not engage. 48 MB is the
	// threshold measured against a 36 MB L3; it scales with the cache the run
	// has, but is kept as a floor so single-cache hosts behave as before.
	const double fitsBytes = std::max(48.0 * 1024 * 1024, cacheBytes * 4.0 / 3.0);
	if (fieldBytes < fitsBytes && !force)
	{
		std::ostringstream why;
		why << "grid fits cache (" << (fieldBytes / (1024 * 1024)) << " MB of field state).";
		plan.why = why.str();
		return plan;
	}

	// Tile boundaries, with one guarantee bolted on for the benefit of
	// boundary-anchored extensions.
	//
	// A core tile narrows from its *left*, so the tail tile [xL,NX) has slab
	// [xL+t+1,NX) at step t. If the tail is short, that slab reaches x=NX-1
	// while x=NX-2 has dropped out of it -- fine for the Yee update, whose
	// stencil wants the neighbour one timestep behind, but fatal for an
	// extension like Mur or an absorbing sheet, which reads the line one or two
	// cells inward at the *same* timestep it writes the edge. Requiring the
	// tail to be at least k+3 lines keeps x=NX-1, NX-2 and NX-3 in one slab for
	// every step of the block; the low edge already has the same property for
	// free, since the first tile holds x=0 and is never narrower than
	// W-k+1 >= k+1 >= 3.
	//
	// The remainder is folded into the previous tile rather than left as a
	// short one, so the last tile is up to k+2 lines wider than the rest. That
	// costs a little cache residency on one tile out of NX/W and buys Mur
	// x-boundaries -- the most common absorbing boundary there is -- a blocked
	// schedule at all.
	std::vector<int> edges;
	for (long x = 0; x < (long)numLines[0]; x += W)
		edges.push_back((int)x);
	edges.push_back((int)numLines[0]);
	const long minTail = k + 3;
	if (edges.size() >= 3 &&
	    (long)numLines[0] - edges.at(edges.size() - 2) < minTail)
		edges.erase(edges.end() - 2);

	plan.active = true;
	plan.k = (unsigned int)k;
	plan.W = (unsigned int)W;
	plan.tileMB = planeBytes * W / (1024 * 1024);
	std::ostringstream detail;
	detail << plan.tileMB << " MB per tile, "
	       << (edges.size() - 1) << " tiles; L3 " << llc.instances << " x "
	       << (llc.instanceBytes / (1024 * 1024)) << " MB (" << llc.source << "); coefficient tables "
	       << (tableBytes / (1024 * 1024)) << " MB; split by "
	       << (m_blk_rowSplit ? "y-row" : "x-line") << ", barrier elision "
	       << (m_blk_elide ? "on" : "off") << ", barrier spin " << m_barrierSpinUs << " us";
	plan.detail = detail.str();
	if (edgesOut)
		*edgesOut = edges;
	return plan;
}

std::vector<TemporalBlockingRequest> Engine_AVX2_Multithread::TemporalBlockingCandidates(unsigned int maxDepth) const
{
	// Tiles of 25 % to 140 % of the L3 the threads occupy, counted as
	// PlanTemporalBlocking() counts them (the fields only), each at k = W/2:
	// the optima measured in PERFORMANCE.md §5.7 lie between 48 and 60 % on a
	// model with large coefficient tables and near the whole cache on one
	// without, always with k = W/2.
	std::vector<TemporalBlockingRequest> out;
	const double planeBytes = 2.0 * (double)m_vs_x * sizeof(f8vector);
	const LLC_Span llc = CpuTopology::Get().LastLevelCaches(m_cores, m_numThreads);
	const double cacheBytes = llc.totalBytes ? (double)llc.totalBytes : 32.0 * 1024 * 1024;
	for (double share : {0.25, 0.35, 0.5, 0.7, 1.0, 1.4})
	{
		const unsigned int W = (unsigned int)std::min<double>(numLines[0], share * cacheBytes / planeBytes);
		const unsigned int k = std::min({32u, W / 2, maxDepth});
		if (k >= 2)
			out.push_back(TemporalBlockingRequest::Forced(k, W));
	}
	return out;
}

void Engine_AVX2_Multithread::SplitRange(int lo, int hi, unsigned int nThreads,
                                         unsigned int id, unsigned int& start,
                                         unsigned int& stop)
{
	if (hi <= lo || nThreads == 0) { start = stop = 0; return; }
	const unsigned int n = (unsigned int)(hi - lo);
	const unsigned int base = n / nThreads;
	const unsigned int rem  = n % nThreads;
	const unsigned int off  = id * base + (id < rem ? id : rem);
	const unsigned int cnt  = base + (id < rem ? 1u : 0u);
	start = (unsigned int)lo + off;
	stop  = start + cnt;
}

// The six slab dispatchers below mirror the flat path's DoPreVoltageUpdates()
// and friends: same extension order (pre-hooks walk m_Eng_exts backwards,
// since it is sorted by descending priority), and a barrier between any two
// extensions that could conflict. Getting either wrong is not a performance
// bug but a correctness one -- two extensions can touch the same cell, and the
// flat path is what defines who goes first.
// The blocked path pays every barrier once per timestep *per tile*, so three
// kinds are left out. An extension that does not implement a hook is skipped
// along with its barrier. With barrier elision on (OPENEMS_AVX2_TBLOCK_ELIDE,
// default), so is one whose SlabTouchesAt() says it cannot reach this slab -- a
// Mur plane at x=0 is only ever in the first tile, a port only in its own --
// and two consecutive hooks that both only read the fields (SlabHookWrites-
// Fields() false) run without a barrier between them, since neither can
// change what the other reads. The barrier after the last hook stays: the
// update or the next pass writes the fields. Every thread evaluates the same
// masks for the same slab and timestep, so they cannot disagree on how many
// barriers this pass has.
#define SLAB_DISPATCH_HOOK(HOOK, BIT)                                          \
	{                                                                          \
		Engine_Extension* ext = m_Eng_exts.at(n);                              \
		if (!(ext->SlabHookMask() & Engine_Extension::BIT))                    \
			continue;                                                          \
		if (m_blk_elide && !ext->SlabTouchesAt(Engine_Extension::BIT, startX, stopX, numTS)) \
			continue;                                                          \
		if (prev && (!m_blk_elide ||                                           \
		             prev->SlabHookWritesFields(Engine_Extension::BIT) ||      \
		             ext->SlabHookWritesFields(Engine_Extension::BIT)))        \
			SyncWait(threadID, SYNC_HOOK);                                     \
		ext->HOOK(startX, stopX, numTS, (int)threadID);                        \
		prev = ext;                                                            \
	}

#define SLAB_DISPATCH_FWD(NAME, HOOK, BIT)                                     \
void Engine_AVX2_Multithread::NAME(unsigned int startX, unsigned int stopX,    \
                                   int numTS, unsigned int threadID)           \
{                                                                              \
	Engine_Extension* prev = nullptr;                                          \
	for (size_t n = 0; n < m_Eng_exts.size(); ++n)                             \
		SLAB_DISPATCH_HOOK(HOOK, BIT)                                          \
	if (prev)                                                                  \
		SyncWait(threadID, SYNC_HOOK);                                         \
}

#define SLAB_DISPATCH_REV(NAME, HOOK, BIT)                                     \
void Engine_AVX2_Multithread::NAME(unsigned int startX, unsigned int stopX,    \
                                   int numTS, unsigned int threadID)           \
{                                                                              \
	Engine_Extension* prev = nullptr;                                          \
	for (int n = (int)m_Eng_exts.size() - 1; n >= 0; --n)                      \
		SLAB_DISPATCH_HOOK(HOOK, BIT)                                          \
	if (prev)                                                                  \
		SyncWait(threadID, SYNC_HOOK);                                         \
}

SLAB_DISPATCH_REV(SlabPreVoltage,     DoPreVoltageUpdatesSlab,  SLAB_PRE_VOLT)
SLAB_DISPATCH_FWD(SlabPostVoltage,    DoPostVoltageUpdatesSlab, SLAB_POST_VOLT)
SLAB_DISPATCH_FWD(SlabApply2Voltages, Apply2VoltagesSlab,       SLAB_APPLY_VOLT)
SLAB_DISPATCH_REV(SlabPreCurrent,     DoPreCurrentUpdatesSlab,  SLAB_PRE_CURR)
SLAB_DISPATCH_FWD(SlabPostCurrent,    DoPostCurrentUpdatesSlab, SLAB_POST_CURR)
SLAB_DISPATCH_FWD(SlabApply2Current,  Apply2CurrentSlab,        SLAB_APPLY_CURR)

#undef SLAB_DISPATCH_HOOK
#undef SLAB_DISPATCH_FWD
#undef SLAB_DISPATCH_REV

namespace
{
// Visit thread \a id's share of the rows [lo,hi) x [0,ny) as runs (x, y0, y1).
// Rows are numbered x-major, which is also their order in memory, so a share
// is one contiguous stretch of the field arrays.
template <typename F>
void ForRowShare(int lo, int hi, unsigned int ny, unsigned int nThreads, unsigned int id, F f)
{
	if (hi <= lo || ny == 0 || nThreads == 0)
		return;
	const uint64_t rows = (uint64_t)(hi - lo) * ny;
	const uint64_t base = rows / nThreads;
	const uint64_t rem  = rows % nThreads;
	uint64_t r = id * base + std::min<uint64_t>(id, rem);
	const uint64_t stop = r + base + (id < rem ? 1 : 0);
	while (r < stop)
	{
		const unsigned int x  = (unsigned int)lo + (unsigned int)(r / ny);
		const unsigned int y0 = (unsigned int)(r % ny);
		const unsigned int y1 = (unsigned int)std::min<uint64_t>(ny, y0 + (stop - r));
		f(x, y0, y1);
		r += y1 - y0;
	}
}
}

void Engine_AVX2_Multithread::UpdateVoltageShare(int lo, int hi, unsigned int threadID)
{
	if (!m_blk_rowSplit)
	{
		unsigned int s, e;
		SplitRange(lo, hi, m_numThreads, threadID, s, e);
		if (e > s) UpdateVoltages(s, e - s);
		return;
	}
	ForRowShare(lo, hi, numLines[1], m_numThreads, threadID,
	            [this](unsigned int x, unsigned int y0, unsigned int y1) { UpdateVoltageRows(x, y0, y1); });
}

void Engine_AVX2_Multithread::UpdateCurrentShare(int lo, int hi, unsigned int threadID)
{
	if (!m_blk_rowSplit)
	{
		unsigned int s, e;
		SplitRange(lo, hi, m_numThreads, threadID, s, e);
		if (e > s) UpdateCurrents(s, e - s);
		return;
	}
	// UpdateCurrents() stops one row short in y; so does each share.
	ForRowShare(lo, hi, numLines[1] - 1, m_numThreads, threadID,
	            [this](unsigned int x, unsigned int y0, unsigned int y1) { UpdateCurrentRows(x, y0, y1); });
}

void Engine_AVX2_Multithread::TrapezoidSweep(int A0, int B0, int k, int dir,
                                             int t0, unsigned int threadID)
{
	const int NX = (int)numLines[0];
	// UpdateCurrents reads volt at x+1, so the engine never updates H on the
	// last x-line -- the same reason the flat path gives its last thread
	// stop_h = stop-1. That line is therefore always "valid" and needs no slope.
	const int NXH = NX - 1;

	int A = A0, B = B0;
	if (A < 0)  A = 0;
	if (B > NX) B = NX;

	for (int t = 0; t < k; ++t)
	{
		// The slope exists only because data outside [A,B) is at the wrong
		// time. At a domain edge there is no outside -- the update kernels
		// clamp -- so the edge is held, not sloped, or its cells never advance.
		int An = (A <= 0)  ? 0  : A + dir;
		int Bn = (B >= NX) ? NX : B - dir;
		if (An < 0)  An = 0;
		if (Bn > NX) Bn = NX;

		// The E sweep range is not symmetric between the two directions: the
		// passes must partition the grid so every cell is updated exactly once
		// per timestep. A narrowing core leaves gaps of width 2t+1 in E but
		// 2t+2 in H, so a widening wedge sweeps E one cell narrower per side.
		int eLo, eHi;
		if (dir > 0) { eLo = An;                     eHi = (Bn >= NX) ? NX : Bn + 1; }
		else         { eLo = (An <= 0) ? 0 : An + 1; eHi = (Bn >= NX) ? NX : Bn; }
		if (eHi > NX)  eHi = NX;
		if (eHi < eLo) eHi = eLo;

		const int ts = t0 + t;
		const unsigned int eS = (unsigned int)eLo, eE = (unsigned int)eHi;

		SlabPreVoltage(eS, eE, ts, threadID);
		UpdateVoltageShare(eLo, eHi, threadID);
		SyncWait(threadID, SYNC_UPDATE);
		SlabPostVoltage(eS, eE, ts, threadID);
		SlabApply2Voltages(eS, eE, ts, threadID);

		const int hHi = (Bn < NXH) ? Bn : NXH;
		// The sweep stops at NX-1 because UpdateCurrents reads volt at x+1, but
		// the extension hooks must not: the flat path calls Apply2Current() and
		// the DoPre/DoPost pair over the whole domain (stopX = UINT_MAX), so an
		// extension that writes the last x-line gets that call there. The tile
		// owning the domain edge therefore hands the hooks [An, NX) while the
		// sweep itself still stops at NX-1. Only that tile can reach NX -- a
		// wedge never does and every other core has Bn < NX -- so the line is
		// still covered exactly once per timestep.
		const int hExt = (Bn >= NX) ? NX : hHi;
		const unsigned int hS = (unsigned int)An;
		const unsigned int hE = (unsigned int)(hExt < An ? An : hExt);

		SlabPreCurrent(hS, hE, ts, threadID);
		UpdateCurrentShare(An, hHi, threadID);
		SyncWait(threadID, SYNC_UPDATE);
		SlabPostCurrent(hS, hE, ts, threadID);
		SlabApply2Current(hS, hE, ts, threadID);

		A = An;
		B = Bn;
	}
}

void Engine_AVX2_Multithread::BlockedWorker(unsigned int threadID)
{
	const unsigned int total = m_iterTS;
	const int startTS = (int)numTS;
	const int64_t t0 = m_syncStats ? HybridBarrier::now() : 0;

	unsigned int done = 0;
	while (done < total)
	{
		const int t0 = startTS + (int)done;
		unsigned int depth = std::min((unsigned int)m_blk_k, total - done);
		for (Engine_Extension* ext : m_Eng_exts)
			depth = ext->SlabBlockDepth(t0, depth);
		const int kk = (int)depth;

		for (size_t i = 0; i + 1 < m_blk_edges.size(); ++i)
			TrapezoidSweep(m_blk_edges[i], m_blk_edges[i + 1], kk, +1, t0, threadID);
		for (size_t i = 1; i + 1 < m_blk_edges.size(); ++i)
			TrapezoidSweep(m_blk_edges[i], m_blk_edges[i], kk, -1, t0, threadID);

		// What an extension can only do once the whole grid has passed a
		// timestep of the block: the steady-state check of a period. Every
		// slab of the block ended at a barrier, so thread 0 sees all of it.
		bool deferred = false;
		for (Engine_Extension* ext : m_Eng_exts)
			deferred = ext->SlabBlockEnd(t0, depth, (int)threadID) || deferred;
		if (deferred)
			SyncWait(threadID, SYNC_HOOK);

		if ((threadID == 0) && (depth >= 2))
			m_blk_blockedTS += depth;
		done += depth;
	}

	if (m_syncStats)
	{
		m_sync[threadID].activeNs += HybridBarrier::now() - t0;
		m_sync[threadID].timesteps += total;
	}

	// Every thread read numTS before the block's first barrier and none reads
	// it again, so the last barrier above orders this write after all of them.
	if (threadID == 0)
		numTS += total;
}

void Engine_AVX2_Multithread::SyncWait(unsigned int threadID, SyncKind kind)
{
	if (!m_syncStats)
	{
		m_IterateBarrier->wait();
		return;
	}
	const int64_t arrive  = HybridBarrier::now();
	const int64_t release = m_IterateBarrier->waitTimed();
	const int64_t back    = HybridBarrier::now();
	SyncStats& s = m_sync[threadID];
	++s.rounds[kind];
	s.waitNs[kind] += std::max<int64_t>(0, release - arrive);
	s.wakeNs[kind] += std::max<int64_t>(0, back - std::max(release, arrive));
}

void Engine_AVX2_Multithread::PrintSyncStats()
{
	if (!m_syncStats || m_sync.empty())
		return;

	SyncStats sum = SyncStats();
	for (const SyncStats& s : m_sync)
	{
		for (int k = 0; k < 2; ++k)
		{
			sum.rounds[k] += s.rounds[k];
			sum.waitNs[k] += s.waitNs[k];
			sum.wakeNs[k] += s.wakeNs[k];
		}
		sum.activeNs += s.activeNs;
		sum.timesteps += s.timesteps;
	}
	if (sum.timesteps == 0 || sum.activeNs <= 0)
		return;

	// Every thread advances every timestep, so per-thread means are sums / threads.
	const double n = (double)m_sync.size();
	const double ts = sum.timesteps / n;
	const double act = (double)sum.activeNs;
	const uint64_t rounds = sum.rounds[0] + sum.rounds[1];
	auto pct = [act](int64_t ns) { return 100.0 * (double)ns / act; };

	std::ostringstream os;
	os << std::fixed << std::setprecision(1);
	os << "AVX2 sync stats: " << m_sync.size() << " threads, " << (uint64_t)ts << " timesteps, ";
	if (m_blk_k > 1)
		os << "blocked k=" << m_blk_k << " W=" << m_blk_W << ", split by " << (m_blk_rowSplit ? "y-row" : "x-line")
		   << ", elision " << (m_blk_elide ? "on" : "off");
	else
		os << "flat";
	os << ", spin " << m_barrierSpinUs << " us\n";
	os << "  barriers per timestep " << rounds / n / ts << " (update " << sum.rounds[0] / n / ts
	   << ", extension " << sum.rounds[1] / n / ts << ")\n";
	os << "  share of thread time waiting for the slowest thread " << pct(sum.waitNs[0] + sum.waitNs[1])
	   << " % (update " << pct(sum.waitNs[0]) << ", extension " << pct(sum.waitNs[1]) << "), waking up "
	   << pct(sum.wakeNs[0] + sum.wakeNs[1]) << " %; per barrier "
	   << (rounds ? (sum.waitNs[0] + sum.waitNs[1]) / 1e3 / rounds : 0.0) << " + "
	   << (rounds ? (sum.wakeNs[0] + sum.wakeNs[1]) / 1e3 / rounds : 0.0) << " us\n";
	os << "  per thread (waiting / waking %):";
	for (size_t t = 0; t < m_sync.size(); ++t)
	{
		const SyncStats& s = m_sync[t];
		const double a = s.activeNs > 0 ? (double)s.activeNs : 1.0;
		os << " " << 100.0 * (s.waitNs[0] + s.waitNs[1]) / a << "/" << 100.0 * (s.wakeNs[0] + s.wakeNs[1]) / a;
	}
	cout << os.str() << endl;

	m_sync.assign(m_sync.size(), SyncStats());
}

// ----------------------------------------------------------------------------
// Thread-count calibration
//
// Thread count does not affect results -- the domain decomposition is by
// x-line and every pass is barrier-separated -- so candidates can be scored on
// *real* timesteps rather than on a warm-up that is thrown away. Calibration is
// therefore free apart from the timesteps that run at a suboptimal width, which
// is why it probes a geometric ladder (a handful of candidates, ~25 ms each)
// instead of climbing one thread at a time.
//
// The predecessor of this code hill-climbed from one thread inside
// NextInterval(), which the main loop only calls every four seconds. On a
// 4-core host that settled in a few intervals; on a 24-core host it needed 44 s
// to reach its optimum, so any run shorter than a minute spent most of its time
// at a fraction of peak throughput.
// ----------------------------------------------------------------------------

void Engine_AVX2_Multithread::BuildCalibrationPlan()
{
	m_cal_plan.clear();
	m_cal_score.clear();
	m_cal_index = 0;
	m_cal_time = 0.0;
	m_cal_steps = 0;
	m_cal_refined = false;
	m_cal_warmup = true;

	if (m_max_numThreads < 1)
		m_max_numThreads = 1;

	for (unsigned int n = 1; n < m_max_numThreads; n *= 2)
		m_cal_plan.push_back(n);
	m_cal_plan.push_back(m_max_numThreads);
}

bool Engine_AVX2_Multithread::CalibrateIterateTS(unsigned int iterTS)
{
	unsigned int remaining = iterTS;

	// Calibrate on whichever schedule the run will actually use. Blocking
	// changes the answer: the flat sweep's roll-off past 8 threads is DRAM
	// contention (§4.3), and blocking removes the contention, so a thread count
	// tuned against the flat path is tuned against a constraint that will not
	// apply. Every candidate count has its own tile (ConfigureTemporalBlocking()
	// in AdvanceCalibration()), and a batch is at least one block deep.
	m_blk_active = (m_blk_k > 1);

	while (remaining > 0 && m_opt_speed)
	{
		const unsigned int batch = std::min(remaining, std::max(CAL_BATCH_TS, m_blk_k));

		const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
		m_iterTS = batch;
		m_blk_active = (m_blk_k > 1) && (batch >= 2);
		m_startBarrier->wait();
		m_stopBarrier->wait();
		const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

		remaining -= batch;

		// The first batch at a new width runs on freshly spawned threads with
		// cold private caches, which systematically penalises the wider
		// configurations -- exactly the ones being judged. Throw it away.
		if (m_cal_warmup)
		{
			m_cal_warmup = false;
			continue;
		}

		m_cal_steps += batch;
		m_cal_time += std::chrono::duration<double>(t1 - t0).count();

		if (m_cal_steps >= CAL_MIN_TS && m_cal_time >= CAL_MIN_SECONDS)
			AdvanceCalibration();
	}

	if (remaining > 0)
	{
		// Calibration ended inside this call: the rest on the settled schedule.
		m_iterTS = remaining;
		m_blk_active = (m_blk_k > 1) && (remaining >= 2);
		m_startBarrier->wait();
		m_stopBarrier->wait();
	}

	m_blk_active = false;
	return true;
}

void Engine_AVX2_Multithread::AdvanceCalibration()
{
	m_cal_score.push_back(m_cal_time > 0.0 ? m_cal_steps / m_cal_time : 0.0);

	if (g_settings.GetVerboseLevel() > 0)
		cout << "AVX2 Multithreaded Engine: " << std::setw(3) << m_cal_plan.at(m_cal_index)
		     << " threads -> " << std::setprecision(1) << std::fixed
		     << m_cal_score.back() << " timesteps/s" << endl;

	++m_cal_index;

	if (m_cal_index >= m_cal_plan.size())
	{
		if (m_cal_refined)
		{
			FinishCalibration();
			return;
		}
		m_cal_refined = true;

		// Bisect the two ladder gaps flanking the winner. The throughput plateau
		// around the optimum is broad, so one refinement pass is enough to land
		// within a couple of percent; a full linear scan would cost more in
		// slow timesteps than it recovers.
		const size_t best = std::max_element(m_cal_score.begin(), m_cal_score.end())
		                    - m_cal_score.begin();
		const unsigned int b = m_cal_plan.at(best);
		const unsigned int lo = (best > 0) ? m_cal_plan.at(best - 1) : b;
		const unsigned int hi = (best + 1 < m_cal_plan.size()) ? m_cal_plan.at(best + 1) : b;

		const unsigned int mids[2] = { (lo + b) / 2, (b + hi) / 2 };
		for (unsigned int i = 0; i < 2; ++i)
		{
			const unsigned int m = mids[i];
			if (m == 0 || m > m_max_numThreads)
				continue;
			if (std::find(m_cal_plan.begin(), m_cal_plan.end(), m) != m_cal_plan.end())
				continue;
			m_cal_plan.push_back(m);
		}

		if (m_cal_index >= m_cal_plan.size())
		{
			FinishCalibration();
			return;
		}
	}

	m_cal_time = 0.0;
	m_cal_steps = 0;
	m_cal_warmup = true;
	this->changeNumThreads(m_cal_plan.at(m_cal_index));
	ConfigureTemporalBlocking(false);
	m_blk_active = (m_blk_k > 1);
}

void Engine_AVX2_Multithread::FinishCalibration()
{
	const size_t best = std::max_element(m_cal_score.begin(), m_cal_score.end())
	                    - m_cal_score.begin();
	const unsigned int bestThreads = m_cal_plan.at(best);

	m_opt_speed = false;
	m_cal_time = 0.0;
	m_cal_steps = 0;

	if (bestThreads != m_numThreads)
		this->changeNumThreads(bestThreads);

	cout << "AVX2 Multithreaded Engine: Best performance found using "
	     << m_numThreads << " threads";
	if (!m_cores.IsAll())
		cout << " on the " << m_cores.ToString() << " cores";
	cout << "." << endl;
	ConfigureTemporalBlocking();
}

void Engine_AVX2_Multithread::NextInterval(float curr_speed)
{
	// Thread-count tuning happens in CalibrateIterateTS(), which self-times on
	// real timesteps instead of waiting for this callback's four-second cadence.
	Engine_AVX2::NextInterval(curr_speed);
}

void Engine_AVX2_Multithread::DoPreVoltageUpdates(int threadID)
{
	for (int n = m_Eng_exts.size() - 1; n >= 0; --n)
	{
		m_Eng_exts.at(n)->DoPreVoltageUpdates(threadID);
		SyncWait((unsigned int)threadID, SYNC_HOOK);
	}
}

void Engine_AVX2_Multithread::DoPostVoltageUpdates(int threadID)
{
	for (size_t n = 0; n < m_Eng_exts.size(); ++n)
	{
		m_Eng_exts.at(n)->DoPostVoltageUpdates(threadID);
		SyncWait((unsigned int)threadID, SYNC_HOOK);
	}
}

void Engine_AVX2_Multithread::Apply2Voltages(int threadID)
{
	for (size_t n = 0; n < m_Eng_exts.size(); ++n)
	{
		m_Eng_exts.at(n)->Apply2Voltages(threadID);
		SyncWait((unsigned int)threadID, SYNC_HOOK);
	}
}

void Engine_AVX2_Multithread::DoPreCurrentUpdates(int threadID)
{
	for (int n = m_Eng_exts.size() - 1; n >= 0; --n)
	{
		m_Eng_exts.at(n)->DoPreCurrentUpdates(threadID);
		SyncWait((unsigned int)threadID, SYNC_HOOK);
	}
}

void Engine_AVX2_Multithread::DoPostCurrentUpdates(int threadID)
{
	for (size_t n = 0; n < m_Eng_exts.size(); ++n)
	{
		m_Eng_exts.at(n)->DoPostCurrentUpdates(threadID);
		SyncWait((unsigned int)threadID, SYNC_HOOK);
	}
}

void Engine_AVX2_Multithread::Apply2Current(int threadID)
{
	for (size_t n = 0; n < m_Eng_exts.size(); ++n)
	{
		m_Eng_exts.at(n)->Apply2Current(threadID);
		SyncWait((unsigned int)threadID, SYNC_HOOK);
	}
}

// ============================================================================
// Worker thread
// ============================================================================

namespace NS_Engine_AVX2_Multithread
{

thread::thread(Engine_AVX2_Multithread* ptr, unsigned int start, unsigned int stop,
               unsigned int stop_h, unsigned int threadID)
{
	m_enginePtr = ptr;
	m_start = start;
	m_stop = stop;
	m_stop_h = stop_h;
	m_threadID = threadID;
}

void thread::operator()()
{
	// Flush denormals for this thread
	Denormal::Disable();
	CpuTopology::Get().Restrict(m_enginePtr->m_cores, m_threadID, m_enginePtr->m_numThreads);

	// Note: the loop must be unconditional. Testing m_stopThreads here would let
	// a worker leave without ever arriving at m_startBarrier, while the shutdown
	// path in Reset()/changeNumThreads() blocks on that same barrier expecting
	// m_numThreads+1 arrivals -> deadlock. Shutdown is signalled solely by the
	// check *after* the barrier, which is also the point where m_stopThreads is
	// safely visible (the barrier's mutex provides the happens-before edge).
	for (;;)
	{
		// Wait for start signal
		m_enginePtr->m_startBarrier->wait();

		if (m_enginePtr->m_stopThreads)
			return;

		if (m_enginePtr->m_blk_active)
		{
			m_enginePtr->BlockedWorker(m_threadID);
			m_enginePtr->m_stopBarrier->wait();
			continue;
		}

		const bool stats = m_enginePtr->m_syncStats;
		const int64_t t0 = stats ? HybridBarrier::now() : 0;
		for (unsigned int iter = 0; iter < m_enginePtr->m_iterTS; ++iter)
		{
			// Pre-voltage extensions
			m_enginePtr->DoPreVoltageUpdates(m_threadID);
			if (m_threadID == 0)
				m_enginePtr->TraceFieldCell("after voltage pre-extensions");

			// Voltage update (x-range for this thread)
			m_enginePtr->UpdateVoltages(m_start, m_stop - m_start + 1);

			// Synchronize
			m_enginePtr->SyncWait(m_threadID, Engine_AVX2_Multithread::SYNC_UPDATE);
			if (m_threadID == 0)
				m_enginePtr->TraceFieldCell("after voltage update");

			// Post-voltage extensions
			m_enginePtr->DoPostVoltageUpdates(m_threadID);
			m_enginePtr->Apply2Voltages(m_threadID);
			if (m_threadID == 0)
				m_enginePtr->TraceFieldCell("after voltage extensions");

			// Pre-current extensions
			m_enginePtr->DoPreCurrentUpdates(m_threadID);
			if (m_threadID == 0)
				m_enginePtr->TraceFieldCell("after current pre-extensions");

			// Current update (last thread processes one fewer x-line for H-field)
			m_enginePtr->UpdateCurrents(m_start, m_stop_h - m_start + 1);

			// Synchronize
			m_enginePtr->SyncWait(m_threadID, Engine_AVX2_Multithread::SYNC_UPDATE);
			if (m_threadID == 0)
				m_enginePtr->TraceFieldCell("after current update");

			// Post-current extensions
			m_enginePtr->DoPostCurrentUpdates(m_threadID);
			m_enginePtr->Apply2Current(m_threadID);
			if (m_threadID == 0)
				m_enginePtr->TraceFieldCell("after current extensions");

			if (m_threadID == 0)
				++m_enginePtr->numTS;
		}
		if (stats)
		{
			m_enginePtr->m_sync[m_threadID].activeNs += HybridBarrier::now() - t0;
			m_enginePtr->m_sync[m_threadID].timesteps += m_enginePtr->m_iterTS;
		}

		m_enginePtr->m_stopBarrier->wait();
	}
}

} // namespace NS_Engine_AVX2_Multithread
