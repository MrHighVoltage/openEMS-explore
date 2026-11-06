/*
*	Copyright (C) 2015 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include "engine_ext_steadystate.h"
#include "operator_ext_steadystate.h"
#include "FDTD/engine_sse.h"
#include "FDTD/engine_interface_fdtd.h"
#if OPENEMS_ENABLE_AVX2
#include "FDTD/engine_avx2.h"
#endif

Engine_Ext_SteadyState::Engine_Ext_SteadyState(Operator_Ext_SteadyState* op_ext): Engine_Extension(op_ext)
{
	m_Op_SS = op_ext;
	m_Priority = ENG_EXT_PRIO_STEADYSTATE;

	m_period = m_Op_SS->m_TS_period;
	m_ringSize = 3*m_period;
	m_records.assign(m_Op_SS->m_E_probe_dir.size()*(size_t)m_ringSize, 0.0);
	m_last_max_diff = 1;
	m_verdictPassed = m_last_max_diff;
	last_total_energy = 0;
	m_Eng_Interface = NULL;
	const char* trace = std::getenv("OPENEMS_STEADYSTATE_TRACE");
	m_trace = trace && trace[0] == '1';
}

Engine_Ext_SteadyState::~Engine_Ext_SteadyState()
{
	delete m_Eng_Interface;
	m_Eng_Interface = NULL;
}

bool Engine_Ext_SteadyState::IsCheckTimestep(unsigned int TS) const
{
	return (m_period>0) && (TS%m_period==0) && (TS>=2*m_period);
}

void Engine_Ext_SteadyState::RecordProbes(unsigned int TS, unsigned int startX, unsigned int stopX)
{
	if (m_ringSize==0)
		return;
	const unsigned int slot = TS%m_ringSize;
	for (size_t n=0;n<m_Op_SS->m_E_probe_dir.size();++n)
	{
		const unsigned int x = m_Op_SS->m_E_probe_pos[0].at(n);
		if ((x<startX) || (x>=stopX))
			continue;
		m_records[n*m_ringSize + slot] = m_Eng->GetVolt(m_Op_SS->m_E_probe_dir.at(n), x, m_Op_SS->m_E_probe_pos[1].at(n), m_Op_SS->m_E_probe_pos[2].at(n));
	}
}

double Engine_Ext_SteadyState::RingRecord(unsigned int TS, size_t n, unsigned int t) const
{
	return m_records[n*m_ringSize + RingTimestep(TS, m_period, t)%m_ringSize];
}

double Engine_Ext_SteadyState::GetLastDiff()
{
	if (m_gpuUpdater)
		m_gpuUpdater();
	// The check of timestep TS runs while the engine advances from TS to
	// TS+1, so a run that has done ts timesteps has passed those of TS < ts.
	// Outside a replay that is every check made.
	const unsigned int ts = m_Eng->InBatchReplay() ? m_Eng->GetNumberOfTimesteps() : UINT_MAX;
	while (!m_verdicts.empty() && (m_verdicts.front().TS < ts))
	{
		PassVerdict(m_verdicts.front());
		m_verdicts.pop_front();
	}
	return m_verdictPassed;
}

void Engine_Ext_SteadyState::PassVerdict(const Verdict& v)
{
	m_verdictPassed = v.diff;
	TraceCheck(v.TS, v.energy, v.diff);
}

void Engine_Ext_SteadyState::TraceCheck(unsigned int TS, double energy, double diff) const
{
	if (!m_trace)
		return;
	// all digits: the traces of two schedules are compared as text
	std::ostringstream os;
	os.precision(17);
	os << "SteadyState check: TS=" << TS << " energy=" << energy << " diff=" << diff << "\n";
	std::cout << os.str() << std::flush;
}

void Engine_Ext_SteadyState::Apply2Voltages()
{
	const unsigned int TS = m_Eng->GetNumberOfTimesteps();
	RecordProbes(TS, 0, UINT_MAX);
	if (IsCheckTimestep(TS))
		CheckPeriod(TS, m_Eng_Interface->CalcFastEnergy(), [this, TS](size_t n, unsigned int t) {return RingRecord(TS, n, t);});
}

void Engine_Ext_SteadyState::SetEngine(Engine* eng)
{
	Engine_Extension::SetEngine(eng);
	m_rows.clear();
#if OPENEMS_ENABLE_AVX2
	if (eng && (eng->GetType()==Engine::AVX2) && (m_period>0))
		m_rows.assign(2*(size_t)static_cast<Engine_AVX2*>(eng)->FastEnergyRowCount(), 0.0);
#endif
}

bool Engine_Ext_SteadyState::SupportsSlabApply() const
{
	return !m_rows.empty();
}

bool Engine_Ext_SteadyState::SlabTouchesAt(SlabHook hook, unsigned int startX, unsigned int stopX, int numTS) const
{
	(void)hook;
	// every slab has rows of the energy of a check
	if (IsCheckTimestep((unsigned int)numTS))
		return true;
	for (size_t n=0;n<m_Op_SS->m_E_probe_dir.size();++n)
	{
		const unsigned int x = m_Op_SS->m_E_probe_pos[0].at(n);
		if ((x>=startX) && (x<stopX))
			return true;
	}
	return false;
}

void Engine_Ext_SteadyState::Apply2VoltagesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
{
	const unsigned int TS = (unsigned int)numTS;
	// a handful of values: not worth splitting across threads
	if (threadID==0)
		RecordProbes(TS, startX, stopX);
#if OPENEMS_ENABLE_AVX2
	if (!IsCheckTimestep(TS) || m_rows.empty() || (threadID<0) || (threadID>=m_NrThreads))
		return;
	// This slab's rows of the energy: x in [startX,stopX), below NX-1, every
	// y below NY-1. The currents of these rows are at timestep TS-1, as in
	// the flat sweep: a narrowing core updated them at TS-1 itself, and a
	// widening wedge's extra lines were advanced to TS-1 by the cores.
	const Engine_AVX2* eng = static_cast<const Engine_AVX2*>(m_Eng);
	const unsigned int ny = eng->GetNumLines(1)-1;
	const unsigned int xEnd = std::min(stopX, eng->GetNumLines(0)-1);
	if (xEnd<=startX)
		return;
	const uint64_t first = (uint64_t)startX*ny, rows = (uint64_t)(xEnd-startX)*ny;
	const uint64_t nThreads = (uint64_t)m_NrThreads, id = (uint64_t)threadID;
	const uint64_t a = first + rows*id/nThreads, b = first + rows*(id+1)/nThreads;
	eng->FastEnergyRows((unsigned int)a, (unsigned int)b, m_rows.data());
#endif
}

unsigned int Engine_Ext_SteadyState::SlabBlockDepth(int t0, unsigned int k) const
{
	(void)t0;
	// one period at most: one check per block, and the ring of three periods
	// still holds the two periods of a check at the end of its block
	return (m_period>0) ? std::min(k, m_period) : k;
}

bool Engine_Ext_SteadyState::SlabBlockEnd(int t0, unsigned int k, int threadID)
{
	if (m_period==0)
		return false;
	// the one multiple of the period the block may hold
	const unsigned int TS = ((unsigned int)t0 + m_period - 1)/m_period*m_period;
	if ((TS>=(unsigned int)t0+k) || !IsCheckTimestep(TS))
		return false;
#if OPENEMS_ENABLE_AVX2
	if (threadID==0)
	{
		const double energy = static_cast<const Engine_AVX2*>(m_Eng)->FastEnergy(m_rows.data());
		CheckPeriod(TS, energy, [this, TS](size_t n, unsigned int t) {return RingRecord(TS, n, t);});
	}
#else
	(void)threadID;
#endif
	return true;
}

void Engine_Ext_SteadyState::CheckPeriod(unsigned int TS, double energy, const std::function<double(size_t, unsigned int)>& record)
{
	const unsigned int p = m_period;
	const unsigned int rel_pos = TS%(2*p);
	const size_t count = m_Op_SS->m_E_probe_dir.size();
	bool no_valid = true;
	m_last_max_diff = 0;
	const double curr_total_energy = energy;
	if (last_total_energy>0)
	{
		m_last_max_diff = std::fabs(curr_total_energy-last_total_energy)/last_total_energy;
		no_valid = false;
	}
	last_total_energy = curr_total_energy;
	unsigned int old_pos = 0;
	unsigned int new_pos = p;
	if (rel_pos<=p)
	{
		new_pos = 0;
		old_pos = p;
	}
	std::vector<double> curr_pow(count, 0), diff_pow(count, 0);
	double max_pow = 0;
	for (size_t n=0;n<count;++n)
	{
		for (unsigned int nt=0;nt<p;++nt)
		{
			const double b_new = record(n, nt+new_pos);
			const double b_old = record(n, nt+old_pos);
			curr_pow[n] += b_new*b_new;
			diff_pow[n] += (b_old-b_new)*(b_old-b_new);
		}
		max_pow = std::max(max_pow, curr_pow[n]);
	}
	for (size_t n=0;n<count;++n)
	{
		if (curr_pow[n]>max_pow*1e-2)
		{
			m_last_max_diff = std::max(m_last_max_diff, diff_pow[n]/curr_pow[n]);
			no_valid = false;
		}
	}
	if ((no_valid) || (m_last_max_diff>1))
		m_last_max_diff = 1;
	// GetLastDiff() hands it on, and the trace prints it, once the run has
	// passed it: an engine may run ahead of the timestep the main loop
	// processes (in-batch sampling), and a run stopping in between never
	// sees the checks it ran ahead to. A run that never asks (--benchmark)
	// keeps a few; a sampled batch never runs further ahead than
	// GetMaxRunAhead(), so that this never passes a verdict its replay needs.
	m_verdicts.push_back(Verdict{TS, curr_total_energy, m_last_max_diff});
	while (m_verdicts.size() > MAX_QUEUED_VERDICTS)
	{
		PassVerdict(m_verdicts.front());
		m_verdicts.pop_front();
	}
}

void Engine_Ext_SteadyState::Apply2Current()
{

}
