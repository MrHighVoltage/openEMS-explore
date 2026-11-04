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

#include <climits>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include "engine_ext_steadystate.h"
#include "operator_ext_steadystate.h"
#include "FDTD/engine_sse.h"
#include "FDTD/engine_interface_fdtd.h"

Engine_Ext_SteadyState::Engine_Ext_SteadyState(Operator_Ext_SteadyState* op_ext): Engine_Extension(op_ext)
{
	m_Op_SS = op_ext;
	m_Priority = ENG_EXT_PRIO_STEADYSTATE;

	m_period = m_Op_SS->m_TS_period;
	m_ringSize = 3*m_period;
	m_records.assign(m_Op_SS->m_E_probe_dir.size()*(size_t)m_ringSize, 0.0);
	m_last_max_diff = 1;
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
	// t is a position in a ring of two periods, which holds TS at TS%(2p) and
	// everything before it back to TS-2p+1: that is timestep TS-((TS-t)%(2p)).
	const unsigned int p2 = 2*m_period;
	const unsigned int ts = TS - ((TS%p2 + p2 - t)%p2);
	return m_records[n*m_ringSize + ts%m_ringSize];
}

double Engine_Ext_SteadyState::GetLastDiff()
{
	if (m_gpuUpdater)
		m_gpuUpdater();
	return m_last_max_diff;
}

void Engine_Ext_SteadyState::UpdateGpuSamples(const float* samples, unsigned int ringPeriods,
	                                           unsigned int completedTS, double totalEnergy)
{
	const unsigned int p = m_Op_SS->m_TS_period;
	const unsigned int completedPeriods = completedTS / p;
	if (!samples || p == 0 || ringPeriods < 3 || completedPeriods < 2)
		return;

	bool noValid = true;
	m_last_max_diff = 0;
	if (last_total_energy > 0)
	{
		m_last_max_diff = std::abs(totalEnergy - last_total_energy) / last_total_energy;
		noValid = false;
	}
	last_total_energy = totalEnergy;

	const unsigned int ringSize = ringPeriods * p;
	const unsigned int oldPeriod = completedPeriods - 2;
	const unsigned int newPeriod = completedPeriods - 1;
	const size_t count = m_Op_SS->m_E_probe_dir.size();
	std::vector<double> currPow(count, 0.0);
	std::vector<double> diffPow(count, 0.0);
	double maxPow = 0;
	for (size_t probe = 0; probe < count; ++probe)
	{
		const float* history = samples + probe * ringSize;
		for (unsigned int nt = 0; nt < p; ++nt)
		{
			const double oldValue = history[(oldPeriod * p + nt) % ringSize];
			const double newValue = history[(newPeriod * p + nt) % ringSize];
			currPow[probe] += newValue * newValue;
			const double delta = oldValue - newValue;
			diffPow[probe] += delta * delta;
		}
		maxPow = std::max(maxPow, currPow[probe]);
	}

	for (size_t probe = 0; probe < count; ++probe)
	{
		if (currPow[probe] > maxPow * 1e-2)
		{
			m_last_max_diff = std::max(m_last_max_diff, diffPow[probe] / currPow[probe]);
			noValid = false;
		}
	}
	if (noValid || m_last_max_diff > 1)
		m_last_max_diff = 1;
	TraceCheck(completedTS, totalEnergy);
}

void Engine_Ext_SteadyState::TraceCheck(unsigned int TS, double energy) const
{
	if (!m_trace)
		return;
	// all digits: the traces of two schedules are compared as text
	std::ostringstream os;
	os.precision(17);
	os << "SteadyState check: TS=" << TS << " energy=" << energy << " diff=" << m_last_max_diff << "\n";
	std::cout << os.str() << std::flush;
}

void Engine_Ext_SteadyState::Apply2Voltages()
{
	const unsigned int TS = m_Eng->GetNumberOfTimesteps();
	RecordProbes(TS, 0, UINT_MAX);
	if (IsCheckTimestep(TS))
		CheckPeriod(TS, m_Eng_Interface->CalcFastEnergy(), [this, TS](size_t n, unsigned int t) {return RingRecord(TS, n, t);});
}

bool Engine_Ext_SteadyState::SlabTouchesAt(SlabHook hook, unsigned int startX, unsigned int stopX, int numTS) const
{
	(void)hook;
	(void)numTS;
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
	// a handful of values: not worth splitting across threads
	if (threadID==0)
		RecordProbes((unsigned int)numTS, startX, stopX);
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
	TraceCheck(TS, curr_total_energy);
}

void Engine_Ext_SteadyState::Apply2Current()
{

}
