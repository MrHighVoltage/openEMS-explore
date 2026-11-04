/*
*	Copyright (C) 2011 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#include "engine_ext_excitation.h"
#include <climits>
#include "operator_ext_excitation.h"
#include "FDTD/engine_sse.h"

Engine_Ext_Excitation::Engine_Ext_Excitation(Operator_Ext_Excitation* op_ext) : Engine_Extension(op_ext)
{
	m_Op_Exc = op_ext;
	m_Priority = ENG_EXT_PRIO_EXCITATION;

	const unsigned int count[2] = {m_Op_Exc->Volt_Count, m_Op_Exc->Curr_Count};
	const unsigned int* index[2] = {m_Op_Exc->Volt_index[0], m_Op_Exc->Curr_index[0]};
	for (int l = 0; l < 2; ++l)
	{
		m_xLo[l] = m_xHi[l] = 0;
		for (unsigned int n = 0; n < count[l]; ++n)
		{
			if (n == 0 || index[l][n] < m_xLo[l]) m_xLo[l] = index[l][n];
			if (n == 0 || index[l][n] + 1 > m_xHi[l]) m_xHi[l] = index[l][n] + 1;
		}
	}
}

bool Engine_Ext_Excitation::SlabTouches(SlabHook hook, unsigned int startX, unsigned int stopX) const
{
	const int l = (hook == SLAB_APPLY_CURR) ? 1 : 0;
	return (m_xLo[l] < stopX) && (startX < m_xHi[l]);
}

Engine_Ext_Excitation::~Engine_Ext_Excitation()
{

}

// startX/stopX bound the x-range to touch and numTS is the timestep to
// evaluate the signal at; first/stride select this thread's stripe of the cell
// list. The whole-grid entry points pass [0,UINT_MAX), the engine's current
// timestep and the whole list, so there is one implementation, not two.
template <typename EngType>
void Engine_Ext_Excitation::Apply2VoltagesImpl(EngType* eng, unsigned int startX, unsigned int stopX, int numTS,
                                               unsigned int first, unsigned int stride)
{
	//soft voltage excitation here (E-field excite)
	int exc_pos;
	unsigned int ny;
	unsigned int pos[3];
	unsigned int length = m_Op_Exc->m_Exc->GetLength();
	FDTD_FLOAT* exc_volt =  m_Op_Exc->m_Exc->GetVoltageSignal();

	int p = numTS+1;
	if (m_Op_Exc->m_Exc->GetSignalPeriod()>0)
		p = (int)m_Op_Exc->m_Exc->GetSignalPeriodTS();

	for (unsigned int n=first; n<m_Op_Exc->Volt_Count; n+=stride)
	{
		if (m_Op_Exc->Volt_index[0][n] < startX || m_Op_Exc->Volt_index[0][n] >= stopX)
			continue;
		exc_pos = numTS - (int)m_Op_Exc->Volt_delay[n];
		exc_pos *= (exc_pos>0);
		exc_pos %= p;
		exc_pos *= (exc_pos<(int)length);
		ny = m_Op_Exc->Volt_dir[n];
		pos[0]=m_Op_Exc->Volt_index[0][n];
		pos[1]=m_Op_Exc->Volt_index[1][n];
		pos[2]=m_Op_Exc->Volt_index[2][n];
		eng->EngType::SetVolt(ny,pos, eng->EngType::GetVolt(ny,pos) + m_Op_Exc->Volt_amp[n]*exc_volt[exc_pos]);
	}
}

void Engine_Ext_Excitation::Apply2Voltages()
{
	const unsigned int startX = 0, stopX = UINT_MAX;
	const int numTS = m_Eng->GetNumberOfTimesteps();
	const unsigned int first = 0, stride = 1;
	ENG_DISPATCH_ARGS(Apply2VoltagesImpl, startX, stopX, numTS, first, stride);
}

void Engine_Ext_Excitation::Apply2VoltagesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
{
	if (threadID < 0 || threadID >= m_NrThreads)
		return;
	const unsigned int first = (unsigned int)threadID, stride = (unsigned int)m_NrThreads;
	ENG_DISPATCH_ARGS(Apply2VoltagesImpl, startX, stopX, numTS, first, stride);
}

template <typename EngType>
void Engine_Ext_Excitation::Apply2CurrentImpl(EngType* eng, unsigned int startX, unsigned int stopX, int numTS,
                                              unsigned int first, unsigned int stride)
{
	//soft current excitation here (H-field excite)
	int exc_pos;
	unsigned int ny;
	unsigned int pos[3];
	unsigned int length = m_Op_Exc->m_Exc->GetLength();
	FDTD_FLOAT* exc_curr =  m_Op_Exc->m_Exc->GetCurrentSignal();

	int p = numTS+1;
	if (m_Op_Exc->m_Exc->GetSignalPeriod()>0)
		p = (int)m_Op_Exc->m_Exc->GetSignalPeriodTS();

	for (unsigned int n=first; n<m_Op_Exc->Curr_Count; n+=stride)
	{
		if (m_Op_Exc->Curr_index[0][n] < startX || m_Op_Exc->Curr_index[0][n] >= stopX)
			continue;
		exc_pos = numTS - (int)m_Op_Exc->Curr_delay[n];
		exc_pos *= (exc_pos>0);
		exc_pos %= p;
		exc_pos *= (exc_pos<(int)length);
		ny = m_Op_Exc->Curr_dir[n];
		pos[0]=m_Op_Exc->Curr_index[0][n];
		pos[1]=m_Op_Exc->Curr_index[1][n];
		pos[2]=m_Op_Exc->Curr_index[2][n];
		eng->EngType::SetCurr(ny,pos, eng->EngType::GetCurr(ny,pos) + m_Op_Exc->Curr_amp[n]*exc_curr[exc_pos]);
	}
}

void Engine_Ext_Excitation::Apply2Current()
{
	const unsigned int startX = 0, stopX = UINT_MAX;
	const int numTS = m_Eng->GetNumberOfTimesteps();
	const unsigned int first = 0, stride = 1;
	ENG_DISPATCH_ARGS(Apply2CurrentImpl, startX, stopX, numTS, first, stride);
}

void Engine_Ext_Excitation::Apply2CurrentSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
{
	if (threadID < 0 || threadID >= m_NrThreads)
		return;
	const unsigned int first = (unsigned int)threadID, stride = (unsigned int)m_NrThreads;
	ENG_DISPATCH_ARGS(Apply2CurrentImpl, startX, stopX, numTS, first, stride);
}
