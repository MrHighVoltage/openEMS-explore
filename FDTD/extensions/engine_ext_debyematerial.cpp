/*
*	Copyright (C) 2026 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#include "engine_ext_debyematerial.h"
#include <climits>
#if OPENEMS_ENABLE_AVX2
#include "FDTD/engine_avx2.h"
#include <type_traits>
#endif
#include "operator_ext_debyematerial.h"
#include "FDTD/engine_sse.h"

Engine_Ext_DebyeMaterial::Engine_Ext_DebyeMaterial(Operator_Ext_DebyeMaterial* op_ext_debye) : Engine_Ext_Dispersive(op_ext_debye)
{
	m_Op_Ext_Deb = op_ext_debye;
	// one ADE slot, which carries the joint correction of all poles; the base
	// allocated volt_ADE[0] for it
	m_Order = m_Op_Ext_Deb->m_Order;
	unsigned int count = m_Op_Ext_Deb->m_LM_Count.at(0);

	volt_pole_ADE = new FDTD_FLOAT**[m_Op_Ext_Deb->m_PoleCount];
	for (int o=0; o<m_Op_Ext_Deb->m_PoleCount; ++o)
	{
		volt_pole_ADE[o] = new FDTD_FLOAT*[3];
		for (int n=0; n<3; ++n)
		{
			volt_pole_ADE[o][n] = new FDTD_FLOAT[count];
			for (unsigned int i=0; i<count; ++i)
				volt_pole_ADE[o][n][i] = 0.0;
		}
	}

	volt_pre_ADE = new FDTD_FLOAT*[3];
	for (int n=0; n<3; ++n)
	{
		volt_pre_ADE[n] = new FDTD_FLOAT[count];
		for (unsigned int i=0; i<count; ++i)
			volt_pre_ADE[n][i] = 0.0;
	}
}

Engine_Ext_DebyeMaterial::~Engine_Ext_DebyeMaterial()
{
	if (volt_pole_ADE!=NULL)
	{
		for (int o=0; o<m_Op_Ext_Deb->m_PoleCount; ++o)
		{
			for (int n=0; n<3; ++n)
				delete[] volt_pole_ADE[o][n];
			delete[] volt_pole_ADE[o];
		}
		delete[] volt_pole_ADE;
		volt_pole_ADE = NULL;
	}

	if (volt_pre_ADE!=NULL)
		for (int n=0; n<3; ++n)
			delete[] volt_pre_ADE[n];
	delete[] volt_pre_ADE;
	volt_pre_ADE = NULL;
}

template <typename Volt>
void Engine_Ext_DebyeMaterial::PreVoltageEntries(const unsigned int* perm, unsigned int j0, unsigned int j1, Volt volt)
{
	const int poles = m_Op_Ext_Deb->m_PoleCount;
	for (unsigned int j=j0; j<j1; ++j)
	{
		const unsigned int i = perm ? perm[j] : j;
		for (int n=0; n<3; ++n)
		{
			const FDTD_FLOAT v = volt(n, i);
			FDTD_FLOAT pre = 0.0;
			for (int o=0; o<poles; ++o)
			{
				// as much of the pole's new state as V^n already fixes
				FDTD_FLOAT W = m_Op_Ext_Deb->v_relax_ADE[o][n][i]*volt_pole_ADE[o][n][i]
				             + m_Op_Ext_Deb->v_drive_ADE[o][n][i]*v;
				pre += W - volt_pole_ADE[o][n][i];
				volt_pole_ADE[o][n][i] = W;
			}
			volt_pre_ADE[n][i] = pre;
		}
	}
}

template <typename Volt>
void Engine_Ext_DebyeMaterial::PostVoltageEntries(const unsigned int* perm, unsigned int j0, unsigned int j1, Volt volt)
{
	const int poles = m_Op_Ext_Deb->m_PoleCount;
	for (unsigned int j=j0; j<j1; ++j)
	{
		const unsigned int i = perm ? perm[j] : j;
		for (int n=0; n<3; ++n)
		{
			const FDTD_FLOAT v_raw = volt(n, i);
			// solved for the V^(n+1) the poles themselves depend on
			const FDTD_FLOAT v = (v_raw - volt_pre_ADE[n][i]) * m_Op_Ext_Deb->v_solve_ADE[n][i];
			volt_ADE[0][n][i] = v_raw - v;   // what Apply2Voltages() subtracts
			// V^(n+1) is now known, so the pole states can be finished
			for (int o=0; o<poles; ++o)
				volt_pole_ADE[o][n][i] += m_Op_Ext_Deb->v_drive_ADE[o][n][i]*v;
		}
	}
}

template <typename EngType>
void Engine_Ext_DebyeMaterial::DoPreVoltageUpdatesImpl(EngType* eng, unsigned int startX, unsigned int stopX, int threadID)
{
#if OPENEMS_ENABLE_AVX2
	if constexpr (std::is_same<EngType, Engine_AVX2>::value)
	{
		if (threadID < 0)
			BuildAVX2Index(eng);
	}
#endif
	unsigned int xLo, xHi;
	if (!SlabXShare(startX, stopX, threadID, xLo, xHi))
		return;
	const unsigned int* perm;
	unsigned int j0, j1;
	if (!SlabEntries(0, xLo, xHi, threadID, perm, j0, j1))
		return;

#if OPENEMS_ENABLE_AVX2
	if constexpr (std::is_same<EngType, Engine_AVX2>::value)
	{
		// Division-free field access; see Engine_Ext_Dispersive::BuildAVX2Index.
		const unsigned int*  off  = m_avx2_off.data()  + m_avx2_start[0];
		const unsigned char* lane = m_avx2_lane.data() + m_avx2_start[0];
		const f8vector* fld = eng->m_volt;
		PreVoltageEntries(perm, j0, j1, [&](int n, unsigned int i) {return fld[off[i] + n].f[lane[i]];});
		return;
	}
#endif
	unsigned int **pos = m_Op_Ext_Deb->m_LM_pos[0];
	PreVoltageEntries(perm, j0, j1,
	                  [&](int n, unsigned int i) {return eng->EngType::GetVolt(n,pos[0][i],pos[1][i],pos[2][i]);});
}

void Engine_Ext_DebyeMaterial::DoPreVoltageUpdates()
{
	const unsigned int startX = 0, stopX = UINT_MAX;
	const int threadID = -1;
	ENG_DISPATCH_ARGS(DoPreVoltageUpdatesImpl, startX, stopX, threadID);
}

void Engine_Ext_DebyeMaterial::DoPreVoltageUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
{
	// numTS is not used: a per-cell recursion driven by the cell's own voltage,
	// see Engine_Ext_LorentzMaterial::DoPreVoltageUpdatesSlab()
	(void)numTS;
	if (threadID < 0 || threadID >= m_NrThreads)
		return;
	BuildSlabIndex();
	ENG_DISPATCH_ARGS(DoPreVoltageUpdatesImpl, startX, stopX, threadID);
}

template <typename EngType>
void Engine_Ext_DebyeMaterial::DoPostVoltageUpdatesImpl(EngType* eng, unsigned int startX, unsigned int stopX, int threadID)
{
#if OPENEMS_ENABLE_AVX2
	if constexpr (std::is_same<EngType, Engine_AVX2>::value)
	{
		if (threadID < 0)
			BuildAVX2Index(eng);
	}
#endif
	unsigned int xLo, xHi;
	if (!SlabXShare(startX, stopX, threadID, xLo, xHi))
		return;
	const unsigned int* perm;
	unsigned int j0, j1;
	if (!SlabEntries(0, xLo, xHi, threadID, perm, j0, j1))
		return;

#if OPENEMS_ENABLE_AVX2
	if constexpr (std::is_same<EngType, Engine_AVX2>::value)
	{
		const unsigned int*  off  = m_avx2_off.data()  + m_avx2_start[0];
		const unsigned char* lane = m_avx2_lane.data() + m_avx2_start[0];
		const f8vector* fld = eng->m_volt;
		PostVoltageEntries(perm, j0, j1, [&](int n, unsigned int i) {return fld[off[i] + n].f[lane[i]];});
		return;
	}
#endif
	unsigned int **pos = m_Op_Ext_Deb->m_LM_pos[0];
	PostVoltageEntries(perm, j0, j1,
	                   [&](int n, unsigned int i) {return eng->EngType::GetVolt(n,pos[0][i],pos[1][i],pos[2][i]);});
}

void Engine_Ext_DebyeMaterial::DoPostVoltageUpdates()
{
	const unsigned int startX = 0, stopX = UINT_MAX;
	const int threadID = -1;
	ENG_DISPATCH_ARGS(DoPostVoltageUpdatesImpl, startX, stopX, threadID);
}

void Engine_Ext_DebyeMaterial::DoPostVoltageUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
{
	(void)numTS;
	if (threadID < 0 || threadID >= m_NrThreads)
		return;
	BuildSlabIndex();
	ENG_DISPATCH_ARGS(DoPostVoltageUpdatesImpl, startX, stopX, threadID);
}
