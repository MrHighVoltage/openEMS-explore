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

#include "engine_ext_lorentzmaterial.h"
#include <climits>
#if OPENEMS_ENABLE_AVX2
#include "FDTD/engine_avx2.h"
#include <type_traits>
#endif
#include "operator_ext_lorentzmaterial.h"
#include "FDTD/engine_sse.h"

Engine_Ext_LorentzMaterial::Engine_Ext_LorentzMaterial(Operator_Ext_LorentzMaterial* op_ext_lorentz) : Engine_Ext_Dispersive(op_ext_lorentz)
{
	m_Op_Ext_Lor = op_ext_lorentz;
	m_Order = m_Op_Ext_Lor->GetDispersionOrder();
	int order = m_Op_Ext_Lor->m_Order;

	curr_Lor_ADE = new FDTD_FLOAT**[order];
	volt_Lor_ADE = new FDTD_FLOAT**[order];
	for (int o=0;o<order;++o)
	{
		curr_Lor_ADE[o] = new FDTD_FLOAT*[3];
		volt_Lor_ADE[o] = new FDTD_FLOAT*[3];
		for (int n=0; n<3; ++n)
		{
			if (m_Op_Ext_Lor->m_curr_Lor_ADE_On[o]==true)
			{
				curr_Lor_ADE[o][n] = new FDTD_FLOAT[m_Op_Ext_Lor->m_LM_Count[o]];
				for (unsigned int i=0; i<m_Op_Ext_Lor->m_LM_Count[o]; ++i)
					curr_Lor_ADE[o][n][i]=0.0;
			}
			else
				curr_Lor_ADE[o][n] = NULL;

			if (m_Op_Ext_Lor->m_volt_Lor_ADE_On[o]==true)
			{
				volt_Lor_ADE[o][n] = new FDTD_FLOAT[m_Op_Ext_Lor->m_LM_Count[o]];
				for (unsigned int i=0; i<m_Op_Ext_Lor->m_LM_Count[o]; ++i)
					volt_Lor_ADE[o][n][i]=0.0;
			}
			else
				volt_Lor_ADE[o][n] = NULL;
		}
	}
}

Engine_Ext_LorentzMaterial::~Engine_Ext_LorentzMaterial()
{
	if (curr_Lor_ADE==NULL && volt_Lor_ADE==NULL)
		return;

	for (int o=0;o<m_Op_Ext_Lor->m_Order;++o)
	{
		for (int n=0; n<3; ++n)
		{
			delete[] curr_Lor_ADE[o][n];
			delete[] volt_Lor_ADE[o][n];
		}
		delete[] curr_Lor_ADE[o];
		delete[] volt_Lor_ADE[o];
	}
	delete[] curr_Lor_ADE;
	curr_Lor_ADE=NULL;

	delete[] volt_Lor_ADE;
	volt_Lor_ADE=NULL;
}

template <typename EngType>
void Engine_Ext_LorentzMaterial::DoPreVoltageUpdatesImpl(EngType* eng, unsigned int startX, unsigned int stopX, int threadID)
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

	for (int o=0;o<m_Order;++o)
	{
		if (m_Op_Ext_Lor->m_volt_ADE_On[o]==false) continue;

		unsigned int **pos = m_Op_Ext_Lor->m_LM_pos[o];

		const unsigned int* perm;
		unsigned int j0, j1;
		if (!SlabEntries(o, xLo, xHi, threadID, perm, j0, j1)) continue;

#if OPENEMS_ENABLE_AVX2
		if constexpr (std::is_same<EngType, Engine_AVX2>::value)
		{
			// Division-free field access; see Engine_Ext_Dispersive::BuildAVX2Index.
			const unsigned int*  off  = m_avx2_off.data()  + m_avx2_start[o];
			const unsigned char* lane = m_avx2_lane.data() + m_avx2_start[o];
			f8vector* fld = eng->m_volt;
			if (m_Op_Ext_Lor->m_volt_Lor_ADE_On[o])
			{
				for (unsigned int j=j0; j<j1; ++j)
				{
					const unsigned int i = perm ? perm[j] : j;
					const f8vector* c = fld + off[i];
					const unsigned int l = lane[i];
					for (int n=0; n<3; ++n)
					{
						volt_Lor_ADE[o][n][i] += m_Op_Ext_Lor->v_Lor_ADE[o][n][i] * volt_ADE[o][n][i];
						volt_ADE[o][n][i] *= m_Op_Ext_Lor->v_int_ADE[o][n][i];
						volt_ADE[o][n][i] += m_Op_Ext_Lor->v_ext_ADE[o][n][i] * (c[n].f[l] - volt_Lor_ADE[o][n][i]);
					}
				}
			}
			else
			{
				for (unsigned int j=j0; j<j1; ++j)
				{
					const unsigned int i = perm ? perm[j] : j;
					const f8vector* c = fld + off[i];
					const unsigned int l = lane[i];
					for (int n=0; n<3; ++n)
					{
						volt_ADE[o][n][i] *= m_Op_Ext_Lor->v_int_ADE[o][n][i];
						volt_ADE[o][n][i] += m_Op_Ext_Lor->v_ext_ADE[o][n][i] * c[n].f[l];
					}
				}
			}
			continue;
		}
#endif

		if (m_Op_Ext_Lor->m_volt_Lor_ADE_On[o])
		{
			for (unsigned int j=j0; j<j1; ++j)
			{
				const unsigned int i = perm ? perm[j] : j;
				volt_Lor_ADE[o][0][i]+=m_Op_Ext_Lor->v_Lor_ADE[o][0][i]*volt_ADE[o][0][i];
				volt_ADE[o][0][i] *= m_Op_Ext_Lor->v_int_ADE[o][0][i];
				volt_ADE[o][0][i] += m_Op_Ext_Lor->v_ext_ADE[o][0][i] * (eng->EngType::GetVoltRaw(0,pos[0][i],pos[1][i],pos[2][i])-volt_Lor_ADE[o][0][i]);

				volt_Lor_ADE[o][1][i]+=m_Op_Ext_Lor->v_Lor_ADE[o][1][i]*volt_ADE[o][1][i];
				volt_ADE[o][1][i] *= m_Op_Ext_Lor->v_int_ADE[o][1][i];
				volt_ADE[o][1][i] += m_Op_Ext_Lor->v_ext_ADE[o][1][i] * (eng->EngType::GetVoltRaw(1,pos[0][i],pos[1][i],pos[2][i])-volt_Lor_ADE[o][1][i]);

				volt_Lor_ADE[o][2][i]+=m_Op_Ext_Lor->v_Lor_ADE[o][2][i]*volt_ADE[o][2][i];
				volt_ADE[o][2][i] *= m_Op_Ext_Lor->v_int_ADE[o][2][i];
				volt_ADE[o][2][i] += m_Op_Ext_Lor->v_ext_ADE[o][2][i] * (eng->EngType::GetVoltRaw(2,pos[0][i],pos[1][i],pos[2][i])-volt_Lor_ADE[o][2][i]);
			}
		}
		else
		{
			for (unsigned int j=j0; j<j1; ++j)
			{
				const unsigned int i = perm ? perm[j] : j;
				volt_ADE[o][0][i] *= m_Op_Ext_Lor->v_int_ADE[o][0][i];
				volt_ADE[o][0][i] += m_Op_Ext_Lor->v_ext_ADE[o][0][i] * eng->EngType::GetVoltRaw(0,pos[0][i],pos[1][i],pos[2][i]);

				volt_ADE[o][1][i] *= m_Op_Ext_Lor->v_int_ADE[o][1][i];
				volt_ADE[o][1][i] += m_Op_Ext_Lor->v_ext_ADE[o][1][i] * eng->EngType::GetVoltRaw(1,pos[0][i],pos[1][i],pos[2][i]);

				volt_ADE[o][2][i] *= m_Op_Ext_Lor->v_int_ADE[o][2][i];
				volt_ADE[o][2][i] += m_Op_Ext_Lor->v_ext_ADE[o][2][i] * eng->EngType::GetVoltRaw(2,pos[0][i],pos[1][i],pos[2][i]);
			}
		}
	}
}

void Engine_Ext_LorentzMaterial::DoPreVoltageUpdates()
{
	const unsigned int startX = 0, stopX = UINT_MAX;
	const int threadID = -1;
	ENG_DISPATCH_ARGS(DoPreVoltageUpdatesImpl, startX, stopX, threadID);
}

void Engine_Ext_LorentzMaterial::DoPreVoltageUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
{
	// numTS is not used: this is a per-cell recursion driven by the cell's own
	// voltage, with no term that reads the absolute timestep. It is still
	// timestep-correct under blocking because the schedule advances each cell
	// exactly once per timestep, in increasing timestep order, so each cell
	// sees the same sequence of voltages it sees under the flat sweep.
	(void)numTS;
	if (threadID < 0 || threadID >= m_NrThreads)
		return;
	BuildSlabIndex();
	ENG_DISPATCH_ARGS(DoPreVoltageUpdatesImpl, startX, stopX, threadID);
}

template <typename EngType>
void Engine_Ext_LorentzMaterial::DoPreCurrentUpdatesImpl(EngType* eng, unsigned int startX, unsigned int stopX, int threadID)
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

	for (int o=0;o<m_Order;++o)
	{
		if (m_Op_Ext_Lor->m_curr_ADE_On[o]==false) continue;

		unsigned int **pos = m_Op_Ext_Lor->m_LM_pos[o];

		const unsigned int* perm;
		unsigned int j0, j1;
		if (!SlabEntries(o, xLo, xHi, threadID, perm, j0, j1)) continue;

#if OPENEMS_ENABLE_AVX2
		if constexpr (std::is_same<EngType, Engine_AVX2>::value)
		{
			// Division-free field access; see Engine_Ext_Dispersive::BuildAVX2Index.
			const unsigned int*  off  = m_avx2_off.data()  + m_avx2_start[o];
			const unsigned char* lane = m_avx2_lane.data() + m_avx2_start[o];
			f8vector* fld = eng->m_curr;
			if (m_Op_Ext_Lor->m_curr_Lor_ADE_On[o])
			{
				for (unsigned int j=j0; j<j1; ++j)
				{
					const unsigned int i = perm ? perm[j] : j;
					const f8vector* c = fld + off[i];
					const unsigned int l = lane[i];
					for (int n=0; n<3; ++n)
					{
						curr_Lor_ADE[o][n][i] += m_Op_Ext_Lor->v_Lor_ADE[o][n][i] * curr_ADE[o][n][i];
						curr_ADE[o][n][i] *= m_Op_Ext_Lor->v_int_ADE[o][n][i];
						curr_ADE[o][n][i] += m_Op_Ext_Lor->v_ext_ADE[o][n][i] * (c[n].f[l] - curr_Lor_ADE[o][n][i]);
					}
				}
			}
			else
			{
				for (unsigned int j=j0; j<j1; ++j)
				{
					const unsigned int i = perm ? perm[j] : j;
					const f8vector* c = fld + off[i];
					const unsigned int l = lane[i];
					for (int n=0; n<3; ++n)
					{
						curr_ADE[o][n][i] *= m_Op_Ext_Lor->v_int_ADE[o][n][i];
						curr_ADE[o][n][i] += m_Op_Ext_Lor->v_ext_ADE[o][n][i] * c[n].f[l];
					}
				}
			}
			continue;
		}
#endif

		if (m_Op_Ext_Lor->m_curr_Lor_ADE_On[o])
		{
			for (unsigned int j=j0; j<j1; ++j)
			{
				const unsigned int i = perm ? perm[j] : j;
				curr_Lor_ADE[o][0][i]+=m_Op_Ext_Lor->i_Lor_ADE[o][0][i]*curr_ADE[o][0][i];
				curr_ADE[o][0][i] *= m_Op_Ext_Lor->i_int_ADE[o][0][i];
				curr_ADE[o][0][i] += m_Op_Ext_Lor->i_ext_ADE[o][0][i] * (eng->EngType::GetCurrRaw(0,pos[0][i],pos[1][i],pos[2][i])-curr_Lor_ADE[o][0][i]);

				curr_Lor_ADE[o][1][i]+=m_Op_Ext_Lor->i_Lor_ADE[o][1][i]*curr_ADE[o][1][i];
				curr_ADE[o][1][i] *= m_Op_Ext_Lor->i_int_ADE[o][1][i];
				curr_ADE[o][1][i] += m_Op_Ext_Lor->i_ext_ADE[o][1][i] * (eng->EngType::GetCurrRaw(1,pos[0][i],pos[1][i],pos[2][i])-curr_Lor_ADE[o][1][i]);

				curr_Lor_ADE[o][2][i]+=m_Op_Ext_Lor->i_Lor_ADE[o][2][i]*curr_ADE[o][2][i];
				curr_ADE[o][2][i] *= m_Op_Ext_Lor->i_int_ADE[o][2][i];
				curr_ADE[o][2][i] += m_Op_Ext_Lor->i_ext_ADE[o][2][i] * (eng->EngType::GetCurrRaw(2,pos[0][i],pos[1][i],pos[2][i])-curr_Lor_ADE[o][2][i]);
			}
		}
		else
		{
			for (unsigned int j=j0; j<j1; ++j)
			{
				const unsigned int i = perm ? perm[j] : j;
				curr_ADE[o][0][i] *= m_Op_Ext_Lor->i_int_ADE[o][0][i];
				curr_ADE[o][0][i] += m_Op_Ext_Lor->i_ext_ADE[o][0][i] * eng->EngType::GetCurrRaw(0,pos[0][i],pos[1][i],pos[2][i]);

				curr_ADE[o][1][i] *= m_Op_Ext_Lor->i_int_ADE[o][1][i];
				curr_ADE[o][1][i] += m_Op_Ext_Lor->i_ext_ADE[o][1][i] * eng->EngType::GetCurrRaw(1,pos[0][i],pos[1][i],pos[2][i]);

				curr_ADE[o][2][i] *= m_Op_Ext_Lor->i_int_ADE[o][2][i];
				curr_ADE[o][2][i] += m_Op_Ext_Lor->i_ext_ADE[o][2][i] * eng->EngType::GetCurrRaw(2,pos[0][i],pos[1][i],pos[2][i]);
			}
		}
	}
}

void Engine_Ext_LorentzMaterial::DoPreCurrentUpdates()
{
	const unsigned int startX = 0, stopX = UINT_MAX;
	const int threadID = -1;
	ENG_DISPATCH_ARGS(DoPreCurrentUpdatesImpl, startX, stopX, threadID);
}

//! \copydoc Engine_Ext_Dispersive::Apply2CurrentSlab
void Engine_Ext_LorentzMaterial::DoPreCurrentUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
{
	(void)numTS;
	if (threadID < 0 || threadID >= m_NrThreads)
		return;
	BuildSlabIndex();
	ENG_DISPATCH_ARGS(DoPreCurrentUpdatesImpl, startX, stopX, threadID);
}
