/*
*	Additional
*	Copyright (C) 2023 Gadi Lahav (gadi@rfwithcare.com)
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

#include "engine_ext_lumpedRLC.h"
#include <climits>
#include "operator_ext_lumpedRLC.h"

#include "FDTD/engine_sse.h"

Engine_Ext_LumpedRLC::Engine_Ext_LumpedRLC(Operator_Ext_LumpedRLC* op_ext_RLC) : Engine_Extension(op_ext_RLC)
{
	// Local pointer of the operator.
	m_Op_Ext_RLC = op_ext_RLC;

	v_Vdn		= new FDTD_FLOAT*[3];
	v_Jn		= new FDTD_FLOAT*[3];
	v_Il		= NULL;

	m_xLo = m_xHi = 0;
	for (unsigned int posIdx = 0 ; posIdx < m_Op_Ext_RLC->RLC_count ; ++posIdx)
	{
		const unsigned int x = m_Op_Ext_RLC->v_RLC_pos[0][posIdx];
		if (posIdx==0 || x<m_xLo) m_xLo = x;
		if (posIdx==0 || x+1>m_xHi) m_xHi = x+1;
	}

	// No additional allocations are required if there are no actual lumped elements.
	if (!(m_Op_Ext_RLC->RLC_count))
		return;

	// Initialize ADE containers for currents and voltages
	v_Il		= new FDTD_FLOAT[m_Op_Ext_RLC->RLC_count];

	for (unsigned int posIdx = 0 ; posIdx < m_Op_Ext_RLC->RLC_count ; ++posIdx)
		v_Il[posIdx] 	= 0.0;

	for (unsigned int k = 0 ; k < 3 ; k++)
	{
		v_Vdn[k] = new FDTD_FLOAT[m_Op_Ext_RLC->RLC_count];
		v_Jn[k] = new FDTD_FLOAT[m_Op_Ext_RLC->RLC_count];

		for (unsigned int posIdx = 0 ; posIdx < m_Op_Ext_RLC->RLC_count ; ++posIdx)
		{
			v_Jn[k][posIdx] = 0.0;
			v_Vdn[k][posIdx] = 0.0;
		}
	}

}

Engine_Ext_LumpedRLC::~Engine_Ext_LumpedRLC()
{
	// Only delete if values were allocated in the first place
	if (m_Op_Ext_RLC->RLC_count)
	{
		delete[] v_Il;

		for (unsigned int k = 0 ; k < 3 ; k++)
		{
			delete[] v_Vdn[k];
			delete[] v_Jn[k];
		}
	}

	delete[] v_Vdn;
	delete[] v_Jn;

	v_Il	= NULL;

	v_Vdn	= NULL;
	v_Jn	= NULL;

	m_Op_Ext_RLC = NULL;


}

// startX/stopX bound the x-range of elements to touch and numTS is the timestep
// being computed. The whole-grid entry points pass [0,UINT_MAX) and the
// engine's own timestep, so there is one implementation, not two.
void Engine_Ext_LumpedRLC::DoPreVoltageUpdatesImpl(int numTS, unsigned int startX, unsigned int stopX)
{
	if (!m_Op_Ext_RLC->RLC_count)
		return;

	unsigned int **pos = m_Op_Ext_RLC->v_RLC_pos;
	const FDTD_FLOAT* Vd_1 = v_Vdn[Slot(numTS, 1)];

	// In pre-process, only update the parallel inductor current:
	for (unsigned int pIdx = 0 ; pIdx < m_Op_Ext_RLC->RLC_count ; pIdx++)
	{
		if (pos[0][pIdx] < startX || pos[0][pIdx] >= stopX)
			continue;
		v_Il[pIdx] += (m_Op_Ext_RLC->v_RLC_i2v[pIdx])*(m_Op_Ext_RLC->v_RLC_ilv[pIdx])*Vd_1[pIdx];
	}
}

void Engine_Ext_LumpedRLC::DoPreVoltageUpdates()
{
	DoPreVoltageUpdatesImpl(m_Eng->GetNumberOfTimesteps(), 0, UINT_MAX);
}

// Lumped elements number in the tens of cells, so the slab hooks run on thread
// 0 and let the engine's barrier serialise them rather than striping a list
// that short. That also sidesteps the question of whether two elements can
// share a cell, which the flat path answers by visiting the list in order.
void Engine_Ext_LumpedRLC::DoPreVoltageUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
{
	if (threadID != 0)
		return;
	DoPreVoltageUpdatesImpl(numTS, startX, stopX);
}

template <typename EngType>
void Engine_Ext_LumpedRLC::Apply2VoltagesImpl(EngType* eng, int numTS, unsigned int startX, unsigned int stopX)
{
	if (!m_Op_Ext_RLC->RLC_count)
		return;

	unsigned int **pos = m_Op_Ext_RLC->v_RLC_pos;
	int *dir = m_Op_Ext_RLC->v_RLC_dir;

	FDTD_FLOAT* Vd_0 = v_Vdn[Slot(numTS, 0)];
	FDTD_FLOAT* Vd_2 = v_Vdn[Slot(numTS, 2)];
	FDTD_FLOAT* J_0  = v_Jn [Slot(numTS, 0)];
	FDTD_FLOAT* J_1  = v_Jn [Slot(numTS, 1)];
	FDTD_FLOAT* J_2  = v_Jn [Slot(numTS, 2)];

	// Read engine calculated node voltage
	for (unsigned int pIdx = 0 ; pIdx < m_Op_Ext_RLC->RLC_count ; pIdx++)
	{
		if (pos[0][pIdx] < startX || pos[0][pIdx] >= stopX)
			continue;
		Vd_0[pIdx] = eng->EngType::GetVoltRaw(dir[pIdx],pos[0][pIdx],pos[1][pIdx],pos[2][pIdx]);
	}

	// Post process: Calculate node voltage with respect to the lumped RLC auxilliary quantity, J
	for (unsigned int pIdx = 0 ; pIdx < m_Op_Ext_RLC->RLC_count ; pIdx++)
	{
		if (pos[0][pIdx] < startX || pos[0][pIdx] >= stopX)
			continue;

		// Calculate updated node voltage, with series and parallel additions
		Vd_0[pIdx] = 		(m_Op_Ext_RLC->v_RLC_vvd[pIdx])*(
							Vd_0[pIdx] - v_Il[pIdx]							// Addition for Parallel inductor
							+
							(m_Op_Ext_RLC->v_RLC_vv2[pIdx])*Vd_2[pIdx]		// Vd[n-2] addition
							+
							(m_Op_Ext_RLC->v_RLC_vj1[pIdx])*J_1[pIdx]		// J[n-1] addition
							+
							(m_Op_Ext_RLC->v_RLC_vj2[pIdx])*J_2[pIdx]);		// J[n-2] addition

		// Update J[0]
		J_0[pIdx] =		(m_Op_Ext_RLC->v_RLC_ib0[pIdx])*(Vd_0[pIdx] - Vd_2[pIdx])
						-
						((m_Op_Ext_RLC->v_RLC_b1[pIdx])*(m_Op_Ext_RLC->v_RLC_ib0[pIdx]))*J_1[pIdx]
						-
						((m_Op_Ext_RLC->v_RLC_b2[pIdx])*(m_Op_Ext_RLC->v_RLC_ib0[pIdx]))*J_2[pIdx];
	}


	// Update node voltage
	for (unsigned int pIdx = 0 ; pIdx < m_Op_Ext_RLC->RLC_count ; pIdx++)
	{
		if (pos[0][pIdx] < startX || pos[0][pIdx] >= stopX)
			continue;
		eng->EngType::SetVolt(dir[pIdx],pos[0][pIdx],pos[1][pIdx],pos[2][pIdx],Vd_0[pIdx]);
	}
}

void Engine_Ext_LumpedRLC::Apply2Voltages()
{
	const int numTS = m_Eng->GetNumberOfTimesteps();
	const unsigned int startX = 0, stopX = UINT_MAX;
	ENG_DISPATCH_ARGS(Apply2VoltagesImpl, numTS, startX, stopX);
}

void Engine_Ext_LumpedRLC::Apply2VoltagesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
{
	if (threadID != 0)
		return;
	ENG_DISPATCH_ARGS(Apply2VoltagesImpl, numTS, startX, stopX);
}
