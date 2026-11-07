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

#include "engine_ext_upml.h"
#include "operator_ext_upml.h"
#include "FDTD/engine.h"
#include "FDTD/engine_sse.h"
#include "tools/useful.h"

Engine_Ext_UPML::Engine_Ext_UPML(Operator_Ext_UPML* op_ext) : Engine_Extension(op_ext)
{
	m_Op_UPML = op_ext;

	//this ABC extension should be executed first!
	m_Priority = ENG_EXT_PRIO_UPML;

#if OPENEMS_ENABLE_AVX2
	m_avx2_packed = false;
	m_v_numVectors = 0;
	m_v_pNy = 0;
	m_v_sx = m_v_sy = 0;
	m_v_vs_y = m_v_vs_x = 0;
	m_v_e_vs_y = m_v_e_vs_x = 0;
	m_v_vv = m_v_vvfo = m_v_vvfn = NULL;
	m_v_ii = m_v_iifo = m_v_iifn = NULL;
	m_v_volt_flux = m_v_curr_flux = NULL;
#endif

	volt_flux.Init("volt_flux", m_Op_UPML->m_numLines);
	curr_flux.Init("curr_flux", m_Op_UPML->m_numLines);

	SetNumberOfThreads(1);
}

Engine_Ext_UPML::~Engine_Ext_UPML()
{
#if OPENEMS_ENABLE_AVX2
	ReleaseAVX2Layout();
#endif
}

void Engine_Ext_UPML::SetEngine(Engine* eng)
{
	Engine_Extension::SetEngine(eng);
#if OPENEMS_ENABLE_AVX2
	// Repack into the engine's f8vector layout where the box allows it.
	// Failure is not fatal -- the scalar path stays valid.
	m_avx2_packed = BuildAVX2Layout();
#endif
}

void Engine_Ext_UPML::SetNumberOfThreads(int nrThread)
{
	Engine_Extension::SetNumberOfThreads(nrThread);

	m_numX = AssignJobs2Threads(m_Op_UPML->m_numLines[0],m_NrThreads,false);
	m_start.resize(m_NrThreads,0);
	m_start.at(0)=0;
	for (size_t n=1; n<m_numX.size(); ++n)
		m_start.at(n) = m_start.at(n-1) + m_numX.at(n-1);
}

// The flat path's split is the one SetNumberOfThreads() precomputed; the
// blocked path's is whatever is left of the box once it is clipped to the
// slab, which changes from call to call and so cannot be precomputed. Both
// answer in PML-local x, which is what the kernels iterate.

bool Engine_Ext_UPML::ThreadLocalX(int threadID, unsigned int& locStart, unsigned int& locStop) const
{
	if (threadID < 0 || threadID >= m_NrThreads)
		return false;
	locStart = m_start.at(threadID);
	locStop  = locStart + m_numX.at(threadID);
	return locStop > locStart;
}

bool Engine_Ext_UPML::SlabTouches(SlabHook hook, unsigned int startX, unsigned int stopX) const
{
	(void)hook;
	// Exactly when SlabShare() below finds a non-empty intersection.
	const unsigned int lo = m_Op_UPML->m_StartPos[0];
	const unsigned int hi = lo + m_Op_UPML->m_numLines[0];
	return (lo < stopX) && (startX < hi);
}

bool Engine_Ext_UPML::SlabLocalX(unsigned int startX, unsigned int stopX, int threadID,
                                 unsigned int& locStart, unsigned int& locStop) const
{
	const unsigned int lo = m_Op_UPML->m_StartPos[0];
	const unsigned int hi = lo + m_Op_UPML->m_numLines[0];
	unsigned int gStart, gStop;
	if (!SlabShare(lo, hi, startX, stopX, m_NrThreads, threadID, gStart, gStop))
		return false;
	locStart = gStart - lo;
	locStop  = gStop  - lo;
	return true;
}

template <typename EngType>
void Engine_Ext_UPML::DoPreVoltageUpdatesImpl(EngType* eng, unsigned int iStart, unsigned int iEnd)
{
	if (m_Eng==NULL)
		return;

	unsigned int pos[3];
	unsigned int loc_pos[3];
	FDTD_FLOAT f_help;

	for (loc_pos[0]=iStart; loc_pos[0]<iEnd; ++loc_pos[0])
	{
		pos[0] = loc_pos[0] + m_Op_UPML->m_StartPos[0];
		for (loc_pos[1]=0; loc_pos[1]<m_Op_UPML->m_numLines[1]; ++loc_pos[1])
		{
			pos[1] = loc_pos[1] + m_Op_UPML->m_StartPos[1];
			for (loc_pos[2]=0; loc_pos[2]<m_Op_UPML->m_numLines[2]; ++loc_pos[2])
			{
				pos[2] = loc_pos[2] + m_Op_UPML->m_StartPos[2];

				f_help = m_Op_UPML->vv(0, loc_pos[0], loc_pos[1], loc_pos[2])   * eng->EngType::GetVoltRaw(0,pos)
						 - m_Op_UPML->vvfo(0, loc_pos[0], loc_pos[1], loc_pos[2]) * volt_flux(0, loc_pos[0], loc_pos[1], loc_pos[2]);
				eng->EngType::SetVolt(0,pos, volt_flux(0, loc_pos[0], loc_pos[1], loc_pos[2]));
				volt_flux(0, loc_pos[0], loc_pos[1], loc_pos[2]) = f_help;

				f_help = m_Op_UPML->vv(1, loc_pos[0], loc_pos[1], loc_pos[2])   * eng->EngType::GetVoltRaw(1,pos)
						 - m_Op_UPML->vvfo(1, loc_pos[0], loc_pos[1], loc_pos[2]) * volt_flux(1, loc_pos[0], loc_pos[1], loc_pos[2]);
				eng->EngType::SetVolt(1,pos, volt_flux(1, loc_pos[0], loc_pos[1], loc_pos[2]));
				volt_flux(1, loc_pos[0], loc_pos[1], loc_pos[2]) = f_help;

				f_help = m_Op_UPML->vv(2, loc_pos[0], loc_pos[1], loc_pos[2])   * eng->EngType::GetVoltRaw(2,pos)
						 - m_Op_UPML->vvfo(2, loc_pos[0], loc_pos[1], loc_pos[2]) * volt_flux(2, loc_pos[0], loc_pos[1], loc_pos[2]);
				eng->EngType::SetVolt(2,pos, volt_flux(2, loc_pos[0], loc_pos[1], loc_pos[2]));
				volt_flux(2, loc_pos[0], loc_pos[1], loc_pos[2]) = f_help;
			}
		}
	}
}

void Engine_Ext_UPML::DoPreVoltageUpdates(int threadID)
{
	unsigned int iStart, iEnd;
	if (!ThreadLocalX(threadID, iStart, iEnd))
		return;
#if OPENEMS_ENABLE_AVX2
	if (m_avx2_packed)
	{
		DoPreVoltageUpdatesAVX2(iStart, iEnd);
		return;
	}
#endif
	ENG_DISPATCH_ARGS(DoPreVoltageUpdatesImpl, iStart, iEnd);
}

void Engine_Ext_UPML::DoPreVoltageUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
{
	(void)numTS; // the UPML update is time-invariant; only the range matters
	unsigned int iStart, iEnd;
	if (!SlabLocalX(startX, stopX, threadID, iStart, iEnd))
		return;
#if OPENEMS_ENABLE_AVX2
	if (m_avx2_packed)
	{
		DoPreVoltageUpdatesAVX2(iStart, iEnd);
		return;
	}
#endif
	ENG_DISPATCH_ARGS(DoPreVoltageUpdatesImpl, iStart, iEnd);
}

template <typename EngType>
void Engine_Ext_UPML::DoPostVoltageUpdatesImpl(EngType* eng, unsigned int iStart, unsigned int iEnd)
{
	if (m_Eng==NULL)
		return;

	unsigned int pos[3];
	unsigned int loc_pos[3];
	FDTD_FLOAT f_help;

	for (loc_pos[0]=iStart; loc_pos[0]<iEnd; ++loc_pos[0])
	{
		pos[0] = loc_pos[0] + m_Op_UPML->m_StartPos[0];
		for (loc_pos[1]=0; loc_pos[1]<m_Op_UPML->m_numLines[1]; ++loc_pos[1])
		{
			pos[1] = loc_pos[1] + m_Op_UPML->m_StartPos[1];
			for (loc_pos[2]=0; loc_pos[2]<m_Op_UPML->m_numLines[2]; ++loc_pos[2])
			{
				pos[2] = loc_pos[2] + m_Op_UPML->m_StartPos[2];

				f_help = volt_flux(0, loc_pos[0], loc_pos[1], loc_pos[2]);
				volt_flux(0, loc_pos[0], loc_pos[1], loc_pos[2]) = eng->EngType::GetVoltRaw(0,pos);
				eng->EngType::SetVolt(0,pos, f_help + m_Op_UPML->vvfn(0, loc_pos[0], loc_pos[1], loc_pos[2]) * volt_flux(0, loc_pos[0], loc_pos[1], loc_pos[2]));

				f_help = volt_flux(1, loc_pos[0], loc_pos[1], loc_pos[2]);
				volt_flux(1, loc_pos[0], loc_pos[1], loc_pos[2]) = eng->EngType::GetVoltRaw(1,pos);
				eng->EngType::SetVolt(1,pos, f_help + m_Op_UPML->vvfn(1, loc_pos[0], loc_pos[1], loc_pos[2]) * volt_flux(1, loc_pos[0], loc_pos[1], loc_pos[2]));

				f_help = volt_flux(2, loc_pos[0], loc_pos[1], loc_pos[2]);
				volt_flux(2, loc_pos[0], loc_pos[1], loc_pos[2]) = eng->EngType::GetVoltRaw(2,pos);
				eng->EngType::SetVolt(2,pos, f_help + m_Op_UPML->vvfn(2, loc_pos[0], loc_pos[1], loc_pos[2]) * volt_flux(2, loc_pos[0], loc_pos[1], loc_pos[2]));
			}
		}
	}
}

void Engine_Ext_UPML::DoPostVoltageUpdates(int threadID)
{
	unsigned int iStart, iEnd;
	if (!ThreadLocalX(threadID, iStart, iEnd))
		return;
#if OPENEMS_ENABLE_AVX2
	if (m_avx2_packed)
	{
		DoPostVoltageUpdatesAVX2(iStart, iEnd);
		return;
	}
#endif
	ENG_DISPATCH_ARGS(DoPostVoltageUpdatesImpl, iStart, iEnd);
}

void Engine_Ext_UPML::DoPostVoltageUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
{
	(void)numTS; // the UPML update is time-invariant; only the range matters
	unsigned int iStart, iEnd;
	if (!SlabLocalX(startX, stopX, threadID, iStart, iEnd))
		return;
#if OPENEMS_ENABLE_AVX2
	if (m_avx2_packed)
	{
		DoPostVoltageUpdatesAVX2(iStart, iEnd);
		return;
	}
#endif
	ENG_DISPATCH_ARGS(DoPostVoltageUpdatesImpl, iStart, iEnd);
}

template <typename EngType>
void Engine_Ext_UPML::DoPreCurrentUpdatesImpl(EngType* eng, unsigned int iStart, unsigned int iEnd)
{
	if (m_Eng==NULL)
		return;

	unsigned int pos[3];
	unsigned int loc_pos[3];
	FDTD_FLOAT f_help;

	for (loc_pos[0]=iStart; loc_pos[0]<iEnd; ++loc_pos[0])
	{
		pos[0] = loc_pos[0] + m_Op_UPML->m_StartPos[0];
		for (loc_pos[1]=0; loc_pos[1]<m_Op_UPML->m_numLines[1]; ++loc_pos[1])
		{
			pos[1] = loc_pos[1] + m_Op_UPML->m_StartPos[1];
			for (loc_pos[2]=0; loc_pos[2]<m_Op_UPML->m_numLines[2]; ++loc_pos[2])
			{
				pos[2] = loc_pos[2] + m_Op_UPML->m_StartPos[2];

				f_help = m_Op_UPML->ii(0, loc_pos[0], loc_pos[1], loc_pos[2])   * eng->EngType::GetCurrRaw(0,pos)
						 - m_Op_UPML->iifo(0, loc_pos[0], loc_pos[1], loc_pos[2]) * curr_flux(0, loc_pos[0], loc_pos[1], loc_pos[2]);
				eng->EngType::SetCurr(0,pos, curr_flux(0, loc_pos[0], loc_pos[1], loc_pos[2]));
				curr_flux(0, loc_pos[0], loc_pos[1], loc_pos[2]) = f_help;

				f_help = m_Op_UPML->ii(1, loc_pos[0], loc_pos[1], loc_pos[2])   * eng->EngType::GetCurrRaw(1,pos)
						 - m_Op_UPML->iifo(1, loc_pos[0], loc_pos[1], loc_pos[2]) * curr_flux(1, loc_pos[0], loc_pos[1], loc_pos[2]);
				eng->EngType::SetCurr(1,pos, curr_flux(1, loc_pos[0], loc_pos[1], loc_pos[2]));
				curr_flux(1, loc_pos[0], loc_pos[1], loc_pos[2]) = f_help;

				f_help = m_Op_UPML->ii(2, loc_pos[0], loc_pos[1], loc_pos[2])   * eng->EngType::GetCurrRaw(2,pos)
						 - m_Op_UPML->iifo(2, loc_pos[0], loc_pos[1], loc_pos[2]) * curr_flux(2, loc_pos[0], loc_pos[1], loc_pos[2]);
				eng->EngType::SetCurr(2,pos, curr_flux(2, loc_pos[0], loc_pos[1], loc_pos[2]));
				curr_flux(2, loc_pos[0], loc_pos[1], loc_pos[2]) = f_help;

			}
		}
	}
}

void Engine_Ext_UPML::DoPreCurrentUpdates(int threadID)
{
	unsigned int iStart, iEnd;
	if (!ThreadLocalX(threadID, iStart, iEnd))
		return;
#if OPENEMS_ENABLE_AVX2
	if (m_avx2_packed)
	{
		DoPreCurrentUpdatesAVX2(iStart, iEnd);
		return;
	}
#endif
	ENG_DISPATCH_ARGS(DoPreCurrentUpdatesImpl, iStart, iEnd);
}

void Engine_Ext_UPML::DoPreCurrentUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
{
	(void)numTS; // the UPML update is time-invariant; only the range matters
	unsigned int iStart, iEnd;
	if (!SlabLocalX(startX, stopX, threadID, iStart, iEnd))
		return;
#if OPENEMS_ENABLE_AVX2
	if (m_avx2_packed)
	{
		DoPreCurrentUpdatesAVX2(iStart, iEnd);
		return;
	}
#endif
	ENG_DISPATCH_ARGS(DoPreCurrentUpdatesImpl, iStart, iEnd);
}

template <typename EngType>
void Engine_Ext_UPML::DoPostCurrentUpdatesImpl(EngType* eng, unsigned int iStart, unsigned int iEnd)
{
	if (m_Eng==NULL)
		return;

	unsigned int pos[3];
	unsigned int loc_pos[3];
	FDTD_FLOAT f_help;

	for (loc_pos[0]=iStart; loc_pos[0]<iEnd; ++loc_pos[0])
	{
		pos[0] = loc_pos[0] + m_Op_UPML->m_StartPos[0];
		for (loc_pos[1]=0; loc_pos[1]<m_Op_UPML->m_numLines[1]; ++loc_pos[1])
		{
			pos[1] = loc_pos[1] + m_Op_UPML->m_StartPos[1];
			for (loc_pos[2]=0; loc_pos[2]<m_Op_UPML->m_numLines[2]; ++loc_pos[2])
			{
				pos[2] = loc_pos[2] + m_Op_UPML->m_StartPos[2];

				f_help = curr_flux(0, loc_pos[0], loc_pos[1], loc_pos[2]);
				curr_flux(0, loc_pos[0], loc_pos[1], loc_pos[2]) = eng->EngType::GetCurrRaw(0,pos);
				eng->EngType::SetCurr(0,pos, f_help + m_Op_UPML->iifn(0, loc_pos[0], loc_pos[1], loc_pos[2]) * curr_flux(0, loc_pos[0], loc_pos[1], loc_pos[2]));

				f_help = curr_flux(1, loc_pos[0], loc_pos[1], loc_pos[2]);
				curr_flux(1, loc_pos[0], loc_pos[1], loc_pos[2]) = eng->EngType::GetCurrRaw(1,pos);
				eng->EngType::SetCurr(1,pos, f_help + m_Op_UPML->iifn(1, loc_pos[0], loc_pos[1], loc_pos[2]) * curr_flux(1, loc_pos[0], loc_pos[1], loc_pos[2]));

				f_help = curr_flux(2, loc_pos[0], loc_pos[1], loc_pos[2]);
				curr_flux(2, loc_pos[0], loc_pos[1], loc_pos[2]) = eng->EngType::GetCurrRaw(2,pos);
				eng->EngType::SetCurr(2,pos, f_help + m_Op_UPML->iifn(2, loc_pos[0], loc_pos[1], loc_pos[2]) * curr_flux(2, loc_pos[0], loc_pos[1], loc_pos[2]));
			}
		}
	}
}

void Engine_Ext_UPML::DoPostCurrentUpdates(int threadID)
{
	unsigned int iStart, iEnd;
	if (!ThreadLocalX(threadID, iStart, iEnd))
		return;
#if OPENEMS_ENABLE_AVX2
	if (m_avx2_packed)
	{
		DoPostCurrentUpdatesAVX2(iStart, iEnd);
		return;
	}
#endif
	ENG_DISPATCH_ARGS(DoPostCurrentUpdatesImpl, iStart, iEnd);
}

void Engine_Ext_UPML::DoPostCurrentUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
{
	(void)numTS; // the UPML update is time-invariant; only the range matters
	unsigned int iStart, iEnd;
	if (!SlabLocalX(startX, stopX, threadID, iStart, iEnd))
		return;
#if OPENEMS_ENABLE_AVX2
	if (m_avx2_packed)
	{
		DoPostCurrentUpdatesAVX2(iStart, iEnd);
		return;
	}
#endif
	ENG_DISPATCH_ARGS(DoPostCurrentUpdatesImpl, iStart, iEnd);
}
