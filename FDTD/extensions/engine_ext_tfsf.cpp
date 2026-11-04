/*
*	Copyright (C) 2012 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#include "engine_ext_tfsf.h"
#include "operator_ext_tfsf.h"
#include "FDTD/engine_sse.h"
#include <climits>
#include <vector>

namespace {

// Value of the shared delay-lookup formula at table index \a n for a signal
// evaluated at \a numTS. Both hooks need this -- they just fill different
// index ranges of the table (see the two Impl functions below) -- so pulling
// it out is what lets them agree bit-for-bit on any index they both touch.
unsigned int DelayLookupEntry(unsigned int n, unsigned int numTS, unsigned int length, int p)
{
	unsigned int val;
	if (numTS < n)
		val = 0;
	else if ((numTS-n >= length) && (p==0))
		val = 0;
	else
		val = numTS - n;
	if (p>0)
		val = val % p;
	return val;
}

// Restrict idx in [0,count) so that base+idx lies in [startX,stopX). This is
// the correctness half of pattern C -- it runs identically on every thread,
// because it decides *which cells exist*, not who owns them. Unsigned-safe:
// startX/stopX may be UINT_MAX (the whole-grid entry points) or sit anywhere
// relative to base without risking wraparound.
void ClampAxisToX(unsigned int base, unsigned int count,
                   unsigned int startX, unsigned int stopX,
                   unsigned int& lo, unsigned int& hi)
{
	lo = (startX > base) ? (startX - base) : 0;
	hi = (stopX  > base) ? (stopX  - base) : 0;
	if (lo > count) lo = count;
	if (hi > count) hi = count;
	if (hi < lo) hi = lo;
}

} // namespace

Engine_Ext_TFSF::Engine_Ext_TFSF(Operator_Ext_TFSF* op_ext) : Engine_Extension(op_ext)
{
	m_Op_TFSF = op_ext;
	m_Priority = ENG_EXT_PRIO_TFSF;
}

Engine_Ext_TFSF::~Engine_Ext_TFSF()
{
}

// One face (direction n, side lowHigh) of the box. pos[n] is fixed at posN for
// the whole face; pos[nP] and pos[nPP] sweep m_Start[nP]/m_Start[nPP] upward
// over i/j respectively, i outer, j inner -- exactly as the original nested
// loops did, which is what ui_pos = i*numJ + j below has to reproduce: that is
// the flattened index the original code built by resetting ui_pos to 0 before
// the i-loop and incrementing it once per innermost iteration, so it counts
// (i,j) pairs in row-major order regardless of which of i/j is skipped here.
//
// Exactly one of n, nP, nPP is the global x axis (index 0). n==0 means the
// whole face sits at a single x (pattern B: keep or drop the whole thing,
// then split the free (i,j) space across threads); otherwise x rides along
// whichever of nP/nPP equals 0 (pattern C: clamp that axis to [startX,stopX)
// identically on every thread, split the other one across threadID).
void Engine_Ext_TFSF::VoltageFace(int n, int nP, int nPP, int lowHigh, unsigned int posN,
                                  unsigned int startX, unsigned int stopX, int nThreads, int threadID,
                                  const unsigned int* delay, FDTD_FLOAT* signal)
{
	const unsigned int numI = m_Op_TFSF->m_numLines[nP];
	const unsigned int numJ = m_Op_TFSF->m_numLines[nPP];
	const unsigned int baseI = m_Op_TFSF->m_Start[nP];
	const unsigned int baseJ = m_Op_TFSF->m_Start[nPP];

	unsigned int iLo = 0, iHi = numI;
	unsigned int jLo = 0, jHi = numJ;

	if (n == 0)
	{
		if (posN < startX || posN >= stopX)
			iLo = iHi = 0;
		else if (!SlabShare(0, numI, 0, UINT_MAX, nThreads, threadID, iLo, iHi))
			iLo = iHi = 0;
	}
	else if (nPP == 0)
	{
		ClampAxisToX(baseJ, numJ, startX, stopX, jLo, jHi);
		if (!SlabShare(0, numI, 0, UINT_MAX, nThreads, threadID, iLo, iHi))
			iLo = iHi = 0;
	}
	else // nP == 0
	{
		ClampAxisToX(baseI, numI, startX, stopX, iLo, iHi);
		if (!SlabShare(0, numJ, 0, UINT_MAX, nThreads, threadID, jLo, jHi))
			jLo = jHi = 0;
	}

	unsigned int pos[3];
	pos[n] = posN;
	for (unsigned int i=iLo; i<iHi; ++i)
	{
		pos[nP] = baseI + i;
		for (unsigned int j=jLo; j<jHi; ++j)
		{
			pos[nPP] = baseJ + j;
			const unsigned int ui_pos = i*numJ + j;

			m_Eng->SetVolt(nP,pos, m_Eng->GetVolt(nP,pos)
						   + (1.0-m_Op_TFSF->m_VoltDelayDelta[n][lowHigh][0][ui_pos])*m_Op_TFSF->m_VoltAmp[n][lowHigh][0][ui_pos]*signal[delay[  m_Op_TFSF->m_VoltDelay[n][lowHigh][0][ui_pos]]]
						   +      m_Op_TFSF->m_VoltDelayDelta[n][lowHigh][0][ui_pos] *m_Op_TFSF->m_VoltAmp[n][lowHigh][0][ui_pos]*signal[delay[1+m_Op_TFSF->m_VoltDelay[n][lowHigh][0][ui_pos]]] );

			m_Eng->SetVolt(nPP,pos, m_Eng->GetVolt(nPP,pos)
						   + (1.0-m_Op_TFSF->m_VoltDelayDelta[n][lowHigh][1][ui_pos])*m_Op_TFSF->m_VoltAmp[n][lowHigh][1][ui_pos]*signal[delay[  m_Op_TFSF->m_VoltDelay[n][lowHigh][1][ui_pos]]]
						   +      m_Op_TFSF->m_VoltDelayDelta[n][lowHigh][1][ui_pos] *m_Op_TFSF->m_VoltAmp[n][lowHigh][1][ui_pos]*signal[delay[1+m_Op_TFSF->m_VoltDelay[n][lowHigh][1][ui_pos]]] );
		}
	}
}

void Engine_Ext_TFSF::DoPostVoltageUpdatesImpl(unsigned int startX, unsigned int stopX, int numTS_in, int nThreads, int threadID)
{
	if (threadID < 0 || threadID >= nThreads)
		return;

	const unsigned int numTS = (unsigned int)numTS_in;
	const unsigned int length = m_Op_TFSF->m_Exc->GetLength();

	int p = (int)m_Op_TFSF->m_Exc->GetSignalPeriodTS();

	// Private to this call -- under blocking this hook runs concurrently on
	// every worker thread with a numTS that is not the engine's own counter
	// (that only advances at block boundaries), so a table shared across
	// calls the way m_DelayLookup used to be would both race between threads
	// and, even single-threaded, read the wrong timestep inside a block.
	std::vector<unsigned int> delay(m_Op_TFSF->m_maxDelay+1);
	for (unsigned int n=0;n<=m_Op_TFSF->m_maxDelay;++n)
		delay[n] = DelayLookupEntry(n, numTS, length, p);

	//get the current signal since an H-field is added ...
	FDTD_FLOAT* signal =  m_Op_TFSF->m_Exc->GetCurrentSignal();

	int nP,nPP;
	for (int n=0;n<3;++n)
	{
		nP = (n+1)%3;
		nPP = (n+2)%3;

		if (m_Op_TFSF->m_ActiveDir[n][0])
			VoltageFace(n, nP, nPP, 0, m_Op_TFSF->m_Start[n], startX, stopX, nThreads, threadID, delay.data(), signal);

		if (m_Op_TFSF->m_ActiveDir[n][1])
			VoltageFace(n, nP, nPP, 1, m_Op_TFSF->m_Stop[n], startX, stopX, nThreads, threadID, delay.data(), signal);
	}
}

void Engine_Ext_TFSF::DoPostVoltageUpdates()
{
	DoPostVoltageUpdatesImpl(0, UINT_MAX, (int)m_Eng->GetNumberOfTimesteps(), 1, 0);
}

// Thread 0 does the whole slab, which is parity with the flat path rather than
// a concession: Engine_Ext_TFSF never overrode the threadID-taking hooks, so
// Engine_Extension's default has always routed it to thread 0 alone.
//
// It also cannot be split the way the volume extensions are. Adjacent faces of
// the box share their edge cells, and each face adds into them, so a cell on an
// edge is written once per face. Any partition that sends two faces to two
// threads has them read-modify-writing that cell concurrently -- a lost update,
// and even with an atomic the summation order would vary and the result would
// stop being bit-identical. Partitioning by *cell* rather than by face would
// work, but the surface is O(N^2) against the sweep's O(N^3) and this is the
// path the flat sweep already takes, so there is nothing to win.
void Engine_Ext_TFSF::DoPostVoltageUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
{
	if (threadID != 0)
		return;
	DoPostVoltageUpdatesImpl(startX, stopX, numTS, 1, 0);
}

// Mirrors VoltageFace -- see its comment for the ui_pos/axis reasoning, which
// is identical here. The only per-face difference is which arrays/accessors
// are used, and that current's lower-side posN is m_Start[n]-1, not m_Start[n]
// (H sits half a cell off E on the Yee grid); that offset is resolved by the
// caller, same as it always was.
void Engine_Ext_TFSF::CurrentFace(int n, int nP, int nPP, int lowHigh, unsigned int posN,
                                  unsigned int startX, unsigned int stopX, int nThreads, int threadID,
                                  const unsigned int* delay, FDTD_FLOAT* signal)
{
	const unsigned int numI = m_Op_TFSF->m_numLines[nP];
	const unsigned int numJ = m_Op_TFSF->m_numLines[nPP];
	const unsigned int baseI = m_Op_TFSF->m_Start[nP];
	const unsigned int baseJ = m_Op_TFSF->m_Start[nPP];

	unsigned int iLo = 0, iHi = numI;
	unsigned int jLo = 0, jHi = numJ;

	if (n == 0)
	{
		if (posN < startX || posN >= stopX)
			iLo = iHi = 0;
		else if (!SlabShare(0, numI, 0, UINT_MAX, nThreads, threadID, iLo, iHi))
			iLo = iHi = 0;
	}
	else if (nPP == 0)
	{
		ClampAxisToX(baseJ, numJ, startX, stopX, jLo, jHi);
		if (!SlabShare(0, numI, 0, UINT_MAX, nThreads, threadID, iLo, iHi))
			iLo = iHi = 0;
	}
	else // nP == 0
	{
		ClampAxisToX(baseI, numI, startX, stopX, iLo, iHi);
		if (!SlabShare(0, numJ, 0, UINT_MAX, nThreads, threadID, jLo, jHi))
			jLo = jHi = 0;
	}

	unsigned int pos[3];
	pos[n] = posN;
	for (unsigned int i=iLo; i<iHi; ++i)
	{
		pos[nP] = baseI + i;
		for (unsigned int j=jLo; j<jHi; ++j)
		{
			pos[nPP] = baseJ + j;
			const unsigned int ui_pos = i*numJ + j;

			m_Eng->SetCurr(nP,pos, m_Eng->GetCurr(nP,pos)
						   + (1.0-m_Op_TFSF->m_CurrDelayDelta[n][lowHigh][0][ui_pos])*m_Op_TFSF->m_CurrAmp[n][lowHigh][0][ui_pos]*signal[delay[  m_Op_TFSF->m_CurrDelay[n][lowHigh][0][ui_pos]]]
						   +      m_Op_TFSF->m_CurrDelayDelta[n][lowHigh][0][ui_pos] *m_Op_TFSF->m_CurrAmp[n][lowHigh][0][ui_pos]*signal[delay[1+m_Op_TFSF->m_CurrDelay[n][lowHigh][0][ui_pos]]] );

			m_Eng->SetCurr(nPP,pos, m_Eng->GetCurr(nPP,pos)
						   + (1.0-m_Op_TFSF->m_CurrDelayDelta[n][lowHigh][1][ui_pos])*m_Op_TFSF->m_CurrAmp[n][lowHigh][1][ui_pos]*signal[delay[  m_Op_TFSF->m_CurrDelay[n][lowHigh][1][ui_pos]]]
						   +      m_Op_TFSF->m_CurrDelayDelta[n][lowHigh][1][ui_pos] *m_Op_TFSF->m_CurrAmp[n][lowHigh][1][ui_pos]*signal[delay[1+m_Op_TFSF->m_CurrDelay[n][lowHigh][1][ui_pos]]] );
		}
	}
}

void Engine_Ext_TFSF::DoPostCurrentUpdatesImpl(unsigned int startX, unsigned int stopX, int numTS_in, int nThreads, int threadID)
{
	if (threadID < 0 || threadID >= nThreads)
		return;

	const unsigned int numTS = (unsigned int)numTS_in;
	const unsigned int length = m_Op_TFSF->m_Exc->GetLength();

	int p = (int)m_Op_TFSF->m_Exc->GetSignalPeriodTS();

	// Same private-to-this-call table as the voltage hook, but filled one entry
	// wider than this hook used to fill it. m_CurrDelay tops out at
	// m_maxDelay-1 and the reads below go as far as delay[1+CurrDelay], i.e.
	// delay[m_maxDelay] -- an index the old loop ("<m_maxDelay") never wrote.
	// It read right only because m_DelayLookup was shared and the voltage hook,
	// which fills the whole table, had already run at the same numTS earlier in
	// the timestep. A private table has no such neighbour, and a blocked
	// schedule has no guaranteed neighbour at all, so fill it here. Same
	// formula and same numTS as the voltage hook used, so the value is the one
	// this hook has always read.
	std::vector<unsigned int> delay(m_Op_TFSF->m_maxDelay+1);
	for (unsigned int n=0;n<=m_Op_TFSF->m_maxDelay;++n)
		delay[n] = DelayLookupEntry(n, numTS, length, p);

	//get the current signal since an E-field is added ...
	FDTD_FLOAT* signal =  m_Op_TFSF->m_Exc->GetVoltageSignal();

	int nP,nPP;
	for (int n=0;n<3;++n)
	{
		if (!m_Op_TFSF->m_ActiveDir[n][0] && !m_Op_TFSF->m_ActiveDir[n][1])
			continue;

		nP = (n+1)%3;
		nPP = (n+2)%3;

		if (m_Op_TFSF->m_ActiveDir[n][0])
			CurrentFace(n, nP, nPP, 0, m_Op_TFSF->m_Start[n]-1, startX, stopX, nThreads, threadID, delay.data(), signal);

		if (m_Op_TFSF->m_ActiveDir[n][1])
			CurrentFace(n, nP, nPP, 1, m_Op_TFSF->m_Stop[n], startX, stopX, nThreads, threadID, delay.data(), signal);
	}
}

void Engine_Ext_TFSF::DoPostCurrentUpdates()
{
	DoPostCurrentUpdatesImpl(0, UINT_MAX, (int)m_Eng->GetNumberOfTimesteps(), 1, 0);
}

void Engine_Ext_TFSF::DoPostCurrentUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
{
	if (threadID != 0)
		return;
	DoPostCurrentUpdatesImpl(startX, stopX, numTS, 1, 0);
}
