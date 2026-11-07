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

#include "engine_ext_mur_abc.h"
#include "operator_ext_mur_abc.h"
#include "FDTD/engine.h"
#include "FDTD/engine_sse.h"
#include "tools/useful.h"
#include "operator_ext_excitation.h"

using std::cerr;
using std::endl;

Engine_Ext_Mur_ABC::Engine_Ext_Mur_ABC(Operator_Ext_Mur_ABC* op_ext) :
	Engine_Extension(op_ext),
	m_Mur_Coeff_nyP (op_ext->m_Mur_Coeff_nyP),
	m_Mur_Coeff_nyPP(op_ext->m_Mur_Coeff_nyPP),
	m_volt_nyP ("volt_nyP",  op_ext->m_numLines),
	m_volt_nyPP("volt_nyPP", op_ext->m_numLines)
{
	m_Op_mur = op_ext;
	m_numLines[0] = m_Op_mur->m_numLines[0];
	m_numLines[1] = m_Op_mur->m_numLines[1];
	m_ny = m_Op_mur->m_ny;
	m_nyP = m_Op_mur->m_nyP;
	m_nyPP = m_Op_mur->m_nyPP;
	m_LineNr = m_Op_mur->m_LineNr;
	m_LineNr_Shift = m_Op_mur->m_LineNr_Shift;

	//find if some excitation is on this mur-abc and find the max length of this excite, so that the abc can start after the excitation is done...
	int maxDelay=-1;
	Operator_Ext_Excitation* Exc_ext = m_Op_mur->m_Op->GetExcitationExtension();
	for (unsigned int n=0; n<Exc_ext->GetVoltCount(); ++n)
	{
		if ( ((Exc_ext->Volt_dir[n]==m_nyP) || (Exc_ext->Volt_dir[n]==m_nyPP)) && (Exc_ext->Volt_index[m_ny][n]==m_LineNr) )
		{
			if ((int)Exc_ext->Volt_delay[n]>maxDelay)
				maxDelay = (int)Exc_ext->Volt_delay[n];
		}
	}
	m_start_TS = 0;
	if (maxDelay>=0)
	{
		m_start_TS = maxDelay + m_Op_mur->m_Op->GetExcitationSignal()->GetLength() + 10; //give it some extra timesteps, for the excitation to travel at least one cell away
		cerr << "Engine_Ext_Mur_ABC::Engine_Ext_Mur_ABC: Warning: Excitation inside the Mur-ABC #" <<  m_ny << "-" << (int)(m_LineNr>0) << " found!!!!  Mur-ABC will be switched on after excitation is done at " << m_start_TS << " timesteps!!! " << endl;
	}

	SetNumberOfThreads(1);
}

Engine_Ext_Mur_ABC::~Engine_Ext_Mur_ABC()
{
}


void Engine_Ext_Mur_ABC::SetNumberOfThreads(int nrThread)
{
	Engine_Extension::SetNumberOfThreads(nrThread);

	m_numX = AssignJobs2Threads(m_numLines[0],m_NrThreads,false);
	m_start.resize(m_NrThreads,0);
	m_start.at(0)=0;
	for (size_t n=1; n<m_numX.size(); ++n)
		m_start.at(n) = m_start.at(n-1) + m_numX.at(n-1);
}


// The plane this extension owns is indexed (i,j) along m_nyP and m_nyPP, and
// which of those is the engine's x axis depends on m_ny: for m_ny==1 it is the
// inner loop j, for m_ny==2 the outer loop i, and for m_ny==0 neither -- the
// whole plane then sits on the single x-line m_LineNr. The two helpers below
// are where that mapping lives; the three hooks just take loop bounds and are
// the same code for the flat sweep and for a slab.
bool Engine_Ext_Mur_ABC::ThreadRange(int threadID,
                                     unsigned int& iStart, unsigned int& iStop,
                                     unsigned int& jStart, unsigned int& jStop)
{
	iStart = iStop = jStart = jStop = 0;
	if ((threadID<0) || (threadID>=m_NrThreads))
		return false;

	iStart = m_start.at(threadID);
	iStop = iStart + m_numX.at(threadID);
	jStop = m_numLines[1];
	return (iStop>iStart) && (jStop>jStart);
}

bool Engine_Ext_Mur_ABC::SlabRange(unsigned int startX, unsigned int stopX, int threadID,
                                   unsigned int& iStart, unsigned int& iStop,
                                   unsigned int& jStart, unsigned int& jStop)
{
	iStart = iStop = jStart = jStop = 0;

	if (m_ny==2)
	{
		// m_nyP is x, so the slab cuts the outer loop and the share-out across
		// threads has to be made over what is left of it. Keeping the plane's
		// own split would leave a slab that covers one thread's stripe to that
		// one thread while the rest idle.
		if (!SlabShare(0, m_numLines[0], startX, stopX, m_NrThreads, threadID, iStart, iStop))
			return false;
		jStop = m_numLines[1];
		return jStop>jStart;
	}

	if (!ThreadRange(threadID, iStart, iStop, jStart, jStop))
		return false;

	if (m_ny==1)
	{
		// m_nyPP is x: cut the inner loop and leave the split over i alone.
		if (jStart<startX) jStart = startX;
		if (jStop>stopX) jStop = stopX;
		return jStop>jStart;
	}

	// m_ny==0: pattern B. The plane writes m_LineNr and reads m_LineNr_Shift one
	// line inward, so it runs in full or not at all, and only for the slab that
	// holds both at the same timestep. The schedule guarantees such a slab
	// exists at either domain edge (see SupportsSlabApply()); this test only
	// picks out which slab it is, it is not what makes the case safe.
	return SlabTouches(SLAB_PRE_VOLT, startX, stopX);
}

bool Engine_Ext_Mur_ABC::SlabTouches(SlabHook hook, unsigned int startX, unsigned int stopX) const
{
	(void)hook;
	// The same test SlabRange() applies to a plane normal to x, so the two
	// cannot disagree; every other orientation crosses every slab.
	if (m_ny!=0)
		return true;
	return (m_LineNr>=startX) && (m_LineNr<stopX) &&
	       ((unsigned int)m_LineNr_Shift>=startX) && ((unsigned int)m_LineNr_Shift<stopX);
}

bool Engine_Ext_Mur_ABC::SupportsSlabApply() const
{
	// Normal to y or z, the shifted read sits on the same x-line as the write,
	// so any slab that owns the line owns everything the extension touches.
	if (m_ny!=0)
		return true;

	// Normal to x, the extension reads one line inward and needs that line at
	// the same timestep as the one it writes. That is exactly the guarantee the
	// slab contract makes for a cell on a domain edge, and the schedule pays
	// for it on both sides: the first tile holds x=0 and is never narrower than
	// W-k+1 >= 3 lines, and ConfigureTemporalBlocking() folds a short tail tile
	// into its neighbour so the last tile is at least k+3 lines -- which is
	// what stops the tail's sloping *left* side from reaching x=NX-1 after
	// x=NX-2 has dropped out of the slab.
	//
	// A Mur ABC is only ever created on a domain face, so there is no interior
	// case to exclude here.
	return true;
}

template <typename EngType>
void Engine_Ext_Mur_ABC::DoPreVoltageUpdatesImpl(EngType* eng, unsigned int iStart, unsigned int iStop, unsigned int jStart, unsigned int jStop)
{
	// See detailed comments in operator_ext_mur_abc.h, not repeated here.
	unsigned int pos[] = {0,0,0};
	unsigned int pos_shift[] = {0,0,0};
	pos[m_ny] = m_LineNr;
	pos_shift[m_ny] = m_LineNr_Shift;

	for (unsigned int i = iStart; i < iStop; i++)
	{
		pos[m_nyP] = i;
		pos_shift[m_nyP] = i;

		for (unsigned int j = jStart; j < jStop; j++)
		{
			pos[m_nyPP] = j;
			pos_shift[m_nyPP] = j;

			m_volt_nyP(i,j) = eng->EngType::GetVoltRaw(m_nyP, pos_shift) -
				m_Op_mur->m_Mur_Coeff_nyP(i,j) * eng->EngType::GetVoltRaw(m_nyP, pos);

			m_volt_nyPP(i,j) = eng->EngType::GetVoltRaw(m_nyPP, pos_shift) -
				m_Op_mur->m_Mur_Coeff_nyPP(i,j) * eng->EngType::GetVoltRaw(m_nyPP, pos);
		}
	}
}

void Engine_Ext_Mur_ABC::DoPreVoltageUpdates(int threadID)
{
	if (m_Eng==NULL) return;
	if (IsActive()==false) return;
	unsigned int iStart,iStop,jStart,jStop;
	if (!ThreadRange(threadID, iStart, iStop, jStart, jStop)) return;
	ENG_DISPATCH_ARGS(DoPreVoltageUpdatesImpl, iStart, iStop, jStart, jStop);
}

void Engine_Ext_Mur_ABC::DoPreVoltageUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
{
	if (m_Eng==NULL) return;
	if (IsActive(numTS)==false) return;
	unsigned int iStart,iStop,jStart,jStop;
	if (!SlabRange(startX, stopX, threadID, iStart, iStop, jStart, jStop)) return;
	ENG_DISPATCH_ARGS(DoPreVoltageUpdatesImpl, iStart, iStop, jStart, jStop);
}

template <typename EngType>
void Engine_Ext_Mur_ABC::DoPostVoltageUpdatesImpl(EngType* eng, unsigned int iStart, unsigned int iStop, unsigned int jStart, unsigned int jStop)
{
	unsigned int pos_shift[] = {0,0,0};
	pos_shift[m_ny] = m_LineNr_Shift;

	for (unsigned int i = iStart; i < iStop; i++)
	{
		pos_shift[m_nyP] = i;

		for (unsigned int j = jStart; j < jStop; j++)
		{
			pos_shift[m_nyPP] = j;

			m_volt_nyP(i,j) +=
				m_Op_mur->m_Mur_Coeff_nyP(i,j) * eng->EngType::GetVoltRaw(m_nyP, pos_shift);
			m_volt_nyPP(i,j) +=
				m_Op_mur->m_Mur_Coeff_nyPP(i,j) * eng->EngType::GetVoltRaw(m_nyPP, pos_shift);
		}
	}
}

void Engine_Ext_Mur_ABC::DoPostVoltageUpdates(int threadID)
{
	if (m_Eng==NULL) return;
	if (IsActive()==false) return;
	unsigned int iStart,iStop,jStart,jStop;
	if (!ThreadRange(threadID, iStart, iStop, jStart, jStop)) return;
	ENG_DISPATCH_ARGS(DoPostVoltageUpdatesImpl, iStart, iStop, jStart, jStop);
}

void Engine_Ext_Mur_ABC::DoPostVoltageUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
{
	if (m_Eng==NULL) return;
	if (IsActive(numTS)==false) return;
	unsigned int iStart,iStop,jStart,jStop;
	if (!SlabRange(startX, stopX, threadID, iStart, iStop, jStart, jStop)) return;
	ENG_DISPATCH_ARGS(DoPostVoltageUpdatesImpl, iStart, iStop, jStart, jStop);
}

template <typename EngType>
void Engine_Ext_Mur_ABC::Apply2VoltagesImpl(EngType* eng, unsigned int iStart, unsigned int iStop, unsigned int jStart, unsigned int jStop)
{
	unsigned int pos[] = {0,0,0};
	pos[m_ny] = m_LineNr;

	for (unsigned int i = iStart; i < iStop; i++)
	{
		pos[m_nyP] = i;

		for (unsigned int j = jStart; j < jStop; j++)
		{
			pos[m_nyPP] = j;

			eng->EngType::SetVolt(m_nyP , pos, m_volt_nyP(i,j));
			eng->EngType::SetVolt(m_nyPP, pos, m_volt_nyPP(i,j));
		}
	}
}

void Engine_Ext_Mur_ABC::Apply2Voltages(int threadID)
{
	if (m_Eng==NULL) return;
	if (IsActive()==false) return;
	unsigned int iStart,iStop,jStart,jStop;
	if (!ThreadRange(threadID, iStart, iStop, jStart, jStop)) return;
	ENG_DISPATCH_ARGS(Apply2VoltagesImpl, iStart, iStop, jStart, jStop);
}

void Engine_Ext_Mur_ABC::Apply2VoltagesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
{
	if (m_Eng==NULL) return;
	if (IsActive(numTS)==false) return;
	unsigned int iStart,iStop,jStart,jStop;
	if (!SlabRange(startX, stopX, threadID, iStart, iStop, jStart, jStop)) return;
	ENG_DISPATCH_ARGS(Apply2VoltagesImpl, iStart, iStop, jStart, jStop);
}
