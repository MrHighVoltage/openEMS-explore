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

#include "engine_ext_dispersive.h"
#include "operator_ext_dispersive.h"
#include "FDTD/engine_sse.h"
#include <climits>
#if OPENEMS_ENABLE_AVX2
#include "FDTD/engine_avx2.h"
#include <type_traits>
#endif

Engine_Ext_Dispersive::Engine_Ext_Dispersive(Operator_Ext_Dispersive* op_ext_disp) : Engine_Extension(op_ext_disp)
{
	m_Op_Ext_Disp = op_ext_disp;
	int order = m_Op_Ext_Disp->m_Order;
	curr_ADE = new FDTD_FLOAT**[order];
	volt_ADE = new FDTD_FLOAT**[order];
	for (int o=0;o<order;++o)
	{
		curr_ADE[o] = new FDTD_FLOAT*[3];
		volt_ADE[o] = new FDTD_FLOAT*[3];
		for (int n=0; n<3; ++n)
		{
			if (m_Op_Ext_Disp->m_curr_ADE_On[o]==true)
			{
				curr_ADE[o][n] = new FDTD_FLOAT[m_Op_Ext_Disp->m_LM_Count[o]];
				for (unsigned int i=0; i<m_Op_Ext_Disp->m_LM_Count[o]; ++i)
					curr_ADE[o][n][i]=0.0;
			}
			else
				curr_ADE[o][n] = NULL;
			if (m_Op_Ext_Disp->m_volt_ADE_On[o]==true)
			{
				volt_ADE[o][n] = new FDTD_FLOAT[m_Op_Ext_Disp->m_LM_Count[o]];
				for (unsigned int i=0; i<m_Op_Ext_Disp->m_LM_Count[o]; ++i)
					volt_ADE[o][n][i]=0.0;
			}
			else
				volt_ADE[o][n] = NULL;
		}
	}
}

Engine_Ext_Dispersive::~Engine_Ext_Dispersive()
{
	if (curr_ADE==NULL && volt_ADE==NULL)
		return;

	for (int o=0;o<m_Op_Ext_Disp->m_Order;++o)
	{
		for (int n=0; n<3; ++n)
		{
			delete[] curr_ADE[o][n];
			delete[] volt_ADE[o][n];
		}
		delete[] curr_ADE[o];
		delete[] volt_ADE[o];
	}
	delete[] curr_ADE;
	curr_ADE=NULL;

	delete[] volt_ADE;
	volt_ADE=NULL;
}

#if OPENEMS_ENABLE_AVX2
void Engine_Ext_Dispersive::BuildAVX2Index(Engine_AVX2* eng)
{
	const unsigned int nv = eng->GetNumVectors();
	if (m_avx2_idx_nv == nv)
		return;

	size_t total = 0;
	m_avx2_start.assign(m_Order, 0);
	for (int o=0; o<m_Order; ++o)
	{
		m_avx2_start[o] = total;
		total += m_Op_Ext_Disp->m_LM_Count.at(o);
	}
	m_avx2_off.resize(total);
	m_avx2_lane.resize(total);

	for (int o=0; o<m_Order; ++o)
	{
		unsigned int **pos = m_Op_Ext_Disp->m_LM_pos[o];
		const size_t base = m_avx2_start[o];
		for (unsigned int i=0; i<m_Op_Ext_Disp->m_LM_Count.at(o); ++i)
		{
			const unsigned int z = pos[2][i];
			m_avx2_off[base+i]  = (z % nv) * eng->m_vs_z
			                    + pos[1][i] * eng->m_vs_y
			                    + pos[0][i] * eng->m_vs_x;
			m_avx2_lane[base+i] = (unsigned char)(z / nv);
		}
	}
	m_avx2_idx_nv = nv;
}
#endif

void Engine_Ext_Dispersive::BuildSlabIndex()
{
	// Every worker thread enters the slab hooks at once, so this cannot be the
	// bare lazy check the flat path gets away with (that one only ever runs on
	// thread 0). The atomic keeps the steady state to a single acquire load.
	if (m_slab_idx_built.load(std::memory_order_acquire))
		return;
	std::lock_guard<std::mutex> lock(m_slab_idx_lock);
	if (m_slab_idx_built.load(std::memory_order_relaxed))
		return;

	const int order = m_Op_Ext_Disp->m_Order;
	size_t total = 0;
	m_slab_start.assign(order, 0);
	for (int o=0; o<order; ++o)
	{
		m_slab_start[o] = total;
		total += m_Op_Ext_Disp->m_LM_Count.at(o);
	}
	m_slab_perm.resize(total);
	m_slab_xoff.assign(order, std::vector<unsigned int>());

	bool any = false;
	unsigned int xlo = 0, xhi = 0;
	for (int o=0; o<order; ++o)
	{
		unsigned int **pos = m_Op_Ext_Disp->m_LM_pos[o];
		const unsigned int cnt = m_Op_Ext_Disp->m_LM_Count.at(o);
		std::vector<unsigned int>& xoff = m_slab_xoff[o];

		unsigned int xmax = 0;
		for (unsigned int i=0; i<cnt; ++i)
		{
			const unsigned int x = pos[0][i];
			if (x > xmax) xmax = x;
			if (!any || x < xlo) xlo = x;
			if (!any || x+1 > xhi) xhi = x+1;
			any = true;
		}

		// Counting sort, stable, so entries sharing an x keep their list order.
		// Today's operators emit the list x-ascending and this comes out the
		// identity, but nothing in Operator_Ext_Dispersive promises that and a
		// future builder breaking it would be silent.
		xoff.assign(cnt ? (size_t)xmax+2 : 1, 0);
		for (unsigned int i=0; i<cnt; ++i)
			++xoff[pos[0][i]+1];
		for (size_t x=1; x<xoff.size(); ++x)
			xoff[x] += xoff[x-1];

		std::vector<unsigned int> cursor(xoff.begin(), xoff.end());
		unsigned int* perm = m_slab_perm.data() + m_slab_start[o];
		for (unsigned int i=0; i<cnt; ++i)
			perm[cursor[pos[0][i]]++] = i;
	}
	m_slab_xlo = any ? xlo : 0;
	m_slab_xhi = any ? xhi : 0;

#if OPENEMS_ENABLE_AVX2
	// Folded in here so the slab path has one build step behind one guard; the
	// flat path still builds it on its own, which is free once this has run.
	if (m_Eng && m_Eng->GetType()==Engine::AVX2)
		BuildAVX2Index((Engine_AVX2*)m_Eng);
#endif

	m_slab_idx_built.store(true, std::memory_order_release);
}

// One x-split shared by every order, rather than an even split of each order's
// index range. A cell can carry an entry in more than one order -- a conducting
// sheet builds two orders over the identical position list -- and Apply2* then
// subtracts from that one engine cell once per order. Splitting each order's
// entries independently could hand those two subtractions to two threads, which
// is both a race and a change of summation order; splitting x keeps every entry
// at a cell on one thread, applied in order 0,1,... as the flat sweep does it.
// The price is that a slab whose cells sit in a thin band of x loads one thread
// more than the others.
bool Engine_Ext_Dispersive::SlabXShare(unsigned int startX, unsigned int stopX, int threadID,
                                       unsigned int& xLo, unsigned int& xHi) const
{
	if (threadID < 0)
	{
		xLo = startX;
		xHi = stopX;
		return true;
	}
	return SlabShare(m_slab_xlo, m_slab_xhi, startX, stopX, m_NrThreads, threadID, xLo, xHi);
}

bool Engine_Ext_Dispersive::SlabEntries(int o, unsigned int xLo, unsigned int xHi, int threadID,
                                        const unsigned int*& perm, unsigned int& j0, unsigned int& j1) const
{
	if (threadID < 0)
	{
		perm = NULL;
		j0 = 0;
		j1 = m_Op_Ext_Disp->m_LM_Count.at(o);
	}
	else
	{
		perm = m_slab_perm.data() + m_slab_start[o];
		j0 = SlabLowerBound(o, xLo);
		j1 = SlabLowerBound(o, xHi);
	}
	return j1 > j0;
}

template <typename EngType>
void Engine_Ext_Dispersive::Apply2VoltagesImpl(EngType* eng, unsigned int startX, unsigned int stopX, int threadID)
{
#if OPENEMS_ENABLE_AVX2
	if constexpr (std::is_same<EngType, Engine_AVX2>::value)
	{
		// The slab path already built this under BuildSlabIndex()'s lock; only
		// the single-threaded flat path may build it here.
		if (threadID < 0)
			BuildAVX2Index(eng);
	}
#endif
	unsigned int xLo, xHi;
	if (!SlabXShare(startX, stopX, threadID, xLo, xHi))
		return;

	for (int o=0;o<m_Op_Ext_Disp->m_Order;++o)
	{
		if (m_Op_Ext_Disp->m_volt_ADE_On[o]==false) continue;

		unsigned int **pos = m_Op_Ext_Disp->m_LM_pos[o];

		const unsigned int* perm;
		unsigned int j0, j1;
		if (!SlabEntries(o, xLo, xHi, threadID, perm, j0, j1)) continue;

#if OPENEMS_ENABLE_AVX2
		if constexpr (std::is_same<EngType, Engine_AVX2>::value)
		{
			// Division-free path: address the f8vectors directly through the
			// cache built by BuildAVX2Index(). Same operands, same order.
			const unsigned int*  off  = m_avx2_off.data()  + m_avx2_start[o];
			const unsigned char* lane = m_avx2_lane.data() + m_avx2_start[o];
			f8vector* volt = eng->m_volt;
			for (unsigned int j=j0; j<j1; ++j)
			{
				const unsigned int i = perm ? perm[j] : j;
				f8vector* c = volt + off[i];
				const unsigned int l = lane[i];
				c[0].f[l] -= volt_ADE[o][0][i];
				c[1].f[l] -= volt_ADE[o][1][i];
				c[2].f[l] -= volt_ADE[o][2][i];
			}
			continue;
		}
#endif

		for (unsigned int j=j0; j<j1; ++j)
		{
			const unsigned int i = perm ? perm[j] : j;
			eng->EngType::SetVolt(0,pos[0][i],pos[1][i],pos[2][i],
				eng->EngType::GetVoltRaw(0,pos[0][i],pos[1][i],pos[2][i]) - volt_ADE[o][0][i]
			);
			eng->EngType::SetVolt(1,pos[0][i],pos[1][i],pos[2][i],
				eng->EngType::GetVoltRaw(1,pos[0][i],pos[1][i],pos[2][i]) - volt_ADE[o][1][i]
			);
			eng->EngType::SetVolt(2,pos[0][i],pos[1][i],pos[2][i],
				eng->EngType::GetVoltRaw(2,pos[0][i],pos[1][i],pos[2][i]) - volt_ADE[o][2][i]
			);
		}
	}
}

void Engine_Ext_Dispersive::Apply2Voltages()
{
	const unsigned int startX = 0, stopX = UINT_MAX;
	const int threadID = -1;
	ENG_DISPATCH_ARGS(Apply2VoltagesImpl, startX, stopX, threadID);
}

void Engine_Ext_Dispersive::Apply2VoltagesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
{
	// numTS is not used: the ADE state is a per-cell recursion advanced by the
	// update hooks, with no term that reads the absolute timestep.
	(void)numTS;
	if (threadID < 0 || threadID >= m_NrThreads)
		return;
	BuildSlabIndex();
	ENG_DISPATCH_ARGS(Apply2VoltagesImpl, startX, stopX, threadID);
}

template <typename EngType>
void Engine_Ext_Dispersive::Apply2CurrentImpl(EngType* eng, unsigned int startX, unsigned int stopX, int threadID)
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

	for (int o=0;o<m_Op_Ext_Disp->m_Order;++o)
	{
		if (m_Op_Ext_Disp->m_curr_ADE_On[o]==false) continue;

		unsigned int **pos = m_Op_Ext_Disp->m_LM_pos[o];

		const unsigned int* perm;
		unsigned int j0, j1;
		if (!SlabEntries(o, xLo, xHi, threadID, perm, j0, j1)) continue;

#if OPENEMS_ENABLE_AVX2
		if constexpr (std::is_same<EngType, Engine_AVX2>::value)
		{
			const unsigned int*  off  = m_avx2_off.data()  + m_avx2_start[o];
			const unsigned char* lane = m_avx2_lane.data() + m_avx2_start[o];
			f8vector* curr = eng->m_curr;
			for (unsigned int j=j0; j<j1; ++j)
			{
				const unsigned int i = perm ? perm[j] : j;
				f8vector* c = curr + off[i];
				const unsigned int l = lane[i];
				c[0].f[l] -= curr_ADE[o][0][i];
				c[1].f[l] -= curr_ADE[o][1][i];
				c[2].f[l] -= curr_ADE[o][2][i];
			}
			continue;
		}
#endif

		for (unsigned int j=j0; j<j1; ++j)
		{
			const unsigned int i = perm ? perm[j] : j;
			eng->EngType::SetCurr(0,pos[0][i],pos[1][i],pos[2][i],
				eng->EngType::GetCurrRaw(0,pos[0][i],pos[1][i],pos[2][i]) - curr_ADE[o][0][i]
			);
			eng->EngType::SetCurr(1,pos[0][i],pos[1][i],pos[2][i],
				eng->EngType::GetCurrRaw(1,pos[0][i],pos[1][i],pos[2][i]) - curr_ADE[o][1][i]
			);
			eng->EngType::SetCurr(2,pos[0][i],pos[1][i],pos[2][i],
				eng->EngType::GetCurrRaw(2,pos[0][i],pos[1][i],pos[2][i]) - curr_ADE[o][2][i]
			);
		}
	}
}

void Engine_Ext_Dispersive::Apply2Current()
{
	const unsigned int startX = 0, stopX = UINT_MAX;
	const int threadID = -1;
	ENG_DISPATCH_ARGS(Apply2CurrentImpl, startX, stopX, threadID);
}

// The blocked schedule never offers the last x-line to the H hooks -- the
// engine cannot update H there, so it holds it back. That costs nothing here:
// Operator::ApplyMagneticBC zeroes IV on that line unconditionally, and the
// operator only builds a non-zero i_ext where IV is non-zero, so curr_ADE stays
// exactly 0 there and the flat sweep's subtraction is a no-op anyway.
void Engine_Ext_Dispersive::Apply2CurrentSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
{
	(void)numTS;
	if (threadID < 0 || threadID >= m_NrThreads)
		return;
	BuildSlabIndex();
	ENG_DISPATCH_ARGS(Apply2CurrentImpl, startX, stopX, threadID);
}
