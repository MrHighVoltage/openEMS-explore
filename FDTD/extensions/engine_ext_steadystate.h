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

#ifndef ENGINE_EXT_STEADYSTATE_H
#define ENGINE_EXT_STEADYSTATE_H

#include "engine_extension.h"
#include "FDTD/engine.h"
#include "FDTD/operator.h"

#include <deque>
#include <functional>
#include <vector>

class Operator_Ext_SteadyState;
class Engine_Interface_FDTD;

class Engine_Ext_SteadyState : public Engine_Extension
{
public:
	Engine_Ext_SteadyState(Operator_Ext_SteadyState* op_ext);
	virtual ~Engine_Ext_SteadyState();

	virtual void Apply2Voltages();
	virtual void Apply2Current();

	// The blocked schedule (Engine_AVX2_Multithread). This extension only
	// reads the fields, and it reads them from the first Apply2Voltages hook,
	// where every slab holds the values the flat sweep reads there: its
	// voltages after the update and the post-voltage hooks, the currents of the
	// timestep before. The slabs of one timestep partition the grid, so
	// - each probe line lies in exactly one slab per timestep, which records
	//   it into a ring of three periods indexed by the absolute timestep;
	// - at a check timestep, every slab sums its rows of the energy estimate
	//   (Engine_AVX2::FastEnergyRows()), and together they cover each row
	//   once;
	// - the check itself waits for the end of the block (SlabBlockEnd()),
	//   when every row is in, and adds the rows in order as the flat sweep
	//   does (Engine_AVX2::FastEnergy()), while the ring has kept the
	//   period's records although the block recorded on.
	// So blocked and flat runs reach the same verdicts, bit for bit, at the
	// same timesteps. A block holds at most one period (SlabBlockDepth()),
	// so one check, and records at most one period past it. Needs the AVX2
	// engine's field layout for the rows: on the others, false. The Vulkan
	// engine does the same on the GPU, with sums by x-plane, and hands the
	// check to CheckPeriod() once its batch is done (Engine_Vulkan::
	// RecordSteadyStateEnergy()).
	virtual bool SupportsSlabApply() const;
	virtual unsigned int SlabHookMask() const {return SLAB_APPLY_VOLT;}
	virtual bool SlabTouchesAt(SlabHook hook, unsigned int startX, unsigned int stopX, int numTS) const;
	virtual bool SlabHookWritesFields(SlabHook hook) const {(void)hook; return false;}
	virtual void Apply2VoltagesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID);
	virtual unsigned int SlabBlockDepth(int t0, unsigned int k) const;
	virtual bool SlabBlockEnd(int t0, unsigned int k, int threadID);

	virtual void SetEngine(Engine* eng);

	void SetEngineInterface(Engine_Interface_FDTD* eng_if) {m_Eng_Interface=eng_if;}
	//! The verdict of the last check: in a replay of in-batch sampling, of
	//! the last check the replayed timestep has passed (one of a timestep
	//! before it, as the run's verdict would have been at its end).
	double GetLastDiff();
	//! Called by GetLastDiff(): the GPU engine evaluates the checks it has computed (Engine_Vulkan::UpdateSteadyStateResult()).
	void SetGpuUpdater(std::function<void()> updater) {m_gpuUpdater = std::move(updater);}

	//! Is \a TS a timestep the last two periods are compared at?
	bool IsCheckTimestep(unsigned int TS) const;

	//! Compare the last two periods of the records at the check timestep \a TS, whose field energy is \a energy
	/*!
	  \a record(n, t) returns the voltage of probe \a n at the position \a t
	  (0 <= t < 2 periods) of a ring of two periods, which holds timestep TS
	  at TS % (2 periods): the pairing of the samples, and which of the two
	  periods normalises the difference, are those of that ring.
	  */
	void CheckPeriod(unsigned int TS, double energy, const std::function<double(size_t, unsigned int)>& record);

	//! The timestep at position \a t of that ring of two periods at the check timestep \a TS
	static unsigned int RingTimestep(unsigned int TS, unsigned int period, unsigned int t)
	{
		// TS sits at TS%(2p), and the ring holds everything back to TS-2p+1
		const unsigned int p2 = 2*period;
		return TS - ((TS%p2 + p2 - t)%p2);
	}

protected:
	//! Record the probes on x-lines in [startX,stopX) at timestep \a TS.
	void RecordProbes(unsigned int TS, unsigned int startX, unsigned int stopX);
	//! The records of the check at \a TS, as CheckPeriod() reads them.
	double RingRecord(unsigned int TS, size_t n, unsigned int t) const;

	//! With OPENEMS_STEADYSTATE_TRACE=1, print the check of timestep \a TS: the energy and the verdict.
	void TraceCheck(unsigned int TS, double energy, double diff) const;

	Operator_Ext_SteadyState* m_Op_SS;
	unsigned int m_period;        //!< timesteps per period
	unsigned int m_ringSize;      //!< 3 periods
	double m_last_max_diff;
	//! [probe][TS % m_ringSize]: the probe voltages of the last three periods
	std::vector<double> m_records;
	//! The rows of the energy estimate of the check timestep in the blocked schedule, see Engine_AVX2::FastEnergyRows()
	std::vector<double> m_rows;

	//! A check GetLastDiff() has not passed yet
	struct Verdict {unsigned int TS; double energy, diff;};
	//! Those checks, oldest first
	std::deque<Verdict> m_verdicts;
	//! The verdict of the last check passed
	double m_verdictPassed;
	//! The check \a v is passed: its verdict counts, and the trace prints it.
	void PassVerdict(const Verdict& v);

	double last_total_energy;
	Engine_Interface_FDTD* m_Eng_Interface;
	std::function<void()> m_gpuUpdater;
	bool m_trace; //!< OPENEMS_STEADYSTATE_TRACE
};


#endif // ENGINE_EXT_STEADYSTATE_H
