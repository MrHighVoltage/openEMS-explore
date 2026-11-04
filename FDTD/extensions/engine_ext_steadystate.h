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

	// The blocked schedule. The probes are recorded per slab, from the first
	// Apply2Voltages hook, where every slab sees the values the flat sweep
	// records; each probe line lies in exactly one slab per timestep. The
	// records go into a ring of three periods indexed by the absolute
	// timestep, so that the check of a period can wait for the end of the
	// block while the block records on. The energy of the check, over the
	// whole grid, is not there yet, so SupportsSlabApply() stays false.
	virtual unsigned int SlabHookMask() const {return SLAB_APPLY_VOLT;}
	virtual bool SlabTouchesAt(SlabHook hook, unsigned int startX, unsigned int stopX, int numTS) const;
	virtual bool SlabHookWritesFields(SlabHook hook) const {(void)hook; return false;}
	virtual void Apply2VoltagesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID);

	void SetEngineInterface(Engine_Interface_FDTD* eng_if) {m_Eng_Interface=eng_if;}
	double GetLastDiff();
	void SetGpuUpdater(std::function<void()> updater) {m_gpuUpdater = std::move(updater);}
	void UpdateGpuSamples(const float* samples, unsigned int ringPeriods,
	                      unsigned int completedTS, double totalEnergy);

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

protected:
	//! Record the probes on x-lines in [startX,stopX) at timestep \a TS.
	void RecordProbes(unsigned int TS, unsigned int startX, unsigned int stopX);
	//! The records of the check at \a TS, as CheckPeriod() reads them.
	double RingRecord(unsigned int TS, size_t n, unsigned int t) const;

	//! With OPENEMS_STEADYSTATE_TRACE=1, print the check of timestep \a TS: the energy and the verdict.
	void TraceCheck(unsigned int TS, double energy) const;

	Operator_Ext_SteadyState* m_Op_SS;
	unsigned int m_period;        //!< timesteps per period
	unsigned int m_ringSize;      //!< 3 periods
	double m_last_max_diff;
	//! [probe][TS % m_ringSize]: the probe voltages of the last three periods
	std::vector<double> m_records;

	double last_total_energy;
	Engine_Interface_FDTD* m_Eng_Interface;
	std::function<void()> m_gpuUpdater;
	bool m_trace; //!< OPENEMS_STEADYSTATE_TRACE
};


#endif // ENGINE_EXT_STEADYSTATE_H
