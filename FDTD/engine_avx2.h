/*
*	Copyright (C) 2026 openEMS contributors
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

#ifndef ENGINE_AVX2_H
#define ENGINE_AVX2_H

#include "engine.h"
#include "operator_avx2.h"

class Engine_AVX2 : public Engine
{
public:
	static Engine_AVX2* New(const Operator_AVX2* op);
	virtual ~Engine_AVX2();

	virtual void Init();
	virtual void Reset();

	//! The generic in-batch sampling of Engine (plans/BATCH_SAMPLING.md).
	virtual bool SupportsBatchSampling() const {return true;}

	// Scalar element access (for post-processing / probes); a probe in a
	// replay of in-batch sampling reads the sample instead (Engine::BatchRead()).
	inline virtual FDTD_FLOAT GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		if (OPENEMS_BS_UNLIKELY(m_bsMode != BS_OFF)) {FDTD_FLOAT v; if (BatchRead(0, n, x, y, z, v)) return v;}
		return m_volt[n + (z % numVectors) * m_vs_z + y * m_vs_y + x * m_vs_x].f[z / numVectors];
	}
	inline virtual FDTD_FLOAT GetVolt(unsigned int n, const unsigned int pos[3]) const
	{
		if (OPENEMS_BS_UNLIKELY(m_bsMode != BS_OFF)) {FDTD_FLOAT v; if (BatchRead(0, n, pos[0], pos[1], pos[2], v)) return v;}
		return m_volt[n + (pos[2] % numVectors) * m_vs_z + pos[1] * m_vs_y + pos[0] * m_vs_x].f[pos[2] / numVectors];
	}
	inline virtual FDTD_FLOAT GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		if (OPENEMS_BS_UNLIKELY(m_bsMode != BS_OFF)) {FDTD_FLOAT v; if (BatchRead(1, n, x, y, z, v)) return v;}
		return m_curr[n + (z % numVectors) * m_vs_z + y * m_vs_y + x * m_vs_x].f[z / numVectors];
	}
	inline virtual FDTD_FLOAT GetCurr(unsigned int n, const unsigned int pos[3]) const
	{
		if (OPENEMS_BS_UNLIKELY(m_bsMode != BS_OFF)) {FDTD_FLOAT v; if (BatchRead(1, n, pos[0], pos[1], pos[2], v)) return v;}
		return m_curr[n + (pos[2] % numVectors) * m_vs_z + pos[1] * m_vs_y + pos[0] * m_vs_x].f[pos[2] / numVectors];
	}

	inline virtual void SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)
	{
		m_volt[n + (z % numVectors) * m_vs_z + y * m_vs_y + x * m_vs_x].f[z / numVectors] = value;
	}
	inline virtual void SetVolt(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value)
	{
		m_volt[n + (pos[2] % numVectors) * m_vs_z + pos[1] * m_vs_y + pos[0] * m_vs_x].f[pos[2] / numVectors] = value;
	}
	inline virtual void SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)
	{
		m_curr[n + (z % numVectors) * m_vs_z + y * m_vs_y + x * m_vs_x].f[z / numVectors] = value;
	}
	inline virtual void SetCurr(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value)
	{
		m_curr[n + (pos[2] % numVectors) * m_vs_z + pos[1] * m_vs_y + pos[0] * m_vs_x].f[pos[2] / numVectors] = value;
	}

protected:
	Engine_AVX2(const Operator_AVX2* op);
	const Operator_AVX2* Op;

	virtual void UpdateVoltages(unsigned int startX, unsigned int numX);
	virtual void UpdateCurrents(unsigned int startX, unsigned int numX);

	//! Engine::SampleFootprint() from the cells' float offsets into m_volt/m_curr, computed once.
	virtual void SampleFootprint(size_t i);
	std::vector<uint64_t> m_bsOffsets; //!< per footprint cell: field<<63 | float offset

	//! The same updates for the y-rows [yStart,yStop) of the single x-line \a x.
	/*!
	  Every cell of a half-step depends only on the other field, so any split of
	  the cells gives the same result bit for bit; these let a caller split a
	  range finer than whole x-lines. UpdateCurrents() covers y < numLines[1]-1,
	  so \a yStop must not exceed that for UpdateCurrentRows().
	*/
	void UpdateVoltageRows(unsigned int x, unsigned int yStart, unsigned int yStop);
	void UpdateCurrentRows(unsigned int x, unsigned int yStart, unsigned int yStop);

	unsigned int numVectors; //!< number of 8-wide AVX vectors along z

public: // public access for efficient extension/interface access
	//! Number of 8-wide z-vectors, i.e. the width of one lane along z.
	//! Global z maps to (vector = z % numVectors, lane = z / numVectors).
	inline unsigned int GetNumVectors() const { return numVectors; }
	//! Grid extent as seen by the engine (Operator::GetNumberOfLines(n, true)).
	inline unsigned int GetNumLines(int n) const { return numLines[n]; }

	// --- The fast energy estimate (Engine_Interface_FDTD::CalcFastEnergy()) ---
	// sum |V|^2 and sum |I|^2 over x < NX-1, y < NY-1 and every z, in rows
	// (x,y) numbered x-major. A row is summed over z in float, the rows in
	// order in double, so the estimate does not depend on who sums a row when:
	// the blocked schedule sums each row as its slab passes the timestep of a
	// steady-state check (Engine_Ext_SteadyState), the flat one all at once,
	// and both give the same bits.
	//! The number of rows.
	unsigned int FastEnergyRowCount() const { return (numLines[0] - 1) * (numLines[1] - 1); }
	//! Sum the rows [first,last) into rows[2r] (|V|^2) and rows[2r+1] (|I|^2).
	void FastEnergyRows(unsigned int first, unsigned int last, double* rows) const;
	//! The estimate EPS0*E + MUE0*H from all rows of \a rows, each field summed in row order.
	double FastEnergy(const double* rows) const;
	//! The same, the rows summed on the fly.
	double FastEnergy() const;
	//! One field's share of the rows [first,last): out[r] = the row's float sum of |V|^2 (or |I|^2 if \a currents).
	/*!
	  The value FastEnergyRows() converts to double, bit for bit: a row's
	  |V|^2 and |I|^2 have separate accumulators, so they can be summed at
	  different times, e.g. each when its field has completed a timestep
	  (Engine_AVX2_Multithread's in-batch sampling, plans/BATCH_SAMPLING.md §4.4).
	*/
	void FastEnergyFieldRows(bool currents, unsigned int first, unsigned int last, float* out) const;
	//! The estimate from all rows of the two fields (FastEnergyFieldRows()), as FastEnergy(const double*).
	double FastEnergy(const float* eRows, const float* hRows) const;

	//! Flat field arrays with interleaved polarisations.
	//! Layout: m_volt[n + z_vec * m_vs_z + y * m_vs_y + x * m_vs_x]
	//! where n=0,1,2 (polarisation) is the fastest dimension.
	f8vector* m_volt;
	f8vector* m_curr;

	// Strides (in f8vector units)
	unsigned int m_vs_n; //!< polarisation stride = 1
	unsigned int m_vs_z; //!< z-vector stride = 3
	unsigned int m_vs_y; //!< y stride = 3 * numVectors
	unsigned int m_vs_x; //!< x stride = 3 * numVectors * numLines[1]
};

#endif // ENGINE_AVX2_H
