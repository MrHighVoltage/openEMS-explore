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

#ifndef PROCESSFIELDS_CALC_H
#define PROCESSFIELDS_CALC_H

/*
 * Template-based field extraction helpers.
 *
 * These eliminate the per-cell virtual function dispatch that occurs when
 * CalcField() goes through Engine_Interface_Base -> GetVolt()/GetCurr().
 * By switching on Engine::EngineType once and calling a templated inner
 * loop with a concrete engine type, all field accesses use static dispatch.
 *
 * The interpolation is Engine_Interface_FDTD's own, InterpolateField() and
 * InterpolateDualField(), as for GetEField() etc., including the seam of a
 * closed mesh (the alpha direction of a closed cylindrical mesh).
 */

#include "FDTD/engine.h"
#include "FDTD/engine_sse.h"
#include "FDTD/engine_interface_fdtd.h"
#include "FDTD/operator.h"
#include "processfields.h"
#include "tools/constants.h"
#include "tools/arraylib/array_nijk.h"

#include <atomic>
#include <thread>
#include <vector>
#include <functional>

namespace FieldCalc {

//--- Material context (avoids accessing Operator protected members) ----------

struct MaterialContext {
	ArrayLib::ArrayNIJK<float>* kappa;
	ArrayLib::ArrayNIJK<float>* epsR;
	ArrayLib::ArrayNIJK<float>* mueR;
};

//--- A dump's footprint, gathered by the GPU engine ---------------------------

/*
 * The raw cells of one field that a dump reads, as Engine_Vulkan gathers
 * them: all three components on the tensor product of three lists of grid
 * lines, laid out [n][a][b][c], with map[d][line] the position of a grid line
 * in its list, or -1.  It stands in for the engine in the templates below, so
 * a dump interpolates exactly as from the whole fields.  A read outside the
 * footprint returns 0 and sets `missed`.
 */
struct FootprintView {
	const float* data = nullptr;
	unsigned int stride = 1;   //!< 2 reads one part of interleaved complex values
	bool curr = false;         //!< the footprint holds curr, else volt
	const int* map[3] = {nullptr, nullptr, nullptr};
	size_t sb = 0, sa = 0, sn = 0;  //!< strides of b, a and n
	mutable std::atomic<bool> missed{false};

	inline FDTD_FLOAT Get(bool c, unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		const int a = map[0][x], b = map[1][y], cc = map[2][z];
		if ((c != curr) || ((a | b | cc) < 0))
		{
			missed.store(true, std::memory_order_relaxed);
			return 0;
		}
		return data[stride * (n * sn + (size_t)a * sa + (size_t)b * sb + (size_t)cc)];
	}
	inline FDTD_FLOAT GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{ return Get(false, n, x, y, z); }
	inline FDTD_FLOAT GetVolt(unsigned int n, const unsigned int pos[3]) const
	{ return Get(false, n, pos[0], pos[1], pos[2]); }
	inline FDTD_FLOAT GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{ return Get(true, n, x, y, z); }
	inline FDTD_FLOAT GetCurr(unsigned int n, const unsigned int pos[3]) const
	{ return Get(true, n, pos[0], pos[1], pos[2]); }
};

//--- Raw field access (non-virtual, templated on engine type) ----------------

template<typename EngT>
static inline double GetRawEField(const EngT* eng, const Operator* op, const MaterialContext&,
                                  unsigned int n, const unsigned int* pos)
{
	double value = eng->EngT::GetVolt(n, pos);
	double delta = op->GetEdgeLength(n, pos);
	return (delta != 0.0) ? value / delta : 0.0;
}

template<typename EngT>
static inline double GetRawJField(const EngT* eng, const Operator* op, const MaterialContext& mat,
                                  unsigned int n, const unsigned int* pos)
{
	double value = eng->EngT::GetVolt(n, pos);
	double delta = op->GetEdgeLength(n, pos);
	if (mat.kappa && (delta != 0.0))
	{
		ArrayLib::ArrayNIJK<float>& kappa = *mat.kappa;
		return value * kappa(n, pos[0], pos[1], pos[2]) / delta;
	}
	return 0.0;
}

template<typename EngT>
static inline double GetRawDField(const EngT* eng, const Operator* op, const MaterialContext& mat,
                                  unsigned int n, const unsigned int* pos)
{
	double value = eng->EngT::GetVolt(n, pos);
	double delta = op->GetEdgeLength(n, pos);
	if (mat.epsR && (delta != 0.0))
	{
		ArrayLib::ArrayNIJK<float>& epsR = *mat.epsR;
		return value * epsR(n, pos[0], pos[1], pos[2]) / delta;
	}
	return 0.0;
}

template<typename EngT>
static inline double GetRawRotHField(const EngT* eng, const Operator* op, const MaterialContext&,
                                     unsigned int n, const unsigned int* pos)
{
	unsigned int nP = (n+1) % 3;
	unsigned int nPP = (n+2) % 3;
	unsigned int locPos[] = {pos[0], pos[1], pos[2]};
	double area = op->GetEdgeArea(n, pos);
	double value = eng->EngT::GetCurr(nPP, pos);
	value -= eng->EngT::GetCurr(nP, pos);
	if (pos[nPP] > 0)
	{
		--locPos[nPP];
		value += eng->EngT::GetCurr(nP, locPos);
		++locPos[nPP];
	}
	if (pos[nP] > 0)
	{
		--locPos[nP];
		value -= eng->EngT::GetCurr(nPP, locPos);
	}
	return (area != 0.0) ? value / area : 0.0;
}

template<typename EngT>
static inline double GetRawHField(const EngT* eng, const Operator* op, const MaterialContext&,
                                  unsigned int n, const unsigned int* pos)
{
	double value = eng->EngT::GetCurr(n, pos[0], pos[1], pos[2]);
	double delta = op->GetEdgeLength(n, pos, true);
	return (delta != 0.0) ? value / delta : 0.0;
}

template<typename EngT>
static inline double GetRawBField(const EngT* eng, const Operator* op, const MaterialContext& mat,
                                  unsigned int n, const unsigned int* pos)
{
	double value = eng->EngT::GetCurr(n, pos[0], pos[1], pos[2]);
	double delta = op->GetEdgeLength(n, pos, true);
	if (mat.mueR && (delta != 0.0))
	{
		ArrayLib::ArrayNIJK<float>& mueR = *mat.mueR;
		return value * mueR(n, pos[0], pos[1], pos[2]) / delta;
	}
	return 0.0;
}

//--- Dispatch helpers for primary (E-type) and dual (H-type) fields ----------

template<typename EngT>
using RawFieldFunc = double (*)(const EngT*, const Operator*, const MaterialContext&, unsigned int, const unsigned int*);

template<typename EngT>
static RawFieldFunc<EngT> GetPrimaryFieldFunc(ProcessFields::DumpType type)
{
	switch (type)
	{
	case ProcessFields::E_FIELD_DUMP:    return &GetRawEField<EngT>;
	case ProcessFields::J_FIELD_DUMP:    return &GetRawJField<EngT>;
	case ProcessFields::D_FIELD_DUMP:    return &GetRawDField<EngT>;
	case ProcessFields::ROTH_FIELD_DUMP: return &GetRawRotHField<EngT>;
	default: return nullptr;
	}
}

template<typename EngT>
static RawFieldFunc<EngT> GetDualFieldFunc(ProcessFields::DumpType type)
{
	switch (type)
	{
	case ProcessFields::H_FIELD_DUMP: return &GetRawHField<EngT>;
	case ProcessFields::B_FIELD_DUMP: return &GetRawBField<EngT>;
	default: return nullptr;
	}
}

//--- Interpolated field extraction ------------------------------------------

//! The interpolated field (\a Dual: H or B, else E, J, D or rotH) at the dump positions of lines [startI, endI) in x
template<typename EngT, bool Dual>
static void ExtractFieldSlice(
	const EngT* eng, const Engine_Interface_FDTD* fdtd_if, const MaterialContext& mat,
	RawFieldFunc<EngT> rawFunc,
	unsigned int numLines[3], unsigned int* posLines[3],
	ArrayLib::ArrayNIJK<FDTD_FLOAT> &field,
	unsigned int startI, unsigned int endI)
{
	const Operator* op = fdtd_if->GetFDTDOperator();
	auto raw = [eng, op, &mat, rawFunc](unsigned int n, const unsigned int* p) {return rawFunc(eng, op, mat, n, p);};
	unsigned int pos[3];
	double out[3];

	for (unsigned int i = startI; i < endI; ++i)
	{
		pos[0] = posLines[0][i];
		for (unsigned int j = 0; j < numLines[1]; ++j)
		{
			pos[1] = posLines[1][j];
			for (unsigned int k = 0; k < numLines[2]; ++k)
			{
				pos[2] = posLines[2][k];
				if (Dual)
					fdtd_if->InterpolateDualField(pos, out, raw);
				else
					fdtd_if->InterpolateField(pos, out, raw);
				field(0, i, j, k) = out[0];
				field(1, i, j, k) = out[1];
				field(2, i, j, k) = out[2];
			}
		}
	}
}

//--- Top-level templated CalcField for a specific engine type ----------------

template<typename EngT>
static bool CalcFieldForEngine(
	const EngT* eng, const Engine_Interface_FDTD* fdtd_if, const MaterialContext& mat,
	ProcessFields::DumpType dumpType,
	unsigned int numLines[3], unsigned int* posLines[3],
	ArrayLib::ArrayNIJK<FDTD_FLOAT> &field,
	unsigned int numThreads)
{
	// Choose primary or dual path
	auto primaryFunc = GetPrimaryFieldFunc<EngT>(dumpType);
	auto dualFunc = GetDualFieldFunc<EngT>(dumpType);

	if (!primaryFunc && !dualFunc)
		return false; // unsupported dump type, fall back to legacy

	// Clamp thread count to available work
	if (numThreads < 1) numThreads = 1;
	if (numThreads > numLines[0]) numThreads = numLines[0];

	if (numThreads <= 1)
	{
		// Single-threaded path
		if (primaryFunc)
			ExtractFieldSlice<EngT, false>(eng, fdtd_if, mat, primaryFunc, numLines, posLines, field, 0, numLines[0]);
		else
			ExtractFieldSlice<EngT, true>(eng, fdtd_if, mat, dualFunc, numLines, posLines, field, 0, numLines[0]);
	}
	else
	{
		// Multi-threaded path: split i-dimension across threads
		std::vector<std::thread> threads;
		threads.reserve(numThreads);

		unsigned int linesPerThread = numLines[0] / numThreads;
		unsigned int remainder = numLines[0] % numThreads;
		unsigned int startI = 0;

		for (unsigned int t = 0; t < numThreads; ++t)
		{
			unsigned int count = linesPerThread + (t < remainder ? 1 : 0);
			unsigned int endI = startI + count;

			if (primaryFunc)
			{
				threads.emplace_back(
					ExtractFieldSlice<EngT, false>,
					eng, fdtd_if, std::cref(mat), primaryFunc, numLines, posLines, std::ref(field), startI, endI);
			}
			else
			{
				threads.emplace_back(
					ExtractFieldSlice<EngT, true>,
					eng, fdtd_if, std::cref(mat), dualFunc, numLines, posLines, std::ref(field), startI, endI);
			}
			startI = endI;
		}

		for (auto& t : threads)
			t.join();
	}

	return true;
}

} // namespace FieldCalc

#endif // PROCESSFIELDS_CALC_H
