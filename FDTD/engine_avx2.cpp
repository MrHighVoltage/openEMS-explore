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

#include "engine_avx2.h"
#include "tools/constants.h"
#include "tools/denormal.h"

#include <immintrin.h>
#include <cstring>
#include <cmath>

using std::cout;
using std::endl;

// --- 32-byte aligned allocation helper ---

static f8vector* Alloc_f8_flat(size_t count)
{
	void* ptr = nullptr;
	if (posix_memalign(&ptr, 32, sizeof(f8vector) * count) != 0)
		return nullptr;
	memset(ptr, 0, sizeof(f8vector) * count);
	return static_cast<f8vector*>(ptr);
}

// --- AVX2 cross-lane shift helpers for z-boundary handling ---

// Shift left by one float: [a b c d e f g h] -> [0 a b c d e f g]
static inline __m256 avx2_shift_left_1f(__m256 val)
{
	__m256i vi = _mm256_castps_si256(val);
	__m256i shifted_in_lane = _mm256_bslli_epi128(vi, 4);
	__m256 perm = _mm256_permute2f128_ps(val, val, 0x00);
	__m256 v3_broadcast = _mm256_permute_ps(perm, 0xFF);
	__m256 shifted = _mm256_castsi256_ps(shifted_in_lane);
	return _mm256_blend_ps(shifted, v3_broadcast, 0x10);
}

// Shift right by one float: [a b c d e f g h] -> [b c d e f g h 0]
static inline __m256 avx2_shift_right_1f(__m256 val)
{
	__m256i vi = _mm256_castps_si256(val);
	__m256i shifted_in_lane = _mm256_bsrli_epi128(vi, 4);
	__m256 perm = _mm256_permute2f128_ps(val, val, 0x11);
	__m256 v4_broadcast = _mm256_permute_ps(perm, 0x00);
	__m256 shifted = _mm256_castsi256_ps(shifted_in_lane);
	return _mm256_blend_ps(shifted, v4_broadcast, 0x08);
}

// ============================================================================
// Engine_AVX2
// ============================================================================

Engine_AVX2* Engine_AVX2::New(const Operator_AVX2* op)
{
	cout << "Create FDTD engine (AVX2 + FMA, compressed flat arrays)" << endl;
	Engine_AVX2* e = new Engine_AVX2(op);
	e->Init();
	return e;
}

Engine_AVX2::Engine_AVX2(const Operator_AVX2* op) : Engine(op)
{
	m_type = AVX2;
	Op = op;
	m_volt = nullptr;
	m_curr = nullptr;
	numVectors = (unsigned int)ceil((double)numLines[2] / 8.0);

	// Strides for interleaved polarisation layout: N-fastest (I-J-K-N ordering)
	m_vs_n = 1;
	m_vs_z = 3;
	m_vs_y = 3 * numVectors;
	m_vs_x = 3 * numVectors * numLines[1];

	// Flush denormals to zero for performance
	Denormal::Disable();
}

Engine_AVX2::~Engine_AVX2()
{
	Reset();
}

void Engine_AVX2::Init()
{
	Engine::Init();

	// Free the base class arrays — we use our own flat AVX2 arrays
	delete volt_ptr;
	volt_ptr = nullptr;
	delete curr_ptr;
	curr_ptr = nullptr;

	size_t totalElems = (size_t)3 * numLines[0] * numLines[1] * numVectors;
	m_volt = Alloc_f8_flat(totalElems);
	m_curr = Alloc_f8_flat(totalElems);
}

void Engine_AVX2::SampleFootprint(size_t i)
{
	// GetVolt() divides by numVectors for every cell; the offsets do it once
	const std::vector<uint64_t>& fp = GetFootprint();
	if (m_bsOffsets.size() != fp.size())
	{
		m_bsOffsets.resize(fp.size());
		unsigned int field, n, x, y, z;
		for (size_t c = 0; c < fp.size(); ++c)
		{
			DecodeFootprintKey(fp[c], field, n, x, y, z);
			const uint64_t vec = n + (z % numVectors) * m_vs_z + y * m_vs_y + (uint64_t)x * m_vs_x;
			m_bsOffsets[c] = ((uint64_t)field<<63) | (vec*8 + z / numVectors);
		}
	}
	const float* volt = m_volt[0].f;
	const float* curr = m_curr[0].f;
	FDTD_FLOAT* out = BatchProbeSample(i);
	for (size_t c = 0; c < m_bsOffsets.size(); ++c)
	{
		const uint64_t off = m_bsOffsets[c];
		out[c] = ((off>>63) ? curr : volt)[off & ~(1ull<<63)];
	}
}

void Engine_AVX2::Reset()
{
	free(m_volt);
	m_volt = nullptr;
	free(m_curr);
	m_curr = nullptr;
	Engine::Reset();
}

// ============================================================================
// Voltage update with AVX2 + FMA + compressed operator + flat arrays
//
// FDTD voltage update equation (per polarisation):
//   V_n = vv_n * V_n + vi_n * (curl_H)_n
//
// Uses FMA: result = fmadd(vv, V, vi * curl_H)
//
// Memory layout (I-J-K-N ordering):
//   All 3 polarisations for same (x,y,z) are contiguous in memory.
//   field[n + z*3 + y*vs_y + x*vs_x]
//
// Operator compression:
//   Single index lookup Op->m_Op_index(x,y,z) into compact compressed tables.
//   The compressed tables are tiny and stay cache-hot (typically <64 KB).
//
// Boundary handling:
//   Branchless: i_shift_x = (x>0)*vs_x, i_shift_y = (y>0)*vs_y
//   At x=0 or y=0, the shift is 0, making "(here) - (here-shift)" = 0.
//
// Load reuse:
//   Each of curr[base], curr[base+1], curr[base+2] ("H at here") feeds two of
//   the three polarisation curls. GCC does not CSE these across the three
//   per-polarisation blocks (confirmed via disassembly), so they are loaded
//   into local __m256 once per z-vector and reused explicitly. Single-
//   threaded throughput: ~7-8% (benchmarked, does not show up in the
//   memory-bandwidth-bound multithreaded engine).
//
// Multithreaded engine is at its memory-bandwidth roofline, not compute-
// bound: on a 4-core/8-thread Skylake host, `perf stat -e
// uncore_imc/data_reads/,data_writes/` measured ~27-38 GB/s of real DRAM
// traffic during a 4-thread run (its optimal thread count -- see
// Engine_AVX2_Multithread's runtime thread-count auto-tune), against a
// ~29 GB/s ceiling from a STREAM-Triad-style probe with the same ~2:1
// read:write mix on the same host. Two consequences already validated here,
// don't redo them without new evidence:
//   - Using all logical (hyperthreaded) cores hurts, not helps: 8 threads
//     ran ~6% *slower* than 4 on this host. SMT siblings add no memory
//     bandwidth, only L1/L2 and load/store-port contention. The runtime
//     auto-tune already discovers this on its own when numThreads=0.
//   - _mm256_stream_ps() for the volt/curr writes (skipping the normal
//     store's write-allocate fetch) was tried and made things *worse*
//     (~15% slower single-threaded, several % slower multithreaded): the
//     3-f8vector-wide polarisation stride (96 bytes) doesn't align to
//     64-byte cache lines, so the non-temporal write-combining buffers
//     can't reliably gather full lines and get flushed early -- worse than
//     the write-allocate they were meant to avoid. Revisit only if the
//     field layout is changed to a cache-line-aligned stride.
// ============================================================================

void Engine_AVX2::UpdateVoltages(unsigned int startX, unsigned int numX)
{
	for (unsigned int x = startX; x < startX + numX; ++x)
		UpdateVoltageRows(x, 0, numLines[1]);
}

// ============================================================================
// The fast energy estimate, by rows
// ============================================================================

namespace
{
// Horizontal sum of the eight lanes, always in the same order.
inline float HorizontalSum(__m256 v)
{
	__m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
	s = _mm_hadd_ps(s, s);
	s = _mm_hadd_ps(s, s);
	return _mm_cvtss_f32(s);
}

// Row (x,y) of one field: sum |F|^2 over its three components and every
// z-vector, in one 8-lane float accumulator, then the lanes in a fixed order.
// This is the definition of a row's value; everything below computes it with
// exactly this sequence of operations per row.
inline float FastEnergyRowField(const f8vector* fld, unsigned int base, unsigned int nZv)
{
	__m256 acc = _mm256_setzero_ps();
	for (unsigned int z = 0; z < nZv; ++z, base += 3)
	{
		const __m256 fx = _mm256_load_ps(fld[base].f);
		const __m256 fy = _mm256_load_ps(fld[base + 1].f);
		const __m256 fz = _mm256_load_ps(fld[base + 2].f);
		acc = _mm256_fmadd_ps(fx, fx, acc);
		acc = _mm256_fmadd_ps(fy, fy, acc);
		acc = _mm256_fmadd_ps(fz, fz, acc);
	}
	return HorizontalSum(acc);
}

// Four rows of one field at once, each with its own accumulator: the same
// operations per row as FastEnergyRowField(), four independent chains instead
// of one, so the FMA latency does not bound it.
inline void FastEnergyRowField4(const f8vector* fld, unsigned int b0, unsigned int b1, unsigned int b2,
                                unsigned int b3, unsigned int nZv, float* out)
{
	__m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
	__m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
	for (unsigned int z = 0; z < nZv; ++z, b0 += 3, b1 += 3, b2 += 3, b3 += 3)
	{
		for (unsigned int n = 0; n < 3; ++n)
		{
			const __m256 f0 = _mm256_load_ps(fld[b0 + n].f);
			const __m256 f1 = _mm256_load_ps(fld[b1 + n].f);
			const __m256 f2 = _mm256_load_ps(fld[b2 + n].f);
			const __m256 f3 = _mm256_load_ps(fld[b3 + n].f);
			a0 = _mm256_fmadd_ps(f0, f0, a0);
			a1 = _mm256_fmadd_ps(f1, f1, a1);
			a2 = _mm256_fmadd_ps(f2, f2, a2);
			a3 = _mm256_fmadd_ps(f3, f3, a3);
		}
	}
	out[0] = HorizontalSum(a0);
	out[1] = HorizontalSum(a1);
	out[2] = HorizontalSum(a2);
	out[3] = HorizontalSum(a3);
}

// Row (x,y) of the estimate: sum |V|^2 and sum |I|^2 over every z-vector.
// Each accumulator takes the operations of FastEnergyRowField() in its order,
// so the values are its values; one loop with two independent chains runs
// about as fast as one of them alone.
inline void FastEnergyRow(const f8vector* volt, const f8vector* curr, unsigned int base,
                          unsigned int nZv, double& e, double& h)
{
	__m256 accE = _mm256_setzero_ps();
	__m256 accH = _mm256_setzero_ps();
	for (unsigned int z = 0; z < nZv; ++z, base += 3)
	{
		const __m256 vx = _mm256_load_ps(volt[base].f);
		const __m256 vy = _mm256_load_ps(volt[base + 1].f);
		const __m256 vz = _mm256_load_ps(volt[base + 2].f);
		accE = _mm256_fmadd_ps(vx, vx, accE);
		accE = _mm256_fmadd_ps(vy, vy, accE);
		accE = _mm256_fmadd_ps(vz, vz, accE);

		const __m256 ix = _mm256_load_ps(curr[base].f);
		const __m256 iy = _mm256_load_ps(curr[base + 1].f);
		const __m256 iz = _mm256_load_ps(curr[base + 2].f);
		accH = _mm256_fmadd_ps(ix, ix, accH);
		accH = _mm256_fmadd_ps(iy, iy, accH);
		accH = _mm256_fmadd_ps(iz, iz, accH);
	}
	e = (double)HorizontalSum(accE);
	h = (double)HorizontalSum(accH);
}
}

void Engine_AVX2::FastEnergyFieldRows(bool currents, unsigned int first, unsigned int last, float* out) const
{
	const f8vector* fld = currents ? m_curr : m_volt;
	const unsigned int ny = numLines[1] - 1;
	if (ny == 0)
		return;  // no rows; grids have >= 3 lines per direction, but do not divide by zero
	unsigned int r = first;
	unsigned int x = r / ny, y = r % ny;
	// the base of row (x,y), advanced row by row without a division
	auto next = [&]() -> unsigned int
	{
		const unsigned int b = x * m_vs_x + y * m_vs_y;
		if (++y == ny)
		{
			y = 0;
			++x;
		}
		return b;
	};
	for (; r + 4 <= last; r += 4)
	{
		const unsigned int b0 = next(), b1 = next(), b2 = next(), b3 = next();
		FastEnergyRowField4(fld, b0, b1, b2, b3, numVectors, out + r);
	}
	for (; r < last; ++r)
		out[r] = FastEnergyRowField(fld, next(), numVectors);
}

double Engine_AVX2::FastEnergy(const float* eRows, const float* hRows) const
{
	// as FastEnergy(const double*): each field's rows in row order, in double
	const unsigned int count = FastEnergyRowCount();
	double E = 0, H = 0;
	for (unsigned int r = 0; r < count; ++r)
	{
		E += (double)eRows[r];
		H += (double)hRows[r];
	}
	return EPS0 * E + MUE0 * H;
}

void Engine_AVX2::FastEnergyRows(unsigned int first, unsigned int last, double* rows) const
{
	const unsigned int ny = numLines[1] - 1;
	for (unsigned int r = first; r < last; ++r)
	{
		const unsigned int base = (r / ny) * m_vs_x + (r % ny) * m_vs_y;
		FastEnergyRow(m_volt, m_curr, base, numVectors, rows[2 * (size_t)r], rows[2 * (size_t)r + 1]);
	}
}

double Engine_AVX2::FastEnergy(const double* rows) const
{
	const unsigned int count = FastEnergyRowCount();
	double E = 0, H = 0;
	for (unsigned int r = 0; r < count; ++r)
	{
		E += rows[2 * (size_t)r];
		H += rows[2 * (size_t)r + 1];
	}
	return EPS0 * E + MUE0 * H;
}

double Engine_AVX2::FastEnergy() const
{
	const unsigned int ny = numLines[1] - 1;
	const unsigned int count = FastEnergyRowCount();
	double E = 0, H = 0;
	for (unsigned int r = 0; r < count; ++r)
	{
		double e, h;
		FastEnergyRow(m_volt, m_curr, (r / ny) * m_vs_x + (r % ny) * m_vs_y, numVectors, e, h);
		E += e;
		H += h;
	}
	return EPS0 * E + MUE0 * H;
}

void Engine_AVX2::UpdateVoltageRows(unsigned int x, unsigned int yStart, unsigned int yStop)
{
	f8vector* __restrict volt = m_volt;
	f8vector* __restrict curr = m_curr;

	const int vs_z = (int)m_vs_z;
	const int vs_y = (int)m_vs_y;
	const int vs_x = (int)m_vs_x;

	unsigned int pos[3];
	pos[0] = x;

	{
		// Branchless: 0 when at boundary, full stride otherwise
		const int i_shift_x = (pos[0] > 0) * vs_x;

		for (pos[1] = yStart; pos[1] < yStop; ++pos[1])
		{
			const int i_shift_y = (pos[1] > 0) * vs_y;

			// --- Main loop: z_vec >= 1 (z-1 stays within allocation) ---
			for (pos[2] = 1; pos[2] < numVectors; ++pos[2])
			{
				const unsigned int index = Op->m_Op_index(pos[0], pos[1], pos[2]);
				const int base = (int)(pos[0] * m_vs_x + pos[1] * m_vs_y + pos[2] * m_vs_z);

				// H-field values at "here" are each used by two of the three
				// curl computations below -- load once and reuse instead of
				// re-reading the same address from memory twice.
				const __m256 curr_here_x = _mm256_load_ps(curr[base].f);
				const __m256 curr_here_y = _mm256_load_ps(curr[base + 1].f);
				const __m256 curr_here_z = _mm256_load_ps(curr[base + 2].f);

				// X-polarisation: V_x = vv_x * V_x + vi_x * (dHz/dy - dHy/dz)
				{
					__m256 v  = _mm256_load_ps(volt[base].f);
					__m256 vv = _mm256_load_ps(Op->f8_vv_Comp[0][index].f);
					__m256 vi = _mm256_load_ps(Op->f8_vi_Comp[0][index].f);

					__m256 curl = _mm256_sub_ps(
						curr_here_z,
						_mm256_load_ps(curr[base + 2 - i_shift_y].f)
					);
					curl = _mm256_sub_ps(curl, curr_here_y);
					curl = _mm256_add_ps(curl,
						_mm256_load_ps(curr[base + 1 - vs_z].f));

					v = _mm256_fmadd_ps(vv, v, _mm256_mul_ps(vi, curl));
					_mm256_store_ps(volt[base].f, v);
				}

				// Y-polarisation: V_y = vv_y * V_y + vi_y * (dHx/dz - dHz/dx)
				{
					__m256 v  = _mm256_load_ps(volt[base + 1].f);
					__m256 vv = _mm256_load_ps(Op->f8_vv_Comp[1][index].f);
					__m256 vi = _mm256_load_ps(Op->f8_vi_Comp[1][index].f);

					__m256 curl = _mm256_sub_ps(
						curr_here_x,
						_mm256_load_ps(curr[base - vs_z].f)
					);
					curl = _mm256_sub_ps(curl, curr_here_z);
					curl = _mm256_add_ps(curl,
						_mm256_load_ps(curr[base + 2 - i_shift_x].f));

					v = _mm256_fmadd_ps(vv, v, _mm256_mul_ps(vi, curl));
					_mm256_store_ps(volt[base + 1].f, v);
				}

				// Z-polarisation: V_z = vv_z * V_z + vi_z * (dHy/dx - dHx/dy)
				{
					__m256 v  = _mm256_load_ps(volt[base + 2].f);
					__m256 vv = _mm256_load_ps(Op->f8_vv_Comp[2][index].f);
					__m256 vi = _mm256_load_ps(Op->f8_vi_Comp[2][index].f);

					__m256 curl = _mm256_sub_ps(
						curr_here_y,
						_mm256_load_ps(curr[base + 1 - i_shift_x].f)
					);
					curl = _mm256_sub_ps(curl, curr_here_x);
					curl = _mm256_add_ps(curl,
						_mm256_load_ps(curr[base - i_shift_y].f));

					v = _mm256_fmadd_ps(vv, v, _mm256_mul_ps(vi, curl));
					_mm256_store_ps(volt[base + 2].f, v);
				}
			}

			// --- Boundary: z_vec = 0 (z-1 wraps to last vector) ---
			{
				const unsigned int index = Op->m_Op_index(pos[0], pos[1], 0);
				const int base = (int)(pos[0] * m_vs_x + pos[1] * m_vs_y);
				const int end  = base + (int)((numVectors - 1) * m_vs_z);

				// z-1 of Hy and Hx: shift from last vector
				__m256 curr_Hy_zm1 = avx2_shift_left_1f(
					_mm256_load_ps(curr[end + 1].f)
				);
				__m256 curr_Hx_zm1 = avx2_shift_left_1f(
					_mm256_load_ps(curr[end].f)
				);

				const __m256 curr_here_x = _mm256_load_ps(curr[base].f);
				const __m256 curr_here_y = _mm256_load_ps(curr[base + 1].f);
				const __m256 curr_here_z = _mm256_load_ps(curr[base + 2].f);

				// X-polarisation
				{
					__m256 v  = _mm256_load_ps(volt[base].f);
					__m256 vv = _mm256_load_ps(Op->f8_vv_Comp[0][index].f);
					__m256 vi = _mm256_load_ps(Op->f8_vi_Comp[0][index].f);

					__m256 curl = _mm256_sub_ps(
						curr_here_z,
						_mm256_load_ps(curr[base + 2 - i_shift_y].f)
					);
					curl = _mm256_sub_ps(curl, curr_here_y);
					curl = _mm256_add_ps(curl, curr_Hy_zm1);

					v = _mm256_fmadd_ps(vv, v, _mm256_mul_ps(vi, curl));
					_mm256_store_ps(volt[base].f, v);
				}

				// Y-polarisation
				{
					__m256 v  = _mm256_load_ps(volt[base + 1].f);
					__m256 vv = _mm256_load_ps(Op->f8_vv_Comp[1][index].f);
					__m256 vi = _mm256_load_ps(Op->f8_vi_Comp[1][index].f);

					__m256 curl = _mm256_sub_ps(
						curr_here_x,
						curr_Hx_zm1
					);
					curl = _mm256_sub_ps(curl, curr_here_z);
					curl = _mm256_add_ps(curl, _mm256_load_ps(curr[base + 2 - i_shift_x].f));

					v = _mm256_fmadd_ps(vv, v, _mm256_mul_ps(vi, curl));
					_mm256_store_ps(volt[base + 1].f, v);
				}

				// Z-polarisation (no z-shift needed for z-pol voltage update)
				{
					__m256 v  = _mm256_load_ps(volt[base + 2].f);
					__m256 vv = _mm256_load_ps(Op->f8_vv_Comp[2][index].f);
					__m256 vi = _mm256_load_ps(Op->f8_vi_Comp[2][index].f);

					__m256 curl = _mm256_sub_ps(
						curr_here_y,
						_mm256_load_ps(curr[base + 1 - i_shift_x].f)
					);
					curl = _mm256_sub_ps(curl, curr_here_x);
					curl = _mm256_add_ps(curl, _mm256_load_ps(curr[base - i_shift_y].f));

					v = _mm256_fmadd_ps(vv, v, _mm256_mul_ps(vi, curl));
					_mm256_store_ps(volt[base + 2].f, v);
				}
			}
		}
	}
}

// ============================================================================
// Current update with AVX2 + FMA + compressed operator + flat arrays
//
// FDTD current update equation (per polarisation):
//   I_n = ii_n * I_n + iv_n * (curl_E)_n
//
// Uses forward differences (E at +1 positions) for the curl of E.
// ============================================================================

void Engine_AVX2::UpdateCurrents(unsigned int startX, unsigned int numX)
{
	for (unsigned int x = startX; x < startX + numX; ++x)
		UpdateCurrentRows(x, 0, numLines[1] - 1);
}

void Engine_AVX2::UpdateCurrentRows(unsigned int x, unsigned int yStart, unsigned int yStop)
{
	f8vector* __restrict volt = m_volt;
	f8vector* __restrict curr = m_curr;

	const int vs_z = (int)m_vs_z;
	const int vs_y = (int)m_vs_y;
	const int vs_x = (int)m_vs_x;

	unsigned int pos[3];
	pos[0] = x;

	{
		for (pos[1] = yStart; pos[1] < yStop; ++pos[1])
		{
			// --- Main loop: z_vec < numVectors - 1 (z+1 within range) ---
			for (pos[2] = 0; pos[2] < numVectors - 1; ++pos[2])
			{
				const unsigned int index = Op->m_Op_index(pos[0], pos[1], pos[2]);
				const int base = (int)(pos[0] * m_vs_x + pos[1] * m_vs_y + pos[2] * m_vs_z);

				// E-field values at "here" are each used by two of the three
				// curl computations below -- load once and reuse.
				const __m256 volt_here_x = _mm256_load_ps(volt[base].f);
				const __m256 volt_here_y = _mm256_load_ps(volt[base + 1].f);
				const __m256 volt_here_z = _mm256_load_ps(volt[base + 2].f);

				// X-polarisation: I_x = ii_x * I_x + iv_x * (dEz/dy - dEy/dz)
				{
					__m256 c  = _mm256_load_ps(curr[base].f);
					__m256 ii = _mm256_load_ps(Op->f8_ii_Comp[0][index].f);
					__m256 iv = _mm256_load_ps(Op->f8_iv_Comp[0][index].f);

					__m256 curl = _mm256_sub_ps(
						volt_here_z,
						_mm256_load_ps(volt[base + 2 + vs_y].f)
					);
					curl = _mm256_sub_ps(curl, volt_here_y);
					curl = _mm256_add_ps(curl,
						_mm256_load_ps(volt[base + 1 + vs_z].f));

					c = _mm256_fmadd_ps(ii, c, _mm256_mul_ps(iv, curl));
					_mm256_store_ps(curr[base].f, c);
				}

				// Y-polarisation: I_y = ii_y * I_y + iv_y * (dEx/dz - dEz/dx)
				{
					__m256 c  = _mm256_load_ps(curr[base + 1].f);
					__m256 ii = _mm256_load_ps(Op->f8_ii_Comp[1][index].f);
					__m256 iv = _mm256_load_ps(Op->f8_iv_Comp[1][index].f);

					__m256 curl = _mm256_sub_ps(
						volt_here_x,
						_mm256_load_ps(volt[base + vs_z].f)
					);
					curl = _mm256_sub_ps(curl, volt_here_z);
					curl = _mm256_add_ps(curl,
						_mm256_load_ps(volt[base + 2 + vs_x].f));

					c = _mm256_fmadd_ps(ii, c, _mm256_mul_ps(iv, curl));
					_mm256_store_ps(curr[base + 1].f, c);
				}

				// Z-polarisation: I_z = ii_z * I_z + iv_z * (dEy/dx - dEx/dy)
				{
					__m256 c  = _mm256_load_ps(curr[base + 2].f);
					__m256 ii = _mm256_load_ps(Op->f8_ii_Comp[2][index].f);
					__m256 iv = _mm256_load_ps(Op->f8_iv_Comp[2][index].f);

					__m256 curl = _mm256_sub_ps(
						volt_here_y,
						_mm256_load_ps(volt[base + 1 + vs_x].f)
					);
					curl = _mm256_sub_ps(curl, volt_here_x);
					curl = _mm256_add_ps(curl,
						_mm256_load_ps(volt[base + vs_y].f));

					c = _mm256_fmadd_ps(ii, c, _mm256_mul_ps(iv, curl));
					_mm256_store_ps(curr[base + 2].f, c);
				}
			}

			// --- Boundary: z_vec = numVectors - 1 (z+1 wraps to first vector) ---
			{
				const unsigned int posZ = numVectors - 1;
				const unsigned int index = Op->m_Op_index(pos[0], pos[1], posZ);
				const int base = (int)(pos[0] * m_vs_x + pos[1] * m_vs_y + posZ * m_vs_z);
				const int base_z0 = (int)(pos[0] * m_vs_x + pos[1] * m_vs_y);

				// z+1 of Ey and Ex: shift from first vector
				__m256 volt_Ey_zp1 = avx2_shift_right_1f(
					_mm256_load_ps(volt[base_z0 + 1].f)
				);
				__m256 volt_Ex_zp1 = avx2_shift_right_1f(
					_mm256_load_ps(volt[base_z0].f)
				);

				const __m256 volt_here_x = _mm256_load_ps(volt[base].f);
				const __m256 volt_here_y = _mm256_load_ps(volt[base + 1].f);
				const __m256 volt_here_z = _mm256_load_ps(volt[base + 2].f);

				// X-polarisation
				{
					__m256 c  = _mm256_load_ps(curr[base].f);
					__m256 ii = _mm256_load_ps(Op->f8_ii_Comp[0][index].f);
					__m256 iv = _mm256_load_ps(Op->f8_iv_Comp[0][index].f);

					__m256 curl = _mm256_sub_ps(
						volt_here_z,
						_mm256_load_ps(volt[base + 2 + vs_y].f)
					);
					curl = _mm256_sub_ps(curl, volt_here_y);
					curl = _mm256_add_ps(curl, volt_Ey_zp1);

					c = _mm256_fmadd_ps(ii, c, _mm256_mul_ps(iv, curl));
					_mm256_store_ps(curr[base].f, c);
				}

				// Y-polarisation
				{
					__m256 c  = _mm256_load_ps(curr[base + 1].f);
					__m256 ii = _mm256_load_ps(Op->f8_ii_Comp[1][index].f);
					__m256 iv = _mm256_load_ps(Op->f8_iv_Comp[1][index].f);

					__m256 curl = _mm256_sub_ps(
						volt_here_x,
						volt_Ex_zp1
					);
					curl = _mm256_sub_ps(curl, volt_here_z);
					curl = _mm256_add_ps(curl, _mm256_load_ps(volt[base + 2 + vs_x].f));

					c = _mm256_fmadd_ps(ii, c, _mm256_mul_ps(iv, curl));
					_mm256_store_ps(curr[base + 1].f, c);
				}

				// Z-polarisation (no z-shift for z-pol current update)
				{
					__m256 c  = _mm256_load_ps(curr[base + 2].f);
					__m256 ii = _mm256_load_ps(Op->f8_ii_Comp[2][index].f);
					__m256 iv = _mm256_load_ps(Op->f8_iv_Comp[2][index].f);

					__m256 curl = _mm256_sub_ps(
						volt_here_y,
						_mm256_load_ps(volt[base + 1 + vs_x].f)
					);
					curl = _mm256_sub_ps(curl, volt_here_x);
					curl = _mm256_add_ps(curl, _mm256_load_ps(volt[base + vs_y].f));

					c = _mm256_fmadd_ps(ii, c, _mm256_mul_ps(iv, curl));
					_mm256_store_ps(curr[base + 2].f, c);
				}
			}
		}
	}
}
