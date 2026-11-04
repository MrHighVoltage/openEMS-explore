/*
*	Copyright (C) 2026 openEMS contributors
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*/

#ifdef WITH_GPU

#include "operator_vulkan.h"
#include "engine_vulkan.h"

#include <iostream>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <cstdlib>
#include <cmath>
#include <stdexcept>

using std::cout;
using std::endl;

// ---------------------------------------------------------------------------
// Compression key: 12 floats (vv[3], vi[3], ii[3], iv[3]) per cell.
// ---------------------------------------------------------------------------
struct VulkanCoeffKey
{
	float vv[3], vi[3], ii[3], iv[3];
};

struct VulkanCoeffKeyHash
{
	size_t operator()(const VulkanCoeffKey& k) const
	{
		// FNV-1a over the raw bytes preserves exact float-bit identity.
		const unsigned char* p = reinterpret_cast<const unsigned char*>(&k);
		size_t h = sizeof(size_t) == 8
			? (size_t)1469598103934665603ull
			: (size_t)2166136261u;
		const size_t prime = sizeof(size_t) == 8
			? (size_t)1099511628211ull
			: (size_t)16777619u;
		for (size_t i = 0; i < sizeof(VulkanCoeffKey); ++i)
		{
			h ^= (size_t)p[i];
			h *= prime;
		}
		return h;
	}
};

struct VulkanCoeffKeyEq
{
	bool operator()(const VulkanCoeffKey& a, const VulkanCoeffKey& b) const
	{
		return memcmp(&a, &b, sizeof(VulkanCoeffKey)) == 0;
	}
};

// ---------------------------------------------------------------------------

namespace
{

unsigned int GetOperatorThreadCount(unsigned int numX)
{
	unsigned int numThreads = std::thread::hardware_concurrency();
	if (numThreads == 0)
		numThreads = 1;

	const char* env = std::getenv("OPENEMS_OPERATOR_THREADS");
	if (env == nullptr)
		env = std::getenv("OPENEMS_GPU_OPERATOR_THREADS");
	if (env != nullptr)
	{
		char* end = nullptr;
		long parsed = std::strtol(env, &end, 10);
		if ((end != env) && (*end == '\0') && (parsed > 0))
			numThreads = (unsigned int)parsed;
	}

	return std::max(1u, std::min(numThreads, numX));
}

template <typename Work>
void RunOperatorRanges(unsigned int xStart, unsigned int xStop, Work work)
{
	if (xStop < xStart)
		return;

	const unsigned int numX = xStop - xStart + 1;
	const unsigned int numThreads = GetOperatorThreadCount(numX);
	std::vector<std::thread> threads;
	threads.reserve(numThreads > 1 ? numThreads - 1 : 0);

	for (unsigned int t = 1; t < numThreads; ++t)
	{
		const unsigned int begin = xStart + (numX * t) / numThreads;
		const unsigned int end = xStart + (numX * (t + 1)) / numThreads - 1;
		threads.emplace_back(work, t, begin, end);
	}

	work(0, xStart, xStart + numX / numThreads - 1);
	for (auto& thread : threads)
		thread.join();
}

} // namespace

Operator_Vulkan::Operator_Vulkan() : Operator()
{
	m_numCompressed = 0;
	m_Compressed = true;
}

Operator_Vulkan::~Operator_Vulkan()
{
}

Operator_Vulkan* Operator_Vulkan::New()
{
	// openEMS::SetupFDTD() announces the operator once the GPU preflight found a device
	Operator_Vulkan* op = new Operator_Vulkan();
	op->Init();
	return op;
}

Engine* Operator_Vulkan::CreateEngine()
{
	m_Engine = Engine_Vulkan::New(this);
	return m_Engine;
}

int Operator_Vulkan::CalcECOperator(DebugFlags debugFlags)
{
	int errCode = Operator::CalcECOperator(debugFlags);
	CompressOperator();
	return errCode;
}

bool Operator_Vulkan::Calc_EC()
{
	if (CSX == nullptr)
	{
		std::cerr << "Operator_Vulkan::Calc_EC: CSX not given or invalid!!!" << std::endl;
		return false;
	}

	MainOp->SetPos(0, 0, 0);
	const unsigned int numThreads = GetOperatorThreadCount(numLines[0]);
	if (g_settings.GetVerboseLevel() > 0)
		cout << "  GPU operator material threads: " << numThreads << endl;

	RunOperatorRanges(0, numLines[0] - 1,
		[this](unsigned int, unsigned int begin, unsigned int end)
		{
			Operator::Calc_EC_Range(begin, end);
		});
	return true;
}

bool Operator_Vulkan::CalcPEC()
{
	m_Nr_PEC[0] = 0;
	m_Nr_PEC[1] = 0;
	m_Nr_PEC[2] = 0;

	const unsigned int numThreads = GetOperatorThreadCount(numLines[0]);
	if (g_settings.GetVerboseLevel() > 0)
		cout << "  GPU operator PEC threads: " << numThreads << endl;

	std::vector<std::array<unsigned int, 3>> counts(numThreads);
	for (auto& count : counts)
		count = {0, 0, 0};

	RunOperatorRanges(0, numLines[0] - 1,
		[this, &counts](unsigned int threadID, unsigned int begin, unsigned int end)
		{
			Operator::CalcPEC_Range(begin, end, counts[threadID].data());
		});

	for (const auto& count : counts)
		for (int n = 0; n < 3; ++n)
			m_Nr_PEC[n] += count[n];

	CalcPEC_Curves();
	return true;
}

void Operator_Vulkan::Calc_ECOperator_Range(unsigned int xStart, unsigned int xStop)
{
	const unsigned int numThreads = GetOperatorThreadCount(xStop - xStart + 1);
	if (g_settings.GetVerboseLevel() > 0)
		cout << "  GPU operator coefficient threads: " << numThreads << endl;

	RunOperatorRanges(xStart, xStop,
		[this](unsigned int, unsigned int begin, unsigned int end)
		{
			Operator::Calc_ECOperator_Range(begin, end);
		});
}

void Operator_Vulkan::CompressOperator()
{
	auto tCompressStart = std::chrono::steady_clock::now();
	bool logDetails = (g_settings.GetVerboseLevel() > 0);
	if (const char* env = std::getenv("OPENEMS_GPU_STARTUP_TRACE"))
	{
		char* end = nullptr;
		long parsed = std::strtol(env, &end, 10);
		if ((end != env) && (*end == '\0'))
			logDetails = (parsed != 0);
	}

	if (g_settings.GetVerboseLevel() > 0)
		cout << "Compressing the FDTD operator (GPU)... this may take a while..." << endl;

	size_t N = (size_t)numLines[0] * numLines[1] * numLines[2];

	m_OpIndex.resize(N);

	// Access base class coefficient arrays (N-I-J-K layout)
	const float* vv_data = (const float*)vv_ptr->data();
	const float* vi_data = (const float*)vi_ptr->data();
	const float* ii_data = (const float*)ii_ptr->data();
	const float* iv_data = (const float*)iv_ptr->data();

	using ShardMap = std::unordered_map<VulkanCoeffKey, uint32_t, VulkanCoeffKeyHash, VulkanCoeffKeyEq>;
	unsigned int hwThreads = std::thread::hardware_concurrency();
	if (hwThreads == 0)
		hwThreads = 1;

	bool forcedThreads = false;
	unsigned int numThreads = hwThreads;
	if (const char* env = std::getenv("OPENEMS_GPU_COMPRESS_THREADS"))
	{
		char* end = nullptr;
		long parsed = std::strtol(env, &end, 10);
		if ((end != env) && (*end == '\0') && (parsed > 0))
		{
			numThreads = (unsigned int)parsed;
			forcedThreads = true;
		}
	}

	if (numThreads > hwThreads)
		numThreads = hwThreads;

	if (!forcedThreads && (N < (size_t)numThreads * 4096))
		numThreads = std::max(1u, (unsigned int)(N / 4096));

	if ((N > 0) && (numThreads > N))
		numThreads = (unsigned int)N;

	if (numThreads == 0)
		numThreads = 1;

	if (logDetails)
		cout << "  Compression cells: " << N << " (" << numLines[0] << "x" << numLines[1] << "x" << numLines[2] << ")" << endl;

	if (g_settings.GetVerboseLevel() > 0)
		cout << "  Compression threads: " << numThreads
		     << (forcedThreads ? " (forced by OPENEMS_GPU_COMPRESS_THREADS)" : " (auto)")
		     << endl;

	size_t numShards = 1;
	while (numShards < (size_t)numThreads * 4)
		numShards <<= 1;
	if (logDetails)
		cout << "  Compression shards: " << numShards << endl;

	std::vector<ShardMap> shards(numShards);
	std::vector<std::mutex> shardLocks(numShards);
	std::mutex uniqueLock;
	std::vector<VulkanCoeffKey> uniqueKeys;
	uniqueKeys.reserve(1024);

	auto worker = [&](size_t startCell, size_t endCell)
	{
		VulkanCoeffKeyHash hasher;
		for (size_t cell = startCell; cell < endCell; ++cell)
		{
			VulkanCoeffKey key;
			for (int n = 0; n < 3; ++n)
			{
				key.vv[n] = vv_data[n * N + cell];
				key.vi[n] = vi_data[n * N + cell];
				key.ii[n] = ii_data[n * N + cell];
				key.iv[n] = iv_data[n * N + cell];
			}

			size_t h = hasher(key);
			size_t shardIdx = h & (numShards - 1);
			uint32_t idx;

			{
				std::lock_guard<std::mutex> lg(shardLocks[shardIdx]);
				auto& shard = shards[shardIdx];
				auto it = shard.find(key);
				if (it != shard.end())
				{
					idx = it->second;
				}
				else
				{
					std::lock_guard<std::mutex> ug(uniqueLock);
					idx = (uint32_t)uniqueKeys.size();
					uniqueKeys.push_back(key);
					shard.emplace(key, idx);
				}
			}

			m_OpIndex[cell] = idx;
		}
	};

	std::vector<std::thread> threads;
	auto tBuildStart = std::chrono::steady_clock::now();
	threads.reserve(numThreads > 0 ? numThreads - 1 : 0);
	for (unsigned int t = 1; t < numThreads; ++t)
	{
		size_t start = (N * t) / numThreads;
		size_t stop  = (N * (t + 1)) / numThreads;
		threads.emplace_back(worker, start, stop);
	}
	worker(0, N / numThreads);
	for (auto& th : threads)
		th.join();
	auto tBuildEnd = std::chrono::steady_clock::now();

	auto validateFinite = [](const std::vector<float>& values, const char* name)
	{
		for (size_t i = 0; i < values.size(); ++i)
		{
			if (!std::isfinite(values[i]))
				throw std::runtime_error(std::string("Operator_Vulkan: non-finite ") +
				                         name + " coefficient at index " + std::to_string(i));
		}
	};

	const size_t uniqueCount = uniqueKeys.size();
	m_numCompressed = (unsigned int)uniqueKeys.size();

	// ---- Is deduplication actually worth it? --------------------------------
	// The shaders reach the coefficients through six independent random gathers
	// per cell (vv/vi/ii/iv, each component-major). While the unique table stays
	// resident in GPU cache those gathers are nearly free and compression is a
	// large win -- a uniform mesh collapses to 6 unique sets and reads only the
	// narrow per-cell index. Once the table outgrows the cache, every gather
	// costs a separate cache line, so a cell that would have streamed 48 bytes
	// of contiguous coefficients instead pulls six scattered lines plus its
	// index. That is exactly what a graded mesh produces: coefficients scale
	// with the local edge length and area, so almost every cell is unique
	// (measured: 493040 of 512000 on an 80^3 power-law mesh, a 23 MB table)
	// and the engine ran 4x slower than the same grid uniformly meshed.
	//
	// So when the table would be too large to cache, drop the indirection and
	// store one entry per cell. The per-cell index then degenerates to the
	// identity and is not uploaded at all (OPIDX_BITS == 0 in the shaders),
	// which also saves the index traffic. The tables in that mode are just the
	// base arrays, which already use the component-major layout the shaders
	// want, so this costs a plain copy and no rearrangement.
	//
	// Measured crossover (80^3 PEC, meshes built from k distinct spacings per
	// axis so the unique count is controllable, Intel HD 530):
	//
	//   table    compressed   per-cell
	//      5 KB   388.6 MC/s   256.4 MC/s
	//    227 KB   345.3        251.3
	//   1824 KB   288.0        257.2
	//   4539 KB   230.6        257.6   <- compression starts losing
	//  11712 KB   159.0        258.5
	//  21955 KB    99.4        260.6
	//
	// Note the per-cell column is flat: it is the streaming bandwidth floor and
	// does not care how varied the mesh is. The real crossover is near 3 MB; the
	// budget sits below it because the two failure directions are not
	// symmetric. Switching too early costs at most ~11% (the 1824 KB row),
	// while staying compressed too long falls off a cliff -- 62% at 22 MB.
	// Vulkan exposes no cache size, so this is a tuned constant rather than a
	// derived one, and it can be overridden for measurement.
	size_t maxTableKB = 2048;
	if (const char* env = std::getenv("OPENEMS_GPU_COMPRESS_MAX_TABLE_KB"))
	{
		char* end = nullptr;
		long parsed = std::strtol(env, &end, 10);
		if ((end != env) && (*end == '\0') && (parsed >= 0))
			maxTableKB = (size_t)parsed;
	}
	const size_t tableBytes = 4 * 3 * (size_t)m_numCompressed * sizeof(float);
	// Never expand: one entry per cell is the worst the identity mode can be.
	m_Compressed = (m_numCompressed < N) && (tableBytes <= maxTableKB * 1024);

	if (!m_Compressed)
	{
		std::vector<VulkanCoeffKey>().swap(uniqueKeys);
		std::vector<ShardMap>().swap(shards);
		std::vector<uint32_t>().swap(m_OpIndex);
		m_numCompressed = (unsigned int)N;

		m_vvComp.assign(vv_data, vv_data + 3 * N);
		m_viComp.assign(vi_data, vi_data + 3 * N);
		m_iiComp.assign(ii_data, ii_data + 3 * N);
		m_ivComp.assign(iv_data, iv_data + 3 * N);

		validateFinite(m_vvComp, "VV");
		validateFinite(m_viComp, "VI");
		validateFinite(m_iiComp, "II");
		validateFinite(m_ivComp, "IV");

		if (g_settings.GetVerboseLevel() > 0)
			cout << "  Unique coefficient sets: " << uniqueCount << " of " << N
			     << " -- table would be " << (tableBytes / 1024)
			     << " KB, over the " << maxTableKB
			     << " KB cache budget; storing one entry per cell instead" << endl;
		return;
	}

	// Rearrange to N-major layout: comp[n * numCompressed + idx]
	// Currently tmp_vv is packed as [vv0_0, vv1_0, vv2_0, vv0_1, vv1_1, vv2_1, ...]
	// Need: [vv0_0, vv0_1, ..., vv0_M, vv1_0, vv1_1, ..., vv1_M, vv2_0, ...]
	m_vvComp.resize(3 * m_numCompressed);
	m_viComp.resize(3 * m_numCompressed);
	m_iiComp.resize(3 * m_numCompressed);
	m_ivComp.resize(3 * m_numCompressed);

	auto writeCompRange = [&](size_t idxStart, size_t idxEnd)
	{
		for (size_t idx = idxStart; idx < idxEnd; ++idx)
		{
			const VulkanCoeffKey& key = uniqueKeys[idx];
			for (int n = 0; n < 3; ++n)
			{
				m_vvComp[n * m_numCompressed + idx] = key.vv[n];
				m_viComp[n * m_numCompressed + idx] = key.vi[n];
				m_iiComp[n * m_numCompressed + idx] = key.ii[n];
				m_ivComp[n * m_numCompressed + idx] = key.iv[n];
			}
		}
	};

	auto tPackStart = std::chrono::steady_clock::now();
	if (numThreads > 1 && m_numCompressed > 4096)
	{
		std::vector<std::thread> outThreads;
		outThreads.reserve(numThreads - 1);
		for (unsigned int t = 1; t < numThreads; ++t)
		{
			size_t start = ((size_t)m_numCompressed * t) / numThreads;
			size_t stop  = ((size_t)m_numCompressed * (t + 1)) / numThreads;
			outThreads.emplace_back(writeCompRange, start, stop);
		}
		writeCompRange(0, (size_t)m_numCompressed / numThreads);
		for (auto& th : outThreads)
			th.join();
	}
	else
	{
		writeCompRange(0, m_numCompressed);
	}

	validateFinite(m_vvComp, "VV");
	validateFinite(m_viComp, "VI");
	validateFinite(m_iiComp, "II");
	validateFinite(m_ivComp, "IV");
	auto tPackEnd = std::chrono::steady_clock::now();
	auto tCompressEnd = std::chrono::steady_clock::now();

	if (logDetails)
	{
		double buildMs = std::chrono::duration<double, std::milli>(tBuildEnd - tBuildStart).count();
		double packMs = std::chrono::duration<double, std::milli>(tPackEnd - tPackStart).count();
		double totalMs = std::chrono::duration<double, std::milli>(tCompressEnd - tCompressStart).count();
		cout << "  Compression timing: build-index=" << buildMs
		     << " ms, pack-tables=" << packMs
		     << " ms, total=" << totalMs << " ms" << endl;
	}

	if (g_settings.GetVerboseLevel() > 0)
	{
		size_t origBytes = 4 * 3 * N * sizeof(float);
		size_t compBytes = N * sizeof(uint32_t) + 4 * 3 * m_numCompressed * sizeof(float);
		cout << "  Unique coefficient sets: " << m_numCompressed
		     << " (compression ratio: " << (double)origBytes / compBytes << "x)" << endl;
		cout << "  Original: " << (origBytes / (1024*1024)) << " MB -> Compressed: "
		     << (compBytes / (1024*1024)) << " MB" << endl;
	}
}

void Operator_Vulkan::ShowStat() const
{
	Operator::ShowStat();

	cout << "GPU operator compression\t: " << (m_Compressed ? "yes" : "no (one entry per cell)") << endl;
	if (m_Compressed)
		cout << "Unique coefficient sets\t\t: " << m_numCompressed << endl;
	cout << "-----------------------------------" << endl;
}

#endif // WITH_GPU
