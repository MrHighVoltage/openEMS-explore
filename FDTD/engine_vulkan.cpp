/*
*	Copyright (C) 2026 openEMS contributors
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*/

#ifdef WITH_GPU

#include "engine_vulkan.h"
#include "operator_vulkan.h"
#include "operator.h"
#include "extensions/engine_extension.h"
#include "extensions/engine_ext_excitation.h"
#include "extensions/operator_ext_excitation.h"
#include "extensions/engine_ext_upml.h"
#include "extensions/operator_ext_upml.h"
#include "extensions/engine_ext_lorentzmaterial.h"
#include "extensions/operator_ext_lorentzmaterial.h"
#include "extensions/engine_ext_debyematerial.h"
#include "extensions/operator_ext_debyematerial.h"
#include "extensions/engine_ext_tfsf.h"
#include "extensions/operator_ext_tfsf.h"
#include "extensions/engine_ext_mur_abc.h"
#include "extensions/operator_ext_mur_abc.h"
#include "extensions/engine_ext_lumpedRLC.h"
#include "extensions/operator_ext_lumpedRLC.h"
#include "extensions/engine_ext_steadystate.h"
#include "extensions/operator_ext_steadystate.h"
#include "extensions/engine_ext_absorbing_bc.h"
#include "extensions/operator_ext_absorbing_bc.h"
#include "Common/processing.h"
#include "gpu_shader_spirv.h"
#include "tools/constants.h"
#include "tools/cpu_topology.h"

#include <iostream>
#include <cstring>
#include <unordered_map>
#include <stdexcept>
#include <functional>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <iomanip>
#include <sstream>

using std::cerr;
using std::cout;
using std::endl;

bool Engine_Vulkan::s_enableProfiling = false;
const char* const Engine_Vulkan::s_profileStageNames[Engine_Vulkan::PROFILE_STAGES] = {
	"pre-V ext", "Yee V", "post-V ext", "V apply+exc", "pre-I ext", "Yee I", "post-I+exc"};

// ---------------------------------------------------------------------------
// Vulkan error checking
// ---------------------------------------------------------------------------
static const char* VkResultName(VkResult result)
{
	switch (result)
	{
		case VK_SUCCESS: return "VK_SUCCESS";
		case VK_NOT_READY: return "VK_NOT_READY";
		case VK_TIMEOUT: return "VK_TIMEOUT";
		case VK_EVENT_SET: return "VK_EVENT_SET";
		case VK_EVENT_RESET: return "VK_EVENT_RESET";
		case VK_INCOMPLETE: return "VK_INCOMPLETE";
		case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
		case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
		case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
		case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
		case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
		case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
		case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
		case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
		case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
		case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
		case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
		case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
		case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
		case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
		default: return "VK_ERROR_UNRECOGNIZED";
	}
}

#define VK_CHECK(call) \
	do { \
		VkResult _r = (call); \
		if (_r != VK_SUCCESS) { \
			cerr << "Vulkan error " << VkResultName(_r) << " (" << _r << ") at " << __FILE__ << ":" << __LINE__ << endl; \
			throw std::runtime_error("Vulkan call failed"); \
		} \
	} while(0)

// ===========================================================================
// Narrow index packing
// ===========================================================================

//! Pack indices into `bits`-wide fields, little-endian within each dword.
//! Mirrors fetchOpIdx()/fetchPmlIdx() in the shaders, which always load a full
//! dword, so the buffer is rounded up to whole dwords and the tail stays zero.
static std::vector<uint32_t> PackNarrowIndices(const uint32_t* src, size_t count,
                                               uint32_t bits)
{
	if (bits == 32)
		return std::vector<uint32_t>(src, src + count);
	const uint32_t perWord = 32u / bits;
	std::vector<uint32_t> packed((count + perWord - 1) / perWord, 0u);
	for (size_t c = 0; c < count; ++c)
		packed[c / perWord] |= (src[c] << ((c % perWord) * bits));
	return packed;
}

//! Byte size of a `count`-entry index buffer at `bits` per entry, rounded up to
//! whole dwords. Never zero: a descriptor may not point at an empty range.
static VkDeviceSize NarrowIndexBufSize(size_t count, uint32_t bits)
{
	const size_t perWord = 32u / bits;
	VkDeviceSize words = (count + perWord - 1) / perWord;
	if (words == 0) words = 1;
	return words * sizeof(uint32_t);
}

//! Deduplication key for one UPML coefficient entry: the six coefficients that
//! the pre/post voltage and current passes read for a single cell-component.
//! Compared by bit pattern -- the values are validated finite before this runs.
namespace {
struct PmlCoeffKey
{
	uint32_t w[6];
	bool operator==(const PmlCoeffKey& o) const
	{
		return std::memcmp(w, o.w, sizeof(w)) == 0;
	}
};
struct PmlCoeffKeyHash
{
	size_t operator()(const PmlCoeffKey& k) const
	{
		size_t h = 1469598103934665603ull;
		for (int i = 0; i < 6; ++i)
		{
			h ^= k.w[i];
			h *= 1099511628211ull;
		}
		return h;
	}
};
} // namespace

// ===========================================================================
// Construction / Destruction
// ===========================================================================

Engine_Vulkan* Engine_Vulkan::New(const Operator* op)
{
	Engine_Vulkan* e = new Engine_Vulkan(op);
	e->Init();
	// upstream's engine line, which benchmark/bench.py reads the backend from
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(e->m_physDevice, &props);
	cout << "Create FDTD engine (GPU, backend: Vulkan (" << props.deviceName << "))" << endl;
	return e;
}

// ---------------------------------------------------------------------------
// Vulkan version negotiation
//
// The engine has a single baseline (Engine_Vulkan::API_VERSION).  Requesting
// an apiVersion the *loader* does not understand makes vkCreateInstance fail
// with VK_ERROR_INCOMPATIBLE_DRIVER on a 1.0 loader, so clamp the request to
// what the loader reports before asking for it, then reject individual
// physical devices whose own apiVersion is below the baseline.
// ---------------------------------------------------------------------------
static uint32_t VulkanLoaderVersion()
{
	auto fn = (PFN_vkEnumerateInstanceVersion)
		vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion");
	if (!fn)
		return VK_API_VERSION_1_0;   // 1.0 loader: the symbol does not exist
	uint32_t ver = VK_API_VERSION_1_0;
	if (fn(&ver) != VK_SUCCESS)
		return VK_API_VERSION_1_0;
	return ver;
}

//! apiVersion to put in VkApplicationInfo: never above what the loader supports.
static uint32_t VulkanRequestedApiVersion()
{
	const uint32_t loader = VulkanLoaderVersion();
	return (loader < Engine_Vulkan::API_VERSION) ? loader : Engine_Vulkan::API_VERSION;
}

//! True when this physical device itself implements the engine baseline.
static bool VulkanDeviceMeetsBaseline(VkPhysicalDevice dev)
{
	VkPhysicalDeviceProperties props{};
	vkGetPhysicalDeviceProperties(dev, &props);
	return props.apiVersion >= Engine_Vulkan::API_VERSION;
}

static std::string VulkanVersionString(uint32_t v)
{
	std::ostringstream os;
	os << VK_API_VERSION_MAJOR(v) << '.' << VK_API_VERSION_MINOR(v)
	   << '.' << VK_API_VERSION_PATCH(v);
	return os.str();
}

namespace {
//! A device the engine can run on: it implements the engine baseline and has a compute queue.
struct VulkanComputeDevice
{
	VkPhysicalDevice dev;
	uint32_t computeQueueFamily;
	VkPhysicalDeviceProperties props;
	VkDeviceSize memory;    //!< the largest device-local heap
};

//! The devices the engine can run on, in vkEnumeratePhysicalDevices() order: the
//! numbers of --gpu-device and OPENEMS_GPU_INDEX. \a sawBelowBaseline tells whether a
//! device was left out for implementing an older Vulkan version.
std::vector<VulkanComputeDevice> ListComputeDevices(VkInstance instance, bool& sawBelowBaseline)
{
	sawBelowBaseline = false;
	uint32_t count = 0;
	vkEnumeratePhysicalDevices(instance, &count, nullptr);
	std::vector<VkPhysicalDevice> devs(count);
	if (count > 0)
		vkEnumeratePhysicalDevices(instance, &count, devs.data());

	std::vector<VulkanComputeDevice> devices;
	for (VkPhysicalDevice d : devs)
	{
		// the single engine baseline is enforced here rather than discovering a
		// missing 1.3 feature later, halfway through pipeline creation
		if (!VulkanDeviceMeetsBaseline(d)) { sawBelowBaseline = true; continue; }
		uint32_t qfCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(d, &qfCount, nullptr);
		std::vector<VkQueueFamilyProperties> qfProps(qfCount);
		vkGetPhysicalDeviceQueueFamilyProperties(d, &qfCount, qfProps.data());
		for (uint32_t i = 0; i < qfCount; ++i)
		{
			if (!(qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT))
				continue;
			VulkanComputeDevice c{};
			c.dev = d;
			c.computeQueueFamily = i;
			vkGetPhysicalDeviceProperties(d, &c.props);
			VkPhysicalDeviceMemoryProperties mp{};
			vkGetPhysicalDeviceMemoryProperties(d, &mp);
			for (uint32_t h = 0; h < mp.memoryHeapCount; ++h)
				if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
					c.memory = std::max(c.memory, mp.memoryHeaps[h].size);
			devices.push_back(c);
			break;
		}
	}
	return devices;
}

//! "n: name (Vulkan, x.x GiB[, integrated])", the device line of upstream's GPU engine
std::string DeviceLine(size_t n, const VulkanComputeDevice& d)
{
	std::ostringstream line;
	line << n << ": " << d.props.deviceName << " (Vulkan, " << std::fixed << std::setprecision(1)
	     << d.memory / 1073741824.0 << " GiB";
	if (d.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
		line << ", integrated";
	else if (d.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
		line << ", CPU";
	line << ")";
	return line.str();
}

//! The device to run on: --gpu-device, else OPENEMS_GPU_INDEX, else the first discrete GPU,
//! else the first device. An invalid --gpu-device throws; an invalid OPENEMS_GPU_INDEX
//! falls back to the automatic choice, with a warning if \a report.
size_t ChooseComputeDevice(const std::vector<VulkanComputeDevice>& devices, bool report)
{
	const int chosen = g_settings.GetGPUDevice();
	if (chosen >= 0)
	{
		if (chosen < (int)devices.size())
			return chosen;
		std::ostringstream err;
		err << "--gpu-device=" << chosen << ", but there " << (devices.size()==1 ? "is " : "are ")
		    << devices.size() << " GPU device" << (devices.size()==1 ? "" : "s");
		for (size_t n = 0; n < devices.size(); ++n)
			err << (n==0 ? ": " : ", ") << DeviceLine(n, devices[n]);
		throw std::runtime_error(err.str());
	}
	if (const char* env = std::getenv("OPENEMS_GPU_INDEX"))
	{
		char* end = nullptr;
		const long idx = std::strtol(env, &end, 10);
		if (end && *end == '\0' && idx >= 0 && idx < (long)devices.size())
			return idx;
		if (report)
			cerr << "Engine_Vulkan: invalid OPENEMS_GPU_INDEX='" << env << "' (valid range: 0.."
			     << (int)devices.size()-1 << "), using automatic GPU selection." << endl;
	}
	for (size_t n = 0; n < devices.size(); ++n)
		if (devices[n].props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
			return n;
	return 0;
}
} // namespace

bool Engine_Vulkan::PreflightAllocationForGrid(unsigned int Nx, unsigned int Ny, unsigned int Nz,
                                               bool strictCoeffReserve, std::string* errMsg)
{
	VkInstance instance = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;
	VkPhysicalDevice phys = VK_NULL_HANDLE;
	VkBuffer bufA = VK_NULL_HANDLE, bufB = VK_NULL_HANDLE, bufStaging = VK_NULL_HANDLE, bufCoeff = VK_NULL_HANDLE;
	VkDeviceMemory memA = VK_NULL_HANDLE, memB = VK_NULL_HANDLE, memStaging = VK_NULL_HANDLE, memCoeff = VK_NULL_HANDLE;

	auto cleanup = [&]()
	{
		if (bufCoeff) vkDestroyBuffer(device, bufCoeff, nullptr);
		if (memCoeff) vkFreeMemory(device, memCoeff, nullptr);
		if (bufStaging) vkDestroyBuffer(device, bufStaging, nullptr);
		if (memStaging) vkFreeMemory(device, memStaging, nullptr);
		if (bufB) vkDestroyBuffer(device, bufB, nullptr);
		if (memB) vkFreeMemory(device, memB, nullptr);
		if (bufA) vkDestroyBuffer(device, bufA, nullptr);
		if (memA) vkFreeMemory(device, memA, nullptr);
		if (device) vkDestroyDevice(device, nullptr);
		if (instance) vkDestroyInstance(instance, nullptr);
	};
	auto fail = [&](const std::string& msg) -> bool
	{
		cleanup();
		if (errMsg) *errMsg = msg;
		return false;
	};

	try
	{
		VkApplicationInfo app{};
		app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		app.pApplicationName = "openEMS Vulkan preflight";
		app.apiVersion = VulkanRequestedApiVersion();

		VkInstanceCreateInfo ici{};
		ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		ici.pApplicationInfo = &app;
		if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS)
			return fail("Vulkan preflight: failed to create Vulkan instance");

		bool sawBelowBaseline = false;
		const std::vector<VulkanComputeDevice> devices = ListComputeDevices(instance, sawBelowBaseline);
		if (devices.empty())
		{
			if (sawBelowBaseline)
				return fail("Vulkan preflight: no GPU implementing Vulkan " +
				            VulkanVersionString(Engine_Vulkan::API_VERSION) +
				            " found (a Vulkan-capable device is present but reports an "
				            "older version -- update the graphics driver)");
			return fail("Vulkan preflight: no compute-capable GPU found");
		}
		// upstream's device list, also with --no-simulation (benchmark/bench.py reads it);
		// the preflight runs once, before the engine opens the device
		if (g_settings.GetVerboseLevel() > 0)
		{
			cout << "GPU devices:" << endl;
			for (size_t n = 0; n < devices.size(); ++n)
				cout << "  " << DeviceLine(n, devices[n]) << endl;
		}
		size_t chosen = 0;
		try
		{
			chosen = ChooseComputeDevice(devices, true);
		}
		catch (const std::exception& e)
		{
			return fail(e.what());
		}
		phys = devices[chosen].dev;
		const uint32_t computeQf = devices[chosen].computeQueueFamily;

		float priority = 1.0f;
		VkDeviceQueueCreateInfo qci{};
		qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		qci.queueFamilyIndex = computeQf;
		qci.queueCount = 1;
		qci.pQueuePriorities = &priority;

		VkDeviceCreateInfo dci{};
		dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		dci.queueCreateInfoCount = 1;
		dci.pQueueCreateInfos = &qci;
		if (vkCreateDevice(phys, &dci, nullptr, &device) != VK_SUCCESS)
			return fail("Vulkan preflight: failed to create logical device");

		auto findMemType = [&](uint32_t bits, VkMemoryPropertyFlags flags) -> uint32_t
		{
			VkPhysicalDeviceMemoryProperties mp{};
			vkGetPhysicalDeviceMemoryProperties(phys, &mp);
			for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
			{
				if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & flags) == flags)
					return i;
			}
			return UINT32_MAX;
		};

		auto allocBuffer = [&](VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memFlags,
		                      VkBuffer& outBuf, VkDeviceMemory& outMem) -> bool
		{
			VkBufferCreateInfo bci{};
			bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bci.size = size;
			bci.usage = usage;
			bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			if (vkCreateBuffer(device, &bci, nullptr, &outBuf) != VK_SUCCESS)
				return false;

			VkMemoryRequirements req{};
			vkGetBufferMemoryRequirements(device, outBuf, &req);
			uint32_t memType = findMemType(req.memoryTypeBits, memFlags);
			if (memType == UINT32_MAX)
				return false;

			VkMemoryAllocateInfo mai{};
			mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			mai.allocationSize = req.size;
			mai.memoryTypeIndex = memType;
			if (vkAllocateMemory(device, &mai, nullptr, &outMem) != VK_SUCCESS)
				return false;

			if (vkBindBufferMemory(device, outBuf, outMem, 0) != VK_SUCCESS)
				return false;

			return true;
		};

		const uint64_t N = (uint64_t)Nx * (uint64_t)Ny * (uint64_t)Nz;
		const VkDeviceSize fieldSize = (VkDeviceSize)(3ull * N * sizeof(FDTD_FLOAT));
		const VkBufferUsageFlags fieldUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
		                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
		                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		const VkBufferUsageFlags stagingUsage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
		                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		const VkMemoryPropertyFlags dLoc = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
		                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

		if (!allocBuffer(fieldSize, fieldUsage, dLoc, bufA, memA) ||
		    !allocBuffer(fieldSize, fieldUsage, dLoc, bufB, memB))
			return fail("Vulkan preflight: unable to allocate field buffers on selected GPU");

		if (!allocBuffer(fieldSize, stagingUsage, host, bufStaging, memStaging))
			return fail("Vulkan preflight: unable to allocate staging buffer on selected GPU/host memory");

		if (strictCoeffReserve)
		{
			const VkDeviceSize coeffUpper = (VkDeviceSize)(N * (uint64_t)sizeof(uint32_t) +
			                                               4ull * 3ull * N * (uint64_t)sizeof(float));
			const VkBufferUsageFlags coeffUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
			                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT;
			if (!allocBuffer(coeffUpper, coeffUsage, dLoc, bufCoeff, memCoeff))
				return fail("Vulkan preflight: strict coefficient reserve failed (GPU memory likely insufficient)");
		}

		cleanup();
		return true;
	}
	catch (...)
	{
		return fail("Vulkan preflight: unexpected error while probing GPU allocations");
	}
}

Engine_Vulkan::Engine_Vulkan(const Operator* op) : Engine(op)
{
	m_type = GPU;

	m_instance       = VK_NULL_HANDLE;
	m_physDevice     = VK_NULL_HANDLE;
	m_device         = VK_NULL_HANDLE;
	m_computeQueue   = VK_NULL_HANDLE;
	m_computeQueueFamily = 0;

	m_voltBuf = m_currBuf = VK_NULL_HANDLE;
	m_voltMem = m_currMem = VK_NULL_HANDLE;
	m_opIndexBuf = VK_NULL_HANDLE;
	m_opIndexMem = VK_NULL_HANDLE;
	m_vvCompBuf = m_viCompBuf = m_iiCompBuf = m_ivCompBuf = VK_NULL_HANDLE;
	m_vvCompMem = m_viCompMem = m_iiCompMem = m_ivCompMem = VK_NULL_HANDLE;
	m_opIndexBufSize = 0;
	m_coeffCompBufSize = 0;
	m_stagingBuf = VK_NULL_HANDLE;
	m_stagingMem = VK_NULL_HANDLE;
	m_fieldBufSize = 0;
	m_fieldN     = 0;
	m_strideYZ   = 0;

	m_excVoltIdxBuf = m_excVoltAmpBuf = m_excVoltDelayBuf = VK_NULL_HANDLE;
	m_excVoltIdxMem = m_excVoltAmpMem = m_excVoltDelayMem = VK_NULL_HANDLE;
	m_excCurrIdxBuf = m_excCurrAmpBuf = m_excCurrDelayBuf = VK_NULL_HANDLE;
	m_excCurrIdxMem = m_excCurrAmpMem = m_excCurrDelayMem = VK_NULL_HANDLE;
	m_excSignalVoltBuf = m_excSignalCurrBuf = VK_NULL_HANDLE;
	m_excSignalVoltMem = m_excSignalCurrMem = VK_NULL_HANDLE;
	m_excVoltCount = m_excCurrCount = 0;
	m_excSignalLen = 0;
	m_excSignalPeriod = 0;

	m_fdtdDescLayout = VK_NULL_HANDLE;
	m_excDescLayout  = VK_NULL_HANDLE;
	m_descPool       = VK_NULL_HANDLE;
	m_updateVoltDescSet = m_updateCurrDescSet = VK_NULL_HANDLE;
	m_excVoltDescSet = m_excCurrDescSet = VK_NULL_HANDLE;

	m_fdtdPipeLayout = VK_NULL_HANDLE;
	m_excPipeLayout  = VK_NULL_HANDLE;
	m_updateVoltPipeline = m_updateCurrPipeline = m_excPipeline = VK_NULL_HANDLE;

	m_cmdPool = VK_NULL_HANDLE;
	for (int i = 0; i < CMD_RING_SIZE; ++i)
		m_cmdBufs[i] = VK_NULL_HANDLE;
	m_utilCmdBuf = VK_NULL_HANDLE;
	for (int i = 0; i < CMD_RING_SIZE; ++i)
		m_fences[i] = VK_NULL_HANDLE;
	m_utilFence = VK_NULL_HANDLE;
	m_cmdIdx = 0;
	m_tbK = 0;
	m_tbW = 0;
	m_tbActive = false;
	m_tbAnnounced = false;
	m_tbQuiet = false;
	for (int i = 0; i < CMD_RING_SIZE; ++i)
		m_gpuInFlight[i] = false;

	m_hostDirty   = false;
	m_hasGPUExcitation  = false;
	m_validateFields    = false;
	m_validateStages    = false;
	m_validateMagnitude = false;
	m_validateTraceIndex = std::numeric_limits<uint32_t>::max();
	m_gpuDrained  = true;
	m_pipelineCache = VK_NULL_HANDLE;
	m_maxTSPerSubmit = 64;
	m_chunkProgressIntervalSec = 4.0;
	m_chunkProgressHasLastPrint = false;
}

Engine_Vulkan::~Engine_Vulkan()
{
	DrainGPU();
	Reset();
}

// ===========================================================================
// Init / Reset
// ===========================================================================

void Engine_Vulkan::Init()
{
	bool startupTrace = (g_settings.GetVerboseLevel() > 0);
	if (const char* env = std::getenv("OPENEMS_GPU_STARTUP_TRACE"))
	{
		char* endPtr = nullptr;
		long v = std::strtol(env, &endPtr, 10);
		if (endPtr && *endPtr == '\0')
			startupTrace = (v != 0);
	}

	auto startupBegin = std::chrono::steady_clock::now();
	unsigned int startupStep = 0;
	auto runStartupStep = [&](const char* label, const std::function<void()>& fn)
	{
		auto t0 = std::chrono::steady_clock::now();
		if (startupTrace)
			cout << "[Vulkan startup] step " << (++startupStep) << ": " << label << "..." << endl;
		fn();
		auto t1 = std::chrono::steady_clock::now();
		if (startupTrace)
		{
			double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
			cout << "[Vulkan startup] done " << label << " in " << ms << " ms" << endl;
		}
	};

	// Base class allocates CPU arrays (volt_ptr, curr_ptr) and extensions
	runStartupStep("Engine::Init", [&]() { Engine::Init(); });

	// Read once: pure-GPU submit chunk size to avoid overly long command buffers.
	if (const char* env = std::getenv("OPENEMS_GPU_MAX_TS_PER_SUBMIT"))
	{
		char* endPtr = nullptr;
		long v = std::strtol(env, &endPtr, 10);
		if (endPtr && *endPtr == '\0' && v > 0)
			m_maxTSPerSubmit = (unsigned int)v;
	}
	// Trapezoidal temporal blocking, read as --gpu-temporal-block reads it:
	// "<k>" or "<k>:<W>" forces it, "off" or "0" turns it off, "auto" or unset
	// leaves it to ConfigureTemporalBlocking().  The command-line options
	// replace these (openEMS::SetupFDTD()).
	if (const char* env = std::getenv("OPENEMS_GPU_TEMPORAL_BLOCK"))
	{
		if (!TemporalBlockingRequest::Parse(env, m_tbRequest))
		{
			cerr << "Engine_Vulkan: cannot read OPENEMS_GPU_TEMPORAL_BLOCK=" << env
			     << " (auto, off, <k> or <k>:<W>); temporal blocking off" << endl;
			m_tbRequest.mode = TemporalBlockingRequest::OFF;
		}
	}
	// How much field state one tile should occupy.  Vulkan exposes no
	// last-level-cache size, so this is a tunable with a per-device-class
	// default: the tile wants to be a fraction of the LLC it must stay in, and
	// a discrete card's is far larger than an integrated one's share of the
	// CPU LLC.  Measured optima: 64 MB tiles on an RX 6800 (128 MB Infinity
	// Cache), 30 MB on a UHD 770 (36 MB shared L3).
	// 0 here means "resolve from the device class in ConfigureTemporalBlocking".
	if (const char* env = std::getenv("OPENEMS_GPU_TEMPORAL_BLOCK_MB"))
	{
		int mb = std::atoi(env);
		if (mb > 0) m_tbRequest.targetMB = (unsigned int)mb;
	}

	if (const char* env = std::getenv("OPENEMS_GPU_CHUNK_PROGRESS_SEC"))
	{
		char* endPtr = nullptr;
		double v = std::strtod(env, &endPtr);
		if (endPtr && *endPtr == '\0' && v > 0.0)
			m_chunkProgressIntervalSec = v;
	}
	if (const char* env = std::getenv("OPENEMS_GPU_VALIDATE_FIELDS"))
	{
		char* endPtr = nullptr;
		long v = std::strtol(env, &endPtr, 10);
		if (endPtr && *endPtr == '\0')
			m_validateFields = (v != 0);
	}
	if (const char* env = std::getenv("OPENEMS_GPU_VALIDATE_STAGES"))
	{
		char* endPtr = nullptr;
		long v = std::strtol(env, &endPtr, 10);
		if (endPtr && *endPtr == '\0')
			m_validateStages = (v != 0);
	}
	if (const char* env = std::getenv("OPENEMS_GPU_VALIDATE_MAGNITUDE"))
	{
		char* endPtr = nullptr;
		long v = std::strtol(env, &endPtr, 10);
		if (endPtr && *endPtr == '\0')
			m_validateMagnitude = (v != 0);
	}
	m_validateFields = m_validateFields || m_validateStages || m_validateMagnitude;
	if (const char* env = std::getenv("OPENEMS_GPU_TRACE_CELL"))
	{
		unsigned int x = 0, y = 0, z = 0;
		if (std::sscanf(env, "%u,%u,%u", &x, &y, &z) == 3 &&
		    x < numLines[0] && y < numLines[1] && z < numLines[2])
			m_validateTraceIndex = x * numLines[1] * numLines[2] + y * numLines[2] + z;
	}

	if (startupTrace)
	{
		cout << "[Vulkan startup] config: max_ts_per_submit=" << m_maxTSPerSubmit
		     << ", chunk_progress_sec=" << m_chunkProgressIntervalSec
		     << ", validate_fields=" << (m_validateFields ? 1 : 0)
		     << ", validate_stages=" << (m_validateStages ? 1 : 0) << endl;
	}

	// Vulkan resources
	runStartupStep("InitVulkan", [&]() { InitVulkan(); });
	runStartupStep("CreateBuffers", [&]() { CreateBuffers(); });
	runStartupStep("CreateDescriptors", [&]() { CreateDescriptors(); });
	runStartupStep("CreatePipelines", [&]() { CreatePipelines(); });
	runStartupStep("UploadCoefficients", [&]() { UploadCoefficients(); });
	runStartupStep("SetupGPUExcitation", [&]() { SetupGPUExcitation(); });
	runStartupStep("SetupGPUExtensions", [&]() { SetupGPUExtensions(); });
	runStartupStep("SetupFusedUPML", [&]() { SetupFusedUPML(); });
	runStartupStep("SetupGPU_EnergyReduction", [&]() { SetupGPU_EnergyReduction(); });
	if (m_validateFields)
		runStartupStep("SetupGPU_FieldValidation", [&]() { SetupGPU_FieldValidation(); });

	// Upload initial (zero) field data to GPU
	runStartupStep("UploadInitialFields", [&]() {
		UploadToDeviceBuffer(m_voltBuf, volt_ptr->data(), m_fieldBufSize);
		UploadToDeviceBuffer(m_currBuf, curr_ptr->data(), m_fieldBufSize);
	});

	m_hostDirty  = false;
	m_liveTS = numTS;

	// Print GPU info
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(m_physDevice, &props);
	cout << "Engine_Vulkan: using " << props.deviceName
	     << " (Vulkan " << VulkanVersionString(props.apiVersion)
	     << ", engine baseline " << VulkanVersionString(Engine_Vulkan::API_VERSION)
	     << ")" << endl;
	if (g_settings.GetVerboseLevel() > 0)
	{
		cout << "  Device features: "
		     << (m_caps.subgroupSizeControl ? "subgroupSizeControl " : "")
		     << (m_caps.computeFullSubgroups ? "computeFullSubgroups " : "")
		     << (m_caps.synchronization2 ? "sync2 " : "")
		     << (m_caps.timelineSemaphore ? "timelineSemaphore " : "")
		     << (m_caps.storageBuffer16 ? "storage16 " : "")
		     << (m_caps.storageBuffer8 ? "storage8 " : "")
		     << endl;
		cout << "  Subgroup size: " << m_caps.minSubgroupSize
		     << ".." << m_caps.maxSubgroupSize << endl;
	}
	cout << "  Field buffer size: " << (m_fieldBufSize / (1024*1024)) << " MB per field" << endl;
	{
		const Operator_Vulkan* opVk = dynamic_cast<const Operator_Vulkan*>(Op);
		if (opVk->IsCompressed())
			cout << "  Compressed operator: " << opVk->GetNumCompressed() << " unique coeff sets"
			     << " (index: " << (m_opIndexBufSize / 1024) << " KB, tables: "
			     << (4 * m_coeffCompBufSize / 1024) << " KB)" << endl;
		else
			cout << "  Uncompressed operator: one coeff set per cell, no index"
			     << " (tables: " << (4 * m_coeffCompBufSize / 1024) << " KB)" << endl;
	}
	if (m_profileEnabled)
		cout << "  GPU profiling enabled (timestamp queries + fence wait accounting"
		     << (m_profileStages ? ", stage timestamps" : "") << ")" << endl;
	if (m_profileNoBarriers)
		cout << "  WARNING: OPENEMS_GPU_PROFILE_NO_BARRIERS -- timestep barriers skipped, results are invalid (timing only)" << endl;
	if (m_hasGPU_UPML)
		cout << "  GPU UPML active (" << m_gpuUPML.size() << " region(s))" << endl;
	if (m_hasFusedUPML)
		cout << "  Fused Yee+UPML kernels active" << endl;
	if (m_hasGPU_Dispersive)
		cout << "  GPU dispersive materials active (" << m_gpuDisp.size() << " order(s))" << endl;
	if (m_hasGPU_TFSF)
		cout << "  GPU TF/SF active (V:" << m_gpuTFSF.voltCount << " I:" << m_gpuTFSF.currCount << " points)" << endl;
	if (m_hasGPU_Mur)
		cout << "  GPU Mur ABC active (" << m_gpuMur.size() << " face(s))" << endl;
	if (m_hasGPU_RLC)
		cout << "  GPU lumped RLC active (" << m_gpuRLC.count << " elements)" << endl;
	if (m_steadyPipeline)
		cout << "  GPU steady-state sampling active (" << m_steadyProbeCount << " probes)" << endl;
	if (!m_gpuLocalABC.empty())
		cout << "  GPU local absorbing sheets active (" << m_gpuLocalABC.size() << " sheet(s))" << endl;
	cout << "  GPU energy reduction active (" << ENERGY_NUM_WG << " workgroups)" << endl;

	if (startupTrace)
	{
		auto startupEnd = std::chrono::steady_clock::now();
		double totalMs = std::chrono::duration<double, std::milli>(startupEnd - startupBegin).count();
		cout << "[Vulkan startup] total init time: " << totalMs << " ms" << endl;
	}
}

void Engine_Vulkan::Reset()
{
	if (m_hasGPU_Probes && (m_probeCacheHits > 0 || m_probeCacheMisses > 0))
	{
		uint64_t total = m_probeCacheHits + m_probeCacheMisses;
		double missPct = total ? (100.0 * (double)m_probeCacheMisses / (double)total) : 0.0;
		cout << "Engine_Vulkan probe cache stats: hits=" << m_probeCacheHits
		     << ", misses=" << m_probeCacheMisses
		     << " (" << missPct << "% miss)" << endl;
		if (m_probeCacheMisses > 0)
			cout << "  Note: each probe cache miss read the live fields, which the pipelined loop had computed ahead." << endl;
	}
	CleanupVulkan();
	Engine::Reset();
}

// ===========================================================================
// Vulkan Initialisation
// ===========================================================================

void Engine_Vulkan::InitVulkan()
{
	// --- Instance ---
	VkApplicationInfo appInfo{};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "openEMS";
	appInfo.apiVersion = VulkanRequestedApiVersion();

	VkInstanceCreateInfo instInfo{};
	instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instInfo.pApplicationInfo = &appInfo;
	VK_CHECK(vkCreateInstance(&instInfo, nullptr, &m_instance));

	// --- Physical device ---
	bool sawBelowBaseline = false;
	const std::vector<VulkanComputeDevice> devices = ListComputeDevices(m_instance, sawBelowBaseline);
	if (devices.empty())
	{
		if (sawBelowBaseline)
			throw std::runtime_error(
				"Engine_Vulkan: no GPU implementing Vulkan " +
				VulkanVersionString(Engine_Vulkan::API_VERSION) +
				" found (a Vulkan device is present but reports an older version "
				"-- update the graphics driver, or build without WITH_GPU)");
		throw std::runtime_error("Engine_Vulkan: no compute-capable GPU found");
	}
	// the preflight has listed the devices and reported an invalid choice
	const VulkanComputeDevice& chosen = devices[ChooseComputeDevice(devices, false)];
	m_physDevice = chosen.dev;
	m_computeQueueFamily = chosen.computeQueueFamily;

	// --- Logical device ---
	float priority = 1.0f;
	std::vector<VkDeviceQueueCreateInfo> queueInfos;
	VkDeviceQueueCreateInfo computeQueueInfo{};
	computeQueueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	computeQueueInfo.queueFamilyIndex = m_computeQueueFamily;
	computeQueueInfo.queueCount = 1;
	computeQueueInfo.pQueuePriorities = &priority;
	queueInfos.push_back(computeQueueInfo);

	// --- Negotiate optional features ---------------------------------------
	// Query what the device supports, then enable exactly the subset the engine
	// can make use of.  Everything here is core in the 1.3 baseline, so this is
	// a supported/not-supported question, not an extension-string question.
	// Enabling nothing else keeps driver-side validation cost and the chance of
	// hitting a buggy optional path to a minimum.
	VkPhysicalDeviceVulkan13Features have13{};
	have13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
	VkPhysicalDeviceVulkan12Features have12{};
	have12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	have12.pNext = &have13;
	VkPhysicalDeviceVulkan11Features have11{};
	have11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
	have11.pNext = &have12;
	VkPhysicalDeviceFeatures2 have2{};
	have2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	have2.pNext = &have11;
	vkGetPhysicalDeviceFeatures2(m_physDevice, &have2);

	VkPhysicalDeviceVulkan13Properties props13{};
	props13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES;
	VkPhysicalDeviceProperties2 props2{};
	props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	props2.pNext = &props13;
	vkGetPhysicalDeviceProperties2(m_physDevice, &props2);

	m_caps = DeviceCaps{};
	m_caps.subgroupSizeControl  = have13.subgroupSizeControl;
	m_caps.computeFullSubgroups = have13.computeFullSubgroups;
	m_caps.synchronization2     = have13.synchronization2;
	m_caps.timelineSemaphore    = have12.timelineSemaphore;
	m_caps.storageBuffer8       = have12.storageBuffer8BitAccess;
	m_caps.shaderInt8           = have12.shaderInt8;
	m_caps.storageBuffer16      = have11.storageBuffer16BitAccess;
	m_caps.shaderInt16          = have2.features.shaderInt16;
	m_caps.minSubgroupSize      = props13.minSubgroupSize;
	m_caps.maxSubgroupSize      = props13.maxSubgroupSize;

	VkPhysicalDeviceVulkan13Features en13{};
	en13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
	en13.subgroupSizeControl  = m_caps.subgroupSizeControl;
	en13.computeFullSubgroups = m_caps.computeFullSubgroups;
	en13.synchronization2     = m_caps.synchronization2;
	VkPhysicalDeviceVulkan12Features en12{};
	en12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	en12.pNext = &en13;
	en12.timelineSemaphore      = m_caps.timelineSemaphore;
	en12.storageBuffer8BitAccess = m_caps.storageBuffer8;
	en12.shaderInt8             = m_caps.shaderInt8;
	VkPhysicalDeviceVulkan11Features en11{};
	en11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
	en11.pNext = &en12;
	en11.storageBuffer16BitAccess = m_caps.storageBuffer16;
	VkPhysicalDeviceFeatures2 en2{};
	en2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	en2.pNext = &en11;
	en2.features.shaderInt16 = m_caps.shaderInt16;

	VkDeviceCreateInfo devInfo{};
	devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	devInfo.queueCreateInfoCount = (uint32_t)queueInfos.size();
	devInfo.pQueueCreateInfos = queueInfos.data();
	// pEnabledFeatures must stay null when a VkPhysicalDeviceFeatures2 is chained.
	devInfo.pNext = &en2;
	VK_CHECK(vkCreateDevice(m_physDevice, &devInfo, nullptr, &m_device));

	vkGetDeviceQueue(m_device, m_computeQueueFamily, 0, &m_computeQueue);

	// --- Command pool + buffer ---
	VkCommandPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex = m_computeQueueFamily;
	poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	VK_CHECK(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_cmdPool));

	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = m_cmdPool;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = 1;
	for (int i = 0; i < CMD_RING_SIZE; ++i)
		VK_CHECK(vkAllocateCommandBuffers(m_device, &allocInfo, &m_cmdBufs[i]));
	VK_CHECK(vkAllocateCommandBuffers(m_device, &allocInfo, &m_utilCmdBuf));

	// --- Fences ---
	VkFenceCreateInfo fenceInfo{};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;   // start signaled so first wait is a no-op
	for (int i = 0; i < CMD_RING_SIZE; ++i)
		VK_CHECK(vkCreateFence(m_device, &fenceInfo, nullptr, &m_fences[i]));
	fenceInfo.flags = 0;   // utility fence starts unsignaled
	VK_CHECK(vkCreateFence(m_device, &fenceInfo, nullptr, &m_utilFence));

	// --- Pipeline cache (speeds up subsequent runs) ---
	VkPipelineCacheCreateInfo cacheInfo{};
	cacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	VK_CHECK(vkCreatePipelineCache(m_device, &cacheInfo, nullptr, &m_pipelineCache));

	SetupProfiling();
}

void Engine_Vulkan::SetupProfiling()
{
	m_profileEnabled = false;
	if (!s_enableProfiling)
		return;

	m_profilePhaseEnabled = false;
	if (const char* envPhase = std::getenv("OPENEMS_GPU_PROFILE_PHASES"))
	{
		int enabled = std::atoi(envPhase);
		m_profilePhaseEnabled = (enabled != 0);
	}
	m_profileQueriesPerSlot = m_profilePhaseEnabled ? 4u : 2u;
	if (const char* env = std::getenv("OPENEMS_GPU_PROFILE_STAGES"))
		m_profileStages = (std::atoi(env) != 0);
	if (const char* env = std::getenv("OPENEMS_GPU_PROFILE_NO_BARRIERS"))
		m_profileNoBarriers = (std::atoi(env) != 0);

	uint32_t qfCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(m_physDevice, &qfCount, nullptr);
	if (qfCount == 0 || m_computeQueueFamily >= qfCount)
		return;

	std::vector<VkQueueFamilyProperties> qfProps(qfCount);
	vkGetPhysicalDeviceQueueFamilyProperties(m_physDevice, &qfCount, qfProps.data());
	if (qfProps[m_computeQueueFamily].timestampValidBits == 0)
	{
		cerr << "Engine_Vulkan: GPU profiling requested, but queue timestamps are unsupported." << endl;
		return;
	}

	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(m_physDevice, &props);
	m_timestampPeriodNs = props.limits.timestampPeriod;

	VkQueryPoolCreateInfo qpi{};
	qpi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
	qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
	qpi.queryCount = m_profileQueriesPerSlot * CMD_RING_SIZE;
	VK_CHECK(vkCreateQueryPool(m_device, &qpi, nullptr, &m_tsQueryPool));

	// A pure-GPU batch holds at most m_maxTSPerSubmit timesteps (IterateTS
	// splits longer calls), so that bounds a slot's stage marks.
	if (m_profileStages)
	{
		m_stageQueriesPerSlot = 1 + PROFILE_STAGES * m_maxTSPerSubmit;
		qpi.queryCount = m_stageQueriesPerSlot * CMD_RING_SIZE;
		VK_CHECK(vkCreateQueryPool(m_device, &qpi, nullptr, &m_stageQueryPool));
	}

	m_profileEnabled = true;
}

void Engine_Vulkan::CollectProfileForSlot(int slot) const
{
	if (!m_profileEnabled || !m_tsQueryPool || slot < 0 || slot >= CMD_RING_SIZE || !m_profilePending[slot])
		return;

	uint64_t ts[4] = {0, 0, 0, 0};
	VkResult r = vkGetQueryPoolResults(
		m_device,
		m_tsQueryPool,
		(uint32_t)(slot * m_profileQueriesPerSlot),
		m_profileQueriesPerSlot,
		sizeof(ts),
		ts,
		sizeof(uint64_t),
		VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

	if (r == VK_SUCCESS && ts[m_profileQueriesPerSlot - 1] >= ts[0])
	{
		double gpuNs = (double)(ts[m_profileQueriesPerSlot - 1] - ts[0]) * (double)m_timestampPeriodNs;
		m_profileGpuMs += gpuNs * 1e-6;
		m_profileBatchCount++;
		m_profileTSCount += m_profileSubmittedTS[slot];
		m_profileFirstTick = std::min(m_profileFirstTick, ts[0]);
		m_profileLastTick = std::max(m_profileLastTick, ts[m_profileQueriesPerSlot - 1]);

		if (m_profilePhaseEnabled && m_profileQueriesPerSlot == 4 &&
		    ts[1] >= ts[0] && ts[2] >= ts[1] && ts[3] >= ts[2])
		{
			double computeNs = (double)(ts[1] - ts[0]) * (double)m_timestampPeriodNs;
			double probeNs   = (double)(ts[2] - ts[1]) * (double)m_timestampPeriodNs;
			double tailNs    = (double)(ts[3] - ts[2]) * (double)m_timestampPeriodNs;
			m_profileComputeMs += computeNs * 1e-6;
			m_profileProbeMs += probeNs * 1e-6;
			m_profileTailMs += tailNs * 1e-6;
		}
	}

	const unsigned int nStage = m_stageSlotTS[slot];
	if (m_stageQueryPool && nStage > 0)
	{
		std::vector<uint64_t> marks(1 + PROFILE_STAGES * nStage);
		r = vkGetQueryPoolResults(m_device, m_stageQueryPool, (uint32_t)(slot * m_stageQueriesPerSlot),
		                          (uint32_t)marks.size(), marks.size() * sizeof(uint64_t), marks.data(),
		                          sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
		if (r == VK_SUCCESS)
		{
			for (size_t i = 1; i < marks.size(); ++i)
				if (marks[i] >= marks[i - 1])
					m_stageMs[(i - 1) % PROFILE_STAGES] += (double)(marks[i] - marks[i - 1]) * (double)m_timestampPeriodNs * 1e-6;
			m_stageTSCount += nStage;
		}
	}

	const_cast<Engine_Vulkan*>(this)->m_profilePending[slot] = false;
	const_cast<Engine_Vulkan*>(this)->m_profileSubmittedTS[slot] = 0;
	const_cast<Engine_Vulkan*>(this)->m_stageSlotTS[slot] = 0;
}

void Engine_Vulkan::PrintProfileSummary() const
{
	if (!m_profileEnabled || m_profileBatchCount == 0)
		return;

	double avgBatchMs = m_profileGpuMs / (double)m_profileBatchCount;
	double avgTsUs = (m_profileTSCount > 0)
	               ? (m_profileGpuMs * 1000.0) / (double)m_profileTSCount
	               : 0.0;
	double waitPerBatchMs = m_profileWaitMs / (double)m_profileBatchCount;

	cout << "Engine_Vulkan profiling summary:" << endl;
	cout << "  Batches: " << m_profileBatchCount
	     << ", timesteps: " << m_profileTSCount << endl;
	cout << "  GPU time: " << m_profileGpuMs << " ms total"
	     << ", avg " << avgBatchMs << " ms/batch"
	     << ", " << avgTsUs << " us/TS" << endl;
	cout << "  Fence wait: " << m_profileWaitMs << " ms total"
	     << ", avg " << waitPerBatchMs << " ms/batch" << endl;
	if (m_profilePhaseEnabled && m_profileGpuMs > 0.0)
	{
		double compPct = 100.0 * (m_profileComputeMs / m_profileGpuMs);
		double probePct = 100.0 * (m_profileProbeMs / m_profileGpuMs);
		double tailPct = 100.0 * (m_profileTailMs / m_profileGpuMs);
		cout << "  Phase split (GPU): compute=" << m_profileComputeMs << " ms (" << compPct << "%), "
		     << "probe=" << m_profileProbeMs << " ms (" << probePct << "%), "
		     << "tail=" << m_profileTailMs << " ms (" << tailPct << "%)" << endl;
	}
	if (m_profileLastTick > m_profileFirstTick)
	{
		// Batches run one after another on one queue, so the span from the
		// first batch's start to the last one's end, minus their busy time, is
		// time the GPU sat waiting for the host.
		const double spanMs = (double)(m_profileLastTick - m_profileFirstTick) * (double)m_timestampPeriodNs * 1e-6;
		cout << "  GPU span: " << spanMs << " ms, idle between batches " << spanMs - m_profileGpuMs
		     << " ms (" << 100.0 * (spanMs - m_profileGpuMs) / spanMs << "%)" << endl;
	}
	if (m_profileRecordedTS > 0)
		cout << "  Compute barriers per timestep: " << (double)m_profileBarriers / (double)m_profileRecordedTS
		     << (m_profileNoBarriers ? " (all skipped: OPENEMS_GPU_PROFILE_NO_BARRIERS, results invalid)" : "") << endl;
	if (m_stageTSCount > 0)
	{
		double sumMs = 0.0;
		for (int s = 0; s < PROFILE_STAGES; ++s)
			sumMs += m_stageMs[s];
		cout << "  Stage split (GPU, " << m_stageTSCount << " timesteps, us/TS):";
		for (int s = 0; s < PROFILE_STAGES; ++s)
			cout << (s ? ", " : " ") << s_profileStageNames[s] << "=" << 1000.0 * m_stageMs[s] / (double)m_stageTSCount
			     << " (" << 100.0 * m_stageMs[s] / sumMs << "%)";
		cout << ", sum=" << 1000.0 * sumMs / (double)m_stageTSCount << endl;
	}
}

void Engine_Vulkan::CmdComputeBarrier(VkCommandBuffer cmd, const VkMemoryBarrier& barrier) const
{
	if (m_profileEnabled)
		++m_profileBarriers;
	if (m_profileNoBarriers)
		return;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                     0, 1, &barrier, 0, nullptr, 0, nullptr);
}

// ===========================================================================
// Buffer helpers
// ===========================================================================

uint32_t Engine_Vulkan::FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const
{
	VkPhysicalDeviceMemoryProperties memProps;
	vkGetPhysicalDeviceMemoryProperties(m_physDevice, &memProps);
	for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
	{
		if ((typeFilter & (1 << i)) &&
		    (memProps.memoryTypes[i].propertyFlags & properties) == properties)
			return i;
	}
	throw std::runtime_error("Engine_Vulkan: failed to find suitable memory type");
}

uint32_t Engine_Vulkan::FindMemoryTypeSoft(uint32_t typeFilter, VkMemoryPropertyFlags properties) const
{
	VkPhysicalDeviceMemoryProperties memProps;
	vkGetPhysicalDeviceMemoryProperties(m_physDevice, &memProps);
	for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
	{
		if ((typeFilter & (1 << i)) &&
		    (memProps.memoryTypes[i].propertyFlags & properties) == properties)
			return i;
	}
	return UINT32_MAX;
}

void Engine_Vulkan::CreateBufferWithMemory(VkDeviceSize size, VkBufferUsageFlags usage,
                                           VkMemoryPropertyFlags memProps,
                                           VkBuffer& buf, VkDeviceMemory& mem)
{
	if (size == 0) { buf = VK_NULL_HANDLE; mem = VK_NULL_HANDLE; return; }
	if (!TryCreateBufferWithMemory(size, usage, memProps, buf, mem))
		throw std::runtime_error("Engine_Vulkan: buffer memory allocation failed");
}

bool Engine_Vulkan::TryCreateBufferWithMemory(VkDeviceSize size, VkBufferUsageFlags usage,
                                              VkMemoryPropertyFlags memProps,
                                              VkBuffer& buf, VkDeviceMemory& mem)
{
	buf = VK_NULL_HANDLE;
	mem = VK_NULL_HANDLE;
	if (size == 0)
		return true;

	VkBufferCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	ci.size  = size;
	ci.usage = usage;
	ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	if (vkCreateBuffer(m_device, &ci, nullptr, &buf) != VK_SUCCESS)
		return false;

	VkMemoryRequirements memReq;
	vkGetBufferMemoryRequirements(m_device, buf, &memReq);

	uint32_t memoryType = FindMemoryTypeSoft(memReq.memoryTypeBits, memProps);
	if (memoryType == UINT32_MAX)
	{
		vkDestroyBuffer(m_device, buf, nullptr);
		buf = VK_NULL_HANDLE;
		return false;
	}

	VkMemoryAllocateInfo ai{};
	ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	ai.allocationSize  = memReq.size;
	ai.memoryTypeIndex = memoryType;
	if (vkAllocateMemory(m_device, &ai, nullptr, &mem) != VK_SUCCESS)
	{
		vkDestroyBuffer(m_device, buf, nullptr);
		buf = VK_NULL_HANDLE;
		return false;
	}
	if (vkBindBufferMemory(m_device, buf, mem, 0) != VK_SUCCESS)
	{
		vkFreeMemory(m_device, mem, nullptr);
		vkDestroyBuffer(m_device, buf, nullptr);
		mem = VK_NULL_HANDLE;
		buf = VK_NULL_HANDLE;
		return false;
	}
	return true;
}

void Engine_Vulkan::CreateBuffers()
{
	size_t N = (size_t)numLines[0] * numLines[1] * numLines[2];
	m_fieldBufSize = 3 * N * sizeof(FDTD_FLOAT);
	m_fieldN   = (uint32_t)N;
	m_strideYZ = numLines[1] * numLines[2];

	VkBufferUsageFlags devRW  = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT   |
	                            VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkBufferUsageFlags devRO  = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	                            VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags dLoc = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
	                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	// The fields stay in device-local memory; the host reads them by DMA.  A
	// host-visible device-local type (the PCI BAR) would let the CPU read them
	// in place, but with a legacy 256 MB BAR the driver silently spills what
	// does not fit into system RAM, and the update kernels then stream the
	// fields over PCIe.  On an RX 6800 that cost 6.4x for a 256x224x224 grid,
	// and about 10x for three runs that each fit but shared the card.
	CreateBufferWithMemory(m_fieldBufSize, devRW, dLoc, m_voltBuf, m_voltMem);
	CreateBufferWithMemory(m_fieldBufSize, devRW, dLoc, m_currBuf, m_currMem);

	// Compressed coefficient buffers (read-only, always device-local)
	const Operator_Vulkan* opVk = dynamic_cast<const Operator_Vulkan*>(Op);
	unsigned int numComp = opVk->GetNumCompressed();
	// Narrow the per-cell operator index to the smallest width that can address
	// every unique coefficient set. opIdx is read once per cell per kernel and
	// is ~10% of all memory traffic on grids that do not fit cache (measured by
	// stubbing the lookup out entirely: +9.7% on Coax, +10.2% on a 320x256x256
	// synthetic grid), so narrowing it is a direct bandwidth saving. Real models
	// span the whole range -- 6 unique sets for a uniform benchmark mesh, 113
	// for Coax, 645/1565/6626 for the stripline and CPW tests -- so all three
	// widths earn their place.
	// Width 0 is the operator's "deduplication did not pay" mode: the tables
	// hold one entry per cell, the index is the identity, and nothing is
	// uploaded for it -- fetchOpIdx() returns the cell number. The buffer is
	// still created at minimum size so the descriptor set stays valid.
	if (!opVk->IsCompressed())
		m_opIdxBits = 0;
	else if (numComp <= 0x100u)
		m_opIdxBits = 8;
	else if (numComp <= 0x10000u)
		m_opIdxBits = 16;
	else
		m_opIdxBits = 32;
	if (m_opIdxBits == 0)
	{
		m_opIndexBufSize = sizeof(uint32_t);
	}
	else
	{
		const size_t opIdxPerWord = 32u / m_opIdxBits;
		// Round up to whole dwords: fetchOpIdx() always loads a full dword.
		m_opIndexBufSize = (VkDeviceSize)((N + opIdxPerWord - 1) / opIdxPerWord) * sizeof(uint32_t);
	}
	m_coeffCompBufSize = 3 * numComp * sizeof(float);
	CreateBufferWithMemory(m_opIndexBufSize,  devRO, dLoc, m_opIndexBuf,  m_opIndexMem);
	CreateBufferWithMemory(m_coeffCompBufSize, devRO, dLoc, m_vvCompBuf, m_vvCompMem);
	CreateBufferWithMemory(m_coeffCompBufSize, devRO, dLoc, m_viCompBuf, m_viCompMem);
	CreateBufferWithMemory(m_coeffCompBufSize, devRO, dLoc, m_iiCompBuf, m_iiCompMem);
	CreateBufferWithMemory(m_coeffCompBufSize, devRO, dLoc, m_ivCompBuf, m_ivCompMem);

	// Staging buffer: coefficient and field uploads, and field downloads
	VkBufferUsageFlags stgUse = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
	                            VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	// Prefer a HOST_CACHED type: this buffer is read back by the CPU in
	// DownloadFromDeviceBuffer(), and a plain HOST_VISIBLE|HOST_COHERENT
	// type can land on write-combined memory -- fast to write, ~10x slower
	// than a cached read for the CPU to read back. Falls back to the plain
	// flags on drivers without a cached host-visible type.
	if (!TryCreateBufferWithMemory(m_fieldBufSize, stgUse, host | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
	                                m_stagingBuf, m_stagingMem))
		CreateBufferWithMemory(m_fieldBufSize, stgUse, host, m_stagingBuf, m_stagingMem);
}

// ===========================================================================
// Descriptor setup
// ===========================================================================

void Engine_Vulkan::WriteDescriptorBuffers(VkDevice dev, VkDescriptorSet set,
                                           const VkBuffer* bufs, const VkDeviceSize* sizes,
                                           uint32_t count)
{
	std::vector<VkDescriptorBufferInfo> infos(count);
	std::vector<VkWriteDescriptorSet>   writes(count);
	for (uint32_t i = 0; i < count; i++)
	{
		infos[i]  = {bufs[i], 0, sizes[i]};
		writes[i] = {};
		writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[i].dstSet          = set;
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[i].pBufferInfo     = &infos[i];
	}
	vkUpdateDescriptorSets(dev, count, writes.data(), 0, nullptr);
}

void Engine_Vulkan::CreateDescriptors()
{
	// --- Layouts ---
	auto makeLayout = [&](uint32_t nBindings) -> VkDescriptorSetLayout
	{
		std::vector<VkDescriptorSetLayoutBinding> bindings(nBindings);
		for (uint32_t i = 0; i < nBindings; i++)
			bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
		VkDescriptorSetLayoutCreateInfo ci{};
		ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		ci.bindingCount = nBindings;
		ci.pBindings    = bindings.data();
		VkDescriptorSetLayout layout;
		VK_CHECK(vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &layout));
		return layout;
	};
	m_fdtdDescLayout = makeLayout(5);
	m_excDescLayout  = makeLayout(5);

	// --- Pool ---
	VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5*2 + 5*2};
	VkDescriptorPoolCreateInfo pi{};
	pi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pi.maxSets       = 4;
	pi.poolSizeCount = 1;
	pi.pPoolSizes    = &poolSize;
	VK_CHECK(vkCreateDescriptorPool(m_device, &pi, nullptr, &m_descPool));

	// --- Allocate sets ---
	VkDescriptorSetLayout layouts[4] = {m_fdtdDescLayout, m_fdtdDescLayout,
	                                     m_excDescLayout,  m_excDescLayout};
	VkDescriptorSetAllocateInfo ai{};
	ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	ai.descriptorPool     = m_descPool;
	ai.descriptorSetCount = 4;
	ai.pSetLayouts        = layouts;
	VkDescriptorSet sets[4];
	VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, sets));
	m_updateVoltDescSet = sets[0];
	m_updateCurrDescSet = sets[1];
	m_excVoltDescSet    = sets[2];
	m_excCurrDescSet    = sets[3];

	// --- Write FDTD update descriptors ---
	{
		VkBuffer      bufs[5] = {m_voltBuf, m_currBuf, m_opIndexBuf, m_vvCompBuf, m_viCompBuf};
		VkDeviceSize sizes[5] = {m_fieldBufSize, m_fieldBufSize, m_opIndexBufSize,
		                          m_coeffCompBufSize, m_coeffCompBufSize};
		WriteDescriptorBuffers(m_device, m_updateVoltDescSet, bufs, sizes, 5);
	}
	{
		VkBuffer      bufs[5] = {m_currBuf, m_voltBuf, m_opIndexBuf, m_iiCompBuf, m_ivCompBuf};
		VkDeviceSize sizes[5] = {m_fieldBufSize, m_fieldBufSize, m_opIndexBufSize,
		                          m_coeffCompBufSize, m_coeffCompBufSize};
		WriteDescriptorBuffers(m_device, m_updateCurrDescSet, bufs, sizes, 5);
	}
	// Excitation descriptors are written in SetupGPUExcitation()
}

// ===========================================================================
// Pipeline setup
// ===========================================================================

VkShaderModule Engine_Vulkan::CreateShaderModule(const unsigned char* code, size_t codeSize)
{
	// Copy to aligned uint32_t buffer
	std::vector<uint32_t> aligned((codeSize + 3) / 4);
	memcpy(aligned.data(), code, codeSize);

	VkShaderModuleCreateInfo ci{};
	ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	ci.codeSize = codeSize;
	ci.pCode    = aligned.data();
	VkShaderModule mod;
	VK_CHECK(vkCreateShaderModule(m_device, &ci, nullptr, &mod));
	return mod;
}

void Engine_Vulkan::CreatePipelines()
{
	// --- Pipeline layouts ---
	// FDTD: push constant = GridPC (12 bytes)
	VkPushConstantRange fdtdPC = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GridPC)};
	VkPipelineLayoutCreateInfo pli{};
	pli.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pli.setLayoutCount         = 1;
	pli.pSetLayouts            = &m_fdtdDescLayout;
	pli.pushConstantRangeCount = 1;
	pli.pPushConstantRanges    = &fdtdPC;
	VK_CHECK(vkCreatePipelineLayout(m_device, &pli, nullptr, &m_fdtdPipeLayout));

	// Excitation: push constant = ExcPC (16 bytes)
	VkPushConstantRange excPC = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ExcPC)};
	pli.pSetLayouts         = &m_excDescLayout;
	pli.pPushConstantRanges = &excPC;
	VK_CHECK(vkCreatePipelineLayout(m_device, &pli, nullptr, &m_excPipeLayout));

	// --- Shader modules ---
	VkShaderModule voltMod = CreateShaderModule(gpu_spirv::update_voltages_data,
	                                            gpu_spirv::update_voltages_size);
	VkShaderModule currMod = CreateShaderModule(gpu_spirv::update_currents_data,
	                                            gpu_spirv::update_currents_size);
	VkShaderModule excMod  = CreateShaderModule(gpu_spirv::apply_excitation_data,
	                                            gpu_spirv::apply_excitation_size);

	// OPIDX_BITS: the compressed-operator index width is known once the operator
	// exists, so specialize it here -- the driver folds fetchOpIdx() to a single
	// path at pipeline creation. Shaders declaring no such constant ignore it.
	// Index widths are known once the operator and the PML tables exist, so
	// specialize both here; the driver folds fetchOpIdx()/fetchPmlIdx() to a
	// single path at pipeline creation. A shader that declares neither, or only
	// one, simply ignores the entries it does not use.
	const uint32_t idxBits[2] = {m_opIdxBits, m_pmlIdxBits};
	VkSpecializationMapEntry opIdxSpecEntry[2] = {
		{0, 0 * sizeof(uint32_t), sizeof(uint32_t)},
		{1, 1 * sizeof(uint32_t), sizeof(uint32_t)},
	};
	VkSpecializationInfo opIdxSpec{};
	opIdxSpec.mapEntryCount = 2;
	opIdxSpec.pMapEntries   = opIdxSpecEntry;
	opIdxSpec.dataSize      = sizeof(idxBits);
	opIdxSpec.pData         = idxBits;

	auto makePipeline = [&](VkShaderModule mod, VkPipelineLayout layout) -> VkPipeline
	{
		VkPipelineShaderStageCreateInfo stage{};
		stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
		stage.module = mod;
		stage.pName  = "main";
		stage.pSpecializationInfo = &opIdxSpec;

		VkComputePipelineCreateInfo ci{};
		ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		ci.stage  = stage;
		ci.layout = layout;
		VkPipeline pipe;
		VK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache, 1, &ci, nullptr, &pipe));
		return pipe;
	};

	m_updateVoltPipeline = makePipeline(voltMod, m_fdtdPipeLayout);
	m_updateCurrPipeline = makePipeline(currMod, m_fdtdPipeLayout);
	m_excPipeline        = makePipeline(excMod,  m_excPipeLayout);

	vkDestroyShaderModule(m_device, voltMod, nullptr);
	vkDestroyShaderModule(m_device, currMod, nullptr);
	vkDestroyShaderModule(m_device, excMod,  nullptr);
}

// ===========================================================================
// Data upload helpers
// ===========================================================================

void Engine_Vulkan::RunSingleCommand(std::function<void(VkCommandBuffer)> func) const
{
	VkCommandBufferBeginInfo bi{};
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkResetCommandBuffer(m_utilCmdBuf, 0);
	VK_CHECK(vkBeginCommandBuffer(m_utilCmdBuf, &bi));

	func(m_utilCmdBuf);

	VK_CHECK(vkEndCommandBuffer(m_utilCmdBuf));

	VkSubmitInfo si{};
	si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	si.commandBufferCount = 1;
	si.pCommandBuffers    = &m_utilCmdBuf;
	vkResetFences(m_device, 1, &m_utilFence);
	VK_CHECK(vkQueueSubmit(m_computeQueue, 1, &si, m_utilFence));
	VK_CHECK(vkWaitForFences(m_device, 1, &m_utilFence, VK_TRUE, UINT64_MAX));
}

void Engine_Vulkan::UploadToDeviceBuffer(VkBuffer dst, const void* data, VkDeviceSize size)
{
	void* mapped;
	VK_CHECK(vkMapMemory(m_device, m_stagingMem, 0, size, 0, &mapped));
	memcpy(mapped, data, size);
	vkUnmapMemory(m_device, m_stagingMem);

	RunSingleCommand([&](VkCommandBuffer cmd)
	{
		VkBufferCopy region = {0, 0, size};
		vkCmdCopyBuffer(cmd, m_stagingBuf, dst, 1, &region);
	});
}

void Engine_Vulkan::DownloadFromDeviceBuffer(VkBuffer src, void* data, VkDeviceSize size) const
{
	RunSingleCommand([&](VkCommandBuffer cmd)
	{
		VkBufferCopy region = {0, 0, size};
		vkCmdCopyBuffer(cmd, src, m_stagingBuf, 1, &region);
	});

	void* mapped;
	VK_CHECK(vkMapMemory(m_device, m_stagingMem, 0, size, 0, &mapped));
	memcpy(data, mapped, size);
	vkUnmapMemory(m_device, m_stagingMem);
}

void Engine_Vulkan::UploadCoefficients()
{
	const Operator_Vulkan* opVk = dynamic_cast<const Operator_Vulkan*>(Op);
	if (m_opIdxBits == 0)
	{
		// Identity index: the shaders never read the buffer.
	}
	else if (m_opIdxBits == 32)
	{
		UploadToDeviceBuffer(m_opIndexBuf, opVk->GetOpIndex(), m_opIndexBufSize);
	}
	else
	{
		const size_t nCells = (size_t)numLines[0] * numLines[1] * numLines[2];
		std::vector<uint32_t> packed =
			PackNarrowIndices(opVk->GetOpIndex(), nCells, m_opIdxBits);
		packed.resize((size_t)m_opIndexBufSize / sizeof(uint32_t), 0u);
		UploadToDeviceBuffer(m_opIndexBuf, packed.data(), m_opIndexBufSize);
	}
	UploadToDeviceBuffer(m_vvCompBuf,  opVk->GetVVComp(),  m_coeffCompBufSize);
	UploadToDeviceBuffer(m_viCompBuf,  opVk->GetVIComp(),  m_coeffCompBufSize);
	UploadToDeviceBuffer(m_iiCompBuf,  opVk->GetIIComp(),  m_coeffCompBufSize);
	UploadToDeviceBuffer(m_ivCompBuf,  opVk->GetIVComp(),  m_coeffCompBufSize);
}

// ===========================================================================
// GpuBuf helpers
// ===========================================================================

void Engine_Vulkan::CreateGpuBuf(GpuBuf& gb, VkDeviceSize size,
                                 VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps)
{
	CreateBufferWithMemory(size, usage, memProps, gb.buffer, gb.memory);
}

void Engine_Vulkan::DestroyGpuBuf(GpuBuf& gb)
{
	if (gb.buffer) { vkDestroyBuffer(m_device, gb.buffer, nullptr); gb.buffer = VK_NULL_HANDLE; }
	if (gb.memory) { vkFreeMemory(m_device, gb.memory, nullptr);    gb.memory = VK_NULL_HANDLE; }
}

void Engine_Vulkan::UploadGpuBuf(GpuBuf& gb, const void* data, VkDeviceSize size)
{
	UploadToDeviceBuffer(gb.buffer, data, size);
}

// ===========================================================================
// GPU Excitation Setup
// ===========================================================================

void Engine_Vulkan::SetupGPUExcitation()
{
	// Find and remove the CPU excitation extension; upload its data to GPU.
	Operator_Ext_Excitation* opExc = const_cast<Operator*>(Op)->GetExcitationExtension();
	Excitation* exc = const_cast<Operator*>(Op)->GetExcitationSignal();

	// Remove CPU excitation first; excitation is always handled on the GPU.
	for (auto it = m_Eng_exts.begin(); it != m_Eng_exts.end(); )
	{
		Engine_Ext_Excitation* eext = dynamic_cast<Engine_Ext_Excitation*>(*it);
		if (eext)
		{
			// Remove CPU excitation; we handle it on GPU
			delete eext;
			it = m_Eng_exts.erase(it);
			continue;
		}

		++it;
	}

	// Every extension runs on the GPU, inside the batched update.  One without
	// a GPU implementation needs a CPU engine: applying it on the host would
	// copy both fields there and back twice per timestep, slower than the CPU.
	auto isGpuNative = [](Engine_Extension* ext)
	{
		return dynamic_cast<Engine_Ext_UPML*>(ext) != nullptr ||
		       dynamic_cast<Engine_Ext_LorentzMaterial*>(ext) != nullptr ||
		       dynamic_cast<Engine_Ext_DebyeMaterial*>(ext) != nullptr ||
		       dynamic_cast<Engine_Ext_TFSF*>(ext) != nullptr ||
		       dynamic_cast<Engine_Ext_Mur_ABC*>(ext) != nullptr ||
		       dynamic_cast<Engine_Ext_LumpedRLC*>(ext) != nullptr ||
		       dynamic_cast<Engine_Ext_SteadyState*>(ext) != nullptr ||
		       dynamic_cast<Engine_Ext_Absorbing_BC*>(ext) != nullptr;
	};
	for (Engine_Extension* ext : m_Eng_exts)
		if (!isGpuNative(ext))
			throw std::runtime_error("Engine_Vulkan: the extension \"" + ext->GetExtensionName() +
			                         "\" has no GPU implementation; use a CPU engine");

	for (auto it = m_Eng_exts.begin(); it != m_Eng_exts.end(); )
	{
		// openEMS keeps a non-owning pointer to this observer. Leave it in
		// the engine-owned list; the engine never runs the list.
		if (dynamic_cast<Engine_Ext_SteadyState*>(*it))
		{
			++it;
			continue;
		}
		delete *it;
		it = m_Eng_exts.erase(it);
	}

	if (!opExc || !exc)
	{
		m_hasGPUExcitation = false;
		return;
	}

	m_hasGPUExcitation = true;
	m_excSignalLen    = exc->GetLength();
	m_excSignalPeriod = (int)exc->GetSignalPeriodTS();
	for (unsigned int i = 0; i < m_excSignalLen; ++i)
	{
		if (!std::isfinite(exc->GetVoltageSignal()[i]) ||
		    !std::isfinite(exc->GetCurrentSignal()[i]))
			throw std::runtime_error("Engine_Vulkan: non-finite excitation signal sample");
	}

	size_t N = (size_t)numLines[0] * numLines[1] * numLines[2];
	uint32_t sYZ = numLines[1] * numLines[2];

	VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	                           VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags dLoc = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	// --- Voltage excitation ---
	m_excVoltCount = opExc->GetVoltCount();
	if (m_excVoltCount > 0)
	{
		// Precompute linear indices, ordered by x.
		//
		// Temporal blocking dispatches the excitation per x-slab, so the
		// entries of one slab have to be a contiguous run.  Sorting by x here
		// costs nothing at setup and lets the blocked path address a slab with
		// a base/end pair, exactly as the field sweep does with gidBase.
		// Stable, so entries within one x keep the operator's order.
		std::vector<uint32_t> linIdx(m_excVoltCount);
		std::vector<uint32_t> order(m_excVoltCount);
		for (unsigned int n = 0; n < m_excVoltCount; n++)
		{
			uint32_t dir = opExc->GetVoltDir()[n];
			uint32_t x   = opExc->GetVoltIndex(0)[n];
			uint32_t y   = opExc->GetVoltIndex(1)[n];
			uint32_t z   = opExc->GetVoltIndex(2)[n];
			if (dir >= 3 || x >= numLines[0] || y >= numLines[1] || z >= numLines[2] ||
			    !std::isfinite(opExc->GetVoltAmp()[n]))
				throw std::runtime_error("Engine_Vulkan: invalid voltage excitation entry");
			linIdx[n] = dir * (uint32_t)N + x * sYZ + y * numLines[2] + z;
			order[n]  = n;
		}
		const unsigned int* vx = opExc->GetVoltIndex(0);
		std::stable_sort(order.begin(), order.end(),
		                 [vx](uint32_t a, uint32_t b) { return vx[a] < vx[b]; });
		{
			std::vector<uint32_t> sortedX(order.size());
			for (size_t n = 0; n < order.size(); ++n)
				sortedX[n] = vx[order[n]];
			BuildSlabEntryIndex(sortedX, m_excVoltXStart);
		}

		std::vector<uint32_t>   sIdx(m_excVoltCount), sDelay(m_excVoltCount);
		std::vector<FDTD_FLOAT> sAmp(m_excVoltCount);
		for (unsigned int n = 0; n < m_excVoltCount; n++)
		{
			sIdx[n]   = linIdx[order[n]];
			sAmp[n]   = opExc->GetVoltAmp()[order[n]];
			sDelay[n] = opExc->GetVoltDelay()[order[n]];
		}

		VkDeviceSize idxSize = m_excVoltCount * sizeof(uint32_t);
		VkDeviceSize ampSize = m_excVoltCount * sizeof(FDTD_FLOAT);
		CreateBufferWithMemory(idxSize, usage, dLoc, m_excVoltIdxBuf,   m_excVoltIdxMem);
		CreateBufferWithMemory(ampSize, usage, dLoc, m_excVoltAmpBuf,   m_excVoltAmpMem);
		CreateBufferWithMemory(idxSize, usage, dLoc, m_excVoltDelayBuf, m_excVoltDelayMem);
		UploadToDeviceBuffer(m_excVoltIdxBuf,   sIdx.data(),   idxSize);
		UploadToDeviceBuffer(m_excVoltAmpBuf,   sAmp.data(),   ampSize);
		UploadToDeviceBuffer(m_excVoltDelayBuf, sDelay.data(), idxSize);
	}

	// --- Current excitation ---
	m_excCurrCount = opExc->GetCurrCount();
	if (m_excCurrCount > 0)
	{
		std::vector<uint32_t> linIdx(m_excCurrCount);
		std::vector<uint32_t> order(m_excCurrCount);
		for (unsigned int n = 0; n < m_excCurrCount; n++)
		{
			uint32_t dir = opExc->GetCurrDir()[n];
			uint32_t x   = opExc->GetCurrIndex(0)[n];
			uint32_t y   = opExc->GetCurrIndex(1)[n];
			uint32_t z   = opExc->GetCurrIndex(2)[n];
			if (dir >= 3 || x >= numLines[0] || y >= numLines[1] || z >= numLines[2] ||
			    !std::isfinite(opExc->GetCurrAmp()[n]))
				throw std::runtime_error("Engine_Vulkan: invalid current excitation entry");
			linIdx[n] = dir * (uint32_t)N + x * sYZ + y * numLines[2] + z;
			order[n]  = n;
		}
		const unsigned int* cx = opExc->GetCurrIndex(0);
		std::stable_sort(order.begin(), order.end(),
		                 [cx](uint32_t a, uint32_t b) { return cx[a] < cx[b]; });
		{
			std::vector<uint32_t> sortedX(order.size());
			for (size_t n = 0; n < order.size(); ++n)
				sortedX[n] = cx[order[n]];
			BuildSlabEntryIndex(sortedX, m_excCurrXStart);
		}

		std::vector<uint32_t>   sIdx(m_excCurrCount), sDelay(m_excCurrCount);
		std::vector<FDTD_FLOAT> sAmp(m_excCurrCount);
		for (unsigned int n = 0; n < m_excCurrCount; n++)
		{
			sIdx[n]   = linIdx[order[n]];
			sAmp[n]   = opExc->GetCurrAmp()[order[n]];
			sDelay[n] = opExc->GetCurrDelay()[order[n]];
		}

		VkDeviceSize idxSize = m_excCurrCount * sizeof(uint32_t);
		VkDeviceSize ampSize = m_excCurrCount * sizeof(FDTD_FLOAT);
		CreateBufferWithMemory(idxSize, usage, dLoc, m_excCurrIdxBuf,   m_excCurrIdxMem);
		CreateBufferWithMemory(ampSize, usage, dLoc, m_excCurrAmpBuf,   m_excCurrAmpMem);
		CreateBufferWithMemory(idxSize, usage, dLoc, m_excCurrDelayBuf, m_excCurrDelayMem);
		UploadToDeviceBuffer(m_excCurrIdxBuf,   sIdx.data(),   idxSize);
		UploadToDeviceBuffer(m_excCurrAmpBuf,   sAmp.data(),   ampSize);
		UploadToDeviceBuffer(m_excCurrDelayBuf, sDelay.data(), idxSize);
	}

	// --- Excitation signal ---
	VkDeviceSize sigSize = m_excSignalLen * sizeof(FDTD_FLOAT);
	CreateBufferWithMemory(sigSize, usage, dLoc, m_excSignalVoltBuf, m_excSignalVoltMem);
	CreateBufferWithMemory(sigSize, usage, dLoc, m_excSignalCurrBuf, m_excSignalCurrMem);
	UploadToDeviceBuffer(m_excSignalVoltBuf, exc->GetVoltageSignal(), sigSize);
	UploadToDeviceBuffer(m_excSignalCurrBuf, exc->GetCurrentSignal(), sigSize);

	// --- Write excitation descriptor sets ---
	if (m_excVoltCount > 0)
	{
		VkBuffer      bufs[5] = {m_voltBuf, m_excVoltAmpBuf, m_excVoltIdxBuf,
		                          m_excVoltDelayBuf, m_excSignalVoltBuf};
		VkDeviceSize sizes[5] = {m_fieldBufSize,
		                          m_excVoltCount * sizeof(FDTD_FLOAT),
		                          m_excVoltCount * sizeof(uint32_t),
		                          m_excVoltCount * sizeof(uint32_t),
		                          sigSize};
		WriteDescriptorBuffers(m_device, m_excVoltDescSet, bufs, sizes, 5);
	}
	if (m_excCurrCount > 0)
	{
		VkBuffer      bufs[5] = {m_currBuf, m_excCurrAmpBuf, m_excCurrIdxBuf,
		                          m_excCurrDelayBuf, m_excSignalCurrBuf};
		VkDeviceSize sizes[5] = {m_fieldBufSize,
		                          m_excCurrCount * sizeof(FDTD_FLOAT),
		                          m_excCurrCount * sizeof(uint32_t),
		                          m_excCurrCount * sizeof(uint32_t),
		                          sigSize};
		WriteDescriptorBuffers(m_device, m_excCurrDescSet, bufs, sizes, 5);
	}
}

// ===========================================================================
// GPU Extension Setup (UPML, Dispersive, TF/SF, Mur ABC, LumpedRLC, Probes)
// ===========================================================================

void Engine_Vulkan::SetupGPUExtensions()
{
	bool startupTrace = (g_settings.GetVerboseLevel() > 0);
	if (const char* env = std::getenv("OPENEMS_GPU_STARTUP_TRACE"))
	{
		char* endPtr = nullptr;
		long v = std::strtol(env, &endPtr, 10);
		if (endPtr && *endPtr == '\0')
			startupTrace = (v != 0);
	}

	auto extBegin = std::chrono::steady_clock::now();

	// Scan operator extensions and setup GPU resources for each
	// Iterate operator extensions
	size_t nExts = Op->GetNumberOfExtentions();

	// Count what we need for descriptor pool sizing
	unsigned int numUPML = 0, numDispOrders = 0, numTFSF = 0, numMur = 0;
	unsigned int numRLC = 0;
	unsigned int totalDescSets = 0;
	unsigned int totalStorageBindings = 0;

	for (size_t eIdx = 0; eIdx < nExts; eIdx++)
	{
		Operator_Extension* ext = Op->GetExtension(eIdx);
		if (auto* upml = dynamic_cast<Operator_Ext_UPML*>(ext))
		{
			numUPML++;
			totalDescSets += 4;
			totalStorageBindings += 18;  // 5+4+5+4
		}
		else if (auto* lor = dynamic_cast<Operator_Ext_LorentzMaterial*>(ext))
		{
			numDispOrders += lor->m_Order;
			totalDescSets += lor->m_Order * 4;
			totalStorageBindings += lor->m_Order * 20;
		}
		else if (dynamic_cast<Operator_Ext_DebyeMaterial*>(ext))
		{
			numDispOrders += 1;
			totalDescSets += 1;
			totalStorageBindings += 7;
		}
		else if (auto* tfsf = dynamic_cast<Operator_Ext_TFSF*>(ext))
		{
			numTFSF++;
			totalDescSets += 2;
			totalStorageBindings += 12;
		}
		else if (auto* mur = dynamic_cast<Operator_Ext_Mur_ABC*>(ext))
		{
			numMur++;
			totalDescSets += 3;
			totalStorageBindings += 17;  // pre 7, post 7, apply 3
		}
		else if (auto* rlc = dynamic_cast<Operator_Ext_LumpedRLC*>(ext))
		{
			if (rlc->RLC_count > 0) {
				numRLC++;
				totalDescSets += 2;
				totalStorageBindings += 22;
			}
		}
	}
	// Probes (placeholder, will be set up later)
	// totalDescSets += 1;
	// totalStorageBindings += 5;

	if (startupTrace)
	{
		cout << "[Vulkan startup] extension scan: op_ext=" << nExts
		     << ", upml=" << numUPML
		     << ", dispersive_orders=" << numDispOrders
		     << ", tfsf=" << numTFSF
		     << ", mur=" << numMur
		     << ", rlc=" << numRLC
		     << ", desc_sets=" << totalDescSets
		     << ", storage_bindings=" << totalStorageBindings
		     << endl;
	}

	if (totalDescSets == 0)
	{
		if (startupTrace)
			cout << "[Vulkan startup] no GPU extension resources required" << endl;
		SetupGPU_LocalABC();
		SetupGPU_SteadyState();
		return;
	}

	// Create extension descriptor set layouts
	CreateExtensionDescriptorLayouts();

	// --- Create extension descriptor pool ---
	VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, totalStorageBindings + 32};
	VkDescriptorPoolCreateInfo pi{};
	pi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pi.maxSets       = totalDescSets + 8;
	pi.poolSizeCount = 1;
	pi.pPoolSizes    = &poolSize;
	pi.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	VK_CHECK(vkCreateDescriptorPool(m_device, &pi, nullptr, &m_extDescPool));

	// Setup each extension family exactly once.
	// Each SetupGPU_* function scans all operator extensions internally.
	if (numUPML > 0)
	{
		if (startupTrace)
			cout << "[Vulkan startup] setup extension family: UPML" << endl;
		SetupGPU_UPML();
	}
	if (numDispOrders > 0)
	{
		if (startupTrace)
			cout << "[Vulkan startup] setup extension family: Dispersive" << endl;
		SetupGPU_Dispersive();
	}
	if (numTFSF > 0)
	{
		if (startupTrace)
			cout << "[Vulkan startup] setup extension family: TFSF" << endl;
		SetupGPU_TFSF();
	}
	if (numMur > 0)
	{
		if (startupTrace)
			cout << "[Vulkan startup] setup extension family: Mur" << endl;
		SetupGPU_Mur();
	}
	if (numRLC > 0)
	{
		if (startupTrace)
			cout << "[Vulkan startup] setup extension family: RLC" << endl;
		SetupGPU_RLC();
	}

	// Create extension compute pipelines
	CreateExtensionPipelines();
	SetupGPU_LocalABC();
	SetupGPU_SteadyState();
	SetupMurFusion();

	if (startupTrace)
	{
		auto extEnd = std::chrono::steady_clock::now();
		double extMs = std::chrono::duration<double, std::milli>(extEnd - extBegin).count();
		cout << "[Vulkan startup] SetupGPUExtensions total: " << extMs << " ms" << endl;
	}
}

void Engine_Vulkan::CreateExtensionDescriptorLayouts()
{
	auto makeLayout = [&](uint32_t nBindings) -> VkDescriptorSetLayout
	{
		std::vector<VkDescriptorSetLayoutBinding> bindings(nBindings);
		for (uint32_t i = 0; i < nBindings; i++)
			bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
		VkDescriptorSetLayoutCreateInfo ci{};
		ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		ci.bindingCount = nBindings;
		ci.pBindings    = bindings.data();
		VkDescriptorSetLayout layout;
		VK_CHECK(vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &layout));
		return layout;
	};

	// UPML: preVolt=5, postVolt=4, preCurr=5, postCurr=4
	// (field, flux, coefficient index, then the coefficient tables that pass reads)
	m_upmlPreVoltDescLayout  = makeLayout(5);
	m_upmlPostVoltDescLayout = makeLayout(4);
	m_upmlPreCurrDescLayout  = makeLayout(5);
	m_upmlPostCurrDescLayout = makeLayout(4);

	// Dispersive: pre=7 (field, ADE, LorADE, posIdx, int, ext, Lor), apply=3
	m_dispPreDescLayout   = makeLayout(7);
	m_dispApplyDescLayout = makeLayout(3);
	// Debye: one set for both passes (field, pole state, pre sum, posIdx, relax, drive, solve)
	m_debyeDescLayout     = makeLayout(7);

	// TF/SF: 6 bindings (field, amp, delta, idx, delay, signal)
	m_tfsfDescLayout = makeLayout(6);

	// Mur: pre=7 (with the saved V(shift)), post=7 (it shares the pre pipeline
	// layout), apply=3
	m_murPreDescLayout   = makeLayout(7);
	m_murPostDescLayout  = makeLayout(7);
	m_murApplyDescLayout = makeLayout(3);

	// RLC: pre=6, apply=16
	m_rlcPreDescLayout   = makeLayout(6);
	m_rlcApplyDescLayout = makeLayout(16);

	// Probes: 5 bindings
	m_probeDescLayout = makeLayout(5);
}

void Engine_Vulkan::CreateExtensionPipelines()
{
	// OPIDX_BITS: the compressed-operator index width is known once the operator
	// exists, so specialize it here -- the driver folds fetchOpIdx() to a single
	// path at pipeline creation. Shaders declaring no such constant ignore it.
	// Index widths are known once the operator and the PML tables exist, so
	// specialize both here; the driver folds fetchOpIdx()/fetchPmlIdx() to a
	// single path at pipeline creation. A shader that declares neither, or only
	// one, simply ignores the entries it does not use.
	const uint32_t idxBits[2] = {m_opIdxBits, m_pmlIdxBits};
	VkSpecializationMapEntry opIdxSpecEntry[2] = {
		{0, 0 * sizeof(uint32_t), sizeof(uint32_t)},
		{1, 1 * sizeof(uint32_t), sizeof(uint32_t)},
	};
	VkSpecializationInfo opIdxSpec{};
	opIdxSpec.mapEntryCount = 2;
	opIdxSpec.pMapEntries   = opIdxSpecEntry;
	opIdxSpec.dataSize      = sizeof(idxBits);
	opIdxSpec.pData         = idxBits;

	auto makePipeline = [&](VkShaderModule mod, VkPipelineLayout layout) -> VkPipeline
	{
		VkPipelineShaderStageCreateInfo stage{};
		stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
		stage.module = mod;
		stage.pName  = "main";
		stage.pSpecializationInfo = &opIdxSpec;
		VkComputePipelineCreateInfo ci{};
		ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		ci.stage  = stage;
		ci.layout = layout;
		VkPipeline pipe;
		VK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache, 1, &ci, nullptr, &pipe));
		return pipe;
	};

	auto makePipeLayout = [&](VkDescriptorSetLayout descLayout, uint32_t pcSize) -> VkPipelineLayout
	{
		VkPushConstantRange pcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize};
		VkPipelineLayoutCreateInfo pli{};
		pli.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pli.setLayoutCount         = 1;
		pli.pSetLayouts            = &descLayout;
		pli.pushConstantRangeCount = 1;
		pli.pPushConstantRanges    = &pcRange;
		VkPipelineLayout layout;
		VK_CHECK(vkCreatePipelineLayout(m_device, &pli, nullptr, &layout));
		return layout;
	};

	// --- UPML pipelines ---
	if (m_hasGPU_UPML)
	{
		m_upmlPrePipeLayout  = makePipeLayout(m_upmlPreVoltDescLayout, sizeof(PmlPC));
		m_upmlPostPipeLayout = makePipeLayout(m_upmlPostVoltDescLayout, sizeof(PmlPC));

		VkShaderModule mod;
		mod = CreateShaderModule(gpu_spirv::upml_pre_voltage_data, gpu_spirv::upml_pre_voltage_size);
		m_upmlPreVoltPipeline = makePipeline(mod, m_upmlPrePipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);

		mod = CreateShaderModule(gpu_spirv::upml_post_voltage_data, gpu_spirv::upml_post_voltage_size);
		m_upmlPostVoltPipeline = makePipeline(mod, m_upmlPostPipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);

		mod = CreateShaderModule(gpu_spirv::upml_pre_current_data, gpu_spirv::upml_pre_current_size);
		m_upmlPreCurrPipeline = makePipeline(mod, m_upmlPrePipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);

		mod = CreateShaderModule(gpu_spirv::upml_post_current_data, gpu_spirv::upml_post_current_size);
		m_upmlPostCurrPipeline = makePipeline(mod, m_upmlPostPipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);
	}

	// --- Dispersive pipelines ---
	if (m_hasGPU_Dispersive)
	{
		m_dispPrePipeLayout   = makePipeLayout(m_dispPreDescLayout, sizeof(DispPC));
		m_dispApplyPipeLayout = makePipeLayout(m_dispApplyDescLayout, sizeof(DispPC));

		VkShaderModule mod;
		mod = CreateShaderModule(gpu_spirv::dispersive_pre_voltage_data, gpu_spirv::dispersive_pre_voltage_size);
		m_dispPreVoltPipeline = makePipeline(mod, m_dispPrePipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);

		mod = CreateShaderModule(gpu_spirv::dispersive_pre_current_data, gpu_spirv::dispersive_pre_current_size);
		m_dispPreCurrPipeline = makePipeline(mod, m_dispPrePipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);

		mod = CreateShaderModule(gpu_spirv::dispersive_apply_voltage_data, gpu_spirv::dispersive_apply_voltage_size);
		m_dispApplyVoltPipeline = makePipeline(mod, m_dispApplyPipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);

		mod = CreateShaderModule(gpu_spirv::dispersive_apply_current_data, gpu_spirv::dispersive_apply_current_size);
		m_dispApplyCurrPipeline = makePipeline(mod, m_dispApplyPipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);

		if (std::any_of(m_gpuDisp.begin(), m_gpuDisp.end(), [](const GpuDispData& d) {return d.debye;}))
		{
			m_debyePipeLayout = makePipeLayout(m_debyeDescLayout, sizeof(DispPC));
			mod = CreateShaderModule(gpu_spirv::debye_pre_voltage_data, gpu_spirv::debye_pre_voltage_size);
			m_debyePreVoltPipeline = makePipeline(mod, m_debyePipeLayout);
			vkDestroyShaderModule(m_device, mod, nullptr);
			mod = CreateShaderModule(gpu_spirv::debye_apply_voltage_data, gpu_spirv::debye_apply_voltage_size);
			m_debyeApplyVoltPipeline = makePipeline(mod, m_debyePipeLayout);
			vkDestroyShaderModule(m_device, mod, nullptr);
		}
	}

	// --- TF/SF pipelines ---
	if (m_hasGPU_TFSF)
	{
		m_tfsfPipeLayout = makePipeLayout(m_tfsfDescLayout, sizeof(TfsfPC));

		VkShaderModule mod;
		mod = CreateShaderModule(gpu_spirv::tfsf_voltage_data, gpu_spirv::tfsf_voltage_size);
		m_tfsfVoltPipeline = makePipeline(mod, m_tfsfPipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);

		mod = CreateShaderModule(gpu_spirv::tfsf_current_data, gpu_spirv::tfsf_current_size);
		m_tfsfCurrPipeline = makePipeline(mod, m_tfsfPipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);
	}

	// --- Mur ABC pipelines ---
	if (m_hasGPU_Mur)
	{
		m_murUpdatePipeLayout = makePipeLayout(m_murPreDescLayout, sizeof(MurPC));
		m_murApplyPipeLayout  = makePipeLayout(m_murApplyDescLayout, sizeof(MurPC));

		VkShaderModule mod;
		mod = CreateShaderModule(gpu_spirv::mur_pre_voltage_data, gpu_spirv::mur_pre_voltage_size);
		m_murPreVoltPipeline = makePipeline(mod, m_murUpdatePipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);

		mod = CreateShaderModule(gpu_spirv::mur_post_voltage_data, gpu_spirv::mur_post_voltage_size);
		m_murPostVoltPipeline = makePipeline(mod, m_murUpdatePipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);

		mod = CreateShaderModule(gpu_spirv::mur_apply_voltage_data, gpu_spirv::mur_apply_voltage_size);
		m_murApplyVoltPipeline = makePipeline(mod, m_murApplyPipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);
	}

	// --- Lumped RLC pipelines ---
	if (m_hasGPU_RLC)
	{
		m_rlcPrePipeLayout   = makePipeLayout(m_rlcPreDescLayout, sizeof(RlcPC));
		m_rlcApplyPipeLayout = makePipeLayout(m_rlcApplyDescLayout, sizeof(RlcPC));

		VkShaderModule mod;
		mod = CreateShaderModule(gpu_spirv::rlc_pre_voltage_data, gpu_spirv::rlc_pre_voltage_size);
		m_rlcPreVoltPipeline = makePipeline(mod, m_rlcPrePipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);

		mod = CreateShaderModule(gpu_spirv::rlc_apply_voltage_data, gpu_spirv::rlc_apply_voltage_size);
		m_rlcApplyVoltPipeline = makePipeline(mod, m_rlcApplyPipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);
	}

	// --- Probe gather pipeline ---
	// (not created here; will be created on demand when probes are set up)
}

// ---------------------------------------------------------------------------
// Per-extension GPU data upload and descriptor writing
// ---------------------------------------------------------------------------

void Engine_Vulkan::SetupGPU_UPML()
{
	// Iterate operator extensions
	size_t nExts = Op->GetNumberOfExtentions();
	VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags dLoc = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	// One table shared by every region, so an index means the same entry
	// wherever it is fetched -- the fused path concatenates the regions and
	// would otherwise need per-region table offsets.
	std::vector<float> tabVv, tabVvfo, tabVvfn, tabIi, tabIifo, tabIifn;
	std::unordered_map<PmlCoeffKey, uint32_t, PmlCoeffKeyHash> coeffLookup;

	for (size_t eIdx = 0; eIdx < nExts; eIdx++)
	{
		Operator_Extension* ext = Op->GetExtension(eIdx);
		auto* upml = dynamic_cast<Operator_Ext_UPML*>(ext);
		if (!upml) continue;

		GpuUPMLData data;
		uint32_t pNx = upml->m_numLines[0];
		uint32_t pNy = upml->m_numLines[1];
		uint32_t pNz = upml->m_numLines[2];
		uint32_t pN  = pNx * pNy * pNz;
		VkDeviceSize pmlBufSize = 3 * pN * sizeof(FDTD_FLOAT);  // [3][pNx][pNy][pNz]

		if (pN == 0) continue;

		data.pc.Nx = numLines[0]; data.pc.Ny = numLines[1]; data.pc.Nz = numLines[2];
		data.pc.pNx = pNx; data.pc.pNy = pNy; data.pc.pNz = pNz;
		data.pc.pStartX = upml->m_StartPos[0];
		data.pc.pStartY = upml->m_StartPos[1];
		data.pc.pStartZ = upml->m_StartPos[2];
		data.totalCells  = pN;

		auto validatePml = [pN](const ArrayLib::ArrayNIJK<FDTD_FLOAT>& values,
		                             const char* name)
		{
			for (uint32_t i = 0; i < 3u * pN; ++i)
			{
				if (!std::isfinite(values.data(i)))
					throw std::runtime_error(std::string("Engine_Vulkan: non-finite UPML ") +
					                         name + " coefficient at index " + std::to_string(i));
			}
		};
		validatePml(upml->vv, "VV");
		validatePml(upml->vvfo, "VVFO");
		validatePml(upml->vvfn, "VVFN");
		validatePml(upml->ii, "II");
		validatePml(upml->iifo, "IIFO");
		validatePml(upml->iifn, "IIFN");

		// Create and upload auxiliary flux buffers (initialized to zero)
		VkBufferUsageFlags rwUsage = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		CreateGpuBuf(data.voltFlux, pmlBufSize, rwUsage, dLoc);
		CreateGpuBuf(data.currFlux, pmlBufSize, rwUsage, dLoc);
		{
			std::vector<float> zeros(3 * pN, 0.0f);
			UploadGpuBuf(data.voltFlux, zeros.data(), pmlBufSize);
			UploadGpuBuf(data.currFlux, zeros.data(), pmlBufSize);
		}

		// Deduplicate this region's coefficients into the shared table.
		// The tuple is the six values one cell-component needs across all four
		// passes, so a single index serves the voltage and current shaders
		// alike. Redundancy is large and robust: the coefficients follow the
		// grading profile (a function of depth into the PML) and the local
		// material, so they barely vary across the transverse plane. Measured
		// over all six regions of an 80^3 PML_8 grid, 821016 entries collapse
		// to ~1100 distinct tuples, and geometric mesh grading only takes that
		// to ~1400 -- unlike the Yee operator table, which grading destroys.
		data.coefIdx.resize(3u * pN);
		for (uint32_t c = 0; c < 3u * pN; ++c)
		{
			PmlCoeffKey key;
			const float vals[6] = {upml->vv.data(c),   upml->vvfo.data(c), upml->vvfn.data(c),
			                       upml->ii.data(c),   upml->iifo.data(c), upml->iifn.data(c)};
			std::memcpy(key.w, vals, sizeof(key.w));

			auto it = coeffLookup.find(key);
			if (it == coeffLookup.end())
			{
				uint32_t idx = (uint32_t)tabVv.size();
				tabVv.push_back(vals[0]);   tabVvfo.push_back(vals[1]); tabVvfn.push_back(vals[2]);
				tabIi.push_back(vals[3]);   tabIifo.push_back(vals[4]); tabIifn.push_back(vals[5]);
				coeffLookup.emplace(key, idx);
				data.coefIdx[c] = idx;
			}
			else
			{
				data.coefIdx[c] = it->second;
			}
		}

		m_gpuUPML.push_back(std::move(data));
	}

	m_hasGPU_UPML = !m_gpuUPML.empty();
	if (!m_hasGPU_UPML)
		return;

	// ---- Shared coefficient tables -------------------------------------
	m_pmlTabCount = (uint32_t)tabVv.size();
	// Narrow the index to the smallest width that can address the table; it is
	// read once per coefficient entry per pass, so its width is most of the
	// remaining PML traffic once the coefficients themselves are deduplicated.
	if (m_pmlTabCount <= 0x100u)
		m_pmlIdxBits = 8;
	else if (m_pmlTabCount <= 0x10000u)
		m_pmlIdxBits = 16;
	else
		m_pmlIdxBits = 32;

	m_pmlTabBufSize = (VkDeviceSize)m_pmlTabCount * sizeof(float);
	auto makeTable = [&](GpuBuf& gb, const std::vector<float>& src)
	{
		CreateGpuBuf(gb, m_pmlTabBufSize, usage, dLoc);
		UploadGpuBuf(gb, src.data(), m_pmlTabBufSize);
	};
	makeTable(m_pmlTabVv,   tabVv);
	makeTable(m_pmlTabVvfo, tabVvfo);
	makeTable(m_pmlTabVvfn, tabVvfn);
	makeTable(m_pmlTabIi,   tabIi);
	makeTable(m_pmlTabIifo, tabIifo);
	makeTable(m_pmlTabIifn, tabIifn);

	// ---- Per-region index buffers and descriptors ----------------------
	for (auto& data : m_gpuUPML)
	{
		const size_t entries = data.coefIdx.size();
		data.pmlIdxBufSize = NarrowIndexBufSize(entries, m_pmlIdxBits);
		CreateGpuBuf(data.pmlIdx, data.pmlIdxBufSize, usage, dLoc);
		{
			std::vector<uint32_t> packed =
				PackNarrowIndices(data.coefIdx.data(), entries, m_pmlIdxBits);
			packed.resize((size_t)data.pmlIdxBufSize / sizeof(uint32_t), 0u);
			UploadGpuBuf(data.pmlIdx, packed.data(), data.pmlIdxBufSize);
		}

		const VkDeviceSize fluxSize = 3 * (VkDeviceSize)data.totalCells * sizeof(FDTD_FLOAT);

		auto allocDescSet = [&](VkDescriptorSetLayout layout) -> VkDescriptorSet
		{
			VkDescriptorSetAllocateInfo ai{};
			ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			ai.descriptorPool     = m_extDescPool;
			ai.descriptorSetCount = 1;
			ai.pSetLayouts        = &layout;
			VkDescriptorSet set;
			VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, &set));
			return set;
		};

		data.preVoltDesc  = allocDescSet(m_upmlPreVoltDescLayout);
		data.postVoltDesc = allocDescSet(m_upmlPostVoltDescLayout);
		data.preCurrDesc  = allocDescSet(m_upmlPreCurrDescLayout);
		data.postCurrDesc = allocDescSet(m_upmlPostCurrDescLayout);

		// preVolt = [volt, volt_flux, pml_idx, tab_vv, tab_vvfo]
		{
			VkBuffer bufs[5]      = {m_voltBuf, data.voltFlux.buffer, data.pmlIdx.buffer,
			                         m_pmlTabVv.buffer, m_pmlTabVvfo.buffer};
			VkDeviceSize sizes[5] = {m_fieldBufSize, fluxSize, data.pmlIdxBufSize,
			                         m_pmlTabBufSize, m_pmlTabBufSize};
			WriteDescriptorBuffers(m_device, data.preVoltDesc, bufs, sizes, 5);
		}
		// postVolt = [volt, volt_flux, pml_idx, tab_vvfn]
		{
			VkBuffer bufs[4]      = {m_voltBuf, data.voltFlux.buffer, data.pmlIdx.buffer,
			                         m_pmlTabVvfn.buffer};
			VkDeviceSize sizes[4] = {m_fieldBufSize, fluxSize, data.pmlIdxBufSize, m_pmlTabBufSize};
			WriteDescriptorBuffers(m_device, data.postVoltDesc, bufs, sizes, 4);
		}
		// preCurr = [curr, curr_flux, pml_idx, tab_ii, tab_iifo]
		{
			VkBuffer bufs[5]      = {m_currBuf, data.currFlux.buffer, data.pmlIdx.buffer,
			                         m_pmlTabIi.buffer, m_pmlTabIifo.buffer};
			VkDeviceSize sizes[5] = {m_fieldBufSize, fluxSize, data.pmlIdxBufSize,
			                         m_pmlTabBufSize, m_pmlTabBufSize};
			WriteDescriptorBuffers(m_device, data.preCurrDesc, bufs, sizes, 5);
		}
		// postCurr = [curr, curr_flux, pml_idx, tab_iifn]
		{
			VkBuffer bufs[4]      = {m_currBuf, data.currFlux.buffer, data.pmlIdx.buffer,
			                         m_pmlTabIifn.buffer};
			VkDeviceSize sizes[4] = {m_fieldBufSize, fluxSize, data.pmlIdxBufSize, m_pmlTabBufSize};
			WriteDescriptorBuffers(m_device, data.postCurrDesc, bufs, sizes, 4);
		}
	}

	if (g_settings.GetVerboseLevel() > 0)
	{
		size_t entries = 0;
		for (const auto& u : m_gpuUPML) entries += u.coefIdx.size();
		size_t before = entries * 6 * sizeof(float);
		size_t after  = (size_t)m_pmlTabCount * 6 * sizeof(float)
		              + entries * m_pmlIdxBits / 8;
		cout << "  UPML coefficients: " << entries << " entries -> " << m_pmlTabCount
		     << " unique (" << (before / 1024) << " KB -> " << (after / 1024)
		     << " KB, " << m_pmlIdxBits << "-bit index)" << endl;
	}
}

void Engine_Vulkan::SetupGPU_Dispersive()
{
	// Iterate operator extensions
	size_t nExts = Op->GetNumberOfExtentions();
	VkBufferUsageFlags usage   = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkBufferUsageFlags rwUsage = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VkMemoryPropertyFlags dLoc = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	uint32_t N  = m_fieldN;
	uint32_t sYZ = m_strideYZ;

	for (size_t eIdx = 0; eIdx < nExts; eIdx++)
	{
		Operator_Extension* ext = Op->GetExtension(eIdx);
		if (auto* deb = dynamic_cast<Operator_Ext_DebyeMaterial*>(ext))
		{
			SetupGPU_Debye(deb);
			continue;
		}
		auto* lor = dynamic_cast<Operator_Ext_LorentzMaterial*>(ext);
		if (!lor) continue;

		for (int order = 0; order < lor->m_Order; order++)
		{
			uint32_t cnt = lor->m_LM_Count[order];
			if (cnt == 0) continue;

			GpuDispData data;
			data.count     = cnt;
			data.voltADEOn = lor->m_volt_ADE_On[order];
			data.currADEOn = lor->m_curr_ADE_On[order];
			data.voltLorADEOn = data.voltADEOn && lor->m_volt_Lor_ADE_On && lor->m_volt_Lor_ADE_On[order];
			data.currLorADEOn = data.currADEOn && lor->m_curr_Lor_ADE_On && lor->m_curr_Lor_ADE_On[order];

			data.voltPC = {cnt, numLines[0], numLines[1], numLines[2], data.voltLorADEOn ? 1u : 0u};
			data.currPC = {cnt, numLines[0], numLines[1], numLines[2], data.currLorADEOn ? 1u : 0u};

			VkDeviceSize adeSize = 3 * cnt * sizeof(FDTD_FLOAT);
			VkDeviceSize idxSize = 3 * cnt * sizeof(uint32_t);

			// Entries are reordered by x so that a temporal-blocking slab maps
			// to a contiguous run of them, exactly as the excitation list is.
			// Stable, so entries sharing an x keep the operator's order.
			//
			// Safe across the two orders of a conducting sheet even though both
			// carry the *identical* position list and both subtract from the
			// same engine cell: each order is its own dispatch, the two run in
			// sequence separated by a barrier (RecordDispersiveSlabPhase()
			// records it between the apply dispatches of a slab), and sorting
			// identical lists by the same key gives both the same permutation.
			// So a cell still sees order 0 then order 1, which is what the flat
			// path does and what makes the result independent of the reordering.
			std::vector<uint32_t> perm(cnt);
			for (uint32_t i = 0; i < cnt; i++)
				perm[i] = i;
			const unsigned int* px = lor->m_LM_pos[order][0];
			std::stable_sort(perm.begin(), perm.end(),
			                 [px](uint32_t a, uint32_t b) { return px[a] < px[b]; });
			{
				std::vector<uint32_t> sortedX(cnt);
				for (uint32_t i = 0; i < cnt; i++)
					sortedX[i] = px[perm[i]];
				BuildSlabEntryIndex(sortedX, data.xStart);
			}

			// Flatten position indices: [3][count]
			std::vector<uint32_t> posIdx(3 * cnt);
			for (uint32_t i = 0; i < cnt; i++)
			{
				posIdx[i]           = lor->m_LM_pos[order][0][perm[i]];
				posIdx[cnt + i]     = lor->m_LM_pos[order][1][perm[i]];
				posIdx[2*cnt + i]   = lor->m_LM_pos[order][2][perm[i]];
			}
			CreateGpuBuf(data.posIdx, idxSize, usage, dLoc);
			UploadGpuBuf(data.posIdx, posIdx.data(), idxSize);

			// ADE state (zero-initialized)
			std::vector<float> zeros(3 * cnt, 0.0f);
			CreateGpuBuf(data.voltADE, adeSize, rwUsage, dLoc);
			UploadGpuBuf(data.voltADE, zeros.data(), adeSize);
			CreateGpuBuf(data.currADE, adeSize, rwUsage, dLoc);
			UploadGpuBuf(data.currADE, zeros.data(), adeSize);

			if (data.voltLorADEOn)
			{
				CreateGpuBuf(data.voltLorADE, adeSize, rwUsage, dLoc);
				UploadGpuBuf(data.voltLorADE, zeros.data(), adeSize);
			}
			if (data.currLorADEOn)
			{
				CreateGpuBuf(data.currLorADE, adeSize, rwUsage, dLoc);
				UploadGpuBuf(data.currLorADE, zeros.data(), adeSize);
			}

			// Flatten coefficient arrays: [3][count]
			auto flattenCoef = [&](FDTD_FLOAT*** arr) -> std::vector<float>
			{
				std::vector<float> flat(3 * cnt);
				for (int n = 0; n < 3; n++)
					for (uint32_t i = 0; i < cnt; i++)
						flat[n * cnt + i] = arr[order][n][perm[i]];
				return flat;
			};

			auto uploadCoef = [&](GpuBuf& gb, FDTD_FLOAT*** arr)
			{
				auto flat = flattenCoef(arr);
				CreateGpuBuf(gb, adeSize, usage, dLoc);
				UploadGpuBuf(gb, flat.data(), adeSize);
			};

			if (data.voltADEOn)
			{
				uploadCoef(data.vIntADE, lor->v_int_ADE);
				uploadCoef(data.vExtADE, lor->v_ext_ADE);
			}
			if (data.currADEOn)
			{
				uploadCoef(data.iIntADE, lor->i_int_ADE);
				uploadCoef(data.iExtADE, lor->i_ext_ADE);
			}
			if (data.voltLorADEOn)
				uploadCoef(data.vLorADE, lor->v_Lor_ADE);
			if (data.currLorADEOn)
				uploadCoef(data.iLorADE, lor->i_Lor_ADE);

			// Allocate descriptor sets
			auto allocDescSet = [&](VkDescriptorSetLayout layout) -> VkDescriptorSet
			{
				VkDescriptorSetAllocateInfo ai{};
				ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				ai.descriptorPool     = m_extDescPool;
				ai.descriptorSetCount = 1;
				ai.pSetLayouts        = &layout;
				VkDescriptorSet set;
				VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, &set));
				return set;
			};

			// Use dummy buffer for unused Lorentz slots when hasLorADE=false
			VkBuffer voltDummy = data.voltADE.buffer;
			VkBuffer currDummy = data.currADE.buffer;
			VkBuffer lorVoltBuf = data.voltLorADEOn ? data.voltLorADE.buffer : voltDummy;
			VkBuffer lorCurrBuf = data.currLorADEOn ? data.currLorADE.buffer : currDummy;
			VkBuffer vIntBuf = data.voltADEOn ? data.vIntADE.buffer : voltDummy;
			VkBuffer vExtBuf = data.voltADEOn ? data.vExtADE.buffer : voltDummy;
			VkBuffer iIntBuf = data.currADEOn ? data.iIntADE.buffer : currDummy;
			VkBuffer iExtBuf = data.currADEOn ? data.iExtADE.buffer : currDummy;
			VkBuffer vLorBuf = data.voltLorADEOn ? data.vLorADE.buffer : voltDummy;
			VkBuffer iLorBuf = data.currLorADEOn ? data.iLorADE.buffer : currDummy;

			data.preVoltDesc = allocDescSet(m_dispPreDescLayout);
			{
				VkBuffer bufs[7]      = {m_voltBuf, data.voltADE.buffer, lorVoltBuf,
				                          data.posIdx.buffer, vIntBuf, vExtBuf, vLorBuf};
				VkDeviceSize sizes[7] = {m_fieldBufSize, adeSize, adeSize, idxSize, adeSize, adeSize, adeSize};
				WriteDescriptorBuffers(m_device, data.preVoltDesc, bufs, sizes, 7);
			}

			data.preCurrDesc = allocDescSet(m_dispPreDescLayout);
			{
				VkBuffer bufs[7]      = {m_currBuf, data.currADE.buffer, lorCurrBuf,
				                          data.posIdx.buffer, iIntBuf, iExtBuf, iLorBuf};
				VkDeviceSize sizes[7] = {m_fieldBufSize, adeSize, adeSize, idxSize, adeSize, adeSize, adeSize};
				WriteDescriptorBuffers(m_device, data.preCurrDesc, bufs, sizes, 7);
			}

			data.applyVoltDesc = allocDescSet(m_dispApplyDescLayout);
			{
				VkBuffer bufs[3]      = {m_voltBuf, data.voltADE.buffer, data.posIdx.buffer};
				VkDeviceSize sizes[3] = {m_fieldBufSize, adeSize, idxSize};
				WriteDescriptorBuffers(m_device, data.applyVoltDesc, bufs, sizes, 3);
			}

			data.applyCurrDesc = allocDescSet(m_dispApplyDescLayout);
			{
				VkBuffer bufs[3]      = {m_currBuf, data.currADE.buffer, data.posIdx.buffer};
				VkDeviceSize sizes[3] = {m_fieldBufSize, adeSize, idxSize};
				WriteDescriptorBuffers(m_device, data.applyCurrDesc, bufs, sizes, 3);
			}

			m_gpuDisp.push_back(std::move(data));
		}
	}

	m_hasGPU_Dispersive = !m_gpuDisp.empty();
}

// A Debye material is one entry of m_gpuDisp, like a Lorentz order, with its
// own two passes (debye_pre_voltage, debye_apply_voltage) on one descriptor set.
void Engine_Vulkan::SetupGPU_Debye(const Operator_Ext_DebyeMaterial* deb)
{
	const uint32_t cnt = deb->m_LM_Count.at(0);
	const uint32_t poles = (uint32_t)deb->m_PoleCount;
	if (cnt == 0 || poles == 0)
		return;
	VkBufferUsageFlags usage   = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkBufferUsageFlags rwUsage = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VkMemoryPropertyFlags dLoc = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	GpuDispData data;
	data.debye     = true;
	data.poles     = poles;
	data.count     = cnt;
	data.voltADEOn = true;    // the passes run in the voltage pre and apply slots
	data.currADEOn = false;
	data.voltPC = {cnt, numLines[0], numLines[1], numLines[2], poles};
	data.currPC = data.voltPC;

	// sorted by x, as the Lorentz entries, so that a slab is a contiguous run
	std::vector<uint32_t> perm(cnt);
	for (uint32_t i = 0; i < cnt; i++)
		perm[i] = i;
	unsigned int* const* pos = deb->m_LM_pos[0];
	std::stable_sort(perm.begin(), perm.end(), [&](uint32_t a, uint32_t b) { return pos[0][a] < pos[0][b]; });
	std::vector<uint32_t> sortedX(cnt);
	for (uint32_t i = 0; i < cnt; i++)
		sortedX[i] = pos[0][perm[i]];
	BuildSlabEntryIndex(sortedX, data.xStart);

	const VkDeviceSize cellSize = 3 * (VkDeviceSize)cnt * sizeof(FDTD_FLOAT);
	const VkDeviceSize poleSize = poles * cellSize;
	const VkDeviceSize idxSize  = 3 * (VkDeviceSize)cnt * sizeof(uint32_t);

	std::vector<uint32_t> posIdx(3 * cnt);
	for (int n = 0; n < 3; n++)
		for (uint32_t i = 0; i < cnt; i++)
			posIdx[n * cnt + i] = pos[n][perm[i]];
	CreateGpuBuf(data.posIdx, idxSize, usage, dLoc);
	UploadGpuBuf(data.posIdx, posIdx.data(), idxSize);

	// state, zero-initialized
	std::vector<float> zeros(poles * 3 * cnt, 0.0f);
	CreateGpuBuf(data.debyeW, poleSize, rwUsage, dLoc);
	UploadGpuBuf(data.debyeW, zeros.data(), poleSize);
	CreateGpuBuf(data.debyePre, cellSize, rwUsage, dLoc);
	UploadGpuBuf(data.debyePre, zeros.data(), cellSize);

	// coefficients: [poles][3][count] and [3][count], in the sorted order
	auto upload = [&](GpuBuf& gb, FDTD_FLOAT* const* const* arr, uint32_t nPoles)
	{
		std::vector<float> flat(nPoles * 3 * cnt);
		for (uint32_t o = 0; o < nPoles; o++)
			for (int n = 0; n < 3; n++)
				for (uint32_t i = 0; i < cnt; i++)
					flat[(o * 3 + n) * cnt + i] = arr[o][n][perm[i]];
		const VkDeviceSize size = flat.size() * sizeof(float);
		CreateGpuBuf(gb, size, usage, dLoc);
		UploadGpuBuf(gb, flat.data(), size);
	};
	upload(data.debyeRelax, deb->v_relax_ADE, poles);
	upload(data.debyeDrive, deb->v_drive_ADE, poles);
	FDTD_FLOAT* const* solve = deb->v_solve_ADE;
	upload(data.debyeSolve, &solve, 1);

	VkDescriptorSetAllocateInfo ai{};
	ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	ai.descriptorPool     = m_extDescPool;
	ai.descriptorSetCount = 1;
	ai.pSetLayouts        = &m_debyeDescLayout;
	VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, &data.debyeDesc));
	VkBuffer bufs[7]      = {m_voltBuf, data.debyeW.buffer, data.debyePre.buffer, data.posIdx.buffer,
	                         data.debyeRelax.buffer, data.debyeDrive.buffer, data.debyeSolve.buffer};
	VkDeviceSize sizes[7] = {m_fieldBufSize, poleSize, cellSize, idxSize, poleSize, poleSize, cellSize};
	WriteDescriptorBuffers(m_device, data.debyeDesc, bufs, sizes, 7);

	m_gpuDisp.push_back(std::move(data));
}

namespace
{
//! Sort four parallel TF/SF arrays by x, then by field index, stably.
/*!
  Injection points on the TF/SF box's edges belong to two faces, so several
  entries write the same field element. The shaders own one run of equal
  indices per thread instead of racing on a non-atomic `+=` (see the note in
  tfsf_voltage.comp), which requires the entries to arrive grouped. The sort is
  stable so that entries sharing an index keep the relative order the CPU
  extension applies them in.

  Sorting by x first is what lets the blocked schedule give an x-slab a
  contiguous entry range.  It cannot split a run: entries sharing a field
  index share its x.  The x of each sorted entry is returned in `sortedX`.
*/
void SortTFSFByFieldIndex(std::vector<uint32_t>& idx, std::vector<float>& amp,
                          std::vector<float>& delta, std::vector<uint32_t>& delay,
                          uint32_t N, uint32_t strideYZ, std::vector<uint32_t>& sortedX)
{
	const size_t n = idx.size();
	const auto xOf = [N, strideYZ](uint32_t i) { return (i % N) / strideYZ; };
	sortedX.resize(n);
	if (n < 2)
	{
		for (size_t i = 0; i < n; ++i) sortedX[i] = xOf(idx[i]);
		return;
	}

	std::vector<uint32_t> order(n);
	for (size_t i = 0; i < n; ++i) order[i] = (uint32_t)i;
	std::stable_sort(order.begin(), order.end(),
	                 [&idx, &xOf](uint32_t a, uint32_t b)
	                 {
	                 	const uint32_t xa = xOf(idx[a]), xb = xOf(idx[b]);
	                 	return (xa != xb) ? (xa < xb) : (idx[a] < idx[b]);
	                 });

	std::vector<uint32_t> sIdx(n), sDelay(n);
	std::vector<float>    sAmp(n), sDelta(n);
	for (size_t i = 0; i < n; ++i)
	{
		sIdx[i]   = idx[order[i]];
		sAmp[i]   = amp[order[i]];
		sDelta[i] = delta[order[i]];
		sDelay[i] = delay[order[i]];
	}
	idx.swap(sIdx); amp.swap(sAmp); delta.swap(sDelta); delay.swap(sDelay);
	for (size_t i = 0; i < n; ++i) sortedX[i] = xOf(idx[i]);
}
} // namespace

void Engine_Vulkan::SetupGPU_TFSF()
{
	// Iterate operator extensions
	size_t nExts = Op->GetNumberOfExtentions();
	VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags dLoc = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	uint32_t N = m_fieldN;
	uint32_t sYZ = m_strideYZ;

	for (size_t eIdx = 0; eIdx < nExts; eIdx++)
	{
		Operator_Extension* ext = Op->GetExtension(eIdx);
		auto* tfsf = dynamic_cast<Operator_Ext_TFSF*>(ext);
		if (!tfsf) continue;

		Excitation* exc = tfsf->m_Exc;
		if (!exc) continue;

		// Count total injection points for voltage and current
		uint32_t voltTotal = 0, currTotal = 0;
		for (int n = 0; n < 3; n++)
		{
			int nP = (n+1)%3, nPP = (n+2)%3;
			uint32_t faceSize = tfsf->m_numLines[nP] * tfsf->m_numLines[nPP];
			for (int side = 0; side < 2; side++)
			{
				if (tfsf->m_ActiveDir[n][side])
				{
					voltTotal += 2 * faceSize;  // 2 tangential components
					currTotal += 2 * faceSize;
				}
			}
		}

		if (voltTotal == 0 && currTotal == 0) continue;

		// Flatten voltage injection points
		std::vector<float>    vAmp(voltTotal), vDelta(voltTotal);
		std::vector<uint32_t> vIdx(voltTotal), vDelay(voltTotal);
		uint32_t vi = 0;

		for (int n = 0; n < 3; n++)
		{
			int nP = (n+1)%3, nPP = (n+2)%3;
			for (int side = 0; side < 2; side++)
			{
				if (!tfsf->m_ActiveDir[n][side]) continue;

				uint32_t linePos = (side == 0) ? tfsf->m_Start[n] : tfsf->m_Stop[n];
				unsigned int ui_pos = 0;
				unsigned int pos[3];

				for (unsigned int i = 0; i < tfsf->m_numLines[nP]; i++)
				{
					for (unsigned int j = 0; j < tfsf->m_numLines[nPP]; j++)
					{
						pos[nP]  = tfsf->m_Start[nP] + i;
						pos[nPP] = tfsf->m_Start[nPP] + j;
						pos[n]   = linePos;

						// nP component
						uint32_t gIdx_nP = nP * N + pos[0] * sYZ + pos[1] * numLines[2] + pos[2];
						vIdx[vi]   = gIdx_nP;
						vAmp[vi]   = tfsf->m_VoltAmp[n][side][0][ui_pos];
						vDelta[vi] = tfsf->m_VoltDelayDelta[n][side][0][ui_pos];
						vDelay[vi] = tfsf->m_VoltDelay[n][side][0][ui_pos];
						vi++;

						// nPP component
						uint32_t gIdx_nPP = nPP * N + pos[0] * sYZ + pos[1] * numLines[2] + pos[2];
						vIdx[vi]   = gIdx_nPP;
						vAmp[vi]   = tfsf->m_VoltAmp[n][side][1][ui_pos];
						vDelta[vi] = tfsf->m_VoltDelayDelta[n][side][1][ui_pos];
						vDelay[vi] = tfsf->m_VoltDelay[n][side][1][ui_pos];
						vi++;

						ui_pos++;
					}
				}
			}
		}

		// Flatten current injection points
		std::vector<float>    cAmp(currTotal), cDelta(currTotal);
		std::vector<uint32_t> cIdx(currTotal), cDelay(currTotal);
		uint32_t ci_idx = 0;

		for (int n = 0; n < 3; n++)
		{
			int nP = (n+1)%3, nPP = (n+2)%3;
			for (int side = 0; side < 2; side++)
			{
				if (!tfsf->m_ActiveDir[n][side]) continue;

				// Current injection: lower face uses Start[n]-1, upper face uses Stop[n]
				uint32_t linePos = (side == 0) ? tfsf->m_Start[n] - 1 : tfsf->m_Stop[n];
				unsigned int ui_pos = 0;
				unsigned int pos[3];

				for (unsigned int i = 0; i < tfsf->m_numLines[nP]; i++)
				{
					for (unsigned int j = 0; j < tfsf->m_numLines[nPP]; j++)
					{
						pos[nP]  = tfsf->m_Start[nP] + i;
						pos[nPP] = tfsf->m_Start[nPP] + j;
						pos[n]   = linePos;

						uint32_t gIdx_nP = nP * N + pos[0] * sYZ + pos[1] * numLines[2] + pos[2];
						cIdx[ci_idx]   = gIdx_nP;
						cAmp[ci_idx]   = tfsf->m_CurrAmp[n][side][0][ui_pos];
						cDelta[ci_idx] = tfsf->m_CurrDelayDelta[n][side][0][ui_pos];
						cDelay[ci_idx] = tfsf->m_CurrDelay[n][side][0][ui_pos];
						ci_idx++;

						uint32_t gIdx_nPP = nPP * N + pos[0] * sYZ + pos[1] * numLines[2] + pos[2];
						cIdx[ci_idx]   = gIdx_nPP;
						cAmp[ci_idx]   = tfsf->m_CurrAmp[n][side][1][ui_pos];
						cDelta[ci_idx] = tfsf->m_CurrDelayDelta[n][side][1][ui_pos];
						cDelay[ci_idx] = tfsf->m_CurrDelay[n][side][1][ui_pos];
						ci_idx++;

						ui_pos++;
					}
				}
			}
		}

		{
			std::vector<uint32_t> sortedX;
			SortTFSFByFieldIndex(vIdx, vAmp, vDelta, vDelay, N, sYZ, sortedX);
			BuildSlabEntryIndex(sortedX, m_tfsfVoltXStart);
			SortTFSFByFieldIndex(cIdx, cAmp, cDelta, cDelay, N, sYZ, sortedX);
			BuildSlabEntryIndex(sortedX, m_tfsfCurrXStart);
		}

		m_gpuTFSF.voltCount = voltTotal;
		m_gpuTFSF.currCount = currTotal;

		// Upload voltage TF/SF data
		if (voltTotal > 0)
		{
			VkDeviceSize fSize = voltTotal * sizeof(float);
			VkDeviceSize iSize = voltTotal * sizeof(uint32_t);
			CreateGpuBuf(m_gpuTFSF.voltAmp,   fSize, usage, dLoc);
			CreateGpuBuf(m_gpuTFSF.voltDelta,  fSize, usage, dLoc);
			CreateGpuBuf(m_gpuTFSF.voltIdx,    iSize, usage, dLoc);
			CreateGpuBuf(m_gpuTFSF.voltDelay,  iSize, usage, dLoc);
			UploadGpuBuf(m_gpuTFSF.voltAmp,   vAmp.data(),   fSize);
			UploadGpuBuf(m_gpuTFSF.voltDelta,  vDelta.data(), fSize);
			UploadGpuBuf(m_gpuTFSF.voltIdx,    vIdx.data(),   iSize);
			UploadGpuBuf(m_gpuTFSF.voltDelay,  vDelay.data(), iSize);

			// Signal: H-field signal for voltage injection
			VkDeviceSize sigSize = exc->GetLength() * sizeof(FDTD_FLOAT);
			CreateGpuBuf(m_gpuTFSF.voltSignal, sigSize, usage, dLoc);
			UploadGpuBuf(m_gpuTFSF.voltSignal, exc->GetCurrentSignal(), sigSize);
		}

		// Upload current TF/SF data
		if (currTotal > 0)
		{
			VkDeviceSize fSize = currTotal * sizeof(float);
			VkDeviceSize iSize = currTotal * sizeof(uint32_t);
			CreateGpuBuf(m_gpuTFSF.currAmp,   fSize, usage, dLoc);
			CreateGpuBuf(m_gpuTFSF.currDelta,  fSize, usage, dLoc);
			CreateGpuBuf(m_gpuTFSF.currIdx,    iSize, usage, dLoc);
			CreateGpuBuf(m_gpuTFSF.currDelay,  iSize, usage, dLoc);
			UploadGpuBuf(m_gpuTFSF.currAmp,   cAmp.data(),   fSize);
			UploadGpuBuf(m_gpuTFSF.currDelta,  cDelta.data(), fSize);
			UploadGpuBuf(m_gpuTFSF.currIdx,    cIdx.data(),   iSize);
			UploadGpuBuf(m_gpuTFSF.currDelay,  cDelay.data(), iSize);

			// Signal: E-field signal for current injection
			VkDeviceSize sigSize = exc->GetLength() * sizeof(FDTD_FLOAT);
			CreateGpuBuf(m_gpuTFSF.currSignal, sigSize, usage, dLoc);
			UploadGpuBuf(m_gpuTFSF.currSignal, exc->GetVoltageSignal(), sigSize);
		}

		// Descriptor sets
		auto allocDescSet = [&](VkDescriptorSetLayout layout) -> VkDescriptorSet
		{
			VkDescriptorSetAllocateInfo ai{};
			ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			ai.descriptorPool     = m_extDescPool;
			ai.descriptorSetCount = 1;
			ai.pSetLayouts        = &layout;
			VkDescriptorSet set;
			VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, &set));
			return set;
		};

		VkDeviceSize sigSize = exc->GetLength() * sizeof(FDTD_FLOAT);

		if (voltTotal > 0)
		{
			m_gpuTFSF.voltDesc = allocDescSet(m_tfsfDescLayout);
			VkDeviceSize fSize = voltTotal * sizeof(float);
			VkDeviceSize iSize = voltTotal * sizeof(uint32_t);
			VkBuffer bufs[6]      = {m_voltBuf, m_gpuTFSF.voltAmp.buffer, m_gpuTFSF.voltDelta.buffer,
			                          m_gpuTFSF.voltIdx.buffer, m_gpuTFSF.voltDelay.buffer,
			                          m_gpuTFSF.voltSignal.buffer};
			VkDeviceSize sizes[6] = {m_fieldBufSize, fSize, fSize, iSize, iSize, sigSize};
			WriteDescriptorBuffers(m_device, m_gpuTFSF.voltDesc, bufs, sizes, 6);
		}

		if (currTotal > 0)
		{
			m_gpuTFSF.currDesc = allocDescSet(m_tfsfDescLayout);
			VkDeviceSize fSize = currTotal * sizeof(float);
			VkDeviceSize iSize = currTotal * sizeof(uint32_t);
			VkBuffer bufs[6]      = {m_currBuf, m_gpuTFSF.currAmp.buffer, m_gpuTFSF.currDelta.buffer,
			                          m_gpuTFSF.currIdx.buffer, m_gpuTFSF.currDelay.buffer,
			                          m_gpuTFSF.currSignal.buffer};
			VkDeviceSize sizes[6] = {m_fieldBufSize, fSize, fSize, iSize, iSize, sigSize};
			WriteDescriptorBuffers(m_device, m_gpuTFSF.currDesc, bufs, sizes, 6);
		}

		m_hasGPU_TFSF = true;
		break;  // Only one TF/SF extension expected
	}
}

void Engine_Vulkan::SetupGPU_Mur()
{
	// Iterate operator extensions
	size_t nExts = Op->GetNumberOfExtentions();
	VkBufferUsageFlags usage   = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkBufferUsageFlags rwUsage = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VkMemoryPropertyFlags dLoc = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	std::vector<float> coeffP, coeffPP;
	for (size_t eIdx = 0; eIdx < nExts; eIdx++)
	{
		Operator_Extension* ext = Op->GetExtension(eIdx);
		auto* mur = dynamic_cast<Operator_Ext_Mur_ABC*>(ext);
		if (!mur) continue;

		GpuMurData data;
		uint32_t numLinesP  = mur->m_numLines[0];
		uint32_t numLinesPP = mur->m_numLines[1];
		uint32_t total      = numLinesP * numLinesPP;
		if (total == 0) continue;

		data.totalCells = total;
		data.pc.Nx = numLines[0]; data.pc.Ny = numLines[1]; data.pc.Nz = numLines[2];
		data.pc.ny          = mur->m_ny;
		data.pc.lineNr      = mur->m_LineNr;
		data.pc.lineNrShift = mur->m_LineNr_Shift;
		data.pc.numLinesP   = numLinesP;
		data.pc.numLinesPP  = numLinesPP;
		data.pc.faceOff     = (uint32_t)coeffP.size();
		coeffP.insert(coeffP.end(), mur->m_Mur_Coeff_nyP.data(), mur->m_Mur_Coeff_nyP.data() + total);
		coeffPP.insert(coeffPP.end(), mur->m_Mur_Coeff_nyPP.data(), mur->m_Mur_Coeff_nyPP.data() + total);
		m_gpuMur.push_back(std::move(data));
	}
	if (m_gpuMur.empty())
		return;

	// One set of buffers for all faces, each face at its faceOff: the fused
	// voltage kernel (SetupMurFusion()) reaches every face through them.
	m_murAllSize = coeffP.size() * sizeof(FDTD_FLOAT);
	{
		std::vector<float> zeros(coeffP.size(), 0.0f);
		for (GpuBuf* b : {&m_murAllNyP, &m_murAllNyPP, &m_murAllVoldNyP, &m_murAllVoldNyPP})
		{
			CreateGpuBuf(*b, m_murAllSize, rwUsage, dLoc);
			UploadGpuBuf(*b, zeros.data(), m_murAllSize);
		}
		CreateGpuBuf(m_murAllCoeffNyP,  m_murAllSize, usage, dLoc);
		CreateGpuBuf(m_murAllCoeffNyPP, m_murAllSize, usage, dLoc);
		UploadGpuBuf(m_murAllCoeffNyP,  coeffP.data(),  m_murAllSize);
		UploadGpuBuf(m_murAllCoeffNyPP, coeffPP.data(), m_murAllSize);
	}

	auto allocDescSet = [&](VkDescriptorSetLayout layout) -> VkDescriptorSet
	{
		VkDescriptorSetAllocateInfo ai{};
		ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		ai.descriptorPool     = m_extDescPool;
		ai.descriptorSetCount = 1;
		ai.pSetLayouts        = &layout;
		VkDescriptorSet set;
		VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, &set));
		return set;
	};
	// pre/post = [volt, mur nyP, mur nyPP, coeff nyP, coeff nyPP, vold nyP, vold nyPP]
	// apply    = [volt, mur nyP, mur nyPP]
	const VkDeviceSize sz = m_murAllSize;
	VkBuffer prePost[7]      = {m_voltBuf, m_murAllNyP.buffer, m_murAllNyPP.buffer,
	                            m_murAllCoeffNyP.buffer, m_murAllCoeffNyPP.buffer,
	                            m_murAllVoldNyP.buffer, m_murAllVoldNyPP.buffer};
	VkDeviceSize prePostSz[7] = {m_fieldBufSize, sz, sz, sz, sz, sz, sz};
	for (GpuMurData& data : m_gpuMur)
	{
		data.preDesc = allocDescSet(m_murPreDescLayout);
		WriteDescriptorBuffers(m_device, data.preDesc, prePost, prePostSz, 7);
		data.postDesc = allocDescSet(m_murPostDescLayout);
		WriteDescriptorBuffers(m_device, data.postDesc, prePost, prePostSz, 7);
		data.applyDesc = allocDescSet(m_murApplyDescLayout);
		WriteDescriptorBuffers(m_device, data.applyDesc, prePost, prePostSz, 3);
	}

	// Two faces with different normals share an edge, and both write the
	// field component along it.  Without a barrier between their apply
	// dispatches the GPU leaves the order undefined.  Engine::SortExtensionByPriority()
	// sorts stably and then reverses, so the CPU applies equal-priority faces
	// in reverse order of creation and the earlier face's value is the one that
	// stays.  Each face therefore skips the edges an earlier face owns.
	for (size_t a = 0; a < m_gpuMur.size(); ++a)
	{
		MurPC& pa = m_gpuMur[a].pc;
		const uint32_t nyP = (pa.ny + 1) % 3, nyPP = (pa.ny + 2) % 3;
		for (size_t b = 0; b < a; ++b)
		{
			const MurPC& pb = m_gpuMur[b].pc;
			if (pb.ny == nyP)
				pa.edgeSkip |= (pb.lineNr == 0) ? 1u : 2u;
			else if (pb.ny == nyPP)
				pa.edgeSkip |= (pb.lineNr == 0) ? 4u : 8u;
		}
	}

	m_hasGPU_Mur = !m_gpuMur.empty();
}

// Fold the Mur ABC's post and apply phases into the Yee voltage update
// (update_voltages_mur.comp): the thread that updates a shift cell writes the
// face's boundary value itself.  That saves two dispatches and two barriers
// per timestep in the flat sweep, and per slab step in the blocked one, where
// on an RX 7900 XTX each barrier costs ~1.8 us.
void Engine_Vulkan::SetupMurFusion()
{
	m_murFused = false;
	if (!m_hasGPU_Mur || m_gpuMur.empty() || m_gpuMur.size() > 6)
		return;
	if (const char* env = std::getenv("OPENEMS_GPU_MUR_FUSION"))
		if (std::atoi(env) == 0)
			return;
	// The flat order is Yee, every post, then the dispersive apply, TF/SF
	// injection, absorbing sheets and UPML post, then every Mur apply.  Folded,
	// the applies move ahead of all of those, so only without them is it the
	// same order.
	if (!m_gpuUPML.empty() || !m_gpuDisp.empty() || m_hasGPU_TFSF || !m_gpuLocalABC.empty())
		return;
	for (const auto& m : m_gpuMur)
		if (m.startTS != 0)
			return;

	const uint32_t nFaces = (uint32_t)m_gpuMur.size();
	std::vector<uint32_t> meta(8 * nFaces, 0);
	for (uint32_t f = 0; f < nFaces; ++f)
	{
		const MurPC& pc = m_gpuMur[f].pc;
		const uint32_t row[8] = {pc.ny, pc.lineNr, pc.lineNrShift, pc.numLinesP, pc.numLinesPP,
		                         pc.edgeSkip, pc.faceOff, 0};
		std::copy(row, row + 8, meta.begin() + 8 * f);
	}
	const VkDeviceSize metaSize = meta.size() * sizeof(uint32_t);
	CreateGpuBuf(m_murFaceBuf, metaSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	UploadGpuBuf(m_murFaceBuf, meta.data(), metaSize);

	const uint32_t nBindings = 12;
	std::vector<VkDescriptorSetLayoutBinding> bindings(nBindings);
	for (uint32_t i = 0; i < nBindings; i++)
		bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
	VkDescriptorSetLayoutCreateInfo dli{};
	dli.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	dli.bindingCount = nBindings;
	dli.pBindings    = bindings.data();
	VK_CHECK(vkCreateDescriptorSetLayout(m_device, &dli, nullptr, &m_murFusedDescLayout));

	VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nBindings};
	VkDescriptorPoolCreateInfo dpi{};
	dpi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	dpi.maxSets       = 1;
	dpi.poolSizeCount = 1;
	dpi.pPoolSizes    = &poolSize;
	VK_CHECK(vkCreateDescriptorPool(m_device, &dpi, nullptr, &m_murFusedDescPool));
	VkDescriptorSetAllocateInfo ai{};
	ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	ai.descriptorPool     = m_murFusedDescPool;
	ai.descriptorSetCount = 1;
	ai.pSetLayouts        = &m_murFusedDescLayout;
	VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, &m_murFusedDescSet));
	const VkDeviceSize sz = m_murAllSize;
	VkBuffer bufs[nBindings] = {m_voltBuf, m_currBuf, m_opIndexBuf, m_vvCompBuf, m_viCompBuf,
	                            m_murAllNyP.buffer, m_murAllNyPP.buffer,
	                            m_murAllCoeffNyP.buffer, m_murAllCoeffNyPP.buffer, m_murFaceBuf.buffer,
	                            m_murAllVoldNyP.buffer, m_murAllVoldNyPP.buffer};
	VkDeviceSize sizes[nBindings] = {m_fieldBufSize, m_fieldBufSize, m_opIndexBufSize,
	                                 m_coeffCompBufSize, m_coeffCompBufSize,
	                                 sz, sz, sz, sz, metaSize, sz, sz};
	WriteDescriptorBuffers(m_device, m_murFusedDescSet, bufs, sizes, nBindings);

	VkPushConstantRange pcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GridPC)};
	VkPipelineLayoutCreateInfo pli{};
	pli.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pli.setLayoutCount         = 1;
	pli.pSetLayouts            = &m_murFusedDescLayout;
	pli.pushConstantRangeCount = 1;
	pli.pPushConstantRanges    = &pcRange;
	VK_CHECK(vkCreatePipelineLayout(m_device, &pli, nullptr, &m_murFusedPipeLayout));

	const uint32_t spec[2] = {m_opIdxBits, nFaces};
	VkSpecializationMapEntry specEntry[2] = {
		{0, 0 * sizeof(uint32_t), sizeof(uint32_t)},
		{1, 1 * sizeof(uint32_t), sizeof(uint32_t)},
	};
	VkSpecializationInfo specInfo{};
	specInfo.mapEntryCount = 2;
	specInfo.pMapEntries   = specEntry;
	specInfo.dataSize      = sizeof(spec);
	specInfo.pData         = spec;
	VkShaderModule mod = CreateShaderModule(gpu_spirv::update_voltages_mur_data,
	                                        gpu_spirv::update_voltages_mur_size);
	VkPipelineShaderStageCreateInfo stage{};
	stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	stage.module = mod;
	stage.pName  = "main";
	stage.pSpecializationInfo = &specInfo;
	VkComputePipelineCreateInfo ci{};
	ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	ci.stage  = stage;
	ci.layout = m_murFusedPipeLayout;
	VK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache, 1, &ci, nullptr, &m_murFusedPipeline));
	vkDestroyShaderModule(m_device, mod, nullptr);

	m_murFused = true;
	cout << "Engine_Vulkan: Mur post/apply folded into the voltage update (" << nFaces << " faces)" << endl;
}

void Engine_Vulkan::SetupGPU_RLC()
{
	// Iterate operator extensions
	size_t nExts = Op->GetNumberOfExtentions();
	VkBufferUsageFlags usage   = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkBufferUsageFlags rwUsage = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VkMemoryPropertyFlags dLoc = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	uint32_t N   = m_fieldN;
	uint32_t sYZ = m_strideYZ;

	for (size_t eIdx = 0; eIdx < nExts; eIdx++)
	{
		Operator_Extension* ext = Op->GetExtension(eIdx);
		auto* rlc = dynamic_cast<Operator_Ext_LumpedRLC*>(ext);
		if (!rlc || rlc->RLC_count == 0) continue;

		uint32_t cnt = rlc->RLC_count;
		m_gpuRLC.count = cnt;

		VkDeviceSize fSize = cnt * sizeof(float);
		VkDeviceSize iSize = cnt * sizeof(uint32_t);

		// Elements in order of x, so that the blocked schedule can give an
		// x-slab a contiguous range of them.  Each element owns its state and
		// rotates its own history, so the order is free; the sort is stable
		// so that elements sharing an x keep the order the CPU visits them in.
		std::vector<uint32_t> order(cnt);
		for (uint32_t i = 0; i < cnt; i++) order[i] = i;
		std::stable_sort(order.begin(), order.end(), [rlc](uint32_t a, uint32_t b)
		                 { return rlc->v_RLC_pos[0][a] < rlc->v_RLC_pos[0][b]; });
		{
			std::vector<uint32_t> sortedX(cnt);
			for (uint32_t i = 0; i < cnt; i++)
				sortedX[i] = rlc->v_RLC_pos[0][order[i]];
			BuildSlabEntryIndex(sortedX, m_rlcXStart);
		}

		// Pre-linearize indices: dir*N + x*sYZ + y*Nz + z
		std::vector<uint32_t> linIdx(cnt);
		for (uint32_t i = 0; i < cnt; i++)
		{
			const uint32_t e = order[i];
			uint32_t dir = rlc->v_RLC_dir[e];
			uint32_t x   = rlc->v_RLC_pos[0][e];
			uint32_t y   = rlc->v_RLC_pos[1][e];
			uint32_t z   = rlc->v_RLC_pos[2][e];
			linIdx[i] = dir * N + x * sYZ + y * numLines[2] + z;
		}

		CreateGpuBuf(m_gpuRLC.linIdx, iSize, usage, dLoc);
		UploadGpuBuf(m_gpuRLC.linIdx, linIdx.data(), iSize);

		// State buffers (zero-init)
		std::vector<float> zeros(cnt, 0.0f);
		CreateGpuBuf(m_gpuRLC.il,   fSize, rwUsage, dLoc); UploadGpuBuf(m_gpuRLC.il,   zeros.data(), fSize);
		CreateGpuBuf(m_gpuRLC.vdn0, fSize, rwUsage, dLoc); UploadGpuBuf(m_gpuRLC.vdn0, zeros.data(), fSize);
		CreateGpuBuf(m_gpuRLC.vdn1, fSize, rwUsage, dLoc); UploadGpuBuf(m_gpuRLC.vdn1, zeros.data(), fSize);
		CreateGpuBuf(m_gpuRLC.vdn2, fSize, rwUsage, dLoc); UploadGpuBuf(m_gpuRLC.vdn2, zeros.data(), fSize);
		CreateGpuBuf(m_gpuRLC.jn0,  fSize, rwUsage, dLoc); UploadGpuBuf(m_gpuRLC.jn0,  zeros.data(), fSize);
		CreateGpuBuf(m_gpuRLC.jn1,  fSize, rwUsage, dLoc); UploadGpuBuf(m_gpuRLC.jn1,  zeros.data(), fSize);
		CreateGpuBuf(m_gpuRLC.jn2,  fSize, rwUsage, dLoc); UploadGpuBuf(m_gpuRLC.jn2,  zeros.data(), fSize);

		// Coefficient buffers
		auto uploadCoef = [&](GpuBuf& gb, FDTD_FLOAT* data_ptr)
		{
			std::vector<float> sorted(cnt);
			for (uint32_t i = 0; i < cnt; i++)
				sorted[i] = data_ptr[order[i]];
			CreateGpuBuf(gb, fSize, usage, dLoc);
			UploadGpuBuf(gb, sorted.data(), fSize);
		};
		uploadCoef(m_gpuRLC.i2v, rlc->v_RLC_i2v);
		uploadCoef(m_gpuRLC.ilv, rlc->v_RLC_ilv);
		uploadCoef(m_gpuRLC.vvd, rlc->v_RLC_vvd);
		uploadCoef(m_gpuRLC.vv2, rlc->v_RLC_vv2);
		uploadCoef(m_gpuRLC.vj1, rlc->v_RLC_vj1);
		uploadCoef(m_gpuRLC.vj2, rlc->v_RLC_vj2);
		uploadCoef(m_gpuRLC.ib0, rlc->v_RLC_ib0);
		uploadCoef(m_gpuRLC.b1,  rlc->v_RLC_b1);
		uploadCoef(m_gpuRLC.b2,  rlc->v_RLC_b2);

		// Descriptor sets
		auto allocDescSet = [&](VkDescriptorSetLayout layout) -> VkDescriptorSet
		{
			VkDescriptorSetAllocateInfo ai{};
			ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			ai.descriptorPool     = m_extDescPool;
			ai.descriptorSetCount = 1;
			ai.pSetLayouts        = &layout;
			VkDescriptorSet set;
			VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, &set));
			return set;
		};

		// preDesc: [il, vdn0, vdn1, vdn2, i2v, ilv]
		m_gpuRLC.preDesc = allocDescSet(m_rlcPreDescLayout);
		{
			VkBuffer bufs[6]      = {m_gpuRLC.il.buffer, m_gpuRLC.vdn0.buffer,
			                          m_gpuRLC.vdn1.buffer, m_gpuRLC.vdn2.buffer,
			                          m_gpuRLC.i2v.buffer, m_gpuRLC.ilv.buffer};
			VkDeviceSize sizes[6] = {fSize, fSize, fSize, fSize, fSize, fSize};
			WriteDescriptorBuffers(m_device, m_gpuRLC.preDesc, bufs, sizes, 6);
		}

		// applyDesc: [volt, il, vdn0, vdn1, vdn2, jn0, jn1, jn2, linIdx, vvd, vv2, vj1, vj2, ib0, b1, b2]
		m_gpuRLC.applyDesc = allocDescSet(m_rlcApplyDescLayout);
		{
			VkBuffer bufs[16] = {
				m_voltBuf, m_gpuRLC.il.buffer,
				m_gpuRLC.vdn0.buffer, m_gpuRLC.vdn1.buffer, m_gpuRLC.vdn2.buffer,
				m_gpuRLC.jn0.buffer, m_gpuRLC.jn1.buffer, m_gpuRLC.jn2.buffer,
				m_gpuRLC.linIdx.buffer,
				m_gpuRLC.vvd.buffer, m_gpuRLC.vv2.buffer,
				m_gpuRLC.vj1.buffer, m_gpuRLC.vj2.buffer,
				m_gpuRLC.ib0.buffer, m_gpuRLC.b1.buffer, m_gpuRLC.b2.buffer
			};
			VkDeviceSize sizes[16] = {
				m_fieldBufSize, fSize,
				fSize, fSize, fSize,
				fSize, fSize, fSize,
				iSize,
				fSize, fSize,
				fSize, fSize,
				fSize, fSize, fSize
			};
			WriteDescriptorBuffers(m_device, m_gpuRLC.applyDesc, bufs, sizes, 16);
		}

		m_hasGPU_RLC = true;
		break;  // Only one RLC extension expected
	}
}

void Engine_Vulkan::SetupGPU_SteadyState()
{
	Operator_Ext_SteadyState* opSteady = nullptr;
	for (size_t i = 0; i < Op->GetNumberOfExtentions(); ++i)
	{
		opSteady = dynamic_cast<Operator_Ext_SteadyState*>(Op->GetExtension(i));
		if (opSteady) break;
	}
	for (Engine_Extension* ext : m_Eng_exts)
	{
		m_gpuSteadyStateExt = dynamic_cast<Engine_Ext_SteadyState*>(ext);
		if (m_gpuSteadyStateExt) break;
	}
	if (!opSteady || !m_gpuSteadyStateExt || opSteady->m_TS_period == 0 ||
	    opSteady->m_E_probe_dir.empty())
		return;

	m_steadyProbeCount = (uint32_t)opSteady->m_E_probe_dir.size();
	m_steadyPeriod = opSteady->m_TS_period;
	m_steadyRingSize = 3 * m_steadyPeriod;
	const uint64_t historyValues = (uint64_t)m_steadyProbeCount * m_steadyRingSize;
	if (historyValues > UINT32_MAX)
		throw std::runtime_error("Vulkan steady-state history exceeds 32-bit shader indexing");

	std::vector<uint32_t> indices(m_steadyProbeCount);
	m_steadyProbeX.resize(m_steadyProbeCount);
	for (uint32_t i = 0; i < m_steadyProbeCount; ++i)
	{
		indices[i] = opSteady->m_E_probe_dir[i] * m_fieldN +
		             opSteady->m_E_probe_pos[0][i] * m_strideYZ +
		             opSteady->m_E_probe_pos[1][i] * numLines[2] +
		             opSteady->m_E_probe_pos[2][i];
		m_steadyProbeX[i] = opSteady->m_E_probe_pos[0][i];
	}
	std::sort(m_steadyProbeX.begin(), m_steadyProbeX.end());

	const VkMemoryPropertyFlags deviceLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	const VkMemoryPropertyFlags hostVisible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
	                                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	const VkDeviceSize indexSize = indices.size() * sizeof(uint32_t);
	const VkDeviceSize historySize = historyValues * sizeof(float);
	CreateGpuBuf(m_steadyProbeIdx, indexSize, storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, deviceLocal);
	UploadGpuBuf(m_steadyProbeIdx, indices.data(), indexSize);
	CreateGpuBuf(m_steadyHistory, historySize, storage, hostVisible);
	VK_CHECK(vkMapMemory(m_device, m_steadyHistory.memory, 0, historySize, 0,
	                     (void**)&m_steadyHistoryMapped));
	std::memset(m_steadyHistoryMapped, 0, historySize);

	VkDescriptorSetLayoutBinding bindings[3];
	for (uint32_t i = 0; i < 3; ++i)
		bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
	VkDescriptorSetLayoutCreateInfo dli{};
	dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	dli.bindingCount = 3;
	dli.pBindings = bindings;
	VK_CHECK(vkCreateDescriptorSetLayout(m_device, &dli, nullptr, &m_steadyDescLayout));

	VkPushConstantRange pcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SteadyPC)};
	VkPipelineLayoutCreateInfo pli{};
	pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pli.setLayoutCount = 1;
	pli.pSetLayouts = &m_steadyDescLayout;
	pli.pushConstantRangeCount = 1;
	pli.pPushConstantRanges = &pcRange;
	VK_CHECK(vkCreatePipelineLayout(m_device, &pli, nullptr, &m_steadyPipeLayout));

	VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
	VkDescriptorPoolCreateInfo dpi{};
	dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	dpi.maxSets = 1;
	dpi.poolSizeCount = 1;
	dpi.pPoolSizes = &poolSize;
	VK_CHECK(vkCreateDescriptorPool(m_device, &dpi, nullptr, &m_steadyDescPool));
	VkDescriptorSetAllocateInfo ai{};
	ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	ai.descriptorPool = m_steadyDescPool;
	ai.descriptorSetCount = 1;
	ai.pSetLayouts = &m_steadyDescLayout;
	VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, &m_steadyDescSet));
	VkBuffer bufs[3] = {m_voltBuf, m_steadyProbeIdx.buffer, m_steadyHistory.buffer};
	VkDeviceSize sizes[3] = {m_fieldBufSize, indexSize, historySize};
	WriteDescriptorBuffers(m_device, m_steadyDescSet, bufs, sizes, 3);

	VkShaderModule mod = CreateShaderModule(gpu_spirv::steady_state_sample_data,
	                                        gpu_spirv::steady_state_sample_size);
	VkPipelineShaderStageCreateInfo stage{};
	stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	stage.module = mod;
	stage.pName = "main";
	VkComputePipelineCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	ci.stage = stage;
	ci.layout = m_steadyPipeLayout;
	VK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache, 1, &ci, nullptr, &m_steadyPipeline));
	vkDestroyShaderModule(m_device, mod, nullptr);

	// The energy of a check, by x-plane, into one of STEADY_CHECK_SLOTS slots
	// that the host reads once the check's batch is done: no drain, and the
	// energy of the check's own timestep.
	m_steadyChunks = (m_strideYZ + STEADY_ENERGY_CHUNK - 1) / STEADY_ENERGY_CHUNK;
	m_steadySlotFloats = 2ull * numLines[0] * m_steadyChunks;
	const VkDeviceSize energySize = STEADY_CHECK_SLOTS * m_steadySlotFloats * sizeof(float);
	CreateGpuBuf(m_steadyEnergy, energySize, storage, hostVisible);
	VK_CHECK(vkMapMemory(m_device, m_steadyEnergy.memory, 0, energySize, 0,
	                     (void**)&m_steadyEnergyMapped));
	VK_CHECK(vkCreateDescriptorSetLayout(m_device, &dli, nullptr, &m_steadyEnergyDescLayout));
	VkPushConstantRange energyPC = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SteadyEnergyPC)};
	pli.pSetLayouts = &m_steadyEnergyDescLayout;
	pli.pPushConstantRanges = &energyPC;
	VK_CHECK(vkCreatePipelineLayout(m_device, &pli, nullptr, &m_steadyEnergyPipeLayout));
	VK_CHECK(vkCreateDescriptorPool(m_device, &dpi, nullptr, &m_steadyEnergyDescPool));
	ai.descriptorPool = m_steadyEnergyDescPool;
	ai.pSetLayouts = &m_steadyEnergyDescLayout;
	VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, &m_steadyEnergyDescSet));
	VkBuffer energyBufs[3] = {m_voltBuf, m_currBuf, m_steadyEnergy.buffer};
	VkDeviceSize energySizes[3] = {m_fieldBufSize, m_fieldBufSize, energySize};
	WriteDescriptorBuffers(m_device, m_steadyEnergyDescSet, energyBufs, energySizes, 3);
	mod = CreateShaderModule(gpu_spirv::reduce_energy_planes_data, gpu_spirv::reduce_energy_planes_size);
	stage.module = mod;
	ci.stage = stage;
	ci.layout = m_steadyEnergyPipeLayout;
	VK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache, 1, &ci, nullptr, &m_steadyEnergyPipeline));
	vkDestroyShaderModule(m_device, mod, nullptr);

	m_gpuSteadyStateExt->SetGpuUpdater([this]() { UpdateSteadyStateResult(); });
}

void Engine_Vulkan::SetupGPU_LocalABC()
{
	std::vector<Operator_Ext_Absorbing_BC*> sheets;
	for (size_t i = 0; i < Op->GetNumberOfExtentions(); ++i)
	{
		if (auto* sheet = dynamic_cast<Operator_Ext_Absorbing_BC*>(Op->GetExtension(i)))
			sheets.push_back(sheet);
	}
	if (sheets.empty()) return;

	VkDescriptorSetLayoutBinding bindings[4];
	for (uint32_t i = 0; i < 4; ++i)
		bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
	VkDescriptorSetLayoutCreateInfo dli{};
	dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	dli.bindingCount = 4;
	dli.pBindings = bindings;
	VK_CHECK(vkCreateDescriptorSetLayout(m_device, &dli, nullptr, &m_localAbcDescLayout));

	VkPushConstantRange pcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(LocalAbcPC)};
	VkPipelineLayoutCreateInfo pli{};
	pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pli.setLayoutCount = 1;
	pli.pSetLayouts = &m_localAbcDescLayout;
	pli.pushConstantRangeCount = 1;
	pli.pPushConstantRanges = &pcRange;
	VK_CHECK(vkCreatePipelineLayout(m_device, &pli, nullptr, &m_localAbcPipeLayout));

	VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)sheets.size() * 8};
	VkDescriptorPoolCreateInfo dpi{};
	dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	dpi.maxSets = (uint32_t)sheets.size() * 2;
	dpi.poolSizeCount = 1;
	dpi.pPoolSizes = &poolSize;
	VK_CHECK(vkCreateDescriptorPool(m_device, &dpi, nullptr, &m_localAbcDescPool));

	VkShaderModule mod = CreateShaderModule(gpu_spirv::local_absorbing_bc_data,
	                                        gpu_spirv::local_absorbing_bc_size);
	VkPipelineShaderStageCreateInfo stage{};
	stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	stage.module = mod;
	stage.pName = "main";
	VkComputePipelineCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	ci.stage = stage;
	ci.layout = m_localAbcPipeLayout;
	VK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache, 1, &ci, nullptr, &m_localAbcPipeline));
	vkDestroyShaderModule(m_device, mod, nullptr);

	const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	const VkMemoryPropertyFlags deviceLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	for (Operator_Ext_Absorbing_BC* sheet : sheets)
	{
		GpuLocalABCData data;
		data.superAbsorption = sheet->m_ABCtype == Operator_Ext_Absorbing_BC::MUR_1ST_SA;
		data.pc.Nx = numLines[0]; data.pc.Ny = numLines[1]; data.pc.Nz = numLines[2];
		data.pc.ny = sheet->m_ny; data.pc.nyP = sheet->m_nyP; data.pc.nyPP = sheet->m_nyPP;
		data.pc.startX = sheet->m_sheetX0[0];
		data.pc.startY = sheet->m_sheetX0[1];
		data.pc.startZ = sheet->m_sheetX0[2];
		data.pc.voltBoundary = sheet->m_sheetX0[sheet->m_ny];
		data.pc.voltShift = data.pc.voltBoundary + (sheet->m_normalSignPositive ? 1 : -1);
		data.pc.currBoundary = data.pc.voltBoundary + (sheet->m_normalSignPositive ? 0 : -1);
		data.pc.currShift = data.pc.voltBoundary + (sheet->m_normalSignPositive ? 1 : -2);
		data.pc.sizeP = sheet->m_numLines[0];
		data.pc.sizePP = sheet->m_numLines[1];
		data.pc.countVolt = data.pc.sizeP * data.pc.sizePP;
		data.pc.countCurr = (data.pc.sizeP - 1) * (data.pc.sizePP - 1);
		data.pc.phase = 0;

		std::vector<float> k1(2 * data.pc.countVolt);
		std::vector<float> k2(2 * data.pc.countVolt, 0.0f);
		for (uint32_t i = 0; i < data.pc.sizeP; ++i)
		{
			for (uint32_t j = 0; j < data.pc.sizePP; ++j)
			{
				const uint32_t idx = i * data.pc.sizePP + j;
				k1[idx] = sheet->m_K1_nyP(i,j);
				k1[data.pc.countVolt + idx] = sheet->m_K1_nyPP(i,j);
				if (data.superAbsorption)
				{
					k2[idx] = sheet->m_K2_nyP(i,j);
					k2[data.pc.countVolt + idx] = sheet->m_K2_nyPP(i,j);
				}
			}
		}

		const VkDeviceSize coeffSize = k1.size() * sizeof(float);
		const VkDeviceSize voltStateSize = 2 * (VkDeviceSize)data.pc.countVolt * sizeof(float);
		const VkDeviceSize currStateSize = 2 * (VkDeviceSize)data.pc.countCurr * sizeof(float);
		CreateGpuBuf(data.k1, coeffSize, usage, deviceLocal);
		CreateGpuBuf(data.k2, coeffSize, usage, deviceLocal);
		UploadGpuBuf(data.k1, k1.data(), coeffSize);
		UploadGpuBuf(data.k2, k2.data(), coeffSize);
		std::vector<float> voltZeros(2 * data.pc.countVolt, 0.0f);
		std::vector<float> currZeros(2 * data.pc.countCurr, 0.0f);
		CreateGpuBuf(data.voltState, voltStateSize, usage, deviceLocal);
		CreateGpuBuf(data.currState, currStateSize, usage, deviceLocal);
		UploadGpuBuf(data.voltState, voltZeros.data(), voltStateSize);
		UploadGpuBuf(data.currState, currZeros.data(), currStateSize);

		VkDescriptorSetLayout layouts[2] = {m_localAbcDescLayout, m_localAbcDescLayout};
		VkDescriptorSet sets[2];
		VkDescriptorSetAllocateInfo ai{};
		ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		ai.descriptorPool = m_localAbcDescPool;
		ai.descriptorSetCount = 2;
		ai.pSetLayouts = layouts;
		VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, sets));
		data.voltDesc = sets[0]; data.currDesc = sets[1];
		VkBuffer voltBufs[4] = {m_voltBuf, data.voltState.buffer, data.k1.buffer, data.k2.buffer};
		VkDeviceSize voltSizes[4] = {m_fieldBufSize, voltStateSize, coeffSize, coeffSize};
		WriteDescriptorBuffers(m_device, data.voltDesc, voltBufs, voltSizes, 4);
		VkBuffer currBufs[4] = {m_currBuf, data.currState.buffer, data.k1.buffer, data.k2.buffer};
		VkDeviceSize currSizes[4] = {m_fieldBufSize, currStateSize, coeffSize, coeffSize};
		WriteDescriptorBuffers(m_device, data.currDesc, currBufs, currSizes, 4);
		m_gpuLocalABC.push_back(std::move(data));
	}
}

void Engine_Vulkan::RecordLocalABCPhase(VkCommandBuffer cmd, uint32_t phase) const
{
	if (!m_localAbcPipeline || m_gpuLocalABC.empty()) return;
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_localAbcPipeline);
	VkMemoryBarrier between{};
	between.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	between.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	between.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	bool dispatched = false;
	// Sheets that cross write the same field cells in their apply phases
	// (2, 5).  The CPU applies equal-priority extensions in reverse order of
	// creation, so those phases go in that order, one sheet at a time.
	const bool writesField = (phase == 2 || phase == 5);
	for (size_t n = 0; n < m_gpuLocalABC.size(); ++n)
	{
		const GpuLocalABCData& data = m_gpuLocalABC[writesField ? m_gpuLocalABC.size() - 1 - n : n];
		if (phase >= 3 && !data.superAbsorption) continue;
		if (writesField && dispatched)
			CmdComputeBarrier(cmd, between);
		LocalAbcPC pc = data.pc;
		pc.phase = phase;
		const uint32_t count = phase >= 3 ? pc.countCurr : pc.countVolt;
		const VkDescriptorSet desc = phase >= 3 ? data.currDesc : data.voltDesc;
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		                        m_localAbcPipeLayout, 0, 1, &desc, 0, nullptr);
		vkCmdPushConstants(cmd, m_localAbcPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
		                   0, sizeof(LocalAbcPC), &pc);
		vkCmdDispatch(cmd, (count + 255) / 256, 1, 1);
		dispatched = true;
	}
	if (dispatched)
	{
		VkMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		CmdComputeBarrier(cmd, barrier);
	}
}

bool Engine_Vulkan::RecordSteadyStateSample(VkCommandBuffer cmd, uint32_t ts, uint32_t xLo, uint32_t xHi) const
{
	if (!m_steadyPipeline) return false;
	// A slab without a probe line records nothing, and needs no dispatch.
	const auto first = std::lower_bound(m_steadyProbeX.begin(), m_steadyProbeX.end(), xLo);
	if (first == m_steadyProbeX.end() || *first >= xHi) return false;
	const SteadyPC pc = {m_steadyProbeCount, m_steadyRingSize, ts, xLo, xHi, m_fieldN, m_strideYZ};
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_steadyPipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
	                        m_steadyPipeLayout, 0, 1, &m_steadyDescSet, 0, nullptr);
	vkCmdPushConstants(cmd, m_steadyPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
	                   0, sizeof(SteadyPC), &pc);
	vkCmdDispatch(cmd, (m_steadyProbeCount + 255) / 256, 1, 1);

	VkMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	CmdComputeBarrier(cmd, barrier);
	return true;
}

void Engine_Vulkan::RecordSteadyStateEnergy(VkCommandBuffer cmd, uint32_t ts, uint32_t xLo, uint32_t xHi) const
{
	if (!m_steadyEnergyPipeline || !m_gpuSteadyStateExt->IsCheckTimestep(ts) || xHi <= xLo)
		return;
	// The slot of the check, as ScheduleSteadyStateChecks() registered it.
	const uint32_t slot = (ts / m_steadyPeriod) % STEADY_CHECK_SLOTS;
	const SteadyEnergyPC pc = {m_fieldN, m_strideYZ, STEADY_ENERGY_CHUNK, m_steadyChunks, xLo,
	                           (uint32_t)(slot * m_steadySlotFloats)};
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_steadyEnergyPipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
	                        m_steadyEnergyPipeLayout, 0, 1, &m_steadyEnergyDescSet, 0, nullptr);
	vkCmdPushConstants(cmd, m_steadyEnergyPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
	                   0, sizeof(SteadyEnergyPC), &pc);
	vkCmdDispatch(cmd, m_steadyChunks, xHi - xLo, 1);

	// The current update after it writes the currents it reads.
	VkMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	CmdComputeBarrier(cmd, barrier);
}

void Engine_Vulkan::ScheduleSteadyStateChecks(unsigned int iterTS)
{
	if (!m_steadyEnergyPipeline || iterTS == 0)
		return;
	const uint32_t p = m_steadyPeriod;
	const uint32_t first = numTS, last = numTS + iterTS - 1;
	// A check needs its two periods of probe records, which the ring of three
	// keeps until the GPU computes the timestep one period after the check,
	// and its slot of partials, until a check STEADY_CHECK_SLOTS periods later
	// takes it. A batch holds at most one period (IterateTS()); a check this
	// batch would overrun is evaluated first, waiting for its batch if need be.
	// In a run that reads GetLastDiff() after every batch, that never waits.
	uint32_t newest = 0;
	for (uint32_t ts = (first + p - 1) / p * p; ts <= last; ts += p)
		if (m_gpuSteadyStateExt->IsCheckTimestep(ts))
			newest = ts;
	while (!m_steadyChecks.empty())
	{
		const uint32_t ts = m_steadyChecks.front().ts;
		if (last <= ts + p && (newest == 0 || newest < ts + STEADY_CHECK_SLOTS * p))
			break;
		EvaluateSteadyStateChecks(ts + 1);
	}
	for (uint32_t ts = (first + p - 1) / p * p; ts <= last; ts += p)
		if (m_gpuSteadyStateExt->IsCheckTimestep(ts))
			m_steadyChecks.push_back({ts, m_cmdIdx, m_submitCount + 1});  // the submit after this recording
}

void Engine_Vulkan::EvaluateSteadyStateChecks(unsigned int before)
{
	while (!m_steadyChecks.empty() && m_steadyChecks.front().ts < before)
	{
		const SteadyCheck c = m_steadyChecks.front();
		// Only for the batch that computed the check, not a drain. A newer
		// serial means the batch was waited for before its slot was reused.
		if (m_gpuInFlight[c.cmdSlot] && m_submitSerial[c.cmdSlot] == c.serial)
			vkWaitForFences(m_device, 1, &m_fences[c.cmdSlot], VK_TRUE, UINT64_MAX);
		m_steadyChecks.pop_front();

		// The planes in order, as the CPU engines add their rows.
		const float* partials = m_steadyEnergyMapped
		                      + ((c.ts / m_steadyPeriod) % STEADY_CHECK_SLOTS) * m_steadySlotFloats;
		double E = 0, H = 0;
		for (uint64_t i = 0; i < m_steadySlotFloats; i += 2)
		{
			E += (double)partials[i];
			H += (double)partials[i + 1];
		}
		const float* history = m_steadyHistoryMapped;
		const uint32_t p = m_steadyPeriod, ring = m_steadyRingSize, ts = c.ts;
		m_gpuSteadyStateExt->CheckPeriod(ts, EPS0 * E + MUE0 * H, [history, p, ring, ts](size_t n, unsigned int t)
		{
			return (double)history[n * ring + Engine_Ext_SteadyState::RingTimestep(ts, p, t) % ring];
		});
	}
}

void Engine_Vulkan::UpdateSteadyStateResult()
{
	// Every check the run has passed, as the CPU engines have them by now.
	EvaluateSteadyStateChecks(numTS);
}

void Engine_Vulkan::SetupProbeCache(ProcessingArray* /*PA*/)
{
	// Finalize probe recording: stop recording mode and build GPU resources
	// from the cell accesses captured during the initial PA->Process().
	m_recordingProbeAccess = false;

	// Probe resources are used even when there are no other GPU extensions.
	// In that case SetupGPUExtensions() may have skipped creating layouts.
	if (m_probeDescLayout == VK_NULL_HANDLE)
	{
		VkDescriptorSetLayoutBinding bindings[5];
		for (uint32_t i = 0; i < 5; ++i)
			bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

		VkDescriptorSetLayoutCreateInfo ci{};
		ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		ci.bindingCount = 5;
		ci.pBindings = bindings;
		VK_CHECK(vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &m_probeDescLayout));
	}

	if (m_recordedCells.empty())
	{
		cerr << "[Vulkan] No probe cell accesses recorded — pipelining disabled." << endl;
		m_hasGPU_Probes = false;
		return;
	}

	// Deduplicate recorded cells
	std::sort(m_recordedCells.begin(), m_recordedCells.end());
	m_recordedCells.erase(std::unique(m_recordedCells.begin(), m_recordedCells.end()),
	                      m_recordedCells.end());

	m_probeCacheCount = (uint32_t)m_recordedCells.size();
	cerr << "[Vulkan] Probe cache: " << m_probeCacheCount << " unique cells recorded." << endl;

	// Build CPU-side cache keys: sorted vector (already sorted by dedup step above)
	// Index into m_probeCacheCPU is simply the position in the sorted vector.
	m_probeCacheKeys.resize(m_probeCacheCount);
	m_probeCacheHits = 0;
	m_probeCacheMisses = 0;
	std::vector<uint32_t> probeIndices(m_probeCacheCount);
	std::vector<uint32_t> probeFieldSel(m_probeCacheCount);

	for (uint32_t i = 0; i < m_probeCacheCount; ++i)
	{
		uint64_t cell = m_recordedCells[i];
		m_probeCacheKeys[i] = cell;
		uint32_t fieldSel = (uint32_t)(cell >> 48);
		uint32_t linearIdx = (uint32_t)(cell & 0x0000FFFFFFFFFFFF);
		probeIndices[i]  = linearIdx;
		probeFieldSel[i] = fieldSel;
	}

	m_probeCacheCPU.resize(m_probeCacheCount, 0.0f);

	// Free the recording vector
	m_recordedCells.clear();
	m_recordedCells.shrink_to_fit();

	// Create GPU buffers for the gather shader
	VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	VkMemoryPropertyFlags dLoc = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	VkMemoryPropertyFlags hostVisible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
	                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	VkDeviceSize idxSize = m_probeCacheCount * sizeof(uint32_t);
	VkDeviceSize outSize = m_probeCacheCount * sizeof(float);

	// Upload probe index buffer (read-only on GPU)
	CreateGpuBuf(m_probeIdxBuf, idxSize, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, dLoc);
	UploadGpuBuf(m_probeIdxBuf, probeIndices.data(), idxSize);

	// Upload field selector buffer (read-only on GPU)
	CreateGpuBuf(m_probeSelBuf, idxSize, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, dLoc);
	UploadGpuBuf(m_probeSelBuf, probeFieldSel.data(), idxSize);

	// Probe results are compact. Let the GPU write them directly to coherent
	// host-visible memory instead of coupling readback to the field allocation.
	CreateGpuBuf(m_probeOutBuf, outSize, usage, hostVisible);
	VK_CHECK(vkMapMemory(m_device, m_probeOutBuf.memory, 0, outSize, 0,
	                     (void**)&m_probeOutMapped));

	// Create probe gather pipeline (re-use existing m_probeDescLayout with 5 bindings)
	// Descriptor pool for 1 set, 5 storage buffer descriptors
	VkDescriptorPoolSize poolSz = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5};
	VkDescriptorPoolCreateInfo dpi{};
	dpi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	dpi.maxSets       = 1;
	dpi.poolSizeCount = 1;
	dpi.pPoolSizes    = &poolSz;
	VK_CHECK(vkCreateDescriptorPool(m_device, &dpi, nullptr, &m_probeDescPool));

	VkDescriptorSetAllocateInfo ai{};
	ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	ai.descriptorPool     = m_probeDescPool;
	ai.descriptorSetCount = 1;
	ai.pSetLayouts        = &m_probeDescLayout;
	VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, &m_probeGatherDescSet));

	// Write descriptors: volt, curr, probeIdx, fieldSel, probeOut
	VkBuffer bufs[5] = {m_voltBuf, m_currBuf, m_probeIdxBuf.buffer,
	                     m_probeSelBuf.buffer, m_probeOutBuf.buffer};
	VkDeviceSize sizes[5] = {m_fieldBufSize, m_fieldBufSize, idxSize, idxSize, outSize};
	WriteDescriptorBuffers(m_device, m_probeGatherDescSet, bufs, sizes, 5);

	// Create the probe gather compute pipeline using m_probePipeLayout
	// The push constant is a single uint (count)
	if (m_probePipeLayout == VK_NULL_HANDLE)
	{
		VkPushConstantRange pcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
		VkPipelineLayoutCreateInfo pli{};
		pli.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pli.setLayoutCount         = 1;
		pli.pSetLayouts            = &m_probeDescLayout;
		pli.pushConstantRangeCount = 1;
		pli.pPushConstantRanges    = &pcRange;
		VK_CHECK(vkCreatePipelineLayout(m_device, &pli, nullptr, &m_probePipeLayout));
	}

	VkShaderModule mod = CreateShaderModule(gpu_spirv::gather_probes_data,
	                                        gpu_spirv::gather_probes_size);
	VkPipelineShaderStageCreateInfo stage{};
	stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	stage.module = mod;
	stage.pName  = "main";

	VkComputePipelineCreateInfo ci{};
	ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	ci.stage  = stage;
	ci.layout = m_probePipeLayout;
	VK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache, 1, &ci, nullptr, &m_probePipeline));
	vkDestroyShaderModule(m_device, mod, nullptr);

	m_hasGPU_Probes = true;
	cerr << "[Vulkan] Probe gather pipeline created (" << m_probeCacheCount
	     << " cells, host-visible direct readback)." << endl;
}

void Engine_Vulkan::SnapshotProbeCache()
{
	if (!m_hasGPU_Probes || m_probeCacheCount == 0) return;
	memcpy(m_probeCacheCPU.data(), m_probeOutMapped, m_probeCacheCount * sizeof(float));
}

void Engine_Vulkan::SubmitSpeculative(unsigned int nTS)
{
	// Submit a batch of nTS timesteps without advancing the public numTS.
	// We save/restore numTS around the call to IterateTS so the engine
	// external state doesn't change until CommitSpeculative().
	m_speculativeTS = nTS;
	unsigned int savedTS = numTS;
	IterateTS(nTS);      // internally advances numTS
	numTS = savedTS;     // revert public numTS
}

void Engine_Vulkan::CommitSpeculative()
{
	if (m_speculativeTS == 0) return;
	numTS += m_speculativeTS;
	m_speculativeTS = 0;
}

// ===========================================================================
// Record extension shader dispatches into command buffer
// ===========================================================================

void Engine_Vulkan::RecordVoltageExtensions(VkCommandBuffer cmd, uint32_t ts) const
{
	VkMemoryBarrier barrier{};
	barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	bool dispatchedPreVolt = false;

	// --- Pre-voltage: Mur, RLC, dispersive and absorbing sheets, then UPML ---
	// Engine::DoPreVoltageUpdates() runs the extensions in reverse priority
	// order, so UPML (ENG_EXT_PRIO_UPML) comes last: a Mur face or a dispersive
	// material reaching into a PML region reads the field there before UPML
	// swaps it for its flux.  The fused path and the blocked one do the same.

	// Mur pre-voltage (saves boundary values before Yee update)
	if (!m_gpuMur.empty())
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_murPreVoltPipeline);
	for (const auto& m : m_gpuMur)
	{
		if (ts < m.startTS) continue;  // Mur not yet active
		uint32_t groups = (m.totalCells + 255) / 256;
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
			                        m_murUpdatePipeLayout, 0, 1, &m.preDesc, 0, nullptr);
		vkCmdPushConstants(cmd, m_murUpdatePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MurPC), &m.pc);
		vkCmdDispatch(cmd, groups, 1, 1);
		dispatchedPreVolt = true;
	}

	// RLC pre-voltage (inductor current update + history rotation)
	if (m_hasGPU_RLC)
	{
		RlcPC pc = {m_gpuRLC.count};
		uint32_t groups = (m_gpuRLC.count + 255) / 256;
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rlcPreVoltPipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		                        m_rlcPrePipeLayout, 0, 1, &m_gpuRLC.preDesc, 0, nullptr);
		vkCmdPushConstants(cmd, m_rlcPrePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RlcPC), &pc);
		vkCmdDispatch(cmd, groups, 1, 1);
		dispatchedPreVolt = dispatchedPreVolt || (groups > 0);
	}

	// Dispersive pre-voltage (ADE update before Yee)
	if (!m_gpuDisp.empty())
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispPreVoltPipeline);
	for (const auto& d : m_gpuDisp)
	{
		if (!d.voltADEOn || d.debye) continue;
		uint32_t groups = (d.count + 255) / 256;
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		                        m_dispPrePipeLayout, 0, 1, &d.preVoltDesc, 0, nullptr);
		vkCmdPushConstants(cmd, m_dispPrePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DispPC), &d.voltPC);
		vkCmdDispatch(cmd, groups, 1, 1);
		dispatchedPreVolt = true;
	}
	dispatchedPreVolt = RecordDebyeVoltPhase(cmd, false, -1, -1, false) || dispatchedPreVolt;
	RecordLocalABCPhase(cmd, 0);

	// UPML pre-voltage, after the reads above
	if (!m_gpuUPML.empty())
	{
		if (dispatchedPreVolt || !m_gpuLocalABC.empty())
			CmdComputeBarrier(cmd, barrier);
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_upmlPreVoltPipeline);
		for (const auto& u : m_gpuUPML)
		{
			uint32_t groups = (u.totalCells + 255) / 256;
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
			                        m_upmlPrePipeLayout, 0, 1, &u.preVoltDesc, 0, nullptr);
			vkCmdPushConstants(cmd, m_upmlPrePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PmlPC), &u.pc);
			vkCmdDispatch(cmd, groups, 1, 1);
			dispatchedPreVolt = true;
		}
	}

	// Barrier: all pre-voltage writes must complete before Yee update
	if (dispatchedPreVolt)
	{
		CmdComputeBarrier(cmd, barrier);
	}
}

void Engine_Vulkan::RecordCurrentExtensions(VkCommandBuffer cmd, uint32_t ts) const
{
	(void)ts;
	VkMemoryBarrier barrier{};
	barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	bool dispatchedPreCurr = false;

	// --- Pre-current: dispersive and absorbing sheets, then UPML ---
	// In reverse priority order, as for the voltage.

	// Dispersive pre-current
	if (!m_gpuDisp.empty())
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispPreCurrPipeline);
	for (const auto& d : m_gpuDisp)
	{
		if (!d.currADEOn) continue;
		uint32_t groups = (d.count + 255) / 256;
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		                        m_dispPrePipeLayout, 0, 1, &d.preCurrDesc, 0, nullptr);
		vkCmdPushConstants(cmd, m_dispPrePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DispPC), &d.currPC);
		vkCmdDispatch(cmd, groups, 1, 1);
		dispatchedPreCurr = true;
	}
	RecordLocalABCPhase(cmd, 3);

	// UPML pre-current, after the reads above
	if (!m_gpuUPML.empty())
	{
		if (dispatchedPreCurr || !m_gpuLocalABC.empty())
			CmdComputeBarrier(cmd, barrier);
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_upmlPreCurrPipeline);
		for (const auto& u : m_gpuUPML)
		{
			uint32_t groups = (u.totalCells + 255) / 256;
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
			                        m_upmlPrePipeLayout, 0, 1, &u.preCurrDesc, 0, nullptr);
			vkCmdPushConstants(cmd, m_upmlPrePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PmlPC), &u.pc);
			vkCmdDispatch(cmd, groups, 1, 1);
			dispatchedPreCurr = true;
		}
	}

	if (dispatchedPreCurr)
	{
		CmdComputeBarrier(cmd, barrier);
	}
}

void Engine_Vulkan::DrainGPU() const
{
	if (m_gpuDrained) return;  // Already drained — skip redundant fence syscalls
	for (int i = 0; i < CMD_RING_SIZE; ++i)
	{
		if (m_gpuInFlight[i])
		{
			auto t0 = std::chrono::steady_clock::now();
			vkWaitForFences(m_device, 1, &m_fences[i], VK_TRUE, UINT64_MAX);
			auto t1 = std::chrono::steady_clock::now();
			if (m_profileEnabled)
			{
				m_profileWaitMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
				CollectProfileForSlot(i);
			}
			m_gpuInFlight[i] = false;
		}
	}
	m_gpuDrained = true;
}

void Engine_Vulkan::SyncFieldsToHost() const
{
	DrainGPU();  // ensure all pending GPU work is complete
	if (!m_hostDirty) return;
	// After SubmitSpeculative() the buffers hold a later timestep than the one
	// being processed, and the processed one is gone.  Field dumps avoid this
	// by gathering at their timestep; whatever still gets here reads ahead.
	if (m_speculativeTS > 0 && !m_warnedStaleRead)
	{
		cerr << "Engine_Vulkan: warning: whole-field read during a speculative batch; "
		     << "the values are from timestep " << (numTS + m_speculativeTS)
		     << ", not " << numTS << "." << endl;
		m_warnedStaleRead = true;
	}
	const_cast<Engine_Vulkan*>(this)->DownloadFromDeviceBuffer(
		m_voltBuf, volt_ptr->data(), m_fieldBufSize);
	const_cast<Engine_Vulkan*>(this)->DownloadFromDeviceBuffer(
		m_currBuf, curr_ptr->data(), m_fieldBufSize);
	m_hostDirty = false;
}

// ===========================================================================
// Field dumps: the footprint gathered, or Fourier-summed, on the GPU
// ===========================================================================

// OPENEMS_GPU_DUMP_GATHER=0 sends every dump through the whole-field path,
// OPENEMS_GPU_DUMP_DFT=0 makes frequency-domain dumps gather each sample and
// sum on the host as the CPU engines do.  Both are for A/B comparisons.
static bool DumpEnvOff(const char* name)
{
	const char* env = std::getenv(name);
	return env && std::string(env) == "0";
}

bool Engine_Vulkan::CheckDumpFootprint(const DumpFootprint& fp, const char* what) const
{
	for (int d = 0; d < 3; ++d)
	{
		const std::vector<uint32_t>& l = fp.lines[d];
		for (size_t i = 0; i < l.size(); ++i)
			if (l[i] >= numLines[d] || (i > 0 && l[i] <= l[i-1]))
			{
				cerr << "Engine_Vulkan::" << what << ": footprint lines out of order or range" << endl;
				return false;
			}
		if (l.empty())
			return false;
	}
	// The shaders index the footprint with 32 bits.
	return fp.Count() < (1ull << 32);
}

int Engine_Vulkan::AddDump(const DumpFootprint& fp, std::function<bool(unsigned int)> due)
{
	DumpReg d;
	d.curr = fp.curr;
	for (int k = 0; k < 3; ++k)
	{
		d.n[k] = (uint32_t)fp.lines[k].size();
		d.lines.insert(d.lines.end(), fp.lines[k].begin(), fp.lines[k].end());
	}
	d.count = (uint32_t)fp.Count();
	d.due = std::move(due);
	m_dumps.push_back(std::move(d));
	return (int)m_dumps.size() - 1;
}

int Engine_Vulkan::RegisterDumpGather(const DumpFootprint& fp, std::function<bool(unsigned int)> due)
{
	if (m_dumpReady || DumpEnvOff("OPENEMS_GPU_DUMP_GATHER") || !CheckDumpFootprint(fp, "RegisterDumpGather"))
		return -1;
	// One ring slot holds the footprints of all gathered dumps.
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(m_physDevice, &props);
	if ((m_dumpSlotFloats + fp.Count()) * sizeof(float) > props.limits.maxStorageBufferRange)
		return -1;
	const int id = AddDump(fp, std::move(due));
	m_dumps[id].slotOff = m_dumpSlotFloats;
	m_dumpSlotFloats += m_dumps[id].count;
	return id;
}

int Engine_Vulkan::RegisterDumpDFT(const DumpFootprint& fp, unsigned int numFreq,
                                   std::function<bool(unsigned int)> due,
                                   std::function<void(unsigned int, std::complex<float>*)> weights)
{
	if (m_dumpReady || numFreq == 0 || DumpEnvOff("OPENEMS_GPU_DUMP_GATHER") || DumpEnvOff("OPENEMS_GPU_DUMP_DFT") ||
	    !CheckDumpFootprint(fp, "RegisterDumpDFT"))
		return -1;
	const VkDeviceSize bytes = (VkDeviceSize)numFreq * fp.Count() * sizeof(std::complex<float>);
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(m_physDevice, &props);
	if (bytes > props.limits.maxStorageBufferRange)
		return -1;
	// The sums may take a quarter of the device-local heap.  Beyond that the
	// dump gathers every sample and sums on the host instead.
	VkPhysicalDeviceMemoryProperties mp;
	vkGetPhysicalDeviceMemoryProperties(m_physDevice, &mp);
	VkDeviceSize heap = 0;
	for (uint32_t i = 0; i < mp.memoryHeapCount; ++i)
		if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
			heap = std::max(heap, mp.memoryHeaps[i].size);
	if (m_dumpSumBytes + bytes > heap / 4)
		return -1;
	GpuBuf sums;
	if (!TryCreateBufferWithMemory(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
	                               VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
	                               sums.buffer, sums.memory))
		return -1;
	const int id = AddDump(fp, std::move(due));
	DumpReg& d = m_dumps[id];
	d.dft = true;
	d.numFreq = numFreq;
	d.weightOff = m_dumpFreqs;
	d.weights = std::move(weights);
	d.sums = sums;
	m_dumpFreqs += numFreq;
	m_dumpSumBytes += bytes;
	return id;
}

VkDescriptorSet Engine_Vulkan::AllocDumpSet(VkBuffer out, VkDeviceSize outBytes)
{
	VkDescriptorSetAllocateInfo ai{};
	ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	ai.descriptorPool     = m_dumpDescPool;
	ai.descriptorSetCount = 1;
	ai.pSetLayouts        = &m_dumpDescLayout;
	VkDescriptorSet set = VK_NULL_HANDLE;
	if (vkAllocateDescriptorSets(m_device, &ai, &set) != VK_SUCCESS)
		return VK_NULL_HANDLE;
	VkBuffer bufs[5] = {m_voltBuf, m_currBuf, m_dumpLinesBuf.buffer, out, m_dumpWeightBuf.buffer};
	VkDeviceSize sizes[5] = {m_fieldBufSize, m_fieldBufSize, VK_WHOLE_SIZE, outBytes, VK_WHOLE_SIZE};
	WriteDescriptorBuffers(m_device, set, bufs, sizes, 5);
	return set;
}

void Engine_Vulkan::EnsureDumpResources()
{
	if (m_dumpReady || m_dumps.empty())
		return;
	m_dumpReady = true;  // closes registration

	std::vector<uint32_t> lines;
	for (DumpReg& d : m_dumps)
	{
		d.lineOff = (uint32_t)lines.size();
		lines.insert(lines.end(), d.lines.begin(), d.lines.end());
	}
	// A few KB, read only at dump timesteps: host-visible, written in place
	// (the staging buffer is one field, which a tiny grid's lines may exceed).
	const VkDeviceSize linesBytes = lines.size() * sizeof(uint32_t);
	CreateGpuBuf(m_dumpLinesBuf, linesBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	void* linesMapped = nullptr;
	VK_CHECK(vkMapMemory(m_device, m_dumpLinesBuf.memory, 0, linesBytes, 0, &linesMapped));
	std::memcpy(linesMapped, lines.data(), linesBytes);
	vkUnmapMemory(m_device, m_dumpLinesBuf.memory);

	// A weight region per command slot, written as its batch is recorded, and
	// one for the synchronous path.
	const VkDeviceSize weightBytes = (VkDeviceSize)std::max<uint32_t>(1, m_dumpFreqs) * (CMD_RING_SIZE + 1) *
	                                 sizeof(std::complex<float>);
	CreateGpuBuf(m_dumpWeightBuf, weightBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	VK_CHECK(vkMapMemory(m_device, m_dumpWeightBuf.memory, 0, weightBytes, 0, (void**)&m_dumpWeightMapped));

	// volt, curr, lines, output (a ring slot or the DFT sums), weights
	VkDescriptorSetLayoutBinding bindings[5];
	for (uint32_t i = 0; i < 5; ++i)
		bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
	VkDescriptorSetLayoutCreateInfo lci{};
	lci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	lci.bindingCount = 5;
	lci.pBindings    = bindings;
	VK_CHECK(vkCreateDescriptorSetLayout(m_device, &lci, nullptr, &m_dumpDescLayout));

	VkPushConstantRange pcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DumpPC)};
	VkPipelineLayoutCreateInfo pli{};
	pli.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pli.setLayoutCount         = 1;
	pli.pSetLayouts            = &m_dumpDescLayout;
	pli.pushConstantRangeCount = 1;
	pli.pPushConstantRanges    = &pcRange;
	VK_CHECK(vkCreatePipelineLayout(m_device, &pli, nullptr, &m_dumpPipeLayout));

	auto makePipeline = [&](const unsigned char* code, size_t size, VkPipeline& pipe)
	{
		VkShaderModule mod = CreateShaderModule(code, size);
		VkComputePipelineCreateInfo ci{};
		ci.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		ci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		ci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
		ci.stage.module = mod;
		ci.stage.pName  = "main";
		ci.layout       = m_dumpPipeLayout;
		VK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache, 1, &ci, nullptr, &pipe));
		vkDestroyShaderModule(m_device, mod, nullptr);
	};
	makePipeline(gpu_spirv::dump_gather_data, gpu_spirv::dump_gather_size, m_dumpGatherPipeline);
	makePipeline(gpu_spirv::dump_dft_data, gpu_spirv::dump_dft_size, m_dumpDFTPipeline);

	uint32_t numSets = (uint32_t)MAX_DUMP_SLOTS;
	for (const DumpReg& d : m_dumps)
		numSets += d.dft ? 1 : 0;
	VkDescriptorPoolSize poolSz = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 * numSets};
	VkDescriptorPoolCreateInfo dpi{};
	dpi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	dpi.maxSets       = numSets;
	dpi.poolSizeCount = 1;
	dpi.pPoolSizes    = &poolSz;
	VK_CHECK(vkCreateDescriptorPool(m_device, &dpi, nullptr, &m_dumpDescPool));

	bool anyDFT = false;
	for (DumpReg& d : m_dumps)
	{
		if (!d.dft)
			continue;
		d.set = AllocDumpSet(d.sums.buffer, VK_WHOLE_SIZE);
		anyDFT = true;
	}
	if (anyDFT)
		RunSingleCommand([&](VkCommandBuffer cmd)
		{
			for (const DumpReg& d : m_dumps)
				if (d.dft)
					vkCmdFillBuffer(cmd, d.sums.buffer, 0, VK_WHOLE_SIZE, 0);
		});

	uint64_t gathered = 0, summed = 0;
	for (const DumpReg& d : m_dumps)
		(d.dft ? summed : gathered) += (uint64_t)d.count * (d.dft ? d.numFreq : 1);
	if (g_settings.GetVerboseLevel() > 0)
		cout << "Engine_Vulkan: " << m_dumps.size() << " field dump(s) on the GPU: "
		     << (gathered * sizeof(float) >> 10) << " KB gathered per dump timestep, "
		     << (summed * sizeof(std::complex<float>) >> 10) << " KB of DFT sums" << endl;
}

Engine_Vulkan::DumpSlot* Engine_Vulkan::TakeDumpSlot(unsigned int ts, unsigned int keepFrom)
{
	for (DumpSlot& s : m_dumpSlots)
		if (s.ts == ts)
			return &s;

	// The emptiest, then oldest, slot nobody will read again: neither one the
	// processing may still read (from keepFrom on) nor one a dump's thread
	// holds (HoldDumpGather()).  With every slot made and none free, wait for
	// those threads to let one go.  Only this thread adds holds, so none
	// appear while it waits.
	auto age = [](const DumpSlot& s) { return s.ts == UINT_MAX ? 0ull : (unsigned long long)s.ts + 1; };
	DumpSlot* pick = nullptr;
	{
		std::unique_lock<std::mutex> lock(m_dumpHoldMutex);
		for (;;)
		{
			bool anyHeld = false;
			for (DumpSlot& s : m_dumpSlots)
			{
				const bool held = m_dumpHolds[&s - m_dumpSlots.data()] > 0;
				anyHeld |= held;
				if ((s.ts == UINT_MAX || s.ts < keepFrom) && !held && (!pick || age(s) < age(*pick)))
					pick = &s;
			}
			if (pick || m_dumpSlots.size() < MAX_DUMP_SLOTS || !anyHeld)
				break;
			m_dumpHoldFreed.wait(lock);
		}
	}
	if (!pick && m_dumpSlots.size() < MAX_DUMP_SLOTS)
	{
		DumpSlot s;
		const VkDeviceSize bytes = std::max<uint64_t>(1, m_dumpSlotFloats) * sizeof(float);
		// The host reads all of it; a cached type keeps that an ordinary read.
		const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		if (!TryCreateBufferWithMemory(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
		                               s.buf.buffer, s.buf.memory) &&
		    !TryCreateBufferWithMemory(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host, s.buf.buffer, s.buf.memory))
			return nullptr;
		VK_CHECK(vkMapMemory(m_device, s.buf.memory, 0, bytes, 0, (void**)&s.mapped));
		s.set = AllocDumpSet(s.buf.buffer, bytes);
		m_dumpSlots.push_back(std::move(s));
		pick = &m_dumpSlots.back();
	}
	if (!pick)
	{
		// More dump timesteps in flight than slots: overwrite the oldest.
		for (DumpSlot& s : m_dumpSlots)
			if (!pick || age(s) < age(*pick))
				pick = &s;
		if (!m_warnedDumpSlots)
			cerr << "Engine_Vulkan: warning: field-dump ring full, timestep " << pick->ts
			     << " is overwritten before it was dumped." << endl;
		m_warnedDumpSlots = true;
	}
	pick->ts = ts;
	pick->cmdSlot = -1;
	pick->has.assign(m_dumps.size(), 0);
	return pick;
}

void Engine_Vulkan::WaitDumpSlot(DumpSlot& slot) const
{
	if (slot.cmdSlot < 0)
		return;
	// A newer serial means the batch was waited for before its slot was reused.
	if (m_gpuInFlight[slot.cmdSlot] && m_submitSerial[slot.cmdSlot] == slot.serial)
		vkWaitForFences(m_device, 1, &m_fences[slot.cmdSlot], VK_TRUE, UINT64_MAX);
	slot.cmdSlot = -1;
}

Engine_Vulkan::DumpPC Engine_Vulkan::DumpPushConstants(const DumpReg& d) const
{
	DumpPC pc;
	pc.n0 = d.n[0];
	pc.n1 = d.n[1];
	pc.n2 = d.n[2];
	pc.lineOff = d.lineOff;
	pc.curr = d.curr ? 1u : 0u;
	pc.NyNz = m_strideYZ;
	pc.Nz = numLines[2];
	pc.N = m_fieldN;
	pc.outOff = (uint32_t)d.slotOff;
	pc.numFreq = d.numFreq;
	pc.weightOff = 0;
	pc.count = d.count;
	return pc;
}

// Everything written before, the fields above all, is visible to the dump
// shaders, and the dump shaders finish before anything after them writes.
static void CmdDumpBarrier(VkCommandBuffer cmd)
{
	VkMemoryBarrier b{};
	b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
	b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                     0, 1, &b, 0, nullptr, 0, nullptr);
}

static void CmdDumpHostBarrier(VkCommandBuffer cmd)
{
	VkMemoryBarrier b{};
	b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	b.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
	                     0, 1, &b, 0, nullptr, 0, nullptr);
}

void Engine_Vulkan::RecordDumpDFT(VkCommandBuffer cmd, DumpReg& d, uint32_t region, unsigned int ts)
{
	const uint32_t weightOff = region * m_dumpFreqs + d.weightOff;
	d.weights(ts, m_dumpWeightMapped + weightOff);
	DumpPC pc = DumpPushConstants(d);
	pc.weightOff = weightOff;
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dumpPipeLayout, 0, 1, &d.set, 0, nullptr);
	vkCmdPushConstants(cmd, m_dumpPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DumpPC), &pc);
	vkCmdDispatch(cmd, (d.count + 255) / 256, 1, 1);
	++d.samples;
}

void Engine_Vulkan::RecordDumpSamples(VkCommandBuffer cmd, unsigned int ts)
{
	if (!m_dumpReady)
		return;
	bool gather = false, dft = false;
	std::vector<char> due(m_dumps.size(), 0);
	for (size_t i = 0; i < m_dumps.size(); ++i)
		if ((due[i] = m_dumps[i].due(ts) ? 1 : 0))
			(m_dumps[i].dft ? dft : gather) = true;
	if (!gather && !dft)
		return;

	CmdDumpBarrier(cmd);
	if (gather)
	{
		DumpSlot* slot = TakeDumpSlot(ts, m_dumpKeepFrom);
		if (slot)
		{
			slot->cmdSlot = m_cmdIdx;
			slot->serial = m_submitCount + 1;  // the submit right after this recording
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dumpGatherPipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dumpPipeLayout, 0, 1, &slot->set, 0, nullptr);
			for (size_t i = 0; i < m_dumps.size(); ++i)
			{
				if (!due[i] || m_dumps[i].dft)
					continue;
				const DumpPC pc = DumpPushConstants(m_dumps[i]);
				vkCmdPushConstants(cmd, m_dumpPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DumpPC), &pc);
				vkCmdDispatch(cmd, (m_dumps[i].count + 255) / 256, 1, 1);
				slot->has[i] = 1;
			}
			CmdDumpHostBarrier(cmd);
		}
	}
	if (dft)
	{
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dumpDFTPipeline);
		for (size_t i = 0; i < m_dumps.size(); ++i)
		{
			if (!due[i] || !m_dumps[i].dft)
				continue;
			RecordDumpDFT(cmd, m_dumps[i], (uint32_t)m_cmdIdx, ts);
			m_dumps[i].recorded.push_back(ts);
		}
	}
	// The next batch writes the fields read here, and its DFT adds to these sums.
	CmdDumpBarrier(cmd);
}

const float* Engine_Vulkan::AcquireDumpGather(int id, unsigned int ts)
{
	if (id < 0 || id >= (int)m_dumps.size() || m_dumps[id].dft)
		return nullptr;
	const DumpReg& d = m_dumps[id];
	for (DumpSlot& s : m_dumpSlots)
		if (s.ts == ts && s.has[id])
		{
			WaitDumpSlot(s);
			return s.mapped + d.slotOff;
		}

	// No batch ended at ts with this dump due (timestep 0): gather now, if
	// the GPU still holds ts.
	DrainGPU();
	if (m_liveTS != ts)
	{
		cerr << "Engine_Vulkan: field dump for timestep " << ts << " was not gathered, and the GPU is at "
		     << m_liveTS << " already." << endl;
		return nullptr;
	}
	EnsureDumpResources();
	DumpSlot* slot = TakeDumpSlot(ts, ts);
	if (!slot)
		return nullptr;
	WaitDumpSlot(*slot);
	const DumpPC pc = DumpPushConstants(d);
	RunSingleCommand([&](VkCommandBuffer cmd)
	{
		CmdDumpBarrier(cmd);
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dumpGatherPipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dumpPipeLayout, 0, 1, &slot->set, 0, nullptr);
		vkCmdPushConstants(cmd, m_dumpPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DumpPC), &pc);
		vkCmdDispatch(cmd, (d.count + 255) / 256, 1, 1);
		CmdDumpHostBarrier(cmd);
	});
	slot->has[id] = 1;
	return slot->mapped + d.slotOff;
}

std::shared_ptr<const float> Engine_Vulkan::HoldDumpGather(int id, unsigned int ts)
{
	const float* data = AcquireDumpGather(id, ts);
	if (!data)
		return nullptr;
	for (size_t i = 0; i < m_dumpSlots.size(); ++i)
		if (m_dumpSlots[i].ts == ts)
		{
			std::lock_guard<std::mutex> lock(m_dumpHoldMutex);
			++m_dumpHolds[i];
			return std::shared_ptr<const float>(data, [this, i](const float*)
			{
				std::lock_guard<std::mutex> lock(m_dumpHoldMutex);
				if (--m_dumpHolds[i] == 0)
					m_dumpHoldFreed.notify_all();
			});
		}
	return nullptr;
}

bool Engine_Vulkan::AddDumpDFTSample(int id, unsigned int ts)
{
	if (id < 0 || id >= (int)m_dumps.size() || !m_dumps[id].dft)
		return false;
	DumpReg& d = m_dumps[id];
	while (!d.recorded.empty() && d.recorded.front() < ts)
	{
		cerr << "Engine_Vulkan: warning: the GPU added timestep " << d.recorded.front()
		     << " to a frequency-domain dump that did not process it." << endl;
		d.recorded.pop_front();
	}
	if (!d.recorded.empty() && d.recorded.front() == ts)
	{
		d.recorded.pop_front();
		return true;
	}

	// Not added by a batch (timestep 0): add it now, if the GPU still holds
	// ts.
	DrainGPU();
	if (m_liveTS != ts)
	{
		cerr << "Engine_Vulkan: frequency-domain dump misses timestep " << ts << ": the GPU is at "
		     << m_liveTS << " already." << endl;
		return false;
	}
	EnsureDumpResources();
	RunSingleCommand([&](VkCommandBuffer cmd)
	{
		CmdDumpBarrier(cmd);
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dumpDFTPipeline);
		RecordDumpDFT(cmd, d, CMD_RING_SIZE, ts);
		CmdDumpBarrier(cmd);
	});
	return true;
}

bool Engine_Vulkan::ReadDumpDFT(int id, std::vector<std::complex<float>>& sums, unsigned int& samples)
{
	if (id < 0 || id >= (int)m_dumps.size() || !m_dumps[id].dft)
		return false;
	DumpReg& d = m_dumps[id];
	DrainGPU();
	EnsureDumpResources();
	if (!d.recorded.empty())
		cerr << "Engine_Vulkan: warning: the GPU added " << d.recorded.size()
		     << " timestep(s) to a frequency-domain dump that did not process them." << endl;
	d.recorded.clear();

	const VkDeviceSize bytes = (VkDeviceSize)d.numFreq * d.count * sizeof(std::complex<float>);
	GpuBuf stage;
	const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	if (!TryCreateBufferWithMemory(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, host | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
	                               stage.buffer, stage.memory) &&
	    !TryCreateBufferWithMemory(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, host, stage.buffer, stage.memory))
		return false;
	RunSingleCommand([&](VkCommandBuffer cmd)
	{
		VkMemoryBarrier b{};
		b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		                     0, 1, &b, 0, nullptr, 0, nullptr);
		VkBufferCopy region = {0, 0, bytes};
		vkCmdCopyBuffer(cmd, d.sums.buffer, stage.buffer, 1, &region);
		b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		b.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
		                     0, 1, &b, 0, nullptr, 0, nullptr);
	});
	void* mapped = nullptr;
	VK_CHECK(vkMapMemory(m_device, stage.memory, 0, bytes, 0, &mapped));
	sums.resize((size_t)d.numFreq * d.count);
	std::memcpy(sums.data(), mapped, bytes);
	vkUnmapMemory(m_device, stage.memory);
	DestroyGpuBuf(stage);
	samples = d.samples;
	return true;
}

void Engine_Vulkan::CleanupDumps()
{
	for (DumpSlot& s : m_dumpSlots)
	{
		if (s.mapped)
			vkUnmapMemory(m_device, s.buf.memory);
		DestroyGpuBuf(s.buf);
	}
	m_dumpSlots.clear();
	for (DumpReg& d : m_dumps)
		DestroyGpuBuf(d.sums);
	m_dumps.clear();
	if (m_dumpWeightMapped)
		vkUnmapMemory(m_device, m_dumpWeightBuf.memory);
	m_dumpWeightMapped = nullptr;
	DestroyGpuBuf(m_dumpWeightBuf);
	DestroyGpuBuf(m_dumpLinesBuf);
	if (m_dumpGatherPipeline) vkDestroyPipeline(m_device, m_dumpGatherPipeline, nullptr);
	if (m_dumpDFTPipeline)    vkDestroyPipeline(m_device, m_dumpDFTPipeline, nullptr);
	if (m_dumpPipeLayout)     vkDestroyPipelineLayout(m_device, m_dumpPipeLayout, nullptr);
	if (m_dumpDescLayout)     vkDestroyDescriptorSetLayout(m_device, m_dumpDescLayout, nullptr);
	if (m_dumpDescPool)       vkDestroyDescriptorPool(m_device, m_dumpDescPool, nullptr);
	m_dumpGatherPipeline = m_dumpDFTPipeline = VK_NULL_HANDLE;
	m_dumpPipeLayout = VK_NULL_HANDLE;
	m_dumpDescLayout = VK_NULL_HANDLE;
	m_dumpDescPool = VK_NULL_HANDLE;
	m_dumpReady = false;
	m_dumpSlotFloats = 0;
	m_dumpFreqs = 0;
	m_dumpSumBytes = 0;
}

void Engine_Vulkan::ValidateGPUFields(unsigned int timestep) const
{
	const uint32_t invalid = std::numeric_limits<uint32_t>::max();
	std::fill_n(m_validateBadMapped, 6, invalid);
	RunSingleCommand([&](VkCommandBuffer cmd)
	{
		RecordGPUFieldValidation(cmd, 0);
		VkMemoryBarrier hostBarrier{};
		hostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		hostBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
			0, 1, &hostBarrier, 0, nullptr, 0, nullptr);
	});

	for (unsigned int component = 0; component < 6; ++component)
	{
		const uint32_t index = m_validateBadMapped[component];
		if (index == invalid)
			continue;

		const uint32_t x = index / m_strideYZ;
		const uint32_t rem = index % m_strideYZ;
		const uint32_t y = rem / numLines[2];
		const uint32_t z = rem % numLines[2];
		const bool voltage = component < 3;
		const unsigned int fieldComponent = voltage ? component : component - 3;
		throw std::runtime_error(
			std::string("Engine_Vulkan: non-finite ") +
			(voltage ? "voltage" : "current") + " field at timestep " +
			std::to_string(timestep) + " component " +
			std::to_string(fieldComponent) + " [" + std::to_string(x) + "," +
			std::to_string(y) + "," + std::to_string(z) + "]");
	}
}

void Engine_Vulkan::RecordGPUFieldValidation(VkCommandBuffer cmd, uint32_t outputBase) const
{
	const auto* opVk = dynamic_cast<const Operator_Vulkan*>(Op);
	const uint32_t numComp = opVk ? opVk->GetNumCompressed() : 0u;
	ValidatePC pc = {m_fieldN, outputBase, m_validateTraceIndex, numComp,
	                 numLines[1], numLines[2], m_validateMagnitude ? 1u : 0u};
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_validatePipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
	                        m_validatePipeLayout, 0, 1, &m_validateDescSet, 0, nullptr);
	vkCmdPushConstants(cmd, m_validatePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
	                   0, sizeof(ValidatePC), &pc);
	vkCmdDispatch(cmd, (m_fieldN + 255) / 256, 1, 1);

	VkMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	vkCmdPipelineBarrier(cmd,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void Engine_Vulkan::ReportGPUFieldValidation(unsigned int timestep) const
{
	DrainGPU();
	const uint32_t invalid = std::numeric_limits<uint32_t>::max();
	static constexpr const char* stages[] = {
		"after voltage pre-extensions", "after voltage update",
		"after voltage extensions", "after current pre-extensions",
		"after current update", "after current extensions"};
	auto traceFloat = [](uint32_t bits) {
		float value;
		std::memcpy(&value, &bits, sizeof(value));
		return value;
	};

	for (unsigned int stage = 0; stage < 6; ++stage)
	{
		if (m_validateTraceIndex != invalid)
		{
			static constexpr const char* stages[] = {
				"after voltage pre-extensions", "after voltage update",
				"after voltage extensions", "after current pre-extensions",
				"after current update", "after current extensions"};
			const uint32_t* trace = m_validateBadMapped + stage * 16 + 6;
			cout << "Engine_Vulkan: trace timestep=" << timestep
			     << " stage=" << stages[stage]
			     << " V0=" << traceFloat(trace[0])
			     << " Cz=" << traceFloat(trace[1])
			     << " Cz_m=" << traceFloat(trace[2])
			     << " Cy=" << traceFloat(trace[3])
			     << " Cy_zm=" << traceFloat(trace[4])
			     << " curl0=" << traceFloat(trace[5])
			     << " op=" << trace[6]
			     << " vv0=" << traceFloat(trace[7])
			     << " vi0=" << traceFloat(trace[8])
			     << " result=" << traceFloat(trace[9]) << endl;
		}
		for (unsigned int component = 0; component < 6; ++component)
		{
			const uint32_t index = m_validateBadMapped[stage * 16 + component];
			if (index == invalid)
				continue;

			const uint32_t x = index / m_strideYZ;
			const uint32_t rem = index % m_strideYZ;
			const uint32_t y = rem / numLines[2];
			const uint32_t z = rem % numLines[2];
			const bool voltage = component < 3;
			const unsigned int fieldComponent = voltage ? component : component - 3;
			throw std::runtime_error(
				std::string("Engine_Vulkan: ") +
				(m_validateMagnitude ? "non-finite or excessive-magnitude " : "non-finite ") +
				(voltage ? "voltage" : "current") + " field " + stages[stage] +
				" at timestep " + std::to_string(timestep) + " component " +
				std::to_string(fieldComponent) + " [" + std::to_string(x) + "," +
				std::to_string(y) + "," + std::to_string(z) + "]");
		}
	}
}

// ===========================================================================
// IterateTS — the main time-stepping loop
// ===========================================================================

void Engine_Vulkan::BuildSlabEntryIndex(const std::vector<uint32_t>& sortedX,
                                        std::vector<uint32_t>& xStart) const
{
	const unsigned int NX = numLines[0];
	xStart.assign(NX + 1, (uint32_t)sortedX.size());
	for (size_t n = sortedX.size(); n-- > 0; )
		xStart[sortedX[n]] = (uint32_t)n;
	// An x with no entries inherits the next populated one, which is what makes
	// [xStart[a], xStart[b]) correct for every a <= b rather than only for
	// populated bounds.
	for (unsigned int x = NX; x-- > 0; )
		if (xStart[x] > xStart[x + 1])
			xStart[x] = xStart[x + 1];
}

// ---------------------------------------------------------------------------
// Trapezoidal temporal blocking
//
// The GPU is at its DRAM roofline on any grid whose field state exceeds the
// last-level cache (OPTIMIZATIONS.md §4.7/C4), and the only lever left is to
// stop generating the traffic.  Advancing one x-slab through k timesteps
// before moving on divides field traffic by roughly k, which turns a
// DRAM-bound sweep into a cache-bound one.
//
// The schedule is the one Engine_AVX2_Multithread already runs and that
// python/Tests/bench_temporal_blocking.c verifies element-for-element against
// the flat sweep.  Two geometry details from that work carry over unchanged
// and are the parts that are easy to get wrong:
//
//   * Domain edges are held, not sloped.  The slope exists because data
//     outside the tile is at the wrong time; at x=0 and x=NX-1 there is no
//     outside, so sloping there leaves edge cells un-advanced.
//   * The E range is not symmetric between the two directions.  A narrowing
//     core leaves gaps of width 2t+1 in E but 2t+2 in H, so a widening wedge
//     sweeps E one cell narrower per side than looks natural.  The symmetric
//     choice double-updates those cells.
// ---------------------------------------------------------------------------

void Engine_Vulkan::TileGeometry(uint64_t& planeBytes, uint64_t& tableBytes,
                                 uint64_t& totalBytes) const
{
	// What a tile has to hold is everything the sweep streams per cell, not
	// just the fields.  With an uncompressed operator a cell's coefficients
	// (vv, vi, ii, iv: 48 B) are twice its fields (volt+curr: 24 B), and a tile
	// sized from the fields alone is three times the target: 1.22x where 3.2x
	// was available on a 5 M-cell grid (PERFORMANCE.md §5.8).  A compressed
	// operator adds its per-cell index; its tables are shared by every cell,
	// so they come off the target once rather than per plane.
	const uint64_t plane     = (uint64_t)numLines[1] * numLines[2];
	const uint64_t cellBytes = 24 + ((m_opIdxBits == 0) ? 48 : m_opIdxBits / 8);
	tableBytes = (m_opIdxBits == 0) ? 0 : 4 * (uint64_t)m_coeffCompBufSize;
	planeBytes = plane * cellBytes;
	totalBytes = planeBytes * (uint64_t)numLines[0] + tableBytes;
}

void Engine_Vulkan::ConfigureTemporalBlocking()
{
	m_tbAnnounced = true;
	const TemporalBlockingPlan plan = PlanTemporalBlocking(m_tbRequest, &m_tbEdges);
	m_tbActive = plan.active;
	m_tbK      = plan.k;
	m_tbW      = plan.W;
	if (m_tbQuiet)
		return;

	if (!plan.note.empty())
		cout << "Engine_Vulkan: temporal blocking — " << plan.note << endl;
	if (plan.active)
		cout << "Engine_Vulkan: trapezoidal temporal blocking active"
		     << (m_tbRequest.mode == TemporalBlockingRequest::AUTO ? " (default)" : "")
		     << " — k=" << plan.k << " W=" << plan.W << " (" << plan.detail << ")" << endl;
	else if (!plan.why.empty())
		cout << "Engine_Vulkan: temporal blocking disabled — " << plan.why << endl;
}

bool Engine_Vulkan::SetTemporalBlocking(const TemporalBlockingRequest& req, bool announce)
{
	// Resolved when the next batch is recorded, like the first time; a batch
	// already in flight keeps the edges it was recorded with.
	m_tbRequest   = req;
	m_tbQuiet     = !announce;
	m_tbAnnounced = false;
	return true;
}

TemporalBlockingPlan Engine_Vulkan::PlanTemporalBlocking(const TemporalBlockingRequest& req) const
{
	return PlanTemporalBlocking(req, nullptr);
}

TemporalBlockingPlan Engine_Vulkan::PlanTemporalBlocking(const TemporalBlockingRequest& req,
                                                         std::vector<int>* edgesOut) const
{
	TemporalBlockingPlan plan;
	if (edgesOut)
		edgesOut->clear();
	const bool isAuto = req.mode == TemporalBlockingRequest::AUTO;
	if (req.mode == TemporalBlockingRequest::OFF || (!isAuto && req.k == 0))
		return plan;

	// Refuse rather than risk: an extension without a slab path would be
	// applied to the whole grid at a timestep only one tile has reached.  Name
	// what refused, so adding a slab path later is an obvious next step rather
	// than a mystery.
	//
	// GPU probes need nothing: the gather is recorded once at the end of each
	// batch, after the last trapezoid, when the whole grid is at one timestep.
	// The steady-state detection samples and sums each slab as it passes the
	// timestep (RecordSlabSweep()), and is evaluated after the batch.

	const int NX = (int)numLines[0];
	uint64_t planeBytes, tableBytes, totalBytes;
	TileGeometry(planeBytes, tableBytes, totalBytes);

	// The default: discrete and integrated GPUs.  A CPU or virtual device
	// (llvmpipe, a passthrough) has no cache of its own to block into.
	VkPhysicalDeviceProperties props{};
	vkGetPhysicalDeviceProperties(m_physDevice, &props);
	const bool integrated = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
	if (isAuto && !integrated && props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
		return plan;

	// Vulkan exposes no cache size, so the tile target is a tunable with a
	// default per device class.  An integrated GPU gets 80 % of one instance of
	// the CPU's L3, the cache it shares on Intel parts up to Raptor Lake: a
	// UHD 770 ran 1.37x faster on 25 % less energy per cell with a 30 MB tile
	// of its 36 MB, an HD 530 at the same speed on 5 % less energy with 7 MB of
	// its 8 MB, and a tile larger than the L3 gave much of that back
	// (PERFORMANCE.md §5.5).  Whether a GPU can use that cache at all is not
	// queryable either and depends on the SoC -- the Vega 7 cannot, and
	// Intel's GPUs left the CPU's ring with Meteor Lake -- but a tile its
	// cache cannot hold costs only the wedges: about 1 % on the Vega 7.
	// --benchmark measures the schedules where the guess matters.
	uint64_t target = (uint64_t)req.targetMB << 20;
	unsigned int llcMB = 0;
	if (target == 0 && integrated)
	{
		// Only an L3: where the CPU's last level is its L2 (Apple, most ARM
		// SoCs), that cache belongs to a cluster of cores, not to the GPU.
		const LLC_Span llc = QueryLastLevelCaches(1);
		const uint64_t l3 = (llc.level >= 3) ? llc.instanceBytes : 0;
		llcMB = (unsigned int)(l3 >> 20);
		target = l3 ? (uint64_t)(0.8 * (double)l3) : 16ull << 20;
	}
	else if (target == 0)
		target = 64ull << 20;
	const unsigned int targetMB = (unsigned int)(target >> 20);

	// A grid between the target and the card's real cache is resident
	// already, and blocking it costs a few percent (PERFORMANCE.md §5.5).
	// The cache size is not queryable, so by default the grid has to be twice
	// the target -- beyond the 96-128 MB of the cards measured -- before
	// blocking is worth its wedges.  Asked for explicitly, the target alone
	// decides, and a tile width asked for explicitly needs only the two tiles
	// checked below.
	if (req.W == 0)
	{
		const uint64_t engageAbove = isAuto ? 2 * target : target;
		if (totalBytes <= engageAbove)
		{
			std::ostringstream why;
			why << "grid already fits the " << (engageAbove >> 20) << " MB "
			    << (isAuto ? "default threshold" : "tile target") << " ("
			    << (totalBytes >> 20) << " MB of fields and coefficients)";
			plan.why = why.str();
			return plan;
		}
	}

	int W = (int)req.W;
	if (W <= 0)
	{
		// Shared tables stay resident whichever tile runs, but never let them
		// take more than half the target.
		const uint64_t budget = target - std::min(tableBytes, target / 2);
		W = (int)std::max<uint64_t>(1, budget / std::max<uint64_t>(1, planeBytes));
	}
	if (W > NX) W = NX;

	// W >= 2k is a hard constraint of the geometry: a core narrows by one line
	// per side per timestep, so a tile narrower than 2k inverts and the wedges
	// that fill behind it start to overlap — which double-updates cells.
	//
	// By default the block is as deep as the tile allows, up to 32.  Depth
	// divides the DRAM traffic, and with fewer, wider wedges it also cuts the
	// barriers per timestep: on the TL model k=W/2 beat k=16 at the same tile
	// on both cards measured (PERFORMANCE.md §5.8).
	int k = isAuto ? std::max(2, std::min(32, W / 2)) : (int)req.k;
	if (W < 2 * k)
	{
		const int kFit = W / 2;
		if (kFit < 2)
		{
			std::ostringstream why;
			why << "a " << W << "-line tile cannot carry a depth-2 trapezoid; the y-z "
			       "cross-section is too large for the " << targetMB << " MB target";
			plan.why = why.str();
			return plan;
		}
		std::ostringstream note;
		note << "k reduced from " << k << " to " << kFit << " to fit a " << W
		     << "-line tile (W >= 2k)";
		plan.note = note.str();
		k = kFit;
	}

	// A local absorbing sheet normal to x reads lines next to the one it
	// writes, at the same timestep, so all of them have to sit in one slab.
	// During a block of k steps a slab boundary only ever sweeps the lines
	// [e-k, e+k] around an interior tile edge e, so an edge at least k lines
	// clear of the sheet's footprint [lo,hi] keeps the sheet inside one core
	// tile for the whole block.  Edges are pushed right past any footprint
	// they would cut; that only widens tiles, so W >= 2k still holds.
	std::vector<std::pair<int, int>> keepWhole;
	for (const GpuLocalABCData& data : m_gpuLocalABC)
		if (data.pc.ny == 0)
		{
			int lo, hi;
			LocalABCFootprintX(data.pc, data.superAbsorption, lo, hi);
			keepWhole.emplace_back(lo, hi);
		}
	const auto clearOfSheets = [&](int e)
	{
		for (bool moved = true; moved; )
		{
			moved = false;
			for (const auto& f : keepWhole)
				if (e >= f.first + 1 - k && e <= f.second + k)
				{
					e = f.second + k + 1;
					moved = true;
				}
		}
		return e;
	};

	std::vector<int> edges;
	edges.push_back(0);
	for (int x = clearOfSheets(W); x < NX; x = clearOfSheets(x + W))
		edges.push_back(x);
	edges.push_back(NX);

	// Fold a short tail into its neighbour.  A tile narrows from its left, so a
	// tail shorter than the block depth can end up as a slab that reaches the
	// last x-line after the line inward from it has already dropped out — fine
	// for the Yee update, whose stencil wants the neighbour one timestep
	// behind, and fatal for any boundary extension that reads one cell inward
	// at the timestep it writes the edge, which is what Mur does
	// (RecordMurSlabPhase()).  It costs one slightly wider tile.
	if (edges.size() >= 3)
	{
		const size_t last = edges.size() - 1;
		if (edges[last] - edges[last - 1] < k + 3)
			edges.erase(edges.begin() + (long)last - 1);
	}

	if (edges.size() < 3)
	{
		std::ostringstream why;
		why << NX << " x-lines make only one tile of width " << W;
		plan.why = why.str();
		return plan;
	}

	plan.active = true;
	plan.k      = (unsigned int)k;
	plan.W      = (unsigned int)W;
	plan.tileMB = (double)W * (double)planeBytes / (1 << 20);
	std::ostringstream detail;
	detail << (edges.size() - 1) << " tiles, " << ((uint64_t)W * planeBytes >> 20) << " MB/tile, "
	       << (totalBytes >> 20) << " MB of fields and coefficients";
	if (llcMB && req.W == 0)
		detail << "; target " << targetMB << " MB = 80 % of the CPU's " << llcMB << " MB L3";
	plan.detail = detail.str();
	if (edgesOut)
		*edgesOut = edges;
	return plan;
}

std::vector<TemporalBlockingRequest> Engine_Vulkan::TemporalBlockingCandidates(unsigned int maxDepth) const
{
	// Tile targets from 4 to 256 MB, each as deep as the default would make
	// it (k = W/2, at most 32): the last-level caches of the GPUs measured
	// range from the 1 MB L2 of a Vega 7 to the 128 MB Infinity Cache of an
	// RX 6800, and at a given tile k = W/2 was never slower (§5.8).
	std::vector<TemporalBlockingRequest> out;
	uint64_t planeBytes, tableBytes, totalBytes;
	TileGeometry(planeBytes, tableBytes, totalBytes);
	for (unsigned int mb : {4u, 6u, 8u, 12u, 16u, 24u, 32u, 48u, 64u, 96u, 128u, 192u, 256u})
	{
		const uint64_t target = (uint64_t)mb << 20;
		const uint64_t budget = target - std::min(tableBytes, target / 2);
		const unsigned int W = (unsigned int)std::min<uint64_t>(numLines[0], budget / std::max<uint64_t>(1, planeBytes));
		const unsigned int k = std::min({32u, W / 2, maxDepth});
		if (k >= 2)
			out.push_back(TemporalBlockingRequest::Forced(k, W));
	}
	return out;
}

void Engine_Vulkan::RecordUPMLSlabPhase(VkCommandBuffer cmd, int phase, int x0, int x1) const
{
	if (m_gpuUPML.empty() || x1 <= x0)
		return;

	VkPipeline       pipe   = VK_NULL_HANDLE;
	VkPipelineLayout layout = VK_NULL_HANDLE;
	switch (phase)
	{
	case 0: pipe = m_upmlPreVoltPipeline;  layout = m_upmlPrePipeLayout;  break;
	case 1: pipe = m_upmlPostVoltPipeline; layout = m_upmlPostPipeLayout; break;
	case 2: pipe = m_upmlPreCurrPipeline;  layout = m_upmlPrePipeLayout;  break;
	default: pipe = m_upmlPostCurrPipeline; layout = m_upmlPostPipeLayout; break;
	}
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);

	for (const auto& u : m_gpuUPML)
	{
		// Clip the global slab against this region's own x-extent.  A region
		// on the far side of the grid contributes nothing to this slab.
		const int rLo = (int)u.pc.pStartX;
		const int rHi = rLo + (int)u.pc.pNx;
		const int lo  = std::max(x0, rLo);
		const int hi  = std::min(x1, rHi);
		if (hi <= lo)
			continue;

		// The shader's gid is region-local with lx slowest, so the clipped
		// range is one contiguous run of it.
		const uint32_t plane = u.pc.pNy * u.pc.pNz;
		PmlPC pc = u.pc;
		pc.gidBase = (uint32_t)(lo - rLo) * plane;
		pc.gidEnd  = (uint32_t)(hi - rLo) * plane;

		VkDescriptorSet desc = (phase == 0) ? u.preVoltDesc
		                     : (phase == 1) ? u.postVoltDesc
		                     : (phase == 2) ? u.preCurrDesc
		                                    : u.postCurrDesc;
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &desc, 0, nullptr);
		vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PmlPC), &pc);
		vkCmdDispatch(cmd, (pc.gidEnd - pc.gidBase + 255) / 256, 1, 1);
	}
}

bool Engine_Vulkan::RecordDispersiveSlabPhase(VkCommandBuffer cmd, int phase, int x0, int x1) const
{
	if (m_gpuDisp.empty() || x1 <= x0)
		return false;

	const bool isVolt  = (phase == 0 || phase == 1);
	const bool isApply = (phase == 1 || phase == 3);

	VkPipeline pipe = isApply ? (isVolt ? m_dispApplyVoltPipeline : m_dispApplyCurrPipeline)
	                          : (isVolt ? m_dispPreVoltPipeline   : m_dispPreCurrPipeline);
	VkPipelineLayout layout = isApply ? m_dispApplyPipeLayout : m_dispPrePipeLayout;

	// Two entries may hold the same cell: the orders of a conducting sheet or
	// of a multi-pole material carry the same position list, and a cell on
	// the face between two dispersive materials is in both lists when its
	// components lie in different ones.  An apply pass rewrites all three
	// components of each of its cells (field -= ADE, also where its ADE is
	// zero), so two of them in flight at once lose one update or apply the
	// two in either order.  They run one after the other, as in the flat
	// path and in its order: the Lorentz entries, then Debye.  The pre passes
	// only read the field and each writes its own state, so they need none.
	VkMemoryBarrier barrier{};
	barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

	bool any = false;
	for (const auto& d : m_gpuDisp)
	{
		if ((isVolt ? !d.voltADEOn : !d.currADEOn) || d.debye)
			continue;
		const uint32_t b = d.xStart[x0], e = d.xStart[x1];
		if (e <= b)
			continue;
		if (isApply && any)
			CmdComputeBarrier(cmd, barrier);

		// The apply pass takes hasLorADE = 0 in the flat path, so keep that.
		DispPC pc = isApply ? DispPC{d.count, numLines[0], numLines[1], numLines[2], 0}
		                    : (isVolt ? d.voltPC : d.currPC);
		pc.entryBase = b;
		pc.entryEnd  = e;

		VkDescriptorSet desc = isApply ? (isVolt ? d.applyVoltDesc : d.applyCurrDesc)
		                               : (isVolt ? d.preVoltDesc   : d.preCurrDesc);
		if (!any)
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &desc, 0, nullptr);
		vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DispPC), &pc);
		vkCmdDispatch(cmd, (e - b + 255) / 256, 1, 1);
		any = true;
	}
	if (isVolt)
		any = RecordDebyeVoltPhase(cmd, isApply, x0, x1, false, isApply && any) || any;
	return any;
}

bool Engine_Vulkan::RecordDebyeVoltPhase(VkCommandBuffer cmd, bool apply, int x0, int x1, bool barrierEach,
                                         bool barrierBefore) const
{
	VkMemoryBarrier barrier{};
	barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	bool any = false;
	for (const auto& d : m_gpuDisp)
	{
		if (!d.debye)
			continue;
		DispPC pc = d.voltPC;
		if (x0 >= 0)
		{
			pc.entryBase = d.xStart[x0];
			pc.entryEnd  = d.xStart[x1];
			if (pc.entryEnd <= pc.entryBase)
				continue;
		}
		const uint32_t n = (x0 >= 0) ? pc.entryEnd - pc.entryBase : d.count;
		if (barrierBefore)
			CmdComputeBarrier(cmd, barrier);
		if (!any)
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
			                  apply ? m_debyeApplyVoltPipeline : m_debyePreVoltPipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_debyePipeLayout, 0, 1, &d.debyeDesc, 0, nullptr);
		vkCmdPushConstants(cmd, m_debyePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DispPC), &pc);
		vkCmdDispatch(cmd, (n + 255) / 256, 1, 1);
		if (barrierEach)
			CmdComputeBarrier(cmd, barrier);
		any = true;
	}
	return any;
}

bool Engine_Vulkan::RecordMurSlabPhase(VkCommandBuffer cmd, int phase, int x0, int x1, uint32_t ts) const
{
	if (m_gpuMur.empty() || x1 <= x0)
		return false;
	if (m_murFused && phase != 0)
		return false;  // post and apply run inside the voltage update

	VkPipeline pipe = (phase == 0) ? m_murPreVoltPipeline
	                : (phase == 1) ? m_murPostVoltPipeline
	                               : m_murApplyVoltPipeline;
	VkPipelineLayout layout = (phase == 2) ? m_murApplyPipeLayout : m_murUpdatePipeLayout;

	bool any = false;
	for (const auto& m : m_gpuMur)
	{
		if (ts < m.startTS)
			continue;
		MurPC pc = m.pc;
		uint32_t threads = 0;
		if (pc.ny == 0)
		{
			// Normal to x: the face writes its boundary line and reads the line
			// next to it at the same timestep, so it belongs to the one slab
			// that holds both (Engine_Ext_Mur_ABC::SlabTouches()).  The schedule
			// guarantees that slab: the first tile never narrows below 3 lines
			// at x=0, and the short-tail fold keeps the last tile >= k+3 lines
			// so its sloping left side never passes x=NX-2.
			const int lo = (int)std::min(pc.lineNr, pc.lineNrShift);
			const int hi = (int)std::max(pc.lineNr, pc.lineNrShift);
			if (lo < x0 || hi >= x1)
				continue;
			threads = m.totalCells;
		}
		else
		{
			// Normal to y or z: the read and the write share their x, so the
			// face is clipped to the slab like any other x-range.
			const uint32_t extX = (pc.ny == 1) ? pc.numLinesPP : pc.numLinesP;
			const uint32_t lo = (uint32_t)x0;
			const uint32_t hi = std::min((uint32_t)x1, extX);
			if (hi <= lo)
				continue;
			pc.xLo = lo;
			pc.xHi = hi;
			threads = (hi - lo) * ((pc.ny == 1) ? pc.numLinesP : pc.numLinesPP);
		}
		const VkDescriptorSet desc = (phase == 0) ? m.preDesc : (phase == 1) ? m.postDesc : m.applyDesc;
		if (!any)
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &desc, 0, nullptr);
		vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MurPC), &pc);
		vkCmdDispatch(cmd, (threads + 255) / 256, 1, 1);
		any = true;
	}
	return any;
}

bool Engine_Vulkan::RecordRLCSlabPhase(VkCommandBuffer cmd, int phase, int x0, int x1) const
{
	if (!m_hasGPU_RLC || x1 <= x0)
		return false;
	const uint32_t b = m_rlcXStart[x0], e = m_rlcXStart[x1];
	if (e <= b)
		return false;

	RlcPC pc = {m_gpuRLC.count, b, e};
	if (phase == 0)
	{
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rlcPreVoltPipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		                        m_rlcPrePipeLayout, 0, 1, &m_gpuRLC.preDesc, 0, nullptr);
		vkCmdPushConstants(cmd, m_rlcPrePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RlcPC), &pc);
	}
	else
	{
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rlcApplyVoltPipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		                        m_rlcApplyPipeLayout, 0, 1, &m_gpuRLC.applyDesc, 0, nullptr);
		vkCmdPushConstants(cmd, m_rlcApplyPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RlcPC), &pc);
	}
	vkCmdDispatch(cmd, (e - b + 255) / 256, 1, 1);
	return true;
}

void Engine_Vulkan::LocalABCFootprintX(const LocalAbcPC& pc, bool superAbsorption, int& lo, int& hi)
{
	lo = (int)std::min(pc.voltBoundary, pc.voltShift);
	hi = (int)std::max(pc.voltBoundary, pc.voltShift);
	if (superAbsorption)
	{
		lo = std::min(lo, (int)std::min(pc.currBoundary, pc.currShift));
		hi = std::max(hi, (int)std::max(pc.currBoundary, pc.currShift));
	}
}

bool Engine_Vulkan::RecordLocalABCSlabPhase(VkCommandBuffer cmd, uint32_t phase, int x0, int x1) const
{
	if (!m_localAbcPipeline || m_gpuLocalABC.empty() || x1 <= x0)
		return false;

	VkMemoryBarrier between{};
	between.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	between.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	between.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	// As in RecordLocalABCPhase(): apply phases in reverse order, one at a time.
	const bool writesField = (phase == 2 || phase == 5);
	bool any = false;
	for (size_t s = 0; s < m_gpuLocalABC.size(); ++s)
	{
		const GpuLocalABCData& data = m_gpuLocalABC[writesField ? m_gpuLocalABC.size() - 1 - s : s];
		if (phase >= 3 && !data.superAbsorption)
			continue;
		LocalAbcPC pc = data.pc;
		pc.phase = phase;
		const bool curr = phase >= 3;
		const uint32_t height = curr ? pc.sizeP - 1 : pc.sizeP;
		const uint32_t width  = curr ? pc.sizePP - 1 : pc.sizePP;
		uint32_t threads = 0;
		if (pc.ny == 0)
		{
			int lo, hi;
			LocalABCFootprintX(pc, data.superAbsorption, lo, hi);
			if (lo < x0 || hi >= x1)
				continue;
			threads = height * width;
		}
		else
		{
			// x is the j axis of a sheet normal to y, the i axis of one normal to z.
			const uint32_t n  = (pc.ny == 1) ? width : height;
			const int      lo = std::max(x0, (int)pc.startX);
			const int      hi = std::min(x1, (int)(pc.startX + n));
			if (hi <= lo)
				continue;
			pc.xLo = (uint32_t)x0;
			pc.xHi = (uint32_t)x1;
			threads = (uint32_t)(hi - lo) * ((pc.ny == 1) ? height : width);
		}
		if (threads == 0)
			continue;
		if (writesField && any)
			CmdComputeBarrier(cmd, between);
		if (!any)
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_localAbcPipeline);
		const VkDescriptorSet desc = curr ? data.currDesc : data.voltDesc;
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		                        m_localAbcPipeLayout, 0, 1, &desc, 0, nullptr);
		vkCmdPushConstants(cmd, m_localAbcPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
		                   0, sizeof(LocalAbcPC), &pc);
		vkCmdDispatch(cmd, (threads + 255) / 256, 1, 1);
		any = true;
	}
	return any;
}

bool Engine_Vulkan::RecordTFSFSlabPhase(VkCommandBuffer cmd, bool volt, int x0, int x1, uint32_t ts) const
{
	if (!m_hasGPU_TFSF || x1 <= x0)
		return false;
	const uint32_t count = volt ? m_gpuTFSF.voltCount : m_gpuTFSF.currCount;
	if (count == 0)
		return false;
	const std::vector<uint32_t>& xStart = volt ? m_tfsfVoltXStart : m_tfsfCurrXStart;
	const uint32_t b = xStart[x0], e = xStart[x1];
	if (e <= b)
		return false;
	Excitation* exc = const_cast<Operator*>(Op)->GetExcitationSignal();
	if (!exc)
		return false;

	const uint32_t sigLen = (uint32_t)exc->GetLength();
	const int sigPeriod = (int)exc->GetSignalPeriodTS();
	// The flat path injects the voltage at ts+1 and the current at ts.
	TfsfPC pc = {count, (int32_t)(volt ? ts + 1 : ts), sigLen, sigPeriod, b, e};
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, volt ? m_tfsfVoltPipeline : m_tfsfCurrPipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tfsfPipeLayout, 0, 1,
	                        volt ? &m_gpuTFSF.voltDesc : &m_gpuTFSF.currDesc, 0, nullptr);
	vkCmdPushConstants(cmd, m_tfsfPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TfsfPC), &pc);
	vkCmdDispatch(cmd, (e - b + 255) / 256, 1, 1);
	return true;
}

void Engine_Vulkan::RecordSlabSweep(VkCommandBuffer cmd, int A0, int B0, int k, int dir,
                                    uint32_t t0, const GridPC& gridV, const GridPC& gridC,
                                    const FusedPC& fused, const VkMemoryBarrier& barrier,
                                    bool hasExcVolt, bool hasExcCurr) const
{
	const int NX = (int)numLines[0];
	// UpdateCurrents reads volt at x+1, so the engine never updates H on the
	// last x-line. That line is therefore always "valid" and needs no slope.
	const int NXH = NX - 1;

	const uint32_t planeV = numLines[1] * numLines[2];
	const uint32_t planeC = (numLines[1] - 1) * (numLines[2] - 1);

	int A = A0, B = B0;
	if (A < 0)  A = 0;
	if (B > NX) B = NX;

	// The interval [A,B) one step on, and the E range of that step.
	const auto step = [NX, dir](int A_, int B_, int& An, int& Bn, int& eLo, int& eHi)
	{
		An = (A_ <= 0)  ? 0  : A_ + dir;
		Bn = (B_ >= NX) ? NX : B_ - dir;
		if (An < 0)  An = 0;
		if (Bn > NX) Bn = NX;
		if (dir > 0) { eLo = An;                     eHi = (Bn >= NX) ? NX : Bn + 1; }
		else         { eLo = (An <= 0) ? 0 : An + 1; eHi = (Bn >= NX) ? NX : Bn; }
		if (eHi > NX)  eHi = NX;
		if (eHi < eLo) eHi = eLo;
	};

	bool preMerged = false;  // this step's Mur/RLC pre-voltage already ran with the last current update
	for (int t = 0; t < k; ++t)
	{
		int An, Bn, eLo, eHi;
		step(A, B, An, Bn, eLo, eHi);

		const uint32_t ts = t0 + (uint32_t)t;

		// --- Pre-voltage: dispersive ADE, Mur, lumped RLC, then UPML (unfused) ---
		// Ordered exactly as the flat path has it: the pre passes run before
		// the update, the apply passes after it and before the excitation.
		// The first four write disjoint state, so they share one barrier, as
		// they do in the flat path.  UPML goes last and alone, as on the CPU
		// (reverse priority order): it swaps field and flux in its cells, and a
		// dispersive material or a Mur face reaching into a PML reads the field.
		const bool unfusedUPML = !m_hasFusedUPML && !m_gpuUPML.empty();
		bool preVolt = RecordDispersiveSlabPhase(cmd, 0, eLo, eHi);
		if (!preMerged)
		{
			preVolt = RecordMurSlabPhase(cmd, 0, eLo, eHi, ts) || preVolt;
			preVolt = RecordRLCSlabPhase(cmd, 0, eLo, eHi) || preVolt;
			preVolt = RecordLocalABCSlabPhase(cmd, 0, eLo, eHi) || preVolt;
		}
		if (unfusedUPML && eHi > eLo)
		{
			if (preVolt)
				CmdComputeBarrier(cmd, barrier);
			RecordUPMLSlabPhase(cmd, 0, eLo, eHi);
			preVolt = true;
		}
		if (preVolt)
			CmdComputeBarrier(cmd, barrier);

		// --- Voltage update on [eLo, eHi) ---
		// With UPML fused into the main kernel this is one dispatch: PML
		// membership is derived per cell from the region metadata, so the
		// fused shader does not care which subset of cells it is given.
		if (m_hasFusedUPML)
		{
			if (eHi > eLo)
			{
				FusedPC pf = fused;
				pf.gidBase = (uint32_t)eLo * planeV;
				pf.gidEnd  = (uint32_t)eHi * planeV;
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_fusedVoltPipeline);
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				                        m_fusedPipeLayout, 0, 1, &m_fusedVoltDescSet, 0, nullptr);
				vkCmdPushConstants(cmd, m_fusedPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
				                   0, sizeof(FusedPC), &pf);
				vkCmdDispatch(cmd, (pf.gidEnd - pf.gidBase + 255) / 256, 1, 1);
				CmdComputeBarrier(cmd, barrier);
			}
		}
		else
		{
			if (eHi > eLo)
			{
				// With m_murFused this also finishes the Mur faces: a face
				// normal to x writes x=0 from x=1 (and NX-1 from NX-2), both of
				// which this slab holds whenever it holds either (see
				// RecordMurSlabPhase()); faces normal to y or z stay on one x.
				GridPC pv = gridV;
				pv.gidBase = (uint32_t)eLo * planeV;
				pv.gidEnd  = std::min((uint32_t)eHi * planeV, gridV.gidEnd);
				const VkPipelineLayout vLayout = m_murFused ? m_murFusedPipeLayout : m_fdtdPipeLayout;
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				                  m_murFused ? m_murFusedPipeline : m_updateVoltPipeline);
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vLayout, 0, 1,
				                        m_murFused ? &m_murFusedDescSet : &m_updateVoltDescSet, 0, nullptr);
				vkCmdPushConstants(cmd, vLayout, VK_SHADER_STAGE_COMPUTE_BIT,
				                   0, sizeof(GridPC), &pv);
				vkCmdDispatch(cmd, (pv.gidEnd - pv.gidBase + 255) / 256, 1, 1);
				CmdComputeBarrier(cmd, barrier);
			}
			if (!m_gpuUPML.empty() && eHi > eLo)
			{
				RecordUPMLSlabPhase(cmd, 1, eLo, eHi);
				CmdComputeBarrier(cmd, barrier);
			}
		}

		// --- Post-voltage: Mur, TF/SF, local absorbing sheets ---
		// The flat path runs the sheets after the TF/SF injection, so with a
		// TF/SF box they get their own group.
		bool postVolt = RecordMurSlabPhase(cmd, 1, eLo, eHi, ts);
		postVolt = RecordTFSFSlabPhase(cmd, true, eLo, eHi, ts) || postVolt;
		if (!m_hasGPU_TFSF)
			postVolt = RecordLocalABCSlabPhase(cmd, 1, eLo, eHi) || postVolt;
		if (postVolt)
			CmdComputeBarrier(cmd, barrier);
		if (m_hasGPU_TFSF && RecordLocalABCSlabPhase(cmd, 1, eLo, eHi))
			CmdComputeBarrier(cmd, barrier);

		// --- Dispersive apply-voltage ---
		if (RecordDispersiveSlabPhase(cmd, 1, eLo, eHi))
		{
				CmdComputeBarrier(cmd, barrier);
		}

		// --- Mur, then lumped RLC apply ---
		// A lumped port puts its resistor and its excitation on the same
		// edges, so the excitation below must see the RLC result.
		if (RecordMurSlabPhase(cmd, 2, eLo, eHi, ts))
			CmdComputeBarrier(cmd, barrier);
		if (RecordRLCSlabPhase(cmd, 1, eLo, eHi))
			CmdComputeBarrier(cmd, barrier);
		if (RecordLocalABCSlabPhase(cmd, 2, eLo, eHi))
			CmdComputeBarrier(cmd, barrier);

		// --- Voltage excitation, restricted to the same x-range ---
		if (hasExcVolt && eHi > eLo)
		{
			const uint32_t b = m_excVoltXStart[eLo], e = m_excVoltXStart[eHi];
			if (e > b)
			{
				ExcPC epc = {m_excVoltCount, (int32_t)ts, m_excSignalLen, m_excSignalPeriod, b, e};
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_excPipeline);
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				                        m_excPipeLayout, 0, 1, &m_excVoltDescSet, 0, nullptr);
				vkCmdPushConstants(cmd, m_excPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
				                   0, sizeof(ExcPC), &epc);
				vkCmdDispatch(cmd, (e - b + 255) / 256, 1, 1);
				CmdComputeBarrier(cmd, barrier);
			}
		}

		// --- Steady-state detection, where the flat path has it ---
		// The probes on these x-lines, and at a check the energy of these
		// planes: their voltages are final for ts, and their currents are
		// those of ts-1 -- a narrowing core advanced its x-lines to ts-1
		// itself, a widening wedge's extra lines the cores did. The E ranges
		// of one timestep partition the grid, so every plane is summed once.
		if (eHi > eLo)
		{
			RecordSteadyStateSample(cmd, ts, (uint32_t)eLo, (uint32_t)eHi);
			RecordSteadyStateEnergy(cmd, ts, (uint32_t)eLo, (uint32_t)eHi);
		}

		// --- Current update on [An, min(Bn, NX-1)) ---
		// The Yee current update never touches the last x-line (it reads volt
		// at x+1), but the flat path's UPML and excitation dispatches both run
		// over the whole domain. The last tile therefore extends those to NX
		// while the Yee sweep still stops at NX-1, so the union over tiles
		// equals the flat dispatch exactly rather than nearly.
		//
		// For UPML this is belt-and-braces: H on that line is never written, so
		// its flux recursion runs on zeros forever and produces zeros. Verified
		// by negative control -- clamping the fused current dispatch to NX-1
		// leaves the PML tests bit-identical. For the excitation it is not
		// belt-and-braces: an entry there adds a nonzero value that the flat
		// sweep applies, so clamping would drop it.
		const int hHi  = (Bn < NXH) ? Bn : NXH;
		const int hExc = (Bn >= NX) ? NX : hHi;

		bool preCurr = RecordDispersiveSlabPhase(cmd, 2, An, hExc);
		preCurr = RecordLocalABCSlabPhase(cmd, 3, An, hExc) || preCurr;
		if (unfusedUPML && hExc > An)
		{
			if (preCurr)
				CmdComputeBarrier(cmd, barrier);
			RecordUPMLSlabPhase(cmd, 2, An, hExc);
			preCurr = true;
		}
		if (preCurr)
			CmdComputeBarrier(cmd, barrier);

		// --- Next step's Mur and RLC pre-voltage, alongside this current update ---
		// Both only read voltages, which are final for step t+1 once this
		// step's voltage half is done, and write state the current half never
		// touches.  So they share the current update's barrier instead of
		// paying for one of their own at the top of the next step -- on the
		// RX 7900 XTX a barrier costs ~1.8 us, and the blocked schedule records
		// one per slab per phase.  Unfused UPML's pre pass comes after them in
		// the next step either way.
		bool nextPre = false;
		preMerged = false;
		if (t + 1 < k)
		{
			int An2, Bn2, eLo2, eHi2;
			step(An, Bn, An2, Bn2, eLo2, eHi2);
			nextPre = RecordMurSlabPhase(cmd, 0, eLo2, eHi2, ts + 1);
			nextPre = RecordRLCSlabPhase(cmd, 0, eLo2, eHi2) || nextPre;
			nextPre = RecordLocalABCSlabPhase(cmd, 0, eLo2, eHi2) || nextPre;
			preMerged = true;
		}

		if (m_hasFusedUPML)
		{
			if (nextPre && hExc <= An)
				CmdComputeBarrier(cmd, barrier);
			if (hExc > An)
			{
				FusedPC pf = fused;
				pf.gidBase = (uint32_t)An * planeV;
				pf.gidEnd  = (uint32_t)hExc * planeV;
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_fusedCurrPipeline);
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				                        m_fusedPipeLayout, 0, 1, &m_fusedCurrDescSet, 0, nullptr);
				vkCmdPushConstants(cmd, m_fusedPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
				                   0, sizeof(FusedPC), &pf);
				vkCmdDispatch(cmd, (pf.gidEnd - pf.gidBase + 255) / 256, 1, 1);
				CmdComputeBarrier(cmd, barrier);
			}
		}
		else
		{
			if (nextPre && hHi <= An)
				CmdComputeBarrier(cmd, barrier);
			if (hHi > An)
			{
				GridPC pc = gridC;
				pc.gidBase = (uint32_t)An * planeC;
				pc.gidEnd  = std::min((uint32_t)hHi * planeC, gridC.gidEnd);
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_updateCurrPipeline);
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				                        m_fdtdPipeLayout, 0, 1, &m_updateCurrDescSet, 0, nullptr);
				vkCmdPushConstants(cmd, m_fdtdPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
				                   0, sizeof(GridPC), &pc);
				vkCmdDispatch(cmd, (pc.gidEnd - pc.gidBase + 255) / 256, 1, 1);
				CmdComputeBarrier(cmd, barrier);
			}
			if (!m_gpuUPML.empty() && hExc > An)
			{
				RecordUPMLSlabPhase(cmd, 3, An, hExc);
				CmdComputeBarrier(cmd, barrier);
			}
		}

		// --- TF/SF current injection ---
		// Over the same range as the excitation below, for the same reason:
		// the flat path applies every entry, including any on the last x-line.
		if (RecordTFSFSlabPhase(cmd, false, An, hExc, ts))
			CmdComputeBarrier(cmd, barrier);
		if (RecordLocalABCSlabPhase(cmd, 4, An, hExc))
			CmdComputeBarrier(cmd, barrier);

		// --- Dispersive apply-current ---
		if (RecordDispersiveSlabPhase(cmd, 3, An, hExc))
		{
				CmdComputeBarrier(cmd, barrier);
		}
		if (RecordLocalABCSlabPhase(cmd, 5, An, hExc))
			CmdComputeBarrier(cmd, barrier);

		// --- Current excitation ---
		// The H sweep stops at NX-1, but the flat path's Apply2Current runs over
		// the whole entry list, so an entry on the last x-line is applied there.
		// Extending the last tile's excitation range to NX is what keeps this
		// path equal to the flat one.  (Engine_AVX2_Multithread's blocked path
		// uses the H range here and therefore drops such an entry.)
		if (hasExcCurr && hExc > An)
		{
			const uint32_t b = m_excCurrXStart[An], e = m_excCurrXStart[hExc];
			if (e > b)
			{
				ExcPC epc = {m_excCurrCount, (int32_t)ts, m_excSignalLen, m_excSignalPeriod, b, e};
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_excPipeline);
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				                        m_excPipeLayout, 0, 1, &m_excCurrDescSet, 0, nullptr);
				vkCmdPushConstants(cmd, m_excPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
				                   0, sizeof(ExcPC), &epc);
				vkCmdDispatch(cmd, (e - b + 255) / 256, 1, 1);
				CmdComputeBarrier(cmd, barrier);
			}
		}

		A = An;
		B = Bn;
	}
}
bool Engine_Vulkan::IterateTS(unsigned int iterTS)
{
	if (iterTS == 0) return true;

	// The caller may still process the timestep this call starts from (the
	// pipelined loop does, after submitting), so the dump ring keeps it.
	struct Depth { int& d; explicit Depth(int& v) : d(v) { ++d; } ~Depth() { --d; } } depth(m_iterateDepth);
	if (m_iterateDepth == 1)
		m_dumpKeepFrom = numTS;

	// On some drivers (notably RADV/AMDGPU), very long command buffers can
	// trigger GPU watchdog resets. Split batches into smaller submits. With
	// steady-state detection, a batch holds at most one period, so that it
	// cannot overwrite the probe records of a check it computes
	// (ScheduleSteadyStateChecks()); runs call for less than half a period.
	unsigned int maxBatch = m_maxTSPerSubmit;
	if (m_steadyEnergyPipeline && m_steadyPeriod < maxBatch)
		maxBatch = m_steadyPeriod;
	if (iterTS > maxBatch)
	{
		if (g_settings.GetVerboseLevel() > 1)
			cout << "Engine_Vulkan: splitting " << iterTS
			     << " timesteps into chunks of " << maxBatch
			     << " (progress every ~" << m_chunkProgressIntervalSec << " s)"
			     << endl;

		unsigned int submitted = 0;
		unsigned int remaining = iterTS;
		while (remaining > 0)
		{
			unsigned int chunk = std::min(remaining, maxBatch);
			if (!IterateTS(chunk))
				return false;
			remaining -= chunk;
			submitted += chunk;

			if (g_settings.GetVerboseLevel() > 0)
			{
				auto now = std::chrono::steady_clock::now();
				bool printNow = !m_chunkProgressHasLastPrint;
				if (!printNow)
				{
					double dt = std::chrono::duration<double>(now - m_chunkProgressLastPrint).count();
					printNow = (dt >= m_chunkProgressIntervalSec);
				}
				if (printNow)
				{
					unsigned int pct = (iterTS > 0)
					                 ? (unsigned int)((100ULL * submitted + (iterTS / 2)) / iterTS)
					                 : 100U;
					cout << "Engine_Vulkan: chunk progress " << submitted << "/" << iterTS
					     << " TS (" << pct << "%)";
					if (m_speculativeTS > 0)
						cout << " [speculative]";
					cout << endl;
					m_chunkProgressLastPrint = now;
					m_chunkProgressHasLastPrint = true;
				}
			}
		}
		return true;
	}

	uint32_t N  = numLines[0] * numLines[1] * numLines[2];
	uint32_t cN = (numLines[0]-1) * (numLines[1]-1) * (numLines[2]-1);
	const Operator_Vulkan* opVk = dynamic_cast<const Operator_Vulkan*>(Op);
	// The voltage sweep covers the full grid, the current sweep the reduced
	// (Nx-1)(Ny-1)(Nz-1) domain, so each needs its own end bound.  Both are
	// the whole domain unless the temporal-blocking proxy narrows them.
	GridPC gridPC  = {numLines[0], numLines[1], numLines[2], opVk->GetNumCompressed(), 0u, N};
	GridPC gridPCc = {numLines[0], numLines[1], numLines[2], opVk->GetNumCompressed(), 0u, cN};
	FusedPC fusedPC = {numLines[0], numLines[1], numLines[2], opVk->GetNumCompressed(),
	                   m_hasFusedUPML ? (uint32_t)m_gpuUPML.size() : 0u};
	uint32_t voltGroups = (N  + 255) / 256;
	uint32_t currGroups = (cN + 255) / 256;
	uint32_t excVoltGroups = (m_excVoltCount + 255) / 256;
	uint32_t excCurrGroups = (m_excCurrCount + 255) / 256;
	if (m_validateStages)
		std::fill_n(m_validateBadMapped, 6 * 16, std::numeric_limits<uint32_t>::max());

	VkMemoryBarrier barrier{};
	barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

	// Fire-and-forget, double-buffered: record the batch into
	// m_cmdBufs[m_cmdIdx], submit it, flip to the other slot.  Probes and
	// field dumps are gathered at the end of the batch; see GetVolt() for the
	// rest.
	EnsureDumpResources();

	bool hasExcVolt = m_hasGPUExcitation && m_excVoltCount > 0;
	bool hasExcCurr = m_hasGPUExcitation && m_excCurrCount > 0;
	bool hasDispVoltADE = false;
	bool hasDispCurrADE = false;
	for (const auto& d : m_gpuDisp)
	{
		hasDispVoltADE = hasDispVoltADE || d.voltADEOn;
		hasDispCurrADE = hasDispCurrADE || d.currADEOn;
		if (hasDispVoltADE && hasDispCurrADE) break;
	}
	bool needPostVoltBarrierFused = (m_hasGPU_Mur || m_hasGPU_TFSF || hasDispVoltADE || m_hasGPU_RLC);
	bool needPostCurrBarrierFused = (m_hasGPU_TFSF || hasDispCurrADE);
	bool needPostVoltBarrier = (m_hasGPU_UPML || m_hasGPU_Mur || m_hasGPU_TFSF || hasDispVoltADE || m_hasGPU_RLC);
	bool needPostCurrBarrier = (m_hasGPU_UPML || m_hasGPU_TFSF || hasDispCurrADE);
	bool hasVoltLateWritesFused = hasExcVolt || m_hasGPU_Mur || m_hasGPU_TFSF || hasDispVoltADE || m_hasGPU_RLC;
	bool hasCurrLateWritesFused = hasExcCurr || m_hasGPU_TFSF || hasDispCurrADE;
	// With m_murFused the Mur post and apply happen inside the voltage
	// update (SetupMurFusion()); only the separate phases need the rest.
	const bool murSeparate = m_hasGPU_Mur && !m_murFused;
	bool hasVoltLateWrites = hasExcVolt || m_hasGPU_UPML || murSeparate || m_hasGPU_TFSF || hasDispVoltADE || m_hasGPU_RLC;
	bool hasCurrLateWrites = hasExcCurr || m_hasGPU_UPML || m_hasGPU_TFSF || hasDispCurrADE;

	// Cache TF/SF signal info (invariant across iterations)
	Excitation* tfsfExc = nullptr;
	int         tfsfSigPeriod = 0;
	uint32_t    tfsfSigLen = 0;
	if (m_hasGPU_TFSF)
	{
		tfsfExc = const_cast<Operator*>(Op)->GetExcitationSignal();
		if (tfsfExc)
		{
			tfsfSigLen = (uint32_t)tfsfExc->GetLength();
			tfsfSigPeriod = (int)tfsfExc->GetSignalPeriodTS();
		}
	}

	// The steady-state checks this batch computes, and room for them.
	ScheduleSteadyStateChecks(iterTS);

	// Wait for this slot if it is still in-flight from a previous round
	if (m_gpuInFlight[m_cmdIdx])
	{
		auto t0 = std::chrono::steady_clock::now();
		vkWaitForFences(m_device, 1, &m_fences[m_cmdIdx], VK_TRUE, UINT64_MAX);
		auto t1 = std::chrono::steady_clock::now();
		if (m_profileEnabled)
		{
			m_profileWaitMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
			CollectProfileForSlot(m_cmdIdx);
		}
		m_gpuInFlight[m_cmdIdx] = false;
	}

	// Begin recording
	VkCommandBuffer cmd = m_cmdBufs[m_cmdIdx];
	VkCommandBufferBeginInfo bi{};
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkResetCommandBuffer(cmd, 0);
	VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
	uint32_t qBaseProfile = 0;

	if (m_profileEnabled && m_tsQueryPool)
	{
		qBaseProfile = (uint32_t)(m_cmdIdx * m_profileQueriesPerSlot);
		vkCmdResetQueryPool(cmd, m_tsQueryPool, qBaseProfile, m_profileQueriesPerSlot);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_tsQueryPool, qBaseProfile);
	}

	// OPENEMS_GPU_PROFILE_STAGES: a timestamp after each stage of every
	// flat-schedule timestep (s_profileStageNames).  A bottom-of-pipe
	// timestamp is written once all earlier work has finished, so two
	// consecutive marks bracket one stage, its barrier included.
	uint32_t qStage = (uint32_t)(m_cmdIdx * m_stageQueriesPerSlot);
	const auto stageMark = [&]()
	{
		if (m_stageQueryPool)
			vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_stageQueryPool, qStage++);
	};
	if (m_stageQueryPool)
	{
		vkCmdResetQueryPool(cmd, m_stageQueryPool, qStage, m_stageQueriesPerSlot);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_stageQueryPool, qStage++);
	}

	// Append all FDTD timesteps
	// ================= Trapezoidal temporal blocking ==================
	// See ConfigureTemporalBlocking() for the schedule and its guards.  A
	// block of kk timesteps is covered by one narrowing trapezoid per tile
	// followed by one widening wedge per interior tile boundary; together
	// they partition the domain, so every cell is updated exactly once per
	// timestep and the result is bit-identical to the flat sweep.
	if (!m_tbAnnounced)
		const_cast<Engine_Vulkan*>(this)->ConfigureTemporalBlocking();
	// A batch of one timestep runs flat: a block of depth one is the flat
	// sweep with the wedges' extra dispatches and barriers. Dirac and step
	// excitations, whose Nyquist interval is one timestep, ask for no more.
	if (m_tbActive && iterTS >= 2)
	{
		// k is clamped to the call's timestep count, so a short IterateTS
		// (openems.cpp passes the number of steps until the next probe or
		// dump) stays correct with no work on the caller's side: the grid
		// is globally coherent at the end of every call.
		const unsigned int k = std::min<unsigned int>(m_tbK, iterTS);
		for (unsigned int base = 0; base < iterTS; base += k)
		{
			const int kk = (int)std::min<unsigned int>(k, iterTS - base);
			const uint32_t t0 = numTS + base;
			for (size_t i = 0; i + 1 < m_tbEdges.size(); ++i)
				RecordSlabSweep(cmd, m_tbEdges[i], m_tbEdges[i + 1], kk, +1, t0,
				                gridPC, gridPCc, fusedPC, barrier, hasExcVolt, hasExcCurr);
			for (size_t i = 1; i + 1 < m_tbEdges.size(); ++i)
				RecordSlabSweep(cmd, m_tbEdges[i], m_tbEdges[i], kk, -1, t0,
				                gridPC, gridPCc, fusedPC, barrier, hasExcVolt, hasExcCurr);
		}
	}
	else
	for (unsigned int iter = 0; iter < iterTS; ++iter)
	{
		uint32_t ts = numTS + iter;

		if (m_hasFusedUPML)
		{
			// ============= FUSED Yee+UPML path =============
			// Single dispatch replaces: UPML pre-volt + Yee volt + UPML post-volt
			// Mur/RLC/Dispersive pre-voltage still dispatch separately before fused
			if (m_hasGPU_Mur || m_hasGPU_RLC || m_hasGPU_Dispersive)
			{
				// Record non-UPML pre-voltage extensions
				// Mur pre-voltage
				if (!m_gpuMur.empty())
					vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_murPreVoltPipeline);
				for (const auto& m : m_gpuMur)
				{
					if (ts < m.startTS) continue;
					uint32_t groups = (m.totalCells + 255) / 256;
					vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
					                        m_murUpdatePipeLayout, 0, 1, &m.preDesc, 0, nullptr);
				vkCmdPushConstants(cmd, m_murUpdatePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MurPC), &m.pc);
					vkCmdDispatch(cmd, groups, 1, 1);
				}
				// RLC pre-voltage
				if (m_hasGPU_RLC)
				{
					RlcPC pc = {m_gpuRLC.count};
					uint32_t groups = (m_gpuRLC.count + 255) / 256;
					vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rlcPreVoltPipeline);
					vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
					                        m_rlcPrePipeLayout, 0, 1, &m_gpuRLC.preDesc, 0, nullptr);
					vkCmdPushConstants(cmd, m_rlcPrePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RlcPC), &pc);
					vkCmdDispatch(cmd, groups, 1, 1);
				}
				// Dispersive pre-voltage
				if (!m_gpuDisp.empty())
					vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispPreVoltPipeline);
				for (const auto& d : m_gpuDisp)
				{
					if (!d.voltADEOn || d.debye) continue;
					uint32_t groups = (d.count + 255) / 256;
					vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
					                        m_dispPrePipeLayout, 0, 1, &d.preVoltDesc, 0, nullptr);
					vkCmdPushConstants(cmd, m_dispPrePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DispPC), &d.voltPC);
					vkCmdDispatch(cmd, groups, 1, 1);
				}
				RecordDebyeVoltPhase(cmd, false, -1, -1, false);
				CmdComputeBarrier(cmd, barrier);
			}
			RecordLocalABCPhase(cmd, 0);
			stageMark();

			// --- Fused voltage update (Yee + UPML pre + UPML post in one dispatch) ---
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_fusedVoltPipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
			                        m_fusedPipeLayout, 0, 1, &m_fusedVoltDescSet, 0, nullptr);
			vkCmdPushConstants(cmd, m_fusedPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
			                   0, sizeof(FusedPC), &fusedPC);
			vkCmdDispatch(cmd, voltGroups, 1, 1);

			CmdComputeBarrier(cmd, barrier);
			stageMark();
			if (m_validateStages)
				RecordGPUFieldValidation(cmd, 16);

			// --- Post-voltage: non-UPML only (Mur, TF/SF) ---
			// (UPML post-voltage is fused into the main dispatch above)
			if (m_hasGPU_Mur || m_hasGPU_TFSF)
			{
				// Mur post-voltage
				if (!m_gpuMur.empty())
					vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_murPostVoltPipeline);
				for (const auto& m : m_gpuMur)
				{
					if (ts < m.startTS) continue;
					uint32_t groups = (m.totalCells + 255) / 256;
					vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
					                        m_murUpdatePipeLayout, 0, 1, &m.postDesc, 0, nullptr);
				vkCmdPushConstants(cmd, m_murUpdatePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MurPC), &m.pc);
					vkCmdDispatch(cmd, groups, 1, 1);
				}
				// TF/SF voltage injection
				if (m_hasGPU_TFSF && m_gpuTFSF.voltCount > 0 && tfsfExc)
				{
					TfsfPC tpc = {m_gpuTFSF.voltCount, (int32_t)(ts+1), tfsfSigLen, tfsfSigPeriod};
					uint32_t groups = (m_gpuTFSF.voltCount + 255) / 256;
					vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tfsfVoltPipeline);
					vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
					                        m_tfsfPipeLayout, 0, 1, &m_gpuTFSF.voltDesc, 0, nullptr);
					vkCmdPushConstants(cmd, m_tfsfPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TfsfPC), &tpc);
					vkCmdDispatch(cmd, groups, 1, 1);
				}
				CmdComputeBarrier(cmd, barrier);
			}
			RecordLocalABCPhase(cmd, 1);
			stageMark();

			// --- Apply2Voltages (dispersive, Mur, RLC — same as non-fused path) ---
			if (!m_gpuDisp.empty())
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispApplyVoltPipeline);
			for (const auto &d : m_gpuDisp)
			{
				if (!d.voltADEOn || d.debye)
					continue;
				DispPC dpc = {d.count, numLines[0], numLines[1], numLines[2], 0};
				uint32_t groups = (d.count + 255) / 256;
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispApplyPipeLayout, 0, 1,
										&d.applyVoltDesc, 0, nullptr);
				vkCmdPushConstants(cmd, m_dispApplyPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DispPC),
								   &dpc);
				vkCmdDispatch(cmd, groups, 1, 1);
				CmdComputeBarrier(cmd, barrier);
			}
			RecordDebyeVoltPhase(cmd, true, -1, -1, true);
			if (!m_gpuMur.empty())
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_murApplyVoltPipeline);
			for (const auto &m : m_gpuMur)
			{
				if (ts < m.startTS)
					continue;
				uint32_t groups = (m.totalCells + 255) / 256;
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_murApplyPipeLayout, 0, 1,
										&m.applyDesc, 0, nullptr);
				vkCmdPushConstants(cmd, m_murApplyPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MurPC), &m.pc);
				vkCmdDispatch(cmd, groups, 1, 1);
			}
			// Mur and RLC can share cells; the CPU applies them in this order.
			if (m_hasGPU_Mur && m_hasGPU_RLC)
				CmdComputeBarrier(cmd, barrier);
			if (m_hasGPU_RLC)
			{
				RlcPC pc = {m_gpuRLC.count};
				uint32_t groups = (m_gpuRLC.count + 255) / 256;
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rlcApplyVoltPipeline);
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rlcApplyPipeLayout, 0, 1,
										&m_gpuRLC.applyDesc, 0, nullptr);
				vkCmdPushConstants(cmd, m_rlcApplyPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RlcPC), &pc);
				vkCmdDispatch(cmd, groups, 1, 1);
			}
			// A lumped port puts its resistor and its excitation on the same
			// edges.  Without this barrier the excitation's += and the RLC's
			// write are unordered.
			if ((m_hasGPU_Mur || m_hasGPU_RLC) && (hasExcVolt || !m_gpuLocalABC.empty()))
				CmdComputeBarrier(cmd, barrier);
			RecordLocalABCPhase(cmd, 2);

			// --- Voltage excitation ---
			// Match Engine::Apply2Voltages(): dispersive and RLC corrections
			// precede the low-priority excitation extension.
			if (hasExcVolt)
			{
				ExcPC epc = {m_excVoltCount, (int32_t)ts, m_excSignalLen, m_excSignalPeriod};
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_excPipeline);
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				                        m_excPipeLayout, 0, 1, &m_excVoltDescSet, 0, nullptr);
				vkCmdPushConstants(cmd, m_excPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
				                   0, sizeof(ExcPC), &epc);
				vkCmdDispatch(cmd, excVoltGroups, 1, 1);
			}

			// Late voltage writes must be visible to pre-current extensions
			// and to the fused Yee current update.
			if (hasVoltLateWritesFused)
			{
				CmdComputeBarrier(cmd, barrier);
			}
			if (m_validateStages)
				RecordGPUFieldValidation(cmd, 32);
			RecordSteadyStateSample(cmd, ts);
			RecordSteadyStateEnergy(cmd, ts, 0, numLines[0]);
			stageMark();

			// --- Fused current update (Yee + UPML pre + UPML post) ---
			// Pre-current non-UPML extensions (dispersive)
			if (m_hasGPU_Dispersive)
			{
				if (!m_gpuDisp.empty())
					vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispPreCurrPipeline);
				for (const auto &d : m_gpuDisp)
				{
					if (!d.currADEOn)
						continue;
					uint32_t groups = (d.count + 255) / 256;
					vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispPrePipeLayout, 0, 1,
											&d.preCurrDesc, 0, nullptr);
					vkCmdPushConstants(cmd, m_dispPrePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DispPC),
									   &d.currPC);
					vkCmdDispatch(cmd, groups, 1, 1);
				}
				CmdComputeBarrier(cmd, barrier);
			}
			RecordLocalABCPhase(cmd, 3);
			stageMark();

			// Fused current dispatch (Yee + UPML pre/post)
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_fusedCurrPipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_fusedPipeLayout, 0, 1,
									&m_fusedCurrDescSet, 0, nullptr);
			vkCmdPushConstants(cmd, m_fusedPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FusedPC), &fusedPC);
			vkCmdDispatch(cmd, voltGroups, 1, 1); // dispatch N threads (same as voltage)

			CmdComputeBarrier(cmd, barrier);
			stageMark();
			if (m_validateStages)
				RecordGPUFieldValidation(cmd, 64);

			// --- Post-current: TF/SF only (UPML is fused) ---
			if (m_hasGPU_TFSF && m_gpuTFSF.currCount > 0 && tfsfExc)
			{
				TfsfPC tpc = {m_gpuTFSF.currCount, (int32_t)(ts), tfsfSigLen, tfsfSigPeriod};
				uint32_t groups = (m_gpuTFSF.currCount + 255) / 256;
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tfsfCurrPipeline);
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tfsfPipeLayout, 0, 1,
										&m_gpuTFSF.currDesc, 0, nullptr);
				vkCmdPushConstants(cmd, m_tfsfPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TfsfPC), &tpc);
				vkCmdDispatch(cmd, groups, 1, 1);
			}
			if (m_hasGPU_TFSF && hasDispCurrADE)
			{
				CmdComputeBarrier(cmd, barrier);
			}
			RecordLocalABCPhase(cmd, 4);
			// Dispersive apply current
			if (!m_gpuDisp.empty())
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispApplyCurrPipeline);
			for (const auto &d : m_gpuDisp)
			{
				if (!d.currADEOn)
					continue;
				DispPC dpc = {d.count, numLines[0], numLines[1], numLines[2], 0};
				uint32_t groups = (d.count + 255) / 256;
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispApplyPipeLayout, 0, 1,
										&d.applyCurrDesc, 0, nullptr);
				vkCmdPushConstants(cmd, m_dispApplyPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DispPC),
								   &dpc);
				vkCmdDispatch(cmd, groups, 1, 1);
				CmdComputeBarrier(cmd, barrier);
			}
			RecordLocalABCPhase(cmd, 5);

			// --- Current excitation ---
			// Match Engine::Apply2Current(): dispersive corrections precede
			// the low-priority excitation extension.
			if (hasExcCurr)
			{
				ExcPC epc = {m_excCurrCount, (int32_t)ts, m_excSignalLen, m_excSignalPeriod};
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_excPipeline);
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_excPipeLayout, 0, 1,
										&m_excCurrDescSet, 0, nullptr);
				vkCmdPushConstants(cmd, m_excPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ExcPC), &epc);
				vkCmdDispatch(cmd, excCurrGroups, 1, 1);
			}
			if (hasCurrLateWritesFused)
			{
				CmdComputeBarrier(cmd, barrier);
			}
			if (m_validateStages)
				RecordGPUFieldValidation(cmd, 80);
		}
		else
		{
		// ============= Original separate-dispatch path =============
		// === Pre-voltage extensions (GPU) ===
		RecordVoltageExtensions(cmd, ts);
		if (m_validateStages)
			RecordGPUFieldValidation(cmd, 0);
		stageMark();

		// --- Voltage update (with Mur post and apply, when fused) ---
		{
			const VkPipelineLayout vLayout = m_murFused ? m_murFusedPipeLayout : m_fdtdPipeLayout;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
			                  m_murFused ? m_murFusedPipeline : m_updateVoltPipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vLayout, 0, 1,
			                        m_murFused ? &m_murFusedDescSet : &m_updateVoltDescSet, 0, nullptr);
			vkCmdPushConstants(cmd, vLayout, VK_SHADER_STAGE_COMPUTE_BIT,
			                   0, sizeof(GridPC), &gridPC);
			vkCmdDispatch(cmd, voltGroups, 1, 1);
		}

		CmdComputeBarrier(cmd, barrier);
		stageMark();
		if (m_validateStages)
			RecordGPUFieldValidation(cmd, 16);

		// === Post-voltage extensions (GPU) ===
		// UPML post-voltage
		if (!m_gpuUPML.empty())
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_upmlPostVoltPipeline);
		for (const auto& u : m_gpuUPML)
		{
			uint32_t groups = (u.totalCells + 255) / 256;
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
			                        m_upmlPostPipeLayout, 0, 1, &u.postVoltDesc, 0, nullptr);
		vkCmdPushConstants(cmd, m_upmlPostPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PmlPC), &u.pc);
			vkCmdDispatch(cmd, groups, 1, 1);
		}
		// UPML post-voltage turns the flux back into the field; Mur and
		// TF/SF come after it on the CPU and may read the same cells.
		if (!m_gpuUPML.empty() && (m_hasGPU_Mur || m_hasGPU_TFSF))
			CmdComputeBarrier(cmd, barrier);

		// Mur post-voltage (accumulates shifted field)
		if (murSeparate)
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_murPostVoltPipeline);
		for (const auto& m : m_gpuMur)
		{
			if (!murSeparate) break;
			if (ts < m.startTS) continue;
			uint32_t groups = (m.totalCells + 255) / 256;
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
			                        m_murUpdatePipeLayout, 0, 1, &m.postDesc, 0, nullptr);
		vkCmdPushConstants(cmd, m_murUpdatePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MurPC), &m.pc);
			vkCmdDispatch(cmd, groups, 1, 1);
		}

		// TF/SF voltage injection
		if (m_hasGPU_TFSF && m_gpuTFSF.voltCount > 0 && tfsfExc)
		{
			TfsfPC tpc = {m_gpuTFSF.voltCount, (int32_t)(ts+1), tfsfSigLen, tfsfSigPeriod};
			uint32_t groups = (m_gpuTFSF.voltCount + 255) / 256;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tfsfVoltPipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
			                        m_tfsfPipeLayout, 0, 1, &m_gpuTFSF.voltDesc, 0, nullptr);
			vkCmdPushConstants(cmd, m_tfsfPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TfsfPC), &tpc);
			vkCmdDispatch(cmd, groups, 1, 1);
		}

		// Barrier for post-voltage writes
		if (m_hasGPU_UPML || murSeparate || m_hasGPU_TFSF)
		{
			CmdComputeBarrier(cmd, barrier);
		}
		RecordLocalABCPhase(cmd, 1);
		stageMark();

		// === Apply2Voltages extensions (GPU) ===
		// Dispersive apply
		if (!m_gpuDisp.empty())
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispApplyVoltPipeline);
		for (const auto &d : m_gpuDisp)
		{
			if (!d.voltADEOn || d.debye)
				continue;
			DispPC dpc = {d.count, numLines[0], numLines[1], numLines[2], 0};
			uint32_t groups = (d.count + 255) / 256;
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispApplyPipeLayout, 0, 1,
									&d.applyVoltDesc, 0, nullptr);
			vkCmdPushConstants(cmd, m_dispApplyPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DispPC), &dpc);
			vkCmdDispatch(cmd, groups, 1, 1);
			CmdComputeBarrier(cmd, barrier);
		}
		RecordDebyeVoltPhase(cmd, true, -1, -1, true);

		// Mur apply (overwrites boundary voltages)
		if (murSeparate)
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_murApplyVoltPipeline);
		for (const auto &m : m_gpuMur)
		{
			if (!murSeparate)
				break;
			if (ts < m.startTS)
				continue;
			uint32_t groups = (m.totalCells + 255) / 256;
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_murApplyPipeLayout, 0, 1, &m.applyDesc,
									0, nullptr);
			vkCmdPushConstants(cmd, m_murApplyPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MurPC), &m.pc);
			vkCmdDispatch(cmd, groups, 1, 1);
		}
		// Mur and RLC can share cells; the CPU applies them in this order.
		if (murSeparate && m_hasGPU_RLC)
			CmdComputeBarrier(cmd, barrier);

		// RLC apply
		if (m_hasGPU_RLC)
		{
			RlcPC pc = {m_gpuRLC.count};
			uint32_t groups = (m_gpuRLC.count + 255) / 256;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rlcApplyVoltPipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rlcApplyPipeLayout, 0, 1,
									&m_gpuRLC.applyDesc, 0, nullptr);
			vkCmdPushConstants(cmd, m_rlcApplyPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RlcPC), &pc);
			vkCmdDispatch(cmd, groups, 1, 1);
		}
		// A lumped port puts its resistor and its excitation on the same
		// edges.  Without this barrier the excitation's += and the RLC's
		// write are unordered.
		if ((murSeparate || m_hasGPU_RLC) && (hasExcVolt || !m_gpuLocalABC.empty()))
			CmdComputeBarrier(cmd, barrier);
		RecordLocalABCPhase(cmd, 2);

		// --- Voltage excitation ---
		// Match Engine::Apply2Voltages(): dispersive and RLC corrections
		// precede the low-priority excitation extension.
		if (hasExcVolt)
		{
			ExcPC epc = {m_excVoltCount, (int32_t)ts, m_excSignalLen, m_excSignalPeriod};
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_excPipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
			                        m_excPipeLayout, 0, 1, &m_excVoltDescSet, 0, nullptr);
			vkCmdPushConstants(cmd, m_excPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
			                   0, sizeof(ExcPC), &epc);
			vkCmdDispatch(cmd, excVoltGroups, 1, 1);
		}

		if (hasVoltLateWrites)
		{
			CmdComputeBarrier(cmd, barrier);
		}
		if (m_validateStages)
			RecordGPUFieldValidation(cmd, 32);
		RecordSteadyStateSample(cmd, ts);
		RecordSteadyStateEnergy(cmd, ts, 0, numLines[0]);
		stageMark();

		// === Pre-current extensions (GPU) ===
		RecordCurrentExtensions(cmd, ts);
		if (m_validateStages)
			RecordGPUFieldValidation(cmd, 48);
		stageMark();

		// --- Current update ---
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_updateCurrPipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_fdtdPipeLayout, 0, 1, &m_updateCurrDescSet,
								0, nullptr);
		vkCmdPushConstants(cmd, m_fdtdPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GridPC), &gridPCc);
		vkCmdDispatch(cmd, currGroups, 1, 1);

		CmdComputeBarrier(cmd, barrier);
		stageMark();
		if (m_validateStages)
			RecordGPUFieldValidation(cmd, 64);

		// === Post-current extensions (GPU) ===
		// UPML post-current
		if (!m_gpuUPML.empty())
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_upmlPostCurrPipeline);
		for (const auto &u : m_gpuUPML)
		{
			uint32_t groups = (u.totalCells + 255) / 256;
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_upmlPostPipeLayout, 0, 1,
									&u.postCurrDesc, 0, nullptr);
			vkCmdPushConstants(cmd, m_upmlPostPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PmlPC), &u.pc);
			vkCmdDispatch(cmd, groups, 1, 1);
		}
		if (!m_gpuUPML.empty() && m_hasGPU_TFSF && m_gpuTFSF.currCount > 0)
			CmdComputeBarrier(cmd, barrier);

		// TF/SF current injection
		if (m_hasGPU_TFSF && m_gpuTFSF.currCount > 0 && tfsfExc)
		{
			TfsfPC tpc = {m_gpuTFSF.currCount, (int32_t)(ts), tfsfSigLen, tfsfSigPeriod};
			uint32_t groups = (m_gpuTFSF.currCount + 255) / 256;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tfsfCurrPipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tfsfPipeLayout, 0, 1,
									&m_gpuTFSF.currDesc, 0, nullptr);
			vkCmdPushConstants(cmd, m_tfsfPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TfsfPC), &tpc);
			vkCmdDispatch(cmd, groups, 1, 1);
		}

		if ((m_hasGPU_UPML || m_hasGPU_TFSF) && hasDispCurrADE)
		{
			CmdComputeBarrier(cmd, barrier);
		}
		RecordLocalABCPhase(cmd, 4);

		// Dispersive apply current
		if (!m_gpuDisp.empty())
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispApplyCurrPipeline);
		for (const auto &d : m_gpuDisp)
		{
			if (!d.currADEOn)
				continue;
			DispPC dpc = {d.count, numLines[0], numLines[1], numLines[2], 0};
			uint32_t groups = (d.count + 255) / 256;
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispApplyPipeLayout, 0, 1,
									&d.applyCurrDesc, 0, nullptr);
			vkCmdPushConstants(cmd, m_dispApplyPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DispPC), &dpc);
			vkCmdDispatch(cmd, groups, 1, 1);
			CmdComputeBarrier(cmd, barrier);
		}
		RecordLocalABCPhase(cmd, 5);

		// --- Current excitation ---
		// Match Engine::Apply2Current(): dispersive corrections precede
		// the low-priority excitation extension.
		if (hasExcCurr)
		{
			ExcPC epc = {m_excCurrCount, (int32_t)ts, m_excSignalLen, m_excSignalPeriod};
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_excPipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_excPipeLayout, 0, 1,
									&m_excCurrDescSet, 0, nullptr);
			vkCmdPushConstants(cmd, m_excPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ExcPC), &epc);
			vkCmdDispatch(cmd, excCurrGroups, 1, 1);
		}

		if (hasCurrLateWrites)
		{
			CmdComputeBarrier(cmd, barrier);
		}
		if (m_validateStages)
			RecordGPUFieldValidation(cmd, 80);
		} // end else (original separate-dispatch path)
		stageMark();
	}

	if (m_validateStages)
	{
		VkMemoryBarrier hostBarrier{};
		hostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		hostBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     VK_PIPELINE_STAGE_HOST_BIT, 0,
		                     1, &hostBarrier, 0, nullptr, 0, nullptr);
	}

	if (m_steadyPipeline)
	{
		VkMemoryBarrier hostBarrier{};
		hostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		hostBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     VK_PIPELINE_STAGE_HOST_BIT, 0,
		                     1, &hostBarrier, 0, nullptr, 0, nullptr);
	}

	if (m_profileEnabled && m_tsQueryPool && m_profilePhaseEnabled)
	{
		// End of compute/update section (before probe gather/copy section)
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_tsQueryPool, qBaseProfile + 1);
	}

	// --- Probe gather: append at end of batch ----------------------
	if (m_hasGPU_Probes && m_probeCacheCount > 0)
	{
		// Final barrier: ensure all Yee/extension writes are visible
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0, 1, &barrier, 0, nullptr, 0, nullptr);

		uint32_t probeGroups = (m_probeCacheCount + 255) / 256;
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_probePipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		                        m_probePipeLayout, 0, 1, &m_probeGatherDescSet, 0, nullptr);
		vkCmdPushConstants(cmd, m_probePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
		                   0, sizeof(uint32_t), &m_probeCacheCount);
		vkCmdDispatch(cmd, probeGroups, 1, 1);

		// The host reads m_probeOutMapped once this batch's fence signals.
		VkMemoryBarrier hostBarrier{};
		hostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		hostBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
			0, 1, &hostBarrier, 0, nullptr, 0, nullptr);
	}

	if (m_profileEnabled && m_tsQueryPool && m_profilePhaseEnabled)
	{
		// End of probe section (or immediately after compute when no probes)
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_tsQueryPool, qBaseProfile + 2);
	}

	// --- Field dumps due at the batch's last timestep ----------------
	RecordDumpSamples(cmd, numTS + iterTS);

	if (m_profileEnabled && m_tsQueryPool)
	{
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
		                   m_tsQueryPool, qBaseProfile + (m_profileQueriesPerSlot - 1));
	}

	// --- End + submit (fire-and-forget) ---
	VK_CHECK(vkEndCommandBuffer(cmd));

	VkSubmitInfo si{};
	si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	si.commandBufferCount = 1;
	si.pCommandBuffers    = &cmd;
	vkResetFences(m_device, 1, &m_fences[m_cmdIdx]);
	VK_CHECK(vkQueueSubmit(m_computeQueue, 1, &si, m_fences[m_cmdIdx]));
	m_submitSerial[m_cmdIdx] = ++m_submitCount;

	m_gpuInFlight[m_cmdIdx] = true;
	if (m_profileEnabled)
	{
		m_profilePending[m_cmdIdx] = true;
		m_profileSubmittedTS[m_cmdIdx] = iterTS;
		m_profileRecordedTS += iterTS;
		m_stageSlotTS[m_cmdIdx] = (m_stageQueryPool && !m_tbActive) ? iterTS : 0;
	}
	m_gpuDrained = false;  // new submission in flight
	m_cmdIdx = (m_cmdIdx + 1) % CMD_RING_SIZE;

	numTS += iterTS;
	m_liveTS = numTS;
	m_hostDirty = true;

	if (m_validateStages)
		ReportGPUFieldValidation(numTS);
	else if (m_validateFields)
		ValidateGPUFields(numTS);

	return true;
}

// ===========================================================================
// Cleanup
// ===========================================================================

void Engine_Vulkan::CleanupVulkan()
{
	if (m_device == VK_NULL_HANDLE) return;
	vkDeviceWaitIdle(m_device);

	// Field-dump gather ring, DFT sums and pipelines
	CleanupDumps();

	// First clean up extension resources
	CleanupExtensions();

	auto destroyBufMem = [&](VkBuffer& buf, VkDeviceMemory& mem)
	{
		if (buf) { vkDestroyBuffer(m_device, buf, nullptr); buf = VK_NULL_HANDLE; }
		if (mem) { vkFreeMemory(m_device, mem, nullptr);    mem = VK_NULL_HANDLE; }
	};

	// Pipelines
	if (m_updateVoltPipeline) vkDestroyPipeline(m_device, m_updateVoltPipeline, nullptr);
	if (m_updateCurrPipeline) vkDestroyPipeline(m_device, m_updateCurrPipeline, nullptr);
	if (m_excPipeline)        vkDestroyPipeline(m_device, m_excPipeline, nullptr);
	if (m_fdtdPipeLayout) vkDestroyPipelineLayout(m_device, m_fdtdPipeLayout, nullptr);
	if (m_excPipeLayout)  vkDestroyPipelineLayout(m_device, m_excPipeLayout, nullptr);

	// Descriptors
	if (m_descPool)       vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
	if (m_fdtdDescLayout) vkDestroyDescriptorSetLayout(m_device, m_fdtdDescLayout, nullptr);
	if (m_excDescLayout)  vkDestroyDescriptorSetLayout(m_device, m_excDescLayout, nullptr);

	// Buffers
	destroyBufMem(m_voltBuf, m_voltMem);
	destroyBufMem(m_currBuf, m_currMem);
	destroyBufMem(m_opIndexBuf, m_opIndexMem);
	destroyBufMem(m_vvCompBuf, m_vvCompMem);
	destroyBufMem(m_viCompBuf, m_viCompMem);
	destroyBufMem(m_iiCompBuf, m_iiCompMem);
	destroyBufMem(m_ivCompBuf, m_ivCompMem);
	destroyBufMem(m_stagingBuf, m_stagingMem);
	destroyBufMem(m_excVoltIdxBuf, m_excVoltIdxMem);
	destroyBufMem(m_excVoltAmpBuf, m_excVoltAmpMem);
	destroyBufMem(m_excVoltDelayBuf, m_excVoltDelayMem);
	destroyBufMem(m_excCurrIdxBuf, m_excCurrIdxMem);
	destroyBufMem(m_excCurrAmpBuf, m_excCurrAmpMem);
	destroyBufMem(m_excCurrDelayBuf, m_excCurrDelayMem);
	destroyBufMem(m_excSignalVoltBuf, m_excSignalVoltMem);
	destroyBufMem(m_excSignalCurrBuf, m_excSignalCurrMem);

	// Command / sync
	for (int i = 0; i < CMD_RING_SIZE; ++i)
		if (m_fences[i]) vkDestroyFence(m_device, m_fences[i], nullptr);
	if (m_utilFence) vkDestroyFence(m_device, m_utilFence, nullptr);
	if (m_cmdPool) vkDestroyCommandPool(m_device, m_cmdPool, nullptr);

	// Optional raw-field validation
	CleanupGPU_FieldValidation();

	// Energy reduction
	CleanupGPU_EnergyReduction();

	// Fused UPML
	CleanupFusedUPML();

	if (m_profileEnabled)
	{
		for (int i = 0; i < CMD_RING_SIZE; ++i)
			CollectProfileForSlot(i);
		PrintProfileSummary();
	}
	if (m_tsQueryPool)
	{
		vkDestroyQueryPool(m_device, m_tsQueryPool, nullptr);
		m_tsQueryPool = VK_NULL_HANDLE;
	}
	if (m_stageQueryPool)
	{
		vkDestroyQueryPool(m_device, m_stageQueryPool, nullptr);
		m_stageQueryPool = VK_NULL_HANDLE;
	}

	// Pipeline cache
	if (m_pipelineCache) vkDestroyPipelineCache(m_device, m_pipelineCache, nullptr);

	// Device / instance
	vkDestroyDevice(m_device, nullptr);
	m_device = VK_NULL_HANDLE;
	vkDestroyInstance(m_instance, nullptr);
	m_instance = VK_NULL_HANDLE;
}

// ===========================================================================
// Extension cleanup
// ===========================================================================

void Engine_Vulkan::CleanupExtensions()
{
	if (m_device == VK_NULL_HANDLE) return;

	// --- Destroy extension pipelines ---
	auto destroyPipeline = [&](VkPipeline& p)
	{
		if (p) { vkDestroyPipeline(m_device, p, nullptr); p = VK_NULL_HANDLE; }
	};
	auto destroyPipeLayout = [&](VkPipelineLayout& pl)
	{
		if (pl) { vkDestroyPipelineLayout(m_device, pl, nullptr); pl = VK_NULL_HANDLE; }
	};
	auto destroyDescLayout = [&](VkDescriptorSetLayout& dl)
	{
		if (dl) { vkDestroyDescriptorSetLayout(m_device, dl, nullptr); dl = VK_NULL_HANDLE; }
	};

	if (m_gpuSteadyStateExt)
		m_gpuSteadyStateExt->SetGpuUpdater({});
	if (m_steadyHistoryMapped)
	{
		vkUnmapMemory(m_device, m_steadyHistory.memory);
		m_steadyHistoryMapped = nullptr;
	}
	destroyPipeline(m_steadyPipeline);
	destroyPipeLayout(m_steadyPipeLayout);
	if (m_steadyDescPool)
	{
		vkDestroyDescriptorPool(m_device, m_steadyDescPool, nullptr);
		m_steadyDescPool = VK_NULL_HANDLE;
	}
	destroyDescLayout(m_steadyDescLayout);
	DestroyGpuBuf(m_steadyProbeIdx);
	DestroyGpuBuf(m_steadyHistory);
	m_steadyDescSet = VK_NULL_HANDLE;
	m_gpuSteadyStateExt = nullptr;
	m_steadyProbeCount = m_steadyPeriod = m_steadyRingSize = 0;
	m_steadyProbeX.clear();
	if (m_steadyEnergyMapped)
	{
		vkUnmapMemory(m_device, m_steadyEnergy.memory);
		m_steadyEnergyMapped = nullptr;
	}
	destroyPipeline(m_steadyEnergyPipeline);
	destroyPipeLayout(m_steadyEnergyPipeLayout);
	if (m_steadyEnergyDescPool)
	{
		vkDestroyDescriptorPool(m_device, m_steadyEnergyDescPool, nullptr);
		m_steadyEnergyDescPool = VK_NULL_HANDLE;
	}
	destroyDescLayout(m_steadyEnergyDescLayout);
	DestroyGpuBuf(m_steadyEnergy);
	m_steadyEnergyDescSet = VK_NULL_HANDLE;
	m_steadyChecks.clear();
	m_steadyChunks = 0;
	m_steadySlotFloats = 0;

	destroyPipeline(m_localAbcPipeline);
	destroyPipeLayout(m_localAbcPipeLayout);
	if (m_localAbcDescPool)
	{
		vkDestroyDescriptorPool(m_device, m_localAbcDescPool, nullptr);
		m_localAbcDescPool = VK_NULL_HANDLE;
	}
	destroyDescLayout(m_localAbcDescLayout);
	for (GpuLocalABCData& data : m_gpuLocalABC)
	{
		DestroyGpuBuf(data.voltState);
		DestroyGpuBuf(data.currState);
		DestroyGpuBuf(data.k1);
		DestroyGpuBuf(data.k2);
	}
	m_gpuLocalABC.clear();

	destroyPipeline(m_upmlPreVoltPipeline);
	destroyPipeline(m_upmlPostVoltPipeline);
	destroyPipeline(m_upmlPreCurrPipeline);
	destroyPipeline(m_upmlPostCurrPipeline);
	destroyPipeline(m_dispPreVoltPipeline);
	destroyPipeline(m_debyePreVoltPipeline);
	destroyPipeline(m_debyeApplyVoltPipeline);
	destroyPipeline(m_dispPreCurrPipeline);
	destroyPipeline(m_dispApplyVoltPipeline);
	destroyPipeline(m_dispApplyCurrPipeline);
	destroyPipeline(m_tfsfVoltPipeline);
	destroyPipeline(m_tfsfCurrPipeline);
	destroyPipeline(m_murPreVoltPipeline);
	destroyPipeline(m_murPostVoltPipeline);
	destroyPipeline(m_murApplyVoltPipeline);
	destroyPipeline(m_rlcPreVoltPipeline);
	destroyPipeline(m_rlcApplyVoltPipeline);
	destroyPipeline(m_probePipeline);

	destroyPipeLayout(m_upmlPrePipeLayout);
	destroyPipeLayout(m_upmlPostPipeLayout);
	destroyPipeLayout(m_dispPrePipeLayout);
	destroyPipeLayout(m_debyePipeLayout);
	destroyPipeLayout(m_dispApplyPipeLayout);
	destroyPipeLayout(m_tfsfPipeLayout);
	destroyPipeLayout(m_murUpdatePipeLayout);
	destroyPipeLayout(m_murApplyPipeLayout);
	destroyPipeLayout(m_rlcPrePipeLayout);
	destroyPipeLayout(m_rlcApplyPipeLayout);
	destroyPipeLayout(m_probePipeLayout);

	// --- Destroy extension descriptor pool (this frees all descriptor sets) ---
	if (m_extDescPool) { vkDestroyDescriptorPool(m_device, m_extDescPool, nullptr); m_extDescPool = VK_NULL_HANDLE; }

	destroyDescLayout(m_upmlPreVoltDescLayout);
	destroyDescLayout(m_upmlPostVoltDescLayout);
	destroyDescLayout(m_upmlPreCurrDescLayout);
	destroyDescLayout(m_upmlPostCurrDescLayout);
	destroyDescLayout(m_dispPreDescLayout);
	destroyDescLayout(m_debyeDescLayout);
	destroyDescLayout(m_dispApplyDescLayout);
	destroyDescLayout(m_tfsfDescLayout);
	destroyDescLayout(m_murPreDescLayout);
	destroyDescLayout(m_murPostDescLayout);
	destroyDescLayout(m_murApplyDescLayout);
	destroyDescLayout(m_rlcPreDescLayout);
	destroyDescLayout(m_rlcApplyDescLayout);
	destroyDescLayout(m_probeDescLayout);

	// --- Destroy UPML GPU data ---
	for (auto& u : m_gpuUPML)
	{
		DestroyGpuBuf(u.voltFlux); DestroyGpuBuf(u.currFlux);
		DestroyGpuBuf(u.pmlIdx);
		// The coefficient tables are shared with the fused path, not per-region.
	}
	m_gpuUPML.clear();
	DestroyGpuBuf(m_pmlTabVv);   DestroyGpuBuf(m_pmlTabVvfo); DestroyGpuBuf(m_pmlTabVvfn);
	DestroyGpuBuf(m_pmlTabIi);   DestroyGpuBuf(m_pmlTabIifo); DestroyGpuBuf(m_pmlTabIifn);
	m_pmlTabCount = 0;

	// --- Destroy dispersive GPU data ---
	for (auto& d : m_gpuDisp)
	{
		DestroyGpuBuf(d.voltADE);    DestroyGpuBuf(d.voltLorADE);
		DestroyGpuBuf(d.currADE);    DestroyGpuBuf(d.currLorADE);
		DestroyGpuBuf(d.posIdx);
		DestroyGpuBuf(d.vIntADE);    DestroyGpuBuf(d.vExtADE); DestroyGpuBuf(d.vLorADE);
		DestroyGpuBuf(d.iIntADE);    DestroyGpuBuf(d.iExtADE); DestroyGpuBuf(d.iLorADE);
		DestroyGpuBuf(d.debyeW);     DestroyGpuBuf(d.debyePre);
		DestroyGpuBuf(d.debyeRelax); DestroyGpuBuf(d.debyeDrive); DestroyGpuBuf(d.debyeSolve);
	}
	m_gpuDisp.clear();

	// --- Destroy TF/SF GPU data ---
	DestroyGpuBuf(m_gpuTFSF.voltAmp);   DestroyGpuBuf(m_gpuTFSF.voltDelta);
	DestroyGpuBuf(m_gpuTFSF.voltIdx);   DestroyGpuBuf(m_gpuTFSF.voltDelay);
	DestroyGpuBuf(m_gpuTFSF.voltSignal);
	DestroyGpuBuf(m_gpuTFSF.currAmp);   DestroyGpuBuf(m_gpuTFSF.currDelta);
	DestroyGpuBuf(m_gpuTFSF.currIdx);   DestroyGpuBuf(m_gpuTFSF.currDelay);
	DestroyGpuBuf(m_gpuTFSF.currSignal);
	m_gpuTFSF = GpuTFSFData{};

	// --- Destroy Mur ABC GPU data ---
	for (GpuBuf* b : {&m_murAllNyP, &m_murAllNyPP, &m_murAllCoeffNyP, &m_murAllCoeffNyPP,
	                  &m_murAllVoldNyP, &m_murAllVoldNyPP, &m_murFaceBuf})
		DestroyGpuBuf(*b);
	m_gpuMur.clear();
	if (m_murFusedPipeline)   vkDestroyPipeline(m_device, m_murFusedPipeline, nullptr);
	if (m_murFusedPipeLayout) vkDestroyPipelineLayout(m_device, m_murFusedPipeLayout, nullptr);
	if (m_murFusedDescPool)   vkDestroyDescriptorPool(m_device, m_murFusedDescPool, nullptr);
	if (m_murFusedDescLayout) vkDestroyDescriptorSetLayout(m_device, m_murFusedDescLayout, nullptr);
	m_murFusedPipeline = VK_NULL_HANDLE; m_murFusedPipeLayout = VK_NULL_HANDLE;
	m_murFusedDescPool = VK_NULL_HANDLE; m_murFusedDescLayout = VK_NULL_HANDLE;
	m_murFused = false;

	// --- Destroy RLC GPU data ---
	DestroyGpuBuf(m_gpuRLC.il);
	DestroyGpuBuf(m_gpuRLC.vdn0); DestroyGpuBuf(m_gpuRLC.vdn1); DestroyGpuBuf(m_gpuRLC.vdn2);
	DestroyGpuBuf(m_gpuRLC.jn0);  DestroyGpuBuf(m_gpuRLC.jn1);  DestroyGpuBuf(m_gpuRLC.jn2);
	DestroyGpuBuf(m_gpuRLC.linIdx);
	DestroyGpuBuf(m_gpuRLC.i2v);  DestroyGpuBuf(m_gpuRLC.ilv);
	DestroyGpuBuf(m_gpuRLC.vvd);  DestroyGpuBuf(m_gpuRLC.vv2);
	DestroyGpuBuf(m_gpuRLC.vj1);  DestroyGpuBuf(m_gpuRLC.vj2);
	DestroyGpuBuf(m_gpuRLC.ib0);  DestroyGpuBuf(m_gpuRLC.b1); DestroyGpuBuf(m_gpuRLC.b2);
	m_gpuRLC = GpuRLCData{};

	// --- Destroy probe cache GPU data (pipelined processing) ---
	if (m_probeOutMapped)
	{
		vkUnmapMemory(m_device, m_probeOutBuf.memory);
		m_probeOutMapped = nullptr;
	}
	DestroyGpuBuf(m_probeIdxBuf);
	DestroyGpuBuf(m_probeSelBuf);
	DestroyGpuBuf(m_probeOutBuf);
	if (m_probeDescPool) { vkDestroyDescriptorPool(m_device, m_probeDescPool, nullptr); m_probeDescPool = VK_NULL_HANDLE; }
	m_probeGatherDescSet = VK_NULL_HANDLE; // freed with pool
	m_probeCacheKeys.clear();
	m_probeCacheCPU.clear();
	m_probeCacheCount = 0;

	// Reset flags
	m_hasGPU_UPML = false;
	m_hasGPU_Dispersive = false;
	m_hasGPU_TFSF = false;
	m_hasGPU_Mur = false;
	m_hasGPU_RLC = false;
	m_hasGPU_Probes = false;
}

// ===========================================================================
// Fused Yee+UPML Pipeline
// ===========================================================================

void Engine_Vulkan::SetupFusedUPML()
{
	if (!m_hasGPU_UPML || m_gpuUPML.empty())
		return;
	if (const char* env = std::getenv("OPENEMS_GPU_DISABLE_FUSED_UPML"))
	{
		char* endPtr = nullptr;
		long v = std::strtol(env, &endPtr, 10);
		if (endPtr && *endPtr == '\0' && v != 0)
		{
			cout << "Engine_Vulkan: fused UPML disabled by environment" << endl;
			return;
		}
	}

	// --- Compute total PML cells and build region metadata ---
	struct PMLRegionGPU { uint32_t startX, startY, startZ, sizeX, sizeY, sizeZ, fluxOffset; };
	std::vector<PMLRegionGPU> regions;
	uint32_t totalPmlCells = 0;

	for (const auto& u : m_gpuUPML)
	{
		PMLRegionGPU r;
		r.startX = u.pc.pStartX;
		r.startY = u.pc.pStartY;
		r.startZ = u.pc.pStartZ;
		r.sizeX  = u.pc.pNx;
		r.sizeY  = u.pc.pNy;
		r.sizeZ  = u.pc.pNz;
		// Each region is packed as [component 0][component 1][component 2].
		// Store an element offset, not a per-component cell offset.
		r.fluxOffset = 3u * totalPmlCells;
		regions.push_back(r);
		totalPmlCells += u.totalCells;
	}
	m_fusedTotalPmlCells = totalPmlCells;

	if (regions.size() > 6)
	{
		cerr << "Engine_Vulkan::SetupFusedUPML: too many PML regions (" << regions.size()
		     << "), max 6 — falling back to separate dispatches" << endl;
		return;
	}

	VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	                           VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkBufferUsageFlags rwUsage = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VkMemoryPropertyFlags dLoc = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	VkDeviceSize fluxBufSize = 3 * (VkDeviceSize)totalPmlCells * sizeof(float);
	if (fluxBufSize == 0) return;
	for (size_t ri = 0; ri < regions.size(); ++ri)
	{
		VkDeviceSize regionEnd = ((VkDeviceSize)regions[ri].fluxOffset +
		                              3 * (VkDeviceSize)m_gpuUPML[ri].totalCells) * sizeof(float);
		if (regionEnd > fluxBufSize)
			throw std::runtime_error("Engine_Vulkan::SetupFusedUPML: packed region exceeds buffer");
	}

	// --- Create concatenated buffers ---
	CreateGpuBuf(m_fusedVoltFlux, fluxBufSize, rwUsage, dLoc);
	CreateGpuBuf(m_fusedCurrFlux, fluxBufSize, rwUsage, dLoc);

	// The coefficient tables are shared with the separate-dispatch path and
	// hold global indices, so only the index needs concatenating here.
	{
		std::vector<uint32_t> allIdx;
		allIdx.reserve(3u * (size_t)totalPmlCells);
		for (const auto& u : m_gpuUPML)
			allIdx.insert(allIdx.end(), u.coefIdx.begin(), u.coefIdx.end());
		if (allIdx.size() != 3u * (size_t)totalPmlCells)
			throw std::runtime_error("Engine_Vulkan::SetupFusedUPML: PML index size mismatch");

		m_fusedPmlIdxBufSize = NarrowIndexBufSize(allIdx.size(), m_pmlIdxBits);
		CreateGpuBuf(m_fusedPmlIdx, m_fusedPmlIdxBufSize, usage, dLoc);
		std::vector<uint32_t> packed =
			PackNarrowIndices(allIdx.data(), allIdx.size(), m_pmlIdxBits);
		packed.resize((size_t)m_fusedPmlIdxBufSize / sizeof(uint32_t), 0u);
		UploadGpuBuf(m_fusedPmlIdx, packed.data(), m_fusedPmlIdxBufSize);
	}

	// --- Upload concatenated data (copy from per-region buffers via staging) ---
	// Initialize flux to zero
	{
		std::vector<float> zeros(3 * totalPmlCells, 0.0f);
		UploadGpuBuf(m_fusedVoltFlux, zeros.data(), fluxBufSize);
		UploadGpuBuf(m_fusedCurrFlux, zeros.data(), fluxBufSize);
	}


	// --- PML region metadata SSBO ---
	VkDeviceSize regionInfoSize = regions.size() * sizeof(PMLRegionGPU);
	CreateGpuBuf(m_fusedPmlRegionInfo, regionInfoSize, usage, dLoc);
	UploadGpuBuf(m_fusedPmlRegionInfo, regions.data(), regionInfoSize);

	// --- Descriptor set layout (11 bindings) ---
	{
		std::vector<VkDescriptorSetLayoutBinding> bindings(11);
		for (uint32_t i = 0; i < 11; i++)
			bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
		VkDescriptorSetLayoutCreateInfo ci{};
		ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		ci.bindingCount = 11;
		ci.pBindings    = bindings.data();
		VK_CHECK(vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &m_fusedDescLayout));
	}

	// --- Pipeline layout (push constant = FusedPC, 20 bytes) ---
	VkPushConstantRange pcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FusedPC)};
	VkPipelineLayoutCreateInfo pli{};
	pli.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pli.setLayoutCount         = 1;
	pli.pSetLayouts            = &m_fusedDescLayout;
	pli.pushConstantRangeCount = 1;
	pli.pPushConstantRanges    = &pcRange;
	VK_CHECK(vkCreatePipelineLayout(m_device, &pli, nullptr, &m_fusedPipeLayout));

	// --- Descriptor pool (2 sets × 11 bindings) ---
	VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 11 * 2};
	VkDescriptorPoolCreateInfo pi{};
	pi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pi.maxSets       = 2;
	pi.poolSizeCount = 1;
	pi.pPoolSizes    = &poolSize;
	VK_CHECK(vkCreateDescriptorPool(m_device, &pi, nullptr, &m_fusedDescPool));

	// --- Allocate descriptor sets ---
	VkDescriptorSetLayout layouts[2] = {m_fusedDescLayout, m_fusedDescLayout};
	VkDescriptorSetAllocateInfo ai{};
	ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	ai.descriptorPool     = m_fusedDescPool;
	ai.descriptorSetCount = 2;
	ai.pSetLayouts        = layouts;
	VkDescriptorSet sets[2];
	VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, sets));
	m_fusedVoltDescSet = sets[0];
	m_fusedCurrDescSet = sets[1];

	// --- Write fused voltage descriptors ---
	{
		VkBuffer bufs[11] = {
			m_voltBuf, m_currBuf, m_opIndexBuf, m_vvCompBuf, m_viCompBuf,
			m_fusedPmlRegionInfo.buffer, m_fusedVoltFlux.buffer, m_fusedPmlIdx.buffer,
			m_pmlTabVv.buffer, m_pmlTabVvfo.buffer, m_pmlTabVvfn.buffer
		};
		VkDeviceSize sizes[11] = {
			m_fieldBufSize, m_fieldBufSize, m_opIndexBufSize,
			m_coeffCompBufSize, m_coeffCompBufSize,
			regionInfoSize, fluxBufSize, m_fusedPmlIdxBufSize,
			m_pmlTabBufSize, m_pmlTabBufSize, m_pmlTabBufSize
		};
		WriteDescriptorBuffers(m_device, m_fusedVoltDescSet, bufs, sizes, 11);
	}

	// --- Write fused current descriptors ---
	{
		VkBuffer bufs[11] = {
			m_currBuf, m_voltBuf, m_opIndexBuf, m_iiCompBuf, m_ivCompBuf,
			m_fusedPmlRegionInfo.buffer, m_fusedCurrFlux.buffer, m_fusedPmlIdx.buffer,
			m_pmlTabIi.buffer, m_pmlTabIifo.buffer, m_pmlTabIifn.buffer
		};
		VkDeviceSize sizes[11] = {
			m_fieldBufSize, m_fieldBufSize, m_opIndexBufSize,
			m_coeffCompBufSize, m_coeffCompBufSize,
			regionInfoSize, fluxBufSize, m_fusedPmlIdxBufSize,
			m_pmlTabBufSize, m_pmlTabBufSize, m_pmlTabBufSize
		};
		WriteDescriptorBuffers(m_device, m_fusedCurrDescSet, bufs, sizes, 11);
	}

	// --- Create compute pipelines ---
	// OPIDX_BITS: the compressed-operator index width is known once the operator
	// exists, so specialize it here -- the driver folds fetchOpIdx() to a single
	// path at pipeline creation. Shaders declaring no such constant ignore it.
	// Index widths are known once the operator and the PML tables exist, so
	// specialize both here; the driver folds fetchOpIdx()/fetchPmlIdx() to a
	// single path at pipeline creation. A shader that declares neither, or only
	// one, simply ignores the entries it does not use.
	const uint32_t idxBits[2] = {m_opIdxBits, m_pmlIdxBits};
	VkSpecializationMapEntry opIdxSpecEntry[2] = {
		{0, 0 * sizeof(uint32_t), sizeof(uint32_t)},
		{1, 1 * sizeof(uint32_t), sizeof(uint32_t)},
	};
	VkSpecializationInfo opIdxSpec{};
	opIdxSpec.mapEntryCount = 2;
	opIdxSpec.pMapEntries   = opIdxSpecEntry;
	opIdxSpec.dataSize      = sizeof(idxBits);
	opIdxSpec.pData         = idxBits;

	auto makePipeline = [&](VkShaderModule mod, VkPipelineLayout layout) -> VkPipeline
	{
		VkPipelineShaderStageCreateInfo stage{};
		stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
		stage.module = mod;
		stage.pName  = "main";
		stage.pSpecializationInfo = &opIdxSpec;
		VkComputePipelineCreateInfo ci{};
		ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		ci.stage  = stage;
		ci.layout = layout;
		VkPipeline pipe;
		VK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache, 1, &ci, nullptr, &pipe));
		return pipe;
	};

	VkShaderModule mod;
	mod = CreateShaderModule(gpu_spirv::fused_update_voltages_data,
	                         gpu_spirv::fused_update_voltages_size);
	m_fusedVoltPipeline = makePipeline(mod, m_fusedPipeLayout);
	vkDestroyShaderModule(m_device, mod, nullptr);

	mod = CreateShaderModule(gpu_spirv::fused_update_currents_data,
	                         gpu_spirv::fused_update_currents_size);
	m_fusedCurrPipeline = makePipeline(mod, m_fusedPipeLayout);
	vkDestroyShaderModule(m_device, mod, nullptr);

	m_hasFusedUPML = true;
	// The fused path owns its concatenated state from this point onward.
	// Release the equivalent per-region allocations retained for fallback.
	for (auto& u : m_gpuUPML)
	{
		DestroyGpuBuf(u.voltFlux); DestroyGpuBuf(u.currFlux);
		DestroyGpuBuf(u.pmlIdx);
		// The coefficient tables are shared with the fused path, not per-region.
	}
	cout << "Engine_Vulkan: fused Yee+UPML kernels created ("
	     << regions.size() << " regions, " << totalPmlCells << " PML cells)" << endl;
}

void Engine_Vulkan::CleanupFusedUPML()
{
	if (m_device == VK_NULL_HANDLE) return;
	DestroyGpuBuf(m_fusedVoltFlux);
	DestroyGpuBuf(m_fusedCurrFlux);
	DestroyGpuBuf(m_fusedPmlIdx);
	DestroyGpuBuf(m_fusedPmlRegionInfo);
	if (m_fusedVoltPipeline) { vkDestroyPipeline(m_device, m_fusedVoltPipeline, nullptr); m_fusedVoltPipeline = VK_NULL_HANDLE; }
	if (m_fusedCurrPipeline) { vkDestroyPipeline(m_device, m_fusedCurrPipeline, nullptr); m_fusedCurrPipeline = VK_NULL_HANDLE; }
	if (m_fusedPipeLayout)   { vkDestroyPipelineLayout(m_device, m_fusedPipeLayout, nullptr); m_fusedPipeLayout = VK_NULL_HANDLE; }
	if (m_fusedDescLayout)   { vkDestroyDescriptorSetLayout(m_device, m_fusedDescLayout, nullptr); m_fusedDescLayout = VK_NULL_HANDLE; }
	if (m_fusedDescPool)     { vkDestroyDescriptorPool(m_device, m_fusedDescPool, nullptr); m_fusedDescPool = VK_NULL_HANDLE; }
	m_fusedVoltDescSet = m_fusedCurrDescSet = VK_NULL_HANDLE;
	m_hasFusedUPML = false;
}

// ===========================================================================
// GPU Energy Reduction
// ===========================================================================

void Engine_Vulkan::SetupGPU_EnergyReduction()
{
	VkDeviceSize partialSize = 2 * ENERGY_NUM_WG * sizeof(float);
	VkMemoryPropertyFlags hostVisible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
	                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	// The reduction writes only 2 KiB. Keep that output directly CPU-visible
	// even when the large field buffers stay in device-local memory on a dGPU.
	CreateGpuBuf(m_energyPartialBuf, partialSize,
	             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostVisible);
	VK_CHECK(vkMapMemory(m_device, m_energyPartialBuf.memory, 0, partialSize, 0,
	                     (void**)&m_energyMapped));
	// Descriptor layout: 3 bindings (volt, curr, partials)
	auto makeLayout = [&](uint32_t nBindings) -> VkDescriptorSetLayout
	{
		std::vector<VkDescriptorSetLayoutBinding> bindings(nBindings);
		for (uint32_t i = 0; i < nBindings; i++)
			bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
		VkDescriptorSetLayoutCreateInfo ci{};
		ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		ci.bindingCount = nBindings;
		ci.pBindings    = bindings.data();
		VkDescriptorSetLayout layout;
		VK_CHECK(vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &layout));
		return layout;
	};
	m_energyDescLayout = makeLayout(3);

	// Pipeline layout: push constant = EnergyPC (8 bytes)
	VkPushConstantRange pcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(EnergyPC)};
	VkPipelineLayoutCreateInfo pli{};
	pli.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pli.setLayoutCount         = 1;
	pli.pSetLayouts            = &m_energyDescLayout;
	pli.pushConstantRangeCount = 1;
	pli.pPushConstantRanges    = &pcRange;
	VK_CHECK(vkCreatePipelineLayout(m_device, &pli, nullptr, &m_energyPipeLayout));

	// Descriptor pool + set
	VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
	VkDescriptorPoolCreateInfo dpi{};
	dpi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	dpi.maxSets       = 1;
	dpi.poolSizeCount = 1;
	dpi.pPoolSizes    = &poolSize;
	VK_CHECK(vkCreateDescriptorPool(m_device, &dpi, nullptr, &m_energyDescPool));

	VkDescriptorSetAllocateInfo ai{};
	ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	ai.descriptorPool     = m_energyDescPool;
	ai.descriptorSetCount = 1;
	ai.pSetLayouts        = &m_energyDescLayout;
	VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, &m_energyDescSet));

	// Write descriptors: volt, curr, partials
	VkBuffer      bufs[3] = {m_voltBuf, m_currBuf, m_energyPartialBuf.buffer};
	VkDeviceSize sizes[3] = {m_fieldBufSize, m_fieldBufSize, partialSize};
	WriteDescriptorBuffers(m_device, m_energyDescSet, bufs, sizes, 3);

	// Compute pipeline
	VkShaderModule mod = CreateShaderModule(gpu_spirv::reduce_energy_data,
	                                        gpu_spirv::reduce_energy_size);
	VkPipelineShaderStageCreateInfo stage{};
	stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	stage.module = mod;
	stage.pName  = "main";

	VkComputePipelineCreateInfo ci{};
	ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	ci.stage  = stage;
	ci.layout = m_energyPipeLayout;
	VK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache, 1, &ci, nullptr, &m_energyPipeline));
	vkDestroyShaderModule(m_device, mod, nullptr);

}

void Engine_Vulkan::SetupGPU_FieldValidation()
{
	const VkDeviceSize resultSize = (m_validateStages ? 6u : 1u) * 16u * sizeof(uint32_t);
	const VkMemoryPropertyFlags hostVisible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
	                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	CreateGpuBuf(m_validateBadBuf, resultSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostVisible);
	VK_CHECK(vkMapMemory(m_device, m_validateBadBuf.memory, 0, resultSize, 0,
	                     (void**)&m_validateBadMapped));

	VkDescriptorSetLayoutBinding bindings[6] = {};
	for (uint32_t i = 0; i < 6; ++i)
		bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = 6;
	layoutInfo.pBindings = bindings;
	VK_CHECK(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_validateDescLayout));

	VkPushConstantRange pcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ValidatePC)};
	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &m_validateDescLayout;
	pipelineLayoutInfo.pushConstantRangeCount = 1;
	pipelineLayoutInfo.pPushConstantRanges = &pcRange;
	VK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_validatePipeLayout));

	VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6};
	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.maxSets = 1;
	poolInfo.poolSizeCount = 1;
	poolInfo.pPoolSizes = &poolSize;
	VK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_validateDescPool));

	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = m_validateDescPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &m_validateDescLayout;
	VK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &m_validateDescSet));

	VkBuffer buffers[6] = {m_voltBuf, m_currBuf, m_validateBadBuf.buffer,
	                       m_opIndexBuf, m_vvCompBuf, m_viCompBuf};
	VkDeviceSize sizes[6] = {m_fieldBufSize, m_fieldBufSize, resultSize,
	                         m_opIndexBufSize, m_coeffCompBufSize, m_coeffCompBufSize};
	WriteDescriptorBuffers(m_device, m_validateDescSet, buffers, sizes, 6);

	// OPIDX_BITS: the compressed-operator index width is known once the operator
	// exists, so specialize it here -- the driver folds fetchOpIdx() to a single
	// path at pipeline creation. Shaders declaring no such constant ignore it.
	// Index widths are known once the operator and the PML tables exist, so
	// specialize both here; the driver folds fetchOpIdx()/fetchPmlIdx() to a
	// single path at pipeline creation. A shader that declares neither, or only
	// one, simply ignores the entries it does not use.
	const uint32_t idxBits[2] = {m_opIdxBits, m_pmlIdxBits};
	VkSpecializationMapEntry opIdxSpecEntry[2] = {
		{0, 0 * sizeof(uint32_t), sizeof(uint32_t)},
		{1, 1 * sizeof(uint32_t), sizeof(uint32_t)},
	};
	VkSpecializationInfo opIdxSpec{};
	opIdxSpec.mapEntryCount = 2;
	opIdxSpec.pMapEntries   = opIdxSpecEntry;
	opIdxSpec.dataSize      = sizeof(idxBits);
	opIdxSpec.pData         = idxBits;

	VkShaderModule module = CreateShaderModule(gpu_spirv::validate_fields_data,
	                                           gpu_spirv::validate_fields_size);
	VkPipelineShaderStageCreateInfo stage{};
	stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	stage.module = module;
	stage.pName = "main";
	stage.pSpecializationInfo = &opIdxSpec;

	VkComputePipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipelineInfo.stage = stage;
	pipelineInfo.layout = m_validatePipeLayout;
	VK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache, 1, &pipelineInfo,
	                                  nullptr, &m_validatePipeline));
	vkDestroyShaderModule(m_device, module, nullptr);
}

void Engine_Vulkan::CleanupGPU_FieldValidation()
{
	if (m_device == VK_NULL_HANDLE) return;
	if (m_validateBadMapped)
	{
		vkUnmapMemory(m_device, m_validateBadBuf.memory);
		m_validateBadMapped = nullptr;
	}
	DestroyGpuBuf(m_validateBadBuf);
	if (m_validatePipeline)   { vkDestroyPipeline(m_device, m_validatePipeline, nullptr); m_validatePipeline = VK_NULL_HANDLE; }
	if (m_validatePipeLayout) { vkDestroyPipelineLayout(m_device, m_validatePipeLayout, nullptr); m_validatePipeLayout = VK_NULL_HANDLE; }
	if (m_validateDescLayout) { vkDestroyDescriptorSetLayout(m_device, m_validateDescLayout, nullptr); m_validateDescLayout = VK_NULL_HANDLE; }
	if (m_validateDescPool)   { vkDestroyDescriptorPool(m_device, m_validateDescPool, nullptr); m_validateDescPool = VK_NULL_HANDLE; }
	m_validateDescSet = VK_NULL_HANDLE;
}

void Engine_Vulkan::CleanupGPU_EnergyReduction()
{
	if (m_device == VK_NULL_HANDLE) return;
	if (m_energyMapped)
	{
		vkUnmapMemory(m_device, m_energyPartialBuf.memory);
		m_energyMapped = nullptr;
	}
	DestroyGpuBuf(m_energyPartialBuf);

	if (m_energyPipeline)   { vkDestroyPipeline(m_device, m_energyPipeline, nullptr); m_energyPipeline = VK_NULL_HANDLE; }
	if (m_energyPipeLayout) { vkDestroyPipelineLayout(m_device, m_energyPipeLayout, nullptr); m_energyPipeLayout = VK_NULL_HANDLE; }
	if (m_energyDescLayout) { vkDestroyDescriptorSetLayout(m_device, m_energyDescLayout, nullptr); m_energyDescLayout = VK_NULL_HANDLE; }
	if (m_energyDescPool)   { vkDestroyDescriptorPool(m_device, m_energyDescPool, nullptr); m_energyDescPool = VK_NULL_HANDLE; }
	m_energyDescSet = VK_NULL_HANDLE; // freed with pool
}

double Engine_Vulkan::CalcFastEnergyGPU() const
{
	// Ensure GPU compute is done
	DrainGPU();

	EnergyPC pc = {m_fieldN, ENERGY_NUM_WG};

	// Dispatch energy reduction
	RunSingleCommand([&](VkCommandBuffer cmd)
	{
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_energyPipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		                        m_energyPipeLayout, 0, 1, &m_energyDescSet, 0, nullptr);
		vkCmdPushConstants(cmd, m_energyPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
		                   0, sizeof(EnergyPC), &pc);
		vkCmdDispatch(cmd, ENERGY_NUM_WG, 1, 1);

		VkMemoryBarrier hostBarrier{};
		hostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		hostBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
			0, 1, &hostBarrier, 0, nullptr, 0, nullptr);
	});

	// Read back the partial sums
	VkDeviceSize partialSize = 2 * ENERGY_NUM_WG * sizeof(float);
	float partials[2 * ENERGY_NUM_WG];

	memcpy(partials, m_energyMapped, partialSize);

	// CPU-side reduction of the partial sums (256 E + 256 H floats)
	double E_energy = 0.0, H_energy = 0.0;
	for (uint32_t i = 0; i < ENERGY_NUM_WG; ++i)
	{
		E_energy += (double)partials[i];
		H_energy += (double)partials[ENERGY_NUM_WG + i];
	}

	return EPS0 * E_energy + MUE0 * H_energy;
}

#endif // WITH_GPU
