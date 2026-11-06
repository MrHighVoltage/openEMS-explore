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

#include "openems.h"
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <thread>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <memory>
#include "tools/signal.h"
#include "tools/useful.h"
#include "FDTD/operator_cylinder.h"
#include "FDTD/operator_cylindermultigrid.h"
#include "FDTD/engine_multithread.h"
#include "FDTD/operator_multithread.h"
#include "FDTD/extensions/operator_ext_excitation.h"
#include "FDTD/extensions/operator_ext_tfsf.h"
#include "FDTD/extensions/operator_ext_mur_abc.h"
#include "FDTD/extensions/operator_ext_upml.h"
#include "FDTD/extensions/operator_ext_lorentzmaterial.h"
#include "FDTD/extensions/operator_ext_debyematerial.h"
#include "FDTD/extensions/operator_ext_lumpedRLC.h"
#include "FDTD/extensions/operator_ext_conductingsheet.h"
#include "FDTD/extensions/operator_ext_steadystate.h"
#include "FDTD/extensions/operator_ext_absorbing_bc.h"
#include "FDTD/extensions/engine_ext_steadystate.h"
#include "FDTD/engine_interface_fdtd.h"
#include "FDTD/engine_interface_sse_fdtd.h"
#include "FDTD/engine_benchmark.h"
#include "tools/cpu_topology.h"
#if OPENEMS_ENABLE_AVX2
#include "FDTD/operator_avx2.h"
#include "FDTD/operator_avx2_multithread.h"
#include "FDTD/engine_avx2.h"
#include "FDTD/engine_avx2_multithread.h"
#include "FDTD/engine_interface_avx2_fdtd.h"
#endif
#ifdef WITH_GPU
#include "FDTD/operator_vulkan.h"
#include "FDTD/engine_vulkan.h"
#endif
#include "Common/processvoltage.h"
#include "Common/processcurrent.h"
#include "Common/processfieldprobe.h"
#include "Common/processmodematch.h"
#include "Common/processfields_td.h"
#include "Common/processfields_fd.h"
#include "Common/processfields_sar.h"
#include <hdf5.h>            // only for H5get_libversion()
#include <vtkVersion.h>

//external libs
#include "tinyxml.h"
#include "ContinuousStructure.h"
#include "CSPropProbeBox.h"
#include "CSPrimBox.h"
#include "CSPropDumpBox.h"

using namespace std;

double CalcDiffTime(timeval t1, timeval t2)
{
	double s_diff = t1.tv_sec - t2.tv_sec;
	s_diff += (t1.tv_usec-t2.tv_usec)*1e-6;
	return s_diff;
}

namespace
{
//! The engine in replay of timestep \a ts of its last batch, for as long as this lives (in-batch sampling).
struct BatchReplayScope
{
	Engine* eng;
	BatchReplayScope(Engine* e, unsigned int ts) : eng(e) {eng->BeginBatchReplay(ts);}
	~BatchReplayScope() {eng->EndBatchReplay();}
	BatchReplayScope(const BatchReplayScope&) = delete;
	BatchReplayScope& operator=(const BatchReplayScope&) = delete;
};
}

openEMS::openEMS()
{
	setlocale(LC_NUMERIC, "en_US.UTF-8");
	FDTD_Op=NULL;
	FDTD_Eng=NULL;
	Eng_Ext_SSD=NULL;
	m_CSX=NULL;
	PA=NULL;
	Enable_Dumps = true;
	DebugMat = false;
	DebugOp = false;
	m_debugCSX = false;
	m_debugBox = m_debugPEC = m_no_simulation = false;
	m_DumpStats = false;
	m_exactEndCriteria = false;

#if OPENEMS_ENABLE_AVX2
	m_engine = EngineType_AVX2_Multithreaded; //default engine type
#else
	m_engine = EngineType_Multithreaded; // default portable engine type
#endif
	m_engineGiven = false;
	m_engine_numThreads = 0;
	m_numThreadsGiven = false;
	m_gpuTBMBOption = 0;
	m_benchmark = false;
	m_benchmarkTS = 0;

	m_Abort = false;
	m_Exc = 0;

	Reset();

	collectCommandLineArguments();
}

openEMS::~openEMS()
{
	Reset();
}

void openEMS::Reset()
{
	if (PA) PA->DeleteAll();
	delete PA;
	PA=0;
	delete FDTD_Eng;
	FDTD_Eng=0;
	delete FDTD_Op;
	FDTD_Op=0;
	delete m_CSX;
	m_CSX=0;
	delete m_Exc;
	m_Exc=0;
Eng_Ext_SSD=0; // non-owning observer; deleted by the engine (m_Eng_exts) in 'delete FDTD_Eng' above

	CylinderCoords = false;
	m_CC_MultiGrid.clear();
	m_CellConstantMaterial=false;
	endCrit = 1e-6;
	m_OverSampling = 4;

	m_TS_method=3;
	m_TS=0;
	m_TS_fac=1.0;
	m_maxTime=0.0;

	for (int n=0;n<6;++n)
	{
		m_BC_type[n]  = 0;
		m_PML_size[n] = 8;
		m_Mur_v_ph[n] = 0;
	}
}

void openEMS::collectCommandLineArguments()
{
	// see commands in tools/global.h
	g_settings.clearOptionDesc();

	// register our supported options to g_settings
	g_settings.appendOptionDesc(optionDesc());
	g_settings.appendOptionDesc(g_settings.optionDesc());
}

OptionDesc
openEMS::optionDesc()
{
	OptionDesc optdesc("Options");

	optdesc.addBoolSwitch("help,h",
		[this](bool val)
		{
			if (!val) return;
			showUsage();
			std::exit(0);
		},
		"Show this help message and exit"
	);

	optdesc.addBoolSwitch("disable-dumps",
		[this](bool val)
		{
			if (!val) return;
			cout << "openEMS - force-disabling all field dumps" << endl;
			SetEnableDumps(!val);
		},
		"Disable all field dumps for faster simulation"
	);

	optdesc.addBoolSwitch("debug-material",
		[this](bool val)
		{
			if (!val) return;
			cout << "openEMS - dumping material to 'material_dump.vtk'" << endl;
			DebugMaterial();
		},
		"Dump material distribution to a vtk file for debugging"
	);

	optdesc.addBoolSwitch("debug-PEC",
		[this](bool val)
		{
			if (!val) return;
			cout << "openEMS - dumping PEC info to 'PEC_dump.vtk'" << endl;
			DebugPEC();
		},
		"Dump metal distribution to a vtk file for debugging"
	);

	optdesc.addBoolSwitch("debug-operator",
		[this](bool val)
		{
			if (!val) return;
			cout << "openEMS - dumping operator to 'operator_dump.vtk'" << endl;
			DebugOperator();
		},
		"Dump operator to vtk file for debugging"
	);

	optdesc.addBoolSwitch("debug-boxes",
		[this](bool val)
		{
			if (!val) return;
			cout << "openEMS - dumping boxes to 'box_dump*.vtk'" << endl;
			DebugBox();
		},
		"Dump e.g. probe boxes to vtk file for debugging"
	);

	optdesc.addBoolSwitch("debug-CSX",
		[this](bool val)
		{
			if (!val) return;
			cout << "openEMS - dumping CSX geometry to 'debugCSX.xml'" << endl;
			DebugCSX();
		},
		"Write CSX geometry file to debugCSX.xml"
	);

	optdesc.addStringOption("engine", "fastest",
		[this](const std::string& val)
		{
			m_engineGiven = (val != "fastest");
			if (val == "fastest")
			{
				// default, don't show console output
#ifdef WITH_GPU
				m_engine = EngineType_GPU;
#elif OPENEMS_ENABLE_AVX2
				m_engine = EngineType_AVX2_Multithreaded;
#else
				m_engine = EngineType_Multithreaded;
#endif
			}
			else if (val == "basic")
			{
				cout << "openEMS - enabled basic engine" << endl;
				m_engine = EngineType_Basic;
			}
			else if (val == "sse")
			{
				cout << "openEMS - enabled sse engine" << endl;
				m_engine = EngineType_SSE;
			}
			else if (val == "sse-compressed")
			{
				cout << "openEMS - enabled compressed sse engine" << endl;
				m_engine = EngineType_SSE_Compressed;
			}
			else if (val == "multithreaded")
			{
				cout << "openEMS - enabled multithreading" << endl;
				m_engine = EngineType_Multithreaded;
			}
#ifdef WITH_GPU
			else if (val == "gpu")
			{
				cout << "openEMS - enabled Vulkan GPU engine" << endl;
				m_engine = EngineType_GPU;
			}
#endif
			else if (val == "avx2")
			{
#if OPENEMS_ENABLE_AVX2
				cout << "openEMS - enabled AVX2+FMA engine" << endl;
				m_engine = EngineType_AVX2;
#else
				cerr << "openEMS - AVX2+FMA engine is unavailable on this platform; using multithreaded engine" << endl;
				m_engine = EngineType_Multithreaded;
#endif
			}
			else if (val == "avx2-multithreaded")
			{
#if OPENEMS_ENABLE_AVX2
				cout << "openEMS - enabled AVX2+FMA multithreaded engine" << endl;
				m_engine = EngineType_AVX2_Multithreaded;
#else
				cerr << "openEMS - AVX2+FMA multithreaded engine is unavailable on this platform; using multithreaded engine" << endl;
				m_engine = EngineType_Multithreaded;
#endif
			}
		},
		"Choose engine type \n\n"
		"  fastest: \tfastest available engine (default)\n"
		"  basic: \tbasic FDTD engine\n"
		"  sse: \tengine using SSE vector extensions\n"
		"  sse-compressed: \tengine using compressed "
		"operator + sse vector extensions\n"
#ifdef WITH_GPU
		"  gpu: \tVulkan GPU-accelerated engine\n"
#endif
		"  avx2: \tengine using AVX2+FMA vector extensions\n"
		"  avx2-multithreaded: \tengine using AVX2+FMA + multithreading\n"
		"  multithreaded: \tengine using compressed "
#ifdef MPI_SUPPORT
		"operator + sse vector extensions + MPI + multithreading\n"
#else
		"operator + sse vector extensions + multithreading\n"
#endif
	);

	// The default -1 tells "not given" from an explicit 0: the parser calls
	// back with the default when an option is absent.
	optdesc.addIntOption("numThreads", -1,
		[this](int val)
		{
			m_numThreadsGiven = (val >= 0);
			this->SetNumberOfThreads(std::max(val, 0));
			if (val > 0)
				cout << "openEMS - fixed number of threads: "
					 << m_engine_numThreads << endl;
		},
		"Force use n threads for the multithreaded engines; 0: find the "
		"fastest count at the start of the run (default)"
	);

	optdesc.addStringOption("cpu-cores", "",
		[this](const std::string& val)
		{
			CoreSet cores;
			if (val.empty() || CoreSet::Parse(val, cores))
				m_cpuCoresOption = val;
			else
				cerr << "openEMS - ignoring --cpu-cores=" << val << " (expected all, physical, performance, "
				        "performance+smt, efficiency or efficiency+smt)" << endl;
		},
		"Cores the worker threads of the multithreaded engines run on: all "
		"(default); physical, one thread per core; performance or efficiency, "
		"one kind of core of a hybrid CPU, one thread per core, and with +smt "
		"every thread of those cores"
	);

	// Temporal blocking. Each replaces the environment variable of the same
	// name (OPENEMS_AVX2_TEMPORAL_BLOCK, ...); see ApplyTemporalBlockingOptions().
	const auto readSchedule = [](const char* option, const std::string& val) -> std::string
	{
		TemporalBlockingRequest req;
		if (val.empty() || TemporalBlockingRequest::Parse(val, req))
			return val;
		cerr << "openEMS - ignoring --" << option << "=" << val
		     << " (expected auto, off, <k> or <k>:<W>)" << endl;
		return "";
	};
	optdesc.addStringOption("avx2-temporal-block", "",
		[this, readSchedule](const std::string& val)
		{
			m_avx2TBOption = readSchedule("avx2-temporal-block", val);
		},
		"Trapezoidal temporal blocking in the avx2-multithreaded engine: off "
		"(default), <k> or <k>:<W> -- blocks of k timesteps over tiles of W "
		"x-lines (default W: half the L3 cache)"
	);
#ifdef WITH_GPU
	optdesc.addStringOption("gpu-temporal-block", "",
		[this, readSchedule](const std::string& val)
		{
			m_gpuTBOption = readSchedule("gpu-temporal-block", val);
		},
		"Trapezoidal temporal blocking in the gpu engine: auto (default), off, "
		"<k> or <k>:<W> -- blocks of k timesteps over tiles of W x-lines "
		"(default W: from the tile target)"
	);

	optdesc.addIntOption("gpu-temporal-block-mb", 0,
		[this](int val)
		{
			m_gpuTBMBOption = (val > 0) ? (unsigned int)val : 0;
		},
		"MB of fields and coefficients one GPU tile should hold (default: 64 on "
		"a discrete card, 80 % of the CPU's L3 cache on an integrated one)"
	);
#endif

#ifdef WITH_GPU
	optdesc.addIntOption("gpu-device", -1,
		[](int val)
		{
			g_settings.SetGPUDevice(val);
		},
		"GPU device of --engine=gpu, a number of the device list printed at "
		"--verbose=1 (default: OPENEMS_GPU_INDEX, else the first discrete GPU)"
	);

	// Accepted so that existing scripts keep working.
	optdesc.addBoolSwitch("gpu-no-rebar-fields",
		[](bool) {},
		"No effect: the GPU field buffers are always device-local"
	);

	optdesc.addBoolSwitch("gpu-profile",
		[this](bool val)
		{
			if (!val) return;
			Engine_Vulkan::SetEnableProfiling(true);
			cout << "openEMS - enabled Vulkan GPU profiling (timestamp queries)" << endl;
		},
		"Enable Vulkan GPU batch profiling (GPU timestamp + fence wait summary)"
	);
#endif

	optdesc.addBoolSwitch("benchmark",
		[this](bool val)
		{
			m_benchmark = val;
			if (val)
				cout << "openEMS - benchmark mode: measuring the engine's settings "
				        "on this model instead of simulating it" << endl;
		},
		"Measure the settings the engine runs this model fastest with -- "
		"threads, cores, temporal-blocking schedule -- and print the options "
		"for them, instead of simulating it. Writes nothing into the "
		"simulation directory."
	);

	optdesc.addIntOption("benchmark-timesteps", 0,
		[this](int val)
		{
			m_benchmarkTS = (val > 0) ? (unsigned int)val : 0;
		},
		"Timesteps per --benchmark run of a schedule (default: about 2 s of "
		"the fastest thread setting; on a GPU 3 s of the flat sweep)"
	);

	optdesc.addBoolSwitch("no-simulation",
		[this](bool val)
		{
			if (!val) return;
			cout << "openEMS - disabling simulation => preprocessing only" << endl;
			m_no_simulation = true;
		},
		"only run preprocessing; do not simulate"
	);

	optdesc.addBoolSwitch("dump-statistics",
		[this](bool val)
		{
			if (!val) return;
			cout << "openEMS - dump simulation statistics to '"
				 << OPENEMS_RUN_STAT_FILE << "' and '"
				 << OPENEMS_STAT_FILE << "'" << endl;
			m_DumpStats = true;
		},
		"dump simulation statistics to '" OPENEMS_RUN_STAT_FILE
		"' and '" OPENEMS_STAT_FILE "'"
	);

	optdesc.addBoolSwitch("exact-endcriteria",
		[this](bool val)
		{
			if (!val) return;
			cout << "openEMS - evaluating the end criteria on a fixed timestep schedule "
			     << "for a machine-independent stopping point (reduces speed)" << endl;
			m_exactEndCriteria = true;
		},
		"evaluate the energy end criteria every Nyquist timesteps for the whole "
		"run, not only during its first minute and then every few seconds of "
		"wall-clock time, so the run stops at the same timestep regardless of "
		"machine speed/load; costs performance, useful mainly for engine/code "
		"verification"
	);

	return optdesc;
}

void openEMS::showUsage()
{
	cout << " Usage: openEMS <FDTD_XML_FILE> [<options>...]" << endl;

	cout << " ";
	g_settings.showOptionUsage(cout);
}

// used by Python binding when running as a shared library
void openEMS::SetLibraryArguments(std::vector<std::string> allOptions)
{
	collectCommandLineArguments();
	g_settings.parseLibraryArguments(allOptions);
}

std::string openEMS::EngineOptionValue() const
{
	switch (m_engine)
	{
	case EngineType_Basic:               return "basic";
	case EngineType_SSE:                 return "sse";
	case EngineType_SSE_Compressed:      return "sse-compressed";
	case EngineType_Multithreaded:       return "multithreaded";
	case EngineType_AVX2:                return "avx2";
	case EngineType_AVX2_Multithreaded:  return "avx2-multithreaded";
#ifdef MPI_SUPPORT
	case EngineType_MPI:                 return "MPI";
#endif
#ifdef WITH_GPU
	case EngineType_GPU:                 return "gpu";
#endif
	}
	return "fastest";
}

void openEMS::ApplyTemporalBlockingOptions()
{
	// The engines read the environment in Init(); an option replaces its
	// variable, and a tile target given alone keeps the schedule the
	// environment or the default asked for.
	const auto overrides = [](const char* option, const char* env)
	{
		if (std::getenv(env))
			cout << "openEMS - --" << option << " overrides " << env << endl;
	};
#ifdef WITH_GPU
	if (!m_gpuTBOption.empty() || m_gpuTBMBOption)
	{
		if (dynamic_cast<Engine_Vulkan*>(FDTD_Eng))
		{
			TemporalBlockingRequest req = FDTD_Eng->GetTemporalBlocking();
			if (!m_gpuTBOption.empty())
			{
				overrides("gpu-temporal-block", "OPENEMS_GPU_TEMPORAL_BLOCK");
				TemporalBlockingRequest::Parse(m_gpuTBOption, req);
			}
			if (m_gpuTBMBOption)
			{
				overrides("gpu-temporal-block-mb", "OPENEMS_GPU_TEMPORAL_BLOCK_MB");
				req.targetMB = m_gpuTBMBOption;
			}
			FDTD_Eng->SetTemporalBlocking(req);
		}
		else
			cerr << "openEMS - --gpu-temporal-block ignored: the engine is not the gpu engine" << endl;
	}
#endif
#if OPENEMS_ENABLE_AVX2
	if (!m_avx2TBOption.empty())
	{
		if (dynamic_cast<Engine_AVX2_Multithread*>(FDTD_Eng))
		{
			TemporalBlockingRequest req = FDTD_Eng->GetTemporalBlocking();
			overrides("avx2-temporal-block", "OPENEMS_AVX2_TEMPORAL_BLOCK");
			TemporalBlockingRequest::Parse(m_avx2TBOption, req);
			FDTD_Eng->SetTemporalBlocking(req);
		}
		else
			cerr << "openEMS - --avx2-temporal-block ignored: the engine is not avx2-multithreaded" << endl;
	}
#else
	if (!m_avx2TBOption.empty())
		cerr << "openEMS - --avx2-temporal-block ignored: built without the AVX2 engines" << endl;
#endif
}

std::string openEMS::TemporalBlockingOptionName() const
{
#ifdef WITH_GPU
	if (dynamic_cast<Engine_Vulkan*>(FDTD_Eng))
		return "gpu-temporal-block";
#endif
#if OPENEMS_ENABLE_AVX2
	if (dynamic_cast<Engine_AVX2_Multithread*>(FDTD_Eng))
		return "avx2-temporal-block";
#endif
	return "";
}

void openEMS::WarnUntunedSettings() const
{
	// The options --benchmark prints, where the run leaves them to defaults.
	std::vector<std::string> unset;
	if (FDTD_Eng->GetNumThreads() > 0)
	{
		if (!m_numThreadsGiven)
			unset.push_back("--numThreads");
		if (m_cpuCoresOption.empty() && CpuTopology::Get().DistinctSets().size() > 1)
			unset.push_back("--cpu-cores");
	}
	const std::string tb = TemporalBlockingOptionName();
	if (!tb.empty())
	{
		std::string env = "OPENEMS_" + tb;
		std::transform(env.begin(), env.end(), env.begin(), [](char c) {return c == '-' ? '_' : (char)std::toupper(c);});
		const std::string& option = (tb == "gpu-temporal-block") ? m_gpuTBOption : m_avx2TBOption;
		if (option.empty() && !std::getenv(env.c_str()))
			unset.push_back("--" + tb);
	}
	if (unset.empty())
		return;
	cerr << "openEMS - Warning: running on default performance settings, not on ones tuned for this model and machine (";
	for (size_t i = 0; i < unset.size(); ++i)
		cerr << (i ? ", " : "") << unset[i];
	cerr << " not set)." << endl
	     << "\tMore speed may be possible: --benchmark measures the settings on this model and prints the options to use." << endl;
}

void openEMS::SetNumberOfThreads(int val)
{
	if ((val<0) || (val>(int)std::thread::hardware_concurrency()))
		val = std::thread::hardware_concurrency();
	m_engine_numThreads = val;
}

string openEMS::GetExtLibsInfo(string prefix)
{
	stringstream str;

	str << prefix << "Used external libraries:" << endl;
	str << prefix << "\t" << ContinuousStructure::GetInfoLine(true) << endl;

	// libhdf5
	unsigned int major, minor, release;
	if (H5get_libversion( &major, &minor, &release ) >= 0)
	{
		str << prefix << "\t" << "hdf5   -- Version: " << major << '.' << minor << '.' << release << endl;
		str << prefix << "\t" << "          compiled against: " H5_VERS_INFO << endl;
	}

	// tinyxml
	str << prefix << "\t" << "tinyxml -- compiled against: " << TIXML_MAJOR_VERSION << '.' << TIXML_MINOR_VERSION << '.' << TIXML_PATCH_VERSION << endl;

	// fparser
	str << prefix << "\t" << "fparser" << endl;

	//vtk
	str << prefix << "\t" << "vtk -- Version: " << vtkVersion::GetVTKMajorVersion() << "." << vtkVersion::GetVTKMinorVersion() << "." << vtkVersion::GetVTKBuildVersion() << endl;
	str << prefix << "\t" << "       compiled against: " << VTK_VERSION << endl;

	return str.str();
}

void openEMS::WelcomeScreen()
{
#if defined(_LP64) || defined(_WIN64)
	string bits = "64bit";
#else
	string bits = "32bit";
#endif

	cout << " ---------------------------------------------------------------------- " << endl;
	cout << " | openEMS " << bits << " -- version " << GIT_VERSION << endl;
	cout << " | (C) 2010-2025 Thorsten Liebig <thorsten.liebig@gmx.de>  GPL license"   << endl;
	cout << " ---------------------------------------------------------------------- " << endl;
	cout << openEMS::GetExtLibsInfo("\t") << endl;
}

bool openEMS::SetupBoundaryConditions()
{
	FDTD_Op->SetBoundaryCondition(m_BC_type); //operator only knows about PEC and PMC, everything else is defined by extensions (see below)

	/**************************** create all operator/engine extensions here !!!! **********************************/
	for (int n=0; n<6; ++n)
	{
		FDTD_Op->SetBCSize(n, 0);
		if (m_BC_type[n]==2) //Mur-ABC
		{
			FDTD_Op->SetBCSize(n, 1);
			Operator_Ext_Mur_ABC* op_ext_mur = new Operator_Ext_Mur_ABC(FDTD_Op);
			op_ext_mur->SetDirection(n/2,n%2);
			if (m_Mur_v_ph[n]>0)
				op_ext_mur->SetPhaseVelocity(m_Mur_v_ph[n]);
			FDTD_Op->AddExtension(op_ext_mur);
		}
		if (m_BC_type[n]==3)
			FDTD_Op->SetBCSize(n, m_PML_size[n]);
	}


	//create the upml
	Operator_Ext_UPML::Create_UPML(FDTD_Op, m_BC_type, m_PML_size, string());

	return true;
}

void openEMS::SetupAbsorbingSheets()
{
	vector<CSProperties*>	cs_props;
	cs_props = m_CSX->GetPropertyByType(CSProperties::ABSORBING_BC);

	for(size_t n = 0 ; n < cs_props.size() ; ++n)
	{
		CSPropAbsorbingBC * cABCprops = dynamic_cast<CSPropAbsorbingBC*>(cs_props.at(n));

		// Now start iterating through primitives
		vector<CSPrimitives*> cs_abc_prims = cABCprops->GetAllPrimitives();
		for (size_t sheetIdx = 0 ; sheetIdx < cs_abc_prims.size() ; ++sheetIdx)
		{

			// Attempt to initialize operator extension
			Operator_Ext_Absorbing_BC* op_ext_abc = new Operator_Ext_Absorbing_BC(FDTD_Op);

			CSPrimitives* cPrimitive = cs_abc_prims.at(sheetIdx);

			// Initialize all necessary parameters so the extension operator can be
			// built later on.
			if (op_ext_abc->SetInitParams(cPrimitive,cABCprops))
				// Finally, add the extension
				FDTD_Op->AddExtension(op_ext_abc);
			else
			{
				cerr << "openEMS::SetupAbsorbingSheets(): Warning: Absorbing sheet #" << sheetIdx << " setup failed.";
				delete op_ext_abc;
			}

		}

	}

}

Engine_Interface_FDTD* openEMS::NewEngineInterface(int multigridlevel)
{
	Operator_CylinderMultiGrid* op_cyl_mg = dynamic_cast<Operator_CylinderMultiGrid*>(FDTD_Op);
	while (op_cyl_mg && multigridlevel>0)
	{
		int mgl = op_cyl_mg->GetMultiGridLevel();
		if (mgl==multigridlevel)
		{
			if (g_settings.GetVerboseLevel()>0)
				cout << __func__ << ": Operator with requested multi-grid level found." << endl;
			return new Engine_Interface_SSE_FDTD(op_cyl_mg);
		}
		Operator_Cylinder* op_cyl_inner = op_cyl_mg->GetInnerOperator();
		op_cyl_mg = dynamic_cast<Operator_CylinderMultiGrid*>(op_cyl_inner);
		if (op_cyl_mg==NULL) //inner most operator reached
		{
			if (g_settings.GetVerboseLevel()>0)
				cout << __func__ << ": Operator with highest multi-grid level chosen." << endl;
			return new Engine_Interface_SSE_FDTD(op_cyl_inner);
		}
		// try next level
	}
	Operator_Cylinder* op_cyl = dynamic_cast<Operator_Cylinder*>(FDTD_Op);
	if (op_cyl)
		return new Engine_Interface_SSE_FDTD(op_cyl);
#if OPENEMS_ENABLE_AVX2
	Operator_AVX2* op_avx2 = dynamic_cast<Operator_AVX2*>(FDTD_Op);
	if (op_avx2)
		return new Engine_Interface_AVX2_FDTD(op_avx2);
#endif
	Operator_sse* op_sse = dynamic_cast<Operator_sse*>(FDTD_Op);
	if (op_sse)
		return new Engine_Interface_SSE_FDTD(op_sse);
	return new Engine_Interface_FDTD(FDTD_Op);
}

void openEMS::SetVerboseLevel(int level)
{
    g_settings.SetVerboseLevel(level);
}

bool openEMS::SetupProcessing()
{
	//*************** setup processing ************//
	if (g_settings.GetVerboseLevel()>0)
		cout << "Setting up processing..." << endl;

	unsigned int Nyquist = FDTD_Op->GetExcitationSignal()->GetNyquistNum();
	PA = new ProcessingArray(Nyquist);

	double start[3];
	double stop[3];
	bool l_MultiBox = false;
	vector<CSProperties*> Probes = m_CSX->GetPropertyByType(CSProperties::PROBEBOX);
	for (size_t i=0; i<Probes.size(); ++i)
	{
		CSPropProbeBox* pb = Probes.at(i)->ToProbeBox();
		if (!pb)
			continue;
		//check whether one or more probe boxes are defined
		l_MultiBox =  (pb->GetQtyPrimitives()>1);

		for (size_t nb=0; nb<pb->GetQtyPrimitives(); ++nb)
		{
			CSPrimitives* prim = pb->GetPrimitive(nb);
			if (prim!=NULL)
			{
				double bnd[6] = {0,0,0,0,0,0};
				prim->GetBoundBox(bnd,true);
				start[0]= bnd[0];
				start[1]=bnd[2];
				start[2]=bnd[4];
				stop[0] = bnd[1];
				stop[1] =bnd[3];
				stop[2] =bnd[5];

				ProcessIntegral* proc = NULL;
				if (pb->GetProbeType()==0)
				{
					CSPrimBox* box = prim->ToBox();
					if (!(box) || box->GetDimension()!=1)
					{
						cerr << "openEMS::SetupProcessing: Error: Probe primitive type or dimension not suitable ... skipping probe " << pb->GetName() << endl;
						continue;
					}
					// use the direction and coordinates of the box
					for (int n=0;n<3;++n)
					{
						start[n] = box->GetCoord(2*n);
						stop[n]  = box->GetCoord(2*n+1);
					}
					ProcessVoltage* procVolt = new ProcessVoltage(NewEngineInterface());
					proc=procVolt;
				}
				else if (pb->GetProbeType()==1)
				{
					ProcessCurrent* procCurr = new ProcessCurrent(NewEngineInterface());
					proc=procCurr;
				}
				else if (pb->GetProbeType()==2)
					proc = new ProcessFieldProbe(NewEngineInterface(),0);
				else if (pb->GetProbeType()==3)
					proc = new ProcessFieldProbe(NewEngineInterface(),1);
				else if ((pb->GetProbeType()==10) || (pb->GetProbeType()==11))
				{
					ProcessModeMatch* pmm = new ProcessModeMatch(NewEngineInterface());
					pmm->SetFieldType(pb->GetProbeType()-10);

					if (!pb->GetModeFile().empty())
						pmm->SetWeightFile(pb->GetModeFile());
					else
					{
						pmm->SetWeightFunction(0, pb->GetModeFunction(0));
						pmm->SetWeightFunction(1, pb->GetModeFunction(1));
						pmm->SetWeightFunction(2, pb->GetModeFunction(2));
					}
					pmm->SetWeightOrigin(pb->GetModeOrigin(0), pb->GetModeOrigin(1), pb->GetModeOrigin(2));
					proc = pmm;
				}
				else
				{
					cerr << "openEMS::SetupFDTD: Warning: Probe type " << pb->GetProbeType() << " of property '" << pb->GetName() << "' is unknown..." << endl;
					continue;
				}
				if (CylinderCoords)
					proc->SetMeshType(Processing::CYLINDRICAL_MESH);
				if ((pb->GetProbeType()==1) || (pb->GetProbeType()==3))
				{
					proc->SetDualTime(true);
					proc->SetDualMesh(true);
				}
				if (pb->GetProbeType()==11)
					proc->SetDualTime(true);
				proc->SetProcessInterval(Nyquist/m_OverSampling);
				if (pb->GetStartTime()>0 || pb->GetStopTime()>0)
					proc->SetProcessStartStopTime(pb->GetStartTime(), pb->GetStopTime());
				proc->AddFrequency(pb->GetFDSamples());
				proc->GetNormalDir(pb->GetNormalDir());
				if (l_MultiBox==false)
					proc->SetName(pb->GetName());
				else
					proc->SetName(pb->GetName(),nb);
				proc->DefineStartStopCoord(start,stop);
				if (g_settings.showProbeDiscretization())
					proc->ShowSnappedCoords();
				proc->SetWeight(pb->GetWeighting());
				PA->AddProcessing(proc);
				prim->SetPrimitiveUsed(true);
			}
		}
	}

	vector<CSProperties*> DumpProps = m_CSX->GetPropertyByType(CSProperties::DUMPBOX);
	for (size_t i=0; i<DumpProps.size(); ++i)
	{
		ProcessFields* ProcField=NULL;

		//check whether one or more probe boxes are defined
		l_MultiBox =  (DumpProps.at(i)->GetQtyPrimitives()>1);

		for (size_t nb=0; nb<DumpProps.at(i)->GetQtyPrimitives(); ++nb)
		{

			CSPrimitives* prim = DumpProps.at(i)->GetPrimitive(nb);
			if (prim!=NULL)
			{
				double bnd[6] = {0,0,0,0,0,0};
				prim->GetBoundBox(bnd,true);
				start[0]= bnd[0];
				start[1]=bnd[2];
				start[2]=bnd[4];
				stop[0] = bnd[1];
				stop[1] =bnd[3];
				stop[2] =bnd[5];
				CSPropDumpBox* db = DumpProps.at(i)->ToDumpBox();
				if (db)
				{
					if ((db->GetDumpType()>=0) && (db->GetDumpType()<=5))
						ProcField = new ProcessFieldsTD(NewEngineInterface(db->GetMultiGridLevel()));
					else if ((db->GetDumpType()>=10) && (db->GetDumpType()<=15))
						ProcField = new ProcessFieldsFD(NewEngineInterface(db->GetMultiGridLevel()));
					else if ( ((db->GetDumpType()>=20) && (db->GetDumpType()<=22)) || (db->GetDumpType()==29) )
					{
						ProcessFieldsSAR* procSAR = new ProcessFieldsSAR(NewEngineInterface(db->GetMultiGridLevel()));
						ProcField = procSAR;
						string method = db->GetAttributeValue("SAR_Method");
						if (!method.empty())
							procSAR->SetSARAveragingMethod(method);
						// use (center)-cell based conductivity only
						procSAR->SetUseCellConductivity(true);
					}
					else
						cerr << "openEMS::SetupFDTD: unknown dump box type... skipping!" << endl;
					if (ProcField)
					{
						ProcField->SetEnable(Enable_Dumps && !m_benchmark);
						ProcField->SetProcessInterval(Nyquist/m_OverSampling);
						if (db->GetStopTime()>0 || db->GetStartTime()>0)
							ProcField->SetProcessStartStopTime(db->GetStartTime(), db->GetStopTime());
						if ((db->GetDumpType()==1) || (db->GetDumpType()==11) || (db->GetDumpType()==5) || (db->GetDumpType()==15))
						{
							ProcField->SetDualTime(true);
							//make dualMesh the default mesh for h- and b-field dumps, maybe overwritten by interpolation type (node-interpolation)
							ProcField->SetDualMesh(true);
						}
						if (db->GetDumpType()>=10)
						{
							ProcField->AddFrequency(db->GetFDSamples());
							ProcField->SetDumpType((ProcessFields::DumpType)(db->GetDumpType()-10));
						}
						else
							ProcField->SetDumpType((ProcessFields::DumpType)db->GetDumpType());

						if (db->GetDumpType()==20)
							ProcField->SetDumpType(ProcessFields::SAR_LOCAL_DUMP);
						if (db->GetDumpType()==21)
							ProcField->SetDumpType(ProcessFields::SAR_1G_DUMP);
						if (db->GetDumpType()==22)
							ProcField->SetDumpType(ProcessFields::SAR_10G_DUMP);
						if (db->GetDumpType()==29)
							ProcField->SetDumpType(ProcessFields::SAR_RAW_DATA);

						//SetupMaterialStorages() has previewed storage needs... refresh here to prevent cleanup!!!
						if ( ProcField->NeedPermittivity() && Enable_Dumps)
							FDTD_Op->SetMaterialStoreFlags(0,true);
						if ( ProcField->NeedConductivity() && Enable_Dumps)
							FDTD_Op->SetMaterialStoreFlags(1,true);
						if ( ProcField->NeedPermeability() && Enable_Dumps)
							FDTD_Op->SetMaterialStoreFlags(2,true);

						ProcField->SetDumpMode((Engine_Interface_Base::InterpolationType)db->GetDumpMode());
						ProcField->SetFileType((ProcessFields::FileType)db->GetFileType());
						if (CylinderCoords)
							ProcField->SetMeshType(Processing::CYLINDRICAL_MESH);
						if (db->GetSubSampling())
							for (int n=0; n<3; ++n)
								ProcField->SetSubSampling(db->GetSubSampling(n),n);
						if (db->GetOptResolution())
							for (int n=0; n<3; ++n)
								ProcField->SetOptResolution(db->GetOptResolution(n),n);

						if (l_MultiBox==false)
							ProcField->SetName(db->GetName());
						else
							ProcField->SetName(db->GetName(),nb);

						ProcField->SetFileName(ProcField->GetName());
						ProcField->DefineStartStopCoord(start,stop);
						if (g_settings.showProbeDiscretization())
							ProcField->ShowSnappedCoords();
						PA->AddProcessing(ProcField);
						prim->SetPrimitiveUsed(true);
					}
				}
			}
		}
	}

	return true;
}

bool openEMS::SetupMaterialStorages()
{
	vector<CSProperties*> DumpProps = m_CSX->GetPropertyByType(CSProperties::DUMPBOX);
	for (size_t i=0; i<DumpProps.size(); ++i)
	{
		CSPropDumpBox* db = DumpProps.at(i)->ToDumpBox();
		if (!db)
			continue;
		if (db->GetQtyPrimitives()==0)
			continue;
		//check for current density dump types
		if ( ((db->GetDumpType()==2) || (db->GetDumpType()==12) || // current density storage
			  (db->GetDumpType()==20) || (db->GetDumpType()==21) || (db->GetDumpType()==22)) && // SAR dump types
			  Enable_Dumps )
			FDTD_Op->SetMaterialStoreFlags(1,true); //tell operator to store kappa material data
		if ( ((db->GetDumpType()==4) || (db->GetDumpType()==14)) || Enable_Dumps) // electric flux density storage
			FDTD_Op->SetMaterialStoreFlags(0,true); //tell operator to store epsR material data
		if ( ((db->GetDumpType()==5) || (db->GetDumpType()==15)) || Enable_Dumps) // magnetic flux density storage
			FDTD_Op->SetMaterialStoreFlags(2,true); //tell operator to store mueR material data
	}
	return true;
}

void openEMS::SetupCylinderMultiGrid(std::string val)
{
	m_CC_MultiGrid.clear();
	m_CC_MultiGrid = SplitString2Double(val,',');
}

#if OPENEMS_ENABLE_AVX2
// OPENEMS_ENABLE_AVX2 only says the build machine could compile the AVX2 engine;
// check the run host too. __builtin_cpu_supports also requires OS YMM support
// (OSXSAVE/XGETBV). OPENEMS_FORCE_NO_AVX2=1 pretends the CPU lacks AVX2 (testing).
static bool CPU_Supports_AVX2_FMA()
{
	const char* envNoAVX2 = std::getenv("OPENEMS_FORCE_NO_AVX2");
	if (envNoAVX2 && envNoAVX2[0] && std::string(envNoAVX2) != "0")
		return false;
	__builtin_cpu_init();
	return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
#endif

bool openEMS::SetupOperator()
{
#if OPENEMS_ENABLE_AVX2
	if (((m_engine == EngineType_AVX2) || (m_engine == EngineType_AVX2_Multithreaded)) && !CPU_Supports_AVX2_FMA())
	{
		cerr << "openEMS - AVX2+FMA engine is unsupported by this CPU; using multithreaded engine" << endl;
		m_engine = EngineType_Multithreaded;
	}
#endif
	// A cylindrical mesh runs on the multithreaded engine only. An explicit --engine=gpu
	// fails rather than report a CPU run as a GPU one (benchmark/bench.py rows).
#ifdef WITH_GPU
	if (CylinderCoords && (m_engine == EngineType_GPU))
	{
		if (m_engineGiven)
		{
			cerr << "openEMS::SetupOperator: Error: the GPU engine does not run on a cylindrical mesh, "
			        "use --engine=multithreaded" << endl;
			return false;
		}
		cout << "openEMS - the GPU engine does not run on a cylindrical mesh, using the multithreaded engine" << endl;
		m_engine = EngineType_Multithreaded;
	}
#endif
	if (CylinderCoords && ((m_engine == EngineType_AVX2) || (m_engine == EngineType_AVX2_Multithreaded)))
	{
		cout << "openEMS - the AVX2 engine does not run on a cylindrical mesh, using the multithreaded engine" << endl;
		m_engine = EngineType_Multithreaded;
	}
	if (CylinderCoords)
	{
		if (m_CC_MultiGrid.size()>0)
		{
			FDTD_Op = Operator_CylinderMultiGrid::New(m_CC_MultiGrid, m_engine_numThreads);
			if (FDTD_Op==NULL)
				FDTD_Op = Operator_Cylinder::New(m_engine_numThreads);
		}
		else
			FDTD_Op = Operator_Cylinder::New(m_engine_numThreads);
	}
	else if (m_engine == EngineType_SSE)
	{
		FDTD_Op = Operator_sse::New();
	}
	else if (m_engine == EngineType_SSE_Compressed)
	{
		FDTD_Op = Operator_SSE_Compressed::New();
	}
	else if (m_engine == EngineType_Multithreaded)
	{
		FDTD_Op = Operator_Multithread::New(m_engine_numThreads);
	}
	else if (m_engine == EngineType_AVX2)
	{
#if OPENEMS_ENABLE_AVX2
		FDTD_Op = Operator_AVX2::New();
#else
		FDTD_Op = Operator_Multithread::New(m_engine_numThreads);
#endif
	}
	else if (m_engine == EngineType_AVX2_Multithreaded)
	{
#if OPENEMS_ENABLE_AVX2
		FDTD_Op = Operator_AVX2_Multithread::New(m_engine_numThreads);
#else
		FDTD_Op = Operator_Multithread::New(m_engine_numThreads);
#endif
	}
#ifdef WITH_GPU
	else if (m_engine == EngineType_GPU)
	{
		FDTD_Op = Operator_Vulkan::New();
	}
#endif
	else
	{
		FDTD_Op = Operator::New();
	}
	return true;
}

void openEMS::Set_BC_Type(int idx, int type)
{
	if ((idx<0) || (idx>5))
		return;
	m_BC_type[idx] = type;
}

int openEMS::Get_BC_Type(int idx)
{
	if ((idx<0) || (idx>5))
		return -1;
	return m_BC_type[idx];
}

void openEMS::Set_BC_PML(int idx, unsigned int size)
{
	if ((idx<0) || (idx>5))
		return;
	m_BC_type[idx] = 3;
	m_PML_size[idx] = size;
}

int openEMS::Get_PML_Size(int idx)
{
	if ((idx<0) || (idx>5))
		return -1;
	if (m_BC_type[idx]!=3)
		return -1; // return -1 if BC was *not* a PML
	return m_PML_size[idx];
}

void openEMS::Set_Mur_PhaseVel(int idx, double val)
{
	if ((idx<0) || (idx>5))
	return;
	m_Mur_v_ph[idx] = val;
}

bool openEMS::ParseFDTDSetup(std::string file)
{
	Reset();

	if (g_settings.GetVerboseLevel()>0)
		cout << "Read openEMS xml file: " << file << " ..." << endl;

	TiXmlDocument doc(file);
	if (!doc.LoadFile())
	{
		cerr << "openEMS: Error File-Loading failed!!! File: " << file << endl;
		exit(-1);
	}

	if (g_settings.GetVerboseLevel()>0)
		cout << "Read openEMS Settings..." << endl;
	TiXmlElement* openEMSxml = doc.FirstChildElement("openEMS");
	if (openEMSxml==NULL)
	{
		cerr << "Can't read openEMS ... " << endl;
		exit(-1);
	}
	TiXmlElement* FDTD_Opts = openEMSxml->FirstChildElement("FDTD");

	if (FDTD_Opts==NULL)
	{
		cerr << "Can't read openEMS FDTD Settings... " << endl;
		exit(-1);
	}

	if (g_settings.GetVerboseLevel()>0)
		cout << "Read Geometry..." << endl;
	ContinuousStructure* csx = new ContinuousStructure();
	string EC(csx->ReadFromXML(openEMSxml));
	if (EC.empty()==false)
		cerr << EC << endl;
	this->SetCSX(csx);


	return this->Parse_XML_FDTDSetup(FDTD_Opts);
}

bool openEMS::Parse_XML_FDTDSetup(TiXmlElement* FDTD_Opts)
{
	double dhelp=0;
	FDTD_Opts->QueryDoubleAttribute("NumberOfTimesteps",&dhelp);
	if (dhelp<0)
		this->SetNumberOfTimeSteps(0);
	else
		this->SetNumberOfTimeSteps((unsigned int)dhelp);

	int ihelp = 0;
	FDTD_Opts->QueryIntAttribute("CylinderCoords",&ihelp);
	if (ihelp==1)
	{
		this->SetCylinderCoords(true);
		const char* cchelp = FDTD_Opts->Attribute("MultiGrid");
		if (cchelp!=NULL)
			this->SetupCylinderMultiGrid(string(cchelp));
	}

	dhelp = 0;
	FDTD_Opts->QueryDoubleAttribute("MaxTime",&dhelp);
	if (dhelp>0)
		this->SetMaxTime(dhelp);

	dhelp = 0;
	FDTD_Opts->QueryDoubleAttribute("endCriteria",&dhelp);
	if (dhelp==0)
		this->SetEndCriteria(1e-6);
	else
		this->SetEndCriteria(dhelp);

	ihelp = 0;
	FDTD_Opts->QueryIntAttribute("OverSampling",&ihelp);
	if (ihelp>1)
		this->SetOverSampling(ihelp);

	// check for cell constant material averaging
	if (FDTD_Opts->QueryIntAttribute("CellConstantMaterial",&ihelp)==TIXML_SUCCESS)
		this->SetCellConstantMaterial(ihelp==1);

	TiXmlElement* BC = FDTD_Opts->FirstChildElement("BoundaryCond");
	if (BC==NULL)
	{
		cerr << "Can't read openEMS boundary cond Settings... " << endl;
		exit(-3);
	}

//	const char* tmp = BC->Attribute("PML_Grading");
//	string pml_gradFunc;
//	if (tmp)
//		pml_gradFunc = string(tmp);

	string bound_names[] = {"xmin","xmax","ymin","ymax","zmin","zmax"};
	string s_bc;
	for (int n=0; n<6; ++n)
	{
		int EC = BC->QueryIntAttribute(bound_names[n].c_str(),&ihelp);
		if (EC==TIXML_SUCCESS)
		{
			this->Set_BC_Type(n, ihelp);
			continue;
		}
		if (EC==TIXML_WRONG_TYPE)
		{
			const char* tmp = BC->Attribute(bound_names[n].c_str());
			if (tmp)
				s_bc = string(tmp);
			else
				cerr << "openEMS::SetupBoundaryConditions: Warning,  boundary condition for \"" << bound_names[n] << "\" unknown... set to PEC " << endl;
			if (s_bc=="PEC")
				this->Set_BC_Type(n, 0);
			else if (s_bc=="PMC")
				this->Set_BC_Type(n, 1);
			else if (s_bc=="MUR")
				this->Set_BC_Type(n, 2);
			else if (strncmp(s_bc.c_str(),"PML_=",4)==0)
				this->Set_BC_PML(n, atoi(s_bc.c_str()+4));
			else
				cerr << "openEMS::SetupBoundaryConditions: Warning,  boundary condition for \"" << bound_names[n] << "\" unknown... set to PEC " << endl;
		}
		else
			cerr << "openEMS::SetupBoundaryConditions: Warning, boundary condition for \"" << bound_names[n] << "\" not found... set to PEC " << endl;
	}

	//read general mur phase velocity
	if (BC->QueryDoubleAttribute("MUR_PhaseVelocity",&dhelp) == TIXML_SUCCESS)
		for (int n=0;n<6;++n)
			this->Set_Mur_PhaseVel(n, dhelp);

	string mur_v_ph_names[6] = {"MUR_PhaseVelocity_xmin", "MUR_PhaseVelocity_xmax", "MUR_PhaseVelocity_ymin", "MUR_PhaseVelocity_ymax", "MUR_PhaseVelocity_zmin", "MUR_PhaseVelocity_zmax"};
	for (int n=0; n<6; ++n)
		if (BC->QueryDoubleAttribute(mur_v_ph_names[n].c_str(),&dhelp) == TIXML_SUCCESS)
			this->Set_Mur_PhaseVel(n, dhelp);

	TiXmlElement* m_Excite_Elem = FDTD_Opts->FirstChildElement("Excitation");
	if (!m_Excite_Elem)
	{
		cerr << "Excitation::setupExcitation: Error, can't read openEMS excitation settings... " << endl;
		return false;
	}

	Excitation* exc = this->InitExcitation();
	double f0=0, fc=0, f_max=0;
	ihelp = -1;
	m_Excite_Elem->QueryIntAttribute("Type",&ihelp);
	switch (ihelp)
	{
	case Excitation::GaussianPulse:
		m_Excite_Elem->QueryDoubleAttribute("f0",&f0);
		m_Excite_Elem->QueryDoubleAttribute("fc",&fc);
		exc->SetupGaussianPulse(f0, fc);
		break;
	case Excitation::Sinusoidal:  // sinusoidal excite
		m_Excite_Elem->QueryDoubleAttribute("f0",&f0);
		exc->SetupSinusoidal(f0);
		break;
	case Excitation::DiracPulse:
		FDTD_Opts->QueryDoubleAttribute("f_max",&f_max);
		exc->SetupDiracPulse(f_max);
		break;
	case Excitation::Step:
		FDTD_Opts->QueryDoubleAttribute("f_max",&f_max);
		exc->SetupStepExcite(f_max);
		break;
	case Excitation::CustomExcite:
		m_Excite_Elem->QueryDoubleAttribute("f0",&f0);
		FDTD_Opts->QueryDoubleAttribute("f_max",&f_max);
		exc->SetupCustomExcite(m_Excite_Elem->Attribute("Function"), f0, f_max);
		break;
	}

	if (FDTD_Opts->QueryIntAttribute("TimeStepMethod",&ihelp)==TIXML_SUCCESS)
		this->SetTimeStepMethod(ihelp);
	if (FDTD_Opts->QueryDoubleAttribute("TimeStep",&dhelp)==TIXML_SUCCESS)
		this->SetTimeStep(dhelp);
	if (FDTD_Opts->QueryDoubleAttribute("TimeStepFactor",&dhelp)==TIXML_SUCCESS)
		this->SetTimeStepFactor(dhelp);
	return true;
}


bool openEMS::Write2XML(TiXmlNode* rootNode)
{
	TiXmlElement main("openEMS");

	TiXmlElement fdtd("FDTD");
	fdtd.SetAttribute("NumberOfTimesteps", this->NrTS);

	if (this->CylinderCoords)
	{
		fdtd.SetAttribute("CylinderCoords", this->CylinderCoords);
		if (this->m_CC_MultiGrid.size()>0)
		{
			string mg = std::to_string(m_CC_MultiGrid.at(0));
			for (unsigned int n=1;n<m_CC_MultiGrid.size();++n)
			{
				mg += "," + std::to_string(m_CC_MultiGrid.at(n));
			}
			fdtd.SetAttribute("MultiGrid", mg);
		}
	}
	if (this->m_maxTime>0)
		fdtd.SetDoubleAttribute("MaxTime", this->m_maxTime);
	fdtd.SetDoubleAttribute("endCriteria", this->endCrit);
	fdtd.SetAttribute("OverSampling", this->m_OverSampling);
	if (this->m_CellConstantMaterial)
		fdtd.SetAttribute("CellConstantMaterial", this->m_CellConstantMaterial);


	TiXmlElement exc("Excitation");
	exc.SetAttribute("Type", m_Exc->GetExciteType());
	switch (m_Exc->GetExciteType())
	{
	case Excitation::GaussianPulse:
		exc.SetDoubleAttribute("f0", m_Exc->GetCenterFreq());
		exc.SetDoubleAttribute("fc", m_Exc->GetCutOffFreq());
		break;
	case Excitation::Sinusoidal:  // sinusoidal excite
		exc.SetDoubleAttribute("f0", m_Exc->GetCenterFreq());
		break;
	case Excitation::DiracPulse:
		fdtd.SetDoubleAttribute("f_max", m_Exc->GetMaxFreq());
		break;
	case Excitation::Step:
		fdtd.SetDoubleAttribute("f_max", m_Exc->GetMaxFreq());
		break;
	case Excitation::CustomExcite:
		exc.SetDoubleAttribute("f0", m_Exc->GetCenterFreq());
		fdtd.SetDoubleAttribute("f_max", m_Exc->GetMaxFreq());
		exc.SetAttribute("Function", m_Exc->GetCustomFunction());
		break;
	}

	fdtd.SetAttribute("TimeStepMethod", m_TS_method);
	if (m_TS>0)
		fdtd.SetDoubleAttribute("TimeStep", m_TS);
	if (m_TS_fac>0 && m_TS_fac!=1)
		fdtd.SetDoubleAttribute("TimeStepFactor", m_TS_fac);
	fdtd.InsertEndChild(exc);

	TiXmlElement BC("BoundaryCond");
	string bound_names[] = {"xmin","xmax","ymin","ymax","zmin","zmax"};
	string BC_names[] = {"PEC", "PMC", "MUR", "PML_"};
	for (int n=0; n<6; ++n)
	{
		if (m_BC_type[n]==3)
			BC.SetAttribute(bound_names[n], "PML_" + std::to_string(m_PML_size[n]));
		else if ((m_BC_type[n]<3) && (m_BC_type[n]>=0))
			BC.SetAttribute(bound_names[n], BC_names[m_BC_type[n]]);
		else
			BC.SetAttribute(bound_names[n], m_BC_type[n]);

		if (m_Mur_v_ph[n]>0)
			BC.SetAttribute("MUR_PhaseVelocity_" + bound_names[n], m_Mur_v_ph[n]);
	}
	fdtd.InsertEndChild(BC);

	main.InsertEndChild(fdtd);
	this->m_CSX->Write2XML(&main);
	rootNode->InsertEndChild(main);
	return true;
}

bool openEMS::Write2XML(std::string file)
{
	setlocale(LC_NUMERIC, "en_US.UTF-8");
	TiXmlDocument doc(file);
	doc.InsertEndChild(TiXmlDeclaration("1.0","UTF-8","yes"));

	if (Write2XML(&doc)==false) return false;

	doc.SaveFile();
	return doc.SaveFile();
}

bool openEMS::ReadFromXML(std::string file)
{
	this->Reset();
	return this->ParseFDTDSetup(file);
}

void openEMS::SetGaussExcite(double f0, double fc)
{
	this->InitExcitation();
	m_Exc->SetupGaussianPulse(f0, fc);
}

void openEMS::SetSinusExcite(double f0)
{
	this->InitExcitation();
	m_Exc->SetupSinusoidal(f0);
}

void openEMS::SetDiracExcite(double f_max)
{
	this->InitExcitation();
	m_Exc->SetupDiracPulse(f_max);
}

void openEMS::SetStepExcite(double f_max)
{
	this->InitExcitation();
	m_Exc->SetupStepExcite(f_max);
}

void openEMS::SetCustomExcite(std::string str, double f0, double fmax)
{
	this->InitExcitation();
	m_Exc->SetupCustomExcite(str, f0, fmax);
}

Excitation* openEMS::InitExcitation()
{
	delete m_Exc;
	m_Exc = new Excitation();
	return m_Exc;
}

void openEMS::SetCSX(ContinuousStructure* csx)
{
	delete m_CSX;
	m_CSX = csx;
	// we destroy it, so tell language bindings not to do it as well
	if (m_CSX)
		m_CSX->SetOwnedExternally(true);
}

ContinuousStructure* openEMS::GetCSX() const
{
	return m_CSX;
}

int openEMS::SetupFDTD()
{
	timeval startTime;
	gettimeofday(&startTime,NULL);

	Signal::SetupHandlerForSIGINT(SIGNAL_EXIT_FORCE);

	if (m_CSX==NULL)
	{
		cerr << "openEMS::SetupFDTD: Error: CSXCAD is not set!" << endl;
		Signal::SetupHandlerForSIGINT(SIGNAL_ORIGINAL);
		return 3;
	}
	if (m_CSX==NULL)
	{
		cerr << "openEMS::SetupFDTD: Error: CSXCAD is not set!" << endl;
		Signal::SetupHandlerForSIGINT(SIGNAL_ORIGINAL);
		return 3;
	}
	std::string ec = m_CSX->Update();
	if (!ec.empty())
		cerr << ec << endl;
	if (g_settings.GetVerboseLevel()>2)
		m_CSX->ShowPropertyStatus(cerr);

	if (CylinderCoords)
		if (m_CSX->GetCoordInputType()!=CYLINDRICAL)
		{
			cerr << "openEMS::SetupFDTD: Warning: Coordinate system found in the CSX file is not a cylindrical. Forcing to cylindrical coordinate system!" << endl;
			m_CSX->SetCoordInputType(CYLINDRICAL); //tell CSX to use cylinder-coords
		}

	if (m_debugCSX)
		m_CSX->Write2XML("debugCSX.xml");

	//*************** setup operator ************//
	if (SetupOperator()==false)
	{
		Signal::SetupHandlerForSIGINT(SIGNAL_ORIGINAL);
		return 2;
	}

	// default material averaging is quarter cell averaging
	FDTD_Op->SetQuarterCellMaterialAvg();

	if (m_CellConstantMaterial)
	{
		FDTD_Op->SetCellConstantMaterial();
		if (g_settings.GetVerboseLevel()>0)
			cout << "Enabling constant cell material assumption." << endl;
	}

	if (m_Exc==NULL)
	{
		cerr << "openEMS::SetupFDTD: Error, excitation is not defined! Abort!" << endl;
		Signal::SetupHandlerForSIGINT(SIGNAL_ORIGINAL);
		return 3;
	}

	FDTD_Op->SetExcitationSignal(m_Exc);
	FDTD_Op->AddExtension(new Operator_Ext_Excitation(FDTD_Op));
	if (!CylinderCoords)
		FDTD_Op->AddExtension(new Operator_Ext_TFSF(FDTD_Op));

	if (FDTD_Op->SetGeometryCSX(m_CSX)==false)
	{
		Signal::SetupHandlerForSIGINT(SIGNAL_ORIGINAL);
		return(2);
	}

	SetupBoundaryConditions();

	FDTD_Op->SetTimeStepMethod(m_TS_method);

	if (m_TS>0)
		FDTD_Op->SetTimestep(m_TS);
	if (m_TS_fac<1)
		FDTD_Op->SetTimestepFactor(m_TS_fac);

	// Is a steady state detection requested
	Operator_Ext_SteadyState* Op_Ext_SSD = NULL;
	if (m_Exc->GetSignalPeriod()>0)
	{
		cout << "Create a steady state detection using a period of " << m_Exc->GetSignalPeriod() << " s" << endl;
		Op_Ext_SSD = new Operator_Ext_SteadyState(FDTD_Op, m_Exc->GetSignalPeriod());
		unsigned int pos[3];
		for (int p=0;p<3;++p)
			pos[p] = FDTD_Op->GetNumberOfLines(p)/2;
		Op_Ext_SSD->Add_E_Probe(pos, 0);
		Op_Ext_SSD->Add_E_Probe(pos, 1);
		Op_Ext_SSD->Add_E_Probe(pos, 2);

		for (int n=0;n<3;++n)
		{
			for (int p=0;p<3;++p)
				pos[p] = FDTD_Op->GetNumberOfLines(p)/2;

			pos[n] = FDTD_Op->GetNumberOfLines(n)/4;
			Op_Ext_SSD->Add_E_Probe(pos, 0);
			Op_Ext_SSD->Add_E_Probe(pos, 1);
			Op_Ext_SSD->Add_E_Probe(pos, 2);

			pos[n] = FDTD_Op->GetNumberOfLines(n)*3/4;
			Op_Ext_SSD->Add_E_Probe(pos, 0);
			Op_Ext_SSD->Add_E_Probe(pos, 1);
			Op_Ext_SSD->Add_E_Probe(pos, 2);
		}
		FDTD_Op->AddExtension(Op_Ext_SSD);
	}

	if (m_CSX->GetQtyPropertyType(CSProperties::LORENTZMATERIAL)>0)
		FDTD_Op->AddExtension(new Operator_Ext_LorentzMaterial(FDTD_Op));
	if (m_CSX->GetQtyPropertyType(CSProperties::DEBYEMATERIAL)>0)
		FDTD_Op->AddExtension(new Operator_Ext_DebyeMaterial(FDTD_Op));
	if (m_CSX->GetQtyPropertyType(CSProperties::CONDUCTINGSHEET)>0)
		FDTD_Op->AddExtension(new Operator_Ext_ConductingSheet(FDTD_Op, m_Exc->GetMaxFreq()));
	if (m_CSX->GetQtyPropertyType(CSProperties::LUMPED_ELEMENT)>0)
		FDTD_Op->AddExtension(new Operator_Ext_LumpedRLC(FDTD_Op));
	if (m_CSX->GetQtyPropertyType(CSProperties::ABSORBING_BC)>0)
		SetupAbsorbingSheets();


	//check all properties to request material storage during operator creation...
	SetupMaterialStorages();

#ifdef WITH_GPU
	if (dynamic_cast<Operator_Vulkan*>(FDTD_Op))
	{
		bool strictCoeffReserve = false;
		if (const char* envStrict = std::getenv("OPENEMS_GPU_PREFLIGHT_STRICT"))
			strictCoeffReserve = (std::atoi(envStrict) != 0);

		std::string preflightErr;
		if (!Engine_Vulkan::PreflightAllocationForGrid(
				FDTD_Op->GetNumberOfLines(0),
				FDTD_Op->GetNumberOfLines(1),
				FDTD_Op->GetNumberOfLines(2),
				strictCoeffReserve,
				&preflightErr))
		{
			cerr << "openEMS::SetupFDTD: GPU preflight failed before operator generation: "
			     << preflightErr << endl;
			cerr << "  Hint: reduce mesh size, disable strict reserve (OPENEMS_GPU_PREFLIGHT_STRICT=0), "
			     << "or use a different GPU via --gpu-device." << endl;
			Signal::SetupHandlerForSIGINT(SIGNAL_ORIGINAL);
			return 2;
		}

		if (strictCoeffReserve)
			cout << "openEMS::SetupFDTD: GPU preflight passed (strict coefficient reserve enabled)." << endl;
		else
			cout << "openEMS::SetupFDTD: GPU preflight passed (core field/staging allocations)." << endl;
		// upstream's operator line, only once a usable device exists: benchmark/bench.py
		// takes it for "the GPU engine can run"
		cout << "Create FDTD operator (GPU)" << endl;
	}
#endif

	/*******************   create the EC-FDTD operator *****************************/
	Operator::DebugFlags debugFlags = Operator::None;
	if (DebugMat)
		debugFlags |= Operator::debugMaterial;
	if (DebugOp)
		debugFlags |= Operator::debugOperator;
	if (m_debugPEC)
		debugFlags |= Operator::debugPEC;

	bool gpuStartupTrace = false;
#ifdef WITH_GPU
	gpuStartupTrace = (dynamic_cast<Operator_Vulkan*>(FDTD_Op) != NULL);
#endif
	if (const char* envTrace = std::getenv("OPENEMS_GPU_STARTUP_TRACE"))
	{
		char* endPtr = nullptr;
		long v = std::strtol(envTrace, &endPtr, 10);
		if (endPtr && *endPtr == '\0')
			gpuStartupTrace = (v != 0);
	}

	auto opBuildStart = std::chrono::steady_clock::now();
	if (gpuStartupTrace)
		cout << "openEMS::SetupFDTD: starting operator generation (CalcECOperator)..." << endl;
	FDTD_Op->CalcECOperator( debugFlags );
	if (gpuStartupTrace)
	{
		auto opBuildEnd = std::chrono::steady_clock::now();
		double opBuildMs = std::chrono::duration<double, std::milli>(opBuildEnd - opBuildStart).count();
		cout << "openEMS::SetupFDTD: finished CalcECOperator in " << opBuildMs << " ms" << endl;
	}
	/*******************************************************************************/

	//reset flags for material storage, if no dump-box resets it to true, it will be cleaned up...
	FDTD_Op->SetMaterialStoreFlags(0,false);
	FDTD_Op->SetMaterialStoreFlags(1,false);
	FDTD_Op->SetMaterialStoreFlags(2,false);
	FDTD_Op->SetMaterialStoreFlags(3,false);

	unsigned int maxTime_TS = (unsigned int)(m_maxTime/FDTD_Op->GetTimestep());
	if ((m_maxTime>0) && (maxTime_TS<NrTS))
		NrTS = maxTime_TS;

	if (!m_Exc->buildExcitationSignal(NrTS))
		exit(2);
	if (!m_benchmark)
	{
		m_Exc->DumpVoltageExcite("et");
		m_Exc->DumpCurrentExcite("ht");
	}

	timeval OpDoneTime;
	gettimeofday(&OpDoneTime,NULL);

	if (g_settings.GetVerboseLevel()>0)
	{
		FDTD_Op->ShowStat();
		FDTD_Op->ShowExtStat();
		cout << "Creation time for operator: " << CalcDiffTime(OpDoneTime,startTime) << " s" << endl;
	}
	cout << "FDTD simulation size: " << FDTD_Op->GetNumberOfLines(0) << "x" << FDTD_Op->GetNumberOfLines(1) << "x" << FDTD_Op->GetNumberOfLines(2) << " --> "  << FDTD_Op->GetNumberCells() << " FDTD cells " << endl;
	cout << "FDTD timestep is: " <<FDTD_Op->GetTimestep()  << " s; Nyquist rate: " <<  m_Exc->GetNyquistNum() << " timesteps @" << CalcNyquistFrequency(m_Exc->GetNyquistNum(),FDTD_Op->GetTimestep()) << " Hz" << endl;
	if (m_Exc->GetNyquistNum()>1000)
		cerr << "openEMS::SetupFDTD: Warning, the timestep seems to be very small --> long simulation. Check your mesh!?" << endl;

	if (m_Exc->GetSignalPeriod()==0)
	{
		cout << "Excitation signal length is: " <<  m_Exc->GetLength() << " timesteps (" <<  m_Exc->GetLength()*FDTD_Op->GetTimestep() << "s)" << endl;
		cout << "Max. number of timesteps: " << NrTS << " ( --> " << (double)NrTS/(double)(m_Exc->GetLength()) << " * Excitation signal length)" << endl;
		if ( ((double)NrTS/(double)m_Exc->GetLength() < 3) && (m_Exc->GetExciteType()==0))
			cerr << "openEMS::SetupFDTD: Warning, max. number of timesteps is smaller than three times the excitation. " << endl << \
					"\tYou may want to choose a higher number of max. timesteps... " << endl;
	}
	else
	{
		int p = (int)m_Exc->GetSignalPeriodTS();
		cout << "Excitation signal period is: " <<  p << " timesteps (" <<  m_Exc->GetSignalPeriod() << "s)" << endl;
		cout << "Max. number of timesteps: " << NrTS << " ( --> " << (double)NrTS/(double)(m_Exc->GetLength()) << " * Excitation signal period)" << endl;
		if (NrTS/p < 3)
			cerr << "openEMS::SetupFDTD: Warning, max. number of timesteps is smaller than three times the excitation signal period. " << endl << \
					"\tYou may want to choose a higher number of max. timesteps... " << endl;
	}

	if (m_no_simulation)
	{
		// simulation was disabled (to generate debug output only)
		Signal::SetupHandlerForSIGINT(SIGNAL_ORIGINAL);
		return 1;
	}

	//create FDTD engine
	auto engCreateStart = std::chrono::steady_clock::now();
	if (gpuStartupTrace)
		cout << "openEMS::SetupFDTD: creating engine..." << endl;
	// --cpu-cores: the engine starts its workers there.
	if (!m_cpuCoresOption.empty())
	{
		CoreSet cores;
		CoreSet::Parse(m_cpuCoresOption, cores);
		if (CpuTopology::Get().Offers(cores))
			FDTD_Op->SetCoreSet(cores);
		else
			cerr << "openEMS - --cpu-cores=" << m_cpuCoresOption << " ignored: threads cannot be kept to "
			        "those cores here (" << CpuTopology::Get().Describe() << ")" << endl;
	}
	FDTD_Eng = FDTD_Op->CreateEngine();
	ApplyTemporalBlockingOptions();
	if (!FDTD_Op->GetCoreSet().IsAll())
	{
		if (FDTD_Eng->GetNumThreads() > 0)
			cout << "openEMS - worker threads on the " << FDTD_Op->GetCoreSet().ToString() << " cores: "
			     << CpuTopology::Get().DescribeSet(FDTD_Op->GetCoreSet()) << endl;
		else
			cerr << "openEMS - --cpu-cores ignored: this engine has no worker threads to place" << endl;
	}
	if (!m_benchmark)
		WarnUntunedSettings();
	if (gpuStartupTrace)
	{
		auto engCreateEnd = std::chrono::steady_clock::now();
		double engCreateMs = std::chrono::duration<double, std::milli>(engCreateEnd - engCreateStart).count();
		cout << "openEMS::SetupFDTD: engine created in " << engCreateMs << " ms" << endl;
	}

	if (Op_Ext_SSD)
	{
		Eng_Ext_SSD = dynamic_cast<Engine_Ext_SteadyState*>(Op_Ext_SSD->GetEngineExtention());
		Eng_Ext_SSD->SetEngineInterface(this->NewEngineInterface());
	}

	//setup all processing classes
	if (SetupProcessing()==false)
	{
		Signal::SetupHandlerForSIGINT(SIGNAL_ORIGINAL);
		return 2;
	}

	// Cleanup all unused material storages...
	FDTD_Op->CleanupMaterialStorage();

	//check and warn for unused properties and primitives
	m_CSX->WarnUnusedPrimitves(cerr);

	// dump all boxes (voltage, current, fields, ...)
	if (m_debugBox)
	{
		PA->DumpBoxes2File("box_dump_");
	}

	Signal::SetupHandlerForSIGINT(SIGNAL_ORIGINAL);
	return 0;
}

string FormatTime(int sec)
{
	stringstream ss;
	if (sec<60)
	{
		ss << setw(9) << sec << "s";
		return ss.str();
	}
	if (sec<3600)
	{
		ss << setw(6) << sec/60 << "m" << setw(2) << setfill('0') << sec%60 << "s";
		return ss.str();
	}
	ss << setw(3) << sec/3600 << "h" << setw(2) << setfill('0')  << (sec%3600)/60 << "m" << setw(2) << setfill('0')  << sec%60 << "s";
	return ss.str();
}

bool openEMS::CheckAbortCond()
{
	if (m_Abort) //abort was set externally
		return true;

	//check whether SIGINT is received
	if (Signal::ReceivedSIGINT())
	{
		cerr << "openEMS::CheckAbortCond(): Received SIGINT, aborting simulation gracefully..." << endl;
		return true;
	}

	//check whether the file "ABORT" exist in current working directory
	ifstream ifile("ABORT");
	if (ifile)
	{
		ifile.close();
		cerr << "openEMS::CheckAbortCond(): Found file \"ABORT\", aborting simulation gracefully..." << endl;
		return true;
	}

	return false;
}

//! Wall-clock time of a run (s) during which the end criteria is evaluated every Nyquist period, see RunFDTD()
static const double EXACT_ENDCRIT_TIME = 60;

void openEMS::RunFDTD()
{
	cout << "Running FDTD engine... this may take a while... grab a cup of coffee?!?" << endl;

	// --benchmark measures the schedules on this model and writes nothing into
	// the simulation directory: the probes stay, since they set the batch
	// lengths, but write into a scratch directory removed afterwards, and the
	// run's length and end criterion give way to the benchmark's.
	struct BenchmarkScope
	{
		std::filesystem::path home, scratch;
		unsigned int* nrTS = nullptr;
		unsigned int savedNrTS = 0;
		double* endCrit = nullptr;
		double savedEndCrit = 0;
		~BenchmarkScope()
		{
			std::error_code ec;
			if (!scratch.empty())
			{
				std::filesystem::current_path(home, ec);
				std::filesystem::remove_all(scratch, ec);
			}
			if (nrTS)
				*nrTS = savedNrTS;
			if (endCrit)
				*endCrit = savedEndCrit;
		}
	} benchScope;
	if (m_benchmark)
	{
		std::error_code ec;
		benchScope.home = std::filesystem::current_path(ec);
		const std::filesystem::path scratch = std::filesystem::temp_directory_path(ec) /
			("openEMS-benchmark-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
		if (!ec && std::filesystem::create_directory(scratch, ec) && !ec)
		{
			benchScope.scratch = scratch;
			std::filesystem::current_path(scratch, ec);
		}
		if (ec || benchScope.scratch.empty())
		{
			cerr << "RunFDTD: --benchmark needs a scratch directory and could not make "
			     << scratch << ": " << ec.message() << endl;
			return;
		}
		benchScope.nrTS = &NrTS;
		benchScope.savedNrTS = NrTS;
		benchScope.endCrit = &endCrit;
		benchScope.savedEndCrit = endCrit;
		NrTS = 2000000000u;
		endCrit = -1;
	}

	Signal::SetupHandlerForSIGINT(SIGNAL_EXIT_GRACEFUL);

	//special handling of a field processing, needed to realize the end criteria...
	ProcessFields* ProcField = new ProcessFields(NewEngineInterface());
	PA->AddProcessing(ProcField);
	double maxE=0,currE=0;
	double change=1;
	bool numericalFailure = false;
	std::string numericalFailureReason;

	auto failNumerically = [&](const std::string& reason)
	{
		if (numericalFailure)
			return;
		numericalFailure = true;
		numericalFailureReason = reason;
		cerr << "RunFDTD: " << reason << ", cancelling simulation." << endl;
	};

	auto sampleEnergy = [&]() -> bool
	{
		currE = ProcField->CalcTotalEnergyEstimate();
		if (!std::isfinite(currE) || currE < 0.0)
		{
			std::ostringstream reason;
			reason << "invalid field energy at timestep "
			       << FDTD_Eng->GetNumberOfTimesteps() << " (value=" << currE << ")";
			failNumerically(reason.str());
			return false;
		}
		if (currE>maxE)
			maxE=currE;
		return true;
	};

	auto validateChange = [&]() -> bool
	{
		if (!std::isfinite(change) || change < 0.0)
		{
			std::ostringstream reason;
			reason << "invalid residual-energy ratio at timestep "
			       << FDTD_Eng->GetNumberOfTimesteps() << " (value=" << change << ")";
			failNumerically(reason.str());
			return false;
		}
		return true;
	};

	//init processings
	PA->InitAll();

	//add all timesteps to end-crit field processing with max excite amplitude
	unsigned int maxExcite = FDTD_Op->GetExcitationSignal()->GetMaxExcitationTimestep();
//	for (unsigned int n=0; n<FDTD_Op->Exc->Volt_Count; ++n)
//		ProcField->AddStep(FDTD_Op->Exc->Volt_delay[n]+maxExcite);
	ProcField->AddStep(maxExcite);

	int prevTS=0,currTS=0;
	double numCells = FDTD_Op->GetNumberCells();
	std::unique_ptr<EngineBenchmark> bench;
	if (m_benchmark)
		bench.reset(new EngineBenchmark(FDTD_Eng, numCells, m_benchmarkTS,
		                                m_numThreadsGiven ? m_engine_numThreads : 0));
	// The benchmark prints its own lines instead of the status every 4 s.
	std::ostream nullOut(nullptr);
	std::ostream& status = bench ? nullOut : cout;

	double speed = 0;
	double t_diff;
	double t_run;

	timeval currTime;
	gettimeofday(&currTime,NULL);
	timeval startTime = currTime;
	timeval prevTime= currTime;

	// The end criteria is evaluated on a fixed timestep schedule, every Nyquist period,
	// during the first EXACT_ENDCRIT_TIME seconds: a wall-clock schedule lets a short
	// simulation run several times longer than necessary, the more the faster the engine.
	// After that only with the wall-clock progress report, which then is a small overshoot
	// and saves the (full-domain) energy estimates -- unless --exact-endcriteria asks
	// for a machine-independent stopping point. The steady-state detection keeps its
	// verdict current itself, at the end of every period, and is read after every
	// batch of timesteps. The benchmark never stops on an end criteria.
	bool exactEndCrit = (Eng_Ext_SSD==NULL) && !bench;
	if (exactEndCrit)
	{
		unsigned int endCritInterval = FDTD_Op->GetExcitationSignal()->GetNyquistNum();
		if (m_exactEndCriteria)
			cout << "Exact-endcriteria: evaluating the end criteria every "
			     << endCritInterval << " timestep(s) (this excitation's Nyquist rate)" << endl;
		else if (g_settings.GetVerboseLevel()>0)
			cout << "Evaluating the end criteria every " << endCritInterval << " timestep(s) during the first "
			     << EXACT_ENDCRIT_TIME << " s, with the progress report after that" << endl;
		ProcField->SetProcessInterval(endCritInterval);
	}

	// OPENEMS_ENDCRIT_TRACE=1: a line for every evaluation of the energy end
	// criterion, all digits, so that the traces of two runs compare as text.
	const char* envEndCritTrace = std::getenv("OPENEMS_ENDCRIT_TRACE");
	const bool endCritTrace = envEndCritTrace && (string(envEndCritTrace) == "1");
	auto traceEndCriteria = [&]()
	{
		if (!endCritTrace)
			return;
		char line[160];
		snprintf(line, sizeof(line), "endcrit trace: ts=%u energy=%.17g maxE=%.17g change=%.17g\n",
		         FDTD_Eng->GetNumberOfTimesteps(), currE, maxE, change);
		cout << line << flush;
	};

	// The end criteria after a batch of timesteps: false stops the run on a numerical failure.
	auto checkEndCriteria = [&]() -> bool
	{
		if (bench)
			return true;
		if (Eng_Ext_SSD)
		{
			change = Eng_Ext_SSD->GetLastDiff();
			return validateChange();
		}
		if (ProcField->CheckTimestep())
		{
			if (!sampleEnergy())
				return false;
			if (exactEndCrit && maxE)
				change = currE/maxE;
			traceEndCriteria();
			return validateChange();
		}
		return true;
	};
	// After the first EXACT_ENDCRIT_TIME seconds, only with the progress report.
	auto relaxEndCriteria = [&]()
	{
		if (!exactEndCrit || m_exactEndCriteria || (CalcDiffTime(currTime,startTime)<=EXACT_ENDCRIT_TIME))
			return;
		exactEndCrit = false;
		if (g_settings.GetVerboseLevel()>0)
			cout << "Evaluating the end criteria with the progress report only from here on" << endl;
		ProcField->ClearProcessInterval();
	};

	// ---- In-batch sampling (plans/BATCH_SAMPLING.md) ----
	// The loop used to end every IterateTS() at the next timestep any
	// processing was due at, which capped a blocked schedule's depth at the
	// probe interval. With in-batch sampling a batch runs on past the
	// timesteps at which only the probes and the energy estimate of the end
	// criterion are due; the engine samples what those read on the way, and
	// the loop replays the batch afterwards: at each timestep the old loop
	// would have ended a batch at, it does what the old loop did there, with
	// the engine in replay of that timestep. OPENEMS_BATCH_SAMPLING=0 keeps
	// the old loop.
	std::string bsOff;
	if (const char* env = std::getenv("OPENEMS_BATCH_SAMPLING"))
		if (string(env) == "0")
			bsOff = "OPENEMS_BATCH_SAMPLING=0";
	if (bsOff.empty() && !FDTD_Eng->SupportsBatchSampling())
		bsOff = "the " + EngineOptionValue() + " engine does not support it";
	const bool bsOn = bsOff.empty();
	// Per processing: served by the replay. The probes are (ProcessIntegral:
	// they read only through the engine's GetVolt()/GetCurr()), and so is
	// ProcField, which reads nothing itself. A batch ends wherever any other
	// processing reads the fields.
	std::vector<char> bsServed;
	size_t bsProcField = 0;
	unsigned int bsRing = 0;     // the most sampled timesteps of one batch
	// The most timesteps of one batch: at first the old loop's cap, the
	// Nyquist interval, then about half a second's worth (adaptBatchLength()).
	const unsigned int bsMinLen = std::max(1u, PA->GetMaxInterval());
	unsigned int bsMaxLen = bsMinLen;
	unsigned int bsDoneTS = 0;   // the timestep last replayed: the one the run has got to
	bool bsStopped = false;      // the run stopped in the replay of a timestep before the batch's last
	struct {unsigned long long batches = 0, timesteps = 0, ends = 0, probes = 0, energies = 0;} bsStats; // -v

	// One batch: the timesteps the old loop would have ended its batches at,
	// and the samples a replay of them needs.
	struct SampledBatch
	{
		std::vector<unsigned int> ends;     // ascending; back() is the batch's last timestep, read live
		std::vector<unsigned int> probeTS;  // the ends at which a probe reads: footprint samples
		std::vector<unsigned int> energyTS; // the ends at which the end criterion reads the energy
		unsigned int longestRun = 0;        // the most timesteps between two samples, or a sample and an end of the batch
		// After the last end, as the old loop leaves them there: the step
		// (capped by NrTS, not by the benchmark), and the processings' step
		// positions. Where a batch may be planned before the one before it
		// is replayed (the pipelined loop), planBatch() continues from these.
		int nextStep = 0;
		std::vector<size_t> nextPositions;
	};
	// The batch from startTS, whose first old-loop batch is step timesteps
	// long: what PA->Process() and checkEndCriteria() would do at each end,
	// looked ahead without doing it (Processing::ProcessAt()), from the
	// processings' step positions now or from \a positions. It ends at the
	// first end at which a processing not served reads the fields, at the
	// ring's capacity, NrTS, the benchmark's boundary, or past bsMaxLen
	// timesteps. Every end costs a few operations per processing; the old
	// loop called PA->Process() at each of them.
	auto planBatch = [&](unsigned int startTS, int step, const std::vector<size_t>* positions = nullptr) -> SampledBatch
	{
		SampledBatch b;
		std::vector<size_t> psPos = positions ? *positions : PA->GetStepPositions();
		std::vector<char> reads;
		const bool energyChecks = !bench && (Eng_Ext_SSD==NULL); // as checkEndCriteria()
		unsigned int ts = startTS, lastSample = startTS, sampled = 0;
		for (;;)
		{
			if (step <= 0)
				throw std::logic_error("in-batch sampling: a batch of no timesteps");
			ts += (unsigned int)step;
			b.ends.push_back(ts);
			if (ts >= NrTS)
				break;
			int next = PA->ProcessAt(ts, psPos, reads);
			bool probe = false, other = false;
			for (size_t i = 0; i < reads.size(); ++i)
			{
				if (!reads[i])
					continue;
				if (bsServed[i])
					probe = true;
				else
					other = true;
			}
			const bool energy = energyChecks && ProcField->CheckTimestepAt(ts, psPos.at(bsProcField));
			// as the old loop after PA->Process()
			if ((next<0) || (next>(int)(NrTS - ts))) next=NrTS - ts;
			b.nextStep = next;
			// The steady-state checks of a batch wait in a queue of
			// Engine_Ext_SteadyState::MAX_QUEUED_VERDICTS for the replay.
			if (other || ((probe || energy) && (sampled >= bsRing))
			    || (bench && (ts >= bench->Boundary())) || (ts - startTS >= bsMaxLen)
			    || (Eng_Ext_SSD && (ts - startTS >= Eng_Ext_SSD->GetMaxRunAhead())))
				break;  // the last end: the processings read the live fields here
			if (probe)
				b.probeTS.push_back(ts);
			if (energy)
				b.energyTS.push_back(ts);
			if (probe || energy)
			{
				++sampled;
				b.longestRun = std::max(b.longestRun, ts - lastSample);
				lastSample = ts;
			}
			step = bench ? bench->Clamp(next, ts) : next;
		}
		b.longestRun = std::max(b.longestRun, b.ends.back() - lastSample);
		b.nextPositions = psPos;
		return b;
	};
	// Replay a batch the engine has run: at each end in order, what the old
	// loop did after an IterateTS(), PA->Process() and the end criteria, with
	// the engine in replay of that timestep. Stops at the first end at which
	// the run stops (an end criterion met, a numerical failure); bsDoneTS is
	// the last end replayed, and bsStopped tells whether that is before the
	// batch's last. \a step is left as the old loop would have it there.
	auto replayBatch = [&](const SampledBatch& b, int& step)
	{
		for (size_t i = 0; i < b.ends.size(); ++i)
		{
			const unsigned int ts = b.ends[i];
			bsDoneTS = ts;
			bool ok;
			{
				BatchReplayScope replay(FDTD_Eng, ts);
				step = PA->Process();
				ok = checkEndCriteria();
			}
			const bool last = (i+1 == b.ends.size());
			if (!ok || (change<=endCrit))
			{
				bsStopped = !last;
				return;
			}
			if ((step<0) || (step>(int)(NrTS - ts))) step=NrTS - ts;
			// The processings must have done what the plan looked ahead to.
			const int next = (bench && !last) ? bench->Clamp(step, ts) : step;
			const bool asPlanned = last ? ((ts >= NrTS) || ((step == b.nextStep) && (PA->GetStepPositions() == b.nextPositions)))
			                            : ((next > 0) && (ts + (unsigned int)next == b.ends[i+1]));
			if (!asPlanned)
			{
				std::ostringstream err;
				err << "in-batch sampling: the replay of timestep " << ts << " continues to timestep "
				    << ts + next << ", the plan to " << (last ? ts + b.nextStep : b.ends[i+1]);
				throw std::logic_error(err.str());
			}
		}
	};
	// After a batch of batchTS timesteps begun at batchTime: the next may run
	// about half a second, at most twice as long as this one, never shorter
	// than the old loop's cap. That keeps the status line and the abort
	// checks at the batch ends as frequent as they were.
	auto adaptBatchLength = [&](unsigned int batchTS, const timeval& batchTime)
	{
		timeval now;
		gettimeofday(&now,NULL);
		const double seconds = CalcDiffTime(now,batchTime);
		double len = (seconds > 0) ? 0.5*batchTS/seconds : 2.0*bsMaxLen;
		len = std::min(len, 2.0*bsMaxLen);
		len = std::min(len, (double)(1u<<30));
		bsMaxLen = std::max(bsMinLen, (unsigned int)len);
	};

	if (m_DumpStats)
		InitRunStatistics(OPENEMS_RUN_STAT_FILE);
	//*************** simulate ************//

#ifdef WITH_GPU
	Engine_Vulkan* gpuEng = dynamic_cast<Engine_Vulkan*>(FDTD_Eng);
#endif

	try
	{
		PA->PreProcess();

		if (bsOn)
		{
			// The footprint: every probe reads its cells once while the engine
			// records them, those with a start time too. What a probe reads does
			// not depend on the timestep.
			bsServed.assign(PA->GetNumberOfProcessings(), 0);
			FDTD_Eng->StartFootprintRecording();
			for (size_t i = 0; i < PA->GetNumberOfProcessings(); ++i)
			{
				Processing* proc = PA->GetProcessing(i);
				if (proc == ProcField)
				{
					bsServed[i] = 1;
					bsProcField = i;
					continue;
				}
				ProcessIntegral* pi = dynamic_cast<ProcessIntegral*>(proc);
				if (!pi)
					continue;
				bsServed[i] = 1;
				if (pi->GetEnable())
					pi->CalcMultipleIntegrals();
			}
			FDTD_Eng->StopFootprintRecording();
			const size_t cells = FDTD_Eng->GetFootprint().size();
			// At most 1024 sampled timesteps per batch, and about 64 MB of them.
			bsRing = 1024;
			if (cells > 0)
				bsRing = (unsigned int)std::min<size_t>(1024, std::max<size_t>(16, (64u<<20)/(cells*sizeof(FDTD_FLOAT))));
			if (const char* env = std::getenv("OPENEMS_BATCH_SAMPLING_RING"))
			{
				if (atoi(env) > 0)
					bsRing = (unsigned int)atoi(env);
				else
					cerr << "RunFDTD: Warning: invalid OPENEMS_BATCH_SAMPLING_RING='" << env << "' (must be >=1), using " << bsRing << endl;
			}
			// the energy estimate, as checkEndCriteria() computes it
			FDTD_Eng->SetBatchEnergyFunction([ProcField]() {return ProcField->CalcTotalEnergyEstimate();});
			cout << "In-batch sampling ON: " << cells << " probe cells, ring " << bsRing << " timesteps" << endl;
		}
		else
			cout << "In-batch sampling OFF: " << bsOff << endl;

#ifdef WITH_GPU
		// Enable probe access recording during initial PA->Process().
		if (gpuEng)
			gpuEng->StartProbeRecording();
#endif

		int step=PA->Process();
		if ((step<0) || (step>(int)NrTS)) step=NrTS;
		if (bench)
		{
			bench->NoteInterval(step);
			step = bench->Clamp(step, FDTD_Eng->GetNumberOfTimesteps());
		}

#ifdef WITH_GPU
	if (gpuEng)
	{
		// The recording above saw only the probes due at this timestep.  One
		// with a start time reads its cells now, or once due it would miss the
		// probe cache and read the live fields, which the pipelined loop has
		// computed ahead by then.
		for (size_t i = 0; i < PA->GetNumberOfProcessings(); ++i)
		{
			ProcessIntegral* pi = dynamic_cast<ProcessIntegral*>(PA->GetProcessing(i));
			if (pi && pi->GetEnable())
				pi->CalcMultipleIntegrals();
		}

		// Build the probe gather pipeline from recorded cell accesses
		gpuEng->SetupProbeCache(PA);

		// Field dumps registered the cells they read in InitProcess(), and the
		// GPU gathers those at each dump's timestep.  A dump it cannot serve
		// reads the whole fields when it runs, which then must still be its
		// timestep: no batch may be computing ahead.  ProcField (added to PA
		// above, purely so the energy estimate gets a processing cadence) is a
		// ProcessFields too, but never dumps.
		bool hostFieldReaders = false;
		for (size_t i = 0; i < PA->GetNumberOfProcessings(); ++i)
		{
			Processing* proc = PA->GetProcessing(i);
			if (proc == ProcField) continue;
			ProcessFields* pf = dynamic_cast<ProcessFields*>(proc);
			if (pf && pf->GetEnable() && !pf->IsServedByGPU()) { hostFieldReaders = true; break; }
		}
		gpuEng->SetHostFieldReaders(hostFieldReaders);
	}

	// OPENEMS_GPU_PIPELINE=0 takes the standard loop, for A/B comparisons.
	const char* envPipeline = std::getenv("OPENEMS_GPU_PIPELINE");
	const bool gpuPipelineOff = envPipeline && string(envPipeline) == "0";

	// Use pipelined loop if probe gather is set up and no CPU extensions
	if (gpuEng && gpuEng->HasPipelinedProcessing() && !gpuPipelineOff)
	{
		// How far the GPU computes ahead while the host processes: by default
		// (auto) up to the next timestep a processing is due at, at most
		// gpuSpecMaxDepth; OPENEMS_GPU_SPEC_DEPTH=<n> fixes it at n timesteps.
		// Auto hides the host's work on a dump behind the GPU's: 160^3 with 63
		// whole-volume dumps runs 1.44 s instead of 3.0 s at depth 1.
		int gpuSpecDepth = 1;
		bool gpuSpecAuto = true;
		int gpuSpecMaxDepth = 64;
		if (const char* envSpecDepth = std::getenv("OPENEMS_GPU_SPEC_DEPTH"))
		{
			string specStr(envSpecDepth);
			if (specStr != "auto" && specStr != "AUTO" && specStr != "Auto")
			{
				int parsedDepth = atoi(envSpecDepth);
				if (parsedDepth > 0)
				{
					gpuSpecAuto = false;
					gpuSpecDepth = parsedDepth;
				}
				else if (parsedDepth < 0)  // "0" stays a shorthand for auto
					cerr << "RunFDTD: Warning: invalid OPENEMS_GPU_SPEC_DEPTH='" << envSpecDepth
					     << "' (use >=1, 0, or auto). Using auto." << endl;
			}
		}
		if (const char* envSpecMaxDepth = std::getenv("OPENEMS_GPU_SPEC_MAX_DEPTH"))
		{
			int parsedMaxDepth = atoi(envSpecMaxDepth);
			if (parsedMaxDepth > 0)
				gpuSpecMaxDepth = parsedMaxDepth;
			else
				cerr << "RunFDTD: Warning: invalid OPENEMS_GPU_SPEC_MAX_DEPTH='" << envSpecMaxDepth
				     << "' (must be >=1). Using default max depth=64." << endl;
		}

		if (!gpuSpecAuto)
			cout << "RunFDTD: GPU speculative depth set to " << gpuSpecDepth << endl;

		// ====== Pipelined GPU main loop ======
		// Pipeline: GPU computes TS(N+1) while CPU processes TS(N).
		//
		// Prime: run first batch, drain, snapshot.
		FDTD_Eng->IterateTS(step);                   // submit first batch
		gpuEng->DrainGPU();                           // wait for first batch
		gpuEng->SnapshotProbeCache();                 // capture probe results

		while ((FDTD_Eng->GetNumberOfTimesteps()<NrTS) && (change>endCrit) && !CheckAbortCond())
		{
			// Every iteration ends with the GPU drained, so the schedule can change here.
			currTS = FDTD_Eng->GetNumberOfTimesteps();
			if (bench && currTS >= bench->Boundary() && !bench->Advance(currTS))
				break;
			// Submit speculative timesteps (GPU starts, numTS NOT advanced).
			// Auto mode uses PA->Process() cadence: submit up to the next processing
			// horizon so probe gathers / dumps still happen on schedule.
			int speculativeNow = gpuSpecAuto ? step : gpuSpecDepth;
			if (speculativeNow > gpuSpecMaxDepth) speculativeNow = gpuSpecMaxDepth;
			// Clamp to requested processing cadence and remaining global timesteps.
			if (step > 0 && speculativeNow > step) speculativeNow = step;
			currTS = FDTD_Eng->GetNumberOfTimesteps();
			int tsLeft = (int)(NrTS - currTS);
			if (bench)
				tsLeft = bench->Clamp(tsLeft, currTS);
			if (speculativeNow > tsLeft) speculativeNow = tsLeft;
			// `step` is the interval that led to this timestep; the next one is
			// only reported by the PA->Process() below.  Stop at the next
			// timestep a processing is due at, so that no batch runs past it.
			for (int s = 1; s < speculativeNow; ++s)
				if (PA->IsDueAt((unsigned int)currTS + s)) { speculativeNow = s; break; }
			if (speculativeNow < 1) break;
			gpuEng->SubmitSpeculative((unsigned int)speculativeNow);

			// Process the state committed by the PREVIOUS iteration (or the
			// prime step above), while the GPU runs the batch just submitted.
			// Probes read the probe cache, field dumps the cells the GPU
			// gathered at the end of the batch that committed this state.
			gpuEng->EnablePipelinedReading(true);
			step = PA->Process();
			gpuEng->EnablePipelinedReading(false);
			if (bench)
				bench->NoteInterval(step);

			// Drain+commit speculative step(s) → advance numTS
			gpuEng->DrainGPU();
			gpuEng->CommitSpeculative();
			gpuEng->SnapshotProbeCache();

			// If step > speculativeNow, we need extra timesteps before next Process.
			if (step > speculativeNow)
			{
				int remaining = step - speculativeNow;
				currTS = FDTD_Eng->GetNumberOfTimesteps();
				if (remaining > (int)(NrTS - currTS)) remaining = NrTS - currTS;
				if (bench)
					remaining = bench->Clamp(remaining, currTS);
				if (remaining > 0)
				{
					FDTD_Eng->IterateTS(remaining);
					gpuEng->DrainGPU();
					gpuEng->SnapshotProbeCache();
				}
			}

			// --- End criteria, on the drained GPU ---
			if (!checkEndCriteria())
				break;

			currTS = FDTD_Eng->GetNumberOfTimesteps();
			if ((step<0) || (step>(int)(NrTS - currTS))) step=NrTS - currTS;

			// --- Timing and status output ---
			gettimeofday(&currTime,NULL);
			t_diff = CalcDiffTime(currTime,prevTime);
			relaxEndCriteria();

			if (t_diff>4)
			{
				t_run = CalcDiffTime(currTime,startTime);
				speed = numCells*(currTS-prevTS)/t_diff;
				status << "[@" <<  FormatTime(t_run) <<  "] Timestep: " << setw(12)  << currTS ;
				status << " || Speed: " << setw(6) << setprecision(1) << std::fixed << speed*1e-6 << " MC/s (" <<  setw(4) << setprecision(3) << std::scientific << t_diff/(currTS-prevTS) << " s/TS)" ;
				if (Eng_Ext_SSD==NULL)
				{
					if (!exactEndCrit) // otherwise already kept current above, every Nyquist period
					{
						if (!sampleEnergy())
							break;
						if (maxE)
							change = currE/maxE;
						traceEndCriteria();
						if (!validateChange())
							break;
					}
					status << " || Energy: ~" << setw(6) << setprecision(2) << std::scientific << currE << " (-" << setw(5)  << setprecision(2) << std::fixed << fabs(10.0*log10(change)) << "dB)" << endl;
				}
				else
					status << " || SteadyState: " << setw(6) << setprecision(2) << std::fixed << 10.0*log10(change) << " dB" << endl;
				prevTime=currTime;
				prevTS=currTS;

				PA->FlushNext();

				if (m_DumpStats)
					DumpRunStatistics(OPENEMS_RUN_STAT_FILE, t_run, currTS, speed, currE);
				FDTD_Eng->NextInterval(speed);
			}
		}

		// Final process for the last snapshot, unless numerical validation stopped the run.
		if (!numericalFailure)
		{
			gpuEng->EnablePipelinedReading(true);
			PA->Process();
			gpuEng->EnablePipelinedReading(false);
		}
	}
	else
#endif
	{
	// ====== Standard (non-pipelined) main loop ======
	// Taken whenever the pipelined loop above isn't usable: CPU extensions on
	// the GPU engine, a field dump the GPU cannot serve, or neither probes nor
	// dumps.  On the GPU, IterateTS() is a fire-and-forget submit, and field
	// dumps gather their cells at the end of that batch.

	// After a batch that ended at currTS: the end criteria relaxed after their
	// first minute, and every 4 s the status line. False stops the run on a
	// numerical failure.
	auto progress = [&]() -> bool
	{
		gettimeofday(&currTime,NULL);

		t_diff = CalcDiffTime(currTime,prevTime);
		relaxEndCriteria();

		if (t_diff>4)
		{
			t_run = CalcDiffTime(currTime,startTime);
			speed = numCells*(currTS-prevTS)/t_diff;
			status << "[@" <<  FormatTime(t_run) <<  "] Timestep: " << setw(12)  << currTS ;
			status << " || Speed: " << setw(6) << setprecision(1) << std::fixed << speed*1e-6 << " MC/s (" <<  setw(4) << setprecision(3) << std::scientific << t_diff/(currTS-prevTS) << " s/TS)" ;
			if (Eng_Ext_SSD==NULL)
			{
				if (!exactEndCrit) // otherwise already kept current above, every Nyquist period
				{
					if (!sampleEnergy())
						return false;
					if (maxE)
						change = currE/maxE;
					traceEndCriteria();
					if (!validateChange())
						return false;
				}
				status << " || Energy: ~" << setw(6) << setprecision(2) << std::scientific << currE << " (-" << setw(5)  << setprecision(2) << std::fixed << fabs(10.0*log10(change)) << "dB)" << endl;
			}
			else
				status << " || SteadyState: " << setw(6) << setprecision(2) << std::fixed << 10.0*log10(change) << " dB" << endl;
			prevTime=currTime;
			prevTS=currTS;

			PA->FlushNext();

			if (m_DumpStats)
				DumpRunStatistics(OPENEMS_RUN_STAT_FILE, t_run, currTS, speed, currE);
			FDTD_Eng->NextInterval(speed);
		}
		return true;
	};

	if (bsOn)
	{
	// With in-batch sampling: a batch runs on past the timesteps at which
	// only probes and the energy estimate are due, and is replayed after.
	while ((FDTD_Eng->GetNumberOfTimesteps()<NrTS) && (change>endCrit) && !numericalFailure && !CheckAbortCond())
	{
		currTS = FDTD_Eng->GetNumberOfTimesteps();
		if (bench)
		{
			if (currTS >= bench->Boundary())
			{
				if (!bench->Advance(currTS))
					break;
			}
			step = bench->Clamp(step, currTS);
		}
		timeval batchTime;
		gettimeofday(&batchTime,NULL);
		const SampledBatch batch = planBatch(currTS, step);
		const unsigned int batchTS = batch.ends.back() - currTS;
		if (bench)
			bench->NoteInterval((int)(FDTD_Eng->SamplesInsideBlocks() ? batchTS : batch.longestRun));
		FDTD_Eng->ScheduleBatchSamples(batch.probeTS, batch.energyTS);
		FDTD_Eng->IterateSampledTS(batchTS);
		++bsStats.batches;
		bsStats.timesteps += batchTS;
		bsStats.ends += batch.ends.size();
		bsStats.probes += batch.probeTS.size();
		bsStats.energies += batch.energyTS.size();
		replayBatch(batch, step);
		if (numericalFailure || bsStopped)
			break;
		adaptBatchLength(batchTS, batchTime);

		currTS = FDTD_Eng->GetNumberOfTimesteps();
		if (!progress())
			break;
	}
	}
	else
	{
	while ((FDTD_Eng->GetNumberOfTimesteps()<NrTS) && (change>endCrit) && !numericalFailure && !CheckAbortCond())
	{
		if (bench)
		{
			currTS = FDTD_Eng->GetNumberOfTimesteps();
			if (currTS >= bench->Boundary())
			{
#ifdef WITH_GPU
				// Time a run to its end, not to its last submit.
				if (gpuEng)
					gpuEng->DrainGPU();
#endif
				if (!bench->Advance(currTS))
					break;
			}
			step = bench->Clamp(step, currTS);
		}
		FDTD_Eng->IterateTS(step);
		step=PA->Process();
		if (bench)
			bench->NoteInterval(step);

		if (!checkEndCriteria())
			break;

//		cout << " do " << step << " steps; current: " << eng.GetNumberOfTimesteps() << endl;
		currTS = FDTD_Eng->GetNumberOfTimesteps();
		if ((step<0) || (step>(int)(NrTS - currTS))) step=NrTS - currTS;

		if (!progress())
			break;
	}
	}
	} // end standard main loop
	} // end simulation try block
	catch (const std::exception& e)
	{
		failNumerically(std::string("FDTD engine exception: ") + e.what());
	}
	catch (...)
	{
		failNumerically("unknown FDTD engine exception");
	}

	// The timestep the run got to. With in-batch sampling the last one
	// replayed, where the old loop would have stopped; the engine may have
	// run on to the end of that batch.
	const unsigned int doneTS = bsOn ? bsDoneTS : FDTD_Eng->GetNumberOfTimesteps();
	if (bsOn && (g_settings.GetVerboseLevel()>0) && bsStats.batches)
	{
		std::ostringstream os;
		os << "In-batch sampling: " << bsStats.batches << " batches of " << bsStats.timesteps << " timesteps ("
		   << setprecision(1) << std::fixed << (double)bsStats.timesteps/bsStats.batches << " per batch), "
		   << bsStats.ends << " timesteps replayed, " << bsStats.probes << " probe and "
		   << bsStats.energies << " energy samples";
		cout << os.str() << endl;
	}

	// Temporal blocking: what share of the timesteps ran blocked.
	{
		const long long blockedTS = FDTD_Eng->GetBlockedTimesteps();
		const TemporalBlockingRequest tbReq = FDTD_Eng->GetTemporalBlocking();
		if ((blockedTS >= 0) && ((tbReq.mode == TemporalBlockingRequest::FORCED) || FDTD_Eng->PlanTemporalBlocking(tbReq).active))
			cout << "Temporal blocking: " << blockedTS << " of " << FDTD_Eng->GetNumberOfTimesteps()
			     << " timesteps ran blocked" << endl;
	}

	if (numericalFailure)
	{
		ofstream marker("openEMS_numerical_failure.txt", ios::out | ios::trunc);
		if (marker)
			marker << numericalFailureReason << endl;
		else
			cerr << "RunFDTD: Warning: unable to write openEMS_numerical_failure.txt" << endl;

#ifdef WITH_GPU
		// A validation exception can leave asynchronous work outstanding.  Drain
		// best-effort before returning, but never mask the original failure.
		if (gpuEng)
		{
			try
			{
				gpuEng->DrainGPU();
			}
			catch (...)
			{
			}
		}
#endif

		gettimeofday(&currTime,NULL);
		t_diff = CalcDiffTime(currTime,startTime);
		cout << "Time for " << doneTS
		     << " iterations with " << FDTD_Op->GetNumberCells()
		     << " cells : " << t_diff << " sec (cancelled)" << endl;

		Signal::SetupHandlerForSIGINT(SIGNAL_ORIGINAL);
		return;
	}

	if (change <= endCrit)
	{
		if (Eng_Ext_SSD == NULL)
			cout << "RunFDTD: residual field energy reached -"
			     << fabs(10.0*log10(endCrit)) << " dB after " << doneTS
			     << " timesteps (-" << setprecision(2) << std::fixed << fabs(10.0*log10(change))
			     << " dB); stopping simulation." << endl;
		else
			cout << "RunFDTD: steady-state criterion of " << setprecision(2) << std::fixed
			     << 10.0*log10(endCrit) << " dB reached after " << doneTS
			     << " timesteps (" << 10.0*log10(change) << " dB); stopping simulation." << endl;
	}

	if (!bench && (change>endCrit) && (FDTD_Op->GetExcitationSignal()->GetExciteType()==0))
		cerr << "RunFDTD: Warning: Max. number of timesteps was reached before the end-criteria of -" << fabs(10.0*log10(endCrit)) << "dB was reached... " << endl << \
				"\tYou may want to choose a higher number of max. timesteps... " << endl;

#ifdef WITH_GPU
	// Include outstanding asynchronous GPU work in runtime and speed reports.
	if (gpuEng)
		gpuEng->DrainGPU();
#endif

	gettimeofday(&currTime,NULL);
	t_diff = CalcDiffTime(currTime,startTime);

	cout << "Time for " << doneTS << " iterations with " << FDTD_Op->GetNumberCells() << " cells : " << t_diff << " sec" << endl;
	cout << "Speed: " << numCells*(double)doneTS/t_diff*1e-6 << " MCells/s " << endl;

	if (m_DumpStats)
		DumpStatistics(OPENEMS_STAT_FILE, t_diff, doneTS);

	//*************** postproc ************//
	PA->PostProcess();

	if (bench)
		bench->Report(cout, EngineOptionValue(), TemporalBlockingOptionName());

	Signal::SetupHandlerForSIGINT(SIGNAL_ORIGINAL);
}

bool openEMS::DumpStatistics(const string& filename, double time, unsigned int numTS)
{
	ofstream stat_file;
	stat_file.open(filename.c_str());

	if (!stat_file.is_open())
	{
		cerr << "openEMS::DumpStatistics: Error, opening file failed..." << endl;
		return false;
	}
	stat_file << std::setprecision( 16 );
	stat_file << FDTD_Op->GetNumberCells() << "\t% number of cells" << endl;
	stat_file << FDTD_Op->GetTimestep() << "\t% timestep (s)" << endl;
	stat_file << numTS << "\t% number of iterations" << endl;
	stat_file << numTS*FDTD_Op->GetTimestep() << "\t% total numerical time (s)" << endl;
	stat_file << time << "\t% simulation time (s)" << endl;
	stat_file << (double)FDTD_Op->GetNumberCells()*(double)numTS/time << "\t% speed (cells/s)" << endl;

	stat_file.close();
	return true;
}

bool openEMS::InitRunStatistics(const string& filename)
{
	ofstream stat_file;
	stat_file.open(filename.c_str(), ios_base::out);

	if (!stat_file.is_open())
	{
		cerr << "openEMS::InitRunStatistics: Error, opening file failed..." << endl;
		return false;
	}
	stat_file << "%time\ttimestep\tspeed\tenergy" << endl;
	stat_file.close();
	return true;
}

bool openEMS::DumpRunStatistics(const string& filename, double time, unsigned int ts, double speed, double energy)
{
	ofstream stat_file;
	stat_file.open(filename.c_str(), ios_base::app);

	if (!stat_file.is_open())
	{
		cerr << "openEMS::DumpRunStatistics: Error, opening file failed..." << endl;
		return false;
	}
	stat_file << time << "\t" << ts << "\t" << speed << "\t" << energy << endl;
	stat_file.close();
	return true;
}
