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

#include <iomanip>
#include "tools/global.h"
#include "tools/vtk_file_writer.h"
#include "tools/hdf5_file_writer.h"
#include "processfields.h"
#include "processfields_calc.h"
#include "FDTD/engine_interface_fdtd.h"
#include "FDTD/engine_avx2.h"
#ifdef WITH_GPU
#include "FDTD/engine_vulkan.h"
#endif

using std::cerr;
using std::endl;

ProcessFields::ProcessFields(Engine_Interface_Base* eng_if) : Processing(eng_if)
{
	m_DumpType = E_FIELD_DUMP;
	// vtk-file is default
	m_fileType = VTK_FILETYPE;
	m_SampleType = NONE;
	m_Vtk_Dump_File = NULL;
	m_HDF5_Dump_File = NULL;
	SetPrecision(6);
	m_dualTime = false;
	m_numCalcThreads = 0;
	m_gpuCurr = false;
	m_gpuGather = -1;

	// dump box should be always inside the snapped lines
	m_SnapMethod = 1;

	for (int n=0; n<3; ++n)
	{
		numLines[n]=0;
		posLines[n]=NULL;
		discLines[n]=NULL;
		subSample[n]=1;
		optResolution[n]=0;
	}
}

ProcessFields::~ProcessFields()
{
	delete m_Vtk_Dump_File;
	m_Vtk_Dump_File = NULL;
	for (int n=0; n<3; ++n)
	{
		delete[] posLines[n];
		posLines[n]=NULL;
		delete[] discLines[n];
		discLines[n]=NULL;
	}
}

std::string ProcessFields::GetFieldNameByType(DumpType type)
{
	switch (type)
	{
	case E_FIELD_DUMP:
		return "E-Field";
	case H_FIELD_DUMP:
		return "H-Field";
	case J_FIELD_DUMP:
		return "J-Field";
	case ROTH_FIELD_DUMP:
		return "RotH-Field";
	case D_FIELD_DUMP:
		return "D-Field";
	case B_FIELD_DUMP:
		return "B-Field";
	case SAR_LOCAL_DUMP:
		return "SAR-local";
	case SAR_1G_DUMP:
		return "SAR_1g";
	case SAR_10G_DUMP:
		return "SAR_10g";
	case SAR_RAW_DATA:
		return "SAR_raw_data";
	}
	return "unknown field";
}

bool ProcessFields::NeedConductivity() const
{
	switch (m_DumpType)
	{
	case J_FIELD_DUMP:
		return true;
	default:
		return false;
	}
	return false;
}

bool ProcessFields::NeedPermittivity() const
{
	switch (m_DumpType)
	{
	case D_FIELD_DUMP:
		return true;
	default:
		return false;
	}
	return false;
}

bool ProcessFields::NeedPermeability() const
{
	switch (m_DumpType)
	{
	case B_FIELD_DUMP:
		return true;
	default:
		return false;
	}
	return false;
}

void ProcessFields::InitProcess()
{
	if (Enabled==false) return;

	CalcMeshPos();

	if (m_fileType==VTK_FILETYPE)
	{
		delete m_Vtk_Dump_File;
		m_Vtk_Dump_File = new VTK_File_Writer(m_filename,(int)m_Mesh_Type);

		#ifdef OUTPUT_IN_DRAWINGUNITS
		double discScaling = 1;
		#else
		double discScaling = Op->GetGridDelta();
		#endif
		m_Vtk_Dump_File->SetMeshLines(discLines,numLines,discScaling);
		m_Vtk_Dump_File->SetNativeDump(g_settings.NativeFieldDumps());
	}
	if (m_fileType==HDF5_FILETYPE)
	{
		delete m_HDF5_Dump_File;
		m_HDF5_Dump_File = new HDF5_File_Writer(m_filename+".h5");

		#ifdef OUTPUT_IN_DRAWINGUNITS
		double discScaling = 1;
		#else
		double discScaling = Op->GetGridDelta();
		#endif
		m_HDF5_Dump_File->WriteRectMesh(numLines,discLines,(int)m_Mesh_Type,discScaling);

		m_HDF5_Dump_File->WriteAttribute("/", "openEMS_HDF5_version", OPENEMS_HDF5_VERSION);
		m_HDF5_Dump_File->WriteAttribute("/", "dump_type", m_DumpType);
		m_HDF5_Dump_File->WriteAttribute("/", "dual_mesh", m_dualMesh);
		if ((m_DumpType!=SAR_RAW_DATA) && (g_settings.GetLegacyHDF5Dumps()))
			m_HDF5_Dump_File->WriteAttribute("/", "legacy_fmt", true);
	}
}

void ProcessFields::SetDumpMode(Engine_Interface_Base::InterpolationType mode)
{
	m_Eng_Interface->SetInterpolationType(mode);
	if (mode==Engine_Interface_Base::CELL_INTERPOLATE)
		m_dualMesh=true;
	else if (mode==Engine_Interface_Base::NODE_INTERPOLATE)
		m_dualMesh=false;
	//else keep the preset/user defined case
}

void ProcessFields::DefineStartStopCoord(double* dstart, double* dstop)
{
	Processing::DefineStartStopCoord(dstart,dstop);

	// normalize order of start and stop
	for (int n=0; n<3; ++n)
	{
		if (start[n]>stop[n])
		{
			unsigned int help = start[n];
			start[n]=stop[n];
			stop[n]=help;
		}
	}
}

double ProcessFields::CalcTotalEnergyEstimate() const
{
	return m_Eng_Interface->CalcFastEnergy();
}

void ProcessFields::SetSubSampling(unsigned int subSampleRate, int dir)
{
	if (dir>2) return;
	if (dir<0)
	{
		subSample[0]=subSampleRate;
		subSample[1]=subSampleRate;
		subSample[2]=subSampleRate;
	}
	else subSample[dir]=subSampleRate;
	m_SampleType = SUBSAMPLE;
}

void ProcessFields::SetOptResolution(double optRes, int dir)
{
	if (dir>2) return;
	if (dir<0)
	{
		optResolution[0]=optRes;
		optResolution[1]=optRes;
		optResolution[2]=optRes;
	}
	else optResolution[dir]=optRes;
	m_SampleType = OPT_RESOLUTION;
}

void ProcessFields::CalcMeshPos()
{
	if ((m_SampleType==SUBSAMPLE) || (m_SampleType==NONE))
	{
		std::vector<unsigned int> tmp_pos;

		for (int n=0; n<3; ++n)
		{
			// construct new discLines
			tmp_pos.clear();
			for (unsigned int i=start[n]; i<=stop[n]; i+=subSample[n])
				tmp_pos.push_back(i);

			numLines[n] = tmp_pos.size();
			delete[] discLines[n];
			discLines[n] = new double[numLines[n]];
			delete[] posLines[n];
			posLines[n] = new unsigned int[numLines[n]];
			for (unsigned int i=0; i<numLines[n]; ++i)
			{
				posLines[n][i] = tmp_pos.at(i);
				discLines[n][i] = Op->GetDiscLine(n,tmp_pos.at(i),m_dualMesh);
			}
		}
	}
	if ((m_SampleType==OPT_RESOLUTION))
	{
		std::vector<unsigned int> tmp_pos;
		double oldPos=0;
		for (int n=0; n<3; ++n)
		{
			// construct new discLines
			tmp_pos.clear();
			tmp_pos.push_back(start[n]);
			oldPos=Op->GetDiscLine(n,start[n],m_dualMesh);
			if (stop[n]==0)
				tmp_pos.push_back(stop[n]);
			else
				for (unsigned int i=start[n]+1; i<=stop[n]-1; ++i)
				{
					if ( (Op->GetDiscLine(n,i+1,m_dualMesh)-oldPos) >= optResolution[n])
					{
						tmp_pos.push_back(i);
						oldPos=Op->GetDiscLine(n,i,m_dualMesh);
					}
				}
			if (start[n]!=stop[n])
				tmp_pos.push_back(stop[n]);
			numLines[n] = tmp_pos.size();
			delete[] discLines[n];
			discLines[n] = new double[numLines[n]];
			delete[] posLines[n];
			posLines[n] = new unsigned int[numLines[n]];
			for (unsigned int i=0; i<numLines[n]; ++i)
			{
				posLines[n][i] = tmp_pos.at(i);
				discLines[n][i] = Op->GetDiscLine(n,tmp_pos.at(i),m_dualMesh);
			}
		}
	}
}

unsigned int ProcessFields::CalcThreads() const
{
	unsigned int nThreads = m_numCalcThreads;
	if (nThreads == 0)
	{
		nThreads = std::thread::hardware_concurrency();
		if (nThreads == 0)
			nThreads = 1;
	}
	return nThreads;
}

static FieldCalc::MaterialContext MaterialContextOf(const Engine_Interface_FDTD* fdtd_if)
{
	FieldCalc::MaterialContext mat;
	mat.kappa = fdtd_if->GetKappaPtr();
	mat.epsR = fdtd_if->GetEpsRPtr();
	mat.mueR = fdtd_if->GetMueRPtr();
	return mat;
}

Engine_Vulkan* ProcessFields::GetVulkanEngine() const
{
#ifdef WITH_GPU
	const Engine_Interface_FDTD* fdtd_if = dynamic_cast<const Engine_Interface_FDTD*>(m_Eng_Interface);
	if (fdtd_if)
		return dynamic_cast<Engine_Vulkan*>(const_cast<Engine*>(fdtd_if->GetFDTDEngine()));
#endif
	return NULL;
}

// The offsets, along every direction, from a dumped line to the raw lines the
// extraction in processfields_calc.h reads for it: the union over the three
// components of the interpolation stencil and, for rotH, of the curl.
static bool GPUStencil(ProcessFields::DumpType type, Engine_Interface_Base::InterpolationType interp,
                       bool& curr, int& lo, int& hi)
{
	switch (type)
	{
	case ProcessFields::E_FIELD_DUMP:
	case ProcessFields::J_FIELD_DUMP:
	case ProcessFields::D_FIELD_DUMP:
		curr = false;
		break;
	case ProcessFields::ROTH_FIELD_DUMP:
	case ProcessFields::H_FIELD_DUMP:
	case ProcessFields::B_FIELD_DUMP:
		curr = true;
		break;
	default:
		return false;
	}
	switch (interp)
	{
	case Engine_Interface_Base::NODE_INTERPOLATE:
		lo = -1; hi = 0;  // E: pos and pos-e_n; H: pos back to pos-e_nP-e_nPP
		break;
	case Engine_Interface_Base::CELL_INTERPOLATE:
		lo = 0; hi = 1;   // E: pos up to pos+e_nP+e_nPP; H: pos and pos+e_n
		break;
	default:
		lo = 0; hi = 0;
		break;
	}
	if (type==ProcessFields::ROTH_FIELD_DUMP)
		lo = -1;          // the curl reaches back one line along nP and nPP
	return true;
}

bool ProcessFields::SetupGPUFootprint(DumpType type)
{
	int lo, hi;
	if (!GPUStencil(type, m_Eng_Interface->GetInterpolationType(), m_gpuCurr, lo, hi))
		return false;
	for (int d=0; d<3; ++d)
	{
		const long gridLines = (long)Op->GetNumberOfLines(d, true);
		std::vector<char> used(gridLines, 0);
		for (unsigned int i=0; i<numLines[d]; ++i)
			for (int o=lo; o<=hi; ++o)
			{
				const long l = (long)posLines[d][i] + o;
				if ((l>=0) && (l<gridLines))
					used[l] = 1;
			}
		m_gpuLines[d].clear();
		m_gpuMap[d].assign(gridLines, -1);
		for (long l=0; l<gridLines; ++l)
			if (used[l])
			{
				m_gpuMap[d][l] = (int)m_gpuLines[d].size();
				m_gpuLines[d].push_back((unsigned int)l);
			}
		if (m_gpuLines[d].empty())
			return false;
	}
	return true;
}

void ProcessFields::RegisterGPUGather(DumpType type)
{
	m_gpuGather = -1;
#ifdef WITH_GPU
	Engine_Vulkan* vk = GetVulkanEngine();
	if ((vk==NULL) || !SetupGPUFootprint(type))
		return;
	Engine_Vulkan::DumpFootprint fp;
	fp.curr = m_gpuCurr;
	for (int d=0; d<3; ++d)
		fp.lines[d].assign(m_gpuLines[d].begin(), m_gpuLines[d].end());
	m_gpuGather = vk->RegisterDumpGather(fp, [this](unsigned int ts) {return IsDueAt(ts);});
#else
	(void)type;
#endif
}

bool ProcessFields::CalcFieldFromFootprint(ArrayLib::ArrayNIJK<FDTD_FLOAT> &field, DumpType type, const float* data, unsigned int stride)
{
	Engine_Interface_FDTD* fdtd_if = dynamic_cast<Engine_Interface_FDTD*>(m_Eng_Interface);
	if ((fdtd_if==NULL) || (data==NULL))
		return false;
	field.Init("Field", numLines);
	FieldCalc::FootprintView view;
	view.data = data;
	view.stride = stride;
	view.curr = m_gpuCurr;
	for (int d=0; d<3; ++d)
		view.map[d] = m_gpuMap[d].data();
	view.sb = m_gpuLines[2].size();
	view.sa = m_gpuLines[1].size() * view.sb;
	view.sn = m_gpuLines[0].size() * view.sa;
	const FieldCalc::MaterialContext mat = MaterialContextOf(fdtd_if);
	if (!FieldCalc::CalcFieldForEngine<FieldCalc::FootprintView>(
			&view, fdtd_if->GetFDTDOperator(), mat, type, m_Eng_Interface->GetInterpolationType(),
			numLines, posLines, field, CalcThreads()))
		return false;
	if (view.missed)
	{
		cerr << "ProcessFields::CalcField: " << m_Name << ": the GPU footprint lacks a cell the dump reads" << endl;
		return false;
	}
	return true;
}

bool ProcessFields::CalcField(ArrayLib::ArrayNIJK<FDTD_FLOAT> &field)
{
	field.Init("Field", numLines);

	Engine_Interface_FDTD* fdtd_if = dynamic_cast<Engine_Interface_FDTD*>(m_Eng_Interface);
	if (fdtd_if)
	{
		const Engine* eng = fdtd_if->GetFDTDEngine();
		const Operator* op = fdtd_if->GetFDTDOperator();
		const FieldCalc::MaterialContext mat = MaterialContextOf(fdtd_if);
		const unsigned int nThreads = CalcThreads();

		bool ok = false;
		switch (eng->GetType())
		{
		case Engine::SSE:
			ok = FieldCalc::CalcFieldForEngine<Engine_sse>(
				static_cast<const Engine_sse*>(eng), op, mat, m_DumpType,
				m_Eng_Interface->GetInterpolationType(), numLines, posLines, field, nThreads);
			break;
		case Engine::AVX2:
			// Covers both Engine_AVX2 and Engine_AVX2_Multithread -- they share
			// this EngineType tag and Engine_AVX2_Multithread adds no new field
			// accessors, so the base Engine_AVX2 scalar GetVolt()/GetCurr() cover
			// both. Without this case, AVX2 fell through to the generic
			// per-cell virtual-dispatch loop below: single-threaded, and each
			// scalar access pays a runtime %/ against numVectors. Measured ~32%
			// slower overall than no-dump on a full-domain every-timestep E-field
			// dump (vs. ~12% for SSE with this fast path), because AVX2 was the
			// only multithreaded engine still hitting that fallback.
			ok = FieldCalc::CalcFieldForEngine<Engine_AVX2>(
				static_cast<const Engine_AVX2*>(eng), op, mat, m_DumpType,
				m_Eng_Interface->GetInterpolationType(), numLines, posLines, field, nThreads);
			break;
#ifdef WITH_GPU
		case Engine::GPU:
		{
			Engine_Vulkan* vk = const_cast<Engine_Vulkan*>(static_cast<const Engine_Vulkan*>(eng));
			const unsigned int ts = m_Eng_Interface->GetNumberOfTimesteps();
			// The batch that ended at ts gathered the cells this dump reads,
			// unless the host copy holds ts anyway, as at timestep 0.
			if ((m_gpuGather>=0) && !vk->HostFieldsAt(ts))
			{
				if (CalcFieldFromFootprint(field, m_DumpType, vk->AcquireDumpGather(m_gpuGather, ts), 1))
					return true;
				field.Init("Field", numLines);
			}
			// The templated extractor intentionally uses qualified base-Engine
			// accessors for thread-safe bulk reads.  Synchronize the Vulkan field
			// buffers first so those reads do not see a stale host copy.
			vk->SyncFieldsToHost();
			ok = FieldCalc::CalcFieldForEngine<Engine>(
				eng, op, mat,
				m_DumpType, m_Eng_Interface->GetInterpolationType(),
				numLines, posLines, field, nThreads);
			break;
		}
#endif
		case Engine::BASIC:
			ok = FieldCalc::CalcFieldForEngine<Engine>(
				eng, op, mat, m_DumpType, m_Eng_Interface->GetInterpolationType(),
				numLines, posLines, field, nThreads);
			break;
		default:
			break;
		}
		if (ok)
			return true;
	}

	unsigned int pos[3];
	double out[3];
	switch (m_DumpType)
	{
	case E_FIELD_DUMP:
		for (unsigned int i=0; i<numLines[0]; ++i)
		{
			pos[0]=posLines[0][i];
			for (unsigned int j=0; j<numLines[1]; ++j)
			{
				pos[1]=posLines[1][j];
				for (unsigned int k=0; k<numLines[2]; ++k)
				{
					pos[2]=posLines[2][k];

					m_Eng_Interface->GetEField(pos,out);
					field(0, i, j, k) = out[0];
					field(1, i, j, k) = out[1];
					field(2, i, j, k) = out[2];
				}
			}
		}
		return true;
	case H_FIELD_DUMP:
		for (unsigned int i=0; i<numLines[0]; ++i)
		{
			pos[0]=posLines[0][i];
			for (unsigned int j=0; j<numLines[1]; ++j)
			{
				pos[1]=posLines[1][j];
				for (unsigned int k=0; k<numLines[2]; ++k)
				{
					pos[2]=posLines[2][k];

					m_Eng_Interface->GetHField(pos,out);
					field(0, i, j, k) = out[0];
					field(1, i, j, k) = out[1];
					field(2, i, j, k) = out[2];
				}
			}
		}
		return true;
	case J_FIELD_DUMP:
		for (unsigned int i=0; i<numLines[0]; ++i)
		{
			pos[0]=posLines[0][i];
			for (unsigned int j=0; j<numLines[1]; ++j)
			{
				pos[1]=posLines[1][j];
				for (unsigned int k=0; k<numLines[2]; ++k)
				{
					pos[2]=posLines[2][k];

					m_Eng_Interface->GetJField(pos,out);
					field(0, i, j, k) = out[0];
					field(1, i, j, k) = out[1];
					field(2, i, j, k) = out[2];
				}
			}
		}
		return true;
	case ROTH_FIELD_DUMP:
		for (unsigned int i=0; i<numLines[0]; ++i)
		{
			pos[0]=posLines[0][i];
			for (unsigned int j=0; j<numLines[1]; ++j)
			{
				pos[1]=posLines[1][j];
				for (unsigned int k=0; k<numLines[2]; ++k)
				{
					pos[2]=posLines[2][k];

					m_Eng_Interface->GetRotHField(pos,out);
					field(0, i, j, k) = out[0];
					field(1, i, j, k) = out[1];
					field(2, i, j, k) = out[2];
				}
			}
		}
		return true;
	case D_FIELD_DUMP:
		for (unsigned int i=0; i<numLines[0]; ++i)
		{
			pos[0]=posLines[0][i];
			for (unsigned int j=0; j<numLines[1]; ++j)
			{
				pos[1]=posLines[1][j];
				for (unsigned int k=0; k<numLines[2]; ++k)
				{
					pos[2]=posLines[2][k];

					m_Eng_Interface->GetDField(pos,out);
					field(0, i, j, k) = out[0];
					field(1, i, j, k) = out[1];
					field(2, i, j, k) = out[2];
				}
			}
		}
		return true;
	case B_FIELD_DUMP:
		for (unsigned int i=0; i<numLines[0]; ++i)
		{
			pos[0]=posLines[0][i];
			for (unsigned int j=0; j<numLines[1]; ++j)
			{
				pos[1]=posLines[1][j];
				for (unsigned int k=0; k<numLines[2]; ++k)
				{
					pos[2]=posLines[2][k];

					m_Eng_Interface->GetBField(pos,out);
					field(0, i, j, k) = out[0];
					field(1, i, j, k) = out[1];
					field(2, i, j, k) = out[2];
				}
			}
		}
		return true;
	default:
		cerr << "ProcessFields::CalcField(): Error, unknown dump type..." << endl;
		return false;
	}
}
