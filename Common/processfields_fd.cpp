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

#include "processfields_fd.h"
#include "Common/operator_base.h"
#ifdef WITH_GPU
#include "FDTD/engine_vulkan.h"
#endif
#include "tools/vtk_file_writer.h"
#include "tools/hdf5_file_writer.h"
#include <iomanip>
#include <sstream>
#include <string>

using namespace std;

ProcessFieldsFD::ProcessFieldsFD(Engine_Interface_Base* eng_if) : ProcessFields(eng_if)
{
	m_gpuDFT = -1;
}

ProcessFieldsFD::~ProcessFieldsFD()
{
	for (size_t n = 0; n<m_FD_Fields.size(); ++n)
	{
		delete m_FD_Fields.at(n);
	}
	m_FD_Fields.clear();
}

void ProcessFieldsFD::InitProcess()
{
	if (Enabled==false) return;

	if (m_FD_Samples.size()==0)
	{
		cerr << "ProcessFieldsFD::InitProcess: No frequencies found... skipping this dump!" << endl;
		Enabled=false;
		return;
	}

	//setup the hdf5 file
	ProcessFields::InitProcess();

	if (m_Vtk_Dump_File)
		m_Vtk_Dump_File->SetHeader(string("openEMS FD Field Dump -- Interpolation: ")+m_Eng_Interface->GetInterpolationTypeString());

	if (m_HDF5_Dump_File)
	{
		m_HDF5_Dump_File->SetCurrentGroup("/FieldData/FD");
		m_HDF5_Dump_File->WriteAttribute("/FieldData/FD","frequency",m_FD_Samples);
	}

	//create data structures...
	for (size_t n = 0; n<m_FD_Samples.size(); ++n)
	{
		ArrayLib::ArrayNIJK<std::complex<float>>* field_fd = new ArrayLib::ArrayNIJK<std::complex<float>>("FD field", numLines);
		m_FD_Fields.push_back(field_fd);
	}

	RegisterGPUDump(m_DumpType);
}

bool ProcessFieldsFD::IsDueAt(unsigned int ts) const
{
	return ProcessFields::IsDueAt(ts) && (m_FD_Interval!=0) && (ts%m_FD_Interval==0);
}

std::complex<float> ProcessFieldsFD::DFTWeight(size_t n, double T) const
{
	std::complex<float> exp_jwt_2_dt = std::exp( (std::complex<float>)(-2.0 * I_UNIT * PI * m_FD_Samples.at(n) * T) );
	exp_jwt_2_dt *= 2; // *2 for single-sided spectrum
	exp_jwt_2_dt *= Op->GetTimestep() * m_FD_Interval; // multiply with timestep-interval
	return exp_jwt_2_dt;
}

void ProcessFieldsFD::RegisterGPUDump(DumpType type)
{
	m_gpuDFT = -1;
#ifdef WITH_GPU
	Engine_Vulkan* vk = GetVulkanEngine();
	if ((vk==NULL) || !SetupGPUFootprint(type))
		return;
	Engine_Vulkan::DumpFootprint fp;
	fp.curr = m_gpuCurr;
	for (int d=0; d<3; ++d)
		fp.lines[d].assign(m_gpuLines[d].begin(), m_gpuLines[d].end());
	m_gpuDFT = vk->RegisterDumpDFT(fp, (unsigned int)m_FD_Samples.size(),
		[this](unsigned int ts) {return IsDueAt(ts);},
		[this](unsigned int ts, std::complex<float>* w)
		{
			// Engine_Interface_FDTD::GetTime() at timestep ts
			const double T = ((double)ts + (double)m_dualTime*0.5) * Op->GetTimestep();
			for (size_t n=0; n<m_FD_Samples.size(); ++n)
				w[n] = DFTWeight(n, T);
		});
	if (m_gpuDFT<0)
		RegisterGPUGather(type);
#else
	(void)type;
#endif
}

bool ProcessFieldsFD::ReadGPUDFT(std::vector<std::complex<float>>& sums)
{
#ifdef WITH_GPU
	unsigned int samples = 0;
	Engine_Vulkan* vk = GetVulkanEngine();
	if ((vk==NULL) || !vk->ReadDumpDFT(m_gpuDFT, sums, samples))
	{
		cerr << "ProcessFieldsFD: " << m_Name << ": cannot read the GPU's DFT sums" << endl;
		return false;
	}
	if (samples!=m_FD_SampleCount)
		cerr << "ProcessFieldsFD: " << m_Name << ": warning: the GPU summed " << samples
		     << " samples, the dump processed " << m_FD_SampleCount << endl;
	return true;
#else
	(void)sums;
	return false;
#endif
}

bool ProcessFieldsFD::InterpolateGPUDFT(const std::vector<std::complex<float>>& sums, DumpType type,
                                        std::vector<ArrayLib::ArrayNIJK<std::complex<float>>*>& fields)
{
	// Interpolation is linear, so interpolating the sums of the raw values
	// equals summing the interpolated fields, up to rounding.
	const size_t count = 3 * m_gpuLines[0].size() * m_gpuLines[1].size() * m_gpuLines[2].size();
	if (sums.size() < count * fields.size())
		return false;
	ArrayLib::ArrayNIJK<FDTD_FLOAT> re, im;
	for (size_t n = 0; n<fields.size(); ++n)
	{
		const float* sum = reinterpret_cast<const float*>(sums.data() + n * count);
		if (!CalcFieldFromFootprint(re, type, sum, 2) || !CalcFieldFromFootprint(im, type, sum + 1, 2))
			return false;
		std::complex<float>* field_fd = fields.at(n)->data();
		const FDTD_FLOAT* field_re = re.data();
		const FDTD_FLOAT* field_im = im.data();
		const size_t N = fields.at(n)->size();
		for (size_t ijk=0; ijk<N; ++ijk)
			field_fd[ijk] = std::complex<float>(field_re[ijk], field_im[ijk]);
	}
	return true;
}

int ProcessFieldsFD::ProcessAt(unsigned int ts, size_t& psPos, bool& readsFields) const
{
	// as Process(), and ProcessFieldsSAR::Process(): the fields are read at
	// the samples of the DFT only
	readsFields = false;
	if (Enabled==false) return -1;
	readsFields = CheckTimestepAt(ts, psPos) && (m_FD_Interval!=0) && (ts%m_FD_Interval==0);
	return NextIntervalAt(ts, psPos);
}

int ProcessFieldsFD::Process()
{
	if (Enabled==false) return -1;
	if (CheckTimestep()==false) return GetNextInterval();

	if ((m_FD_Interval==0) || (m_Eng_Interface->GetNumberOfTimesteps()%m_FD_Interval!=0))
		return GetNextInterval();

#ifdef WITH_GPU
	if (m_gpuDFT>=0)
	{
		// The GPU adds the sample to its sums; DumpFDData() interpolates them.
		if (GetVulkanEngine()->AddDumpDFTSample(m_gpuDFT, m_Eng_Interface->GetNumberOfTimesteps()))
			++m_FD_SampleCount;
		return GetNextInterval();
	}
#endif

	ArrayLib::ArrayNIJK<FDTD_FLOAT> tmp_field_td;
	if (!CalcField(tmp_field_td))
		return -1;
	FDTD_FLOAT* field_td = tmp_field_td.data();

	std::complex<float>* field_fd = NULL;

	double T = m_Eng_Interface->GetTime(m_dualTime);
	for (size_t n = 0; n<m_FD_Samples.size(); ++n)
	{
		std::complex<float> exp_jwt_2_dt = DFTWeight(n, T);
		unsigned int N = m_FD_Fields.at(n)->size();
		field_fd = m_FD_Fields.at(n)->data();
		for (unsigned int ijk=0;ijk<N;++ijk)
			field_fd[ijk] += field_td[ijk] * exp_jwt_2_dt;
	}
	++m_FD_SampleCount;
	return GetNextInterval();
}

void ProcessFieldsFD::PostProcess()
{
	DumpFDData();
}

void ProcessFieldsFD::DumpFDData()
{
	if (m_gpuDFT>=0)
	{
		std::vector<std::complex<float>> sums;
		if (!ReadGPUDFT(sums) || !InterpolateGPUDFT(sums, m_DumpType, m_FD_Fields))
			cerr << "ProcessFieldsFD::DumpFDData: " << m_Name << ": the GPU's DFT sums are lost" << endl;
	}

	if (m_fileType==VTK_FILETYPE)
	{
		ArrayLib::ArrayNIJK<float> tmp_field("field", numLines);
		std::complex<float>* field_fd = NULL;
		float* field = tmp_field.data();
		double angle=0;
		int Nr_Ph = 21;

		for (size_t n = 0; n<m_FD_Samples.size(); ++n)
		{
			std::string str_freq;
			double freq = m_FD_Samples.at(n);
			if ((freq-long(freq))==0)
				// if the frequency is a natural number, export as long
				str_freq = std::to_string(long(freq));
			else
				str_freq = std::to_string(freq);

			unsigned int N = m_FD_Fields.at(n)->size();
			field_fd = m_FD_Fields.at(n)->data();
			//dump multiple phase to vtk-files
			for (int p=0; p<Nr_Ph; ++p)
			{
				angle = 2.0 * PI * p / Nr_Ph;
				std::complex<float> exp_jwt = std::exp( (std::complex<float>)(I_UNIT * angle) );
				for (unsigned int ijk=0;ijk<N;++ijk)
					field[ijk] = real(field_fd[ijk] * exp_jwt);

				stringstream ss;
				ss << m_filename << fixed << "_f=" << str_freq << "_p=" << std::setw( 3 ) << std::setfill( '0' ) <<(int)(angle * 180 / PI);

				m_Vtk_Dump_File->SetFilename(ss.str());
				m_Vtk_Dump_File->ClearAllFields();
				m_Vtk_Dump_File->AddVectorField(GetFieldNameByType(m_DumpType),tmp_field);
				if (m_Vtk_Dump_File->Write()==false)
					cerr << "ProcessFieldsFD::Process: can't dump to file... abort! " << endl;
			}

			{
				//dump magnitude to vtk-files
				for (unsigned int ijk=0;ijk<N;++ijk)
					field[ijk] = abs(field_fd[ijk]);

				stringstream ss;
				ss << m_filename << fixed << "_f=" << str_freq << "_abs";
				m_Vtk_Dump_File->SetFilename(ss.str());
				m_Vtk_Dump_File->ClearAllFields();
				m_Vtk_Dump_File->AddVectorField(GetFieldNameByType(m_DumpType),tmp_field);
				if (m_Vtk_Dump_File->Write()==false)
					cerr << "ProcessFieldsFD::Process: can't dump to file... abort! " << endl;
			}

			{
				//dump phase to vtk-files
				for (unsigned int ijk=0;ijk<N;++ijk)
					field[ijk] = arg(field_fd[ijk]);

				stringstream ss;
				ss << m_filename << fixed << "_f=" << str_freq << "_arg";
				m_Vtk_Dump_File->SetFilename(ss.str());
				m_Vtk_Dump_File->ClearAllFields();
				m_Vtk_Dump_File->AddVectorField(GetFieldNameByType(m_DumpType),tmp_field);
				if (m_Vtk_Dump_File->Write()==false)
					cerr << "ProcessFieldsFD::Process: can't dump to file... abort! " << endl;
			}
		}
		return;
	}

	if (m_fileType==HDF5_FILETYPE)
	{
		for (size_t n = 0; n<m_FD_Samples.size(); ++n)
		{
			stringstream ss;
			ss << "f" << n;
			// size_t datasize[]={numLines[0],numLines[1],numLines[2]};
			if (m_HDF5_Dump_File->WriteVectorField<std::complex<float>>(ss.str(), *m_FD_Fields.at(n), g_settings.GetLegacyHDF5Dumps())==false)
				cerr << "ProcessFieldsFD::Process: can't dump to file...! " << endl;

			if (g_settings.GetLegacyHDF5Dumps())
			{
				if (m_HDF5_Dump_File->WriteAttribute("/FieldData/FD/"+ss.str()+"_real", "frequency", m_FD_Samples.at(n))==false)
					cerr << "ProcessFieldsFD::Process: can't dump to file...! " << endl;
				if (m_HDF5_Dump_File->WriteAttribute("/FieldData/FD/"+ss.str()+"_imag", "frequency", m_FD_Samples.at(n))==false)
					cerr << "ProcessFieldsFD::Process: can't dump to file...! " << endl;
			}
			else if (m_HDF5_Dump_File->WriteAttribute("/FieldData/FD/"+ss.str(), "frequency", m_FD_Samples.at(n))==false)
				cerr << "ProcessFieldsFD::Process: can't dump to file...! " << endl;
		}
		return;
	}

	cerr << "ProcessFieldsFD::Process: unknown File-Type" << endl;
}
