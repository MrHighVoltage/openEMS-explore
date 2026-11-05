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

#include "processfields_td.h"
#include "async_field_writer.h"
#include "Common/operator_base.h"
#include "tools/vtk_file_writer.h"
#include "tools/hdf5_file_writer.h"
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>

using namespace std;

// The one thread that writes the files of all time-domain dumps, with
// m_asyncBuffers slots from each.  The dumps' own threads (m_GatherCalc)
// would otherwise take turns on HDF5, which serializes every call
// (HDF5_GlobalMutex()): every turn wakes a sleeping thread, and that takes
// longer than a small write.
static AsyncFieldWriter& FileWriter()
{
	static AsyncFieldWriter* writer = new AsyncFieldWriter(0);  // never deleted: no joining at exit
	return *writer;
}

ProcessFieldsTD::ProcessFieldsTD(Engine_Interface_Base* eng_if) : ProcessFields(eng_if)
{
	pad_length = 8;
	m_asyncBuffers = 4;
	// A/B gate: OPENEMS_ASYNC_DUMP_BUFFERS=0 falls back to synchronous writes.
	if (const char* env = std::getenv("OPENEMS_ASYNC_DUMP_BUFFERS"))
		m_asyncBuffers = (unsigned int)std::max(0, atoi(env));
	m_GatherCalc = NULL;
	m_fileWriterSlots = 0;
}

ProcessFieldsTD::~ProcessFieldsTD()
{
	// the queued writes use the file writers, which ProcessFields deletes
	delete m_GatherCalc;
	m_GatherCalc = NULL;
	if (m_fileWriterSlots > 0)
	{
		FileWriter().Flush();
		FileWriter().AddCapacity(-m_fileWriterSlots);
	}
}

void ProcessFieldsTD::InitProcess()
{
	if (Enabled==false) return;

	ProcessFields::InitProcess();

	if (m_Vtk_Dump_File)
		m_Vtk_Dump_File->SetHeader(string("openEMS TD Field Dump -- Interpolation: ")+m_Eng_Interface->GetInterpolationTypeString());

	if (m_HDF5_Dump_File)
	{
		m_HDF5_Dump_File->SetCurrentGroup("/FieldData/TD");
		m_HDF5_Dump_File->OpenFile();
	}

	RegisterGPUGather(m_DumpType);

	delete m_GatherCalc;
	m_GatherCalc = NULL;
	if ((m_asyncBuffers > 0) && (m_gpuGather >= 0))
		m_GatherCalc = new AsyncFieldWriter(m_asyncBuffers);
	FileWriter().AddCapacity((int)m_asyncBuffers - m_fileWriterSlots);
	m_fileWriterSlots = (int)m_asyncBuffers;
}

void ProcessFieldsTD::FlushData()
{
	if (m_GatherCalc)
		m_GatherCalc->Flush();
	if (m_asyncBuffers > 0)
		FileWriter().Flush();
	if (m_HDF5_Dump_File)
		m_HDF5_Dump_File->FlushFile();
}

int ProcessFieldsTD::Process()
{
	if (Enabled==false) return -1;
	if (CheckTimestep()==false) return GetNextInterval();

	string filename = m_filename;

	std::shared_ptr<AsyncFieldWriter::FieldArray> field = std::make_shared<AsyncFieldWriter::FieldArray>();
	// The cells the GPU gathered for this timestep, held: m_GatherCalc
	// interpolates them, while this thread drives the GPU on.
	std::shared_ptr<const float> gathered = m_GatherCalc ? HoldGPUFootprint() : nullptr;
	bool success = gathered || CalcField(*field);

	if (m_asyncBuffers > 0)
	{
		AsyncFieldWriter::WriteFunc write;
		if (m_fileType==VTK_FILETYPE)
		{
			unsigned int ts = m_Eng_Interface->GetNumberOfTimesteps();
			std::string fieldName = GetFieldNameByType(m_DumpType);
			VTK_File_Writer* vtkWriter = m_Vtk_Dump_File;
			write = [vtkWriter, ts, fieldName](AsyncFieldWriter::FieldArray &data) -> bool
				{
					vtkWriter->SetTimestep(ts);
					vtkWriter->ClearAllFields();
					vtkWriter->AddVectorField(fieldName, data);
					return vtkWriter->Write();
				};
		}
		else if (m_fileType==HDF5_FILETYPE)
		{
			int padLength = pad_length;
			unsigned int ts = m_Eng_Interface->GetNumberOfTimesteps();
			float time = (float)m_Eng_Interface->GetTime(m_dualTime);
			HDF5_File_Writer* hdf5Writer = m_HDF5_Dump_File;
			write = [hdf5Writer, padLength, ts, time](AsyncFieldWriter::FieldArray &data) -> bool
				{
					stringstream ss;
					ss << std::setw(padLength) << std::setfill('0') << ts;
					bool ok = hdf5Writer->WriteVectorField<float>(ss.str(), data, g_settings.GetLegacyHDF5Dumps());
					ok &= hdf5Writer->WriteAttribute("/FieldData/TD/"+ss.str(), "time", time);
					return ok;
				};
		}
		else
		{
			success = false;
			cerr << "ProcessFieldsTD::Process: unknown File-Type" << endl;
		}
		if (write && gathered)
		{
			const DumpType type = m_DumpType;
			m_GatherCalc->Submit(field, [this, type, gathered, field, write](AsyncFieldWriter::FieldArray &data) mutable -> bool
				{
					const bool ok = CalcFieldFromFootprint(data, type, gathered.get(), 1);
					gathered.reset();  // the GPU may reuse the ring slot
					if (ok)
						FileWriter().Submit(field, write);
					return ok;
				});
		}
		else if (write)
			FileWriter().Submit(field, write);
	}
	else if (m_fileType==VTK_FILETYPE)
	{
		m_Vtk_Dump_File->SetTimestep(m_Eng_Interface->GetNumberOfTimesteps());
		m_Vtk_Dump_File->ClearAllFields();
		m_Vtk_Dump_File->AddVectorField(GetFieldNameByType(m_DumpType), *field);
		success &= m_Vtk_Dump_File->Write();
	}
	else if (m_fileType==HDF5_FILETYPE)
	{
		stringstream ss;
		ss << std::setw( pad_length ) << std::setfill( '0' ) << m_Eng_Interface->GetNumberOfTimesteps();
		success &= m_HDF5_Dump_File->WriteVectorField<float>(ss.str(), *field, g_settings.GetLegacyHDF5Dumps());
		float time = (float)m_Eng_Interface->GetTime(m_dualTime);
		success &= m_HDF5_Dump_File->WriteAttribute("/FieldData/TD/"+ss.str(), "time", time);
	}
	else
	{
		success = false;
		cerr << "ProcessFieldsTD::Process: unknown File-Type" << endl;
	}

	if (success==false)
	{
		SetEnable(false);
		cerr << "ProcessFieldsTD::Process: can't dump to file... disabled! " << endl;
	}

	return GetNextInterval();
}
