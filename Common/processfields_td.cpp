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

ProcessFieldsTD::ProcessFieldsTD(Engine_Interface_Base* eng_if) : ProcessFields(eng_if)
{
	pad_length = 8;
	m_asyncBuffers = 4;
	// A/B gate: OPENEMS_ASYNC_DUMP_BUFFERS=0 falls back to synchronous writes.
	if (const char* env = std::getenv("OPENEMS_ASYNC_DUMP_BUFFERS"))
		m_asyncBuffers = (unsigned int)std::max(0, atoi(env));
	m_AsyncWriter = NULL;
}

ProcessFieldsTD::~ProcessFieldsTD()
{
	delete m_AsyncWriter;
	m_AsyncWriter = NULL;
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

	if (m_asyncBuffers > 0)
	{
		delete m_AsyncWriter;
		m_AsyncWriter = new AsyncFieldWriter(m_asyncBuffers);
	}

	RegisterGPUGather(m_DumpType);
}

void ProcessFieldsTD::FlushData()
{
	if (m_AsyncWriter)
		m_AsyncWriter->Flush();
	if (m_HDF5_Dump_File)
		m_HDF5_Dump_File->FlushFile();
}

int ProcessFieldsTD::Process()
{
	if (Enabled==false) return -1;
	if (CheckTimestep()==false) return GetNextInterval();

	string filename = m_filename;

	std::shared_ptr<AsyncFieldWriter::FieldArray> field =
		std::make_shared<AsyncFieldWriter::FieldArray>("TD_field", numLines);
	bool success = CalcField(*field);

	if (m_AsyncWriter)
	{
		if (m_fileType==VTK_FILETYPE)
		{
			unsigned int ts = m_Eng_Interface->GetNumberOfTimesteps();
			std::string fieldName = GetFieldNameByType(m_DumpType);
			VTK_File_Writer* vtkWriter = m_Vtk_Dump_File;
			m_AsyncWriter->Submit(field,
				[vtkWriter, ts, fieldName](AsyncFieldWriter::FieldArray &data) -> bool
				{
					vtkWriter->SetTimestep(ts);
					vtkWriter->ClearAllFields();
					vtkWriter->AddVectorField(fieldName, data);
					return vtkWriter->Write();
				});
		}
		else if (m_fileType==HDF5_FILETYPE)
		{
			int padLength = pad_length;
			unsigned int ts = m_Eng_Interface->GetNumberOfTimesteps();
			float time = (float)m_Eng_Interface->GetTime(m_dualTime);
			HDF5_File_Writer* hdf5Writer = m_HDF5_Dump_File;
			m_AsyncWriter->Submit(field,
				[hdf5Writer, padLength, ts, time](AsyncFieldWriter::FieldArray &data) -> bool
				{
					stringstream ss;
					ss << std::setw(padLength) << std::setfill('0') << ts;
					bool ok = hdf5Writer->WriteVectorField<float>(ss.str(), data, g_settings.GetLegacyHDF5Dumps());
					ok &= hdf5Writer->WriteAttribute("/FieldData/TD/"+ss.str(), "time", time);
					return ok;
				});
		}
		else
		{
			success = false;
			cerr << "ProcessFieldsTD::Process: unknown File-Type" << endl;
		}
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
