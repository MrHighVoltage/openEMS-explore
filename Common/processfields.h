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

#ifndef PROCESSFIELDS_H
#define PROCESSFIELDS_H

#include "processing.h"
#include "tools/array_ops.h"
#include "tools/arraylib/array_nijk.h"


class VTK_File_Writer;
class HDF5_File_Writer;
class Engine_Vulkan;

class ProcessFields : public Processing
{
public:
	ProcessFields(Engine_Interface_Base* eng_if);
	virtual ~ProcessFields();

	//! File type definition.
	enum FileType { VTK_FILETYPE, HDF5_FILETYPE};

	//! Dump type definitions.
	/*!
	  Current dump types are electric field (E_FIELD_DUMP), magnetic field (H_FIELD_DUMP),
	  (conduction) electric current density (kappa*E) (J_FIELD_DUMP) and total current density (rotH)
	  */
	enum DumpType { E_FIELD_DUMP=0, H_FIELD_DUMP=1, J_FIELD_DUMP=2, ROTH_FIELD_DUMP=3, D_FIELD_DUMP=4, B_FIELD_DUMP=5, SAR_LOCAL_DUMP=20, SAR_1G_DUMP=21, SAR_10G_DUMP=22, SAR_RAW_DATA=29};

	virtual std::string GetProcessingName() const {return "common field processing";}

	virtual void InitProcess();

	virtual void DefineStartStopCoord(double* dstart, double* dstop);

	//! Define a field dump sub sampling rate for a given direction (default: \a dir = -1 means all directions)
	virtual void SetSubSampling(unsigned int subSampleRate, int dir=-1);

	//! Define a field dump optimal resolution for a given direction (default: \a dir = -1 means all directions)
	virtual void SetOptResolution(double optRes, int dir=-1);

	//! Set the filename for a hdf5 data group file (HDF5 FileType only) \sa SetFileType()
	void SetFileName(std::string fn) {m_filename=fn;}
	std::string SetFileName() const {return m_filename;}

	//! Define the Dump-Mode
	void SetDumpMode(Engine_Interface_Base::InterpolationType mode);
	//! This method will dump all fields on a main cell node using 2 E-field and 4 H-fields per direction.
	void SetDumpMode2Node() {SetDumpMode(Engine_Interface_Base::NODE_INTERPOLATE);}
	//! This method will dump all fields in the center of a main cell (dual-node) using 4 E-field and 2 H-fields per direction.
	void SetDumpMode2Cell() {SetDumpMode(Engine_Interface_Base::CELL_INTERPOLATE);}

	//! Set dump type: 0 for E-fields, 1 for H-fields, 2 for D-fields, 3 for B-fields, 4 for J-fields, etc...
	virtual void SetDumpType(DumpType type) {m_DumpType=type;}

	double CalcTotalEnergyEstimate() const;

	void SetFileType(FileType fileType) {m_fileType=fileType;}

	static std::string GetFieldNameByType(DumpType type);

	virtual bool NeedConductivity() const;
	virtual bool NeedPermittivity() const;
	virtual bool NeedPermeability() const;

	void SetNumCalcThreads(unsigned int n) {m_numCalcThreads = n;}
	unsigned int GetNumCalcThreads() const {return m_numCalcThreads;}

	//! True if the GPU engine gathers the cells this dump reads, so it never needs the whole fields.
	virtual bool IsServedByGPU() const {return m_gpuGather>=0;}

protected:
	DumpType m_DumpType;
	FileType m_fileType;

	VTK_File_Writer* m_Vtk_Dump_File;
	HDF5_File_Writer* m_HDF5_Dump_File;

	enum SampleType {NONE, SUBSAMPLE, OPT_RESOLUTION} m_SampleType;
	virtual void CalcMeshPos();

	//! field dump sub-sampling (if enabled)
	unsigned int subSample[3];

	//! field dump optimal resolution (if enabled)
	double optResolution[3];

	//! dump mesh information
	unsigned int numLines[3];	//number of lines to dump
	unsigned int* posLines[3];	//grid positions to dump
	double* discLines[3];		//mesh disc lines to dump
	unsigned int m_numCalcThreads;
	unsigned int CalcThreads() const;

	//! Calculate the defined field.
	bool CalcField(ArrayLib::ArrayNIJK<FDTD_FLOAT> &field);

	//! GPU engine: the raw cells a dump reads (its footprint), all three
	//! components of one field on m_gpuLines; m_gpuMap[d][line] is a grid
	//! line's position in m_gpuLines[d], or -1.
	bool m_gpuCurr;
	std::vector<unsigned int> m_gpuLines[3];
	std::vector<int> m_gpuMap[3];
	//! The engine's id for the gathered footprint, -1 = none.
	int m_gpuGather;
	//! The Vulkan engine behind the interface, or NULL.
	Engine_Vulkan* GetVulkanEngine() const;
	//! Set up the footprint for dumping \a type; false if the GPU cannot serve that type.
	bool SetupGPUFootprint(DumpType type);
	//! Have the GPU gather the footprint for \a type at every timestep IsDueAt() reports.
	void RegisterGPUGather(DumpType type);
	//! CalcField() for \a type from a gathered footprint.  A \a stride of 2
	//! reads one part of interleaved complex values.
	bool CalcFieldFromFootprint(ArrayLib::ArrayNIJK<FDTD_FLOAT> &field, DumpType type, const float* data, unsigned int stride);
};

#endif // PROCESSFIELDS_H
