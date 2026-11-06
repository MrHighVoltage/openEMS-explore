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

#ifndef PROCESSFIELDS_FD_H
#define PROCESSFIELDS_FD_H

#include "processfields.h"
#include "tools/arraylib/array_nijk.h"
#include <complex>

class ProcessFieldsFD : public ProcessFields
{
public:
	ProcessFieldsFD(Engine_Interface_Base* eng_if);
	virtual ~ProcessFieldsFD();

	virtual std::string GetProcessingName() const {return "frequency domain field dump";}

	virtual void InitProcess();

	virtual int Process();
	virtual int ProcessAt(unsigned int ts, size_t& psPos, bool& readsFields) const;
	virtual void PostProcess();

	virtual bool IsDueAt(unsigned int ts) const;
	virtual bool IsServedByGPU() const {return (m_gpuDFT>=0) || ProcessFields::IsServedByGPU();}

protected:
	virtual void DumpFDData();

	//! The DFT weight of the sample at time \a T for frequency m_FD_Samples[n].
	std::complex<float> DFTWeight(size_t n, double T) const;

	//! The engine's id for the DFT sums the GPU keeps, -1 = none.
	int m_gpuDFT;
	//! Register a dump of \a type with the GPU engine: the DFT sums if they
	//! fit, else a gather of the footprint at every sample.
	void RegisterGPUDump(DumpType type);
	//! The GPU's DFT sums, [k][n][a][b][c] over the footprint.
	bool ReadGPUDFT(std::vector<std::complex<float>>& sums);
	//! The dump of \a type at every frequency, interpolated from \a sums.
	bool InterpolateGPUDFT(const std::vector<std::complex<float>>& sums, DumpType type,
	                       std::vector<ArrayLib::ArrayNIJK<std::complex<float>>*>& fields);

	//! frequency domain field storage
	std::vector<ArrayLib::ArrayNIJK<std::complex<float>>*> m_FD_Fields;
};

#endif // PROCESSFIELDS_FD_H
