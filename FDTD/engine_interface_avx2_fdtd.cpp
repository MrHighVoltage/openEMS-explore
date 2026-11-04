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

#include "engine_interface_avx2_fdtd.h"

using std::cerr;
using std::endl;

Engine_Interface_AVX2_FDTD::Engine_Interface_AVX2_FDTD(Operator_AVX2* op)
	: Engine_Interface_FDTD(op)
{
	m_Op_AVX2 = op;
	m_Eng_AVX2 = dynamic_cast<Engine_AVX2*>(m_Op_AVX2->GetEngine());
	if (m_Eng_AVX2 == nullptr)
	{
		cerr << "Engine_Interface_AVX2_FDTD::Engine_Interface_AVX2_FDTD: "
		     << "Error: AVX2 Engine is not set! Exit!" << endl;
		exit(1);
	}
}

Engine_Interface_AVX2_FDTD::~Engine_Interface_AVX2_FDTD()
{
	m_Op_AVX2 = nullptr;
	m_Eng_AVX2 = nullptr;
}

double Engine_Interface_AVX2_FDTD::CalcFastEnergy() const
{
	if (m_Eng_AVX2->GetType() != Engine::AVX2)
		return Engine_Interface_FDTD::CalcFastEnergy();
	// by rows, see Engine_AVX2::FastEnergy()
	return m_Eng_AVX2->FastEnergy();
}
