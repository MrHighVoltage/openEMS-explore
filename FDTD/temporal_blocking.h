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

#ifndef TEMPORAL_BLOCKING_H
#define TEMPORAL_BLOCKING_H

#include <string>

//! A trapezoidal temporal-blocking schedule, as asked for.
/*!
  From --gpu-temporal-block / --avx2-temporal-block or the environment
  variables of the same name (OPENEMS_GPU_TEMPORAL_BLOCK, ...). Engine_Vulkan
  and Engine_AVX2_Multithread resolve it against their grid; every schedule
  computes the same fields bit for bit as the flat sweep (PERFORMANCE.md §5.1).
*/
struct TemporalBlockingRequest
{
	enum Mode
	{
		AUTO,   //!< the engine's default: on for a grid larger than the cache, on CPU and GPU alike
		OFF,    //!< the flat sweep
		FORCED  //!< blocks of k timesteps
	};
	Mode mode = AUTO;
	unsigned int k = 0;        //!< FORCED: timesteps per block
	unsigned int W = 0;        //!< FORCED: tile width in x-lines; 0 = from the tile target
	unsigned int targetMB = 0; //!< GPU: MB of fields and coefficients per tile; 0 = per device class

	//! Read "auto", "off" (or "0"), "<k>" or "<k>:<W>"; false, and \a req untouched, on anything else.
	static bool Parse(const std::string& text, TemporalBlockingRequest& req);
	//! What Parse() reads back.
	std::string ToString() const;

	static TemporalBlockingRequest Off();
	static TemporalBlockingRequest Forced(unsigned int k, unsigned int W);
};

//! What an engine resolves a TemporalBlockingRequest to on its grid.
struct TemporalBlockingPlan
{
	bool active = false;
	unsigned int k = 0;  //!< timesteps per block
	unsigned int W = 0;  //!< tile width in x-lines
	double tileMB = 0;   //!< what one tile holds, as the engine counts it
	std::string detail;  //!< active: tiles, sizes and where they came from
	std::string note;    //!< active: an adjustment, e.g. k lowered to fit the tile
	std::string why;     //!< not active although asked for: the reason
};

#endif // TEMPORAL_BLOCKING_H
