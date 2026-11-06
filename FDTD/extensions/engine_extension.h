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

#ifndef ENGINE_EXTENSION_H
#define ENGINE_EXTENSION_H

#define ENG_EXT_PRIO_DEFAULT 0 //default engine extension priority

// priority definitions for some important extensions
#define ENG_EXT_PRIO_STEADYSTATE		+2e6  //steady state extension priority
#define ENG_EXT_PRIO_UPML				+1e6  //unaxial pml extension priority
#define ENG_EXT_PRIO_CYLINDER			+1e5  //cylindrial extension priority
#define ENG_EXT_PRIO_TFSF				+5e4  //total-field/scattered-field extension priority
#define ENG_EXT_PRIO_EXCITATION			-1000 //excitation priority
#define ENG_EXT_PRIO_CYLINDERMULTIGRID	-3000 //cylindrial multi-grid extension priority

#include <string>

class Operator_Extension;
class Engine;

//! Abstract base-class for all engine extensions
class Engine_Extension
{
public:
	virtual ~Engine_Extension();

	virtual void SetNumberOfThreads(int nrThread);

	//! This method will be called __before__ the main engine does the usual voltage updates. This method may __not__ change the engine voltages!!!
	virtual void DoPreVoltageUpdates() {}
	virtual void DoPreVoltageUpdates(int threadID);
	//! This method will be called __after__ the main engine does the usual voltage updates. This method may __not__ change the engine voltages!!!
	virtual void DoPostVoltageUpdates() {}
	virtual void DoPostVoltageUpdates(int threadID);
	//! This method will be called __after__ all updates to the voltages and extensions and may add/set its results to the engine voltages, but may __not__ rely on the current value of the engine voltages!!!
	virtual void Apply2Voltages() {}
	virtual void Apply2Voltages(int threadID);

	//! This method will be called __before__ the main engine does the usual current updates. This method may __not__ change the engine current!!!
	virtual void DoPreCurrentUpdates() {}
	virtual void DoPreCurrentUpdates(int threadID);
	//! This method will be called __after__ the main engine does the usual current updates. This method may __not__ change the engine current!!!
	virtual void DoPostCurrentUpdates() {}
	virtual void DoPostCurrentUpdates(int threadID);
	//! This method will be called __after__ all updates to the current and extensions and may add/set its results to the engine current, but may __not__ rely on the current value of the engine current!!!
	virtual void Apply2Current() {}
	virtual void Apply2Current(int threadID);

	// ------------------------------------------------------------------------
	// Slab interface -- the temporally blocked schedule
	//
	// Temporal blocking (Engine_AVX2_Multithread's trapezoidal path,
	// OPTIMIZATIONS.md 4.7/C2) advances a tile of x-lines through several
	// timesteps before moving on, so at any moment different parts of the grid
	// are at different timesteps. An extension that can only be told "apply
	// yourself to everything, at whatever timestep the engine says it is on"
	// cannot participate; the engine then falls back to the flat sweep, so an
	// extension is safe by omission.
	//
	// The six hooks below mirror the six flat ones one-for-one and are called
	// in the same order at the same points, once per timestep per tile, on
	// every worker thread, barrier-separated. Implementing them means honouring
	// this contract:
	//
	//   * Write only to cells with x in [startX,stopX). Anything outside is at
	//     a different timestep and would be corrupted.
	//   * Read only cells in that same range, which are all at timestep numTS.
	//     A boundary-anchored extension is the exception, and the schedule
	//     guarantees exactly this much for it: whenever a slab contains x=0 or
	//     x=NX-1, it also contains the two cells inward from that edge, at the
	//     same timestep. That covers Mur (one cell in) and an absorbing sheet
	//     (two, for its current shift). It does *not* extend to an edge-shaped
	//     extension sitting anywhere else -- an interior sheet normal to x is
	//     cut by a sweeping tile boundary and cannot be slabbed.
	//   * Take the timestep from \a numTS. The engine's own counter only
	//     advances at block boundaries and is wrong inside one.
	//   * Partition the work across threadID in [0,GetNumberOfThreads()) and
	//     touch nothing another thread will touch. The engine barriers between
	//     extensions, not within one.
	//   * Carry no state that depends on call *order*. Ring buffers rotated
	//     once per call, scratch buffers rebuilt per call, and similar are all
	//     broken by a schedule that visits the same timestep many times for
	//     different slabs; index such state by numTS instead.
	//   * Write voltages only in the three voltage hooks and currents only in
	//     the three current hooks. In-batch sampling takes a slab's voltages
	//     of a timestep once its Apply2VoltagesSlab() is done and its currents
	//     once its Apply2CurrentSlab() is (Engine_AVX2_Multithread::
	//     SampleVoltages(), plans/BATCH_SAMPLING.md §4.4). Every slab-capable
	//     extension in the tree keeps to it; one that cannot needs the
	//     sampling points moved first.
	//
	// Three shapes cover every extension in the tree:
	//   A  volume or cell list -- intersect the extension's x-extent with
	//      [startX,stopX), then split that intersection across threadID.
	//   B  plane anchored at a fixed x -- do nothing unless the slab contains
	//      the plane, else keep the extension's own thread split.
	//   C  plane or surface spanning x -- restrict whichever loop axis is x,
	//      split the remaining axis across threadID.
	// ------------------------------------------------------------------------

	//! Can this extension be applied to a sub-range of x at an explicitly given
	//! timestep, rather than to the whole grid at the engine's current one?
	virtual bool SupportsSlabApply() const {return false;}

	//! Which of the six slab hooks below this extension actually implements.
	enum SlabHook
	{
		SLAB_PRE_VOLT   = 1 << 0,
		SLAB_POST_VOLT  = 1 << 1,
		SLAB_APPLY_VOLT = 1 << 2,
		SLAB_PRE_CURR   = 1 << 3,
		SLAB_POST_CURR  = 1 << 4,
		SLAB_APPLY_CURR = 1 << 5
	};

	//! Bitwise OR of the SlabHook values this extension overrides.
	/*!
	  The engine barriers after every extension it calls, so an unimplemented
	  hook is not free: it costs a barrier across all threads for a call that
	  does nothing. That is cheap once per timestep on the flat sweep and
	  expensive on the blocked one, which pays it once per timestep *per tile*.
	  Most extensions implement two or three of the six -- declaring which lets
	  the engine skip the call and the barrier together. Every thread reads the
	  same mask, so they stay in step.

	  Get this wrong by omitting a hook you do implement and it is never called,
	  silently. It is declared next to the hooks for that reason.
	*/
	virtual unsigned int SlabHookMask() const {return 0;}

	//! Can \a hook do anything at all for the slab [startX,stopX)?
	/*!
	  The engine calls every hook once per timestep per tile and barriers
	  after it, also for tiles the extension never reaches -- a Mur plane at
	  x=0 is only ever in the first one. Returning false lets the engine skip
	  the call and its barrier for that slab. Return false only when the hook
	  would provably do nothing there on every thread; the answer may depend
	  on the slab, but not on the timestep or the thread. The default, true,
	  is always safe.
	*/
	virtual bool SlabTouches(SlabHook hook, unsigned int startX, unsigned int stopX) const
	{(void)hook; (void)startX; (void)stopX; return true;}

	//! SlabTouches() for the slab [startX,stopX) at the timestep \a numTS.
	/*!
	  The engine asks this one. It is for an extension whose reach changes
	  with the timestep: the steady-state detection reads a few probe lines at
	  every timestep, but every slab at the end of a period. All threads ask
	  with the same slab and timestep, so the answer may depend on both, but
	  not on the thread. The default is SlabTouches().
	*/
	virtual bool SlabTouchesAt(SlabHook hook, unsigned int startX, unsigned int stopX, int numTS) const
	{(void)numTS; return SlabTouches(hook, startX, stopX);}

	//! Does \a hook write the engine's fields?
	/*!
	  A hook that only reads the fields, and writes nothing but this
	  extension's own state, cannot change what another such hook reads, so
	  the engine runs consecutive hooks of that kind without a barrier between
	  them; it still barriers before anything that writes the fields. Return
	  false only for hooks that honour exactly that. The default, true, is
	  always safe.
	*/
	virtual bool SlabHookWritesFields(SlabHook hook) const {(void)hook; return true;}

	//! The deepest block from timestep \a t0 on that this extension allows, at most \a k (and at least 1).
	virtual unsigned int SlabBlockDepth(int t0, unsigned int k) const {(void)t0; return k;}

	//! Work that needs a timestep of the whole grid, deferred to the end of the block [t0, t0+k).
	/*!
	  Called on every thread after the last slab of every block, once every
	  slab hook of the block has run on every thread. The work itself belongs
	  on thread 0. Return true if there is any, on every thread alike: the
	  engine then holds the other threads until thread 0 is done, so the next
	  block cannot overwrite what it reads.
	*/
	virtual bool SlabBlockEnd(int t0, unsigned int k, int threadID) {(void)t0; (void)k; (void)threadID; return false;}

	//! \copydoc DoPreVoltageUpdates
	virtual void DoPreVoltageUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
	{(void)startX; (void)stopX; (void)numTS; (void)threadID;}
	//! \copydoc DoPostVoltageUpdates
	virtual void DoPostVoltageUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
	{(void)startX; (void)stopX; (void)numTS; (void)threadID;}
	//! \copydoc Apply2Voltages
	virtual void Apply2VoltagesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
	{(void)startX; (void)stopX; (void)numTS; (void)threadID;}

	//! \copydoc DoPreCurrentUpdates
	virtual void DoPreCurrentUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
	{(void)startX; (void)stopX; (void)numTS; (void)threadID;}
	//! \copydoc DoPostCurrentUpdates
	virtual void DoPostCurrentUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
	{(void)startX; (void)stopX; (void)numTS; (void)threadID;}
	//! \copydoc Apply2Current
	virtual void Apply2CurrentSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID)
	{(void)startX; (void)stopX; (void)numTS; (void)threadID;}

	//! Number of threads the slab hooks are partitioned across.
	int GetNumberOfThreads() const {return m_NrThreads;}

	//! Set the Engine to this extension. This will usually done automatically by Engine::AddExtension
	virtual void SetEngine(Engine* eng) {m_Eng=eng;}

	//! Get the priority for this extension
	virtual int GetPriority() const {return m_Priority;}

	//! Set the priority for this extension
	virtual void SetPriority(int val) {m_Priority=val;}

	virtual bool operator< (const Engine_Extension& other);

	virtual std::string GetExtensionName() const;

protected:
	Engine_Extension(Operator_Extension* op_ext);

	//! Intersect [lo,hi) with the slab [startX,stopX), then hand this thread its share.
	/*!
	  The even split every pattern-A slab hook needs, in one place: the
	  extensions differ in what they iterate, not in how they divide it. Returns
	  false when this thread has nothing to do, which is the common case for a
	  slab that misses the extension's box entirely.
	*/
	static bool SlabShare(unsigned int lo, unsigned int hi,
	                      unsigned int startX, unsigned int stopX,
	                      int nThreads, int threadID,
	                      unsigned int& start, unsigned int& stop);

	Operator_Extension* m_Op_ext;
	Engine* m_Eng;

	int m_Priority;

	int m_NrThreads;
};

#endif // ENGINE_EXTENSION_H
