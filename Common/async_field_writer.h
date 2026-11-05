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

#ifndef ASYNC_FIELD_WRITER_H
#define ASYNC_FIELD_WRITER_H

/*
 * Asynchronous field-data writer with a bounded queue.
 *
 * ProcessFieldsTD hands off a (field, writeCallback) pair via Submit().
 * A background thread picks up queued jobs and executes the write.
 * Each job owns its field array until the write completes.
 *
 * If the queue is full (all slots occupied by pending writes),
 * Submit() blocks until a slot is freed — this provides natural
 * back-pressure so that RAM usage is bounded to at most
 *   (numBuffers + 1) * sizeof(one field dump).
 */

#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <deque>
#include <memory>
#include "tools/constants.h"
#include "tools/arraylib/array_nijk.h"

class AsyncFieldWriter
{
public:
	using FieldArray = ArrayLib::ArrayNIJK<FDTD_FLOAT>;

	//! Callback signature: perform the write, return success.
	//! The callback must not retain the field array after it returns.
	using WriteFunc = std::function<bool(FieldArray &field)>;

	//! Create the writer with \a numBuffers slots (at least one).
	explicit AsyncFieldWriter(unsigned int numBuffers = 4);

	//! Drain any pending writes and join the writer thread.
	~AsyncFieldWriter();

	//! Submit a field and its write callback.
	//! Blocks if the queue is full (back-pressure).
	//! Shared ownership of \a field is transferred to the writer.
	void Submit(std::shared_ptr<FieldArray> field, WriteFunc writeFunc);

	//! Change the number of slots by \a delta, for a writer that several
	//! sources share and each brings its slots to.
	void AddCapacity(int delta);

	//! Block until all queued writes have completed.
	void Flush();

	//! Return the number of pending (queued but not yet written) jobs.
	unsigned int PendingCount() const;

private:
	struct Job
	{
		std::shared_ptr<FieldArray> field;
		WriteFunc writeFunc;
	};

	void WriterLoop();

	std::deque<Job> m_jobs;
	int m_capacity;
	unsigned int m_inFlight;

	mutable std::mutex m_mutex;
	std::condition_variable m_notFull;
	std::condition_variable m_notEmpty;
	std::condition_variable m_flushed;

	std::thread m_thread;
	bool m_stop;
};

#endif // ASYNC_FIELD_WRITER_H
