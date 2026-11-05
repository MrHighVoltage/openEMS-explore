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

#include "async_field_writer.h"
#include <algorithm>
#include <iostream>

using std::cerr;
using std::endl;

AsyncFieldWriter::AsyncFieldWriter(unsigned int numBuffers)
	: m_capacity((int)numBuffers)
	, m_inFlight(0)
	, m_stop(false)
{
	m_thread = std::thread(&AsyncFieldWriter::WriterLoop, this);
}

AsyncFieldWriter::~AsyncFieldWriter()
{
	// Signal the writer thread to stop and drain remaining jobs
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		m_stop = true;
	}
	m_notEmpty.notify_one();
	if (m_thread.joinable())
		m_thread.join();
}

void AsyncFieldWriter::Submit(std::shared_ptr<FieldArray> field, WriteFunc writeFunc)
{
	std::unique_lock<std::mutex> lock(m_mutex);

	// Block until a slot is available (back-pressure)
	m_notFull.wait(lock, [this]() { return (int)m_jobs.size() < std::max(1, m_capacity); });

	m_jobs.push_back(Job{std::move(field), std::move(writeFunc)});

	lock.unlock();
	m_notEmpty.notify_one();
}

void AsyncFieldWriter::AddCapacity(int delta)
{
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		m_capacity += delta;
	}
	m_notFull.notify_all();
}

void AsyncFieldWriter::Flush()
{
	std::unique_lock<std::mutex> lock(m_mutex);
	m_flushed.wait(lock, [this]() { return m_jobs.empty() && m_inFlight == 0; });
}

unsigned int AsyncFieldWriter::PendingCount() const
{
	std::lock_guard<std::mutex> lock(m_mutex);
	return (unsigned int)m_jobs.size();
}

void AsyncFieldWriter::WriterLoop()
{
	for (;;)
	{
		Job job;

		// Wait for a job
		{
			std::unique_lock<std::mutex> lock(m_mutex);
			m_notEmpty.wait(lock, [this]() { return !m_jobs.empty() || m_stop; });

			if (m_jobs.empty() && m_stop)
				return; // No more work, thread exits

			job = std::move(m_jobs.front());
			m_jobs.pop_front();
			++m_inFlight;
		}

		// Notify Submit() that a slot freed up
		m_notFull.notify_one();

		// Execute the write (outside the lock)
		bool ok = false;
		try
		{
			ok = job.writeFunc(*job.field);
		}
		catch (...)
		{
			cerr << "AsyncFieldWriter: exception during write" << endl;
		}

		if (!ok)
			cerr << "AsyncFieldWriter: write failed" << endl;

		// The job owns the field data. Notify only after its write completed.
		{
			std::lock_guard<std::mutex> lock(m_mutex);
			--m_inFlight;
			if (m_jobs.empty() && m_inFlight == 0)
				m_flushed.notify_all();
		}
	}
}
