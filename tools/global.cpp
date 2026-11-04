/*
*	Copyright (C) 2024 Yifeng Li <tomli@tomli.me>
*	Copyright (C) 2010 Sebastian Held <sebastian.held@gmx.de>
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

#include <cstring>
#include <iostream>
#include "global.h"

using namespace std;

// create global object
Global g_settings;

//! \brief This function initializes the object
Global::Global()
{
	m_showProbeDiscretization = false;
	m_nativeFieldDumps = false;
	m_legacyHDF5 = false;
	m_VerboseLevel = 0;
	m_SavedVerboseLevel = 0;
	m_GPUDevice = -1;
	m_optionDesc = NULL;
}

OptionDesc
Global::optionDesc()
{
	OptionDesc optdesc("Additional global arguments");

	optdesc.addBoolSwitch(
		"showProbeDiscretization",
		[this](bool val)
		{
			if (!val) return;
			cout << "openEMS - showing probe discretization information" << endl;
			m_showProbeDiscretization = true;
		},
		"Show probe discretization information"
	);

	optdesc.addBoolSwitch(
		"nativeFieldDumps",
		[this](bool val)
		{
			if (!val) return;
			cout << "openEMS - dumping all fields using the native field components" << endl;
			m_nativeFieldDumps = true;
		},
		"Dump all fields using the native field components"
	);

	optdesc.addBoolSwitch(
		"legacyHDF5Dumps",
		[this](bool val)
		{
			if (!val) return;
			cout << "openEMS - dumping all fields using the legacy HDF5 file format as required for Octave/Matlab import" << endl;
			m_legacyHDF5 = true;
		},
		"Dump all fields using the legacy HDF5 file format as required for Octave/Matlab import"
	);

	optdesc.addUintOption(
		"verbose,v",
		0, // default
		1, // implicit (bare -v means 1)
		[this](unsigned int val)
		{
			// Don't apply settings if the default value 0 is unchanged,
			// Apply settings and print messages if we have a non-default value
			// or if the non-default value is changed back to default in another
			// call (when running as a shared library).
			if (val == 0 && m_VerboseLevel == 0) return;

			m_VerboseLevel = val;
			cout << "openEMS - verbose level " << m_VerboseLevel << endl;
		},
		"Verbose level, select debug level 1 to 3, "
		"also accept -v, -vv, -vvv"
	);

	return optdesc;
}

void Global::clearOptionDesc()
{
	delete m_optionDesc;
	m_optionDesc = NULL;
}

void Global::appendOptionDesc(OptionDesc desc)
{
	if (m_optionDesc == NULL)
		m_optionDesc = new OptionDesc();

	m_optionDesc->merge(desc);
}

void Global::parseLibraryArguments(std::vector<std::string> allOptions)
{
	for (std::string& option : allOptions)
	{
		if (option.length() == 1)
			option = "-" + option;
		else
			option = "--" + option;
	}

	m_optionDesc->parse(allOptions);
}

void Global::parseCommandLineArguments(int argc, const char* argv[])
{
	// Handle repeated "-vv" and "-vvv" syntax by rewriting argv[]
	// to the equivalent --verbose=N form.
	std::pair<std::string, std::string> replaceTable[] =
	{
		{"-vv",  "--verbose=2"},
		{"-vvv", "--verbose=3"},
	};

	for (int i = 0; i < argc; i++)
	{
		for (const auto& entry : replaceTable)
		{
			if (std::string(argv[i]) == entry.first)
				argv[i] = entry.second.c_str();
		}
	}

	m_optionDesc->parse(argc, argv);
}

void Global::showOptionUsage(std::ostream& ostr)
{
	m_optionDesc->printUsage(ostr);
}

bool Global::hasOption(std::string option)
{
	// No longer backed by a variables_map; this method is unused
	// externally and retained only for API compatibility.
	(void)option;
	return false;
}

void Global::clearOptions()
{
	// Previously cleared the boost variables_map.
	// With the callback-based parser, state is set directly on members,
	// so there is nothing to clear here.
}
