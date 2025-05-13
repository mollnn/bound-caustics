/*
This file is part of Mitsuba, a physically based rendering system.

Copyright (c) 2007-2014 by Wenzel Jakob and others.

Mitsuba is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License Version 3
as published by the Free Software Foundation.

Mitsuba is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#if !defined(__PARSABLE_LOGGER_H)
#define __PARSABLE_LOGGER_H

#include <mitsuba/mitsuba.h>
#include <fstream>
#include <sstream>

MTS_NAMESPACE_BEGIN

// Generates a log that is easy to parse
class ParsableLogger {
public:

    ParsableLogger() {}

    void open(const std::string & logFile)
    {
        m_outputFile.open(logFile.c_str(), std::ofstream::out | std::ofstream::trunc);
    }

    template <typename StreamableObject>
    ParsableLogger & operator<<(const StreamableObject & obj) {
        m_outputFile << obj;
        m_consoleLogger << obj;
        return *this;
    }

    void outputToConsole(ELogLevel level) {
        SLog(level, m_consoleLogger.str().c_str());
        m_consoleLogger.str("");
    }

    void openTag(const std::string &tag) {
        m_outputFile << std::endl << "<<" << tag << ">>" << std::endl << std::endl;
    }

    void closeTag(const std::string &tag) {
        m_outputFile << std::endl << "<</" << tag << ">>" << std::endl << std::endl;
        m_outputFile.flush();
    }

    void startIteration(size_t i) {
        m_outputFile << std::endl << "<<ITERATION" << i << ">>" << std::endl << std::endl;
    }

    void endIteration(size_t i) {
        m_outputFile << std::endl << "<</ITERATION" << i << ">>" << std::endl << std::endl;
        m_outputFile.flush();
    }

    void close() {
        m_outputFile.close();
    }

private:

    std::ofstream m_outputFile;
    std::ostringstream m_consoleLogger;
};

MTS_NAMESPACE_END

#endif /* __PARSABLE_LOGGER */
