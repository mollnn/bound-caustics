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

#pragma once
#if !defined(__MITSUBA_CGEN_UNIT_H_)
#define __MITSUBA_CGEN_UNIT_H_

#include "../upsmcmc.h"

MTS_NAMESPACE_BEGIN

// The photon generation is divided to work units. Each unit corresponds to a given
// number of photon paths.
class CameraGeneratorWorkUnit : public WorkUnit {
public:
    inline void set(const WorkUnit *wu) {
        m_cameraPathsCount = static_cast<const CameraGeneratorWorkUnit *>(wu)->m_cameraPathsCount;
        m_timeout = static_cast<const CameraGeneratorWorkUnit *>(wu)->m_timeout;
        m_workerIndex = static_cast<const CameraGeneratorWorkUnit *>(wu)->m_workerIndex;
    }

    inline void setWorkerIndex(size_t index) {
        m_workerIndex = index;
    }

    inline size_t getWorkerIndex() const {
        return m_workerIndex;
    }

    inline int getCameraPathsCount() const {
        return m_cameraPathsCount;
    }

    inline void setCameraPathsCount(int count) {
        m_cameraPathsCount = count;
    }

    inline int getTimeout() const {
        return m_timeout;
    }

    inline void setTimeout(int timeout) {
        m_timeout = timeout;
    }

    inline void load(Stream *stream) {
        SLog(EError, "Streaming is disabled");
    }

    inline void save(Stream *stream) const {
        SLog(EError, "Streaming is disabled");
    }

    inline std::string toString() const {
        return "CameraGeneratorWorkUnit[]";
    }

    MTS_DECLARE_CLASS()
private:
    int m_timeout;
    int m_cameraPathsCount;
    size_t m_workerIndex;
};

MTS_NAMESPACE_END

#endif // __MITSUBA_CGEN_UNIT_H_