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

#if !defined(__CGEN_PROC_H)
#define __CGEN_PROC_H

// === Include MTS
#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/renderjob.h>
#include <mitsuba/core/bitmap.h>
#include "../upsmcmc.h"
#include "cgen_result.h"
#include "../path_utils.h"

// === Include STL
#include <fstream>

MTS_NAMESPACE_BEGIN

// Runs several workers for generating camera paths
class CameraGeneratorProcess : public ParallelProcess {
public:
    CameraGeneratorProcess(Timer * timeoutTimer, const UPSMCMCConfiguration &config);

    /// Resets process at the beginning of each iteration
    void startIteration() {
        m_workCounter = 0;
        m_accum->clearAndKeepStats();
    }

    // Photon access
    CameraGeneratorResult * getResult() {
        return m_accum.get();
    }

    /* ParallelProcess impl. */
    void processResult(const WorkResult *wr, bool cancelled);
    ref<WorkProcessor> createWorkProcessor() const;
    EStatus generateWork(WorkUnit *unit, int worker);

    MTS_DECLARE_CLASS()
protected:

    /// Virtual destructor
    virtual ~CameraGeneratorProcess() { }

private:
    const UPSMCMCConfiguration &m_config;
    ref<Timer> m_timeoutTimer;
    ref<Mutex> m_resultMutex;
    int m_workCounter;
    mutable ref_vector<WorkProcessor> m_workers;
    ref<CameraGeneratorResult> m_accum;
    bool m_generateCameraPaths;
};

MTS_NAMESPACE_END

#endif /* __CGEN_PROC */
