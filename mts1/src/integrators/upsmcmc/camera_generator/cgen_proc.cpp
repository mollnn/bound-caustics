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

#include "cgen_proc.h"
#include "cgen_unit.h"
#include <mitsuba/render/scene.h>
// Workers
#include "cworker.h"

MTS_NAMESPACE_BEGIN

CameraGeneratorProcess::CameraGeneratorProcess(Timer * timeoutTimer,
const UPSMCMCConfiguration &conf) :
    m_config(conf), m_timeoutTimer(timeoutTimer) {
    m_resultMutex = new Mutex();
    m_workCounter = 0;
    m_accum = new CameraGeneratorResult(m_config, true);
}

ref<WorkProcessor> CameraGeneratorProcess::createWorkProcessor() const {
    WorkProcessor * w;
    if (m_workers.size() == m_workCounter) {
        w = new CAMWorker(m_config);
        m_workers.push_back(w);
    }
    return m_workers[m_workCounter];
}

void CameraGeneratorProcess::processResult(const WorkResult *wr, bool cancelled) {
    LockGuard lock(m_resultMutex);
    const CameraGeneratorResult *result = static_cast<const CameraGeneratorResult*>(wr);
    m_accum->accum(result);
}

ParallelProcess::EStatus CameraGeneratorProcess::generateWork(WorkUnit *unit, int worker) {
    int timeout = 0;
    if (m_workCounter >= m_config.nCores || timeout < 0)
        return EFailure;

    CameraGeneratorWorkUnit *workUnit = static_cast<CameraGeneratorWorkUnit *>(unit);

    workUnit->setTimeout(timeout);
    int paths = (int)(m_config.subPaths / (Float)m_config.nCores);
    if (m_workCounter == m_config.nCores - 1)
        workUnit->setCameraPathsCount(m_config.subPaths - (m_config.nCores - 1) * paths);
    else
        workUnit->setCameraPathsCount(paths);
    workUnit->setWorkerIndex(m_workCounter);
    ++m_workCounter;
    return ESuccess;
}

MTS_IMPLEMENT_CLASS(CameraGeneratorProcess, false, ParallelProcess)
MTS_IMPLEMENT_CLASS(CameraGeneratorWorkUnit, false, WorkUnit)
MTS_IMPLEMENT_CLASS(CameraGeneratorResult, false, WorkResult)

MTS_NAMESPACE_END
