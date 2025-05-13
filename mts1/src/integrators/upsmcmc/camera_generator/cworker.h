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

#if !defined(__CAM_WORKER_H)
#define __CAM_WORKER_H

#include <mitsuba/core/plugin.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/bidir/util.h>

#include "../upsmcmc.h"
#include "cgen_result.h"
#include "../path_utils.h"
#include "cgen_unit.h"
#include "../upsmcmc_sampler.h"

MTS_NAMESPACE_BEGIN

// Simple Monte Carlo camera generator
class CAMWorker : public WorkProcessor {
public:
    CAMWorker(const UPSMCMCConfiguration &conf)
        : m_config(conf), m_prepared(false) {
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        SLog(EError, "Streaming is disabled");
    }

    ref<WorkUnit> createWorkUnit() const {
        return new CameraGeneratorWorkUnit();
    }

    ref<WorkResult> createWorkResult() const {
        return new CameraGeneratorResult(m_config);
    }

    void prepare();

    void process(const WorkUnit *workUnit, WorkResult *workResult, const bool &stop);

    ref<WorkProcessor> clone() const {
        return new CAMWorker(m_config);
    }

    MTS_DECLARE_CLASS()
private:
    const UPSMCMCConfiguration & m_config;
    ref<Scene> m_scene;
    ref<Sampler> m_sampler;
    ref<PathUtils> m_pathUtils;
    MemoryPool *m_pool;
    bool m_prepared;
    size_t m_index;
};

MTS_NAMESPACE_END

#endif /* __CAM_WORKER */