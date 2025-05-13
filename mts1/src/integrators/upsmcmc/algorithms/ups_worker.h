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

#if !defined(__UPS_WORKER_H)
#define __UPS_WORKER_H

#include <mitsuba/core/plugin.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/bidir/util.h>

#include "../upsmcmc.h"
#include "../upsmcmc_result.h"
#include "../path_utils.h"
#include "../camera_storage.h"

MTS_NAMESPACE_BEGIN

// Worker that generates paths using UPS/VCM algorithm.
class UPSWorker : public WorkProcessor {
public:
    UPSWorker(const UPSMCMCConfiguration &conf, const CameraStorage * cameraStorage)
        : m_config(conf), m_cameraStorage(cameraStorage), m_prepared(false) {
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        SLog(EError, "Streaming is disabled");
    }

    ref<WorkUnit> createWorkUnit() const {
        return new SeedWorkUnitEx();
    }

    ref<WorkResult> createWorkResult() const {
        return new UPSMCMCResult(m_config, Bitmap::ESpectrum,
            m_film->getCropSize(), m_film->getReconstructionFilter());
    }

    void prepare();

    void process(const WorkUnit *workUnit, WorkResult *workResult, const bool &stop);

    ref<WorkProcessor> clone() const {
        return new UPSWorker(m_config, m_cameraStorage);
    }

    MTS_DECLARE_CLASS()
private:

    const UPSMCMCConfiguration & m_config;
    const CameraStorage * m_cameraStorage;
    ref<Scene> m_scene;
    ref<Sampler> m_sampler;
    ref<PathUtils> m_pathUtils;
    ref<Film> m_film;
    MemoryPool *m_pool;
    bool m_prepared;
    size_t m_index;
};

MTS_NAMESPACE_END

#endif /* __UPS_WORKER */