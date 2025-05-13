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

#if !defined(__UPSMCMC_PROC_H)
#define __UPSMCMC_PROC_H

// === Include MTS
#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/renderjob.h>
#include <mitsuba/core/bitmap.h>
#include "upsmcmc.h"
#include "upsmcmc_result.h"
#include "camera_storage.h"
#include "path_utils.h"

// === Include STL
#include <fstream>

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                           Parallel process                           */
/* ==================================================================== */

class UPSMCMCProcess : public ParallelProcess {
public:
	UPSMCMCProcess(const RenderJob *parent, RenderQueue *queue,
        Timer * timeoutTimer,
		UPSMCMCConfiguration &config, 
		const std::vector<PathSeedEx> &seeds,
		Scene* scene,
        const CameraStorage * cameraStorage);

    /// Resets process at the beginning of each iteration
	void startIteration() {
		m_workCounter = 0;
	}

    /// Displays overall contribution
	void develop(bool finalCall);

    /// Prints statistics
	void printStats() const;

	/* ParallelProcess impl. */
	void processResult(const WorkResult *wr, bool cancelled);
	ref<WorkProcessor> createWorkProcessor() const;
	void bindResource(const std::string &name, int id);
	EStatus generateWork(WorkUnit *unit, int worker);

	MTS_DECLARE_CLASS()
protected:

	/// Virtual destructor
	virtual ~UPSMCMCProcess() { }

private:
	ref<const RenderJob> m_job;
	RenderQueue *m_queue;
	UPSMCMCConfiguration &m_config;
    const CameraStorage * m_cameraStorage;
	ref<Bitmap> m_developBuffer;
	ref<UPSMCMCResult> m_accum;
	size_t m_sampleCount;
	const std::vector<PathSeedEx> &m_seeds;
	ref<Mutex> m_resultMutex;
	ref<Film> m_film;
	int m_workCounter;
	ref<Timer> m_timeoutTimer;
	mutable ref_vector<WorkProcessor> m_workers;
	Scene* m_scene;
    Float m_lastFlush;
};

MTS_NAMESPACE_END

#endif /* __MMLT_PROC */
