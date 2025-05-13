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

#include <mitsuba/bidir/util.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/core/plugin.h>
#include "upsmcmc.h"
#include "upsmcmc_proc.h"
#include "upsmcmc_sampler.h"
#include "path_utils.h"
#include "camera_storage.h"
#include "camera_generator/cgen_proc.h"

MTS_NAMESPACE_BEGIN

class UPSMCMC : public Integrator {
public:
	UPSMCMC(const Properties &props) : Integrator(props) {
		// Load USP MCMC Properties
		m_config.maxDepth = props.getInteger("maxDepth", 10);
		m_config.seedSamples = props.getInteger("seedSamples", 10000);
		m_config.timeout = props.getInteger("timeout", 0);
		m_config.maxTimeImgDump = props.getInteger("maxTimeImgDump", INT_MAX);
		m_config.parseAlgorithm(props.getString("algorithm", "UPSMCMC"));
        m_config.initialRadius = props.getFloat("initialRadius", 1.f);
        m_config.pathsPerIter = props.getInteger("pathsPerIter", 1000000);
        m_config.alpha = props.getFloat("alpha", 1.f);
        m_config.rrDepth = props.getInteger("rrDepth", 5);
        m_config.seed = props.getSize("seed", 0);
        m_config.primarySpace.pLarge = props.getFloat("pLarge", 0.3f);
        m_config.primarySpace.adaptivity = props.getBoolean("adaptivity", true);
        m_config.primarySpace.acceptanceGoal = props.getFloat("acceptanceGoal", 0.234f);
	}

	/// Unserialize from a binary data stream
	UPSMCMC(Stream *stream, InstanceManager *manager)
		: Integrator(stream, manager) {
		SLog(EError, "Streaming is disabled");
	}

	virtual ~UPSMCMC() { }

	void serialize(Stream *stream, InstanceManager *manager) const {
		Integrator::serialize(stream, manager);
		SLog(EError, "Streaming is disabled");
	}

	bool preprocess(const Scene *scene, RenderQueue *queue,
		const RenderJob *job, int sceneResID, int sensorResID,
		int samplerResID) {
		Integrator::preprocess(scene, queue, job, sceneResID,
			sensorResID, samplerResID);

		if (scene->getSubsurfaceIntegrators().size() > 0)
			Log(EError, "Subsurface integrators are not supported by UPSMCMC!");

        m_config.sceneAABB = scene->getAABB();

		return true;
	}

	void cancel() {
		Scheduler::getInstance()->cancel(m_process);
		m_canceled = true;
	}

    void initCameraShooting(ref<Timer> renderingTimer,
        int sceneResID, int sensorResID, int samplerResID, ref<Sampler> origSampler) {
        
        m_cameraGeneratorProcess = new CameraGeneratorProcess(renderingTimer, m_config);
        m_cameraGeneratorProcess->bindResource("scene", sceneResID);
        m_cameraGeneratorProcess->bindResource("sensor", sensorResID);
        // Create samplers
        ref<Scheduler> scheduler = Scheduler::getInstance();

        /* Create a sampler instance for every core */
        std::vector<SerializableObject *> samplers(scheduler->getCoreCount());
        for (size_t i = 0; i < scheduler->getCoreCount(); ++i) {
            ref<Sampler> clonedSampler = origSampler->clone();
            clonedSampler->incRef();
            samplers[i] = clonedSampler.get();
        }

        // Register and bind the samplers
        int cameraSamplerResID = scheduler->registerMultiResource(samplers);
        for (size_t i = 0; i < scheduler->getCoreCount(); ++i)
            samplers[i]->decRef();
        m_cameraGeneratorProcess->bindResource("sampler", cameraSamplerResID);
    }

    bool shootCameraPaths() {
        ref<Scheduler> scheduler = Scheduler::getInstance();
        m_cameraStorage->clearPaths();
        // Run one iteration of camera paths shooting
        m_process = m_cameraGeneratorProcess;
        m_cameraGeneratorProcess->startIteration();
        scheduler->schedule(m_cameraGeneratorProcess);
        scheduler->wait(m_cameraGeneratorProcess);
        m_process = NULL;
        if (m_cameraGeneratorProcess->getReturnStatus() != ParallelProcess::ESuccess) {
            m_canceled = true;
            return false;
        }
        // Store camera paths
        CameraGeneratorResult * result = m_cameraGeneratorProcess->getResult();
        m_cameraStorage->movePaths(result->getPathsCache());
		m_cameraStorage->build(true);
        return true;
    }

   int getSamplerResID(int samplerResID) {
        ref<Scheduler> scheduler = Scheduler::getInstance();
        /* Create a sampler instance for each worker */
        ref<UPSMCMCSampler> mltSampler = new UPSMCMCSampler(m_config, m_config.seed);
        std::vector<SerializableObject *> mltSamplers(scheduler->getCoreCount());
        for (size_t i = 0; i < mltSamplers.size(); ++i) {
            ref<Sampler> clonedSampler = mltSampler->clone();
            clonedSampler->incRef();
            mltSamplers[i] = clonedSampler.get();
        }
        int mltSamplerResID = scheduler->registerMultiResource(mltSamplers);
        for (size_t i = 0; i < scheduler->getCoreCount(); ++i)
            mltSamplers[i]->decRef();
        return mltSamplerResID;
    }

	bool render(Scene *scene, RenderQueue *queue, const RenderJob *job,
		int sceneResID, int sensorResID, int samplerResID) {
		ref<Scheduler> scheduler = Scheduler::getInstance();
		ref<Sensor> sensor = scene->getSensor();
		ref<Sampler> sampler = sensor->getSampler();
		const Film *film = sensor->getFilm();
		m_config.nCores = (int)scheduler->getCoreCount();
		size_t sampleCount = sampler->getSampleCount();
		m_canceled = false;

		ref<Timer> renderingTimer = new Timer;
        m_random = new Random(13337);

		Vector2i cropSize = film->getCropSize();
		Assert(cropSize.x > 0 && cropSize.y > 0);
		Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT
			" %s, " SSE_STR ", approx. " SIZE_T_FMT " mutations/pixel) ..",
			cropSize.x, cropSize.y,
            m_config.nCores, m_config.nCores == 1 ? "core" : "cores", sampleCount);

        m_config.logger = new ParsableLogger();
        m_config.logger->open(scene->getDestinationFile().string() + ".log");
        m_config.dumpToLog();

		// Create photon storage
        m_cameraStorage = new CameraStorage(m_config, film->getCropSize());
		
		// Compute iterations count
		const size_t cropArea = (size_t)cropSize.x * cropSize.y;
        m_config.subPaths = cropArea * ((m_config.pathsPerIter + cropArea - 1) / cropArea);
		m_config.maxIterations = sampleCount;

		// Sampler for MCMC
		ref<ReplayableSampler> rplSampler = new ReplayableSampler();
        int newSamplerResID = getSamplerResID(samplerResID);
		int rplSamplerResID = scheduler->registerResource(rplSampler);

		// Path Sampling Utils
		ref<PathUtils> pathUtils = new PathUtils(scene,
            rplSampler, rplSampler, rplSampler, m_config);

		// Create parallel-process for MCMC
		std::vector<PathSeedEx> pathSeeds;
        ref<UPSMCMCProcess> process = NULL;

		// Progress bar
		ProgressReporter * progress = new ProgressReporter("Rendering", m_config.maxIterations, job);

        // We need to create camera generator process
        initCameraShooting(renderingTimer, sceneResID, sensorResID, samplerResID, sampler);
        // Set normalization and start depth for UPS (is done at updateNormalization for all other algorithms)
		if (m_config.isUPS()) {
            m_config.normalization.clear();
            m_config.normalization.push_back(1.f);
        }
		if (m_config.isUPS_MCMC()) {
            m_config.normalization.clear();
            // Three chains - two for light paths techniques and other for path-tracing techniques
            m_config.normalization.push_back(1.f);
            m_config.normalization.push_back(1.f);
            m_config.nonVisible = 1.f;
            m_config.normalization.push_back(1.f);
        }
		// Cycle over all iterations
		size_t iteration = 0;
		while (iteration < m_config.maxIterations && !m_canceled) {
            m_config.logger->startIteration(iteration);
			// Compute merging radius
			m_config.currentIteration = iteration;
			m_config.currentRadiusScale = (Float)(m_config.initialRadius * std::pow(iteration + 1, (m_config.alpha - 1) * 0.5f));
			m_config.currentRadiusScale = std::max(m_config.currentRadiusScale, (Float)1e-7); // Purely for numeric stability
            // Fire camera paths
            Log(EInfo, "Performing camera pass %i", (int)iteration);
			if (!shootCameraPaths()) {
                m_config.logger->endIteration(iteration);
                break;
            }
            m_config.logger->outputToConsole(EInfo);
			// Compute initial samples
			if (m_config.isUPS_MCMC()) {
                Log(EInfo, "Performing seeding pass %i", (int)iteration);
                pathUtils->generateSeeds(m_config.seedSamples, m_config.nCores, m_cameraStorage, pathSeeds);
            }
            // Get process
            if (process == NULL) {
                process = new UPSMCMCProcess(job, queue, renderingTimer, m_config, pathSeeds, scene, m_cameraStorage);
                process->bindResource("scene", sceneResID);
                process->bindResource("sensor", sensorResID);
                process->bindResource("sampler", newSamplerResID);
                process->bindResource("rplSampler", rplSamplerResID);
            }
			// Run one iteration of MCMC
            Log(EInfo, "Performing algorithm %s pass %i",m_config.tostrAlgorithm().c_str(),(int)iteration);
			m_process = process;
			process->startIteration();
			scheduler->schedule(process);
			scheduler->wait(process);
			m_process = NULL;
			process->develop(false);
			if (process->getReturnStatus() != ParallelProcess::ESuccess)
				m_canceled = true;
            m_config.logger->endIteration(iteration);
			progress->update(++iteration);
            if (m_config.timeout > 0) {
                int timeout = static_cast<int>(static_cast<int64_t>(m_config.timeout * 1000) -
                    static_cast<int64_t>(renderingTimer->getMilliseconds()));
                if (timeout < 0) {
                    break;
                }
            }
		}
        m_config.logger->openTag("RENDERING");
        (*m_config.logger) << "Total iterations: " << iteration << '\n';
        (*m_config.logger) << "Total time (in seconds): " << (renderingTimer->getMilliseconds() / 1000) << '\n';
        m_config.logger->closeTag("RENDERING");
        process->develop(true);
        process->printStats();
		delete progress;
		scheduler->unregisterResource(rplSamplerResID);

        m_config.logger->close();
        delete m_config.logger;
		return !m_canceled; 
	}

	MTS_DECLARE_CLASS()
private:
	ref<ParallelProcess> m_process;
    ref<CameraGeneratorProcess> m_cameraGeneratorProcess;
    ref<CameraStorage> m_cameraStorage;
	UPSMCMCConfiguration m_config;
	bool m_canceled;
    ref<Random> m_random;
};

MTS_IMPLEMENT_CLASS_S(UPSMCMC, false, Integrator)
MTS_EXPORT_PLUGIN(UPSMCMC, "UPS MCMC combination");
MTS_NAMESPACE_END
