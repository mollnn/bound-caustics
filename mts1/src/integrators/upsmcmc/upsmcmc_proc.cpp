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

#include "upsmcmc_proc.h"
#include <mitsuba/render/scene.h>
// Workers
#include "algorithms/ups_mcmc_worker.h"
#include "algorithms/ups_worker.h"

MTS_NAMESPACE_BEGIN

UPSMCMCProcess::UPSMCMCProcess(const RenderJob *parent, RenderQueue *queue,
    Timer * timeoutTimer,
	UPSMCMCConfiguration &conf, 
	const std::vector<PathSeedEx> &seeds,
	Scene* scene,
    const CameraStorage * cameraStorage) : m_job(parent), m_queue(queue),
    m_timeoutTimer(timeoutTimer),
	m_config(conf), m_seeds(seeds), m_scene(scene),  m_cameraStorage(cameraStorage) {
	m_resultMutex = new Mutex();
	m_workCounter = 0;
	m_sampleCount = 0;
    m_lastFlush = 0;
}

ref<WorkProcessor> UPSMCMCProcess::createWorkProcessor() const {
	WorkProcessor * w;
    if (m_workers.size() == m_workCounter) {
        switch (m_config.algorithm) {
            case UPSMCMCConfiguration::EUPS_MCMC:
                w = new UPSMCMCWorker(m_config, m_cameraStorage); break;
            case UPSMCMCConfiguration::EUPS:
                w = new UPSWorker(m_config, m_cameraStorage); break;
            default:
            SLog(EError, "Unsupported algorithm");
        }
        m_workers.push_back(w);
    }
	return m_workers[m_workCounter];
}

void UPSMCMCProcess::develop(bool finalCall) {
    size_t pixelCount = m_developBuffer->getPixelCount();
    Spectrum *target = (Spectrum *)m_developBuffer->getData();
    m_developBuffer->clear();

    if (m_config.isUPS_MCMC())
        m_config.nonVisible = (Float)m_accum->getStats().chains[1].largeNoZero / (Float)m_accum->getStats().chains[1].largeTotal;

    // For each chain
    for (size_t c = 0; c < m_config.chainCount(); ++c) {
        const ImageBlock * chainResult = m_accum->getImageBlock(c);
        const Spectrum *accum = (Spectrum *)chainResult->getBitmap()->getData();
        /* Compute the luminance correction factor */
        Float avgLuminance = m_accum->getSampleCount(c) / (Float)pixelCount;
		if (m_config.isUPS_MCMC() && c > 0) {
            m_config.normalization[c] = (Float)m_accum->getStats().chains[c].luminanceAccum / (Float)m_accum->getStats().chains[c].lumSamples;
            SLog(EInfo, "Normalization is: %f", m_config.normalization[c]);
        }
        Float luminanceFactor = m_config.normalization[c] / avgLuminance;

        if (avgLuminance > 0) {
            for (size_t i = 0; i < pixelCount; ++i) {
                target[i] += accum[i] * luminanceFactor;
            }
        }
    }
        
	m_film->setBitmap(m_developBuffer);
    Float timerSec = Float(m_timeoutTimer->getMilliseconds() / 1000);
    if ((timerSec - m_lastFlush) > m_config.maxTimeImgDump) {
        m_lastFlush = timerSec;
        /// Path computation
        fs::path path = m_scene->getDestinationFile();
        std::stringstream ss;
        ss << path.stem().filename().string() << "_pass_" << (m_config.currentIteration + 1);
        fs::path filename = path.parent_path() / fs::path(ss.str());

        /// Develop image
        m_film->setDestinationFile(filename, 0);
        m_film->develop(m_scene, 0.f);

        /// Revert destination file
        m_film->setDestinationFile(m_scene->getDestinationFile(), 0);
    }

	m_queue->signalRefresh(m_job);
}

void UPSMCMCProcess::printStats() const {
    m_config.logger->openTag("ALGORITHM_STATS");
    const UPSMCMCStats & stats = m_accum->getStats();
    for (size_t t = 0; t < stats.chains.size(); ++t) {
        const UPSMCMCStats::ChainStats & chain = stats.chains[t];
        m_config.logger->openTag(formatString("CHAIN #%i", (t + 1)));
        Float acceptRatio = ((chain.largeAccepted + chain.smallAccepted) / (Float)(chain.largeTotal + chain.smallTotal));
        (*m_config.logger) << "Acceptance ratio: " << acceptRatio << " (Small: " << (chain.smallAccepted / (Float)chain.smallTotal)
            << " Large: " << (chain.largeAccepted / (Float)chain.largeTotal) << ")\n";
        (*m_config.logger) << "Swap ratio: " << (chain.swapsAccepted / (Float)chain.swapsTotal) << " \n";
        (*m_config.logger) << "Visibility ratio: " << m_config.nonVisible << " \n";
        m_config.logger->closeTag(formatString("CHAIN #%i", (t + 1)));
    }
    m_config.logger->outputToConsole(EInfo);
    m_config.logger->closeTag("ALGORITHM_STATS");
}

void UPSMCMCProcess::processResult(const WorkResult *wr, bool cancelled) {
	LockGuard lock(m_resultMutex);
    const UPSMCMCResult *result = static_cast<const UPSMCMCResult*>(wr);
	m_accum->accum(result);
	m_sampleCount += m_config.subPaths;
}

ParallelProcess::EStatus UPSMCMCProcess::generateWork(WorkUnit *unit, int worker) {
	int timeout = 0;
    if (m_config.timeout > 0) {
		timeout = static_cast<int>(static_cast<int64_t>(m_config.timeout * 1000) -
			static_cast<int64_t>(m_timeoutTimer->getMilliseconds()));
	}
	if (m_workCounter >= m_config.nCores || timeout < 0)
		return EFailure;

	SeedWorkUnitEx *workUnit = static_cast<SeedWorkUnitEx *>(unit);

	if (m_config.isUPS_MCMC()) {
        workUnit->clearSeeds();
        workUnit->addSeed(m_seeds[(m_workCounter << 1)]);
        workUnit->addSeed(m_seeds[(m_workCounter << 1) + 1]);
    }
    workUnit->setWorkerIndex(m_workCounter);
	workUnit->setTimeout(timeout);
    ++m_workCounter;
	return ESuccess;
}

void UPSMCMCProcess::bindResource(const std::string &name, int id) {
	ParallelProcess::bindResource(name, id);
	if (name == "sensor") {
		m_film = static_cast<Sensor *>(Scheduler::getInstance()->getResource(id))->getFilm();
		
        m_accum = new UPSMCMCResult(m_config, Bitmap::ESpectrum, m_film->getCropSize());
		m_accum->clear();
		m_developBuffer = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, m_film->getCropSize());
	}
}

MTS_IMPLEMENT_CLASS(UPSMCMCProcess, false, ParallelProcess)

MTS_NAMESPACE_END
