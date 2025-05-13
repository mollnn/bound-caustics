#include "ups_mcmc_worker.h"

MTS_NAMESPACE_BEGIN

void UPSMCMCWorker::prepare() {
    if (m_prepared)
        return;
    // === Basic initialization
    Scene *scene = static_cast<Scene *>(getResource("scene"));
    m_origSampler = static_cast<UPSMCMCSampler *>(getResource("sampler"));
    m_sensor = static_cast<Sensor *>(getResource("sensor"));
    m_scene = new Scene(scene);
    m_film = m_sensor->getFilm();
    m_scene->setSensor(m_sensor);
    m_scene->setSampler(m_origSampler);
    m_scene->removeSensor(scene->getSensor());
    m_scene->addSensor(m_sensor);
    m_scene->setSensor(m_sensor);
    m_scene->wakeup(NULL, m_resources);
    m_scene->initializeBidirectional();

    for (int i = 0; i < 2; ++i) { // For both chains
        Chain & c = m_chain[i];
        c.rplSampler = static_cast<ReplayableSampler*>(
            static_cast<Sampler *>(getResource("rplSampler"))->clone().get());

        c.emitter = new UPSMCMCSampler(m_origSampler);
        c.sensor = new UPSMCMCSampler(m_origSampler);
        c.direct = new UPSMCMCSampler(m_origSampler);
        c.emitter->setLargeStep(true);
        c.emitter->accept();
        c.sensor->setLargeStep(true);
        c.sensor->accept();
        c.direct->setLargeStep(true);
        c.direct->accept();
        c.pathUtils = new PathUtils(m_scene, c.emitter, c.sensor, c.direct, m_config);
    }
    m_prepared = true;
}

static inline Float t(Float luminance, size_t chain) {
    if (!std::isfinite(luminance) || std::isnan(luminance))
        return 0;
    return chain == 0 ? luminance > 0.f : luminance;
}


inline Float UPSMCMCWorker::mist(Float luminance, size_t chain) {
    return (1.f / m_config.normalization[chain + 1]) / (1.f / m_config.normalization[1] + luminance / m_config.normalization[2]);
}

void UPSMCMCWorker::process(const WorkUnit *workUnit, WorkResult *workResult, const bool &stop) {
    UPSMCMCResult *result = static_cast<UPSMCMCResult *>(workResult);
    const SeedWorkUnitEx *wu = static_cast<const SeedWorkUnitEx *>(workUnit);
    SplatList * current[2];
    current[0] = new SplatList();
    current[1] = new SplatList();
    SplatList * proposed = new SplatList();

    // Clear results
    result->clear();

    ref<Timer> timer = new Timer();
    ref<Random> random = m_origSampler->getRandom();

    m_index = wu->getWorkerIndex();
    const Vector2i res = result->getResolution();
    size_t mutationCount = m_config.subPaths / m_config.nCores;
    size_t m_pixelIndex = mutationCount * m_index;


    // Seed the chains
    for (int i = 0; i < 2; ++i) {
        Chain & c = m_chain[i];
#if 1
        c.emitter->reset();
        c.sensor->reset();
        c.direct->reset();
        c.emitter->setRandom(c.rplSampler->getRandom());
        c.sensor->setRandom(c.rplSampler->getRandom());
        c.direct->setRandom(c.rplSampler->getRandom());

        /* Generate the initial sample by replaying the seeding random
        number stream at the appropriate position. Afterwards, revert
        back to this worker's own source of random numbers */
        const PathSeedEx & seed = wu->getSeed(i);
        c.rplSampler->setSampleIndex(seed.sampleIndex);

        // === Create proposed path
        c.pathUtils->sampleLightAndDoUPS(*current[i], m_cameraStorage, seed.cameraPathIndex);
        c.pathUtils->clearSplats();
        if (std::abs((current[i]->luminance - seed.luminance)
            / seed.luminance) > Epsilon)
            Log(EError, "Error when reconstructing a seed path: luminance "
            "= %f, but expected luminance = %f", current[i]->luminance, seed.luminance);

        c.emitter->setRandom(random);
        c.sensor->setRandom(random);
        c.direct->setRandom(random);
        c.rplSampler->updateSampleIndex(c.rplSampler->getSampleIndex()
            + c.sensor->getSampleIndex()
            + c.emitter->getSampleIndex()
            + c.direct->getSampleIndex());

        c.sensor->accept();
        c.emitter->accept();
        c.direct->accept();
#else
        // Find first sample
        const int index = (int)(m_pixelIndex % m_config.subPaths);
        c.sensor->setLargeStep(true);
        c.emitter->setLargeStep(true);
        c.direct->setLargeStep(true);
        while (current[i]->luminance == 0) {
            // === Create proposed path
            c.pathUtils->sampleLightAndDoUPS(*proposed, m_cameraStorage, index);
            c.pathUtils->clearSplats();
            if (proposed->luminance > 0) {
                std::swap(proposed, current[i]);
                // === Update samplers
                c.emitter->accept();
                c.direct->accept();
                c.sensor->accept();
            }
            else {
                c.emitter->reject(false);
                c.direct->reject(false);
                c.sensor->reject(false);
            }
        }
#endif
    }
    
    /* MCMC main loop */
    Float cumulativeWeight[] = { 0, 0 };
    size_t chain = 0;
    for (uint64_t mutationCtr = 0; mutationCtr < mutationCount && !stop; ++mutationCtr, chain = (chain + 1) % 2) {
        // === For time constant comparison
        if (wu->getTimeout() > 0 && (mutationCtr % 8192) == 0
            && (int)timer->getMilliseconds() > wu->getTimeout())
            break;

        UPSMCMCStats::ChainStats & stats = result->getStats().chains[chain + 1];

        Chain & c = m_chain[chain];

        // Compute pixel
        int index = (int)(m_pixelIndex % m_config.subPaths);
        ++m_pixelIndex;

        // === pLarge state ?
        bool largeStep = chain == 0 && random->nextFloat() < c.sensor->largeStepProb();

        // Propagate the decision to all samplers
        c.sensor->setLargeStep(true);
        c.emitter->setLargeStep(largeStep);
        c.direct->setLargeStep(largeStep);

        // === Create proposed path
        c.pathUtils->sampleLightAndDoUPS(*proposed, m_cameraStorage, index);
        c.pathUtils->clearSplats();

        Float currentTarget = t(current[chain]->luminance, chain);
        Float proposedTarget = t(proposed->luminance, chain);

        bool nonZeroProposal = proposedTarget != 0;
        /// Update stats
        if (largeStep) {
            stats.largeTotal++;
            ++result->getStats().chains[1].lumSamples;
            ++result->getStats().chains[2].lumSamples;
            if (nonZeroProposal) {
                stats.largeNoZero++;
                result->getStats().chains[1].luminanceAccum += t(proposed->luminance, 0);
                result->getStats().chains[2].luminanceAccum += t(proposed->luminance, 1);
                if (proposed->luminance / (Float)mutationCount > m_config.normalization[2]) {
                    // Simple hack to remove fireflies from normalization estimation
                    SLog(EWarn, "Skipped luminance sample: %f Diff: %f", proposed->luminance, (proposed->luminance / (Float)mutationCount) / (m_config.normalization[2]));
                    result->getStats().chains[2].luminanceAccum -= proposed->luminance;
                    --result->getStats().chains[2].lumSamples;
                }
            }
        }
        else {
            stats.smallTotal++;
        }
        if (!proposed->pathTracing.second.isZero())
            result->putSample(proposed->pathTracing.first, proposed->pathTracing.second, 0);
        result->increaseSampleCount(1, 0); // Always contribute to the second chain

        Float a = std::min((Float)1, (proposedTarget / currentTarget));

        if (a == 1 || a > random->nextFloat()) {
            // === For all splat, splat it one the screen
            Float mis = mist(current[chain]->luminance, chain);
            for (size_t k = 0; k < current[chain]->size(); ++k) {
                Spectrum value = current[chain]->getValue(k) * cumulativeWeight[chain] * mis;
                if (!value.isZero())
                    result->putSample(current[chain]->getPosition(k), value, chain + 1);
            }
            result->increaseSampleCount(cumulativeWeight[chain], chain + 1);

            // === Update current state
            cumulativeWeight[chain] = a;
            std::swap(proposed, current[chain]);

            // === Update samplers
            c.emitter->accept();
            c.direct->accept();
            c.sensor->accept();


            // === Update statistics
            if (largeStep) {
                ++stats.largeAccepted;
            }
            else {
                ++stats.smallAccepted;
            }
        }
        else {
            cumulativeWeight[chain] += 1 - a;

            if (a > 0) {
                Float mis = mist(proposed->luminance, chain);
                for (size_t k = 0; k < proposed->size(); ++k) {
                    Spectrum value = proposed->getValue(k) * a * mis;
                    if (!value.isZero())
                        result->putSample(proposed->getPosition(k), value, chain + 1);
                }
                result->increaseSampleCount(a, chain + 1);
            }

            // === Update sampler
            c.emitter->reject(nonZeroProposal);
            c.direct->reject(nonZeroProposal);
            c.sensor->reject(nonZeroProposal);

        }

        if (chain == 1) {
            // Try swap
            ++stats.swapsTotal;
            Float a = std::min((Float)1, (current[0]->luminance / current[1]->luminance));
            if (a == 1 || a > random->nextFloat()) {
                ++stats.swapsAccepted;
                /* Splat before swap */
                for (int i = 0; i < 2; ++i) {
                    Float mis = mist(current[i]->luminance, i);
                    for (size_t k = 0; k < current[i]->size(); ++k) {
                        Spectrum value = current[i]->getValue(k) * cumulativeWeight[i] * mis;
                        if (!value.isZero())
                            result->putSample(current[i]->getPosition(k), value, i + 1);
                    }
                    result->increaseSampleCount(cumulativeWeight[i], i + 1);
                }
                std::swap(m_chain[0], m_chain[1]);
                m_chain[0].emitter->swapStats(m_chain[1].emitter);
                m_chain[0].sensor->swapStats(m_chain[1].sensor);
                m_chain[0].direct->swapStats(m_chain[1].direct);
                std::swap(current[0], current[1]);
                cumulativeWeight[0] = cumulativeWeight[1] = 0;
            }
        }
    }

    /* Perform the last splat */
    for (int i = 0; i < 2; ++i) {
        Float mis1 = mist(current[i]->luminance, i);
        for (size_t k = 0; k < current[i]->size(); ++k) {
            Spectrum value = current[i]->getValue(k) * cumulativeWeight[i] * mis1;
            if (!value.isZero())
                result->putSample(current[i]->getPosition(k), value, i + 1);
        }
        result->increaseSampleCount(cumulativeWeight[i], i + 1);
        delete current[i];
        BDAssert(m_chain[i].pathUtils->getMemoryPool().unused());
    }
    delete proposed;
}

MTS_IMPLEMENT_CLASS(UPSMCMCWorker, false, WorkProcessor)

MTS_NAMESPACE_END