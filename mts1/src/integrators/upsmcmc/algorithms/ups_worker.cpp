#include "ups_worker.h"

MTS_NAMESPACE_BEGIN

void UPSWorker::prepare() {
    if (m_prepared)
        return;
    // === Basic initialization
    if (m_prepared)
        return;
    Scene *scene = static_cast<Scene *>(getResource("scene"));
    m_scene = new Scene(scene);
    m_sampler = static_cast<UPSMCMCSampler *>(getResource("sampler"));
    Sensor *newSensor = static_cast<Sensor *>(getResource("sensor"));
    m_scene->removeSensor(scene->getSensor());
    m_scene->addSensor(newSensor);
    m_scene->setSensor(newSensor);
    m_scene->initializeBidirectional();
    m_film = newSensor->getFilm();

    m_pathUtils = new PathUtils(m_scene,
        m_sampler, m_sampler, m_sampler, m_config);

    m_pool = &m_pathUtils->getMemoryPool();

    m_prepared = true;
}

void UPSWorker::process(const WorkUnit *workUnit, WorkResult *workResult, const bool &stop) {
    UPSMCMCResult *result = static_cast<UPSMCMCResult *>(workResult);
    const SeedWorkUnitEx *wu = static_cast<const SeedWorkUnitEx *>(workUnit);
    SplatList *current1 = new SplatList(), *current2 = new SplatList(), *proposed = new SplatList();

    // Clear results
    result->clear();

    ref<Timer> timer = new Timer();

    m_index = wu->getWorkerIndex();
    const Vector2i res = result->getResolution();
    const size_t pathsCount = m_config.subPaths / m_config.nCores;
    size_t m_pixelIndex = pathsCount * m_index;

    /* VCM/UPS main loop */
    SplatList splatList;
    for (uint64_t ctr = 0; ctr < pathsCount && !stop; ++ctr) {
        // === For time constant comparison
        if (wu->getTimeout() > 0 && (ctr % 8192) == 0
            && (int)timer->getMilliseconds() > wu->getTimeout())
            break;

        // Compute pixel
        int index = (int)(m_pixelIndex % m_config.subPaths);
        Point2i pixel(index % res.x, index / res.x);
        ++m_pixelIndex;

        // === Create proposed path
        m_pathUtils->sampleLightAndDoUPS(splatList, m_cameraStorage, index);
        m_pathUtils->clearSplats();

        if (!splatList.pathTracing.second.isZero())
            result->putSample(splatList.pathTracing.first, splatList.pathTracing.second, 0);

        // === For all splat, splat it one the screen
        for (size_t k = 0; k < splatList.size(); ++k) {
            Spectrum value = splatList.getValue(k);
            if (!value.isZero())
                result->putSample(splatList.getPosition(k), value, 0);
        }
        result->increaseSampleCount(1, 0);

    }

    if (!m_pool->unused())
        Log(EError, "Internal error: detected a memory pool leak!");
}

MTS_IMPLEMENT_CLASS(UPSWorker, false, WorkProcessor)

MTS_NAMESPACE_END