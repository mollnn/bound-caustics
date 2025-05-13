#include "cworker.h"
#include <mitsuba/bidir/util.h>

MTS_NAMESPACE_BEGIN

void CAMWorker::prepare() {
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

    m_pathUtils = new PathUtils(m_scene,
        m_sampler, m_sampler, m_sampler, m_config);

    m_pool = &m_pathUtils->getMemoryPool();


    m_prepared = true;
}

void CAMWorker::process(const WorkUnit *workUnit, WorkResult *workResult, const bool &stop) {
    CameraGeneratorResult *result = static_cast<CameraGeneratorResult *>(workResult);
    const CameraGeneratorWorkUnit *wu = static_cast<const CameraGeneratorWorkUnit *>(workUnit);

    result->clear();
    ref<Timer> timer = new Timer();
    m_index = wu->getWorkerIndex();
    const Vector2i res = m_scene->getSensor()->getFilm()->getCropSize();
    const size_t pixelCount = res.x * res.y;
    size_t m_pixelIndex = (pixelCount / m_config.nCores) * m_index;

    Path cameraPath;
    for (size_t ctr = 0; ctr < wu->getCameraPathsCount()
        && !stop; ++ctr) {
        if (wu->getTimeout() > 0 && (ctr % 8192) == 0 &&
            (int)timer->getMilliseconds() > wu->getTimeout())
            break;
        
        // Compute pixel
        int index = (int)(m_pixelIndex % pixelCount);
        Point2i pixel(index % res.x, index / res.x);
        ++m_pixelIndex;

        // Generate path using standard MC and add it to results.
        m_pathUtils->generateCameraPath(cameraPath, pixel);
        result->addPath(cameraPath, m_pathUtils->getSensorSubPathRadius(), m_pathUtils->getSensorSubPathSamplePosition());
        cameraPath.release(*m_pool);
    }

    if (!m_pool->unused())
        Log(EError, "Internal error: detected a memory pool leak!");
}


MTS_IMPLEMENT_CLASS(CAMWorker, false, WorkProcessor)


MTS_NAMESPACE_END