#include "camera_storage.h"

MTS_NAMESPACE_BEGIN

void CameraStorage::movePaths(PathsCache & pathsCache) {
    m_pathsCache.clear();
    m_pathsCache.move(pathsCache);
}

void CameraStorage::build(bool buildGrid) {
    if (buildGrid) {
        m_grid.clear();
        m_nodes.clear();
        for (int pathId = 0; pathId < (int)m_pathsCache.pathsCount(); ++pathId) {
            const PathsCache::SimplePath & sp = m_pathsCache.getPath(pathId);
            for (int vertexId = 2; vertexId <= sp.m_length; ++vertexId) { // Ignores first two vertices and also first hit
                const PathVertex & pv = m_pathsCache.getPathVertex(sp, vertexId);
                if (pv.isSurfaceInteraction() && pv.getIntersection().shape) {
                    int bsdfType = pv.getIntersection().getBSDF()->getType();
                    // Only add vertices on diffuse/glossy surfaces
                    if ((bsdfType & BSDF::EDiffuseReflection) || (bsdfType & BSDF::EGlossyReflection))
                        m_nodes.push_back(Node(pathId, vertexId, pv.getPosition(), sp.m_radius));
                }
            }
        }
        m_grid.build(m_nodes);
    }
    m_pathsCache.buildSelector(m_random);
}

MTS_IMPLEMENT_CLASS(CameraStorage, false, Object)

MTS_NAMESPACE_END