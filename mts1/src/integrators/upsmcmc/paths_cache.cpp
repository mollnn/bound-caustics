#include "paths_cache.h"

MTS_NAMESPACE_BEGIN

void PathsCache::addPath(const Path & path, Float radius, const Point2 & samplePos) {
    if (path.length() == 0)
        return;
    // Insert the location of the 1 + last vertex and 1 + last edge
    m_paths.push_back(SimplePath((int)m_vertices.size(), (int)m_edges.size(), path.length(), 
        radius, samplePos));
    // Add vertices and edges
    for (int i = 0; i < path.length(); ++i)
    {
        m_vertices.push_back(*path.vertex(i));
        m_edges.push_back(*path.edge(i));
    }
    m_vertices.push_back(*path.vertex(path.length()));
}

void PathsCache::clear() {
    m_paths.clear();
    m_vertices.dealloc();
    m_edges.dealloc();
}

void PathsCache::move(PathsCache & pathsCache) {
    // Move vertices & edges
    int verIndex = (int)m_vertices.move_effective(pathsCache.m_vertices);
    int edgeIndex = (int)m_edges.move_effective(pathsCache.m_edges);
    // Move paths and correct their indices
    size_t start = m_paths.size();
    m_paths.insert(m_paths.end(), pathsCache.m_paths.begin(), pathsCache.m_paths.end());
    for_each(m_paths.begin() + start, m_paths.end(), [&](SimplePath & path){
        path.m_firstVertexId += verIndex;
        path.m_firstEdgeId += edgeIndex;
    });
    pathsCache.m_paths.clear();
}

bool PathsCache::clonePath(Path & path, MemoryPool & pool, const SimplePath & sp, int limitVertices) const {
    if (limitVertices > (sp.m_length + 1))
        return false;
    path.release(pool);
    int length = limitVertices == -1 ? sp.m_length : limitVertices - 1;
    for (int i = 0; i < length; ++i) {
        PathVertex * v = pool.allocVertex();
        *v = getPathVertex(sp, i);
        PathEdge * e = pool.allocEdge();
        *e = getPathEdge(sp, i);
        path.append(e, v);
    }
    PathVertex * v = pool.allocVertex();
    *v = getPathVertex(sp, length);
    path.append(v);
    return true;
}

MTS_IMPLEMENT_CLASS(PathsCache, false, Object)

MTS_NAMESPACE_END