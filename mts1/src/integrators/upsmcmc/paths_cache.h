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

#if !defined(__PATHS_CACHE_H)
#define __PATHS_CACHE_H

#include "upsmcmc.h"
#include "growing_array.h"

MTS_NAMESPACE_BEGIN

// Caches paths that will be later used as photon paths
class PathsCache : public Object {
public:

    // Stores indices of the first vertex and the first edge of a given path and the path length
    struct SimplePath {
        SimplePath(){}
        SimplePath(int lv, int le, int length, Float radius, const Point2 & samplePos) :
        m_firstVertexId(lv), m_firstEdgeId(le), m_length(length),
        m_radius(radius), m_samplePos(samplePos)
        {
        }
        int m_firstVertexId;
        int m_firstEdgeId;
        int m_length;
        int m_primarySampleLength;
        Float m_radius;
        Point2 m_samplePos;
    };

    PathsCache(const UPSMCMCConfiguration & config, size_t estimatedPathCount) :m_config(config)
    {
        m_paths.reserve(estimatedPathCount);
    }

    void addPath(const Path & path, Float radius, const Point2 & samplePos);

    void clear();

    // Move contents of <pathsCache> to this cache.
    void move(PathsCache & pathsCache);

    inline size_t pathsCount() const {
        return m_paths.size();
    }

    inline const SimplePath & getPath(size_t index) const {
        return m_paths[index];
    }

    inline const PathVertex & getPathVertex(const SimplePath & sp, size_t vertexId) const {
        return m_vertices[sp.m_firstVertexId + vertexId];
    }

    inline PathVertex & getPathVertex(const SimplePath & sp, size_t vertexId) {
        return m_vertices[sp.m_firstVertexId + vertexId];
    }

    inline const PathEdge & getPathEdge(const SimplePath & sp, size_t edgeId) const {
        return m_edges[sp.m_firstEdgeId + edgeId];
    }

    inline PathEdge & getPathEdge(const SimplePath & sp, size_t edgeId) {
        return m_edges[sp.m_firstEdgeId + edgeId];
    }

    inline const size_t estimatedVertexCount() const {
        return m_vertices.size();
    }

    inline const size_t estimatedEdgeCount() const {
        return m_edges.size();
    }

    // Selects a path and get weight into account
    inline SimplePath selectPathProportionally(size_t index) const {
        return getPath(m_selector[index]);
    }

    bool clonePath(Path & path, MemoryPool & pool, const SimplePath & sp, int limitVertices = -1) const;

    void buildSelector(Random * random) {
        m_selector.resize(m_paths.size());
        for (size_t i = 0; i < m_selector.size(); ++i)
            m_selector[i] = i;
        for (size_t i = 0; i < m_selector.size(); ++i) {
            size_t i1 = random->nextSize(m_selector.size());
            std::swap(m_selector[i1], m_selector[i]);
        }
    }

    ~PathsCache() {
        clear();
    }

    MTS_DECLARE_CLASS()
private:

    const UPSMCMCConfiguration & m_config;

    std::vector<SimplePath> m_paths;

    GrowingArray<PathVertex> m_vertices;

    GrowingArray<PathEdge> m_edges;

    std::vector<size_t> m_selector;
};

MTS_NAMESPACE_END

#endif /* __PATHS_CACHE */
