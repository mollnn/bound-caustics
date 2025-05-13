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

#if !defined(__CAMERA_STORAGE_H)
#define __CAMERA_STORAGE_H

#include <functional>
#include "spheres_grid.h"
#include "paths_cache.h"

MTS_NAMESPACE_BEGIN

class CameraStorage : public Object {
public:
    CameraStorage(const UPSMCMCConfiguration & config, Vector2i resolution) :m_config(config),
        m_pathsCache(config, resolution.x * resolution.y)
    {
        m_random = new Random(1234);
    }

    void clearPaths() {
        m_pathsCache.clear();
    }

    void movePaths(PathsCache & pathsCache);

    void build(bool buildGrid);

    void gatherPointQuery(const Point & p, const std::function<void(int, int)>& functor) const {
        m_grid.query(p, [&](int index) {
            const Node & node = m_nodes[index];
            functor(node.getPathIndex(), node.getVertexIndex());
        });
    }

    const PathsCache & getCameraPaths() const {
        return m_pathsCache;
    }

    ~CameraStorage() {
    }

    MTS_DECLARE_CLASS()
private:

    struct Node {
        Node() {}

        Node(int pathId, int vertexId, const Point & p, Float radius) {
            this->pathId = pathId;
            this->vertexId = vertexId;
            position = p;
            this->radius = radius;
        }

        int getPathIndex() const {
            return pathId;
        }

        int getVertexIndex() const {
            return vertexId;
        }

        const Point & getPosition() const {
            return position;
        }
        int pathId;
        int vertexId;
        Point position;
        Float radius;
    };

    std::vector<Node> m_nodes;

    SpheresGrid<Node> m_grid;

    const UPSMCMCConfiguration & m_config;

    PathsCache m_pathsCache;

    ref<Random> m_random;
};

MTS_NAMESPACE_END

#endif /* __CAMERA_STORAGE */
