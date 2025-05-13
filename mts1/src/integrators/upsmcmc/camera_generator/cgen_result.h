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

#pragma once
#if !defined(__MITSUBA_PGEN_RESULT_H_)
#define __MITSUBA_PGEN_RESULT_H_

#include "../paths_cache.h"

MTS_NAMESPACE_BEGIN

// Stores generated photons
class CameraGeneratorResult : public WorkResult {
public:
    // If accum = true (i.e. this instance will be used for results accumulation), more space for photons is reserved. 
    inline CameraGeneratorResult(const UPSMCMCConfiguration &config, bool accum = false):
        m_config(config), 
        m_pathsCache(config,
            accum ? config.subPaths : (config.subPaths / config.nCores + config.nCores))
    {
        clearAndKeepStats();
    }

    virtual ~CameraGeneratorResult() {
    }

    // Accumulate results
    void accum(const CameraGeneratorResult *workResult) {
        m_pathsCache.move(workResult->m_pathsCache); // Photons
    }

    // Adds weighted photon path
    void addPath(const Path & path, Float radius, const Point2 & samplePos = Point2(-1)) {
        m_pathsCache.addPath(path, radius, samplePos);
    }
    
    // Photons access
    inline PathsCache & getPathsCache() {
        return m_pathsCache;
    }

    void clearAndKeepStats() {
        m_pathsCache.clear();
    }

    void clear() {
        clearAndKeepStats();
    }

    virtual void load(Stream *stream) {
        SLog(EError, "Streaming is disabled");
    }

    virtual void save(Stream *stream) const {
        SLog(EError, "Streaming is disabled");
    }

    std::string toString() const {
        return "";
    }

    MTS_DECLARE_CLASS()

private:
    const UPSMCMCConfiguration & m_config;
    // Must be mutable, so we can perform efficient accumulation 
    mutable PathsCache m_pathsCache;
};

MTS_NAMESPACE_END

#endif /* __MITSUBA_PGEN_RESULT_H_ */