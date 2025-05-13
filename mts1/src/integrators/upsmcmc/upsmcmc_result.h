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
#if !defined(__MITSUBA_UPSMCMC_RESULT_H_)
#define __MITSUBA_UPSMCMC_RESULT_H_

#include "upsmcmc_stats.h"

MTS_NAMESPACE_BEGIN

class UPSMCMCResult : public WorkResult {
public:
    inline UPSMCMCResult(const UPSMCMCConfiguration &config, Bitmap::EPixelFormat fmt, const Vector2i &size,
        const ReconstructionFilter *filter = NULL, int channels = -1) :
        m_config(config), m_stats(config.chainCount()),
        m_resolution(size)
    {
        m_chains.resize(config.chainCount());
        for (size_t i = 0; i < m_chains.size(); ++i) {
            ChainResult & c = m_chains[i];
            c.m_image = new ImageBlock(fmt, size, filter, channels);
            c.m_samples = 0;
        }
        m_stats.clear();
    }

    virtual ~UPSMCMCResult() {
    }

    inline void putSample(const Point2 &sample, const Spectrum &spec, size_t chainNo) {
        ChainResult & chain = m_chains[chainNo];
        chain.m_image->put(sample, spec, 1.0f);
    }

    inline void increaseSampleCount(Float sampleValue, size_t chainNo) {
        ChainResult & chain = m_chains[chainNo];
        chain.m_samples += sampleValue;
    }

    // Accumulate results
    void accum(const UPSMCMCResult *workResult) {
        for (size_t i = 0; i < m_chains.size(); ++i) {
            ChainResult & ourChain = m_chains[i];
            const ChainResult & otherChain = workResult->m_chains[i];
            ourChain.m_image->put(otherChain.m_image.get());
            ourChain.m_samples += otherChain.m_samples;
        }
        m_stats.add(workResult->m_stats);
    }

    inline const Vector2i & getResolution() const {
        return m_resolution;
    }

    inline UPSMCMCStats& getStats() {
        return m_stats;
    }

    inline const UPSMCMCStats& getStats() const {
        return m_stats;
    }

    inline const ImageBlock *getImageBlock(size_t chainNo) const {
        return m_chains[chainNo].m_image.get();
    }

    inline const Float getSampleCount(size_t chainNo) const {
        return m_chains[chainNo].m_samples;
    }

    void clear() {
        for_each(m_chains.begin(), m_chains.end(), [](ChainResult & cr){
            cr.m_image->clear();
            cr.m_samples = 0;
        });
        m_stats.clear();
    }

    void clearAndKeepStats() {
        for_each(m_chains.begin(), m_chains.end(), [](ChainResult & cr){
            cr.m_image->clear();
            cr.m_samples = 0;
        });
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
    /// Holds result of a single chain
    struct ChainResult {
        ref<ImageBlock> m_image; ///< Holds image buffer for this chain
        Float m_samples; ///< Number of samples
    };
    std::vector<ChainResult> m_chains;
    UPSMCMCStats m_stats;
    const UPSMCMCConfiguration & m_config;
    // Resolution
    Vector2i m_resolution;
};

MTS_NAMESPACE_END

#endif /* __MITSUBA_UPSMCMC_RESULT_H_ */