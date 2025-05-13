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
#if !defined(__MITSUBA_UPSMCMC_STATS_H_)
#define __MITSUBA_UPSMCMC_STATS_H_

#include "upsmcmc.h"

MTS_NAMESPACE_BEGIN

struct UPSMCMCStats {

    struct ChainStats {
        size_t largeNoZero; ///< Large mutation that had non-zero illumination
        size_t largeAccepted; ///< Accepted large mutations
        size_t largeTotal; ///< All large mutations
        size_t smallAccepted; ///< Accepted small mutations
        size_t smallTotal; ///< All small mutations
        size_t lumSamples;
        Float luminanceAccum; ///< Accumulated luminance from large steps
        size_t swapsAccepted;
        size_t swapsTotal;
        size_t connectionsMC;
        size_t connectionsVIS;
        size_t connectionsMCNonZero;
        size_t connectionsVISNonZero;
    };

    std::vector<ChainStats> chains;

    UPSMCMCStats(size_t chainCount) {
        chains.resize(chainCount);
    }

    void clear() {
        for (auto it = chains.begin(); it != chains.end(); ++it) {
            it->largeNoZero = 0;
            it->largeAccepted = 0;
            it->largeTotal = 0;
            it->smallAccepted = 0;
            it->smallTotal = 0;
            it->swapsAccepted = 0;
            it->swapsTotal = 0;
            it->lumSamples = 0;
            it->luminanceAccum = 0.f;
            it->connectionsMC = 0;
            it->connectionsVIS = 0;
            it->connectionsMCNonZero = 0;
            it->connectionsVISNonZero = 0;
        }
    }

    void clearLumSamples() {
        for (auto it = chains.begin(); it != chains.end(); ++it) {
            it->lumSamples = 0;
            it->luminanceAccum = 0.f;
        }
    }

    /// Accumulates statistics
    void add(const UPSMCMCStats& other) {
        SAssert(other.chains.size() == chains.size());
        for (size_t i = 0; i < chains.size(); ++i) {
            ChainStats & chain = chains[i];
            const ChainStats & otherChain = other.chains[i];
            chain.largeNoZero += otherChain.largeNoZero;
            chain.largeAccepted += otherChain.largeAccepted;
            chain.largeTotal += otherChain.largeTotal;
            chain.smallAccepted += otherChain.smallAccepted;
            chain.smallTotal += otherChain.smallTotal;
            chain.swapsAccepted += otherChain.swapsAccepted;
            chain.swapsTotal += otherChain.swapsTotal;
            chain.lumSamples += otherChain.lumSamples;
            chain.luminanceAccum += otherChain.luminanceAccum;
            chain.connectionsMC += otherChain.connectionsMC;
            chain.connectionsVIS += otherChain.connectionsVIS;
            chain.connectionsMCNonZero += otherChain.connectionsMCNonZero;
            chain.connectionsVISNonZero += otherChain.connectionsVISNonZero;
        }
    }

    std::string toString() const {
        std::stringstream ss;
        ss << "UPSMCMCStats[";

        ss << "]";
        return ss.str();
    }

    void load(Stream * stream) {
        SLog(EError, "Streaming is disabled");
    }

    void save(Stream * stream) const {
        SLog(EError, "Streaming is disabled");
    }
};

MTS_NAMESPACE_END

#endif /* __MITSUBA_UPSMCMC_STATS_H_ */