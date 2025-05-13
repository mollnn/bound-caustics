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

#if !defined(__UPSMCMC_H)
#define __UPSMCMC_H

#include <mitsuba/core/bitmap.h>
#include <mitsuba/bidir/manifold.h>
#include <mitsuba/bidir/mut_manifold.h>
#include <mitsuba/bidir/pathsampler.h>
#include "parsable_logger.h"

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                         Configuration storage                        */
/* ==================================================================== */

/**
* \brief Stores all configuration parameters used
* by the UPS MCMC rendering implementation
*/
struct UPSMCMCConfiguration {

	// All supported MCMC algorithms
	enum EALGORITHMS {
        EUPS = 0,
        EUPS_MCMC = 1
	};

	// MMLT + PSMLT specific settings
	struct PrimarySpace {
		// Large mutation initial probability
		Float pLarge;
		// Adaptive mutations?
		bool adaptivity;
		// Adaptivity acceptance goal
		Float acceptanceGoal;
	} primarySpace;

	int maxDepth; // Max path depth
	int seedSamples; // Number of samples used for luminance computation
	std::vector<Float> normalization; // Normalization (for each used chain)
	size_t timeout; // Timeout if set
	int maxTimeImgDump; // How often should we dump image
	
	int algorithm; // MCMC algorithm (PSMLT = 0, MMLT = 1, MLT = 2) 
    Float initialRadius; // Initial photon query radius (0 = infer based on scene size and sensor resolution)
    Float alpha; // Alpha parameter from the paper (influences the speed, at which the photon radius is reduced)
	Float currentRadiusScale; // Radius scale for the current iteration
    int rrDepth;// Russian-roulette for psmlt/photon generation
	size_t maxIterations; // Maximum number of iterations = sample count
	size_t currentIteration; // Current iteration
    int nCores; // Number of computing cores
    AABB sceneAABB; // Scene bounding box
    Float nonVisible;
    size_t seed;
    size_t subPaths; // Number of both camera and light supbaths in one iteration
    size_t pathsPerIter; // Number of paths per iteration
    mutable ParsableLogger * logger; // Logger

	inline UPSMCMCConfiguration() { }

	// Helper function, all alphabetical characters of s are transformed to upper case.
	std::string strtoupper(const std::string & s) {
		std::string res = s;
		for_each(res.begin(), res.end(), [](char & c){c = toupper(c); });
		return res;
	}

	// Parses algorithm value from string
	void parseAlgorithm(const std::string & alg) {
		std::string t = strtoupper(alg);
		if (t == "UPS")
			algorithm = EUPS;
		else if (t == "UPSMCMC")
			algorithm = EUPS_MCMC;
		else {
			SLog(EWarn, "Unknown algorithm: %s switching to UPS.", t.c_str());
			algorithm = EUPS;
		}
	}

	// Returns algorithm name
	std::string tostrAlgorithm() const  {
		switch (algorithm) {
        case EUPS: return "UPS";
        case EUPS_MCMC: return "UPSMCMC";
		default:SLog(EError, "Internal error - algorithm not set");
		}
		return "";// Should never get here
	}

    bool isUPS() const {
        return algorithm == EUPS;
    }

    bool isUPS_MCMC() const {
        return algorithm == EUPS_MCMC;
    }

    size_t chainCount() const {
        return normalization.size();
    }

    // Dumps settings to the console
    void dumpToLog() {
        (*logger).openTag("CONFIG");
        // First dump algorithm independent settings:
        (*logger) << "Maximum path length: " << maxDepth << '\n';
        (*logger) << "Timeout: " << timeout << '\n';
        (*logger) << "Algorithm: " << tostrAlgorithm() << '\n';
        (*logger) << "Paths per iteration: " << pathsPerIter << '\n';
        (*logger) << "Initial radius: " << initialRadius << '\n';
        (*logger) << "Alpha: " << alpha << '\n';
        (*logger) << "Russian roulette depth: " << rrDepth << '\n';
        (*logger) << tostrAlgorithm() << " Specific settings\n";
        (*logger) << "Initial large-step probability: " << primarySpace.pLarge << '\n';
        (*logger) << "Adaptivity: " << (primarySpace.adaptivity ? formatString("%f goal acceptance", primarySpace.acceptanceGoal).c_str() : "no") << '\n';
        (*logger).closeTag("CONFIG");
        (*logger).outputToConsole(EInfo);
    }

	inline UPSMCMCConfiguration(Stream *stream) {
		SLog(EError, "Streaming is disabled");
	}

	inline void serialize(Stream *stream) const {
		SLog(EError, "Streaming is disabled");
	}
};

// OTHER COMMON DEFINITIONS

MTS_NAMESPACE_END

#endif /* __UPSMCMC_H */
