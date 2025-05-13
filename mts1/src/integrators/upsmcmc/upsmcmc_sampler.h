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


#if !defined(__UPSMCMC_SAMPLER_H)
#define __UPSMCMC_SAMPLER_H

#include <mitsuba/render/sampler.h>
#include <mitsuba/core/random.h>
#include "upsmcmc.h"

MTS_NAMESPACE_BEGIN

/**
* Sampler implementation as described in
* 'A Simple and Robust Mutation Strategy for the
* Metropolis Light Transport Algorithm' by Kelemen et al.
*/
class UPSMCMCSampler : public Sampler {
public:
    // Construct a new MLT sampler
    UPSMCMCSampler(const UPSMCMCConfiguration &conf, size_t seed);

    /**
    * \brief Construct a new sampler, which operates on the
    * same random number generator as \a sampler.
    */
    UPSMCMCSampler(UPSMCMCSampler *sampler);

    /// Unserialize from a binary data stream
    UPSMCMCSampler(Stream *stream, InstanceManager *manager);

    /// Set up the internal state
    void configure();

    /// Serialize to a binary data stream
    void serialize(Stream *stream, InstanceManager *manager) const;

    /// Returns large step probability
    inline Float largeStepProb() const {
        return m_probLargeStep;
    }

    /// Set whether the current step should be large
    inline void setLargeStep(bool value) { m_largeStep = value; }

    /// Check if the current step is a large step
    inline bool isLargeStep() const { return m_largeStep; }

    /// Retrieve the next component value from the current sample
    virtual Float next1D();

    /// Retrieve the next two component values from the current sample
    virtual Point2 next2D();

    /// Return a string description
    virtual std::string toString() const;

    /// 1D mutation routine
    inline Float mutate(Float value) {
        Float sample = m_random->nextFloat();
        bool add;

        if (sample < 0.5f) {
            add = true;
            sample *= 2.0f;
        }
        else {
            add = false;
            sample = 2.0f * (sample - 0.5f);
        }

        Float dv = m_adaptivity ? powf(sample, (1 / m_smallMutationSize) + 1.f) :
            m_s2 * math::fastexp(sample * m_logRatio);
        if (add) {
            value += dv;
            if (value >= 1)
                value -= 1;
        }
        else {
            value -= dv;
            if (value < 0)
                value += 1;
        }
        return value;
    }

    /// Return a primary sample
    Float primarySample(size_t i);

    /// Reset (& start with a large mutation)
    void reset();

    /// Accept a mutation
    void accept();

    /// Reject a mutation
    void reject(bool nonZero);

    void disbableMutations() {
        m_time--;
    }

    void enableMutations() {
        m_time++;
    }

    void resetStats() {
        m_totalSmallMutations = 0;
        m_totalLargeMutations = 0;
        m_acceptedSmallMutations = 0;
        m_acceptedLargeMutations = 0;
        m_nonZeroLargeMutations = 0;
        m_smallMutationSize = m_s2;
        m_mutationUpdates = 1;
    }

    void swapStats(UPSMCMCSampler * other) {
        std::swap(other->m_totalSmallMutations, m_totalSmallMutations);
        std::swap(other->m_totalLargeMutations, m_totalLargeMutations);
        std::swap(other->m_acceptedSmallMutations, m_acceptedSmallMutations);
        std::swap(other->m_acceptedLargeMutations, m_acceptedLargeMutations);
        std::swap(other->m_nonZeroLargeMutations, m_nonZeroLargeMutations);
        std::swap(other->m_smallMutationSize, m_smallMutationSize);
        std::swap(other->m_mutationUpdates, m_mutationUpdates);
    }

    void swapAttempt() {
        ++m_totalLargeMutations;
    }

    void swapSuccess() {
        ++m_acceptedLargeMutations;
    }
    /// Recompute adaptive step size
    void recomputeStepSize();

    /// Replace the underlying random number generator
    inline void setRandom(Random *random) { m_random = random; }

    /// Return the underlying random number generator
    inline Random *getRandom() { return m_random; }

    /* The following functions do nothing in this implementation */
    virtual void advance() { }
    virtual void generate(const Point2i &pos) { }

    /* The following functions are unsupported by this implementation */
    void request1DArray(size_t size) { Log(EError, "request1DArray(): Unsupported!"); }
    void request2DArray(size_t size) { Log(EError, "request2DArray(): Unsupported!"); }
    void setSampleIndex(size_t sampleIndex) { Log(EError, "setSampleIndex(): Unsupported!"); }
    ref<Sampler> clone();

    void getPrimarySpaceSample(std::vector<Float> & out) {
        out.resize(m_u.size());
        for (size_t i = 0; i < m_u.size(); ++i)
            out[i] = m_u[i].value;
    }

    void setPrimarySpaceSample(const Float * in, size_t length) {
        m_u.resize(length, SampleStruct(0));
        for (size_t i = 0; i < m_u.size(); ++i) {
            m_u[i].value = in[i];
            m_u[i].modify = m_time;
        }
    }

    MTS_DECLARE_CLASS()
protected:
    /// Virtual destructor
    virtual ~UPSMCMCSampler();
protected:
    struct SampleStruct {
        Float value;
        size_t modify;

        inline SampleStruct(Float value) : value(value), modify(0) { }
    };

    ref<Random> m_random;
    Float m_s1, m_s2, m_logRatio;
    bool m_largeStep;
    std::vector<std::pair<size_t, SampleStruct> > m_backup;
    std::vector<SampleStruct> m_u;
    size_t m_time, m_largeStepTime;
    Float m_probLargeStep;
    bool m_adaptivity;

    // Adaptivity stuff
    size_t m_totalSmallMutations;
    size_t m_totalLargeMutations;
    size_t m_acceptedSmallMutations;
    size_t m_acceptedLargeMutations;
    size_t m_nonZeroLargeMutations;
    Float m_goalAcceptanceSmall;
    Float m_initialLargeStepProbability;
    Float m_smallMutationSize;
    size_t m_mutationUpdates;
};

MTS_NAMESPACE_END

#endif /* __PSSMLT_SAMPLER_H */
