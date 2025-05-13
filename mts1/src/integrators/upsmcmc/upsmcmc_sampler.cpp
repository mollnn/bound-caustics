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

#include "upsmcmc_sampler.h"

MTS_NAMESPACE_BEGIN

UPSMCMCSampler::UPSMCMCSampler(const UPSMCMCConfiguration &config, size_t seed) : Sampler(Properties()) {
    m_random = new Random(seed);
    m_s1 = 1.0f / 1024.0f;
    m_s2 = 1.0f / 64.0f;
    m_probLargeStep = config.primarySpace.pLarge;
    m_goalAcceptanceSmall = config.primarySpace.acceptanceGoal;
    m_initialLargeStepProbability = config.primarySpace.pLarge;
    m_adaptivity = config.primarySpace.adaptivity;
    configure();
}

UPSMCMCSampler::UPSMCMCSampler(UPSMCMCSampler *sampler) : Sampler(Properties()),
m_random(sampler->m_random) {
    m_s1 = sampler->m_s1;
    m_s2 = sampler->m_s2;
    m_probLargeStep = sampler->m_probLargeStep;
    m_goalAcceptanceSmall = sampler->m_goalAcceptanceSmall;
    m_initialLargeStepProbability = sampler->m_initialLargeStepProbability;
    m_adaptivity = sampler->m_adaptivity;
    configure();
}

UPSMCMCSampler::UPSMCMCSampler(Stream *stream, InstanceManager *manager)
: Sampler(stream, manager) {
    SLog(EError, "Streaming is disabled");
}

void UPSMCMCSampler::serialize(Stream *stream, InstanceManager *manager) const {
    SLog(EError, "Streaming is disabled");
}

void UPSMCMCSampler::configure() {
    m_logRatio = -math::fastlog(m_s2 / m_s1);
    m_time = 0;
    m_largeStepTime = 0;
    m_largeStep = false;
    m_sampleIndex = 0;
    m_sampleCount = 0;
    m_totalSmallMutations = 0;
    m_totalLargeMutations = 0;
    m_acceptedSmallMutations = 0;
    m_acceptedLargeMutations = 0;
    m_nonZeroLargeMutations = 0;
    m_smallMutationSize = m_s2;
    m_mutationUpdates = 1;
}

UPSMCMCSampler::~UPSMCMCSampler() { }

void UPSMCMCSampler::accept() {
    if (m_largeStep) {
        m_largeStepTime = m_time;
        ++m_acceptedLargeMutations;
        ++m_totalLargeMutations;
        ++m_nonZeroLargeMutations;
    }
    else {
        ++m_acceptedSmallMutations;
        ++m_totalSmallMutations;
    }
    m_time++;
    m_backup.clear();
    m_sampleIndex = 0;
    recomputeStepSize();
}

void UPSMCMCSampler::reset() {
    m_time = m_sampleIndex = m_largeStepTime = 0;
    m_u.clear();
}

void UPSMCMCSampler::reject(bool nonZero) {
    if (m_largeStep) {
        ++m_totalLargeMutations;
        m_nonZeroLargeMutations += (int)nonZero;
    }
    else {
        ++m_totalSmallMutations;
    }
    for (size_t i = 0; i<m_backup.size(); ++i)
        m_u[m_backup[i].first] = m_backup[i].second;
    m_backup.clear();
    m_sampleIndex = 0;
    recomputeStepSize();
}

void UPSMCMCSampler::recomputeStepSize() {
    if (!m_adaptivity)
        return;
    // Large mutation
    if (m_largeStep) {
        /*if (m_totalSmallMutations == 0 || m_totalLargeMutations == 0) {
            m_probLargeStep = m_initialLargeStepProbability;
        }
        else {

            Float n0 = m_nonZeroLargeMutations / (Float)m_totalLargeMutations;
            Float nl = m_acceptedLargeMutations / (Float)m_totalLargeMutations;
            Float ns = m_acceptedLargeMutations / (Float)m_totalSmallMutations;

            if ((nl / n0) < 0.1f)
                m_probLargeStep = 0.25f;
            else
                m_probLargeStep = std::min(std::max(ns / (2.f*(ns - nl)), 0.25f), 1.f);
        }*/
    }
    else {
        // Small mutation
        if (m_totalSmallMutations > 0) {
            Float goal = m_goalAcceptanceSmall;
            Float ratio = (m_acceptedSmallMutations + m_acceptedLargeMutations) / (Float)(m_totalSmallMutations + m_totalLargeMutations);
            /*if (m_totalLargeMutations * 10 < m_totalSmallMutations)
                goal /= 1.f - m_initialLargeStepProbability;*/
            Float newSize = m_smallMutationSize + (ratio - goal) / m_mutationUpdates;
            if (newSize > 0.f && newSize < 1.f) {
                m_smallMutationSize = newSize;
                ++m_mutationUpdates;
            }
        }
    }
}

Float UPSMCMCSampler::primarySample(size_t i) {
    // === Lazy update
    while (i >= m_u.size())
        m_u.push_back(SampleStruct(m_random->nextFloat()));

    // === Overwise test number
    if (m_u[i].modify < m_time) {
        if (m_largeStep) {
            // Generate large step
            m_backup.push_back(std::pair<size_t, SampleStruct>(i, m_u[i]));
            m_u[i].modify = m_time;
            m_u[i].value = m_random->nextFloat();
        }
        else {
            // === If this random number is reseted by a large step
            // but we didn't alread do it.
            // we generate the random number for large step
            if (m_u[i].modify < m_largeStepTime) {
                m_u[i].modify = m_largeStepTime;
                m_u[i].value = m_random->nextFloat();
            }

            // === Rattraper le retard dans la mutation
            // Jusqu'a t-1
            while (m_u[i].modify + 1 < m_time) {
                m_u[i].value = mutate(m_u[i].value);
                m_u[i].modify++; // Said new value in modify
            }

            // === Save it
            m_backup.push_back(std::pair<size_t, SampleStruct>(i, m_u[i]));

            // Mut it
            m_u[i].value = mutate(m_u[i].value);
            m_u[i].modify++;
        }
    }

    return m_u[i].value;
}

ref<Sampler> UPSMCMCSampler::clone() {
    ref<UPSMCMCSampler> sampler = new UPSMCMCSampler(this);
    sampler->m_sampleCount = m_sampleCount;
    sampler->m_sampleIndex = m_sampleIndex;
    sampler->m_random = new Random(m_random);
    return sampler.get();
}

Float UPSMCMCSampler::next1D() {
    return primarySample(m_sampleIndex++);
}

Point2 UPSMCMCSampler::next2D() {
    /// Enforce a specific order of evaluation
    Float value1 = primarySample(m_sampleIndex++);
    Float value2 = primarySample(m_sampleIndex++);
    return Point2(value1, value2);
}

std::string UPSMCMCSampler::toString() const {
    std::ostringstream oss;
    oss << "UPSMCMCSampler[" << endl
        << "  mutation size = " << m_smallMutationSize << endl
        << "  accept ratio = " << m_acceptedSmallMutations / (Float)m_totalSmallMutations << endl
        << "  updates = " << m_mutationUpdates << endl
        << "]";
    return oss.str();
}

MTS_IMPLEMENT_CLASS_S(UPSMCMCSampler, false, Sampler)
MTS_NAMESPACE_END
