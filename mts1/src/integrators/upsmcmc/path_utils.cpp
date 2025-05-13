#include "path_utils.h"
#include <mitsuba/bidir/rsampler.h>
#include <mitsuba/bidir/util.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/plugin.h>

MTS_NAMESPACE_BEGIN

PathUtils::PathUtils(const Scene *scene, Sampler *emitterSampler,
Sampler *sensorSampler, Sampler *directSampler,
const UPSMCMCConfiguration & config) :
PathSampler(PathSampler::EBidirectional, scene, emitterSampler, sensorSampler, directSampler,
    config.maxDepth, config.rrDepth,
    false, true), m_config(config) {
}

void PathUtils::sampleLightAndDoUPS(SplatList &list, const CameraStorage * cameraStorage, size_t cameraPathIndex) {
    Assert(cameraStorage);
    list.clear();
    Spectrum *importanceWeights = NULL, *radianceWeights = NULL;

    // Get light path
    const Sensor *sensor = m_scene->getSensor();
    m_time = sensor->getShutterOpen(); /* TODO if blur is necessary*/
    // Perform walk from the light
    m_emitterSubpath.initialize(m_scene, m_time, EImportance, m_pool);
    // Length is limited by max. depth and Russian roulette may be applied
    m_emitterSubpath.randomWalk(m_scene, m_emitterSampler, m_emitterDepth,
        m_rrDepth, EImportance, m_pool);
    // Get camera path
	m_cameraPath = cameraStorage->getCameraPaths().selectPathProportionally(cameraPathIndex);
    if (!cameraStorage->getCameraPaths().clonePath(m_sensorSubpath, m_pool, m_cameraPath, -1))
        return;
    /* Compute the combined weights along the light sub-path */
    importanceWeights = (Spectrum *)alloca(m_emitterSubpath.vertexCount() * sizeof(Spectrum));
    importanceWeights[0] = Spectrum(1.0f);
    for (size_t i = 1; i < m_emitterSubpath.vertexCount(); ++i)
        importanceWeights[i] = importanceWeights[i - 1] *
        m_emitterSubpath.vertex(i - 1)->weight[EImportance] *
        m_emitterSubpath.vertex(i - 1)->rrWeight *
        m_emitterSubpath.edge(i - 1)->weight[EImportance];

    /* Compute the combined weights along the camera sub-path */
    radianceWeights = (Spectrum *)alloca(m_sensorSubpath.vertexCount()  * sizeof(Spectrum));

    radianceWeights[0] = Spectrum(1.0f);
    for (size_t i = 1; i<m_sensorSubpath.vertexCount(); ++i)
        radianceWeights[i] = radianceWeights[i - 1] *
        m_sensorSubpath.vertex(i - 1)->weight[ERadiance] *
        m_sensorSubpath.vertex(i - 1)->rrWeight *
        m_sensorSubpath.edge(i - 1)->weight[ERadiance];

    // Sets the "default" splatting position for all paths that use at least 3 vertices from m_sensorSubpath
    m_samplePos = Point2(0.0f);
    if (m_sensorSubpath.vertexCount() > 2) {
        m_samplePos = m_cameraPath.m_samplePos;
        list.append(m_cameraPath.m_samplePos, Spectrum(0.0f));
        list.pathTracing.first = m_samplePos;
        m_radius = m_cameraPath.m_radius;
        m_mergeWeight = m_radius * m_radius * M_PI * m_config.subPaths;
    }

    for (int s = (int)m_emitterSubpath.vertexCount() - 1; s >= 0; --s) {
        /* Determine the range of camera vertices to be traversed,
        while respecting the specified maximum path length */
        int minT = std::max(2 - s, 0),
            maxT = (int)m_sensorSubpath.vertexCount() - 1;
        maxT = std::min(maxT, m_maxDepth + 1 - s);
        for (int t = maxT; t >= minT; --t) {
            computeBDPTTechnique(list, s, t, importanceWeights, radianceWeights);
        }
        if (s > 2 && s <= m_maxDepth && m_emitterSubpath.vertex(s)->isSurfaceInteraction()
            && m_emitterSubpath.vertex(s)->isConnectable())
            computeMergingTechnique(list, s, -1, importanceWeights, cameraStorage);
    }
}

void PathUtils::clearSplats() {
    /* CLEANUP - Release any used edges and vertices back to the memory pool */
    m_sensorSubpath.release(m_pool);
    m_emitterSubpath.release(m_pool);
    m_connectionSubpath.release(m_pool);
}

void PathUtils::computeBDPTTechnique(SplatList &list, int s, int t,
    const Spectrum * importanceWeights, const Spectrum * radianceWeights)
{
    PathVertex tempEndpoint, tempSample, vsTemp, vtTemp;
    PathEdge tempEdge, connectionEdge;
    Point2 samplePos(0.0f);

    PathVertex
        *vsPred = m_emitterSubpath.vertexOrNull(s - 1),
        *vtPred = m_sensorSubpath.vertexOrNull(t - 1),
        *vs = m_emitterSubpath.vertex(s),
        *vt = m_sensorSubpath.vertex(t);
    PathEdge
        *vsEdge = m_emitterSubpath.edgeOrNull(s - 1),
        *vtEdge = m_sensorSubpath.edgeOrNull(t - 1);

    RestoreMeasureHelper rmh0(vs), rmh1(vt);

    /* Will be set to true if direct sampling was used */
    bool sampleDirect = false;

    /* Number of edges of the combined subpaths */
    int depth = s + t - 1;
    
    if (depth == 1 && s == 1 && t == 1)
        return;

    /* Allowed remaining number of ENull vertices that can
    be bridged via pathConnect (negative=arbitrarily many) */
    int remaining = m_maxDepth - depth;

    /* Will receive the path weight of the (s, t)-connection */
    Spectrum value(0.f);

    /* Measure associated with the connection vertices */
    EMeasure vsMeasure = EArea, vtMeasure = EArea;


    /* Account for the terms of the measurement contribution
    function that are coupled to the connection endpoints */
    if (vs->isEmitterSupernode()) {
        /* If possible, convert 'vt' into an emitter sample */
        if (!vt->cast(m_scene, PathVertex::EEmitterSample) || vt->isDegenerate())
            return;

        value = radianceWeights[t] *
            vs->eval(m_scene, vsPred, vt, EImportance) *
            vt->eval(m_scene, vtPred, vs, ERadiance);
    }
    else if (vt->isSensorSupernode()) {
        /* If possible, convert 'vs' into an sensor sample */
        if (!vs->cast(m_scene, PathVertex::ESensorSample) || vs->isDegenerate())
            return;

        /* Make note of the changed pixel sample position */
        if (!vs->getSamplePosition(vsPred, samplePos))
            return;

        value = importanceWeights[s] *
            vs->eval(m_scene, vsPred, vt, EImportance) *
            vt->eval(m_scene, vtPred, vs, ERadiance);
    }
    else if (m_sampleDirect && ((t == 1 && s > 1) || (s == 1 && t > 1))) {
        /* s==1/t==1 path: use a direct sampling strategy if requested */
        if (s == 1) {
            if (vt->isDegenerate())
                return;

            /* Generate a position on an emitter using direct sampling */
			value = radianceWeights[t] * vt->sampleDirect(m_scene,m_config.isUPS_MCMC() ? m_sensorSampler : m_directSampler,
                &tempEndpoint, &tempEdge, &tempSample, EImportance);

            if (value.isZero())
                return;
            vs = &tempSample; vsPred = &tempEndpoint; vsEdge = &tempEdge;
            value *= vt->eval(m_scene, vtPred, vs, ERadiance);
            vsMeasure = vs->getAbstractEmitter()->needsDirectionSample() ? EArea : EDiscrete;
            vt->measure = EArea;
        }
        else {
            if (vs->isDegenerate())
                return;
            /* Generate a position on the sensor using direct sampling */
            value = importanceWeights[s] * vs->sampleDirect(m_scene, m_directSampler,
                &tempEndpoint, &tempEdge, &tempSample, ERadiance);
            if (value.isZero())
                return;
            vt = &tempSample; vtPred = &tempEndpoint; vtEdge = &tempEdge;
            value *= vs->eval(m_scene, vsPred, vt, EImportance);
            vtMeasure = vt->getAbstractEmitter()->needsDirectionSample() ? EArea : EDiscrete;
            vs->measure = EArea;
        }

        sampleDirect = true;
    }
    else {
        /* Can't connect degenerate endpoints */
        if (vs->isDegenerate() || vt->isDegenerate())
            return;

        value = importanceWeights[s] * radianceWeights[t] *
            vs->eval(m_scene, vsPred, vt, EImportance) *
            vt->eval(m_scene, vtPred, vs, ERadiance);

        /* Temporarily force vertex measure to EArea. Needed to
        handle BSDFs with diffuse + specular components */
        vs->measure = vt->measure = EArea;
    }

    /* Attempt to connect the two endpoints, which could result in
    the creation of additional vertices (index-matched boundaries etc.) */
    int interactions = remaining;
    if (value.isZero() || !connectionEdge.pathConnectAndCollapse(
        m_scene, vsEdge, vs, vt, vtEdge, interactions))
        return;

    depth += interactions;

    if (m_excludeDirectIllum && depth <= 2)
        return;

    /* Account for the terms of the measurement contribution
    function that are coupled to the connection edge */
    if (!sampleDirect)
        value *= connectionEdge.evalCached(vs, vt, PathEdge::EGeneralizedGeometricTerm);
    else
        value *= connectionEdge.evalCached(vs, vt, PathEdge::ETransmittance |
        (s == 1 ? PathEdge::ECosineRad : PathEdge::ECosineImp));

    /* Determine the pixel sample position when necessary */
    if (vt->isSensorSample() && !vt->getSamplePosition(vs, samplePos))
        return;

    if (sampleDirect) {
        /* A direct sampling strategy was used, which generated
        two new vertices at one of the path ends. Temporarily
        modify the path to reflect this change */
        if (t == 1)
            m_sensorSubpath.swapEndpoints(vtPred, vtEdge, vt);
        else
            m_emitterSubpath.swapEndpoints(vsPred, vsEdge, vs);
    }

    Float tmpRadius = getRayRadius(m_time, s, t) * m_config.currentRadiusScale;
    Float tmpMergeWeight = tmpRadius * tmpRadius * M_PI * m_config.subPaths;
    std::swap(m_radius, tmpRadius);
    std::swap(m_mergeWeight, tmpMergeWeight);

    /* Compute the multiple importance sampling weight */
    Float misW = depth > 1 ? miWeight(&connectionEdge, EBDPT, s, t) : 1.f;
    value *= misW;

    std::swap(m_radius, tmpRadius);
    std::swap(m_mergeWeight, tmpMergeWeight);

    if (sampleDirect) {
        /* Now undo the previous change */
        if (t == 1)
            m_sensorSubpath.swapEndpoints(vtPred, vtEdge, vt);
        else
            m_emitterSubpath.swapEndpoints(vsPred, vsEdge, vs);
    }

    if (t < 2) {
        list.append(samplePos, value);
    }
    else {
        BDAssert(m_sensorSubpath.vertexCount() > 2);
		if (m_config.isUPS_MCMC() && s < 2)
            list.pathTracing.second += value;
        else
            list.accum(0, value);
    }
}

void PathUtils::computeMergingTechnique(SplatList &list, int s, int tt,
    const Spectrum * importanceWeights, const CameraStorage * cameraStorage) {
    Path::swap(m_tempPath, m_sensorSubpath);
    Float tmpRadius = m_radius;
    Float tmpMergeWeight = m_mergeWeight;

    const PathsCache & cameraPaths = cameraStorage->getCameraPaths();
    const PathVertex & pv = *m_emitterSubpath.vertex(s);
    // Select the preceding vertex
    const PathVertex & ppv = *m_emitterSubpath.vertex(s-1);
    int maxVertexId = m_maxDepth + 2 - s;
    // Execute gather point query at the selected gather point
    cameraStorage->gatherPointQuery(pv.getPosition(), [&](int pathId, int t) {
        // Select the path
        const PathsCache::SimplePath & sp = cameraPaths.getPath(pathId);
        
        const PathVertex & cameraVertex = cameraPaths.getPathVertex(sp, t);
        const Intersection & its = cameraVertex.getIntersection();

        Normal photonNormal(pv.getGeometricNormal());
        Vector wi = ppv.getPosition() - pv.getPosition();
        Float pLength = wi.length();
        wi /= pLength;
        Float wiDotGeoN = absDot(photonNormal, wi);

        if ((t > maxVertexId)
            || (m_excludeDirectIllum && s == 2 && t == 2)
            || dot(photonNormal, its.shFrame.n) < 1e-1f
#ifndef MTS_NOSHADINGNORMAL
            || wiDotGeoN < 1e-2f
#endif
            )
            return;
        m_radius = sp.m_radius;
        m_mergeWeight = m_radius * m_radius * M_PI * m_config.subPaths;

        // Compute gather point power and copy vertices to sensor sub-path for MIS calculation:
        m_sensorSubpath.clear();
        Spectrum power(1.f);
        for (int i = 0; i < t; ++i) {
            const PathVertex & v = cameraPaths.getPathVertex(sp, i);
            const PathEdge & e = cameraPaths.getPathEdge(sp, i);
            power *= v.weight[ERadiance] * v.rrWeight * e.weight[ERadiance];
            // The Path interface doesn't allow adding const vertices,edges. But we will not modify them anyway!
            m_sensorSubpath.append(const_cast<PathEdge *>(&e), const_cast<PathVertex *>(&v));
        }
        m_sensorSubpath.append(const_cast<PathVertex *>(&cameraVertex));

        // Add light path weight
        power *= importanceWeights[s];

        BSDFSamplingRecord bRecImp(its, its.toLocal(wi), its.wi, EImportance);
        BSDFSamplingRecord bRecRad(its, bRecImp.wo, bRecImp.wi, ERadiance);

        const Spectrum bsdfVal = its.getBSDF()->eval(bRecImp);


#ifdef MTS_NOSHADINGNORMAL
        Spectrum value = power * bsdfVal / std::abs(Frame::cosTheta(bRecImp.wo));
#else
        Spectrum value = power * bsdfVal;
#endif
        if (value.isZero())
            return;
#ifndef MTS_NOSHADINGNORMAL
        // Account for non-symmetry due to shading normals
        value *= std::abs(Frame::cosTheta(bRecImp.wi) /
            (wiDotGeoN * std::abs(Frame::cosTheta(bRecImp.wo))));
#endif
        // Prepare pdfs for MIS computation
        m_pdfImportance = its.getBSDF()->pdf(bRecImp);
        m_pdfRadiance = its.getBSDF()->pdf(bRecRad);
        // Convert pdfs to area measure if connection is possible
        const PathEdge * impEdge = m_sensorSubpath.edge(t - 1);
        const PathVertex * impSucc = m_sensorSubpath.vertex(t - 1);
        const PathVertex * impCur = m_sensorSubpath.vertex(t);
        if (impSucc->isConnectable()) {
            m_pdfImportance /= impEdge->length * impEdge->length;
            m_pdfImportance *= std::abs(
                (impSucc->isOnSurface() ? dot(-impEdge->d, impSucc->getGeometricNormal()) : 1));
        }
        else
            m_pdfImportance /= std::abs(
            (impCur->isOnSurface() ? dot(impEdge->d, impCur->getGeometricNormal()) : 1));
        if (ppv.isConnectable()) {
            m_pdfRadiance /= pLength * pLength;
            m_pdfRadiance *= std::abs(
                (ppv.isOnSurface() ? dot(-wi, ppv.getGeometricNormal()) : 1));
        }
        else
            m_pdfRadiance /= std::abs(
            (pv.isOnSurface() ? dot(wi, pv.getGeometricNormal()) : 1));

        value /= m_mergeWeight;

        const Float misw = miWeight(NULL, EMERGING, s - 1, t); //UPS MIS weight
        value *= misw;

        list.append(sp.m_samplePos, value);
    });
    std::swap(tmpRadius,m_radius);
    std::swap(tmpMergeWeight,m_mergeWeight);
    Path::swap(m_tempPath, m_sensorSubpath);
}

#if 1 // POWER HEURISTIC
inline double MISw(double v) {
    return v * v;
}
#else // BALANCE HEURISTIC
inline double MISw(double v) {
    return v;
}
#endif

Float PathUtils::miWeight(const PathEdge *connectionEdge, int technique, int s, int t) {
    int k = s + t + 1, // Path edges
        n = k + 1; // Path vertices
    const PathVertex
        *vsPred = m_emitterSubpath.vertexOrNull(s - 1),
        *vtPred = m_sensorSubpath.vertexOrNull(t - 1),
        *vs = m_emitterSubpath.vertex(s),
        *vt = m_sensorSubpath.vertex(t);

    /* pdfImp[i] and pdfRad[i] store the area/volume density of vertex
    'i' when sampled from the adjacent vertex in the emitter
    and sensor direction, respectively. */

    Float ratioEmitterDirect = 0.0f, ratioSensorDirect = 0.0f;
    Float *pdfImp = (Float *)alloca(n * sizeof(Float)),
        *pdfRad = (Float *)alloca(n * sizeof(Float));
    bool  *connectable = (bool *)alloca(n * sizeof(bool)),
        *isNull = (bool *)alloca(n * sizeof(bool)),
        *isDiffuseSurface = (bool *)alloca(n * sizeof(bool));

    /* Keep track of which vertices are connectable / null interactions */
    int pos = 0;
    for (int i = 0; i <= s; ++i) {
        const PathVertex *v = m_emitterSubpath.vertex(i);
        connectable[pos] = v->isConnectable();
        isNull[pos] = v->isNullInteraction() && !connectable[pos];
        isDiffuseSurface[pos] = connectable[pos] && v->isSurfaceInteraction() && pos > 2;
        pos++;
    }

    for (int i = t; i >= 0; --i) {
        const PathVertex *v = m_sensorSubpath.vertex(i);
        connectable[pos] = v->isConnectable();
        isNull[pos] = v->isNullInteraction() && !connectable[pos];
        isDiffuseSurface[pos] = connectable[pos] && v->isSurfaceInteraction() && pos > 2;
        pos++;
    }

    bool sampleDirect = m_sampleDirect;
    if (k <= 3) // i.e. directly visible light source
        sampleDirect = false;
    EMeasure vsMeasure = EArea, vtMeasure = EArea;

    if (sampleDirect) {
        /* When direct sampling is enabled, we may be able to create certain
        connections that otherwise would have failed (e.g. to an
        orthographic camera or a directional light source) */
        const AbstractEmitter *emitter = (s > 0 ? m_emitterSubpath.vertex(1) : vt)->getAbstractEmitter();
        const AbstractEmitter *sensor = (t > 0 ? m_sensorSubpath.vertex(1) : vs)->getAbstractEmitter();

        EMeasure emitterDirectMeasure = emitter->getDirectMeasure();
        EMeasure sensorDirectMeasure = sensor->getDirectMeasure();

        connectable[0] = emitterDirectMeasure != EDiscrete && emitterDirectMeasure != EInvalidMeasure;
        connectable[1] = emitterDirectMeasure != EInvalidMeasure;
        connectable[k - 1] = sensorDirectMeasure != EInvalidMeasure;
        connectable[k] = sensorDirectMeasure != EDiscrete && sensorDirectMeasure != EInvalidMeasure;

        /* The following is needed to handle orthographic cameras &
        directional light sources together with direct sampling */
        if (t == 1)
            vtMeasure = sensor->needsDirectionSample() ? EArea : EDiscrete;
        else if (s == 1)
            vsMeasure = emitter->needsDirectionSample() ? EArea : EDiscrete;
    }

    /* Collect importance transfer area/volume densities from vertices */
    pos = 0;
    pdfImp[pos++] = 1.0;
    for (int i = 0; i < s; ++i) {
        pdfImp[pos++] = m_emitterSubpath.vertex(i)->pdf[EImportance]
            * m_emitterSubpath.edge(i)->pdf[EImportance];
        //SAssert(s < 2 || (1.f / rrPdf) == m_emitterSubpath.vertex(i)->rrWeight);
    }

    if (technique == EBDPT) {
        pdfImp[pos++] = vs->evalPdf(m_scene, vsPred, vt, EImportance, vsMeasure)
            * connectionEdge->pdf[EImportance];
    }
    else { // MERGING
        pdfImp[pos++] = m_emitterSubpath.vertex(s)->pdf[EImportance]
            * m_emitterSubpath.edge(s)->pdf[EImportance];
    }

    if (t > 0) {
        if (technique == EBDPT) {
            pdfImp[pos++] = vt->evalPdf(m_scene, vs, vtPred, EImportance, vtMeasure)
                * m_sensorSubpath.edge(t - 1)->pdf[EImportance];
        }
        else { // MERGING
            pdfImp[pos++] = m_pdfImportance * m_sensorSubpath.edge(t - 1)->pdf[EImportance]; // Solid angle pdf if connection is not available
        }

        for (int i = t - 1; i > 0; --i) {
            pdfImp[pos++] = m_sensorSubpath.vertex(i)->pdf[EImportance]
                * m_sensorSubpath.edge(i - 1)->pdf[EImportance];
        }
    }

    /* Collect radiance transfer area/volume densities from vertices */
    pos = n-1;
    pdfRad[pos--] = 1.0;
    for (int i = 0; i < t; ++i) {
        pdfRad[pos--] = m_sensorSubpath.vertex(i)->pdf[ERadiance]
            * m_sensorSubpath.edge(i)->pdf[ERadiance];
        //SAssert(t < 2 || (1.f / rrPdf) == m_sensorSubpath.vertex(i)->rrWeight);
    }

    if (technique == EBDPT) {
        pdfRad[pos--] = vt->evalPdf(m_scene, vtPred, vs, ERadiance, vtMeasure)
            * connectionEdge->pdf[ERadiance];
    }
    else {// MERGING
        pdfRad[pos--] = m_pdfRadiance * m_emitterSubpath.edge(s)->pdf[ERadiance];// Solid angle pdf if connection is not available
    }

    if (s > 0) {
        if (technique == EBDPT) {
            pdfRad[pos--] = vs->evalPdf(m_scene, vt, vsPred, ERadiance, vsMeasure)
                * m_emitterSubpath.edge(s - 1)->pdf[ERadiance];
        }
        else {// MERGING
            pdfRad[pos--] = m_emitterSubpath.vertex(s)->pdf[ERadiance]
                * m_emitterSubpath.edge(s - 1)->pdf[ERadiance];

        }

        for (int i = s - 1; i > 0; --i) {
            pdfRad[pos--] = m_emitterSubpath.vertex(i)->pdf[ERadiance]
                * m_emitterSubpath.edge(i - 1)->pdf[ERadiance];
        }

    }


    /* When the path contains specular surface interactions, it is possible
    to compute the correct MI weights even without going through all the
    trouble of computing the proper generalized geometric terms (described
    in the SIGGRAPH 2012 specular manifolds paper). The reason is that these
    all cancel out. But to make sure that that's actually true, we need to
    convert some of the area densities in the 'pdfRad' and 'pdfImp' arrays
    into the projected solid angle measure */
    for (int i = 1; i <= k - 3; ++i) {
        // skip the last camera vertex for EBDPT (because of the connection) and the last but one camera vertex for EMERGING (it might already have solid angle pdf!)
        if ((technique == EBDPT && i == s) || (technique == EMERGING && i == s + 1) 
            || !(connectable[i] && !connectable[i + 1]))
            continue;
        int ss = technique == EMERGING ? s + 1 : s;
        const PathVertex *cur = i <= ss ? m_emitterSubpath.vertex(i) : m_sensorSubpath.vertex(k - i);
        const PathVertex *succ = i + 1 <= ss ? m_emitterSubpath.vertex(i + 1) : m_sensorSubpath.vertex(k - i - 1);
        const PathEdge *edge = i < ss ? m_emitterSubpath.edge(i) : m_sensorSubpath.edge(k - i - 1);

        pdfImp[i + 1] *= edge->length * edge->length / std::abs(
            (succ->isOnSurface() ? dot(edge->d, succ->getGeometricNormal()) : 1) *
            (cur->isOnSurface() ? dot(edge->d, cur->getGeometricNormal()) : 1));
    }

    for (int i = k - 1; i >= 3; --i) {
        // skip the last light vertex for EBDPT (because of the connection) and the last but one light vertex for EMERGING (it might already have solid angle pdf!)
        if ((technique == EBDPT && i - 1 == s) || (technique == EMERGING && i == s + 1) 
            || !(connectable[i] && !connectable[i - 1]))
            continue;
        int ss = technique == EMERGING ? s + 1 : s;
        const PathVertex *cur = i <= ss ? m_emitterSubpath.vertex(i) : m_sensorSubpath.vertex(k - i);
        const PathVertex *succ = i - 1 <= ss ? m_emitterSubpath.vertex(i - 1) : m_sensorSubpath.vertex(k - i + 1);
        const PathEdge *edge = i <= ss ? m_emitterSubpath.edge(i - 1) : m_sensorSubpath.edge(k - i);

        pdfRad[i - 1] *= edge->length * edge->length / std::abs(
            (succ->isOnSurface() ? dot(edge->d, succ->getGeometricNormal()) : 1) *
            (cur->isOnSurface() ? dot(edge->d, cur->getGeometricNormal()) : 1));
    }

    int emitterRefIndirection = 2, sensorRefIndirection = k - 2;

    /* One more array sweep before the actual useful work starts -- phew! :)
    "Collapse" edges/vertices that were caused by BSDF::ENull interactions.
    The BDPT implementation is smart enough to connect straight through those,
    so they shouldn't be treated as Dirac delta events in what follows */
    for (int i = 1; i <= k - 3; ++i) {
        if (!connectable[i] || !isNull[i + 1])
            continue;

        int start = i + 1, end = start;
        while (isNull[end + 1])
            ++end;

        if (!connectable[end + 1]) {
            /// The chain contains a non-ENull interaction
            isNull[start] = false;
            continue;
        }

        const PathVertex *before = i <= s ? m_emitterSubpath.vertex(i) : m_sensorSubpath.vertex(k - i);
        const PathVertex *after = end + 1 <= s ? m_emitterSubpath.vertex(end + 1) : m_sensorSubpath.vertex(k - end - 1);

        Vector d = before->getPosition() - after->getPosition();
        Float lengthSquared = d.lengthSquared();
        d /= std::sqrt(lengthSquared);

        Float geoTerm = std::abs(
            (before->isOnSurface() ? dot(before->getGeometricNormal(), d) : 1) *
            (after->isOnSurface() ? dot(after->getGeometricNormal(), d) : 1)) / lengthSquared;

        pdfRad[start - 1] *= pdfRad[end] * geoTerm;
        pdfRad[end] = 1;
        pdfImp[start] *= pdfImp[end + 1] * geoTerm;
        pdfImp[end + 1] = 1;

       // When an ENull chain starts right after the emitter / before the sensor,
        //we must keep track of the reference vertex for direct sampling strategies. 
        if (start == 2)
            emitterRefIndirection = end + 1;
        else if (end == k - 2)
            sensorRefIndirection = start - 1;

        i = end;
    }

    double initial = 1.0f;

    /* When direct sampling strategies are enabled, we must
    account for them here as well */
    if (sampleDirect) {
        // Direct connection probability of the emitter 
        const PathVertex *sample = s>0 ? m_emitterSubpath.vertex(1) : vt;
        const PathVertex *ref = emitterRefIndirection <= s
            ? m_emitterSubpath.vertex(emitterRefIndirection) : m_sensorSubpath.vertex(k - emitterRefIndirection);
        EMeasure measure = sample->getAbstractEmitter()->getDirectMeasure();

        if (connectable[1] && connectable[emitterRefIndirection])
            ratioEmitterDirect = ref->evalPdfDirect(m_scene, sample, EImportance,
            measure == ESolidAngle ? EArea : measure) / pdfImp[1];

        // Direct connection probability of the sensor 
        sample = t>0 ? m_sensorSubpath.vertex(1) : vs;
        ref = sensorRefIndirection <= s ? m_emitterSubpath.vertex(sensorRefIndirection)
            : m_sensorSubpath.vertex(k - sensorRefIndirection);
        measure = sample->getAbstractEmitter()->getDirectMeasure();

        if (connectable[k - 1] && connectable[sensorRefIndirection])
            ratioSensorDirect = ref->evalPdfDirect(m_scene, sample, ERadiance,
            measure == ESolidAngle ? EArea : measure) / pdfRad[k - 1];

        if (technique == EBDPT) {
            if (s == 1)
                initial /= ratioEmitterDirect;
            else if (t == 1)
                initial /= ratioSensorDirect;
        }
    }

    if (technique == EMERGING) {
        initial /= getMergeWeight();
    }
    else
        initial /= getPhotonConnectionWeight(s);
    double weight = 1, pdf = initial;

    /* With all of the above information, the MI weight can now be computed.
    Since the goal is to evaluate the power heuristic, the absolute area
    product density of each strategy is interestingly not required. Instead,
    an incremental scheme can be used that only finds the densities relative
    to the (s,t) strategy, which can be done using a linear sweep. For
    details, refer to the Veach thesis, p.306. */
    for (int i = s + 1; i<k; ++i) {
        double next = pdf;
        
        if (technique == EBDPT || i > s + 1) {
            next *= (double)pdfImp[i];

            if (isDiffuseSurface[i]) {
                double value = next * getMergeWeight();
                weight += MISw(value);
            }
        }

        next /= (double)pdfRad[i];

        double value = next * getPhotonConnectionWeight(i);

        if (sampleDirect) {
            if (i == 1)
                value *= ratioEmitterDirect;
            else if (i == sensorRefIndirection)
                value *= ratioSensorDirect;
        }


        if (connectable[i] && (connectable[i + 1] || isNull[i + 1]))
           weight += MISw(value);

        pdf = next;
    }

    

    /* As above, but now compute pdf[i] with i<s (this is done by
    evaluating the inverse of the previous expressions). */
    pdf = initial;
    for (int i = technique == EBDPT ? s - 1 : s; i >= 0; --i) {
        double next = pdf;
            
        if (technique == EBDPT || i < s) {
            next *= (double)pdfRad[i + 1];
            if (isDiffuseSurface[i + 1]) {
                double value = next * getMergeWeight();
                weight += MISw(value);
            }
        }

        next /= (double)pdfImp[i + 1];

        double value = next * getPhotonConnectionWeight(i);

        if (sampleDirect) {
            if (i == 1)
                value *= ratioEmitterDirect;
            else if (i == sensorRefIndirection)
                value *= ratioSensorDirect;
        }

        if (connectable[i] && (connectable[i + 1] || isNull[i + 1]))
            weight += MISw(value);

        pdf = next;
    }
    return (Float)(1.0 / weight);
}

void PathUtils::generateCameraPath(Path & result, const Point2i &offset) {
    const Sensor *sensor = m_scene->getSensor();

    result.clear();

    /* Uniformly sample a scene time */
    m_time = sensor->getShutterOpen();
    if (sensor->needsTimeSample())
        m_time = sensor->sampleTime(m_sensorSampler->next1D());

    /* Initialize the path */
    result.initialize(m_scene, m_time, ERadiance, m_pool);


    /* Perform random walk */
    if (offset == Point2i(-1))
        result.randomWalk(m_scene, m_sensorSampler, m_sensorDepth,
            m_rrDepth, ERadiance, m_pool);
    else {
        result.randomWalkFromPixel(m_scene, m_sensorSampler,
            m_sensorDepth, offset, m_rrDepth, m_pool);
    }
    
    if (result.vertexCount() > 2) {
        result.vertex(1)->getSamplePosition(result.vertex(2), m_samplePos);
        m_radius = getRayRadius(m_time, result.vertex(1)->getPositionSamplingRecord().p,
            result.vertex(2)->getPosition()) * m_config.currentRadiusScale;
    }
    else {
        m_radius = 0.f;
    }
}

Float PathUtils::getRayRadius(Float time, int s, int t) const {
    if (t < 2 && t + s < 3)
        return 1.f; // No radius needed
    if (t > 1) {
        return getRayRadius(time, m_sensorSubpath.vertex(1)->getPosition(),
            m_sensorSubpath.vertex(2)->getPosition());
    }
    else if (t == 1) {
        return getRayRadius(time, m_sensorSubpath.vertex(1)->getPosition(),
            m_emitterSubpath.vertex(s)->getPosition());
    }
    else {
        return getRayRadius(time, m_emitterSubpath.vertex(s)->getPosition(),
            m_emitterSubpath.vertex(s-1)->getPosition());
    }
}

Float PathUtils::getRayRadius(Float time, const Point & p0, const Point & p1) const {
    // First compute ray differentials from the camera
    Ray r;
    r.setOrigin(p0);
    Float distance;
    Vector d = p1 - r.o;
    distance = d.length();
    r.setDirection(d / distance);
    r.time = time;
    const Sensor *sensor = m_scene->getSensor();
    RayDifferential rd;
    sensor->getRayDifferential(r, rd);
    Point posProj = rd.o + rd.d * distance;
    Point rX = rd.rxOrigin + rd.rxDirection * distance;
    Point rY = rd.ryOrigin + rd.ryDirection * distance;
    Float dX = (rX - posProj).lengthSquared();
    Float dY = (rY - posProj).lengthSquared();

    Float radiusSqr = std::max(dX, dY);
    return sqrt(radiusSqr);
}

void PathUtils::generateSeeds(size_t sampleCount, size_t seedCount, const CameraStorage * cameraStorage, std::vector<PathSeedEx> &seeds) {
    BDAssert(m_sensorSampler == m_emitterSampler);
    BDAssert(m_sensorSampler->getClass()->derivesFrom(MTS_CLASS(ReplayableSampler)));

    ref<Timer> timer = new Timer();
    std::vector<PathSeedEx> tempSeeds; // For both chains
    tempSeeds.reserve(sampleCount);

    seeds.clear();
    seeds.reserve(seedCount << 1); // Two chains, twice as many seeds

    SplatList splatList;
    Float luminance;
    const PathsCache &pc = cameraStorage->getCameraPaths();
    int cameraPath = 0;

    for (size_t i = 0; i<sampleCount; ++i) {
        size_t sampleIndex = m_sensorSampler->getSampleIndex();
        luminance = 0.0f;

        sampleLightAndDoUPS(splatList, cameraStorage, cameraPath);
        clearSplats();
        if (splatList.luminance > 0.f) {
            tempSeeds.push_back(PathSeedEx(sampleIndex, splatList.luminance, cameraPath));
        }
        cameraPath = (cameraPath + 1) % pc.pathsCount();
    }
    BDAssert(m_pool.unused());

    if (tempSeeds.empty()) {
        // Do random initialization
        for (size_t i = 0; i < (seedCount << 1); ++i) {
            size_t sampleIndex = m_sensorSampler->getSampleIndex();
            sampleLightAndDoUPS(splatList, cameraStorage, cameraPath);
            clearSplats();
            seeds.push_back(PathSeedEx(sampleIndex, splatList.luminance, cameraPath));
            cameraPath = (cameraPath + 1) % pc.pathsCount();
        }
        BDAssert(m_pool.unused());
        Log(EInfo, "Done -- selected initial samples", timer->getMilliseconds());
        return;
    }

    DiscreteDistribution seedPDF(tempSeeds.size());
    for (size_t i = 0; i<tempSeeds.size(); ++i)
        seedPDF.append(tempSeeds[i].luminance);
    seedPDF.normalize();

    for (size_t i = 0; i < seedCount; ++i) {
        // Visibility chain - we pick them with uniform probability
        seeds.push_back(tempSeeds.at(std::min(size_t(m_sensorSampler->next1D() * tempSeeds.size()), tempSeeds.size()-1)));
        // Contribution chain - we pick the according to luminance
        seeds.push_back(tempSeeds.at(seedPDF.sample(m_sensorSampler->next1D())));
    }

    Log(EInfo, "Done -- selected initial samples", timer->getMilliseconds());
}


MTS_IMPLEMENT_CLASS(PathUtils, false, PathSampler)
MTS_IMPLEMENT_CLASS(SeedWorkUnitEx, false, WorkUnit)
MTS_NAMESPACE_END

