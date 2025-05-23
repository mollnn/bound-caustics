// clang-format off
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

#include <mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>
#include "../../utils/mbglints/distr.h"
#include <fstream>
#include <mutex>

std::mutex mu;
std::ofstream ofs("pathcut_log.txt");
#include "../../utils/mbglints/glintbounce.h"

MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLength("Path tracer", "Average path length", EAverage);

/*! \plugin{path}{Path tracer}
 * \order{2}
 * \parameters{
 *     \parameter{maxDepth}{\Integer}{Specifies the longest path depth
 *         in the generated output image (where \code{-1} corresponds to $\infty$).
 *         A value of \code{1} will only render directly visible light sources.
 *         \code{2} will lead to single-bounce (direct-only) illumination,
 *         and so on. \default{\code{-1}}
 *     }
 *     \parameter{rrDepth}{\Integer}{Specifies the minimum path depth, after
 *        which the implementation will start to use the ``russian roulette''
 *        path termination criterion. \default{\code{5}}
 *     }
 *     \parameter{strictNormals}{\Boolean}{Be strict about potential
 *        inconsistencies involving shading normals? See the description below
 *        for details.\default{no, i.e. \code{false}}
 *     }
 *     \parameter{hideEmitters}{\Boolean}{Hide directly visible emitters?
 *        See page~\pageref{sec:hideemitters} for details.
 *        \default{no, i.e. \code{false}}
 *     }
 * }
 *
 * This integrator implements a basic path tracer and is a \emph{good default choice}
 * when there is no strong reason to prefer another method.
 *
 * To use the path tracer appropriately, it is instructive to know roughly how
 * it works: its main operation is to trace many light paths using \emph{random walks}
 * starting from the sensor. A single random walk is shown below, which entails
 * casting a ray associated with a pixel in the output image and searching for
 * the first visible intersection. A new direction is then chosen at the intersection,
 * and the ray-casting step repeats over and over again (until one of several
 * stopping criteria applies).
 * \begin{center}
 * \includegraphics[width=.7\textwidth]{images/integrator_path_figure.pdf}
 * \end{center}
 * At every intersection, the path tracer tries to create a connection to
 * the light source in an attempt to find a \emph{complete} path along which
 * light can flow from the emitter to the sensor. This of course only works
 * when there is no occluding object between the intersection and the emitter.
 *
 * This directly translates into a category of scenes where
 * a path tracer can be expected to produce reasonable results: this is the case
 * when the emitters are easily ``accessible'' by the contents of the scene. For instance,
 * an interior scene that is lit by an area light will be considerably harder
 * to render when this area light is inside a glass enclosure (which
 * effectively counts as an occluder).
 *
 * Like the \pluginref{direct} plugin, the path tracer internally relies on multiple importance
 * sampling to combine BSDF and emitter samples. The main difference in comparison
 * to the former plugin is that it considers light paths of arbitrary length to compute
 * both direct and indirect illumination.
 *
 * For good results, combine the path tracer with one of the
 * low-discrepancy sample generators (i.e. \pluginref{ldsampler},
 * \pluginref{halton}, or \pluginref{sobol}).
 *
 * \paragraph{Strict normals:}\label{sec:strictnormals}
 * Triangle meshes often rely on interpolated shading normals
 * to suppress the inherently faceted appearance of the underlying geometry. These
 * ``fake'' normals are not without problems, however. They can lead to paradoxical
 * situations where a light ray impinges on an object from a direction that is classified as ``outside''
 * according to the shading normal, and ``inside'' according to the true geometric normal.
 *
 * The \code{strictNormals}
 * parameter specifies the intended behavior when such cases arise. The default (\code{false}, i.e. ``carry on'')
 * gives precedence to information given by the shading normal and considers such light paths to be valid.
 * This can theoretically cause light ``leaks'' through boundaries, but it is not much of a problem in practice.
 *
 * When set to \code{true}, the path tracer detects inconsistencies and ignores these paths. When objects
 * are poorly tesselated, this latter option may cause them to lose a significant amount of the incident
 * radiation (or, in other words, they will look dark).
 *
 * The bidirectional integrators in Mitsuba (\pluginref{bdpt}, \pluginref{pssmlt}, \pluginref{mlt} ...)
 * implicitly have \code{strictNormals} set to \code{true}. Hence, another use of this parameter
 * is to match renderings created by these methods.
 *
 * \remarks{
 *    \item This integrator does not handle participating media
 *    \item This integrator has poor convergence properties when rendering
 *    caustics and similar effects. In this case, \pluginref{bdpt} or
 *    one of the photon mappers may be preferable.
 * }
 */


class PathCutPathTracer : public MonteCarloIntegrator {
public:
    PathCutPathTracer(const Properties &props)
        : MonteCarloIntegrator(props) { 
        m_useResultant = props.getBoolean("useResultant", false);
        m_methodMask = props.getInteger("methodMask", 0);
        m_pathcutBounce = props.getInteger("pathcutBounce", 1);
        m_cutoffMatrix = props.getInteger("cutoffMatrix", 6);
        m_cutoffResultant = props.getInteger("cutoffResultant", 12);
        m_pathcutThres = props.getFloat("pathcutThres", 0.000001);
        m_distrPath = props.getString("distrPath", "");
        g_distr_max = props.getFloat("distr_max", 1e+2);
        g_distr_min = props.getFloat("distr_min", 1e-2);
        g_spec_var = props.getFloat("spec_var", 1e-2);
        g_force_gamma = props.getFloat("force_gamma", 0);
        g_force_sample = props.getInteger("force_sample", 0);
        g_use_max_var = props.getBoolean("use_max_var", true);
        g_p0 = BounderVec3(
            props.getFloat("lightPosition_x", 3.0f),
            props.getFloat("lightPosition_y", 1.0f),
            props.getFloat("lightPosition_z", 4.0f)
        );
        REFRACTION_2D = props.getBoolean("REFRACTION_2D", true);
        p0_p2[0][0] = g_p0;
        p0_p2[1][0] = g_p0;
        g_bounce = m_pathcutBounce;

        AR = props.getFloat("AR", 1e1);
        Am = props.getFloat("Am", 1e-3); // minimum irradiance
        AM = props.getFloat("AM", 1e5);  // maximum irradiance
        INF_AREA_TOL = props.getFloat("INF_AREA_TOL", 0.01);
        u1TOLERATE = props.getFloat("u1TOLERATE", 1);
        U1T = props.getFloat("U1T", 0.001);

        res = props.getInteger("res", 64);
        SHADING_NORMAL = props.getBoolean("SHADING_NORMAL", true);
        SHADING_NORMAL2 = props.getBoolean("SHADING_NORMAL2", true);

        CHAIN_TYPE = props.getInteger("CHAIN_TYPE", 1);
        CHAIN_LENGTH = (CHAIN_TYPE < 10) ? 1 : 2;

        dump_bound = props.getInteger("dump_bound", 0);

        if (m_distrPath != "") {
            m_distr.load(m_distrPath, m_pathcutBounce);
        }
        Log(EInfo, "m_useResultant %d", m_useResultant);
        Log(EInfo, "m_methodMask %d", m_methodMask);
        Log(EInfo, "m_pathcutBounce %d", m_pathcutBounce);
        Log(EInfo, "m_cutoffMatrix %d", m_cutoffMatrix);
        Log(EInfo, "m_cutoffResultant %d", m_cutoffResultant);
        Log(EInfo, "m_pathcutThres %fs", m_pathcutThres);
        Log(EInfo, "m_distrPath %s", m_distrPath.c_str());
        Log(EInfo, "AR %f", AR);
        Log(EInfo, "Am %f", Am);
        Log(EInfo, "AM %f", AM);
        Log(EInfo, "INF_AREA_TOL %f", INF_AREA_TOL);
        Log(EInfo, "u1TOLERATE %f", u1TOLERATE);
        Log(EInfo, "U1T %f", U1T);
        Log(EInfo, "CHAIN_TYPE %d", CHAIN_TYPE);
        Log(EInfo, "REFRACTION_2D %d", REFRACTION_2D);
        Log(EInfo, "dump_bound %d", dump_bound);
        Log(EInfo, "m_lightPos_x %f", g_p0[0]);
        Log(EInfo, "m_lightPos_y %f", g_p0[1]);
        Log(EInfo, "m_lightPos_z %f", g_p0[2]);

        #ifdef RESULTANT_SIMD
            if (m_useResultant)
                Log(EInfo, "Resultant with SIMD acceleration");
        #else 
            if (m_useResultant)
                Log(EInfo, "Resultant without SIMD acceleration");
        #endif
    }

    /// Unserialize from a binary data stream
    PathCutPathTracer(Stream *stream, InstanceManager *manager)
        : MonteCarloIntegrator(stream, manager) {  }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        bool scattered = false;

        /* Perform the first ray intersection (or ignore if the
           intersection has already been provided). */
        rRec.rayIntersect(ray);
        ray.mint = Epsilon;

        Spectrum throughput(1.0f);
        Float eta = 1.0f;

        while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
            if (!its.isValid()) {
                /* If no intersection could be found, potentially return
                   radiance from a environment luminaire if it exists */
                if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
                    && (!m_hideEmitters || scattered))
                    Li += throughput * scene->evalEnvironment(ray);
                break;
            }

            const BSDF *bsdf = its.getBSDF(ray);

            /* Possibly include emitted radiance if requested */
            if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
                && (!m_hideEmitters || scattered))
                Li += throughput * its.Le(-ray.d);

            /* Include radiance from a subsurface scattering model if requested */
            if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
                Li += throughput * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);

            if ((rRec.depth >= m_maxDepth && m_maxDepth > 0)
                || (m_strictNormals && dot(ray.d, its.geoFrame.n)
                    * Frame::cosTheta(its.wi) >= 0)) {

                /* Only continue if:
                   1. The current path length is below the specifed maximum
                   2. If 'strictNormals'=true, when the geometric and shading
                      normals classify the incident direction to the same side */
                break;
            }

            /* ==================================================================== */
            /*                     Direct illumination sampling                     */
            /* ==================================================================== */

            /* Estimate the direct illumination if this is requested */
            DirectSamplingRecord dRec(its);

            if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
                (bsdf->getType() & BSDF::ESmooth)) {
                Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
                if (!value.isZero()) {
                    const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

                    /* Allocate a record for querying the BSDF */
                    BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);

                    /* Evaluate BSDF * cos(theta) */
                    const Spectrum bsdfVal = bsdf->eval(bRec);

                    /* Prevent light leaks due to the use of shading normals */
                    if (!bsdfVal.isZero() && (!m_strictNormals
                            || dot(its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {

                        /* Calculate prob. of having generated that direction
                           using BSDF sampling */
                        Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                            ? bsdf->pdf(bRec) : 0;

                        /* Weight using the power heuristic */
                        Float weight = miWeight(dRec.pdf, bsdfPdf);
                        Li += throughput * value * bsdfVal * weight;
                    }
                }
            }

            /* ==================================================================== */
            /*                     Specular path sampling                           */
            /* ==================================================================== */
            if (bsdf->getType() & BSDF::ESmooth) {

                // todo: move to class member
                // todo: bsdf
                // todo: spectrum
                if (bgr == nullptr) {
                    bgr = new BounceGlintRenderer((Scene*)scene, false);
                    static std::mutex data_mutex;
                    data_mutex.lock();
                    if (m_shapes.empty()) {
                        BounceGlintRenderer::init_shapes(scene, m_shapes);
                    }
                    data_mutex.unlock();
                }
                bgr->use_resultant = m_useResultant;
                bgr->methodMask = m_methodMask;
                // todo: cutoff as param
                bgr->setEmitter((Scene*)scene, true);
                bgr->config_cutoff_matrix = m_cutoffMatrix;
                bgr->config_cutoff_resultant = m_cutoffResultant;
                bgr->setCameraFake(its.p.x, its.p.y, its.p.z, 0, 1, 0);
                SpecularDistribution* distr_ptr = nullptr;
                if (m_distrPath != "") distr_ptr = &m_distr;
                auto ans = bgr->renderOneImage("", m_pathcutBounce, 1, m_pathcutThres, distr_ptr, rRec.sampler, false);
                Spectrum value(0.0);
                for (auto [d, v]: ans) {
                    BSDFSamplingRecord brec_tmp(its, its.toLocal(Vector3f(d[0], d[1], d[2])), ERadiance);
                    float vv[3] = { v[0], v[1], v[2] };
                    value += bsdf->eval(brec_tmp) * Spectrum(vv);
                }
                // value[0] += 0.3 * ans.size();
                Li += throughput * value;
            }

            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */

            /* Sample BSDF * cos(theta) */
            Float bsdfPdf;
            BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
            Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
            if (bsdfWeight.isZero())
                break;

            scattered |= bRec.sampledType != BSDF::ENull;

            /* Prevent light leaks due to the use of shading normals */
            const Vector wo = its.toWorld(bRec.wo);
            Float woDotGeoN = dot(its.geoFrame.n, wo);
            if (m_strictNormals && woDotGeoN * Frame::cosTheta(bRec.wo) <= 0)
                break;

            bool hitEmitter = false;
            Spectrum value;

            /* Trace a ray in this direction */
            ray = Ray(its.p, wo, ray.time);
            if (scene->rayIntersect(ray, its)) {
                /* Intersected something - check if it was a luminaire */
                if (its.isEmitter()) {
                    value = its.Le(-ray.d);
                    dRec.setQuery(ray, its);
                    hitEmitter = true;
                }
            } else {
                /* Intersected nothing -- perhaps there is an environment map? */
                const Emitter *env = scene->getEnvironmentEmitter();

                if (env) {
                    if (m_hideEmitters && !scattered)
                        break;

                    value = env->evalEnvironment(ray);
                    if (!env->fillDirectSamplingRecord(dRec, ray))
                        break;
                    hitEmitter = true;
                } else {
                    break;
                }
            }

            /* Keep track of the throughput and relative
               refractive index along the path */
            throughput *= bsdfWeight;
            eta *= bRec.eta;

            /* If a luminaire was hit, estimate the local illumination and
               weight using the power heuristic */
            if (hitEmitter &&
                (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
                /* Compute the prob. of generating that direction using the
                   implemented direct illumination sampling technique */
                const Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ?
                    scene->pdfEmitterDirect(dRec) : 0;
                Li += throughput * value * miWeight(bsdfPdf, lumPdf);
            }

            /* ==================================================================== */
            /*                         Indirect illumination                        */
            /* ==================================================================== */

            /* Set the recursive query type. Stop if no surface was hit by the
               BSDF sample or if indirect illumination was not requested */
            if (!its.isValid() || !(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                break;
            rRec.type = RadianceQueryRecord::ERadianceNoEmission;

            if (rRec.depth++ >= m_rrDepth) {
                /* Russian roulette: try to keep path weights equal to one,
                   while accounting for the solid angle compression at refractive
                   index boundaries. Stop with at least some probability to avoid
                   getting stuck (e.g. due to total internal reflection) */

                Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
                if (rRec.nextSample1D() >= q)
                    break;
                throughput /= q;
            }
        }

        /* Store statistics */
        avgPathLength.incrementBase();
        avgPathLength += rRec.depth;

        return Li;
    }

    inline Float miWeight(Float pdfA, Float pdfB) const {
        pdfA *= pdfA;
        pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        MonteCarloIntegrator::serialize(stream, manager);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "PathCutPathTracer[" << endl
            << "  maxDepth = " << m_maxDepth << "," << endl
            << "  rrDepth = " << m_rrDepth << "," << endl
            << "  strictNormals = " << m_strictNormals << endl
            << "]";
        return oss.str();
    }

    bool m_useResultant = false; 
    int m_methodMask = 0;
    int m_cutoffMatrix = 6;
    int m_cutoffResultant = 12;
    int m_pathcutBounce = 1;
    float m_pathcutThres = 0.000001;
    thread_local static BounceGlintRenderer* bgr;
    string m_distrPath = "";
    static SpecularDistribution m_distr;
    MTS_DECLARE_CLASS()
};

thread_local BounceGlintRenderer* PathCutPathTracer::bgr = nullptr;
SpecularDistribution PathCutPathTracer::m_distr;

MTS_IMPLEMENT_CLASS_S(PathCutPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(PathCutPathTracer, "MI path cuts path tracer");
MTS_NAMESPACE_END