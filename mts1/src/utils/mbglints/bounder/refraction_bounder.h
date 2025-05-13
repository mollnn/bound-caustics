#include "polynomial4.h"

class RefractionBounder : public Bounder
{
public:
    RefractionBounder(int k, const int id) : Bounder(k, id) {}

    inline void get_rational_single_reflection_deriv(BVP3_4D<1, 1, 2> &x1, const BVP3_4D<1, 1, 2> &n1, const BVP3_4D<1, 1, 2> &d0, const BVP_4D<1, 1, 1> &eta, const BVP3_4D<1, 1, 1> &p20, BVP3_4D<1, 1, 1> &p21, BVP3_4D<1, 1, 1> &p22, BVP_4D<2, 2, 7> &u2p, BVP_4D<2, 2, 7> &v2p, BVP_4D<2, 2, 6> &k2, BVP_4D<1, 1, 3> &C)
    {
        BVP_4D<1, 1, 5> beta = n1.dot(n1) * d0.dot(d0) - (n1.dot(n1) * d0.dot(d0) - n1.dot(d0) * n1.dot(d0)) * eta * eta;
        BoundingVal I_beta = beta.bound();
        I_beta.m = std::max(I_beta.m, beta_MIN * 0.5);
        I_beta.M = std::max(I_beta.M, beta_MIN * 0.99);
        double mid = I_beta.m;

        auto [a1n, a0n, delta_xin, delta_xi1n] = compute_easy_approx_for_sqrt(I_beta.m, I_beta.M);
        // check dimension
        BVP_4D<1, 1, 1> a1(a1n);
        BVP_4D<1, 1, 1> a0(a0n);
        double c[2] = {-delta_xin, 2 * delta_xin};
        BVP_4D<1, 2, 1> xi(c);
        double c2[2] = {0, delta_xi1n};
        BVP_4D<2, 1, 1> zeta(c2);
        BVP_4D<2, 2, 5> sqrt_beta = a1 * beta + a0 + xi * (beta - BVP_4D<1, 1, 1>(mid)) + zeta;
        BVP3_4D<2, 2, 6> d1 = (d0 * (n1.dot(n1)) - n1 * n1.dot(d0)) * eta - n1 * sqrt_beta;

        u2p = d1.cross(p22).dot(x1 - p20);
        v2p = (x1 - p20).cross(p21).dot(d1);
        k2 = d1.cross(p22).dot(p21);
        C = d0.dot(d0);
    }

    inline void get_rational_single_reflection_norm(BVP3_4D<1, 1, 2> &x1, const BVP3_4D<1, 1, 2> &n1, const BVP3_4D<1, 1, 2> &d0, const BVP_4D<1, 1, 1> &eta, const BVP3_4D<1, 1, 1> &p20, BVP3_4D<1, 1, 1> &p21, BVP3_4D<1, 1, 1> &p22, BVP3_4D<1, 1, 1> &p11, BVP3_4D<1, 1, 1> &p12, BVP_4D<1, 2, 7> &u2p, BVP_4D<1, 2, 7> &v2p, BVP_4D<1, 2, 6> &k2, BVP_4D<1, 1, 3> &C, BVP_4D<1, 1, 3> &s0, BVP_4D<1, 2, 7> &s1, BVP_4D<1, 1, 3> &s2)
    {
        BVP_4D<1, 1, 5> beta = n1.dot(n1) * d0.dot(d0) - (n1.dot(n1) * d0.dot(d0) - n1.dot(d0) * n1.dot(d0)) * eta * eta;
        BoundingVal I_beta = beta.bound();
        I_beta.m = std::max(I_beta.m, 0.0);
        I_beta.M = std::max(I_beta.M, beta_MIN * 0.99);
        double mid = I_beta.m;

        auto [a1n, a0n, delta_xin] = compute_line_approx_for_sqrt(I_beta.m, I_beta.M);
        BVP_4D<1, 1, 1> a1(a1n);
        BVP_4D<1, 1, 1> a0(a0n);
        double c[2] = {-delta_xin, 2 * delta_xin};
        BVP_4D<1, 2, 1> xi(c);
        BVP_4D<1, 2, 5> sqrt_beta = a1 * beta + a0 + xi;
        BVP3_4D<1, 2, 6> d1 = (d0 * (n1.dot(n1)) - n1 * n1.dot(d0)) * eta - n1 * sqrt_beta;

        u2p = d1.cross(p22).dot(x1 - p20);
        v2p = (x1 - p20).cross(p21).dot(d1);
        k2 = d1.cross(p22).dot(p21);
        C = d0.dot(d0);

        s0 = n1.dot(d0) * BVP_4D<1, 1, 1>(-1.0);
        s1 = (x1 - p20).cross(p21).dot(p22) * k2;
        s2 = p11.cross(p12).dot(d0) * (p11.cross(p12).dot(n1)) * BVP_4D<1, 1, 1>(-1.0);
    }

    inline std::tuple<bool, bool, bool, bool, bool, double, double, BounderDomain> positional_check(const BounderDomain &u1_BounderDomain, BVP_4D<1, 2, 7> &u2p, BVP_4D<1, 2, 7> &v2p, BVP_4D<1, 2, 6> &k2, BVP_4D<1, 1, 3> &C, BVP_4D<1, 1, 3> &s0, BVP_4D<1, 2, 7> &s1, BVP_4D<1, 1, 3> &s2)
    {
        BoundingVal u2_b = u2p.fbound(k2);
        BoundingVal v2_b = v2p.fbound(k2);
        BoundingVal u2p_b = u2p.bound();
        BoundingVal v2p_b = v2p.bound();
        BoundingVal u2q_b = k2.bound();

        if (u2q_b.m * u2q_b.M < 0 && u2p_b.m * u2p_b.M > 0 && v2p_b.m * v2p_b.M > 0)
        {
            BoundingVal inv_u2_b = k2.fbound(u2p);
            BoundingVal inv_v2_b = k2.fbound(v2p);

            if (inv_u2_b.m * inv_u2_b.M > 0 && inv_v2_b.m * inv_v2_b.M > 0)
            {
                u2_b.m = 1.0 / inv_u2_b.M;
                u2_b.M = 1.0 / inv_u2_b.m;
                v2_b.m = 1.0 / inv_v2_b.M;
                v2_b.M = 1.0 / inv_v2_b.m;
                u2q_b.m = 1.0;
                u2q_b.M = 1.0;
            }
            else if (inv_u2_b.M < 1 || inv_v2_b.M < 1)
            {
                u2_b.m = 1e9;
                u2_b.M = 1e9;
                v2_b.m = 1e9;
                v2_b.M = 1e9;
                u2q_b.m = 1.0;
                u2q_b.M = 1.0;
            }
        }

        if (u2q_b.m * u2q_b.M < 0)
        {
            BoundingVal inv_u2_b = k2.fbound(u2p);
            BoundingVal inv_v2_b = k2.fbound(v2p);
            if ((u2p_b.m * u2p_b.M > 0 && inv_u2_b.M < 1) || ((v2p_b.m * v2p_b.M) > 0 && inv_v2_b.M < 1))
            {
                u2_b.m = 1e9;
                u2_b.M = 1e9;
                v2_b.m = 1e9;
                v2_b.M = 1e9;
                u2q_b.m = 1.0;
                u2q_b.M = 1.0;
            }
        }

        BoundingVal s0_b = s0.bound();
        BoundingVal s1_b = s1.bound();
        BoundingVal s2_b = s2.bound();

        bool pbad_u2 = u2q_b.m * u2q_b.M > 0 && (u2_b.m < -u1TOLERATE || u2_b.M > 1 + u1TOLERATE || v2_b.M < -u1TOLERATE || v2_b.m > 1 + u1TOLERATE);
        bool bad_u = u2q_b.m * u2q_b.M > 0 && (u2_b.M < 0 || u2_b.m > 1 || v2_b.M < 0 || v2_b.m > 1) || u1_BounderDomain.um + u1_BounderDomain.vm > 1;
        bool bad_s = s0_b.M < 0 || s1_b.M < 0;
        bool pbad_s = s0_b.m < 0 || s1_b.m < 0;
        bool pbad_u1 = u1_BounderDomain.uM + u1_BounderDomain.vM > 1 + u1TOLERATE;

        return {bad_u, bad_s, pbad_u1, pbad_u2, pbad_s, u2q_b.m, u2q_b.M, BounderDomain(u2_b.m, u2_b.M, v2_b.m, v2_b.M)};
    };

    inline void irradiance_explicit(const BounderDomain &u1_BounderDomain, BVP_4D<2, 2, 7> &u2p, BVP_4D<2, 2, 7> &v2p, BVP_4D<2, 2, 6> &k2, BVP_4D<1, 1, 3> &C, BVP_4D<5, 5, 21> &Jp, BVP_4D<5, 5, 25> &Jq)
    {
        BVP_4D<2, 2, 7> u2p_du = u2p.du();
        BVP_4D<2, 2, 7> u2p_dv = u2p.dv();
        BVP_4D<2, 2, 7> v2p_du = v2p.du();
        BVP_4D<2, 2, 7> v2p_dv = v2p.dv();
        BVP_4D<2, 2, 6> k2_du = k2.du();
        BVP_4D<2, 2, 6> k2_dv = k2.dv();
        double u_scale = 1 / (u1_BounderDomain.uM - u1_BounderDomain.um);
        double v_scale = 1 / (u1_BounderDomain.vM - u1_BounderDomain.vm);

        u2p_du = u2p_du.scalarMul(u_scale);
        v2p_du = v2p_du.scalarMul(u_scale);
        k2_du = k2_du.scalarMul(u_scale);
        u2p_dv = u2p_dv.scalarMul(v_scale);
        v2p_dv = v2p_dv.scalarMul(v_scale);
        k2_dv = k2_dv.scalarMul(v_scale);

        BVP_4D<3, 3, 12> u2u1p = u2p_du * k2 - k2_du * u2p;
        BVP_4D<3, 3, 12> u2v1p = u2p_dv * k2 - k2_dv * u2p;
        BVP_4D<3, 3, 12> v2u1p = v2p_du * k2 - k2_du * v2p;
        BVP_4D<3, 3, 12> v2v1p = v2p_dv * k2 - k2_dv * v2p;

        BVP_4D<3, 3, 11> denom = k2 * k2;

        BVP_4D<5, 5, 23> fq = u2u1p * v2v1p;
        BVP_4D<5, 5, 23> fq1 = u2v1p * v2u1p;
        fq = fq - fq1;

        Jq = fq * C;
        Jp = denom * denom;
    }

    void bound_bfs(const TriangleSequence &tseq, const BounderDomain &bfs_node) override
    {
#ifdef BOUNDER_TIMER
        auto start_time = std::chrono::high_resolution_clock::now();
#endif
        double ans = 0.0;
        BounderVec3 PL = tseq.pL;
        BounderVec3 P10 = tseq.p10, P11 = tseq.p11, P12 = tseq.p12;
        BounderVec3 N10 = tseq.n10, N11 = tseq.n11, N12 = tseq.n12;
        BounderVec3 P20 = tseq.p20, P21 = tseq.p21, P22 = tseq.p22;
        double u1m = bfs_node.um, u1M = bfs_node.uM, v1m = bfs_node.vm, v1M = bfs_node.vM;

        BounderVec3 p10_ = P10 + P11 * u1m + P12 * v1m;
        BounderVec3 p11_ = P11 * (u1M - u1m);
        BounderVec3 p12_ = P12 * (v1M - v1m);
        BounderVec3 n10_ = N10 + N11 * u1m + N12 * v1m;
        BounderVec3 n11_ = N11 * (u1M - u1m);
        BounderVec3 n12_ = N12 * (v1M - v1m);

        BVP3_4D<1, 1, 1> pL{PL};
        BVP3_4D<1, 1, 1> p10{p10_};
        BVP3_4D<1, 1, 1> p11{p11_};
        BVP3_4D<1, 1, 1> p12{p12_};
        BVP3_4D<1, 1, 1> n10{n10_};
        BVP3_4D<1, 1, 1> n11{n11_};
        BVP3_4D<1, 1, 1> n12{n12_};
        BVP3_4D<1, 1, 1> p20{P20};
        BVP3_4D<1, 1, 1> p21{P21};
        BVP3_4D<1, 1, 1> p22{P22};

        BVP3_4D<1, 1, 2> x1 = p10 + p11 * u_4d + p12 * v_4d;
        BVP3_4D<1, 1, 2> n1 = n10 + n11 * u_4d + n12 * v_4d;
        BVP3_4D<1, 1, 2> d0 = x1 - pL;
        BVP_4D<1, 1, 1> eta(eta_VAL);

#ifdef BOUNDER_TIMER
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time) * 1e-9;
        construct_time[thread_k] += duration.count();
#endif

        // ##################### Positional stage #####################

#ifdef BOUNDER_TIMER
        start_time = std::chrono::high_resolution_clock::now();
#endif

#ifdef COUNTER
        int pos_begin = global_counter;
        pos_num++;
#endif

        BVP_4D<1, 2, 7> u2p;
        BVP_4D<1, 2, 7> v2p;
        BVP_4D<1, 2, 6> k2;
        BVP_4D<1, 1, 3> C;
        BVP_4D<1, 1, 3> s0;
        BVP_4D<1, 2, 7> s1;
        BVP_4D<1, 1, 3> s2;

        get_rational_single_reflection_norm(x1, n1, d0, eta, p20, p21, p22, p11, p12, u2p, v2p, k2, C, s0, s1, s2);

        BounderVec3 pD0 = P20, pD1 = P21, pD2 = P22;
        double areaD = 0.0;

        auto [bad_u, bad_s, pbad_u1, pbad_u2, pbad_s, uDqm, uDqM, uD_BounderDomain] = positional_check(bfs_node, u2p, v2p, k2, C, s0, s1, s2);

#ifdef COUNTER
        int pos_end = global_counter;
        pos_counter += (pos_end - pos_begin);
#endif

        if ((u1M - u1m) * (v1M - v1m) < 1e-4)
        {
            pbad_s = false;
        }
        if (bad_u || bad_s)
        {
#ifdef BOUNDER_TIMER
            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time) * 1e-9;
            pos_time[thread_k] += duration.count();
#endif
            return;
        }
        if (pbad_s || pbad_u1 || pbad_u2 || uDqm * uDqM < 0)
        {
            ans = -1;
            if (u1M - u1m < U1T)
            {
                ans = 0;
                if (uDqm * uDqM < 0)
                {
                    uD_BounderDomain.um = 0;
                    uD_BounderDomain.uM = 1;
                    uD_BounderDomain.vm = 0;
                    uD_BounderDomain.vM = 1;
                }
            }
        }

        areaD = (uD_BounderDomain.uM - uD_BounderDomain.um) * (uD_BounderDomain.vM - uD_BounderDomain.vm);

#ifdef BOUNDER_TIMER
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time) * 1e-9;
        pos_time[thread_k] += duration.count();
#endif

        // ##################### Irradiance stage #####################

#ifdef BOUNDER_TIMER
        start_time = std::chrono::high_resolution_clock::now();
#endif

#ifdef COUNTER
        int irr_begin = global_counter;
#endif

        bool bad_approx = true;

        if (ans >= 0)
        {
            BVP_4D<2, 2, 7> u2p_d;
            BVP_4D<2, 2, 7> v2p_d;
            BVP_4D<2, 2, 6> k2_d;
            BVP_4D<1, 1, 3> C_d;
            BVP_4D<5, 5, 21> Jp;
            BVP_4D<5, 5, 25> Jq;

            get_rational_single_reflection_deriv(x1, n1, d0, eta, p20, p21, p22, u2p_d, v2p_d, k2_d, C_d);

            irradiance_explicit(bfs_node, u2p_d, v2p_d, k2_d, C_d, Jp, Jq);

            BoundingVal Idenom = Jq.bound();
            BoundingVal I = Jp.fbound(Jq);

            auto [rm, rM] = point_triangle_dist_range(PL, P10, P11, P12, bfs_node);
            double h = std::abs((P11.cross(P12).normalized()).dot(P10 - PL));
            double x_u = P11.cross(P12).norm() / pD1.cross(pD2).norm();
            double fqm = Idenom.m, fqM = Idenom.M, fm = I.m, fM = I.M;
            double absfm = std::min(std::abs(fm), std::abs(fM));
            if (fm * fM < 0)
            {
                absfm = 0.0;
            }
            double absfM = std::max(std::abs(fm), std::abs(fM));
            double valM = absfM * h / rm * x_u;
            double valm = absfm * h / rM * x_u;
            valm = std::max(Am, valm);
            valM = std::min(AM, valM);

            if (fqm * fqM < 0)
            {
                valM = AM;
            }
            bad_approx = valM / valm > AR;
            if (bad_approx && areaD > INF_AREA_TOL && queue.size() < MAX_SUBDIV && u1M - u1m >= U1T)
            {
                ans = -1;
            }
            else
            {
                ans = valM;
            }
        }

#ifdef COUNTER
        int irr_end = global_counter;
        irr_counter += (irr_end - irr_begin);
#endif

        if (ans < 0)
        {
            double mid_u1 = (u1m + u1M) * 0.5, mid_v1 = (v1m + v1M) * 0.5;
            queue.push_back(BounderDomain(u1m, mid_u1, v1m, mid_v1));
            queue.push_back(BounderDomain(mid_u1, u1M, v1m, mid_v1));
            queue.push_back(BounderDomain(u1m, mid_u1, mid_v1, v1M));
            queue.push_back(BounderDomain(mid_u1, u1M, mid_v1, v1M));
#ifdef BOUNDER_TIMER
            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time) * 1e-9;
            irr_time[thread_k] += duration.count();
#endif
        }
        else
        {
#ifdef BOUNDER_TIMER
            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time) * 1e-9;
            irr_time[thread_k] += duration.count();
#endif
            splat(uD_BounderDomain, ans);
        }
    }
};