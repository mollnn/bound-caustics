#pragma once
#include "polynomial.h"

class ReflectionBounder : public Bounder
{
public:
    ReflectionBounder(const int k, const int id) : Bounder(k, id) {}

    inline void get_rational_single_reflection(const TriangleSequence &tseq, const BounderDomain &u1_domain, BVP<5> &u2p, BVP<5> &v2p, BVP<4> &k2, BVP<3> &C, BVP<3> &s0, BVP<5> &s1, BVP<3> &s2)
    {
#ifdef BOUNDER_TIMER
        auto start_time = std::chrono::high_resolution_clock::now();
#endif

        BounderVec3 PL = tseq.pL;
        BounderVec3 P10 = tseq.p10, P11 = tseq.p11, P12 = tseq.p12;
        BounderVec3 N10 = tseq.n10, N11 = tseq.n11, N12 = tseq.n12;
        BounderVec3 P20 = tseq.p20, P21 = tseq.p21, P22 = tseq.p22;

        double u1m = u1_domain.um;
        double u1M = u1_domain.uM;
        double v1m = u1_domain.vm;
        double v1M = u1_domain.vM;

        BounderVec3 p10_ = P10 + P11 * u1m + P12 * v1m;
        BounderVec3 p11_ = P11 * (u1M - u1m);
        BounderVec3 p12_ = P12 * (v1M - v1m);
        BounderVec3 n10_ = N10 + N11 * u1m + N12 * v1m;
        BounderVec3 n11_ = N11 * (u1M - u1m);
        BounderVec3 n12_ = N12 * (v1M - v1m);

        // BVP3<1> pL = {BVP<1>(PL[0]), BVP<1>(PL[1]), BVP<1>(PL[2])};
        // BVP3<1> p10 = {BVP<1>(p10_[0]), BVP<1>(p10_[1]), BVP<1>(p10_[2])};
        // BVP3<1> p11 = {BVP<1>(p11_[0]), BVP<1>(p11_[1]), BVP<1>(p11_[2])};
        // BVP3<1> p12 = {BVP<1>(p12_[0]), BVP<1>(p12_[1]), BVP<1>(p12_[2])};
        // BVP3<1> n10 = {BVP<1>(n10_[0]), BVP<1>(n10_[1]), BVP<1>(n10_[2])};
        // BVP3<1> n11 = {BVP<1>(n11_[0]), BVP<1>(n11_[1]), BVP<1>(n11_[2])};
        // BVP3<1> n12 = {BVP<1>(n12_[0]), BVP<1>(n12_[1]), BVP<1>(n12_[2])};
        // BVP3<1> p20 = {BVP<1>(P20[0]), BVP<1>(P20[1]), BVP<1>(P20[2])};
        // BVP3<1> p21 = {BVP<1>(P21[0]), BVP<1>(P21[1]), BVP<1>(P21[2])};
        // BVP3<1> p22 = {BVP<1>(P22[0]), BVP<1>(P22[1]), BVP<1>(P22[2])};

        BVP3<1> pL{PL};
        BVP3<1> p10{p10_};
        BVP3<1> p11{p11_};
        BVP3<1> p12{p12_};
        BVP3<1> n10{n10_};
        BVP3<1> n11{n11_};
        BVP3<1> n12{n12_};
        BVP3<1> p20{P20};
        BVP3<1> p21{P21};
        BVP3<1> p22{P22};

        BVP3<2> x1 = p10 + p11 * u + p12 * v;
        BVP3<2> n1 = n10 + n11 * u + n12 * v;
        BVP3<2> d0 = x1 - pL;
        BVP3<4> d1 = d0 * (n1.dot(n1)) - n1 * (n1.dot(d0)) - n1 * (n1.dot(d0));

#ifdef BOUNDER_TIMER
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time) * 1e-9;
        construct_time[thread_k] += duration.count();
#endif

        u2p = d1.cross(p22).dot(x1 - p20);
        v2p = (x1 - p20).cross(p21).dot(d1);
        k2 = d1.cross(p22).dot(p21);
        C = d0.dot(d0);

        s0 = n1.dot(d0) * BVP<1>{-1.0};
        s1 = (x1 - p20).cross(p21).dot(p22) * k2;
        s2 = p11.cross(p12).dot(d0) * (p11.cross(p12).dot(n1)) * BVP<1>{-1.0};

        // std::cout << "u2p " << u2p.size << std::endl;
        // std::cout << "v2p " << v2p.size << std::endl;
        // std::cout << "k2 " << k2.size << std::endl;
        // std::cout << "C " << C.size << std::endl;
        // std::cout << "s0 " << s0.size << std::endl;
        // std::cout << "s1 " << s1.size << std::endl;
        // std::cout << "s2 " << s2.size << std::endl;
        // std::cout << "-------------------------------------" << std::endl;
    }

    inline std::tuple<bool, bool, bool, bool, bool, double, double, BounderDomain> positional_check(const BounderDomain &u1_domain, BVP<5> &u2p, BVP<5> &v2p, BVP<4> &k2, BVP<3> &C, BVP<3> &s0, BVP<5> &s1, BVP<3> &s2)
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
        bool bad_u = u2q_b.m * u2q_b.M > 0 && (u2_b.M < 0 || u2_b.m > 1 || v2_b.M < 0 || v2_b.m > 1) || u1_domain.um + u1_domain.vm > 1;
        bool bad_s = s0_b.M < 0 || s1_b.M < 0 || s2_b.M < 0;
        bool pbad_s = s0_b.m < 0 || s1_b.m < 0 || s2_b.m < 0;
        bool pbad_u1 = u1_domain.uM + u1_domain.vM > 1 + u1TOLERATE;
        return {bad_u, bad_s, pbad_u1, pbad_u2, pbad_s, u2q_b.m, u2q_b.M, BounderDomain(u2_b.m, u2_b.M, v2_b.m, v2_b.M)};
    };

    inline void irradiance_explicit(const BounderDomain &u1_domain, BVP<5> &u2p, BVP<5> &v2p, BVP<4> &k2, BVP<3> &C, BVP<13> &Jp, BVP<17> &Jq)
    {
        BVP<5> u2p_du = u2p.du();
        BVP<5> u2p_dv = u2p.dv();
        BVP<5> v2p_du = v2p.du();
        BVP<5> v2p_dv = v2p.dv();
        BVP<4> k2_du = k2.du();
        BVP<4> k2_dv = k2.dv();
        double u_scale = 1 / (u1_domain.uM - u1_domain.um);
        double v_scale = 1 / (u1_domain.vM - u1_domain.vm);

        u2p_du = u2p_du.scalarMul(u_scale);
        v2p_du = v2p_du.scalarMul(u_scale);
        k2_du = k2_du.scalarMul(u_scale);
        u2p_dv = u2p_dv.scalarMul(v_scale);
        v2p_dv = v2p_dv.scalarMul(v_scale);
        k2_dv = k2_dv.scalarMul(v_scale);

        BVP<8> u2u1p = u2p_du * k2 - k2_du * u2p;
        BVP<8> u2v1p = u2p_dv * k2 - k2_dv * u2p;
        BVP<8> v2u1p = v2p_du * k2 - k2_du * v2p;
        BVP<8> v2v1p = v2p_dv * k2 - k2_dv * v2p;

        BVP<7> denom = k2 * k2;

        BVP<15> fq = u2u1p * v2v1p;
        BVP<15> fq1 = u2v1p * v2u1p;
        fq = fq - fq1;

        Jq = fq * C;
        Jp = denom * denom;

        // std::cout << "u2p_du " << u2p_du.size << std::endl;
        // std::cout << "u2p_dv " << u2p_dv.size << std::endl;
        // std::cout << "v2p_du " << v2p_du.size << std::endl;
        // std::cout << "v2p_dv " << v2p_dv.size << std::endl;
        // std::cout << "k2_du " << k2_du.size << std::endl;
        // std::cout << "k2_dv " << k2_dv.size << std::endl;
        // std::cout << "u2u1p " << u2u1p.size << std::endl;
        // std::cout << "u2v1p " << u2v1p.size << std::endl;
        // std::cout << "v2u1p " << v2u1p.size << std::endl;
        // std::cout << "v2v1p " << v2v1p.size << std::endl;
        // std::cout << "fq " << fq.size << std::endl;
        // std::cout << "fq1 " << fq1.size << std::endl;
        // std::cout << "fp " << fp.size << std::endl;
        // std::cout << "Jp " << Jp.size << std::endl;
        // std::cout << "Jq " << Jq.size << std::endl;
        // std::cout << "-------------------------------------" << std::endl;
    }

    void bound_bfs(const TriangleSequence &tseq, const BounderDomain &bfs_node) override
    {
#ifdef BOUNDER_TIMER
        auto start_time = std::chrono::high_resolution_clock::now();
#endif

#ifdef COUNTER
        int pos_begin = global_counter;
#endif

        double ans = 0.0;
        BounderVec3 PL = tseq.pL;
        BounderVec3 P10 = tseq.p10, P11 = tseq.p11, P12 = tseq.p12;
        BounderVec3 N10 = tseq.n10, N11 = tseq.n11, N12 = tseq.n12;
        BounderVec3 P20 = tseq.p20, P21 = tseq.p21, P22 = tseq.p22;
        double u1m = bfs_node.um, u1M = bfs_node.uM, v1m = bfs_node.vm, v1M = bfs_node.vM;

        // ##################### Positional stage #####################

        BVP<5> u2p;
        BVP<5> v2p;
        BVP<4> k2;
        BVP<3> C;
        BVP<3> s0;
        BVP<5> s1;
        BVP<3> s2;

        get_rational_single_reflection(tseq, bfs_node, u2p, v2p, k2, C, s0, s1, s2);

        BounderVec3 pD0 = P20, pD1 = P21, pD2 = P22;
        double areaD = 0.0;
        BVP<13> Jp;
        BVP<17> Jq;

        auto [bad_u, bad_s, pbad_u1, pbad_u2, pbad_s, uDqm, uDqM, uD_domain] = positional_check(bfs_node, u2p, v2p, k2, C, s0, s1, s2);

#ifdef COUNTER
        int pos_end = global_counter;
        pos_counter += (pos_end - pos_begin);
#endif

        // pos_counter1 += global_counter;
        // pos_counter2++;
        // global_counter = 0;

        if ((u1M - u1m) * (v1M - v1m) < 1e-4)
        {
            pbad_s = false;
        }
        if (bad_u || bad_s)
        {
#ifdef BOUNDER_TIMER
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time) * 1e-9;
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
                    uD_domain.um = 0;
                    uD_domain.uM = 1;
                    uD_domain.vm = 0;
                    uD_domain.vM = 1;
                }
            }
        }

        areaD = (uD_domain.uM - uD_domain.um) * (uD_domain.vM - uD_domain.vm);

#ifdef BOUNDER_TIMER
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time) * 1e-9;
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
            irradiance_explicit(bfs_node, u2p, v2p, k2, C, Jp, Jq);
            BoundingVal Idenom = Jq.bound();
            BoundingVal I = Jp.fbound(Jq);

            // irrad_counter1 += global_counter;
            // irrad_counter2++;
            // global_counter = 0;

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
            splat(uD_domain, ans);
        }
    }
};