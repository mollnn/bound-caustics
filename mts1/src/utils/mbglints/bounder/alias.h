#pragma once
#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <vector>
#include <complex>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <tuple>
#include <unordered_set>
#include <omp.h>
#include "../localEigen/Eigen"
#define BOUNDER_TIMER
#undef COUNTER                             // pos : irr = 6559164510 : 35487129870 = 1 : 5.4
#undef DEBUG
#pragma comment(linker, "/STACK:8589934592") // avoid stack overflow

const int RES = 1024;
const int N_MAT = 28;
const int N_MAT_XY = 6;
const int MAX_THREAD = 32;
const int JACOBIAN_COMPRESS = 1; // 0-accurate 1-compressed   accurate is almost for one bounce only
const double eta_VAL = 0.6666666666666666;
// # NOT IMPORTANT PARAMETERS
const double beta_STRONG_THRES = 1e99;
const double beta_MIN = 1e-9;
const double SPLAT_SUBDIV_THRES = 1e-3;

// PY VAR
int CHAIN_TYPE = 1; // 1 for reflection and 2 for refraction
bool REFRACTION_2D = true;
int CHAIN_LENGTH = (CHAIN_TYPE < 10) ? 1 : 2;
int res = 32;
bool SHADING_NORMAL = true;  // the first bounce (close to light)
bool SHADING_NORMAL2 = true; // the second bounce (close to receiver)

// MESH CORE HYPERPARAMETERS
double AR = 4;    // approx ratio
double Am = 1e-3; // minimum irradiance
double AM = 1e5;  // maximum irradiance
double INF_AREA_TOL = 0.0001;
double u1TOLERATE = 1;
double U1T = 0.0001;

double g_distr_min = 1e-2;
double g_distr_max = 1e+2;
double g_spec_var = 1e-2;
double g_force_gamma = 0;
int g_force_sample = 0;
bool g_use_max_var = true;

int dump_bound = 0;

// SOME PARAMETERS THAT MAY SPEED UP THE ALGORITHM
int MAX_SUBDIV = 9999999; // not strictly enforced when the positional denom contains zero

// PERFORMANCE
size_t global_counter = 0;
size_t pos_counter = 0;
size_t irr_counter = 0;
// std::chrono::duration<double, std::nano> duration;
//  std::unordered_set<int> plus_matrix_size;
//  std::unordered_set<int> mul_matrix_size;
//  std::unordered_set<int> scal_matrix_size;
//  std::unordered_set<int> bound_matrix_size;
//  std::unordered_set<int> fbound_matrix_size;
//  std::unordered_set<int> du_matrix_size;
//  std::unordered_set<int> dv_matrix_size;
//  std::unordered_set<int> T_matrix_size;
double pos_time[MAX_THREAD] = {};
double construct_time[MAX_THREAD] = {};
double irr_time[MAX_THREAD] = {};
double total_time[MAX_THREAD] = {};
double splat_time[MAX_THREAD] = {};

// TYPE DEFINE
typedef double PolyMatrix[N_MAT][N_MAT];
typedef double PolyMatrix_3D[N_MAT_XY][N_MAT][N_MAT];
typedef double PolyMatrix_4D[N_MAT_XY][N_MAT_XY][N_MAT][N_MAT];
typedef double BounderMatrix[RES][RES];
typedef Eigen::Vector3d BounderVec3;

struct BounderTriangle
{
    BounderVec3 vertices[3];
    BounderVec3 normals[3];
};

struct BoundingVal
{
    BoundingVal(double m_, double M_) : m(m_), M(M_)
    {
    }

    double m;
    double M;
};

class BounderDomain
{
public:
    BounderDomain(double um_, double uM_, double vm_, double vM_) : um(um_), uM(uM_), vm(vm_), vM(vM_)
    {
    }
    BounderDomain() : um(0.0), uM(0.0), vm(0.0), vM(0.0) {}
    double um;
    double uM;
    double vm;
    double vM;
};

class TriangleSequence
{
public:
    TriangleSequence(BounderVec3 &pL_, BounderVec3 &p10_, BounderVec3 &p11_, BounderVec3 &p12_,
                     BounderVec3 &n10_, BounderVec3 &n11_, BounderVec3 &n12_,
                     BounderVec3 &p20_, BounderVec3 &p21_, BounderVec3 &p22_)
        : pL(pL_), p10(p10_), p11(p11_), p12(p12_),
          n10(n10_), n11(n11_), n12(n12_),
          p20(p20_), p21(p21_), p22(p22_)
    {
    }

    BounderVec3 pL;
    BounderVec3 p10, p11, p12;
    BounderVec3 n10, n11, n12;
    BounderVec3 p20, p21, p22;
};

// GLOBAL VAR
std::vector<std::pair<double, int>> mesh_mem[MAX_THREAD][RES][RES];
double factorial[N_MAT];
PolyMatrix binomial;
PolyMatrix *c2b_cache[N_MAT] = {nullptr};
std::vector<BounderTriangle> mesh;
BounderVec3 p0_p2[2][4] = {{{3.0, 1.0, 4.0}, {-3.0, 0.0, -3.0}, {6.0, 0.000001, 0.0}, {0.0, 0.000002, 6.0}},
                           {{1, 2, 1}, {-3.0, 0.0, -3.0}, {6.0, 0.000001, 0.0}, {0.0, 0.000002, 6.0}}};
// for pool
// BounderVec3 p0_p2[2][4] = {{{1, 2, 1}, {-8.0, -2.0, -8.0}, {16.0, 0.000001, 0.0}, {0.0, 0.000002, 16.0}},
//                            {{1, 2, 1}, {-8.0, -2.0, -8.0}, {16.0, 0.000001, 0.0}, {0.0, 0.000002, 16.0}}};
BounderVec3 g_p0 = {0.0, 0.0, 0.0};

class Bounder
{
public:
    Bounder(const int k, const int id) : thread_k(k), queue(), queue_head(0), tri_id(id)
    {
    }

    virtual void bound_bfs(const TriangleSequence &tseq, const BounderDomain &bfs_node) = 0;

    int thread_k;
    int tri_id;
    std::vector<BounderDomain> queue;
    int queue_head;

    void compute_bound3d(const TriangleSequence &tseq, const BounderDomain &u1_domian)
    {
        if (u1_domian.uM - u1_domian.um <= 0 || u1_domian.vM - u1_domian.vm <= 0)
        {
            return;
        }

        queue.push_back(u1_domian);

        while (queue.size() > queue_head)
        {
            BounderDomain currentNode = queue[queue_head];
            bound_bfs(tseq, currentNode);
            queue_head += 1;
        }
    }

    inline void splat(const BounderDomain &u1_domian, const double val)
    {
#ifdef BOUNDER_TIMER
        auto start_time = std::chrono::high_resolution_clock::now();
#endif

        if (u1_domian.um > 1 || u1_domian.uM < 0 || u1_domian.vm > 1 || u1_domian.vM < 0 || val == 0.f)
        {
#ifdef BOUNDER_TIMER
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time) * 1e-9;
            splat_time[thread_k] += duration.count();
#endif
            return;
        }

        int a = std::max(0, std::min(res, static_cast<int>(std::max(0.0, u1_domian.vm) * res)));
        int b = std::max(0, std::min(res, static_cast<int>(std::min(1.0, u1_domian.vM) * res) + 1));
        int c = std::max(0, std::min(res, static_cast<int>(std::max(0.0, u1_domian.um) * res)));
        int d = std::max(0, std::min(res, static_cast<int>(std::min(1.0, u1_domian.uM) * res) + 1));

        for (int i = a; i < b; ++i)
        {
            for (int j = c; j < d; ++j)
            {
                if (mesh_mem[thread_k][i][j].empty() || (std::get<1>(mesh_mem[thread_k][i][j].back()) != tri_id))
                {
                    mesh_mem[thread_k][i][j].emplace_back(std::make_pair(val, tri_id));
                }
                else
                {
                    if (std::get<0>(mesh_mem[thread_k][i][j].back()) < val)
                    {
                        mesh_mem[thread_k][i][j].back().first = val;
                    }
                }
            }
        }

#ifdef BOUNDER_TIMER
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time) * 1e-9;
        splat_time[thread_k] += duration.count();
#endif
    }
};
