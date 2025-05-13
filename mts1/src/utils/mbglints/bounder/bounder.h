#pragma once
#include "refraction_bounder_2d.h"
#include "reflection_bounder.h"
#include "refraction_bounder.h"

void run_bounder(const std::string &name)
{
    initializeFactorial();
    initializeBinomial();
    initializeCache();
    initializeMesh(name, {0.0, 0.0, 0.0});
    BounderDomain u1_domain(0.0, 1.0, 0.0, 1.0);

    std::cout << "begin bounder" << std::endl;

#pragma omp parallel for schedule(dynamic, res / MAX_THREAD)
    for (int k = 0; k < mesh.size(); k++)
    {
#ifdef BOUNDER_TIMER
        auto start_time = std::chrono::high_resolution_clock::now();
#endif
        BounderTriangle tr = mesh[k];
        BounderVec3 p10 = tr.vertices[0], p11 = tr.vertices[1] - tr.vertices[0], p12 = tr.vertices[2] - tr.vertices[0];
        BounderVec3 n10 = tr.normals[0], n11 = tr.normals[1] - tr.normals[0], n12 = tr.normals[2] - tr.normals[0];
        TriangleSequence tseq(p0_p2[CHAIN_TYPE - 1][0], p10, p11, p12, n10, n11, n12, p0_p2[CHAIN_TYPE - 1][1], p0_p2[CHAIN_TYPE - 1][2], p0_p2[CHAIN_TYPE - 1][3]);

        int id = omp_get_thread_num() % MAX_THREAD;

        if (CHAIN_TYPE == 1)
        {
            ReflectionBounder reflectionBounder(id, k);
            reflectionBounder.compute_bound3d(tseq, u1_domain);
        }
        else if (CHAIN_TYPE == 2)
        {
            if (REFRACTION_2D)
            {
                RefractionBounder2d refractionBounder2d(id, k);
                refractionBounder2d.compute_bound3d(tseq, u1_domain);
            }
            else
            {
                RefractionBounder refractionBounder(id, k);
                refractionBounder.compute_bound3d(tseq, u1_domain);
            }
        }

#ifdef BOUNDER_TIMER
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time) * 1e-9;
        auto duration_since_epoch = std::chrono::duration_cast<std::chrono::seconds>(end_time.time_since_epoch());
        total_time[id] += duration.count();
#endif
    }

    // #ifdef BOUNDER_TIMER
    //     auto start_time = std::chrono::high_resolution_clock::now();
    // #endif
    //     one_thread(0, u1_domain);
    // #ifdef BOUNDER_TIMER
    //     auto end_time = std::chrono::high_resolution_clock::now();
    //     auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time) * 1e-9;
    //     total_time[0] += duration.count();
    // #endif

    if (dump_bound == 1)
    {
        std::ofstream bounder_ofs1("../../results/sample_map.txt", std::ios::out | std::ios::trunc);
        for (int i = 0; i < res; ++i)
        {
            for (int j = 0; j < res; ++j)
            {
                int len = 0;
                for (int t = 0; t < MAX_THREAD; ++t)
                {
                    len += mesh_mem[t][i][j].size();
                }
                bounder_ofs1 << len;

                for (int k = 0; k < MAX_THREAD; ++k)
                {
                    for (int n = 0; n < mesh_mem[k][i][j].size(); n++)
                    {
                        bounder_ofs1 << " " << std::setprecision(14) << std::fixed << std::get<0>(mesh_mem[k][i][j][n]);
                        bounder_ofs1 << " " << std::get<1>(mesh_mem[k][i][j][n]);
                    }
                }
                bounder_ofs1 << std::endl;
            }
        }
    }
    else if (dump_bound == 2)
    {
        std::ofstream bounder_ofs2("../../results/matrix_map.txt", std::ios::out | std::ios::trunc);
        for (int i = 0; i < res; ++i)
        {
            for (int j = 0; j < res; ++j)
            {
                double sum = 0.0;
                for (int k = 0; k < MAX_THREAD; ++k)
                {
                    for (int n = 0; n < mesh_mem[k][i][j].size(); n++)
                    {
                        sum += std::min(std::get<0>(mesh_mem[k][i][j][n]), 1.0);
                    }
                }
                bounder_ofs2 << std::setw(16) << std::setprecision(6) << std::fixed << sum;
            }
            bounder_ofs2 << std::endl;
        }
    }
}
