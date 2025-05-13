// author: hao zhang, zhimin fan
#include "resultant_cuda.cuh"
#include <iostream>
#include <chrono>
#include "bezout.cuh"

__global__ void solve_one(double3 pD, double3 pL,
                      double3 *p10, double3 *n10,
                      double3 *p11, double3 *n11,
                      double3 *p12, double3 *n12,
                      double *polys, double *Cxzs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double n1x[4];
    n1x[0] = n10[idx].x;
    n1x[2] = n11[idx].x;
    n1x[1] = n12[idx].x;
    n1x[3] = 0.0f;
    double n1y[4];
    n1y[0] = n10[idx].y;
    n1y[2] = n11[idx].y;
    n1y[1] = n12[idx].y;
    n1y[3] = 0.0f;
    double n1z[4];
    n1z[0] = n10[idx].z;
    n1z[2] = n11[idx].z;
    n1z[1] = n12[idx].z;
    n1z[3] = 0.0f;

    double DDx[4];
    DDx[0] = p10[idx].x - pD.x;
    DDx[2] = p11[idx].x;
    DDx[1] = p12[idx].x;
    DDx[3] = 0.0f;
    double DDy[4];
    DDy[0] = p10[idx].y - pD.y;
    DDy[2] = p11[idx].y;
    DDy[1] = p12[idx].y;
    DDy[3] = 0.0f;
    double DDz[4];
    DDz[0] = p10[idx].z - pD.z;
    DDz[2] = p11[idx].z;
    DDz[1] = p12[idx].z;
    DDz[3] = 0.0f;
    double DDs[9] = {0};
    bpm(DDx, DDx, DDs, 2, 2);
    bpma(DDy, DDy, DDs, 1, 2, 2);
    bpma(DDz, DDz, DDs, 1, 2, 2);

    double D1x[4];
    D1x[0] = p10[idx].x - pL.x;
    D1x[2] = p11[idx].x;
    D1x[1] = p12[idx].x;
    D1x[3] = 0.0f;
    double D1y[4];
    D1y[0] = p10[idx].y - pL.y;
    D1y[2] = p11[idx].y;
    D1y[1] = p12[idx].y;
    D1y[3] = 0.0f;
    double D1z[4];
    D1z[0] = p10[idx].z - pL.z;
    D1z[2] = p11[idx].z;
    D1z[1] = p12[idx].z;
    D1z[3] = 0.0f;
    double D1s[9] = {0};
    bpm(D1x, D1x, D1s, 2, 2);
    bpma(D1y, D1y, D1s, 1, 2, 2);
    bpma(D1z, D1z, D1s, 1, 2, 2);

    double Azy[9] = {0};
    double Azy2[25] = {0};
    bpm(DDz, n1y, Azy, 2, 2);
    bpma(DDy, n1z, Azy, -1, 2, 2);
    bpm(Azy, Azy, Azy2, 3, 3);
    double Bzy[9] = {0};
    double Bzy2[25] = {0};
    bpm(D1z, n1y, Bzy, 2, 2);
    bpma(D1y, n1z, Bzy, -1, 2, 2);
    bpm(Bzy, Bzy, Bzy2, 3, 3);
    double Czy[49] = {0};
    bpm(Azy2, D1s, Czy, 5, 3);
    bpma(Bzy2, DDs, Czy, -1, 5, 3);

    double Axz[9] = {0};
    double Axz2[25] = {0};
    bpm(DDx, n1z, Axz, 2, 2);
    bpma(DDz, n1x, Axz, -1, 2, 2);
    bpm(Axz, Axz, Axz2, 3, 3);
    double Bxz[9] = {0};
    double Bxz2[25] = {0};
    bpm(D1x, n1z, Bxz, 2, 2);
    bpma(D1z, n1x, Bxz, -1, 2, 2);
    bpm(Bxz, Bxz, Bxz2, 3, 3);
    double Cxz[49] = {0};
    bpm(Axz2, D1s, Cxz, 5, 3);
    bpma(Bxz2, DDs, Cxz, -1, 5, 3);

    for (int i = 0; i < 6; ++i)
        for (int j = 0; i + j < 6; ++j)
            Cxzs[idx * 36 + i * 6 + j] = Cxz[i * 7 + j];

    double bezoutMat[25 * numCoefficients] = {0};
    bezout_matrix(Czy, Cxz, bezoutMat);
    double *poly = polys + idx * numCoefficients;
    matrix5x5Determinant(bezoutMat, poly);
}

void solve_cuda(
    const double3 &pD,
    const double3 &pL,
    const std::vector<double3> &p10s,
    const std::vector<double3> &n10s,
    const std::vector<double3> &p11s,
    const std::vector<double3> &n11s,
    const std::vector<double3> &p12s,
    const std::vector<double3> &n12s,
    double *polys, double *Cxzs)
{
    int N = p10s.size();
    int threadsPerBlock = 64;
    int blocksPerGrid = (N - 1) / threadsPerBlock + 1;

    double3 *d_p10, *d_n10, *d_p11, *d_n11, *d_p12, *d_n12;
    double *d_polys, *d_Cxzs;
    auto start_malloc = std::chrono::high_resolution_clock::now();
    cudaMalloc((void **)&d_p10, N * sizeof(double3));
    cudaMalloc((void **)&d_n10, N * sizeof(double3));
    cudaMalloc((void **)&d_p11, N * sizeof(double3));
    cudaMalloc((void **)&d_n11, N * sizeof(double3));
    cudaMalloc((void **)&d_p12, N * sizeof(double3));
    cudaMalloc((void **)&d_n12, N * sizeof(double3));
    cudaMalloc((void **)&d_polys, N * numCoefficients * sizeof(double));
    cudaMalloc((void **)&d_Cxzs, N * 36 * sizeof(double));
    auto end_malloc = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration_malloc_us = end_malloc - start_malloc;

    auto start_memcpy = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_p10, p10s.data(), N * sizeof(double3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n10, n10s.data(), N * sizeof(double3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p11, p11s.data(), N * sizeof(double3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n11, n11s.data(), N * sizeof(double3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p12, p12s.data(), N * sizeof(double3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n12, n12s.data(), N * sizeof(double3), cudaMemcpyHostToDevice);
    auto end_memcpy = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration_memcpy_us = end_memcpy - start_memcpy;

    auto start_kernel = std::chrono::high_resolution_clock::now();
    solve_one<<<blocksPerGrid, threadsPerBlock>>>(pD, pL, d_p10, d_n10, d_p11, d_n11, d_p12, d_n12, d_polys, d_Cxzs);

    cudaDeviceSynchronize();
    auto end_kernel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration_kernel_us = end_kernel - start_kernel;

    auto start_memcpy_ = std::chrono::high_resolution_clock::now();
    cudaMemcpy(polys, d_polys, N * numCoefficients * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Cxzs, d_Cxzs, N * 36 * sizeof(double), cudaMemcpyDeviceToHost);
    auto end_memcpy_ = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration_memcpy__us = end_memcpy_ - start_memcpy_;

    auto start_free = std::chrono::high_resolution_clock::now();
    cudaFree(d_p10); cudaFree(d_n10); cudaFree(d_p11); cudaFree(d_n11); cudaFree(d_p12); cudaFree(d_n12);
    cudaFree(d_polys); cudaFree(d_Cxzs);
    auto end_free = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration_free_us = end_free - start_free;

    // auto start_cout = std::chrono::high_resolution_clock::now();
    // std::cout << "profiling cuda Malloc totTime: " << duration_malloc_us.count() / 1e6 << " s" << std::endl;
    // std::cout << "profiling cuda Free totTime: " << duration_free_us.count() / 1e6 << " s" << std::endl;
    // std::cout << "profiling cuda Memcpy1 Time: " << duration_memcpy_us.count() / N << " us" << std::endl;
    // std::cout << "profiling cuda Memcpy2 Time: " << duration_memcpy__us.count() / N << " us" << std::endl;
    // std::cout << "profiling cuda Kernel Time: " << duration_kernel_us.count() / N << " us" << std::endl;
    // std::cout << "profiling cuda Total3 Time: " << (duration_memcpy_us + duration_memcpy__us + duration_kernel_us).count() / N << " us" << std::endl;
    // auto end_cout = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::micro> duration_cout_us = end_cout - start_cout;
    // std::cout << "profiling cuda Cout Time: " << duration_cout_us.count() / N << " us" << std::endl;
}
