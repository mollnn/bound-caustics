#include "cuda_common.cuh"

// uni ploy mul
__device__ void upm(double *polyA, double *polyB, double *result, int n, int m)
{
    for (int i = 0; i < n + m - 1; ++i)
    {
        result[i] = 0.0f;
        for (int j = 0; j <= i && j < n; ++j)
        {
            if (i - j >= m)
                continue;
            result[i] += polyA[j] * polyB[i - j];
        }
    }
}

__device__ void upms(double *polyA, double *polyB, double *result, int n, int m)
{
    for (int i = 0; i < n + m - 1; ++i)
    {
        for (int j = 0; j <= i && j < n; ++j)
        {
            if (i - j >= m)
                continue;
            result[i] -= polyA[j] * polyB[i - j];
        }
    }
}

// uni ploy add
__device__ void upa(double *polyA, double *polyB, double *result, int n)
{
    for (int i = 0; i < n; ++i)
    {
        result[i] = polyA[i] + polyB[i];
    }
}

// uni ploy sub
__device__ void ups(double *polyA, double *polyB, double *result, int n)
{
    for (int i = 0; i < n; ++i)
    {
        result[i] = polyA[i] - polyB[i];
    }
}

// bi ploy mul
__device__ void bpm(double *polyA, double *polyB, double *result, int n, int m)
{
    for (int i = 0; i < n + m - 1; ++i)
        for (int j = 0; i + j < n + m - 1; ++j)
            result[i * n + j] = 0.0f;
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < m; ++k)
            for (int j = 0; i + j < n; ++j)
                for (int l = 0; k + l < m; ++l)
                    result[(i + k) * (n + m - 1) + j + l] += polyA[i * n + j] * polyB[k * m + l];
}

// bi ploy mul
__device__ void bpma(double *polyA, double *polyB, double *result, double f, int n, int m)
{
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < m; ++k)
            for (int j = 0; i + j < n; ++j)
                for (int l = 0; k + l < m; ++l)
                    result[(i + k) * (n + m - 1) + j + l] += f * polyA[i * n + j] * polyB[k * m + l];
}