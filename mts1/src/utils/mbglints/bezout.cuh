#include "determinant.cuh"

__device__ void bezout_matrix(double *a, double *b, double *f)
{
    const int n = 5; // !
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = i; j < n; ++j)
        {
            upm(a + i * 7, b + (j + 1) * 7, f + (i * n + j) * numCoefficients, 6 - i, 6 - (j + 1));
            upms(a + (j + 1) * 7, b + i * 7, f + (i * n + j) * numCoefficients, 6 - (j + 1), 6 - i);
        }
    }

    for (size_t i = 1; i < n - 1; ++i)
        for (size_t j = i; j < n - 1; ++j)
            upa(f + (i * n + j) * numCoefficients, f + ((i - 1) * n + (j + 1)) * numCoefficients, f + (i * n + j) * numCoefficients, numCoefficients);

    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < i; ++j)
            for (size_t k = 0; k < numCoefficients; ++k)
                f[(i * n + j) * numCoefficients + k] = f[(j * n + i) * numCoefficients + k];
}