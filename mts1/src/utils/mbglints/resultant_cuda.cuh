#include <cuda_runtime.h>
#include <vector>
#include <utility>

#ifdef __cplusplus
extern "C" {
#endif

void solve_cuda(
    const double3 &pD,
    const double3 &pL,
    const std::vector<double3> &p10s,
    const std::vector<double3> &n10s,
    const std::vector<double3> &p11s,
    const std::vector<double3> &n11s,
    const std::vector<double3> &p12s,
    const std::vector<double3> &n12s,
    double *polys, double *Cxzs);

#ifdef __cplusplus
}
#endif

