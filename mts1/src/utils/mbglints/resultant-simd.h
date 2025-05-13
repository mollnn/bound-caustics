// clang-format off
#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <immintrin.h>
#include <ios>
#include <iostream>
#include <vector>
#include "opt_onebounce6x6.h"

namespace resultant_simd {

constexpr size_t N_DEGREE = 4;
constexpr size_t N_POLY = 9;
constexpr size_t N_MAT = N_DEGREE + 1;

int global_poly_cutoff = N_POLY - 1;
double global_poly_cutoff_eps = 1e-9;
int global_method_mask = 0;

constexpr size_t BVP_ADD_SIZE(size_t bvp_1_size, size_t bvp_2_size) {
  return std::max(bvp_1_size, bvp_2_size);
}

constexpr size_t BVP_MUL_SIZE(size_t bvp_1_size, size_t bvp_2_size) {
  return bvp_1_size + bvp_2_size - 1;
}

template <size_t Size> struct UnivariatePolyMatrix {
  // TODO marco define max order 

  UnivariatePolyMatrix<Size>() {
    for (int i = 0; i < Size; ++i)
      for (int j = 0; j < Size; ++j) {
        for (int k = 0; k < 8; ++k) {
          matrix[i][j][k] = _mm256_setzero_pd();
        }
      }
  }

  __m256d matrix[Size][Size][8];

  std::vector<std::vector<
      std::pair<Resultant::UnivariatePolynomial, std::pair<double, double>>>>
  determinant() const {
    printf(
        "No general implementation -- UnivariatePolyMatrix::determinant()\n");
  }
};

template <size_t Size> struct BVP {
public:
  BVP() { printf("No general implementation -- BVP<%ld>()\n", Size); }

  explicit BVP(const double d) { coeffs[0][0] = _mm256_broadcast_sd(&d); }

  explicit BVP(const double *_coeffs) {
    printf("No general implementation -- BVP<%ld>()\n", Size);
  }

  template <size_t Other_Size, size_t Res_size = BVP_ADD_SIZE(Size, Other_Size)>
  BVP<Res_size> operator+(const BVP<Other_Size> &other) const {
    printf("No general implementation -- BVP<%ld>+()\n", Size);
  }

  template <size_t Other_Size, size_t Res_size = BVP_ADD_SIZE(Size, Other_Size)>
  BVP<Res_size> operator-(const BVP<Other_Size> &other) const {
    printf("No general implementation -- BVP<%ld>-()\n", Size);
  }

  template <size_t Other_Size, size_t Res_size = BVP_MUL_SIZE(Size, Other_Size)>
  BVP<Res_size> operator*(const BVP<Other_Size> &other) const {
    printf("No general implementation -- BVP<%ld>*()\n", Size);
  }

  BVP<Size> operator*(const double) const {
    printf("No general implementation -- BVP<%ld>*(double)\n", Size);
  }

  void print() const {
    printf("No general implementation -- BVP<%ld>print()\n", Size);
  }

public:
  __m256d coeffs[Size][Size];
};

template <size_t Size> struct BVP3 {

public:
  BVP3() {}

  BVP3(const double *_x, const double *_y, const double *_z)
      : x(_x), y(_y), z(_z) {}

  BVP3(const BVP<Size> &_x, const BVP<Size> &_y, const BVP<Size> &_z)
      : x(_x), y(_y), z(_z) {}

  template <size_t Other_Size, size_t Res_Size = BVP_MUL_SIZE(Size, Other_Size)>
  BVP<Res_Size> dot(const BVP3<Other_Size> &other) const {
    return x * other.x + y * other.y + z * other.z;
  }

  template <size_t Other_Size, size_t Res_size = BVP_MUL_SIZE(Size, Other_Size)>
  BVP3<Res_size> cross(const BVP3<Other_Size> &other) const {
    return {
        y * other.z - z * other.y, //
        z * other.x - x * other.z, //
        x * other.y - y * other.x  //
    };
  }

  template <size_t Other_Size, size_t Res_Size = BVP_MUL_SIZE(Size, Other_Size)>
  BVP3<Res_Size> operator*(const BVP3<Other_Size> &other) const {
    return {x * other.x, y * other.y, z * other.z};
  }

  template <size_t Other_Size, size_t Res_Size = BVP_MUL_SIZE(Size, Other_Size)>
  BVP3<Res_Size> operator*(const BVP<Other_Size> &other) const {
    return {x * other, y * other, z * other};
  }

  template <size_t Other_Size, size_t Res_Size = BVP_ADD_SIZE(Size, Other_Size)>
  BVP3<Res_Size> operator-(const BVP3<Other_Size> &other) const {
    return {x - other.x, y - other.y, z - other.z};
  }

  BVP3<Size> operator*(const double d) const { return {x * d, y * d, z * d}; }

  void print() const {
    x.print();
    y.print();
    z.print();
  }

public:
  BVP<Size> x, y, z;
};

template <size_t Poly1_Size, size_t Poly2_Size, size_t Matrix_Size>
UnivariatePolyMatrix<Matrix_Size> bezout_matrix(const BVP<Poly1_Size> &poly1,
                                                const BVP<Poly2_Size> &poly2) {
  printf("No general implementation -- bezout_matrix<%ld. %ld, %ld>()\n",
         Poly1_Size, Poly2_Size, Matrix_Size);
}

}; // namespace resultant_simd

namespace resultant_simd {
/**
 * Size = 1 Specification
 */
template <> BVP<1ul>::BVP() { coeffs[0][0] = _mm256_setzero_pd(); }

template <> BVP<1ul>::BVP(const double *_coeffs) {
  coeffs[0][0] = _mm256_load_pd(&_coeffs[0]);
}

template <>
template <>
BVP<1ul> BVP<1ul>::operator-(const BVP<1ul> &other) const {
  BVP<1ul> result;
  result.coeffs[0][0] = _mm256_sub_pd(coeffs[0][0], other.coeffs[0][0]);
  return result;
}

/**
 * Size = 2 Specification
 */

template <> BVP<2ul>::BVP() {
  coeffs[0][0] = _mm256_setzero_pd();
  coeffs[0][1] = _mm256_setzero_pd();
  coeffs[1][0] = _mm256_setzero_pd();
}

//! loadu_pd ???
template <> BVP<2ul>::BVP(const double *_coeffs) {
  coeffs[0][0] = _mm256_load_pd(&_coeffs[0]);
  coeffs[0][1] = _mm256_load_pd(&_coeffs[4]);
  coeffs[1][0] = _mm256_load_pd(&_coeffs[8]);
}

template <>
template <>
BVP<2ul> BVP<1ul>::operator-(const BVP<2ul> &other) const {

  BVP<2ul> result;

  result.coeffs[0][0] = _mm256_sub_pd(coeffs[0][0], other.coeffs[0][0]);
  result.coeffs[0][1] = _mm256_sub_pd(result.coeffs[0][1], other.coeffs[0][1]);
  result.coeffs[1][0] = _mm256_sub_pd(result.coeffs[1][0], other.coeffs[1][0]);

  return result;
};

template <>
template <>
BVP<2ul> BVP<2ul>::operator-(const BVP<2ul> &other) const {
  BVP<2ul> res;
  res.coeffs[0][0] = _mm256_sub_pd(coeffs[0][0], other.coeffs[0][0]);
  res.coeffs[0][1] = _mm256_sub_pd(coeffs[0][1], other.coeffs[0][1]);
  res.coeffs[1][0] = _mm256_sub_pd(coeffs[1][0], other.coeffs[1][0]);
  return res;
}

template <>
template <>
BVP<2ul> BVP<2ul>::operator*(const BVP<1ul> &other) const {

  BVP<2ul> result;

  /**
   * result.coeffs[0][0] += coeffs[0][0] * other.coeffs[0][0];
   */
  result.coeffs[0][0] = _mm256_mul_pd(coeffs[0][0], other.coeffs[0][0]);

  /**
   * result.coeffs[0][1] += coeffs[0][1] * other.coeffs[0][0];
   */

  result.coeffs[0][1] = _mm256_mul_pd(coeffs[0][1], other.coeffs[0][0]);

  /**
   * result.coeffs[1][0] += coeffs[1][0] * other.coeffs[0][0];
   */
  result.coeffs[1][0] = _mm256_mul_pd(coeffs[1][0], other.coeffs[0][0]);

  return result;
}

template <>
template <>
BVP<2ul> BVP<2ul>::operator-(const BVP<1ul> &other) const {
  BVP<2ul> result;

  result.coeffs[0][0] = _mm256_sub_pd(coeffs[0][0], other.coeffs[0][0]);
  result.coeffs[0][1] = coeffs[0][1];
  result.coeffs[1][0] = coeffs[1][0];

  return result;
}

/**
 * Size = 3 Specification
 */

template <> BVP<3ul>::BVP() {
  coeffs[0][0] = _mm256_setzero_pd();
  coeffs[0][1] = _mm256_setzero_pd();
  coeffs[0][2] = _mm256_setzero_pd();
  coeffs[1][0] = _mm256_setzero_pd();
  coeffs[1][1] = _mm256_setzero_pd();
  coeffs[2][0] = _mm256_setzero_pd();
}

template <> BVP<3ul>::BVP(const double *_coeffs) {
  coeffs[0][0] = _mm256_load_pd(&_coeffs[0]);
  coeffs[0][1] = _mm256_load_pd(&_coeffs[4]);
  coeffs[0][2] = _mm256_load_pd(&_coeffs[8]);
  coeffs[1][0] = _mm256_load_pd(&_coeffs[12]);
  coeffs[1][1] = _mm256_load_pd(&_coeffs[16]);
  coeffs[2][0] = _mm256_load_pd(&_coeffs[20]);
}

template <>
template <>
BVP<3ul> BVP<3ul>::operator+(const BVP<3ul> &other) const {
  BVP<3ul> res;
  res.coeffs[0][0] = _mm256_add_pd(coeffs[0][0], other.coeffs[0][0]);
  res.coeffs[0][1] = _mm256_add_pd(coeffs[0][1], other.coeffs[0][1]);
  res.coeffs[0][2] = _mm256_add_pd(coeffs[0][2], other.coeffs[0][2]);
  res.coeffs[1][0] = _mm256_add_pd(coeffs[1][0], other.coeffs[1][0]);
  res.coeffs[1][1] = _mm256_add_pd(coeffs[1][1], other.coeffs[1][1]);
  res.coeffs[2][0] = _mm256_add_pd(coeffs[2][0], other.coeffs[2][0]);
  return res;
}

template <>
template <>
BVP<3ul> BVP<3ul>::operator-(const BVP<3ul> &other) const {
  BVP<3ul> res;
  res.coeffs[0][0] = _mm256_sub_pd(coeffs[0][0], other.coeffs[0][0]);
  res.coeffs[0][1] = _mm256_sub_pd(coeffs[0][1], other.coeffs[0][1]);
  res.coeffs[0][2] = _mm256_sub_pd(coeffs[0][2], other.coeffs[0][2]);
  res.coeffs[1][0] = _mm256_sub_pd(coeffs[1][0], other.coeffs[1][0]);
  res.coeffs[1][1] = _mm256_sub_pd(coeffs[1][1], other.coeffs[1][1]);
  res.coeffs[2][0] = _mm256_sub_pd(coeffs[2][0], other.coeffs[2][0]);
  return res;
}

template <> void BVP<3ul>::print() const {
  double pd[4];
  for (int y = 0; y < 3; ++y) {
    for (int x = 0; x < 3; ++x) {
      _mm256_store_pd(pd, coeffs[y][x]);

      printf("%.2lf ", pd[0]);
    }
    printf("\n");
  }
  printf("\n");
}

template <>
template <>
BVP<3ul> BVP<2ul>::operator*(const BVP<2ul> &other) const {

  BVP<3ul> result;

  /**
   * result.coeffs[0][0] += coeffs[0][0] * other.coeffs[0][0];
   */
  result.coeffs[0][0] = _mm256_mul_pd(coeffs[0][0], other.coeffs[0][0]);

  /**
   * result.coeffs[0][1] += coeffs[0][0] * other.coeffs[0][1];
   * result.coeffs[0][1] += coeffs[0][1] * other.coeffs[0][0];
   */
  __m256d tmp1 = _mm256_mul_pd(coeffs[0][0], other.coeffs[0][1]);
  __m256d tmp2 = _mm256_mul_pd(coeffs[0][1], other.coeffs[0][0]);
  result.coeffs[0][1] = _mm256_add_pd(tmp1, tmp2);

  /**
   *   result.coeffs[0][2] += coeffs[0][1] * other.coeffs[0][1];
   */
  result.coeffs[0][2] = _mm256_mul_pd(coeffs[0][1], other.coeffs[0][1]);

  /**
   * result.coeffs[1][0] += coeffs[0][0] * other.coeffs[1][0];
   * result.coeffs[1][0] += coeffs[1][0] * other.coeffs[0][0];
   */
  tmp1 = _mm256_mul_pd(coeffs[0][0], other.coeffs[1][0]);
  tmp2 = _mm256_mul_pd(coeffs[1][0], other.coeffs[0][0]);
  result.coeffs[1][0] = _mm256_add_pd(tmp1, tmp2);

  /**
   * result.coeffs[1][1] += coeffs[0][1] * other.coeffs[1][0];
   * result.coeffs[1][1] += coeffs[1][0] * other.coeffs[0][1];
   */
  tmp1 = _mm256_mul_pd(coeffs[0][1], other.coeffs[1][0]);
  tmp2 = _mm256_mul_pd(coeffs[1][0], other.coeffs[0][1]);
  result.coeffs[1][1] = _mm256_add_pd(tmp1, tmp2);

  /**
   *   result.coeffs[2][0] += coeffs[1][0] * other.coeffs[1][0];
   */

  result.coeffs[2][0] = _mm256_mul_pd(coeffs[1][0], other.coeffs[1][0]);

  return result;
}

/**
 * Size = 5 Specification
 */

template <> BVP<5ul>::BVP() {
  coeffs[0][0] = _mm256_setzero_pd();
  coeffs[0][1] = _mm256_setzero_pd();
  coeffs[0][2] = _mm256_setzero_pd();
  coeffs[0][3] = _mm256_setzero_pd();
  coeffs[0][4] = _mm256_setzero_pd();
  coeffs[1][0] = _mm256_setzero_pd();
  coeffs[1][1] = _mm256_setzero_pd();
  coeffs[1][2] = _mm256_setzero_pd();
  coeffs[1][3] = _mm256_setzero_pd();
  coeffs[2][0] = _mm256_setzero_pd();
  coeffs[2][1] = _mm256_setzero_pd();
  coeffs[2][2] = _mm256_setzero_pd();
  coeffs[3][0] = _mm256_setzero_pd();
  coeffs[3][1] = _mm256_setzero_pd();
  coeffs[4][0] = _mm256_setzero_pd();
}

template <> BVP<5ul>::BVP(const double *_coeffs) {
  coeffs[0][0] = _mm256_load_pd(&_coeffs[0]);
  coeffs[0][1] = _mm256_load_pd(&_coeffs[4]);
  coeffs[0][2] = _mm256_load_pd(&_coeffs[8]);
  coeffs[0][3] = _mm256_load_pd(&_coeffs[12]);
  coeffs[0][4] = _mm256_load_pd(&_coeffs[16]);
  coeffs[1][0] = _mm256_load_pd(&_coeffs[20]);
  coeffs[1][1] = _mm256_load_pd(&_coeffs[24]);
  coeffs[1][2] = _mm256_load_pd(&_coeffs[28]);
  coeffs[1][3] = _mm256_load_pd(&_coeffs[32]);
  coeffs[2][0] = _mm256_load_pd(&_coeffs[36]);
  coeffs[2][1] = _mm256_load_pd(&_coeffs[40]);
  coeffs[2][2] = _mm256_load_pd(&_coeffs[44]);
  coeffs[3][0] = _mm256_load_pd(&_coeffs[48]);
  coeffs[3][1] = _mm256_load_pd(&_coeffs[52]);
  coeffs[4][0] = _mm256_load_pd(&_coeffs[56]);
}

template <>
template <>
BVP<5ul> BVP<5ul>::operator+(const BVP<5ul> &other) const {
  BVP<5ul> res;
  res.coeffs[0][0] = _mm256_add_pd(coeffs[0][0], other.coeffs[0][0]);
  res.coeffs[0][1] = _mm256_add_pd(coeffs[0][1], other.coeffs[0][1]);
  res.coeffs[0][2] = _mm256_add_pd(coeffs[0][2], other.coeffs[0][2]);
  res.coeffs[0][3] = _mm256_add_pd(coeffs[0][3], other.coeffs[0][3]);
  res.coeffs[0][4] = _mm256_add_pd(coeffs[0][4], other.coeffs[0][4]);
  res.coeffs[1][0] = _mm256_add_pd(coeffs[1][0], other.coeffs[1][0]);
  res.coeffs[1][1] = _mm256_add_pd(coeffs[1][1], other.coeffs[1][1]);
  res.coeffs[1][2] = _mm256_add_pd(coeffs[1][2], other.coeffs[1][2]);
  res.coeffs[1][3] = _mm256_add_pd(coeffs[1][3], other.coeffs[1][3]);
  res.coeffs[2][0] = _mm256_add_pd(coeffs[2][0], other.coeffs[2][0]);
  res.coeffs[2][1] = _mm256_add_pd(coeffs[2][1], other.coeffs[2][1]);
  res.coeffs[2][2] = _mm256_add_pd(coeffs[2][2], other.coeffs[2][2]);
  res.coeffs[3][0] = _mm256_add_pd(coeffs[3][0], other.coeffs[3][0]);
  res.coeffs[3][1] = _mm256_add_pd(coeffs[3][1], other.coeffs[3][1]);
  res.coeffs[4][0] = _mm256_add_pd(coeffs[4][0], other.coeffs[4][0]);
  return res;
}

template <>
template <>
BVP<5ul> BVP<3ul>::operator*(const BVP<3ul> &other) const {

  BVP<5ul> result;

  /**
   * result.coeffs[0][0] += coeffs[0][0] * other.coeffs[0][0];
   */
  result.coeffs[0][0] = _mm256_mul_pd(coeffs[0][0], other.coeffs[0][0]);

  {
    /**
     * result.coeffs[0][1] += coeffs[0][0] * other.coeffs[0][1];
     * result.coeffs[0][1] += coeffs[0][1] * other.coeffs[0][0];
     */
    __m256d tmp1 = _mm256_mul_pd(coeffs[0][0], other.coeffs[0][1]);
    __m256d tmp2 = _mm256_mul_pd(coeffs[0][1], other.coeffs[0][0]);
    result.coeffs[0][1] = _mm256_add_pd(tmp1, tmp2);
  }

  {
    //   result.coeffs[0][2] += coeffs[0][0] * other.coeffs[0][2];
    //   result.coeffs[0][2] += coeffs[0][1] * other.coeffs[0][1];
    //   result.coeffs[0][2] += coeffs[0][2] * other.coeffs[0][0];

    __m256d tmp1 = _mm256_mul_pd(coeffs[0][0], other.coeffs[0][2]);
    __m256d tmp2 = _mm256_mul_pd(coeffs[0][1], other.coeffs[0][1]);
    tmp1 = _mm256_add_pd(tmp1, tmp2);
    result.coeffs[0][2] = _mm256_mul_pd(coeffs[0][2], other.coeffs[0][0]);
    result.coeffs[0][2] = _mm256_add_pd(result.coeffs[0][2], tmp1);
  }

  {
    //   result.coeffs[0][3] += coeffs[0][1] * other.coeffs[0][2];
    //   result.coeffs[0][3] += coeffs[0][2] * other.coeffs[0][1];
    __m256d tmp1 = _mm256_mul_pd(coeffs[0][1], other.coeffs[0][2]);
    __m256d tmp2 = _mm256_mul_pd(coeffs[0][2], other.coeffs[0][1]);
    result.coeffs[0][3] = _mm256_add_pd(tmp1, tmp2);
  }

  //   result.coeffs[0][4] += coeffs[0][2] * other.coeffs[0][2];
  result.coeffs[0][4] = _mm256_mul_pd(coeffs[0][2], other.coeffs[0][2]);

  {
    //   result.coeffs[1][0] += coeffs[0][0] * other.coeffs[1][0];
    //   result.coeffs[1][0] += coeffs[1][0] * other.coeffs[0][0];
    __m256d tmp1 = _mm256_mul_pd(coeffs[0][0], other.coeffs[1][0]);
    __m256d tmp2 = _mm256_mul_pd(coeffs[1][0], other.coeffs[0][0]);
    result.coeffs[1][0] = _mm256_add_pd(tmp1, tmp2);
  }
  {
    //   result.coeffs[1][1] += coeffs[0][0] * other.coeffs[1][1];
    //   result.coeffs[1][1] += coeffs[0][1] * other.coeffs[1][0];
    //   result.coeffs[1][1] += coeffs[1][0] * other.coeffs[0][1];
    //   result.coeffs[1][1] += coeffs[1][1] * other.coeffs[0][0];
    __m256d tmp1 = _mm256_mul_pd(coeffs[0][0], other.coeffs[1][1]);
    __m256d tmp2 = _mm256_mul_pd(coeffs[0][1], other.coeffs[1][0]);
    tmp1 = _mm256_add_pd(tmp1, tmp2);

    __m256d tmp3 = _mm256_mul_pd(coeffs[1][0], other.coeffs[0][1]);
    __m256d tmp4 = _mm256_mul_pd(coeffs[1][1], other.coeffs[0][0]);
    tmp3 = _mm256_add_pd(tmp3, tmp4);

    result.coeffs[1][1] = _mm256_add_pd(tmp1, tmp3);
  }

  {
    //   result.coeffs[1][2] += coeffs[0][1] * other.coeffs[1][1];
    //   result.coeffs[1][2] += coeffs[0][2] * other.coeffs[1][0];
    //   result.coeffs[1][2] += coeffs[1][0] * other.coeffs[0][2];
    //   result.coeffs[1][2] += coeffs[1][1] * other.coeffs[0][1];

    __m256d tmp1 = _mm256_mul_pd(coeffs[0][1], other.coeffs[1][1]);
    __m256d tmp2 = _mm256_mul_pd(coeffs[0][2], other.coeffs[1][0]);
    tmp1 = _mm256_add_pd(tmp1, tmp2);

    __m256d tmp3 = _mm256_mul_pd(coeffs[1][0], other.coeffs[0][2]);
    __m256d tmp4 = _mm256_mul_pd(coeffs[1][1], other.coeffs[0][1]);
    tmp3 = _mm256_add_pd(tmp3, tmp4);

    result.coeffs[1][2] = _mm256_add_pd(tmp1, tmp3);
  }

  {
    //   result.coeffs[1][3] += coeffs[0][2] * other.coeffs[1][1];
    //   result.coeffs[1][3] += coeffs[1][1] * other.coeffs[0][2];

    __m256d tmp1 = _mm256_mul_pd(coeffs[0][2], other.coeffs[1][1]);
    __m256d tmp2 = _mm256_mul_pd(coeffs[1][1], other.coeffs[0][2]);
    result.coeffs[1][3] = _mm256_add_pd(tmp1, tmp2);
  }

  {
    //   result.coeffs[2][0] += coeffs[0][0] * other.coeffs[2][0];
    //   result.coeffs[2][0] += coeffs[1][0] * other.coeffs[1][0];
    //   result.coeffs[2][0] += coeffs[2][0] * other.coeffs[0][0];

    __m256d tmp1 = _mm256_mul_pd(coeffs[0][0], other.coeffs[2][0]);
    __m256d tmp2 = _mm256_mul_pd(coeffs[1][0], other.coeffs[1][0]);
    tmp1 = _mm256_add_pd(tmp1, tmp2);
    result.coeffs[2][0] = _mm256_mul_pd(coeffs[2][0], other.coeffs[0][0]);
    result.coeffs[2][0] = _mm256_add_pd(result.coeffs[2][0], tmp1);
  }

  {
    //   result.coeffs[2][1] += coeffs[0][1] * other.coeffs[2][0];
    //   result.coeffs[2][1] += coeffs[1][0] * other.coeffs[1][1];
    //   result.coeffs[2][1] += coeffs[1][1] * other.coeffs[1][0];
    //   result.coeffs[2][1] += coeffs[2][0] * other.coeffs[0][1];

    __m256d tmp1 = _mm256_mul_pd(coeffs[0][1], other.coeffs[2][0]);
    __m256d tmp2 = _mm256_mul_pd(coeffs[1][0], other.coeffs[1][1]);
    tmp1 = _mm256_add_pd(tmp1, tmp2);

    __m256d tmp3 = _mm256_mul_pd(coeffs[1][1], other.coeffs[1][0]);
    __m256d tmp4 = _mm256_mul_pd(coeffs[2][0], other.coeffs[0][1]);
    tmp3 = _mm256_add_pd(tmp3, tmp4);

    result.coeffs[2][1] = _mm256_add_pd(tmp1, tmp3);
  }

  {
    //   result.coeffs[2][2] += coeffs[0][2] * other.coeffs[2][0];
    //   result.coeffs[2][2] += coeffs[1][1] * other.coeffs[1][1];
    //   result.coeffs[2][2] += coeffs[2][0] * other.coeffs[0][2];

    __m256d tmp1 = _mm256_mul_pd(coeffs[0][2], other.coeffs[2][0]);
    __m256d tmp2 = _mm256_mul_pd(coeffs[1][1], other.coeffs[1][1]);
    tmp1 = _mm256_add_pd(tmp1, tmp2);
    result.coeffs[2][2] = _mm256_mul_pd(coeffs[2][0], other.coeffs[0][2]);
    result.coeffs[2][2] = _mm256_add_pd(result.coeffs[2][2], tmp1);
  }

  {
    //   result.coeffs[3][0] += coeffs[1][0] * other.coeffs[2][0];
    //   result.coeffs[3][0] += coeffs[2][0] * other.coeffs[1][0];

    __m256d tmp1 = _mm256_mul_pd(coeffs[1][0], other.coeffs[2][0]);
    __m256d tmp2 = _mm256_mul_pd(coeffs[2][0], other.coeffs[1][0]);
    result.coeffs[3][0] = _mm256_add_pd(tmp1, tmp2);
  }

  {
    //   result.coeffs[3][1] += coeffs[1][1] * other.coeffs[2][0];
    //   result.coeffs[3][1] += coeffs[2][0] * other.coeffs[1][1];

    __m256d tmp1 = _mm256_mul_pd(coeffs[1][1], other.coeffs[2][0]);
    __m256d tmp2 = _mm256_mul_pd(coeffs[2][0], other.coeffs[1][1]);
    result.coeffs[3][1] = _mm256_add_pd(tmp1, tmp2);
  }

  //   result.coeffs[4][0] += coeffs[2][0] * other.coeffs[2][0];

  result.coeffs[4][0] = _mm256_mul_pd(coeffs[2][0], other.coeffs[2][0]);

  return result;
}

template<>
BVP<7ul>::BVP() {
  coeffs[0][0] = _mm256_setzero_pd();
  coeffs[0][1] = _mm256_setzero_pd();
  coeffs[0][2] = _mm256_setzero_pd();
  coeffs[0][3] = _mm256_setzero_pd();
  coeffs[0][4] = _mm256_setzero_pd();
  coeffs[0][5] = _mm256_setzero_pd();
  coeffs[0][6] = _mm256_setzero_pd();

  coeffs[1][0] = _mm256_setzero_pd();
  coeffs[1][1] = _mm256_setzero_pd();
  coeffs[1][2] = _mm256_setzero_pd();
  coeffs[1][3] = _mm256_setzero_pd();
  coeffs[1][4] = _mm256_setzero_pd();
  coeffs[1][5] = _mm256_setzero_pd();

  coeffs[2][0] = _mm256_setzero_pd();
  coeffs[2][1] = _mm256_setzero_pd();
  coeffs[2][2] = _mm256_setzero_pd();
  coeffs[2][3] = _mm256_setzero_pd();
  coeffs[2][4] = _mm256_setzero_pd();

  coeffs[3][0] = _mm256_setzero_pd();
  coeffs[3][1] = _mm256_setzero_pd();
  coeffs[3][2] = _mm256_setzero_pd();
  coeffs[3][3] = _mm256_setzero_pd();
 
  coeffs[4][0] = _mm256_setzero_pd();
  coeffs[4][1] = _mm256_setzero_pd();
  coeffs[4][2] = _mm256_setzero_pd();
 
  coeffs[5][0] = _mm256_setzero_pd();
  coeffs[5][1] = _mm256_setzero_pd();

  coeffs[6][0] = _mm256_setzero_pd();
}

template <>
template <>
BVP<7ul> BVP<5ul>::operator*(const BVP<3ul> &other) const {
  BVP<7ul> result;
  result.coeffs[0][0] = _mm256_add_pd(
      result.coeffs[0][0], _mm256_mul_pd(coeffs[0][0], other.coeffs[0][0]));
  result.coeffs[0][1] = _mm256_add_pd(
      result.coeffs[0][1], _mm256_mul_pd(coeffs[0][0], other.coeffs[0][1]));
  result.coeffs[0][2] = _mm256_add_pd(
      result.coeffs[0][2], _mm256_mul_pd(coeffs[0][0], other.coeffs[0][2]));
  result.coeffs[0][1] = _mm256_add_pd(
      result.coeffs[0][1], _mm256_mul_pd(coeffs[0][1], other.coeffs[0][0]));
  result.coeffs[0][2] = _mm256_add_pd(
      result.coeffs[0][2], _mm256_mul_pd(coeffs[0][1], other.coeffs[0][1]));
  result.coeffs[0][3] = _mm256_add_pd(
      result.coeffs[0][3], _mm256_mul_pd(coeffs[0][1], other.coeffs[0][2]));
  result.coeffs[0][2] = _mm256_add_pd(
      result.coeffs[0][2], _mm256_mul_pd(coeffs[0][2], other.coeffs[0][0]));
  result.coeffs[0][3] = _mm256_add_pd(
      result.coeffs[0][3], _mm256_mul_pd(coeffs[0][2], other.coeffs[0][1]));
  result.coeffs[0][4] = _mm256_add_pd(
      result.coeffs[0][4], _mm256_mul_pd(coeffs[0][2], other.coeffs[0][2]));
  result.coeffs[0][3] = _mm256_add_pd(
      result.coeffs[0][3], _mm256_mul_pd(coeffs[0][3], other.coeffs[0][0]));
  result.coeffs[0][4] = _mm256_add_pd(
      result.coeffs[0][4], _mm256_mul_pd(coeffs[0][3], other.coeffs[0][1]));
  result.coeffs[0][5] = _mm256_add_pd(
      result.coeffs[0][5], _mm256_mul_pd(coeffs[0][3], other.coeffs[0][2]));
  result.coeffs[0][4] = _mm256_add_pd(
      result.coeffs[0][4], _mm256_mul_pd(coeffs[0][4], other.coeffs[0][0]));
  result.coeffs[0][5] = _mm256_add_pd(
      result.coeffs[0][5], _mm256_mul_pd(coeffs[0][4], other.coeffs[0][1]));
  result.coeffs[0][6] = _mm256_add_pd(
      result.coeffs[0][6], _mm256_mul_pd(coeffs[0][4], other.coeffs[0][2]));
  result.coeffs[1][0] = _mm256_add_pd(
      result.coeffs[1][0], _mm256_mul_pd(coeffs[0][0], other.coeffs[1][0]));
  result.coeffs[1][1] = _mm256_add_pd(
      result.coeffs[1][1], _mm256_mul_pd(coeffs[0][0], other.coeffs[1][1]));
  result.coeffs[1][1] = _mm256_add_pd(
      result.coeffs[1][1], _mm256_mul_pd(coeffs[0][1], other.coeffs[1][0]));
  result.coeffs[1][2] = _mm256_add_pd(
      result.coeffs[1][2], _mm256_mul_pd(coeffs[0][1], other.coeffs[1][1]));
  result.coeffs[1][2] = _mm256_add_pd(
      result.coeffs[1][2], _mm256_mul_pd(coeffs[0][2], other.coeffs[1][0]));
  result.coeffs[1][3] = _mm256_add_pd(
      result.coeffs[1][3], _mm256_mul_pd(coeffs[0][2], other.coeffs[1][1]));
  result.coeffs[1][3] = _mm256_add_pd(
      result.coeffs[1][3], _mm256_mul_pd(coeffs[0][3], other.coeffs[1][0]));
  result.coeffs[1][4] = _mm256_add_pd(
      result.coeffs[1][4], _mm256_mul_pd(coeffs[0][3], other.coeffs[1][1]));
  result.coeffs[1][4] = _mm256_add_pd(
      result.coeffs[1][4], _mm256_mul_pd(coeffs[0][4], other.coeffs[1][0]));
  result.coeffs[1][5] = _mm256_add_pd(
      result.coeffs[1][5], _mm256_mul_pd(coeffs[0][4], other.coeffs[1][1]));
  result.coeffs[2][0] = _mm256_add_pd(
      result.coeffs[2][0], _mm256_mul_pd(coeffs[0][0], other.coeffs[2][0]));
  result.coeffs[2][1] = _mm256_add_pd(
      result.coeffs[2][1], _mm256_mul_pd(coeffs[0][1], other.coeffs[2][0]));
  result.coeffs[2][2] = _mm256_add_pd(
      result.coeffs[2][2], _mm256_mul_pd(coeffs[0][2], other.coeffs[2][0]));
  result.coeffs[2][3] = _mm256_add_pd(
      result.coeffs[2][3], _mm256_mul_pd(coeffs[0][3], other.coeffs[2][0]));
  result.coeffs[2][4] = _mm256_add_pd(
      result.coeffs[2][4], _mm256_mul_pd(coeffs[0][4], other.coeffs[2][0]));
  result.coeffs[1][0] = _mm256_add_pd(
      result.coeffs[1][0], _mm256_mul_pd(coeffs[1][0], other.coeffs[0][0]));
  result.coeffs[1][1] = _mm256_add_pd(
      result.coeffs[1][1], _mm256_mul_pd(coeffs[1][0], other.coeffs[0][1]));
  result.coeffs[1][2] = _mm256_add_pd(
      result.coeffs[1][2], _mm256_mul_pd(coeffs[1][0], other.coeffs[0][2]));
  result.coeffs[1][1] = _mm256_add_pd(
      result.coeffs[1][1], _mm256_mul_pd(coeffs[1][1], other.coeffs[0][0]));
  result.coeffs[1][2] = _mm256_add_pd(
      result.coeffs[1][2], _mm256_mul_pd(coeffs[1][1], other.coeffs[0][1]));
  result.coeffs[1][3] = _mm256_add_pd(
      result.coeffs[1][3], _mm256_mul_pd(coeffs[1][1], other.coeffs[0][2]));
  result.coeffs[1][2] = _mm256_add_pd(
      result.coeffs[1][2], _mm256_mul_pd(coeffs[1][2], other.coeffs[0][0]));
  result.coeffs[1][3] = _mm256_add_pd(
      result.coeffs[1][3], _mm256_mul_pd(coeffs[1][2], other.coeffs[0][1]));
  result.coeffs[1][4] = _mm256_add_pd(
      result.coeffs[1][4], _mm256_mul_pd(coeffs[1][2], other.coeffs[0][2]));
  result.coeffs[1][3] = _mm256_add_pd(
      result.coeffs[1][3], _mm256_mul_pd(coeffs[1][3], other.coeffs[0][0]));
  result.coeffs[1][4] = _mm256_add_pd(
      result.coeffs[1][4], _mm256_mul_pd(coeffs[1][3], other.coeffs[0][1]));
  result.coeffs[1][5] = _mm256_add_pd(
      result.coeffs[1][5], _mm256_mul_pd(coeffs[1][3], other.coeffs[0][2]));
  result.coeffs[2][0] = _mm256_add_pd(
      result.coeffs[2][0], _mm256_mul_pd(coeffs[1][0], other.coeffs[1][0]));
  result.coeffs[2][1] = _mm256_add_pd(
      result.coeffs[2][1], _mm256_mul_pd(coeffs[1][0], other.coeffs[1][1]));
  result.coeffs[2][1] = _mm256_add_pd(
      result.coeffs[2][1], _mm256_mul_pd(coeffs[1][1], other.coeffs[1][0]));
  result.coeffs[2][2] = _mm256_add_pd(
      result.coeffs[2][2], _mm256_mul_pd(coeffs[1][1], other.coeffs[1][1]));
  result.coeffs[2][2] = _mm256_add_pd(
      result.coeffs[2][2], _mm256_mul_pd(coeffs[1][2], other.coeffs[1][0]));
  result.coeffs[2][3] = _mm256_add_pd(
      result.coeffs[2][3], _mm256_mul_pd(coeffs[1][2], other.coeffs[1][1]));
  result.coeffs[2][3] = _mm256_add_pd(
      result.coeffs[2][3], _mm256_mul_pd(coeffs[1][3], other.coeffs[1][0]));
  result.coeffs[2][4] = _mm256_add_pd(
      result.coeffs[2][4], _mm256_mul_pd(coeffs[1][3], other.coeffs[1][1]));
  result.coeffs[3][0] = _mm256_add_pd(
      result.coeffs[3][0], _mm256_mul_pd(coeffs[1][0], other.coeffs[2][0]));
  result.coeffs[3][1] = _mm256_add_pd(
      result.coeffs[3][1], _mm256_mul_pd(coeffs[1][1], other.coeffs[2][0]));
  result.coeffs[3][2] = _mm256_add_pd(
      result.coeffs[3][2], _mm256_mul_pd(coeffs[1][2], other.coeffs[2][0]));
  result.coeffs[3][3] = _mm256_add_pd(
      result.coeffs[3][3], _mm256_mul_pd(coeffs[1][3], other.coeffs[2][0]));
  result.coeffs[2][0] = _mm256_add_pd(
      result.coeffs[2][0], _mm256_mul_pd(coeffs[2][0], other.coeffs[0][0]));
  result.coeffs[2][1] = _mm256_add_pd(
      result.coeffs[2][1], _mm256_mul_pd(coeffs[2][0], other.coeffs[0][1]));
  result.coeffs[2][2] = _mm256_add_pd(
      result.coeffs[2][2], _mm256_mul_pd(coeffs[2][0], other.coeffs[0][2]));
  result.coeffs[2][1] = _mm256_add_pd(
      result.coeffs[2][1], _mm256_mul_pd(coeffs[2][1], other.coeffs[0][0]));
  result.coeffs[2][2] = _mm256_add_pd(
      result.coeffs[2][2], _mm256_mul_pd(coeffs[2][1], other.coeffs[0][1]));
  result.coeffs[2][3] = _mm256_add_pd(
      result.coeffs[2][3], _mm256_mul_pd(coeffs[2][1], other.coeffs[0][2]));
  result.coeffs[2][2] = _mm256_add_pd(
      result.coeffs[2][2], _mm256_mul_pd(coeffs[2][2], other.coeffs[0][0]));
  result.coeffs[2][3] = _mm256_add_pd(
      result.coeffs[2][3], _mm256_mul_pd(coeffs[2][2], other.coeffs[0][1]));
  result.coeffs[2][4] = _mm256_add_pd(
      result.coeffs[2][4], _mm256_mul_pd(coeffs[2][2], other.coeffs[0][2]));
  result.coeffs[3][0] = _mm256_add_pd(
      result.coeffs[3][0], _mm256_mul_pd(coeffs[2][0], other.coeffs[1][0]));
  result.coeffs[3][1] = _mm256_add_pd(
      result.coeffs[3][1], _mm256_mul_pd(coeffs[2][0], other.coeffs[1][1]));
  result.coeffs[3][1] = _mm256_add_pd(
      result.coeffs[3][1], _mm256_mul_pd(coeffs[2][1], other.coeffs[1][0]));
  result.coeffs[3][2] = _mm256_add_pd(
      result.coeffs[3][2], _mm256_mul_pd(coeffs[2][1], other.coeffs[1][1]));
  result.coeffs[3][2] = _mm256_add_pd(
      result.coeffs[3][2], _mm256_mul_pd(coeffs[2][2], other.coeffs[1][0]));
  result.coeffs[3][3] = _mm256_add_pd(
      result.coeffs[3][3], _mm256_mul_pd(coeffs[2][2], other.coeffs[1][1]));
  result.coeffs[4][0] = _mm256_add_pd(
      result.coeffs[4][0], _mm256_mul_pd(coeffs[2][0], other.coeffs[2][0]));
  result.coeffs[4][1] = _mm256_add_pd(
      result.coeffs[4][1], _mm256_mul_pd(coeffs[2][1], other.coeffs[2][0]));
  result.coeffs[4][2] = _mm256_add_pd(
      result.coeffs[4][2], _mm256_mul_pd(coeffs[2][2], other.coeffs[2][0]));
  result.coeffs[3][0] = _mm256_add_pd(
      result.coeffs[3][0], _mm256_mul_pd(coeffs[3][0], other.coeffs[0][0]));
  result.coeffs[3][1] = _mm256_add_pd(
      result.coeffs[3][1], _mm256_mul_pd(coeffs[3][0], other.coeffs[0][1]));
  result.coeffs[3][2] = _mm256_add_pd(
      result.coeffs[3][2], _mm256_mul_pd(coeffs[3][0], other.coeffs[0][2]));
  result.coeffs[3][1] = _mm256_add_pd(
      result.coeffs[3][1], _mm256_mul_pd(coeffs[3][1], other.coeffs[0][0]));
  result.coeffs[3][2] = _mm256_add_pd(
      result.coeffs[3][2], _mm256_mul_pd(coeffs[3][1], other.coeffs[0][1]));
  result.coeffs[3][3] = _mm256_add_pd(
      result.coeffs[3][3], _mm256_mul_pd(coeffs[3][1], other.coeffs[0][2]));
  result.coeffs[4][0] = _mm256_add_pd(
      result.coeffs[4][0], _mm256_mul_pd(coeffs[3][0], other.coeffs[1][0]));
  result.coeffs[4][1] = _mm256_add_pd(
      result.coeffs[4][1], _mm256_mul_pd(coeffs[3][0], other.coeffs[1][1]));
  result.coeffs[4][1] = _mm256_add_pd(
      result.coeffs[4][1], _mm256_mul_pd(coeffs[3][1], other.coeffs[1][0]));
  result.coeffs[4][2] = _mm256_add_pd(
      result.coeffs[4][2], _mm256_mul_pd(coeffs[3][1], other.coeffs[1][1]));
  result.coeffs[5][0] = _mm256_add_pd(
      result.coeffs[5][0], _mm256_mul_pd(coeffs[3][0], other.coeffs[2][0]));
  result.coeffs[5][1] = _mm256_add_pd(
      result.coeffs[5][1], _mm256_mul_pd(coeffs[3][1], other.coeffs[2][0]));
  result.coeffs[4][0] = _mm256_add_pd(
      result.coeffs[4][0], _mm256_mul_pd(coeffs[4][0], other.coeffs[0][0]));
  result.coeffs[4][1] = _mm256_add_pd(
      result.coeffs[4][1], _mm256_mul_pd(coeffs[4][0], other.coeffs[0][1]));
  result.coeffs[4][2] = _mm256_add_pd(
      result.coeffs[4][2], _mm256_mul_pd(coeffs[4][0], other.coeffs[0][2]));
  result.coeffs[5][0] = _mm256_add_pd(
      result.coeffs[5][0], _mm256_mul_pd(coeffs[4][0], other.coeffs[1][0]));
  result.coeffs[5][1] = _mm256_add_pd(
      result.coeffs[5][1], _mm256_mul_pd(coeffs[4][0], other.coeffs[1][1]));
  result.coeffs[6][0] = _mm256_add_pd(
      result.coeffs[6][0], _mm256_mul_pd(coeffs[4][0], other.coeffs[2][0]));

  return result;
}

template <>
template <>
BVP<7ul> BVP<7ul>::operator-(const BVP<7ul> &other) const {
  BVP<7ul> result;

  result.coeffs[0][0] = _mm256_sub_pd(coeffs[0][0], other.coeffs[0][0]);
  result.coeffs[0][1] = _mm256_sub_pd(coeffs[0][1], other.coeffs[0][1]);
  result.coeffs[0][2] = _mm256_sub_pd(coeffs[0][2], other.coeffs[0][2]);
  result.coeffs[0][3] = _mm256_sub_pd(coeffs[0][3], other.coeffs[0][3]);
  result.coeffs[0][4] = _mm256_sub_pd(coeffs[0][4], other.coeffs[0][4]);
  result.coeffs[0][5] = _mm256_sub_pd(coeffs[0][5], other.coeffs[0][5]);
  result.coeffs[0][6] = _mm256_sub_pd(coeffs[0][6], other.coeffs[0][6]);
  result.coeffs[1][0] = _mm256_sub_pd(coeffs[1][0], other.coeffs[1][0]);
  result.coeffs[1][1] = _mm256_sub_pd(coeffs[1][1], other.coeffs[1][1]);
  result.coeffs[1][2] = _mm256_sub_pd(coeffs[1][2], other.coeffs[1][2]);
  result.coeffs[1][3] = _mm256_sub_pd(coeffs[1][3], other.coeffs[1][3]);
  result.coeffs[1][4] = _mm256_sub_pd(coeffs[1][4], other.coeffs[1][4]);
  result.coeffs[1][5] = _mm256_sub_pd(coeffs[1][5], other.coeffs[1][5]);
  result.coeffs[2][0] = _mm256_sub_pd(coeffs[2][0], other.coeffs[2][0]);
  result.coeffs[2][1] = _mm256_sub_pd(coeffs[2][1], other.coeffs[2][1]);
  result.coeffs[2][2] = _mm256_sub_pd(coeffs[2][2], other.coeffs[2][2]);
  result.coeffs[2][3] = _mm256_sub_pd(coeffs[2][3], other.coeffs[2][3]);
  result.coeffs[2][4] = _mm256_sub_pd(coeffs[2][4], other.coeffs[2][4]);
  result.coeffs[3][0] = _mm256_sub_pd(coeffs[3][0], other.coeffs[3][0]);
  result.coeffs[3][1] = _mm256_sub_pd(coeffs[3][1], other.coeffs[3][1]);
  result.coeffs[3][2] = _mm256_sub_pd(coeffs[3][2], other.coeffs[3][2]);
  result.coeffs[3][3] = _mm256_sub_pd(coeffs[3][3], other.coeffs[3][3]);
  result.coeffs[4][0] = _mm256_sub_pd(coeffs[4][0], other.coeffs[4][0]);
  result.coeffs[4][1] = _mm256_sub_pd(coeffs[4][1], other.coeffs[4][1]);
  result.coeffs[4][2] = _mm256_sub_pd(coeffs[4][2], other.coeffs[4][2]);
  result.coeffs[5][0] = _mm256_sub_pd(coeffs[5][0], other.coeffs[5][0]);
  result.coeffs[5][1] = _mm256_sub_pd(coeffs[5][1], other.coeffs[5][1]);
  result.coeffs[6][0] = _mm256_sub_pd(coeffs[6][0], other.coeffs[6][0]);

  return result;
}

template <> BVP<7ul> BVP<7ul>::operator*(const double d) const {
  BVP<7ul> result;

  __m256d scalar = _mm256_broadcast_sd(&d);

  result.coeffs[0][0] = _mm256_mul_pd(coeffs[0][0], scalar);
  result.coeffs[0][1] = _mm256_mul_pd(coeffs[0][1], scalar);
  result.coeffs[0][2] = _mm256_mul_pd(coeffs[0][2], scalar);
  result.coeffs[0][3] = _mm256_mul_pd(coeffs[0][3], scalar);
  result.coeffs[0][4] = _mm256_mul_pd(coeffs[0][4], scalar);
  result.coeffs[0][5] = _mm256_mul_pd(coeffs[0][5], scalar);
  result.coeffs[0][6] = _mm256_mul_pd(coeffs[0][6], scalar);
  result.coeffs[1][0] = _mm256_mul_pd(coeffs[1][0], scalar);
  result.coeffs[1][1] = _mm256_mul_pd(coeffs[1][1], scalar);
  result.coeffs[1][2] = _mm256_mul_pd(coeffs[1][2], scalar);
  result.coeffs[1][3] = _mm256_mul_pd(coeffs[1][3], scalar);
  result.coeffs[1][4] = _mm256_mul_pd(coeffs[1][4], scalar);
  result.coeffs[1][5] = _mm256_mul_pd(coeffs[1][5], scalar);
  result.coeffs[2][0] = _mm256_mul_pd(coeffs[2][0], scalar);
  result.coeffs[2][1] = _mm256_mul_pd(coeffs[2][1], scalar);
  result.coeffs[2][2] = _mm256_mul_pd(coeffs[2][2], scalar);
  result.coeffs[2][3] = _mm256_mul_pd(coeffs[2][3], scalar);
  result.coeffs[2][4] = _mm256_mul_pd(coeffs[2][4], scalar);
  result.coeffs[3][0] = _mm256_mul_pd(coeffs[3][0], scalar);
  result.coeffs[3][1] = _mm256_mul_pd(coeffs[3][1], scalar);
  result.coeffs[3][2] = _mm256_mul_pd(coeffs[3][2], scalar);
  result.coeffs[3][3] = _mm256_mul_pd(coeffs[3][3], scalar);
  result.coeffs[4][0] = _mm256_mul_pd(coeffs[4][0], scalar);
  result.coeffs[4][1] = _mm256_mul_pd(coeffs[4][1], scalar);
  result.coeffs[4][2] = _mm256_mul_pd(coeffs[4][2], scalar);
  result.coeffs[5][0] = _mm256_mul_pd(coeffs[5][0], scalar);
  result.coeffs[5][1] = _mm256_mul_pd(coeffs[5][1], scalar);
  result.coeffs[6][0] = _mm256_mul_pd(coeffs[6][0], scalar);
  return result;
}

template <size_t Size>
void depack_bvp(const BVP<Size> &bvp_packed,
                Resultant::BivariatePolynomial *bvp_4) {
  printf("No general implementation -- depack_bvp<%ld>()\n", Size);
}

template <>
void depack_bvp(const BVP<3ul> &bvp_packed,
                Resultant::BivariatePolynomial *bvp_4) {
  //
  double coeff_00[4];
  double coeff_01[4];
  double coeff_02[4];
  double coeff_10[4];
  double coeff_11[4];
  double coeff_20[4];

  _mm256_store_pd(coeff_00, bvp_packed.coeffs[0][0]);
  _mm256_store_pd(coeff_01, bvp_packed.coeffs[0][1]);
  _mm256_store_pd(coeff_02, bvp_packed.coeffs[0][2]);
  _mm256_store_pd(coeff_10, bvp_packed.coeffs[1][0]);
  _mm256_store_pd(coeff_11, bvp_packed.coeffs[1][1]);
  _mm256_store_pd(coeff_20, bvp_packed.coeffs[2][0]);

  for (int i = 0; i < 4; ++i) {
    std::vector<std::vector<double>> coefs(3);

    coefs[0] = {coeff_00[i], coeff_01[i], coeff_02[i]};
    coefs[1] = {coeff_10[i], coeff_11[i], 0.0};
    coefs[2] = {coeff_20[i], 0.0, 0.0};

    bvp_4[i] = Resultant::BivariatePolynomial(coefs);
  }
}

template <>
void depack_bvp(const BVP<5ul> &bvp_packed,
                Resultant::BivariatePolynomial *bvp_4) {
  double coeff_00[4];
  double coeff_01[4];
  double coeff_02[4];
  double coeff_03[4];
  double coeff_04[4];

  double coeff_10[4];
  double coeff_11[4];
  double coeff_12[4];
  double coeff_13[4];

  double coeff_20[4];
  double coeff_21[4];
  double coeff_22[4];

  double coeff_30[4];
  double coeff_31[4];

  double coeff_40[4];

  _mm256_store_pd(coeff_00, bvp_packed.coeffs[0][0]);
  _mm256_store_pd(coeff_01, bvp_packed.coeffs[0][1]);
  _mm256_store_pd(coeff_02, bvp_packed.coeffs[0][2]);
  _mm256_store_pd(coeff_03, bvp_packed.coeffs[0][3]);
  _mm256_store_pd(coeff_04, bvp_packed.coeffs[0][4]);

  _mm256_store_pd(coeff_10, bvp_packed.coeffs[1][0]);
  _mm256_store_pd(coeff_11, bvp_packed.coeffs[1][1]);
  _mm256_store_pd(coeff_12, bvp_packed.coeffs[1][2]);
  _mm256_store_pd(coeff_13, bvp_packed.coeffs[1][3]);

  _mm256_store_pd(coeff_20, bvp_packed.coeffs[2][0]);
  _mm256_store_pd(coeff_21, bvp_packed.coeffs[2][1]);
  _mm256_store_pd(coeff_22, bvp_packed.coeffs[2][2]);

  _mm256_store_pd(coeff_30, bvp_packed.coeffs[3][0]);
  _mm256_store_pd(coeff_31, bvp_packed.coeffs[3][1]);

  _mm256_store_pd(coeff_40, bvp_packed.coeffs[4][0]);

  for (int i = 0; i < 4; ++i) {
    std::vector<std::vector<double>> coefs(5);

    coefs[0] = {coeff_00[i], coeff_01[i], coeff_02[i], coeff_03[i],
                coeff_04[i]};
    coefs[1] = {coeff_10[i], coeff_11[i], coeff_12[i], coeff_13[i], 0.0};
    coefs[2] = {coeff_20[i], coeff_21[i], coeff_22[i], 0.0, 0.0};
    coefs[3] = {coeff_30[i], coeff_31[i], 0.0, 0.0, 0.0};
    coefs[4] = {coeff_40[i], 0.0, 0.0, 0.0, 0.0};

    bvp_4[i] = Resultant::BivariatePolynomial(coefs);
  }
}

template <size_t X_INDEX, size_t Y_INDEX>
void reflection_unipoly_cross(const BVP<5ul> &poly1, const BVP<3ul> &poly2,
                              UnivariatePolyMatrix<4ul> *bezout_matrix) {
  printf("No general implementation -- reflection_unipoly_cross<%ld, %ld>*()\n",
         X_INDEX, Y_INDEX);
}

template <>
void reflection_unipoly_cross<0ul, 0ul>(
    const BVP<5UL> &poly1, const BVP<3UL> &poly2,
    UnivariatePolyMatrix<4UL> *bezout_matrix) {
  bezout_matrix->matrix[0][0][0] =
      _mm256_add_pd(bezout_matrix->matrix[0][0][0],
                    _mm256_mul_pd(poly1.coeffs[0][0], poly2.coeffs[1][0]));
  bezout_matrix->matrix[0][0][1] =
      _mm256_add_pd(bezout_matrix->matrix[0][0][1],
                    _mm256_mul_pd(poly1.coeffs[0][0], poly2.coeffs[1][1]));
  bezout_matrix->matrix[0][0][1] =
      _mm256_add_pd(bezout_matrix->matrix[0][0][1],
                    _mm256_mul_pd(poly1.coeffs[0][1], poly2.coeffs[1][0]));
  bezout_matrix->matrix[0][0][2] =
      _mm256_add_pd(bezout_matrix->matrix[0][0][2],
                    _mm256_mul_pd(poly1.coeffs[0][1], poly2.coeffs[1][1]));
  bezout_matrix->matrix[0][0][2] =
      _mm256_add_pd(bezout_matrix->matrix[0][0][2],
                    _mm256_mul_pd(poly1.coeffs[0][2], poly2.coeffs[1][0]));
  bezout_matrix->matrix[0][0][3] =
      _mm256_add_pd(bezout_matrix->matrix[0][0][3],
                    _mm256_mul_pd(poly1.coeffs[0][2], poly2.coeffs[1][1]));
  bezout_matrix->matrix[0][0][3] =
      _mm256_add_pd(bezout_matrix->matrix[0][0][3],
                    _mm256_mul_pd(poly1.coeffs[0][3], poly2.coeffs[1][0]));
  bezout_matrix->matrix[0][0][4] =
      _mm256_add_pd(bezout_matrix->matrix[0][0][4],
                    _mm256_mul_pd(poly1.coeffs[0][3], poly2.coeffs[1][1]));
  bezout_matrix->matrix[0][0][4] =
      _mm256_add_pd(bezout_matrix->matrix[0][0][4],
                    _mm256_mul_pd(poly1.coeffs[0][4], poly2.coeffs[1][0]));
  bezout_matrix->matrix[0][0][5] =
      _mm256_add_pd(bezout_matrix->matrix[0][0][5],
                    _mm256_mul_pd(poly1.coeffs[0][4], poly2.coeffs[1][1]));
  bezout_matrix->matrix[0][0][0] =
      _mm256_sub_pd(bezout_matrix->matrix[0][0][0],
                    _mm256_mul_pd(poly2.coeffs[0][0], poly1.coeffs[1][0]));
  bezout_matrix->matrix[0][0][1] =
      _mm256_sub_pd(bezout_matrix->matrix[0][0][1],
                    _mm256_mul_pd(poly2.coeffs[0][0], poly1.coeffs[1][1]));
  bezout_matrix->matrix[0][0][2] =
      _mm256_sub_pd(bezout_matrix->matrix[0][0][2],
                    _mm256_mul_pd(poly2.coeffs[0][0], poly1.coeffs[1][2]));
  bezout_matrix->matrix[0][0][3] =
      _mm256_sub_pd(bezout_matrix->matrix[0][0][3],
                    _mm256_mul_pd(poly2.coeffs[0][0], poly1.coeffs[1][3]));
  bezout_matrix->matrix[0][0][1] =
      _mm256_sub_pd(bezout_matrix->matrix[0][0][1],
                    _mm256_mul_pd(poly2.coeffs[0][1], poly1.coeffs[1][0]));
  bezout_matrix->matrix[0][0][2] =
      _mm256_sub_pd(bezout_matrix->matrix[0][0][2],
                    _mm256_mul_pd(poly2.coeffs[0][1], poly1.coeffs[1][1]));
  bezout_matrix->matrix[0][0][3] =
      _mm256_sub_pd(bezout_matrix->matrix[0][0][3],
                    _mm256_mul_pd(poly2.coeffs[0][1], poly1.coeffs[1][2]));
  bezout_matrix->matrix[0][0][4] =
      _mm256_sub_pd(bezout_matrix->matrix[0][0][4],
                    _mm256_mul_pd(poly2.coeffs[0][1], poly1.coeffs[1][3]));
  bezout_matrix->matrix[0][0][2] =
      _mm256_sub_pd(bezout_matrix->matrix[0][0][2],
                    _mm256_mul_pd(poly2.coeffs[0][2], poly1.coeffs[1][0]));
  bezout_matrix->matrix[0][0][3] =
      _mm256_sub_pd(bezout_matrix->matrix[0][0][3],
                    _mm256_mul_pd(poly2.coeffs[0][2], poly1.coeffs[1][1]));
  bezout_matrix->matrix[0][0][4] =
      _mm256_sub_pd(bezout_matrix->matrix[0][0][4],
                    _mm256_mul_pd(poly2.coeffs[0][2], poly1.coeffs[1][2]));
  bezout_matrix->matrix[0][0][5] =
      _mm256_sub_pd(bezout_matrix->matrix[0][0][5],
                    _mm256_mul_pd(poly2.coeffs[0][2], poly1.coeffs[1][3]));
}

template <>
void reflection_unipoly_cross<0ul, 1ul>(
    const BVP<5UL> &poly1, const BVP<3UL> &poly2,
    UnivariatePolyMatrix<4UL> *bezout_matrix) {
  bezout_matrix->matrix[0][1][0] =
      _mm256_add_pd(bezout_matrix->matrix[0][1][0],
                    _mm256_mul_pd(poly1.coeffs[0][0], poly2.coeffs[2][0]));
  bezout_matrix->matrix[0][1][1] =
      _mm256_add_pd(bezout_matrix->matrix[0][1][1],
                    _mm256_mul_pd(poly1.coeffs[0][1], poly2.coeffs[2][0]));
  bezout_matrix->matrix[0][1][2] =
      _mm256_add_pd(bezout_matrix->matrix[0][1][2],
                    _mm256_mul_pd(poly1.coeffs[0][2], poly2.coeffs[2][0]));
  bezout_matrix->matrix[0][1][3] =
      _mm256_add_pd(bezout_matrix->matrix[0][1][3],
                    _mm256_mul_pd(poly1.coeffs[0][3], poly2.coeffs[2][0]));
  bezout_matrix->matrix[0][1][4] =
      _mm256_add_pd(bezout_matrix->matrix[0][1][4],
                    _mm256_mul_pd(poly1.coeffs[0][4], poly2.coeffs[2][0]));
  bezout_matrix->matrix[0][1][0] =
      _mm256_sub_pd(bezout_matrix->matrix[0][1][0],
                    _mm256_mul_pd(poly2.coeffs[0][0], poly1.coeffs[2][0]));
  bezout_matrix->matrix[0][1][1] =
      _mm256_sub_pd(bezout_matrix->matrix[0][1][1],
                    _mm256_mul_pd(poly2.coeffs[0][0], poly1.coeffs[2][1]));
  bezout_matrix->matrix[0][1][2] =
      _mm256_sub_pd(bezout_matrix->matrix[0][1][2],
                    _mm256_mul_pd(poly2.coeffs[0][0], poly1.coeffs[2][2]));
  bezout_matrix->matrix[0][1][1] =
      _mm256_sub_pd(bezout_matrix->matrix[0][1][1],
                    _mm256_mul_pd(poly2.coeffs[0][1], poly1.coeffs[2][0]));
  bezout_matrix->matrix[0][1][2] =
      _mm256_sub_pd(bezout_matrix->matrix[0][1][2],
                    _mm256_mul_pd(poly2.coeffs[0][1], poly1.coeffs[2][1]));
  bezout_matrix->matrix[0][1][3] =
      _mm256_sub_pd(bezout_matrix->matrix[0][1][3],
                    _mm256_mul_pd(poly2.coeffs[0][1], poly1.coeffs[2][2]));
  bezout_matrix->matrix[0][1][2] =
      _mm256_sub_pd(bezout_matrix->matrix[0][1][2],
                    _mm256_mul_pd(poly2.coeffs[0][2], poly1.coeffs[2][0]));
  bezout_matrix->matrix[0][1][3] =
      _mm256_sub_pd(bezout_matrix->matrix[0][1][3],
                    _mm256_mul_pd(poly2.coeffs[0][2], poly1.coeffs[2][1]));
  bezout_matrix->matrix[0][1][4] =
      _mm256_sub_pd(bezout_matrix->matrix[0][1][4],
                    _mm256_mul_pd(poly2.coeffs[0][2], poly1.coeffs[2][2]));
}

template <>
void reflection_unipoly_cross<0ul, 2ul>(
    const BVP<5UL> &poly1, const BVP<3UL> &poly2,
    UnivariatePolyMatrix<4UL> *bezout_matrix) {
  bezout_matrix->matrix[0][2][0] =
      _mm256_sub_pd(bezout_matrix->matrix[0][2][0],
                    _mm256_mul_pd(poly2.coeffs[0][0], poly1.coeffs[3][0]));
  bezout_matrix->matrix[0][2][1] =
      _mm256_sub_pd(bezout_matrix->matrix[0][2][1],
                    _mm256_mul_pd(poly2.coeffs[0][0], poly1.coeffs[3][1]));
  bezout_matrix->matrix[0][2][1] =
      _mm256_sub_pd(bezout_matrix->matrix[0][2][1],
                    _mm256_mul_pd(poly2.coeffs[0][1], poly1.coeffs[3][0]));
  bezout_matrix->matrix[0][2][2] =
      _mm256_sub_pd(bezout_matrix->matrix[0][2][2],
                    _mm256_mul_pd(poly2.coeffs[0][1], poly1.coeffs[3][1]));
  bezout_matrix->matrix[0][2][2] =
      _mm256_sub_pd(bezout_matrix->matrix[0][2][2],
                    _mm256_mul_pd(poly2.coeffs[0][2], poly1.coeffs[3][0]));
  bezout_matrix->matrix[0][2][3] =
      _mm256_sub_pd(bezout_matrix->matrix[0][2][3],
                    _mm256_mul_pd(poly2.coeffs[0][2], poly1.coeffs[3][1]));
}

template <>
void reflection_unipoly_cross<0ul, 3ul>(
    const BVP<5UL> &poly1, const BVP<3UL> &poly2,
    UnivariatePolyMatrix<4UL> *bezout_matrix) {
  bezout_matrix->matrix[0][3][0] =
      _mm256_sub_pd(bezout_matrix->matrix[0][3][0],
                    _mm256_mul_pd(poly2.coeffs[0][0], poly1.coeffs[4][0]));
  bezout_matrix->matrix[0][3][1] =
      _mm256_sub_pd(bezout_matrix->matrix[0][3][1],
                    _mm256_mul_pd(poly2.coeffs[0][1], poly1.coeffs[4][0]));
  bezout_matrix->matrix[0][3][2] =
      _mm256_sub_pd(bezout_matrix->matrix[0][3][2],
                    _mm256_mul_pd(poly2.coeffs[0][2], poly1.coeffs[4][0]));
}

template <>
void reflection_unipoly_cross<1ul, 1ul>(
    const BVP<5UL> &poly1, const BVP<3UL> &poly2,
    UnivariatePolyMatrix<4UL> *bezout_matrix) {

  bezout_matrix->matrix[1][1][0] =
      _mm256_add_pd(bezout_matrix->matrix[1][1][0],
                    _mm256_mul_pd(poly1.coeffs[1][0], poly2.coeffs[2][0]));
  bezout_matrix->matrix[1][1][1] =
      _mm256_add_pd(bezout_matrix->matrix[1][1][1],
                    _mm256_mul_pd(poly1.coeffs[1][1], poly2.coeffs[2][0]));
  bezout_matrix->matrix[1][1][2] =
      _mm256_add_pd(bezout_matrix->matrix[1][1][2],
                    _mm256_mul_pd(poly1.coeffs[1][2], poly2.coeffs[2][0]));
  bezout_matrix->matrix[1][1][3] =
      _mm256_add_pd(bezout_matrix->matrix[1][1][3],
                    _mm256_mul_pd(poly1.coeffs[1][3], poly2.coeffs[2][0]));
  bezout_matrix->matrix[1][1][0] =
      _mm256_sub_pd(bezout_matrix->matrix[1][1][0],
                    _mm256_mul_pd(poly2.coeffs[1][0], poly1.coeffs[2][0]));
  bezout_matrix->matrix[1][1][1] =
      _mm256_sub_pd(bezout_matrix->matrix[1][1][1],
                    _mm256_mul_pd(poly2.coeffs[1][0], poly1.coeffs[2][1]));
  bezout_matrix->matrix[1][1][2] =
      _mm256_sub_pd(bezout_matrix->matrix[1][1][2],
                    _mm256_mul_pd(poly2.coeffs[1][0], poly1.coeffs[2][2]));
  bezout_matrix->matrix[1][1][1] =
      _mm256_sub_pd(bezout_matrix->matrix[1][1][1],
                    _mm256_mul_pd(poly2.coeffs[1][1], poly1.coeffs[2][0]));
  bezout_matrix->matrix[1][1][2] =
      _mm256_sub_pd(bezout_matrix->matrix[1][1][2],
                    _mm256_mul_pd(poly2.coeffs[1][1], poly1.coeffs[2][1]));
  bezout_matrix->matrix[1][1][3] =
      _mm256_sub_pd(bezout_matrix->matrix[1][1][3],
                    _mm256_mul_pd(poly2.coeffs[1][1], poly1.coeffs[2][2]));
}

template <>
void reflection_unipoly_cross<1ul, 2ul>(
    const BVP<5UL> &poly1, const BVP<3UL> &poly2,
    UnivariatePolyMatrix<4UL> *bezout_matrix) {

  bezout_matrix->matrix[1][2][0] =
      _mm256_sub_pd(bezout_matrix->matrix[1][2][0],
                    _mm256_mul_pd(poly2.coeffs[1][0], poly1.coeffs[3][0]));
  bezout_matrix->matrix[1][2][1] =
      _mm256_sub_pd(bezout_matrix->matrix[1][2][1],
                    _mm256_mul_pd(poly2.coeffs[1][0], poly1.coeffs[3][1]));
  bezout_matrix->matrix[1][2][1] =
      _mm256_sub_pd(bezout_matrix->matrix[1][2][1],
                    _mm256_mul_pd(poly2.coeffs[1][1], poly1.coeffs[3][0]));
  bezout_matrix->matrix[1][2][2] =
      _mm256_sub_pd(bezout_matrix->matrix[1][2][2],
                    _mm256_mul_pd(poly2.coeffs[1][1], poly1.coeffs[3][1]));
}

template <>
void reflection_unipoly_cross<1ul, 3ul>(
    const BVP<5UL> &poly1, const BVP<3UL> &poly2,
    UnivariatePolyMatrix<4UL> *bezout_matrix) {
  bezout_matrix->matrix[1][3][0] =
      _mm256_sub_pd(bezout_matrix->matrix[1][3][0],
                    _mm256_mul_pd(poly2.coeffs[1][0], poly1.coeffs[4][0]));
  bezout_matrix->matrix[1][3][1] =
      _mm256_sub_pd(bezout_matrix->matrix[1][3][1],
                    _mm256_mul_pd(poly2.coeffs[1][1], poly1.coeffs[4][0]));
}

template <>
void reflection_unipoly_cross<2ul, 2ul>(
    const BVP<5UL> &poly1, const BVP<3UL> &poly2,
    UnivariatePolyMatrix<4UL> *bezout_matrix) {
  bezout_matrix->matrix[2][2][0] =
      _mm256_sub_pd(bezout_matrix->matrix[2][2][0],
                    _mm256_mul_pd(poly2.coeffs[2][0], poly1.coeffs[3][0]));
  bezout_matrix->matrix[2][2][1] =
      _mm256_sub_pd(bezout_matrix->matrix[2][2][1],
                    _mm256_mul_pd(poly2.coeffs[2][0], poly1.coeffs[3][1]));
}

template <>
void reflection_unipoly_cross<2ul, 3ul>(
    const BVP<5UL> &poly1, const BVP<3UL> &poly2,
    UnivariatePolyMatrix<4UL> *bezout_matrix) {
  bezout_matrix->matrix[2][3][0] =
      _mm256_sub_pd(bezout_matrix->matrix[2][3][0],
                    _mm256_mul_pd(poly2.coeffs[2][0], poly1.coeffs[4][0]));
}

/**
 * The two polys with degree 4 and 2, thus the bezout matrix is 4x4,
 * see the paper
 */
template <>
UnivariatePolyMatrix<4ul> bezout_matrix<5ul, 3ul, 4ul>(const BVP<5ul> &poly1,
                                                       const BVP<3ul> &poly2) {

  UnivariatePolyMatrix<4ul> matrix;

  // f[0][0] = a[0] * b[1] - b[0] * a[1];
  reflection_unipoly_cross<0ul, 0ul>(poly1, poly2, &matrix);

  // f[0][1] = a[0] * b[2] - b[0] * a[2];
  reflection_unipoly_cross<0ul, 1ul>(poly1, poly2, &matrix);

  // f[0][2] = -b[0] * a[3];
  reflection_unipoly_cross<0ul, 2ul>(poly1, poly2, &matrix);

  // f[0][3] = -b[0] * a[4];
  reflection_unipoly_cross<0ul, 3ul>(poly1, poly2, &matrix);

  // f[1][1] = a[1] * b[2] - b[1] * a[2];
  reflection_unipoly_cross<1ul, 1ul>(poly1, poly2, &matrix);

  // f[1][2] = -b[1] * a[3];
  reflection_unipoly_cross<1ul, 2ul>(poly1, poly2, &matrix);

  // f[1][3] = -b[1] * a[4];
  reflection_unipoly_cross<1ul, 3ul>(poly1, poly2, &matrix);

  // f[2][2] = -b[2] * a[3];
  reflection_unipoly_cross<2ul, 2ul>(poly1, poly2, &matrix);

  // f[2][3] = -b[2] * a[4];
  reflection_unipoly_cross<2ul, 3ul>(poly1, poly2, &matrix);

  // f[1][1] = f[1][1] + f[0][2];
  matrix.matrix[1][1][0] =
      _mm256_add_pd(matrix.matrix[1][1][0], matrix.matrix[0][2][0]);
  matrix.matrix[1][1][1] =
      _mm256_add_pd(matrix.matrix[1][1][1], matrix.matrix[0][2][1]);
  matrix.matrix[1][1][2] =
      _mm256_add_pd(matrix.matrix[1][1][2], matrix.matrix[0][2][2]);
  matrix.matrix[1][1][3] =
      _mm256_add_pd(matrix.matrix[1][1][3], matrix.matrix[0][2][3]);

  // f[1][2] = f[1][2] + f[0][3];
  matrix.matrix[1][2][0] =
      _mm256_add_pd(matrix.matrix[1][2][0], matrix.matrix[0][3][0]);
  matrix.matrix[1][2][1] =
      _mm256_add_pd(matrix.matrix[1][2][1], matrix.matrix[0][3][1]);
  matrix.matrix[1][2][2] =
      _mm256_add_pd(matrix.matrix[1][2][2], matrix.matrix[0][3][2]);

  // f[2][2] = f[2][2] + f[1][3];
  matrix.matrix[2][2][0] =
      _mm256_add_pd(matrix.matrix[2][2][0], matrix.matrix[1][3][0]);
  matrix.matrix[2][2][1] =
      _mm256_add_pd(matrix.matrix[2][2][1], matrix.matrix[1][3][1]);

  // f[1][0] = f[0][1];
  matrix.matrix[1][0][0] = matrix.matrix[0][1][0];
  matrix.matrix[1][0][1] = matrix.matrix[0][1][1];
  matrix.matrix[1][0][2] = matrix.matrix[0][1][2];
  matrix.matrix[1][0][3] = matrix.matrix[0][1][3];
  matrix.matrix[1][0][4] = matrix.matrix[0][1][4];

  // f[2][0] = f[0][2];
  matrix.matrix[2][0][0] = matrix.matrix[0][2][0];
  matrix.matrix[2][0][1] = matrix.matrix[0][2][1];
  matrix.matrix[2][0][2] = matrix.matrix[0][2][2];
  matrix.matrix[2][0][3] = matrix.matrix[0][2][3];

  // f[2][1] = f[1][2];
  matrix.matrix[2][1][0] = matrix.matrix[1][2][0];
  matrix.matrix[2][1][1] = matrix.matrix[1][2][1];
  matrix.matrix[2][1][2] = matrix.matrix[1][2][2];

  // f[3][0] = f[0][3];
  matrix.matrix[3][0][0] = matrix.matrix[0][3][0];
  matrix.matrix[3][0][1] = matrix.matrix[0][3][1];
  matrix.matrix[3][0][2] = matrix.matrix[0][3][2];

  // f[3][1] = f[1][3];
  matrix.matrix[3][1][0] = matrix.matrix[1][3][0];
  matrix.matrix[3][1][1] = matrix.matrix[1][3][1];

  // f[3][2] = f[2][3];
  matrix.matrix[3][2][0] = matrix.matrix[2][3][0];

  return matrix;
}

/**
 * Bezout matrix in single refraction case, the matrix is 6x6,
 * see the paper
 */
template<>
UnivariatePolyMatrix<6ul> bezout_matrix<7ul, 3ul, 6ul>(const BVP<7ul> &poly1, const BVP<3ul> &poly2) {
  UnivariatePolyMatrix<6ul> matrix;

  matrix.matrix[0][0][0] =
      _mm256_add_pd(matrix.matrix[0][0][0],
                    _mm256_mul_pd(poly1.coeffs[0][0], poly2.coeffs[1][0]));
  matrix.matrix[0][0][1] =
      _mm256_add_pd(matrix.matrix[0][0][1],
                    _mm256_mul_pd(poly1.coeffs[0][0], poly2.coeffs[1][1]));
  matrix.matrix[0][0][1] =
      _mm256_add_pd(matrix.matrix[0][0][1],
                    _mm256_mul_pd(poly1.coeffs[0][1], poly2.coeffs[1][0]));
  matrix.matrix[0][0][2] =
      _mm256_add_pd(matrix.matrix[0][0][2],
                    _mm256_mul_pd(poly1.coeffs[0][1], poly2.coeffs[1][1]));
  matrix.matrix[0][0][2] =
      _mm256_add_pd(matrix.matrix[0][0][2],
                    _mm256_mul_pd(poly1.coeffs[0][2], poly2.coeffs[1][0]));
  matrix.matrix[0][0][3] =
      _mm256_add_pd(matrix.matrix[0][0][3],
                    _mm256_mul_pd(poly1.coeffs[0][2], poly2.coeffs[1][1]));
  matrix.matrix[0][0][3] =
      _mm256_add_pd(matrix.matrix[0][0][3],
                    _mm256_mul_pd(poly1.coeffs[0][3], poly2.coeffs[1][0]));
  matrix.matrix[0][0][4] =
      _mm256_add_pd(matrix.matrix[0][0][4],
                    _mm256_mul_pd(poly1.coeffs[0][3], poly2.coeffs[1][1]));
  matrix.matrix[0][0][4] =
      _mm256_add_pd(matrix.matrix[0][0][4],
                    _mm256_mul_pd(poly1.coeffs[0][4], poly2.coeffs[1][0]));
  matrix.matrix[0][0][5] =
      _mm256_add_pd(matrix.matrix[0][0][5],
                    _mm256_mul_pd(poly1.coeffs[0][4], poly2.coeffs[1][1]));
  matrix.matrix[0][0][5] =
      _mm256_add_pd(matrix.matrix[0][0][5],
                    _mm256_mul_pd(poly1.coeffs[0][5], poly2.coeffs[1][0]));
  matrix.matrix[0][0][6] =
      _mm256_add_pd(matrix.matrix[0][0][6],
                    _mm256_mul_pd(poly1.coeffs[0][5], poly2.coeffs[1][1]));
  matrix.matrix[0][0][6] =
      _mm256_add_pd(matrix.matrix[0][0][6],
                    _mm256_mul_pd(poly1.coeffs[0][6], poly2.coeffs[1][0]));
  matrix.matrix[0][0][7] =
      _mm256_add_pd(matrix.matrix[0][0][7],
                    _mm256_mul_pd(poly1.coeffs[0][6], poly2.coeffs[1][1]));
  matrix.matrix[0][0][0] =
      _mm256_sub_pd(matrix.matrix[0][0][0],
                    _mm256_mul_pd(poly1.coeffs[1][0], poly2.coeffs[0][0]));
  matrix.matrix[0][0][1] =
      _mm256_sub_pd(matrix.matrix[0][0][1],
                    _mm256_mul_pd(poly1.coeffs[1][1], poly2.coeffs[0][0]));
  matrix.matrix[0][0][2] =
      _mm256_sub_pd(matrix.matrix[0][0][2],
                    _mm256_mul_pd(poly1.coeffs[1][2], poly2.coeffs[0][0]));
  matrix.matrix[0][0][3] =
      _mm256_sub_pd(matrix.matrix[0][0][3],
                    _mm256_mul_pd(poly1.coeffs[1][3], poly2.coeffs[0][0]));
  matrix.matrix[0][0][4] =
      _mm256_sub_pd(matrix.matrix[0][0][4],
                    _mm256_mul_pd(poly1.coeffs[1][4], poly2.coeffs[0][0]));
  matrix.matrix[0][0][5] =
      _mm256_sub_pd(matrix.matrix[0][0][5],
                    _mm256_mul_pd(poly1.coeffs[1][5], poly2.coeffs[0][0]));
  matrix.matrix[0][0][1] =
      _mm256_sub_pd(matrix.matrix[0][0][1],
                    _mm256_mul_pd(poly1.coeffs[1][0], poly2.coeffs[0][1]));
  matrix.matrix[0][0][2] =
      _mm256_sub_pd(matrix.matrix[0][0][2],
                    _mm256_mul_pd(poly1.coeffs[1][1], poly2.coeffs[0][1]));
  matrix.matrix[0][0][3] =
      _mm256_sub_pd(matrix.matrix[0][0][3],
                    _mm256_mul_pd(poly1.coeffs[1][2], poly2.coeffs[0][1]));
  matrix.matrix[0][0][4] =
      _mm256_sub_pd(matrix.matrix[0][0][4],
                    _mm256_mul_pd(poly1.coeffs[1][3], poly2.coeffs[0][1]));
  matrix.matrix[0][0][5] =
      _mm256_sub_pd(matrix.matrix[0][0][5],
                    _mm256_mul_pd(poly1.coeffs[1][4], poly2.coeffs[0][1]));
  matrix.matrix[0][0][6] =
      _mm256_sub_pd(matrix.matrix[0][0][6],
                    _mm256_mul_pd(poly1.coeffs[1][5], poly2.coeffs[0][1]));
  matrix.matrix[0][0][2] =
      _mm256_sub_pd(matrix.matrix[0][0][2],
                    _mm256_mul_pd(poly1.coeffs[1][0], poly2.coeffs[0][2]));
  matrix.matrix[0][0][3] =
      _mm256_sub_pd(matrix.matrix[0][0][3],
                    _mm256_mul_pd(poly1.coeffs[1][1], poly2.coeffs[0][2]));
  matrix.matrix[0][0][4] =
      _mm256_sub_pd(matrix.matrix[0][0][4],
                    _mm256_mul_pd(poly1.coeffs[1][2], poly2.coeffs[0][2]));
  matrix.matrix[0][0][5] =
      _mm256_sub_pd(matrix.matrix[0][0][5],
                    _mm256_mul_pd(poly1.coeffs[1][3], poly2.coeffs[0][2]));
  matrix.matrix[0][0][6] =
      _mm256_sub_pd(matrix.matrix[0][0][6],
                    _mm256_mul_pd(poly1.coeffs[1][4], poly2.coeffs[0][2]));
  matrix.matrix[0][0][7] =
      _mm256_sub_pd(matrix.matrix[0][0][7],
                    _mm256_mul_pd(poly1.coeffs[1][5], poly2.coeffs[0][2]));
  matrix.matrix[0][1][0] =
      _mm256_add_pd(matrix.matrix[0][1][0],
                    _mm256_mul_pd(poly1.coeffs[0][0], poly2.coeffs[2][0]));
  matrix.matrix[0][1][1] =
      _mm256_add_pd(matrix.matrix[0][1][1],
                    _mm256_mul_pd(poly1.coeffs[0][1], poly2.coeffs[2][0]));
  matrix.matrix[0][1][2] =
      _mm256_add_pd(matrix.matrix[0][1][2],
                    _mm256_mul_pd(poly1.coeffs[0][2], poly2.coeffs[2][0]));
  matrix.matrix[0][1][3] =
      _mm256_add_pd(matrix.matrix[0][1][3],
                    _mm256_mul_pd(poly1.coeffs[0][3], poly2.coeffs[2][0]));
  matrix.matrix[0][1][4] =
      _mm256_add_pd(matrix.matrix[0][1][4],
                    _mm256_mul_pd(poly1.coeffs[0][4], poly2.coeffs[2][0]));
  matrix.matrix[0][1][5] =
      _mm256_add_pd(matrix.matrix[0][1][5],
                    _mm256_mul_pd(poly1.coeffs[0][5], poly2.coeffs[2][0]));
  matrix.matrix[0][1][6] =
      _mm256_add_pd(matrix.matrix[0][1][6],
                    _mm256_mul_pd(poly1.coeffs[0][6], poly2.coeffs[2][0]));
  matrix.matrix[0][1][0] =
      _mm256_sub_pd(matrix.matrix[0][1][0],
                    _mm256_mul_pd(poly1.coeffs[2][0], poly2.coeffs[0][0]));
  matrix.matrix[0][1][1] =
      _mm256_sub_pd(matrix.matrix[0][1][1],
                    _mm256_mul_pd(poly1.coeffs[2][1], poly2.coeffs[0][0]));
  matrix.matrix[0][1][2] =
      _mm256_sub_pd(matrix.matrix[0][1][2],
                    _mm256_mul_pd(poly1.coeffs[2][2], poly2.coeffs[0][0]));
  matrix.matrix[0][1][3] =
      _mm256_sub_pd(matrix.matrix[0][1][3],
                    _mm256_mul_pd(poly1.coeffs[2][3], poly2.coeffs[0][0]));
  matrix.matrix[0][1][4] =
      _mm256_sub_pd(matrix.matrix[0][1][4],
                    _mm256_mul_pd(poly1.coeffs[2][4], poly2.coeffs[0][0]));
  matrix.matrix[0][1][1] =
      _mm256_sub_pd(matrix.matrix[0][1][1],
                    _mm256_mul_pd(poly1.coeffs[2][0], poly2.coeffs[0][1]));
  matrix.matrix[0][1][2] =
      _mm256_sub_pd(matrix.matrix[0][1][2],
                    _mm256_mul_pd(poly1.coeffs[2][1], poly2.coeffs[0][1]));
  matrix.matrix[0][1][3] =
      _mm256_sub_pd(matrix.matrix[0][1][3],
                    _mm256_mul_pd(poly1.coeffs[2][2], poly2.coeffs[0][1]));
  matrix.matrix[0][1][4] =
      _mm256_sub_pd(matrix.matrix[0][1][4],
                    _mm256_mul_pd(poly1.coeffs[2][3], poly2.coeffs[0][1]));
  matrix.matrix[0][1][5] =
      _mm256_sub_pd(matrix.matrix[0][1][5],
                    _mm256_mul_pd(poly1.coeffs[2][4], poly2.coeffs[0][1]));
  matrix.matrix[0][1][2] =
      _mm256_sub_pd(matrix.matrix[0][1][2],
                    _mm256_mul_pd(poly1.coeffs[2][0], poly2.coeffs[0][2]));
  matrix.matrix[0][1][3] =
      _mm256_sub_pd(matrix.matrix[0][1][3],
                    _mm256_mul_pd(poly1.coeffs[2][1], poly2.coeffs[0][2]));
  matrix.matrix[0][1][4] =
      _mm256_sub_pd(matrix.matrix[0][1][4],
                    _mm256_mul_pd(poly1.coeffs[2][2], poly2.coeffs[0][2]));
  matrix.matrix[0][1][5] =
      _mm256_sub_pd(matrix.matrix[0][1][5],
                    _mm256_mul_pd(poly1.coeffs[2][3], poly2.coeffs[0][2]));
  matrix.matrix[0][1][6] =
      _mm256_sub_pd(matrix.matrix[0][1][6],
                    _mm256_mul_pd(poly1.coeffs[2][4], poly2.coeffs[0][2]));
  matrix.matrix[0][2][0] =
      _mm256_sub_pd(matrix.matrix[0][2][0],
                    _mm256_mul_pd(poly1.coeffs[3][0], poly2.coeffs[0][0]));
  matrix.matrix[0][2][1] =
      _mm256_sub_pd(matrix.matrix[0][2][1],
                    _mm256_mul_pd(poly1.coeffs[3][1], poly2.coeffs[0][0]));
  matrix.matrix[0][2][2] =
      _mm256_sub_pd(matrix.matrix[0][2][2],
                    _mm256_mul_pd(poly1.coeffs[3][2], poly2.coeffs[0][0]));
  matrix.matrix[0][2][3] =
      _mm256_sub_pd(matrix.matrix[0][2][3],
                    _mm256_mul_pd(poly1.coeffs[3][3], poly2.coeffs[0][0]));
  matrix.matrix[0][2][1] =
      _mm256_sub_pd(matrix.matrix[0][2][1],
                    _mm256_mul_pd(poly1.coeffs[3][0], poly2.coeffs[0][1]));
  matrix.matrix[0][2][2] =
      _mm256_sub_pd(matrix.matrix[0][2][2],
                    _mm256_mul_pd(poly1.coeffs[3][1], poly2.coeffs[0][1]));
  matrix.matrix[0][2][3] =
      _mm256_sub_pd(matrix.matrix[0][2][3],
                    _mm256_mul_pd(poly1.coeffs[3][2], poly2.coeffs[0][1]));
  matrix.matrix[0][2][4] =
      _mm256_sub_pd(matrix.matrix[0][2][4],
                    _mm256_mul_pd(poly1.coeffs[3][3], poly2.coeffs[0][1]));
  matrix.matrix[0][2][2] =
      _mm256_sub_pd(matrix.matrix[0][2][2],
                    _mm256_mul_pd(poly1.coeffs[3][0], poly2.coeffs[0][2]));
  matrix.matrix[0][2][3] =
      _mm256_sub_pd(matrix.matrix[0][2][3],
                    _mm256_mul_pd(poly1.coeffs[3][1], poly2.coeffs[0][2]));
  matrix.matrix[0][2][4] =
      _mm256_sub_pd(matrix.matrix[0][2][4],
                    _mm256_mul_pd(poly1.coeffs[3][2], poly2.coeffs[0][2]));
  matrix.matrix[0][2][5] =
      _mm256_sub_pd(matrix.matrix[0][2][5],
                    _mm256_mul_pd(poly1.coeffs[3][3], poly2.coeffs[0][2]));
  matrix.matrix[0][3][0] =
      _mm256_sub_pd(matrix.matrix[0][3][0],
                    _mm256_mul_pd(poly1.coeffs[4][0], poly2.coeffs[0][0]));
  matrix.matrix[0][3][1] =
      _mm256_sub_pd(matrix.matrix[0][3][1],
                    _mm256_mul_pd(poly1.coeffs[4][1], poly2.coeffs[0][0]));
  matrix.matrix[0][3][2] =
      _mm256_sub_pd(matrix.matrix[0][3][2],
                    _mm256_mul_pd(poly1.coeffs[4][2], poly2.coeffs[0][0]));
  matrix.matrix[0][3][1] =
      _mm256_sub_pd(matrix.matrix[0][3][1],
                    _mm256_mul_pd(poly1.coeffs[4][0], poly2.coeffs[0][1]));
  matrix.matrix[0][3][2] =
      _mm256_sub_pd(matrix.matrix[0][3][2],
                    _mm256_mul_pd(poly1.coeffs[4][1], poly2.coeffs[0][1]));
  matrix.matrix[0][3][3] =
      _mm256_sub_pd(matrix.matrix[0][3][3],
                    _mm256_mul_pd(poly1.coeffs[4][2], poly2.coeffs[0][1]));
  matrix.matrix[0][3][2] =
      _mm256_sub_pd(matrix.matrix[0][3][2],
                    _mm256_mul_pd(poly1.coeffs[4][0], poly2.coeffs[0][2]));
  matrix.matrix[0][3][3] =
      _mm256_sub_pd(matrix.matrix[0][3][3],
                    _mm256_mul_pd(poly1.coeffs[4][1], poly2.coeffs[0][2]));
  matrix.matrix[0][3][4] =
      _mm256_sub_pd(matrix.matrix[0][3][4],
                    _mm256_mul_pd(poly1.coeffs[4][2], poly2.coeffs[0][2]));
  matrix.matrix[0][4][0] =
      _mm256_sub_pd(matrix.matrix[0][4][0],
                    _mm256_mul_pd(poly1.coeffs[5][0], poly2.coeffs[0][0]));
  matrix.matrix[0][4][1] =
      _mm256_sub_pd(matrix.matrix[0][4][1],
                    _mm256_mul_pd(poly1.coeffs[5][1], poly2.coeffs[0][0]));
  matrix.matrix[0][4][1] =
      _mm256_sub_pd(matrix.matrix[0][4][1],
                    _mm256_mul_pd(poly1.coeffs[5][0], poly2.coeffs[0][1]));
  matrix.matrix[0][4][2] =
      _mm256_sub_pd(matrix.matrix[0][4][2],
                    _mm256_mul_pd(poly1.coeffs[5][1], poly2.coeffs[0][1]));
  matrix.matrix[0][4][2] =
      _mm256_sub_pd(matrix.matrix[0][4][2],
                    _mm256_mul_pd(poly1.coeffs[5][0], poly2.coeffs[0][2]));
  matrix.matrix[0][4][3] =
      _mm256_sub_pd(matrix.matrix[0][4][3],
                    _mm256_mul_pd(poly1.coeffs[5][1], poly2.coeffs[0][2]));
  matrix.matrix[0][5][0] =
      _mm256_sub_pd(matrix.matrix[0][5][0],
                    _mm256_mul_pd(poly1.coeffs[6][0], poly2.coeffs[0][0]));
  matrix.matrix[0][5][1] =
      _mm256_sub_pd(matrix.matrix[0][5][1],
                    _mm256_mul_pd(poly1.coeffs[6][0], poly2.coeffs[0][1]));
  matrix.matrix[0][5][2] =
      _mm256_sub_pd(matrix.matrix[0][5][2],
                    _mm256_mul_pd(poly1.coeffs[6][0], poly2.coeffs[0][2]));
  matrix.matrix[1][1][0] =
      _mm256_add_pd(matrix.matrix[1][1][0],
                    _mm256_mul_pd(poly1.coeffs[1][0], poly2.coeffs[2][0]));
  matrix.matrix[1][1][1] =
      _mm256_add_pd(matrix.matrix[1][1][1],
                    _mm256_mul_pd(poly1.coeffs[1][1], poly2.coeffs[2][0]));
  matrix.matrix[1][1][2] =
      _mm256_add_pd(matrix.matrix[1][1][2],
                    _mm256_mul_pd(poly1.coeffs[1][2], poly2.coeffs[2][0]));
  matrix.matrix[1][1][3] =
      _mm256_add_pd(matrix.matrix[1][1][3],
                    _mm256_mul_pd(poly1.coeffs[1][3], poly2.coeffs[2][0]));
  matrix.matrix[1][1][4] =
      _mm256_add_pd(matrix.matrix[1][1][4],
                    _mm256_mul_pd(poly1.coeffs[1][4], poly2.coeffs[2][0]));
  matrix.matrix[1][1][5] =
      _mm256_add_pd(matrix.matrix[1][1][5],
                    _mm256_mul_pd(poly1.coeffs[1][5], poly2.coeffs[2][0]));
  matrix.matrix[1][1][0] =
      _mm256_sub_pd(matrix.matrix[1][1][0],
                    _mm256_mul_pd(poly1.coeffs[2][0], poly2.coeffs[1][0]));
  matrix.matrix[1][1][1] =
      _mm256_sub_pd(matrix.matrix[1][1][1],
                    _mm256_mul_pd(poly1.coeffs[2][1], poly2.coeffs[1][0]));
  matrix.matrix[1][1][2] =
      _mm256_sub_pd(matrix.matrix[1][1][2],
                    _mm256_mul_pd(poly1.coeffs[2][2], poly2.coeffs[1][0]));
  matrix.matrix[1][1][3] =
      _mm256_sub_pd(matrix.matrix[1][1][3],
                    _mm256_mul_pd(poly1.coeffs[2][3], poly2.coeffs[1][0]));
  matrix.matrix[1][1][4] =
      _mm256_sub_pd(matrix.matrix[1][1][4],
                    _mm256_mul_pd(poly1.coeffs[2][4], poly2.coeffs[1][0]));
  matrix.matrix[1][1][1] =
      _mm256_sub_pd(matrix.matrix[1][1][1],
                    _mm256_mul_pd(poly1.coeffs[2][0], poly2.coeffs[1][1]));
  matrix.matrix[1][1][2] =
      _mm256_sub_pd(matrix.matrix[1][1][2],
                    _mm256_mul_pd(poly1.coeffs[2][1], poly2.coeffs[1][1]));
  matrix.matrix[1][1][3] =
      _mm256_sub_pd(matrix.matrix[1][1][3],
                    _mm256_mul_pd(poly1.coeffs[2][2], poly2.coeffs[1][1]));
  matrix.matrix[1][1][4] =
      _mm256_sub_pd(matrix.matrix[1][1][4],
                    _mm256_mul_pd(poly1.coeffs[2][3], poly2.coeffs[1][1]));
  matrix.matrix[1][1][5] =
      _mm256_sub_pd(matrix.matrix[1][1][5],
                    _mm256_mul_pd(poly1.coeffs[2][4], poly2.coeffs[1][1]));
  matrix.matrix[1][2][0] =
      _mm256_sub_pd(matrix.matrix[1][2][0],
                    _mm256_mul_pd(poly1.coeffs[3][0], poly2.coeffs[1][0]));
  matrix.matrix[1][2][1] =
      _mm256_sub_pd(matrix.matrix[1][2][1],
                    _mm256_mul_pd(poly1.coeffs[3][1], poly2.coeffs[1][0]));
  matrix.matrix[1][2][2] =
      _mm256_sub_pd(matrix.matrix[1][2][2],
                    _mm256_mul_pd(poly1.coeffs[3][2], poly2.coeffs[1][0]));
  matrix.matrix[1][2][3] =
      _mm256_sub_pd(matrix.matrix[1][2][3],
                    _mm256_mul_pd(poly1.coeffs[3][3], poly2.coeffs[1][0]));
  matrix.matrix[1][2][1] =
      _mm256_sub_pd(matrix.matrix[1][2][1],
                    _mm256_mul_pd(poly1.coeffs[3][0], poly2.coeffs[1][1]));
  matrix.matrix[1][2][2] =
      _mm256_sub_pd(matrix.matrix[1][2][2],
                    _mm256_mul_pd(poly1.coeffs[3][1], poly2.coeffs[1][1]));
  matrix.matrix[1][2][3] =
      _mm256_sub_pd(matrix.matrix[1][2][3],
                    _mm256_mul_pd(poly1.coeffs[3][2], poly2.coeffs[1][1]));
  matrix.matrix[1][2][4] =
      _mm256_sub_pd(matrix.matrix[1][2][4],
                    _mm256_mul_pd(poly1.coeffs[3][3], poly2.coeffs[1][1]));
  matrix.matrix[1][3][0] =
      _mm256_sub_pd(matrix.matrix[1][3][0],
                    _mm256_mul_pd(poly1.coeffs[4][0], poly2.coeffs[1][0]));
  matrix.matrix[1][3][1] =
      _mm256_sub_pd(matrix.matrix[1][3][1],
                    _mm256_mul_pd(poly1.coeffs[4][1], poly2.coeffs[1][0]));
  matrix.matrix[1][3][2] =
      _mm256_sub_pd(matrix.matrix[1][3][2],
                    _mm256_mul_pd(poly1.coeffs[4][2], poly2.coeffs[1][0]));
  matrix.matrix[1][3][1] =
      _mm256_sub_pd(matrix.matrix[1][3][1],
                    _mm256_mul_pd(poly1.coeffs[4][0], poly2.coeffs[1][1]));
  matrix.matrix[1][3][2] =
      _mm256_sub_pd(matrix.matrix[1][3][2],
                    _mm256_mul_pd(poly1.coeffs[4][1], poly2.coeffs[1][1]));
  matrix.matrix[1][3][3] =
      _mm256_sub_pd(matrix.matrix[1][3][3],
                    _mm256_mul_pd(poly1.coeffs[4][2], poly2.coeffs[1][1]));
  matrix.matrix[1][4][0] =
      _mm256_sub_pd(matrix.matrix[1][4][0],
                    _mm256_mul_pd(poly1.coeffs[5][0], poly2.coeffs[1][0]));
  matrix.matrix[1][4][1] =
      _mm256_sub_pd(matrix.matrix[1][4][1],
                    _mm256_mul_pd(poly1.coeffs[5][1], poly2.coeffs[1][0]));
  matrix.matrix[1][4][1] =
      _mm256_sub_pd(matrix.matrix[1][4][1],
                    _mm256_mul_pd(poly1.coeffs[5][0], poly2.coeffs[1][1]));
  matrix.matrix[1][4][2] =
      _mm256_sub_pd(matrix.matrix[1][4][2],
                    _mm256_mul_pd(poly1.coeffs[5][1], poly2.coeffs[1][1]));
  matrix.matrix[1][5][0] =
      _mm256_sub_pd(matrix.matrix[1][5][0],
                    _mm256_mul_pd(poly1.coeffs[6][0], poly2.coeffs[1][0]));
  matrix.matrix[1][5][1] =
      _mm256_sub_pd(matrix.matrix[1][5][1],
                    _mm256_mul_pd(poly1.coeffs[6][0], poly2.coeffs[1][1]));
  matrix.matrix[2][2][0] =
      _mm256_sub_pd(matrix.matrix[2][2][0],
                    _mm256_mul_pd(poly1.coeffs[3][0], poly2.coeffs[2][0]));
  matrix.matrix[2][2][1] =
      _mm256_sub_pd(matrix.matrix[2][2][1],
                    _mm256_mul_pd(poly1.coeffs[3][1], poly2.coeffs[2][0]));
  matrix.matrix[2][2][2] =
      _mm256_sub_pd(matrix.matrix[2][2][2],
                    _mm256_mul_pd(poly1.coeffs[3][2], poly2.coeffs[2][0]));
  matrix.matrix[2][2][3] =
      _mm256_sub_pd(matrix.matrix[2][2][3],
                    _mm256_mul_pd(poly1.coeffs[3][3], poly2.coeffs[2][0]));
  matrix.matrix[2][3][0] =
      _mm256_sub_pd(matrix.matrix[2][3][0],
                    _mm256_mul_pd(poly1.coeffs[4][0], poly2.coeffs[2][0]));
  matrix.matrix[2][3][1] =
      _mm256_sub_pd(matrix.matrix[2][3][1],
                    _mm256_mul_pd(poly1.coeffs[4][1], poly2.coeffs[2][0]));
  matrix.matrix[2][3][2] =
      _mm256_sub_pd(matrix.matrix[2][3][2],
                    _mm256_mul_pd(poly1.coeffs[4][2], poly2.coeffs[2][0]));
  matrix.matrix[2][4][0] =
      _mm256_sub_pd(matrix.matrix[2][4][0],
                    _mm256_mul_pd(poly1.coeffs[5][0], poly2.coeffs[2][0]));
  matrix.matrix[2][4][1] =
      _mm256_sub_pd(matrix.matrix[2][4][1],
                    _mm256_mul_pd(poly1.coeffs[5][1], poly2.coeffs[2][0]));
  matrix.matrix[2][5][0] =
      _mm256_sub_pd(matrix.matrix[2][5][0],
                    _mm256_mul_pd(poly1.coeffs[6][0], poly2.coeffs[2][0]));
  matrix.matrix[1][1][0] =
      _mm256_add_pd(matrix.matrix[1][1][0], matrix.matrix[0][2][0]);
  matrix.matrix[1][1][1] =
      _mm256_add_pd(matrix.matrix[1][1][1], matrix.matrix[0][2][1]);
  matrix.matrix[1][1][2] =
      _mm256_add_pd(matrix.matrix[1][1][2], matrix.matrix[0][2][2]);
  matrix.matrix[1][1][3] =
      _mm256_add_pd(matrix.matrix[1][1][3], matrix.matrix[0][2][3]);
  matrix.matrix[1][1][4] =
      _mm256_add_pd(matrix.matrix[1][1][4], matrix.matrix[0][2][4]);
  matrix.matrix[1][1][5] =
      _mm256_add_pd(matrix.matrix[1][1][5], matrix.matrix[0][2][5]);
  matrix.matrix[1][1][6] =
      _mm256_add_pd(matrix.matrix[1][1][6], matrix.matrix[0][2][6]);
  matrix.matrix[1][2][0] =
      _mm256_add_pd(matrix.matrix[1][2][0], matrix.matrix[0][3][0]);
  matrix.matrix[1][2][1] =
      _mm256_add_pd(matrix.matrix[1][2][1], matrix.matrix[0][3][1]);
  matrix.matrix[1][2][2] =
      _mm256_add_pd(matrix.matrix[1][2][2], matrix.matrix[0][3][2]);
  matrix.matrix[1][2][3] =
      _mm256_add_pd(matrix.matrix[1][2][3], matrix.matrix[0][3][3]);
  matrix.matrix[1][2][4] =
      _mm256_add_pd(matrix.matrix[1][2][4], matrix.matrix[0][3][4]);
  matrix.matrix[1][2][5] =
      _mm256_add_pd(matrix.matrix[1][2][5], matrix.matrix[0][3][5]);
  matrix.matrix[1][3][0] =
      _mm256_add_pd(matrix.matrix[1][3][0], matrix.matrix[0][4][0]);
  matrix.matrix[1][3][1] =
      _mm256_add_pd(matrix.matrix[1][3][1], matrix.matrix[0][4][1]);
  matrix.matrix[1][3][2] =
      _mm256_add_pd(matrix.matrix[1][3][2], matrix.matrix[0][4][2]);
  matrix.matrix[1][3][3] =
      _mm256_add_pd(matrix.matrix[1][3][3], matrix.matrix[0][4][3]);
  matrix.matrix[1][3][4] =
      _mm256_add_pd(matrix.matrix[1][3][4], matrix.matrix[0][4][4]);
  matrix.matrix[1][4][0] =
      _mm256_add_pd(matrix.matrix[1][4][0], matrix.matrix[0][5][0]);
  matrix.matrix[1][4][1] =
      _mm256_add_pd(matrix.matrix[1][4][1], matrix.matrix[0][5][1]);
  matrix.matrix[1][4][2] =
      _mm256_add_pd(matrix.matrix[1][4][2], matrix.matrix[0][5][2]);
  matrix.matrix[1][4][3] =
      _mm256_add_pd(matrix.matrix[1][4][3], matrix.matrix[0][5][3]);
  matrix.matrix[2][2][0] =
      _mm256_add_pd(matrix.matrix[2][2][0], matrix.matrix[1][3][0]);
  matrix.matrix[2][2][1] =
      _mm256_add_pd(matrix.matrix[2][2][1], matrix.matrix[1][3][1]);
  matrix.matrix[2][2][2] =
      _mm256_add_pd(matrix.matrix[2][2][2], matrix.matrix[1][3][2]);
  matrix.matrix[2][2][3] =
      _mm256_add_pd(matrix.matrix[2][2][3], matrix.matrix[1][3][3]);
  matrix.matrix[2][2][4] =
      _mm256_add_pd(matrix.matrix[2][2][4], matrix.matrix[1][3][4]);
  matrix.matrix[2][3][0] =
      _mm256_add_pd(matrix.matrix[2][3][0], matrix.matrix[1][4][0]);
  matrix.matrix[2][3][1] =
      _mm256_add_pd(matrix.matrix[2][3][1], matrix.matrix[1][4][1]);
  matrix.matrix[2][3][2] =
      _mm256_add_pd(matrix.matrix[2][3][2], matrix.matrix[1][4][2]);
  matrix.matrix[2][3][3] =
      _mm256_add_pd(matrix.matrix[2][3][3], matrix.matrix[1][4][3]);
  matrix.matrix[2][4][0] =
      _mm256_add_pd(matrix.matrix[2][4][0], matrix.matrix[1][5][0]);
  matrix.matrix[2][4][1] =
      _mm256_add_pd(matrix.matrix[2][4][1], matrix.matrix[1][5][1]);
  matrix.matrix[2][4][2] =
      _mm256_add_pd(matrix.matrix[2][4][2], matrix.matrix[1][5][2]);
  matrix.matrix[3][3][0] =
      _mm256_add_pd(matrix.matrix[3][3][0], matrix.matrix[2][4][0]);
  matrix.matrix[3][3][1] =
      _mm256_add_pd(matrix.matrix[3][3][1], matrix.matrix[2][4][1]);
  matrix.matrix[3][3][2] =
      _mm256_add_pd(matrix.matrix[3][3][2], matrix.matrix[2][4][2]);
  matrix.matrix[3][4][0] =
      _mm256_add_pd(matrix.matrix[3][4][0], matrix.matrix[2][5][0]);
  matrix.matrix[3][4][1] =
      _mm256_add_pd(matrix.matrix[3][4][1], matrix.matrix[2][5][1]);
  matrix.matrix[4][4][0] =
      _mm256_add_pd(matrix.matrix[4][4][0], matrix.matrix[3][5][0]);
  matrix.matrix[1][0][0] = matrix.matrix[0][1][0];
  matrix.matrix[1][0][1] = matrix.matrix[0][1][1];
  matrix.matrix[1][0][2] = matrix.matrix[0][1][2];
  matrix.matrix[1][0][3] = matrix.matrix[0][1][3];
  matrix.matrix[1][0][4] = matrix.matrix[0][1][4];
  matrix.matrix[1][0][5] = matrix.matrix[0][1][5];
  matrix.matrix[1][0][6] = matrix.matrix[0][1][6];
  matrix.matrix[1][0][7] = matrix.matrix[0][1][7];
  matrix.matrix[2][0][0] = matrix.matrix[0][2][0];
  matrix.matrix[2][0][1] = matrix.matrix[0][2][1];
  matrix.matrix[2][0][2] = matrix.matrix[0][2][2];
  matrix.matrix[2][0][3] = matrix.matrix[0][2][3];
  matrix.matrix[2][0][4] = matrix.matrix[0][2][4];
  matrix.matrix[2][0][5] = matrix.matrix[0][2][5];
  matrix.matrix[2][0][6] = matrix.matrix[0][2][6];
  matrix.matrix[2][1][0] = matrix.matrix[1][2][0];
  matrix.matrix[2][1][1] = matrix.matrix[1][2][1];
  matrix.matrix[2][1][2] = matrix.matrix[1][2][2];
  matrix.matrix[2][1][3] = matrix.matrix[1][2][3];
  matrix.matrix[2][1][4] = matrix.matrix[1][2][4];
  matrix.matrix[2][1][5] = matrix.matrix[1][2][5];
  matrix.matrix[3][0][0] = matrix.matrix[0][3][0];
  matrix.matrix[3][0][1] = matrix.matrix[0][3][1];
  matrix.matrix[3][0][2] = matrix.matrix[0][3][2];
  matrix.matrix[3][0][3] = matrix.matrix[0][3][3];
  matrix.matrix[3][0][4] = matrix.matrix[0][3][4];
  matrix.matrix[3][0][5] = matrix.matrix[0][3][5];
  matrix.matrix[3][1][0] = matrix.matrix[1][3][0];
  matrix.matrix[3][1][1] = matrix.matrix[1][3][1];
  matrix.matrix[3][1][2] = matrix.matrix[1][3][2];
  matrix.matrix[3][1][3] = matrix.matrix[1][3][3];
  matrix.matrix[3][1][4] = matrix.matrix[1][3][4];
  matrix.matrix[3][2][0] = matrix.matrix[2][3][0];
  matrix.matrix[3][2][1] = matrix.matrix[2][3][1];
  matrix.matrix[3][2][2] = matrix.matrix[2][3][2];
  matrix.matrix[3][2][3] = matrix.matrix[2][3][3];
  matrix.matrix[4][0][0] = matrix.matrix[0][4][0];
  matrix.matrix[4][0][1] = matrix.matrix[0][4][1];
  matrix.matrix[4][0][2] = matrix.matrix[0][4][2];
  matrix.matrix[4][0][3] = matrix.matrix[0][4][3];
  matrix.matrix[4][0][4] = matrix.matrix[0][4][4];
  matrix.matrix[4][1][0] = matrix.matrix[1][4][0];
  matrix.matrix[4][1][1] = matrix.matrix[1][4][1];
  matrix.matrix[4][1][2] = matrix.matrix[1][4][2];
  matrix.matrix[4][1][3] = matrix.matrix[1][4][3];
  matrix.matrix[4][2][0] = matrix.matrix[2][4][0];
  matrix.matrix[4][2][1] = matrix.matrix[2][4][1];
  matrix.matrix[4][2][2] = matrix.matrix[2][4][2];
  matrix.matrix[4][3][0] = matrix.matrix[3][4][0];
  matrix.matrix[4][3][1] = matrix.matrix[3][4][1];
  matrix.matrix[5][0][0] = matrix.matrix[0][5][0];
  matrix.matrix[5][0][1] = matrix.matrix[0][5][1];
  matrix.matrix[5][0][2] = matrix.matrix[0][5][2];
  matrix.matrix[5][0][3] = matrix.matrix[0][5][3];
  matrix.matrix[5][1][0] = matrix.matrix[1][5][0];
  matrix.matrix[5][1][1] = matrix.matrix[1][5][1];
  matrix.matrix[5][1][2] = matrix.matrix[1][5][2];
  matrix.matrix[5][2][0] = matrix.matrix[2][5][0];
  matrix.matrix[5][2][1] = matrix.matrix[2][5][1];
  matrix.matrix[5][3][0] = matrix.matrix[3][5][0];

  return matrix;
}



template <>
std::vector<std::vector<
    std::pair<Resultant::UnivariatePolynomial, std::pair<double, double>>>>
UnivariatePolyMatrix<4ul>::determinant() const {

  // clang-format off
  // (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * (A[2][2] * A[3][3] - A[2][3] * A[3][2]) - //
  // (A[0][0] * A[1][2] - A[0][2] * A[1][0]) * (A[2][1] * A[3][3] - A[2][3] * A[3][1]) + //
  // (A[0][0] * A[1][3] - A[0][3] * A[1][0]) * (A[2][1] * A[3][2] - A[2][2] * A[3][1]) + //
  // (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * (A[2][0] * A[3][3] - A[2][3] * A[3][0]) - //
  // (A[0][1] * A[1][3] - A[0][3] * A[1][1]) * (A[2][0] * A[3][2] - A[2][2] * A[3][0]) + //
  // (A[0][2] * A[1][3] - A[0][3] * A[1][2]) * (A[2][0] * A[3][1] - A[2][1] * A[3][0]);  //
  // clang-format on

  //! Notice, A[3][3] is zero

  __m256d res_poly[9]; // max degree is 8

  res_poly[0] = _mm256_setzero_pd();
  res_poly[1] = _mm256_setzero_pd();
  res_poly[2] = _mm256_setzero_pd();
  res_poly[3] = _mm256_setzero_pd();
  res_poly[4] = _mm256_setzero_pd();
  res_poly[5] = _mm256_setzero_pd();
  res_poly[6] = _mm256_setzero_pd();
  res_poly[7] = _mm256_setzero_pd();
  res_poly[8] = _mm256_setzero_pd();
  // (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * (A[2][2] * A[3][3] - A[2][3] *
  // A[3][2])

  {
    __m256d temp_1[9];
    temp_1[0] = _mm256_setzero_pd();
    temp_1[1] = _mm256_setzero_pd();
    temp_1[2] = _mm256_setzero_pd();
    temp_1[3] = _mm256_setzero_pd();
    temp_1[4] = _mm256_setzero_pd();
    temp_1[5] = _mm256_setzero_pd();
    temp_1[6] = _mm256_setzero_pd();
    temp_1[7] = _mm256_setzero_pd();
    temp_1[8] = _mm256_setzero_pd();

    // A[0][0] * A[1][1]
    temp_1[0] = _mm256_add_pd(temp_1[0],
                              _mm256_mul_pd(matrix[0][0][0], matrix[1][1][0]));
    temp_1[1] = _mm256_add_pd(temp_1[1],
                              _mm256_mul_pd(matrix[0][0][0], matrix[1][1][1]));
    temp_1[2] = _mm256_add_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][0][0], matrix[1][1][2]));
    temp_1[3] = _mm256_add_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][0][0], matrix[1][1][3]));
    temp_1[1] = _mm256_add_pd(temp_1[1],
                              _mm256_mul_pd(matrix[0][0][1], matrix[1][1][0]));
    temp_1[2] = _mm256_add_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][0][1], matrix[1][1][1]));
    temp_1[3] = _mm256_add_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][0][1], matrix[1][1][2]));
    temp_1[4] = _mm256_add_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][0][1], matrix[1][1][3]));
    temp_1[2] = _mm256_add_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][0][2], matrix[1][1][0]));
    temp_1[3] = _mm256_add_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][0][2], matrix[1][1][1]));
    temp_1[4] = _mm256_add_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][0][2], matrix[1][1][2]));
    temp_1[5] = _mm256_add_pd(temp_1[5],
                              _mm256_mul_pd(matrix[0][0][2], matrix[1][1][3]));
    temp_1[3] = _mm256_add_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][0][3], matrix[1][1][0]));
    temp_1[4] = _mm256_add_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][0][3], matrix[1][1][1]));
    temp_1[5] = _mm256_add_pd(temp_1[5],
                              _mm256_mul_pd(matrix[0][0][3], matrix[1][1][2]));
    temp_1[6] = _mm256_add_pd(temp_1[6],
                              _mm256_mul_pd(matrix[0][0][3], matrix[1][1][3]));
    temp_1[4] = _mm256_add_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][0][4], matrix[1][1][0]));
    temp_1[5] = _mm256_add_pd(temp_1[5],
                              _mm256_mul_pd(matrix[0][0][4], matrix[1][1][1]));
    temp_1[6] = _mm256_add_pd(temp_1[6],
                              _mm256_mul_pd(matrix[0][0][4], matrix[1][1][2]));
    temp_1[7] = _mm256_add_pd(temp_1[7],
                              _mm256_mul_pd(matrix[0][0][4], matrix[1][1][3]));
    temp_1[5] = _mm256_add_pd(temp_1[5],
                              _mm256_mul_pd(matrix[0][0][5], matrix[1][1][0]));
    temp_1[6] = _mm256_add_pd(temp_1[6],
                              _mm256_mul_pd(matrix[0][0][5], matrix[1][1][1]));
    temp_1[7] = _mm256_add_pd(temp_1[7],
                              _mm256_mul_pd(matrix[0][0][5], matrix[1][1][2]));
    temp_1[8] = _mm256_add_pd(temp_1[8],
                              _mm256_mul_pd(matrix[0][0][5], matrix[1][1][3]));

    // A[0][1] * A[1][0]

    temp_1[0] = _mm256_sub_pd(temp_1[0],
                              _mm256_mul_pd(matrix[0][1][0], matrix[1][0][0]));
    temp_1[1] = _mm256_sub_pd(temp_1[1],
                              _mm256_mul_pd(matrix[0][1][0], matrix[1][0][1]));
    temp_1[2] = _mm256_sub_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][1][0], matrix[1][0][2]));
    temp_1[3] = _mm256_sub_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][1][0], matrix[1][0][3]));
    temp_1[4] = _mm256_sub_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][1][0], matrix[1][0][4]));
    temp_1[1] = _mm256_sub_pd(temp_1[1],
                              _mm256_mul_pd(matrix[0][1][1], matrix[1][0][0]));
    temp_1[2] = _mm256_sub_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][1][1], matrix[1][0][1]));
    temp_1[3] = _mm256_sub_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][1][1], matrix[1][0][2]));
    temp_1[4] = _mm256_sub_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][1][1], matrix[1][0][3]));
    temp_1[5] = _mm256_sub_pd(temp_1[5],
                              _mm256_mul_pd(matrix[0][1][1], matrix[1][0][4]));
    temp_1[2] = _mm256_sub_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][1][2], matrix[1][0][0]));
    temp_1[3] = _mm256_sub_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][1][2], matrix[1][0][1]));
    temp_1[4] = _mm256_sub_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][1][2], matrix[1][0][2]));
    temp_1[5] = _mm256_sub_pd(temp_1[5],
                              _mm256_mul_pd(matrix[0][1][2], matrix[1][0][3]));
    temp_1[6] = _mm256_sub_pd(temp_1[6],
                              _mm256_mul_pd(matrix[0][1][2], matrix[1][0][4]));
    temp_1[3] = _mm256_sub_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][1][3], matrix[1][0][0]));
    temp_1[4] = _mm256_sub_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][1][3], matrix[1][0][1]));
    temp_1[5] = _mm256_sub_pd(temp_1[5],
                              _mm256_mul_pd(matrix[0][1][3], matrix[1][0][2]));
    temp_1[6] = _mm256_sub_pd(temp_1[6],
                              _mm256_mul_pd(matrix[0][1][3], matrix[1][0][3]));
    temp_1[7] = _mm256_sub_pd(temp_1[7],
                              _mm256_mul_pd(matrix[0][1][3], matrix[1][0][4]));
    temp_1[4] = _mm256_sub_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][1][4], matrix[1][0][0]));
    temp_1[5] = _mm256_sub_pd(temp_1[5],
                              _mm256_mul_pd(matrix[0][1][4], matrix[1][0][1]));
    temp_1[6] = _mm256_sub_pd(temp_1[6],
                              _mm256_mul_pd(matrix[0][1][4], matrix[1][0][2]));
    temp_1[7] = _mm256_sub_pd(temp_1[7],
                              _mm256_mul_pd(matrix[0][1][4], matrix[1][0][3]));
    temp_1[8] = _mm256_sub_pd(temp_1[8],
                              _mm256_mul_pd(matrix[0][1][4], matrix[1][0][4]));

    //(A[2][2] * A[3][3] - A[2][3] *A[3][2]) = - A[2][3] *A[3][2]
    // 0 degree
    __m256d scalr = _mm256_mul_pd(matrix[2][3][0], matrix[3][2][0]);
    //    set_1_pd ??

    // add to final result
    res_poly[0] = _mm256_sub_pd(res_poly[0], _mm256_mul_pd(temp_1[0], scalr));
    res_poly[1] = _mm256_sub_pd(res_poly[1], _mm256_mul_pd(temp_1[1], scalr));
    res_poly[2] = _mm256_sub_pd(res_poly[2], _mm256_mul_pd(temp_1[2], scalr));
    res_poly[3] = _mm256_sub_pd(res_poly[3], _mm256_mul_pd(temp_1[3], scalr));
    res_poly[4] = _mm256_sub_pd(res_poly[4], _mm256_mul_pd(temp_1[4], scalr));
    res_poly[5] = _mm256_sub_pd(res_poly[5], _mm256_mul_pd(temp_1[5], scalr));
    res_poly[6] = _mm256_sub_pd(res_poly[6], _mm256_mul_pd(temp_1[6], scalr));
    res_poly[7] = _mm256_sub_pd(res_poly[7], _mm256_mul_pd(temp_1[7], scalr));
    res_poly[8] = _mm256_sub_pd(res_poly[8], _mm256_mul_pd(temp_1[8], scalr));
  }

  // (A[0][0] * A[1][2] - A[0][2] * A[1][0]) * (A[2][1] * A[3][3] - A[2][3] *
  // A[3][1])

  {
    // (A[0][0] * A[1][2]
    __m256d temp_1[8];

    temp_1[0] = _mm256_setzero_pd();
    temp_1[1] = _mm256_setzero_pd();
    temp_1[2] = _mm256_setzero_pd();
    temp_1[3] = _mm256_setzero_pd();
    temp_1[4] = _mm256_setzero_pd();
    temp_1[5] = _mm256_setzero_pd();
    temp_1[6] = _mm256_setzero_pd();
    temp_1[7] = _mm256_setzero_pd();

    temp_1[0] = _mm256_add_pd(temp_1[0],
                              _mm256_mul_pd(matrix[0][0][0], matrix[1][2][0]));
    temp_1[1] = _mm256_add_pd(temp_1[1],
                              _mm256_mul_pd(matrix[0][0][0], matrix[1][2][1]));
    temp_1[2] = _mm256_add_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][0][0], matrix[1][2][2]));
    temp_1[1] = _mm256_add_pd(temp_1[1],
                              _mm256_mul_pd(matrix[0][0][1], matrix[1][2][0]));
    temp_1[2] = _mm256_add_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][0][1], matrix[1][2][1]));
    temp_1[3] = _mm256_add_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][0][1], matrix[1][2][2]));
    temp_1[2] = _mm256_add_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][0][2], matrix[1][2][0]));
    temp_1[3] = _mm256_add_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][0][2], matrix[1][2][1]));
    temp_1[4] = _mm256_add_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][0][2], matrix[1][2][2]));
    temp_1[3] = _mm256_add_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][0][3], matrix[1][2][0]));
    temp_1[4] = _mm256_add_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][0][3], matrix[1][2][1]));
    temp_1[5] = _mm256_add_pd(temp_1[5],
                              _mm256_mul_pd(matrix[0][0][3], matrix[1][2][2]));
    temp_1[4] = _mm256_add_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][0][4], matrix[1][2][0]));
    temp_1[5] = _mm256_add_pd(temp_1[5],
                              _mm256_mul_pd(matrix[0][0][4], matrix[1][2][1]));
    temp_1[6] = _mm256_add_pd(temp_1[6],
                              _mm256_mul_pd(matrix[0][0][4], matrix[1][2][2]));
    temp_1[5] = _mm256_add_pd(temp_1[5],
                              _mm256_mul_pd(matrix[0][0][5], matrix[1][2][0]));
    temp_1[6] = _mm256_add_pd(temp_1[6],
                              _mm256_mul_pd(matrix[0][0][5], matrix[1][2][1]));
    temp_1[7] = _mm256_add_pd(temp_1[7],
                              _mm256_mul_pd(matrix[0][0][5], matrix[1][2][2]));

    // A[0][2] * A[1][0]
    temp_1[0] = _mm256_sub_pd(temp_1[0],
                              _mm256_mul_pd(matrix[0][2][0], matrix[1][0][0]));
    temp_1[1] = _mm256_sub_pd(temp_1[1],
                              _mm256_mul_pd(matrix[0][2][0], matrix[1][0][1]));
    temp_1[2] = _mm256_sub_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][2][0], matrix[1][0][2]));
    temp_1[3] = _mm256_sub_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][2][0], matrix[1][0][3]));
    temp_1[4] = _mm256_sub_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][2][0], matrix[1][0][4]));
    temp_1[1] = _mm256_sub_pd(temp_1[1],
                              _mm256_mul_pd(matrix[0][2][1], matrix[1][0][0]));
    temp_1[2] = _mm256_sub_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][2][1], matrix[1][0][1]));
    temp_1[3] = _mm256_sub_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][2][1], matrix[1][0][2]));
    temp_1[4] = _mm256_sub_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][2][1], matrix[1][0][3]));
    temp_1[5] = _mm256_sub_pd(temp_1[5],
                              _mm256_mul_pd(matrix[0][2][1], matrix[1][0][4]));
    temp_1[2] = _mm256_sub_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][2][2], matrix[1][0][0]));
    temp_1[3] = _mm256_sub_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][2][2], matrix[1][0][1]));
    temp_1[4] = _mm256_sub_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][2][2], matrix[1][0][2]));
    temp_1[5] = _mm256_sub_pd(temp_1[5],
                              _mm256_mul_pd(matrix[0][2][2], matrix[1][0][3]));
    temp_1[6] = _mm256_sub_pd(temp_1[6],
                              _mm256_mul_pd(matrix[0][2][2], matrix[1][0][4]));
    temp_1[3] = _mm256_sub_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][2][3], matrix[1][0][0]));
    temp_1[4] = _mm256_sub_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][2][3], matrix[1][0][1]));
    temp_1[5] = _mm256_sub_pd(temp_1[5],
                              _mm256_mul_pd(matrix[0][2][3], matrix[1][0][2]));
    temp_1[6] = _mm256_sub_pd(temp_1[6],
                              _mm256_mul_pd(matrix[0][2][3], matrix[1][0][3]));
    temp_1[7] = _mm256_sub_pd(temp_1[7],
                              _mm256_mul_pd(matrix[0][2][3], matrix[1][0][4]));

    // (A[2][1] * A[3][3] - A[2][3] * A[3][1]) = -A[2][3] * A[3][1]
    // 1 degree
    __m256d temp_2[2];
    temp_2[0] = _mm256_mul_pd(matrix[2][3][0], matrix[3][1][0]);
    temp_2[1] = _mm256_mul_pd(matrix[2][3][0], matrix[3][1][1]);

    __m256d temp[9];

    temp[0] = _mm256_mul_pd(temp_1[0], temp_2[0]);
    temp[1] = _mm256_mul_pd(temp_1[1], temp_2[0]);
    temp[2] = _mm256_mul_pd(temp_1[2], temp_2[0]);
    temp[3] = _mm256_mul_pd(temp_1[3], temp_2[0]);
    temp[4] = _mm256_mul_pd(temp_1[4], temp_2[0]);
    temp[5] = _mm256_mul_pd(temp_1[5], temp_2[0]);
    temp[6] = _mm256_mul_pd(temp_1[6], temp_2[0]);
    temp[7] = _mm256_mul_pd(temp_1[7], temp_2[0]);
    temp[8] = _mm256_setzero_pd();

    temp[1] = _mm256_add_pd(temp[1], _mm256_mul_pd(temp_1[0], temp_2[1]));
    temp[2] = _mm256_add_pd(temp[2], _mm256_mul_pd(temp_1[1], temp_2[1]));
    temp[3] = _mm256_add_pd(temp[3], _mm256_mul_pd(temp_1[2], temp_2[1]));
    temp[4] = _mm256_add_pd(temp[4], _mm256_mul_pd(temp_1[3], temp_2[1]));
    temp[5] = _mm256_add_pd(temp[5], _mm256_mul_pd(temp_1[4], temp_2[1]));
    temp[6] = _mm256_add_pd(temp[6], _mm256_mul_pd(temp_1[5], temp_2[1]));
    temp[7] = _mm256_add_pd(temp[7], _mm256_mul_pd(temp_1[6], temp_2[1]));
    temp[8] = _mm256_add_pd(temp[8], _mm256_mul_pd(temp_1[7], temp_2[1]));

    // add to final result
    res_poly[0] = _mm256_add_pd(res_poly[0], temp[0]);
    res_poly[1] = _mm256_add_pd(res_poly[1], temp[1]);
    res_poly[2] = _mm256_add_pd(res_poly[2], temp[2]);
    res_poly[3] = _mm256_add_pd(res_poly[3], temp[3]);
    res_poly[4] = _mm256_add_pd(res_poly[4], temp[4]);
    res_poly[5] = _mm256_add_pd(res_poly[5], temp[5]);
    res_poly[6] = _mm256_add_pd(res_poly[6], temp[6]);
    res_poly[7] = _mm256_add_pd(res_poly[7], temp[7]);
    res_poly[8] = _mm256_add_pd(res_poly[8], temp[8]);
  }

  //(A[0][0] * A[1][3] - A[0][3] * A[1][0]) * (A[2][1] * A[3][2] - A[2][2] *
  // A[3][1])

  {
    // A[0][0] * A[1][3]
    __m256d temp_1[7];

    temp_1[0] = _mm256_setzero_pd();
    temp_1[1] = _mm256_setzero_pd();
    temp_1[2] = _mm256_setzero_pd();
    temp_1[3] = _mm256_setzero_pd();
    temp_1[4] = _mm256_setzero_pd();
    temp_1[5] = _mm256_setzero_pd();
    temp_1[6] = _mm256_setzero_pd();

    temp_1[0] = _mm256_add_pd(temp_1[0],
                              _mm256_mul_pd(matrix[0][0][0], matrix[1][3][0]));
    temp_1[1] = _mm256_add_pd(temp_1[1],
                              _mm256_mul_pd(matrix[0][0][0], matrix[1][3][1]));
    temp_1[1] = _mm256_add_pd(temp_1[1],
                              _mm256_mul_pd(matrix[0][0][1], matrix[1][3][0]));
    temp_1[2] = _mm256_add_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][0][1], matrix[1][3][1]));
    temp_1[2] = _mm256_add_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][0][2], matrix[1][3][0]));
    temp_1[3] = _mm256_add_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][0][2], matrix[1][3][1]));
    temp_1[3] = _mm256_add_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][0][3], matrix[1][3][0]));
    temp_1[4] = _mm256_add_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][0][3], matrix[1][3][1]));
    temp_1[4] = _mm256_add_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][0][4], matrix[1][3][0]));
    temp_1[5] = _mm256_add_pd(temp_1[5],
                              _mm256_mul_pd(matrix[0][0][4], matrix[1][3][1]));
    temp_1[5] = _mm256_add_pd(temp_1[5],
                              _mm256_mul_pd(matrix[0][0][5], matrix[1][3][0]));
    temp_1[6] = _mm256_add_pd(temp_1[6],
                              _mm256_mul_pd(matrix[0][0][5], matrix[1][3][1]));

    // A[0][3] * A[1][0]

    temp_1[0] = _mm256_sub_pd(temp_1[0],
                              _mm256_mul_pd(matrix[0][3][0], matrix[1][0][0]));
    temp_1[1] = _mm256_sub_pd(temp_1[1],
                              _mm256_mul_pd(matrix[0][3][0], matrix[1][0][1]));
    temp_1[2] = _mm256_sub_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][3][0], matrix[1][0][2]));
    temp_1[3] = _mm256_sub_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][3][0], matrix[1][0][3]));
    temp_1[4] = _mm256_sub_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][3][0], matrix[1][0][4]));
    temp_1[1] = _mm256_sub_pd(temp_1[1],
                              _mm256_mul_pd(matrix[0][3][1], matrix[1][0][0]));
    temp_1[2] = _mm256_sub_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][3][1], matrix[1][0][1]));
    temp_1[3] = _mm256_sub_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][3][1], matrix[1][0][2]));
    temp_1[4] = _mm256_sub_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][3][1], matrix[1][0][3]));
    temp_1[5] = _mm256_sub_pd(temp_1[5],
                              _mm256_mul_pd(matrix[0][3][1], matrix[1][0][4]));
    temp_1[2] = _mm256_sub_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][3][2], matrix[1][0][0]));
    temp_1[3] = _mm256_sub_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][3][2], matrix[1][0][1]));
    temp_1[4] = _mm256_sub_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][3][2], matrix[1][0][2]));
    temp_1[5] = _mm256_sub_pd(temp_1[5],
                              _mm256_mul_pd(matrix[0][3][2], matrix[1][0][3]));
    temp_1[6] = _mm256_sub_pd(temp_1[6],
                              _mm256_mul_pd(matrix[0][3][2], matrix[1][0][4]));

    // A[2][1] * A[3][2]

    __m256d temp_2[3];

    temp_2[0] = _mm256_mul_pd(matrix[2][1][0], matrix[3][2][0]);
    temp_2[1] = _mm256_mul_pd(matrix[2][1][1], matrix[3][2][0]);
    temp_2[2] = _mm256_mul_pd(matrix[2][1][2], matrix[3][2][0]);

    // A[2][2] * A[3][1])

    temp_2[0] = _mm256_sub_pd(temp_2[0],
                              _mm256_mul_pd(matrix[2][2][0], matrix[3][1][0]));
    temp_2[1] = _mm256_sub_pd(temp_2[1],
                              _mm256_mul_pd(matrix[2][2][0], matrix[3][1][1]));
    temp_2[1] = _mm256_sub_pd(temp_2[1],
                              _mm256_mul_pd(matrix[2][2][1], matrix[3][1][0]));
    temp_2[2] = _mm256_sub_pd(temp_2[2],
                              _mm256_mul_pd(matrix[2][2][1], matrix[3][1][1]));

    // temp_1 * temp_2 and add to final result

    res_poly[0] =
        _mm256_add_pd(res_poly[0], _mm256_mul_pd(temp_1[0], temp_2[0]));
    res_poly[1] =
        _mm256_add_pd(res_poly[1], _mm256_mul_pd(temp_1[0], temp_2[1]));
    res_poly[2] =
        _mm256_add_pd(res_poly[2], _mm256_mul_pd(temp_1[0], temp_2[2]));
    res_poly[1] =
        _mm256_add_pd(res_poly[1], _mm256_mul_pd(temp_1[1], temp_2[0]));
    res_poly[2] =
        _mm256_add_pd(res_poly[2], _mm256_mul_pd(temp_1[1], temp_2[1]));
    res_poly[3] =
        _mm256_add_pd(res_poly[3], _mm256_mul_pd(temp_1[1], temp_2[2]));
    res_poly[2] =
        _mm256_add_pd(res_poly[2], _mm256_mul_pd(temp_1[2], temp_2[0]));
    res_poly[3] =
        _mm256_add_pd(res_poly[3], _mm256_mul_pd(temp_1[2], temp_2[1]));
    res_poly[4] =
        _mm256_add_pd(res_poly[4], _mm256_mul_pd(temp_1[2], temp_2[2]));
    res_poly[3] =
        _mm256_add_pd(res_poly[3], _mm256_mul_pd(temp_1[3], temp_2[0]));
    res_poly[4] =
        _mm256_add_pd(res_poly[4], _mm256_mul_pd(temp_1[3], temp_2[1]));
    res_poly[5] =
        _mm256_add_pd(res_poly[5], _mm256_mul_pd(temp_1[3], temp_2[2]));
    res_poly[4] =
        _mm256_add_pd(res_poly[4], _mm256_mul_pd(temp_1[4], temp_2[0]));
    res_poly[5] =
        _mm256_add_pd(res_poly[5], _mm256_mul_pd(temp_1[4], temp_2[1]));
    res_poly[6] =
        _mm256_add_pd(res_poly[6], _mm256_mul_pd(temp_1[4], temp_2[2]));
    res_poly[5] =
        _mm256_add_pd(res_poly[5], _mm256_mul_pd(temp_1[5], temp_2[0]));
    res_poly[6] =
        _mm256_add_pd(res_poly[6], _mm256_mul_pd(temp_1[5], temp_2[1]));
    res_poly[7] =
        _mm256_add_pd(res_poly[7], _mm256_mul_pd(temp_1[5], temp_2[2]));
    res_poly[6] =
        _mm256_add_pd(res_poly[6], _mm256_mul_pd(temp_1[6], temp_2[0]));
    res_poly[7] =
        _mm256_add_pd(res_poly[7], _mm256_mul_pd(temp_1[6], temp_2[1]));
    res_poly[8] =
        _mm256_add_pd(res_poly[8], _mm256_mul_pd(temp_1[6], temp_2[2]));
  }

  // (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * (A[2][0] * A[3][3] - A[2][3] *
  // A[3][0])
  {
    // A[0][1] * A[1][2]
    __m256d temp_1[7];
    temp_1[0] = _mm256_setzero_pd();
    temp_1[1] = _mm256_setzero_pd();
    temp_1[2] = _mm256_setzero_pd();
    temp_1[3] = _mm256_setzero_pd();
    temp_1[4] = _mm256_setzero_pd();
    temp_1[5] = _mm256_setzero_pd();
    temp_1[6] = _mm256_setzero_pd();

    temp_1[0] = _mm256_add_pd(temp_1[0],
                              _mm256_mul_pd(matrix[0][1][0], matrix[1][2][0]));
    temp_1[1] = _mm256_add_pd(temp_1[1],
                              _mm256_mul_pd(matrix[0][1][0], matrix[1][2][1]));
    temp_1[2] = _mm256_add_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][1][0], matrix[1][2][2]));
    temp_1[1] = _mm256_add_pd(temp_1[1],
                              _mm256_mul_pd(matrix[0][1][1], matrix[1][2][0]));
    temp_1[2] = _mm256_add_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][1][1], matrix[1][2][1]));
    temp_1[3] = _mm256_add_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][1][1], matrix[1][2][2]));
    temp_1[2] = _mm256_add_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][1][2], matrix[1][2][0]));
    temp_1[3] = _mm256_add_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][1][2], matrix[1][2][1]));
    temp_1[4] = _mm256_add_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][1][2], matrix[1][2][2]));
    temp_1[3] = _mm256_add_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][1][3], matrix[1][2][0]));
    temp_1[4] = _mm256_add_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][1][3], matrix[1][2][1]));
    temp_1[5] = _mm256_add_pd(temp_1[5],
                              _mm256_mul_pd(matrix[0][1][3], matrix[1][2][2]));
    temp_1[4] = _mm256_add_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][1][4], matrix[1][2][0]));
    temp_1[5] = _mm256_add_pd(temp_1[5],
                              _mm256_mul_pd(matrix[0][1][4], matrix[1][2][1]));
    temp_1[6] = _mm256_add_pd(temp_1[6],
                              _mm256_mul_pd(matrix[0][1][4], matrix[1][2][2]));

    // A[0][2] * A[1][1]
    temp_1[0] = _mm256_sub_pd(temp_1[0],
                              _mm256_mul_pd(matrix[0][2][0], matrix[1][1][0]));
    temp_1[1] = _mm256_sub_pd(temp_1[1],
                              _mm256_mul_pd(matrix[0][2][0], matrix[1][1][1]));
    temp_1[2] = _mm256_sub_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][2][0], matrix[1][1][2]));
    temp_1[3] = _mm256_sub_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][2][0], matrix[1][1][3]));
    temp_1[1] = _mm256_sub_pd(temp_1[1],
                              _mm256_mul_pd(matrix[0][2][1], matrix[1][1][0]));
    temp_1[2] = _mm256_sub_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][2][1], matrix[1][1][1]));
    temp_1[3] = _mm256_sub_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][2][1], matrix[1][1][2]));
    temp_1[4] = _mm256_sub_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][2][1], matrix[1][1][3]));
    temp_1[2] = _mm256_sub_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][2][2], matrix[1][1][0]));
    temp_1[3] = _mm256_sub_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][2][2], matrix[1][1][1]));
    temp_1[4] = _mm256_sub_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][2][2], matrix[1][1][2]));
    temp_1[5] = _mm256_sub_pd(temp_1[5],
                              _mm256_mul_pd(matrix[0][2][2], matrix[1][1][3]));
    temp_1[3] = _mm256_sub_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][2][3], matrix[1][1][0]));
    temp_1[4] = _mm256_sub_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][2][3], matrix[1][1][1]));
    temp_1[5] = _mm256_sub_pd(temp_1[5],
                              _mm256_mul_pd(matrix[0][2][3], matrix[1][1][2]));
    temp_1[6] = _mm256_sub_pd(temp_1[6],
                              _mm256_mul_pd(matrix[0][2][3], matrix[1][1][3]));

    // A[2][0] * A[3][3] - A[2][3] *A[3][0] = -A[2][3] *A[3][0]
    // 2 degree
    __m256d temp_2[3];
    temp_2[0] = _mm256_mul_pd(matrix[2][3][0], matrix[3][0][0]);
    temp_2[1] = _mm256_mul_pd(matrix[2][3][0], matrix[3][0][1]);
    temp_2[2] = _mm256_mul_pd(matrix[2][3][0], matrix[3][0][2]);

    // temp_1 multiply temp_2 and add to final result
    res_poly[0] =
        _mm256_sub_pd(res_poly[0], _mm256_mul_pd(temp_1[0], temp_2[0]));
    res_poly[1] =
        _mm256_sub_pd(res_poly[1], _mm256_mul_pd(temp_1[0], temp_2[1]));
    res_poly[2] =
        _mm256_sub_pd(res_poly[2], _mm256_mul_pd(temp_1[0], temp_2[2]));
    res_poly[1] =
        _mm256_sub_pd(res_poly[1], _mm256_mul_pd(temp_1[1], temp_2[0]));
    res_poly[2] =
        _mm256_sub_pd(res_poly[2], _mm256_mul_pd(temp_1[1], temp_2[1]));
    res_poly[3] =
        _mm256_sub_pd(res_poly[3], _mm256_mul_pd(temp_1[1], temp_2[2]));
    res_poly[2] =
        _mm256_sub_pd(res_poly[2], _mm256_mul_pd(temp_1[2], temp_2[0]));
    res_poly[3] =
        _mm256_sub_pd(res_poly[3], _mm256_mul_pd(temp_1[2], temp_2[1]));
    res_poly[4] =
        _mm256_sub_pd(res_poly[4], _mm256_mul_pd(temp_1[2], temp_2[2]));
    res_poly[3] =
        _mm256_sub_pd(res_poly[3], _mm256_mul_pd(temp_1[3], temp_2[0]));
    res_poly[4] =
        _mm256_sub_pd(res_poly[4], _mm256_mul_pd(temp_1[3], temp_2[1]));
    res_poly[5] =
        _mm256_sub_pd(res_poly[5], _mm256_mul_pd(temp_1[3], temp_2[2]));
    res_poly[4] =
        _mm256_sub_pd(res_poly[4], _mm256_mul_pd(temp_1[4], temp_2[0]));
    res_poly[5] =
        _mm256_sub_pd(res_poly[5], _mm256_mul_pd(temp_1[4], temp_2[1]));
    res_poly[6] =
        _mm256_sub_pd(res_poly[6], _mm256_mul_pd(temp_1[4], temp_2[2]));
    res_poly[5] =
        _mm256_sub_pd(res_poly[5], _mm256_mul_pd(temp_1[5], temp_2[0]));
    res_poly[6] =
        _mm256_sub_pd(res_poly[6], _mm256_mul_pd(temp_1[5], temp_2[1]));
    res_poly[7] =
        _mm256_sub_pd(res_poly[7], _mm256_mul_pd(temp_1[5], temp_2[2]));
    res_poly[6] =
        _mm256_sub_pd(res_poly[6], _mm256_mul_pd(temp_1[6], temp_2[0]));
    res_poly[7] =
        _mm256_sub_pd(res_poly[7], _mm256_mul_pd(temp_1[6], temp_2[1]));
    res_poly[8] =
        _mm256_sub_pd(res_poly[8], _mm256_mul_pd(temp_1[6], temp_2[2]));
  }

  //(A[0][1] * A[1][3] - A[0][3] * A[1][1]) * (A[2][0] * A[3][2] - A[2][2] *
  // A[3][0])

  {
    // A[0][1] * A[1][3]
    __m256d temp_1[6];
    temp_1[0] = _mm256_setzero_pd();
    temp_1[1] = _mm256_setzero_pd();
    temp_1[2] = _mm256_setzero_pd();
    temp_1[3] = _mm256_setzero_pd();
    temp_1[4] = _mm256_setzero_pd();
    temp_1[5] = _mm256_setzero_pd();

    temp_1[0] = _mm256_add_pd(temp_1[0],
                              _mm256_mul_pd(matrix[0][1][0], matrix[1][3][0]));
    temp_1[1] = _mm256_add_pd(temp_1[1],
                              _mm256_mul_pd(matrix[0][1][0], matrix[1][3][1]));
    temp_1[1] = _mm256_add_pd(temp_1[1],
                              _mm256_mul_pd(matrix[0][1][1], matrix[1][3][0]));
    temp_1[2] = _mm256_add_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][1][1], matrix[1][3][1]));
    temp_1[2] = _mm256_add_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][1][2], matrix[1][3][0]));
    temp_1[3] = _mm256_add_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][1][2], matrix[1][3][1]));
    temp_1[3] = _mm256_add_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][1][3], matrix[1][3][0]));
    temp_1[4] = _mm256_add_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][1][3], matrix[1][3][1]));
    temp_1[4] = _mm256_add_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][1][4], matrix[1][3][0]));
    temp_1[5] = _mm256_add_pd(temp_1[5],
                              _mm256_mul_pd(matrix[0][1][4], matrix[1][3][1]));

    // A[0][3] * A[1][1]
    temp_1[0] = _mm256_sub_pd(temp_1[0],
                              _mm256_mul_pd(matrix[0][3][0], matrix[1][1][0]));
    temp_1[1] = _mm256_sub_pd(temp_1[1],
                              _mm256_mul_pd(matrix[0][3][0], matrix[1][1][1]));
    temp_1[2] = _mm256_sub_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][3][0], matrix[1][1][2]));
    temp_1[3] = _mm256_sub_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][3][0], matrix[1][1][3]));
    temp_1[1] = _mm256_sub_pd(temp_1[1],
                              _mm256_mul_pd(matrix[0][3][1], matrix[1][1][0]));
    temp_1[2] = _mm256_sub_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][3][1], matrix[1][1][1]));
    temp_1[3] = _mm256_sub_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][3][1], matrix[1][1][2]));
    temp_1[4] = _mm256_sub_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][3][1], matrix[1][1][3]));
    temp_1[2] = _mm256_sub_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][3][2], matrix[1][1][0]));
    temp_1[3] = _mm256_sub_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][3][2], matrix[1][1][1]));
    temp_1[4] = _mm256_sub_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][3][2], matrix[1][1][2]));
    temp_1[5] = _mm256_sub_pd(temp_1[5],
                              _mm256_mul_pd(matrix[0][3][2], matrix[1][1][3]));

    // A[2][0] * A[3][2]

    __m256d temp_2[4];
    temp_2[0] = _mm256_mul_pd(matrix[2][0][0], matrix[3][2][0]);
    temp_2[1] = _mm256_mul_pd(matrix[2][0][1], matrix[3][2][0]);
    temp_2[2] = _mm256_mul_pd(matrix[2][0][2], matrix[3][2][0]);
    temp_2[3] = _mm256_mul_pd(matrix[2][0][3], matrix[3][2][0]);

    // A[2][2] * A[3][0])

    temp_2[0] = _mm256_sub_pd(temp_2[0],
                              _mm256_mul_pd(matrix[2][2][0], matrix[3][0][0]));
    temp_2[1] = _mm256_sub_pd(temp_2[1],
                              _mm256_mul_pd(matrix[2][2][0], matrix[3][0][1]));
    temp_2[2] = _mm256_sub_pd(temp_2[2],
                              _mm256_mul_pd(matrix[2][2][0], matrix[3][0][2]));
    temp_2[1] = _mm256_sub_pd(temp_2[1],
                              _mm256_mul_pd(matrix[2][2][1], matrix[3][0][0]));
    temp_2[2] = _mm256_sub_pd(temp_2[2],
                              _mm256_mul_pd(matrix[2][2][1], matrix[3][0][1]));
    temp_2[3] = _mm256_sub_pd(temp_2[3],
                              _mm256_mul_pd(matrix[2][2][1], matrix[3][0][2]));

    // temp_1 multiply temp_2

    res_poly[0] =
        _mm256_sub_pd(res_poly[0], _mm256_mul_pd(temp_1[0], temp_2[0]));
    res_poly[1] =
        _mm256_sub_pd(res_poly[1], _mm256_mul_pd(temp_1[0], temp_2[1]));
    res_poly[2] =
        _mm256_sub_pd(res_poly[2], _mm256_mul_pd(temp_1[0], temp_2[2]));
    res_poly[3] =
        _mm256_sub_pd(res_poly[3], _mm256_mul_pd(temp_1[0], temp_2[3]));
    res_poly[1] =
        _mm256_sub_pd(res_poly[1], _mm256_mul_pd(temp_1[1], temp_2[0]));
    res_poly[2] =
        _mm256_sub_pd(res_poly[2], _mm256_mul_pd(temp_1[1], temp_2[1]));
    res_poly[3] =
        _mm256_sub_pd(res_poly[3], _mm256_mul_pd(temp_1[1], temp_2[2]));
    res_poly[4] =
        _mm256_sub_pd(res_poly[4], _mm256_mul_pd(temp_1[1], temp_2[3]));
    res_poly[2] =
        _mm256_sub_pd(res_poly[2], _mm256_mul_pd(temp_1[2], temp_2[0]));
    res_poly[3] =
        _mm256_sub_pd(res_poly[3], _mm256_mul_pd(temp_1[2], temp_2[1]));
    res_poly[4] =
        _mm256_sub_pd(res_poly[4], _mm256_mul_pd(temp_1[2], temp_2[2]));
    res_poly[5] =
        _mm256_sub_pd(res_poly[5], _mm256_mul_pd(temp_1[2], temp_2[3]));
    res_poly[3] =
        _mm256_sub_pd(res_poly[3], _mm256_mul_pd(temp_1[3], temp_2[0]));
    res_poly[4] =
        _mm256_sub_pd(res_poly[4], _mm256_mul_pd(temp_1[3], temp_2[1]));
    res_poly[5] =
        _mm256_sub_pd(res_poly[5], _mm256_mul_pd(temp_1[3], temp_2[2]));
    res_poly[6] =
        _mm256_sub_pd(res_poly[6], _mm256_mul_pd(temp_1[3], temp_2[3]));
    res_poly[4] =
        _mm256_sub_pd(res_poly[4], _mm256_mul_pd(temp_1[4], temp_2[0]));
    res_poly[5] =
        _mm256_sub_pd(res_poly[5], _mm256_mul_pd(temp_1[4], temp_2[1]));
    res_poly[6] =
        _mm256_sub_pd(res_poly[6], _mm256_mul_pd(temp_1[4], temp_2[2]));
    res_poly[7] =
        _mm256_sub_pd(res_poly[7], _mm256_mul_pd(temp_1[4], temp_2[3]));
    res_poly[5] =
        _mm256_sub_pd(res_poly[5], _mm256_mul_pd(temp_1[5], temp_2[0]));
    res_poly[6] =
        _mm256_sub_pd(res_poly[6], _mm256_mul_pd(temp_1[5], temp_2[1]));
    res_poly[7] =
        _mm256_sub_pd(res_poly[7], _mm256_mul_pd(temp_1[5], temp_2[2]));
    res_poly[8] =
        _mm256_sub_pd(res_poly[8], _mm256_mul_pd(temp_1[5], temp_2[3]));
  }

  // (A[0][2] * A[1][3] - A[0][3] * A[1][2]) * (A[2][0] * A[3][1] - A[2][1] *
  // A[3][0]);
  {
    // A[0][2] * A[1][3]
    __m256d temp_1[5];
    temp_1[0] = _mm256_setzero_pd();
    temp_1[1] = _mm256_setzero_pd();
    temp_1[2] = _mm256_setzero_pd();
    temp_1[3] = _mm256_setzero_pd();
    temp_1[4] = _mm256_setzero_pd();

    temp_1[0] = _mm256_add_pd(temp_1[0],
                              _mm256_mul_pd(matrix[0][2][0], matrix[1][3][0]));
    temp_1[1] = _mm256_add_pd(temp_1[1],
                              _mm256_mul_pd(matrix[0][2][0], matrix[1][3][1]));
    temp_1[1] = _mm256_add_pd(temp_1[1],
                              _mm256_mul_pd(matrix[0][2][1], matrix[1][3][0]));
    temp_1[2] = _mm256_add_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][2][1], matrix[1][3][1]));
    temp_1[2] = _mm256_add_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][2][2], matrix[1][3][0]));
    temp_1[3] = _mm256_add_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][2][2], matrix[1][3][1]));
    temp_1[3] = _mm256_add_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][2][3], matrix[1][3][0]));
    temp_1[4] = _mm256_add_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][2][3], matrix[1][3][1]));
    // A[0][3] * A[1][2]
    temp_1[0] = _mm256_sub_pd(temp_1[0],
                              _mm256_mul_pd(matrix[0][3][0], matrix[1][2][0]));
    temp_1[1] = _mm256_sub_pd(temp_1[1],
                              _mm256_mul_pd(matrix[0][3][0], matrix[1][2][1]));
    temp_1[2] = _mm256_sub_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][3][0], matrix[1][2][2]));
    temp_1[1] = _mm256_sub_pd(temp_1[1],
                              _mm256_mul_pd(matrix[0][3][1], matrix[1][2][0]));
    temp_1[2] = _mm256_sub_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][3][1], matrix[1][2][1]));
    temp_1[3] = _mm256_sub_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][3][1], matrix[1][2][2]));
    temp_1[2] = _mm256_sub_pd(temp_1[2],
                              _mm256_mul_pd(matrix[0][3][2], matrix[1][2][0]));
    temp_1[3] = _mm256_sub_pd(temp_1[3],
                              _mm256_mul_pd(matrix[0][3][2], matrix[1][2][1]));
    temp_1[4] = _mm256_sub_pd(temp_1[4],
                              _mm256_mul_pd(matrix[0][3][2], matrix[1][2][2]));

    // A[2][0] * A[3][1]
    __m256d temp_2[5];
    temp_2[0] = _mm256_setzero_pd();
    temp_2[1] = _mm256_setzero_pd();
    temp_2[2] = _mm256_setzero_pd();
    temp_2[3] = _mm256_setzero_pd();
    temp_2[4] = _mm256_setzero_pd();

    temp_2[0] = _mm256_add_pd(temp_2[0],
                              _mm256_mul_pd(matrix[2][0][0], matrix[3][1][0]));
    temp_2[1] = _mm256_add_pd(temp_2[1],
                              _mm256_mul_pd(matrix[2][0][0], matrix[3][1][1]));
    temp_2[1] = _mm256_add_pd(temp_2[1],
                              _mm256_mul_pd(matrix[2][0][1], matrix[3][1][0]));
    temp_2[2] = _mm256_add_pd(temp_2[2],
                              _mm256_mul_pd(matrix[2][0][1], matrix[3][1][1]));
    temp_2[2] = _mm256_add_pd(temp_2[2],
                              _mm256_mul_pd(matrix[2][0][2], matrix[3][1][0]));
    temp_2[3] = _mm256_add_pd(temp_2[3],
                              _mm256_mul_pd(matrix[2][0][2], matrix[3][1][1]));
    temp_2[3] = _mm256_add_pd(temp_2[3],
                              _mm256_mul_pd(matrix[2][0][3], matrix[3][1][0]));
    temp_2[4] = _mm256_add_pd(temp_2[4],
                              _mm256_mul_pd(matrix[2][0][3], matrix[3][1][1]));

    // A[2][1] *A[3][0];
    temp_2[0] = _mm256_sub_pd(temp_2[0],
                              _mm256_mul_pd(matrix[2][1][0], matrix[3][0][0]));
    temp_2[1] = _mm256_sub_pd(temp_2[1],
                              _mm256_mul_pd(matrix[2][1][0], matrix[3][0][1]));
    temp_2[2] = _mm256_sub_pd(temp_2[2],
                              _mm256_mul_pd(matrix[2][1][0], matrix[3][0][2]));
    temp_2[1] = _mm256_sub_pd(temp_2[1],
                              _mm256_mul_pd(matrix[2][1][1], matrix[3][0][0]));
    temp_2[2] = _mm256_sub_pd(temp_2[2],
                              _mm256_mul_pd(matrix[2][1][1], matrix[3][0][1]));
    temp_2[3] = _mm256_sub_pd(temp_2[3],
                              _mm256_mul_pd(matrix[2][1][1], matrix[3][0][2]));
    temp_2[2] = _mm256_sub_pd(temp_2[2],
                              _mm256_mul_pd(matrix[2][1][2], matrix[3][0][0]));
    temp_2[3] = _mm256_sub_pd(temp_2[3],
                              _mm256_mul_pd(matrix[2][1][2], matrix[3][0][1]));
    temp_2[4] = _mm256_sub_pd(temp_2[4],
                              _mm256_mul_pd(matrix[2][1][2], matrix[3][0][2]));

    // multiple temp_1 and temp_2 add to result
    res_poly[0] =
        _mm256_add_pd(res_poly[0], _mm256_mul_pd(temp_1[0], temp_2[0]));
    res_poly[1] =
        _mm256_add_pd(res_poly[1], _mm256_mul_pd(temp_1[0], temp_2[1]));
    res_poly[2] =
        _mm256_add_pd(res_poly[2], _mm256_mul_pd(temp_1[0], temp_2[2]));
    res_poly[3] =
        _mm256_add_pd(res_poly[3], _mm256_mul_pd(temp_1[0], temp_2[3]));
    res_poly[4] =
        _mm256_add_pd(res_poly[4], _mm256_mul_pd(temp_1[0], temp_2[4]));
    res_poly[1] =
        _mm256_add_pd(res_poly[1], _mm256_mul_pd(temp_1[1], temp_2[0]));
    res_poly[2] =
        _mm256_add_pd(res_poly[2], _mm256_mul_pd(temp_1[1], temp_2[1]));
    res_poly[3] =
        _mm256_add_pd(res_poly[3], _mm256_mul_pd(temp_1[1], temp_2[2]));
    res_poly[4] =
        _mm256_add_pd(res_poly[4], _mm256_mul_pd(temp_1[1], temp_2[3]));
    res_poly[5] =
        _mm256_add_pd(res_poly[5], _mm256_mul_pd(temp_1[1], temp_2[4]));
    res_poly[2] =
        _mm256_add_pd(res_poly[2], _mm256_mul_pd(temp_1[2], temp_2[0]));
    res_poly[3] =
        _mm256_add_pd(res_poly[3], _mm256_mul_pd(temp_1[2], temp_2[1]));
    res_poly[4] =
        _mm256_add_pd(res_poly[4], _mm256_mul_pd(temp_1[2], temp_2[2]));
    res_poly[5] =
        _mm256_add_pd(res_poly[5], _mm256_mul_pd(temp_1[2], temp_2[3]));
    res_poly[6] =
        _mm256_add_pd(res_poly[6], _mm256_mul_pd(temp_1[2], temp_2[4]));
    res_poly[3] =
        _mm256_add_pd(res_poly[3], _mm256_mul_pd(temp_1[3], temp_2[0]));
    res_poly[4] =
        _mm256_add_pd(res_poly[4], _mm256_mul_pd(temp_1[3], temp_2[1]));
    res_poly[5] =
        _mm256_add_pd(res_poly[5], _mm256_mul_pd(temp_1[3], temp_2[2]));
    res_poly[6] =
        _mm256_add_pd(res_poly[6], _mm256_mul_pd(temp_1[3], temp_2[3]));
    res_poly[7] =
        _mm256_add_pd(res_poly[7], _mm256_mul_pd(temp_1[3], temp_2[4]));
    res_poly[4] =
        _mm256_add_pd(res_poly[4], _mm256_mul_pd(temp_1[4], temp_2[0]));
    res_poly[5] =
        _mm256_add_pd(res_poly[5], _mm256_mul_pd(temp_1[4], temp_2[1]));
    res_poly[6] =
        _mm256_add_pd(res_poly[6], _mm256_mul_pd(temp_1[4], temp_2[2]));
    res_poly[7] =
        _mm256_add_pd(res_poly[7], _mm256_mul_pd(temp_1[4], temp_2[3]));
    res_poly[8] =
        _mm256_add_pd(res_poly[8], _mm256_mul_pd(temp_1[4], temp_2[4]));
  }

  // packed res_poly to Resultant::UnivariatePolynomial
  std::vector<std::vector<
      std::pair<Resultant::UnivariatePolynomial, std::pair<double, double>>>>
      result;

  std::vector<
      std::pair<Resultant::UnivariatePolynomial, std::pair<double, double>>>
      res_0;
  std::vector<
      std::pair<Resultant::UnivariatePolynomial, std::pair<double, double>>>
      res_1;
  std::vector<
      std::pair<Resultant::UnivariatePolynomial, std::pair<double, double>>>
      res_2;
  std::vector<
      std::pair<Resultant::UnivariatePolynomial, std::pair<double, double>>>
      res_3;

  double unpacked_poly[9][4];

  _mm256_store_pd(&unpacked_poly[0][0], res_poly[0]);
  _mm256_store_pd(&unpacked_poly[1][0], res_poly[1]);
  _mm256_store_pd(&unpacked_poly[2][0], res_poly[2]);
  _mm256_store_pd(&unpacked_poly[3][0], res_poly[3]);
  _mm256_store_pd(&unpacked_poly[4][0], res_poly[4]);
  _mm256_store_pd(&unpacked_poly[5][0], res_poly[5]);
  _mm256_store_pd(&unpacked_poly[6][0], res_poly[6]);
  _mm256_store_pd(&unpacked_poly[7][0], res_poly[7]);
  _mm256_store_pd(&unpacked_poly[8][0], res_poly[8]);

  std::vector<double> poly_0 = {
      unpacked_poly[0][0], unpacked_poly[1][0], unpacked_poly[2][0],
      unpacked_poly[3][0], unpacked_poly[4][0], unpacked_poly[5][0],
      unpacked_poly[6][0], unpacked_poly[7][0], unpacked_poly[8][0]};
  std::vector<double> poly_1 = {
      unpacked_poly[0][1], unpacked_poly[1][1], unpacked_poly[2][1],
      unpacked_poly[3][1], unpacked_poly[4][1], unpacked_poly[5][1],
      unpacked_poly[6][1], unpacked_poly[7][1], unpacked_poly[8][1]};
  std::vector<double> poly_2 = {
      unpacked_poly[0][2], unpacked_poly[1][2], unpacked_poly[2][2],
      unpacked_poly[3][2], unpacked_poly[4][2], unpacked_poly[5][2],
      unpacked_poly[6][2], unpacked_poly[7][2], unpacked_poly[8][2]};
  std::vector<double> poly_3 = {
      unpacked_poly[0][3], unpacked_poly[1][3], unpacked_poly[2][3],
      unpacked_poly[3][3], unpacked_poly[4][3], unpacked_poly[5][3],
      unpacked_poly[6][3], unpacked_poly[7][3], unpacked_poly[8][3]};

  res_0.push_back(std::make_pair(Resultant::UnivariatePolynomial(poly_0),
                                 std::make_pair(0.0, 1.0)));
  res_1.push_back(std::make_pair(Resultant::UnivariatePolynomial(poly_1),
                                 std::make_pair(0.0, 1.0)));
  res_2.push_back(std::make_pair(Resultant::UnivariatePolynomial(poly_2),
                                 std::make_pair(0.0, 1.0)));
  res_3.push_back(std::make_pair(Resultant::UnivariatePolynomial(poly_3),
                                 std::make_pair(0.0, 1.0)));

  result.emplace_back(res_0);
  result.emplace_back(res_1);
  result.emplace_back(res_2);
  result.emplace_back(res_3);

  return result;
}

// clang-format on

#define OPTIMIZED_ONEBOUNCE_DET

#ifndef OPTIMIZED_ONEBOUNCE_DET

template <>
std::vector<std::vector<
    std::pair<Resultant::UnivariatePolynomial, std::pair<double, double>>>>
UnivariatePolyMatrix<6ul>::determinant() const {
  __m256d n_6_temp[17];
  n_6_temp[0] = _mm256_setzero_pd();
  n_6_temp[1] = _mm256_setzero_pd();
  n_6_temp[2] = _mm256_setzero_pd();
  n_6_temp[3] = _mm256_setzero_pd();
  n_6_temp[4] = _mm256_setzero_pd();
  n_6_temp[5] = _mm256_setzero_pd();
  n_6_temp[6] = _mm256_setzero_pd();
  n_6_temp[7] = _mm256_setzero_pd();
  n_6_temp[8] = _mm256_setzero_pd();
  n_6_temp[9] = _mm256_setzero_pd();
  n_6_temp[10] = _mm256_setzero_pd();
  n_6_temp[11] = _mm256_setzero_pd();
  n_6_temp[12] = _mm256_setzero_pd();
  n_6_temp[13] = _mm256_setzero_pd();
  n_6_temp[14] = _mm256_setzero_pd();
  n_6_temp[15] = _mm256_setzero_pd();
  n_6_temp[16] = _mm256_setzero_pd();
  {
    __m256d n_5_temp[17];
    n_5_temp[0] = _mm256_setzero_pd();
    n_5_temp[1] = _mm256_setzero_pd();
    n_5_temp[2] = _mm256_setzero_pd();
    n_5_temp[3] = _mm256_setzero_pd();
    n_5_temp[4] = _mm256_setzero_pd();
    n_5_temp[5] = _mm256_setzero_pd();
    n_5_temp[6] = _mm256_setzero_pd();
    n_5_temp[7] = _mm256_setzero_pd();
    n_5_temp[8] = _mm256_setzero_pd();
    n_5_temp[9] = _mm256_setzero_pd();
    n_5_temp[10] = _mm256_setzero_pd();
    n_5_temp[11] = _mm256_setzero_pd();
    n_5_temp[12] = _mm256_setzero_pd();
    n_5_temp[13] = _mm256_setzero_pd();
    n_5_temp[14] = _mm256_setzero_pd();
    n_5_temp[15] = _mm256_setzero_pd();
    n_5_temp[16] = _mm256_setzero_pd();
    {
      __m256d n_4_temp[17];
      n_4_temp[0] = _mm256_setzero_pd();
      n_4_temp[1] = _mm256_setzero_pd();
      n_4_temp[2] = _mm256_setzero_pd();
      n_4_temp[3] = _mm256_setzero_pd();
      n_4_temp[4] = _mm256_setzero_pd();
      n_4_temp[5] = _mm256_setzero_pd();
      n_4_temp[6] = _mm256_setzero_pd();
      n_4_temp[7] = _mm256_setzero_pd();
      n_4_temp[8] = _mm256_setzero_pd();
      n_4_temp[9] = _mm256_setzero_pd();
      n_4_temp[10] = _mm256_setzero_pd();
      n_4_temp[11] = _mm256_setzero_pd();
      n_4_temp[12] = _mm256_setzero_pd();
      n_4_temp[13] = _mm256_setzero_pd();
      n_4_temp[14] = _mm256_setzero_pd();
      n_4_temp[15] = _mm256_setzero_pd();
      n_4_temp[16] = _mm256_setzero_pd();
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[5];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][1]));
          __m256d temp_2[5];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][2]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][2]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][2]));
          __m256d temp_3[5];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][3][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][3][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][0], temp_3[4]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][3][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][1], temp_3[4]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][3][2], temp_3[4]));
        }
        {
          __m256d temp_1[6];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][3][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][3][4], matrix[1][5][1]));
          __m256d temp_2[6];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][3]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][3]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][3]));
          __m256d temp_3[6];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][4][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][0], temp_3[5]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][1], temp_3[5]));
        }
        {
          __m256d temp_1[7];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][3][0], matrix[1][4][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][0], matrix[1][4][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][0], matrix[1][4][2]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][1], matrix[1][4][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][1], matrix[1][4][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][1], matrix[1][4][2]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][2], matrix[1][4][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][2], matrix[1][4][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][2], matrix[1][4][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][3], matrix[1][4][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][3], matrix[1][4][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][3][3], matrix[1][4][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][4], matrix[1][4][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][3][4], matrix[1][4][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][3][4], matrix[1][4][2]));
          __m256d temp_2[7];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][3][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][3][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][0], matrix[1][3][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][0], matrix[1][3][3]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][3][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][3][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][1], matrix[1][3][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][1], matrix[1][3][3]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][3][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][3][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][2], matrix[1][3][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][2], matrix[1][3][3]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][3][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][3][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][3], matrix[1][3][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][3], matrix[1][3][3]));
          __m256d temp_3[7];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
        }
        n_4_temp[0] = _mm256_sub_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][2][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][2][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][2][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][2][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][2][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][2][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][2][0], n_3_temp[6]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][2][1], n_3_temp[0]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][2][1], n_3_temp[1]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][2][1], n_3_temp[2]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][2][1], n_3_temp[3]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][2][1], n_3_temp[4]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][2][1], n_3_temp[5]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][2][1], n_3_temp[6]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][2][2], n_3_temp[0]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][2][2], n_3_temp[1]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][2][2], n_3_temp[2]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][2][2], n_3_temp[3]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][2][2], n_3_temp[4]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][2][2], n_3_temp[5]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][2][2], n_3_temp[6]));
      }
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[5];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][1]));
          __m256d temp_2[5];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][2]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][2]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][2]));
          __m256d temp_3[5];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][2][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][2][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][2][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][0], temp_3[4]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][2][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][2][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][1], temp_3[4]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][2][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][2][2], temp_3[4]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][3], temp_3[0]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][3], temp_3[1]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][3], temp_3[2]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][2][3], temp_3[3]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][2][3], temp_3[4]));
        }
        {
          __m256d temp_1[7];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][2][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][4], matrix[1][5][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][5], matrix[1][5][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][2][5], matrix[1][5][1]));
          __m256d temp_2[7];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][4]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][4]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][4]));
          __m256d temp_3[7];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][4][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][0], temp_3[5]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][0], temp_3[6]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][1], temp_3[5]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][4][1], temp_3[6]));
        }
        {
          __m256d temp_1[8];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][2][0], matrix[1][4][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][0], matrix[1][4][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][0], matrix[1][4][2]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][1], matrix[1][4][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][1], matrix[1][4][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][1], matrix[1][4][2]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][2], matrix[1][4][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][2], matrix[1][4][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][2], matrix[1][4][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][3], matrix[1][4][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][3], matrix[1][4][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][3], matrix[1][4][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][4], matrix[1][4][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][4], matrix[1][4][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][2][4], matrix[1][4][2]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][5], matrix[1][4][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][2][5], matrix[1][4][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][2][5], matrix[1][4][2]));
          __m256d temp_2[8];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][2][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][2][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][0], matrix[1][2][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][0], matrix[1][2][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][0], matrix[1][2][4]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][2][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][2][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][1], matrix[1][2][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][1], matrix[1][2][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][1], matrix[1][2][4]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][2][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][2][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][2], matrix[1][2][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][2], matrix[1][2][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][2], matrix[1][2][4]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][2][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][2][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][3], matrix[1][2][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][3], matrix[1][2][3]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][4][3], matrix[1][2][4]));
          __m256d temp_3[8];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][5][0], temp_3[7]));
        }
        n_4_temp[0] = _mm256_add_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][3][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_add_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][3][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_add_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][3][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_add_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][3][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_add_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][3][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_add_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][3][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_add_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][3][0], n_3_temp[6]));
        n_4_temp[7] = _mm256_add_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][3][0], n_3_temp[7]));
        n_4_temp[1] = _mm256_add_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][3][1], n_3_temp[0]));
        n_4_temp[2] = _mm256_add_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][3][1], n_3_temp[1]));
        n_4_temp[3] = _mm256_add_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][3][1], n_3_temp[2]));
        n_4_temp[4] = _mm256_add_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][3][1], n_3_temp[3]));
        n_4_temp[5] = _mm256_add_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][3][1], n_3_temp[4]));
        n_4_temp[6] = _mm256_add_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][3][1], n_3_temp[5]));
        n_4_temp[7] = _mm256_add_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][3][1], n_3_temp[6]));
        n_4_temp[8] = _mm256_add_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][3][1], n_3_temp[7]));
      }
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[6];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][3][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][3][4], matrix[1][5][1]));
          __m256d temp_2[6];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][3]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][3]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][3]));
          __m256d temp_3[6];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][2][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][2][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][2][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][0], temp_3[5]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][2][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][2][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][1], temp_3[4]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][2][1], temp_3[5]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][2][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][2][2], temp_3[4]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][2][2], temp_3[5]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][3], temp_3[0]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][3], temp_3[1]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][3], temp_3[2]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][2][3], temp_3[3]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][2][3], temp_3[4]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][2][3], temp_3[5]));
        }
        {
          __m256d temp_1[7];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][2][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][4], matrix[1][5][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][5], matrix[1][5][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][2][5], matrix[1][5][1]));
          __m256d temp_2[7];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][4]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][4]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][4]));
          __m256d temp_3[7];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][3][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][3][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][0], temp_3[5]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][3][0], temp_3[6]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][3][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][3][1], temp_3[5]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][3][1], temp_3[6]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][2], temp_3[0]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][2], temp_3[1]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][2], temp_3[2]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][2], temp_3[3]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][3][2], temp_3[4]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][3][2], temp_3[5]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][3][2], temp_3[6]));
        }
        {
          __m256d temp_1[9];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][2][0], matrix[1][3][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][0], matrix[1][3][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][0], matrix[1][3][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][0], matrix[1][3][3]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][1], matrix[1][3][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][1], matrix[1][3][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][1], matrix[1][3][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][1], matrix[1][3][3]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][2], matrix[1][3][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][2], matrix[1][3][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][2], matrix[1][3][2]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][2], matrix[1][3][3]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][3], matrix[1][3][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][3], matrix[1][3][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][3], matrix[1][3][2]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][2][3], matrix[1][3][3]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][4], matrix[1][3][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][4], matrix[1][3][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][2][4], matrix[1][3][2]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][2][4], matrix[1][3][3]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][5], matrix[1][3][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][2][5], matrix[1][3][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][2][5], matrix[1][3][2]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][2][5], matrix[1][3][3]));
          __m256d temp_2[9];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][3][0], matrix[1][2][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][3][0], matrix[1][2][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][3][0], matrix[1][2][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][3][0], matrix[1][2][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][0], matrix[1][2][4]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][3][1], matrix[1][2][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][3][1], matrix[1][2][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][3][1], matrix[1][2][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][1], matrix[1][2][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][1], matrix[1][2][4]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][3][2], matrix[1][2][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][3][2], matrix[1][2][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][2], matrix[1][2][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][2], matrix[1][2][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][3][2], matrix[1][2][4]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][3][3], matrix[1][2][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][3], matrix[1][2][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][3], matrix[1][2][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][3][3], matrix[1][2][3]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][3][3], matrix[1][2][4]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][4], matrix[1][2][0]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][4], matrix[1][2][1]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][3][4], matrix[1][2][2]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][3][4], matrix[1][2][3]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][3][4], matrix[1][2][4]));
          __m256d temp_3[9];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][5][0], temp_3[7]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][5][0], temp_3[8]));
        }
        n_4_temp[0] = _mm256_sub_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][4][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][4][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][4][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][4][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][4][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][4][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][4][0], n_3_temp[6]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][4][0], n_3_temp[7]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][4][0], n_3_temp[8]));
      }
      n_5_temp[0] = _mm256_add_pd(n_5_temp[0],
                                  _mm256_mul_pd(matrix[4][1][0], n_4_temp[0]));
      n_5_temp[1] = _mm256_add_pd(n_5_temp[1],
                                  _mm256_mul_pd(matrix[4][1][0], n_4_temp[1]));
      n_5_temp[2] = _mm256_add_pd(n_5_temp[2],
                                  _mm256_mul_pd(matrix[4][1][0], n_4_temp[2]));
      n_5_temp[3] = _mm256_add_pd(n_5_temp[3],
                                  _mm256_mul_pd(matrix[4][1][0], n_4_temp[3]));
      n_5_temp[4] = _mm256_add_pd(n_5_temp[4],
                                  _mm256_mul_pd(matrix[4][1][0], n_4_temp[4]));
      n_5_temp[5] = _mm256_add_pd(n_5_temp[5],
                                  _mm256_mul_pd(matrix[4][1][0], n_4_temp[5]));
      n_5_temp[6] = _mm256_add_pd(n_5_temp[6],
                                  _mm256_mul_pd(matrix[4][1][0], n_4_temp[6]));
      n_5_temp[7] = _mm256_add_pd(n_5_temp[7],
                                  _mm256_mul_pd(matrix[4][1][0], n_4_temp[7]));
      n_5_temp[8] = _mm256_add_pd(n_5_temp[8],
                                  _mm256_mul_pd(matrix[4][1][0], n_4_temp[8]));
      n_5_temp[1] = _mm256_add_pd(n_5_temp[1],
                                  _mm256_mul_pd(matrix[4][1][1], n_4_temp[0]));
      n_5_temp[2] = _mm256_add_pd(n_5_temp[2],
                                  _mm256_mul_pd(matrix[4][1][1], n_4_temp[1]));
      n_5_temp[3] = _mm256_add_pd(n_5_temp[3],
                                  _mm256_mul_pd(matrix[4][1][1], n_4_temp[2]));
      n_5_temp[4] = _mm256_add_pd(n_5_temp[4],
                                  _mm256_mul_pd(matrix[4][1][1], n_4_temp[3]));
      n_5_temp[5] = _mm256_add_pd(n_5_temp[5],
                                  _mm256_mul_pd(matrix[4][1][1], n_4_temp[4]));
      n_5_temp[6] = _mm256_add_pd(n_5_temp[6],
                                  _mm256_mul_pd(matrix[4][1][1], n_4_temp[5]));
      n_5_temp[7] = _mm256_add_pd(n_5_temp[7],
                                  _mm256_mul_pd(matrix[4][1][1], n_4_temp[6]));
      n_5_temp[8] = _mm256_add_pd(n_5_temp[8],
                                  _mm256_mul_pd(matrix[4][1][1], n_4_temp[7]));
      n_5_temp[9] = _mm256_add_pd(n_5_temp[9],
                                  _mm256_mul_pd(matrix[4][1][1], n_4_temp[8]));
      n_5_temp[2] = _mm256_add_pd(n_5_temp[2],
                                  _mm256_mul_pd(matrix[4][1][2], n_4_temp[0]));
      n_5_temp[3] = _mm256_add_pd(n_5_temp[3],
                                  _mm256_mul_pd(matrix[4][1][2], n_4_temp[1]));
      n_5_temp[4] = _mm256_add_pd(n_5_temp[4],
                                  _mm256_mul_pd(matrix[4][1][2], n_4_temp[2]));
      n_5_temp[5] = _mm256_add_pd(n_5_temp[5],
                                  _mm256_mul_pd(matrix[4][1][2], n_4_temp[3]));
      n_5_temp[6] = _mm256_add_pd(n_5_temp[6],
                                  _mm256_mul_pd(matrix[4][1][2], n_4_temp[4]));
      n_5_temp[7] = _mm256_add_pd(n_5_temp[7],
                                  _mm256_mul_pd(matrix[4][1][2], n_4_temp[5]));
      n_5_temp[8] = _mm256_add_pd(n_5_temp[8],
                                  _mm256_mul_pd(matrix[4][1][2], n_4_temp[6]));
      n_5_temp[9] = _mm256_add_pd(n_5_temp[9],
                                  _mm256_mul_pd(matrix[4][1][2], n_4_temp[7]));
      n_5_temp[10] = _mm256_add_pd(n_5_temp[10],
                                   _mm256_mul_pd(matrix[4][1][2], n_4_temp[8]));
    }
    {
      __m256d n_4_temp[17];
      n_4_temp[0] = _mm256_setzero_pd();
      n_4_temp[1] = _mm256_setzero_pd();
      n_4_temp[2] = _mm256_setzero_pd();
      n_4_temp[3] = _mm256_setzero_pd();
      n_4_temp[4] = _mm256_setzero_pd();
      n_4_temp[5] = _mm256_setzero_pd();
      n_4_temp[6] = _mm256_setzero_pd();
      n_4_temp[7] = _mm256_setzero_pd();
      n_4_temp[8] = _mm256_setzero_pd();
      n_4_temp[9] = _mm256_setzero_pd();
      n_4_temp[10] = _mm256_setzero_pd();
      n_4_temp[11] = _mm256_setzero_pd();
      n_4_temp[12] = _mm256_setzero_pd();
      n_4_temp[13] = _mm256_setzero_pd();
      n_4_temp[14] = _mm256_setzero_pd();
      n_4_temp[15] = _mm256_setzero_pd();
      n_4_temp[16] = _mm256_setzero_pd();
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[5];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][1]));
          __m256d temp_2[5];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][2]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][2]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][2]));
          __m256d temp_3[5];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][3][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][3][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][0], temp_3[4]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][3][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][1], temp_3[4]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][3][2], temp_3[4]));
        }
        {
          __m256d temp_1[6];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][3][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][3][4], matrix[1][5][1]));
          __m256d temp_2[6];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][3]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][3]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][3]));
          __m256d temp_3[6];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][4][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][0], temp_3[5]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][1], temp_3[5]));
        }
        {
          __m256d temp_1[7];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][3][0], matrix[1][4][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][0], matrix[1][4][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][0], matrix[1][4][2]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][1], matrix[1][4][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][1], matrix[1][4][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][1], matrix[1][4][2]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][2], matrix[1][4][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][2], matrix[1][4][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][2], matrix[1][4][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][3], matrix[1][4][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][3], matrix[1][4][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][3][3], matrix[1][4][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][4], matrix[1][4][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][3][4], matrix[1][4][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][3][4], matrix[1][4][2]));
          __m256d temp_2[7];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][3][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][3][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][0], matrix[1][3][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][0], matrix[1][3][3]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][3][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][3][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][1], matrix[1][3][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][1], matrix[1][3][3]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][3][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][3][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][2], matrix[1][3][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][2], matrix[1][3][3]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][3][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][3][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][3], matrix[1][3][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][3], matrix[1][3][3]));
          __m256d temp_3[7];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
        }
        n_4_temp[0] = _mm256_sub_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][1][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][1][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][1][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][1][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][1][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][1][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][1][0], n_3_temp[6]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][1][1], n_3_temp[0]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][1][1], n_3_temp[1]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][1][1], n_3_temp[2]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][1][1], n_3_temp[3]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][1][1], n_3_temp[4]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][1][1], n_3_temp[5]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][1][1], n_3_temp[6]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][1][2], n_3_temp[0]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][1][2], n_3_temp[1]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][1][2], n_3_temp[2]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][1][2], n_3_temp[3]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][1][2], n_3_temp[4]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][1][2], n_3_temp[5]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][1][2], n_3_temp[6]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][1][3], n_3_temp[0]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][1][3], n_3_temp[1]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][1][3], n_3_temp[2]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][1][3], n_3_temp[3]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][1][3], n_3_temp[4]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][1][3], n_3_temp[5]));
        n_4_temp[9] = _mm256_sub_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][1][3], n_3_temp[6]));
      }
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[5];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][1]));
          __m256d temp_2[5];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][2]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][2]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][2]));
          __m256d temp_3[5];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][1][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][1][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][1][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][0], temp_3[4]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][1][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][1][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][1], temp_3[4]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][1][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][2], temp_3[4]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][3], temp_3[0]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][3], temp_3[1]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][3], temp_3[2]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][3], temp_3[3]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][1][3], temp_3[4]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][4], temp_3[0]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][4], temp_3[1]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][4], temp_3[2]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][1][4], temp_3[3]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][1][4], temp_3[4]));
        }
        {
          __m256d temp_1[8];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][1][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][4], matrix[1][5][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][5], matrix[1][5][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][5], matrix[1][5][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][6], matrix[1][5][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][6], matrix[1][5][1]));
          __m256d temp_2[8];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][5]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][5]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][5]));
          __m256d temp_3[8];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][4][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][0], temp_3[5]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][0], temp_3[6]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][4][0], temp_3[7]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][1], temp_3[5]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][4][1], temp_3[6]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][4][1], temp_3[7]));
        }
        {
          __m256d temp_1[9];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][1][0], matrix[1][4][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][0], matrix[1][4][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][0], matrix[1][4][2]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][1], matrix[1][4][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][1], matrix[1][4][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][1], matrix[1][4][2]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][2], matrix[1][4][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][2], matrix[1][4][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][2], matrix[1][4][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][3], matrix[1][4][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][3], matrix[1][4][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][3], matrix[1][4][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][4], matrix[1][4][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][4], matrix[1][4][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][4], matrix[1][4][2]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][5], matrix[1][4][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][5], matrix[1][4][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][5], matrix[1][4][2]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][6], matrix[1][4][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][6], matrix[1][4][1]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][1][6], matrix[1][4][2]));
          __m256d temp_2[9];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][1][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][1][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][0], matrix[1][1][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][0], matrix[1][1][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][0], matrix[1][1][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][0], matrix[1][1][5]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][1][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][1][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][1], matrix[1][1][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][1], matrix[1][1][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][1], matrix[1][1][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][1], matrix[1][1][5]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][1][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][1][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][2], matrix[1][1][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][2], matrix[1][1][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][2], matrix[1][1][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][4][2], matrix[1][1][5]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][1][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][1][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][3], matrix[1][1][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][3], matrix[1][1][3]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][4][3], matrix[1][1][4]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][4][3], matrix[1][1][5]));
          __m256d temp_3[9];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][5][0], temp_3[7]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][5][0], temp_3[8]));
        }
        n_4_temp[0] = _mm256_add_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][3][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_add_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][3][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_add_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][3][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_add_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][3][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_add_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][3][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_add_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][3][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_add_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][3][0], n_3_temp[6]));
        n_4_temp[7] = _mm256_add_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][3][0], n_3_temp[7]));
        n_4_temp[8] = _mm256_add_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][3][0], n_3_temp[8]));
        n_4_temp[1] = _mm256_add_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][3][1], n_3_temp[0]));
        n_4_temp[2] = _mm256_add_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][3][1], n_3_temp[1]));
        n_4_temp[3] = _mm256_add_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][3][1], n_3_temp[2]));
        n_4_temp[4] = _mm256_add_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][3][1], n_3_temp[3]));
        n_4_temp[5] = _mm256_add_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][3][1], n_3_temp[4]));
        n_4_temp[6] = _mm256_add_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][3][1], n_3_temp[5]));
        n_4_temp[7] = _mm256_add_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][3][1], n_3_temp[6]));
        n_4_temp[8] = _mm256_add_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][3][1], n_3_temp[7]));
        n_4_temp[9] = _mm256_add_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][3][1], n_3_temp[8]));
      }
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[6];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][3][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][3][4], matrix[1][5][1]));
          __m256d temp_2[6];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][3]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][3]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][3]));
          __m256d temp_3[6];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][1][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][1][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][1][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][0], temp_3[5]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][1][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][1][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][1], temp_3[4]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][1], temp_3[5]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][1][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][2], temp_3[4]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][1][2], temp_3[5]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][3], temp_3[0]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][3], temp_3[1]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][3], temp_3[2]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][3], temp_3[3]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][1][3], temp_3[4]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][1][3], temp_3[5]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][4], temp_3[0]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][4], temp_3[1]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][4], temp_3[2]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][1][4], temp_3[3]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][1][4], temp_3[4]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][1][4], temp_3[5]));
        }
        {
          __m256d temp_1[8];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][1][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][4], matrix[1][5][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][5], matrix[1][5][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][5], matrix[1][5][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][6], matrix[1][5][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][6], matrix[1][5][1]));
          __m256d temp_2[8];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][5]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][5]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][5]));
          __m256d temp_3[8];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][3][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][3][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][0], temp_3[5]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][3][0], temp_3[6]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][3][0], temp_3[7]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][3][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][3][1], temp_3[5]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][3][1], temp_3[6]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][3][1], temp_3[7]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][2], temp_3[0]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][2], temp_3[1]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][2], temp_3[2]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][2], temp_3[3]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][3][2], temp_3[4]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][3][2], temp_3[5]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][3][2], temp_3[6]));
          n_3_temp[9] = _mm256_sub_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][3][2], temp_3[7]));
        }
        {
          __m256d temp_1[10];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[9] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][1][0], matrix[1][3][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][0], matrix[1][3][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][0], matrix[1][3][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][0], matrix[1][3][3]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][1], matrix[1][3][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][1], matrix[1][3][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][1], matrix[1][3][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][1], matrix[1][3][3]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][2], matrix[1][3][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][2], matrix[1][3][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][2], matrix[1][3][2]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][2], matrix[1][3][3]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][3], matrix[1][3][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][3], matrix[1][3][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][3], matrix[1][3][2]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][3], matrix[1][3][3]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][4], matrix[1][3][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][4], matrix[1][3][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][4], matrix[1][3][2]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][4], matrix[1][3][3]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][5], matrix[1][3][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][5], matrix[1][3][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][5], matrix[1][3][2]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][1][5], matrix[1][3][3]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][6], matrix[1][3][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][6], matrix[1][3][1]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][1][6], matrix[1][3][2]));
          temp_1[9] = _mm256_add_pd(
              temp_1[9], _mm256_mul_pd(matrix[0][1][6], matrix[1][3][3]));
          __m256d temp_2[10];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[9] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][3][0], matrix[1][1][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][3][0], matrix[1][1][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][3][0], matrix[1][1][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][3][0], matrix[1][1][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][0], matrix[1][1][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][0], matrix[1][1][5]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][3][1], matrix[1][1][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][3][1], matrix[1][1][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][3][1], matrix[1][1][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][1], matrix[1][1][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][1], matrix[1][1][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][3][1], matrix[1][1][5]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][3][2], matrix[1][1][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][3][2], matrix[1][1][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][2], matrix[1][1][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][2], matrix[1][1][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][3][2], matrix[1][1][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][3][2], matrix[1][1][5]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][3][3], matrix[1][1][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][3], matrix[1][1][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][3], matrix[1][1][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][3][3], matrix[1][1][3]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][3][3], matrix[1][1][4]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][3][3], matrix[1][1][5]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][4], matrix[1][1][0]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][4], matrix[1][1][1]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][3][4], matrix[1][1][2]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][3][4], matrix[1][1][3]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][3][4], matrix[1][1][4]));
          temp_2[9] = _mm256_add_pd(
              temp_2[9], _mm256_mul_pd(matrix[0][3][4], matrix[1][1][5]));
          __m256d temp_3[10];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[9] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[9] = _mm256_add_pd(temp_3[9], temp_1[9]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          temp_3[9] = _mm256_sub_pd(temp_3[9], temp_2[9]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][5][0], temp_3[7]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][5][0], temp_3[8]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][5][0], temp_3[9]));
        }
        n_4_temp[0] = _mm256_sub_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][4][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][4][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][4][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][4][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][4][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][4][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][4][0], n_3_temp[6]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][4][0], n_3_temp[7]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][4][0], n_3_temp[8]));
        n_4_temp[9] = _mm256_sub_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][4][0], n_3_temp[9]));
      }
      n_5_temp[0] = _mm256_sub_pd(n_5_temp[0],
                                  _mm256_mul_pd(matrix[4][2][0], n_4_temp[0]));
      n_5_temp[1] = _mm256_sub_pd(n_5_temp[1],
                                  _mm256_mul_pd(matrix[4][2][0], n_4_temp[1]));
      n_5_temp[2] = _mm256_sub_pd(n_5_temp[2],
                                  _mm256_mul_pd(matrix[4][2][0], n_4_temp[2]));
      n_5_temp[3] = _mm256_sub_pd(n_5_temp[3],
                                  _mm256_mul_pd(matrix[4][2][0], n_4_temp[3]));
      n_5_temp[4] = _mm256_sub_pd(n_5_temp[4],
                                  _mm256_mul_pd(matrix[4][2][0], n_4_temp[4]));
      n_5_temp[5] = _mm256_sub_pd(n_5_temp[5],
                                  _mm256_mul_pd(matrix[4][2][0], n_4_temp[5]));
      n_5_temp[6] = _mm256_sub_pd(n_5_temp[6],
                                  _mm256_mul_pd(matrix[4][2][0], n_4_temp[6]));
      n_5_temp[7] = _mm256_sub_pd(n_5_temp[7],
                                  _mm256_mul_pd(matrix[4][2][0], n_4_temp[7]));
      n_5_temp[8] = _mm256_sub_pd(n_5_temp[8],
                                  _mm256_mul_pd(matrix[4][2][0], n_4_temp[8]));
      n_5_temp[9] = _mm256_sub_pd(n_5_temp[9],
                                  _mm256_mul_pd(matrix[4][2][0], n_4_temp[9]));
      n_5_temp[1] = _mm256_sub_pd(n_5_temp[1],
                                  _mm256_mul_pd(matrix[4][2][1], n_4_temp[0]));
      n_5_temp[2] = _mm256_sub_pd(n_5_temp[2],
                                  _mm256_mul_pd(matrix[4][2][1], n_4_temp[1]));
      n_5_temp[3] = _mm256_sub_pd(n_5_temp[3],
                                  _mm256_mul_pd(matrix[4][2][1], n_4_temp[2]));
      n_5_temp[4] = _mm256_sub_pd(n_5_temp[4],
                                  _mm256_mul_pd(matrix[4][2][1], n_4_temp[3]));
      n_5_temp[5] = _mm256_sub_pd(n_5_temp[5],
                                  _mm256_mul_pd(matrix[4][2][1], n_4_temp[4]));
      n_5_temp[6] = _mm256_sub_pd(n_5_temp[6],
                                  _mm256_mul_pd(matrix[4][2][1], n_4_temp[5]));
      n_5_temp[7] = _mm256_sub_pd(n_5_temp[7],
                                  _mm256_mul_pd(matrix[4][2][1], n_4_temp[6]));
      n_5_temp[8] = _mm256_sub_pd(n_5_temp[8],
                                  _mm256_mul_pd(matrix[4][2][1], n_4_temp[7]));
      n_5_temp[9] = _mm256_sub_pd(n_5_temp[9],
                                  _mm256_mul_pd(matrix[4][2][1], n_4_temp[8]));
      n_5_temp[10] = _mm256_sub_pd(n_5_temp[10],
                                   _mm256_mul_pd(matrix[4][2][1], n_4_temp[9]));
    }
    {
      __m256d n_4_temp[17];
      n_4_temp[0] = _mm256_setzero_pd();
      n_4_temp[1] = _mm256_setzero_pd();
      n_4_temp[2] = _mm256_setzero_pd();
      n_4_temp[3] = _mm256_setzero_pd();
      n_4_temp[4] = _mm256_setzero_pd();
      n_4_temp[5] = _mm256_setzero_pd();
      n_4_temp[6] = _mm256_setzero_pd();
      n_4_temp[7] = _mm256_setzero_pd();
      n_4_temp[8] = _mm256_setzero_pd();
      n_4_temp[9] = _mm256_setzero_pd();
      n_4_temp[10] = _mm256_setzero_pd();
      n_4_temp[11] = _mm256_setzero_pd();
      n_4_temp[12] = _mm256_setzero_pd();
      n_4_temp[13] = _mm256_setzero_pd();
      n_4_temp[14] = _mm256_setzero_pd();
      n_4_temp[15] = _mm256_setzero_pd();
      n_4_temp[16] = _mm256_setzero_pd();
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[5];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][1]));
          __m256d temp_2[5];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][2]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][2]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][2]));
          __m256d temp_3[5];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][2][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][2][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][2][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][0], temp_3[4]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][2][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][2][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][1], temp_3[4]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][2][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][2][2], temp_3[4]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][3], temp_3[0]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][3], temp_3[1]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][3], temp_3[2]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][2][3], temp_3[3]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][2][3], temp_3[4]));
        }
        {
          __m256d temp_1[7];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][2][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][4], matrix[1][5][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][5], matrix[1][5][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][2][5], matrix[1][5][1]));
          __m256d temp_2[7];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][4]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][4]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][4]));
          __m256d temp_3[7];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][4][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][0], temp_3[5]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][0], temp_3[6]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][1], temp_3[5]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][4][1], temp_3[6]));
        }
        {
          __m256d temp_1[8];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][2][0], matrix[1][4][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][0], matrix[1][4][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][0], matrix[1][4][2]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][1], matrix[1][4][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][1], matrix[1][4][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][1], matrix[1][4][2]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][2], matrix[1][4][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][2], matrix[1][4][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][2], matrix[1][4][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][3], matrix[1][4][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][3], matrix[1][4][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][3], matrix[1][4][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][4], matrix[1][4][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][4], matrix[1][4][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][2][4], matrix[1][4][2]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][5], matrix[1][4][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][2][5], matrix[1][4][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][2][5], matrix[1][4][2]));
          __m256d temp_2[8];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][2][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][2][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][0], matrix[1][2][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][0], matrix[1][2][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][0], matrix[1][2][4]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][2][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][2][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][1], matrix[1][2][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][1], matrix[1][2][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][1], matrix[1][2][4]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][2][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][2][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][2], matrix[1][2][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][2], matrix[1][2][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][2], matrix[1][2][4]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][2][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][2][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][3], matrix[1][2][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][3], matrix[1][2][3]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][4][3], matrix[1][2][4]));
          __m256d temp_3[8];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][5][0], temp_3[7]));
        }
        n_4_temp[0] = _mm256_sub_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][1][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][1][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][1][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][1][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][1][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][1][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][1][0], n_3_temp[6]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][1][0], n_3_temp[7]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][1][1], n_3_temp[0]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][1][1], n_3_temp[1]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][1][1], n_3_temp[2]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][1][1], n_3_temp[3]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][1][1], n_3_temp[4]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][1][1], n_3_temp[5]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][1][1], n_3_temp[6]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][1][1], n_3_temp[7]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][1][2], n_3_temp[0]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][1][2], n_3_temp[1]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][1][2], n_3_temp[2]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][1][2], n_3_temp[3]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][1][2], n_3_temp[4]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][1][2], n_3_temp[5]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][1][2], n_3_temp[6]));
        n_4_temp[9] = _mm256_sub_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][1][2], n_3_temp[7]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][1][3], n_3_temp[0]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][1][3], n_3_temp[1]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][1][3], n_3_temp[2]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][1][3], n_3_temp[3]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][1][3], n_3_temp[4]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][1][3], n_3_temp[5]));
        n_4_temp[9] = _mm256_sub_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][1][3], n_3_temp[6]));
        n_4_temp[10] = _mm256_sub_pd(
            n_4_temp[10], _mm256_mul_pd(matrix[3][1][3], n_3_temp[7]));
      }
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[5];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][1]));
          __m256d temp_2[5];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][2]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][2]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][2]));
          __m256d temp_3[5];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][1][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][1][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][1][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][0], temp_3[4]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][1][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][1][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][1], temp_3[4]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][1][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][2], temp_3[4]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][3], temp_3[0]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][3], temp_3[1]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][3], temp_3[2]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][3], temp_3[3]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][1][3], temp_3[4]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][4], temp_3[0]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][4], temp_3[1]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][4], temp_3[2]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][1][4], temp_3[3]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][1][4], temp_3[4]));
        }
        {
          __m256d temp_1[8];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][1][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][4], matrix[1][5][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][5], matrix[1][5][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][5], matrix[1][5][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][6], matrix[1][5][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][6], matrix[1][5][1]));
          __m256d temp_2[8];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][5]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][5]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][5]));
          __m256d temp_3[8];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][4][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][0], temp_3[5]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][0], temp_3[6]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][4][0], temp_3[7]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][1], temp_3[5]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][4][1], temp_3[6]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][4][1], temp_3[7]));
        }
        {
          __m256d temp_1[9];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][1][0], matrix[1][4][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][0], matrix[1][4][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][0], matrix[1][4][2]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][1], matrix[1][4][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][1], matrix[1][4][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][1], matrix[1][4][2]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][2], matrix[1][4][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][2], matrix[1][4][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][2], matrix[1][4][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][3], matrix[1][4][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][3], matrix[1][4][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][3], matrix[1][4][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][4], matrix[1][4][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][4], matrix[1][4][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][4], matrix[1][4][2]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][5], matrix[1][4][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][5], matrix[1][4][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][5], matrix[1][4][2]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][6], matrix[1][4][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][6], matrix[1][4][1]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][1][6], matrix[1][4][2]));
          __m256d temp_2[9];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][1][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][1][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][0], matrix[1][1][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][0], matrix[1][1][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][0], matrix[1][1][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][0], matrix[1][1][5]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][1][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][1][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][1], matrix[1][1][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][1], matrix[1][1][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][1], matrix[1][1][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][1], matrix[1][1][5]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][1][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][1][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][2], matrix[1][1][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][2], matrix[1][1][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][2], matrix[1][1][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][4][2], matrix[1][1][5]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][1][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][1][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][3], matrix[1][1][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][3], matrix[1][1][3]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][4][3], matrix[1][1][4]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][4][3], matrix[1][1][5]));
          __m256d temp_3[9];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][5][0], temp_3[7]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][5][0], temp_3[8]));
        }
        n_4_temp[0] = _mm256_add_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][2][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_add_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][2][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_add_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][2][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_add_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][2][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_add_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][2][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_add_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][2][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_add_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][2][0], n_3_temp[6]));
        n_4_temp[7] = _mm256_add_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][2][0], n_3_temp[7]));
        n_4_temp[8] = _mm256_add_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][2][0], n_3_temp[8]));
        n_4_temp[1] = _mm256_add_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][2][1], n_3_temp[0]));
        n_4_temp[2] = _mm256_add_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][2][1], n_3_temp[1]));
        n_4_temp[3] = _mm256_add_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][2][1], n_3_temp[2]));
        n_4_temp[4] = _mm256_add_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][2][1], n_3_temp[3]));
        n_4_temp[5] = _mm256_add_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][2][1], n_3_temp[4]));
        n_4_temp[6] = _mm256_add_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][2][1], n_3_temp[5]));
        n_4_temp[7] = _mm256_add_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][2][1], n_3_temp[6]));
        n_4_temp[8] = _mm256_add_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][2][1], n_3_temp[7]));
        n_4_temp[9] = _mm256_add_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][2][1], n_3_temp[8]));
        n_4_temp[2] = _mm256_add_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][2][2], n_3_temp[0]));
        n_4_temp[3] = _mm256_add_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][2][2], n_3_temp[1]));
        n_4_temp[4] = _mm256_add_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][2][2], n_3_temp[2]));
        n_4_temp[5] = _mm256_add_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][2][2], n_3_temp[3]));
        n_4_temp[6] = _mm256_add_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][2][2], n_3_temp[4]));
        n_4_temp[7] = _mm256_add_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][2][2], n_3_temp[5]));
        n_4_temp[8] = _mm256_add_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][2][2], n_3_temp[6]));
        n_4_temp[9] = _mm256_add_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][2][2], n_3_temp[7]));
        n_4_temp[10] = _mm256_add_pd(
            n_4_temp[10], _mm256_mul_pd(matrix[3][2][2], n_3_temp[8]));
      }
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[7];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][2][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][4], matrix[1][5][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][5], matrix[1][5][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][2][5], matrix[1][5][1]));
          __m256d temp_2[7];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][4]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][4]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][4]));
          __m256d temp_3[7];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][1][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][1][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][1][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][0], temp_3[6]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][1][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][1][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][1], temp_3[4]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][1], temp_3[5]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][1][1], temp_3[6]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][1][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][2], temp_3[4]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][1][2], temp_3[5]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][1][2], temp_3[6]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][3], temp_3[0]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][3], temp_3[1]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][3], temp_3[2]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][3], temp_3[3]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][1][3], temp_3[4]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][1][3], temp_3[5]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][1][3], temp_3[6]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][4], temp_3[0]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][4], temp_3[1]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][4], temp_3[2]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][1][4], temp_3[3]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][1][4], temp_3[4]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][1][4], temp_3[5]));
          n_3_temp[10] = _mm256_add_pd(
              n_3_temp[10], _mm256_mul_pd(matrix[2][1][4], temp_3[6]));
        }
        {
          __m256d temp_1[8];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][1][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][4], matrix[1][5][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][5], matrix[1][5][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][5], matrix[1][5][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][6], matrix[1][5][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][6], matrix[1][5][1]));
          __m256d temp_2[8];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][5]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][5]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][5]));
          __m256d temp_3[8];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][2][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][2][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][2][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][0], temp_3[5]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][2][0], temp_3[6]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][2][0], temp_3[7]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][2][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][2][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][2][1], temp_3[5]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][2][1], temp_3[6]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][2][1], temp_3[7]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][2][2], temp_3[0]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][2], temp_3[1]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][2], temp_3[2]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][2], temp_3[3]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][2][2], temp_3[4]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][2][2], temp_3[5]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][2][2], temp_3[6]));
          n_3_temp[9] = _mm256_sub_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][2][2], temp_3[7]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][3], temp_3[0]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][3], temp_3[1]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][3], temp_3[2]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][2][3], temp_3[3]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][2][3], temp_3[4]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][2][3], temp_3[5]));
          n_3_temp[9] = _mm256_sub_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][2][3], temp_3[6]));
          n_3_temp[10] = _mm256_sub_pd(
              n_3_temp[10], _mm256_mul_pd(matrix[2][2][3], temp_3[7]));
        }
        {
          __m256d temp_1[11];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[9] = _mm256_setzero_pd();
          temp_1[10] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][1][0], matrix[1][2][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][0], matrix[1][2][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][0], matrix[1][2][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][0], matrix[1][2][3]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][0], matrix[1][2][4]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][1], matrix[1][2][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][1], matrix[1][2][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][1], matrix[1][2][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][1], matrix[1][2][3]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][1], matrix[1][2][4]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][2], matrix[1][2][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][2], matrix[1][2][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][2], matrix[1][2][2]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][2], matrix[1][2][3]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][2], matrix[1][2][4]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][3], matrix[1][2][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][3], matrix[1][2][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][3], matrix[1][2][2]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][3], matrix[1][2][3]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][3], matrix[1][2][4]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][4], matrix[1][2][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][4], matrix[1][2][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][4], matrix[1][2][2]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][4], matrix[1][2][3]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][1][4], matrix[1][2][4]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][5], matrix[1][2][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][5], matrix[1][2][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][5], matrix[1][2][2]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][1][5], matrix[1][2][3]));
          temp_1[9] = _mm256_add_pd(
              temp_1[9], _mm256_mul_pd(matrix[0][1][5], matrix[1][2][4]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][6], matrix[1][2][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][6], matrix[1][2][1]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][1][6], matrix[1][2][2]));
          temp_1[9] = _mm256_add_pd(
              temp_1[9], _mm256_mul_pd(matrix[0][1][6], matrix[1][2][3]));
          temp_1[10] = _mm256_add_pd(
              temp_1[10], _mm256_mul_pd(matrix[0][1][6], matrix[1][2][4]));
          __m256d temp_2[11];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[9] = _mm256_setzero_pd();
          temp_2[10] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][2][0], matrix[1][1][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][2][0], matrix[1][1][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][2][0], matrix[1][1][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][2][0], matrix[1][1][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][2][0], matrix[1][1][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][2][0], matrix[1][1][5]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][2][1], matrix[1][1][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][2][1], matrix[1][1][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][2][1], matrix[1][1][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][2][1], matrix[1][1][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][2][1], matrix[1][1][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][2][1], matrix[1][1][5]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][2][2], matrix[1][1][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][2][2], matrix[1][1][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][2][2], matrix[1][1][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][2][2], matrix[1][1][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][2][2], matrix[1][1][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][2][2], matrix[1][1][5]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][2][3], matrix[1][1][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][2][3], matrix[1][1][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][2][3], matrix[1][1][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][2][3], matrix[1][1][3]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][2][3], matrix[1][1][4]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][2][3], matrix[1][1][5]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][2][4], matrix[1][1][0]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][2][4], matrix[1][1][1]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][2][4], matrix[1][1][2]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][2][4], matrix[1][1][3]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][2][4], matrix[1][1][4]));
          temp_2[9] = _mm256_add_pd(
              temp_2[9], _mm256_mul_pd(matrix[0][2][4], matrix[1][1][5]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][2][5], matrix[1][1][0]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][2][5], matrix[1][1][1]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][2][5], matrix[1][1][2]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][2][5], matrix[1][1][3]));
          temp_2[9] = _mm256_add_pd(
              temp_2[9], _mm256_mul_pd(matrix[0][2][5], matrix[1][1][4]));
          temp_2[10] = _mm256_add_pd(
              temp_2[10], _mm256_mul_pd(matrix[0][2][5], matrix[1][1][5]));
          __m256d temp_3[11];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[9] = _mm256_setzero_pd();
          temp_3[10] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[9] = _mm256_add_pd(temp_3[9], temp_1[9]);
          temp_3[10] = _mm256_add_pd(temp_3[10], temp_1[10]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          temp_3[9] = _mm256_sub_pd(temp_3[9], temp_2[9]);
          temp_3[10] = _mm256_sub_pd(temp_3[10], temp_2[10]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][5][0], temp_3[7]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][5][0], temp_3[8]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][5][0], temp_3[9]));
          n_3_temp[10] = _mm256_add_pd(
              n_3_temp[10], _mm256_mul_pd(matrix[2][5][0], temp_3[10]));
        }
        n_4_temp[0] = _mm256_sub_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][4][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][4][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][4][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][4][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][4][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][4][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][4][0], n_3_temp[6]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][4][0], n_3_temp[7]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][4][0], n_3_temp[8]));
        n_4_temp[9] = _mm256_sub_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][4][0], n_3_temp[9]));
        n_4_temp[10] = _mm256_sub_pd(
            n_4_temp[10], _mm256_mul_pd(matrix[3][4][0], n_3_temp[10]));
      }
      n_5_temp[0] = _mm256_add_pd(n_5_temp[0],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[0]));
      n_5_temp[1] = _mm256_add_pd(n_5_temp[1],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[1]));
      n_5_temp[2] = _mm256_add_pd(n_5_temp[2],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[2]));
      n_5_temp[3] = _mm256_add_pd(n_5_temp[3],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[3]));
      n_5_temp[4] = _mm256_add_pd(n_5_temp[4],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[4]));
      n_5_temp[5] = _mm256_add_pd(n_5_temp[5],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[5]));
      n_5_temp[6] = _mm256_add_pd(n_5_temp[6],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[6]));
      n_5_temp[7] = _mm256_add_pd(n_5_temp[7],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[7]));
      n_5_temp[8] = _mm256_add_pd(n_5_temp[8],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[8]));
      n_5_temp[9] = _mm256_add_pd(n_5_temp[9],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[9]));
      n_5_temp[10] = _mm256_add_pd(
          n_5_temp[10], _mm256_mul_pd(matrix[4][3][0], n_4_temp[10]));
    }
    n_6_temp[0] =
        _mm256_sub_pd(n_6_temp[0], _mm256_mul_pd(matrix[5][0][0], n_5_temp[0]));
    n_6_temp[1] =
        _mm256_sub_pd(n_6_temp[1], _mm256_mul_pd(matrix[5][0][0], n_5_temp[1]));
    n_6_temp[2] =
        _mm256_sub_pd(n_6_temp[2], _mm256_mul_pd(matrix[5][0][0], n_5_temp[2]));
    n_6_temp[3] =
        _mm256_sub_pd(n_6_temp[3], _mm256_mul_pd(matrix[5][0][0], n_5_temp[3]));
    n_6_temp[4] =
        _mm256_sub_pd(n_6_temp[4], _mm256_mul_pd(matrix[5][0][0], n_5_temp[4]));
    n_6_temp[5] =
        _mm256_sub_pd(n_6_temp[5], _mm256_mul_pd(matrix[5][0][0], n_5_temp[5]));
    n_6_temp[6] =
        _mm256_sub_pd(n_6_temp[6], _mm256_mul_pd(matrix[5][0][0], n_5_temp[6]));
    n_6_temp[7] =
        _mm256_sub_pd(n_6_temp[7], _mm256_mul_pd(matrix[5][0][0], n_5_temp[7]));
    n_6_temp[8] =
        _mm256_sub_pd(n_6_temp[8], _mm256_mul_pd(matrix[5][0][0], n_5_temp[8]));
    n_6_temp[9] =
        _mm256_sub_pd(n_6_temp[9], _mm256_mul_pd(matrix[5][0][0], n_5_temp[9]));
    n_6_temp[10] = _mm256_sub_pd(n_6_temp[10],
                                 _mm256_mul_pd(matrix[5][0][0], n_5_temp[10]));
    n_6_temp[1] =
        _mm256_sub_pd(n_6_temp[1], _mm256_mul_pd(matrix[5][0][1], n_5_temp[0]));
    n_6_temp[2] =
        _mm256_sub_pd(n_6_temp[2], _mm256_mul_pd(matrix[5][0][1], n_5_temp[1]));
    n_6_temp[3] =
        _mm256_sub_pd(n_6_temp[3], _mm256_mul_pd(matrix[5][0][1], n_5_temp[2]));
    n_6_temp[4] =
        _mm256_sub_pd(n_6_temp[4], _mm256_mul_pd(matrix[5][0][1], n_5_temp[3]));
    n_6_temp[5] =
        _mm256_sub_pd(n_6_temp[5], _mm256_mul_pd(matrix[5][0][1], n_5_temp[4]));
    n_6_temp[6] =
        _mm256_sub_pd(n_6_temp[6], _mm256_mul_pd(matrix[5][0][1], n_5_temp[5]));
    n_6_temp[7] =
        _mm256_sub_pd(n_6_temp[7], _mm256_mul_pd(matrix[5][0][1], n_5_temp[6]));
    n_6_temp[8] =
        _mm256_sub_pd(n_6_temp[8], _mm256_mul_pd(matrix[5][0][1], n_5_temp[7]));
    n_6_temp[9] =
        _mm256_sub_pd(n_6_temp[9], _mm256_mul_pd(matrix[5][0][1], n_5_temp[8]));
    n_6_temp[10] = _mm256_sub_pd(n_6_temp[10],
                                 _mm256_mul_pd(matrix[5][0][1], n_5_temp[9]));
    n_6_temp[11] = _mm256_sub_pd(n_6_temp[11],
                                 _mm256_mul_pd(matrix[5][0][1], n_5_temp[10]));
    n_6_temp[2] =
        _mm256_sub_pd(n_6_temp[2], _mm256_mul_pd(matrix[5][0][2], n_5_temp[0]));
    n_6_temp[3] =
        _mm256_sub_pd(n_6_temp[3], _mm256_mul_pd(matrix[5][0][2], n_5_temp[1]));
    n_6_temp[4] =
        _mm256_sub_pd(n_6_temp[4], _mm256_mul_pd(matrix[5][0][2], n_5_temp[2]));
    n_6_temp[5] =
        _mm256_sub_pd(n_6_temp[5], _mm256_mul_pd(matrix[5][0][2], n_5_temp[3]));
    n_6_temp[6] =
        _mm256_sub_pd(n_6_temp[6], _mm256_mul_pd(matrix[5][0][2], n_5_temp[4]));
    n_6_temp[7] =
        _mm256_sub_pd(n_6_temp[7], _mm256_mul_pd(matrix[5][0][2], n_5_temp[5]));
    n_6_temp[8] =
        _mm256_sub_pd(n_6_temp[8], _mm256_mul_pd(matrix[5][0][2], n_5_temp[6]));
    n_6_temp[9] =
        _mm256_sub_pd(n_6_temp[9], _mm256_mul_pd(matrix[5][0][2], n_5_temp[7]));
    n_6_temp[10] = _mm256_sub_pd(n_6_temp[10],
                                 _mm256_mul_pd(matrix[5][0][2], n_5_temp[8]));
    n_6_temp[11] = _mm256_sub_pd(n_6_temp[11],
                                 _mm256_mul_pd(matrix[5][0][2], n_5_temp[9]));
    n_6_temp[12] = _mm256_sub_pd(n_6_temp[12],
                                 _mm256_mul_pd(matrix[5][0][2], n_5_temp[10]));
  }
  {
    __m256d n_5_temp[17];
    n_5_temp[0] = _mm256_setzero_pd();
    n_5_temp[1] = _mm256_setzero_pd();
    n_5_temp[2] = _mm256_setzero_pd();
    n_5_temp[3] = _mm256_setzero_pd();
    n_5_temp[4] = _mm256_setzero_pd();
    n_5_temp[5] = _mm256_setzero_pd();
    n_5_temp[6] = _mm256_setzero_pd();
    n_5_temp[7] = _mm256_setzero_pd();
    n_5_temp[8] = _mm256_setzero_pd();
    n_5_temp[9] = _mm256_setzero_pd();
    n_5_temp[10] = _mm256_setzero_pd();
    n_5_temp[11] = _mm256_setzero_pd();
    n_5_temp[12] = _mm256_setzero_pd();
    n_5_temp[13] = _mm256_setzero_pd();
    n_5_temp[14] = _mm256_setzero_pd();
    n_5_temp[15] = _mm256_setzero_pd();
    n_5_temp[16] = _mm256_setzero_pd();
    {
      __m256d n_4_temp[17];
      n_4_temp[0] = _mm256_setzero_pd();
      n_4_temp[1] = _mm256_setzero_pd();
      n_4_temp[2] = _mm256_setzero_pd();
      n_4_temp[3] = _mm256_setzero_pd();
      n_4_temp[4] = _mm256_setzero_pd();
      n_4_temp[5] = _mm256_setzero_pd();
      n_4_temp[6] = _mm256_setzero_pd();
      n_4_temp[7] = _mm256_setzero_pd();
      n_4_temp[8] = _mm256_setzero_pd();
      n_4_temp[9] = _mm256_setzero_pd();
      n_4_temp[10] = _mm256_setzero_pd();
      n_4_temp[11] = _mm256_setzero_pd();
      n_4_temp[12] = _mm256_setzero_pd();
      n_4_temp[13] = _mm256_setzero_pd();
      n_4_temp[14] = _mm256_setzero_pd();
      n_4_temp[15] = _mm256_setzero_pd();
      n_4_temp[16] = _mm256_setzero_pd();
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[5];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][1]));
          __m256d temp_2[5];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][2]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][2]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][2]));
          __m256d temp_3[5];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][3][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][3][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][0], temp_3[4]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][3][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][1], temp_3[4]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][3][2], temp_3[4]));
        }
        {
          __m256d temp_1[6];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][3][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][3][4], matrix[1][5][1]));
          __m256d temp_2[6];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][3]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][3]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][3]));
          __m256d temp_3[6];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][4][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][0], temp_3[5]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][1], temp_3[5]));
        }
        {
          __m256d temp_1[7];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][3][0], matrix[1][4][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][0], matrix[1][4][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][0], matrix[1][4][2]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][1], matrix[1][4][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][1], matrix[1][4][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][1], matrix[1][4][2]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][2], matrix[1][4][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][2], matrix[1][4][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][2], matrix[1][4][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][3], matrix[1][4][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][3], matrix[1][4][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][3][3], matrix[1][4][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][4], matrix[1][4][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][3][4], matrix[1][4][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][3][4], matrix[1][4][2]));
          __m256d temp_2[7];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][3][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][3][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][0], matrix[1][3][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][0], matrix[1][3][3]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][3][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][3][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][1], matrix[1][3][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][1], matrix[1][3][3]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][3][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][3][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][2], matrix[1][3][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][2], matrix[1][3][3]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][3][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][3][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][3], matrix[1][3][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][3], matrix[1][3][3]));
          __m256d temp_3[7];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
        }
        n_4_temp[0] = _mm256_sub_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][2][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][2][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][2][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][2][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][2][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][2][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][2][0], n_3_temp[6]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][2][1], n_3_temp[0]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][2][1], n_3_temp[1]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][2][1], n_3_temp[2]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][2][1], n_3_temp[3]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][2][1], n_3_temp[4]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][2][1], n_3_temp[5]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][2][1], n_3_temp[6]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][2][2], n_3_temp[0]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][2][2], n_3_temp[1]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][2][2], n_3_temp[2]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][2][2], n_3_temp[3]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][2][2], n_3_temp[4]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][2][2], n_3_temp[5]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][2][2], n_3_temp[6]));
      }
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[5];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][1]));
          __m256d temp_2[5];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][2]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][2]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][2]));
          __m256d temp_3[5];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][2][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][2][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][2][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][0], temp_3[4]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][2][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][2][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][1], temp_3[4]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][2][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][2][2], temp_3[4]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][3], temp_3[0]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][3], temp_3[1]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][3], temp_3[2]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][2][3], temp_3[3]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][2][3], temp_3[4]));
        }
        {
          __m256d temp_1[7];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][2][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][4], matrix[1][5][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][5], matrix[1][5][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][2][5], matrix[1][5][1]));
          __m256d temp_2[7];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][4]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][4]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][4]));
          __m256d temp_3[7];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][4][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][0], temp_3[5]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][0], temp_3[6]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][1], temp_3[5]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][4][1], temp_3[6]));
        }
        {
          __m256d temp_1[8];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][2][0], matrix[1][4][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][0], matrix[1][4][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][0], matrix[1][4][2]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][1], matrix[1][4][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][1], matrix[1][4][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][1], matrix[1][4][2]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][2], matrix[1][4][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][2], matrix[1][4][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][2], matrix[1][4][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][3], matrix[1][4][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][3], matrix[1][4][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][3], matrix[1][4][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][4], matrix[1][4][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][4], matrix[1][4][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][2][4], matrix[1][4][2]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][5], matrix[1][4][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][2][5], matrix[1][4][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][2][5], matrix[1][4][2]));
          __m256d temp_2[8];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][2][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][2][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][0], matrix[1][2][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][0], matrix[1][2][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][0], matrix[1][2][4]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][2][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][2][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][1], matrix[1][2][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][1], matrix[1][2][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][1], matrix[1][2][4]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][2][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][2][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][2], matrix[1][2][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][2], matrix[1][2][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][2], matrix[1][2][4]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][2][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][2][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][3], matrix[1][2][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][3], matrix[1][2][3]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][4][3], matrix[1][2][4]));
          __m256d temp_3[8];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][5][0], temp_3[7]));
        }
        n_4_temp[0] = _mm256_add_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][3][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_add_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][3][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_add_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][3][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_add_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][3][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_add_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][3][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_add_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][3][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_add_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][3][0], n_3_temp[6]));
        n_4_temp[7] = _mm256_add_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][3][0], n_3_temp[7]));
        n_4_temp[1] = _mm256_add_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][3][1], n_3_temp[0]));
        n_4_temp[2] = _mm256_add_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][3][1], n_3_temp[1]));
        n_4_temp[3] = _mm256_add_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][3][1], n_3_temp[2]));
        n_4_temp[4] = _mm256_add_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][3][1], n_3_temp[3]));
        n_4_temp[5] = _mm256_add_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][3][1], n_3_temp[4]));
        n_4_temp[6] = _mm256_add_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][3][1], n_3_temp[5]));
        n_4_temp[7] = _mm256_add_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][3][1], n_3_temp[6]));
        n_4_temp[8] = _mm256_add_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][3][1], n_3_temp[7]));
      }
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[6];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][3][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][3][4], matrix[1][5][1]));
          __m256d temp_2[6];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][3]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][3]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][3]));
          __m256d temp_3[6];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][2][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][2][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][2][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][0], temp_3[5]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][2][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][2][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][1], temp_3[4]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][2][1], temp_3[5]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][2][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][2][2], temp_3[4]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][2][2], temp_3[5]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][3], temp_3[0]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][3], temp_3[1]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][3], temp_3[2]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][2][3], temp_3[3]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][2][3], temp_3[4]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][2][3], temp_3[5]));
        }
        {
          __m256d temp_1[7];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][2][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][4], matrix[1][5][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][5], matrix[1][5][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][2][5], matrix[1][5][1]));
          __m256d temp_2[7];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][4]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][4]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][4]));
          __m256d temp_3[7];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][3][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][3][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][0], temp_3[5]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][3][0], temp_3[6]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][3][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][3][1], temp_3[5]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][3][1], temp_3[6]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][2], temp_3[0]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][2], temp_3[1]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][2], temp_3[2]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][2], temp_3[3]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][3][2], temp_3[4]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][3][2], temp_3[5]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][3][2], temp_3[6]));
        }
        {
          __m256d temp_1[9];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][2][0], matrix[1][3][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][0], matrix[1][3][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][0], matrix[1][3][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][0], matrix[1][3][3]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][1], matrix[1][3][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][1], matrix[1][3][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][1], matrix[1][3][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][1], matrix[1][3][3]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][2], matrix[1][3][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][2], matrix[1][3][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][2], matrix[1][3][2]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][2], matrix[1][3][3]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][3], matrix[1][3][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][3], matrix[1][3][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][3], matrix[1][3][2]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][2][3], matrix[1][3][3]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][4], matrix[1][3][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][4], matrix[1][3][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][2][4], matrix[1][3][2]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][2][4], matrix[1][3][3]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][5], matrix[1][3][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][2][5], matrix[1][3][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][2][5], matrix[1][3][2]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][2][5], matrix[1][3][3]));
          __m256d temp_2[9];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][3][0], matrix[1][2][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][3][0], matrix[1][2][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][3][0], matrix[1][2][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][3][0], matrix[1][2][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][0], matrix[1][2][4]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][3][1], matrix[1][2][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][3][1], matrix[1][2][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][3][1], matrix[1][2][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][1], matrix[1][2][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][1], matrix[1][2][4]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][3][2], matrix[1][2][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][3][2], matrix[1][2][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][2], matrix[1][2][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][2], matrix[1][2][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][3][2], matrix[1][2][4]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][3][3], matrix[1][2][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][3], matrix[1][2][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][3], matrix[1][2][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][3][3], matrix[1][2][3]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][3][3], matrix[1][2][4]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][4], matrix[1][2][0]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][4], matrix[1][2][1]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][3][4], matrix[1][2][2]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][3][4], matrix[1][2][3]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][3][4], matrix[1][2][4]));
          __m256d temp_3[9];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][5][0], temp_3[7]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][5][0], temp_3[8]));
        }
        n_4_temp[0] = _mm256_sub_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][4][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][4][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][4][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][4][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][4][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][4][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][4][0], n_3_temp[6]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][4][0], n_3_temp[7]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][4][0], n_3_temp[8]));
      }
      n_5_temp[0] = _mm256_add_pd(n_5_temp[0],
                                  _mm256_mul_pd(matrix[4][0][0], n_4_temp[0]));
      n_5_temp[1] = _mm256_add_pd(n_5_temp[1],
                                  _mm256_mul_pd(matrix[4][0][0], n_4_temp[1]));
      n_5_temp[2] = _mm256_add_pd(n_5_temp[2],
                                  _mm256_mul_pd(matrix[4][0][0], n_4_temp[2]));
      n_5_temp[3] = _mm256_add_pd(n_5_temp[3],
                                  _mm256_mul_pd(matrix[4][0][0], n_4_temp[3]));
      n_5_temp[4] = _mm256_add_pd(n_5_temp[4],
                                  _mm256_mul_pd(matrix[4][0][0], n_4_temp[4]));
      n_5_temp[5] = _mm256_add_pd(n_5_temp[5],
                                  _mm256_mul_pd(matrix[4][0][0], n_4_temp[5]));
      n_5_temp[6] = _mm256_add_pd(n_5_temp[6],
                                  _mm256_mul_pd(matrix[4][0][0], n_4_temp[6]));
      n_5_temp[7] = _mm256_add_pd(n_5_temp[7],
                                  _mm256_mul_pd(matrix[4][0][0], n_4_temp[7]));
      n_5_temp[8] = _mm256_add_pd(n_5_temp[8],
                                  _mm256_mul_pd(matrix[4][0][0], n_4_temp[8]));
      n_5_temp[1] = _mm256_add_pd(n_5_temp[1],
                                  _mm256_mul_pd(matrix[4][0][1], n_4_temp[0]));
      n_5_temp[2] = _mm256_add_pd(n_5_temp[2],
                                  _mm256_mul_pd(matrix[4][0][1], n_4_temp[1]));
      n_5_temp[3] = _mm256_add_pd(n_5_temp[3],
                                  _mm256_mul_pd(matrix[4][0][1], n_4_temp[2]));
      n_5_temp[4] = _mm256_add_pd(n_5_temp[4],
                                  _mm256_mul_pd(matrix[4][0][1], n_4_temp[3]));
      n_5_temp[5] = _mm256_add_pd(n_5_temp[5],
                                  _mm256_mul_pd(matrix[4][0][1], n_4_temp[4]));
      n_5_temp[6] = _mm256_add_pd(n_5_temp[6],
                                  _mm256_mul_pd(matrix[4][0][1], n_4_temp[5]));
      n_5_temp[7] = _mm256_add_pd(n_5_temp[7],
                                  _mm256_mul_pd(matrix[4][0][1], n_4_temp[6]));
      n_5_temp[8] = _mm256_add_pd(n_5_temp[8],
                                  _mm256_mul_pd(matrix[4][0][1], n_4_temp[7]));
      n_5_temp[9] = _mm256_add_pd(n_5_temp[9],
                                  _mm256_mul_pd(matrix[4][0][1], n_4_temp[8]));
      n_5_temp[2] = _mm256_add_pd(n_5_temp[2],
                                  _mm256_mul_pd(matrix[4][0][2], n_4_temp[0]));
      n_5_temp[3] = _mm256_add_pd(n_5_temp[3],
                                  _mm256_mul_pd(matrix[4][0][2], n_4_temp[1]));
      n_5_temp[4] = _mm256_add_pd(n_5_temp[4],
                                  _mm256_mul_pd(matrix[4][0][2], n_4_temp[2]));
      n_5_temp[5] = _mm256_add_pd(n_5_temp[5],
                                  _mm256_mul_pd(matrix[4][0][2], n_4_temp[3]));
      n_5_temp[6] = _mm256_add_pd(n_5_temp[6],
                                  _mm256_mul_pd(matrix[4][0][2], n_4_temp[4]));
      n_5_temp[7] = _mm256_add_pd(n_5_temp[7],
                                  _mm256_mul_pd(matrix[4][0][2], n_4_temp[5]));
      n_5_temp[8] = _mm256_add_pd(n_5_temp[8],
                                  _mm256_mul_pd(matrix[4][0][2], n_4_temp[6]));
      n_5_temp[9] = _mm256_add_pd(n_5_temp[9],
                                  _mm256_mul_pd(matrix[4][0][2], n_4_temp[7]));
      n_5_temp[10] = _mm256_add_pd(n_5_temp[10],
                                   _mm256_mul_pd(matrix[4][0][2], n_4_temp[8]));
      n_5_temp[3] = _mm256_add_pd(n_5_temp[3],
                                  _mm256_mul_pd(matrix[4][0][3], n_4_temp[0]));
      n_5_temp[4] = _mm256_add_pd(n_5_temp[4],
                                  _mm256_mul_pd(matrix[4][0][3], n_4_temp[1]));
      n_5_temp[5] = _mm256_add_pd(n_5_temp[5],
                                  _mm256_mul_pd(matrix[4][0][3], n_4_temp[2]));
      n_5_temp[6] = _mm256_add_pd(n_5_temp[6],
                                  _mm256_mul_pd(matrix[4][0][3], n_4_temp[3]));
      n_5_temp[7] = _mm256_add_pd(n_5_temp[7],
                                  _mm256_mul_pd(matrix[4][0][3], n_4_temp[4]));
      n_5_temp[8] = _mm256_add_pd(n_5_temp[8],
                                  _mm256_mul_pd(matrix[4][0][3], n_4_temp[5]));
      n_5_temp[9] = _mm256_add_pd(n_5_temp[9],
                                  _mm256_mul_pd(matrix[4][0][3], n_4_temp[6]));
      n_5_temp[10] = _mm256_add_pd(n_5_temp[10],
                                   _mm256_mul_pd(matrix[4][0][3], n_4_temp[7]));
      n_5_temp[11] = _mm256_add_pd(n_5_temp[11],
                                   _mm256_mul_pd(matrix[4][0][3], n_4_temp[8]));
    }
    {
      __m256d n_4_temp[17];
      n_4_temp[0] = _mm256_setzero_pd();
      n_4_temp[1] = _mm256_setzero_pd();
      n_4_temp[2] = _mm256_setzero_pd();
      n_4_temp[3] = _mm256_setzero_pd();
      n_4_temp[4] = _mm256_setzero_pd();
      n_4_temp[5] = _mm256_setzero_pd();
      n_4_temp[6] = _mm256_setzero_pd();
      n_4_temp[7] = _mm256_setzero_pd();
      n_4_temp[8] = _mm256_setzero_pd();
      n_4_temp[9] = _mm256_setzero_pd();
      n_4_temp[10] = _mm256_setzero_pd();
      n_4_temp[11] = _mm256_setzero_pd();
      n_4_temp[12] = _mm256_setzero_pd();
      n_4_temp[13] = _mm256_setzero_pd();
      n_4_temp[14] = _mm256_setzero_pd();
      n_4_temp[15] = _mm256_setzero_pd();
      n_4_temp[16] = _mm256_setzero_pd();
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[5];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][1]));
          __m256d temp_2[5];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][2]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][2]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][2]));
          __m256d temp_3[5];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][3][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][3][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][0], temp_3[4]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][3][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][1], temp_3[4]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][3][2], temp_3[4]));
        }
        {
          __m256d temp_1[6];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][3][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][3][4], matrix[1][5][1]));
          __m256d temp_2[6];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][3]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][3]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][3]));
          __m256d temp_3[6];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][4][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][0], temp_3[5]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][1], temp_3[5]));
        }
        {
          __m256d temp_1[7];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][3][0], matrix[1][4][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][0], matrix[1][4][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][0], matrix[1][4][2]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][1], matrix[1][4][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][1], matrix[1][4][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][1], matrix[1][4][2]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][2], matrix[1][4][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][2], matrix[1][4][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][2], matrix[1][4][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][3], matrix[1][4][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][3], matrix[1][4][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][3][3], matrix[1][4][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][4], matrix[1][4][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][3][4], matrix[1][4][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][3][4], matrix[1][4][2]));
          __m256d temp_2[7];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][3][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][3][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][0], matrix[1][3][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][0], matrix[1][3][3]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][3][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][3][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][1], matrix[1][3][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][1], matrix[1][3][3]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][3][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][3][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][2], matrix[1][3][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][2], matrix[1][3][3]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][3][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][3][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][3], matrix[1][3][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][3], matrix[1][3][3]));
          __m256d temp_3[7];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
        }
        n_4_temp[0] = _mm256_sub_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][0][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][0][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][0][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][0][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][0][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][0][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][0][0], n_3_temp[6]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][0][1], n_3_temp[0]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][0][1], n_3_temp[1]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][0][1], n_3_temp[2]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][0][1], n_3_temp[3]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][0][1], n_3_temp[4]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][0][1], n_3_temp[5]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][0][1], n_3_temp[6]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][0][2], n_3_temp[0]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][0][2], n_3_temp[1]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][0][2], n_3_temp[2]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][0][2], n_3_temp[3]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][0][2], n_3_temp[4]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][0][2], n_3_temp[5]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][0][2], n_3_temp[6]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][0][3], n_3_temp[0]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][0][3], n_3_temp[1]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][0][3], n_3_temp[2]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][0][3], n_3_temp[3]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][0][3], n_3_temp[4]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][0][3], n_3_temp[5]));
        n_4_temp[9] = _mm256_sub_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][0][3], n_3_temp[6]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][0][4], n_3_temp[0]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][0][4], n_3_temp[1]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][0][4], n_3_temp[2]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][0][4], n_3_temp[3]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][0][4], n_3_temp[4]));
        n_4_temp[9] = _mm256_sub_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][0][4], n_3_temp[5]));
        n_4_temp[10] = _mm256_sub_pd(
            n_4_temp[10], _mm256_mul_pd(matrix[3][0][4], n_3_temp[6]));
      }
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[5];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][1]));
          __m256d temp_2[5];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][2]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][2]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][2]));
          __m256d temp_3[5];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][0][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][0][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][0][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][0], temp_3[4]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][0][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][0][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][1], temp_3[4]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][0][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][2], temp_3[4]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][3], temp_3[0]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][3], temp_3[1]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][3], temp_3[2]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][3], temp_3[3]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][3], temp_3[4]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][4], temp_3[0]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][4], temp_3[1]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][4], temp_3[2]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][4], temp_3[3]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][0][4], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][5], temp_3[0]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][5], temp_3[1]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][5], temp_3[2]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][0][5], temp_3[3]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][0][5], temp_3[4]));
        }
        {
          __m256d temp_1[9];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][0][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][4], matrix[1][5][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][5], matrix[1][5][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][5], matrix[1][5][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][6], matrix[1][5][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][6], matrix[1][5][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][7], matrix[1][5][0]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][7], matrix[1][5][1]));
          __m256d temp_2[9];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][5]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][6]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][5]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][6]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][5]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][6]));
          __m256d temp_3[9];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][4][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][0], temp_3[5]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][0], temp_3[6]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][4][0], temp_3[7]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][4][0], temp_3[8]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][1], temp_3[5]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][4][1], temp_3[6]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][4][1], temp_3[7]));
          n_3_temp[9] = _mm256_sub_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][4][1], temp_3[8]));
        }
        {
          __m256d temp_1[10];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[9] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][0][0], matrix[1][4][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][0], matrix[1][4][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][0], matrix[1][4][2]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][1], matrix[1][4][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][1], matrix[1][4][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][1], matrix[1][4][2]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][2], matrix[1][4][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][2], matrix[1][4][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][2], matrix[1][4][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][3], matrix[1][4][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][3], matrix[1][4][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][3], matrix[1][4][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][4], matrix[1][4][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][4], matrix[1][4][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][4], matrix[1][4][2]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][5], matrix[1][4][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][5], matrix[1][4][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][5], matrix[1][4][2]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][6], matrix[1][4][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][6], matrix[1][4][1]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][6], matrix[1][4][2]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][7], matrix[1][4][0]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][7], matrix[1][4][1]));
          temp_1[9] = _mm256_add_pd(
              temp_1[9], _mm256_mul_pd(matrix[0][0][7], matrix[1][4][2]));
          __m256d temp_2[10];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[9] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][5]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][6]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][5]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][6]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][5]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][6]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][3]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][4]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][5]));
          temp_2[9] = _mm256_add_pd(
              temp_2[9], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][6]));
          __m256d temp_3[10];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[9] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[9] = _mm256_add_pd(temp_3[9], temp_1[9]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          temp_3[9] = _mm256_sub_pd(temp_3[9], temp_2[9]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][5][0], temp_3[7]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][5][0], temp_3[8]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][5][0], temp_3[9]));
        }
        n_4_temp[0] = _mm256_add_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][3][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_add_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][3][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_add_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][3][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_add_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][3][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_add_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][3][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_add_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][3][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_add_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][3][0], n_3_temp[6]));
        n_4_temp[7] = _mm256_add_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][3][0], n_3_temp[7]));
        n_4_temp[8] = _mm256_add_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][3][0], n_3_temp[8]));
        n_4_temp[9] = _mm256_add_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][3][0], n_3_temp[9]));
        n_4_temp[1] = _mm256_add_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][3][1], n_3_temp[0]));
        n_4_temp[2] = _mm256_add_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][3][1], n_3_temp[1]));
        n_4_temp[3] = _mm256_add_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][3][1], n_3_temp[2]));
        n_4_temp[4] = _mm256_add_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][3][1], n_3_temp[3]));
        n_4_temp[5] = _mm256_add_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][3][1], n_3_temp[4]));
        n_4_temp[6] = _mm256_add_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][3][1], n_3_temp[5]));
        n_4_temp[7] = _mm256_add_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][3][1], n_3_temp[6]));
        n_4_temp[8] = _mm256_add_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][3][1], n_3_temp[7]));
        n_4_temp[9] = _mm256_add_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][3][1], n_3_temp[8]));
        n_4_temp[10] = _mm256_add_pd(
            n_4_temp[10], _mm256_mul_pd(matrix[3][3][1], n_3_temp[9]));
      }
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[6];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][3][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][3][4], matrix[1][5][1]));
          __m256d temp_2[6];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][3]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][3]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][3]));
          __m256d temp_3[6];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][0][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][0][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][0][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][0], temp_3[5]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][0][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][0][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][1], temp_3[4]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][1], temp_3[5]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][0][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][2], temp_3[4]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][2], temp_3[5]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][3], temp_3[0]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][3], temp_3[1]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][3], temp_3[2]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][3], temp_3[3]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][3], temp_3[4]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][0][3], temp_3[5]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][4], temp_3[0]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][4], temp_3[1]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][4], temp_3[2]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][4], temp_3[3]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][0][4], temp_3[4]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][0][4], temp_3[5]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][5], temp_3[0]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][5], temp_3[1]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][5], temp_3[2]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][0][5], temp_3[3]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][0][5], temp_3[4]));
          n_3_temp[10] = _mm256_add_pd(
              n_3_temp[10], _mm256_mul_pd(matrix[2][0][5], temp_3[5]));
        }
        {
          __m256d temp_1[9];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][0][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][4], matrix[1][5][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][5], matrix[1][5][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][5], matrix[1][5][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][6], matrix[1][5][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][6], matrix[1][5][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][7], matrix[1][5][0]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][7], matrix[1][5][1]));
          __m256d temp_2[9];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][5]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][6]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][5]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][6]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][5]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][6]));
          __m256d temp_3[9];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][3][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][3][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][0], temp_3[5]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][3][0], temp_3[6]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][3][0], temp_3[7]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][3][0], temp_3[8]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][3][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][3][1], temp_3[5]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][3][1], temp_3[6]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][3][1], temp_3[7]));
          n_3_temp[9] = _mm256_sub_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][3][1], temp_3[8]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][2], temp_3[0]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][2], temp_3[1]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][2], temp_3[2]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][2], temp_3[3]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][3][2], temp_3[4]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][3][2], temp_3[5]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][3][2], temp_3[6]));
          n_3_temp[9] = _mm256_sub_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][3][2], temp_3[7]));
          n_3_temp[10] = _mm256_sub_pd(
              n_3_temp[10], _mm256_mul_pd(matrix[2][3][2], temp_3[8]));
        }
        {
          __m256d temp_1[11];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[9] = _mm256_setzero_pd();
          temp_1[10] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][0][0], matrix[1][3][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][0], matrix[1][3][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][0], matrix[1][3][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][0], matrix[1][3][3]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][1], matrix[1][3][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][1], matrix[1][3][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][1], matrix[1][3][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][1], matrix[1][3][3]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][2], matrix[1][3][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][2], matrix[1][3][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][2], matrix[1][3][2]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][2], matrix[1][3][3]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][3], matrix[1][3][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][3], matrix[1][3][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][3], matrix[1][3][2]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][3], matrix[1][3][3]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][4], matrix[1][3][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][4], matrix[1][3][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][4], matrix[1][3][2]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][4], matrix[1][3][3]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][5], matrix[1][3][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][5], matrix[1][3][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][5], matrix[1][3][2]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][5], matrix[1][3][3]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][6], matrix[1][3][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][6], matrix[1][3][1]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][6], matrix[1][3][2]));
          temp_1[9] = _mm256_add_pd(
              temp_1[9], _mm256_mul_pd(matrix[0][0][6], matrix[1][3][3]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][7], matrix[1][3][0]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][7], matrix[1][3][1]));
          temp_1[9] = _mm256_add_pd(
              temp_1[9], _mm256_mul_pd(matrix[0][0][7], matrix[1][3][2]));
          temp_1[10] = _mm256_add_pd(
              temp_1[10], _mm256_mul_pd(matrix[0][0][7], matrix[1][3][3]));
          __m256d temp_2[11];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[9] = _mm256_setzero_pd();
          temp_2[10] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][3][0], matrix[1][0][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][3][0], matrix[1][0][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][3][0], matrix[1][0][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][3][0], matrix[1][0][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][0], matrix[1][0][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][0], matrix[1][0][5]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][3][0], matrix[1][0][6]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][3][1], matrix[1][0][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][3][1], matrix[1][0][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][3][1], matrix[1][0][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][1], matrix[1][0][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][1], matrix[1][0][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][3][1], matrix[1][0][5]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][3][1], matrix[1][0][6]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][3][2], matrix[1][0][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][3][2], matrix[1][0][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][2], matrix[1][0][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][2], matrix[1][0][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][3][2], matrix[1][0][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][3][2], matrix[1][0][5]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][3][2], matrix[1][0][6]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][3][3], matrix[1][0][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][3], matrix[1][0][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][3], matrix[1][0][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][3][3], matrix[1][0][3]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][3][3], matrix[1][0][4]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][3][3], matrix[1][0][5]));
          temp_2[9] = _mm256_add_pd(
              temp_2[9], _mm256_mul_pd(matrix[0][3][3], matrix[1][0][6]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][4], matrix[1][0][0]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][4], matrix[1][0][1]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][3][4], matrix[1][0][2]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][3][4], matrix[1][0][3]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][3][4], matrix[1][0][4]));
          temp_2[9] = _mm256_add_pd(
              temp_2[9], _mm256_mul_pd(matrix[0][3][4], matrix[1][0][5]));
          temp_2[10] = _mm256_add_pd(
              temp_2[10], _mm256_mul_pd(matrix[0][3][4], matrix[1][0][6]));
          __m256d temp_3[11];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[9] = _mm256_setzero_pd();
          temp_3[10] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[9] = _mm256_add_pd(temp_3[9], temp_1[9]);
          temp_3[10] = _mm256_add_pd(temp_3[10], temp_1[10]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          temp_3[9] = _mm256_sub_pd(temp_3[9], temp_2[9]);
          temp_3[10] = _mm256_sub_pd(temp_3[10], temp_2[10]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][5][0], temp_3[7]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][5][0], temp_3[8]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][5][0], temp_3[9]));
          n_3_temp[10] = _mm256_add_pd(
              n_3_temp[10], _mm256_mul_pd(matrix[2][5][0], temp_3[10]));
        }
        n_4_temp[0] = _mm256_sub_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][4][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][4][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][4][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][4][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][4][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][4][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][4][0], n_3_temp[6]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][4][0], n_3_temp[7]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][4][0], n_3_temp[8]));
        n_4_temp[9] = _mm256_sub_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][4][0], n_3_temp[9]));
        n_4_temp[10] = _mm256_sub_pd(
            n_4_temp[10], _mm256_mul_pd(matrix[3][4][0], n_3_temp[10]));
      }
      n_5_temp[0] = _mm256_sub_pd(n_5_temp[0],
                                  _mm256_mul_pd(matrix[4][2][0], n_4_temp[0]));
      n_5_temp[1] = _mm256_sub_pd(n_5_temp[1],
                                  _mm256_mul_pd(matrix[4][2][0], n_4_temp[1]));
      n_5_temp[2] = _mm256_sub_pd(n_5_temp[2],
                                  _mm256_mul_pd(matrix[4][2][0], n_4_temp[2]));
      n_5_temp[3] = _mm256_sub_pd(n_5_temp[3],
                                  _mm256_mul_pd(matrix[4][2][0], n_4_temp[3]));
      n_5_temp[4] = _mm256_sub_pd(n_5_temp[4],
                                  _mm256_mul_pd(matrix[4][2][0], n_4_temp[4]));
      n_5_temp[5] = _mm256_sub_pd(n_5_temp[5],
                                  _mm256_mul_pd(matrix[4][2][0], n_4_temp[5]));
      n_5_temp[6] = _mm256_sub_pd(n_5_temp[6],
                                  _mm256_mul_pd(matrix[4][2][0], n_4_temp[6]));
      n_5_temp[7] = _mm256_sub_pd(n_5_temp[7],
                                  _mm256_mul_pd(matrix[4][2][0], n_4_temp[7]));
      n_5_temp[8] = _mm256_sub_pd(n_5_temp[8],
                                  _mm256_mul_pd(matrix[4][2][0], n_4_temp[8]));
      n_5_temp[9] = _mm256_sub_pd(n_5_temp[9],
                                  _mm256_mul_pd(matrix[4][2][0], n_4_temp[9]));
      n_5_temp[10] = _mm256_sub_pd(
          n_5_temp[10], _mm256_mul_pd(matrix[4][2][0], n_4_temp[10]));
      n_5_temp[1] = _mm256_sub_pd(n_5_temp[1],
                                  _mm256_mul_pd(matrix[4][2][1], n_4_temp[0]));
      n_5_temp[2] = _mm256_sub_pd(n_5_temp[2],
                                  _mm256_mul_pd(matrix[4][2][1], n_4_temp[1]));
      n_5_temp[3] = _mm256_sub_pd(n_5_temp[3],
                                  _mm256_mul_pd(matrix[4][2][1], n_4_temp[2]));
      n_5_temp[4] = _mm256_sub_pd(n_5_temp[4],
                                  _mm256_mul_pd(matrix[4][2][1], n_4_temp[3]));
      n_5_temp[5] = _mm256_sub_pd(n_5_temp[5],
                                  _mm256_mul_pd(matrix[4][2][1], n_4_temp[4]));
      n_5_temp[6] = _mm256_sub_pd(n_5_temp[6],
                                  _mm256_mul_pd(matrix[4][2][1], n_4_temp[5]));
      n_5_temp[7] = _mm256_sub_pd(n_5_temp[7],
                                  _mm256_mul_pd(matrix[4][2][1], n_4_temp[6]));
      n_5_temp[8] = _mm256_sub_pd(n_5_temp[8],
                                  _mm256_mul_pd(matrix[4][2][1], n_4_temp[7]));
      n_5_temp[9] = _mm256_sub_pd(n_5_temp[9],
                                  _mm256_mul_pd(matrix[4][2][1], n_4_temp[8]));
      n_5_temp[10] = _mm256_sub_pd(n_5_temp[10],
                                   _mm256_mul_pd(matrix[4][2][1], n_4_temp[9]));
      n_5_temp[11] = _mm256_sub_pd(
          n_5_temp[11], _mm256_mul_pd(matrix[4][2][1], n_4_temp[10]));
    }
    {
      __m256d n_4_temp[17];
      n_4_temp[0] = _mm256_setzero_pd();
      n_4_temp[1] = _mm256_setzero_pd();
      n_4_temp[2] = _mm256_setzero_pd();
      n_4_temp[3] = _mm256_setzero_pd();
      n_4_temp[4] = _mm256_setzero_pd();
      n_4_temp[5] = _mm256_setzero_pd();
      n_4_temp[6] = _mm256_setzero_pd();
      n_4_temp[7] = _mm256_setzero_pd();
      n_4_temp[8] = _mm256_setzero_pd();
      n_4_temp[9] = _mm256_setzero_pd();
      n_4_temp[10] = _mm256_setzero_pd();
      n_4_temp[11] = _mm256_setzero_pd();
      n_4_temp[12] = _mm256_setzero_pd();
      n_4_temp[13] = _mm256_setzero_pd();
      n_4_temp[14] = _mm256_setzero_pd();
      n_4_temp[15] = _mm256_setzero_pd();
      n_4_temp[16] = _mm256_setzero_pd();
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[5];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][1]));
          __m256d temp_2[5];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][2]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][2]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][2]));
          __m256d temp_3[5];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][2][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][2][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][2][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][0], temp_3[4]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][2][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][2][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][1], temp_3[4]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][2][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][2][2], temp_3[4]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][3], temp_3[0]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][3], temp_3[1]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][3], temp_3[2]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][2][3], temp_3[3]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][2][3], temp_3[4]));
        }
        {
          __m256d temp_1[7];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][2][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][4], matrix[1][5][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][5], matrix[1][5][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][2][5], matrix[1][5][1]));
          __m256d temp_2[7];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][4]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][4]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][4]));
          __m256d temp_3[7];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][4][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][0], temp_3[5]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][0], temp_3[6]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][1], temp_3[5]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][4][1], temp_3[6]));
        }
        {
          __m256d temp_1[8];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][2][0], matrix[1][4][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][0], matrix[1][4][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][0], matrix[1][4][2]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][1], matrix[1][4][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][1], matrix[1][4][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][1], matrix[1][4][2]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][2], matrix[1][4][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][2], matrix[1][4][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][2], matrix[1][4][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][3], matrix[1][4][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][3], matrix[1][4][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][3], matrix[1][4][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][4], matrix[1][4][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][4], matrix[1][4][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][2][4], matrix[1][4][2]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][5], matrix[1][4][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][2][5], matrix[1][4][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][2][5], matrix[1][4][2]));
          __m256d temp_2[8];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][2][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][2][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][0], matrix[1][2][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][0], matrix[1][2][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][0], matrix[1][2][4]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][2][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][2][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][1], matrix[1][2][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][1], matrix[1][2][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][1], matrix[1][2][4]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][2][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][2][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][2], matrix[1][2][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][2], matrix[1][2][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][2], matrix[1][2][4]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][2][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][2][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][3], matrix[1][2][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][3], matrix[1][2][3]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][4][3], matrix[1][2][4]));
          __m256d temp_3[8];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][5][0], temp_3[7]));
        }
        n_4_temp[0] = _mm256_sub_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][0][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][0][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][0][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][0][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][0][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][0][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][0][0], n_3_temp[6]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][0][0], n_3_temp[7]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][0][1], n_3_temp[0]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][0][1], n_3_temp[1]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][0][1], n_3_temp[2]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][0][1], n_3_temp[3]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][0][1], n_3_temp[4]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][0][1], n_3_temp[5]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][0][1], n_3_temp[6]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][0][1], n_3_temp[7]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][0][2], n_3_temp[0]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][0][2], n_3_temp[1]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][0][2], n_3_temp[2]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][0][2], n_3_temp[3]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][0][2], n_3_temp[4]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][0][2], n_3_temp[5]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][0][2], n_3_temp[6]));
        n_4_temp[9] = _mm256_sub_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][0][2], n_3_temp[7]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][0][3], n_3_temp[0]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][0][3], n_3_temp[1]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][0][3], n_3_temp[2]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][0][3], n_3_temp[3]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][0][3], n_3_temp[4]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][0][3], n_3_temp[5]));
        n_4_temp[9] = _mm256_sub_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][0][3], n_3_temp[6]));
        n_4_temp[10] = _mm256_sub_pd(
            n_4_temp[10], _mm256_mul_pd(matrix[3][0][3], n_3_temp[7]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][0][4], n_3_temp[0]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][0][4], n_3_temp[1]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][0][4], n_3_temp[2]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][0][4], n_3_temp[3]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][0][4], n_3_temp[4]));
        n_4_temp[9] = _mm256_sub_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][0][4], n_3_temp[5]));
        n_4_temp[10] = _mm256_sub_pd(
            n_4_temp[10], _mm256_mul_pd(matrix[3][0][4], n_3_temp[6]));
        n_4_temp[11] = _mm256_sub_pd(
            n_4_temp[11], _mm256_mul_pd(matrix[3][0][4], n_3_temp[7]));
      }
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[5];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][1]));
          __m256d temp_2[5];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][2]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][2]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][2]));
          __m256d temp_3[5];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][0][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][0][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][0][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][0], temp_3[4]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][0][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][0][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][1], temp_3[4]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][0][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][2], temp_3[4]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][3], temp_3[0]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][3], temp_3[1]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][3], temp_3[2]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][3], temp_3[3]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][3], temp_3[4]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][4], temp_3[0]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][4], temp_3[1]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][4], temp_3[2]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][4], temp_3[3]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][0][4], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][5], temp_3[0]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][5], temp_3[1]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][5], temp_3[2]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][0][5], temp_3[3]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][0][5], temp_3[4]));
        }
        {
          __m256d temp_1[9];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][0][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][4], matrix[1][5][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][5], matrix[1][5][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][5], matrix[1][5][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][6], matrix[1][5][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][6], matrix[1][5][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][7], matrix[1][5][0]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][7], matrix[1][5][1]));
          __m256d temp_2[9];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][5]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][6]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][5]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][6]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][5]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][6]));
          __m256d temp_3[9];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][4][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][0], temp_3[5]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][0], temp_3[6]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][4][0], temp_3[7]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][4][0], temp_3[8]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][1], temp_3[5]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][4][1], temp_3[6]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][4][1], temp_3[7]));
          n_3_temp[9] = _mm256_sub_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][4][1], temp_3[8]));
        }
        {
          __m256d temp_1[10];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[9] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][0][0], matrix[1][4][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][0], matrix[1][4][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][0], matrix[1][4][2]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][1], matrix[1][4][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][1], matrix[1][4][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][1], matrix[1][4][2]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][2], matrix[1][4][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][2], matrix[1][4][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][2], matrix[1][4][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][3], matrix[1][4][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][3], matrix[1][4][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][3], matrix[1][4][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][4], matrix[1][4][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][4], matrix[1][4][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][4], matrix[1][4][2]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][5], matrix[1][4][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][5], matrix[1][4][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][5], matrix[1][4][2]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][6], matrix[1][4][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][6], matrix[1][4][1]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][6], matrix[1][4][2]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][7], matrix[1][4][0]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][7], matrix[1][4][1]));
          temp_1[9] = _mm256_add_pd(
              temp_1[9], _mm256_mul_pd(matrix[0][0][7], matrix[1][4][2]));
          __m256d temp_2[10];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[9] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][5]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][6]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][5]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][6]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][5]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][6]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][3]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][4]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][5]));
          temp_2[9] = _mm256_add_pd(
              temp_2[9], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][6]));
          __m256d temp_3[10];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[9] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[9] = _mm256_add_pd(temp_3[9], temp_1[9]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          temp_3[9] = _mm256_sub_pd(temp_3[9], temp_2[9]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][5][0], temp_3[7]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][5][0], temp_3[8]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][5][0], temp_3[9]));
        }
        n_4_temp[0] = _mm256_add_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][2][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_add_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][2][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_add_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][2][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_add_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][2][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_add_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][2][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_add_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][2][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_add_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][2][0], n_3_temp[6]));
        n_4_temp[7] = _mm256_add_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][2][0], n_3_temp[7]));
        n_4_temp[8] = _mm256_add_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][2][0], n_3_temp[8]));
        n_4_temp[9] = _mm256_add_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][2][0], n_3_temp[9]));
        n_4_temp[1] = _mm256_add_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][2][1], n_3_temp[0]));
        n_4_temp[2] = _mm256_add_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][2][1], n_3_temp[1]));
        n_4_temp[3] = _mm256_add_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][2][1], n_3_temp[2]));
        n_4_temp[4] = _mm256_add_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][2][1], n_3_temp[3]));
        n_4_temp[5] = _mm256_add_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][2][1], n_3_temp[4]));
        n_4_temp[6] = _mm256_add_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][2][1], n_3_temp[5]));
        n_4_temp[7] = _mm256_add_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][2][1], n_3_temp[6]));
        n_4_temp[8] = _mm256_add_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][2][1], n_3_temp[7]));
        n_4_temp[9] = _mm256_add_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][2][1], n_3_temp[8]));
        n_4_temp[10] = _mm256_add_pd(
            n_4_temp[10], _mm256_mul_pd(matrix[3][2][1], n_3_temp[9]));
        n_4_temp[2] = _mm256_add_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][2][2], n_3_temp[0]));
        n_4_temp[3] = _mm256_add_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][2][2], n_3_temp[1]));
        n_4_temp[4] = _mm256_add_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][2][2], n_3_temp[2]));
        n_4_temp[5] = _mm256_add_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][2][2], n_3_temp[3]));
        n_4_temp[6] = _mm256_add_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][2][2], n_3_temp[4]));
        n_4_temp[7] = _mm256_add_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][2][2], n_3_temp[5]));
        n_4_temp[8] = _mm256_add_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][2][2], n_3_temp[6]));
        n_4_temp[9] = _mm256_add_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][2][2], n_3_temp[7]));
        n_4_temp[10] = _mm256_add_pd(
            n_4_temp[10], _mm256_mul_pd(matrix[3][2][2], n_3_temp[8]));
        n_4_temp[11] = _mm256_add_pd(
            n_4_temp[11], _mm256_mul_pd(matrix[3][2][2], n_3_temp[9]));
      }
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[7];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][2][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][2][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][2][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][2][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][2][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][4], matrix[1][5][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][2][5], matrix[1][5][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][2][5], matrix[1][5][1]));
          __m256d temp_2[7];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][0], matrix[1][2][4]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][1], matrix[1][2][4]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][2], matrix[1][2][4]));
          __m256d temp_3[7];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][0][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][0][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][0][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][0], temp_3[6]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][0][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][0][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][1], temp_3[4]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][1], temp_3[5]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][1], temp_3[6]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][0][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][2], temp_3[4]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][2], temp_3[5]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][0][2], temp_3[6]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][3], temp_3[0]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][3], temp_3[1]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][3], temp_3[2]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][3], temp_3[3]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][3], temp_3[4]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][0][3], temp_3[5]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][0][3], temp_3[6]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][4], temp_3[0]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][4], temp_3[1]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][4], temp_3[2]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][4], temp_3[3]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][0][4], temp_3[4]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][0][4], temp_3[5]));
          n_3_temp[10] = _mm256_add_pd(
              n_3_temp[10], _mm256_mul_pd(matrix[2][0][4], temp_3[6]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][5], temp_3[0]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][5], temp_3[1]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][5], temp_3[2]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][0][5], temp_3[3]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][0][5], temp_3[4]));
          n_3_temp[10] = _mm256_add_pd(
              n_3_temp[10], _mm256_mul_pd(matrix[2][0][5], temp_3[5]));
          n_3_temp[11] = _mm256_add_pd(
              n_3_temp[11], _mm256_mul_pd(matrix[2][0][5], temp_3[6]));
        }
        {
          __m256d temp_1[9];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][0][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][4], matrix[1][5][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][5], matrix[1][5][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][5], matrix[1][5][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][6], matrix[1][5][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][6], matrix[1][5][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][7], matrix[1][5][0]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][7], matrix[1][5][1]));
          __m256d temp_2[9];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][5]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][6]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][5]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][6]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][5]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][6]));
          __m256d temp_3[9];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][2][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][2][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][2][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][0], temp_3[5]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][2][0], temp_3[6]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][2][0], temp_3[7]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][2][0], temp_3[8]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][2][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][2][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][2][1], temp_3[5]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][2][1], temp_3[6]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][2][1], temp_3[7]));
          n_3_temp[9] = _mm256_sub_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][2][1], temp_3[8]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][2][2], temp_3[0]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][2], temp_3[1]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][2], temp_3[2]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][2], temp_3[3]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][2][2], temp_3[4]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][2][2], temp_3[5]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][2][2], temp_3[6]));
          n_3_temp[9] = _mm256_sub_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][2][2], temp_3[7]));
          n_3_temp[10] = _mm256_sub_pd(
              n_3_temp[10], _mm256_mul_pd(matrix[2][2][2], temp_3[8]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][2][3], temp_3[0]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][2][3], temp_3[1]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][2][3], temp_3[2]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][2][3], temp_3[3]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][2][3], temp_3[4]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][2][3], temp_3[5]));
          n_3_temp[9] = _mm256_sub_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][2][3], temp_3[6]));
          n_3_temp[10] = _mm256_sub_pd(
              n_3_temp[10], _mm256_mul_pd(matrix[2][2][3], temp_3[7]));
          n_3_temp[11] = _mm256_sub_pd(
              n_3_temp[11], _mm256_mul_pd(matrix[2][2][3], temp_3[8]));
        }
        {
          __m256d temp_1[12];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[9] = _mm256_setzero_pd();
          temp_1[10] = _mm256_setzero_pd();
          temp_1[11] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][0][0], matrix[1][2][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][0], matrix[1][2][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][0], matrix[1][2][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][0], matrix[1][2][3]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][0], matrix[1][2][4]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][1], matrix[1][2][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][1], matrix[1][2][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][1], matrix[1][2][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][1], matrix[1][2][3]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][1], matrix[1][2][4]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][2], matrix[1][2][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][2], matrix[1][2][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][2], matrix[1][2][2]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][2], matrix[1][2][3]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][2], matrix[1][2][4]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][3], matrix[1][2][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][3], matrix[1][2][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][3], matrix[1][2][2]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][3], matrix[1][2][3]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][3], matrix[1][2][4]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][4], matrix[1][2][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][4], matrix[1][2][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][4], matrix[1][2][2]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][4], matrix[1][2][3]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][4], matrix[1][2][4]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][5], matrix[1][2][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][5], matrix[1][2][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][5], matrix[1][2][2]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][5], matrix[1][2][3]));
          temp_1[9] = _mm256_add_pd(
              temp_1[9], _mm256_mul_pd(matrix[0][0][5], matrix[1][2][4]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][6], matrix[1][2][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][6], matrix[1][2][1]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][6], matrix[1][2][2]));
          temp_1[9] = _mm256_add_pd(
              temp_1[9], _mm256_mul_pd(matrix[0][0][6], matrix[1][2][3]));
          temp_1[10] = _mm256_add_pd(
              temp_1[10], _mm256_mul_pd(matrix[0][0][6], matrix[1][2][4]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][7], matrix[1][2][0]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][7], matrix[1][2][1]));
          temp_1[9] = _mm256_add_pd(
              temp_1[9], _mm256_mul_pd(matrix[0][0][7], matrix[1][2][2]));
          temp_1[10] = _mm256_add_pd(
              temp_1[10], _mm256_mul_pd(matrix[0][0][7], matrix[1][2][3]));
          temp_1[11] = _mm256_add_pd(
              temp_1[11], _mm256_mul_pd(matrix[0][0][7], matrix[1][2][4]));
          __m256d temp_2[12];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[9] = _mm256_setzero_pd();
          temp_2[10] = _mm256_setzero_pd();
          temp_2[11] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][2][0], matrix[1][0][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][2][0], matrix[1][0][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][2][0], matrix[1][0][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][2][0], matrix[1][0][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][2][0], matrix[1][0][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][2][0], matrix[1][0][5]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][2][0], matrix[1][0][6]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][2][1], matrix[1][0][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][2][1], matrix[1][0][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][2][1], matrix[1][0][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][2][1], matrix[1][0][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][2][1], matrix[1][0][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][2][1], matrix[1][0][5]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][2][1], matrix[1][0][6]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][2][2], matrix[1][0][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][2][2], matrix[1][0][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][2][2], matrix[1][0][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][2][2], matrix[1][0][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][2][2], matrix[1][0][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][2][2], matrix[1][0][5]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][2][2], matrix[1][0][6]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][2][3], matrix[1][0][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][2][3], matrix[1][0][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][2][3], matrix[1][0][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][2][3], matrix[1][0][3]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][2][3], matrix[1][0][4]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][2][3], matrix[1][0][5]));
          temp_2[9] = _mm256_add_pd(
              temp_2[9], _mm256_mul_pd(matrix[0][2][3], matrix[1][0][6]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][2][4], matrix[1][0][0]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][2][4], matrix[1][0][1]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][2][4], matrix[1][0][2]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][2][4], matrix[1][0][3]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][2][4], matrix[1][0][4]));
          temp_2[9] = _mm256_add_pd(
              temp_2[9], _mm256_mul_pd(matrix[0][2][4], matrix[1][0][5]));
          temp_2[10] = _mm256_add_pd(
              temp_2[10], _mm256_mul_pd(matrix[0][2][4], matrix[1][0][6]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][2][5], matrix[1][0][0]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][2][5], matrix[1][0][1]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][2][5], matrix[1][0][2]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][2][5], matrix[1][0][3]));
          temp_2[9] = _mm256_add_pd(
              temp_2[9], _mm256_mul_pd(matrix[0][2][5], matrix[1][0][4]));
          temp_2[10] = _mm256_add_pd(
              temp_2[10], _mm256_mul_pd(matrix[0][2][5], matrix[1][0][5]));
          temp_2[11] = _mm256_add_pd(
              temp_2[11], _mm256_mul_pd(matrix[0][2][5], matrix[1][0][6]));
          __m256d temp_3[12];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[9] = _mm256_setzero_pd();
          temp_3[10] = _mm256_setzero_pd();
          temp_3[11] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[9] = _mm256_add_pd(temp_3[9], temp_1[9]);
          temp_3[10] = _mm256_add_pd(temp_3[10], temp_1[10]);
          temp_3[11] = _mm256_add_pd(temp_3[11], temp_1[11]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          temp_3[9] = _mm256_sub_pd(temp_3[9], temp_2[9]);
          temp_3[10] = _mm256_sub_pd(temp_3[10], temp_2[10]);
          temp_3[11] = _mm256_sub_pd(temp_3[11], temp_2[11]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][5][0], temp_3[7]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][5][0], temp_3[8]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][5][0], temp_3[9]));
          n_3_temp[10] = _mm256_add_pd(
              n_3_temp[10], _mm256_mul_pd(matrix[2][5][0], temp_3[10]));
          n_3_temp[11] = _mm256_add_pd(
              n_3_temp[11], _mm256_mul_pd(matrix[2][5][0], temp_3[11]));
        }
        n_4_temp[0] = _mm256_sub_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][4][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][4][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][4][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][4][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][4][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][4][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][4][0], n_3_temp[6]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][4][0], n_3_temp[7]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][4][0], n_3_temp[8]));
        n_4_temp[9] = _mm256_sub_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][4][0], n_3_temp[9]));
        n_4_temp[10] = _mm256_sub_pd(
            n_4_temp[10], _mm256_mul_pd(matrix[3][4][0], n_3_temp[10]));
        n_4_temp[11] = _mm256_sub_pd(
            n_4_temp[11], _mm256_mul_pd(matrix[3][4][0], n_3_temp[11]));
      }
      n_5_temp[0] = _mm256_add_pd(n_5_temp[0],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[0]));
      n_5_temp[1] = _mm256_add_pd(n_5_temp[1],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[1]));
      n_5_temp[2] = _mm256_add_pd(n_5_temp[2],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[2]));
      n_5_temp[3] = _mm256_add_pd(n_5_temp[3],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[3]));
      n_5_temp[4] = _mm256_add_pd(n_5_temp[4],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[4]));
      n_5_temp[5] = _mm256_add_pd(n_5_temp[5],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[5]));
      n_5_temp[6] = _mm256_add_pd(n_5_temp[6],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[6]));
      n_5_temp[7] = _mm256_add_pd(n_5_temp[7],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[7]));
      n_5_temp[8] = _mm256_add_pd(n_5_temp[8],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[8]));
      n_5_temp[9] = _mm256_add_pd(n_5_temp[9],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[9]));
      n_5_temp[10] = _mm256_add_pd(
          n_5_temp[10], _mm256_mul_pd(matrix[4][3][0], n_4_temp[10]));
      n_5_temp[11] = _mm256_add_pd(
          n_5_temp[11], _mm256_mul_pd(matrix[4][3][0], n_4_temp[11]));
    }
    n_6_temp[0] =
        _mm256_add_pd(n_6_temp[0], _mm256_mul_pd(matrix[5][1][0], n_5_temp[0]));
    n_6_temp[1] =
        _mm256_add_pd(n_6_temp[1], _mm256_mul_pd(matrix[5][1][0], n_5_temp[1]));
    n_6_temp[2] =
        _mm256_add_pd(n_6_temp[2], _mm256_mul_pd(matrix[5][1][0], n_5_temp[2]));
    n_6_temp[3] =
        _mm256_add_pd(n_6_temp[3], _mm256_mul_pd(matrix[5][1][0], n_5_temp[3]));
    n_6_temp[4] =
        _mm256_add_pd(n_6_temp[4], _mm256_mul_pd(matrix[5][1][0], n_5_temp[4]));
    n_6_temp[5] =
        _mm256_add_pd(n_6_temp[5], _mm256_mul_pd(matrix[5][1][0], n_5_temp[5]));
    n_6_temp[6] =
        _mm256_add_pd(n_6_temp[6], _mm256_mul_pd(matrix[5][1][0], n_5_temp[6]));
    n_6_temp[7] =
        _mm256_add_pd(n_6_temp[7], _mm256_mul_pd(matrix[5][1][0], n_5_temp[7]));
    n_6_temp[8] =
        _mm256_add_pd(n_6_temp[8], _mm256_mul_pd(matrix[5][1][0], n_5_temp[8]));
    n_6_temp[9] =
        _mm256_add_pd(n_6_temp[9], _mm256_mul_pd(matrix[5][1][0], n_5_temp[9]));
    n_6_temp[10] = _mm256_add_pd(n_6_temp[10],
                                 _mm256_mul_pd(matrix[5][1][0], n_5_temp[10]));
    n_6_temp[11] = _mm256_add_pd(n_6_temp[11],
                                 _mm256_mul_pd(matrix[5][1][0], n_5_temp[11]));
    n_6_temp[1] =
        _mm256_add_pd(n_6_temp[1], _mm256_mul_pd(matrix[5][1][1], n_5_temp[0]));
    n_6_temp[2] =
        _mm256_add_pd(n_6_temp[2], _mm256_mul_pd(matrix[5][1][1], n_5_temp[1]));
    n_6_temp[3] =
        _mm256_add_pd(n_6_temp[3], _mm256_mul_pd(matrix[5][1][1], n_5_temp[2]));
    n_6_temp[4] =
        _mm256_add_pd(n_6_temp[4], _mm256_mul_pd(matrix[5][1][1], n_5_temp[3]));
    n_6_temp[5] =
        _mm256_add_pd(n_6_temp[5], _mm256_mul_pd(matrix[5][1][1], n_5_temp[4]));
    n_6_temp[6] =
        _mm256_add_pd(n_6_temp[6], _mm256_mul_pd(matrix[5][1][1], n_5_temp[5]));
    n_6_temp[7] =
        _mm256_add_pd(n_6_temp[7], _mm256_mul_pd(matrix[5][1][1], n_5_temp[6]));
    n_6_temp[8] =
        _mm256_add_pd(n_6_temp[8], _mm256_mul_pd(matrix[5][1][1], n_5_temp[7]));
    n_6_temp[9] =
        _mm256_add_pd(n_6_temp[9], _mm256_mul_pd(matrix[5][1][1], n_5_temp[8]));
    n_6_temp[10] = _mm256_add_pd(n_6_temp[10],
                                 _mm256_mul_pd(matrix[5][1][1], n_5_temp[9]));
    n_6_temp[11] = _mm256_add_pd(n_6_temp[11],
                                 _mm256_mul_pd(matrix[5][1][1], n_5_temp[10]));
    n_6_temp[12] = _mm256_add_pd(n_6_temp[12],
                                 _mm256_mul_pd(matrix[5][1][1], n_5_temp[11]));
  }
  {
    __m256d n_5_temp[17];
    n_5_temp[0] = _mm256_setzero_pd();
    n_5_temp[1] = _mm256_setzero_pd();
    n_5_temp[2] = _mm256_setzero_pd();
    n_5_temp[3] = _mm256_setzero_pd();
    n_5_temp[4] = _mm256_setzero_pd();
    n_5_temp[5] = _mm256_setzero_pd();
    n_5_temp[6] = _mm256_setzero_pd();
    n_5_temp[7] = _mm256_setzero_pd();
    n_5_temp[8] = _mm256_setzero_pd();
    n_5_temp[9] = _mm256_setzero_pd();
    n_5_temp[10] = _mm256_setzero_pd();
    n_5_temp[11] = _mm256_setzero_pd();
    n_5_temp[12] = _mm256_setzero_pd();
    n_5_temp[13] = _mm256_setzero_pd();
    n_5_temp[14] = _mm256_setzero_pd();
    n_5_temp[15] = _mm256_setzero_pd();
    n_5_temp[16] = _mm256_setzero_pd();
    {
      __m256d n_4_temp[17];
      n_4_temp[0] = _mm256_setzero_pd();
      n_4_temp[1] = _mm256_setzero_pd();
      n_4_temp[2] = _mm256_setzero_pd();
      n_4_temp[3] = _mm256_setzero_pd();
      n_4_temp[4] = _mm256_setzero_pd();
      n_4_temp[5] = _mm256_setzero_pd();
      n_4_temp[6] = _mm256_setzero_pd();
      n_4_temp[7] = _mm256_setzero_pd();
      n_4_temp[8] = _mm256_setzero_pd();
      n_4_temp[9] = _mm256_setzero_pd();
      n_4_temp[10] = _mm256_setzero_pd();
      n_4_temp[11] = _mm256_setzero_pd();
      n_4_temp[12] = _mm256_setzero_pd();
      n_4_temp[13] = _mm256_setzero_pd();
      n_4_temp[14] = _mm256_setzero_pd();
      n_4_temp[15] = _mm256_setzero_pd();
      n_4_temp[16] = _mm256_setzero_pd();
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[5];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][1]));
          __m256d temp_2[5];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][2]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][2]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][2]));
          __m256d temp_3[5];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][3][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][3][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][0], temp_3[4]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][3][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][1], temp_3[4]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][3][2], temp_3[4]));
        }
        {
          __m256d temp_1[6];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][3][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][3][4], matrix[1][5][1]));
          __m256d temp_2[6];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][3]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][3]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][3]));
          __m256d temp_3[6];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][4][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][0], temp_3[5]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][1], temp_3[5]));
        }
        {
          __m256d temp_1[7];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][3][0], matrix[1][4][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][0], matrix[1][4][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][0], matrix[1][4][2]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][1], matrix[1][4][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][1], matrix[1][4][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][1], matrix[1][4][2]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][2], matrix[1][4][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][2], matrix[1][4][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][2], matrix[1][4][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][3], matrix[1][4][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][3], matrix[1][4][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][3][3], matrix[1][4][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][4], matrix[1][4][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][3][4], matrix[1][4][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][3][4], matrix[1][4][2]));
          __m256d temp_2[7];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][3][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][3][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][0], matrix[1][3][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][0], matrix[1][3][3]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][3][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][3][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][1], matrix[1][3][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][1], matrix[1][3][3]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][3][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][3][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][2], matrix[1][3][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][2], matrix[1][3][3]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][3][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][3][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][3], matrix[1][3][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][3], matrix[1][3][3]));
          __m256d temp_3[7];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
        }
        n_4_temp[0] = _mm256_sub_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][1][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][1][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][1][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][1][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][1][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][1][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][1][0], n_3_temp[6]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][1][1], n_3_temp[0]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][1][1], n_3_temp[1]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][1][1], n_3_temp[2]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][1][1], n_3_temp[3]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][1][1], n_3_temp[4]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][1][1], n_3_temp[5]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][1][1], n_3_temp[6]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][1][2], n_3_temp[0]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][1][2], n_3_temp[1]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][1][2], n_3_temp[2]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][1][2], n_3_temp[3]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][1][2], n_3_temp[4]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][1][2], n_3_temp[5]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][1][2], n_3_temp[6]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][1][3], n_3_temp[0]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][1][3], n_3_temp[1]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][1][3], n_3_temp[2]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][1][3], n_3_temp[3]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][1][3], n_3_temp[4]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][1][3], n_3_temp[5]));
        n_4_temp[9] = _mm256_sub_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][1][3], n_3_temp[6]));
      }
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[5];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][1]));
          __m256d temp_2[5];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][2]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][2]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][2]));
          __m256d temp_3[5];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][1][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][1][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][1][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][0], temp_3[4]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][1][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][1][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][1], temp_3[4]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][1][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][2], temp_3[4]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][3], temp_3[0]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][3], temp_3[1]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][3], temp_3[2]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][3], temp_3[3]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][1][3], temp_3[4]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][4], temp_3[0]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][4], temp_3[1]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][4], temp_3[2]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][1][4], temp_3[3]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][1][4], temp_3[4]));
        }
        {
          __m256d temp_1[8];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][1][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][4], matrix[1][5][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][5], matrix[1][5][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][5], matrix[1][5][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][6], matrix[1][5][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][6], matrix[1][5][1]));
          __m256d temp_2[8];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][5]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][5]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][5]));
          __m256d temp_3[8];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][4][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][0], temp_3[5]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][0], temp_3[6]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][4][0], temp_3[7]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][1], temp_3[5]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][4][1], temp_3[6]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][4][1], temp_3[7]));
        }
        {
          __m256d temp_1[9];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][1][0], matrix[1][4][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][0], matrix[1][4][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][0], matrix[1][4][2]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][1], matrix[1][4][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][1], matrix[1][4][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][1], matrix[1][4][2]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][2], matrix[1][4][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][2], matrix[1][4][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][2], matrix[1][4][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][3], matrix[1][4][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][3], matrix[1][4][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][3], matrix[1][4][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][4], matrix[1][4][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][4], matrix[1][4][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][4], matrix[1][4][2]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][5], matrix[1][4][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][5], matrix[1][4][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][5], matrix[1][4][2]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][6], matrix[1][4][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][6], matrix[1][4][1]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][1][6], matrix[1][4][2]));
          __m256d temp_2[9];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][1][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][1][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][0], matrix[1][1][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][0], matrix[1][1][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][0], matrix[1][1][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][0], matrix[1][1][5]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][1][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][1][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][1], matrix[1][1][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][1], matrix[1][1][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][1], matrix[1][1][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][1], matrix[1][1][5]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][1][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][1][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][2], matrix[1][1][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][2], matrix[1][1][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][2], matrix[1][1][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][4][2], matrix[1][1][5]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][1][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][1][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][3], matrix[1][1][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][3], matrix[1][1][3]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][4][3], matrix[1][1][4]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][4][3], matrix[1][1][5]));
          __m256d temp_3[9];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][5][0], temp_3[7]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][5][0], temp_3[8]));
        }
        n_4_temp[0] = _mm256_add_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][3][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_add_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][3][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_add_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][3][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_add_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][3][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_add_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][3][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_add_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][3][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_add_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][3][0], n_3_temp[6]));
        n_4_temp[7] = _mm256_add_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][3][0], n_3_temp[7]));
        n_4_temp[8] = _mm256_add_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][3][0], n_3_temp[8]));
        n_4_temp[1] = _mm256_add_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][3][1], n_3_temp[0]));
        n_4_temp[2] = _mm256_add_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][3][1], n_3_temp[1]));
        n_4_temp[3] = _mm256_add_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][3][1], n_3_temp[2]));
        n_4_temp[4] = _mm256_add_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][3][1], n_3_temp[3]));
        n_4_temp[5] = _mm256_add_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][3][1], n_3_temp[4]));
        n_4_temp[6] = _mm256_add_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][3][1], n_3_temp[5]));
        n_4_temp[7] = _mm256_add_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][3][1], n_3_temp[6]));
        n_4_temp[8] = _mm256_add_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][3][1], n_3_temp[7]));
        n_4_temp[9] = _mm256_add_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][3][1], n_3_temp[8]));
      }
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[6];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][3][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][3][4], matrix[1][5][1]));
          __m256d temp_2[6];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][3]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][3]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][3]));
          __m256d temp_3[6];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][1][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][1][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][1][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][0], temp_3[5]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][1][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][1][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][1], temp_3[4]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][1], temp_3[5]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][1][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][2], temp_3[4]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][1][2], temp_3[5]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][3], temp_3[0]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][3], temp_3[1]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][3], temp_3[2]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][3], temp_3[3]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][1][3], temp_3[4]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][1][3], temp_3[5]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][4], temp_3[0]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][4], temp_3[1]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][4], temp_3[2]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][1][4], temp_3[3]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][1][4], temp_3[4]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][1][4], temp_3[5]));
        }
        {
          __m256d temp_1[8];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][1][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][4], matrix[1][5][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][5], matrix[1][5][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][5], matrix[1][5][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][6], matrix[1][5][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][6], matrix[1][5][1]));
          __m256d temp_2[8];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][5]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][5]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][5]));
          __m256d temp_3[8];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][3][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][3][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][0], temp_3[5]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][3][0], temp_3[6]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][3][0], temp_3[7]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][3][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][3][1], temp_3[5]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][3][1], temp_3[6]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][3][1], temp_3[7]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][2], temp_3[0]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][2], temp_3[1]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][2], temp_3[2]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][2], temp_3[3]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][3][2], temp_3[4]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][3][2], temp_3[5]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][3][2], temp_3[6]));
          n_3_temp[9] = _mm256_sub_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][3][2], temp_3[7]));
        }
        {
          __m256d temp_1[10];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[9] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][1][0], matrix[1][3][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][0], matrix[1][3][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][0], matrix[1][3][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][0], matrix[1][3][3]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][1], matrix[1][3][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][1], matrix[1][3][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][1], matrix[1][3][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][1], matrix[1][3][3]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][2], matrix[1][3][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][2], matrix[1][3][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][2], matrix[1][3][2]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][2], matrix[1][3][3]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][3], matrix[1][3][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][3], matrix[1][3][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][3], matrix[1][3][2]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][3], matrix[1][3][3]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][4], matrix[1][3][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][4], matrix[1][3][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][4], matrix[1][3][2]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][4], matrix[1][3][3]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][5], matrix[1][3][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][5], matrix[1][3][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][5], matrix[1][3][2]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][1][5], matrix[1][3][3]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][6], matrix[1][3][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][6], matrix[1][3][1]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][1][6], matrix[1][3][2]));
          temp_1[9] = _mm256_add_pd(
              temp_1[9], _mm256_mul_pd(matrix[0][1][6], matrix[1][3][3]));
          __m256d temp_2[10];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[9] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][3][0], matrix[1][1][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][3][0], matrix[1][1][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][3][0], matrix[1][1][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][3][0], matrix[1][1][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][0], matrix[1][1][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][0], matrix[1][1][5]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][3][1], matrix[1][1][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][3][1], matrix[1][1][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][3][1], matrix[1][1][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][1], matrix[1][1][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][1], matrix[1][1][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][3][1], matrix[1][1][5]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][3][2], matrix[1][1][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][3][2], matrix[1][1][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][2], matrix[1][1][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][2], matrix[1][1][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][3][2], matrix[1][1][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][3][2], matrix[1][1][5]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][3][3], matrix[1][1][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][3], matrix[1][1][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][3], matrix[1][1][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][3][3], matrix[1][1][3]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][3][3], matrix[1][1][4]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][3][3], matrix[1][1][5]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][4], matrix[1][1][0]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][4], matrix[1][1][1]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][3][4], matrix[1][1][2]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][3][4], matrix[1][1][3]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][3][4], matrix[1][1][4]));
          temp_2[9] = _mm256_add_pd(
              temp_2[9], _mm256_mul_pd(matrix[0][3][4], matrix[1][1][5]));
          __m256d temp_3[10];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[9] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[9] = _mm256_add_pd(temp_3[9], temp_1[9]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          temp_3[9] = _mm256_sub_pd(temp_3[9], temp_2[9]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][5][0], temp_3[7]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][5][0], temp_3[8]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][5][0], temp_3[9]));
        }
        n_4_temp[0] = _mm256_sub_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][4][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][4][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][4][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][4][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][4][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][4][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][4][0], n_3_temp[6]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][4][0], n_3_temp[7]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][4][0], n_3_temp[8]));
        n_4_temp[9] = _mm256_sub_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][4][0], n_3_temp[9]));
      }
      n_5_temp[0] = _mm256_add_pd(n_5_temp[0],
                                  _mm256_mul_pd(matrix[4][0][0], n_4_temp[0]));
      n_5_temp[1] = _mm256_add_pd(n_5_temp[1],
                                  _mm256_mul_pd(matrix[4][0][0], n_4_temp[1]));
      n_5_temp[2] = _mm256_add_pd(n_5_temp[2],
                                  _mm256_mul_pd(matrix[4][0][0], n_4_temp[2]));
      n_5_temp[3] = _mm256_add_pd(n_5_temp[3],
                                  _mm256_mul_pd(matrix[4][0][0], n_4_temp[3]));
      n_5_temp[4] = _mm256_add_pd(n_5_temp[4],
                                  _mm256_mul_pd(matrix[4][0][0], n_4_temp[4]));
      n_5_temp[5] = _mm256_add_pd(n_5_temp[5],
                                  _mm256_mul_pd(matrix[4][0][0], n_4_temp[5]));
      n_5_temp[6] = _mm256_add_pd(n_5_temp[6],
                                  _mm256_mul_pd(matrix[4][0][0], n_4_temp[6]));
      n_5_temp[7] = _mm256_add_pd(n_5_temp[7],
                                  _mm256_mul_pd(matrix[4][0][0], n_4_temp[7]));
      n_5_temp[8] = _mm256_add_pd(n_5_temp[8],
                                  _mm256_mul_pd(matrix[4][0][0], n_4_temp[8]));
      n_5_temp[9] = _mm256_add_pd(n_5_temp[9],
                                  _mm256_mul_pd(matrix[4][0][0], n_4_temp[9]));
      n_5_temp[1] = _mm256_add_pd(n_5_temp[1],
                                  _mm256_mul_pd(matrix[4][0][1], n_4_temp[0]));
      n_5_temp[2] = _mm256_add_pd(n_5_temp[2],
                                  _mm256_mul_pd(matrix[4][0][1], n_4_temp[1]));
      n_5_temp[3] = _mm256_add_pd(n_5_temp[3],
                                  _mm256_mul_pd(matrix[4][0][1], n_4_temp[2]));
      n_5_temp[4] = _mm256_add_pd(n_5_temp[4],
                                  _mm256_mul_pd(matrix[4][0][1], n_4_temp[3]));
      n_5_temp[5] = _mm256_add_pd(n_5_temp[5],
                                  _mm256_mul_pd(matrix[4][0][1], n_4_temp[4]));
      n_5_temp[6] = _mm256_add_pd(n_5_temp[6],
                                  _mm256_mul_pd(matrix[4][0][1], n_4_temp[5]));
      n_5_temp[7] = _mm256_add_pd(n_5_temp[7],
                                  _mm256_mul_pd(matrix[4][0][1], n_4_temp[6]));
      n_5_temp[8] = _mm256_add_pd(n_5_temp[8],
                                  _mm256_mul_pd(matrix[4][0][1], n_4_temp[7]));
      n_5_temp[9] = _mm256_add_pd(n_5_temp[9],
                                  _mm256_mul_pd(matrix[4][0][1], n_4_temp[8]));
      n_5_temp[10] = _mm256_add_pd(n_5_temp[10],
                                   _mm256_mul_pd(matrix[4][0][1], n_4_temp[9]));
      n_5_temp[2] = _mm256_add_pd(n_5_temp[2],
                                  _mm256_mul_pd(matrix[4][0][2], n_4_temp[0]));
      n_5_temp[3] = _mm256_add_pd(n_5_temp[3],
                                  _mm256_mul_pd(matrix[4][0][2], n_4_temp[1]));
      n_5_temp[4] = _mm256_add_pd(n_5_temp[4],
                                  _mm256_mul_pd(matrix[4][0][2], n_4_temp[2]));
      n_5_temp[5] = _mm256_add_pd(n_5_temp[5],
                                  _mm256_mul_pd(matrix[4][0][2], n_4_temp[3]));
      n_5_temp[6] = _mm256_add_pd(n_5_temp[6],
                                  _mm256_mul_pd(matrix[4][0][2], n_4_temp[4]));
      n_5_temp[7] = _mm256_add_pd(n_5_temp[7],
                                  _mm256_mul_pd(matrix[4][0][2], n_4_temp[5]));
      n_5_temp[8] = _mm256_add_pd(n_5_temp[8],
                                  _mm256_mul_pd(matrix[4][0][2], n_4_temp[6]));
      n_5_temp[9] = _mm256_add_pd(n_5_temp[9],
                                  _mm256_mul_pd(matrix[4][0][2], n_4_temp[7]));
      n_5_temp[10] = _mm256_add_pd(n_5_temp[10],
                                   _mm256_mul_pd(matrix[4][0][2], n_4_temp[8]));
      n_5_temp[11] = _mm256_add_pd(n_5_temp[11],
                                   _mm256_mul_pd(matrix[4][0][2], n_4_temp[9]));
      n_5_temp[3] = _mm256_add_pd(n_5_temp[3],
                                  _mm256_mul_pd(matrix[4][0][3], n_4_temp[0]));
      n_5_temp[4] = _mm256_add_pd(n_5_temp[4],
                                  _mm256_mul_pd(matrix[4][0][3], n_4_temp[1]));
      n_5_temp[5] = _mm256_add_pd(n_5_temp[5],
                                  _mm256_mul_pd(matrix[4][0][3], n_4_temp[2]));
      n_5_temp[6] = _mm256_add_pd(n_5_temp[6],
                                  _mm256_mul_pd(matrix[4][0][3], n_4_temp[3]));
      n_5_temp[7] = _mm256_add_pd(n_5_temp[7],
                                  _mm256_mul_pd(matrix[4][0][3], n_4_temp[4]));
      n_5_temp[8] = _mm256_add_pd(n_5_temp[8],
                                  _mm256_mul_pd(matrix[4][0][3], n_4_temp[5]));
      n_5_temp[9] = _mm256_add_pd(n_5_temp[9],
                                  _mm256_mul_pd(matrix[4][0][3], n_4_temp[6]));
      n_5_temp[10] = _mm256_add_pd(n_5_temp[10],
                                   _mm256_mul_pd(matrix[4][0][3], n_4_temp[7]));
      n_5_temp[11] = _mm256_add_pd(n_5_temp[11],
                                   _mm256_mul_pd(matrix[4][0][3], n_4_temp[8]));
      n_5_temp[12] = _mm256_add_pd(n_5_temp[12],
                                   _mm256_mul_pd(matrix[4][0][3], n_4_temp[9]));
    }
    {
      __m256d n_4_temp[17];
      n_4_temp[0] = _mm256_setzero_pd();
      n_4_temp[1] = _mm256_setzero_pd();
      n_4_temp[2] = _mm256_setzero_pd();
      n_4_temp[3] = _mm256_setzero_pd();
      n_4_temp[4] = _mm256_setzero_pd();
      n_4_temp[5] = _mm256_setzero_pd();
      n_4_temp[6] = _mm256_setzero_pd();
      n_4_temp[7] = _mm256_setzero_pd();
      n_4_temp[8] = _mm256_setzero_pd();
      n_4_temp[9] = _mm256_setzero_pd();
      n_4_temp[10] = _mm256_setzero_pd();
      n_4_temp[11] = _mm256_setzero_pd();
      n_4_temp[12] = _mm256_setzero_pd();
      n_4_temp[13] = _mm256_setzero_pd();
      n_4_temp[14] = _mm256_setzero_pd();
      n_4_temp[15] = _mm256_setzero_pd();
      n_4_temp[16] = _mm256_setzero_pd();
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[5];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][1]));
          __m256d temp_2[5];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][2]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][2]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][2]));
          __m256d temp_3[5];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][3][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][3][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][0], temp_3[4]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][3][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][1], temp_3[4]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][3][2], temp_3[4]));
        }
        {
          __m256d temp_1[6];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][3][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][3][4], matrix[1][5][1]));
          __m256d temp_2[6];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][3]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][3]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][3]));
          __m256d temp_3[6];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][4][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][0], temp_3[5]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][1], temp_3[5]));
        }
        {
          __m256d temp_1[7];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][3][0], matrix[1][4][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][0], matrix[1][4][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][0], matrix[1][4][2]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][1], matrix[1][4][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][1], matrix[1][4][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][1], matrix[1][4][2]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][2], matrix[1][4][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][2], matrix[1][4][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][2], matrix[1][4][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][3], matrix[1][4][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][3], matrix[1][4][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][3][3], matrix[1][4][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][4], matrix[1][4][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][3][4], matrix[1][4][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][3][4], matrix[1][4][2]));
          __m256d temp_2[7];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][3][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][3][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][0], matrix[1][3][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][0], matrix[1][3][3]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][3][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][3][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][1], matrix[1][3][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][1], matrix[1][3][3]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][3][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][3][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][2], matrix[1][3][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][2], matrix[1][3][3]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][3][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][3][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][3], matrix[1][3][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][3], matrix[1][3][3]));
          __m256d temp_3[7];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
        }
        n_4_temp[0] = _mm256_sub_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][0][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][0][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][0][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][0][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][0][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][0][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][0][0], n_3_temp[6]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][0][1], n_3_temp[0]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][0][1], n_3_temp[1]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][0][1], n_3_temp[2]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][0][1], n_3_temp[3]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][0][1], n_3_temp[4]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][0][1], n_3_temp[5]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][0][1], n_3_temp[6]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][0][2], n_3_temp[0]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][0][2], n_3_temp[1]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][0][2], n_3_temp[2]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][0][2], n_3_temp[3]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][0][2], n_3_temp[4]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][0][2], n_3_temp[5]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][0][2], n_3_temp[6]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][0][3], n_3_temp[0]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][0][3], n_3_temp[1]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][0][3], n_3_temp[2]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][0][3], n_3_temp[3]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][0][3], n_3_temp[4]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][0][3], n_3_temp[5]));
        n_4_temp[9] = _mm256_sub_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][0][3], n_3_temp[6]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][0][4], n_3_temp[0]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][0][4], n_3_temp[1]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][0][4], n_3_temp[2]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][0][4], n_3_temp[3]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][0][4], n_3_temp[4]));
        n_4_temp[9] = _mm256_sub_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][0][4], n_3_temp[5]));
        n_4_temp[10] = _mm256_sub_pd(
            n_4_temp[10], _mm256_mul_pd(matrix[3][0][4], n_3_temp[6]));
      }
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[5];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][1]));
          __m256d temp_2[5];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][2]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][2]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][2]));
          __m256d temp_3[5];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][0][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][0][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][0][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][0], temp_3[4]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][0][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][0][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][1], temp_3[4]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][0][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][2], temp_3[4]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][3], temp_3[0]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][3], temp_3[1]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][3], temp_3[2]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][3], temp_3[3]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][3], temp_3[4]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][4], temp_3[0]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][4], temp_3[1]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][4], temp_3[2]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][4], temp_3[3]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][0][4], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][5], temp_3[0]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][5], temp_3[1]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][5], temp_3[2]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][0][5], temp_3[3]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][0][5], temp_3[4]));
        }
        {
          __m256d temp_1[9];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][0][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][4], matrix[1][5][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][5], matrix[1][5][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][5], matrix[1][5][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][6], matrix[1][5][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][6], matrix[1][5][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][7], matrix[1][5][0]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][7], matrix[1][5][1]));
          __m256d temp_2[9];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][5]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][6]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][5]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][6]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][5]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][6]));
          __m256d temp_3[9];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][4][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][0], temp_3[5]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][0], temp_3[6]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][4][0], temp_3[7]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][4][0], temp_3[8]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][1], temp_3[5]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][4][1], temp_3[6]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][4][1], temp_3[7]));
          n_3_temp[9] = _mm256_sub_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][4][1], temp_3[8]));
        }
        {
          __m256d temp_1[10];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[9] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][0][0], matrix[1][4][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][0], matrix[1][4][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][0], matrix[1][4][2]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][1], matrix[1][4][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][1], matrix[1][4][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][1], matrix[1][4][2]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][2], matrix[1][4][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][2], matrix[1][4][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][2], matrix[1][4][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][3], matrix[1][4][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][3], matrix[1][4][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][3], matrix[1][4][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][4], matrix[1][4][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][4], matrix[1][4][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][4], matrix[1][4][2]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][5], matrix[1][4][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][5], matrix[1][4][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][5], matrix[1][4][2]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][6], matrix[1][4][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][6], matrix[1][4][1]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][6], matrix[1][4][2]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][7], matrix[1][4][0]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][7], matrix[1][4][1]));
          temp_1[9] = _mm256_add_pd(
              temp_1[9], _mm256_mul_pd(matrix[0][0][7], matrix[1][4][2]));
          __m256d temp_2[10];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[9] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][5]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][6]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][5]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][6]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][5]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][6]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][3]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][4]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][5]));
          temp_2[9] = _mm256_add_pd(
              temp_2[9], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][6]));
          __m256d temp_3[10];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[9] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[9] = _mm256_add_pd(temp_3[9], temp_1[9]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          temp_3[9] = _mm256_sub_pd(temp_3[9], temp_2[9]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][5][0], temp_3[7]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][5][0], temp_3[8]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][5][0], temp_3[9]));
        }
        n_4_temp[0] = _mm256_add_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][3][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_add_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][3][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_add_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][3][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_add_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][3][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_add_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][3][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_add_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][3][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_add_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][3][0], n_3_temp[6]));
        n_4_temp[7] = _mm256_add_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][3][0], n_3_temp[7]));
        n_4_temp[8] = _mm256_add_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][3][0], n_3_temp[8]));
        n_4_temp[9] = _mm256_add_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][3][0], n_3_temp[9]));
        n_4_temp[1] = _mm256_add_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][3][1], n_3_temp[0]));
        n_4_temp[2] = _mm256_add_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][3][1], n_3_temp[1]));
        n_4_temp[3] = _mm256_add_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][3][1], n_3_temp[2]));
        n_4_temp[4] = _mm256_add_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][3][1], n_3_temp[3]));
        n_4_temp[5] = _mm256_add_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][3][1], n_3_temp[4]));
        n_4_temp[6] = _mm256_add_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][3][1], n_3_temp[5]));
        n_4_temp[7] = _mm256_add_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][3][1], n_3_temp[6]));
        n_4_temp[8] = _mm256_add_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][3][1], n_3_temp[7]));
        n_4_temp[9] = _mm256_add_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][3][1], n_3_temp[8]));
        n_4_temp[10] = _mm256_add_pd(
            n_4_temp[10], _mm256_mul_pd(matrix[3][3][1], n_3_temp[9]));
      }
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[6];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][3][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][3][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][3][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][3][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][3][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][3][4], matrix[1][5][1]));
          __m256d temp_2[6];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][3][3]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][3][3]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][3][3]));
          __m256d temp_3[6];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][0][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][0][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][0][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][0], temp_3[5]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][0][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][0][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][1], temp_3[4]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][1], temp_3[5]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][0][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][2], temp_3[4]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][2], temp_3[5]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][3], temp_3[0]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][3], temp_3[1]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][3], temp_3[2]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][3], temp_3[3]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][3], temp_3[4]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][0][3], temp_3[5]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][4], temp_3[0]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][4], temp_3[1]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][4], temp_3[2]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][4], temp_3[3]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][0][4], temp_3[4]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][0][4], temp_3[5]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][5], temp_3[0]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][5], temp_3[1]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][5], temp_3[2]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][0][5], temp_3[3]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][0][5], temp_3[4]));
          n_3_temp[10] = _mm256_add_pd(
              n_3_temp[10], _mm256_mul_pd(matrix[2][0][5], temp_3[5]));
        }
        {
          __m256d temp_1[9];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][0][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][4], matrix[1][5][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][5], matrix[1][5][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][5], matrix[1][5][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][6], matrix[1][5][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][6], matrix[1][5][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][7], matrix[1][5][0]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][7], matrix[1][5][1]));
          __m256d temp_2[9];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][5]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][6]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][5]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][6]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][5]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][6]));
          __m256d temp_3[9];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][3][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][3][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][0], temp_3[5]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][3][0], temp_3[6]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][3][0], temp_3[7]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][3][0], temp_3[8]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][3][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][3][1], temp_3[5]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][3][1], temp_3[6]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][3][1], temp_3[7]));
          n_3_temp[9] = _mm256_sub_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][3][1], temp_3[8]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][3][2], temp_3[0]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][3][2], temp_3[1]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][3][2], temp_3[2]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][3][2], temp_3[3]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][3][2], temp_3[4]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][3][2], temp_3[5]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][3][2], temp_3[6]));
          n_3_temp[9] = _mm256_sub_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][3][2], temp_3[7]));
          n_3_temp[10] = _mm256_sub_pd(
              n_3_temp[10], _mm256_mul_pd(matrix[2][3][2], temp_3[8]));
        }
        {
          __m256d temp_1[11];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[9] = _mm256_setzero_pd();
          temp_1[10] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][0][0], matrix[1][3][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][0], matrix[1][3][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][0], matrix[1][3][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][0], matrix[1][3][3]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][1], matrix[1][3][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][1], matrix[1][3][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][1], matrix[1][3][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][1], matrix[1][3][3]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][2], matrix[1][3][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][2], matrix[1][3][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][2], matrix[1][3][2]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][2], matrix[1][3][3]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][3], matrix[1][3][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][3], matrix[1][3][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][3], matrix[1][3][2]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][3], matrix[1][3][3]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][4], matrix[1][3][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][4], matrix[1][3][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][4], matrix[1][3][2]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][4], matrix[1][3][3]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][5], matrix[1][3][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][5], matrix[1][3][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][5], matrix[1][3][2]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][5], matrix[1][3][3]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][6], matrix[1][3][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][6], matrix[1][3][1]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][6], matrix[1][3][2]));
          temp_1[9] = _mm256_add_pd(
              temp_1[9], _mm256_mul_pd(matrix[0][0][6], matrix[1][3][3]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][7], matrix[1][3][0]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][7], matrix[1][3][1]));
          temp_1[9] = _mm256_add_pd(
              temp_1[9], _mm256_mul_pd(matrix[0][0][7], matrix[1][3][2]));
          temp_1[10] = _mm256_add_pd(
              temp_1[10], _mm256_mul_pd(matrix[0][0][7], matrix[1][3][3]));
          __m256d temp_2[11];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[9] = _mm256_setzero_pd();
          temp_2[10] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][3][0], matrix[1][0][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][3][0], matrix[1][0][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][3][0], matrix[1][0][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][3][0], matrix[1][0][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][0], matrix[1][0][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][0], matrix[1][0][5]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][3][0], matrix[1][0][6]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][3][1], matrix[1][0][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][3][1], matrix[1][0][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][3][1], matrix[1][0][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][1], matrix[1][0][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][1], matrix[1][0][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][3][1], matrix[1][0][5]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][3][1], matrix[1][0][6]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][3][2], matrix[1][0][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][3][2], matrix[1][0][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][2], matrix[1][0][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][2], matrix[1][0][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][3][2], matrix[1][0][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][3][2], matrix[1][0][5]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][3][2], matrix[1][0][6]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][3][3], matrix[1][0][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][3], matrix[1][0][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][3], matrix[1][0][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][3][3], matrix[1][0][3]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][3][3], matrix[1][0][4]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][3][3], matrix[1][0][5]));
          temp_2[9] = _mm256_add_pd(
              temp_2[9], _mm256_mul_pd(matrix[0][3][3], matrix[1][0][6]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][3][4], matrix[1][0][0]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][3][4], matrix[1][0][1]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][3][4], matrix[1][0][2]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][3][4], matrix[1][0][3]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][3][4], matrix[1][0][4]));
          temp_2[9] = _mm256_add_pd(
              temp_2[9], _mm256_mul_pd(matrix[0][3][4], matrix[1][0][5]));
          temp_2[10] = _mm256_add_pd(
              temp_2[10], _mm256_mul_pd(matrix[0][3][4], matrix[1][0][6]));
          __m256d temp_3[11];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[9] = _mm256_setzero_pd();
          temp_3[10] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[9] = _mm256_add_pd(temp_3[9], temp_1[9]);
          temp_3[10] = _mm256_add_pd(temp_3[10], temp_1[10]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          temp_3[9] = _mm256_sub_pd(temp_3[9], temp_2[9]);
          temp_3[10] = _mm256_sub_pd(temp_3[10], temp_2[10]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][5][0], temp_3[7]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][5][0], temp_3[8]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][5][0], temp_3[9]));
          n_3_temp[10] = _mm256_add_pd(
              n_3_temp[10], _mm256_mul_pd(matrix[2][5][0], temp_3[10]));
        }
        n_4_temp[0] = _mm256_sub_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][4][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][4][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][4][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][4][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][4][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][4][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][4][0], n_3_temp[6]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][4][0], n_3_temp[7]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][4][0], n_3_temp[8]));
        n_4_temp[9] = _mm256_sub_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][4][0], n_3_temp[9]));
        n_4_temp[10] = _mm256_sub_pd(
            n_4_temp[10], _mm256_mul_pd(matrix[3][4][0], n_3_temp[10]));
      }
      n_5_temp[0] = _mm256_sub_pd(n_5_temp[0],
                                  _mm256_mul_pd(matrix[4][1][0], n_4_temp[0]));
      n_5_temp[1] = _mm256_sub_pd(n_5_temp[1],
                                  _mm256_mul_pd(matrix[4][1][0], n_4_temp[1]));
      n_5_temp[2] = _mm256_sub_pd(n_5_temp[2],
                                  _mm256_mul_pd(matrix[4][1][0], n_4_temp[2]));
      n_5_temp[3] = _mm256_sub_pd(n_5_temp[3],
                                  _mm256_mul_pd(matrix[4][1][0], n_4_temp[3]));
      n_5_temp[4] = _mm256_sub_pd(n_5_temp[4],
                                  _mm256_mul_pd(matrix[4][1][0], n_4_temp[4]));
      n_5_temp[5] = _mm256_sub_pd(n_5_temp[5],
                                  _mm256_mul_pd(matrix[4][1][0], n_4_temp[5]));
      n_5_temp[6] = _mm256_sub_pd(n_5_temp[6],
                                  _mm256_mul_pd(matrix[4][1][0], n_4_temp[6]));
      n_5_temp[7] = _mm256_sub_pd(n_5_temp[7],
                                  _mm256_mul_pd(matrix[4][1][0], n_4_temp[7]));
      n_5_temp[8] = _mm256_sub_pd(n_5_temp[8],
                                  _mm256_mul_pd(matrix[4][1][0], n_4_temp[8]));
      n_5_temp[9] = _mm256_sub_pd(n_5_temp[9],
                                  _mm256_mul_pd(matrix[4][1][0], n_4_temp[9]));
      n_5_temp[10] = _mm256_sub_pd(
          n_5_temp[10], _mm256_mul_pd(matrix[4][1][0], n_4_temp[10]));
      n_5_temp[1] = _mm256_sub_pd(n_5_temp[1],
                                  _mm256_mul_pd(matrix[4][1][1], n_4_temp[0]));
      n_5_temp[2] = _mm256_sub_pd(n_5_temp[2],
                                  _mm256_mul_pd(matrix[4][1][1], n_4_temp[1]));
      n_5_temp[3] = _mm256_sub_pd(n_5_temp[3],
                                  _mm256_mul_pd(matrix[4][1][1], n_4_temp[2]));
      n_5_temp[4] = _mm256_sub_pd(n_5_temp[4],
                                  _mm256_mul_pd(matrix[4][1][1], n_4_temp[3]));
      n_5_temp[5] = _mm256_sub_pd(n_5_temp[5],
                                  _mm256_mul_pd(matrix[4][1][1], n_4_temp[4]));
      n_5_temp[6] = _mm256_sub_pd(n_5_temp[6],
                                  _mm256_mul_pd(matrix[4][1][1], n_4_temp[5]));
      n_5_temp[7] = _mm256_sub_pd(n_5_temp[7],
                                  _mm256_mul_pd(matrix[4][1][1], n_4_temp[6]));
      n_5_temp[8] = _mm256_sub_pd(n_5_temp[8],
                                  _mm256_mul_pd(matrix[4][1][1], n_4_temp[7]));
      n_5_temp[9] = _mm256_sub_pd(n_5_temp[9],
                                  _mm256_mul_pd(matrix[4][1][1], n_4_temp[8]));
      n_5_temp[10] = _mm256_sub_pd(n_5_temp[10],
                                   _mm256_mul_pd(matrix[4][1][1], n_4_temp[9]));
      n_5_temp[11] = _mm256_sub_pd(
          n_5_temp[11], _mm256_mul_pd(matrix[4][1][1], n_4_temp[10]));
      n_5_temp[2] = _mm256_sub_pd(n_5_temp[2],
                                  _mm256_mul_pd(matrix[4][1][2], n_4_temp[0]));
      n_5_temp[3] = _mm256_sub_pd(n_5_temp[3],
                                  _mm256_mul_pd(matrix[4][1][2], n_4_temp[1]));
      n_5_temp[4] = _mm256_sub_pd(n_5_temp[4],
                                  _mm256_mul_pd(matrix[4][1][2], n_4_temp[2]));
      n_5_temp[5] = _mm256_sub_pd(n_5_temp[5],
                                  _mm256_mul_pd(matrix[4][1][2], n_4_temp[3]));
      n_5_temp[6] = _mm256_sub_pd(n_5_temp[6],
                                  _mm256_mul_pd(matrix[4][1][2], n_4_temp[4]));
      n_5_temp[7] = _mm256_sub_pd(n_5_temp[7],
                                  _mm256_mul_pd(matrix[4][1][2], n_4_temp[5]));
      n_5_temp[8] = _mm256_sub_pd(n_5_temp[8],
                                  _mm256_mul_pd(matrix[4][1][2], n_4_temp[6]));
      n_5_temp[9] = _mm256_sub_pd(n_5_temp[9],
                                  _mm256_mul_pd(matrix[4][1][2], n_4_temp[7]));
      n_5_temp[10] = _mm256_sub_pd(n_5_temp[10],
                                   _mm256_mul_pd(matrix[4][1][2], n_4_temp[8]));
      n_5_temp[11] = _mm256_sub_pd(n_5_temp[11],
                                   _mm256_mul_pd(matrix[4][1][2], n_4_temp[9]));
      n_5_temp[12] = _mm256_sub_pd(
          n_5_temp[12], _mm256_mul_pd(matrix[4][1][2], n_4_temp[10]));
    }
    {
      __m256d n_4_temp[17];
      n_4_temp[0] = _mm256_setzero_pd();
      n_4_temp[1] = _mm256_setzero_pd();
      n_4_temp[2] = _mm256_setzero_pd();
      n_4_temp[3] = _mm256_setzero_pd();
      n_4_temp[4] = _mm256_setzero_pd();
      n_4_temp[5] = _mm256_setzero_pd();
      n_4_temp[6] = _mm256_setzero_pd();
      n_4_temp[7] = _mm256_setzero_pd();
      n_4_temp[8] = _mm256_setzero_pd();
      n_4_temp[9] = _mm256_setzero_pd();
      n_4_temp[10] = _mm256_setzero_pd();
      n_4_temp[11] = _mm256_setzero_pd();
      n_4_temp[12] = _mm256_setzero_pd();
      n_4_temp[13] = _mm256_setzero_pd();
      n_4_temp[14] = _mm256_setzero_pd();
      n_4_temp[15] = _mm256_setzero_pd();
      n_4_temp[16] = _mm256_setzero_pd();
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[5];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][1]));
          __m256d temp_2[5];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][2]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][2]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][2]));
          __m256d temp_3[5];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][1][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][1][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][1][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][0], temp_3[4]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][1][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][1][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][1], temp_3[4]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][1][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][2], temp_3[4]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][3], temp_3[0]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][3], temp_3[1]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][3], temp_3[2]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][3], temp_3[3]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][1][3], temp_3[4]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][4], temp_3[0]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][4], temp_3[1]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][4], temp_3[2]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][1][4], temp_3[3]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][1][4], temp_3[4]));
        }
        {
          __m256d temp_1[8];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][1][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][4], matrix[1][5][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][5], matrix[1][5][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][5], matrix[1][5][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][6], matrix[1][5][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][6], matrix[1][5][1]));
          __m256d temp_2[8];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][5]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][5]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][5]));
          __m256d temp_3[8];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][4][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][0], temp_3[5]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][0], temp_3[6]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][4][0], temp_3[7]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][1], temp_3[5]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][4][1], temp_3[6]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][4][1], temp_3[7]));
        }
        {
          __m256d temp_1[9];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][1][0], matrix[1][4][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][0], matrix[1][4][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][0], matrix[1][4][2]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][1], matrix[1][4][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][1], matrix[1][4][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][1], matrix[1][4][2]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][2], matrix[1][4][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][2], matrix[1][4][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][2], matrix[1][4][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][3], matrix[1][4][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][3], matrix[1][4][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][3], matrix[1][4][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][4], matrix[1][4][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][4], matrix[1][4][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][4], matrix[1][4][2]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][5], matrix[1][4][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][5], matrix[1][4][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][5], matrix[1][4][2]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][6], matrix[1][4][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][6], matrix[1][4][1]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][1][6], matrix[1][4][2]));
          __m256d temp_2[9];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][1][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][1][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][0], matrix[1][1][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][0], matrix[1][1][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][0], matrix[1][1][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][0], matrix[1][1][5]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][1][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][1][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][1], matrix[1][1][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][1], matrix[1][1][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][1], matrix[1][1][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][1], matrix[1][1][5]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][1][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][1][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][2], matrix[1][1][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][2], matrix[1][1][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][2], matrix[1][1][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][4][2], matrix[1][1][5]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][1][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][1][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][3], matrix[1][1][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][3], matrix[1][1][3]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][4][3], matrix[1][1][4]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][4][3], matrix[1][1][5]));
          __m256d temp_3[9];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][5][0], temp_3[7]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][5][0], temp_3[8]));
        }
        n_4_temp[0] = _mm256_sub_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][0][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][0][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][0][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][0][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][0][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][0][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][0][0], n_3_temp[6]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][0][0], n_3_temp[7]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][0][0], n_3_temp[8]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][0][1], n_3_temp[0]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][0][1], n_3_temp[1]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][0][1], n_3_temp[2]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][0][1], n_3_temp[3]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][0][1], n_3_temp[4]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][0][1], n_3_temp[5]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][0][1], n_3_temp[6]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][0][1], n_3_temp[7]));
        n_4_temp[9] = _mm256_sub_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][0][1], n_3_temp[8]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][0][2], n_3_temp[0]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][0][2], n_3_temp[1]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][0][2], n_3_temp[2]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][0][2], n_3_temp[3]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][0][2], n_3_temp[4]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][0][2], n_3_temp[5]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][0][2], n_3_temp[6]));
        n_4_temp[9] = _mm256_sub_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][0][2], n_3_temp[7]));
        n_4_temp[10] = _mm256_sub_pd(
            n_4_temp[10], _mm256_mul_pd(matrix[3][0][2], n_3_temp[8]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][0][3], n_3_temp[0]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][0][3], n_3_temp[1]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][0][3], n_3_temp[2]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][0][3], n_3_temp[3]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][0][3], n_3_temp[4]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][0][3], n_3_temp[5]));
        n_4_temp[9] = _mm256_sub_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][0][3], n_3_temp[6]));
        n_4_temp[10] = _mm256_sub_pd(
            n_4_temp[10], _mm256_mul_pd(matrix[3][0][3], n_3_temp[7]));
        n_4_temp[11] = _mm256_sub_pd(
            n_4_temp[11], _mm256_mul_pd(matrix[3][0][3], n_3_temp[8]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][0][4], n_3_temp[0]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][0][4], n_3_temp[1]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][0][4], n_3_temp[2]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][0][4], n_3_temp[3]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][0][4], n_3_temp[4]));
        n_4_temp[9] = _mm256_sub_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][0][4], n_3_temp[5]));
        n_4_temp[10] = _mm256_sub_pd(
            n_4_temp[10], _mm256_mul_pd(matrix[3][0][4], n_3_temp[6]));
        n_4_temp[11] = _mm256_sub_pd(
            n_4_temp[11], _mm256_mul_pd(matrix[3][0][4], n_3_temp[7]));
        n_4_temp[12] = _mm256_sub_pd(
            n_4_temp[12], _mm256_mul_pd(matrix[3][0][4], n_3_temp[8]));
      }
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[5];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][5][1]));
          __m256d temp_2[5];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][4][2]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][4][2]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][4][2]));
          __m256d temp_3[5];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][0][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][0][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][0][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][0], temp_3[4]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][0][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][0][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][1], temp_3[4]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][0][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][2], temp_3[4]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][3], temp_3[0]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][3], temp_3[1]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][3], temp_3[2]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][3], temp_3[3]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][3], temp_3[4]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][4], temp_3[0]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][4], temp_3[1]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][4], temp_3[2]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][4], temp_3[3]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][0][4], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][5], temp_3[0]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][5], temp_3[1]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][5], temp_3[2]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][0][5], temp_3[3]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][0][5], temp_3[4]));
        }
        {
          __m256d temp_1[9];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][0][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][4], matrix[1][5][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][5], matrix[1][5][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][5], matrix[1][5][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][6], matrix[1][5][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][6], matrix[1][5][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][7], matrix[1][5][0]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][7], matrix[1][5][1]));
          __m256d temp_2[9];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][5]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][6]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][5]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][6]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][5]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][6]));
          __m256d temp_3[9];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][4][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][0], temp_3[5]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][0], temp_3[6]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][4][0], temp_3[7]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][4][0], temp_3[8]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][4][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][4][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][4][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][4][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][4][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][4][1], temp_3[5]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][4][1], temp_3[6]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][4][1], temp_3[7]));
          n_3_temp[9] = _mm256_sub_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][4][1], temp_3[8]));
        }
        {
          __m256d temp_1[10];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[9] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][0][0], matrix[1][4][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][0], matrix[1][4][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][0], matrix[1][4][2]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][1], matrix[1][4][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][1], matrix[1][4][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][1], matrix[1][4][2]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][2], matrix[1][4][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][2], matrix[1][4][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][2], matrix[1][4][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][3], matrix[1][4][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][3], matrix[1][4][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][3], matrix[1][4][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][4], matrix[1][4][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][4], matrix[1][4][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][4], matrix[1][4][2]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][5], matrix[1][4][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][5], matrix[1][4][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][5], matrix[1][4][2]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][6], matrix[1][4][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][6], matrix[1][4][1]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][6], matrix[1][4][2]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][7], matrix[1][4][0]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][7], matrix[1][4][1]));
          temp_1[9] = _mm256_add_pd(
              temp_1[9], _mm256_mul_pd(matrix[0][0][7], matrix[1][4][2]));
          __m256d temp_2[10];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[9] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][5]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][0], matrix[1][0][6]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][5]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][4][1], matrix[1][0][6]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][5]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][4][2], matrix[1][0][6]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][3]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][4]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][5]));
          temp_2[9] = _mm256_add_pd(
              temp_2[9], _mm256_mul_pd(matrix[0][4][3], matrix[1][0][6]));
          __m256d temp_3[10];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[9] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[9] = _mm256_add_pd(temp_3[9], temp_1[9]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          temp_3[9] = _mm256_sub_pd(temp_3[9], temp_2[9]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][5][0], temp_3[7]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][5][0], temp_3[8]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][5][0], temp_3[9]));
        }
        n_4_temp[0] = _mm256_add_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][1][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_add_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][1][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_add_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][1][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_add_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][1][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_add_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][1][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_add_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][1][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_add_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][1][0], n_3_temp[6]));
        n_4_temp[7] = _mm256_add_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][1][0], n_3_temp[7]));
        n_4_temp[8] = _mm256_add_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][1][0], n_3_temp[8]));
        n_4_temp[9] = _mm256_add_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][1][0], n_3_temp[9]));
        n_4_temp[1] = _mm256_add_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][1][1], n_3_temp[0]));
        n_4_temp[2] = _mm256_add_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][1][1], n_3_temp[1]));
        n_4_temp[3] = _mm256_add_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][1][1], n_3_temp[2]));
        n_4_temp[4] = _mm256_add_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][1][1], n_3_temp[3]));
        n_4_temp[5] = _mm256_add_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][1][1], n_3_temp[4]));
        n_4_temp[6] = _mm256_add_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][1][1], n_3_temp[5]));
        n_4_temp[7] = _mm256_add_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][1][1], n_3_temp[6]));
        n_4_temp[8] = _mm256_add_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][1][1], n_3_temp[7]));
        n_4_temp[9] = _mm256_add_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][1][1], n_3_temp[8]));
        n_4_temp[10] = _mm256_add_pd(
            n_4_temp[10], _mm256_mul_pd(matrix[3][1][1], n_3_temp[9]));
        n_4_temp[2] = _mm256_add_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][1][2], n_3_temp[0]));
        n_4_temp[3] = _mm256_add_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][1][2], n_3_temp[1]));
        n_4_temp[4] = _mm256_add_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][1][2], n_3_temp[2]));
        n_4_temp[5] = _mm256_add_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][1][2], n_3_temp[3]));
        n_4_temp[6] = _mm256_add_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][1][2], n_3_temp[4]));
        n_4_temp[7] = _mm256_add_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][1][2], n_3_temp[5]));
        n_4_temp[8] = _mm256_add_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][1][2], n_3_temp[6]));
        n_4_temp[9] = _mm256_add_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][1][2], n_3_temp[7]));
        n_4_temp[10] = _mm256_add_pd(
            n_4_temp[10], _mm256_mul_pd(matrix[3][1][2], n_3_temp[8]));
        n_4_temp[11] = _mm256_add_pd(
            n_4_temp[11], _mm256_mul_pd(matrix[3][1][2], n_3_temp[9]));
        n_4_temp[3] = _mm256_add_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][1][3], n_3_temp[0]));
        n_4_temp[4] = _mm256_add_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][1][3], n_3_temp[1]));
        n_4_temp[5] = _mm256_add_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][1][3], n_3_temp[2]));
        n_4_temp[6] = _mm256_add_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][1][3], n_3_temp[3]));
        n_4_temp[7] = _mm256_add_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][1][3], n_3_temp[4]));
        n_4_temp[8] = _mm256_add_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][1][3], n_3_temp[5]));
        n_4_temp[9] = _mm256_add_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][1][3], n_3_temp[6]));
        n_4_temp[10] = _mm256_add_pd(
            n_4_temp[10], _mm256_mul_pd(matrix[3][1][3], n_3_temp[7]));
        n_4_temp[11] = _mm256_add_pd(
            n_4_temp[11], _mm256_mul_pd(matrix[3][1][3], n_3_temp[8]));
        n_4_temp[12] = _mm256_add_pd(
            n_4_temp[12], _mm256_mul_pd(matrix[3][1][3], n_3_temp[9]));
      }
      {
        __m256d n_3_temp[17];
        n_3_temp[0] = _mm256_setzero_pd();
        n_3_temp[1] = _mm256_setzero_pd();
        n_3_temp[2] = _mm256_setzero_pd();
        n_3_temp[3] = _mm256_setzero_pd();
        n_3_temp[4] = _mm256_setzero_pd();
        n_3_temp[5] = _mm256_setzero_pd();
        n_3_temp[6] = _mm256_setzero_pd();
        n_3_temp[7] = _mm256_setzero_pd();
        n_3_temp[8] = _mm256_setzero_pd();
        n_3_temp[9] = _mm256_setzero_pd();
        n_3_temp[10] = _mm256_setzero_pd();
        n_3_temp[11] = _mm256_setzero_pd();
        n_3_temp[12] = _mm256_setzero_pd();
        n_3_temp[13] = _mm256_setzero_pd();
        n_3_temp[14] = _mm256_setzero_pd();
        n_3_temp[15] = _mm256_setzero_pd();
        n_3_temp[16] = _mm256_setzero_pd();
        {
          __m256d temp_1[8];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][1][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][1][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][1][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][1][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][1][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][4], matrix[1][5][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][1][5], matrix[1][5][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][5], matrix[1][5][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][1][6], matrix[1][5][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][1][6], matrix[1][5][1]));
          __m256d temp_2[8];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][0], matrix[1][1][5]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][1], matrix[1][1][5]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][5][2], matrix[1][1][5]));
          __m256d temp_3[8];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][0][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][0][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][0][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][0], temp_3[6]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][0], temp_3[7]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][0][1], temp_3[0]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][0][1], temp_3[1]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][1], temp_3[2]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][1], temp_3[3]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][1], temp_3[4]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][1], temp_3[5]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][1], temp_3[6]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][0][1], temp_3[7]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][0][2], temp_3[0]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][2], temp_3[1]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][2], temp_3[2]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][2], temp_3[3]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][2], temp_3[4]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][2], temp_3[5]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][0][2], temp_3[6]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][0][2], temp_3[7]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][0][3], temp_3[0]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][3], temp_3[1]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][3], temp_3[2]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][3], temp_3[3]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][3], temp_3[4]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][0][3], temp_3[5]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][0][3], temp_3[6]));
          n_3_temp[10] = _mm256_add_pd(
              n_3_temp[10], _mm256_mul_pd(matrix[2][0][3], temp_3[7]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][0][4], temp_3[0]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][4], temp_3[1]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][4], temp_3[2]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][4], temp_3[3]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][0][4], temp_3[4]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][0][4], temp_3[5]));
          n_3_temp[10] = _mm256_add_pd(
              n_3_temp[10], _mm256_mul_pd(matrix[2][0][4], temp_3[6]));
          n_3_temp[11] = _mm256_add_pd(
              n_3_temp[11], _mm256_mul_pd(matrix[2][0][4], temp_3[7]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][0][5], temp_3[0]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][0][5], temp_3[1]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][0][5], temp_3[2]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][0][5], temp_3[3]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][0][5], temp_3[4]));
          n_3_temp[10] = _mm256_add_pd(
              n_3_temp[10], _mm256_mul_pd(matrix[2][0][5], temp_3[5]));
          n_3_temp[11] = _mm256_add_pd(
              n_3_temp[11], _mm256_mul_pd(matrix[2][0][5], temp_3[6]));
          n_3_temp[12] = _mm256_add_pd(
              n_3_temp[12], _mm256_mul_pd(matrix[2][0][5], temp_3[7]));
        }
        {
          __m256d temp_1[9];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][0][0], matrix[1][5][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][0], matrix[1][5][1]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][1], matrix[1][5][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][1], matrix[1][5][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][2], matrix[1][5][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][2], matrix[1][5][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][3], matrix[1][5][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][3], matrix[1][5][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][4], matrix[1][5][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][4], matrix[1][5][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][5], matrix[1][5][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][5], matrix[1][5][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][6], matrix[1][5][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][6], matrix[1][5][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][7], matrix[1][5][0]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][7], matrix[1][5][1]));
          __m256d temp_2[9];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][5]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][0], matrix[1][0][6]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][5]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][5][1], matrix[1][0][6]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][5]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][5][2], matrix[1][0][6]));
          __m256d temp_3[9];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          n_3_temp[0] = _mm256_sub_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][1][0], temp_3[0]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][1][0], temp_3[1]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][1][0], temp_3[2]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][0], temp_3[3]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][0], temp_3[4]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][0], temp_3[5]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][0], temp_3[6]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][1][0], temp_3[7]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][1][0], temp_3[8]));
          n_3_temp[1] = _mm256_sub_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][1][1], temp_3[0]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][1][1], temp_3[1]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][1], temp_3[2]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][1], temp_3[3]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][1], temp_3[4]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][1], temp_3[5]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][1][1], temp_3[6]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][1][1], temp_3[7]));
          n_3_temp[9] = _mm256_sub_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][1][1], temp_3[8]));
          n_3_temp[2] = _mm256_sub_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][1][2], temp_3[0]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][2], temp_3[1]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][2], temp_3[2]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][2], temp_3[3]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][2], temp_3[4]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][1][2], temp_3[5]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][1][2], temp_3[6]));
          n_3_temp[9] = _mm256_sub_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][1][2], temp_3[7]));
          n_3_temp[10] = _mm256_sub_pd(
              n_3_temp[10], _mm256_mul_pd(matrix[2][1][2], temp_3[8]));
          n_3_temp[3] = _mm256_sub_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][1][3], temp_3[0]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][3], temp_3[1]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][3], temp_3[2]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][3], temp_3[3]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][1][3], temp_3[4]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][1][3], temp_3[5]));
          n_3_temp[9] = _mm256_sub_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][1][3], temp_3[6]));
          n_3_temp[10] = _mm256_sub_pd(
              n_3_temp[10], _mm256_mul_pd(matrix[2][1][3], temp_3[7]));
          n_3_temp[11] = _mm256_sub_pd(
              n_3_temp[11], _mm256_mul_pd(matrix[2][1][3], temp_3[8]));
          n_3_temp[4] = _mm256_sub_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][1][4], temp_3[0]));
          n_3_temp[5] = _mm256_sub_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][1][4], temp_3[1]));
          n_3_temp[6] = _mm256_sub_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][1][4], temp_3[2]));
          n_3_temp[7] = _mm256_sub_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][1][4], temp_3[3]));
          n_3_temp[8] = _mm256_sub_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][1][4], temp_3[4]));
          n_3_temp[9] = _mm256_sub_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][1][4], temp_3[5]));
          n_3_temp[10] = _mm256_sub_pd(
              n_3_temp[10], _mm256_mul_pd(matrix[2][1][4], temp_3[6]));
          n_3_temp[11] = _mm256_sub_pd(
              n_3_temp[11], _mm256_mul_pd(matrix[2][1][4], temp_3[7]));
          n_3_temp[12] = _mm256_sub_pd(
              n_3_temp[12], _mm256_mul_pd(matrix[2][1][4], temp_3[8]));
        }
        {
          __m256d temp_1[13];
          temp_1[0] = _mm256_setzero_pd();
          temp_1[1] = _mm256_setzero_pd();
          temp_1[2] = _mm256_setzero_pd();
          temp_1[3] = _mm256_setzero_pd();
          temp_1[4] = _mm256_setzero_pd();
          temp_1[5] = _mm256_setzero_pd();
          temp_1[6] = _mm256_setzero_pd();
          temp_1[7] = _mm256_setzero_pd();
          temp_1[8] = _mm256_setzero_pd();
          temp_1[9] = _mm256_setzero_pd();
          temp_1[10] = _mm256_setzero_pd();
          temp_1[11] = _mm256_setzero_pd();
          temp_1[12] = _mm256_setzero_pd();
          temp_1[0] = _mm256_add_pd(
              temp_1[0], _mm256_mul_pd(matrix[0][0][0], matrix[1][1][0]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][0], matrix[1][1][1]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][0], matrix[1][1][2]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][0], matrix[1][1][3]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][0], matrix[1][1][4]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][0], matrix[1][1][5]));
          temp_1[1] = _mm256_add_pd(
              temp_1[1], _mm256_mul_pd(matrix[0][0][1], matrix[1][1][0]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][1], matrix[1][1][1]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][1], matrix[1][1][2]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][1], matrix[1][1][3]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][1], matrix[1][1][4]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][1], matrix[1][1][5]));
          temp_1[2] = _mm256_add_pd(
              temp_1[2], _mm256_mul_pd(matrix[0][0][2], matrix[1][1][0]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][2], matrix[1][1][1]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][2], matrix[1][1][2]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][2], matrix[1][1][3]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][2], matrix[1][1][4]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][2], matrix[1][1][5]));
          temp_1[3] = _mm256_add_pd(
              temp_1[3], _mm256_mul_pd(matrix[0][0][3], matrix[1][1][0]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][3], matrix[1][1][1]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][3], matrix[1][1][2]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][3], matrix[1][1][3]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][3], matrix[1][1][4]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][3], matrix[1][1][5]));
          temp_1[4] = _mm256_add_pd(
              temp_1[4], _mm256_mul_pd(matrix[0][0][4], matrix[1][1][0]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][4], matrix[1][1][1]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][4], matrix[1][1][2]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][4], matrix[1][1][3]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][4], matrix[1][1][4]));
          temp_1[9] = _mm256_add_pd(
              temp_1[9], _mm256_mul_pd(matrix[0][0][4], matrix[1][1][5]));
          temp_1[5] = _mm256_add_pd(
              temp_1[5], _mm256_mul_pd(matrix[0][0][5], matrix[1][1][0]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][5], matrix[1][1][1]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][5], matrix[1][1][2]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][5], matrix[1][1][3]));
          temp_1[9] = _mm256_add_pd(
              temp_1[9], _mm256_mul_pd(matrix[0][0][5], matrix[1][1][4]));
          temp_1[10] = _mm256_add_pd(
              temp_1[10], _mm256_mul_pd(matrix[0][0][5], matrix[1][1][5]));
          temp_1[6] = _mm256_add_pd(
              temp_1[6], _mm256_mul_pd(matrix[0][0][6], matrix[1][1][0]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][6], matrix[1][1][1]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][6], matrix[1][1][2]));
          temp_1[9] = _mm256_add_pd(
              temp_1[9], _mm256_mul_pd(matrix[0][0][6], matrix[1][1][3]));
          temp_1[10] = _mm256_add_pd(
              temp_1[10], _mm256_mul_pd(matrix[0][0][6], matrix[1][1][4]));
          temp_1[11] = _mm256_add_pd(
              temp_1[11], _mm256_mul_pd(matrix[0][0][6], matrix[1][1][5]));
          temp_1[7] = _mm256_add_pd(
              temp_1[7], _mm256_mul_pd(matrix[0][0][7], matrix[1][1][0]));
          temp_1[8] = _mm256_add_pd(
              temp_1[8], _mm256_mul_pd(matrix[0][0][7], matrix[1][1][1]));
          temp_1[9] = _mm256_add_pd(
              temp_1[9], _mm256_mul_pd(matrix[0][0][7], matrix[1][1][2]));
          temp_1[10] = _mm256_add_pd(
              temp_1[10], _mm256_mul_pd(matrix[0][0][7], matrix[1][1][3]));
          temp_1[11] = _mm256_add_pd(
              temp_1[11], _mm256_mul_pd(matrix[0][0][7], matrix[1][1][4]));
          temp_1[12] = _mm256_add_pd(
              temp_1[12], _mm256_mul_pd(matrix[0][0][7], matrix[1][1][5]));
          __m256d temp_2[13];
          temp_2[0] = _mm256_setzero_pd();
          temp_2[1] = _mm256_setzero_pd();
          temp_2[2] = _mm256_setzero_pd();
          temp_2[3] = _mm256_setzero_pd();
          temp_2[4] = _mm256_setzero_pd();
          temp_2[5] = _mm256_setzero_pd();
          temp_2[6] = _mm256_setzero_pd();
          temp_2[7] = _mm256_setzero_pd();
          temp_2[8] = _mm256_setzero_pd();
          temp_2[9] = _mm256_setzero_pd();
          temp_2[10] = _mm256_setzero_pd();
          temp_2[11] = _mm256_setzero_pd();
          temp_2[12] = _mm256_setzero_pd();
          temp_2[0] = _mm256_add_pd(
              temp_2[0], _mm256_mul_pd(matrix[0][1][0], matrix[1][0][0]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][1][0], matrix[1][0][1]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][1][0], matrix[1][0][2]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][1][0], matrix[1][0][3]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][1][0], matrix[1][0][4]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][1][0], matrix[1][0][5]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][1][0], matrix[1][0][6]));
          temp_2[1] = _mm256_add_pd(
              temp_2[1], _mm256_mul_pd(matrix[0][1][1], matrix[1][0][0]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][1][1], matrix[1][0][1]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][1][1], matrix[1][0][2]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][1][1], matrix[1][0][3]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][1][1], matrix[1][0][4]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][1][1], matrix[1][0][5]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][1][1], matrix[1][0][6]));
          temp_2[2] = _mm256_add_pd(
              temp_2[2], _mm256_mul_pd(matrix[0][1][2], matrix[1][0][0]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][1][2], matrix[1][0][1]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][1][2], matrix[1][0][2]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][1][2], matrix[1][0][3]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][1][2], matrix[1][0][4]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][1][2], matrix[1][0][5]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][1][2], matrix[1][0][6]));
          temp_2[3] = _mm256_add_pd(
              temp_2[3], _mm256_mul_pd(matrix[0][1][3], matrix[1][0][0]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][1][3], matrix[1][0][1]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][1][3], matrix[1][0][2]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][1][3], matrix[1][0][3]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][1][3], matrix[1][0][4]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][1][3], matrix[1][0][5]));
          temp_2[9] = _mm256_add_pd(
              temp_2[9], _mm256_mul_pd(matrix[0][1][3], matrix[1][0][6]));
          temp_2[4] = _mm256_add_pd(
              temp_2[4], _mm256_mul_pd(matrix[0][1][4], matrix[1][0][0]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][1][4], matrix[1][0][1]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][1][4], matrix[1][0][2]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][1][4], matrix[1][0][3]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][1][4], matrix[1][0][4]));
          temp_2[9] = _mm256_add_pd(
              temp_2[9], _mm256_mul_pd(matrix[0][1][4], matrix[1][0][5]));
          temp_2[10] = _mm256_add_pd(
              temp_2[10], _mm256_mul_pd(matrix[0][1][4], matrix[1][0][6]));
          temp_2[5] = _mm256_add_pd(
              temp_2[5], _mm256_mul_pd(matrix[0][1][5], matrix[1][0][0]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][1][5], matrix[1][0][1]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][1][5], matrix[1][0][2]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][1][5], matrix[1][0][3]));
          temp_2[9] = _mm256_add_pd(
              temp_2[9], _mm256_mul_pd(matrix[0][1][5], matrix[1][0][4]));
          temp_2[10] = _mm256_add_pd(
              temp_2[10], _mm256_mul_pd(matrix[0][1][5], matrix[1][0][5]));
          temp_2[11] = _mm256_add_pd(
              temp_2[11], _mm256_mul_pd(matrix[0][1][5], matrix[1][0][6]));
          temp_2[6] = _mm256_add_pd(
              temp_2[6], _mm256_mul_pd(matrix[0][1][6], matrix[1][0][0]));
          temp_2[7] = _mm256_add_pd(
              temp_2[7], _mm256_mul_pd(matrix[0][1][6], matrix[1][0][1]));
          temp_2[8] = _mm256_add_pd(
              temp_2[8], _mm256_mul_pd(matrix[0][1][6], matrix[1][0][2]));
          temp_2[9] = _mm256_add_pd(
              temp_2[9], _mm256_mul_pd(matrix[0][1][6], matrix[1][0][3]));
          temp_2[10] = _mm256_add_pd(
              temp_2[10], _mm256_mul_pd(matrix[0][1][6], matrix[1][0][4]));
          temp_2[11] = _mm256_add_pd(
              temp_2[11], _mm256_mul_pd(matrix[0][1][6], matrix[1][0][5]));
          temp_2[12] = _mm256_add_pd(
              temp_2[12], _mm256_mul_pd(matrix[0][1][6], matrix[1][0][6]));
          __m256d temp_3[13];
          temp_3[0] = _mm256_setzero_pd();
          temp_3[1] = _mm256_setzero_pd();
          temp_3[2] = _mm256_setzero_pd();
          temp_3[3] = _mm256_setzero_pd();
          temp_3[4] = _mm256_setzero_pd();
          temp_3[5] = _mm256_setzero_pd();
          temp_3[6] = _mm256_setzero_pd();
          temp_3[7] = _mm256_setzero_pd();
          temp_3[8] = _mm256_setzero_pd();
          temp_3[9] = _mm256_setzero_pd();
          temp_3[10] = _mm256_setzero_pd();
          temp_3[11] = _mm256_setzero_pd();
          temp_3[12] = _mm256_setzero_pd();
          temp_3[0] = _mm256_add_pd(temp_3[0], temp_1[0]);
          temp_3[1] = _mm256_add_pd(temp_3[1], temp_1[1]);
          temp_3[2] = _mm256_add_pd(temp_3[2], temp_1[2]);
          temp_3[3] = _mm256_add_pd(temp_3[3], temp_1[3]);
          temp_3[4] = _mm256_add_pd(temp_3[4], temp_1[4]);
          temp_3[5] = _mm256_add_pd(temp_3[5], temp_1[5]);
          temp_3[6] = _mm256_add_pd(temp_3[6], temp_1[6]);
          temp_3[7] = _mm256_add_pd(temp_3[7], temp_1[7]);
          temp_3[8] = _mm256_add_pd(temp_3[8], temp_1[8]);
          temp_3[9] = _mm256_add_pd(temp_3[9], temp_1[9]);
          temp_3[10] = _mm256_add_pd(temp_3[10], temp_1[10]);
          temp_3[11] = _mm256_add_pd(temp_3[11], temp_1[11]);
          temp_3[12] = _mm256_add_pd(temp_3[12], temp_1[12]);
          temp_3[0] = _mm256_sub_pd(temp_3[0], temp_2[0]);
          temp_3[1] = _mm256_sub_pd(temp_3[1], temp_2[1]);
          temp_3[2] = _mm256_sub_pd(temp_3[2], temp_2[2]);
          temp_3[3] = _mm256_sub_pd(temp_3[3], temp_2[3]);
          temp_3[4] = _mm256_sub_pd(temp_3[4], temp_2[4]);
          temp_3[5] = _mm256_sub_pd(temp_3[5], temp_2[5]);
          temp_3[6] = _mm256_sub_pd(temp_3[6], temp_2[6]);
          temp_3[7] = _mm256_sub_pd(temp_3[7], temp_2[7]);
          temp_3[8] = _mm256_sub_pd(temp_3[8], temp_2[8]);
          temp_3[9] = _mm256_sub_pd(temp_3[9], temp_2[9]);
          temp_3[10] = _mm256_sub_pd(temp_3[10], temp_2[10]);
          temp_3[11] = _mm256_sub_pd(temp_3[11], temp_2[11]);
          temp_3[12] = _mm256_sub_pd(temp_3[12], temp_2[12]);
          n_3_temp[0] = _mm256_add_pd(
              n_3_temp[0], _mm256_mul_pd(matrix[2][5][0], temp_3[0]));
          n_3_temp[1] = _mm256_add_pd(
              n_3_temp[1], _mm256_mul_pd(matrix[2][5][0], temp_3[1]));
          n_3_temp[2] = _mm256_add_pd(
              n_3_temp[2], _mm256_mul_pd(matrix[2][5][0], temp_3[2]));
          n_3_temp[3] = _mm256_add_pd(
              n_3_temp[3], _mm256_mul_pd(matrix[2][5][0], temp_3[3]));
          n_3_temp[4] = _mm256_add_pd(
              n_3_temp[4], _mm256_mul_pd(matrix[2][5][0], temp_3[4]));
          n_3_temp[5] = _mm256_add_pd(
              n_3_temp[5], _mm256_mul_pd(matrix[2][5][0], temp_3[5]));
          n_3_temp[6] = _mm256_add_pd(
              n_3_temp[6], _mm256_mul_pd(matrix[2][5][0], temp_3[6]));
          n_3_temp[7] = _mm256_add_pd(
              n_3_temp[7], _mm256_mul_pd(matrix[2][5][0], temp_3[7]));
          n_3_temp[8] = _mm256_add_pd(
              n_3_temp[8], _mm256_mul_pd(matrix[2][5][0], temp_3[8]));
          n_3_temp[9] = _mm256_add_pd(
              n_3_temp[9], _mm256_mul_pd(matrix[2][5][0], temp_3[9]));
          n_3_temp[10] = _mm256_add_pd(
              n_3_temp[10], _mm256_mul_pd(matrix[2][5][0], temp_3[10]));
          n_3_temp[11] = _mm256_add_pd(
              n_3_temp[11], _mm256_mul_pd(matrix[2][5][0], temp_3[11]));
          n_3_temp[12] = _mm256_add_pd(
              n_3_temp[12], _mm256_mul_pd(matrix[2][5][0], temp_3[12]));
        }
        n_4_temp[0] = _mm256_sub_pd(
            n_4_temp[0], _mm256_mul_pd(matrix[3][4][0], n_3_temp[0]));
        n_4_temp[1] = _mm256_sub_pd(
            n_4_temp[1], _mm256_mul_pd(matrix[3][4][0], n_3_temp[1]));
        n_4_temp[2] = _mm256_sub_pd(
            n_4_temp[2], _mm256_mul_pd(matrix[3][4][0], n_3_temp[2]));
        n_4_temp[3] = _mm256_sub_pd(
            n_4_temp[3], _mm256_mul_pd(matrix[3][4][0], n_3_temp[3]));
        n_4_temp[4] = _mm256_sub_pd(
            n_4_temp[4], _mm256_mul_pd(matrix[3][4][0], n_3_temp[4]));
        n_4_temp[5] = _mm256_sub_pd(
            n_4_temp[5], _mm256_mul_pd(matrix[3][4][0], n_3_temp[5]));
        n_4_temp[6] = _mm256_sub_pd(
            n_4_temp[6], _mm256_mul_pd(matrix[3][4][0], n_3_temp[6]));
        n_4_temp[7] = _mm256_sub_pd(
            n_4_temp[7], _mm256_mul_pd(matrix[3][4][0], n_3_temp[7]));
        n_4_temp[8] = _mm256_sub_pd(
            n_4_temp[8], _mm256_mul_pd(matrix[3][4][0], n_3_temp[8]));
        n_4_temp[9] = _mm256_sub_pd(
            n_4_temp[9], _mm256_mul_pd(matrix[3][4][0], n_3_temp[9]));
        n_4_temp[10] = _mm256_sub_pd(
            n_4_temp[10], _mm256_mul_pd(matrix[3][4][0], n_3_temp[10]));
        n_4_temp[11] = _mm256_sub_pd(
            n_4_temp[11], _mm256_mul_pd(matrix[3][4][0], n_3_temp[11]));
        n_4_temp[12] = _mm256_sub_pd(
            n_4_temp[12], _mm256_mul_pd(matrix[3][4][0], n_3_temp[12]));
      }
      n_5_temp[0] = _mm256_add_pd(n_5_temp[0],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[0]));
      n_5_temp[1] = _mm256_add_pd(n_5_temp[1],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[1]));
      n_5_temp[2] = _mm256_add_pd(n_5_temp[2],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[2]));
      n_5_temp[3] = _mm256_add_pd(n_5_temp[3],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[3]));
      n_5_temp[4] = _mm256_add_pd(n_5_temp[4],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[4]));
      n_5_temp[5] = _mm256_add_pd(n_5_temp[5],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[5]));
      n_5_temp[6] = _mm256_add_pd(n_5_temp[6],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[6]));
      n_5_temp[7] = _mm256_add_pd(n_5_temp[7],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[7]));
      n_5_temp[8] = _mm256_add_pd(n_5_temp[8],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[8]));
      n_5_temp[9] = _mm256_add_pd(n_5_temp[9],
                                  _mm256_mul_pd(matrix[4][3][0], n_4_temp[9]));
      n_5_temp[10] = _mm256_add_pd(
          n_5_temp[10], _mm256_mul_pd(matrix[4][3][0], n_4_temp[10]));
      n_5_temp[11] = _mm256_add_pd(
          n_5_temp[11], _mm256_mul_pd(matrix[4][3][0], n_4_temp[11]));
      n_5_temp[12] = _mm256_add_pd(
          n_5_temp[12], _mm256_mul_pd(matrix[4][3][0], n_4_temp[12]));
    }
    n_6_temp[0] =
        _mm256_sub_pd(n_6_temp[0], _mm256_mul_pd(matrix[5][2][0], n_5_temp[0]));
    n_6_temp[1] =
        _mm256_sub_pd(n_6_temp[1], _mm256_mul_pd(matrix[5][2][0], n_5_temp[1]));
    n_6_temp[2] =
        _mm256_sub_pd(n_6_temp[2], _mm256_mul_pd(matrix[5][2][0], n_5_temp[2]));
    n_6_temp[3] =
        _mm256_sub_pd(n_6_temp[3], _mm256_mul_pd(matrix[5][2][0], n_5_temp[3]));
    n_6_temp[4] =
        _mm256_sub_pd(n_6_temp[4], _mm256_mul_pd(matrix[5][2][0], n_5_temp[4]));
    n_6_temp[5] =
        _mm256_sub_pd(n_6_temp[5], _mm256_mul_pd(matrix[5][2][0], n_5_temp[5]));
    n_6_temp[6] =
        _mm256_sub_pd(n_6_temp[6], _mm256_mul_pd(matrix[5][2][0], n_5_temp[6]));
    n_6_temp[7] =
        _mm256_sub_pd(n_6_temp[7], _mm256_mul_pd(matrix[5][2][0], n_5_temp[7]));
    n_6_temp[8] =
        _mm256_sub_pd(n_6_temp[8], _mm256_mul_pd(matrix[5][2][0], n_5_temp[8]));
    n_6_temp[9] =
        _mm256_sub_pd(n_6_temp[9], _mm256_mul_pd(matrix[5][2][0], n_5_temp[9]));
    n_6_temp[10] = _mm256_sub_pd(n_6_temp[10],
                                 _mm256_mul_pd(matrix[5][2][0], n_5_temp[10]));
    n_6_temp[11] = _mm256_sub_pd(n_6_temp[11],
                                 _mm256_mul_pd(matrix[5][2][0], n_5_temp[11]));
    n_6_temp[12] = _mm256_sub_pd(n_6_temp[12],
                                 _mm256_mul_pd(matrix[5][2][0], n_5_temp[12]));
  };

  // packed res_poly to Resultant::UnivariatePolynomial

  std::vector<std::vector<
      std::pair<Resultant::UnivariatePolynomial, std::pair<double, double>>>>
      result;

  std::vector<
      std::pair<Resultant::UnivariatePolynomial, std::pair<double, double>>>
      res_0;
  std::vector<
      std::pair<Resultant::UnivariatePolynomial, std::pair<double, double>>>
      res_1;
  std::vector<
      std::pair<Resultant::UnivariatePolynomial, std::pair<double, double>>>
      res_2;
  std::vector<
      std::pair<Resultant::UnivariatePolynomial, std::pair<double, double>>>
      res_3;

  double unpacked_poly[17][4];

  _mm256_store_pd(&unpacked_poly[0][0], n_6_temp[0]);
  _mm256_store_pd(&unpacked_poly[1][0], n_6_temp[1]);
  _mm256_store_pd(&unpacked_poly[2][0], n_6_temp[2]);
  _mm256_store_pd(&unpacked_poly[3][0], n_6_temp[3]);
  _mm256_store_pd(&unpacked_poly[4][0], n_6_temp[4]);
  _mm256_store_pd(&unpacked_poly[5][0], n_6_temp[5]);
  _mm256_store_pd(&unpacked_poly[6][0], n_6_temp[6]);
  _mm256_store_pd(&unpacked_poly[7][0], n_6_temp[7]);
  _mm256_store_pd(&unpacked_poly[8][0], n_6_temp[8]);
  _mm256_store_pd(&unpacked_poly[9][0], n_6_temp[9]);
  _mm256_store_pd(&unpacked_poly[10][0], n_6_temp[10]);
  _mm256_store_pd(&unpacked_poly[11][0], n_6_temp[11]);
  _mm256_store_pd(&unpacked_poly[12][0], n_6_temp[12]);
  _mm256_store_pd(&unpacked_poly[13][0], n_6_temp[13]);
  _mm256_store_pd(&unpacked_poly[14][0], n_6_temp[14]);
  _mm256_store_pd(&unpacked_poly[15][0], n_6_temp[15]);
  _mm256_store_pd(&unpacked_poly[16][0], n_6_temp[16]);

  std::vector<double> poly_0 = {
      unpacked_poly[0][0],  unpacked_poly[1][0],  unpacked_poly[2][0],
      unpacked_poly[3][0],  unpacked_poly[4][0],  unpacked_poly[5][0],
      unpacked_poly[6][0],  unpacked_poly[7][0],  unpacked_poly[8][0],
      unpacked_poly[9][0],  unpacked_poly[10][0], unpacked_poly[11][0],
      unpacked_poly[12][0], unpacked_poly[13][0], unpacked_poly[14][0],
      unpacked_poly[15][0], unpacked_poly[16][0]};
  std::vector<double> poly_1 = {
      unpacked_poly[0][1],  unpacked_poly[1][1],  unpacked_poly[2][1],
      unpacked_poly[3][1],  unpacked_poly[4][1],  unpacked_poly[5][1],
      unpacked_poly[6][1],  unpacked_poly[7][1],  unpacked_poly[8][1],
      unpacked_poly[9][1],  unpacked_poly[10][1], unpacked_poly[11][1],
      unpacked_poly[12][1], unpacked_poly[13][1], unpacked_poly[14][1],
      unpacked_poly[15][1], unpacked_poly[16][1]};
  std::vector<double> poly_2 = {
      unpacked_poly[0][2],  unpacked_poly[1][2],  unpacked_poly[2][2],
      unpacked_poly[3][2],  unpacked_poly[4][2],  unpacked_poly[5][2],
      unpacked_poly[6][2],  unpacked_poly[7][2],  unpacked_poly[8][2],
      unpacked_poly[9][2],  unpacked_poly[10][2], unpacked_poly[11][2],
      unpacked_poly[12][2], unpacked_poly[13][2], unpacked_poly[14][2],
      unpacked_poly[15][2], unpacked_poly[16][2]};
  std::vector<double> poly_3 = {
      unpacked_poly[0][3],  unpacked_poly[1][3],  unpacked_poly[2][3],
      unpacked_poly[3][3],  unpacked_poly[4][3],  unpacked_poly[5][3],
      unpacked_poly[6][3],  unpacked_poly[7][3],  unpacked_poly[8][3],
      unpacked_poly[9][3],  unpacked_poly[10][3], unpacked_poly[11][3],
      unpacked_poly[12][3], unpacked_poly[13][3], unpacked_poly[14][3],
      unpacked_poly[15][3], unpacked_poly[16][3]};

  res_0.push_back(std::make_pair(Resultant::UnivariatePolynomial(poly_0),
                                 std::make_pair(0.0, 1.0)));
  res_1.push_back(std::make_pair(Resultant::UnivariatePolynomial(poly_1),
                                 std::make_pair(0.0, 1.0)));
  res_2.push_back(std::make_pair(Resultant::UnivariatePolynomial(poly_2),
                                 std::make_pair(0.0, 1.0)));
  res_3.push_back(std::make_pair(Resultant::UnivariatePolynomial(poly_3),
                                 std::make_pair(0.0, 1.0)));

  result.emplace_back(res_0);
  result.emplace_back(res_1);
  result.emplace_back(res_2);
  result.emplace_back(res_3);

  return result;
}

#else
template <>
std::vector<std::vector<
    std::pair<Resultant::UnivariatePolynomial, std::pair<double, double>>>>
UnivariatePolyMatrix<6ul>::determinant() const {

  __m256d res_poly[17];
  res_poly[0] = _mm256_setzero_pd();
  res_poly[1] = _mm256_setzero_pd();
  res_poly[2] = _mm256_setzero_pd();
  res_poly[3] = _mm256_setzero_pd();
  res_poly[4] = _mm256_setzero_pd();
  res_poly[5] = _mm256_setzero_pd();
  res_poly[6] = _mm256_setzero_pd();
  res_poly[7] = _mm256_setzero_pd();
  res_poly[8] = _mm256_setzero_pd();
  res_poly[9] = _mm256_setzero_pd();
  res_poly[10] = _mm256_setzero_pd();
  res_poly[11] = _mm256_setzero_pd();
  res_poly[12] = _mm256_setzero_pd();
  res_poly[13] = _mm256_setzero_pd();
  res_poly[14] = _mm256_setzero_pd();
  res_poly[15] = _mm256_setzero_pd();
  res_poly[16] = _mm256_setzero_pd();

  opt_onebounce_6x6::det6x6(matrix, res_poly);

  double unpacked_poly[17][4];

  std::vector<std::vector<
      std::pair<Resultant::UnivariatePolynomial, std::pair<double, double>>>>
      result;

  std::vector<
      std::pair<Resultant::UnivariatePolynomial, std::pair<double, double>>>
      res_0;
  std::vector<
      std::pair<Resultant::UnivariatePolynomial, std::pair<double, double>>>
      res_1;
  std::vector<
      std::pair<Resultant::UnivariatePolynomial, std::pair<double, double>>>
      res_2;
  std::vector<
      std::pair<Resultant::UnivariatePolynomial, std::pair<double, double>>>
      res_3;

  _mm256_store_pd(&unpacked_poly[0][0], res_poly[0]);
  _mm256_store_pd(&unpacked_poly[1][0], res_poly[1]);
  _mm256_store_pd(&unpacked_poly[2][0], res_poly[2]);
  _mm256_store_pd(&unpacked_poly[3][0], res_poly[3]);
  _mm256_store_pd(&unpacked_poly[4][0], res_poly[4]);
  _mm256_store_pd(&unpacked_poly[5][0], res_poly[5]);
  _mm256_store_pd(&unpacked_poly[6][0], res_poly[6]);
  _mm256_store_pd(&unpacked_poly[7][0], res_poly[7]);
  _mm256_store_pd(&unpacked_poly[8][0], res_poly[8]);
  _mm256_store_pd(&unpacked_poly[9][0], res_poly[9]);
  _mm256_store_pd(&unpacked_poly[10][0], res_poly[10]);
  _mm256_store_pd(&unpacked_poly[11][0], res_poly[11]);
  _mm256_store_pd(&unpacked_poly[12][0], res_poly[12]);
  _mm256_store_pd(&unpacked_poly[13][0], res_poly[13]);
  _mm256_store_pd(&unpacked_poly[14][0], res_poly[14]);
  _mm256_store_pd(&unpacked_poly[15][0], res_poly[15]);
  _mm256_store_pd(&unpacked_poly[16][0], res_poly[16]);

  std::vector<double> poly_0 = {
      unpacked_poly[0][0],  unpacked_poly[1][0],  unpacked_poly[2][0],
      unpacked_poly[3][0],  unpacked_poly[4][0],  unpacked_poly[5][0],
      unpacked_poly[6][0],  unpacked_poly[7][0],  unpacked_poly[8][0],
      unpacked_poly[9][0],  unpacked_poly[10][0], unpacked_poly[11][0],
      unpacked_poly[12][0], unpacked_poly[13][0], unpacked_poly[14][0],
      unpacked_poly[15][0], unpacked_poly[16][0]};
  std::vector<double> poly_1 = {
      unpacked_poly[0][1],  unpacked_poly[1][1],  unpacked_poly[2][1],
      unpacked_poly[3][1],  unpacked_poly[4][1],  unpacked_poly[5][1],
      unpacked_poly[6][1],  unpacked_poly[7][1],  unpacked_poly[8][1],
      unpacked_poly[9][1],  unpacked_poly[10][1], unpacked_poly[11][1],
      unpacked_poly[12][1], unpacked_poly[13][1], unpacked_poly[14][1],
      unpacked_poly[15][1], unpacked_poly[16][1]};
  std::vector<double> poly_2 = {
      unpacked_poly[0][2],  unpacked_poly[1][2],  unpacked_poly[2][2],
      unpacked_poly[3][2],  unpacked_poly[4][2],  unpacked_poly[5][2],
      unpacked_poly[6][2],  unpacked_poly[7][2],  unpacked_poly[8][2],
      unpacked_poly[9][2],  unpacked_poly[10][2], unpacked_poly[11][2],
      unpacked_poly[12][2], unpacked_poly[13][2], unpacked_poly[14][2],
      unpacked_poly[15][2], unpacked_poly[16][2]};
  std::vector<double> poly_3 = {
      unpacked_poly[0][3],  unpacked_poly[1][3],  unpacked_poly[2][3],
      unpacked_poly[3][3],  unpacked_poly[4][3],  unpacked_poly[5][3],
      unpacked_poly[6][3],  unpacked_poly[7][3],  unpacked_poly[8][3],
      unpacked_poly[9][3],  unpacked_poly[10][3], unpacked_poly[11][3],
      unpacked_poly[12][3], unpacked_poly[13][3], unpacked_poly[14][3],
      unpacked_poly[15][3], unpacked_poly[16][3]};

  res_0.push_back(std::make_pair(Resultant::UnivariatePolynomial(poly_0),
                                 std::make_pair(0.0, 1.0)));
  res_1.push_back(std::make_pair(Resultant::UnivariatePolynomial(poly_1),
                                 std::make_pair(0.0, 1.0)));
  res_2.push_back(std::make_pair(Resultant::UnivariatePolynomial(poly_2),
                                 std::make_pair(0.0, 1.0)));
  res_3.push_back(std::make_pair(Resultant::UnivariatePolynomial(poly_3),
                                 std::make_pair(0.0, 1.0)));

  result.emplace_back(res_0);
  result.emplace_back(res_1);
  result.emplace_back(res_2);
  result.emplace_back(res_3);

  return result;
}

#endif
// clang-format off

// clang-format on

std::atomic<double> coeff_time{0.0};
std::atomic<double> bezout_time{0.0};
std::atomic<double> det_time{0.0};
std::atomic<double> solve_time{0.0};
std::atomic<long long> simd_count{0};

void update_coeff(double d) {
  double current;
  do {
    current = coeff_time.load();

  } while (!coeff_time.compare_exchange_weak(current, current + d));
}

void update_bezout(double d) {
  double current;

  do {
    current = bezout_time.load();
  } while (!bezout_time.compare_exchange_weak(current, current + d));
}

void update_det(double d) {
  double current;

  do {
    current = det_time.load();
  } while (!det_time.compare_exchange_weak(current, current + d));
}

void update_count(int i) {
  long long current;
  do {
    current = simd_count.load();
  } while (!simd_count.compare_exchange_weak(current, current + i));
}

void update_solve(double d) {
  double current;

  do {
    current = solve_time.load();
  } while (!solve_time.compare_exchange_weak(current, current + d));
}

#define ENABLE_TIME_COUNT

std::vector<std::vector<std::tuple<double, double, double, double>>>
solve(int chain_type,                 // 1R, 2T
      const std::vector<double> &pD_, /** Camera xyz */
      const std::vector<double> &pL_, /** Light xyz */
      const std::vector<double> &pX_, const std::vector<double> &pY_,
      const std::vector<double> &pZ_, const std::vector<double> &nX_,
      const std::vector<double> &nY_, const std::vector<double> &nZ_,
      bool use_fft, int cutoff_matrix, int cutoff_resultant,
      float cutoff_eps_resultant, int methodMask) {

  assert(chain_type == 1 || chain_type == 2);

#ifdef ENABLE_TIME_COUNT
  update_count(4);
#endif
  //
  global_poly_cutoff_eps = cutoff_eps_resultant;
  global_poly_cutoff = cutoff_resultant;
  global_method_mask = methodMask;

  auto ___coeff_begin = std::chrono::high_resolution_clock::now();

  //* Directly initialize x1 and n1hat

  /**
   * | p10.x, p11.x | | p10.y, p11.y | | p10.z, p11.z |
   * | p12.x,       | | p12.y,       | | p12.z,       |
   */

  BVP3<2ul> x1(pX_.data(), pY_.data(), pZ_.data());

  /**
   * | n10.x, n11.x | | n10.y, n11.y | | n10.z, n11.z |
   * | n12.x,       | | n12.y,       | | n12.z,       |
   */

  BVP3<2ul> n1hat(nX_.data(), nY_.data(), nZ_.data());

  BVP3<1ul> xD((BVP<1ul>(pD_[0])), BVP<1ul>(pD_[1]), BVP<1ul>(pD_[2]));
  BVP3<1ul> xL((BVP<1ul>(pL_[0])), BVP<1ul>(pL_[1]), BVP<1ul>(pL_[2]));

  BVP3<1ul> p11(&pX_[8], &pY_[8], &pZ_[8]);
  BVP3<1ul> p12(&pX_[4], &pY_[4], &pZ_[4]);

  double valid_v_min = 0;
  double valid_v_max = 1;

  Resultant::BivariatePolynomial u2hat;
  Resultant::BivariatePolynomial v2hat;
  Resultant::BivariatePolynomial kappa2;

  if (chain_type == 1) {
    BVP3 d0 = x1 - xD;
    BVP3 d1 = xL - x1;

    BVP d0_dot_n1hat = d0.dot(n1hat);
    BVP d1_dot_n1hat = d1.dot(n1hat);

    BVP3 t1hat1 = n1hat.cross(p11);
    BVP3 t1hat2 = n1hat.cross(p12);

    BVP d0_dot_t1hat2 = d0.dot(t1hat2);
    BVP d1_dot_t1hat2 = d1.dot(t1hat2);

    BVP Czy = d0_dot_n1hat * d1_dot_t1hat2 + d0_dot_t1hat2 * d1_dot_n1hat;

    BVP3 s = xL - xD;
    BVP3 cop = (d0.cross(s)).cross(n1hat.cross(s));
    BVP Cxz = cop.x;

    // BVP to poly matrix

    auto ___coeff_end = std::chrono::high_resolution_clock::now();

#ifdef ENABLE_TIME_COUNT
    update_coeff(std::chrono::duration_cast<std::chrono::nanoseconds>(
                     ___coeff_end - ___coeff_begin)
                     .count() *
                 1e-3);
#endif

    auto ___bezout_begin = std::chrono::high_resolution_clock::now();
    UnivariatePolyMatrix<4ul> bezout = bezout_matrix<5ul, 3ul, 4ul>(Czy, Cxz);
    auto ___bezout_end = std::chrono::high_resolution_clock::now();

#ifdef ENABLE_TIME_COUNT
    update_bezout(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      ___bezout_end - ___bezout_begin)
                      .count() *
                  1e-3f);
#endif

    auto ___det_begin = std::chrono::high_resolution_clock::now();
    auto dets = bezout.determinant();
    auto ___det_end = std::chrono::high_resolution_clock::now();

#ifdef ENABLE_TIME_COUNT
    update_det(std::chrono::duration_cast<std::chrono::nanoseconds>(
                   ___det_end - ___det_begin)
                   .count() *
               1e-3f);
#endif

    std::vector<std::vector<std::tuple<double, double, double, double>>> res(4);
    Resultant::BivariatePolynomial Cxz_4[4];

    depack_bvp(Cxz, Cxz_4);

    //! SIMD Optimization only support m=0
    res[0] = Resultant::solve_equ(
        dets[0], Resultant::UnivariatePolynomialMatrix(), Cxz_4[0], u2hat,
        v2hat, kappa2, 1, valid_v_min, valid_v_max);
    res[1] = Resultant::solve_equ(
        dets[1], Resultant::UnivariatePolynomialMatrix(), Cxz_4[1], u2hat,
        v2hat, kappa2, 1, valid_v_min, valid_v_max);
    res[2] = Resultant::solve_equ(
        dets[2], Resultant::UnivariatePolynomialMatrix(), Cxz_4[2], u2hat,
        v2hat, kappa2, 1, valid_v_min, valid_v_max);
    res[3] = Resultant::solve_equ(
        dets[3], Resultant::UnivariatePolynomialMatrix(), Cxz_4[3], u2hat,
        v2hat, kappa2, 1, valid_v_min, valid_v_max);

    return res;
  } else if (chain_type == 2) {
    // refraction
    constexpr double eta = 1.5;

    BVP3 d0 = x1 - xD;
    BVP3 d1 = xL - x1;

    BVP d0_norm2 = d0.dot(d0);
    BVP d1_norm2 = d1.dot(d1);

    BVP3 c0 = d0.cross(n1hat);
    BVP3 c1 = d1.cross(n1hat);

    BVP3 c = ((c0 * c0 * d1_norm2) * (eta * eta)) - c1 * c1 * d0_norm2;
    BVP Czy = c.x;

    BVP3 s = xL - xD;
    BVP3 cop = (d0.cross(s)).cross(n1hat.cross(s));
    BVP Cxz = cop.x + cop.y + cop.z;

    auto ___coeff_end = std::chrono::high_resolution_clock::now();

#ifdef ENABLE_TIME_COUNT
    update_coeff(std::chrono::duration_cast<std::chrono::nanoseconds>(
                     ___coeff_end - ___coeff_begin)
                     .count() *
                 1e-3);
#endif

    auto ___bezout_begin = std::chrono::high_resolution_clock::now();
    UnivariatePolyMatrix<6ul> bezout = bezout_matrix<7ul, 3ul, 6ul>(Czy, Cxz);
    auto ___bezout_end = std::chrono::high_resolution_clock::now();

#ifdef ENABLE_TIME_COUNT
    update_bezout(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      ___bezout_end - ___bezout_begin)
                      .count() *
                  1e-3f);
#endif

    auto ___det_begin = std::chrono::high_resolution_clock::now();
    auto dets = bezout.determinant();
    auto ___det_end = std::chrono::high_resolution_clock::now();
#ifdef ENABLE_TIME_COUNT
    update_det(std::chrono::duration_cast<std::chrono::nanoseconds>(
                   ___det_end - ___det_begin)
                   .count() *
               1e-3f);
#endif

    std::vector<std::vector<std::tuple<double, double, double, double>>> res(4);
    Resultant::BivariatePolynomial Cxz_4[4];

    depack_bvp(Cxz, Cxz_4);

    auto solve_time_begin = std::chrono::high_resolution_clock::now();
    //  //! SIMD Optimization only support m=0
    res[0] = Resultant::solve_equ(
        dets[0], Resultant::UnivariatePolynomialMatrix(), Cxz_4[0], u2hat,
        v2hat, kappa2, 1, valid_v_min, valid_v_max);
    res[1] = Resultant::solve_equ(
        dets[1], Resultant::UnivariatePolynomialMatrix(), Cxz_4[1], u2hat,
        v2hat, kappa2, 1, valid_v_min, valid_v_max);
    res[2] = Resultant::solve_equ(
        dets[2], Resultant::UnivariatePolynomialMatrix(), Cxz_4[2], u2hat,
        v2hat, kappa2, 1, valid_v_min, valid_v_max);
    res[3] = Resultant::solve_equ(
        dets[3], Resultant::UnivariatePolynomialMatrix(), Cxz_4[3], u2hat,
        v2hat, kappa2, 1, valid_v_min, valid_v_max);
    auto solve_time_end = std::chrono::high_resolution_clock::now();

#ifdef ENABLE_TIME_COUNT
    update_solve(std::chrono::duration_cast<std::chrono::nanoseconds>(
                     solve_time_end - solve_time_begin)
                     .count() *
                 1e-3f);
#endif
    return res;

  } else {
    printf("ERROR : SIMD acceleraction only support one bounce specular\n");
    exit(1);
  }
}

class LogHelper {
public:
  LogHelper() = default;

  ~LogHelper() {
#ifdef RESULTANT_SIMD
    std::cout << "Average : \n";

    long long count = simd_count.load();

    double t;

    t = coeff_time.load();
    std::cout << "SIMD Coeff time : " << t / count << "us" << std::endl;

    t = bezout_time.load();
    std::cout << "SIMD Bezout time : " << t / count << "us" << std::endl;

    t = det_time.load();
    std::cout << "SIMD det time : " << t / count << "us" << std::endl;

    t = solve_time.load();
    std::cout << "SIMD solve time : " << t / count << "us" << std::endl;

#endif
  }
} helper;

} // namespace resultant_simd