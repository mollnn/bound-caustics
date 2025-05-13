#pragma once
#include "alias.h"

void initializeFactorial()
{
    factorial[0] = 1.0;
    for (int i = 1; i < N_MAT; ++i)
    {
        factorial[i] = factorial[i - 1] * i;
    }
}

void initializeBinomial()
{
    for (int n = 0; n < N_MAT; ++n)
    {
        for (int i = 0; i <= n; ++i)
        {
            binomial[n][i] = factorial[n] / (factorial[i] * factorial[n - i]);
        }
    }
}

void initializeCache()
{
    c2b_cache[3] = (PolyMatrix *)malloc(N_MAT * N_MAT * sizeof(double));
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            (*(c2b_cache[3]))[i][j] = binomial[i][j] / binomial[3 - 1][j];
        }
    }

    c2b_cache[4] = (PolyMatrix *)malloc(N_MAT * N_MAT * sizeof(double));
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            (*(c2b_cache[4]))[i][j] = binomial[i][j] / binomial[4 - 1][j];
        }
    }

    c2b_cache[5] = (PolyMatrix *)malloc(N_MAT * N_MAT * sizeof(double));
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            (*(c2b_cache[5]))[i][j] = binomial[i][j] / binomial[5 - 1][j];
        }
    }

    c2b_cache[6] = (PolyMatrix *)malloc(N_MAT * N_MAT * sizeof(double));
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            (*(c2b_cache[6]))[i][j] = binomial[i][j] / binomial[6 - 1][j];
        }
    }

    c2b_cache[7] = (PolyMatrix *)malloc(N_MAT * N_MAT * sizeof(double));
    for (int i = 0; i < 7; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            (*(c2b_cache[7]))[i][j] = binomial[i][j] / binomial[7 - 1][j];
        }
    }

    c2b_cache[13] = (PolyMatrix *)malloc(N_MAT * N_MAT * sizeof(double));
    for (int i = 0; i < 13; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            (*(c2b_cache[13]))[i][j] = binomial[i][j] / binomial[13 - 1][j];
        }
    }

    c2b_cache[17] = (PolyMatrix *)malloc(N_MAT * N_MAT * sizeof(double));
    for (int i = 0; i < 17; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            (*(c2b_cache[17]))[i][j] = binomial[i][j] / binomial[17 - 1][j];
        }
    }

    c2b_cache[24] = (PolyMatrix *)malloc(N_MAT * N_MAT * sizeof(double));
    for (size_t i = 0; i < 24; ++i)
    {
        for (size_t j = 0; j <= i; ++j)
        {
            (*(c2b_cache[24]))[i][j] = binomial[i][j] / binomial[24 - 1][j];
        }
    }

    c2b_cache[25] = (PolyMatrix *)malloc(N_MAT * N_MAT * sizeof(double));
    for (int i = 0; i < 25; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            (*(c2b_cache[25]))[i][j] = binomial[i][j] / binomial[25 - 1][j];
        }
    }
}

void initializeMesh(const std::string &filename, const BounderVec3 &offset = BounderVec3(0.0, 0.0, 0.0), double scale = 1.0)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<BounderVec3> vertices;
    std::vector<BounderVec3> normals;
    std::string line;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string type;
        ss >> type;

        if (type == "v")
        {
            double x, y, z;
            ss >> x >> y >> z;
            vertices.emplace_back(x, y, z);
        }
        else if (type == "vn")
        {
            double x, y, z;
            ss >> x >> y >> z;
            normals.emplace_back(x, y, z);
        }
        else if (type == "f")
        {
            int vertex_indices[3] = {};
            int normal_indices[3] = {};
            for (int i = 0; i < 3; ++i)
            {
                std::string vertex_data;
                ss >> vertex_data;
                size_t first_slash = vertex_data.find('/');
                size_t second_slash = vertex_data.find('/', first_slash + 1);

                int vertex_index = std::stoi(vertex_data.substr(0, first_slash));
                int normal_index = std::stoi(vertex_data.substr(second_slash + 1));

                vertex_indices[i] = vertex_index - 1;
                normal_indices[i] = normal_index - 1;
            }

            BounderTriangle face;
            for (int i = 0; i < 3; ++i)
            {
                face.vertices[i] = vertices[vertex_indices[i]] * scale + offset;
                face.normals[i] = normals[normal_indices[i]];
            }

            mesh.push_back(face);
        }
    }

    file.close();
}

constexpr size_t BVP_ADD_SIZE(size_t bvp_1_size, size_t bvp_2_size)
{
    return std::max(bvp_1_size, bvp_2_size);
}

constexpr size_t BVP_MUL_SIZE(size_t bvp_1_size, size_t bvp_2_size)
{
    return bvp_1_size + bvp_2_size - 1;
}

template <size_t Size>
inline void transpose(const PolyMatrix &mat, PolyMatrix &result)
{
    // printf("No general implementation -- transpose<%ld>()\n", Size);
    for (size_t i = 0; i < Size; ++i)
    {
        for (size_t j = 0; j < Size; ++j)
        {
            result[j][i] = mat[i][j];
        }
    }
}

template <size_t Size>
inline void C2B2d(const PolyMatrix &a, PolyMatrix &result)
{
    PolyMatrix *iU = c2b_cache[Size];

    PolyMatrix tmp;
    for (size_t i = 0; i < Size; ++i)
    {
        for (size_t j = 0; j < Size; ++j)
        {
            tmp[i][j] = 0;
            for (size_t k = 0; i - k >= 0 && k + j < Size; ++k)
            {
#ifdef COUNTER
                global_counter += 2;
#endif
                tmp[i][j] += (*iU)[i][k] * a[k][j];
            }
        }
    }

    PolyMatrix tmp_t;
    transpose<Size>(tmp, tmp_t);

    for (size_t i = 0; i < Size; ++i)
    {
        for (size_t j = 0; j < Size; ++j)
        {
            tmp[i][j] = 0;
            for (size_t k = 0; k < Size && i - k >= 0; ++k)
            {
#ifdef COUNTER
                global_counter += 2;
#endif
                tmp[i][j] += (*iU)[i][k] * tmp_t[k][j];
            }
        }
    }

    transpose<Size>(tmp, result);
}

inline std::tuple<double, double> intersect3d(const BounderVec3 &o, const BounderVec3 &d, const BounderVec3 &p0, const BounderVec3 &p1, const BounderVec3 &p2, bool rectangle = false, bool ignore_check = false)
{
#ifdef COUNTER
    global_counter += 20;
#endif
    double u = (d.cross(p2)).dot(o - p0);
    double v = ((o - p0).cross(p1)).dot(d);
    double k = (d.cross(p2)).dot(p1);
    u /= k;
    v /= k;
    BounderVec3 x = p0 + u * p1 + v * p2;
    BounderVec3 m = p1.cross(p2);
    double t = m.dot(d) * m.dot(x - o);

    if ((u < 0 || v < 0 || u > 1 || v > 1 || (u + v > 1 && !rectangle)) && !ignore_check)
    {
        return {-1.0, -1.0};
    }
    return {u, v};
}

inline std::tuple<double, double> point_triangle_dist_range(BounderVec3 &pL, BounderVec3 &p10, BounderVec3 &p11, BounderVec3 &p12, const BounderDomain &domain)
{
#ifdef COUNTER
    global_counter += 25;
#endif
    double rm = 1e19, rM = 0;
    double u1m = domain.um, u1M = domain.uM, v1m = domain.vm, v1M = domain.vM;
    BounderVec3 tmp_p = p10 + u1m * p11 + v1m * p12;
    double r = (tmp_p - pL).norm();
    rm = std::min(rm, r);
    rM = std::max(rM, r);
    tmp_p = p10 + u1m * p11 + v1M * p12;
    r = (tmp_p - pL).norm();
    rm = std::min(rm, r);
    rM = std::max(rM, r);
    tmp_p = p10 + u1M * p11 + v1m * p12;
    r = (tmp_p - pL).norm();
    rm = std::min(rm, r);
    rM = std::max(rM, r);
    tmp_p = p10 + u1M * p11 + v1M * p12;
    r = (tmp_p - pL).norm();
    rm = std::min(rm, r);
    rM = std::max(rM, r);

    auto [u, v] = intersect3d(pL, p11.cross(p12), p10, p11, p12);
    if (u != -1.0)
    {
        u = std::max(u1m, std::min(u1M, u));
        v = std::max(v1m, std::min(v1M, v));
        tmp_p = p10 + u * p11 + v * p12;
        r = (tmp_p - pL).norm();
        rm = std::min(rm, r);
        rM = std::max(rM, r);
    }
    return {rm, rM};
}

std::tuple<double, double, double, double> compute_easy_approx_for_sqrt(double l, double r)
{
#ifdef COUNTER
    global_counter += 20;
#endif
    double a = (1 / std::sqrt(l) + 1 / std::sqrt(r)) / 4;
    double delta_xi = (1 / std::sqrt(l) - 1 / std::sqrt(r)) / 4;
    double mid = l;
    double b = std::sqrt(mid) - a * mid;
    double delta_xi1 = std::sqrt(r) - std::sqrt(l) - (r - l) / std::sqrt(r) / 2;
    return std::make_tuple(a, b, delta_xi, delta_xi1);
}

std::tuple<double, double, double> compute_line_approx_for_sqrt(double l, double r)
{
#ifdef COUNTER
    global_counter += 20;
#endif
    double a = std::sqrt(r) - std::sqrt(l);
    a /= (r - l);
    double b = std::sqrt(l) - a * l;
    double xm = 1.0 / (4.0 * a * a);
    double delta_xi = 0.5 * (std::sqrt(xm) - a * xm - b);
    b += delta_xi;
    return std::make_tuple(a, b, delta_xi);
}

inline double bern_trans_coef(size_t N, size_t I, size_t J)
{
    return binomial[I][J] / binomial[N][J];
}

template <size_t y_Size, size_t uv_Size>
inline void C2B3d(const PolyMatrix_3D &a, PolyMatrix_3D &result)
{
    PolyMatrix_3D t;
    for (size_t i = 0; i < y_Size; i++)
    {
        C2B2d<uv_Size>(a[i], t[i]);
    }

    for (size_t i = 0; i < y_Size; i++)
    {
        for (size_t k1 = 0; k1 < uv_Size; k1++)
        {
            for (size_t k2 = 0; k2 < uv_Size; k2++)
            {
#ifdef COUNTER
                global_counter += 2;
#endif
                result[i][k1][k2] = bern_trans_coef(y_Size - 1, i, 0) * t[0][k1][k2];
            }
        }

        for (size_t j = 1; j < i + 1; j++)
        {
            for (size_t k1 = 0; k1 < uv_Size; k1++)
            {
                for (size_t k2 = 0; k2 < uv_Size; k2++)
                {
#ifdef COUNTER
                    global_counter += 2;
#endif
                    result[i][k1][k2] += bern_trans_coef(y_Size - 1, i, j) * t[j][k1][k2];
                }
            }
        }
    }
}

#ifdef DEBUG

std::ofstream ofs("./matrix_size.txt", std::ios::out | std::ios::trunc);

template <size_t x_Size, size_t y_Size, size_t uv_Size>
inline void print4DMartix(PolyMatrix_4D &a)
{
    ofs << x_Size << " " << y_Size << " " << uv_Size << std::endl;
    for (size_t i = 0; i < x_Size; i++)
    {
        for (size_t j = 0; j < y_Size; j++)
        {
            ofs << "i j: " << i << " " << j << std::endl;
            for (size_t k = 0; k < uv_Size; k++)
            {
                for (size_t l = 0; l < uv_Size; l++)
                {
                    ofs << std::setw(12) << std::setprecision(3) << a[i][j][k][l] << " ";
                }
                ofs << std::endl;
            }
        }
    }
}

#endif

template <size_t x_Size, size_t y_Size, size_t uv_Size>
inline void C2B4d(const PolyMatrix_4D &a, PolyMatrix_4D &result)
{
    PolyMatrix_4D t;
    for (size_t i = 0; i < x_Size; i++)
    {
        C2B3d<y_Size, uv_Size>(a[i], t[i]);
    }

#ifdef DEBUG

    // ofs << "bound" << std::endl;
    // print4DMartix<x_Size, y_Size, uv_Size>(t);

#endif

    for (size_t i = 0; i < x_Size; i++)
    {
        for (size_t k1 = 0; k1 < y_Size; k1++)
        {
            for (size_t k2 = 0; k2 < uv_Size; k2++)
            {
                for (size_t k3 = 0; k3 < uv_Size; k3++)
                {
#ifdef COUNTER
                    global_counter += 2;
#endif
                    result[i][k1][k2][k3] = bern_trans_coef(x_Size - 1, i, 0) * t[0][k1][k2][k3];
                }
            }
        }
        for (size_t j = 1; j < i + 1; j++)
        {
            for (size_t k1 = 0; k1 < y_Size; k1++)
            {
                for (size_t k2 = 0; k2 < uv_Size; k2++)
                {
                    for (size_t k3 = 0; k3 < uv_Size; k3++)
                    {
#ifdef COUNTER
                        global_counter += 2;
#endif
                        result[i][k1][k2][k3] += bern_trans_coef(x_Size - 1, i, j) * t[j][k1][k2][k3];
                    }
                }
            }
        }
    }
}

template <>
inline void C2B3d<4, 25>(const PolyMatrix_3D &a, PolyMatrix_3D &result)
{
    PolyMatrix_3D t;
    for (size_t i = 0; i < 4; i++)
    {
        C2B2d<25>(a[i], t[i]); // 17
    }

    for (size_t i = 0; i < 4; i++)
    {
        for (size_t k1 = 0; k1 < 25; k1++)
        {
            for (size_t k2 = 0; k2 < 25; k2++)
            {
#ifdef COUNTER
                global_counter += 2;
#endif
                result[i][k1][k2] = bern_trans_coef(5 - 1, i, 0) * t[0][k1][k2];
            }
        }

        for (size_t j = 1; j < i + 1; j++)
        {
            for (size_t k1 = 0; k1 < 25; k1++)
            {
                for (size_t k2 = 0; k2 < 25; k2++)
                {
#ifdef COUNTER
                    global_counter += 2;
#endif
                    result[i][k1][k2] += bern_trans_coef(5 - 1, i, j) * t[j][k1][k2];
                }
            }
        }
    }

    for (size_t k1 = 0; k1 < 25; k1++)
    {
        for (size_t k2 = 0; k2 < 25; k2++)
        {
#ifdef COUNTER
            global_counter += 2;
#endif
            result[4][k1][k2] = bern_trans_coef(5 - 1, 4, 0) * t[0][k1][k2];
        }
    }

    for (size_t j = 1; j < 4; j++)
    {
        for (size_t k1 = 0; k1 < 25; k1++)
        {
            for (size_t k2 = 0; k2 < 25; k2++)
            {
#ifdef COUNTER
                global_counter += 2;
#endif
                result[4][k1][k2] += bern_trans_coef(5 - 1, 4, j) * t[j][k1][k2];
            }
        }
    }
}

template <>
inline void C2B3d<3, 25>(const PolyMatrix_3D &a, PolyMatrix_3D &result)
{
    PolyMatrix_3D t;
    for (size_t i = 0; i < 3; i++)
    {
        C2B2d<25>(a[i], t[i]); // 13
    }

    for (size_t i = 0; i < 3; i++)
    {
        for (size_t k1 = 0; k1 < 25; k1++)
        {
            for (size_t k2 = 0; k2 < 25; k2++)
            {
#ifdef COUNTER
                global_counter += 2;
#endif
                result[i][k1][k2] = bern_trans_coef(5 - 1, i, 0) * t[0][k1][k2];
            }
        }

        for (size_t j = 1; j < i + 1; j++)
        {
            for (size_t k1 = 0; k1 < 25; k1++)
            {
                for (size_t k2 = 0; k2 < 25; k2++)
                {
#ifdef COUNTER
                    global_counter += 2;
#endif
                    result[i][k1][k2] += bern_trans_coef(5 - 1, i, j) * t[j][k1][k2];
                }
            }
        }
    }

    for (int i = 3; i < 5; i++)
    {
        for (size_t k1 = 0; k1 < 25; k1++)
        {
            for (size_t k2 = 0; k2 < 25; k2++)
            {
#ifdef COUNTER
                global_counter += 2;
#endif
                result[i][k1][k2] = bern_trans_coef(5 - 1, i, 0) * t[0][k1][k2];
            }
        }

        for (size_t j = 1; j < 3; j++)
        {
            for (size_t k1 = 0; k1 < 25; k1++)
            {
                for (size_t k2 = 0; k2 < 25; k2++)
                {
#ifdef COUNTER
                    global_counter += 2;
#endif
                    result[i][k1][k2] += bern_trans_coef(5 - 1, i, j) * t[j][k1][k2];
                }
            }
        }
    }
}

template <>
inline void C2B3d<2, 25>(const PolyMatrix_3D &a, PolyMatrix_3D &result)
{
    PolyMatrix_3D t;
    for (size_t i = 0; i < 2; i++)
    {
        C2B2d<25>(a[i], t[i]);
    }

    for (size_t i = 0; i < 2; i++)
    {
        for (size_t k1 = 0; k1 < 25; k1++)
        {
            for (size_t k2 = 0; k2 < 25; k2++)
            {
#ifdef COUNTER
                global_counter += 2;
#endif
                result[i][k1][k2] = bern_trans_coef(5 - 1, i, 0) * t[0][k1][k2];
            }
        }

        for (size_t j = 1; j < i + 1; j++)
        {
            for (size_t k1 = 0; k1 < 25; k1++)
            {
                for (size_t k2 = 0; k2 < 25; k2++)
                {
#ifdef COUNTER
                    global_counter += 2;
#endif
                    result[i][k1][k2] += bern_trans_coef(5 - 1, i, j) * t[j][k1][k2];
                }
            }
        }
    }

    for (int i = 2; i < 5; i++)
    {
        for (size_t k1 = 0; k1 < 25; k1++)
        {
            for (size_t k2 = 0; k2 < 25; k2++)
            {
#ifdef COUNTER
                global_counter += 2;
#endif
                result[i][k1][k2] = bern_trans_coef(5 - 1, i, 0) * t[0][k1][k2];
            }
        }

        for (size_t j = 1; j < 2; j++)
        {
            for (size_t k1 = 0; k1 < 25; k1++)
            {
                for (size_t k2 = 0; k2 < 25; k2++)
                {
#ifdef COUNTER
                    global_counter += 2;
#endif
                    result[i][k1][k2] += bern_trans_coef(5 - 1, i, j) * t[j][k1][k2];
                }
            }
        }
    }
}

template <>
inline void C2B3d<1, 25>(const PolyMatrix_3D &a, PolyMatrix_3D &result)
{
    PolyMatrix t;
    C2B2d<25>(a[0], t); // 5

    for (int i = 0; i < 5; i++)
    {
        for (size_t k1 = 0; k1 < 25; k1++)
        {
            for (size_t k2 = 0; k2 < 25; k2++)
            {
#ifdef COUNTER
                global_counter += 2;
#endif
                result[i][k1][k2] = bern_trans_coef(5 - 1, i, 0) * t[k1][k2];
            }
        }
    }
}

template <>
inline void C2B4d<5, 5, 25>(const PolyMatrix_4D &a, PolyMatrix_4D &result)
{
    PolyMatrix_4D t;
    C2B3d<5, 25>(a[0], t[0]);
    C2B3d<4, 25>(a[1], t[1]);
    C2B3d<3, 25>(a[2], t[2]);
    C2B3d<2, 25>(a[3], t[3]);
    C2B3d<1, 25>(a[4], t[4]);

    for (size_t i = 0; i < 5; i++)
    {
        for (size_t k1 = 0; k1 < 5; k1++)
        {
            for (size_t k2 = 0; k2 < 25; k2++)
            {
                for (size_t k3 = 0; k3 < 25; k3++)
                {
#ifdef COUNTER
                    global_counter += 2;
#endif
                    result[i][k1][k2][k3] = bern_trans_coef(5 - 1, i, 0) * t[0][k1][k2][k3];
                }
            }
        }
        for (size_t j = 1; j < std::min((size_t)5, i + 1); j++)
        {
            for (size_t k1 = 0; k1 < 5; k1++)
            {
                for (size_t k2 = 0; k2 < 25; k2++)
                {
                    for (size_t k3 = 0; k3 < 25; k3++)
                    {
#ifdef COUNTER
                        global_counter += 2;
#endif
                        result[i][k1][k2][k3] += bern_trans_coef(5 - 1, i, j) * t[j][k1][k2][k3];
                    }
                }
            }
        }
    }
}