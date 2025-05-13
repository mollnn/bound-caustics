#pragma once
#include "utils.h"

template <size_t x_Size, size_t y_Size, size_t uv_Size>
struct BVP_4D
{
public:
    BVP_4D()
    {
        for (size_t i = 0; i < x_Size; i++)
        {
            for (size_t j = 0; j < y_Size; j++)
            {
                for (size_t k = 0; k < uv_Size; k++)
                {
                    for (size_t l = 0; l < uv_Size; l++)
                    {
                        coeffs[i][j][k][l] = 0;
                    }
                }
            }
        }
    }

    explicit BVP_4D(double d) { coeffs[0][0][0][0] = d; }

    BVP_4D(double *c)
    {
        printf("No general implementation -- BVP_4D<>\n");
    }

    BVP_4D(BVP_4D<x_Size, y_Size, uv_Size> &&other)
    {
        // printf("BVP_4D(BVP_4D<x_Size, y_Size, uv_Size> &&other)\n");
        auto start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < x_Size; i++)
        {
            for (size_t j = 0; j < y_Size; j++)
            {
                for (size_t k = 0; k < uv_Size; k++)
                {
                    for (size_t l = 0; l < uv_Size; l++)
                    {
                        coeffs[i][j][k][l] = std::move(other.coeffs[i][j][k][l]);
                    }
                }
            }
        }
    }

    BVP_4D &operator=(BVP_4D<x_Size, y_Size, uv_Size> &&other)
    {
        // printf("BVP_4D &operator=(BVP_4D<x_Size, y_Size, uv_Size> &&other)\n");
        if (this != &other)
        {

            for (size_t i = 0; i < x_Size; i++)
            {
                for (size_t j = 0; j < y_Size; j++)
                {
                    for (size_t k = 0; k < uv_Size; k++)
                    {
                        for (size_t l = 0; l < uv_Size; l++)
                        {
                            coeffs[i][j][k][l] = std::move(other.coeffs[i][j][k][l]);
                        }
                    }
                }
            }
        }
        return *this;
    }

    BVP_4D(const BVP_4D<x_Size, y_Size, uv_Size> &other)
    {
        // printf("BVP_4D(const BVP_4D<x_Size, y_Size, uv_Size> &other)\n");
        for (size_t i = 0; i < x_Size; i++)
        {
            for (size_t j = 0; j < y_Size; j++)
            {
                for (size_t k = 0; k < uv_Size; k++)
                {
                    for (size_t l = 0; l < uv_Size; l++)
                    {
                        coeffs[i][j][k][l] = other.coeffs[i][j][k][l];
                    }
                }
            }
        }
    }

    BVP_4D &operator=(const BVP_4D<x_Size, y_Size, uv_Size> &other)
    {
        // printf("BVP_4D &operator=(const BVP_4D<x_Size, y_Size, uv_Size> &other)\n");
        if (this != &other)
        {

            for (size_t i = 0; i < x_Size; i++)
            {
                for (size_t j = 0; j < y_Size; j++)
                {
                    for (size_t k = 0; k < uv_Size; k++)
                    {
                        for (size_t l = 0; l < uv_Size; l++)
                        {
                            coeffs[i][j][k][l] = other.coeffs[i][j][k][l];
                        }
                    }
                }
            }
        }
        return *this;
    }

    template <size_t Other_x_Size, size_t Other_y_Size, size_t Other_uv_Size, size_t Res_x_Size = BVP_ADD_SIZE(x_Size, Other_x_Size), size_t Res_y_Size = BVP_ADD_SIZE(y_Size, Other_y_Size), size_t Res_uv_Size = BVP_ADD_SIZE(uv_Size, Other_uv_Size)>
    BVP_4D<Res_x_Size, Res_y_Size, Res_uv_Size> operator+(const BVP_4D<Other_x_Size, Other_y_Size, Other_uv_Size> &other) const
    {
        BVP_4D<Res_x_Size, Res_y_Size, Res_uv_Size> res;
        for (int i = 0; i < Res_x_Size; i++)
        {
            for (int j = 0; j < Res_y_Size; j++)
            {
                for (int k = 0; k < Res_uv_Size; k++)
                {
                    for (int l = 0; l + k < Res_uv_Size; l++)
                    {
#ifdef COUNTER
                        global_counter++;
#endif
                        res.coeffs[i][j][k][l] = coeffs[i][j][k][l] + other.coeffs[i][j][k][l];
                    }
                }
            }
        }

#ifdef DEBUG
        // if (x_Size != Other_x_Size || y_Size != Other_y_Size || uv_Size != Other_uv_Size)
        // {
        //     ofs << "+++++op1" << std::endl;
        //     print4DMartix<x_Size, y_Size, uv_Size>(coeffs);
        //     ofs << "+++++op2" << std::endl;
        //     print4DMartix<Other_x_Size, Other_y_Size, Other_uv_Size>(other.coeffs);
        //     ofs << "+++++res" << std::endl;
        //     print4DMartix<Res_x_Size, Res_y_Size, Res_uv_Size>(res.coeffs);
        // }
#endif

        return res;
    }

    template <size_t Other_x_Size, size_t Other_y_Size, size_t Other_uv_Size, size_t Res_x_Size = BVP_ADD_SIZE(x_Size, Other_x_Size), size_t Res_y_Size = BVP_ADD_SIZE(y_Size, Other_y_Size), size_t Res_uv_Size = BVP_ADD_SIZE(uv_Size, Other_uv_Size)>
    BVP_4D<Res_x_Size, Res_y_Size, Res_uv_Size> operator-(const BVP_4D<Other_x_Size, Other_y_Size, Other_uv_Size> &other) const
    {
        BVP_4D<Res_x_Size, Res_y_Size, Res_uv_Size> res;
        for (int i = 0; i < Res_x_Size; i++)
        {
            for (int j = 0; j < Res_y_Size; j++)
            {
                for (int k = 0; k < Res_uv_Size; k++)
                {
                    for (int l = 0; l + k < Res_uv_Size; l++)
                    {
#ifdef COUNTER
                        global_counter++;
#endif
                        res.coeffs[i][j][k][l] = coeffs[i][j][k][l] - other.coeffs[i][j][k][l];
                    }
                }
            }
        }

#ifdef DEBUG
        // if (x_Size == 5 && y_Size == 5 && uv_Size == 23 && Other_x_Size == 5 && Other_y_Size == 5 && Other_uv_Size == 23)
        // {
        // ofs << "------op1" << std::endl;
        // print4DMartix<x_Size, y_Size, uv_Size>(coeffs);
        // ofs << "------op2" << std::endl;
        // print4DMartix<Other_x_Size, Other_y_Size, Other_uv_Size>(other.coeffs);
        // ofs << "------res" << std::endl;
        // print4DMartix<Res_x_Size, Res_y_Size, Res_uv_Size>(res.coeffs);
        // }
#endif

        return res;
    }

    template <size_t Other_x_Size, size_t Other_y_Size, size_t Other_uv_Size, size_t Res_x_Size = BVP_MUL_SIZE(x_Size, Other_x_Size), size_t Res_y_Size = BVP_MUL_SIZE(y_Size, Other_y_Size), size_t Res_uv_Size = BVP_MUL_SIZE(uv_Size, Other_uv_Size)>
    BVP_4D<Res_x_Size, Res_y_Size, Res_uv_Size> operator*(const BVP_4D<Other_x_Size, Other_y_Size, Other_uv_Size> &other) const
    {
        BVP_4D<Res_x_Size, Res_y_Size, Res_uv_Size> res;

        for (int i1 = 0; i1 < x_Size; i1++)
        {
            for (int i2 = 0; i2 < Other_x_Size; i2++)
            {
                for (int j1 = 0; j1 < y_Size; j1++)
                {
                    for (int j2 = 0; j2 < Other_y_Size; j2++)
                    {
                        for (int k1 = 0; k1 < uv_Size; k1++)
                        {
                            for (int k2 = 0; k2 < Other_uv_Size; k2++)
                            {
                                for (int l1 = 0; l1 + k1 < uv_Size; l1++)
                                {
                                    for (int l2 = 0; l2 + k2 < Other_uv_Size; l2++)
                                    {
#ifdef COUNTER
                                        global_counter += 2;
#endif
                                        res.coeffs[i1 + i2][j1 + j2][k1 + k2][l1 + l2] += coeffs[i1][j1][k1][l1] * other.coeffs[i2][j2][k2][l2];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

#ifdef DEBUG

        if (x_Size == 3 && y_Size == 3 && uv_Size == 11 && Other_x_Size == 3 && Other_y_Size == 3 && Other_uv_Size == 11)
        {
            // ofs << "*****op1" << std::endl;
            print4DMartix<x_Size, y_Size, uv_Size>(coeffs);
            // ofs << "*****op2" << std::endl;
            // print4DMartix<Other_x_Size, Other_y_Size, Other_uv_Size>(other.coeffs);
        }
#endif

        return res;
    }

    BVP_4D<x_Size, y_Size, uv_Size> scalarMul(const double other) const
    {
        BVP_4D<x_Size, y_Size, uv_Size> res;
        for (int i = 0; i < x_Size; i++)
        {
            for (int j = 0; j < y_Size; j++)
            {
                for (int k = 0; k < uv_Size; k++)
                {
                    for (int l = 0; l + k < uv_Size; l++)
                    {
#ifdef COUNTER
                        global_counter++;
#endif
                        res.coeffs[i][j][k][l] = coeffs[i][j][k][l] * other;
                    }
                }
            }
        }

#ifdef DEBUG

        // ofs << "****double_res" << std::endl;
        // print4DMartix<x_Size, y_Size, uv_Size>(res.coeffs);

#endif

        return res;
    }

    BoundingVal bound() const
    {
        PolyMatrix_4D b_coeffs;
        C2B4d<x_Size, y_Size, uv_Size>(coeffs, b_coeffs);
        double maxVal = INT_MIN;
        double minVal = INT_MAX;
        for (int i = 0; i < x_Size; i++)
        {
            for (int j = 0; j < y_Size; j++)
            {
                for (int k = 0; k < uv_Size; k++)
                {
                    for (int l = 0; l < uv_Size; l++)
                    {
                        if (b_coeffs[i][j][k][l] > maxVal)
                        {
                            maxVal = b_coeffs[i][j][k][l];
                        }
                        if (b_coeffs[i][j][k][l] < minVal)
                        {
                            minVal = b_coeffs[i][j][k][l];
                        }
                    }
                }
            }
        }
        return BoundingVal(minVal, maxVal);
    }

    template <size_t Other_x_Size, size_t Other_y_Size, size_t Other_uv_Size, size_t Res_x_Size = BVP_ADD_SIZE(x_Size, Other_x_Size), size_t Res_y_Size = BVP_ADD_SIZE(y_Size, Other_y_Size), size_t Res_uv_Size = BVP_ADD_SIZE(uv_Size, Other_uv_Size)>
    BoundingVal fbound(const BVP_4D<Other_x_Size, Other_y_Size, Other_uv_Size> &other) const
    {
        PolyMatrix_4D b_coeffs;
        PolyMatrix_4D other_b_coeffs;
        C2B4d<Res_x_Size, Res_y_Size, Res_uv_Size>(coeffs, b_coeffs);
        C2B4d<Res_x_Size, Res_y_Size, Res_uv_Size>(other.coeffs, other_b_coeffs);
        double maxVal = INT_MIN;
        double minVal = INT_MAX;
        for (int i = 0; i < Res_x_Size; i++)
        {
            for (int j = 0; j < Res_y_Size; j++)
            {
                for (int k = 0; k < Res_uv_Size; k++)
                {
                    for (int l = 0; l < Res_uv_Size; l++)
                    {
#ifdef COUNTER
                        global_counter++;
#endif
                        double val = b_coeffs[i][j][k][l] / other_b_coeffs[i][j][k][l];
                        if (val > maxVal)
                        {
                            maxVal = val;
                        }
                        if (val < minVal)
                        {
                            minVal = val;
                        }
                    }
                }
            }
        }

#ifdef DEBUG
        // if (x_Size != Other_x_Size || y_Size != Other_y_Size || uv_Size != Other_uv_Size)
        // {
        //     ofs << "fbound_op1" << std::endl;
        //     print4DMartix<x_Size, y_Size, uv_Size>(coeffs);
        //     ofs << "fbounding1" << std::endl;
        //     print4DMartix<Res_x_Size, Res_y_Size, Res_uv_Size>(b_coeffs);
        //     ofs << "fbound_op2" << std::endl;
        //     print4DMartix<Other_x_Size, Other_y_Size, Other_uv_Size>(other.coeffs);
        //     ofs << "fbounding2" << std::endl;
        //     print4DMartix<Res_x_Size, Res_y_Size, Res_uv_Size>(other_b_coeffs);
        // }
#endif

        return BoundingVal(minVal, maxVal);
    }

    BVP_4D<x_Size, y_Size, uv_Size> du() const
    {
        BVP_4D<x_Size, y_Size, uv_Size> res;
        for (int i = 0; i < x_Size; i++)
        {
            for (int j = 0; j < y_Size; j++)
            {
                for (int k = 1; k < uv_Size; k++)
                {
                    for (int l = 0; l + k < uv_Size; l++)
                    {
#ifdef COUNTER
                        global_counter++;
#endif
                        res.coeffs[i][j][k - 1][l] = k * coeffs[i][j][k][l];
                    }
                }
            }
        }

#ifdef DEBUG

        // ofs << "du" << std::endl;
        // print4DMartix<x_Size, y_Size, uv_Size>(res.coeffs);

#endif

        return res;
    }

    BVP_4D<x_Size, y_Size, uv_Size> dv() const
    {
        BVP_4D<x_Size, y_Size, uv_Size> res;
        for (int i = 0; i < x_Size; i++)
        {
            for (int j = 0; j < y_Size; j++)
            {
                for (int k = 0; k < uv_Size; k++)
                {
                    for (int l = 1; l + k < uv_Size; l++)
                    {
#ifdef COUNTER
                        global_counter++;
#endif
                        res.coeffs[i][j][k][l - 1] = l * coeffs[i][j][k][l];
                    }
                }
            }
        }

#ifdef DEBUG

        // ofs << "dv" << std::endl;
        // print4DMartix<x_Size, y_Size, uv_Size>(res.coeffs);

#endif

        return res;
    }

public:
    mutable PolyMatrix_4D coeffs;
};

template <size_t x_Size, size_t y_Size, size_t uv_Size>
struct BVP3_4D
{
    BVP3_4D(BVP_4D<x_Size, y_Size, uv_Size> &&_x, BVP_4D<x_Size, y_Size, uv_Size> &&_y, BVP_4D<x_Size, y_Size, uv_Size> &&_z)
    {
        // printf("BVP3_4D(BVP_4D<x_Size, y_Size, uv_Size> &&_x, BVP_4D<x_Size, y_Size, uv_Size> &&_y, BVP_4D<x_Size, y_Size, uv_Size> &&_z)\n");
        coeffs[0] = std::move(_x);
        coeffs[1] = std::move(_y);
        coeffs[2] = std::move(_z);
    }

    BVP3_4D(const BVP_4D<x_Size, y_Size, uv_Size> &_x, const BVP_4D<x_Size, y_Size, uv_Size> &_y, const BVP_4D<x_Size, y_Size, uv_Size> &_z)
    {
        // printf("BVP3_4D(const BVP_4D<x_Size, y_Size, uv_Size> &_x, const BVP_4D<x_Size, y_Size, uv_Size> &_y, const BVP_4D<x_Size, y_Size, uv_Size> &_z)\n");
        coeffs[0] = _x;
        coeffs[1] = _y;
        coeffs[2] = _z;
    }

    BVP3_4D(BVP3_4D<x_Size, y_Size, uv_Size> &&other)
    {
        // printf("BVP3_4D(BVP3_4D<x_Size, y_Size, uv_Size> &&other)\n");
        coeffs[0] = std::move(other.coeffs[0]);
        coeffs[1] = std::move(other.coeffs[1]);
        coeffs[2] = std::move(other.coeffs[2]);
    }

    BVP3_4D(const BVP3_4D<x_Size, y_Size, uv_Size> &other)
    {
        // printf("BVP3_4D(const BVP3_4D<x_Size, y_Size, uv_Size> &other)\n");
        coeffs[0] = other.coeffs[0];
        coeffs[1] = other.coeffs[1];
        coeffs[2] = other.coeffs[2];
    }

    BVP3_4D &operator=(BVP3_4D<x_Size, y_Size, uv_Size> &&other)
    {
        // printf("BVP3_4D &operator=(BVP3_4D<x_Size, y_Size, uv_Size> &&other)\n");
        if (this != &other)
        {
            coeffs[0] = std::move(other.coeffs[0]);
            coeffs[1] = std::move(other.coeffs[1]);
            coeffs[2] = std::move(other.coeffs[2]);
        }
        return *this;
    }

    BVP3_4D &operator=(const BVP3_4D<x_Size, y_Size, uv_Size> &other)
    {
        // printf("BVP3_4D &operator=(const BVP3_4D<x_Size, y_Size, uv_Size> &other)\n");
        if (this != &other)
        {
            coeffs[0] = other.coeffs[0];
            coeffs[1] = other.coeffs[1];
            coeffs[2] = other.coeffs[2];
        }
        return *this;
    }

    BVP3_4D(BounderVec3 &p)
    {
        coeffs[0] = BVP_4D<1, 1, 1>{p[0]};
        coeffs[1] = BVP_4D<1, 1, 1>{p[1]};
        coeffs[2] = BVP_4D<1, 1, 1>{p[2]};
    }

    template <size_t Other_x_Size, size_t Other_y_Size, size_t Other_uv_Size, size_t Res_x_Size = BVP_ADD_SIZE(x_Size, Other_x_Size), size_t Res_y_Size = BVP_ADD_SIZE(y_Size, Other_y_Size), size_t Res_uv_Size = BVP_ADD_SIZE(uv_Size, Other_uv_Size)>
    BVP3_4D<Res_x_Size, Res_y_Size, Res_uv_Size> operator+(const BVP3_4D<Other_x_Size, Other_y_Size, Other_uv_Size> &other) const
    {
        return {coeffs[0] + other.coeffs[0], coeffs[1] + other.coeffs[1], coeffs[2] + other.coeffs[2]};
    }

    template <size_t Other_x_Size, size_t Other_y_Size, size_t Other_uv_Size, size_t Res_x_Size = BVP_ADD_SIZE(x_Size, Other_x_Size), size_t Res_y_Size = BVP_ADD_SIZE(y_Size, Other_y_Size), size_t Res_uv_Size = BVP_ADD_SIZE(uv_Size, Other_uv_Size)>
    BVP3_4D<Res_x_Size, Res_y_Size, Res_uv_Size> operator-(const BVP3_4D<Other_x_Size, Other_y_Size, Other_uv_Size> &other) const
    {
        return {coeffs[0] - other.coeffs[0], coeffs[1] - other.coeffs[1], coeffs[2] - other.coeffs[2]};
    }

    template <size_t Other_x_Size, size_t Other_y_Size, size_t Other_uv_Size, size_t Res_x_Size = BVP_MUL_SIZE(x_Size, Other_x_Size), size_t Res_y_Size = BVP_MUL_SIZE(y_Size, Other_y_Size), size_t Res_uv_Size = BVP_MUL_SIZE(uv_Size, Other_uv_Size)>
    BVP3_4D<Res_x_Size, Res_y_Size, Res_uv_Size> operator*(const BVP3_4D<Other_x_Size, Other_y_Size, Other_uv_Size> &other) const
    {
        return {coeffs[0] * other.coeffs[0], coeffs[1] * other.coeffs[1], coeffs[2] * other.coeffs[2]};
    }

    template <size_t Other_x_Size, size_t Other_y_Size, size_t Other_uv_Size, size_t Res_x_Size = BVP_MUL_SIZE(x_Size, Other_x_Size), size_t Res_y_Size = BVP_MUL_SIZE(y_Size, Other_y_Size), size_t Res_uv_Size = BVP_MUL_SIZE(uv_Size, Other_uv_Size)>
    BVP3_4D<Res_x_Size, Res_y_Size, Res_uv_Size> operator*(const BVP_4D<Other_x_Size, Other_y_Size, Other_uv_Size> &other) const
    {
        return {coeffs[0] * other, coeffs[1] * other, coeffs[2] * other};
    }

    template <size_t Other_x_Size, size_t Other_y_Size, size_t Other_uv_Size, size_t Res_x_Size = BVP_MUL_SIZE(x_Size, Other_x_Size), size_t Res_y_Size = BVP_MUL_SIZE(y_Size, Other_y_Size), size_t Res_uv_Size = BVP_MUL_SIZE(uv_Size, Other_uv_Size)>
    BVP_4D<Res_x_Size, Res_y_Size, Res_uv_Size> dot(const BVP3_4D<Other_x_Size, Other_y_Size, Other_uv_Size> &other) const
    { // dot product
        return coeffs[0] * other.coeffs[0] + coeffs[1] * other.coeffs[1] + coeffs[2] * other.coeffs[2];
    }

    template <size_t Other_x_Size, size_t Other_y_Size, size_t Other_uv_Size, size_t Res_x_Size = BVP_MUL_SIZE(x_Size, Other_x_Size), size_t Res_y_Size = BVP_MUL_SIZE(y_Size, Other_y_Size), size_t Res_uv_Size = BVP_MUL_SIZE(uv_Size, Other_uv_Size)>
    BVP3_4D<Res_x_Size, Res_y_Size, Res_uv_Size> cross(const BVP3_4D<Other_x_Size, Other_y_Size, Other_uv_Size> &other) const
    { // cross product
        return {
            coeffs[1] * other.coeffs[2] - coeffs[2] * other.coeffs[1], // yz
            coeffs[2] * other.coeffs[0] - coeffs[0] * other.coeffs[2], // zx
            coeffs[0] * other.coeffs[1] - coeffs[1] * other.coeffs[0]  // xy
        };
    }

    BVP3_4D<x_Size, y_Size, uv_Size> du() const
    {
        return {coeffs[0].du(), coeffs[1].du(), coeffs[2].du()};
    }

    BVP3_4D<x_Size, y_Size, uv_Size> dv() const
    {
        return {coeffs[0].dv(), coeffs[1].dv(), coeffs[2].dv()};
    }

    mutable BVP_4D<x_Size, y_Size, uv_Size> coeffs[3];
};

template <>
BVP_4D<1, 1, 2>::BVP_4D(double *c)
{
    coeffs[0][0][0][0] = c[0];
    coeffs[0][0][0][1] = c[1];
    coeffs[0][0][1][0] = c[2];
    coeffs[0][0][1][1] = c[3];
}

double u_4d_c[] = {0.0, 1.0, 0.0, 0.0};
double v_4d_c[] = {0.0, 0.0, 1.0, 0.0};

BVP_4D<1, 1, 2> u_4d{u_4d_c};
BVP_4D<1, 1, 2> v_4d{v_4d_c};

template <>
BVP_4D<1, 2, 1>::BVP_4D(double *c)
{
    coeffs[0][0][0][0] = c[0];
    coeffs[0][1][0][0] = c[1];
}

template <>
BVP_4D<2, 1, 1>::BVP_4D(double *c)
{
    coeffs[0][0][0][0] = c[0];
    coeffs[1][0][0][0] = c[1];
}

// init

template <>
template <>
BoundingVal BVP_4D<1, 2, 7>::fbound(const BVP_4D<1, 2, 6> &other) const
{
    for (int i = 0; i < 7; i++)
    {
        other.coeffs[0][0][i][6] = 0.0;
        other.coeffs[0][1][i][6] = 0.0;
        other.coeffs[0][0][6][i] = 0.0;
        other.coeffs[0][1][6][i] = 0.0;
    }

    PolyMatrix_4D b_coeffs;
    PolyMatrix_4D other_b_coeffs;
    C2B4d<1, 2, 7>(coeffs, b_coeffs);
    C2B4d<1, 2, 7>(other.coeffs, other_b_coeffs);
    double maxVal = INT_MIN;
    double minVal = INT_MAX;
    for (int i = 0; i < 1; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < 7; k++)
            {
                for (int l = 0; l < 7; l++)
                {
#ifdef COUNTER
                    global_counter++;
#endif
                    double val = b_coeffs[i][j][k][l] / other_b_coeffs[i][j][k][l];
                    if (val > maxVal)
                    {
                        maxVal = val;
                    }
                    if (val < minVal)
                    {
                        minVal = val;
                    }
                }
            }
        }
    }

    return BoundingVal(minVal, maxVal);
}

template <>
template <>
BoundingVal BVP_4D<1, 2, 6>::fbound(const BVP_4D<1, 2, 7> &other) const
{
    for (int i = 0; i < 7; i++)
    {
        coeffs[0][0][i][6] = 0.0;
        coeffs[0][1][i][6] = 0.0;
        coeffs[0][0][6][i] = 0.0;
        coeffs[0][1][6][i] = 0.0;
    }

    PolyMatrix_4D b_coeffs;
    PolyMatrix_4D other_b_coeffs;
    C2B4d<1, 2, 7>(coeffs, b_coeffs);
    C2B4d<1, 2, 7>(other.coeffs, other_b_coeffs);
    double maxVal = INT_MIN;
    double minVal = INT_MAX;
    for (int i = 0; i < 1; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < 7; k++)
            {
                for (int l = 0; l < 7; l++)
                {
#ifdef COUNTER
                    global_counter++;
#endif
                    double val = b_coeffs[i][j][k][l] / other_b_coeffs[i][j][k][l];
                    if (val > maxVal)
                    {
                        maxVal = val;
                    }
                    if (val < minVal)
                    {
                        minVal = val;
                    }
                }
            }
        }
    }

    return BoundingVal(minVal, maxVal);
}

template <>
template <>
BoundingVal BVP_4D<5, 5, 21>::fbound(const BVP_4D<5, 5, 25> &other) const
{
    // 21: begin at 0 0 - 0 4, 2 0 - 2 3
    // 25: begin at 2 0, 2 1, 2 2, 3 0, 3 1, 4 0
    for (int i = 0; i < 5; i++)
    {
        // HERE?
        for (int j = 0; i + j < 5; j++)
        {
            for (int k = 21; k < 25; k++)
            {
                for (int l = 0; l < 25; l++)
                {
                    coeffs[i][j][k][l] = 0.0;
                }
            }

            for (int k = 21; k < 25; k++)
            {
                for (int l = 0; l < 21; l++)
                {
                    coeffs[i][j][l][k] = 0.0;
                }
            }
        }
    }

    PolyMatrix_4D b_coeffs;
    PolyMatrix_4D other_b_coeffs;
    C2B4d<5, 5, 25>(coeffs, b_coeffs); // 25
    C2B4d<5, 5, 25>(other.coeffs, other_b_coeffs);
    double maxVal = INT_MIN;
    double minVal = INT_MAX;
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            for (int k = 0; k < 25; k++)
            {
                for (int l = 0; l < 25; l++)
                {
#ifdef COUNTER
                    global_counter++;
#endif
                    double val = b_coeffs[i][j][k][l] / other_b_coeffs[i][j][k][l];
                    if (val > maxVal)
                    {
                        maxVal = val;
                    }
                    if (val < minVal)
                    {
                        minVal = val;
                    }
                }
            }
        }
    }

    return BoundingVal(minVal, maxVal);
}

template <>
template <>
BVP_4D<1, 1, 2> BVP_4D<1, 1, 1>::operator+(const BVP_4D<1, 1, 2> &other) const
{
    BVP_4D<1, 1, 2> res(other);
    res.coeffs[0][0][0][0] += coeffs[0][0][0][0];

#ifdef COUNTER
    global_counter++;
#endif

    return res;
}

template <>
template <>
BVP_4D<1, 1, 2> BVP_4D<1, 1, 2>::operator+(const BVP_4D<1, 1, 1> &other) const
{
    BVP_4D<1, 1, 2> res(*this);
    res.coeffs[0][0][0][0] += other.coeffs[0][0][0][0];

#ifdef COUNTER
    global_counter++;
#endif

    return res;
}

template <>
template <>
BVP_4D<1, 1, 2> BVP_4D<1, 1, 2>::operator-(const BVP_4D<1, 1, 1> &other) const
{
    BVP_4D<1, 1, 2> res(*this);
    res.coeffs[0][0][0][0] -= other.coeffs[0][0][0][0];

#ifdef COUNTER
    global_counter++;
#endif

    return res;
}

template <>
template <>
BVP_4D<1, 1, 5> BVP_4D<1, 1, 5>::operator+(const BVP_4D<1, 1, 1> &other) const
{
    BVP_4D<1, 1, 5> res(*this);
    res.coeffs[0][0][0][0] += other.coeffs[0][0][0][0];

#ifdef COUNTER
    global_counter++;
#endif

    return res;
}

template <>
template <>
BVP_4D<1, 2, 5> BVP_4D<1, 1, 5>::operator+(const BVP_4D<1, 2, 1> &other) const
{
    BVP_4D<1, 2, 5> res;

    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; i + j < 5; j++)
        {
            res.coeffs[0][0][i][j] = coeffs[0][0][i][j];
        }
    }

    res.coeffs[0][0][0][0] += other.coeffs[0][0][0][0];
    res.coeffs[0][1][0][0] += other.coeffs[0][1][0][0];

#ifdef COUNTER
    global_counter += 2;
#endif

    return res;
}

template <>
template <>
BVP_4D<1, 2, 6> BVP_4D<1, 1, 4>::operator-(const BVP_4D<1, 2, 6> &other) const
{
    BVP_4D<1, 2, 6> res(other.scalarMul(-1));

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; i + j < 4; j++)
        {
#ifdef COUNTER
            global_counter++;
#endif
            res.coeffs[0][0][i][j] += coeffs[0][0][i][j];
        }
    }

    return res;
}

template <>
template <>
BVP_4D<2, 2, 6> BVP_4D<1, 1, 4>::operator-(const BVP_4D<2, 2, 6> &other) const
{
    BVP_4D<2, 2, 6> res(other.scalarMul(-1));

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; i + j < 4; j++)
        {
#ifdef COUNTER
            global_counter++;
#endif
            res.coeffs[0][0][i][j] += coeffs[0][0][i][j];
        }
    }

    return res;
}

template <>
template <>
BVP_4D<1, 2, 5> BVP_4D<1, 1, 5>::operator+(const BVP_4D<1, 2, 5> &other) const
{
    BVP_4D<1, 2, 5> res(other);

    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; i + j < 5; j++)
        {
#ifdef COUNTER
            global_counter++;
#endif
            res.coeffs[0][0][i][j] += coeffs[0][0][i][j];
        }
    }

    return res;
}

template <>
template <>
BVP_4D<1, 1, 5> BVP_4D<1, 1, 5>::operator-(const BVP_4D<1, 1, 1> &other) const
{
    BVP_4D<1, 1, 5> res(*this);

    res.coeffs[0][0][0][0] -= other.coeffs[0][0][0][0];

#ifdef COUNTER
    global_counter++;
#endif

    return res;
}

template <>
template <>
BVP_4D<2, 2, 5> BVP_4D<1, 2, 5>::operator+(const BVP_4D<2, 1, 1> &other) const
{
    BVP_4D<2, 2, 5> res;

    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; i + j < 5; j++)
        {
#ifdef COUNTER
            global_counter++;
#endif
            res.coeffs[0][1][i][j] = coeffs[0][1][i][j];
            res.coeffs[0][0][i][j] = coeffs[0][0][i][j];
        }
    }

    res.coeffs[0][0][0][0] += other.coeffs[0][0][0][0];
    res.coeffs[1][0][0][0] += other.coeffs[1][0][0][0];

    return res;
}

// optimize

template <>
template <>
BVP_4D<5, 5, 23> BVP_4D<5, 5, 23>::operator-(const BVP_4D<5, 5, 23> &other) const
{
    BVP_4D<5, 5, 23> res;

    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5 - i; j++)
        {
            for (int k = 0; k < 21 - 4 * i; k++)
            {
                for (int l = 0; l + k < 21 - 4 * i; l++)
                {
#ifdef COUNTER
                    global_counter++;
#endif
                    res.coeffs[i][j][k][l] = coeffs[i][j][k][l] - other.coeffs[i][j][k][l];
                }
            }
        }
    }

    return res;
}

template <>
template <>
BVP_4D<5, 5, 23> BVP_4D<3, 3, 12>::operator*(const BVP_4D<3, 3, 12> &other) const
{
    BVP_4D<5, 5, 23> res;

    for (int i1 = 0; i1 < 3; i1++)
    {
        for (int i2 = 0; i2 < 3; i2++)
        {
            for (int j1 = 0; j1 < 3 - i1; j1++)
            {
                for (int j2 = 0; j2 < 3 - i2; j2++)
                {
                    for (int k1 = 0; k1 < 11 - 4 * i1; k1++)
                    {
                        for (int k2 = 0; k2 < 11 - 4 * i2; k2++)
                        {
                            for (int l1 = 0; l1 + k1 < 11 - 4 * i1; l1++)
                            {
                                for (int l2 = 0; l2 + k2 < 11 - 4 * i2; l2++)
                                {
#ifdef COUNTER
                                    global_counter += 2;
#endif
                                    res.coeffs[i1 + i2][j1 + j2][k1 + k2][l1 + l2] += coeffs[i1][j1][k1][l1] * other.coeffs[i2][j2][k2][l2];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return res;
}

template <>
template <>
BVP_4D<5, 5, 21> BVP_4D<3, 3, 11>::operator*(const BVP_4D<3, 3, 11> &other) const
{
    BVP_4D<5, 5, 21> res;

    for (int i1 = 0; i1 < 3; i1++)
    {
        for (int i2 = 0; i2 < 3; i2++)
        {
            for (int j1 = 0; j1 < 3 - i1; j1++)
            {
                for (int j2 = 0; j2 < 3 - i2; j2++)
                {
                    for (int k1 = 0; k1 < 11 - 4 * i1; k1++)
                    {
                        for (int k2 = 0; k2 < 11 - 4 * i2; k2++)
                        {
                            for (int l1 = 0; l1 + k1 < 11 - 4 * i1; l1++)
                            {
                                for (int l2 = 0; l2 + k2 < 11 - 4 * i2; l2++)
                                {
#ifdef COUNTER
                                    global_counter += 2;
#endif
                                    res.coeffs[i1 + i2][j1 + j2][k1 + k2][l1 + l2] += coeffs[i1][j1][k1][l1] * other.coeffs[i2][j2][k2][l2];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return res;
}