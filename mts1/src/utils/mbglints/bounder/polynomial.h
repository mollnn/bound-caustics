#pragma once
#include "utils.h"

template <size_t Size>
class BVP
{
public:
    BVP()
    {
        for (int i = 0; i < Size; i++)
        {
            for (int j = 0; j < Size; j++)
            {
                coeffs[i][j] = 0;
            }
        }
    }

    explicit BVP(const double d) { coeffs[0][0] = d; }

    BVP(const double *c)
    {
        printf("No general implementation -- BVP<%ld>()\n", Size);
    }

    BVP(BVP<Size> &&other)
    {
        for (size_t i = 0; i < Size; ++i)
        {
            for (size_t j = 0; j < Size; ++j)
            {
                coeffs[i][j] = std::move(other.coeffs[i][j]);
            }
        }
    }

    BVP &operator=(BVP<Size> &&other)
    {
        if (this != &other)
        {
            for (int i = 0; i < Size; i++)
            {
                for (int j = 0; j < Size; j++)
                {
                    coeffs[i][j] = std::move(other.coeffs[i][j]);
                }
            }
        }
        return *this;
    }

    BVP(const BVP<Size> &other)
    {
        // std::cout<< "visit dup" << std::endl;
        for (int i = 0; i < Size; i++)
        {
            for (int j = 0; j < Size; j++)
            {
                coeffs[i][j] = other.coeffs[i][j];
            }
        }
    }

    BVP &operator=(const BVP<Size> &other)
    {
        // std::cout<< "visit =" << std::endl;
        if (this != &other)
        {
            for (int i = 0; i < Size; i++)
            {
                for (int j = 0; j < Size; j++)
                {
                    coeffs[i][j] = other.coeffs[i][j];
                }
            }
        }
        return *this;
    }

    template <size_t Other_Size, size_t Res_Size = BVP_ADD_SIZE(Size, Other_Size)>
    BVP<Res_Size> operator+(const BVP<Other_Size> &other) const
    {
        // printf("No general implementation -- BVP<%ld>+()\n", Size);
        for (size_t i = Size; i < Res_Size; i++)
        {
            for (size_t j = 0; j < Res_Size; j++)
            {
                coeffs[i][j] = 0.f;
                coeffs[j][i] = 0.f;
            }
        }

        for (size_t i = Other_Size; i < Res_Size; i++)
        {
            for (size_t j = 0; j < Res_Size; j++)
            {
                other.coeffs[i][j] = 0.f;
                other.coeffs[j][i] = 0.f;
            }
        }

        BVP<Res_Size> res;

        for (size_t i = 0; i < Res_Size; ++i)
        {
            for (size_t j = 0; i + j < Res_Size; ++j)
            {
#ifdef COUNTER
                global_counter++;
#endif
                res.coeffs[i][j] = coeffs[i][j] + other.coeffs[i][j];
            }
        }

        return res;
    }

    template <size_t Other_Size, size_t Res_Size = BVP_ADD_SIZE(Size, Other_Size)>
    BVP<Res_Size> operator-(const BVP<Other_Size> &other) const
    {
        // printf("No general implementation -- BVP<%ld>-()\n", Size);
        for (size_t i = Size; i < Res_Size; i++)
        {
            for (size_t j = 0; j < Res_Size; j++)
            {
                coeffs[i][j] = 0.f;
                coeffs[j][i] = 0.f;
            }
        }

        for (size_t i = Other_Size; i < Res_Size; i++)
        {
            for (size_t j = 0; j < Res_Size; j++)
            {
                other.coeffs[i][j] = 0.f;
                other.coeffs[j][i] = 0.f;
            }
        }

        BVP<Res_Size> res;

        for (size_t i = 0; i < Res_Size; ++i)
        {
            for (size_t j = 0; i + j < Res_Size; ++j)
            {
#ifdef COUNTER
                global_counter++;
#endif
                res.coeffs[i][j] = coeffs[i][j] - other.coeffs[i][j];
            }
        }

        return res;
    }

    template <size_t Other_Size, size_t Res_Size = BVP_MUL_SIZE(Size, Other_Size)>
    BVP<Res_Size> operator*(const BVP<Other_Size> &other) const
    {
        // printf("No general implementation -- BVP<%ld>*BVP<%ld>()\n", Size, Other_Size);
        BVP<Res_Size> result;
        for (size_t i = 0; i < Size; ++i)
        {
            for (size_t k = 0; k < Other_Size; ++k)
            {
                for (size_t j = 0; i + j < Size; ++j)
                {
                    for (size_t l = 0; k + l < Other_Size; ++l)
                    {
#ifdef COUNTER
                        global_counter += 2;
#endif
                        result.coeffs[i + k][j + l] += coeffs[i][j] * other.coeffs[k][l];
                    }
                }
            }
        }
        return result;
    }

    BVP<Size> scalarMul(const double other) const
    {
        // printf("No general implementation -- BVP<%ld>*(double)\n", Size);
        BVP<Size> result;

        for (size_t i = 0; i < Size; ++i)
        {
            for (size_t j = 0; i + j < Size; ++j)
            {
#ifdef COUNTER
                global_counter++;
#endif
                result.coeffs[i][j] = coeffs[i][j] * other;
            }
        }
        return result;
    }

    BoundingVal bound() const
    {
        PolyMatrix b_coeffs;
        C2B2d<Size>(coeffs, b_coeffs);
        double maxVal = INT_MIN;
        double minVal = INT_MAX;

        // bound_matrix_size.emplace(size);

        for (int i = 0; i < Size; ++i)
        {
            for (int j = 0; j < Size; ++j)
            {
                if (b_coeffs[i][j] > maxVal)
                {
                    maxVal = b_coeffs[i][j];
                }
                if (b_coeffs[i][j] < minVal)
                {
                    minVal = b_coeffs[i][j];
                }
            }
        }
        // std::cout << "bound:" << size << std::endl;
        // printMatrix(b_coeffs);
        return BoundingVal(minVal, maxVal);
    }

    template <size_t Other_Size, size_t Res_Size = BVP_ADD_SIZE(Size, Other_Size)>
    BoundingVal fbound(const BVP<Other_Size> &other) const
    {
        for (size_t i = Size; i < Res_Size; i++)
        {
            for (size_t j = 0; j < Res_Size; j++)
            {
                coeffs[i][j] = 0.f;
                coeffs[j][i] = 0.f;
            }
        }

        for (size_t i = Other_Size; i < Res_Size; i++)
        {
            for (size_t j = 0; j < Res_Size; j++)
            {
                other.coeffs[i][j] = 0.f;
                other.coeffs[j][i] = 0.f;
            }
        }
        
        PolyMatrix b_coeffs;
        PolyMatrix other_b_coeffs;
        C2B2d<Res_Size>(coeffs, b_coeffs);
        C2B2d<Res_Size>(other.coeffs, other_b_coeffs);
        double maxVal = INT_MIN;
        double minVal = INT_MAX;

        // fbound_matrix_size.emplace(degree);

        for (int i = 0; i < Res_Size; ++i)
        {
            for (int j = 0; j < Res_Size; ++j)
            {
#ifdef COUNTER
                global_counter++;
#endif
                double val = b_coeffs[i][j] / other_b_coeffs[i][j];
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
        // std::cout << "fbound_numer:" << size << std::endl;
        // printMatrix(b_coeffs);
        // std::cout << "fbound_denom:" << other.size << std::endl;
        // printMatrix(other_b_coeffs);
        return BoundingVal(minVal, maxVal);
    }

    BVP<Size> du() const
    {
        BVP<Size> res;

        // du_matrix_size.emplace(size);

        for (int i = 1; i < Size; i++)
        {
            for (int j = 0; i + j < Size; j++)
            {
#ifdef COUNTER
                global_counter++;
#endif
                res.coeffs[i - 1][j] = i * coeffs[i][j];
            }
        }
        // std::cout << "du:" << size << std::endl;
        // printMatrix(res.coeffs);
        // std::cout << "coef:" << size << std::endl;
        // printMatrix(coeffs);
        return res;
    }

    BVP<Size> dv() const
    {
        BVP<Size> res;

        // dv_matrix_size.emplace(size);

        for (int i = 0; i < Size; i++)
        {
            for (int j = 1; i + j < Size; j++)
            {
#ifdef COUNTER
                global_counter++;
#endif
                res.coeffs[i][j - 1] = j * coeffs[i][j];
            }
        }
        return res;
    }

public:
    // TODO: Change to Size
    mutable PolyMatrix coeffs;
};

template <size_t Size>
class BVP3
{
public:
    BVP3(BVP<Size> &&_x, BVP<Size> &&_y, BVP<Size> &&_z)
    {
        coeffs[0] = std::move(_x);
        coeffs[1] = std::move(_y);
        coeffs[2] = std::move(_z);
    }

    BVP3(const BVP<Size> &_x, const BVP<Size> &_y, const BVP<Size> &_z)
    {
        coeffs[0] = _x;
        coeffs[1] = _y;
        coeffs[2] = _z;
    }

    BVP3(BVP3<Size> &&other)
    {
        coeffs[0] = std::move(other.coeffs[0]);
        coeffs[1] = std::move(other.coeffs[1]);
        coeffs[2] = std::move(other.coeffs[2]);
    }

    BVP3(const BVP3<Size> &other)
    {
        coeffs[0] = other.coeffs[0];
        coeffs[1] = other.coeffs[1];
        coeffs[2] = other.coeffs[2];
    }

    // BVP3 &operator=(const BVP<Size> &_x, const BVP<Size> &_y, const BVP<Size> &_z)
    // {
    //     x = _x;
    //     y = _y;
    //     z = _z;
    //     return *this;
    // }

    // BVP3 &operator=(const BVP<Size> &&_x, const BVP<Size> &&_y, const BVP<Size> &&_z)
    // {
    //     x = std::move(_x);
    //     y = std::move(_y);
    //     z = std::move(_z);
    //     return *this;
    // }

    BVP3 &operator=(BVP3<Size> &&other)
    {
        if (this != &other)
        {
            coeffs[0] = std::move(other.coeffs[0]);
            coeffs[1] = std::move(other.coeffs[1]);
            coeffs[2] = std::move(other.coeffs[2]);
        }
        return *this;
    }

    BVP3 &operator=(const BVP3<Size> &other)
    {
        if (this != &other)
        {
            coeffs[0] = other.coeffs[0];
            coeffs[1] = other.coeffs[1];
            coeffs[2] = other.coeffs[2];
        }
        return *this;
    }

    BVP3(BounderVec3 &p)
    {
        coeffs[0] = BVP<1>{p[0]};
        coeffs[1] = BVP<1>{p[1]};
        coeffs[2] = BVP<1>{p[2]};
    }

    template <size_t Other_Size, size_t Res_Size = BVP_MUL_SIZE(Size, Other_Size)>
    BVP<Res_Size> dot(const BVP3<Other_Size> &other) const
    {
        return coeffs[0] * other.coeffs[0] + coeffs[1] * other.coeffs[1] + coeffs[2] * other.coeffs[2];
    }

    template <size_t Other_Size, size_t Res_Size = BVP_MUL_SIZE(Size, Other_Size)>
    BVP3<Res_Size> cross(const BVP3<Other_Size> &other) const
    {
        return {
            coeffs[1] * other.coeffs[2] - coeffs[2] * other.coeffs[1], // yz
            coeffs[2] * other.coeffs[0] - coeffs[0] * other.coeffs[2], // zx
            coeffs[0] * other.coeffs[1] - coeffs[1] * other.coeffs[0]  // xy
        };
    }

    template <size_t Other_Size, size_t Res_Size = BVP_MUL_SIZE(Size, Other_Size)>
    BVP3<Res_Size> operator*(const BVP3<Other_Size> &other) const
    {
        return {coeffs[0] * other.coeffs[0], coeffs[1] * other.coeffs[1], coeffs[2] * other.coeffs[2]};
    }

    template <size_t Other_Size, size_t Res_Size = BVP_MUL_SIZE(Size, Other_Size)>
    BVP3<Res_Size> operator*(const BVP<Other_Size> &other) const
    {
        return {coeffs[0] * other, coeffs[1] * other, coeffs[2] * other};
    }

    template <size_t Other_Size, size_t Res_Size = BVP_ADD_SIZE(Size, Other_Size)>
    BVP3<Res_Size> operator-(const BVP3<Other_Size> &other) const
    {
        return {coeffs[0] - other.coeffs[0], coeffs[1] - other.coeffs[1], coeffs[2] - other.coeffs[2]};
    }

    template <size_t Other_Size, size_t Res_Size = BVP_ADD_SIZE(Size, Other_Size)>
    BVP3<Res_Size> operator+(const BVP3<Other_Size> &other) const
    {
        return {coeffs[0] + other.coeffs[0], coeffs[1] + other.coeffs[1], coeffs[2] + other.coeffs[2]};
    }

    BVP3<Size> du() const { return {coeffs[0].du(), coeffs[1].du(), coeffs[2].du()}; }

    BVP3<Size> dv() const { return {coeffs[0].dv(), coeffs[1].dv(), coeffs[2].dv()}; }

public:
    mutable BVP<Size> coeffs[3];
};

template <>
BVP<2>::BVP(const double *c)
{
    coeffs[0][0] = c[0];
    coeffs[0][1] = c[1];
    coeffs[1][0] = c[2];
    coeffs[1][1] = c[3];
}

double u_a[] = {0.0, 0.0, 1.0, 0.0};
double u_b[] = {0.0, 1.0, 0.0, 0.0};

BVP<2> u{u_a};
BVP<2> v{u_b};