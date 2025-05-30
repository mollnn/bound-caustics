from alias import *
from transform import *
from utils import *

@njit(fastmath=True)
def batch_get_rational_d1(data):
    PL, p10, p11, p12, n10, n11, n12 = \
        data[0], data[1], data[2], data[3], data[4], data[5], data[6]
    batch_size = len(p10[0])
    
    u = BPoly(np.array([[[[0], [1]]]], dtype=np.float64))
    v = BPoly(np.array([[[[0, 1]]]], dtype=np.float64))

    p10 = BatchBPoly3(p10)
    p11 = BatchBPoly3(p11)
    p12 = BatchBPoly3(p12)
    n10 = BatchBPoly3(n10)
    n11 = BatchBPoly3(n11)
    n12 = BatchBPoly3(n12)
    x0 =  BatchBPoly3(PL)

    η = BPoly(np.array([[[[η_VAL]]]], dtype=np.float64))
    x1 = p10 + p11 * u + p12 * v
    n1 = n10 + n11 * u + n12 * v if SHADING_NORMAL else n10
    d0 = x1 - x0

    β = n1.dot(n1) * d0.dot(d0) - (n1.dot(n1) * d0.dot(d0) - n1.dot(d0) * n1.dot(d0)) * η * η
    Iβ = β.bound()
    
    for i in range(batch_size):
        Iβ[0, i] = Iβ[0, i] if Iβ[0, i] >= β_MIN * 0.5 else β_MIN * 0.5
        Iβ[1, i] = Iβ[1, i] if Iβ[1, i] >= β_MIN * 0.99 else β_MIN * 0.99

    mid = Iβ[0]
    a1, a0, Δξ, Δξ1 = compute_easy_approx_for_sqrt(Iβ[0], Iβ[1])
    a1, a0 = BatchBPoly(a1), BatchBPoly(a0)
    temp_ξ = np.empty((1, 2, 1, 1, batch_size), dtype=np.float64)
    for i in range(batch_size):
        temp_ξ[0, 0, 0, 0, i] = -Δξ[i]
        temp_ξ[0, 1, 0, 0, i] = Δξ[i]
    ξ = BatchBPoly(temp_ξ)
    sqrtβ = a1 * β + a0 + ξ * (β - BatchBPoly(mid))
        
    Iβ = β.bound()

    d1 = (d0 * (n1.dot(n1)) - n1 * n1.dot(d0)) * η - n1 * sqrtβ

    return d1

@njit(fastmath=True)
def batch_get_rational_double_refract_half_deriv(data, u1m, u1M, v1m, v1M):
    PL, P10, P11, P12, N10, N11, N12, P20, P21, P22, N20, N21, N22, P30, P31, P32 = \
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15]
    batch_size = len(u1m)

    p10_ = P10 + P11 * u1m + P12 * v1m
    p11_ = P11 * (u1M - u1m)
    p12_ = P12 * (v1M - v1m)
    n10_ = N10 + N11 * u1m + N12 * v1m
    n11_ = N11 * (u1M - u1m)
    n12_ = N12 * (v1M - v1m)
    u = BPoly(np.array([[[[0], [1]]]], dtype=np.float64))
    v = BPoly(np.array([[[[0, 1]]]], dtype=np.float64))

    p10 = BatchBPoly3(p10_)
    p11 = BatchBPoly3(p11_)
    p12 = BatchBPoly3(p12_)
    n10 = BatchBPoly3(n10_)
    n11 = BatchBPoly3(n11_)
    n12 = BatchBPoly3(n12_)
    x0 =  BatchBPoly3(PL)
    p20 = BatchBPoly3(P20)
    p21 = BatchBPoly3(P21)
    p22 = BatchBPoly3(P22)

    η = BPoly(np.array([[[[η_VAL]]]], dtype=np.float64))
    x1 = p10 + p11 * u + p12 * v
    n1 = n10 + n11 * u + n12 * v if SHADING_NORMAL else n10
    d0 = x1 - x0

    β = n1.dot(n1) * d0.dot(d0) - (n1.dot(n1) * d0.dot(d0) - n1.dot(d0) * n1.dot(d0)) * η * η
    Iβ = β.bound()
    
    for i in range(batch_size):
        Iβ[0, i] = Iβ[0, i] if Iβ[0, i] >= β_MIN * 0.5 else β_MIN * 0.5
        Iβ[1, i] = Iβ[1, i] if Iβ[1, i] >= β_MIN * 0.99 else β_MIN * 0.99

    mid = Iβ[0]
    a1, a0, Δξ, Δξ1 = compute_easy_approx_for_sqrt(Iβ[0], Iβ[1])
    a1, a0 = BatchBPoly(a1), BatchBPoly(a0)
    temp_ξ = np.empty((1, 2, 1, 1, batch_size), dtype=np.float64)
    for i in range(batch_size):
        temp_ξ[0, 0, 0, 0, i] = -Δξ[i]
        temp_ξ[0, 1, 0, 0, i] = Δξ[i]
    ξ = BatchBPoly(temp_ξ)
    temp_ξ1 = np.empty((2, 1, 1, 1, batch_size), dtype=np.float64)
    for i in range(batch_size):
        temp_ξ1[0, 0, 0, 0, i] = 0
        temp_ξ1[1, 0, 0, 0, i] = Δξ1[i]
    ξ1 = BatchBPoly(temp_ξ1)
    sqrtβ = a1 * β + a0 + ξ * (β - BatchBPoly(mid)) + ξ1
        
    Iβ = β.bound()

    d1 = (d0 * (n1.dot(n1)) - n1 * n1.dot(d0)) * η - n1 * sqrtβ
    C = d0.dot(d0)

    u2p = d1.cross(p22).dot(x1 - p20)
    v2p = (x1 - p20).cross(p21).dot(d1)
    k2 = d1.cross(p22).dot(p21)

    return u2p, v2p, k2, C
    

@njit(fastmath=True)
def batch_get_rational_double_refract_1(cache_c2b, data, u1m, u1M, v1m, v1M, approx_method="deriv"):
    PL, P10, P11, P12, N10, N11, N12, P20, P21, P22, N20, N21, N22, P30, P31, P32 = \
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15]
    batch_size = len(u1m)

    p10_ = P10 + P11 * u1m + P12 * v1m
    p11_ = P11 * (u1M - u1m)
    p12_ = P12 * (v1M - v1m)
    n10_ = N10 + N11 * u1m + N12 * v1m
    n11_ = N11 * (u1M - u1m)
    n12_ = N12 * (v1M - v1m)
    u = BPoly(np.array([[[[0], [1]]]], dtype=np.float64))
    v = BPoly(np.array([[[[0, 1]]]], dtype=np.float64))

    p10 = BatchBPoly3(p10_)
    p11 = BatchBPoly3(p11_)
    p12 = BatchBPoly3(p12_)
    n10 = BatchBPoly3(n10_)
    n11 = BatchBPoly3(n11_)
    n12 = BatchBPoly3(n12_)
    x0 =  BatchBPoly3(PL)
    p20 = BatchBPoly3(P20)
    p21 = BatchBPoly3(P21)
    p22 = BatchBPoly3(P22)
    n20 = BatchBPoly3(N20)
    n21 = BatchBPoly3(N21)
    n22 = BatchBPoly3(N22)
    p30 = BatchBPoly3(P30)
    p31 = BatchBPoly3(P31)
    p32 = BatchBPoly3(P32)

    η = BPoly(np.array([[[[η_VAL]]]], dtype=np.float64))
    x1 = p10 + p11 * u + p12 * v
    n1 = n10 + n11 * u + n12 * v if SHADING_NORMAL else n10
    d0 = x1 - x0

    β = n1.dot(n1) * d0.dot(d0) - (n1.dot(n1) * d0.dot(d0) - n1.dot(d0) * n1.dot(d0)) * η * η
    if approx_method == "primal":
        β = batch_reduce_all(β, cache_c2b, new_axis=3)
    Iβ = β.bound()
    
    if approx_method == "deriv" or approx_method == "half_deriv":
        for i in range(batch_size):
            Iβ[0, i] = Iβ[0, i] if Iβ[0, i] >= β_MIN * 0.5 else β_MIN * 0.5
            Iβ[1, i] = Iβ[1, i] if Iβ[1, i] >= β_MIN * 0.99 else β_MIN * 0.99
    else:
        for i in range(batch_size):
            Iβ[0, i] = Iβ[0, i] if Iβ[0, i] >= 0 else 0
            Iβ[1, i] = Iβ[1, i] if Iβ[1, i] >= β_MIN * 0.99 else β_MIN * 0.99

    mid = Iβ[0]
    if approx_method == "deriv" or approx_method == "half_deriv":
        a1, a0, Δξ, Δξ1 = compute_easy_approx_for_sqrt(Iβ[0], Iβ[1])
        a1, a0 = BatchBPoly(a1), BatchBPoly(a0)
        temp_ξ = np.empty((1, 2, 1, 1, batch_size), dtype=np.float64)
        for i in range(batch_size):
            temp_ξ[0, 0, 0, 0, i] = -Δξ[i]
            temp_ξ[0, 1, 0, 0, i] = Δξ[i]
        ξ = BatchBPoly(temp_ξ)
        temp_ξ1 = np.empty((2, 1, 1, 1, 1, batch_size), dtype=np.float64)
        for i in range(batch_size):
            temp_ξ1[0, 0, 0, 0, i] = 0
            temp_ξ1[1, 0, 0, 0, i] = Δξ1[i]
        ξ1 = BatchBPoly(temp_ξ1)
        sqrtβ = a1 * β + a0 + ξ * (β - BatchBPoly(mid)) + ξ1
    elif approx_method == "primal":
        a1, a0, Δξ = compute_line_approx_for_sqrt(Iβ[0], Iβ[1])
        a1, a0 = BatchBPoly(a1), BatchBPoly(a0)
        temp_ξ = np.empty((1, 2, 1, 1, batch_size), dtype=np.float64)
        for i in range(batch_size):
            temp_ξ[0, 0, 0, 0, i] = -Δξ[i]
            temp_ξ[0, 1, 0, 0, i] = Δξ[i]
        ξ = BatchBPoly(temp_ξ)
        sqrtβ = a1 * β + a0 + ξ
    else:
        raise Exception("bad approx_method")
    Iβ = β.bound()

    d1 = (d0 * (n1.dot(n1)) - n1 * n1.dot(d0)) * η - n1 * sqrtβ
    C = d0.dot(d0)

    u2p = d1.cross(p22).dot(x1 - p20)
    v2p = (x1 - p20).cross(p21).dot(d1)
    k2 = d1.cross(p22).dot(p21)

    Ik2 = k2.bound()
    k2_aligned = k2.align_degree_to(u2p)
    Iu2 = u2p.fbound(k2_aligned)
    Iv2 = v2p.fbound(k2_aligned)

    return Iu2, Iv2, u2p, v2p, k2, p20, p21, p22, n20, n21, n22, p30, p31, p32, d0, n1, x1, d1, C, β, Ik2

@njit(fastmath=True)
def batch_get_rational_double_refract_2(cache_c2b, u2p, v2p, k2, p20, p21, p22, n20, n21, n22, p30, p31, p32, d0, n1, x1, d1, C, β, approx_method="deriv"):
    batch_size = u2p.a.shape[-1]
    k2_original = k2 * BPoly(np.array(1, dtype=np.float64))

    Ik2 = k2.bound()
    weight = np.ones(batch_size, dtype=np.float64)
    if SHADING_NORMAL2:
        for i in range(batch_size):
            if Ik2[0, i] * Ik2[1, i] > 0 and Ik2[0, i] < 0:
                weight[i] = -1.
    k2 = k2 * BatchBPoly(weight)
    u2p = u2p * BatchBPoly(weight)
    v2p = v2p * BatchBPoly(weight)

    x2 = p20 * k2 + p21 * u2p + p22 * v2p  # / k2
    n2 = (n20 * k2 + n21 * u2p + n22 * v2p if SHADING_NORMAL2 else n20) * BPoly(np.array([[[[-1.0]]]], dtype=np.float64))

    η2 = BatchBPoly(np.array([[[[1.0 / η_VAL for _ in range(batch_size)]]]], dtype=np.float64))

    n2_n2 = n2.dot(n2)
    n2_d1 = n2.dot(d1)
    d1_d1 = d1.dot(d1)
    β2 = n2_n2 * d1_d1 - (n2_n2 * d1_d1 - n2_d1 * n2_d1) * η2 * η2  # / k2 ** 2
    β2q = n2_n2 * d1_d1
    if approx_method == "primal":
        β2 = batch_reduce_all(β2, cache_c2b, new_axis=2)
        β2q = batch_reduce_all(β2q, cache_c2b, new_axis=2)

    Iβ2 = β2.bound()
    
    if approx_method == "deriv":
        for i in range(batch_size):
            Iβ2[0, i] = Iβ2[0, i] if Iβ2[0, i] >= β_MIN * 0.5 else β_MIN * 0.5
            Iβ2[1, i] = Iβ2[1, i] if Iβ2[1, i] >= β_MIN * 0.99 else β_MIN * 0.99
    else:
        for i in range(batch_size):
            Iβ2[0, i] = Iβ2[0, i] if Iβ2[0, i] >= 0 else 0
            Iβ2[1, i] = Iβ2[1, i] if Iβ2[1, i] >= β_MIN * 0.99 else β_MIN * 0.99

    mid = Iβ2[0]
    if approx_method == "deriv":
        a1, a0, Δζ, Δζ1 = compute_easy_approx_for_sqrt(Iβ2[0], Iβ2[1])
        a1, a0 = BatchBPoly(a1), BatchBPoly(a0)
        temp_ζ = np.empty((2, 1, 1, 1, batch_size), dtype=np.float64)
        for i in range(batch_size):
            temp_ζ[0, 0, 0, 0, i] = -Δζ[i]
            temp_ζ[1, 0, 0, 0, i] = Δζ[i]
        ζ = BatchBPoly(temp_ζ)
        temp_ξ1 = np.empty((2, 1, 1, 1, 1, 1, batch_size), dtype=np.float64)
        for i in range(batch_size):
            temp_ξ1[0, 0, 0, 0, i] = 0
            temp_ξ1[1, 0, 0, 0, i] = Δζ1[i]
        ξ1 = BatchBPoly(temp_ξ1)
        sqrtβ2 = a1 * β2 + a0 + ζ * (β2 - BatchBPoly(mid)) + ξ1 # / k2
    elif approx_method == "primal":
        a1, a0, Δζ = compute_line_approx_for_sqrt(Iβ2[0], Iβ2[1])
        a1, a0 = BatchBPoly(a1), BatchBPoly(a0)
        temp_ζ = np.empty((2, 1, 1, 1, batch_size), dtype=np.float64)
        for i in range(batch_size):
            temp_ζ[0, 0, 0, 0, i] = -Δζ[i]
            temp_ζ[1, 0, 0, 0, i] = Δζ[i]
        ζ = BatchBPoly(temp_ζ)
        sqrtβ2 = a1 * β2 + a0 + ζ
    else:
        raise Exception("bad approx_method")
    Iβ2 = β2.bound()

    d2 = (d1 * n2_n2 - n2 * n2_d1) * η2 - n2 * sqrtβ2  # / k2 ** 2

    u3p = d2.cross(p32).dot(x2 - p30 * k2)
    v3p = (x2 - p30 * k2).cross(p31).dot(d2)
    k3 = d2.cross(p32).dot(p31) * k2

    s0, s1, s2, s3 = None, None, None, None
    if approx_method == "primal":
        s0 = n1.dot(d0) * BatchBPoly(np.array([[[[-1 for _ in range(batch_size)]]]], dtype=np.float64))
        s1 = (x1 - p20).cross(p21).dot(p22) * k2_original
        s2 = n2_d1 * BatchBPoly(np.array([[[[-1 for _ in range(batch_size)]]]], dtype=np.float64))
        if SHADING_NORMAL2:
            s2 = s2 * k2
        s3 = (x2 - p30 * k2).cross(p31).dot(p32) * k3

    return u3p, v3p, k3, u2p, v2p, k2, C, s0, s1, s2, s3, β, β2, β2q


@njit(fastmath=True)
def batch_get_rational_double_refract_2_deriv(u2p, v2p, k2, p20, p21, p22, n20, n21, n22, p30, p31, p32, d1):
    batch_size = u2p.a.shape[-1]
    Ik2 = k2.bound()
    weight = np.ones(batch_size, dtype=np.float64)
    if SHADING_NORMAL2:
        for i in range(batch_size):
            if Ik2[0, i] * Ik2[1, i] > 0 and Ik2[0, i] < 0:
                weight[i] = -1.
    k2 = k2 * BatchBPoly(weight)
    u2p = u2p * BatchBPoly(weight)
    v2p = v2p * BatchBPoly(weight)

    x2 = p20 * k2 + p21 * u2p + p22 * v2p
    n2 = (n20 * k2 + n21 * u2p + n22 * v2p if SHADING_NORMAL2 else n20) * BPoly(np.array([[[[-1.0]]]], dtype=np.float64))

    η2 = BatchBPoly(np.array([[[[1.0 / η_VAL for _ in range(batch_size)]]]], dtype=np.float64))

    n2_n2 = n2.dot(n2)
    n2_d1 = n2.dot(d1)
    d1_d1 = d1.dot(d1)
    β2 = n2_n2 * d1_d1 - (n2_n2 * d1_d1 - n2_d1 * n2_d1) * η2 * η2
    # β2q = n2_n2 * d1_d1

    Iβ2 = β2.bound()
    
    for i in range(batch_size):
        Iβ2[0, i] = Iβ2[0, i] if Iβ2[0, i] >= β_MIN * 0.5 else β_MIN * 0.5
        Iβ2[1, i] = Iβ2[1, i] if Iβ2[1, i] >= β_MIN * 0.99 else β_MIN * 0.99

    mid = Iβ2[0]
    a1, a0, Δζ, Δζ1 = compute_easy_approx_for_sqrt(Iβ2[0], Iβ2[1])
    a1, a0 = BatchBPoly(a1), BatchBPoly(a0)
    temp_ζ = np.empty((2, 1, 1, 1, batch_size), dtype=np.float64)
    for i in range(batch_size):
        temp_ζ[0, 0, 0, 0, i] = -Δζ[i]
        temp_ζ[1, 0, 0, 0, i] = Δζ[i]
    ζ = BatchBPoly(temp_ζ)
    temp_ξ1 = np.empty((2, 1, 1, 1, 1, 1, batch_size), dtype=np.float64)
    for i in range(batch_size):
        temp_ξ1[0, 0, 0, 0, i] = 0
        temp_ξ1[1, 0, 0, 0, i] = Δζ1[i]
    ξ1 = BatchBPoly(temp_ξ1)
    sqrtβ2 = a1 * β2 + a0 + ζ * (β2 - BatchBPoly(mid)) + ξ1
    Iβ2 = β2.bound()

    d2 = (d1 * n2_n2 - n2 * n2_d1) * η2 - n2 * sqrtβ2

    u3p = d2.cross(p32).dot(x2 - p30 * k2)
    v3p = (x2 - p30 * k2).cross(p31).dot(d2)
    k3 = d2.cross(p32).dot(p31) * k2

    return u3p, v3p, k3, u2p, v2p, k2

