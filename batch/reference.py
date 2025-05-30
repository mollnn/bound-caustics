# ray tracing (within given triangle tuples) routines. should not be used by the algorithm. only for generating reference.
from alias import *
from utils import *
from formula import *

def ptrace(PL, P10, P11, P12, N10, N11, N12, P20, P21, P22, N20, N21, N22, P30, P31, P32, u, v, ignore_check=False):
    x1 = P10 + P11 * u + P12 * v
    n1 = N10 + N11 * u + N12 * v
    d0 = x1 - PL
    if CHAIN_TYPE == 1 or CHAIN_TYPE == 11:
        d1 = reflect3d(d0, n1)
    elif CHAIN_TYPE == 2 or CHAIN_TYPE == 22:
        d1 = refract3d(d0, n1, η_VAL)
    if d1 is None:
        return None, None, None, None
    u2, v2 = intersect3d(x1, d1, P20, P21, P22, False if CHAIN_LENGTH > 1 else True, ignore_check=ignore_check)
    if u2 is None:
        return None, None, None, None
    if CHAIN_LENGTH == 1:
        return u2, v2, None, None
    x2 = P20 + P21 * u2 + P22 * v2
    n2 = N20 + N21 * u2 + N22 * v2
    d1 = x2 - x1
    if CHAIN_TYPE == 11:
        d2 = reflect3d(d1, n2)
    elif CHAIN_TYPE == 22: 
        d2 = refract3d(d1, -n2, 1 / η_VAL)
    if d2 is None:
        return None, None, None, None
    u3, v3 = intersect3d(x2, d2, P30, P31, P32, True, ignore_check=ignore_check)
    if u3 is None:
        return None, None, None, None
    return u2, v2, u3, v3

def ptrace_get_uD(PL, P10, P11, P12, N10, N11, N12, P20, P21, P22, N20, N21, N22, P30, P31, P32, u, v, ignore_check=False):
    return ptrace(PL, P10, P11, P12, N10, N11, N12, P20, P21, P22, N20, N21, N22, P30, P31, P32, u, v, ignore_check)[CHAIN_LENGTH * 2 - 2:CHAIN_LENGTH * 2]

def render_ptrace(PL, P10, P11, P12, N10, N11, N12, P20, P21, P22, N20, N21, N22, P30, P31, P32, res, spp):
    ans = np.zeros((res, res))
    PD0, PD1, PD2 = (P20, P21, P22) if CHAIN_LENGTH == 1 else (P30, P31, P32)
    photon_energy = res ** 2 / spp / norm(cross(PD1, PD2))
    for _ in range(spp):
        u = np.random.rand()
        v = np.random.rand()
        if u + v > 1:
            u, v = 1 - u, 1 - v
        u2, v2, u3, v3 = ptrace(PL, P10, P11, P12, N10, N11, N12, P20, P21, P22, N20, N21, N22, P30, P31, P32, u, v)
        if u2 is None:
            continue
        x1 = P10 + P11 * u + P12 * v
        r = norm(x1 - PL)
        m1 = cross(P11, P12)
        m1 = m1 / norm(m1)
        d1 = x1 - PL
        d1 = d1 / norm(d1)
        cos_theta = abs(d1.dot(m1))
        pdf = 2 * r * r / (cos_theta * norm(cross(P11, P12)) + 1e-9)
        uD, vD = (u2, v2) if CHAIN_LENGTH == 1 else (u3, v3)
        if np.isnan(uD) or np.isnan(vD):
            continue
        if np.isnan(photon_energy / pdf) or np.isinf(photon_energy / pdf):
            continue
        ans[int(vD * res)][int(uD * res)] += photon_energy / pdf
    return ans

def ptrace_finite_diff(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, u, v, du, dv, ignore_check=False): # du = 0 XOR dv = 0
    eps = du + dv
    u_ = u + du
    v_ = v + dv
    u2, v2, u3, v3 = ptrace(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, u, v, ignore_check)
    u2_, v2_, u3_, v3_ = ptrace(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, u_, v_, ignore_check)
    if u2 is None or u2_ is None:
        return None, None
    uD = u2 if CHAIN_LENGTH == 1 else u3
    vD = v2 if CHAIN_LENGTH == 1 else v3
    uD_ = u2_ if CHAIN_LENGTH == 1 else u3_
    vD_ = v2_ if CHAIN_LENGTH == 1 else v3_
    duD = (uD_ - uD) / eps
    dvD = (vD_ - vD) / eps
    return duD, dvD

def ptrace_jacobian_matrix(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, u, v, ignore_check=False):
    eps = 1e-6
    duD_du1, dvD_du1 = ptrace_finite_diff(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, u, v, eps, 0, ignore_check)
    if duD_du1 is None:
        duD_du1, dvD_du1 = ptrace_finite_diff(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, u, v, -eps, 0, ignore_check)
    if duD_du1 is None:
        return None
    duD_dv1, dvD_dv1 = ptrace_finite_diff(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, u, v, 0, eps, ignore_check)
    if dvD_dv1 is None:
        duD_dv1, dvD_dv1 = ptrace_finite_diff(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, u, v, 0, -eps, ignore_check)
    if dvD_dv1 is None:
        return None
    return np.array([[duD_du1, dvD_du1], [duD_dv1, dvD_dv1]])

def eval_path_contribution_extra_factor(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, u, v):
    x1 = p10 + p11 * u + p12 * v
    d0 = x1 - p0
    C = dot(d0, d0)
    pD1 = p21 if CHAIN_LENGTH == 1 else p31
    pD2 = p22 if CHAIN_LENGTH == 1 else p32
    return 1.0 / (C ** 1.5 * norm(cross(p11, p12)) * norm(cross(pD1, pD2)))

def eval_path_contribution(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, u, v):
    J = ptrace_jacobian_matrix(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, u, v)
    if J is None:
        return 0
    return 1.0 / np.abs(J[0][0] * J[1][1] - J[0][1] * J[1][0]) * eval_path_contribution_extra_factor(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, u, v)

def draw_3d_lts_viz(ax, p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32):
    N_SPP_SQRT = 16
    plot_point(ax, p0, facecolor=COLOR2)
    for i in range(N_SPP_SQRT):
        for j in range(N_SPP_SQRT):
            u1, v1 = (i + 0.5) / N_SPP_SQRT, (j + 0.5) / N_SPP_SQRT
            if u1 + v1 > 1: continue
            u2, v2, u3, v3 = ptrace(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, u1, v1)
            if u2 is None:
                continue
            x1 = p10 + u1 * p11 + v1 * p12
            x2 = p20 + u2 * p21 + v2 * p22
            plot_ray_endpoint(ax, p0, x1, COLOR2)
            plot_ray_endpoint(ax, x1, x2, COLOR2)
            plot_point(ax, x1)
            plot_point(ax, x2)
            if CHAIN_LENGTH >= 2:
                x3 = p30 + u3 * p31 + v3 * p32
                plot_ray_endpoint(ax, x2, x3, COLOR2)
                plot_point(ax, x3)

    pD0, pD1, pD2 = (p20, p21, p22) if CHAIN_LENGTH == 1 else (p30, p31, p32)
    plot_triangle(ax, pD0, pD1, pD2, COLOR0S)
    plot_triangle(ax, p10, p11, p12, COLOR0)
    plot_dir(ax, p10, n10, COLOR0)
    plot_dir(ax, p10 + p11, normalize(n10 + n11), COLOR0)
    plot_dir(ax, p10 + p12, normalize(n10 + n12), COLOR0)
    if CHAIN_LENGTH >= 2:
        plot_triangle(ax, p20, p21, p22, COLOR0)
        plot_dir(ax, p20, n20, COLOR0)
        plot_dir(ax, p20 + p21, normalize(n20 + n21), COLOR0)
        plot_dir(ax, p20 + p22, normalize(n20 + n22), COLOR0)
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 2])
    ax.set_zlim([0, 2])
    ax.set_xlabel("x")
