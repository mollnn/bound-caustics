import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from bernstein import *
bound_bins = [0 for i in range(10000)]


def compute_bound(pL, p0, p1, n0, n1, left=0, right=1):
    u1 = Symbol('u') * (right - left) + left + 0.00000001
    p10x, p10y, p11x, p11y, p20x, p20y, p21x, p21y, n10x, n10y, n11x, n11y, x0x, x0y = symbols('p10x p10y p11x p11y p20x p20y p21x p21y n10x n10y n11x n11y x0x x0y')
    value_dict = {p10x: p0[0], p10y: p0[1], p11x: p1[0], p11y: p1[1], p20x: -1, p20y: -1, p21x: 2, p21y: 0, n10x: n0[0], n10y: n0[1], n11x: n1[0], n11y: n1[1], x0x: pL[0], x0y: pL[1]}

    x1x, x1y, n1x, n1y = p10x + p11x * u1, p10y + p11y * u1, n10x + n11x * u1, n10y + n11y * u1
    d0x, d0y = x1x - x0x, x1y - x0y
    d1x, d1y = -2 * (d0x * n1x + d0y * n1y) * n1x + (n1x * n1x + n1y * n1y) * d0x, -2 * (d0x * n1x + d0y * n1y) * n1y + (n1x * n1x + n1y * n1y) * d0y

    u2hat = (x1x - p20x) * d1y - (x1y - p20y) * d1x
    kappa2 = p21x * d1y - p21y * d1x
    u2hat = simplify(u2hat.subs(value_dict))
    kappa2 = simplify(kappa2.subs(value_dict))
    u2_numer_bern = coef_a2b(Poly(u2hat).coeffs()[::-1])
    u2_denom_bern = coef_a2b(Poly(kappa2).coeffs()[::-1], len(u2_numer_bern) - 1)

    u2_frac_bern = [i / j for (i, j) in zip(u2_numer_bern, u2_denom_bern)]

    u2u1_numer = diff(u2hat) * kappa2 - diff(kappa2) * u2hat
    u2u1_denom = kappa2 * kappa2

    u2u1_numer, u2u1_denom = u2u1_denom, u2u1_numer  

    u2u1_numer_bern = coef_a2b(Poly(u2u1_numer).coeffs()[::-1])
    u2u1_denom_bern = coef_a2b(Poly(u2u1_denom).coeffs()[::-1])
    u2u1_frac_bern = [i / j for (i, j) in zip(u2u1_numer_bern, u2u1_denom_bern)]
    J = right - left
    return min(u2u1_frac_bern) * J, max(u2u1_frac_bern) * J, min(u2u1_denom_bern) / J, max(u2u1_denom_bern) / J,  min(u2_frac_bern), max(u2_frac_bern), min(u2_denom_bern), max(u2_denom_bern)


def compute_the_bound_recursive(pL, p0, p1, n0, n1, left=0, right=1, show=0):
    mn, mx, dmn, dmx, p_mn, p_mx, p_dmn, p_dmx = compute_bound(pL, p0, p1, n0, n1, left, right)
    print("[%.3f, %.3f]: range [%.3f, %.3f], denom [%.3f, %.3f], position range [%.3f, %.3f], denom [%.3f, %.3f]" %
          (left, right, mn, mx, dmn, dmx, p_mn, p_mx, p_dmn, p_dmx))
    amx = max(abs(mn), abs(mx))
    amn = min(abs(mn), abs(mx))
    if p_dmn * p_dmx > 0 and (p_mx < 0 or p_mn > 1):
        return
    if (dmn * dmx < 0 or (amx / amn > 2 and amx > 0.1) or p_dmn * p_dmx < 0) and right - left > 1e-1:
        mid = (left + right) / 2
        compute_the_bound_recursive(pL, p0, p1, n0, n1, left, mid, show)
        compute_the_bound_recursive(pL, p0, p1, n0, n1, mid, right, show)
    else:
        if dmn * dmx > 0 and p_dmn * p_dmx > 0:
            amx = float(amx)
            amn = float(amn)
        else:
            amx = 1e9
            amn = 0
        p_mn = float(p_mn)
        p_mx = float(p_mx)
        p_mn = max(0.0, p_mn)
        p_mx = min(1.0, p_mx)
        if show==0:
            plt.fill_between([p_mn, p_mx], 0.0, [amx, amx], facecolor=(154/255, 200/255, 219/255), alpha=0.4)
            for i in range(int(p_mn * 10000), int(p_mx * 10000 + 1)):
                if 0 <= i < 10000:
                    bound_bins[i] = max(bound_bins[i], amx)
        elif show==1:
            plt.fill_between([left, right], [p_mn, p_mn], [p_mx, p_mx], facecolor=(154/255, 200/255, 219/255), alpha=0.4)
        elif show==2:
            plt.fill_between([left, right], [amn, amn], [amx, amx], facecolor=(154/255, 200/255, 219/255), alpha=0.4)
