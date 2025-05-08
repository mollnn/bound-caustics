from matplotlib import pyplot as plt
import numpy as np

ref_bins = [0 for i in range(10000)]
ref_bins2 = [0 for i in range(10000)]

def compute_ref(pL, p0, p1, n0, n1, output_u1=0):
    global ref_bins
    global ref_bins2
    p10x, p10y, p11x, p11y, n10x, n10y, n11x, n11y = p0[0], p0[1], p1[0], p1[1], n0[0], n0[1], n1[0], n1[1]
    x0x, x0y, p20x, p20y, p21x, p21y = pL[0], pL[1], -1, -1, 2, 0

    u = np.linspace(0, 1, 8192)
    u2 = (-((n10x + n11x*u)*(-2*(n10x + n11x*u)*(p10x + p11x*u - x0x) - 2*(n10y + n11y*u)*(p10y + p11y*u - x0y)) + ((n10x + n11x*u)**2 + (n10y + n11y*u)**2)*(p10x + p11x*u - x0x))*(p10y + p11y*u - p20y) + ((n10y + n11y*u)*(-2*(n10x + n11x*u)*(p10x + p11x*u - x0x) - 2*(n10y + n11y*u)*(p10y + p11y*u - x0y)) + ((n10x + n11x*u)**2 + (n10y + n11y*u)**2)*(p10y + p11y*u - x0y)) *(p10x + p11x*u - p20x))/(p21x*((n10y + n11y*u)*(-2*(n10x + n11x*u)*(p10x + p11x*u - x0x) - 2*(n10y + n11y*u)*(p10y + p11y*u - x0y)) + ((n10x + n11x*u)**2 + (n10y + n11y*u)**2)*(p10y + p11y*u - x0y)) - p21y*((n10x + n11x*u)*(-2*(n10x + n11x*u)*(p10x + p11x*u - x0x) - 2*(n10y + n11y*u)*(p10y + p11y*u - x0y)) + ((n10x + n11x*u)**2 + (n10y + n11y*u)**2)*(p10x + p11x*u - x0x)))
    du2_du1 = (p11x*((n10y + n11y*u)*(-2*(n10x + n11x*u)*(p10x + p11x*u - x0x) - 2*(n10y + n11y*u)*(p10y + p11y*u - x0y)) + ((n10x + n11x*u)**2 + (n10y + n11y*u)**2)*(p10y + p11y*u - x0y)) - p11y*((n10x + n11x*u)*(-2*(n10x + n11x*u)*(p10x + p11x*u - x0x) - 2*(n10y + n11y*u)*(p10y + p11y*u - x0y)) + ((n10x + n11x*u)**2 + (n10y + n11y*u)**2)*(p10x + p11x*u - x0x)) + (p10x + p11x*u - p20x)*(n11y*(-2*(n10x + n11x*u)*(p10x + p11x*u - x0x) - 2*(n10y + n11y*u)*(p10y + p11y*u - x0y)) + p11y*((n10x + n11x*u)**2 + (n10y + n11y*u)**2) + (n10y + n11y*u)*(-2*n11x*(p10x + p11x*u - x0x) - 2*n11y*(p10y + p11y*u - x0y) - 2*p11x*(n10x + n11x*u) - 2*p11y*(n10y + n11y*u)) + (2*n11x*(n10x + n11x*u) + 2*n11y*(n10y + n11y*u))*(p10y + p11y*u - x0y)) - (p10y + p11y*u - p20y)*(n11x*(-2*(n10x + n11x*u)*(p10x + p11x*u - x0x) - 2*(n10y + n11y*u)*(p10y + p11y*u - x0y)) + p11x*((n10x + n11x*u)**2 + (n10y + n11y*u)**2) + (n10x + n11x*u)*(-2*n11x*(p10x + p11x*u - x0x) - 2*n11y*(p10y + p11y*u - x0y) - 2*p11x*(n10x + n11x*u) - 2*p11y*(n10y + n11y*u)) + (2*n11x*(n10x + n11x*u) + 2*n11y*(n10y + n11y*u))*(p10x + p11x*u - x0x)))/(p21x*((n10y + n11y*u)*(-2*(n10x + n11x*u)*(p10x + p11x*u - x0x) - 2*(n10y + n11y*u)*(p10y + p11y*u - x0y)) + ((n10x + n11x*u)**2 + (n10y + n11y*u)**2)*(p10y + p11y*u - x0y)) - p21y*((n10x + n11x*u)*(-2*(n10x + n11x*u)*(p10x + p11x*u - x0x) - 2*(n10y + n11y*u)*(p10y + p11y*u - x0y)) + ((n10x + n11x*u)**2 + (n10y + n11y*u)**2)*(p10x + p11x*u - x0x))) + (-p21x*(n11y*(-2*(n10x + n11x*u)*(p10x + p11x*u - x0x) - 2*(n10y + n11y*u)*(p10y + p11y*u - x0y)) + p11y*((n10x + n11x*u)**2 + (n10y + n11y*u)**2) + (n10y + n11y*u)*(-2*n11x*(p10x + p11x*u - x0x) - 2*n11y*(p10y + p11y*u - x0y) - 2*p11x*(n10x + n11x*u) - 2*p11y*(n10y + n11y*u)) + (2*n11x*(n10x + n11x*u) + 2*n11y*(n10y + n11y*u))*(p10y + p11y*u - x0y)) + p21y*(n11x*(-2*(n10x + n11x*u)*(p10x + p11x*u - x0x) - 2*(n10y + n11y*u)*(p10y + p11y*u - x0y)) + p11x*((n10x + n11x*u)**2 + (n10y + n11y*u)**2) + (n10x + n11x*u)*(-2*n11x*(p10x + p11x*u - x0x) - 2*n11y*(p10y + p11y*u - x0y) - 2*p11x*(n10x + n11x*u) - 2*p11y*(n10y + n11y*u)) + (2*n11x*(n10x + n11x*u) + 2*n11y*(n10y + n11y*u))*(p10x + p11x*u - x0x)))*(-((n10x + n11x*u)*(-2*(n10x + n11x*u)*(p10x + p11x*u - x0x) - 2*(n10y + n11y*u)*(p10y + p11y*u - x0y)) + ((n10x + n11x*u)**2 + (n10y + n11y*u)**2)*(p10x + p11x*u - x0x))*(p10y + p11y*u - p20y) + ((n10y + n11y*u)*(-2*(n10x + n11x*u)*(p10x + p11x*u - x0x) - 2*(n10y + n11y*u)*(p10y + p11y*u - x0y)) + ((n10x + n11x*u)**2 + (n10y + n11y*u)**2)*(p10y + p11y*u - x0y))*(p10x + p11x*u - p20x))/(p21x*((n10y + n11y*u)*(-2*(n10x + n11x*u)*(p10x + p11x*u - x0x) - 2*(n10y + n11y*u)*(p10y + p11y*u - x0y)) + ((n10x + n11x*u)**2 + (n10y + n11y*u)**2)*(p10y + p11y*u - x0y)) - p21y*((n10x + n11x*u)*(-2*(n10x + n11x*u)*(p10x + p11x*u - x0x) - 2*(n10y + n11y*u)*(p10y + p11y*u - x0y)) + ((n10x + n11x*u)**2 + (n10y + n11y*u)**2)*(p10x + p11x*u - x0x)))**2

    # plt.plot(u, u2, c=(200 / 255, 36 / 255, 35 / 255), label="xD")
    if output_u1 == 1:
        plt.scatter(u[(u2 > 0) & (u2 < 1)], u2[(u2 > 0) & (u2 < 1)], s=0.05)
        return 0, 0
    if output_u1 == 2:
        plt.scatter(u[(u2 > 0) & (u2 < 1)], np.abs(1 / du2_du1)[(u2 > 0) & (u2 < 1)], s=0.05)
        return 0, 0

    plt.scatter(u2[(u2 > 0) & (u2 < 1)], np.abs(1 / du2_du1)[(u2 > 0) & (u2 < 1)], s=0.05)
    # plt.legend()
    for i, j in zip(u2[(u2 > 0) & (u2 < 1)].tolist(), (1 / du2_du1)[(u2 > 0) & (u2 < 1)].tolist()):
        if j > 0:
            ref_bins[int(i * 10000)] = max(ref_bins[int(i * 10000)], abs(j))
        else:
            ref_bins2[int(i * 10000)] = max(ref_bins2[int(i * 10000)], abs(j))
    return min(1 / du2_du1), max(1 / du2_du1)
