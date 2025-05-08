import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from bernstein import *


def plot_segment(p0, p1, n0, n1):
    plt.plot([p0[0], p0[0] + p1[0]], [p0[1], p0[1] + p1[1]], color='g')
    plt.quiver(p0[0], p0[1], n0[0] * 0.02, n0[1] * 0.02, color='b', scale=0.1)
    plt.quiver(p0[0] + p1[0], p0[1] + p1[1], (n0[0] + n1[0]) * 0.02, (n0[1] + n1[1]) * 0.02, color='m', scale=0.1)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)


def reflect(i, n):
    n_unit = n / np.linalg.norm(n)
    i_unit = i / np.linalg.norm(i)
    r = i - 2 * np.dot(i_unit, n_unit) * n_unit
    return r / np.linalg.norm(r)


def intersect(o, d, p0, p1):
    if np.cross(p0 - o, p1) / np.cross(d, p1) > 0:
        u = np.cross(o - p0, d) / np.cross(p1, d)
        if u >= 0 and u <= 1:
            return u
    return None


def draw_rays(pL, p0, p1, n0, n1):
    succ = False
    N_RAYS = 10000
    for _ in range(N_RAYS):
        t = _ * np.pi * 2 / N_RAYS
        i = np.array([np.cos(t), np.sin(t)])
        u = intersect(pL, i, p0, p1)
        if u is None or i.dot(n0 + n1 * u) > 0 or np.cross(i, p1) * np.cross(n0 + n1 * u, p1) > 0:
            continue
        r = reflect(i, n0 + n1 * u)
        if r.dot(n0 + n1 * u) < 0 or np.cross(r, p1) * np.cross(n0 + n1 * u, p1) < 0:
            continue
        plt.plot([pL[0], p0[0] + u * p1[0]], [pL[1], p0[1] + u * p1[1]], 'k', lw=0.5, alpha=0.05)
        plt.plot([p0[0] + u * p1[0], p0[0] + u * p1[0] + r[0] * 10], [p0[1] + u * p1[1], p0[1] + u * p1[1] + r[1] * 10], 'r', lw=0.5, alpha=0.05)
        succ = True
    return succ
