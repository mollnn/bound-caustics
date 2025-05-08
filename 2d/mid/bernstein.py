import math

import numpy as np

def factorial(n):
    return math.factorial(n)

def C(n, m):
    return factorial(n) / (factorial(m) * factorial(n - m))


def coef_a2b(a, force_d = None):
    d = len(a) - 1
    if force_d is not None:
        d = force_d
    b = []
    for j in range(d + 1):
        tmp = 0
        for i in range(min(j + 1, len(a))):
            tmp += C(j, i) / C(d, i) * a[i]
        b.append(tmp)
    return b


def eval_poly_a(a, x):
    n = len(a)
    ans = 0
    for i in range(n):
        ans += a[i] * x**i
    return ans


def eval_bernstein(d, j, x):
    return C(d, j) * x**j * (1 - x) ** (d - j)


def eval_poly_b(b, x):
    d = len(b) - 1
    ans = 0
    for j in range(d + 1):
        ans += b[j] * eval_bernstein(d, j, x)
    return ans


def compute_iU(n):
    u = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1):
            u[i][j] = C(i, j) / C(n - 1, j)
    return u

def coef_A2B2d(a):
    n = len(a)
    iU = compute_iU(n)
    print(iU)
    b = np.transpose(iU @ np.transpose(iU @ a))
    return b

def eval_poly_a2d(a, x, y):
    d = len(a) - 1
    ans = 0
    for i in range(d + 1):
        for j in range(d + 1):
            ans += a[i][j] * x ** i * y ** j
    return ans

def eval_poly_b2d(b, x, y):
    d = len(b) - 1
    ans = 0
    for i in range(d + 1):
        for j in range(d + 1):
            ans += b[i][j] * eval_bernstein(d, i, x) * eval_bernstein(d, j, y)
    return ans

from matplotlib import pyplot as plt
def plot_bernstein(b, title, color=None, legend=None):
    print(title, b)
    d = len(b) - 1
    x_values = np.linspace(0, 1, 100)
    y_values = [eval_poly_b(b, x) for x in x_values]
    control_x = np.linspace(0, 1, d + 1)
    control_y = b
    if title != 'u2' and title != 'du1/du2':
        plt.plot(x_values, y_values, c=color)
    plt.scatter(control_x, control_y, label=legend, c=color)
    if legend != None:
        plt.legend()
    plt.title(title)
    plt.xlabel('u1')
