from alias import *
from numba import njit, jit
from interval import *

# @njit
def reflect3d(i, n):
    normal = n / np.linalg.norm(n)
    incident = i / np.linalg.norm(i)
    r = incident - 2 * np.dot(incident, normal) * normal
    return r / np.linalg.norm(r)

# @njit
def refract3d(d0, n1, η):
    d0_n1 = d0.dot(n1)
    if d0_n1 > 0:
        return None
    d0_sqr = d0.dot(d0)
    n1_sqr = n1.dot(n1)
    β = n1_sqr * d0_sqr - η * η * (n1_sqr * d0_sqr - (n1.dot(d0)) ** 2)
    if β < 0:
        return None
    d1t = η * (n1.dot(n1) * d0 - (d0.dot(n1)) * n1)
    d1n = sqrt(β) * n1
    d1 = d1t - d1n
    return d1 / np.linalg.norm(d1)

# @njit
def intersect3d(o, d, p0, p1, p2, rectangle=False, ignore_check=False):
    u = np.cross(d, p2).dot(o - p0)
    v = np.cross(o - p0, p1).dot(d)
    k = np.cross(d, p2).dot(p1)
    u, v = u / k, v / k
    x = p0 + u * p1 + v * p2
    m = np.cross(p1, p2)
    t = np.dot(m, d) * np.dot(m, x - o)
    if (u < 0 or v < 0 or u > 1 or v > 1 or (u + v > 1 and rectangle == False) or t < 0) and ignore_check==False:
        return None, None
    return u, v

def point_triangle_dist_range(pL, p10, p11, p12, u1m, u1M, v1m, v1M):
    rm, rM = 1e19, 0
    tmp_p = p10 + u1m * p11 + v1m * p12
    r = norm(tmp_p - pL)
    rm, rM = min(rm, r), max(rM, r)
    tmp_p = p10 + u1m * p11 + v1M * p12
    r = norm(tmp_p - pL)
    rm, rM = min(rm, r), max(rM, r)
    tmp_p = p10 + u1M * p11 + v1m * p12
    r = norm(tmp_p - pL)
    rm, rM = min(rm, r), max(rM, r)
    tmp_p = p10 + u1M * p11 + v1M * p12
    r = norm(tmp_p - pL)
    rm, rM = min(rm, r), max(rM, r)

    u, v = intersect3d(pL, cross(p11, p12), p10, p11, p12)
    if u is not None:
        u = max(u1m, min(u1M, u))
        v = max(v1m, min(v1M, v))
        tmp_p = p10 + u * p11 + v * p12
        r = norm(tmp_p - pL)
        rm, rM = min(rm, r), max(rM, r)
    return rm, rM

def read_obj(filename, offset=[0, 0, 0], scale=1):
    fp = open(filename, "r")
    fl = fp.readlines()
    v = [[]]
    vn = [[]]
    ans = []
    for s in fl:
        a = s.split()
        if len(a) > 0:
            if a[0] == "v":
                v.append(np.array([float(a[1]), float(a[2]), float(a[3])]))
            elif a[0] == "vn":
                vn.append(np.array([float(a[1]), float(a[2]), float(a[3])]))
            elif a[0] == "f":
                b = a[1:]
                b = [i.split("/") for i in b]
                if len(vn) == 0:
                    print("No normals")
                    exit(0)
                ans.append(
                    [
                        v[int(b[0][0])] * scale + offset,
                        v[int(b[1][0])] * scale + offset,
                        v[int(b[2][0])] * scale + offset,
                        vn[int(b[0][-1])],
                        vn[int(b[1][-1])],
                        vn[int(b[2][-1])],
                    ]
                )
    return np.array(ans) # caveat: three point form, not edge delta form

def random_gen_data():
    p20 = np.array([0, 0, 0])
    p21 = np.array([2, 0, 0])
    p22 = np.array([0, 2, 0])
    p10 = np.random.rand(3) * 1
    p11 = np.random.rand(3)
    p12 = np.random.rand(3)
    n10 = np.random.rand(3) - 0.5
    n11 = np.random.rand(3) - 0.5
    n12 = np.random.rand(3) - 0.5
    p0 = np.random.rand(3) * 2 - 0.5
    n10 = n10 / norm(n10)
    return p0, p10, p11, p12, n10, n11, n12, p20, p21, p22

def random_gen_data_TT():
    while True:
        x = random_gen_data_TT_impl()
        if x is not None:
            return x

def random_gen_data_TT_impl():
    pL = np.random.rand(3) * 2 + np.array([0, 0, 1])
    p10 = np.random.rand(3) * 2
    p11 = np.random.rand(3) * 0.5
    p12 = np.random.rand(3) * 0.5
    p20 = np.random.rand(3) * 2
    p21 = np.random.rand(3)
    p22 = np.random.rand(3)
    n10 = np.random.rand(3) + np.array([-0.5, -0.5, 0])
    n20 = np.random.rand(3) + np.array([-0.5, -0.5, -1])
    if np.abs(dot(n10, normalize(cross(p11, p12)))) < 0.5:
        return None
    if np.abs(dot(n20, normalize(cross(p21, p22)))) < 0.5:
        return None
    n11 = np.zeros(3)
    n12 = np.zeros(3)
    n21 = np.zeros(3)
    n22 = np.zeros(3)
    if SHADING_NORMAL:
        n11 = np.random.rand(3) - 0.5
        n12 = np.random.rand(3) - 0.5
    n11 *= 2
    n12 *= 2
    if np.abs(dot(n10 + n11, normalize(cross(p11, p12)))) < 0.5:
        return None
    if np.abs(dot(n10 + n12, normalize(cross(p11, p12)))) < 0.5:
        return None
    if np.abs(dot(n10 + 0.5 * n11 + 0.5 * n12, normalize(cross(p11, p12)))) < 0.5:
        return None
    if SHADING_NORMAL2:
        n21 = np.random.rand(3) - 0.5
        n22 = np.random.rand(3) - 0.5
    if np.abs(dot(n20 + n21, normalize(cross(p21, p22)))) < 0.5:
        return None
    if np.abs(dot(n20 + n22, normalize(cross(p21, p22)))) < 0.5:
        return None
    if np.abs(dot(n20 + 0.5 * n21 + 0.5 * n22, normalize(cross(p21, p22)))) < 0.5:
        return None
    p30, p31, p32 = array([0.0, 0, 0]), array([3, 0.0, 0]), array([0.0, 3, 0])
    return pL, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32


def create_plt3d_space():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    return ax

def plot_triangle(ax, p0, e1, e2, color='b'):
    p1 = p0 + e1
    p2 = p0 + e2
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=color)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color)
    ax.plot([p2[0], p0[0]], [p2[1], p0[1]], [p2[2], p0[2]], color=color)
    ax.plot_trisurf([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]], [p0[2], p1[2], p2[2]], color=color, alpha=0.34)

def plot_ray_endpoint(ax, p0, p1, color='r'):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=color, alpha=0.5)
    ax.quiver(p0[0], p0[1], p0[2], p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2], color=color, length=0.5, alpha=0.5)

def plot_ray_dir(ax, p0, d, color='r'):
    ax.plot([p0[0], p0[0]+d[0]*1], [p0[1], p0[1]+d[1]*1], [p0[2], p0[2]+d[2]*1], color=color, alpha=0.5)
    ax.quiver(p0[0], p0[1], p0[2], d[0], d[1], d[2], color=color, length=0.5, alpha=0.5)

def plot_dir(ax, p0, d, color='r'):
    ax.quiver(p0[0], p0[1], p0[2], d[0], d[1], d[2], color=color, length=0.5)

def plot_point(ax, p, radius=15, facecolor='w', edgecolor='k'):
    ax.scatter(p[0], p[1], p[2], s=radius, c=facecolor, edgecolors=edgecolor)

def plt_imshow_with_title(subplot_spec, title, data, vmin, vmax, cmap):
    plt.subplot(subplot_spec)
    plt.title(title)
    plt.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar()
    plt.axis("off")
    

@njit(cache=True)
def compute_easy_approx_for_sqrt(l, r):
    a = (1 / np.sqrt(l) + 1 / np.sqrt(r)) / 4
    Δξ = (1 / np.sqrt(l) - 1 / np.sqrt(r)) / 4  
    mid = l
    b = np.sqrt(mid) - a * mid
    Δξ1 = np.sqrt(r) - np.sqrt(l) - (r - l) / np.sqrt(r) / 2
    return a, b, Δξ, Δξ1


@njit(cache=True)
def compute_line_approx_for_sqrt(l, r):
    a = np.sqrt(r) - np.sqrt(l)
    a /= r - l
    b = np.sqrt(l) - a * l
    xm = 1 / 4 /a / a
    Δξ = 0.5 * (np.sqrt(xm) - a * xm - b)
    b += Δξ
    return a, b, Δξ


def validity(x, ref):
    mask_ref = np.where(ref > 1e-2, 1, 0)
    if np.sum(mask_ref) == 0:
        return 1
    return np.sum(np.minimum(x / ref, 1) * mask_ref) / (np.sum(mask_ref) + 1e-9)

def tightness(x, ref):
    mask_x = np.where(x > 1e-2, 1, 0)
    if np.sum(mask_x) == 0:
        return 1
    return np.sum(np.minimum(ref / x, 1) * mask_x) / (np.sum(mask_x) + 1e-9)

def position_tightness(x, ref):
    mask_x = np.where(x > 1e-2, 1, 0)
    mask_ref = np.where(ref > 1e-2, 1, 0)
    if np.sum(mask_x) == 0:
        return 1
    return np.sum((1 - np.maximum(mask_x - mask_ref, 0)) * mask_x) / (np.sum(mask_x) + 1e-9)

def irradiance_tightness(x, ref):
    mask_ref = np.where(ref > 1e-2, 1, 0)
    if np.sum(mask_ref) == 0:
        return 1
    return np.sum(np.minimum(ref / x, 1) * mask_ref) / (np.sum(mask_ref) + 1e-9)

def bounding_metrics(x, ref):
    return np.array([validity(x, ref), tightness(x, ref), position_tightness(x, ref), irradiance_tightness(x, ref)])

def show_bpoly(a):
        vm = np.max(np.abs(a))
        for _ in range(2):
            for i in range(a.shape[2]):
                for j in range(a.shape[3]):
                    plt.subplot(2 * a.shape[2], a.shape[3], (i * 2 + _) * a.shape[3] + j + 1)
                    plt.imshow(a[_, 0, i, j], cmap='coolwarm', vmin=-vm, vmax=vm)
        plt.show()

@njit(cache=True, fastmath=True)
def batch_norm(vec, batch_size):
    ans = np.zeros(batch_size, dtype=np.float64)
    for i in range(batch_size):
        ans[i] = (vec[0, i] ** 2 + vec[1, i] ** 2 + vec[2, i] ** 2) ** 0.5
    return ans

@njit(cache=True, fastmath=True)
def batch_cross(vec1, vec2, batch_size):
    ans = np.zeros((3, batch_size), dtype=np.float64)
    for i in range(batch_size):
        ans[0, i] = vec1[1, i] * vec2[2, i] - vec1[2, i] * vec2[1, i]
        ans[1, i] = vec1[2, i] * vec2[0, i] - vec1[0, i] * vec2[2, i]
        ans[2, i] = vec1[0, i] * vec2[1, i] - vec1[1, i] * vec2[0, i]
    return ans

@njit(cache=True, fastmath=True)
def batch_dot(vec1, vec2, batch_size):
    ans = np.zeros(batch_size, dtype=np.float64)
    for i in range(batch_size):
        ans[i] = vec1[0, i] * vec2[0, i] + vec1[1, i] * vec2[1, i] + vec1[2, i] * vec2[2, i]
    return ans

@njit(cache=True, fastmath=True)
def batch_normalize(vec, batch_size):
    ans = np.zeros((3, batch_size), dtype=np.float64)
    for i in range(batch_size):
        norm_val = (vec[0, i] ** 2 + vec[1, i] ** 2 + vec[2, i] ** 2) ** 0.5
        if norm_val > 1e-9:
            ans[0, i] = vec[0, i] / norm_val
            ans[1, i] = vec[1, i] / norm_val
            ans[2, i] = vec[2, i] / norm_val
    return ans

@njit(cache=True, fastmath=True)
def batch_intersect3d(o, d, p0, p1, p2, batch_size):
    u = batch_dot(batch_cross(d, p2, batch_size), o - p0, batch_size)
    v = batch_dot(batch_cross(o - p0, p1, batch_size), d, batch_size)
    k = batch_dot(batch_cross(d, p2, batch_size), p1, batch_size)
    u, v = u / k, v / k
    x = p0 + u * p1 + v * p2
    m = batch_cross(p1, p2, batch_size)
    t = batch_dot(m, d, batch_size) * batch_dot(m, x - o, batch_size)
    return u, v, t

@njit(cache=True, fastmath=True)
def batch_point_triangle_dist_range(pL, p10, p11, p12, u1m, u1M, v1m, v1M, batch_size):
    rm, rM = np.ones(batch_size, dtype=np.float64) * 1e19, np.zeros(batch_size, dtype=np.float64)
    tmp_p = p10 + u1m * p11 + v1m * p12
    r = batch_norm(tmp_p - pL, batch_size)
    rm, rM = np.minimum(rm, r), np.maximum(rM, r)
    tmp_p = p10 + u1m * p11 + v1M * p12
    r = batch_norm(tmp_p - pL, batch_size)
    rm, rM = np.minimum(rm, r), np.maximum(rM, r)
    tmp_p = p10 + u1M * p11 + v1m * p12
    r = batch_norm(tmp_p - pL, batch_size)
    rm, rM = np.minimum(rm, r), np.maximum(rM, r)
    tmp_p = p10 + u1M * p11 + v1M * p12
    r = batch_norm(tmp_p - pL, batch_size)
    rm, rM = np.minimum(rm, r), np.maximum(rM, r)

    u, v, t = batch_intersect3d(pL, batch_cross(p11, p12, batch_size), p10, p11, p12, batch_size)

    tmp_p = p10 + u * p11 + v * p12
    for i in range(batch_size):
        if u[i] < 0 or v[i] < 0 or u[i] > 1 or v[i] > 1 or u[i] + v[i] > 1 or t[i] < 0:
            continue
        
        u[i] = max(u1m[i], min(u1M[i], u[i]))
        v[i] = max(v1m[i], min(v1M[i], v[i]))
        r = norm(tmp_p[:, i] - pL[:, i])
        rm[i], rM[i] = min(rm[i], r), max(rM[i], r)

    return rm, rM

if __name__ == '__main__':
    batch_size = 64
    pL = np.random.rand(3, batch_size)
    p10 = np.random.rand(3, batch_size)
    p11 = np.random.rand(3, batch_size)
    p12 = np.random.rand(3, batch_size)
    u1m = np.random.rand(batch_size)
    u1M = np.random.rand(batch_size)
    v1m = np.random.rand(batch_size)
    v1M = np.random.rand(batch_size)

    batch_point_triangle_dist_range(pL, p10, p11, p12, u1m, u1M, v1m, v1M, batch_size)
    batch_point_triangle_dist_range(pL, p10, p11, p12, u1m, u1M, v1m, v1M, batch_size)
    
    t = time.perf_counter()    
    for _ in range(1000):
        batch_point_triangle_dist_range(pL, p10, p11, p12, u1m, u1M, v1m, v1M, batch_size)
    print(f"Time: {time.perf_counter() - t:.5f}s")


    