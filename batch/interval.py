from alias import *

@jitclass([('l', float64[:]), ('r', float64[:])])
class BatchInterval1D:
    def __init__(self, l, r):
        batch_size = len(l)
        self.l = np.zeros(batch_size, dtype=np.float64)
        self.r = np.zeros(batch_size, dtype=np.float64)
        for i in range(batch_size):
            if l[i] < r[i]:
                self.l[i] = l[i]
                self.r[i] = r[i]
            else:
                self.l[i] = r[i]
                self.r[i] = l[i]

class Interval1D:
    def __init__(self, _l, _r = None):
        if _r is None:
            l, r = _l
        else:
            l, r = _l, _r
        if l < r:
            self.l = l
            self.r = r
        else:
            self.l = r
            self.r = l

    def __repr__(self):
        return f"Interval1D({self.l:.2f}, {self.r:.2f})"
    
    def __str__(self):
        return f"[{self.l:.2f}, {self.r:.2f}]"
    
    def __add__(self, other):
        return Interval1D(self.l + other.l, self.r + other.r)
    
    def __sub__(self, other):
        return Interval1D(self.l - other.r, self.r - other.l)
    
    def __mul__(self, other):
        return Interval1D(min(self.l * other.l, self.l * other.r, self.r * other.l, self.r * other.r), max(self.l * other.l, self.l * other.r, self.r * other.l, self.r * other.r))

    def __truediv__(self, other):
        return Interval1D(min(self.l / other.l, self.l / other.r, self.r / other.l, self.r / other.r), max(self.l / other.l, self.l / other.r, self.r / other.l, self.r / other.r))
    
    def contain(self, x):
        return self.l <= x <= self.r
    
    def sqrt(self):
        return Interval1D(np.sqrt(self.l), np.sqrt(self.r))
    
    def sqr(self):
        mn = min(self.l * self.l, self.r * self.r)
        mx = max(self.l * self.l, self.r * self.r)
        if self.l <= 0 and self.r >= 0:
            mn = 1e-99
        return Interval1D(mn, mx)
    
class Interval3D:
    def __init__(self, x, y = None, z = None):
        if y is None:
            x, y, z = x
        if x.__class__ == Interval1D:
            self.x = x
            self.y = y
            self.z = z
        elif np.array(x).ndim == 0:
            self.x = Interval1D(x, x)
            self.y = Interval1D(y, y)
            self.z = Interval1D(z, z)
        else:
            self.x = Interval1D(x)
            self.y = Interval1D(y)
            self.z = Interval1D(z)

    def __repr__(self):
        return f"Interval3D({self.x}, {self.y}, {self.z})"
    
    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"
    
    def __add__(self, other):
        return Interval3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Interval3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, other):
        if other.__class__ == Interval1D:
            return Interval3D(self.x * other, self.y * other, self.z * other)
        else:
            return Interval3D(self.x * other.x, self.y * other.y, self.z * other.z)
    
    def __truediv__(self, other):
        if other.__class__ == Interval1D:
            return Interval3D(self.x / other, self.y / other, self.z / other)
        else:
            return Interval3D(self.x / other.x, self.y / other.y, self.z / other.z)
    
    def contain(self, p):
        return self.x.contain(p[0]) and self.y.contain(p[1]) and self.z.contain(p[2])

    def contain0(self):
        return self.x.contain(0) and self.y.contain(0) and self.z.contain(0)

    def norm(self):
        return (self.x.sqr() + self.y.sqr() + self.z.sqr()).sqrt()

    def normalized(self):
        return self / self.norm()
    
    def union(self, other):
        return Interval3D((min(self.x.l, other.x.l), max(self.x.r, other.x.r)), 
                          (min(self.y.l, other.y.l), max(self.y.r, other.y.r)), 
                          (min(self.z.l, other.z.l), max(self.z.r, other.z.r)))

@njit(fastmath=True)
def interval_nb_add(l1, l2, r1, r2):
    return l1 + l2, r1 + r2

@njit(fastmath=True)
def interval_nb_sub(l1, l2, r1, r2):
    return l1 - r2, r1 - l2

@njit(fastmath=True)
def interval_nb_mul(l1, l2, r1, r2):
    return min(l1 * l2, l1 * r2, r1 * l2, r1 * r2), max(l1 * l2, l1 * r2, r1 * l2, r1 * r2)

@njit(fastmath=True)
def interval3D_nb_cross(Alx, Aly, Alz, Arx, Ary, Arz, Blx, Bly, Blz, Brx, Bry, Brz):   # sign may be wrong
    Iyz = interval_nb_mul(Aly, Blz, Ary, Brz)
    Izy = interval_nb_mul(Bly, Alz, Bry, Arz)
    Iyx = interval_nb_mul(Aly, Blx, Ary, Brx)
    Ixy = interval_nb_mul(Bly, Alx, Bry, Arx)
    Ixz = interval_nb_mul(Alx, Blz, Arx, Brz)
    Izx = interval_nb_mul(Blx, Alz, Brx, Arz)
    Ix = interval_nb_sub(Iyz[0], Izy[0], Iyz[1], Izy[1])
    Iy = interval_nb_sub(Izx[0], Ixz[0], Izx[1], Ixz[1])
    Iz = interval_nb_sub(Ixy[0], Iyx[0], Ixy[1], Iyx[1])
    return Ix[0], Iy[0], Iz[0], Ix[1], Iy[1], Iz[1]

@njit(fastmath=True)
def interval_nb_truediv(l1, l2, r1, r2):
    return min(l1 / l2, l1 / r2, r1 / l2, r1 / r2), max(l1 / l2, l1 / r2, r1 / l2, r1 / r2)

@njit(fastmath=True)
def interval_nb_contain(l, r, x):
    return l <= x <= r

@njit(fastmath=True)
def interval_nb_sqrt(l, r):
    return np.sqrt(l), np.sqrt(r)

@njit(fastmath=True)
def interval_nb_sqr(l, r):
    mn = min(l * l, r * r)
    mx = max(l * l, r * r)
    if l <= 0 and r >= 0:
        mn = 1e-99
    return mn, mx

@njit(fastmath=True)
def interval3D_nb_add(l1, l2, l3, r1, r2, r3, l4, l5, l6, r4, r5, r6):
    return l1 + l4, l2 + l5, l3 + l6, r1 + r4, r2 + r5, r3 + r6

@njit(fastmath=True)
def interval3D_nb_sub(l1, l2, l3, r1, r2, r3, l4, l5, l6, r4, r5, r6):
    return l1 - r4, l2 - r5, l3 - r6, r1 - l4, r2 - l5, r3 - l6

@njit(fastmath=True)
def interval3D_nb_mul(l1, l2, l3, r1, r2, r3, l4, l5, l6, r4, r5, r6):
    return min(l1 * l4, l1 * r4, r1 * l4, r1 * r4), min(l2 * l5, l2 * r5, r2 * l5, r2 * r5), min(l3 * l6, l3 * r6, r3 * l6, r3 * r6), max(l1 * l4, l1 * r4, r1 * l4, r1 * r4), max(l2 * l5, l2 * r5, r2 * l5, r2 * r5), max(l3 * l6, l3 * r6, r3 * l6, r3 * r6)

@njit(fastmath=True)
def interval3D_nb_truediv(l1, l2, l3, r1, r2, r3, l4, l5, l6, r4, r5, r6):
    return min(l1 / l4, l1 / r4, r1 / l4, r1 / r4), min(l2 / l5, l2 / r5, r2 / l5, r2 / r5), min(l3 / l6, l3 / r6, r3 / l6, r3 / r6), max(l1 / l4, l1 / r4, r1 / l4, r1 / r4), max(l2 / l5, l2 / r5, r2 / l5, r2 / r5), max(l3 / l6, l3 / r6, r3 / l6, r3 / r6)

@njit(fastmath=True)
def interval3D_nb_contain(l1, l2, l3, r1, r2, r3, p0, p1, p2):
    return l1 <= p0 <= r1 and l2 <= p1 <= r2 and l3 <= p2 <= r3

@njit(fastmath=True)
def interval3D_nb_contain0(l1, l2, l3, r1, r2, r3):
    return l1 <= 0 <= r1 and l2 <= 0 <= r2 and l3 <= 0 <= r3

@njit(fastmath=True)
def interval3D_nb_norm(l1, l2, l3, r1, r2, r3):
    a = interval_nb_sqr(l1, r1)
    b = interval_nb_sqr(l2, r2)
    c = interval_nb_sqr(l3, r3)
    return interval_nb_sqrt(a[0] + b[0] + c[0], a[1] + b[1] + c[1])

@njit(fastmath=True)
def interval3D_nb_normalized(l1, l2, l3, r1, r2, r3):
    norm = interval3D_nb_norm(l1, l2, l3, r1, r2, r3)
    return interval3D_nb_truediv(l1, l2, l3, r1, r2, r3, norm[0], norm[0], norm[0], norm[1], norm[1], norm[1])

@njit(fastmath=True)
def interval3D_nb_union(l1, l2, l3, r1, r2, r3, l4, l5, l6, r4, r5, r6):
    return min(l1, l4), min(l2, l5), min(l3, l6), max(r1, r4), max(r2, r5), max(r3, r6)
                         
@njit(fastmath=True)
def interval_direction_nb(l1, l2, l3, r1, r2, r3, l4, l5, l6, r4, r5, r6):
    a = interval3D_nb_sub(l4, l5, l6, r4, r5, r6, l1, l2, l3, r1, r2, r3)
    return interval3D_nb_normalized(a[0], a[1], a[2], a[3], a[4], a[5])

@njit(fastmath=True)
def interval_halfvector_nb(l1, l2, l3, r1, r2, r3, l4, l5, l6, r4, r5, r6, eta1):
    a = interval3D_nb_mul(l4, l5, l6, r4, r5, r6, eta1, eta1, eta1, eta1, eta1, eta1) 
    b = interval3D_nb_sub(a[0], a[1], a[2], a[3], a[4], a[5], l1, l2, l3, r1, r2, r3)
    return interval3D_nb_normalized(b[0], b[1], b[2], b[3], b[4], b[5])

@njit(fastmath=True)
def interval_position_nb(l1, l2, l3, r1, r2, r3, l4, l5, l6, r4, r5, r6, l7, l8, l9, r7, r8, r9):
    p1 = interval3D_nb_add(l1, l2, l3, r1, r2, r3, l4, l5, l6, r4, r5, r6)
    p2 = interval3D_nb_add(l1, l2, l3, r1, r2, r3, l7, l8, l9, r7, r8, r9)
    u = interval3D_nb_union(p1[0], p1[1], p1[2], p1[3], p1[4], p1[5], p2[0], p2[1], p2[2], p2[3], p2[4], p2[5])
    return interval3D_nb_union(l1, l2, l3, r1, r2, r3, u[0], u[1], u[2], u[3], u[4], u[5])

@njit(fastmath=True)
def interval_normal_nb(l1, l2, l3, r1, r2, r3, l4, l5, l6, r4, r5, r6, l7, l8, l9, r7, r8, r9):
    a = interval_position_nb(l1, l2, l3, r1, r2, r3, l4, l5, l6, r4, r5, r6, l7, l8, l9, r7, r8, r9)
    return interval3D_nb_normalized(a[0], a[1], a[2], a[3], a[4], a[5])

@njit(fastmath=True)
def interval_position_rect_nb(l1, l2, l3, r1, r2, r3, l4, l5, l6, r4, r5, r6, l7, l8, l9, r7, r8, r9):
    p1 = interval3D_nb_add(l1, l2, l3, r1, r2, r3, l4, l5, l6, r4, r5, r6)
    p2 = interval3D_nb_add(l1, l2, l3, r1, r2, r3, l7, l8, l9, r7, r8, r9)
    p3 = interval3D_nb_add(l7, l8, l9, r7, r8, r9, p1[0], p1[1], p1[2], p1[3], p1[4], p1[5])
    u1 = interval3D_nb_union(p1[0], p1[1], p1[2], p1[3], p1[4], p1[5], p2[0], p2[1], p2[2], p2[3], p2[4], p2[5])
    u2 = interval3D_nb_union(p3[0], p3[1], p3[2], p3[3], p3[4], p3[5], u1[0], u1[1], u1[2], u1[3], u1[4], u1[5])
    return interval3D_nb_union(l1, l2, l3, r1, r2, r3, u2[0], u2[1], u2[2], u2[3], u2[4], u2[5])

@njit(fastmath=True)
def interval_normal_rect_nb(l1, l2, l3, r1, r2, r3, l4, l5, l6, r4, r5, r6, l7, l8, l9, r7, r8, r9):
    a = interval_position_rect_nb(l1, l2, l3, r1, r2, r3, l4, l5, l6, r4, r5, r6, l7, l8, l9, r7, r8, r9)
    return interval3D_nb_normalized(a[0], a[1], a[2], a[3], a[4], a[5])

@njit(fastmath=True)
def interval_test_impl_nb(p0_l1, p0_l2, p0_l3, p0_r1, p0_r2, p0_r3, 
                       p10_l1, p10_l2, p10_l3, p10_r1, p10_r2, p10_r3, 
                       p11_l1, p11_l2, p11_l3, p11_r1, p11_r2, p11_r3, 
                       p12_l1, p12_l2, p12_l3, p12_r1, p12_r2, p12_r3, 
                       n10_l1, n10_l2, n10_l3, n10_r1, n10_r2, n10_r3, 
                       n11_l1, n11_l2, n11_l3, n11_r1, n11_r2, n11_r3, 
                       n12_l1, n12_l2, n12_l3, n12_r1, n12_r2, n12_r3, 
                       p20_l1, p20_l2, p20_l3, p20_r1, p20_r2, p20_r3, 
                       p21_l1, p21_l2, p21_l3, p21_r1, p21_r2, p21_r3, 
                       p22_l1, p22_l2, p22_l3, p22_r1, p22_r2, p22_r3, 
                       n20_l1, n20_l2, n20_l3, n20_r1, n20_r2, n20_r3, 
                       n21_l1, n21_l2, n21_l3, n21_r1, n21_r2, n21_r3, 
                       n22_l1, n22_l2, n22_l3, n22_r1, n22_r2, n22_r3, 
                       p30_l1, p30_l2, p30_l3, p30_r1, p30_r2, p30_r3, 
                       p31_l1, p31_l2, p31_l3, p31_r1, p31_r2, p31_r3, 
                       p32_l1, p32_l2, p32_l3, p32_r1, p32_r2, p32_r3):
    Ix1 = interval_position_nb(p10_l1, p10_l2, p10_l3, p10_r1, p10_r2, p10_r3, 
                               p11_l1, p11_l2, p11_l3, p11_r1, p11_r2, p11_r3, 
                               p12_l1, p12_l2, p12_l3, p12_r1, p12_r2, p12_r3)
    Ix2 = interval_position_nb(p20_l1, p20_l2, p20_l3, p20_r1, p20_r2, p20_r3, 
                               p21_l1, p21_l2, p21_l3, p21_r1, p21_r2, p21_r3, 
                               p22_l1, p22_l2, p22_l3, p22_r1, p22_r2, p22_r3)
    Ix3 = interval_position_rect_nb(p30_l1, p30_l2, p30_l3, p30_r1, p30_r2, p30_r3, 
                                    p31_l1, p31_l2, p31_l3, p31_r1, p31_r2, p31_r3, 
                                    p32_l1, p32_l2, p32_l3, p32_r1, p32_r2, p32_r3)
    Id0 = interval_direction_nb(p0_l1, p0_l2, p0_l3, p0_r1, p0_r2, p0_r3, Ix1[0], Ix1[1], Ix1[2], Ix1[3], Ix1[4], Ix1[5])
    Id1 = interval_direction_nb(Ix1[0], Ix1[1], Ix1[2], Ix1[3], Ix1[4], Ix1[5], Ix2[0], Ix2[1], Ix2[2], Ix2[3], Ix2[4], Ix2[5])
    Id2 = interval_direction_nb(Ix2[0], Ix2[1], Ix2[2], Ix2[3], Ix2[4], Ix2[5], Ix3[0], Ix3[1], Ix3[2], Ix3[3], Ix3[4], Ix3[5])
    Ih1 = interval_halfvector_nb(Id0[0], Id0[1], Id0[2], Id0[3], Id0[4], Id0[5], Id1[0], Id1[1], Id1[2], Id1[3], Id1[4], Id1[5], 1.0 / η_VAL)
    Ih2 = interval_halfvector_nb(Id1[0], Id1[1], Id1[2], Id1[3], Id1[4], Id1[5], Id2[0], Id2[1], Id2[2], Id2[3], Id2[4], Id2[5], η_VAL)
    t = interval_normal_nb(n10_l1, n10_l2, n10_l3, n10_r1, n10_r2, n10_r3, n11_l1, n11_l2, n11_l3, n11_r1, n11_r2, n11_r3, n12_l1, n12_l2, n12_l3, n12_r1, n12_r2, n12_r3)
    In1 = interval3D_nb_mul(t[0], t[1], t[2], t[3], t[4], t[5], -1, -1, -1, -1, -1, -1)
    t = interval_normal_nb(n20_l1, n20_l2, n20_l3, n20_r1, n20_r2, n20_r3, 
                             n21_l1, n21_l2, n21_l3, n21_r1, n21_r2, n21_r3, 
                             n22_l1, n22_l2, n22_l3, n22_r1, n22_r2, n22_r3)
    In2 = interval3D_nb_mul(t[0], t[1], t[2], t[3], t[4], t[5], -1, -1, -1, -1, -1, -1)
    t = interval3D_nb_sub(Ih1[0], Ih1[1], Ih1[2], Ih1[3], Ih1[4], Ih1[5], In1[0], In1[1], In1[2], In1[3], In1[4], In1[5])
    flag1 = interval3D_nb_contain0(t[0], t[1], t[2], t[3], t[4], t[5])
    t = interval3D_nb_sub(Ih2[0], Ih2[1], Ih2[2], Ih2[3], Ih2[4], Ih2[5], In2[0], In2[1], In2[2], In2[3], In2[4], In2[5])
    flag2 = interval3D_nb_contain0(t[0], t[1], t[2], t[3], t[4], t[5])
    return flag1 and flag2

@njit(fastmath=True)
def rotate(p0, p1, p2, sin_theta, cos_theta, sin_phi, cos_phi):
    p1x = p0 * cos_theta - p2 * sin_theta
    p1y = p0 * sin_theta + p2 * cos_theta
    p1z = p1
    p2x = p1x * cos_phi + p1z * sin_phi
    p2y = p1y
    p2z = -p1x * sin_phi + p1z * cos_phi
    return p2x, p2y, p2z

@njit(fastmath=True)
def rotate_impl(p0_l1, p0_l2, p0_l3,
                p10_l1, p10_l2, p10_l3,
                p11_l1, p11_l2, p11_l3,
                p12_l1, p12_l2, p12_l3,
                n10_l1, n10_l2, n10_l3,
                n11_l1, n11_l2, n11_l3,
                n12_l1, n12_l2, n12_l3,
                p20_l1, p20_l2, p20_l3,
                p21_l1, p21_l2, p21_l3,
                p22_l1, p22_l2, p22_l3,
                n20_l1, n20_l2, n20_l3,
                n21_l1, n21_l2, n21_l3,
                n22_l1, n22_l2, n22_l3,
                p30_l1, p30_l2, p30_l3,
                p31_l1, p31_l2, p31_l3,
                p32_l1, p32_l2, p32_l3,
                sin_theta, cos_theta, sin_phi, cos_phi):
    p0 = rotate(p0_l1, p0_l2, p0_l3, sin_theta, cos_theta, sin_phi, cos_phi)
    p10 = rotate(p10_l1, p10_l2, p10_l3, sin_theta, cos_theta, sin_phi, cos_phi)
    p11 = rotate(p11_l1, p11_l2, p11_l3, sin_theta, cos_theta, sin_phi, cos_phi)
    p12 = rotate(p12_l1, p12_l2, p12_l3, sin_theta, cos_theta, sin_phi, cos_phi)
    n10 = rotate(n10_l1, n10_l2, n10_l3, sin_theta, cos_theta, sin_phi, cos_phi)
    n11 = rotate(n11_l1, n11_l2, n11_l3, sin_theta, cos_theta, sin_phi, cos_phi)
    n12 = rotate(n12_l1, n12_l2, n12_l3, sin_theta, cos_theta, sin_phi, cos_phi)
    p20 = rotate(p20_l1, p20_l2, p20_l3, sin_theta, cos_theta, sin_phi, cos_phi)
    p21 = rotate(p21_l1, p21_l2, p21_l3, sin_theta, cos_theta, sin_phi, cos_phi)
    p22 = rotate(p22_l1, p22_l2, p22_l3, sin_theta, cos_theta, sin_phi, cos_phi)
    n20 = rotate(n20_l1, n20_l2, n20_l3, sin_theta, cos_theta, sin_phi, cos_phi)
    n21 = rotate(n21_l1, n21_l2, n21_l3, sin_theta, cos_theta, sin_phi, cos_phi)
    n22 = rotate(n22_l1, n22_l2, n22_l3, sin_theta, cos_theta, sin_phi, cos_phi)
    p30 = rotate(p30_l1, p30_l2, p30_l3, sin_theta, cos_theta, sin_phi, cos_phi)
    p31 = rotate(p31_l1, p31_l2, p31_l3, sin_theta, cos_theta, sin_phi, cos_phi)
    p32 = rotate(p32_l1, p32_l2, p32_l3, sin_theta, cos_theta, sin_phi, cos_phi)
    return p0[0], p0[1], p0[2], p10[0], p10[1], p10[2],p11[0], p11[1], p11[2], p12[0], p12[1], p12[2], n10[0], n10[1], n10[2], n11[0], n11[1], n11[2], n12[0], n12[1], n12[2], p20[0], p20[1], p20[2], p21[0], p21[1], p21[2], p22[0], p22[1], p22[2], n20[0], n20[1], n20[2], n21[0], n21[1], n21[2], n22[0], n22[1], n22[2], p30[0], p30[1], p30[2], p31[0], p31[1], p31[2], p32[0], p32[1], p32[2]

@njit(fastmath=True)
def interval_test_nb(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, sin_theta, cos_theta, sin_phi, cos_phi):
    p0_l1, p0_l2, p0_l3, p10_l1, p10_l2, p10_l3, p11_l1, p11_l2, p11_l3, p12_l1, p12_l2, p12_l3, n10_l1, n10_l2, n10_l3, n11_l1, n11_l2, n11_l3, n12_l1, n12_l2, n12_l3, p20_l1, p20_l2, p20_l3, p21_l1, p21_l2, p21_l3, p22_l1, p22_l2, p22_l3, n20_l1, n20_l2, n20_l3, n21_l1, n21_l2, n21_l3, n22_l1, n22_l2, n22_l3, p30_l1, p30_l2, p30_l3, p31_l1, p31_l2, p31_l3, p32_l1, p32_l2, p32_l3, =  rotate_impl(p0[0], p0[1], p0[2],
                                                                                                                                                                                                                                                                                                                                                                                                            p10[0], p10[1], p10[2],
                                                                                                                                                                                                                                                                                                                                                                                                            p11[0], p11[1], p11[2],
                                                                                                                                                                                                                                                                                                                                                                                                            p12[0], p12[1], p12[2],
                                                                                                                                                                                                                                                                                                                                                                                                            n10[0], n10[1], n10[2],
                                                                                                                                                                                                                                                                                                                                                                                                            n11[0], n11[1], n11[2],
                                                                                                                                                                                                                                                                                                                                                                                                            n12[0], n12[1], n12[2],
                                                                                                                                                                                                                                                                                                                                                                                                            p20[0], p20[1], p20[2],
                                                                                                                                                                                                                                                                                                                                                                                                            p21[0], p21[1], p21[2],
                                                                                                                                                                                                                                                                                                                                                                                                            p22[0], p22[1], p22[2],
                                                                                                                                                                                                                                                                                                                                                                                                            n20[0], n20[1], n20[2],
                                                                                                                                                                                                                                                                                                                                                                                                            n21[0], n21[1], n21[2],
                                                                                                                                                                                                                                                                                                                                                                                                            n22[0], n22[1], n22[2],
                                                                                                                                                                                                                                                                                                                                                                                                            p30[0], p30[1], p30[2],
                                                                                                                                                                                                                                                                                                                                                                                                            p31[0], p31[1], p31[2],
                                                                                                                                                                                                                                                                                                                                                                                                            p32[0], p32[1], p32[2],
                                                                                                                                                                                                                                                                                                                                                                                                            sin_theta, cos_theta, sin_phi, cos_phi)
    return interval_test_impl_nb(p0_l1, p0_l2, p0_l3, p0_l1, p0_l2, p0_l3,
                                 p10_l1, p10_l2, p10_l3, p10_l1, p10_l2, p10_l3, 
                                 p11_l1, p11_l2, p11_l3, p11_l1, p11_l2, p11_l3, 
                                 p12_l1, p12_l2, p12_l3, p12_l1, p12_l2, p12_l3, 
                                 n10_l1, n10_l2, n10_l3, n10_l1, n10_l2, n10_l3, 
                                 n11_l1, n11_l2, n11_l3, n11_l1, n11_l2, n11_l3, 
                                 n12_l1, n12_l2, n12_l3, n12_l1, n12_l2, n12_l3, 
                                 p20_l1, p20_l2, p20_l3, p20_l1, p20_l2, p20_l3, 
                                 p21_l1, p21_l2, p21_l3, p21_l1, p21_l2, p21_l3, 
                                 p22_l1, p22_l2, p22_l3, p22_l1, p22_l2, p22_l3, 
                                 n20_l1, n20_l2, n20_l3, n20_l1, n20_l2, n20_l3, 
                                 n21_l1, n21_l2, n21_l3, n21_l1, n21_l2, n21_l3, 
                                 n22_l1, n22_l2, n22_l3, n22_l1, n22_l2, n22_l3, 
                                 p30_l1, p30_l2, p30_l3, p30_l1, p30_l2, p30_l3, 
                                 p31_l1, p31_l2, p31_l3, p31_l1, p31_l2, p31_l3,
                                 p32_l1, p32_l2, p32_l3, p32_l1, p32_l2, p32_l3)

@njit(fastmath=True)
def interval_test_all_impl(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32):
    if interval_test_nb(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, 0.0, 1.0, 0.0, 1.0) == False:
        return False
    if interval_test_nb(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, 0.0, 1.0, 0.3826834323650898, 0.9238795325112867) == False:
        return False
    if interval_test_nb(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, 0.0, 1.0, 0.7071067811865476, 0.7071067811865476) == False:
        return False
    if interval_test_nb(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, 0.0, 1.0, 0.9238795325112867, 0.38268343236508984) == False:
        return False
    if interval_test_nb(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, 0.3826834323650898, 0.9238795325112867, 0.0, 1.0) == False:
        return False
    if interval_test_nb(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, 0.3826834323650898, 0.9238795325112867, 0.3826834323650898, 0.9238795325112867) == False:
        return False
    if interval_test_nb(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, 0.3826834323650898, 0.9238795325112867, 0.7071067811865476, 0.7071067811865476) == False:
        return False
    if interval_test_nb(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, 0.3826834323650898, 0.9238795325112867, 0.9238795325112867, 0.38268343236508984) == False:
        return False
    if interval_test_nb(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, 0.7071067811865476, 0.7071067811865476, 0.0, 1.0) == False:
        return False
    if interval_test_nb(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, 0.7071067811865476, 0.7071067811865476, 0.3826834323650898, 0.9238795325112867) == False:
        return False
    if interval_test_nb(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, 0.7071067811865476, 0.7071067811865476, 0.7071067811865476, 0.7071067811865476) == False:
        return False
    if interval_test_nb(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, 0.7071067811865476, 0.7071067811865476, 0.9238795325112867, 0.38268343236508984) == False:
        return False
    if interval_test_nb(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, 0.9238795325112867, 0.38268343236508984, 0.0, 1.0) == False:
        return False
    if interval_test_nb(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, 0.9238795325112867, 0.38268343236508984, 0.3826834323650898, 0.9238795325112867) == False:
        return False
    if interval_test_nb(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, 0.9238795325112867, 0.38268343236508984, 0.7071067811865476, 0.7071067811865476) == False:
        return False
    if interval_test_nb(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32, 0.9238795325112867, 0.38268343236508984, 0.9238795325112867, 0.38268343236508984) == False:
        return False
    
    return True

@njit(fastmath=True)
def interval_test_first_bounce_nb(p0_l1, p0_l2, p0_l3, p0_r1, p0_r2, p0_r3, 
                                  p10_l1, p10_l2, p10_l3, p10_r1, p10_r2, p10_r3, 
                                  p11_l1, p11_l2, p11_l3, p11_r1, p11_r2, p11_r3, 
                                  p12_l1, p12_l2, p12_l3, p12_r1, p12_r2, p12_r3, 
                                  n10_l1, n10_l2, n10_l3, n10_r1, n10_r2, n10_r3, 
                                  n11_l1, n11_l2, n11_l3, n11_r1, n11_r2, n11_r3, 
                                  n12_l1, n12_l2, n12_l3, n12_r1, n12_r2, n12_r3, 
                                  p20_l1, p20_l2, p20_l3, p20_r1, p20_r2, p20_r3, 
                                  p21_l1, p21_l2, p21_l3, p21_r1, p21_r2, p21_r3, 
                                  p22_l1, p22_l2, p22_l3, p22_r1, p22_r2, p22_r3):
    Ix1 = interval_position_rect_nb(p10_l1, p10_l2, p10_l3, p10_r1, p10_r2, p10_r3, 
                                    p11_l1, p11_l2, p11_l3, p11_r1, p11_r2, p11_r3, 
                                    p12_l1, p12_l2, p12_l3, p12_r1, p12_r2, p12_r3)
    Ix2 = interval_position_rect_nb(p20_l1, p20_l2, p20_l3, p20_r1, p20_r2, p20_r3, 
                                    p21_l1, p21_l2, p21_l3, p21_r1, p21_r2, p21_r3, 
                                    p22_l1, p22_l2, p22_l3, p22_r1, p22_r2, p22_r3)
    Id0 = interval_direction_nb(p0_l1, p0_l2, p0_l3, p0_r1, p0_r2, p0_r3, Ix1[0], Ix1[1], Ix1[2], Ix1[3], Ix1[4], Ix1[5])
    Id1 = interval_direction_nb(Ix1[0], Ix1[1], Ix1[2], Ix1[3], Ix1[4], Ix1[5], Ix2[0], Ix2[1], Ix2[2], Ix2[3], Ix2[4], Ix2[5])

    Ih1 = interval_halfvector_nb(Id0[0], Id0[1], Id0[2], Id0[3], Id0[4], Id0[5], Id1[0], Id1[1], Id1[2], Id1[3], Id1[4], Id1[5], 1.0 / η_VAL)
    t = interval_normal_rect_nb(n10_l1, n10_l2, n10_l3, n10_r1, n10_r2, n10_r3, n11_l1, n11_l2, n11_l3, n11_r1, n11_r2, n11_r3, n12_l1, n12_l2, n12_l3, n12_r1, n12_r2, n12_r3)
    In1 = interval3D_nb_mul(t[0], t[1], t[2], t[3], t[4], t[5], -1, -1, -1, -1, -1, -1)
    t = interval3D_nb_sub(Ih1[0], Ih1[1], Ih1[2], Ih1[3], Ih1[4], Ih1[5], In1[0], In1[1], In1[2], In1[3], In1[4], In1[5])
    flag1 = interval3D_nb_contain0(t[0], t[1], t[2], t[3], t[4], t[5])
    return flag1

@njit(fastmath=True)
def interval_test_all(tseq):
    return interval_test_all_impl(tseq.pL, tseq.p10, tseq.p11, tseq.p12, tseq.n10, tseq.n11, tseq.n12, tseq.p20, tseq.p21, tseq.p22, tseq.n20, tseq.n21, tseq.n22, tseq.p30, tseq.p31, tseq.p32)

@njit(fastmath=True)
def find_u1_domain_impl(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22):
    queue = np.zeros((103, 4), dtype=np.float64)
    queue[0] = np.array([0, 1, 0, 1])
    ptr = 0
    lenq = 1
    ans_u1m, ans_u1M, ans_v1m, ans_v1M = 1, 0, 1, 0
    while(ptr < lenq):
        u1m, u1M, v1m, v1M = queue[ptr]
        ptr += 1
        p10_, p11_, p12_ = p10 + u1m * p11 + v1m * p12, p11 * (u1M - u1m), p12 * (v1M - v1m)
        n10_, n11_, n12_ = n10 + u1m * n11 + v1m * n12, n11 * (u1M - u1m), n12 * (v1M - v1m)
        if interval_test_first_bounce_nb(p0[0], p0[1], p0[2], p0[0], p0[1], p0[2], 
                                         p10_[0], p10_[1], p10_[2], p10_[0], p10_[1], p10_[2],
                                         p11_[0], p11_[1], p11_[2], p11_[0], p11_[1], p11_[2],
                                         p12_[0], p12_[1], p12_[2], p12_[0], p12_[1], p12_[2],
                                         n10_[0], n10_[1], n10_[2], n10_[0], n10_[1], n10_[2],
                                         n11_[0], n11_[1], n11_[2], n11_[0], n11_[1], n11_[2],
                                         n12_[0], n12_[1], n12_[2], n12_[0], n12_[1], n12_[2],
                                         p20[0], p20[1], p20[2], p20[0], p20[1], p20[2],
                                         p21[0], p21[1], p21[2], p21[0], p21[1], p21[2],
                                         p22[0], p22[1], p22[2], p22[0], p22[1], p22[2]):
            if lenq < 100:
                queue[lenq] = np.array([u1m, (u1m + u1M) / 2, v1m, (v1m + v1M) / 2])
                queue[lenq + 1] = np.array([(u1m + u1M) / 2, u1M, v1m, (v1m + v1M) / 2])
                queue[lenq + 2] = np.array([u1m, (u1m + u1M) / 2, (v1m + v1M) / 2, v1M])
                queue[lenq + 3] = np.array([(u1m + u1M) / 2, u1M, (v1m + v1M) / 2, v1M])
                lenq += 4
            else:
                ans_u1m = min(ans_u1m, u1m)
                ans_u1M = max(ans_u1M, u1M)
                ans_v1m = min(ans_v1m, v1m)
                ans_v1M = max(ans_v1M, v1M)
    return ans_u1m, ans_u1M, ans_v1m, ans_v1M

@njit(fastmath=True)
def find_u1_domain(tseq):
    p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32 = tseq.pL, tseq.p10, tseq.p11, tseq.p12, tseq.n10, tseq.n11, tseq.n12, tseq.p20, tseq.p21, tseq.p22, tseq.n20, tseq.n21, tseq.n22, tseq.p30, tseq.p31, tseq.p32
    return find_u1_domain_impl(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22)


@njit(fastmath=True)
def batch_interval3D_nb_sub(BIx1, Ix2):
    batch_size = BIx1.shape[1]
    result = np.zeros((6, batch_size), dtype=np.float64)
    
    for i in range(batch_size):
        temp = interval3D_nb_sub(BIx1[0, i], BIx1[1, i], BIx1[2, i], BIx1[3, i], BIx1[4, i], BIx1[5, i],
                                Ix2[0], Ix2[1], Ix2[2], Ix2[3], Ix2[4], Ix2[5])
        result[0, i] = temp[0]
        result[1, i] = temp[1] 
        result[2, i] = temp[2]
        result[3, i] = temp[3]
        result[4, i] = temp[4]
        result[5, i] = temp[5]
        
    return result

@njit(fastmath=True)
def batch_interval3D_nb_cross(Alx, Aly, Alz, Arx, Ary, Arz, Blx, Bly, Blz, Brx, Bry, Brz):   # sign may be wrong
    batch_size = Alx.shape[0]
    Ix = np.zeros((2, batch_size), dtype=np.float64)
    Iy = np.zeros((2, batch_size), dtype=np.float64)
    Iz = np.zeros((2, batch_size), dtype=np.float64)
    for i in range(batch_size):
        Iyz = interval_nb_mul(Aly[i], Blz[i], Ary[i], Brz[i])
        Izy = interval_nb_mul(Bly[i], Alz[i], Bry[i], Arz[i])
        Iyx = interval_nb_mul(Aly[i], Blx[i], Ary[i], Brx[i])
        Ixy = interval_nb_mul(Bly[i], Alx[i], Bry[i], Arx[i])
        Ixz = interval_nb_mul(Alx[i], Blz[i], Arx[i], Brz[i])
        Izx = interval_nb_mul(Blx[i], Alz[i], Brx[i], Arz[i])
        t_Ix = interval_nb_sub(Iyz[0], Izy[0], Iyz[1], Izy[1])
        t_Iy = interval_nb_sub(Izx[0], Ixz[0], Izx[1], Ixz[1])
        t_Iz = interval_nb_sub(Ixy[0], Iyx[0], Ixy[1], Iyx[1])

        Ix[0, i] = t_Ix[0]
        Ix[1, i] = t_Ix[1]
        Iy[0, i] = t_Iy[0]
        Iy[1, i] = t_Iy[1]
        Iz[0, i] = t_Iz[0]
        Iz[1, i] = t_Iz[1]
    return Ix[0], Iy[0], Iz[0], Ix[1], Iy[1], Iz[1]

@njit(fastmath=True)
def batch_interval3D_nb_contain0(l1, l2, l3, r1, r2, r3):
    batch_size = l1.shape[0]
    result = np.zeros(batch_size, dtype=np.bool_)
    for i in range(batch_size):
        result[i] = l1[i] <= 0 <= r1[i] and l2[i] <= 0 <= r2[i] and l3[i] <= 0 <= r3[i]
    return result
