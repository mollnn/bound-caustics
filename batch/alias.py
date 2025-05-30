# alias, constants, and hyperparameters
import os
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy import dot, cross, sqrt
import math
from scipy.signal import convolve2d, convolve
import time
from tqdm import tqdm
from copy import deepcopy
import pickle
from scipy.signal import convolve, fftconvolve, oaconvolve
import numba
from numba import jit, njit, int32, int64, float64, types, optional
from numba.experimental import jitclass
from numba.typed import Dict

np.random.seed(1)

bvh_node_spec = [
    ('bbox_min', float64[:]),
    ('bbox_max', float64[:]),
    ('triangle_id', int32),
    ('left', int32),
    ('right', int32)
]

@jitclass(bvh_node_spec)
class BVHNode:
    def __init__(self):
        self.bbox_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        self.bbox_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
        self.triangle_id = -1
        self.left = -1
        self.right = -1


def build_bvh(mesh, obj_tri_cnt_prefix):
    bvh_nodes = []
    def build_bvh_rec(mesh, triangle_index, obj_index):
        if len(triangle_index) == 0:
            return -1

        node = BVHNode()
        cur_node_index = len(bvh_nodes)
        
        vertices = mesh.reshape(-1, 3)
        node.bbox_min = np.min(vertices, axis=0)
        node.bbox_max = np.max(vertices, axis=0)

        bvh_nodes.append(node)

        if len(triangle_index) == 1:
            node.triangle_id = triangle_index[0]
            return cur_node_index

        axis = np.argmax(node.bbox_max - node.bbox_min)
        mid = (node.bbox_min[axis] + node.bbox_max[axis]) / 2

        centroids = np.mean(mesh, axis=1)
        left_mask = centroids[:, axis] <= mid
        right_mask = ~left_mask
        
        if not np.any(left_mask) or not np.any(right_mask):
            mid_idx = len(triangle_index) // 2
            left_mask = np.zeros_like(triangle_index, dtype=bool)
            left_mask[:mid_idx] = True
            right_mask = ~left_mask

        if np.any(left_mask):
            node.left = build_bvh_rec(mesh[left_mask], triangle_index[left_mask], obj_index)
        if np.any(right_mask):
            node.right = build_bvh_rec(mesh[right_mask], triangle_index[right_mask], obj_index)

        return cur_node_index
    
    bvh_roots = []
    for i in range(len(obj_tri_cnt_prefix) - 1):
        tri_start = obj_tri_cnt_prefix[i]
        tri_end = obj_tri_cnt_prefix[i + 1]
        bvh_roots.append(build_bvh_rec(mesh[tri_start:tri_end], np.arange(tri_start, tri_end), i))

    return bvh_nodes, bvh_roots

def read_obj(filename, offset=[0, 0, 0], scale=1):
    fp = open(filename, "r")
    fl = fp.readlines()
    v = [[]]
    vn = [[]]
    ans = []
    obj_tri_cnt_prefix = []
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
            elif a[0] == "g":
                if len(a) > 1 and a[1] == "default":
                    obj_tri_cnt_prefix.append(len(ans))

    obj_tri_cnt_prefix.append(len(ans))

    mesh = np.array(ans)
    bvhs = build_bvh(mesh[:, 0:3, :], obj_tri_cnt_prefix)
    
    return mesh, bvhs, obj_tri_cnt_prefix # caveat: three point form, not edge delta form

# diamonds

# p0 = np.array([1, 2.0, -2])
# p30 = np.array([-3, 0.0, -3])
# p31 = np.array([6, 0.000001, 0])
# p32 = np.array([0, 0.000002, 6])
# pD0, pD1, pD2 = p30, p31, p32
# mesh, bvhs, obj_tri_cnt_prefix = read_obj("../data/diamonds.obj", [0.0, 0.0, 0])
# print(obj_tri_cnt_prefix)
# ground = np.array([p30, p30+p31, p30+p32])

# constraint_file_name = "constraints.txt"

# RES = 512
# N_THREAD = 16
# JACOBIAN_COMPRESS = 1   # 0-accurate 1-compressed   accurate is almost for one bounce only

# CHAIN_TYPE = 22 # 1 for reflection and 2 for refraction
# CHAIN_LENGTH = 1 if CHAIN_TYPE < 10 else 2

# η_VAL = 0.6666666666666666
# SHADING_NORMAL = False # the first bounce (close to light)
# SHADING_NORMAL2 = False # the second bounce (close to receiver)

# # CORE HYPERPARAMETERS
# INF_AREA_TOL = SPLAT_SUBDIV_THRES = 1e-4
# U1T = u1TOLERATE = 0.125001
# AR = 1e1 # approx ratio

# # NOT IMPORTANT PARAMETERS
# Am = 1e-5 # minimum irradiance
# AM = 1e2 # maximum irradiance
# β_STRONG_THRES = 1e99  # please fix explicit before run this code
# β_MIN = 1e-9

# BATCH_SIZE = 64


# slab

p0 = np.array([1, 2.0, -2])
p30 = np.array([-3, 0.0, -3])
p31 = np.array([6, 0.000001, 0])
p32 = np.array([0, 0.000002, 6])
pD0, pD1, pD2 = p30, p31, p32
mesh, bvhs, obj_tri_cnt_prefix = read_obj("../data/slab10k.obj", [0.0, 0.0, 0])
print(obj_tri_cnt_prefix)
ground = np.array([p30, p30+p31, p30+p32])

constraint_file_name = "constraints.txt"

RES = 512
N_THREAD = 16
JACOBIAN_COMPRESS = 1   # 0-accurate 1-compressed   accurate is almost for one bounce only

SKIP_IRRADIANCE = True

CHAIN_TYPE = 22 # 1 for reflection and 2 for refraction
CHAIN_LENGTH = 1 if CHAIN_TYPE < 10 else 2

η_VAL = 0.6666666666666666
SHADING_NORMAL = True # the first bounce (close to light)
SHADING_NORMAL2 = True # the second bounce (close to receiver)

# CORE HYPERPARAMETERS
INF_AREA_TOL = SPLAT_SUBDIV_THRES = 1e-4
U1T = u1TOLERATE = 0.500001
AR = 1e1 # approx ratio

# NOT IMPORTANT PARAMETERS
Am = 1e-5 # minimum irradiance
AM = 1e2 # maximum irradiance
β_STRONG_THRES = 1e9
β_MIN = 1e-9

BATCH_SIZE = 64

def normalize(x):
    return x / norm(x)

def array(a):
    return np.array(a)

def DUMMY_VEC3(): 
    return array([0, 0, 0]) + np.random.rand(3) * 1e-6

COLOR0 = (70/255, 100/255, 100/255)
COLOR0S = (140/255, 197/255, 190/255)
COLOR1 = (7/255, 65/255, 102/255)
COLOR2 = (204/255, 1/255, 31/255)
COLOR3 = (250/255, 218/255, 221/255)

spec = [('pL', float64[:]), ('p10', float64[:]), ('p11', float64[:]), ('p12', float64[:]), ('n10', float64[:]), ('n11', float64[:]), ('n12', float64[:]), ('p20', float64[:]), ('p21', float64[:]), ('p22', float64[:]), ('n20', float64[:]), ('n21', float64[:]), ('n22', float64[:]), ('p30', float64[:]), ('p31', float64[:]), ('p32', float64[:])]
@jitclass(spec)
class TriangleSequence:
    def __init__(self, pL, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32):
        self.pL, self.p10, self.p11, self.p12, self.n10, self.n11, self.n12, self.p20, self.p21, self.p22, self.n20, self.n21, self.n22, self.p30, self.p31, self.p32 = pL, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32
