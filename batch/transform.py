# transform between bernstein polynomials and power basis polynomials, and degree reduction routines
from alias import *
from polynomial import *

BCS = 1

fac = [1]
    
@njit(cache=True, fastmath=True)
def inv_matU(n):
    u = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1):
            u[i][j] = binomial[i, j] / binomial[n - 1, j]
    return u

cache_c2b = Dict.empty(key_type=int64, value_type=float64[:,:])

@njit(fastmath=True)
def C2B2d(a, cache_c2b, force_len=None):
    n = len(a)
    if force_len is not None:
        n = force_len
    if n not in cache_c2b:
        cache_c2b[n] = inv_matU(n)
    iU = np.ascontiguousarray(cache_c2b[n])
    ap = np.zeros((n, n))
    ap[:len(a), :len(a)] = a
    tmp = iU @ ap
    tmp = np.transpose(tmp)
    tmp = iU @ tmp
    b = np.transpose(tmp)
    return b

@njit(cache=True, fastmath=True)
def C2B2d_impl(a, iU, n):
    ap = np.zeros((n, n))
    ap[:len(a), :len(a)] = a
    tmp = iU @ ap
    tmp = np.transpose(tmp)
    tmp = iU @ tmp
    b = np.transpose(tmp)
    return b

######################

# compress the first four dims, reduce the degree of the last two dims

lin_reg_x_n = [1, 1, 2, 2]

@njit(cache=True, fastmath=True)
def lsq_wrapper(a, b):
    # return np.linalg.lstsq(a, b)[0]
    return np.linalg.solve(a.T @ a, a.T @ b)

def get_linreg_x_impl(bern_input):
    global lin_reg_x_n
    linreg_x = np.zeros((bern_input.shape[0] , bern_input.shape[1], 3))
    if bern_input.shape[0] > 1:
        for i in range(bern_input.shape[0]):
            linreg_x[i, :, 0] = i / (bern_input.shape[0] - 1)
    if bern_input.shape[1] > 1:
        for i in range(bern_input.shape[1]):
            linreg_x[:, i, 1] = i / (bern_input.shape[1] - 1)
    linreg_x[:, :, 2] = 1
    
    linreg_x = linreg_x.reshape((-1, 3))
    N = lin_reg_x_n[2] * lin_reg_x_n[3]
    linreg_ext = np.zeros((N, len(linreg_x)))
    for k in range(lin_reg_x_n[2]):
        tk = linreg_x[:, 0] ** k
        for l in range(lin_reg_x_n[3]):
            linreg_ext[k * lin_reg_x_n[3] + l] = tk * linreg_x[:, 1] ** l
    linreg_x = linreg_ext.T
    return linreg_x

@njit(cache=True, fastmath=True)
def batch_get_linreg_x(bern_input):
    lin_reg_x_n = [1, 1, 2, 2]
    linreg_x = np.zeros((bern_input.shape[0] , bern_input.shape[1], 3))
    if bern_input.shape[0] > 1:
        for i0 in range(bern_input.shape[0]):
            w = i0 / (bern_input.shape[0] - 1)
            for i1 in range(bern_input.shape[1]):
                linreg_x[i0, i1, 0] = w
    if bern_input.shape[1] > 1:
        for i1 in range(bern_input.shape[1]):
            w = i1 / (bern_input.shape[1] - 1)
            for i0 in range(bern_input.shape[0]):
                linreg_x[i0, i1, 1] = w
    for i0 in range(bern_input.shape[0]):
        for i1 in range(bern_input.shape[1]):
                linreg_x[i0, i1, 2] = 1
    
    linreg_x = linreg_x.reshape((-1, 3))
    N = lin_reg_x_n[2] * lin_reg_x_n[3]
    linreg_ext = np.zeros((N, len(linreg_x)))
    for k in range(lin_reg_x_n[2]):
        tk = np.empty((linreg_x.shape[0]), dtype=np.float64)
        for i in range(linreg_x.shape[0]):
            tk[i] = linreg_x[i, 0] ** k
        for l in range(lin_reg_x_n[3]):
            for i in range(linreg_x.shape[0]):
                linreg_ext[k * lin_reg_x_n[3] + l, i] = tk[i] * linreg_x[i, 1] ** l

    linreg_x = linreg_ext.T
    return linreg_x

linreg_x_cache = {}

def get_linreg_x(bern_input):
    global linreg_x_cache
    hash = bern_input.shape[0] * 1000 + bern_input.shape[1]
    if hash not in linreg_x_cache.keys():
        linreg_x_cache[hash] = get_linreg_x_impl(bern_input)
    return linreg_x_cache[hash]

@njit(cache=True)
def broadcast(ans_bern_hd_, bern_input_full):
    ans_bern_hd = np.zeros_like(bern_input_full)
    for i in range(len(bern_input_full)):
        for j in range(len(bern_input_full[0])):
            for k in range(len(bern_input_full[0][0])):
                for l in range(len(bern_input_full[0][0][0])):
                    ans_bern_hd[i][j][k][l] = ans_bern_hd_
    return ans_bern_hd

def reduce_all(bern_input_, new_axis = 0):   # 5d (high deg) -> 1 new + 2 null + 2d (low deg)
    global lin_reg_x_n
    siz = bern_input_.a.shape
    bern_input = bern_input_.elevate([siz[0] - 1, siz[1] - 1, siz[2] - 1, siz[3] - 1, max(siz[4], siz[5]) - 1, max(siz[4], siz[5]) - 1]).a
    bern_input_full = bern_input[:]
    bern_input = np.average(bern_input, axis=0)
    bern_input = np.average(bern_input, axis=0)
    bern_input = np.average(bern_input, axis=0)
    bern_input = np.average(bern_input, axis=0)
    linreg_x = get_linreg_x(bern_input)
    flat_bern_input_full = bern_input_full.flatten()
    flat_bern_input = bern_input.flatten()
    linreg_ans = lsq_wrapper(linreg_x, flat_bern_input)
    ans_mono = np.reshape(linreg_ans, lin_reg_x_n[2:])
    ans_bern = np.array(C2B2d(ans_mono, cache_c2b))
    ans_bern_hd_ = np.array(C2B2d(ans_mono, cache_c2b, len(bern_input_full[0][0][0][0])))
    ans_bern_hd = broadcast(ans_bern_hd_, bern_input_full)
    ans_bern_hd = np.array(ans_bern_hd).flatten()
    max_error = np.max(np.abs(flat_bern_input_full - ans_bern_hd))
    ans_bern = ans_bern.reshape([1] + lin_reg_x_n)
    final_ans = np.stack([ans_bern - max_error, ans_bern + max_error])
    if new_axis == -1:
        final_ans = (final_ans[:1] + final_ans[1:]) / 2
    elif new_axis != 0:
        final_ans = np.swapaxes(final_ans, 0, new_axis)
    return BPoly(final_ans)

@njit(cache=True, fastmath=True)
def batch_bern_average(bern_input):
    bern_input_avg = np.zeros(bern_input[0].shape, dtype=np.float64)
    for i in range(bern_input.shape[0]):
        bern_input_avg += bern_input[i]
    bern_input_avg /= bern_input.shape[0]
    return bern_input_avg

@njit(cache=True, fastmath=True)
def batch_flatten(a, inner_batch=True):
    if inner_batch:
        batch_size = a.shape[-1]
        num = int(a.size / batch_size)
        ret = np.empty((batch_size, num))
        for i in range(batch_size):
            ret[i] = a[..., i].flatten()
    else:
        batch_size = a.shape[0]
        num = int(a.size / batch_size)
        ret = np.empty((batch_size, num))
        for i in range(batch_size):
            ret[i] = a[i].flatten()
    return ret


@njit(fastmath=True)
def batch_reduce_all(bern_input_, cache_c2b, new_axis = 0, target_degree=2):   # 5d (high deg) -> 1 new + 2 null + 2d (low deg)
    lin_reg_x_n = [1, 1, target_degree, target_degree]

    siz = bern_input_.a.shape
    batch_size = siz[-1]
    bern_input = bern_input_.elevate(np.array([siz[0] - 1, siz[1] - 1, siz[2] - 1, siz[3] - 1, max(siz[4], siz[5]) - 1, max(siz[4], siz[5]) - 1], dtype=np.int64)).a
    bern_input_full = bern_input[:]
    bern_input = batch_bern_average(bern_input)
    bern_input = batch_bern_average(bern_input)
    bern_input = batch_bern_average(bern_input)
    bern_input = batch_bern_average(bern_input)
    linreg_x = batch_get_linreg_x(bern_input)

    flat_bern_input_full = batch_flatten(bern_input_full)
    flat_bern_input = batch_flatten(bern_input)
    ans_bern = np.empty((batch_size, lin_reg_x_n[2], lin_reg_x_n[3]), dtype=np.float64)
    ans_bern_hd = np.zeros_like(flat_bern_input_full)
    for i in range(batch_size):
        linreg_ans = lsq_wrapper(linreg_x, flat_bern_input[i])
        ans_mono = np.reshape(linreg_ans, (lin_reg_x_n[2], lin_reg_x_n[3]))
        ans_bern[i] = C2B2d(ans_mono, cache_c2b)
        ans_bern_hd_ = C2B2d(ans_mono, cache_c2b, len(bern_input_full[0][0][0][0]))
        temp = np.zeros(bern_input_full.shape[:-1], dtype=np.float64)
        shape = bern_input_full.shape
        for i0 in range(shape[0]):
            for i1 in range(shape[1]):
                for i2 in range(shape[2]):
                    for i3 in range(shape[3]):
                        temp[i0, i1, i2, i3] = ans_bern_hd_

        ans_bern_hd[i] = temp.flatten()
    
    max_error = np.zeros((batch_size), dtype=np.float64)
    errors = np.abs(flat_bern_input_full - ans_bern_hd)
    for i in range(batch_size):
        max_error[i] = np.max(errors[i])
    ans = np.zeros((2, 1, 1, 1, target_degree, target_degree, batch_size), dtype=np.float64)
    for i0 in range(target_degree):
        for i1 in range(target_degree):
            for i in range(batch_size):
                ans[0, 0, 0, 0, i0, i1, i] = ans_bern[i, i0, i1] - max_error[i]
                ans[1, 0, 0, 0, i0, i1, i] = ans_bern[i, i0, i1] + max_error[i]
    if new_axis == -1:
        final_ans = np.zeros((1, 1, 1, 1, target_degree, target_degree, batch_size), dtype=np.float64)
        for i0 in range(target_degree):
            for i1 in range(target_degree):
                for i in range(batch_size):
                    final_ans[0, 0, 0, 0, i0, i1, i] = (ans[0, 0, 0, 0, i0, i1, i] + ans[1, 0, 0, 0, i0, i1, i]) * 0.5
    else:
        final_ans = np.swapaxes(ans, 0, new_axis)

    return BatchBPoly(final_ans)

######################################
# ex version: compress first two dim only. the mid two dims are unchanged. reduce the degree of the last two dims.

lin_reg_x_n_ex = 2

@njit(cache=True, fastmath=True)
def get_linreg_x_ex_impl_ex_ex(bern_input):
    linreg_x = np.zeros((bern_input.shape[0] , bern_input.shape[1], 3))
    if bern_input.shape[0] > 1:
        for i in range(bern_input.shape[0]):
            linreg_x[i, :, 0] = i / (bern_input.shape[0] - 1)
    if bern_input.shape[1] > 1:
        for i in range(bern_input.shape[1]):
            linreg_x[:, i, 1] = i / (bern_input.shape[1] - 1)
    linreg_x[:, :, 2] = 1
    
    linreg_x = linreg_x.reshape((-1, 3))
    N = lin_reg_x_n_ex * lin_reg_x_n_ex
    linreg_ext = np.zeros((N, len(linreg_x)))
    for k in range(lin_reg_x_n_ex):
        tk = linreg_x[:, 0] ** k
        for l in range(lin_reg_x_n_ex):
            linreg_ext[k * lin_reg_x_n_ex + l] = tk * linreg_x[:, 1] ** l
    return linreg_ext.T

cache_r2d = Dict.empty(key_type=int64, value_type=float64[:,:])

@njit(fastmath=True)
def get_linreg_x_ex(cache_r2d, bern_input):
    hash = bern_input.shape[0] * 1000 + bern_input.shape[1]
    if hash not in cache_r2d:
        cache_r2d[hash] = get_linreg_x_ex_impl_ex_ex(bern_input)
    return cache_r2d[hash]


@njit(cache=True, fastmath=True)
def batch_get_linreg_x_ex(cache_r2d, bern_input):
    batch_size = bern_input.shape[-1]
    linreg_x = np.zeros((bern_input.shape[0] , bern_input.shape[1], 3, batch_size))
    if bern_input.shape[0] > 1:
        for i in range(bern_input.shape[0]):
            w = i / (bern_input.shape[0] - 1)
            for j in range(bern_input.shape[1]):
                for k in range(batch_size):
                    linreg_x[i, j, 0, k] = w
    if bern_input.shape[1] > 1:
        for i in range(bern_input.shape[0]):
            for j in range(bern_input.shape[1]):
                w = j / (bern_input.shape[1] - 1)
                for k in range(batch_size):
                    linreg_x[i, j, 1, k] = w
    for i in range(bern_input.shape[0]):
        for j in range(bern_input.shape[1]):
            for k in range(batch_size):
                linreg_x[i, j, 2, k] = 1
    
    linreg_x = linreg_x.reshape((-1, 3, batch_size))
    N = lin_reg_x_n_ex * lin_reg_x_n_ex
    linreg_ext = np.zeros((N, len(linreg_x), batch_size))
    for k in range(lin_reg_x_n_ex):
        tk = np.empty((linreg_x.shape[0], batch_size), dtype=np.float64)
        for i in range(linreg_x.shape[0]):
            for j in range(batch_size):
                tk[i, j] = linreg_x[i, 0, j] ** k
        for l in range(lin_reg_x_n_ex):
            for i in range(linreg_x.shape[0]):
                for j in range(batch_size): 
                    linreg_ext[k * lin_reg_x_n_ex + l, i, j] = tk[i, j] * linreg_x[i, 1, j] ** l
                    
    ret = np.empty((batch_size, len(linreg_x), N), dtype=np.float64)
    for k in range(ret.shape[0]):
        for i in range(ret.shape[1]):
            for j in range(ret.shape[2]):
                ret[k, i, j] = linreg_ext[j, i, k]
    return ret

@njit(cache=True, fastmath=True)
def bern_compress_ex(bern_input_, linreg_x, iU, iU_big, new_axis = 0): 
    bern_input = bern_input_
    bern_input_full = bern_input[:]
    bern_input_avg = np.zeros_like(bern_input[0][0])
    for i in range(len(bern_input)):
        for j in range(len(bern_input[0])):
            bern_input_avg += bern_input[i][j]
    bern_input_avg /= len(bern_input) * len(bern_input[0])
    bern_input = bern_input_avg
    flat_bern_input_full = bern_input_full.flatten()
    flat_bern_input = np.ascontiguousarray(bern_input.flatten())
    linreg_ans = np.ascontiguousarray(lsq_wrapper(linreg_x, flat_bern_input))
    ans_mono = linreg_ans.reshape((lin_reg_x_n_ex, lin_reg_x_n_ex))
    ans_bern = C2B2d_impl(ans_mono, iU, len(ans_mono))
    ans_bern_hd_ = C2B2d_impl(ans_mono, iU_big, len(bern_input_full[0][0]))
    ans_bern_hd = np.zeros_like(bern_input_full)
    for i in range(len(bern_input_full)):
        for j in range(len(bern_input_full[0])):
            ans_bern_hd[i][j] = ans_bern_hd_
    ans_bern_hd = ans_bern_hd.flatten()
    max_error = np.max(flat_bern_input_full - ans_bern_hd) 
    min_error = np.min(flat_bern_input_full - ans_bern_hd)
    ans_bern = ans_bern.reshape((1, lin_reg_x_n_ex, lin_reg_x_n_ex))
    final_ans = np.zeros((2, 1, lin_reg_x_n_ex, lin_reg_x_n_ex), dtype=np.float64)
    final_ans[0] = ans_bern + min_error
    final_ans[1] = ans_bern + max_error
    if new_axis != 0:
        final_ans = np.swapaxes(final_ans, 0, new_axis)
    return final_ans, max_error


@njit(fastmath=True)
def reduce2d(cache_r2d, cache_c2b, bern_input_, new_axis = 0):   
    bern_input = bern_input_.elevate([bern_input_.a.shape[0] - 1, bern_input_.a.shape[1] - 1, bern_input_.a.shape[2] - 1, bern_input_.a.shape[3] - 1,  max(bern_input_.a.shape[4], bern_input_.a.shape[5]) - 1, max(bern_input_.a.shape[4], bern_input_.a.shape[5]) - 1])
    linreg_x = get_linreg_x_ex(cache_r2d, bern_input.a[0][0][0][0])
    ans = np.zeros((2, 1, bern_input.a.shape[2], bern_input.a.shape[3], lin_reg_x_n_ex, lin_reg_x_n_ex))
    iU = inv_matU(lin_reg_x_n_ex)
    iU_big = inv_matU(bern_input.a.shape[4])
    for i in range(bern_input.a.shape[2]):
        for j in range(bern_input.a.shape[3]):
            ans[:, :, i, j], tmp = bern_compress_ex(bern_input.a[:, :, i, j, :, :], linreg_x, iU, iU_big, 0)
    if new_axis != 0:
        ans = np.swapaxes(ans, 0, new_axis)
    return BPoly(ans)


@njit(fastmath=True)
def batch_reduce2d(cache_r2d, bern_input_, new_axis = 0):
    batch_size = bern_input_.a.shape[-1]   
    bern_input = bern_input_.elevate([bern_input_.a.shape[0] - 1, bern_input_.a.shape[1] - 1, bern_input_.a.shape[2] - 1, bern_input_.a.shape[3] - 1,  max(bern_input_.a.shape[4], bern_input_.a.shape[5]) - 1, max(bern_input_.a.shape[4], bern_input_.a.shape[5]) - 1])
    linreg_x = np.ascontiguousarray(get_linreg_x_ex(cache_r2d, bern_input.a[0][0][0][0]))
    ans = np.zeros((batch_size, 2, 1, bern_input.a.shape[2], bern_input.a.shape[3], lin_reg_x_n_ex, lin_reg_x_n_ex))
    iU = np.ascontiguousarray(inv_matU(lin_reg_x_n_ex))
    iU_big = np.ascontiguousarray(inv_matU(bern_input.a.shape[4]))
    for i in range(bern_input.a.shape[2]):
        for j in range(bern_input.a.shape[3]):
            for k in range(batch_size):
                ans[k, :, :, i, j], _ = bern_compress_ex(bern_input.a[:, :, i, j, :, :, k], linreg_x, iU, iU_big, 0)
    batch_ans = np.zeros((2, 1, bern_input.a.shape[2], bern_input.a.shape[3], lin_reg_x_n_ex, lin_reg_x_n_ex, batch_size))
    for i0 in range(2):
        for i1 in range(1):
            for i2 in range(bern_input.a.shape[2]):
                for i3 in range(bern_input.a.shape[3]):
                    for i4 in range(lin_reg_x_n_ex):
                        for i5 in range(lin_reg_x_n_ex):
                            for i6 in range(batch_size):
                                batch_ans[i0, i1, i2, i3, i4, i5, i6] = ans[i6, i0, i1, i2, i3, i4, i5]
    if new_axis != 0:
        batch_ans = np.swapaxes(batch_ans, 0, new_axis)
    return BatchBPoly(batch_ans)

######################################

@njit(fastmath=True)
def msm_r2d(cache_r2d, cache_c2b, a_, b_, c_, d_, axis=0): # mul sub mul with auto reduce 2d
    a = a_
    b = BPoly(np.swapaxes(b_.a, 0, 1))
    ab = reduce2d(cache_r2d, cache_c2b, a * b, 0)
    c = c_
    d = BPoly(np.swapaxes(d_.a, 0, 1))
    cd = reduce2d(cache_r2d, cache_c2b, c * d, 1)
    return reduce2d(cache_r2d, cache_c2b, ab - cd, axis)

@njit(fastmath=True)
def batch_msm_r2d(cache_r2d, cache_c2b, a_, b_, c_, d_, axis=0): # mul sub mul with auto reduce 2d
    a = a_
    b = BatchBPoly(np.swapaxes(b_.a, 0, 1))
    ab = batch_reduce2d(cache_r2d, a * b, 0)
    c = c_
    d = BatchBPoly(np.swapaxes(d_.a, 0, 1))
    cd = batch_reduce2d(cache_r2d, c * d, 1)
    return batch_reduce2d(cache_r2d, ab - cd, axis)

@njit(fastmath=True)
def reduce2d_batch4(cache_r2d, cache_c2b, in1, in2, in3, in4, new_axis):
    out1 = reduce2d(cache_r2d, cache_c2b, in1, new_axis)
    out2 = reduce2d(cache_r2d, cache_c2b, in2, new_axis)
    out3 = reduce2d(cache_r2d, cache_c2b, in3, new_axis)
    out4 = reduce2d(cache_r2d, cache_c2b, in4, new_axis)
    return out1, out2, out3, out4

@njit(fastmath=True)
def batch_reduce2d_batch4(cache_r2d, in1, in2, in3, in4, new_axis):
    out1 = batch_reduce2d(cache_r2d, in1, new_axis)
    out2 = batch_reduce2d(cache_r2d, in2, new_axis)
    out3 = batch_reduce2d(cache_r2d, in3, new_axis)
    out4 = batch_reduce2d(cache_r2d, in4, new_axis)
    return out1, out2, out3, out4


if __name__ == "__main__":  # some test code to show numba performance issues. to be deleted
    def check(batch_size, batch_ans, ans, mode=0):
        flag = True
        for i in range(batch_size):
            # print(i)
            # print(batch_ans.x[..., i])
            # print(ans[i].x)
            if mode == 0:
                flag = flag and np.allclose(batch_ans.x[..., i], ans[i].x, atol=1e-7)
                flag = flag and np.allclose(batch_ans.y[..., i], ans[i].y, atol=1e-7)
                flag = flag and np.allclose(batch_ans.z[..., i], ans[i].z, atol=1e-7)
            elif mode == 1:
                flag = flag and np.allclose(batch_ans.a[..., i], ans[i].a, atol=1e-7)
            else:
                flag = flag and np.allclose(batch_ans[..., i], ans[i], atol=1e-7)
        return flag
    
    def run(a, batch_size, new_axis=0):
        ans = []
        for j in range(batch_size):
            ans.append(reduce_all(a[j], new_axis))
        return ans

    @ njit(fastmath=True)
    def run_batch(a, cache_c2b, new_axis=0):
        return batch_reduce_all(a, cache_c2b, new_axis)
    
    @ njit(fastmath=True)
    def run_batch2(a, cache_r2d, new_axis=0):
        return batch_reduce2d(cache_r2d, a, new_axis)
    
    for batch_size in [2]:
        for array_size in [8]:
            x = [np.random.rand(array_size, array_size) for _ in range(batch_size)]

            a = [BPoly(np.array(x[i], dtype=np.float64)) for i in range(batch_size)]
            
            bx = np.empty((array_size, array_size, batch_size), dtype=np.float64)
            for i in range(batch_size):
                for j in range(array_size):
                    for k in range(array_size):
                        bx[j, k, i] = x[i][j][k]

            ba = BatchBPoly(np.array(bx, dtype=np.float64))

            t0 = time.perf_counter()
            for _ in range(1000):
                run_batch(ba, cache_c2b)
            t = time.perf_counter()-t0
            print("batch_reduce_all:")
            print(t, " ms")
            
            t0 = time.perf_counter()
            for _ in range(1000):
                run_batch2(ba, cache_r2d)
            t = time.perf_counter()-t0
            print("batch_reduce2d:")
            print(t, " ms")

            t0 = time.perf_counter()
            for _ in range(1000):
                run_batch(ba, cache_c2b)
            t = time.perf_counter()-t0
            print("batch_reduce_all:")
            print(t, " ms")
            
            t0 = time.perf_counter()
            for _ in range(1000):
                run_batch2(ba, cache_r2d)
            t = time.perf_counter()-t0
            print("batch_reduce2d:")
            print(t, " ms")
