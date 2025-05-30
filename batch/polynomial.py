# (bernstein-form) polynomial classes
from alias import *


@njit(cache=True, fastmath=True)
def initialize_factorial(max_n):
    fac = np.ones(max_n + 1, dtype=np.float64)
    for i in range(1, max_n + 1):
        fac[i] = fac[i - 1] * i
    return fac


@njit(cache=True, fastmath=True)
def initialize_binomial(max_n, factorial):
    bino = np.ones((max_n + 1, max_n + 1), dtype=np.float64)
    for n in range(max_n + 1):
        for i in range(n + 1):
            bino[n, i] = factorial[n] / (factorial[i] * factorial[n - i])
    return bino


max_n = 144
factorial = initialize_factorial(max_n)
binomial = initialize_binomial(max_n, factorial)
mat_left = Dict.empty(key_type=int64, value_type=float64[:, :])
mat_right = Dict.empty(key_type=int64, value_type=float64[:, :])


@njit(cache=True, fastmath=True)
def evalB_basis(d, j, x):
    return binomial[d, j] * np.power(x, j) * np.power(1 - x, d - j)


@njit(cache=True, fastmath=True)
def nb_convolve(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    result = np.zeros((arr1.shape[0] + arr2.shape[0] - 1, arr1.shape[1] + arr2.shape[1] - 1, arr1.shape[2] + arr2.shape[2] - 1,
                      arr1.shape[3] + arr2.shape[3] - 1, arr1.shape[4] + arr2.shape[4] - 1, arr1.shape[5] + arr2.shape[5] - 1), dtype=np.float64)
    for i1 in range(arr1.shape[0]):
        for i2 in range(arr1.shape[1]):
            for i3 in range(arr1.shape[2]):
                for i4 in range(arr1.shape[3]):
                    for i5 in range(arr1.shape[4]):
                        for i6 in range(arr1.shape[5]):
                            for j1 in range(arr2.shape[0]):
                                for j2 in range(arr2.shape[1]):
                                    for j3 in range(arr2.shape[2]):
                                        for j4 in range(arr2.shape[3]):
                                            for j5 in range(arr2.shape[4]):
                                                for j6 in range(arr2.shape[5]):
                                                    result[i1 + j1, i2 + j2, i3 + j3, i4 + j4, i5 + j5, i6 +
                                                           j6] += arr1[i1, i2, i3, i4, i5, i6] * arr2[j1, j2, j3, j4, j5, j6]
    return result


@njit(cache=True, fastmath=True)
def batch_nb_convolve(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    result = np.zeros((arr1.shape[0] + arr2.shape[0] - 1,
                       arr1.shape[1] + arr2.shape[1] - 1,
                       arr1.shape[2] + arr2.shape[2] - 1,
                       arr1.shape[3] + arr2.shape[3] - 1,
                       arr1.shape[4] + arr2.shape[4] - 1,
                       arr1.shape[5] + arr2.shape[5] - 1,
                       arr1.shape[6]), dtype=np.float64)
    for i1 in range(arr1.shape[0]):
        for i2 in range(arr1.shape[1]):
            for i3 in range(arr1.shape[2]):
                for i4 in range(arr1.shape[3]):
                    for i5 in range(arr1.shape[4]):
                        for i6 in range(arr1.shape[5]):
                            for j1 in range(arr2.shape[0]):
                                for j2 in range(arr2.shape[1]):
                                    for j3 in range(arr2.shape[2]):
                                        for j4 in range(arr2.shape[3]):
                                            for j5 in range(arr2.shape[4]):
                                                for j6 in range(arr2.shape[5]):
                                                    for i0 in range(arr1.shape[6]):
                                                        result[i1 + j1, i2 + j2, i3 + j3, i4 + j4, i5 + j5, i6 + j6, i0] += \
                                                            arr1[i1, i2, i3, i4, i5, i6, i0] * \
                                                            arr2[j1, j2, j3, j4, j5, j6, i0]
    return result


@njit(cache=True, fastmath=True)
def mat_element_left(n, m, k):
    if k > m:
        return 0
    ans = 0
    for j in range(n - m + 1):
        ans += binomial[n - k, j] * binomial[n - k - j, m - k]
    ans *= (binomial[n, k] / binomial[n, m])
    return ans


@njit(cache=True, fastmath=True)
def mat_element_right(n, m, k):
    if k < m:
        return 0
    ans = 0
    for j in range(m + 1):
        ans += binomial[k, j] * binomial[k - j, m - j]
    ans *= (binomial[n, k] / binomial[n, m])
    return ans


@njit(cache=True, fastmath=True)
def matrix_left(n, mat_left):
    if n in mat_left:
        return mat_left[n]

    matrix = np.zeros((n + 1, n + 1), dtype=np.float64)
    for m in range(n + 1):
        for k in range(n + 1):
            matrix[m, k] = mat_element_left(n, m, k)

    matrix *= 0.5 ** n
    matrix = np.ascontiguousarray(matrix)
    mat_left[n] = matrix

    return matrix


@njit(cache=True, fastmath=True)
def matrix_right(n, mat_right):
    if n in mat_right:
        return mat_right[n]

    matrix = np.zeros((n + 1, n + 1), dtype=np.float64)
    for m in range(n + 1):
        for k in range(n + 1):
            matrix[m, k] = mat_element_right(n, m, k)

    matrix *= 0.5 ** n
    matrix = np.ascontiguousarray(matrix)
    mat_right[n] = matrix

    return matrix


@njit(cache=True, fastmath=True)
def bpoly_nb_change_variable(a: np.ndarray, u_left: bool, v_left: bool, mat_left: np.ndarray, mat_right: np.ndarray) -> np.ndarray:
    n = np.array(a.shape, dtype=np.int64) - 1

    ans = np.zeros(a.shape, dtype=np.float64)
    mat = np.ascontiguousarray(matrix_left(
        n[4], mat_left) if u_left else matrix_right(n[4], mat_right))
    for i in range(ans.shape[0]):
        for j in range(ans.shape[1]):
            for k in range(ans.shape[2]):
                for l in range(ans.shape[3]):
                    temp_a = np.ascontiguousarray(a[i, j, k, l, :, :])
                    ans[i, j, k, l, :, :] = mat @ temp_a
    ans = np.transpose(ans, (0, 1, 2, 3, 5, 4))
    mat = np.ascontiguousarray(matrix_left(
        n[5], mat_left) if v_left else matrix_right(n[5], mat_right))
    for i in range(ans.shape[0]):
        for j in range(ans.shape[1]):
            for k in range(ans.shape[2]):
                for l in range(ans.shape[3]):
                    temp_ans = np.ascontiguousarray(ans[i, j, k, l, :, :])
                    ans[i, j, k, l, :, :] = mat @ temp_ans
    ans = np.transpose(ans, (0, 1, 2, 3, 5, 4))
    return ans


@njit(cache=True, fastmath=True)
def batch_bpoly_nb_change_variable(a, u_left, v_left, mat_left, mat_right):
    batch_size = a.shape[-1]
    n = np.array(a.shape[:-1], dtype=np.int64) - 1

    ans = np.zeros(a.shape, dtype=np.float64)
    mat = np.empty((batch_size, n[4] + 1, n[4] + 1), dtype=np.float64)
    for i in range(batch_size):
        mat[i] = matrix_left(n[4], mat_left) if u_left[i] else matrix_right(n[4], mat_right)
    for i in range(ans.shape[0]):
        for j in range(ans.shape[1]):
            for k in range(ans.shape[2]):
                for l in range(ans.shape[3]):
                    for o in range(batch_size):
                        temp_a = np.ascontiguousarray(a[i, j, k, l, :, :, o])
                        temp_mat = np.ascontiguousarray(mat[o])
                        ans[i, j, k, l, :, :, o] = temp_mat @ temp_a
    ans = np.transpose(ans, (0, 1, 2, 3, 5, 4, 6))
    mat = np.empty((batch_size, n[5] + 1, n[5] + 1), dtype=np.float64)
    for i in range(batch_size):
        mat[i] = matrix_left(n[5], mat_left) if v_left[i] else matrix_right(n[5], mat_right)
    for i in range(ans.shape[0]):
        for j in range(ans.shape[1]):
            for k in range(ans.shape[2]):
                for l in range(ans.shape[3]):
                    for o in range(batch_size):
                        temp_ans = np.ascontiguousarray(ans[i, j, k, l, :, :, o])
                        temp_mat = np.ascontiguousarray(mat[o])
                        ans[i, j, k, l, :, :, o] = temp_mat @ temp_ans
    ans = np.transpose(ans, (0, 1, 2, 3, 5, 4, 6))
    return ans


@njit(cache=True, fastmath=True)
def bpoly_nb_eval(a, x_):
    x = x_[:]
    if len(x) == 1:
        x = np.array([1, 1, 1, 1, x[0], 1], dtype=np.float64)
    elif len(x) == 2:
        x = np.array([1, 1, 1, 1, x[0], x[1]], dtype=np.float64)
    elif len(x) == 3:
        x = np.array([1, 1, 1, x[0], x[1], x[2]], dtype=np.float64)
    elif len(x) == 4:
        x = np.array([1, 1, x[0], x[1], x[2], x[3]], dtype=np.float64)
    elif len(x) == 5:
        x = np.array([1, x[0], x[1], x[2], x[3], x[4]], dtype=np.float64)

    n = np.array(a.shape) - 1
    ans = 0
    for i1 in range(n[0] + 1):
        for i2 in range(n[1] + 1):
            for i3 in range(n[2] + 1):
                for i4 in range(n[3] + 1):
                    for i5 in range(n[4] + 1):
                        for i6 in range(n[5] + 1):
                            ans += a[i1, i2, i3, i4, i5, i6] * evalB_basis(n[0], i1, x[0]) * evalB_basis(n[1], i2, x[1]) * evalB_basis(
                                n[2], i3, x[2]) * evalB_basis(n[3], i4, x[3]) * evalB_basis(n[4], i5, x[4]) * evalB_basis(n[5], i6, x[5])

    return ans


@njit(cache=True, fastmath=True)
def bpoly_nb_elevate(a, target_degree):
    n = np.array(a.shape) - 1
    r = np.array([target_degree[0] - n[0], target_degree[1] - n[1], target_degree[2] -
                 n[2], target_degree[3] - n[3], target_degree[4] - n[4], target_degree[5] - n[5]])

    a_wrt_scaled_basis = np.zeros(
        (n[0] + 1, n[1] + 1, n[2] + 1, n[3] + 1, n[4] + 1, n[5] + 1), dtype=np.float64)
    for i1 in range(n[0] + 1):
        for i2 in range(n[1] + 1):
            for i3 in range(n[2] + 1):
                for i4 in range(n[3] + 1):
                    for i5 in range(n[4] + 1):
                        for i6 in range(n[5] + 1):
                            a_wrt_scaled_basis[i1, i2, i3, i4, i5, i6] = binomial[n[0], i1] * binomial[n[1], i2] * binomial[n[2],
                                                                                                                            i3] * binomial[n[3], i4] * binomial[n[4], i5] * binomial[n[5], i6] * a[i1, i2, i3, i4, i5, i6]
    binomial_array = np.zeros(
        (r[0] + 1, r[1] + 1, r[2] + 1, r[3] + 1, r[4] + 1, r[5] + 1), dtype=np.float64)
    for i1 in range(r[0] + 1):
        for i2 in range(r[1] + 1):
            for i3 in range(r[2] + 1):
                for i4 in range(r[3] + 1):
                    for i5 in range(r[4] + 1):
                        for i6 in range(r[5] + 1):
                            binomial_array[i1, i2, i3, i4, i5, i6] = binomial[r[0], i1] * binomial[r[1], i2] * \
                                binomial[r[2], i3] * binomial[r[3], i4] * \
                                binomial[r[4], i5] * binomial[r[5], i6]

    a_new = nb_convolve(binomial_array, a_wrt_scaled_basis)
    binomial_a = np.zeros((target_degree[0] + 1, target_degree[1] + 1, target_degree[2] + 1,
                          target_degree[3] + 1, target_degree[4] + 1, target_degree[5] + 1), dtype=np.float64)
    for i1 in range(target_degree[0] + 1):
        for i2 in range(target_degree[1] + 1):
            for i3 in range(target_degree[2] + 1):
                for i4 in range(target_degree[3] + 1):
                    for i5 in range(target_degree[4] + 1):
                        for i6 in range(target_degree[5] + 1):
                            binomial_a[i1, i2, i3, i4, i5, i6] = binomial[target_degree[0], i1] * binomial[target_degree[1], i2] * \
                                binomial[target_degree[2], i3] * binomial[target_degree[3], i4] * \
                                binomial[target_degree[4], i5] * \
                                binomial[target_degree[5], i6]

    a_new /= binomial_a
    return a_new


@njit(cache=True, fastmath=True)
def batch_bpoly_nb_elevate(a, target_degree):
    batch_size = a.shape[-1]
    n = np.array(a.shape[:-1]) - 1
    # r = target_degree - n
    r = np.array([target_degree[0] - n[0], target_degree[1] - n[1], target_degree[2] - n[2], target_degree[3] - n[3], target_degree[4] - n[4], target_degree[5] - n[5]])    

    a_wrt_scaled_basis = np.zeros(
        (n[0] + 1, n[1] + 1, n[2] + 1, n[3] + 1, n[4] + 1, n[5] + 1, batch_size), dtype=np.float64)

    for i1 in range(n[0] + 1):
        for i2 in range(n[1] + 1):
            for i3 in range(n[2] + 1):
                for i4 in range(n[3] + 1):
                    for i5 in range(n[4] + 1):
                        for i6 in range(n[5] + 1):
                            weight = binomial[n[0], i1] * binomial[n[1], i2] * binomial[n[2], i3] * binomial[n[3], i4] * binomial[n[4], i5] * binomial[n[5], i6]
                            for i0 in range(batch_size):
                                a_wrt_scaled_basis[i1, i2, i3, i4, i5, i6, i0] = weight * a[i1, i2, i3, i4, i5, i6, i0]
      

    binomial_array = np.zeros(
        (r[0] + 1, r[1] + 1, r[2] + 1, r[3] + 1, r[4] + 1, r[5] + 1, batch_size), dtype=np.float64)

    for i1 in range(r[0] + 1):
        for i2 in range(r[1] + 1):
            for i3 in range(r[2] + 1):
                for i4 in range(r[3] + 1):
                    for i5 in range(r[4] + 1):
                        for i6 in range(r[5] + 1):
                            weight = binomial[r[0], i1] * binomial[r[1], i2] * binomial[r[2], i3] * binomial[r[3], i4] * binomial[r[4], i5] * binomial[r[5], i6]
                            for i0 in range(batch_size):
                                binomial_array[i1, i2, i3, i4, i5, i6, i0] = weight

    a_new = batch_nb_convolve(binomial_array, a_wrt_scaled_basis)
    
    for i1 in range(target_degree[0] + 1):
        for i2 in range(target_degree[1] + 1):
            for i3 in range(target_degree[2] + 1):
                for i4 in range(target_degree[3] + 1):
                    for i5 in range(target_degree[4] + 1):
                        for i6 in range(target_degree[5] + 1):
                            weight = (
                                binomial[target_degree[0], i1] * binomial[target_degree[1], i2] * 
                                binomial[target_degree[2], i3] * binomial[target_degree[3], i4] * 
                                binomial[target_degree[4], i5] * binomial[target_degree[5], i6]
                                )
                            for i0 in range(batch_size):
                                a_new[i1, i2, i3, i4, i5, i6, i0] /= weight

    return a_new


@njit(cache=True, fastmath=True)
def bpoly_nb_add(a, other):
    n = np.maximum(np.array(a.shape), np.array(other.shape)) - 1
    a = a
    for i in range(len(a.shape)):
        if a.shape[i] != n[i] + 1:
            a = bpoly_nb_elevate(a, n)
            break
    b = other
    for i in range(len(other.shape)):
        if other.shape[i] != n[i] + 1:
            b = bpoly_nb_elevate(other, n)
            break
    return a + b


@njit(cache=True, fastmath=True)
def batch_bpoly_nb_add(a, b):
    a_shape = a.shape[:-1]
    b_shape = b.shape[:-1]
    n = np.maximum(np.array(a_shape), np.array(b_shape)) - 1
    
    ret_1 = np.zeros((n[0] + 1, n[1] + 1, n[2] + 1, n[3] + 1, n[4] + 1, n[5] + 1, a.shape[-1]), dtype=np.float64)
    if a_shape[0] != n[0] + 1 or a_shape[1] != n[1] + 1 or a_shape[2] != n[2] + 1 or a_shape[3] != n[3] + 1 or a_shape[4] != n[4] + 1 or a_shape[5] != n[5] + 1:
        ret_1 = batch_bpoly_nb_elevate(a, n)
    else:
        ret_1 = a

    ret_2 = np.zeros((n[0] + 1, n[1] + 1, n[2] + 1, n[3] + 1, n[4] + 1, n[5] + 1, a.shape[-1]), dtype=np.float64)
    if b_shape[0] != n[0] + 1 or b_shape[1] != n[1] + 1 or b_shape[2] != n[2] + 1 or b_shape[3] != n[3] + 1 or b_shape[4] != n[4] + 1 or b_shape[5] != n[5] + 1:
        ret_2 = batch_bpoly_nb_elevate(b, n)
    else:
        ret_2 = b

    ret = np.empty_like(ret_1)
    for i1 in range(n[0] + 1):
        for i2 in range(n[1] + 1):
            for i3 in range(n[2] + 1):
                for i4 in range(n[3] + 1):
                    for i5 in range(n[4] + 1):
                        for i6 in range(n[5] + 1):
                            for i0 in range(a.shape[-1]):
                                ret[i1, i2, i3, i4, i5, i6, i0] = ret_1[i1, i2, i3, i4, i5, i6, i0] + ret_2[i1, i2, i3, i4, i5, i6, i0] 

    return ret


@njit(cache=True, fastmath=True)
def bpoly_nb_sub(a, other):
    n = np.maximum(np.array(a.shape), np.array(other.shape)) - 1
    a = a
    for i in range(len(a.shape)):
        if a.shape[i] != n[i] + 1:
            a = bpoly_nb_elevate(a, n)
            break
    b = other
    for i in range(len(other.shape)):
        if other.shape[i] != n[i] + 1:
            b = bpoly_nb_elevate(other, n)
            break
    return a - b


@njit(cache=True, fastmath=True)
def batch_bpoly_nb_sub(a, b):
    a_shape = a.shape[:-1]
    b_shape = b.shape[:-1]
    n = np.maximum(np.array(a_shape), np.array(b_shape)) - 1
    ret_1 = a
    if a_shape[0] != n[0] + 1 or a_shape[1] != n[1] + 1 or a_shape[2] != n[2] + 1 or a_shape[3] != n[3] + 1 or a_shape[4] != n[4] + 1 or a_shape[5] != n[5] + 1:
        ret_1 = batch_bpoly_nb_elevate(a, n)

    ret_2 = b
    if b_shape[0] != n[0] + 1 or b_shape[1] != n[1] + 1 or b_shape[2] != n[2] + 1 or b_shape[3] != n[3] + 1 or b_shape[4] != n[4] + 1 or b_shape[5] != n[5] + 1:
        ret_2 = batch_bpoly_nb_elevate(b, n)

    return ret_1 - ret_2

@njit(cache=True, fastmath=True)
def bpoly_nb_mul(a, other):
    n = np.array(a.shape) - 1
    m = np.array(other.shape) - 1
    target_degree = n + m

    a_wrt_scaled_basis_1 = np.zeros(
        (n[0] + 1, n[1] + 1, n[2] + 1, n[3] + 1, n[4] + 1, n[5] + 1), dtype=np.float64)
    for i1 in range(n[0] + 1):
        for i2 in range(n[1] + 1):
            for i3 in range(n[2] + 1):
                for i4 in range(n[3] + 1):
                    for i5 in range(n[4] + 1):
                        for i6 in range(n[5] + 1):
                            a_wrt_scaled_basis_1[i1, i2, i3, i4, i5, i6] = binomial[n[0], i1] * binomial[n[1], i2] * binomial[n[2],
                                                                                                                              i3] * binomial[n[3], i4] * binomial[n[4], i5] * binomial[n[5], i6] * a[i1, i2, i3, i4, i5, i6]
    a_wrt_scales_basis_2 = np.zeros(
        (m[0] + 1, m[1] + 1, m[2] + 1, m[3] + 1, m[4] + 1, m[5] + 1), dtype=np.float64)
    for i1 in range(m[0] + 1):
        for i2 in range(m[1] + 1):
            for i3 in range(m[2] + 1):
                for i4 in range(m[3] + 1):
                    for i5 in range(m[4] + 1):
                        for i6 in range(m[5] + 1):
                            a_wrt_scales_basis_2[i1, i2, i3, i4, i5, i6] = binomial[m[0], i1] * binomial[m[1], i2] * binomial[m[2],
                                                                                                                              i3] * binomial[m[3], i4] * binomial[m[4], i5] * binomial[m[5], i6] * other[i1, i2, i3, i4, i5, i6]

    a_new = nb_convolve(a_wrt_scaled_basis_1, a_wrt_scales_basis_2)

    binomial_a = np.zeros((target_degree[0] + 1, target_degree[1] + 1, target_degree[2] + 1,
                          target_degree[3] + 1, target_degree[4] + 1, target_degree[5] + 1), np.float64)
    for i1 in range(target_degree[0] + 1):
        for i2 in range(target_degree[1] + 1):
            for i3 in range(target_degree[2] + 1):
                for i4 in range(target_degree[3] + 1):
                    for i5 in range(target_degree[4] + 1):
                        for i6 in range(target_degree[5] + 1):
                            binomial_a[i1, i2, i3, i4, i5, i6] = binomial[target_degree[0], i1] * binomial[target_degree[1], i2] * \
                                binomial[target_degree[2], i3] * binomial[target_degree[3], i4] * \
                                binomial[target_degree[4], i5] * \
                                binomial[target_degree[5], i6]
    
    a_new /= binomial_a
    return a_new


@njit(cache=True, fastmath=True)
def batch_bpoly_nb_mul(a, b):
    batch_size = a.shape[-1]
    n = np.array(a.shape[:-1]) - 1
    m = np.array(b.shape[:-1]) - 1
    target_degree = n + m

    a_wrt_scaled_basis_1 = np.zeros(
        (n[0] + 1, n[1] + 1, n[2] + 1, n[3] + 1, n[4] + 1, n[5] + 1, batch_size), dtype=np.float64)

    for i1 in range(n[0] + 1):
        for i2 in range(n[1] + 1):
            for i3 in range(n[2] + 1):
                for i4 in range(n[3] + 1):
                    for i5 in range(n[4] + 1):
                        for i6 in range(n[5] + 1):
                            weight = binomial[n[0], i1] * binomial[n[1], i2] * binomial[n[2], i3] * binomial[n[3], i4] * binomial[n[4], i5] * binomial[n[5], i6]
                            for i0 in range(batch_size):
                                a_wrt_scaled_basis_1[i1, i2, i3, i4, i5, i6, i0] = weight * a[i1, i2, i3, i4, i5, i6, i0]

    a_wrt_scaled_basis_2 = np.zeros(
        (m[0] + 1, m[1] + 1, m[2] + 1, m[3] + 1, m[4] + 1, m[5] + 1, batch_size), dtype=np.float64)

    for i1 in range(m[0] + 1):
        for i2 in range(m[1] + 1):
            for i3 in range(m[2] + 1):
                for i4 in range(m[3] + 1):
                    for i5 in range(m[4] + 1):
                        for i6 in range(m[5] + 1):
                            weight = binomial[m[0], i1] * binomial[m[1], i2] * binomial[m[2], i3] * binomial[m[3], i4] * binomial[m[4], i5] * binomial[m[5], i6]
                            for i0 in range(batch_size):
                                a_wrt_scaled_basis_2[i1, i2, i3, i4, i5, i6, i0] = weight * b[i1, i2, i3, i4, i5, i6, i0]

    a_new = batch_nb_convolve(a_wrt_scaled_basis_1, a_wrt_scaled_basis_2)

    for i1 in range(target_degree[0] + 1):
        for i2 in range(target_degree[1] + 1):
            for i3 in range(target_degree[2] + 1):
                for i4 in range(target_degree[3] + 1):
                    for i5 in range(target_degree[4] + 1):
                        for i6 in range(target_degree[5] + 1):
                            weight = (
                                binomial[target_degree[0], i1] * binomial[target_degree[1], i2] * 
                                binomial[target_degree[2], i3] * binomial[target_degree[3], i4] * 
                                binomial[target_degree[4], i5] * binomial[target_degree[5], i6]
                                )
                            for i0 in range(batch_size):
                                a_new[i1, i2, i3, i4, i5, i6, i0] /= weight

    return a_new


@njit(cache=True, fastmath=True)
def bpoly_nb_du(a):
    n = np.array(a.shape) - 1
    if n[4] == 0:
        return np.zeros((1, 1, 1, 1, 1, 1), dtype=np.float64)

    a_new = np.zeros((n[0] + 1, n[1] + 1, n[2] + 1, n[3] + 1, n[4], n[5] + 1), dtype=np.float64)

    for i1 in range(n[0] + 1):
        for i2 in range(n[1] + 1):
            for i3 in range(n[2] + 1):
                for i4 in range(n[3] + 1):
                    for k in range(n[4]):
                        for i6 in range(n[5] + 1):
                            a_new[i1, i2, i3, i4, k, i6] = n[4] * \
                                (a[i1, i2, i3, i4, k + 1, i6] - a[i1, i2, i3, i4, k, i6])

    return a_new


@njit(cache=True, fastmath=True)
def batch_bpoly_nb_du(a):
    batch_size = a.shape[-1]
    n = np.array(a.shape[:-1]) - 1

    a_new = np.zeros((n[0] + 1, n[1] + 1, n[2] + 1, n[3] + 1, n[4], n[5] + 1, batch_size), dtype=np.float64)

    for i1 in range(n[0] + 1):
        for i2 in range(n[1] + 1):
            for i3 in range(n[2] + 1):
                for i4 in range(n[3] + 1):
                    for k in range(n[4]):
                        for i6 in range(n[5] + 1):
                            for i0 in range(batch_size):
                                a_new[i1, i2, i3, i4, k, i6, i0] = n[4] * (a[i1, i2, i3, i4, k + 1, i6, i0] - a[i1, i2, i3, i4, k, i6, i0])

    return a_new


@njit(cache=True, fastmath=True)
def bpoly_nb_dv(a):
    n = np.array(a.shape) - 1
    if n[5] == 0:
        return np.zeros((1, 1, 1, 1, 1, 1), dtype=np.float64)

    a_new = np.zeros((n[0] + 1, n[1] + 1, n[2] + 1, n[3] + 1, n[4] + 1, n[5]), dtype=np.float64)

    for i1 in range(n[0] + 1):
        for i2 in range(n[1] + 1):
            for i3 in range(n[2] + 1):
                for i4 in range(n[3] + 1):
                    for i5 in range(n[4] + 1):
                        for k in range(n[5]):
                            a_new[i1, i2, i3, i4, i5, k] = n[5] * \
                                (a[i1, i2, i3, i4, i5, k + 1] - a[i1, i2, i3, i4, i5, k])

    return a_new


@njit(cache=True, fastmath=True)
def batch_bpoly_nb_dv(a):
    batch_size = a.shape[-1]
    n = np.array(a.shape[:-1]) - 1

    a_new = np.zeros((n[0] + 1, n[1] + 1, n[2] + 1, n[3] + 1, n[4] + 1, n[5], batch_size), dtype=np.float64)

    for i1 in range(n[0] + 1):
        for i2 in range(n[1] + 1):
            for i3 in range(n[2] + 1):
                for i4 in range(n[3] + 1):
                    for i5 in range(n[4] + 1):
                        for k in range(n[5]):
                            for i0 in range(batch_size):
                                a_new[i1, i2, i3, i4, i5, k, i0] = n[5] * (a[i1, i2, i3, i4, i5, k + 1, i0] - a[i1, i2, i3, i4, i5, k, i0])

    return a_new


@njit(cache=True, fastmath=True)
def bpoly_nb_dx(a):
    n = np.array(a.shape) - 1
    if n[2] == 0:
        return np.zeros((1, 1, 1, 1, 1, 1), dtype=np.float64)

    a_new = np.zeros((n[0] + 1, n[1] + 1, n[2], n[3] + 1, n[4] + 1, n[5] + 1), dtype=np.float64)

    for i1 in range(n[0] + 1):
        for i2 in range(n[1] + 1):
            for k in range(n[2]):
                for i4 in range(n[3] + 1):
                    for i5 in range(n[4] + 1):
                        for i6 in range(n[5] + 1):
                            a_new[i1, i2, k, i4, i5, i6] = n[2] * \
                                (a[i1, i2, k + 1, i4, i5, i6] -
                                 a[i1, i2, k, i4, i5, i6])

    return a_new


@njit(cache=True, fastmath=True)
def batch_bpoly_nb_dx(a):
    batch_size = a.shape[-1]
    n = np.array(a.shape[:-1]) - 1

    a_new = np.zeros((n[0] + 1, n[1] + 1, n[2], n[3] + 1, n[4] + 1, n[5] + 1, batch_size), dtype=np.float64)

    for i1 in range(n[0] + 1):
        for i2 in range(n[1] + 1):
            for k in range(n[2]):
                for i4 in range(n[3] + 1):
                    for i5 in range(n[4] + 1):
                        for i6 in range(n[5] + 1):
                            for i0 in range(batch_size):
                                a_new[i1, i2, k, i4, i5, i6, i0] = n[2] * \
                                    (a[i1, i2, k + 1, i4, i5, i6, i0] - a[i1, i2, k, i4, i5, i6, i0])

    return a_new


@njit(cache=True, fastmath=True)
def bpoly_nb_dy(a):
    n = np.array(a.shape) - 1
    if n[3] == 0:
        return np.zeros((1, 1, 1, 1, 1, 1), dtype=np.float64)

    a_new = np.zeros((n[0] + 1, n[1] + 1, n[2] + 1, n[3],
                     n[4] + 1, n[5] + 1), dtype=np.float64)

    for i1 in range(n[0] + 1):
        for i2 in range(n[1] + 1):
            for i3 in range(n[2] + 1):
                for k in range(n[3]):
                    for i5 in range(n[4] + 1):
                        for i6 in range(n[5] + 1):
                            a_new[i1, i2, i3, k, i5, i6] = n[3] * \
                                (a[i1, i2, i3, k + 1, i5, i6] -
                                 a[i1, i2, i3, k, i5, i6])

    return a_new


@njit(cache=True, fastmath=True)
def batch_bpoly_nb_dy(a):
    batch_size = a.shape[-1]
    n = np.array(a.shape[:-1]) - 1

    a_new = np.zeros((n[0] + 1, n[1] + 1, n[2] + 1, n[3], n[4] + 1, n[5] + 1, batch_size), dtype=np.float64)

    for i1 in range(n[0] + 1):
        for i2 in range(n[1] + 1):
            for i3 in range(n[2] + 1):
                for k in range(n[3]):
                    for i5 in range(n[4] + 1):
                        for i6 in range(n[5] + 1):
                            for i0 in range(batch_size):
                                a_new[i1, i2, i3, k, i5, i6, i0] = n[3] * \
                                    (a[i1, i2, i3, k + 1, i5, i6, i0] - a[i1, i2, i3, k, i5, i6, i0])

    return a_new


@njit(cache=True, fastmath=True)
def resolve_a(a_):
    if a_.ndim == 0:
        self_a = a_.reshape((1, 1, 1, 1, 1, 1))
    elif a_.ndim == 1:
        self_a = a_.reshape((1, 1, 1, 1, a_.size, 1))
    elif a_.ndim == 2:
        self_a = a_.reshape((1, 1, 1, 1, a_.shape[0], a_.shape[1]))
    elif a_.ndim == 3:
        self_a = a_.reshape((1, 1, 1, a_.shape[0], a_.shape[1], a_.shape[2]))
    elif a_.ndim == 4:
        self_a = a_.reshape(
            (1, 1, a_.shape[0], a_.shape[1], a_.shape[2], a_.shape[3]))
    elif a_.ndim == 5:
        self_a = a_.reshape(
            (1, a_.shape[0], a_.shape[1], a_.shape[2], a_.shape[3], a_.shape[4]))
    else:
        self_a = a_
    return self_a


spec = [('a', float64[:, :, :, :, :, :]),]


@jitclass(spec=spec)
class BPoly:
    def __init__(self, a_):
        self.a = resolve_a(a_)

    def eval(self, x_):
        return bpoly_nb_eval(self.a, x_)

    def __add__(self, other):
        return BPoly(bpoly_nb_add(self.a, other.a))

    def __sub__(self, other):
        return BPoly(bpoly_nb_sub(self.a, other.a))

    def __mul__(self, other):
        return BPoly(bpoly_nb_mul(self.a, other.a))

    def elevate(self, target_degree):
        return BPoly(bpoly_nb_elevate(self.a, target_degree))

    def du(self):
        return BPoly(bpoly_nb_du(self.a))

    def dv(self):
        return BPoly(bpoly_nb_dv(self.a))

    def dx(self):
        return BPoly(bpoly_nb_dx(self.a))

    def dy(self):
        return BPoly(bpoly_nb_dy(self.a))

    def subdiv(self, u_left, v_left, mat_left, mat_right):  # the last 2 dims
        return BPoly(bpoly_nb_change_variable(self.a, u_left, v_left, mat_left, mat_right))

    def subdiv_mid(self, x_left, y_left, mat_left, mat_right):  # the mid 2 dims
        p = BPoly(np.transpose(self.a, (0, 1, 4, 5, 2, 3)))
        p = p.subdiv(x_left, y_left, mat_left, mat_right)
        p.a = np.transpose(p.a, (0, 1, 4, 5, 2, 3))
        return p

    def bound(self):
        return np.min(self.a), np.max(self.a)

    def align_degree_to(self, target_bpoly):
        target_degree = np.array(target_bpoly.a.shape) - 1
        src_degree = np.array(self.a.shape) - 1
        assert np.all(target_degree >= src_degree)
        return self.elevate(target_degree)

    def fbound(self, denom):  # fraction bound given a denominator
        assert self.a.shape == denom.a.shape
        frac_a = self.a / denom.a
        # frac_a = np.nan_to_num(frac_a, nan=0, posinf=0, neginf=0)
        return np.min(frac_a), np.max(frac_a)


spec = [('x', float64[:, :, :, :, :, :]),
        ('y', float64[:, :, :, :, :, :]), ('z', float64[:, :, :, :, :, :])]


@jitclass(spec)
class BPoly3:
    def __init__(self, a):
        self.x = resolve_a(a[0])
        self.y = resolve_a(a[1])
        self.z = resolve_a(a[2])

    def eval(self, x_):
        return np.stack((bpoly_nb_eval(self.x, x_), bpoly_nb_eval(self.y, x_), bpoly_nb_eval(self.z, x_)))

    def __add__(self, other):
        return BPoly3(np.stack((bpoly_nb_add(self.x, other.x), bpoly_nb_add(self.y, other.y), bpoly_nb_add(self.z, other.z))))

    def __sub__(self, other):
        return BPoly3(np.stack((bpoly_nb_sub(self.x, other.x), bpoly_nb_sub(self.y, other.y), bpoly_nb_sub(self.z, other.z))))

    def __mul__(self, other):
        if isinstance(other, BPoly3):
            return BPoly3(np.stack((bpoly_nb_mul(self.x, other.x), bpoly_nb_mul(self.y, other.y), bpoly_nb_mul(self.z, other.z))))
        elif isinstance(other, BPoly):
            return BPoly3(np.stack((bpoly_nb_mul(self.x, other.a), bpoly_nb_mul(self.y, other.a), bpoly_nb_mul(self.z, other.a))))

    def du(self):
        return BPoly3(np.stack((bpoly_nb_du(self.x), bpoly_nb_du(self.y), bpoly_nb_du(self.z))))

    def dv(self):
        return BPoly3(np.stack((bpoly_nb_dv(self.x), bpoly_nb_dv(self.y), bpoly_nb_dv(self.z))))

    def dx(self):
        return BPoly3(np.stack((bpoly_nb_dx(self.x), bpoly_nb_dx(self.y), bpoly_nb_dx(self.z))))

    def dy(self):
        return BPoly3(np.stack((bpoly_nb_dy(self.x), bpoly_nb_dy(self.y), bpoly_nb_dy(self.z))))

    def dot(self, other):
        return BPoly(bpoly_nb_add(bpoly_nb_mul(self.x, other.x), bpoly_nb_add(bpoly_nb_mul(self.y, other.y), bpoly_nb_mul(self.z, other.z))))

    def cross(self, other):
        return BPoly3(np.stack((bpoly_nb_sub(bpoly_nb_mul(self.y, other.z), bpoly_nb_mul(self.z, other.y)), bpoly_nb_sub(bpoly_nb_mul(self.z, other.x), bpoly_nb_mul(self.x, other.z)), bpoly_nb_sub(bpoly_nb_mul(self.x, other.y), bpoly_nb_mul(self.y, other.x)))))


@njit(cache=True, fastmath=True)
def batch_resolve_a(a_):
    if a_.ndim == 1:
        self_a = a_.reshape((1, 1, 1, 1, 1, 1, a_.size))
    elif a_.ndim == 2:
        self_a = a_.reshape((1, 1, 1, 1, a_.shape[0], 1, a_.shape[1]))
    elif a_.ndim == 3:
        self_a = a_.reshape((1, 1, 1, 1, a_.shape[0], a_.shape[1], a_.shape[2]))
    elif a_.ndim == 4:
        self_a = a_.reshape((1, 1, 1, a_.shape[0], a_.shape[1], a_.shape[2], a_.shape[3]))
    elif a_.ndim == 5:
        self_a = a_.reshape((1, 1, a_.shape[0], a_.shape[1], a_.shape[2], a_.shape[3], a_.shape[4]))
    elif a_.ndim == 6:
        self_a = a_.reshape((1, a_.shape[0], a_.shape[1], a_.shape[2], a_.shape[3], a_.shape[4], a_.shape[5]))
    else:
        self_a = a_
    return self_a


spec = [('a', float64[:, :, :, :, :, :, :])]

@jitclass(spec)
class BatchBPoly:
    def __init__(self, batch_a):
        self.a = batch_resolve_a(batch_a)

    def __add__(self, other):
        return BatchBPoly(batch_bpoly_nb_add(self.a, other.a))

    def __sub__(self, other):
        return BatchBPoly(batch_bpoly_nb_sub(self.a, other.a))

    def __mul__(self, other):
        if isinstance(other, BatchBPoly):
            return BatchBPoly(batch_bpoly_nb_mul(self.a, other.a))
        elif isinstance(other, BPoly):
            batchh_size = self.a.shape[-1]
            b = np.zeros(other.a.shape + (batchh_size,), dtype=np.float64)
            for i0 in range(other.a.shape[0]):
                for i1 in range(other.a.shape[1]):
                    for i2 in range(other.a.shape[2]):
                        for i3 in range(other.a.shape[3]):
                            for i4 in range(other.a.shape[4]):
                                for i5 in range(other.a.shape[5]):
                                    for i6 in range(batchh_size):
                                        b[i0, i1, i2, i3, i4, i5, i6] = other.a[i0, i1, i2, i3, i4, i5]
            return BatchBPoly(batch_bpoly_nb_mul(self.a, b))

        b = np.zeros(self.a.shape, dtype=np.float64)
        return BatchBPoly(batch_bpoly_nb_mul(self.a, b))

    def elevate(self, target_degree):
        return BatchBPoly(batch_bpoly_nb_elevate(self.a, target_degree))

    def du(self):
        n = np.array(self.a.shape[:-1]) - 1
        if n[4] == 0:
            shape = (1, 1, 1, 1, 1, 1, self.a.shape[-1])
            return BatchBPoly(np.zeros(shape, dtype=np.float64))
    
        return BatchBPoly(batch_bpoly_nb_du(self.a))

    def dv(self):
        n = np.array(self.a.shape[:-1]) - 1
        if n[5] == 0:
            shape = (1, 1, 1, 1, 1, 1, self.a.shape[-1])
            return BatchBPoly(np.zeros(shape, dtype=np.float64))
    
        return BatchBPoly(batch_bpoly_nb_dv(self.a))

    def dx(self):
        n = np.array(self.a.shape[:-1]) - 1
        if n[2] == 0:
            shape = (1, 1, 1, 1, 1, 1, self.a.shape[-1])
            return BatchBPoly(np.zeros(shape, dtype=np.float64))
    
        return BatchBPoly(batch_bpoly_nb_dx(self.a))

    def dy(self):
        n = np.array(self.a.shape[:-1]) - 1
        if n[3] == 0:
            shape = (1, 1, 1, 1, 1, 1, self.a.shape[-1])
            return BatchBPoly(np.zeros(shape, dtype=np.float64))
    
        return BatchBPoly(batch_bpoly_nb_dy(self.a))

    def subdiv(self, u_left, v_left, mat_left, mat_right):  # the last 2 dims
        return BatchBPoly(batch_bpoly_nb_change_variable(self.a, u_left, v_left, mat_left, mat_right))

    def subdiv_mid(self, x_left, y_left, mat_left, mat_right):  # the mid 2 dims
        p = BatchBPoly(np.transpose(self.a, (0, 1, 4, 5, 2, 3, 6)))
        p = p.subdiv(x_left, y_left, mat_left, mat_right)
        p.a = np.transpose(p.a, (0, 1, 4, 5, 2, 3, 6))
        return p

    def bound(self):
        shape = self.a.shape
        batch_size = shape[-1]
        ret = np.empty((2, batch_size), dtype=np.float64)
        for i in range(batch_size):
            max = self.a[0, 0, 0, 0, 0, 0, i]
            min = self.a[0, 0, 0, 0, 0, 0, i]
            for i1 in range(shape[0]):
                for i2 in range(shape[1]):
                    for i3 in range(shape[2]):
                        for i4 in range(shape[3]):
                            for i5 in range(shape[4]):
                                for i6 in range(shape[5]):
                                    max = max if max >= self.a[i1, i2, i3, i4, i5, i6, i] else self.a[i1, i2, i3, i4, i5, i6, i]
                                    min = min if min <= self.a[i1, i2, i3, i4, i5, i6, i] else self.a[i1, i2, i3, i4, i5, i6, i]
            ret[0, i] = min
            ret[1, i] = max
        return ret

    def align_degree_to(self, target_bpoly):
        target_degree = target_bpoly.a.shape[:-1]
        return self.elevate([target_degree[0] - 1, target_degree[1] - 1, target_degree[2] - 1, target_degree[3] - 1, target_degree[4] - 1, target_degree[5] - 1])

    def fbound(self, denom):
        shape = self.a.shape
        batch_size = shape[-1]
        ret = np.empty((2, batch_size), dtype=np.float64)
        frac_a = self.a / denom.a
        for i in range(batch_size):
            ret[0, i] = np.min(frac_a[..., i])
            ret[1, i] = np.max(frac_a[..., i])
        return ret


spec = [('x', float64[:, :, :, :, :, :, :]),
        ('y', float64[:, :, :, :, :, :, :]),
        ('z', float64[:, :, :, :, :, :, :])]

@jitclass(spec)
class BatchBPoly3:
    def __init__(self, batch_a):
        self.x = batch_resolve_a(batch_a[0])
        self.y = batch_resolve_a(batch_a[1])
        self.z = batch_resolve_a(batch_a[2])

    def __add__(self, other):
        return BatchBPoly3(np.stack((batch_bpoly_nb_add(self.x, other.x), batch_bpoly_nb_add(self.y, other.y), batch_bpoly_nb_add(self.z, other.z))))

    def __sub__(self, other):
        return BatchBPoly3(np.stack((batch_bpoly_nb_sub(self.x, other.x), batch_bpoly_nb_sub(self.y, other.y), batch_bpoly_nb_sub(self.z, other.z))))

    def __mul__(self, other):
        if isinstance(other, BPoly3):
            shape = other.x.shape + (self.x.shape[-1],)
            x = np.empty(shape, dtype=np.float64)
            y = np.empty(shape, dtype=np.float64)
            z = np.empty(shape, dtype=np.float64)
            for i1 in range(shape[0]):
                for i2 in range(shape[1]):
                    for i3 in range(shape[2]):
                        for i4 in range(shape[3]):
                            for i5 in range(shape[4]):
                                for i6 in range(shape[5]):
                                    for i0 in range(shape[6]):
                                        x[i1, i2, i3, i4, i5, i6, i0] = other.x[i1, i2, i3, i4, i5, i6]
                                        y[i1, i2, i3, i4, i5, i6, i0] = other.y[i1, i2, i3, i4, i5, i6]
                                        z[i1, i2, i3, i4, i5, i6, i0] = other.z[i1, i2, i3, i4, i5, i6]

            return BatchBPoly3(np.stack((batch_bpoly_nb_mul(self.x, x), batch_bpoly_nb_mul(self.y, y), batch_bpoly_nb_mul(self.z, z))))
        elif isinstance(other, BPoly):
            shape = other.a.shape + (self.x.shape[-1],)
            x = np.empty(shape, dtype=np.float64)
            y = np.empty(shape, dtype=np.float64)
            z = np.empty(shape, dtype=np.float64)
            for i1 in range(shape[0]):
                for i2 in range(shape[1]):
                    for i3 in range(shape[2]):
                        for i4 in range(shape[3]):
                            for i5 in range(shape[4]):
                                for i6 in range(shape[5]):
                                    for i0 in range(shape[6]):
                                        x[i1, i2, i3, i4, i5, i6, i0] = other.a[i1, i2, i3, i4, i5, i6]
                                        y[i1, i2, i3, i4, i5, i6, i0] = other.a[i1, i2, i3, i4, i5, i6]
                                        z[i1, i2, i3, i4, i5, i6, i0] = other.a[i1, i2, i3, i4, i5, i6]
            return BatchBPoly3(np.stack((batch_bpoly_nb_mul(self.x, x), batch_bpoly_nb_mul(self.y, y), batch_bpoly_nb_mul(self.z, z))))
        elif isinstance(other, BatchBPoly):
            return BatchBPoly3(np.stack((batch_bpoly_nb_mul(self.x, other.a), batch_bpoly_nb_mul(self.y, other.a), batch_bpoly_nb_mul(self.z, other.a))))
        else:  # BatchBPoly3
            return BatchBPoly3(np.stack((batch_bpoly_nb_mul(self.x, other.x), batch_bpoly_nb_mul(self.y, other.y), batch_bpoly_nb_mul(self.z, other.z))))

    def du(self):
        n = np.array(self.x.shape[:-1]) - 1
        if n[4] == 0:
            shape = (3, 1, 1, 1, 1, 1, 1, self.x.shape[-1])
            return BatchBPoly3(np.zeros(shape, dtype=np.float64))
    
        return BatchBPoly3(np.stack((batch_bpoly_nb_du(self.x), batch_bpoly_nb_du(self.y), batch_bpoly_nb_du(self.z))))

    def dv(self):
        n = np.array(self.x.shape[:-1]) - 1
        if n[5] == 0:
            shape = (3, 1, 1, 1, 1, 1, 1, self.x.shape[-1])
            return BatchBPoly3(np.zeros(shape, dtype=np.float64))
        
        return BatchBPoly3(np.stack((batch_bpoly_nb_dv(self.x), batch_bpoly_nb_dv(self.y), batch_bpoly_nb_dv(self.z))))

    def dx(self):
        n = np.array(self.x.shape[:-1]) - 1
        if n[2] == 0:
            shape = (3, 1, 1, 1, 1, 1, 1, self.x.shape[-1])
            return BatchBPoly3(np.zeros(shape, dtype=np.float64))
    
        return BatchBPoly3(np.stack((batch_bpoly_nb_dx(self.x), batch_bpoly_nb_dx(self.y), batch_bpoly_nb_dx(self.z))))


    def dy(self):
        n = np.array(self.x.shape[:-1]) - 1
        if n[3] == 0:
            shape = (3, 1, 1, 1, 1, 1, 1, self.x.shape[-1])
            return BatchBPoly3(np.zeros(shape, dtype=np.float64))
    
        return BatchBPoly3(np.stack((batch_bpoly_nb_dy(self.x), batch_bpoly_nb_dy(self.y), batch_bpoly_nb_dy(self.z))))


    def dot(self, other):
        return BatchBPoly(batch_bpoly_nb_add(
            batch_bpoly_nb_mul(self.x, other.x), \
            batch_bpoly_nb_add(batch_bpoly_nb_mul(self.y, other.y), \
                               batch_bpoly_nb_mul(self.z, other.z))))

    def cross(self, other):
        x = batch_bpoly_nb_sub(batch_bpoly_nb_mul(self.y, other.z), batch_bpoly_nb_mul(self.z, other.y))
        y = batch_bpoly_nb_sub(batch_bpoly_nb_mul(self.z, other.x), batch_bpoly_nb_mul(self.x, other.z))
        z = batch_bpoly_nb_sub(batch_bpoly_nb_mul(self.x, other.y), batch_bpoly_nb_mul(self.y, other.x))

        return BatchBPoly3(np.stack((x, y, z)))


if __name__ == "__main__":  # some test code to show numba performance issues. to be deleted
    @ njit(fastmath=True)
    def run(a, b, batch_size, mode=0):
        for j in range(batch_size):
            if mode == 0 or mode == 1:
                c = a[j] + b[j]
            if mode == 0 or mode == 2:
                c = a[j] - b[j]
            if mode == 0 or mode == 3:
                c = a[j] * b[j]

    @ njit(fastmath=True)
    def run_batch(a, b, mode=0):
        if mode == 0 or mode == 1:
            c = a + b
        if mode == 0 or mode == 2:
            c = a - b
        if mode == 0 or mode == 3:
            c = a * b

    @ njit(fastmath=True)
    def run_arr(a, b):
        for i in range(1000000):
            c = a+b

    def run_py(a, b):
        for i in range(1000000):
            c = a+b

    first = True

    for batch_size in [10]:
        for array_size in [8, 35]:
            x = [np.random.rand(array_size, array_size) for _ in range(batch_size)]
            y = [np.random.rand(array_size, array_size) for _ in range(batch_size)]
            z = [np.random.rand(array_size, array_size) for _ in range(batch_size)]

            a = [BPoly3(np.array([x[i], y[i], z[i]], dtype=np.float64)) for i in range(batch_size)]
            
            bx = np.empty((array_size, array_size, batch_size), dtype=np.float64)
            by = np.empty((array_size, array_size, batch_size), dtype=np.float64)
            bz = np.empty((array_size, array_size, batch_size), dtype=np.float64)
            for i in range(batch_size):
                for j in range(array_size):
                    for k in range(array_size):
                        bx[j, k, i] = x[i][j][k]
                        by[j, k, i] = y[i][j][k]
                        bz[j, k, i] = z[i][j][k]

            ba = BatchBPoly3(np.array([bx, by, bz], dtype=np.float64))

            x = [np.random.rand(array_size, array_size) for _ in range(batch_size)]
            y = [np.random.rand(array_size, array_size) for _ in range(batch_size)]
            z = [np.random.rand(array_size, array_size) for _ in range(batch_size)]

            b = [BPoly3(np.array([x[i], y[i], z[i]], dtype=np.float64)) for i in range(batch_size)]

            bx = np.empty((array_size, array_size, batch_size), dtype=np.float64)
            by = np.empty((array_size, array_size, batch_size), dtype=np.float64)
            bz = np.empty((array_size, array_size, batch_size), dtype=np.float64)
            for i in range(batch_size):
                for j in range(array_size):
                    for k in range(array_size):
                        bx[j, k, i] = x[i][j][k]
                        by[j, k, i] = y[i][j][k]
                        bz[j, k, i] = z[i][j][k]
                        
            bb = BatchBPoly3(np.array([bx, by, bz], dtype=np.float64))

            if first:
                run(a, b, batch_size)
                run_batch(ba, bb)
            first = False
            
            print("array_size: ", array_size, "batch_size: ", batch_size)
            
            print('***')
            # t0 = time.perf_counter()
            # for _ in range(10):
            #     run(a, b, batch_size, mode=3)
            # t = time.perf_counter()-t0
            # print("unbatch:")
            # print(t, " ms")
            
            t0 = time.perf_counter()
            for _ in range(100):
                run_batch(ba, bb, mode=3)
            t = time.perf_counter()-t0
            print("batch:")
            print(t, " ms")
            
            print('+++')
            # t0 = time.perf_counter()
            # for _ in range(10):
            #     run(a, b, batch_size, mode=1)
            # t = time.perf_counter()-t0
            # print("unbatch:")
            # print(t, " ms")
            
            t0 = time.perf_counter()
            for _ in range(100):
                run_batch(ba, bb, mode=1)
            t = time.perf_counter()-t0
            print("batch:")
            print(t, " ms")
            
            print('***')
            # t0 = time.perf_counter()
            # for _ in range(10):
            #     run(a, b, batch_size, mode=3)
            # t = time.perf_counter()-t0
            # print("unbatch:")
            # print(t, " ms")
            
            t0 = time.perf_counter()
            for _ in range(100):
                run_batch(ba, bb, mode=3)
            t = time.perf_counter()-t0
            print("batch:")
            print(t, " ms")

            print('+++')
            # t0 = time.perf_counter()
            # for _ in range(10):
            #     run(a, b, batch_size, mode=1)
            # t = time.perf_counter()-t0
            # print("unbatch:")
            # print(t, " ms")
            
            t0 = time.perf_counter()
            for _ in range(100):
                run_batch(ba, bb, mode=1)
            t = time.perf_counter()-t0
            print("batch:")
            print(t, " ms")
