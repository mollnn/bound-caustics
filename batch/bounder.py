# caustic bounding generator, the main algorithm
from alias import *
from transform import *
from formula import *
from utils import *

class Bounder:
    def __init__(self):
        self.batch_buf = np.zeros((BATCH_SIZE, RES, RES), dtype=np.float64)
        self.triangle_data = np.zeros((16, 3, BATCH_SIZE), dtype=np.float64)
        self.triangle_id_data = np.zeros((BATCH_SIZE, 2), dtype=np.int32)
        self.data_use_cnt = -np.ones(BATCH_SIZE, dtype=np.int32)
        self.max_batch_size = BATCH_SIZE

        self.kernel_data = [
            [np.zeros(self.max_batch_size, dtype=np.int32), 
             np.zeros((4, self.max_batch_size), dtype=np.float64)],

            [np.zeros(self.max_batch_size, dtype=np.int32),
             np.zeros((4, self.max_batch_size), dtype=np.float64), # u1m u1M v1m v1M
             np.zeros((1, 1, 1, 2, 5, 5, self.max_batch_size), dtype=np.float64), # u2p
             np.zeros((1, 1, 1, 2, 5, 5, self.max_batch_size), dtype=np.float64), # v2p
             np.zeros((1, 1, 1, 2, 4, 4, self.max_batch_size), dtype=np.float64), # k2
             np.zeros((3, 1, 1, 1, 1, 1, 1, self.max_batch_size), dtype=np.float64), # p20
             np.zeros((3, 1, 1, 1, 1, 1, 1, self.max_batch_size), dtype=np.float64), # p21
             np.zeros((3, 1, 1, 1, 1, 1, 1, self.max_batch_size), dtype=np.float64), # p22
             np.zeros((3, 1, 1, 1, 1, 1, 1, self.max_batch_size), dtype=np.float64), # n20
             np.zeros((3, 1, 1, 1, 1, 1, 1, self.max_batch_size), dtype=np.float64), # n21
             np.zeros((3, 1, 1, 1, 1, 1, 1, self.max_batch_size), dtype=np.float64), # n22
             np.zeros((3, 1, 1, 1, 1, 1, 1, self.max_batch_size), dtype=np.float64), # p30
             np.zeros((3, 1, 1, 1, 1, 1, 1, self.max_batch_size), dtype=np.float64), # p31
             np.zeros((3, 1, 1, 1, 1, 1, 1, self.max_batch_size), dtype=np.float64), # p32
             np.zeros((3, 1, 1, 1, 1, 2, 2, self.max_batch_size), dtype=np.float64), # d0
             np.zeros((3, 1, 1, 1, 1, 2, 2, self.max_batch_size), dtype=np.float64), # n1
             np.zeros((3, 1, 1, 1, 1, 2, 2, self.max_batch_size), dtype=np.float64), # x1
             np.zeros((3, 1, 1, 1, 2, 4, 4, self.max_batch_size), dtype=np.float64), # d1
             np.zeros((1, 1, 1, 1, 3, 3, self.max_batch_size), dtype=np.float64), 
             np.zeros((1, 1, 1, 2, 2, 2, self.max_batch_size), dtype=np.float64)],

            [np.zeros(self.max_batch_size, dtype=np.int32),
             np.zeros((4, self.max_batch_size), dtype=np.float64), # u1m u1M v1m v1M
             np.zeros((1, 1, 1, 2, 5, 5, self.max_batch_size), dtype=np.float64), # u2p
             np.zeros((1, 1, 1, 2, 5, 5, self.max_batch_size), dtype=np.float64), # v2p
             np.zeros((1, 1, 1, 2, 4, 4, self.max_batch_size), dtype=np.float64), # k2
             np.zeros((1, 1, 2, 5, 16, 16, self.max_batch_size), dtype=np.float64), # u3p
             np.zeros((1, 1, 2, 5, 16, 16, self.max_batch_size), dtype=np.float64), # v3p
             np.zeros((1, 1, 2, 5, 15, 15, self.max_batch_size), dtype=np.float64), # k3
             np.zeros((1, 1, 1, 1, 3, 3, self.max_batch_size), dtype=np.float64), # s0
             np.zeros((1, 1, 1, 2, 5, 5, self.max_batch_size), dtype=np.float64), # s1
             np.zeros((1, 1, 1, 4, 11, 11, self.max_batch_size), dtype=np.float64), # s2
             np.zeros((1, 1, 2, 6, 19, 19, self.max_batch_size), dtype=np.float64), # s3
             np.zeros(self.max_batch_size, dtype=np.int32) # tir
             ],

            [np.zeros(self.max_batch_size, dtype=np.int32), 
             np.zeros((4, self.max_batch_size), dtype=np.float64), 
             np.zeros((4, self.max_batch_size), dtype=np.float64), 
             np.zeros(self.max_batch_size, dtype=np.int32), 
             np.zeros(self.max_batch_size, dtype=np.float64)],

            [np.zeros(self.max_batch_size, dtype=np.int32),
             np.zeros((4, self.max_batch_size), dtype=np.float64),
             np.zeros((4, self.max_batch_size), dtype=np.float64),
             np.zeros(self.max_batch_size, dtype=np.float64),
             np.zeros(self.max_batch_size, dtype=np.float64),
             np.zeros(self.max_batch_size, dtype=np.float64),
             np.zeros((2, 2, 3, 7, 7, 5, 5, self.max_batch_size), dtype=np.float64)]
        ]

        self.kernel_launched_size = [0, 0, 0, 0, 0]

        self.kernel_data_pool = [
            [[] for _ in range(len(self.kernel_data[0]))],
            [[] for _ in range(len(self.kernel_data[1]))],
            [[] for _ in range(len(self.kernel_data[2]))],
            [[] for _ in range(len(self.kernel_data[3]))],
            [[] for _ in range(len(self.kernel_data[4]))]
        ]
        
        self.dfs_level_max = 4

        self.kernel_func = {
            0: self.kernel_1,
            1: self.kernel_2,
            2: self.kernel_3,
            3: self.kernel_4,
            4: self.kernel_5
        }

        self.kernel_time = [0, 0, 0, 0, 0]
        self.launch_size_counter = [0, 0, 0, 0, 0]
        self.launch_counter = [0, 0, 0, 0, 0]

        self.irradiance_method_time = [0, 0]
        self.irradiance_method_counter = [0, 0]

    def precompile(self):
        bba = BatchBPoly(np.array(np.random.rand(2, 2, 10), dtype=np.float64))
        bbb = BatchBPoly(np.array(np.random.rand(2, 2, 10), dtype=np.float64))
        c = bba + bbb
        c = bba - bbb
        c = bba * bbb
        c = bba.du()
        C = bba.dv()
        c = bba.dx()
        c = bba.dy()
        c = bba.bound()
        c = bba.align_degree_to(bbb)
        c = bba.fbound(c)
        bba.subdiv_mid(np.ones((10), dtype=bool), np.zeros((10), dtype=bool), mat_left, mat_right)
        bba3 = BatchBPoly3(np.array(np.random.rand(3, 2, 2, 10), dtype=np.float64))
        bbb3 = BatchBPoly3(np.array(np.random.rand(3, 2, 2, 10), dtype=np.float64))
        c = bba3 + bbb3
        c = bba3 - bbb3
        c = bba3 * bbb3
        c = bba3.du()
        C = bba3.dv()
        c = bba3.dx()
        c = bba3.dy()
        c = bba3.dot(bbb3)
        c = bba3.cross(bbb3)

        a = np.array([1.00000,2.00000,-2.00000,-0.19080,1.58721,-0.78662,-0.01309,0.04030,0.03519,0.03649,0.03573,0.01973,-0.19075,0.58707,-0.78674,-0.01308,0.04024,0.03512,0.03747,0.03585,0.01962,-0.82244,0.91464,0.56242,0.03013,-0.02977,0.03675,0.02616,0.02751,0.03975,-0.82190,-0.08511,0.56324,0.03008,-0.02966,0.03664,0.02615,0.02744,0.03963,-3.00000,0.00000,-3.00000,6.00000,0.00000,0.00000,0.00000,0.00000,6.00000], dtype=np.float64)
        b = np.array([1.00000,2.00000,-2.00000,0.29982,1.63410,-0.71276,-0.01713,0.04452,0.03485,0.03737,0.02280,0.03838,0.29944,0.63393,-0.71307,-0.01706,0.04438,0.03472,0.03726,0.02275,0.03824,-0.49109,0.94053,0.86908,0.04563,-0.02880,0.02186,0.04407,0.02994,0.02496,-0.49152,-0.05926,0.86885,0.04547,-0.02871,0.02183,0.04392,0.02984,0.02490,-3.00000,0.00000,-3.00000,6.00000,0.00000,0.00000,0.00000,0.00000,6.00000], dtype=np.float64)
        a = a.reshape((16, 3))
        b = b.reshape((16, 3))

        precompile_batch_size = 2

        data = np.zeros((16, 3, precompile_batch_size), dtype=np.float64)
        # u1v1 = np.zeros((4, BATCH_SIZE), dtype=np.float64)
        for i in range(0, precompile_batch_size, 2):
            data[:, :, i] = a[:, :]
            data[:, :, i + 1] = b[:, :]

        u1m, u1M, v1m, v1M = np.zeros(precompile_batch_size, dtype=np.float64), np.ones(precompile_batch_size, dtype=np.float64), np.zeros(precompile_batch_size, dtype=np.float64), np.ones(precompile_batch_size, dtype=np.float64)

        pL = data[0, :, :precompile_batch_size]
        p10 = data[1, :, :precompile_batch_size]
        p11 = data[2, :, :precompile_batch_size]
        p12 = data[3, :, :precompile_batch_size]
        pD1 = data[8, :, :precompile_batch_size]
        pD2 = data[9, :, :precompile_batch_size]
        batch_point_triangle_dist_range(pL, p10, p11, p12, u1m, u1M, v1m, v1M, precompile_batch_size)

        # self.compute_bound3d_batched(data, u1v1, BATCH_SIZE)
        kernel_data_2 = batch_get_rational_double_refract_1(cache_c2b, data, u1m, u1M, v1m, v1M, "primal")
        shape_check_index = [2, 3, 4, 14, 15, 16, 17, 18, 19]
        for i in shape_check_index:
            if isinstance(kernel_data_2[i], BatchBPoly3):
                data_shape = (3,) + kernel_data_2[i].x.shape[:-1]
            elif isinstance(kernel_data_2[i], BatchBPoly):
                data_shape = kernel_data_2[i].a.shape[:-1]
            else:
                assert False
            if data_shape != self.kernel_data[1][i].shape[:-1]:
                self.kernel_data[1][i] = np.zeros(data_shape + (self.max_batch_size,), dtype=np.float64)

        Iu2, Iv2, u2p, v2p, k2, p20, p21, p22, n20, n21, n22, p30, p31, p32, d0, n1, x1, d1, C, β, Ik2 = kernel_data_2

        u3p, v3p, k3, u2p, v2p, k2, C, s0, s1, s2, s3, β, β2, β2q = \
            batch_get_rational_double_refract_2(cache_c2b, u2p, v2p, k2, p20, p21, p22, n20, n21, n22, p30, p31, p32, d0, n1, x1, d1, C, β, "primal")
        
        kernel_data_3 = [0, 0, u2p, v2p, k2, u3p, v3p, k3, s0, s1, s2, s3]
        shape_check_index = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for i in shape_check_index:
            if isinstance(kernel_data_3[i], BatchBPoly3):
                data_shape = (3,) + kernel_data_3[i].x.shape[:-1]
            elif isinstance(kernel_data_3[i], BatchBPoly):
                data_shape = kernel_data_3[i].a.shape[:-1]
            else:
                assert False
            if data_shape != self.kernel_data[2][i].shape[:-1]:
                self.kernel_data[2][i] = np.zeros(data_shape + (self.max_batch_size,), dtype=np.float64)

        bad_u, bad_s, pbad_u1, pbad_u2, pbad_s, uDqm, uDqM, uDm, uDM, vDm, vDM = batch_positional_check(
                u1m, u1M, v1m, v1M, u2p, v2p, k2, u3p, v3p, k3, s0, s1, s2, s3)
        
        Jp, Jq = batch_irradiance_explicit(cache_c2b, data, u1m, u1M, v1m, v1M)
        
        b = BatchBPoly3(np.array([[[1.0 for _ in range(precompile_batch_size)]], [[0.0 for _ in range(precompile_batch_size)]], [[0.0 for _ in range(precompile_batch_size)]]]))
        Jp, Jq = self.batch_irradiance_implicit(data, u1m, u1M, v1m, v1M, uDm, uDM, vDm, vDM, b)

        if Jp.a.shape[:-1] != self.kernel_data[4][-1].shape[1:-1]:
            self.kernel_data[4][-1] = np.zeros((2,) + Jp.a.shape[:-1] + (self.max_batch_size,), dtype=np.float64)

        IJq = Jq.bound()
        IJp = Jp.fbound(Jq)
        areaD = np.random.rand(precompile_batch_size)
        compute_radiance_ans(pL, p10, p11, p12, pD1, pD2, u1m, u1M, v1m, v1M, IJp, IJq, areaD, precompile_batch_size)

    def get_result(self, sample_map_record, final=False):
        result_index = np.where(self.data_use_cnt == 0)[0]

        for batch_id in result_index:
            if self.batch_buf[batch_id].mean() > 0:
                batch_data = self.batch_buf[batch_id]
                nonzero_v, nonzero_u = np.where(batch_data > 0)
                if len(nonzero_v) > 0:
                    i, j = self.triangle_id_data[batch_id]
                    tmp_record = [i, j]
                    for v, u in zip(nonzero_v, nonzero_u):
                        tmp_record.extend([v, u, batch_data[v, u]])
                    sample_map_record.append(tmp_record)

        self.data_use_cnt[result_index] = -1
    
    def check_memory(self):
        # if self.triangle_data.shape[-1] >= 450:
        #     return len(np.where(self.data_use_cnt == -1)[0]) >= self.max_batch_size
        return True

    def set_triangle_data(self, triangle_data, u1v1_data, triangle_id_data, task_num):
        if task_num == 0: return

        useless_index = np.where(self.data_use_cnt == -1)[0]
        split_index = len(useless_index)
        if split_index < task_num:
            pre_num = self.triangle_data.shape[-1]
            self.triangle_data[:, :, useless_index] = triangle_data[:, :, :split_index]
            self.triangle_data = np.concatenate((self.triangle_data, triangle_data[:, :, split_index:task_num]), axis=-1)
            self.triangle_id_data[useless_index] = triangle_id_data[:split_index, :]
            self.triangle_id_data = np.concatenate((self.triangle_id_data, triangle_id_data[split_index:task_num, :]), axis=0)
            self.batch_buf[useless_index] = 0
            self.batch_buf = np.concatenate((self.batch_buf, np.zeros((task_num - split_index, RES, RES), dtype=np.float64)), axis=0)
            
            self.data_use_cnt = np.concatenate((self.data_use_cnt, np.zeros(task_num - split_index, dtype=np.int32)), axis=0)
            index_array = np.concatenate((useless_index, np.arange(pre_num, self.triangle_data.shape[-1], dtype=np.int32)), axis=0)
        else:
            useless_index = useless_index[:task_num]
            self.triangle_data[:, :, useless_index] = triangle_data[:, :, :task_num]
            self.triangle_id_data[useless_index] = triangle_id_data[:task_num, :]
            self.batch_buf[useless_index] = 0
            index_array = useless_index

        self.data_use_cnt[index_array] = 1
        
        u1v1_index_array = np.arange(task_num, dtype=np.int32)
        
        if len(index_array) > 0:
            if self.kernel_launched_size[0] < self.max_batch_size:
                begin = self.kernel_launched_size[0]
                num = min(self.max_batch_size - begin, len(index_array))
                end = begin + num
                copy_index = index_array[:num]
                copy_index_u1v1 = u1v1_index_array[:num]

                self.kernel_data[0][0][begin:end] = copy_index

                self.kernel_data[0][1][0, begin:end] = u1v1_data[0][copy_index_u1v1]
                self.kernel_data[0][1][1, begin:end] = u1v1_data[1][copy_index_u1v1]
                self.kernel_data[0][1][2, begin:end] = u1v1_data[2][copy_index_u1v1]
                self.kernel_data[0][1][3, begin:end] = u1v1_data[3][copy_index_u1v1]

                self.kernel_launched_size[0] += num
                index_array = index_array[num:]
                u1v1_index_array = u1v1_index_array[num:]

            if len(index_array) > 0:
                self.kernel_data_pool[0][0].append(index_array)
                self.kernel_data_pool[0][1].append((u1v1_data[0][u1v1_index_array], u1v1_data[1][u1v1_index_array], u1v1_data[2][u1v1_index_array], u1v1_data[3][u1v1_index_array]))

    def set_data_from_pool(self, kernel_id):
        num = self.max_batch_size - self.kernel_launched_size[kernel_id]
        while len(self.kernel_data_pool[kernel_id][0]) > 0 and num > 0:
            data_len = 0
            if isinstance(self.kernel_data_pool[kernel_id][0][-1], tuple):
                data_len = len(self.kernel_data_pool[kernel_id][0][-1][0])
            else:
                data_len = len(self.kernel_data_pool[kernel_id][0][-1])
            t_n = min(num, data_len)
            if t_n == 0:
                break
            begin = self.kernel_launched_size[kernel_id]
            end = begin + t_n
            for i in range(len(self.kernel_data[kernel_id])):
                data = self.kernel_data_pool[kernel_id][i].pop()

                if isinstance(data, tuple):
                    if len(data) == 2:
                        self.kernel_data[kernel_id][i][0, ..., begin:end] = data[0][..., :t_n]
                        self.kernel_data[kernel_id][i][1, ..., begin:end] = data[1][..., :t_n]
                        if t_n < len(data[0]):
                            self.kernel_data_pool[kernel_id][i].append((data[0][..., t_n:], data[1][..., t_n:]))
                    elif len(data) == 3:
                        self.kernel_data[kernel_id][i][0, ..., begin:end] = data[0][..., :t_n]
                        self.kernel_data[kernel_id][i][1, ..., begin:end] = data[1][..., :t_n]
                        self.kernel_data[kernel_id][i][2, ..., begin:end] = data[2][..., :t_n]
                        if t_n < len(data[0]):
                            self.kernel_data_pool[kernel_id][i].append((data[0][..., t_n:], data[1][..., t_n:], data[2][..., t_n:]))
                    elif len(data) == 4:
                        self.kernel_data[kernel_id][i][0, ..., begin:end] = data[0][..., :t_n]
                        self.kernel_data[kernel_id][i][1, ..., begin:end] = data[1][..., :t_n]
                        self.kernel_data[kernel_id][i][2, ..., begin:end] = data[2][..., :t_n]
                        self.kernel_data[kernel_id][i][3, ..., begin:end] = data[3][..., :t_n]
                        if t_n < len(data[0]):
                            self.kernel_data_pool[kernel_id][i].append((data[0][..., t_n:], data[1][..., t_n:], data[2][..., t_n:], data[3][..., t_n:]))
                    else:
                        assert False
                elif len(data) > 0:
                    self.kernel_data[kernel_id][i][..., begin:end] = data[..., :t_n]
                    if t_n < len(data):
                        self.kernel_data_pool[kernel_id][i].append(data[..., t_n:])
                else:   
                    assert False

            self.kernel_launched_size[kernel_id] += t_n
            num -= t_n

    def compute_bound3d_batched(self, lazy=True):
        bfs_flag = True
        early_exit = False
        
        lazy_bar = 1
        while bfs_flag and (not early_exit or not lazy):
            early_exit = True
            for kernel_id in range(5):
                if self.kernel_launched_size[kernel_id] == 0:
                    continue
                if lazy and self.kernel_launched_size[kernel_id] < self.max_batch_size * lazy_bar:
                    continue
                early_exit = False

                t = time.perf_counter()
                self.kernel_func[kernel_id](self.kernel_data[kernel_id], self.kernel_launched_size[kernel_id])
                self.kernel_time[kernel_id] += time.perf_counter() - t
                self.launch_size_counter[kernel_id] += self.kernel_launched_size[kernel_id]
                self.launch_counter[kernel_id] += 1

                self.kernel_launched_size[kernel_id] = 0

            for kernel_id in range(5):
                self.set_data_from_pool(kernel_id)

            if not lazy:
                bfs_flag = any(self.kernel_launched_size)
            else:
                if self.kernel_launched_size[0] >= self.max_batch_size * lazy_bar:
                    bfs_flag = True

    def kernel_1(self, data, batch_size):
        if batch_size == self.max_batch_size:
            triangle_index = data[0]
            u1m, u1M, v1m, v1M = data[1][0], data[1][1], data[1][2], data[1][3]
        else:
            triangle_index = data[0][:batch_size]
            u1m = data[1][0, :batch_size]
            u1M = data[1][1, :batch_size]
            v1m = data[1][2, :batch_size]
            v1M = data[1][3, :batch_size]

        triangle_data = np.empty((16, 3, batch_size), dtype=np.float64)
        triangle_data[:, 0] = self.triangle_data[:, 0, triangle_index]
        triangle_data[:, 1] = self.triangle_data[:, 1, triangle_index]
        triangle_data[:, 2] = self.triangle_data[:, 2, triangle_index]
        
        Iu2, Iv2, u2p, v2p, k2, p20, p21, p22, n20, n21, n22, p30, p31, p32, d0, n1, x1, d1, C, β, Ik2 = \
            batch_get_rational_double_refract_1(cache_c2b, triangle_data, u1m, u1M, v1m, v1M, "primal")

        flow_flag = np.ones(batch_size, dtype=np.int32)

        condition_mask = ((Iu2[0] > 1) | (Iu2[1] < 0) | (Iv2[0] > 1) | (Iv2[1] < 0) | (Iu2[0] + Iv2[0] > 1)) & (Ik2[0] * Ik2[1] > 0)
        flow_flag[condition_mask] = 0

        break_index = np.where(flow_flag == 0)[0]
        for i in data[0][break_index]:
            self.data_use_cnt[i] -= 1

        continue_index = np.where(flow_flag == 1)[0]

        # block_index = self.kernel_cur_block_index[1]
        if len(continue_index) > 0:
            if self.kernel_launched_size[1] < self.max_batch_size:
                begin = self.kernel_launched_size[1]
                num = min(self.max_batch_size - begin, len(continue_index))
                end = begin + num

                copy_index = continue_index[:num]

                self.kernel_data[1][0][begin:end] = data[0][copy_index]
                self.kernel_data[1][1][0, begin:end] = u1m[copy_index]
                self.kernel_data[1][1][1, begin:end] = u1M[copy_index]
                self.kernel_data[1][1][2, begin:end] = v1m[copy_index]
                self.kernel_data[1][1][3, begin:end] = v1M[copy_index]
                self.kernel_data[1][2][..., begin:end] = u2p.a[..., copy_index]
                self.kernel_data[1][3][..., begin:end] = v2p.a[..., copy_index]
                self.kernel_data[1][4][..., begin:end] = k2.a[..., copy_index]
                self.kernel_data[1][5][0, ..., begin:end] = p20.x[..., copy_index]
                self.kernel_data[1][5][1, ..., begin:end] = p20.y[..., copy_index]
                self.kernel_data[1][5][2, ..., begin:end] = p20.z[..., copy_index]
                self.kernel_data[1][6][0, ..., begin:end] = p21.x[..., copy_index]
                self.kernel_data[1][6][1, ..., begin:end] = p21.y[..., copy_index]
                self.kernel_data[1][6][2, ..., begin:end] = p21.z[..., copy_index]
                self.kernel_data[1][7][0, ..., begin:end] = p22.x[..., copy_index]
                self.kernel_data[1][7][1, ..., begin:end] = p22.y[..., copy_index]
                self.kernel_data[1][7][2, ..., begin:end] = p22.z[..., copy_index]
                self.kernel_data[1][8][0, ..., begin:end] = n20.x[..., copy_index]
                self.kernel_data[1][8][1, ..., begin:end] = n20.y[..., copy_index]
                self.kernel_data[1][8][2, ..., begin:end] = n20.z[..., copy_index]
                self.kernel_data[1][9][0, ..., begin:end] = n21.x[..., copy_index]
                self.kernel_data[1][9][1, ..., begin:end] = n21.y[..., copy_index]
                self.kernel_data[1][9][2, ..., begin:end] = n21.z[..., copy_index]
                self.kernel_data[1][10][0, ..., begin:end] = n22.x[..., copy_index]
                self.kernel_data[1][10][1, ..., begin:end] = n22.y[..., copy_index]
                self.kernel_data[1][10][2, ..., begin:end] = n22.z[..., copy_index]
                self.kernel_data[1][11][0, ..., begin:end] = p30.x[..., copy_index]
                self.kernel_data[1][11][1, ..., begin:end] = p30.y[..., copy_index]
                self.kernel_data[1][11][2, ..., begin:end] = p30.z[..., copy_index]
                self.kernel_data[1][12][0, ..., begin:end] = p31.x[..., copy_index]
                self.kernel_data[1][12][1, ..., begin:end] = p31.y[..., copy_index]
                self.kernel_data[1][12][2, ..., begin:end] = p31.z[..., copy_index]
                self.kernel_data[1][13][0, ..., begin:end] = p32.x[..., copy_index]
                self.kernel_data[1][13][1, ..., begin:end] = p32.y[..., copy_index]
                self.kernel_data[1][13][2, ..., begin:end] = p32.z[..., copy_index]
                self.kernel_data[1][14][0, ..., begin:end] = d0.x[..., copy_index]
                self.kernel_data[1][14][1, ..., begin:end] = d0.y[..., copy_index]
                self.kernel_data[1][14][2, ..., begin:end] = d0.z[..., copy_index]
                self.kernel_data[1][15][0, ..., begin:end] = n1.x[..., copy_index]
                self.kernel_data[1][15][1, ..., begin:end] = n1.y[..., copy_index]
                self.kernel_data[1][15][2, ..., begin:end] = n1.z[..., copy_index]
                self.kernel_data[1][16][0, ..., begin:end] = x1.x[..., copy_index]
                self.kernel_data[1][16][1, ..., begin:end] = x1.y[..., copy_index]
                self.kernel_data[1][16][2, ..., begin:end] = x1.z[..., copy_index]
                self.kernel_data[1][17][0, ..., begin:end] = d1.x[..., copy_index]
                self.kernel_data[1][17][1, ..., begin:end] = d1.y[..., copy_index]
                self.kernel_data[1][17][2, ..., begin:end] = d1.z[..., copy_index]
                self.kernel_data[1][18][..., begin:end] = C.a[..., copy_index]
                self.kernel_data[1][19][..., begin:end] = β.a[..., copy_index]

                self.kernel_launched_size[1] += num
                continue_index = continue_index[num:]

            if len(continue_index) > 0:
                self.kernel_data_pool[1][0].append(data[0][continue_index])
                self.kernel_data_pool[1][1].append((u1m[continue_index], u1M[continue_index], v1m[continue_index], v1M[continue_index]))
                self.kernel_data_pool[1][2].append(u2p.a[..., continue_index])
                self.kernel_data_pool[1][3].append(v2p.a[..., continue_index])
                self.kernel_data_pool[1][4].append(k2.a[..., continue_index])
                self.kernel_data_pool[1][5].append((p20.x[..., continue_index], p20.y[..., continue_index], p20.z[..., continue_index]))
                self.kernel_data_pool[1][6].append((p21.x[..., continue_index], p21.y[..., continue_index], p21.z[..., continue_index]))
                self.kernel_data_pool[1][7].append((p22.x[..., continue_index], p22.y[..., continue_index], p22.z[..., continue_index]))
                self.kernel_data_pool[1][8].append((n20.x[..., continue_index], n20.y[..., continue_index], n20.z[..., continue_index]))
                self.kernel_data_pool[1][9].append((n21.x[..., continue_index], n21.y[..., continue_index], n21.z[..., continue_index]))
                self.kernel_data_pool[1][10].append((n22.x[..., continue_index], n22.y[..., continue_index], n22.z[..., continue_index]))
                self.kernel_data_pool[1][11].append((p30.x[..., continue_index], p30.y[..., continue_index], p30.z[..., continue_index]))
                self.kernel_data_pool[1][12].append((p31.x[..., continue_index], p31.y[..., continue_index], p31.z[..., continue_index]))
                self.kernel_data_pool[1][13].append((p32.x[..., continue_index], p32.y[..., continue_index], p32.z[..., continue_index]))
                self.kernel_data_pool[1][14].append((d0.x[..., continue_index], d0.y[..., continue_index], d0.z[..., continue_index]))
                self.kernel_data_pool[1][15].append((n1.x[..., continue_index], n1.y[..., continue_index], n1.z[..., continue_index]))
                self.kernel_data_pool[1][16].append((x1.x[..., continue_index], x1.y[..., continue_index], x1.z[..., continue_index]))
                self.kernel_data_pool[1][17].append((d1.x[..., continue_index], d1.y[..., continue_index], d1.z[..., continue_index]))
                self.kernel_data_pool[1][18].append(C.a[..., continue_index])
                self.kernel_data_pool[1][19].append(β.a[..., continue_index])

    def kernel_2(self, data, batch_size):
        if batch_size == self.max_batch_size:
            u1m, u1M, v1m, v1M, u2p, v2p, k2, p20, p21, p22, n20, n21, n22, p30, p31, p32, d0, n1, x1, d1, C, β = \
                data[1][0], data[1][1], data[1][2], data[1][3], \
                BatchBPoly(data[2]), BatchBPoly(data[3]), BatchBPoly(data[4]), \
                BatchBPoly3(data[5]), BatchBPoly3(data[6]), BatchBPoly3(data[7]), \
                BatchBPoly3(data[8]), BatchBPoly3(data[9]), BatchBPoly3(data[10]), \
                BatchBPoly3(data[11]), BatchBPoly3(data[12]), BatchBPoly3(data[13]), \
                BatchBPoly3(data[14]), BatchBPoly3(data[15]), BatchBPoly3(data[16]), \
                BatchBPoly3(data[17]), BatchBPoly(data[18]), BatchBPoly(data[19])
        else:
            u1m, u1M, v1m, v1M, u2p, v2p, k2, p20, p21, p22, n20, n21, n22, p30, p31, p32, d0, n1, x1, d1, C, β = \
                data[1][0, :batch_size], data[1][1, :batch_size], data[1][2, :batch_size], data[1][3, :batch_size], \
                BatchBPoly(data[2][..., :batch_size]), BatchBPoly(data[3][..., :batch_size]), BatchBPoly(data[4][..., :batch_size]), \
                BatchBPoly3(np.stack((data[5][0, ..., :batch_size], data[5][1, ..., :batch_size], data[5][2, ..., :batch_size]))), \
                BatchBPoly3(np.stack((data[6][0, ..., :batch_size], data[6][1, ..., :batch_size], data[6][2, ..., :batch_size]))), \
                BatchBPoly3(np.stack((data[7][0, ..., :batch_size], data[7][1, ..., :batch_size], data[7][2, ..., :batch_size]))), \
                BatchBPoly3(np.stack((data[8][0, ..., :batch_size], data[8][1, ..., :batch_size], data[8][2, ..., :batch_size]))), \
                BatchBPoly3(np.stack((data[9][0, ..., :batch_size], data[9][1, ..., :batch_size], data[9][2, ..., :batch_size]))), \
                BatchBPoly3(np.stack((data[10][0, ..., :batch_size], data[10][1, ..., :batch_size], data[10][2, ..., :batch_size]))), \
                BatchBPoly3(np.stack((data[11][0, ..., :batch_size], data[11][1, ..., :batch_size], data[11][2, ..., :batch_size]))), \
                BatchBPoly3(np.stack((data[12][0, ..., :batch_size], data[12][1, ..., :batch_size], data[12][2, ..., :batch_size]))), \
                BatchBPoly3(np.stack((data[13][0, ..., :batch_size], data[13][1, ..., :batch_size], data[13][2, ..., :batch_size]))), \
                BatchBPoly3(np.stack((data[14][0, ..., :batch_size], data[14][1, ..., :batch_size], data[14][2, ..., :batch_size]))), \
                BatchBPoly3(np.stack((data[15][0, ..., :batch_size], data[15][1, ..., :batch_size], data[15][2, ..., :batch_size]))), \
                BatchBPoly3(np.stack((data[16][0, ..., :batch_size], data[16][1, ..., :batch_size], data[16][2, ..., :batch_size]))), \
                BatchBPoly3(np.stack((data[17][0, ..., :batch_size], data[17][1, ..., :batch_size], data[17][2, ..., :batch_size]))), \
                BatchBPoly(data[18][..., :batch_size]), BatchBPoly(data[19][..., :batch_size])
            
        u3p, v3p, k3, u2p, v2p, k2, C, s0, s1, s2, s3, β, β2, β2q = \
            batch_get_rational_double_refract_2(cache_c2b, u2p, v2p, k2, p20, p21, p22, n20, n21, n22, p30, p31, p32, d0, n1, x1, d1, C, β, "primal")

        Iβ = β.bound()
        Iβ2 = β2.bound()
        
        Iβ2f = β2.fbound(β2q)
        Iβ2f = BatchInterval1D(Iβ2f[0], Iβ2f[1])
        tir = np.zeros(batch_size, dtype=np.int32) if CHAIN_TYPE != 22 else np.where(Iβ2f.l < β_STRONG_THRES, 1, 0)

        flow_flag = np.ones(batch_size, dtype=np.int32)
        if CHAIN_TYPE in [2, 22]:
            condition_mask = (Iβ2[1] < β_MIN) if CHAIN_TYPE == 22 else (Iβ[1] < β_MIN)
            flow_flag[condition_mask] = 0

        break_index = np.where(flow_flag == 0)[0]
        for i in data[0][break_index]:
            self.data_use_cnt[i] -= 1

        continue_index = np.where(flow_flag == 1)[0]

        if len(continue_index) > 0:
            if self.kernel_launched_size[2] < self.max_batch_size:
                begin = self.kernel_launched_size[2]
                num = min(self.max_batch_size - begin, len(continue_index))
                end = begin + num

                copy_index = continue_index[:num]

                self.kernel_data[2][0][begin:end] = data[0][copy_index]
                self.kernel_data[2][1][0, begin:end] = u1m[copy_index]
                self.kernel_data[2][1][1, begin:end] = u1M[copy_index]
                self.kernel_data[2][1][2, begin:end] = v1m[copy_index]
                self.kernel_data[2][1][3, begin:end] = v1M[copy_index]
                self.kernel_data[2][2][..., begin:end] = u2p.a[..., copy_index]
                self.kernel_data[2][3][..., begin:end] = v2p.a[..., copy_index]
                self.kernel_data[2][4][..., begin:end] = k2.a[..., copy_index]
                self.kernel_data[2][5][..., begin:end] = u3p.a[..., copy_index]
                self.kernel_data[2][6][..., begin:end] = v3p.a[..., copy_index]
                self.kernel_data[2][7][..., begin:end] = k3.a[..., copy_index]
                self.kernel_data[2][8][..., begin:end] = s0.a[..., copy_index]
                self.kernel_data[2][9][..., begin:end] = s1.a[..., copy_index]
                self.kernel_data[2][10][..., begin:end] = s2.a[..., copy_index]
                self.kernel_data[2][11][..., begin:end] = s3.a[..., copy_index]
                self.kernel_data[2][12][begin:end] = tir[copy_index]

                self.kernel_launched_size[2] += num
                continue_index = continue_index[num:]

            if len(continue_index) > 0:
                self.kernel_data_pool[2][0].append(data[0][continue_index])
                self.kernel_data_pool[2][1].append((u1m[continue_index], u1M[continue_index], v1m[continue_index], v1M[continue_index]))
                self.kernel_data_pool[2][2].append(u2p.a[..., continue_index])
                self.kernel_data_pool[2][3].append(v2p.a[..., continue_index])
                self.kernel_data_pool[2][4].append(k2.a[..., continue_index])
                self.kernel_data_pool[2][5].append(u3p.a[..., continue_index])
                self.kernel_data_pool[2][6].append(v3p.a[..., continue_index])
                self.kernel_data_pool[2][7].append(k3.a[..., continue_index])
                self.kernel_data_pool[2][8].append(s0.a[..., continue_index])
                self.kernel_data_pool[2][9].append(s1.a[..., continue_index])
                self.kernel_data_pool[2][10].append(s2.a[..., continue_index])
                self.kernel_data_pool[2][11].append(s3.a[..., continue_index])
                self.kernel_data_pool[2][12].append(tir[continue_index])
    
    def kernel_3(self, data, batch_size):
        if batch_size == self.max_batch_size:
            u1m, u1M, v1m, v1M, u2p, v2p, k2, u3p, v3p, k3, s0, s1, s2, s3, tir = \
                data[1][0], data[1][1], data[1][2], data[1][3], \
                BatchBPoly(data[2]), BatchBPoly(data[3]), BatchBPoly(data[4]), \
                BatchBPoly(data[5]), BatchBPoly(data[6]), BatchBPoly(data[7]), \
                BatchBPoly(data[8]), BatchBPoly(data[9]), BatchBPoly(data[10]), \
                BatchBPoly(data[11]), data[12]
        else:
            u1m, u1M, v1m, v1M, u2p, v2p, k2, u3p, v3p, k3, s0, s1, s2, s3, tir = \
                data[1][0, :batch_size], data[1][1, :batch_size], data[1][2, :batch_size], data[1][3, :batch_size], \
                BatchBPoly(data[2][..., :batch_size]), BatchBPoly(data[3][..., :batch_size]), BatchBPoly(data[4][..., :batch_size]), \
                BatchBPoly(data[5][..., :batch_size]), BatchBPoly(data[6][..., :batch_size]), BatchBPoly(data[7][..., :batch_size]), \
                BatchBPoly(data[8][..., :batch_size]), BatchBPoly(data[9][..., :batch_size]), BatchBPoly(data[10][..., :batch_size]), \
                BatchBPoly(data[11][..., :batch_size]), data[12][:batch_size]
        
        bad_u, bad_s, pbad_u1, pbad_u2, pbad_s, uDqm, uDqM, uDm, uDM, vDm, vDM = batch_positional_check(
                u1m, u1M, v1m, v1M, u2p, v2p, k2, u3p, v3p, k3, s0, s1, s2, s3)
        
        flow_flag = np.ones(batch_size, dtype=np.int32)

        for i in range(batch_size):
            if (u1M[i] - u1m[i]) * (v1M[i] - v1m[i]) < 1e-4:
                pbad_s[i] = False  # hack

        for i in range(batch_size):
            if bad_s[i] or bad_u[i]:
                flow_flag[i] = 0
            elif pbad_s[i] or pbad_u1[i] or pbad_u2[i] or uDqm[i] * uDqM[i] < 0:
                flow_flag[i] = -1
                if u1M[i] - u1m[i] < U1T:
                    flow_flag[i] = 1
                    if uDqm[i] * uDqM[i] < 0:
                        uDm[i], uDM[i], vDm[i], vDM[i] = 0, 1, 0, 1

        break_index = np.where(flow_flag == 0)[0]
        for i in data[0][break_index]:
            self.data_use_cnt[i] -= 1
        
        areaD = (uDM - uDm) * (vDM - vDm)

        if SKIP_IRRADIANCE:
            continue_index = np.where(flow_flag == -1)[0]
            if len(continue_index) > 0:
                self.batch_bfs_node_append(data[0], continue_index, u1m, u1M, v1m, v1M)

            continue_index = np.where(flow_flag == 1)[0]
            if len(continue_index) > 0:
                areaD = areaD[continue_index]
                u1m = u1m[continue_index]
                u1M = u1M[continue_index]
                v1m = v1m[continue_index]
                v1M = v1M[continue_index]
                uDm = uDm[continue_index]
                uDM = uDM[continue_index]
                vDm = vDm[continue_index]
                vDM = vDM[continue_index]
                triangle_data_index = data[0][continue_index]

                flow_flag = np.ones(len(continue_index), dtype=np.int32)
                bad_approx_mask = (areaD > INF_AREA_TOL) & (u1M - u1m >= U1T)
                flow_flag[bad_approx_mask] = 0

                continue_index = np.where(flow_flag == 0)[0]
                if len(continue_index) > 0:
                    self.batch_bfs_node_append(triangle_data_index, continue_index, u1m, u1M, v1m, v1M)
                
                continue_index = np.where(flow_flag == 1)[0]
                if len(continue_index) > 0:
                    self.batch_splat(uDm[continue_index], uDM[continue_index], vDm[continue_index], vDM[continue_index], np.ones(len(continue_index), dtype=np.float64), triangle_data_index[continue_index])
                    
                    for i in triangle_data_index[continue_index]:
                        self.data_use_cnt[i] -= 1

            return
        
        continue_index = np.where(flow_flag == -1)[0]
        if len(continue_index) > 0:
            self.batch_bfs_node_append(data[0], continue_index, u1m, u1M, v1m, v1M)

        continue_index = np.where(flow_flag == 1)[0]

        if len(continue_index) > 0:
            if self.kernel_launched_size[3] < self.max_batch_size:
                begin = self.kernel_launched_size[3]
                num = min(self.max_batch_size - begin, len(continue_index))
                end = begin + num

                copy_index = continue_index[:num]
                self.kernel_data[3][0][begin:end] = data[0][copy_index]
                self.kernel_data[3][1][0, begin:end] = u1m[copy_index]
                self.kernel_data[3][1][1, begin:end] = u1M[copy_index]
                self.kernel_data[3][1][2, begin:end] = v1m[copy_index]
                self.kernel_data[3][1][3, begin:end] = v1M[copy_index]
                self.kernel_data[3][2][0, begin:end] = uDm[copy_index]
                self.kernel_data[3][2][1, begin:end] = uDM[copy_index]
                self.kernel_data[3][2][2, begin:end] = vDm[copy_index]
                self.kernel_data[3][2][3, begin:end] = vDM[copy_index]
                self.kernel_data[3][3][begin:end] = tir[copy_index]
                self.kernel_data[3][4][begin:end] = areaD[copy_index]

                self.kernel_launched_size[3] += num
                continue_index = continue_index[num:]
            
            if len(continue_index) > 0:
                self.kernel_data_pool[3][0].append(data[0][continue_index])
                self.kernel_data_pool[3][1].append((u1m[continue_index], u1M[continue_index], v1m[continue_index], v1M[continue_index]))
                self.kernel_data_pool[3][2].append((uDm[continue_index], uDM[continue_index], vDm[continue_index], vDM[continue_index]))
                self.kernel_data_pool[3][3].append(tir[continue_index])
                self.kernel_data_pool[3][4].append(areaD[continue_index])

    def kernel_4(self, data, batch_size):
        if batch_size == self.max_batch_size:
            u1m, u1M, v1m, v1M, uDm, uDM, vDm, vDM, tir, areaD = \
                data[1][0], data[1][1], data[1][2], data[1][3], \
                data[2][0], data[2][1], data[2][2], data[2][3], \
                data[3], data[4]
        else:
            u1m, u1M, v1m, v1M, uDm, uDM, vDm, vDM, tir, areaD = \
                data[1][0, :batch_size], data[1][1, :batch_size], data[1][2, :batch_size], data[1][3, :batch_size], \
                data[2][0, :batch_size], data[2][1, :batch_size], data[2][2, :batch_size], data[2][3, :batch_size], \
                data[3][:batch_size], data[4][:batch_size]
        
        IJq = np.zeros((2, batch_size), dtype=np.float64)
        IJp = np.zeros((2, batch_size), dtype=np.float64)
        Jp_a = np.zeros((2, 3, 7, 7, 5, 5, batch_size), dtype=np.float64)
        Jq_a = np.zeros((2, 3, 7, 7, 5, 5, batch_size), dtype=np.float64)

        flow_flag = np.ones(batch_size, dtype=np.int32)
        flow_flag[tir == 1] = 0
        explicit_flow = np.where(flow_flag == 1)[0]
        implicit_flow = np.where(flow_flag == 0)[0]

        explicit_num = len(explicit_flow)
        if explicit_num > 0:
            triangle_data_index = data[0][explicit_flow]
            t_data = np.empty((16, 3, explicit_num), dtype=np.float64)
            t_data[:, 0] = self.triangle_data[:, 0, triangle_data_index]
            t_data[:, 1] = self.triangle_data[:, 1, triangle_data_index]
            t_data[:, 2] = self.triangle_data[:, 2, triangle_data_index]
            t_u1m = u1m[explicit_flow]
            t_u1M = u1M[explicit_flow]
            t_v1m = v1m[explicit_flow]
            t_v1M = v1M[explicit_flow]

            t = time.perf_counter()
            Jp, Jq = batch_irradiance_explicit(cache_c2b, t_data, t_u1m, t_u1M, t_v1m, t_v1M)
            self.irradiance_method_time[0] += time.perf_counter() - t
            self.irradiance_method_counter[0] += explicit_num

            IJq[0, explicit_flow], IJq[1, explicit_flow] = Jq.bound()
            IJp[0, explicit_flow], IJp[1, explicit_flow] = Jp.fbound(Jq)
        
        implicit_num = len(implicit_flow)
        if implicit_num > 0:
            triangle_data_index = data[0][implicit_flow]
            t_data = np.empty((16, 3, implicit_num), dtype=np.float64)
            t_data[:, 0] = self.triangle_data[:, 0, triangle_data_index]
            t_data[:, 1] = self.triangle_data[:, 1, triangle_data_index]
            t_data[:, 2] = self.triangle_data[:, 2, triangle_data_index]
            t_u1m = u1m[implicit_flow]
            t_u1M = u1M[implicit_flow]
            t_v1m = v1m[implicit_flow]
            t_v1M = v1M[implicit_flow]
            t_uDm = uDm[implicit_flow]
            t_uDM = uDM[implicit_flow]
            t_vDm = vDm[implicit_flow]
            t_vDM = vDM[implicit_flow]

            b = BatchBPoly3(np.array([[[1.0 for _ in range(implicit_num)]], [[0.0 for _ in range(implicit_num)]], [[0.0 for _ in range(implicit_num)]]]))

            t = time.perf_counter()
            Jp, Jq = self.batch_irradiance_implicit(t_data, t_u1m, t_u1M, t_v1m, t_v1M, t_uDm, t_uDM, t_vDm, t_vDM, b)
            self.irradiance_method_time[1] += time.perf_counter() - t
            self.irradiance_method_counter[1] += implicit_num

            Jp_a[..., implicit_flow] = Jp.a
            Jq_a[..., implicit_flow] = Jq.a

            IJq[0, implicit_flow], IJq[1, implicit_flow] = Jq.bound()
            IJp[0, implicit_flow], IJp[1, implicit_flow] = Jp.fbound(Jq)

        pL = self.triangle_data[0][:, data[0][:batch_size]]
        p10 = self.triangle_data[1][:, data[0][:batch_size]]
        p11 = self.triangle_data[2][:, data[0][:batch_size]]
        p12 = self.triangle_data[3][:, data[0][:batch_size]]
        if CHAIN_TYPE in [1, 2]:
            pD1 = self.triangle_data[8][:, data[0][:batch_size]]
            pD2 = self.triangle_data[9][:, data[0][:batch_size]]
        else:
            pD1 = self.triangle_data[14][:, data[0][:batch_size]]
            pD2 = self.triangle_data[15][:, data[0][:batch_size]]
        ans, flow_flag, bad_q, h_rm_xu, h_rM_xu = compute_radiance_ans(pL, p10, p11, p12, pD1, pD2, u1m, u1M, v1m, v1M, IJp, IJq, areaD, batch_size)
        
        continue_index = np.where(flow_flag == 0)[0]
        if len(continue_index) > 0:
            self.batch_bfs_node_append(data[0], continue_index, u1m, u1M, v1m, v1M)

        continue_index = np.where(flow_flag == 1)[0]
        if len(continue_index) > 0:
            tir = tir[continue_index]
            splat_index = np.where(tir == 0)[0]
            if len(splat_index) > 0:
                t_ans = ans[continue_index][splat_index]

                t_uDm = uDm[continue_index][splat_index]
                t_uDM = uDM[continue_index][splat_index]
                t_vDm = vDm[continue_index][splat_index]
                t_vDM = vDM[continue_index][splat_index]
                index_global = data[0][continue_index][splat_index]
                self.batch_splat(t_uDm, t_uDM, t_vDm, t_vDM, t_ans, index_global)

                for i in index_global:
                    self.data_use_cnt[i] -= 1

            splat_subdivided = np.where(tir == 1)[0]
            if len(splat_subdivided) > 0:
                bad_q = bad_q[continue_index][splat_subdivided]
                splat_index = np.where(bad_q == 0)[0]

                t_ans = ans[continue_index][splat_subdivided][splat_index]
                t_ans = np.minimum(t_ans, AM)

                t_uDm = uDm[continue_index][splat_subdivided][splat_index]
                t_uDM = uDM[continue_index][splat_subdivided][splat_index]
                t_vDm = vDm[continue_index][splat_subdivided][splat_index]
                t_vDM = vDM[continue_index][splat_subdivided][splat_index]
                index_global = data[0][continue_index][splat_subdivided][splat_index]
                self.batch_splat(t_uDm, t_uDM, t_vDm, t_vDM, t_ans, index_global)

                for i in index_global:
                    self.data_use_cnt[i] -= 1

                continue_index = continue_index[splat_subdivided][np.where(bad_q == 1)[0]]

                if len(continue_index) > 0:
                    if self.kernel_launched_size[4] < self.max_batch_size:
                        begin = self.kernel_launched_size[4]
                        num = min(self.max_batch_size - begin, len(continue_index))
                        end = begin + num

                        copy_index = continue_index[:num]

                        self.kernel_data[4][0][begin:end] = data[0][copy_index]
                        self.kernel_data[4][1][0, begin:end] = u1m[copy_index]
                        self.kernel_data[4][1][1, begin:end] = u1M[copy_index]
                        self.kernel_data[4][1][2, begin:end] = v1m[copy_index]
                        self.kernel_data[4][1][3, begin:end] = v1M[copy_index]
                        self.kernel_data[4][2][0, begin:end] = uDm[copy_index]
                        self.kernel_data[4][2][1, begin:end] = uDM[copy_index]
                        self.kernel_data[4][2][2, begin:end] = vDm[copy_index]
                        self.kernel_data[4][2][3, begin:end] = vDM[copy_index]
                        self.kernel_data[4][3][begin:end] = h_rm_xu[copy_index]
                        self.kernel_data[4][4][begin:end] = h_rM_xu[copy_index]
                        self.kernel_data[4][5][begin:end] = areaD[copy_index]
                        self.kernel_data[4][6][0, ..., begin:end] = Jp_a[..., copy_index]
                        self.kernel_data[4][6][1, ..., begin:end] = Jq_a[..., copy_index]

                        self.kernel_launched_size[4] += num
                        continue_index = continue_index[num:]

                    if len(continue_index) > 0:
                        self.kernel_data_pool[4][0].append(data[0][continue_index])
                        self.kernel_data_pool[4][1].append((u1m[continue_index], u1M[continue_index], v1m[continue_index], v1M[continue_index]))
                        self.kernel_data_pool[4][2].append((uDm[continue_index], uDM[continue_index], vDm[continue_index], vDM[continue_index]))
                        self.kernel_data_pool[4][3].append(h_rm_xu[continue_index])
                        self.kernel_data_pool[4][4].append(h_rM_xu[continue_index])
                        self.kernel_data_pool[4][5].append(areaD[continue_index])
                        self.kernel_data_pool[4][6].append((Jp_a[..., continue_index], Jq_a[..., continue_index]))

    def batch_bfs_node_append(self, triangle_data_index, continue_index, u1m, u1M, v1m, v1M):
        next_kernel_1_size = len(continue_index)
        mid_u1, mid_v1 = (u1m + u1M) * 0.5, (v1m + v1M) * 0.5
        
        for i in triangle_data_index[continue_index]:
            self.data_use_cnt[i] += 3

        mode = 0
        mode_num = next_kernel_1_size
        next_kernel_1 = np.tile(continue_index, 4)
        while len(next_kernel_1) > 0:
            if self.kernel_launched_size[0] < self.max_batch_size:
                begin = self.kernel_launched_size[0]
                num = min(self.max_batch_size - begin, min(mode_num, len(next_kernel_1)))
                end = begin + num
                copy_index = next_kernel_1[:num]
                
                t_triangle_data_index = triangle_data_index[copy_index]

                self.kernel_data[0][0][begin:end] = t_triangle_data_index
                if mode == 0:
                    self.kernel_data[0][1][0, begin:end] = u1m[copy_index]
                    self.kernel_data[0][1][1, begin:end] = mid_u1[copy_index]
                    self.kernel_data[0][1][2, begin:end] = v1m[copy_index]
                    self.kernel_data[0][1][3, begin:end] = mid_v1[copy_index]
                elif mode == 1:
                    self.kernel_data[0][1][0, begin:end] = mid_u1[copy_index]
                    self.kernel_data[0][1][1, begin:end] = u1M[copy_index]
                    self.kernel_data[0][1][2, begin:end] = v1m[copy_index]
                    self.kernel_data[0][1][3, begin:end] = mid_v1[copy_index]
                elif mode == 2:
                    self.kernel_data[0][1][0, begin:end] = u1m[copy_index]
                    self.kernel_data[0][1][1, begin:end] = mid_u1[copy_index]
                    self.kernel_data[0][1][2, begin:end] = mid_v1[copy_index]
                    self.kernel_data[0][1][3, begin:end] = v1M[copy_index]
                elif mode == 3:
                    self.kernel_data[0][1][0, begin:end] = mid_u1[copy_index]
                    self.kernel_data[0][1][1, begin:end] = u1M[copy_index]
                    self.kernel_data[0][1][2, begin:end] = mid_v1[copy_index]
                    self.kernel_data[0][1][3, begin:end] = v1M[copy_index]

                next_kernel_1 = next_kernel_1[num:]
                mode_num -= num
                if mode_num == 0:
                    mode = (mode + 1) % 4
                    mode_num = next_kernel_1_size
                self.kernel_launched_size[0] += num

            else:
                num = min(mode_num, len(next_kernel_1))

                copy_index = next_kernel_1[:num]

                self.kernel_data_pool[0][0].append(triangle_data_index[copy_index])
                if mode == 0:
                    self.kernel_data_pool[0][1].append((u1m[copy_index], mid_u1[copy_index], v1m[copy_index], mid_v1[copy_index]))
                elif mode == 1:
                    self.kernel_data_pool[0][1].append((mid_u1[copy_index], u1M[copy_index], v1m[copy_index], mid_v1[copy_index]))
                elif mode == 2:
                    self.kernel_data_pool[0][1].append((u1m[copy_index], mid_u1[copy_index], mid_v1[copy_index], v1M[copy_index]))
                elif mode == 3:
                    self.kernel_data_pool[0][1].append((mid_u1[copy_index], u1M[copy_index], mid_v1[copy_index], v1M[copy_index]))
                
                next_kernel_1 = next_kernel_1[num:]
                mode_num -= num
                if mode_num == 0:
                    mode = (mode + 1) % 4
                    mode_num = next_kernel_1_size
        

    def kernel_5(self, data, batch_size):
        if batch_size == self.max_batch_size:
            triangle_data_index, u1m, u1M, v1m, v1M, uDm, uDM, vDm, vDM, h_rm_xu, h_rM_xu, areaD, Jp_a, Jq_a = \
                data[0], \
                data[1][0], data[1][1], data[1][2], data[1][3], \
                data[2][0], data[2][1], data[2][2], data[2][3], \
                data[3], data[4], data[5], \
                data[6][0], data[6][1]
        else:
            triangle_data_index, u1m, u1M, v1m, v1M, uDm, uDM, vDm, vDM, h_rm_xu, h_rM_xu, areaD, Jp_a, Jq_a = \
                data[0][:batch_size], \
                data[1][0, :batch_size], data[1][1, :batch_size], data[1][2, :batch_size], data[1][3, :batch_size], \
                data[2][0, :batch_size], data[2][1, :batch_size], data[2][2, :batch_size], data[2][3, :batch_size], \
                data[3][:batch_size], data[4][:batch_size], data[5][:batch_size],\
                data[6][0, ..., :batch_size], data[6][1, ..., :batch_size]
        
        triangle_data = np.empty((16, 3, batch_size), dtype=np.float64)
        triangle_data[:, 0] = self.triangle_data[:, 0, triangle_data_index]
        triangle_data[:, 1] = self.triangle_data[:, 1, triangle_data_index]
        triangle_data[:, 2] = self.triangle_data[:, 2, triangle_data_index]

        b = BatchBPoly3(np.array([[[0.0 for _ in range(batch_size)]], [[1.0 for _ in range(batch_size)]], [[0.0 for _ in range(batch_size)]]]))

        t_Jp2, t_Jq2 = self.batch_irradiance_implicit(triangle_data, u1m, u1M, v1m, v1M, uDm, uDM, vDm, vDM, b)

        log2_values = np.log2(areaD / SPLAT_SUBDIV_THRES)
        levels = np.zeros(batch_size, dtype=np.int32)
        for i in range(batch_size):
            if areaD[i] <= SPLAT_SUBDIV_THRES:
                levels[i] = 0
            else:
                levels[i] = max(0, min(4, int(log2_values[i] / 2)))

        for level in range(self.dfs_level_max + 1):
            flow_index = np.where(levels == level)[0]

            if len(flow_index) == 0:
                continue

            Jp, Jq = BatchBPoly(Jp_a[..., flow_index]), BatchBPoly(Jq_a[..., flow_index])
            Jp2, Jq2 = BatchBPoly(t_Jp2.a[..., flow_index]), BatchBPoly(t_Jq2.a[..., flow_index])

            self.splat_dfs(level, triangle_data_index[flow_index], u1m[flow_index], u1M[flow_index], v1m[flow_index], v1M[flow_index], uDm[flow_index], uDM[flow_index], vDm[flow_index], vDM[flow_index], 
                           h_rm_xu[flow_index], h_rM_xu[flow_index], Jp, Jq, Jp2, Jq2, len(flow_index))

            for i in triangle_data_index[flow_index]:
                self.data_use_cnt[i] -= 1

    def splat_dfs(self, level, triangle_data_index, u1m, u1M, v1m, v1M, uDm, uDM, vDm, vDM, h_rm_xu, h_rM_xu, Jp, Jq, Jp2, Jq2, batch_size):
        if level == 0:
            IJq = Jq.bound()
            IJp = Jp.fbound(Jq)
            ans1 = self.splat_get_ans(IJp, IJq, h_rm_xu, h_rM_xu, batch_size)
            IJq2 = Jq2.bound()
            IJp2 = Jp2.fbound(Jq2)
            ans2 = self.splat_get_ans(IJp2, IJq2, h_rm_xu, h_rM_xu, batch_size)
            ans = np.minimum(ans1, ans2)
            self.batch_splat(uDm, uDM, vDm, vDM, ans, triangle_data_index)
        else:
            u1m_, u1M_ = u1m, u1M
            v1m_, v1M_ = v1m, v1M
            for iD in range(2):
                uDm_ = uDm if iD == 0 else (uDm + uDM) * 0.5
                uDM_ = (uDm + uDM) * 0.5 if iD == 0 else uDM
                for jD in range(2):
                    vDm_ = vDm if jD == 0 else (vDm + vDM) * 0.5
                    vDM_ = (vDm + vDM) * 0.5 if jD == 0 else vDM
                    u_left = np.ones(batch_size, dtype=bool) if iD == 0 else np.zeros(batch_size, dtype=bool)
                    u_right = np.ones(batch_size, dtype=bool) if jD == 0 else np.zeros(batch_size, dtype=bool)
                    Jp_ = Jp.subdiv_mid(u_left, u_right, mat_left, mat_right)
                    Jq_ = Jq.subdiv_mid(u_left, u_right, mat_left, mat_right)
                    Jp2_ = Jp2.subdiv_mid(u_left, u_right, mat_left, mat_right)
                    Jq2_ = Jq2.subdiv_mid(u_left, u_right, mat_left, mat_right)

                    self.splat_dfs(level - 1, triangle_data_index, u1m_, u1M_, v1m_, v1M_, uDm_, uDM_, vDm_, vDM_, h_rm_xu, h_rM_xu, Jp_, Jq_, Jp2_, Jq2_, batch_size)


    def splat_get_ans(self, IJp, IJq, h_rm_x_u, h_rM_x_u, batch_size):
        Idenom = BatchInterval1D(IJq[0], IJq[1])
        I = BatchInterval1D(IJp[0], IJp[1])

        fqm, fqM, fm, fM = Idenom.l, Idenom.r, I.l, I.r
        
        ans = np.zeros(batch_size, dtype=np.float64)
        
        for i in range(batch_size):
            absfm = min(abs(fm[i]), abs(fM[i])) if fm[i] * fM[i] >= 0 else 0
            absfM = max(abs(fm[i]), abs(fM[i]))

            valM = absfM * h_rm_x_u[i]
            valm = absfm * h_rM_x_u[i]
            valm = max(Am, valm)
            valM = min(AM, valM)
            
            ans[i] = valM if fqm[i] * fqM[i] >= 0 else AM

        return ans

    def batch_splat(self, um, uM, vm, vM, ans, index_global):
        batch_size = len(index_global)
        for i in range(batch_size):
            if um[i] <= 1 and uM[i] >= 0 and vm[i] <= 1 and vM[i] >= 0:
                a = max(0, min(RES, int(max(0.0, vm[i]) * RES)))
                b = max(0, min(RES, int(min(1.0, vM[i]) * RES)+1))
                c = max(0, min(RES, int(max(0.0, um[i]) * RES)))
                d = max(0, min(RES, int(min(1.0, uM[i]) * RES)+1))
                e = np.ones((b-a, d-c)) * ans[i]
                self.batch_buf[index_global[i]][a:b, c:d] = np.maximum(self.batch_buf[index_global[i]][a:b, c:d], e)

    def batch_irradiance_implicit(self, data, u1m, u1M, v1m, v1M, uDm, uDM, vDm, vDM, b):
        u2p, v2p, k2, C = batch_get_rational_double_refract_half_deriv(data, u1m, u1M, v1m, v1M)
        u2p.a = u2p.a.swapaxes(0, 3).swapaxes(1, 2)
        v2p.a = v2p.a.swapaxes(0, 3).swapaxes(1, 2)
        k2.a = k2.a.swapaxes(0, 3).swapaxes(1, 2)

        return batch_irradiance_implicit(cache_r2d, cache_c2b, data, u1m, u1M, v1m, v1M, uDm, uDM, vDm, vDM, b, u2p, v2p, k2, C)


@njit(cache=True, fastmath=True)
def batch_positional_check(u1m, u1M, v1m, v1M, u2p, v2p, k2, u3p, v3p, k3, s0, s1, s2, s3):
    batch_size = len(u1m)
    k2 = k2.align_degree_to(u2p)
    k3 = k3.align_degree_to(u3p)
    u2m, u2M = u2p.fbound(k2)
    v2m, v2M = v2p.fbound(k2)
    u2pm, u2pM = u2p.bound()
    v2pm, v2pM = v2p.bound()
    u2qm, u2qM = k2.bound()
    uDm, uDM = u3p.fbound(k3)
    uDpm, uDpM = u3p.bound()
    vDpm, vDpM = v3p.bound()
    vDm, vDM = v3p.fbound(k3)
    uDqm, uDqM = k3.bound()
    inv_uDm, inv_uDM = k3.fbound(u3p)
    inv_vDm, inv_vDM = k3.fbound(v3p)
    inv_u2m, inv_u2M = k2.fbound(u2p)
    inv_v2m, inv_v2M = k2.fbound(v2p)
    for i in range(batch_size):
        if uDqm[i] * uDqM[i] < 0 and uDpm[i] * uDpM[i] > 0 and vDpm[i] * vDpM[i] > 0:  # denom includes 0, try reciprocal
            if inv_uDm[i] * inv_uDM[i] > 0 and inv_vDm[i] * inv_vDM[i] > 0:
                uDm[i], uDM[i] = 1.0 / inv_uDM[i], 1.0 / inv_uDm[i]
                vDm[i], vDM[i] = 1.0 / inv_vDM[i], 1.0 / inv_vDm[i]
                uDqm[i], uDqM[i] = 1.0, 1.0  # pseudo, means no problem
            elif inv_uDM[i] < 1 or inv_vDM[i] < 1:
                uDm[i], uDM[i] = 1e9, 1e9
                vDm[i], vDM[i] = 1e9, 1e9
                uDqm[i], uDqM[i] = 1.0, 1.0
        if u2qm[i] * u2qM[i] < 0 and u2pm[i] * u2pM[i] > 0 and v2pm[i] * v2pM[i] > 0:  # denom includes 0, try reciprocal
            if inv_u2m[i] * inv_u2M[i] > 0 and inv_v2m[i] * inv_v2M[i] > 0:
                u2m[i], u2M[i] = 1.0 / inv_u2M[i], 1.0 / inv_u2m[i]
                v2m[i], v2M[i] = 1.0 / inv_v2M[i], 1.0 / inv_v2m[i]
                u2qm[i], u2qM[i] = 1.0, 1.0
            elif inv_u2M[i] < 1 or inv_v2M[i] < 1:
                u2m[i], u2M[i] = 1e9, 1e9
                v2m[i], v2M[i] = 1e9, 1e9
                u2qm[i], u2qM[i] = 1.0, 1.0
        if uDqm[i] * uDqM[i] < 0:  # denom includes 0, try reciprocal
            if (uDpm[i] * uDpM[i] > 0 and inv_uDM[i] < 1) or (vDpm[i] * vDpM[i] > 0 and inv_vDM[i] < 1):
                uDm[i], uDM[i] = 1e9, 1e9
                vDm[i], vDM[i] = 1e9, 1e9
                uDqm[i], uDqM[i] = 1.0, 1.0
        if u2qm[i] * u2qM[i] < 0:  # denom includes 0, try reciprocal
            if (u2pm[i] * u2pM[i] > 0 and inv_u2M[i] < 1) or (v2pm[i] * v2pM[i] > 0 and inv_v2M[i] < 1):
                u2m[i], u2M[i] = 1e9, 1e9
                v2m[i], v2M[i] = 1e9, 1e9
                u2qm[i], u2qM[i] = 1.0, 1.0

    s0m, s0M = s0.bound()
    s1m, s1M = s1.bound()
    s2m, s2M = s2.bound()
    s3m, s3M = s3.bound()
    bad_u2 = np.zeros(batch_size, dtype=numba.types.bool_)
    pbad_u2 = np.zeros(batch_size, dtype=numba.types.bool_)
    bad_u = np.zeros(batch_size, dtype=numba.types.bool_)
    bad_s = np.zeros(batch_size, dtype=numba.types.bool_)
    pbad_s = np.zeros(batch_size, dtype=numba.types.bool_)
    pbad_u1 = np.zeros(batch_size, dtype=numba.types.bool_)
    for i in range(batch_size):
        bad_u2[i] = (u2qm[i] * u2qM[i] > 0 and (u2M[i] < 0 or u2m[i] > 1 or v2M[i] < 0 or v2m[i] > 1 or u2m[i] + v2m[i] > 1)) and CHAIN_LENGTH >= 2
        pbad_u2[i] = (u2qm[i] * u2qM[i] < 0 or (u2m[i] < -u1TOLERATE or u2M[i] > 1 + u1TOLERATE or v2M[i] < - u1TOLERATE or v2m[i] > 1 + u1TOLERATE or u2m[i] + v2m[i] > 1 + u1TOLERATE)) and CHAIN_LENGTH >= 2
        bad_u[i] = uDqm[i] * uDqM[i] > 0 and (uDM[i] < 0 or uDm[i] > 1 or vDM[i] < 0 or vDm[i] > 1) or u1m[i] + v1m[i] > 1
        bad_u[i] = bad_u[i] or bad_u2[i]
        bad_s[i] = (s0M[i] < 0 or s1M[i] < 0 or s2M[i] < 0 or s3M[i] < 0) and CHAIN_LENGTH >= 2
        pbad_s[i] = s0m[i] < 0 or s1m[i] < 0 or s2m[i] < 0 or s3m[i] < 0
        pbad_u1[i] = u1M[i] + v1M[i] > 1 + u1TOLERATE
        
    return bad_u, bad_s, pbad_u1, pbad_u2, pbad_s, uDqm, uDqM, uDm, uDM, vDm, vDM


@njit(fastmath=True)
def batch_irradiance_implicit_impl(cache_r2d, cache_c2b, u, v, p10_, p11_, p12_, p20_, p21_, p22_, n20_, n21_, n22_, p30_, p31_, p32_, C, u3, v3, u2p, v2p, k2, scale, b):
    x1 = p10_ + p11_ * u + p12_ * v
    x2s = p20_ * k2 + p21_ * u2p + p22_ * v2p
    n2s = (n20_ * k2 + n21_ * u2p + n22_ * v2p) if SHADING_NORMAL else n20_
    x3 = p30_ + p31_ * u3 + p32_ * v3
    d1s = x2s - x1 * k2
    d2s = x3 * k2 - x2s

    batch_size = b.x.shape[-1]

    G = d1s.cross(n2s).dot(x3 - x1)
    Gu1, Gv1, Gu3, Gv3 = batch_reduce2d_batch4(cache_r2d, G.du(), G.dv(), G.dx(), G.dy(), 0)
    η_poly = BatchBPoly(np.array([η_VAL * η_VAL for _ in range(batch_size)], dtype=np.float64))

    Fu1, Fv1, Fu3, Fv3 = None, None, None, None

    if False:  # Accurate
        c1, c2 = d1s.cross(n2s).dot(b), d2s.cross(n2s).dot(b)
        F = d2s.dot(d2s) * c1 * c1 - d1s.dot(d1s) * c2 * c2 * η_poly
        Fu1, Fv1, Fu3, Fv3 = reduce2d_batch4(
            cache_r2d, cache_c2b, F.du(), F.dv(), F.dx(), F.dy(), 0)
    else:  # Approximate (the following code is just a manually derived version of the above three lines)

        c1, c2 = d1s.cross(n2s).dot(b), d2s.cross(n2s).dot(b)
        d2s_dot_d2s = d2s.dot(d2s)
        d2s_dot_d2s_du = d2s_dot_d2s.du()
        d2s_dot_d2s_dv = d2s_dot_d2s.dv()
        d2s_dot_d2s_dx = d2s_dot_d2s.dx()
        d2s_dot_d2s_dy = d2s_dot_d2s.dy()

        d1s_dot_d1s = d1s.dot(d1s)
        d1s_dot_d1s_du = d1s_dot_d1s.du()
        d1s_dot_d1s_dv = d1s_dot_d1s.dv()

        c1_dot_c1 = c1 * c1
        c1_dot_c1_du = c1_dot_c1.du()
        c1_dot_c1_dv = c1_dot_c1.dv()

        c2 = batch_reduce2d(cache_r2d, c2, 1)
        c2_du = batch_reduce2d(cache_r2d, c2.du(), 0)
        c2_dv = batch_reduce2d(cache_r2d, c2.dv(), 0)
        c2_dx = batch_reduce2d(cache_r2d, c2.dx(), 0)
        c2_dy = batch_reduce2d(cache_r2d, c2.dy(), 0)
        c2_dot_c2 = c2 * c2
        c2_dot_c2_du = c2_du * c2 * BPoly(np.array(2.0))
        c2_dot_c2_dv = c2_dv * c2 * BPoly(np.array(2.0))
        c2_dot_c2_dx = c2_dx * c2 * BPoly(np.array(2.0))
        c2_dot_c2_dy = c2_dy * c2 * BPoly(np.array(2.0))
        # c2_dot_c2 = c2 * c2
        # c2_dot_c2_du = c2_dot_c2.du()
        # c2_dot_c2_dv = c2_dot_c2.dv()
        # c2_dot_c2_dx = c2_dot_c2.dx()
        # c2_dot_c2_dy = c2_dot_c2.dy()

        d2s_dot_d2s_du = batch_reduce2d(cache_r2d, d2s_dot_d2s_du, 0)
        d2s_dot_d2s_dv = batch_reduce2d(cache_r2d, d2s_dot_d2s_dv, 0)
        d2s_dot_d2s_dx = batch_reduce2d(cache_r2d, d2s_dot_d2s_dx, 0)
        d2s_dot_d2s_dy = batch_reduce2d(cache_r2d, d2s_dot_d2s_dy, 0)

        c1_dot_c1 = batch_reduce2d(cache_r2d, c1_dot_c1, 1)

        d2s_dot_d2s = batch_reduce2d(cache_r2d, d2s_dot_d2s, 0)
        c1_dot_c1_du = batch_reduce2d(cache_r2d, c1_dot_c1_du, 1)
        c1_dot_c1_dv = batch_reduce2d(cache_r2d, c1_dot_c1_dv, 1)

        d1s_dot_d1s_du = batch_reduce2d(cache_r2d, d1s_dot_d1s_du, 0)
        d1s_dot_d1s_dv = batch_reduce2d(cache_r2d, d1s_dot_d1s_dv, 0)

        c2_dot_c2 = batch_reduce2d(cache_r2d, c2_dot_c2, 1)

        d1s_dot_d1s = batch_reduce2d(cache_r2d, d1s_dot_d1s, 0)

        c2_dot_c2_du = batch_reduce2d(cache_r2d, c2_dot_c2_du, 1)
        c2_dot_c2_dv = batch_reduce2d(cache_r2d, c2_dot_c2_dv, 1)
        c2_dot_c2_dx = batch_reduce2d(cache_r2d, c2_dot_c2_dx, 1)
        c2_dot_c2_dy = batch_reduce2d(cache_r2d, c2_dot_c2_dy, 1)

        Fu1_a1 = batch_reduce2d(cache_r2d, d2s_dot_d2s_du * c1_dot_c1, 0)
        Fu1_a2 = batch_reduce2d(cache_r2d, d2s_dot_d2s * c1_dot_c1_du, 1)
        Fu1_a = batch_reduce2d(cache_r2d, Fu1_a1 + Fu1_a2, 0)
        Fu1_b1 = batch_reduce2d(cache_r2d, d1s_dot_d1s_du * c2_dot_c2 * η_poly, 0)
        Fu1_b2 = batch_reduce2d(cache_r2d, d1s_dot_d1s * c2_dot_c2_du * η_poly, 1)
        Fu1_b = batch_reduce2d(cache_r2d, Fu1_b1 + Fu1_b2, 1)
        Fu1 = batch_reduce2d(cache_r2d, Fu1_a - Fu1_b, 0)

        Fv1_a1 = batch_reduce2d(cache_r2d, d2s_dot_d2s_dv * c1_dot_c1, 0)
        Fv1_a2 = batch_reduce2d(cache_r2d, d2s_dot_d2s * c1_dot_c1_dv, 1)
        Fv1_a = batch_reduce2d(cache_r2d, Fv1_a1 + Fv1_a2, 0)
        Fv1_b1 = batch_reduce2d(cache_r2d, d1s_dot_d1s_dv * c2_dot_c2 * η_poly, 0)
        Fv1_b2 = batch_reduce2d(cache_r2d, d1s_dot_d1s * c2_dot_c2_dv * η_poly, 1)
        Fv1_b = batch_reduce2d(cache_r2d, Fv1_b1 + Fv1_b2, 1)
        Fv1 = batch_reduce2d(cache_r2d, Fv1_a - Fv1_b, 0)

        Fu3_a = batch_reduce2d(cache_r2d, d2s_dot_d2s_dx * c1_dot_c1, 0)
        Fu3_b = batch_reduce2d(cache_r2d, d1s_dot_d1s * c2_dot_c2_dx * η_poly, 1)
        Fu3 = batch_reduce2d(cache_r2d, Fu3_a - Fu3_b, 0)

        Fv3_a = batch_reduce2d(cache_r2d, d2s_dot_d2s_dy * c1_dot_c1, 0)
        Fv3_b = batch_reduce2d(cache_r2d, d1s_dot_d1s * c2_dot_c2_dy * η_poly, 1)
        Fv3 = batch_reduce2d(cache_r2d, Fv3_a - Fv3_b, 0)

    denom = batch_msm_r2d(cache_r2d, cache_c2b, Fu1, Gv1, Fv1, Gu1)
    u1u3p = batch_msm_r2d(cache_r2d, cache_c2b, Fv1, Gu3, Fu3, Gv1)
    u1v3p = batch_msm_r2d(cache_r2d, cache_c2b, Fv1, Gv3, Fv3, Gv1)
    v1u3p = batch_msm_r2d(cache_r2d, cache_c2b, Fu3, Gu1, Fu1, Gu3)
    v1v3p = batch_msm_r2d(cache_r2d, cache_c2b, Fv3, Gu1, Fu1, Gv3)

    Jp = batch_msm_r2d(cache_r2d, cache_c2b, u1u3p, v1v3p, u1v3p, v1u3p)
    tJq = denom * denom * C
    Jq_a = np.swapaxes(tJq.a, 0, 1)
    Jq = BatchBPoly(Jq_a)
    max_shape = np.maximum(np.array(Jp.a.shape[:-1]), np.array(Jq.a.shape[:-1])) - 1
    return Jp.elevate(max_shape) * BatchBPoly(scale), Jq.elevate(max_shape)

@njit(fastmath=True)
def batch_irradiance_implicit(cache_r2d, cache_c2b, data, u1m, u1M, v1m, v1M, uDm, uDM, vDm, vDM, b, u2p, v2p, k2, C):
    scale = ((u1M - u1m) * (v1M - v1m) / (uDM - uDm) / (vDM - vDm))

    pL, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32 = \
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15]

    p10_ = p10 + p11 * u1m + p12 * v1m
    p11_ = p11 * (u1M - u1m)
    p12_ = p12 * (v1M - v1m)

    p10_ = BatchBPoly3(p10_)
    p11_ = BatchBPoly3(p11_)
    p12_ = BatchBPoly3(p12_)
    p20_ = BatchBPoly3(p20)
    p21_ = BatchBPoly3(p21)
    p22_ = BatchBPoly3(p22)
    p30_ = BatchBPoly3(p30)
    p31_ = BatchBPoly3(p31)
    p32_ = BatchBPoly3(p32)
    n20_ = BatchBPoly3(n20)
    n21_ = BatchBPoly3(n21)
    n22_ = BatchBPoly3(n22)

    batch_size = len(scale)
    
    u = BPoly(np.array([[[[0], [1]]]], dtype=np.float64))
    v = BPoly(np.array([[[[0, 1]]]], dtype=np.float64))
    u3_a = np.empty((2, 1, 1, 1, batch_size), dtype=np.float64)
    v3_a = np.empty((2, 1, 1, batch_size), dtype=np.float64)
    for i in range(batch_size):
        u3_a[0, 0, 0, 0, i] = uDm[i]
        u3_a[1, 0, 0, 0, i] = uDM[i]
        v3_a[0, 0, 0, i] = vDm[i]
        v3_a[1, 0, 0, i] = vDM[i]
    u3 = BatchBPoly(u3_a)
    v3 = BatchBPoly(v3_a)

    ans = batch_irradiance_implicit_impl(cache_r2d, cache_c2b, u, v, p10_, p11_, p12_, p20_, p21_, p22_, n20_, n21_, n22_, p30_, p31_, p32_, C, u3, v3, u2p, v2p, k2, scale, b)
    return ans

@njit(fastmath=True)
def batch_irradiance_explicit(cache_c2b, data, u1m, u1M, v1m, v1M):
    Iu2, Iv2, u2p, v2p, k2, p20, p21, p22, n20, n21, n22, p30, p31, p32, d0, n1, x1, d1, C, β, Ik2 = \
        batch_get_rational_double_refract_1(cache_c2b, data, u1m, u1M, v1m, v1M, "deriv")
    u3p, v3p, k3, u2p, v2p, k2 = \
        batch_get_rational_double_refract_2_deriv(u2p, v2p, k2, p20, p21, p22, n20, n21, n22, p30, p31, p32, d1)
    
    u3p_du, u3p_dv, v3p_du, v3p_dv, k3_du, k3_dv = u3p.du(), u3p.dv(), v3p.du(), v3p.dv(), k3.du(), k3.dv()
    u_scale_array = 1. / (u1M - u1m)
    v_scale_array = 1. / (v1M - v1m)
    u_scale = BatchBPoly(u_scale_array)
    v_scale = BatchBPoly(v_scale_array)
    u3p_du, v3p_du, k3_du = u3p_du * u_scale, v3p_du * u_scale, k3_du * u_scale
    u3p_dv, v3p_dv, k3_dv = u3p_dv * v_scale, v3p_dv * v_scale, k3_dv * v_scale
    
    u3p_du = batch_reduce_all(u3p_du, cache_c2b, new_axis=0)
    u3p_dv = batch_reduce_all(u3p_dv, cache_c2b, new_axis=0)
    v3p_du = batch_reduce_all(v3p_du, cache_c2b, new_axis=0)
    v3p_dv = batch_reduce_all(v3p_dv, cache_c2b, new_axis=0)
    k3_du = batch_reduce_all(k3_du, cache_c2b, new_axis=2)
    k3_dv = batch_reduce_all(k3_dv, cache_c2b, new_axis=2)
    k3 = batch_reduce_all(k3, cache_c2b, new_axis=1)
    u3p = batch_reduce_all(u3p, cache_c2b, new_axis=3)
    v3p = batch_reduce_all(v3p, cache_c2b, new_axis=3)

    u3u1p = u3p_du * k3 - k3_du * u3p
    u3v1p = u3p_dv * k3 - k3_dv * u3p
    v3u1p = v3p_du * k3 - k3_du * v3p
    v3v1p = v3p_dv * k3 - k3_dv * v3p

    denom = k3 * k3

    u3u1p = batch_reduce_all(u3u1p, cache_c2b, new_axis=0)
    v3v1p = batch_reduce_all(v3v1p, cache_c2b, new_axis=1)
    u3v1p = batch_reduce_all(u3v1p, cache_c2b, new_axis=2)
    v3u1p = batch_reduce_all(v3u1p, cache_c2b, new_axis=3)

    fq = u3u1p * v3v1p
    fq = batch_reduce_all(fq, cache_c2b, new_axis=0)
    fq1 = u3v1p * v3u1p
    fq = fq - fq1
    fq = fq * C
    fp = denom * denom
    target_degree = np.zeros((len(fp.a.shape) - 1,), dtype=np.int32)
    for i in range(len(fp.a.shape) - 1):
        target_degree[i] = fp.a.shape[i] if fp.a.shape[i] > fq.a.shape[i] else fq.a.shape[i]
    # np.maximum(fp.a.shape[:-1], fq.a.shape[:-1])
    Jp = fp.elevate(target_degree)
    Jq = fq.elevate(target_degree)
    return Jp, Jq

@njit(fastmath=True)
def compute_radiance_ans(pL, p10, p11, p12, pD1, pD2, u1m, u1M, v1m, v1M, IJp, IJq, areaD, batch_size):
    rm, rM = batch_point_triangle_dist_range(pL, p10, p11, p12, u1m, u1M, v1m, v1M, batch_size)
    h = np.abs(batch_dot(batch_normalize(batch_cross(p11, p12, batch_size), batch_size), p10 - pL, batch_size))
    x_u = batch_norm(batch_cross(p11, p12, batch_size), batch_size) / batch_norm(batch_cross(pD1, pD2, batch_size), batch_size)

    Idenom = BatchInterval1D(IJq[0], IJq[1])
    I = BatchInterval1D(IJp[0], IJp[1])

    fqm, fqM, fm, fM = Idenom.l, Idenom.r, I.l, I.r
    
    bad_q = np.zeros(batch_size, dtype=np.int32)
    h_rm_xu = h / rm * x_u
    h_rM_xu = h / rM * x_u

    mask = fm * fM >= 0
    absfm = np.where(mask, np.minimum(np.abs(fm), np.abs(fM)), 0)
    absfM = np.maximum(np.abs(fm), np.abs(fM))

    valM = np.minimum(absfM * h_rm_xu, AM)
    valm = np.maximum(absfm * h_rM_xu, Am)

    bad_q_mask = ((fqm * fqM < 0) | (valM / valm > AR))
    bad_q[bad_q_mask] = 1

    ans = valM

    max_valM_mask = fqm * fqM < 0
    ans[max_valM_mask] = AM

    flow_flag = np.ones(batch_size, dtype=np.int32)
    bad_approx_mask = (valM / valm > AR) & (areaD > INF_AREA_TOL) & (u1M - u1m >= U1T)
    flow_flag[bad_approx_mask] = 0

    return ans, flow_flag, bad_q, h_rm_xu, h_rM_xu
