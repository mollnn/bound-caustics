from alias import *
from utils import *
from formula import *
from bounder import *
from reference import *
import time
import numba
from numba import njit
import multiprocessing


@njit(fastmath=True)
def get_valid_triangle_sequence_with_group(mesh, bvhs, obj_tri_cnt_prefix, tid, n_thread):
    init_u1v1 = []
    valid_triangle_index = []

    bvh_nodes, bvh_roots = bvhs

    obj_num = len(obj_tri_cnt_prefix) - 1

    for obj_idx in range(obj_num):
        obj_tri_start = obj_tri_cnt_prefix[obj_idx]
        obj_tri_end = obj_tri_cnt_prefix[obj_idx + 1]

        for batch_start in range(tid + obj_tri_start, obj_tri_end, n_thread * BATCH_SIZE):
            batch_end = min(batch_start + n_thread * BATCH_SIZE, obj_tri_end)
            batch_size = math.ceil((batch_end - batch_start) / n_thread)

            data = np.zeros((7, 3, batch_size), dtype=np.float64)
            BIx1 = np.zeros((6, batch_size), dtype=np.float64)

            for b_idx, i in enumerate(range(batch_start, batch_end, n_thread)):
                tr = mesh[i]
                p10, p11, p12 = tr[0], tr[1] - tr[0], tr[2] - tr[0]
                n10, n11, n12 = tr[3], tr[4] - tr[3], tr[5] - tr[3]

                data[0, :, b_idx] = p0
                data[1, :, b_idx] = p10
                data[2, :, b_idx] = p11
                data[3, :, b_idx] = p12
                data[4, :, b_idx] = n10
                data[5, :, b_idx] = n11
                data[6, :, b_idx] = n12
                Ix1 = interval_position_nb(p10[0], p10[1], p10[2], p10[0], p10[1], p10[2], \
                                        p11[0], p11[1], p11[2], p11[0], p11[1], p11[2], \
                                        p12[0], p12[1], p12[2], p12[0], p12[1], p12[2], \
                                        )
                BIx1[0, b_idx] = Ix1[0]
                BIx1[1, b_idx] = Ix1[1]
                BIx1[2, b_idx] = Ix1[2]
                BIx1[3, b_idx] = Ix1[3]
                BIx1[4, b_idx] = Ix1[4]
                BIx1[5, b_idx] = Ix1[5]

            d1 = batch_get_rational_d1(data)
            d1x = BatchBPoly(d1.x)
            d1y = BatchBPoly(d1.y)
            d1z = BatchBPoly(d1.z)
            Id1x = d1x.bound()
            Id1y = d1y.bound()
            Id1z = d1z.bound()

            triangle_index = np.zeros((2, batch_size), dtype=np.int32)
            triangle_index[0, :] = np.arange(batch_start, batch_end, n_thread)
            triangle_index[1, :] = np.arange(batch_size)

            queue = [bvh_roots[obj_idx]]
            queue_mask = [triangle_index]
            queue_index = 0

            while queue_index < len(queue):
                bvh_node_index = queue[queue_index]
                node = bvh_nodes[bvh_node_index]
                triangle_index_node = queue_mask[queue_index]
                queue_index += 1

                Ix2 = (node.bbox_min[0], node.bbox_min[1], node.bbox_min[2], node.bbox_max[0], node.bbox_max[1], node.bbox_max[2])

                Iw = batch_interval3D_nb_sub(BIx1[:, triangle_index_node[1]], Ix2)
                Iits = batch_interval3D_nb_cross(Iw[0], Iw[1], Iw[2], Iw[3], Iw[4], Iw[5], Id1x[0, triangle_index_node[1]], Id1y[0, triangle_index_node[1]], Id1z[0, triangle_index_node[1]], Id1x[1, triangle_index_node[1]], Id1y[1, triangle_index_node[1]], Id1z[1, triangle_index_node[1]])
                flag_its = batch_interval3D_nb_contain0(Iits[0], Iits[1], Iits[2], Iits[3], Iits[4], Iits[5])

                next_index = np.where(flag_its == True)[0]

                if len(next_index) == 0:
                    continue

                if node.triangle_id != -1:
                    tr2 = mesh[node.triangle_id]

                    p20, p21, p22 = tr2[0], tr2[1] - tr2[0], tr2[2] - tr2[0]
                    n20, n21, n22 = tr2[3], tr2[4] - tr2[3], tr2[5] - tr2[3]

                    for tr1_idx in triangle_index_node[0, next_index]:
                        tr1 = mesh[tr1_idx]
                        p10, p11, p12 = tr1[0], tr1[1] - tr1[0], tr1[2] - tr1[0]
                        n10, n11, n12 = tr1[3], tr1[4] - tr1[3], tr1[5] - tr1[3]

                        if interval_test_all_impl(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32) == False:
                            continue

                        u1m, u1M, v1m, v1M = find_u1_domain_impl(p0, p10, p11, p12, n10, n11, n12, p20, p21, p22)

                        if u1M - u1m <= 0 or v1M - v1m <= 0:
                            continue

                        init_u1v1.append([u1m, u1M, v1m, v1M])
                        valid_triangle_index.append([tr1_idx, node.triangle_id])
                else:
                    if node.left != -1:
                        queue.append(node.left)
                        queue_mask.append(triangle_index_node[:, next_index])
                    if node.right != -1:
                        queue.append(node.right)
                        queue_mask.append(triangle_index_node[:, next_index])

    return init_u1v1, valid_triangle_index

def get_dummy_mesh():
    a = np.array([1.00000,2.00000,-2.00000,-0.19080,1.58721,-0.78662,-0.01309,0.04030,0.03519,0.03649,0.03573,0.01973,-0.19075,0.58707,-0.78674,-0.01308,0.04024,0.03512,0.03747,0.03585,0.01962,-0.82244,0.91464,0.56242,0.03013,-0.02977,0.03675,0.02616,0.02751,0.03975,-0.82190,-0.08511,0.56324,0.03008,-0.02966,0.03664,0.02615,0.02744,0.03963,-3.00000,0.00000,-3.00000,6.00000,0.00000,0.00000,0.00000,0.00000,6.00000], dtype=np.float64)
    mesh = np.zeros((2, 6, 3), dtype=np.float64)
    mesh[0][0] = a[3:6]
    mesh[0][1] = a[6:9] + a[3:6]
    mesh[0][2] = a[9:12] + a[3:6]
    mesh[0][3] = a[12:15]
    mesh[0][4] = a[15:18] + a[12:15]
    mesh[0][5] = a[18:21] + a[12:15]
    mesh[1][0] = a[21:24]
    mesh[1][1] = a[24:27] + a[21:24]
    mesh[1][2] = a[27:30] + a[21:24]
    mesh[1][3] = a[30:33]
    mesh[1][4] = a[33:36] + a[30:33]
    mesh[1][5] = a[36:39] + a[30:33]
    return mesh, build_bvh(mesh[:, 0:3, :], [0, 2])

def run_mesh(args):
    tid, debug_mode = args
    # precompile
    print("Tid %d precompile" % tid)
    dummy_mesh, dummy_bvh = get_dummy_mesh()
    get_valid_triangle_sequence_with_group(dummy_mesh, dummy_bvh, [0, 2], 0, N_THREAD)
    bounder = Bounder()
    bounder.precompile()

    print("Tid %d start" % tid)
    sample_map_record = []

    t0 = time.perf_counter()
    init_u1v1, valid_triangle_index = get_valid_triangle_sequence_with_group(mesh, bvhs, obj_tri_cnt_prefix, tid, N_THREAD)
    t_interval_domain = time.perf_counter() - t0
    print("Tid %d tot_time_interval_domain: %.3f s" % (tid, t_interval_domain))

    task_batch_num = BATCH_SIZE

    batch_triangle_data = np.zeros((16, 3, task_batch_num), dtype=np.float64)
    batch_init_u1v1 = np.zeros((4, task_batch_num), dtype=np.float64)
    batch_id = np.zeros((task_batch_num, 2), dtype=np.int32)
    
    t_bound = 0
    t_get_result = 0
    t1 = time.perf_counter()
    batch_num = math.ceil(len(valid_triangle_index) / task_batch_num)
    for batch_i in range(batch_num):
        batch_size = min(task_batch_num, len(valid_triangle_index) - batch_i * task_batch_num)
        for inner_i in range(batch_size):
            valid_i = batch_i * task_batch_num + inner_i
            i, j = valid_triangle_index[valid_i]
            tr = mesh[i]
            tr2 = mesh[j]

            p10, p11, p12 = tr[0], tr[1] - tr[0], tr[2] - tr[0]
            n10, n11, n12 = tr[3], tr[4] - tr[3], tr[5] - tr[3]
            p20, p21, p22 = tr2[0], tr2[1] - tr2[0], tr2[2] - tr2[0]
            n20, n21, n22 = tr2[3], tr2[4] - tr2[3], tr2[5] - tr2[3]
            
            tesq_list = [p0, p10, p11, p12, n10, n11, n12, p20, p21, p22, n20, n21, n22, p30, p31, p32]
            for idx in range(16):
                batch_triangle_data[idx, :, inner_i] = tesq_list[idx]

            batch_init_u1v1[0, inner_i] = init_u1v1[valid_i][0]
            batch_init_u1v1[1, inner_i] = init_u1v1[valid_i][1]
            batch_init_u1v1[2, inner_i] = init_u1v1[valid_i][2]
            batch_init_u1v1[3, inner_i] = init_u1v1[valid_i][3]
            batch_id[inner_i, 0] = i
            batch_id[inner_i, 1] = j

        if batch_i == batch_num - 1:
            t1_bound = time.perf_counter()
            if batch_size > 0:
                bounder.set_triangle_data(batch_triangle_data, batch_init_u1v1, batch_id, batch_size)
            bounder.compute_bound3d_batched(lazy=False)
            t2_bound = time.perf_counter()
            t_bound += t2_bound - t1_bound
            bounder.get_result(sample_map_record, True)
            t_get_result += time.perf_counter() - t2_bound
        else:
            if debug_mode == 0:
                if not bounder.check_memory():
                    t1_bound = time.perf_counter()
                    bounder.compute_bound3d_batched(lazy=False)
                    t2_bound = time.perf_counter()
                    t_bound += t2_bound - t1_bound
                    bounder.get_result(sample_map_record, True)
                    t_get_result += time.perf_counter() - t2_bound
                t1_bound = time.perf_counter()
                bounder.set_triangle_data(batch_triangle_data, batch_init_u1v1, batch_id, batch_size)
                bounder.compute_bound3d_batched(lazy=True)
                t2_bound = time.perf_counter()
                t_bound += t2_bound - t1_bound
                bounder.get_result(sample_map_record, False)
                t_get_result += time.perf_counter() - t2_bound
            elif debug_mode == 1:
                t1_bound = time.perf_counter()
                bounder.set_triangle_data(batch_triangle_data, batch_init_u1v1, batch_id, batch_size)
                bounder.compute_bound3d_batched(lazy=False)
                t2_bound = time.perf_counter()
                t_bound += t2_bound - t1_bound
                bounder.get_result(sample_map_record, True)
                t_get_result += time.perf_counter() - t2_bound

    t2 = time.perf_counter()

    avg_kernel_time = [0, 0, 0, 0, 0]
    for i in range(5):
        if bounder.launch_size_counter[i] > 0:
            avg_kernel_time[i] = bounder.kernel_time[i] / bounder.launch_size_counter[i]

    print('Tid %d, bound computation time: %.3f s, get result time: %.3f s, total time: %.3f s\n'
          ' k1 total: %.3f s, avg: %.3f us\n'
          ' k2 total: %.3f s, avg: %.3f us\n'
          ' k3 total: %.3f s, avg: %.3f us\n'
          ' k4 total: %.3f s, avg: %.3f us\n'
          ' k5 total: %.3f s, avg: %.3f us\n'
          % (tid, t_bound, t_get_result, t2 - t1,
             bounder.kernel_time[0], avg_kernel_time[0] * 1e6,
             bounder.kernel_time[1], avg_kernel_time[1] * 1e6,
             bounder.kernel_time[2], avg_kernel_time[2] * 1e6,
             bounder.kernel_time[3], avg_kernel_time[3] * 1e6,
             bounder.kernel_time[4], avg_kernel_time[4] * 1e6
        )
    )

    # profile_data = [len(valid_triangle_index), t_interval_domain, t_bound, t2 - t1, bounder.kernel_time, avg_kernel_time, bounder.launch_size_counter, bounder.launch_counter]

    # with open("results/profile_%d.pkl" % tid, "wb") as f:
    #     pickle.dump(profile_data, f)

    with open("../results/sample_map_%d.pkl" % tid, "wb") as f:
        pickle.dump(sample_map_record, f)


if __name__ == "__main__":
    import sys
    argv = sys.argv
    mode = 1 if len(argv) < 2 else int(argv[1])

#     run_mesh([0, mode])
# if False:
    with multiprocessing.Pool(N_THREAD) as pool:
        args = [[i, mode] for i in range(N_THREAD)]
        pool.map(run_mesh, args)

#     total_bound_time = 0
#     total_interval_domain = 0
#     total_triangle_pair = 0
#     for _ in range(N_THREAD):
#         tmp = pickle.load(open(f"results/profile_{_}.pkl", "rb"))
#         total_triangle_pair += tmp[0]
#         total_interval_domain += tmp[1]
#         total_bound_time += tmp[2]
#     print("total bound time: %.3f s, total triangle pair: %d, total interval domain: %.3f s" % (total_bound_time, total_triangle_pair, total_interval_domain))

# if False:
#     time_data = [0, 0, [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
#     launch_data = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
#     for _ in range(N_THREAD):
#         tmp = pickle.load(open(f"results/profile_{_}.pkl", "rb"))
#         time_data[0] += tmp[0]
#         time_data[1] += tmp[1]
#         for i in range(5):
#             time_data[2][i] += tmp[2][i]
#             time_data[3][i] += tmp[3][i]
#             launch_data[0][i] += tmp[4][i]
#             launch_data[1][i] += tmp[5][i]
        
#     avg_kernel_launch_size = [0, 0, 0, 0, 0]
#     for i in range(5):
#         if launch_data[1][i] > 0:
#             avg_kernel_launch_size[i] = launch_data[0][i] / launch_data[1][i]

#     time_data[0] /= N_THREAD
#     time_data[1] /= N_THREAD
#     for i in range(5):
#         time_data[2][i] /= N_THREAD
#         time_data[3][i] /= N_THREAD
#         launch_data[0][i] /= N_THREAD
#         launch_data[1][i] /= N_THREAD

#     print(
#         'total time: %.3f s\n'
#         'bound computation time: %.3f s\n'
#         ' k1 total time: %.3f s, avg time: %.3f us, avg launch cnt: %.1f, avg launch size: %.1f\n'
#         ' k2 total time: %.3f s, avg time: %.3f us, avg launch cnt: %.1f, avg launch size: %.1f\n'
#         ' k3 total time: %.3f s, avg time: %.3f us, avg launch cnt: %.1f, avg launch size: %.1f\n'
#         ' k4 total time: %.3f s, avg time: %.3f us, avg launch cnt: %.1f, avg launch size: %.1f\n'
#         ' k5 total time: %.3f s, avg time: %.3f us, avg launch cnt: %.1f, avg launch size: %.1f\n'
#           % (time_data[1], time_data[0],
#              time_data[2][0], time_data[3][0] * 1e6, launch_data[1][0], avg_kernel_launch_size[0],
#              time_data[2][1], time_data[3][1] * 1e6, launch_data[1][1], avg_kernel_launch_size[1],
#              time_data[2][2], time_data[3][2] * 1e6, launch_data[1][2], avg_kernel_launch_size[2],
#              time_data[2][3], time_data[3][3] * 1e6, launch_data[1][3], avg_kernel_launch_size[3],
#              time_data[2][4], time_data[3][4] * 1e6, launch_data[1][4], avg_kernel_launch_size[4]
#         )
#     )

# if False:

    ans = [[[] for _ in range(RES)] for _ in range(RES)]
    ans_with_ij = [[[] for _ in range(RES)] for _ in range(RES)]

    for _ in range(N_THREAD):
        tmp = pickle.load(open(f"../results/sample_map_{_}.pkl", "rb"))
        for record in tmp:
            i, j = record[0], record[1]
            for k in range(2, len(record), 3):
                v, u, val = record[k], record[k+1], record[k+2]
                ans[v][u].append(val)
                ans_with_ij[v][u].append(val)
                ans_with_ij[v][u].append(i)
                ans_with_ij[v][u].append(j)

    with open("../results/sample_map.txt", "w") as f:
        for v in range(RES):
            for u in range(RES):
                s = ' '.join(str(round(x, 6)) for x in ans_with_ij[v][u])
                f.write(f"{len(ans[v][u])} {s}\n")
        f.close()

    img = np.zeros((RES, RES))
    for v in range(RES):
        for u in range(RES):
            img[v][u] = np.sum(min(i, 10) for i in ans[v][u])

    print("average: ", np.average(img))

    plt.xlabel("u")
    plt.ylabel("v")
    plt.imshow(np.log10(img + 1e-9), vmin=-2, vmax=4)
    plt.show()
