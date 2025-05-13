#pragma once
#include <algorithm>
#include <iostream>
#include <fstream>
#include "../../../include/mitsuba/render/sampler.h"
#include "./bounder/bounder.h"
#include <mutex> 

#define COMPACT_MEMORY

namespace mitsuba
{
	float rn(ref<Sampler> sampler)
	{
		return sampler->next1D();
	}
	struct BoundedSampler
	{
		std::vector<float> a; // bound value
		std::vector<float> s; // prefix sum
		std::vector<float> p;
		// above arraies will be released after initialization
		
		std::vector<int> id; // we can use id to shuffle the outside triangle_id_map, so we do not count id's memory
		std::vector<std::vector<float>> b;  // each bin. THE ACTUAL DISTRIBUTION STORAGE
		std::vector<float> bt; // b[0] is not needed, so we should not count bt's memory
		
		void clear()
		{
			a.swap(std::vector<float>());
			s.swap(std::vector<float>());
			p.swap(std::vector<float>());
			id.swap(std::vector<int>());
			for (auto &i: b)
			{
				i.swap(std::vector<float>());
			}
			b.swap(std::vector<std::vector<float>>());
			bt.swap(std::vector<float>());
		}

		void bin_packing()
		{
			int i = 0;
			std::vector<float> a;
			float sum = 0;
			for (; i < p.size(); i++)
			{
				if (sum + p[i] <= 1)
				{
					sum += p[i];
					a.push_back(p[i]);
				}
				else
				{
					// for (auto &x : a)
					// 	x /= sum;
					bt.push_back(sum);
					for (int j = 1; j < a.size(); j++)
						a[j] += a[j - 1];
					for (int j = a.size(); j > 1; j--)
						a[j - 1] = a[j - 2];
					a[0] = 0;
					b.push_back(a);
					a.clear();
					sum = 0;
					i--;
				}
			}
			if (a.size())
			{
				// for (auto &x : a)
				// 	x /= sum;
				bt.push_back(sum);
				for (int j = 1; j < a.size(); j++)
					a[j] += a[j - 1];
				for (int j = a.size(); j > 1; j--)
					a[j - 1] = a[j - 2];
				a[0] = 0;
				b.push_back(a);
			}
		}

		float slope = 0;

		void init(const std::vector<float>& h)
		{
			std::vector<std::pair<float, int>> tmp;
			for (int i = 0; i < h.size(); i++)
			{
				if (h[i] > 0)
					tmp.push_back({ h[i], i });
			}
			sort(tmp.begin(), tmp.end());

			int n = tmp.size();
			if (n == 0)
				return;
			a.resize(n);
			id.resize(n);
			s.resize(n);

			for (int i = 0; i < n; i++)
			{
				a[i] = tmp[i].first;
				id[i] = tmp[i].second;
			}
			s[0] = a[0];
			for (int i = 1; i < n; i++)
			{

				s[i] = s[i - 1] + a[i];
			}
		}

		float determine_slope(float expected_samples)
		{
			int n = a.size();
			int l = 0, r = n;
			while (l < r)
			{
				int m = (l + r) / 2;
				if (1.0f * s[m] / (a[m] + 1e-9) < expected_samples + m - n + 1)
				{
					r = m;
				}
				else
				{
					l = m + 1;
				}
			}
			if (l == 0)
			{
				return 1e-9;
			}
			if (l == n)
			{
				return (s[l - 1] + 1e-9) / expected_samples;
			}
			return (s[l - 1] + 1e-9) / (expected_samples + l - n);
		}

		float deterine_slope_var(float exp_var)
		{
			int n = a.size();
			float l = 1e-99, r = 1e9;
			while (r - l > 1e-5)
			{
				float mid = (l + r) / 2;
				float v = 0;
				for (int i = 0; i < n; i++)
				{
					float p = std::min(1.0f, a[i] / mid);
					if (g_use_max_var)
						v += a[i] * a[i] * std::max((p < 1 ? 1.0 : 0.0), (1.0 / p - 1) * (1.0 / p - 1)); // maximum variance
					else
						v += a[i] * a[i] * (1.0 / p - 1); // expected variance
				}
				if (v > exp_var)
					r = mid;
				else
					l = mid;
			}
			return l;
		}

		void prepare_p(float ev)
		{
			if (g_force_gamma > 0)
			{
				slope = 1.0 / g_force_gamma;
			}
			else if (g_force_sample == 0)
			{
				slope = deterine_slope_var(ev);
			}
			else
			{
				slope = determine_slope(g_force_sample);
			}
			int cnt = 0;
			if (a.size() == 0)
			{
				return;
			}
			p.resize(a.size());
			for (int i = 0; i < a.size(); i++)
			{
				p[i] = std::min(1.0f, std::max(1e-15f, a[i] / slope));
			}		
			bin_packing();
			a.swap(std::vector<float>());
			s.swap(std::vector<float>());
			p.swap(std::vector<float>());
		}

		std::vector<std::pair<int, float>> sample(ref<Sampler> sample)
		{
			if (b.size() == 0)
			{
				return {};
			}
			std::vector<std::pair<int, float>> res;

			int cnt=0;

			for (int i=0;i<b.size();i++)
			{
				auto &x=b[i];
				float r = sample->next1D();
				if (r < bt[i])
				{
					int z = std::lower_bound(x.begin(), x.end(), r) - x.begin() - 1;
					if (z < 0) z = 0;
					int pos = z + cnt;
					float pdf = z + 1 == x.size() ? bt[i] - x.back() : x[z + 1] - x[z];
					res.push_back({id[pos], pdf});
				}
				cnt += x.size();
			}


			return res;
		}

		// todo: optimize when some p[i] are very small, pack them as a regular distributino to do logn sampling
	};

#ifdef COMPACT_MEMORY
	struct CompactBoundedSampler
	{
		std::vector<int> id; // we can use id to shuffle the outside triangle_id_map, so we do not count id's memory
		std::vector<std::vector<float>> b;  // each bin. THE ACTUAL DISTRIBUTION STORAGE
		std::vector<float> bt; // b[0] is not needed, so we should not count bt's memory

		std::vector<std::pair<int, float>> sample(ref<Sampler> sample)
		{
			if (b.size() == 0)
			{
				return {};
			}
			std::vector<std::pair<int, float>> res;

			int cnt=0;

			for (int i=0;i<b.size();i++)
			{
				auto &x=b[i];
				float r = sample->next1D();
				if (r < bt[i])
				{
					int z = std::lower_bound(x.begin(), x.end(), r) - x.begin() - 1;
					if (z < 0) z = 0;
					int pos = z + cnt;
					float pdf = z + 1 == x.size() ? bt[i] - x.back() : x[z + 1] - x[z];
					res.push_back({id[pos], pdf});
				}
				cnt += x.size();
			}

			return res;
		}
	};
#endif

	BoundedSampler bounded_samplers_global[RES][RES];
	std::vector<std::pair<int, int>> triangle_id_map[RES][RES];  // 8B actually for compatibility. we report 2B for single bounce (only one uint16_t). likewise, 4B for double bounce.
	#ifdef COMPACT_MEMORY
	CompactBoundedSampler cbsg[RES][RES];
	#endif
	struct SpecularDistribution
	{
		void load(const std::string& name, int bounce)
		{
			std::string::size_type pos = name.find_last_of(".");
			if (pos != std::string::npos)
			{
				std::string ext = name.substr(pos);
				if (ext == ".txt")
				{
					{
						std::string line;
						int cnt = 0;
						std::ifstream count_file(name);
						while (std::getline(count_file, line)) {
							cnt++;
						}
						count_file.close();
						
						res = (int)sqrt(cnt);   // overwrite scene xml res
					}
					std::ifstream ifs(name);
					int sum = 0;
					for (int i = 0; i < res; i++)
					{
						for (int j = 0; j < res; j++)
						{
							int n;
							ifs >> n;
							std::vector<float> a(n);
							for (int k = 0; k < n; k++)
							{
								int ti, tj;
								ifs >> a[k];
								a[k] = std::max(g_distr_min, std::min(g_distr_max, a[k] * 1.0));
								ifs >> ti;
								tj = ti;
								if (bounce == 2)
									ifs >> tj;
								triangle_id_map[i][j].push_back({ ti, tj });
							}
							bounded_samplers_global[i][j].init(a);
							sum += n;
						}
					}
				}
				else if (ext == ".obj")
				{
					run_bounder(name);

                    float max_pos_time = 0.0f;
                    float max_irr_time = 0.0f;
                    float max_splat_time = 0.0f;
                    float avg_total_time = 0.0f;
                    float avg_pos_time = 0.0f;
                    float avg_irr_time = 0.0f;
                    float avg_construct_time = 0.0f;
                    float avg_splat_time = 0.0f;
                    int actual_threads = 0;

                    for (int i = 0; i < MAX_THREAD; i++)
                    {
                        if (pos_time[i] > max_pos_time)
                        {
                            max_pos_time = pos_time[i];
                        }
                        if (irr_time[i] > max_irr_time)
                        {
                            max_irr_time = irr_time[i];
                        }
                        if (splat_time[i] > max_splat_time)
                        {
                            max_splat_time = splat_time[i];
                        }
                        avg_pos_time += pos_time[i];
                        avg_construct_time += construct_time[i];
                        avg_irr_time += irr_time[i];
                        avg_total_time += total_time[i];
                        avg_splat_time += splat_time[i];

                        if (total_time[i] != 0)
                        {
                            std::cout << "Thread " << i << ": " << total_time[i] << " s" << std::endl;
                            actual_threads++;
                        }
                    }

#ifdef COUNTER
                    std::cout << "pos counter: " << pos_counter << std::endl;
                    std::cout << "irr counter: " << irr_counter << std::endl;
#endif
                    std::cout << "actual threads: " << actual_threads << std::endl;
                    std::cout << "avg construct time: " << avg_construct_time / actual_threads << " s" << std::endl;
                    std::cout << "avg pos time: " << avg_pos_time / actual_threads << " s" << std::endl;
                    std::cout << "avg irr time: " << avg_irr_time / actual_threads << " s" << std::endl;
                    std::cout << "avg splat time: " << avg_splat_time / actual_threads << " s" << std::endl;
                    std::cout << "avg bounding time: " << (avg_pos_time + avg_irr_time) / actual_threads << " s" << std::endl;
                    std::cout << "actual total time: " << avg_total_time << " s" << std::endl;
                    std::cout << "calculate total time: " << avg_pos_time + avg_irr_time + avg_splat_time << " s" << std::endl;
                    std::cout << "max pos time: " << max_pos_time << " s" << std::endl;
                    std::cout << "max irr time: " << max_irr_time << " s" << std::endl;
                    std::cout << "max rec time: " << max_splat_time << " s" << std::endl;

					std::cout << "merge and init " << std::endl;

                    auto start_time = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, res / MAX_THREAD)
                    for (int idx = 0; idx < res * res; ++idx)
                    {
                        int i = idx / res;
                        int j = idx % res;
                    
                        std::vector<float> a;
                        for (int k = 0; k < actual_threads; ++k)
                        {
                            for (int n = 0; n < mesh_mem[k][i][j].size(); n++)
                            {
                                triangle_id_map[i][j].push_back({ std::get<1>(mesh_mem[k][i][j][n]), std::get<1>(mesh_mem[k][i][j][n]) });
                                a.push_back(std::max(g_distr_min, std::min(g_distr_max, std::get<0>(mesh_mem[k][i][j][n]) * 1.0)));
                                // a.push_back(std::get<0>(mesh_mem[k][i][j][n]));
                            }
							mesh_mem[k][i][j].swap(std::vector<std::pair<double, int>>());
                        }
                        bounded_samplers_global[i][j].init(a);
                    }
                    auto end_time = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time) * 1e-9;
                    std::cout << "merge and init time: " << duration.count() << " s" << std::endl;
                    std::cout << "avg rec time: " << (avg_splat_time + duration.count()) / actual_threads << " s" << std::endl;
                }
			}

			int max_grid_size = 0;
			long long sum_grid_size = 0;
			for (int i = 0; i < res; i++)
			{
				for (int j = 0; j < res; j++)
				{
					max_grid_size = std::max(1ull * max_grid_size, bounded_samplers_global[i][j].a.size());
					sum_grid_size += bounded_samplers_global[i][j].a.size();
				}
			}
			std::cout << "max_grid_size " << max_grid_size << std::endl;
			std::cout << "sum_grid_size " << sum_grid_size << std::endl;

			std::cout << "prepare p " << std::endl;

			double distr_mem_size = 0;
#pragma omp parallel for
			for (int i = 0; i < res; i++)
			{
				double s = 0;
				for (int j = 0; j < res; j++)
				{
					bounded_samplers_global[i][j].prepare_p(g_spec_var);
					#ifdef COMPACT_MEMORY
					for (int k=0;k<bounded_samplers_global[i][j].id.size();k++)
					{
						cbsg[i][j].id.push_back(triangle_id_map[i][j][bounded_samplers_global[i][j].id[k]].first);
					}
					cbsg[i][j].id.shrink_to_fit();
					cbsg[i][j].b = bounded_samplers_global[i][j].b;
					cbsg[i][j].b.shrink_to_fit();
					cbsg[i][j].bt = bounded_samplers_global[i][j].bt;
					cbsg[i][j].bt.shrink_to_fit();
					bounded_samplers_global[i][j].clear();
					triangle_id_map[i][j].clear();
					triangle_id_map[i][j].shrink_to_fit();
					s += (cbsg[i][j].id.size() * (4 + 2) + cbsg[i][j].b.size() * (2.125));   // 17 bit for int. we can steal one bit from float
					#endif
				}
				distr_mem_size += s;
			}

			std::cout << "done  distr storage = " << distr_mem_size / 1048576 << " MB" << std::endl;
		}

		std::vector<std::tuple<int, int, float>> sample(float u, float v, ref<Sampler> sampler)
		{
			int i = std::min(res - 1, std::max(0, int(v * res)));
			int j = std::min(res - 1, std::max(0, int(u * res)));
			#ifdef COMPACT_MEMORY
			auto ans = cbsg[i][j].sample(sampler);
			#else
			auto ans = bounded_samplers_global[i][j].sample(sampler);
			#endif
			std::vector<std::tuple<int, int, float>> res;
			for (auto [id, p] : ans)
			{
				#ifdef COMPACT_MEMORY
				res.push_back({id, id, p});
				#else
				res.push_back({ triangle_id_map[i][j][id].first, triangle_id_map[i][j][id].second, p });
				#endif
			}
			return res;
		}
	};
}