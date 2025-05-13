#pragma once
#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <iomanip>
#include <complex>
#include <fstream>
#include <sstream> // to print some info
#include <limits>
#include <array>
#include "resultant_qz.h"
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
// #include <Python.h>
// #include <numpy/arrayobject.h>
#include "cyPolynomial.h"
#pragma comment(linker, "/STACK:8589934592") // avoid stack overflow
using namespace Eigen;

namespace Resultant
{
	template <bool use_unit_root = false>
	inline std::pair<std::vector<std::pair<UnivariatePolynomial, std::pair<double, double>>>, UnivariatePolynomialMatrix> resultant_elimination(const BivariatePolynomial &poly1, const BivariatePolynomial &poly2, int cutoff_matrix, int cutoff_resultant, double valid_v_min, double valid_v_max)
	{
		auto ___time_begin = std::chrono::high_resolution_clock::now();

		auto poly2_ = poly1 * BivariatePolynomial(1, 0) + poly2; // d(poly1) >= d(poly2)

		std::vector<UnivariatePolynomial> poly1Mat = poly1.toUnivariatePolynomials();
		std::vector<UnivariatePolynomial> poly2Mat = poly2_.toUnivariatePolynomials();

		UnivariatePolynomialMatrix bezoutMat = bezout_matrix(poly1Mat, poly2Mat);

		std::pair<std::vector<pair<UnivariatePolynomial, pair<double, double>>>, UnivariatePolynomialMatrix> res;
		res.second = bezoutMat;
		// print values of bezout matrix (only when log is enabled)
		if (global_method_mask & METHODMASK_SPLINE_GAUSSIAN_WITH_LOG || global_method_mask & METHODMASK_DICHOTOMY_GAUSSIAN_WITH_LOG)
		{
			int num_samples = 1001;
			std::vector<double> x(num_samples, 0);
			std::vector<double> y(num_samples, 0);
			y[0] = evalMatrixPolynomialDeterminant(bezoutMat, 0);
			int signal = (y[0] > 0) ? 1 : (y[0] < 0 ? -1 : 0);
			bool has_roots = false;
			for (int i = 1; i < num_samples; i++)
			{
				x[i] = static_cast<double>(i) / (num_samples - 1);
				y[i] = evalMatrixPolynomialDeterminant(bezoutMat, x[i]);
				if (signal * y[i] <= 0)
					has_roots = true;
			}
			if (has_roots)
			{
				for (int i = 0; i < num_samples; i++)
				{
					ss << x[i] << " " << y[i] << " ";
				}
			}
		}

		auto ___time_end = std::chrono::high_resolution_clock::now();
		___perf_time_bezout += std::chrono::duration_cast<std::chrono::nanoseconds>(___time_end - ___time_begin).count() * 1e-3;
		___time_begin = ___time_end;

		auto det = (global_method_mask & METHODMASK_LAGRANGE_GAUSSIAN) ? determinantLagrange(bezoutMat, valid_v_min, valid_v_max) : (global_method_mask & METHODMASK_DICHOTOMY_GAUSSIAN || (global_method_mask & METHODMASK_DICHOTOMY_GAUSSIAN_WITH_LOG)) ? std::vector<std::pair<UnivariatePolynomial, std::pair<double, double>>>()
																																: ((global_method_mask & METHODMASK_SPLINE_GAUSSIAN) || (global_method_mask & METHODMASK_SPLINE_GAUSSIAN_WITH_LOG))		  ? cubicSplineInterpolator(bezoutMat, valid_v_min, valid_v_max)
																																																														  : determinant(bezoutMat);
		// det.resize(cutoff_resultant + 1);
		// det.normalize();
		res.first = det;

		___time_end = std::chrono::high_resolution_clock::now();
		___perf_time_determinant += std::chrono::duration_cast<std::chrono::nanoseconds>(___time_end - ___time_begin).count() * 1e-3;
		___time_begin = ___time_end;

		return res;
	}

#ifdef QZ_SOLVER
	template <bool use_unit_root = false>
	inline std::vector<double> resultant_elimination_qz(const BivariatePolynomial &poly1, const BivariatePolynomial &poly2, int cutoff_matrix, int cutoff_resultant)
	{
		std::vector<double> result_u;
		auto ___time_begin = std::chrono::high_resolution_clock::now();

		std::vector<UnivariatePolynomial> poly1Mat = poly1.toUnivariatePolynomials();
		std::vector<UnivariatePolynomial> poly2Mat = poly2.toUnivariatePolynomials();

		UnivariatePolynomialMatrix bezoutMat = bezout_matrix(poly1Mat, poly2Mat);

		auto ___time_end = std::chrono::high_resolution_clock::now();
		___perf_time_bezout += std::chrono::duration_cast<std::chrono::nanoseconds>(___time_end - ___time_begin).count() * 1e-3;
		___time_begin = ___time_end;

		result_u = eigen_solver.eval_eigen(bezoutMat);
		___time_end = std::chrono::high_resolution_clock::now();
		___perf_time_bezout += std::chrono::duration_cast<std::chrono::nanoseconds>(___time_end - ___time_begin).count() * 1e-3;
		___time_begin = ___time_end;
		return result_u;
	}
#endif

	std::vector<std::tuple<double, double, double, double>> solve_equ(const std::vector<pair<UnivariatePolynomial, pair<double, double>>> &poly_vec_u, const UnivariatePolynomialMatrix &bezout_matrix, BivariatePolynomial &Cxz,
																	  const BivariatePolynomial &u2hat, const BivariatePolynomial &v2hat, const BivariatePolynomial &kappa2, int chain_type, double valid_v_min, double valid_v_max)
	{
		perf_count++;

		auto ___time_begin = std::chrono::high_resolution_clock::now();
		auto ___time_end = ___time_begin;

		// Solver begin
		double roots[9 * (N_POLY - 1)], roots2[9 * (N_POLY - 1)];
		int numRoots = 0;
		double doubleervalMin = -0.01;
		double doubleervalMax = 1.01;
		double errorThreshold = 1e-9;
		double x_tolerance = 0.0001;
		if (global_method_mask & METHODMASK_DICHOTOMY_GAUSSIAN || global_method_mask & METHODMASK_DICHOTOMY_GAUSSIAN_WITH_LOG)
		{
			int n = global_poly_cutoff + 1;

			int max_iteration = max_iteration_dichotomy;
			std::vector<double> x(n);
			std::vector<double> y(n);

			int idx = 0;

			std::vector<int> logOfBisection(n - 1, 0);
			std::vector<double> logOfBisection_sol(n - 1, 0);
			std::vector<double> logOfBisection_solval(n - 1, 0);

			for (int i = 0; i < n; i++)
			{
				x[i] = i * 1.0f / (n - 1) * (valid_v_max - valid_v_min) + valid_v_min;
				y[i] = evalMatrixPolynomialDeterminant(bezout_matrix, x[i]);
				if (y[i] == 0.0)
				{
					roots[idx++] = x[i];
					numRoots++;
				}
				if (i > 0 && y[i] * y[i - 1] <= 0.0 && y[i - 1] != 0.0 && y[i] != 0.0)
				{
					logOfBisection[i - 1] = 1;
					double lb = x[i - 1], rb = x[i];
					double mid_value = 0;
					double mid = 0;
					for (int k = 1; k <= max_iteration; k++)
					{
						mid = (lb + rb) / 2;
						mid_value = evalMatrixPolynomialDeterminant(bezout_matrix, mid);
						// std::cout << k << " " << mid << " " << mid_value << endl;
						if (mid_value * y[i] < 0.0)
						{
							lb = mid;
						}
						else
						{
							rb = mid;
						}
					}
					// std::cout << "final" << " " << mid << " " << mid_value << endl;
					roots[idx++] = mid;
					logOfBisection_sol[i - 1] = mid;
					logOfBisection_solval[i - 1] = mid_value;
					numRoots++;
				}
			}

			if (global_method_mask & METHODMASK_DICHOTOMY_GAUSSIAN_WITH_LOG && !ss.str().empty())
			{
				ofstream ofs_resultant;
				if (isFirstCall)
				{
					ofs_resultant.open("resultant_info.txt", std::ios::trunc);
					isFirstCall = false;
				}
				else
				{
					ofs_resultant.open("resultant_info.txt", std::ios::app);
				}
				if (ofs_resultant.is_open())
				{
					ofs_resultant << ss.str() << std::endl;
					ofs_resultant << n - 1 << std::endl;
					for (int i = 0; i < n - 1; i++)
					{
						ofs_resultant << logOfBisection[i] << " " << logOfBisection_sol[i] << " " << logOfBisection_solval[i] << endl;
					}
					ofs_resultant.close();
				}
			}
		}
		else if (global_method_mask & METHODMASK_SPLINE_GAUSSIAN || global_method_mask & METHODMASK_SPLINE_GAUSSIAN_WITH_LOG)
		{ // cubic spline interpolation method
			double errorThreshold = 1e-9;

			int idx = 0;
			for (int i = 0; i < poly_vec_u.size(); i++)
			{
				cy::Polynomial<double, 3> temp_poly;
				for (int j = 0; j < 4; j++)
				{
					temp_poly.coef[j] = poly_vec_u[i].first.coeffs[j];
				}
				double lb = poly_vec_u[i].second.first;
				double rb = poly_vec_u[i].second.second;
				double bias = 1e233;
				double bias2 = 0;
				int le0 = 0, ge0 = 0;
				const int NSAMPLES = 16;
				for (int k = 0; k <= NSAMPLES; k++)
				{
					double x = lb + (rb - lb) * k / NSAMPLES;
					double y = temp_poly.coef[0] + x * (temp_poly.coef[1] + x * (temp_poly.coef[2] + x * temp_poly.coef[3]));
					if (y < 0)
						le0++;
					if (y > 0)
						ge0++;
					bias = std::min(bias, std::abs(y));
					bias2 += std::abs(y);
				}
				bias *= 1.001;
				bias += bias2 / NSAMPLES * 0.001;
				for (int k = 0; k < 3; k++)
				{
					if (k == 0)
						if (le0 != 0 || ge0 == 0)
							continue;
					if (k == 2)
						if (ge0 != 0 || le0 == 0)
							continue;
					double temp_roots[3];
					int temp_num_roots = temp_poly.Roots(temp_roots, poly_vec_u[i].second.first - x_tolerance, poly_vec_u[i].second.second + x_tolerance, errorThreshold);
					for (int j = 0; j < temp_num_roots; j++)
					{
						if (temp_roots[j] >= poly_vec_u[i].second.first - x_tolerance && temp_roots[j] <= poly_vec_u[i].second.second + x_tolerance)
						{ // to ensure that roots belong to [poly_u[i].second.first, poly_u[i].second.second], actually, it's not necessary
							double x = temp_roots[j];
							// for (int itr = 0; itr < 1; itr ++) {
							//     double eps = 1e-4;
							//     double y = evalMatrixPolynomialDeterminant(bezout_matrix, x);
							//     double y1 = evalMatrixPolynomialDeterminant(bezout_matrix, x + eps);
							//     double slope = (y1 - y) / eps;
							//     double x_new = x - y / slope * std::pow(0.7, itr);
							//     x_new = max(x_new, -0.1);
							//     x_new = min(x_new, 1.1);
							//     double y_new = evalMatrixPolynomialDeterminant(bezout_matrix, x_new);
							//     if (std::abs(y_new) < std::abs(y)) {
							//         // std::cout << "newton succ " << y << " " << y_new << std::endl;
							//         x = x_new;
							//     }
							// }
							roots[idx++] = x;
							numRoots++;
						}
					}
					temp_poly.coef[0] += bias;
				}
				temp_poly.coef[0] -= bias * 2;
			}
			// print info
			if (!ss.str().empty())
			{
				ofstream ofs_resultant;
				if (isFirstCall)
				{
					ofs_resultant.open("resultant_info.txt", std::ios::trunc);
					isFirstCall = false;
				}
				else
				{
					ofs_resultant.open("resultant_info.txt", std::ios::app);
				}
				if (ofs_resultant.is_open())
				{
					ofs_resultant << ss.str() << std::endl;
					ofs_resultant << poly_vec_u.size() << std::endl;
					for (int i = 0; i < poly_vec_u.size(); i++)
					{
						ofs_resultant << poly_vec_u[i].first.coeffs[0] << " " << poly_vec_u[i].first.coeffs[1] << " " << poly_vec_u[i].first.coeffs[2] << " " << poly_vec_u[i].first.coeffs[3] << std::endl;
					}
					ofs_resultant.close();
				}
			}
		}
		else
		{ // other method
			// todo: add tolerance
			UnivariatePolynomial poly_u = poly_vec_u[0].first;

			int maxdeg = 1;
			double maxcoef = 0;
			for (int i = 1; i < N_POLY; i++)
			{
				if (abs(poly_u.coeffs[i]) > maxcoef)
				{
					maxcoef = abs(poly_u.coeffs[i]);
				}
			}
			for (int i = 1; i < N_POLY; i++)
			{
				if (abs(poly_u.coeffs[i]) > maxcoef * global_poly_cutoff_eps)
				{
					maxdeg = i;
				}
			}

			static int stats_max_deg_sum = 0;
			static int stats_max_deg_cnt = 0;
			static int stats_max_deg_max = 0;
			stats_max_deg_sum += maxdeg;
			stats_max_deg_cnt += 1;
			stats_max_deg_max = std::max(stats_max_deg_max, maxdeg);
			// if (rand() % 100) std::cout << "maxdeg " << stats_max_deg_max << std::endl;

			const int const_deg = N_POLY - 1; // ! N of cy::Polynomial is degree (= length - 1)
			cy::Polynomial<double, const_deg> poly;
			for (int i = 0; i <= const_deg; i++)
			{
				poly.coef[i] = poly_u.coeffs[i];
			}
			numRoots = poly.Roots(roots, valid_v_min - 0.01, valid_v_max + 0.01, errorThreshold);
		}
		ss.str("");
		ss.clear();

		___time_end = std::chrono::high_resolution_clock::now();
		___perf_time_solver1 += std::chrono::duration_cast<std::chrono::nanoseconds>(___time_end - ___time_begin).count() * 1e-3;
		___time_begin = ___time_end;

		std::vector<std::tuple<double, double, double, double>> solutions;

		// delete dup
		std::vector<double> rootsVec(roots, roots + numRoots);

		// std::sort(rootsVec.begin(), rootsVec.end());

		std::vector<double> result;
#ifdef RESULTANT_TRACE
		rss << "solver1: ";
#endif
		for (auto root : rootsVec)
		{
			// if (root > 0 && root < 1 && (result.empty() || std::abs(root - result.back()) > 0.01))
#ifdef RESULTANT_TRACE
			rss << root << ", ";
#endif
			result.push_back(root);
		}
#ifdef RESULTANT_TRACE
		rss << endl;
#endif

		// solve back
		for (int i = 0; i < result.size(); i++)
		{
			double gamma = result[i];
			auto unipoly = Cxz.evaluateAtV(gamma);

			int maxdeg = 1;
			double maxcoef = 0;
			for (int i = 1; i < unipoly.size; i++)
				if (abs(unipoly.coeffs[i]) > maxcoef)
					maxcoef = abs(unipoly.coeffs[i]);
			for (int i = 1; i < unipoly.size; i++)
				if (abs(unipoly.coeffs[i]) > maxcoef * global_poly_cutoff_eps)
					maxdeg = i;

			int numRoots2 = 0;

			{									// todo: add some if-else above
				const int const_deg = N_DEGREE; // ! N of cy::Polynomial is degree (= length - 1)
				cy::Polynomial<double, const_deg> poly2;
				for (int k = 0; k <= const_deg; k++)
					poly2.coef[k] = unipoly.coeffs[k];
				numRoots2 = poly2.Roots(roots2, doubleervalMin, doubleervalMax, errorThreshold);
			}

			for (int j = 0; j < numRoots2; j++)
			{
				double beta = roots2[j];
				if (beta < -x_tolerance || beta > 1 + x_tolerance || beta + gamma > 1 + x_tolerance)
					continue;
				double alpha = 1.0 - beta - gamma;

				double u2hat_val = u2hat.evaluateAtV(gamma).evaluateAtU(beta);
				double v2hat_val = v2hat.evaluateAtV(gamma).evaluateAtU(beta);
				double kappa2_val = kappa2.evaluateAtV(gamma).evaluateAtU(beta);

				double beta2 = u2hat_val / kappa2_val;
				double gamma2 = v2hat_val / kappa2_val;
				if (chain_type < 10)
				{
					beta2 = 0.333;
					gamma2 = 0.333;
				}
				double alpha2 = 1 - beta2 - gamma2;

				if (chain_type > 10)
					if (alpha2 < 0 || beta2 < 0 || gamma2 < 0)
					{
						continue;
					}
				solutions.push_back({alpha, beta, alpha2, beta2});
			}
		}

		___time_end = std::chrono::high_resolution_clock::now();
		___perf_time_solver2 += std::chrono::duration_cast<std::chrono::nanoseconds>(___time_end - ___time_begin).count() * 1e-3;

		return solutions;
	}

#ifdef QZ_SOLVER
	std::vector<std::tuple<double, double, double, double>> solve_equ_qz(std::vector<double> result_u, const BivariatePolynomial &Cxz,
																		 const BivariatePolynomial &u2hat, const BivariatePolynomial &v2hat, const BivariatePolynomial &kappa2, int chain_type)
	{
		auto ___time_begin = std::chrono::high_resolution_clock::now();
		auto ___time_end = ___time_begin;
		int numRoots = result_u.size();
		double result_v[N_POLY];
		double doubleervalMin = 0;
		double doubleervalMax = 1;
		double errorThreshold = 1e-9;

		std::vector<std::tuple<double, double, double, double>> solutions;

		for (int i = 0; i < numRoots; i++)
		{
			double gamma = result_u[i];
			if (gamma < 0)
				gamma = 0;
			if (gamma > 1)
				gamma = 1;
			auto unipoly = Cxz.evaluateAtV(gamma);

			int maxdeg = 1;
			double maxcoef = 0;
			for (int i = 1; i < unipoly.size; i++)
				if (abs(unipoly.coeffs[i]) > maxcoef)
					maxcoef = abs(unipoly.coeffs[i]);
			for (int i = 1; i < unipoly.size; i++)
				if (abs(unipoly.coeffs[i]) > maxcoef * global_poly_cutoff_eps)
					maxdeg = i;
			int numRoots2 = 0;

			{									// todo: add some if-else above
				const int const_deg = N_DEGREE; // ! N of cy::Polynomial is degree (= length - 1)
				cy::Polynomial<double, const_deg> poly2;
				for (int k = 0; k <= const_deg; k++)
					poly2.coef[k] = unipoly.coeffs[k];
				numRoots2 = poly2.Roots(result_v, doubleervalMin, doubleervalMax, errorThreshold);
			}

			for (int j = 0; j < numRoots2; j++)
			{
				double beta = result_v[j];
				if (beta < 0)
					beta = 0;
				if (beta > 1)
					beta = 1;
				if (beta > 1 - gamma)
					beta = 1 - gamma;
				if (beta + gamma <= 1.0)
				{
					double alpha = 1.0 - beta - gamma;

					double u2hat_val = u2hat.evaluateAtV(gamma).evaluateAtU(beta);
					double v2hat_val = v2hat.evaluateAtV(gamma).evaluateAtU(beta);
					double kappa2_val = kappa2.evaluateAtV(gamma).evaluateAtU(beta);

					double beta2 = u2hat_val / kappa2_val;
					double gamma2 = v2hat_val / kappa2_val;
					if (chain_type < 10)
					{
						beta2 = 0.333;
						gamma2 = 0.333;
					}
					double alpha2 = 1 - beta2 - gamma2;

					if (chain_type > 10)
						if (alpha2 < 0 || beta2 < 0 || gamma2 < 0)
						{
							continue;
						}

					solutions.push_back({alpha, beta, alpha2, beta2});
				}
			}
		}

		___time_end = std::chrono::high_resolution_clock::now();
		___perf_time_solver2 += std::chrono::duration_cast<std::chrono::nanoseconds>(___time_end - ___time_begin).count() * 1e-3;

		return solutions;
	}
#endif

}