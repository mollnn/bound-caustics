/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#if !defined(__UTIL_H_)
#define __UTIL_H_


#include <Eigen/Dense>
#include <vector>

using namespace Eigen;

Eigen::Vector3d myreflect(const Eigen::Vector3d &wi, const Eigen::Vector3d  &n);
Eigen::Vector3d myrefract(const Eigen::Vector3d &wi, const Eigen::Vector3d  &n, double eta);
void coordinateSystem(const Eigen::Vector3d &a, Eigen::Vector3d &b, Eigen::Vector3d &c);
void mycoordinateSystem(const Eigen::Vector3d &a, Eigen::Vector3d &b, Eigen::Vector3d &c);
double myfresnelDielectricExt(double cosThetaI,
	double &cosThetaT, double eta);
double myfresnelDielectric(double cosThetaI,
	double cosThetaT, double eta);
Eigen::Vector3d myfresnelConductorExact(double cosThetaI,
	const Eigen::Vector3d &eta, const Eigen::Vector3d &k);


inline Vector3d divide (const Vector3d &first, const Vector3d &other)  {
	Vector3d value = first;

	for (int i=0; i<3; i++)
		value[i] /= other[i];

	return value;
}
inline Vector3d multiply(const Vector3d &first, const Vector3d &other)  {
	Vector3d value = first;

	for (int i=0; i<3; i++)
		value[i] *= other[i];

	return value;
}

inline Vector3d sqrt(const Vector3d &first)  {
	Vector3d value = first;

	for (int i=0; i<3; i++)
		value[i] =sqrt(value[i]);

	return value;
}
double myfresnelDielectric(double cosThetaI, double cosThetaT, double eta) {
	if (eta == 1)
		return 0.0f;

	double Rs = (cosThetaI - eta * cosThetaT)
		/ (cosThetaI + eta * cosThetaT);
	double Rp = (eta * cosThetaI - cosThetaT)
		/ (eta * cosThetaI + cosThetaT);

	/* No polarization -- return the unpolarized reflectance */
	return 0.5f * (Rs * Rs + Rp * Rp);
}


double myfresnelDielectricExt(double cosThetaI_, double &cosThetaT_, double eta) {
	if (eta == 1) {
		cosThetaT_ = -cosThetaI_;
		return 0.0f;
	}

	/* Using Snell's law, calculate the squared sine of the
	angle between the Vector3d  and the transmitted ray */
	double scale = (cosThetaI_ > 0) ? 1 / eta : eta,
		cosThetaTSqr = 1 - (1 - cosThetaI_*cosThetaI_) * (scale*scale);

	/* Check for total internal reflection */
	if (cosThetaTSqr <= 0.0f) {
		cosThetaT_ = 0.0f;
		return 1.0f;
	}

	/* Find the absolute cosines of the incident/transmitted rays */
	double cosThetaI = std::abs(cosThetaI_);
	double cosThetaT = std::sqrt(cosThetaTSqr);

	double Rs = (cosThetaI - eta * cosThetaT)
		/ (cosThetaI + eta * cosThetaT);
	double Rp = (eta * cosThetaI - cosThetaT)
		/ (eta * cosThetaI + cosThetaT);

	cosThetaT_ = (cosThetaI_ > 0) ? -cosThetaT : cosThetaT;

	/* No polarization -- return the unpolarized reflectance */
	return 0.5f * (Rs * Rs + Rp * Rp);
}

Eigen::Vector3d myfresnelConductorExact(double cosThetaI, const Eigen::Vector3d &eta, const Eigen::Vector3d &k) {
	/* Modified from "Optics" by K.D. Moeller, University Science Books, 1988 */

	double cosThetaI2 = cosThetaI*cosThetaI,
	      sinThetaI2 = 1-cosThetaI2,
		  sinThetaI4 = sinThetaI2*sinThetaI2;

	Eigen::Vector3d temp1 = multiply(eta, eta) - multiply(k, k) - Eigen::Vector3d(sinThetaI2, sinThetaI2, sinThetaI2),
		a2pb2 = sqrt(multiply(temp1, temp1) + multiply(multiply(k, k),multiply(eta, eta)) * 4),
			 a = sqrt((a2pb2 + temp1) * 0.5f);

	Eigen::Vector3d term1 = a2pb2 + Eigen::Vector3d(cosThetaI2, cosThetaI2, cosThetaI2),
	         term2 = a*(2*cosThetaI);

	Eigen::Vector3d Rs2 = divide((term1 - term2), (term1 + term2));
	// do not find the same function in eigan


	Eigen::Vector3d term3 = a2pb2*cosThetaI2 + Eigen::Vector3d(sinThetaI4, sinThetaI4, sinThetaI4),
	         term4 = term2*sinThetaI2;

	Eigen::Vector3d Rp2 = divide(multiply(Rs2, (term3 - term4)), (term3 + term4));

	return 0.5f * (Rp2 + Rs2);


}

Eigen::Vector3d myreflect(const Eigen::Vector3d &wi, const Eigen::Vector3d  &n) {
	return 2 * wi.dot(n) * n - wi;
}

inline double signum(double value) {
#if defined(__WINDOWS__)
	return (double)_copysign(1.0, value);
#elif defined(SINGLE_PRECISION)
	return copysignf((double) 1.0, value);
#elif defined(DOUBLE_PRECISION)
	return copysign((double) 1.0, value);
#endif
}

Eigen::Vector3d myrefract(const Eigen::Vector3d &wi, const Eigen::Vector3d  &n, double eta) {
	if (eta == 1)
		return -wi;

	double cosThetaI = wi.dot(n);
	if (cosThetaI > 0)
		eta = 1 / eta;

	/* Using Snell's law, calculate the squared sine of the
	angle between the Vector3d  and the transmitted ray */
	double cosThetaTSqr = 1 - (1 - cosThetaI*cosThetaI) * (eta*eta);

	/* Check for total internal reflection */
	if (cosThetaTSqr <= 0.0f)
		return Eigen::Vector3d(0.0f,0.0,0.0);

	return n * (cosThetaI * eta - signum(cosThetaI)
		* std::sqrt(cosThetaTSqr)) - wi * eta;
}

void coordinateSystem(const Vector3d &a, Vector3d &b, Vector3d &c) {
	if (std::abs(a[0]) > std::abs(a[1])) {
		double invLen = 1.0f / std::sqrt(a[0] * a[0] + a[2] * a[2]);
		c = Vector3d(a[2] * invLen, 0.0f, -a[0] * invLen);
	}
	else {
		double invLen = 1.0f / std::sqrt(a[1] * a[1] + a[2] * a[2]);
		c = Vector3d(0.0f, a[2] * invLen, -a[1] * invLen);
	}
	b = c.cross(a);
}

void mycoordinateSystem(const Vector3d &a, Vector3d &b, Vector3d &c) {
	if (std::abs(a[0]) > std::abs(a[1])) {
		double invLen = 1.0f / std::sqrt(a[0] * a[0] + a[2] * a[2]);
		c = Vector3d(a[2] * invLen, 0.0f, -a[0] * invLen);
	}
	else {
		double invLen = 1.0f / std::sqrt(a[1] * a[1] + a[2] * a[2]);
		c = Vector3d(0.0f, a[2] * invLen, -a[1] * invLen);
	}
	b = c.cross(a);
}


#endif /* __MITSUBA_CORE_UTIL_H_ */
