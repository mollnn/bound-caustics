#pragma once
#if !defined(__BSDF_H_)
#define __BSDF_H_

#include "util.h"
#include "localEigen/Dense"
#include <mitsuba/core/fresolver.h>
#include "../../bsdfs/ior.h"
#include <boost/algorithm/string.hpp>
#include <mitsuba/render/bsdf.h>
#include "../../BSDFs/microfacet.h"
using namespace Eigen;
using namespace std;

typedef Eigen::Vector3d Vec3;
typedef Eigen::Vector4i Vec4i;
typedef Eigen::Vector4d Vec4d;
typedef Eigen::Vector3d Vec3d;

MTS_NAMESPACE_BEGIN

class MyBSDF{

public:
	enum EBSDF {
		ERefelction = 0,
		ERefracion,
		EDielectricRefracion,
		EDielectricReflection
	};

	virtual Eigen::Vector3d eval(const Eigen::Vector3d &out, const Eigen::Vector3d &n) const = 0;

	EBSDF getType(){ return m_type; }

	double schlick(Eigen::Vector3d n, Eigen::Vector3d v) {
		double F0 = pow((m_eta - 1.0f), 2.0f) / pow((m_eta + 1.0f), 2.0f);
		return F0 + (1.0f - F0)*pow(1.0f - n.dot(v), 5.0f);
	}
	virtual double getEta() const = 0;

	virtual Vec3 computeWo(const Vec3 &wi, const Vec3 &n) const = 0;
	virtual int getSign() const = 0;
	EBSDF m_type;
	double m_eta;

};

class Reflection :public MyBSDF{

public:
	Reflection(string materialName){
		m_type = ERefelction;
		ref<FileResolver> fResolver = Thread::getThread()->getFileResolver();

		Spectrum intEta, intK;
		if (boost::to_lower_copy(materialName) == "none") {
			intEta = Spectrum(0.0f);
			intK = Spectrum(1.0f);
		}
		else {
			intEta.fromContinuousSpectrum(InterpolatedSpectrum(
				fResolver->resolve("data/ior/" + materialName + ".eta.spd")));
			intK.fromContinuousSpectrum(InterpolatedSpectrum(
				fResolver->resolve("data/ior/" + materialName + ".k.spd")));
		}

		Float extEta = lookupIOR("air");

		m_ref_eta = Vec3(intEta[0], intEta[1], intEta[2]) / extEta;
		m_ref_k = Vec3(intK[0], intK[1], intK[2]) / extEta;
	};

	Eigen::Vector3d eval(const Eigen::Vector3d &out, const Eigen::Vector3d &n) const
	{
		double cosThetaI = out.dot(n);
		Eigen::Vector3d F = myfresnelConductorExact(cosThetaI, m_ref_eta, m_ref_k);
		return F;
	}

	Vec3 computeWo(const Vec3 &wi, const Vec3 &n) const
	{
		return myreflect(Vec3(-wi[0], -wi[1], -wi[2]), n);
	}
	virtual int getSign() const{ return -1; }
	double getEta() const { return 1.0; }

private:
	Eigen::Vector3d m_ref_eta; //not used for now
	Eigen::Vector3d m_ref_k; // not used for now
	Eigen::Vector3d SpecularColor;

};

class DilectricRefraction :public MyBSDF{
public:
	DilectricRefraction(const double&_eta, const Vec3 &transmClor){
		m_type = EDielectricRefracion;
		m_eta = _eta;
		m_transmittanceColor = transmClor;
	};

	Eigen::Vector3d eval(const Eigen::Vector3d &out, const Eigen::Vector3d &n) const
	{
		double cosThetaI = out.dot(n);
		double cosThetaT;
		double F = 1.0 - myfresnelDielectricExt(cosThetaI, cosThetaT, m_eta); //
		double factor = (cosThetaT < 0 ? 1.0 / m_eta : m_eta);
		return Eigen::Vector3d(F * m_transmittanceColor[0], F* m_transmittanceColor[1], F* m_transmittanceColor[2]) * factor * factor;
	}

	Vec3 computeWo(const Vec3 &wi, const Vec3 &n) const
	{
		return myrefract(Vec3(-wi[0], -wi[1], -wi[2]), n, m_eta);
	}

	virtual int getSign() const{ return 1; }
	double getEta() const { return m_eta; }

private:

	Vec3 m_transmittanceColor;
};


class DilectricReflection :public MyBSDF{
public:
	DilectricReflection(const double&_eta){
		m_type = EDielectricReflection;
		m_eta = _eta;
	};


	Eigen::Vector3d eval(const Eigen::Vector3d &out, const Eigen::Vector3d &n) const
	{
		double cosThetaI = out.dot(n);
		double cosThetaT;
		double F = myfresnelDielectricExt(cosThetaI, cosThetaT, m_eta); //

		return Eigen::Vector3d(F, F, F);
	}

	Vec3 computeWo(const Vec3 &wi, const Vec3 &n) const
	{
		return myreflect(Vec3(-wi[0], -wi[1], -wi[2]), n);
	}

	virtual int getSign() const{ return -2; }
	double getEta() const { return 1.0; } 


};

MTS_NAMESPACE_END
#endif /* __BSDF_H_ */