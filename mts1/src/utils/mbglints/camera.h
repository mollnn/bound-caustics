#pragma once
#if !defined(__CAMERA_H_)
#define __CAMERA_H_

#include "path.h"
#include "util.h"
#include "shape.h"
#include "localEigen/Dense"
using namespace Eigen;

typedef Eigen::Vector3d Vec3;
typedef Eigen::Vector2d Vec2;
typedef Eigen::Vector4d Vec4;
typedef Eigen::Vector2i Vec2i;



double deg2rad(double degree)
{
	return degree * M_PI / 180.0f;
}

Vec3 getNormal(const Vec3 & vert1, const Vec3 & vert2, const Vec3 & vert3)
{
	Vec3 v2v3 = vert3 - vert2;
	Vec3 v1v2 = vert2 - vert1;
	Vec3 N = v2v3.cross(v1v2);
	N.normalize();

	return N;
}


Vec4 plane(const Vec3 &P, const Vec3 &v1, const Vec3 &v2, const Vec3 &v3, const Vec3 &ng)
{
	Vec3 v3_to_2 = (v2 - v3).normalized();
	Vec3 p1 = v3 + (P - v3).dot(v3_to_2) * v3_to_2;
	Vec3 nA = getNormal(v2, v3, v3 + ng);

	double w = nA.dot(v2);
	Vec4 L = Vec4(nA[0], nA[1], nA[2], -w);

	double length = L.dot(Vec4(v1[0], v1[1], v1[2], 1));
	L /= length;
	return L;

}
Vec3 reflect(const Vec3 &wi, const Vec3 &n) {
	return 2 * wi.dot(n) * n - wi;
}

enum EPostOption
{
	NoProcess = 0, // Default no post-processing
	BloomTwoPass,
	BloomUpDown,
};


class Camera{
public:
	Camera(const Vec3 &_pos, const Vec3 &_dir, const Vec3 &_up, const Vec3 &_left, const Vec2i &_res, double _fov,
		const mitsuba::Transform &_cameraToSample) :
		m_pos(_pos), m_dir(_dir), m_resolution(_res), m_fov(_fov){
		m_cameraToSample = _cameraToSample;
		m_sampleToCamera = m_cameraToSample.inverse();
		m_s = _left;
		m_t = _up;

		// if (global_log_flag) cout << "m_s:" << m_s[0] << ", " << m_s[1] << "," << m_s[2] << endl;
		// if (global_log_flag) cout << "m_t:" << m_t[0] << ", " << m_t[1] << "," << m_t[2] << endl;

		m_scale = tan(deg2rad(m_fov * 0.5));
		m_imageAspectRatio = (double)m_resolution[1] / (double)m_resolution[0];

		m_A = 4 * m_imageAspectRatio * m_scale * m_scale;
		// if (global_log_flag) cout << "M_A1: " << m_resolution[0] * m_resolution[1] / m_A << endl;
		
		// postprocessing
		postProcessOption = EPostOption::NoProcess;
	};

	void setMember(const Vec3 &_pos, const Vec3 &_dir, const Vec3 &_up, const Vec3 &_left, const Vec2i &_res, double _fov,
		const mitsuba::Transform &_cameraToSample) {
		m_pos = _pos;
		m_dir = _dir;
		m_resolution = _res;
		m_fov = _fov;
		m_cameraToSample = _cameraToSample;
		m_sampleToCamera = m_cameraToSample.inverse();
		m_s = _left;
		m_t = _up;

		// if (global_log_flag) cout << "m_s:" << m_s[0] << ", " << m_s[1] << "," << m_s[2] << endl;
		// if (global_log_flag) cout << "m_t:" << m_t[0] << ", " << m_t[1] << "," << m_t[2] << endl;

		m_scale = tan(deg2rad(m_fov * 0.5));
		m_imageAspectRatio = (double)m_resolution[1] / (double)m_resolution[0];

		m_A = 4 * m_imageAspectRatio * m_scale * m_scale;
		// if (global_log_flag) cout << "M_A1: " << m_resolution[0] * m_resolution[1] / m_A << endl;
		
		// postprocessing
		postProcessOption = EPostOption::NoProcess;
	}

	void reflectionAutoDiff(const PathVert &vert, const DVec4 &P, DVec4 &D)
	{
		DVec3 n = vert.Lalpha.dot(P) * vert.Nalpha + vert.Lbeta.dot(P) * vert.Nbeta + vert.Lgamma.dot(P) * vert.Ngamma;
		DVec3 N = n / sqrt(n.dot(n));

		DVec4 N4 = DVec4(N[0], N[1], N[2], 0);
		D = D - 2.0f*(D.dot(N4))*N4;
	}

	bool refractionAutoDiff(const PathVert &vert, const DVec4 &P, DVec4 &D)
	{
		DVec3 n = vert.Lalpha.dot(P) * vert.Nalpha + vert.Lbeta.dot(P) * vert.Nbeta + vert.Lgamma.dot(P) * vert.Ngamma;
		DVec3 N = n / sqrt(n.dot(n));
		DVec4 N4 = DVec4(N[0], N[1], N[2], 0);
		DFloat DdotN = D.dot(N4);

		double eta;
		if (vert.m_bsdf->m_eta == 1 || g_bounce == 2)    // pretty hack, not support T
		{
			if (DdotN < 0)
			{
				eta = 1.0 / vert.m_bsdf->m_eta;
			}
			else
			{
				N4 = -N4;
				DdotN = D.dot(N4);
				eta = vert.m_bsdf->m_eta;
			}
		}
		else 
		{
			eta = 1.5;
		}
		DFloat term2 = -sqrt(1 - eta*eta*(1 - DdotN * DdotN));
		DFloat mu = eta * DdotN - term2;

		D = eta * D - mu * N4;
		return true;
	}

	void propagationAutoDiff(const PathVert &next_vert, DVec4 &P, const DVec4 &D)
	{
		Vec3 N = next_vert.m_N_G;
		double w = N.dot(next_vert.m_pos);
		Vec4 N4 = Vec4(N[0], N[1], N[2], -w);
		DFloat temp = D.dot(Vec4(N[0], N[1], N[2], 0));
		DFloat t = -(P.dot(N4)) / temp;
		P = P + t*D;
	}

	DVec4 pathAutoDiff(const Path &path, DFloat x, DFloat y, DVec4 &outPxy, bool &valid)
	{
		const int pathLength = path.getLength() - 2;

		DVec4 P_xy = DVec4(m_pos[0], m_pos[1], m_pos[2], 1);
		DVec3 D3 = (m_dir + x * m_s + y * m_t).normalized();

		DVec4 D_xy = DVec4(D3[0], D3[1], D3[2], 0);

		valid = true;
		for (int i = 0; i < pathLength; i++)
		{
			//vert start from the light side, but we have to move to the camera side
			const PathVert &vert = path.getVert(pathLength - i);

			//the first operator is propagation
			propagationAutoDiff(vert, P_xy, D_xy);

			if (vert.m_bsdf->getType() == MyBSDF::ERefelction)
				reflectionAutoDiff(vert, P_xy, D_xy);
			else
				valid = refractionAutoDiff(vert, P_xy, D_xy);
		}

		propagationAutoDiff(path.getVert(0), P_xy, D_xy);

		outPxy = P_xy;
		return D_xy;
	}

	bool  computeDifferentialAutoDiff(const Path &path, Vec3 &out_dDdx, Vec3 &out_dDdy, Vec3 &out_dPdx, Vec3 &out_dPdy, double& fac)
	{
		const int pathLength = path.getLength() - 2;
		Vec3 dworld = (path.getVert(pathLength).m_pos - path.getVert(pathLength + 1).m_pos).normalized();
		Vec3 dLocal = toLocal(dworld); 
		dLocal /= dLocal[2];//the z unit is 1
		fac = 1 / std::pow(dLocal[0] * dLocal[0] + dLocal[1] * dLocal[1] + 1, 1.5);
		// fac = 1.0;
		DFloat x = DFloat(dLocal[0], 2, 0);
		DFloat y = DFloat(dLocal[1], 2, 1);
		DVec4 P;
		bool valid = true;
		DVec4 D = pathAutoDiff(path, x, y, P, valid);
		
		out_dDdx = Vec3(D[0].derivatives()[0], D[1].derivatives()[0], D[2].derivatives()[0]);
		out_dDdy = Vec3(D[0].derivatives()[1], D[1].derivatives()[1], D[2].derivatives()[1]);

		out_dPdx = Vec3(P[0].derivatives()[0], P[1].derivatives()[0], P[2].derivatives()[0]);
		out_dPdy = Vec3(P[0].derivatives()[1], P[1].derivatives()[1], P[2].derivatives()[1]);
		return valid;
	}



	std::pair<Vec3, Vec3> splat(const Path *path)
	{
		Vec3 surfacePos = path->getLastVertPos();
		Vec3 camToSurfaceDir = (surfacePos - m_pos).normalized();

		//from world space to local Space
		Vec3 localDir = toLocal(camToSurfaceDir);
		Vec2 pixel = dirToPixel(localDir);

		Vec3 value;

		if ((pixel[0] < m_resolution[0] && pixel[0] >= 0 && pixel[1] < m_resolution[1] && pixel[1] >= 0) || global_log_flag == false)
		{
			//compute direction ray differential
			Vec3 dDdx, dDdy, dPdx, dPdy;
			double fac;
			bool valid = computeDifferentialAutoDiff(*path, dDdx, dDdy, dPdx, dPdy, fac);
			if (!valid)
				return {};

			double temp;
			temp = dPdx.norm() * dPdy.norm() * std::max(0.01, dPdx.normalized().cross(dPdy.normalized()).norm());

			double curvatureFac = 1.0 / temp;
			value = path->computeContribution() * curvatureFac * fac;

			path->ans = value;

			if(!isfinite(value.x()))
			{
				value = Vec3(0);
			}

			if (global_log_flag == true)
				value *= m_resolution[0] * m_resolution[1] / m_A;
		}

		return std::make_pair(camToSurfaceDir, value);
	}

	std::vector<std::pair<Vec3, Vec3>> splat(const std::vector<Path*> &pathList, const Scene *scene)
	{
		const int threadCount = omp_get_max_threads();
		if (global_log_flag) cout << "The thread cout for splatting is " << threadCount << endl;

		Vec3 sum(0,0,0);
		std::vector<std::pair<Vec3, Vec3>> ans;

		for (int i = 0; i <pathList.size(); i++)
		{
			int tid = omp_get_thread_num();
			const Path *path = pathList[i];	
			auto [dir, tmp] = splat(path);	
			if (isfinite(tmp.x()))
			{
			sum += tmp;
			ans.push_back({dir, tmp});
			}
		}

		return ans;
	}

	void writeOutput(string output)
	{
	}

	void writeOutput2(string output)
	{
	}

	Vec3 getPos() const { return m_pos; }
	Vec3 getDir() const { return m_dir; }
	void setPos(const Vec3 &_pos) { m_pos = _pos; }
	void setPostProcessOption(EPostOption __postProcessOption) {postProcessOption = __postProcessOption; }

	Vec2i getResolution(){
		return m_resolution;
	}

	Vec2i getResolution() const{
		return m_resolution;
	}

	bool pathOcclusion(const Path *path, const Scene *scene)
	{
		for (int i = 0; i < path->getLength() - 1; i++)
		{
			const PathVert &current = path->getVert(i);
			const PathVert &next = path->getVert(i + 1);

			Vec3 pos = current.m_pos;
			Vec3 pos2 = next.m_pos;
			Vec3 dir = (pos2 - pos).normalized();
			RayDifferential ray;
			ray.time = 0;
			ray.hasDifferentials = false;

			ray = Ray(Point3(pos[0], pos[1], pos[2]), Vector3(dir[0], dir[1], dir[2]), 0);
			ray.maxt = (pos2 - pos).norm() - Epsilon;
			ray.mint = Epsilon;
			Intersection its;
			if (scene->rayIntersect(ray, its))
			{
				return true;
			}

		}
		return false;

	}

private:	
	mitsuba::Transform m_cameraToSample; 
	mitsuba::Transform m_sampleToCamera;

	Vec2i m_resolution;
	Vec3 m_pos;
	Vec3 m_dir;
	Vec3 m_s, m_t;
	double m_fov;
	double m_scale;
	double m_imageAspectRatio;
	double m_A; //the area of the image plane
	EPostOption postProcessOption;

	/// Convert from world coordinates to local coordinates
	inline Vec3 toLocal(const Vec3 &v) const {
		return Vec3(
			v.dot(m_s),
			v.dot(m_t),
			v.dot(m_dir));
	}

	inline Vec3 toWorld(const Vec3 &v) const {
		return m_s * v[0] + m_t * v[1] + m_dir * v[2];
	}

	inline Vec2 dirToPixel(const Vec3 &v) const {
		Point localV(v[0], v[1], v[2]);

		Point screenSample = m_cameraToSample(localV);
		if (screenSample.x < 0 || screenSample.x > 1 ||
			screenSample.y < 0 || screenSample.y > 1)
			return Vec2(0, 0);

		return Vec2(
			screenSample.x * m_resolution[0],
			screenSample.y * m_resolution[1]);
	}


	inline Vec3 pixelToDir(const Vec2 &pixelSample) const {
		Point nearP = m_sampleToCamera(Point(
			pixelSample[0] / m_resolution[0],
			pixelSample[1] / m_resolution[1], 0.0f));
		Vec3 d = Vec3(nearP[0], nearP[1], nearP[2]).normalized();
		return d;
	}


};


#endif /* __CAMERA_H_ */