// glintbounce.cpp : Defines the entry point for the console application.
//
// clang-format off

int g_bounce = 0;

#include "localEigen/Dense"
#include "iostream"
#include <cstdio>
bool global_log_flag = true;
#include "camera.h"
#include "util.h"
#include <time.h>
#include "shape.h"
#include <mitsuba/render/util.h>
#include "unsupported/Eigen/AutoDiff"
#include "intervalpath.h"
#include <queue>
#include <stack>
#include <mitsuba/render/scene.h>
#include <mitsuba/core/ray.h>
#include <mutex>
#include <atomic>
#include <iomanip>
#include <chrono>
//#include "encoder.h"
//#include "decoder.h"
#include "resultant.h"
#include "resultant-simd.h"
#include <mitsuba/render/sampler.h>
#include "distr.h"

// GPU Version (comment the line below if not required)
// #define MBGLINTS_RESULTANT_ENABLE_CUDA

#ifdef MBGLINTS_RESULTANT_ENABLE_CUDA
#include "resultant_cuda.cuh"
#endif

using namespace std;
using namespace Eigen;
using namespace mitsuba;

typedef Eigen::Vector3d Vec3;
typedef Eigen::Vector2i Vec2i;
typedef Eigen::Vector3i Vec3i;

MTS_NAMESPACE_BEGIN

typedef Eigen::Vector3d Vec3;
typedef Eigen::Vector3i Vec3i;
typedef Eigen::Vector2d Vec2;
typedef Eigen::Vector4d Vec4;

typedef Eigen::Matrix<double, 6, 1> Vec6;

//-----------------------------------------------------
typedef AutoDiffScalar<Eigen::Vector2d> DFloat;

typedef Eigen::Matrix<DFloat, 4, 1> DVec4;
typedef Eigen::Matrix<DFloat, 3, 1> DVec3;
typedef Eigen::Matrix<DFloat, 2, 1> DVec2;

typedef Eigen::Matrix<DFloat, 2, 1, 0, 2, 1> DMatrix2;
typedef Eigen::Matrix<double, 3, 2> Matrix3x2;
typedef Eigen::Matrix<double, 2, 3> Matrix2x3;
typedef Eigen::Matrix<double, 2, 2> Matrix2;

// we only need four derivative to track for two bounce case
//-----------------------------------------------------
typedef AutoDiffScalar<Vec4> DFloat4;
typedef Eigen::Matrix<DFloat4, 4, 1> DVec44;
typedef Eigen::Matrix<DFloat4, 6, 1> DVec46;
typedef Eigen::Matrix<DFloat4, 3, 1> DVec43;
typedef Eigen::Matrix<DFloat4, 2, 1> DVec42;
typedef Eigen::Matrix<double, 6, 4> Matrix6x4;
typedef Eigen::Matrix<double, 4, 6> Matrix4x6;

// for three bounce
typedef Eigen::Matrix<double, 6, 1> Vec6;
typedef AutoDiffScalar<Vec6> DFloat6;
typedef Eigen::Matrix<DFloat6, 6, 1> DVec66;
typedef Eigen::Matrix<DFloat6, 9, 1> DVec69;
typedef Eigen::Matrix<DFloat6, 3, 1> DVec63;
typedef Eigen::Matrix<DFloat6, 2, 1> DVec62;

typedef Eigen::Matrix<double, 6, 9> Matrix6x9;
typedef Eigen::Matrix<double, 9, 6> Matrix9x6;

// for four bounce
typedef Eigen::Matrix<double, 8, 1> Vec8;
typedef AutoDiffScalar<Vec8> DFloat8;
typedef Eigen::Matrix<DFloat8, 8, 1> DVec88;
typedef Eigen::Matrix<DFloat8, 12, 1> DVec8_12;
typedef Eigen::Matrix<DFloat8, 3, 1> DVec83;

typedef Eigen::Matrix<double, 8, 12> Matrix8x12;
typedef Eigen::Matrix<double, 12, 8> Matrix12x8;

// for five bounce
typedef Eigen::Matrix<double, 10, 1> Vec10;
typedef AutoDiffScalar<Vec10> DFloat10;
typedef Eigen::Matrix<DFloat10, 10, 1> DVec10_10;
typedef Eigen::Matrix<DFloat10, 15, 1> DVec10_15;
typedef Eigen::Matrix<DFloat10, 3, 1> DVec10_3;

typedef Eigen::Matrix<double, 10, 15> Matrix10x15;
typedef Eigen::Matrix<double, 15, 10> Matrix15x10;

typedef Eigen::Matrix<double, 2, 2> Matrix2;

double total_solver_time = 0.0;
long long total_solver_count = 0;

std::vector<MyShape *> m_shapes;

#define M_PI 3.14159265358979323846f

enum ELightTransportMode
{
	ER = 0,
	ET,
	ERR,
	ETT,
	ETRT
};

class BounceGlintRenderer
{
public:
	// Constructor for height field
	BounceGlintRenderer(const Scene *scene, string hfPath, double scale)
	{
		brsolver_records.reserve(1e4);
		// m_errorThresh = 0.001;
		HFMesh = true;

		m_scene = scene;
		//set emitter
		setEmitter(scene);
		//for camera
		setCamera(scene);

		outputSomeInform();

	};


	// Constructor for mesh
	BounceGlintRenderer(Scene *scene, bool useNeural)
	{
		brsolver_records.reserve(1e4);
		m_errorThresh = 0.00001;

		HFMesh = false;
		// assume there is only one emitter
		m_scene = scene;
		m_sampler = scene->getSampler();

		// set emitter
		setEmitter(scene);
		// for camera
		setCamera(scene);

		outputSomeInform();
	}

	static void init_shapes(const Scene* scene, std::vector<MyShape *>& m_shapes) {
		ref_vector<Shape> shapes = scene->getShapes();
		for (int i = 0; i < shapes.size(); i++)
		{
			const Shape *shape = shapes[i];
			if (shape->getEmitter() != NULL)
				continue;

			const BSDF *bsdf = shape->getBSDF();
			MyBSDF *mybsdf1;
			MyBSDF *mybsdf2;
			if (bsdf->hasComponent(BSDF::EDeltaTransmission))
			{
				double eta = bsdf->getEta();
				Spectrum transmittanceColor = bsdf->getProperties().getSpectrum("specularTransmittance", Spectrum(1.0));
				mybsdf1 = new DilectricRefraction(eta, Vec3(transmittanceColor[0], transmittanceColor[1], transmittanceColor[2]));
				mybsdf2 = new DilectricReflection(eta);
				//if (global_log_flag) cout << "Trnsmittance Eta:" << eta << endl;
			}
			else if (bsdf->hasComponent(BSDF::EDeltaReflection))
			{
				mybsdf1 = new Reflection(bsdf->getProperties().getString("material", "Ag"));
				mybsdf2 = NULL;
				//if (global_log_flag) cout << "Use Reflectance" << endl;
			}
			else
			{
				//if (global_log_flag) cout << "The BSDF type of the shape is invalid for glint." << endl;
				continue;
			}

			const Sensor *cam = scene->getSensor();
			const Transform &transf = cam->getWorldTransform()->eval(0);
			Point p = transf(Point(0, 0, 0));

			Vec3 camPos(p.x, p.y, p.z);

			MyShape *myshape = new MyShape(shape, mybsdf1, mybsdf2, camPos, scene);
			m_shapes.push_back(myshape);
			//if (global_log_flag) cout << "Add one shape..." << endl;
		}
	}

	~BounceGlintRenderer()
	{
		delete m_camera;
		for (int i = 0; i < m_shapes.size(); i++)
		{
			delete m_shapes[i];
		}
		std::vector<MyShape *>().swap(m_shapes);
	}

	void outputSomeInform()
	{
		// {
		// 	Vec3 camPos = m_camera->getPos();
		// 	Vec3 camDir = m_camera->getDir();
		// 	cout << "scene CamPos:" << camPos[0] << "," << camPos[1] << "," << camPos[2] << endl;
		// 	cout << "scene LightPos:" << m_lightPos[0] << "," << m_lightPos[1] << "," << m_lightPos[2] << endl;
		// 	cout << "scene CamDir:" << camDir[0] << "," << camDir[1] << "," << camDir[2] << endl;
		// }

		// int totalTriangle = 0;
		// for (int k = 0; k < m_shapes.size(); k++)
		// {
		// 	const int triaCount1 = m_shapes[k]->m_triangles.size();
		// 	totalTriangle += triaCount1;
		// 	cout << "scene Shape " << k << ":  " << triaCount1 << endl;
		// }
		// cout << "scene Total Triangle Count:  " << totalTriangle << endl;
		// {
		// 	long long memoryCost = 0;
		// 	for (int k = 0; k < m_shapes.size(); k++)
		// 	{
		// 		memoryCost += m_shapes[k]->getHiearchySize();
		// 	}
		// 	double memCost = memoryCost / (1024.0f * 1024.0f);
		// 	//if (global_log_flag) cout << "scene The memory cost is: " << memCost << "MB" << endl;
		// }
	}

	void setEmitter(const Scene *scene, bool resample = false)
	{
		// assume there is only one emitter
		const Emitter *emitter = NULL;
		if (!resample)
			m_lightCount = 0;
		for (int i = 0; i < scene->getEmitters().size(); i++)
		{
			const Emitter *e = scene->getEmitters()[i].get();
			if (e->getType() == Emitter::EDeltaPosition && !resample)
			{
				emitter = e;
				const Transform &trafo = e->getWorldTransform()->eval(0);
				Point pos = trafo(Point(0.0f));
				m_lightPos = Vec3(pos.x, pos.y, pos.z);
				Spectrum intensity = e->getProperties().getSpectrum("intensity");
				m_lightIntensity = Vec3(intensity[0], intensity[1], intensity[2]);

				m_lightCount++;
				m_lightPosList.push_back(m_lightPos);
				m_lightIntensList.push_back(m_lightIntensity);
			}
			else if (e->getType() == Emitter::EOnSurface)
			{
				emitter = e;

				// for light intensity
				float surArea = e->getShape()->getSurfaceArea();
				PositionSamplingRecord pRec(0);
				Point2 samplePoint = m_sampler->next2D();

				e->getShape()->samplePosition(pRec, samplePoint);
				auto pdf = e->getShape()->pdfPosition(pRec);

				// decide coeff
				Spectrum intensity = e->getProperties().getSpectrum("radiance") / (pdf);
				m_lightIntensity = Vec3(intensity[0], intensity[1], intensity[2]);

				auto pos = pRec.p;

				m_lightPos = Vec3(pos.x, pos.y, pos.z);
				if (!resample)
				{
					m_lightCount++;
					m_lightPosList.push_back(m_lightPos);
					m_lightIntensList.push_back(m_lightIntensity);
				}
				else
				{
					m_lightPosList[i] = m_lightPos;
					m_lightIntensList[i] = m_lightIntensity;
				}
			}
			else if (e->isEnvironmentEmitter())
			{
				m_env = e;
			}
		}
		if (emitter == NULL && !resample)
		{
			cout << "error No valid emitter is specified!" << endl;
			m_lightPos = Vec3(-20.0f, 0.0f, 20.0f);
		}
	}

	void setOnePointEmitter(int i)
	{
		// assume there is only one emitter
		m_lightPos = m_lightPosList[i];
		m_lightIntensity = m_lightIntensList[i];
	}

	void setCamera(const Scene *scene)
	{
		const Sensor *cam = scene->getSensor();
		const Transform &transf = cam->getWorldTransform()->eval(0);
		Point p = transf(Point(0, 0, 0));
		Vector d = transf(Vector(0, 0, 1));

		Vec3 camPos(p.x, p.y, p.z);
		Vec3 camDir(d.x, d.y, d.z);

		const Matrix4x4 &m = transf.getMatrix();
		Vec3 up = Vec3(m(0, 1), m(1, 1), m(2, 1));
		Vec3 left = Vec3(m(0, 0), m(1, 0), m(2, 0));

		double fov = cam->getProperties().getFloat("fov", 10.0);
		Vector2i res = cam->getFilm()->getSize();

		const Vector2i &filmSize = cam->getFilm()->getSize();
		const Vector2i &cropSize = cam->getFilm()->getCropSize();
		const Point2i &cropOffset = cam->getFilm()->getCropOffset();

		Vector2 relSize((Float)cropSize.x / (Float)filmSize.x,
						(Float)cropSize.y / (Float)filmSize.y);
		Point2 relOffset((Float)cropOffset.x / (Float)filmSize.x,
						 (Float)cropOffset.y / (Float)filmSize.y);

		double aspect = cam->getAspect();
		Transform m_cameraToSample =
			Transform::scale(Vector(1.0f / relSize.x, 1.0f / relSize.y, 1.0f)) * Transform::translate(Vector(-relOffset.x, -relOffset.y, 0.0f)) * Transform::scale(Vector(-0.5f, -0.5f * aspect, 1.0f)) * Transform::translate(Vector(-1.0f, -1.0f / aspect, 0.0f)) * Transform::perspective(((PerspectiveCamera *)cam)->getXFov(), ((ProjectiveCamera *)cam)->getNearClip(), ((ProjectiveCamera *)cam)->getFarClip());

		m_camera = new Camera(camPos, camDir, up, left, Vec2i(res.x, res.y), fov, m_cameraToSample);
	}

	void setCameraFake(double px, double py, double pz, double nx, double ny, double nz)
	{
		// todo: accurate fake
		Vec3 camPos(px, py, pz);
		Vec3 camDir(nx, ny, nz);

		Vec3 up = Vec3(1, 0, 0);
		Vec3 left = Vec3(0, 0, 1);

		double fov = 120;
		Vector2i res(1);

		Vector2 relSize(1, 1);
		Point2 relOffset(0, 0);

		double aspect = 1;
		Transform m_cameraToSample = Transform::scale(Vector(-0.5f, -0.5f * aspect, 1.0f)) * Transform::translate(Vector(-1.0f, -1.0f / aspect, 0.0f)) * Transform::perspective(fov, 0.001, 1000);
		if (CHAIN_TYPE == 2 && g_bounce == 1)
		{
			if (m_camera)
			{
				m_camera->setPos(camPos);
			}
			else 
			{
				m_camera = new Camera(camPos, camDir, up, left, Vec2i(res.x, res.y), fov, m_cameraToSample);
			}
		}
		else m_camera->setMember(camPos, camDir, up, left, Vec2i(res.x, res.y), fov, m_cameraToSample);
	}

	void setPostProcessOption(EPostOption __p)
	{
		m_camera->setPostProcessOption(__p);
	}

	double evalError(const DVec46 &fx)
	{
		double error = 0.0f;
		for (int i = 0; i < 6; i++)
		{
			error += abs(fx[i].value());
		}
		return error;
	}
	double evalError(const Vec6 &fx)
	{
		double error = 0.0f;
		for (int i = 0; i < 6; i++)
		{
			error += abs(fx[i]);
		}
		return error;
	}
	double evalError(const Vec3 &fx)
	{
		double error = 0.0f;
		for (int i = 0; i < 3; i++)
		{
			error += abs(fx[i]);
		}
		return error;
	}
	double evalError(const DVec69 &fx)
	{
		double error = 0.0f;
		for (int i = 0; i < 9; i++)
		{
			error += abs(fx[i].value());
		}
		return error;
	}
	double evalError(const DVec3 &fx)
	{
		double error = 0.0f;
		for (int i = 0; i < 3; i++)
		{
			error += abs(fx[i].value());
		}
		return error;
	}
	double evalError(const DVec8_12 &fx)
	{
		double error = 0.0f;
		for (int i = 0; i < 12; i++)
		{
			error += abs(fx[i].value());
		}
		return error;
	}
	double evalError(const DVec10_15 &fx)
	{
		double error = 0.0f;
		for (int i = 0; i < 15; i++)
		{
			error += abs(fx[i].value());
		}
		return error;
	}

	// Functions for Newton solver
	// <DFloat6, DVec69, Vec6, Matrix9x6, Matrix6x9, DVec66>
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
	T2 f_Bounce(T1 alpha[], T1 beta[],
				Vec3 v[][3], Vec3 vn[][3],
				int shape[], int bounce, const Vec3 &viewPos, double eta[])
	{
		T1 gamma[MAX_Bounce];

		for (int i = 0; i < bounce; i++)
		{
			gamma[i] = 1 - alpha[i] - beta[i];
			// if (global_log_flag) std::cout << "fbouncelog alpha " << alpha[i] << " " << beta[i] << " " << gamma[i] << std::endl;
		}

		T7 p[MAX_Bounce], n[MAX_Bounce];
		// this should bounce + 2, the first for camera, and the last for light
		for (int k = 0; k < bounce; k++)
		{
			p[k + 1] = alpha[k] * v[k][0] + beta[k] * v[k][1] + gamma[k] * v[k][2];
			n[k + 1] = (alpha[k] * vn[k][0] + beta[k] * vn[k][1] + gamma[k] * vn[k][2]);
			// if (global_log_flag) std::cout << "fbouncelog pos  " << p[k+1][0] << " " << p[k+1][1] << " " << p[k+1][2] << std::endl;
			// if (global_log_flag) std::cout << "fbouncelog norm " << n[k+1][0] << " " << n[k+1][1] << " " << n[k+1][2] << std::endl;
		}
		p[0] = viewPos;
		p[bounce + 1] = m_lightPos;
		// if (global_log_flag) std::cout << "fbouncelog viewpos " << p[0][0] << " " << p[0][1] << " " << p[0][2] << std::endl;
		// if (global_log_flag) std::cout << "fbouncelog lightpos " << p[2][0] << " " << p[2][1] << " " << p[2][2] << std::endl;
		T2 result; // DVec69
		for (int k = 1; k < bounce + 1; k++)
		{
			// * custom constraint design

			T7 in = (p[k - 1] - p[k]).normalized();
			T7 out = (p[k + 1] - p[k]).normalized();
			// if (global_log_flag) std::cout << "fbouncelog in " << in[0] << " " << in[1] << " " << in[2] << std::endl;
			// if (global_log_flag) std::cout << "fbouncelog out " << out[0] << " " << out[1] << " " << out[2] << std::endl;

			// eta is on the inside direction
			double realEta = eta[k];
			if (in.dot(n[k]) < 0)
				realEta = 1.0 / realEta;

			T7 h = (in + realEta * out);
			// if (global_log_flag) std::cout << "fbouncelog hv " << h[0] << " " << h[1] << " " << h[2] << std::endl;

			// * compute delta of directions

			// T7 d = n[k] + m_shapes[shape[k - 1]]->m_bsdf1->getSign() * h;
			// if (global_log_flag) std::cout << "fbouncelog delta " << d[0] << " " << d[1] << " " <<d[2] << std::endl;

			T7 d = h.cross(n[k]);
			result[3 * (k - 1)] = d[0];
			result[3 * (k - 1) + 1] = d[1];
			result[3 * (k - 1) + 2] = d[2];
		}

		return result;
	}

	// <DFloat6, DVec69, Vec6, Matrix9x6, Matrix6x9, DVec66, DVec63>
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
	int SolverRouter(int iShape[], int iTriangle[],
					 std::vector<Path *> &path, int &pathcutAfterNewton, const int bounce, int mode[], double eta[], double pdf = 1.f)
	{
		if (use_resultant) {
			return ResultantSolver_in<T1,T2,T3,T4,T5,T6,T7>(iShape, iTriangle, path, pathcutAfterNewton, bounce, mode, eta, pdf);
		}
		else {
			return NewtonSolver<T1, T2, T3, T4, T5, T6, T7>(iShape, iTriangle, path, pathcutAfterNewton, bounce, mode, eta, pdf);
		}
	}

	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
	int NewtonSolver(int iShape[], int iTriangle[],
					 std::vector<Path *> &path, int &pathcutAfterNewton, const int bounce, int mode[], double eta[], double pdf = 1.f)
	{
		std::vector<std::pair<double, double>> solutions, fake;
		std::vector<std::tuple<double, double, double, double>> seeds;
		if ((methodMask > 0) && use_resultant == false)
		{
			int k = methodMask;
			if (methodMask >= 9999) k = 1;
			// for (int i = 0; i < k; i++)
			// {
			// 	for (int j = 0; i + j < k; j++)
			// 	{
			// 		seeds.push_back({(i + 0.5) / k, (j + 0.5) / k});
			// 	}
			// }
			while (k--) {
				double r1 = rand() * 1.0 / 32767;
				double r2 = rand() * 1.0 / 32767;
				double r3 = rand() * 1.0 / 32767;
				double r4 = rand() * 1.0 / 32767;
				if (r1 + r2 > 1) {
					r1 = 1 - r1;
					r2 = 1 - r2;
				}
				if (r3 + r4 > 1) {
					r3 = 1 - r3;
					r4 = 1 - r4;
				}
				seeds.push_back({r1, r2, r3, r4});
			}
		}
		else
		{
			seeds.push_back({1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0});
		}
		if (methodMask < 9999)
		{
			int ans = 0;
			for (auto [u, v, u2, v2] : seeds)
			{
				ans += NewtonSolverImpl<T1, T2, T3, T4, T5, T6, T7>(iShape, iTriangle, path, pathcutAfterNewton, bounce, mode, eta, u, v, u2, v2, solutions, pdf);
			}
			return ans;
		}
		else 
		{
			for (auto [u, v, u2, v2] : seeds)
			{
				int ans = NewtonSolverImpl<T1, T2, T3, T4, T5, T6, T7>(iShape, iTriangle, path, pathcutAfterNewton, bounce, mode, eta, u, v, u2, v2, solutions, 1e99);
			}
			if (solutions.size() == 0)
			{
				return 0;
			}
			double inv_pdf = 1.0;
			for (int i = 0; i < 256; i++)
			{
				double r1 = rand() * 1.0 / 32767;
				double r2 = rand() * 1.0 / 32767;
				double r3 = rand() * 1.0 / 32767;
				double r4 = rand() * 1.0 / 32767;
				if (r1 + r2 > 1) {
					r1 = 1 - r1;
					r2 = 1 - r2;
				}
				if (r3 + r4 > 1) {
					r3 = 1 - r3;
					r4 = 1 - r4;
				}
				fake.clear();
				int ans = NewtonSolverImpl<T1, T2, T3, T4, T5, T6, T7>(iShape, iTriangle, path, pathcutAfterNewton, bounce, mode, eta, r1, r2, r3, r4, fake, pdf / inv_pdf);
				if (fake.size() > 0)
				{
					auto [u1, v1] = fake[0];
					auto [u10, v10] = solutions[0];
					if (abs(u1 - u10) + abs(v1 - v10) < 1e-2)
					{
						break;
					}
				}
				inv_pdf += 1;
			}
		}
	}

	// <DFloat6, DVec69, Vec6, Matrix9x6, Matrix6x9, DVec66, DVec63>
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
	int NewtonSolverImpl(int iShape[], int iTriangle[],
					 std::vector<Path *> &path, int &pathcutAfterNewton, const int bounce, int mode[], double eta[], double u1, double v1, double u2, double v2, std::vector<std::pair<double, double>>& solutions, double pdf = 1)
	{
		T1 alpha1[MAX_Bounce]; // DFloat6
		T1 beta1[MAX_Bounce];
		Vec3i idx[MAX_Bounce];
		Vec3 v[MAX_Bounce][3];
		Vec3 vn[MAX_Bounce][3];
		//	Vec3 vc[MAX_Bounce];
		T1 alpha[MAX_Bounce], beta[MAX_Bounce];

		// u1 = 0.12956214136517497;
		// v1 = 0.3997698747221097;
		// u2 = 0.11405610171303947;
		// v2 = 0.24744782986248784;

		const double inv_three = 1.0f / 3.0f;
		const int dCount = 2 * bounce;
		for (int i = 0; i < bounce; i++)
		{
			MyShape *shape = m_shapes[iShape[i]];
			alpha1[i] = T1(i == 0 ? u1: u2, dCount, 2 * i);
			beta1[i] = T1(i == 0 ? v1: v2, dCount, 2 * i + 1);
			alpha[i] = T1(i == 0 ? u1: u2, dCount, 2 * i);
			beta[i] = T1(i == 0 ? v1: v2, dCount, 2 * i + 1);
			Vec3i iIndex = shape->m_triangles[iTriangle[i]].index;

			const vector<Vec3> &verts = shape->m_verts;
			const vector<Vec3> &vns = shape->m_vertexNormal;
			for (int k = 0; k < 3; k++)
			{
				v[i][k] = verts[iIndex[k]];
				vn[i][k] = vns[iIndex[k]];
			}
		}

		T2 fx; // DVec69
		const int itrCount = config_itrCount;
		bool found = false;
		const int largeD = 3 * bounce;
		const int shortD = 2 * bounce;

		// * NEWTON BEGIN

		auto time_begin = std::chrono::high_resolution_clock::now();
		
		for (int i = 0; i < itrCount; i++)
		{
			bool valid = true;
			for (int k = 0; k < bounce; k++)
			{
				alpha[k] = T1(alpha1[k].value(), dCount, (2 * k));
				beta[k] = T1(beta1[k].value(), dCount, (2 * k + 1));

				if (alpha[k].value() > 10 || alpha[k].value() < -10 || beta[k].value() > 10 || beta[k].value() < -10)
				{
					valid = false;
					break;
				}
			}
			if (!valid)
			{
				found = false;
				break;
			}
			fx = f_Bounce<T1, T2, T3, T4, T5, T6, T7>(alpha, beta, v, vn, iShape, bounce, m_camera->getPos(), eta);
			// fx = f_Bounce_constH<T1, T2, T3, T4, T5, T6, T7>(alpha, beta, vn, vc, shape, bounce, m_camera->getPos(), eta);

			double error = evalError(fx);
			if (error < m_errorThresh)
			{
				found = true;
				break;
			}

			// 9:3*bounce
			T3 jacobian[3 * MAX_Bounce]; // Vec6
			for (int k = 0; k < largeD; k++)
				jacobian[k] = fx[k].derivatives();

			T4 J; // Matrix9x6
			for (int j = 0; j < shortD; j++)
			{
				for (int k = 0; k < largeD; k++)
				{
					J(k, j) = jacobian[k][j];
				}
			}

			T5 Jt = J.transpose(); ////Matrix6x9
			T5 invJ = (Jt * J).inverse() * Jt;

			T6 alpha_beta = invJ * fx; // DVec66

			for (int k = 0; k < bounce; k++)
			{
				alpha1[k] = alpha[k] - alpha_beta[2 * k];
				beta1[k] = beta[k] - alpha_beta[2 * k + 1];
			}
		}

		auto time_end = std::chrono::high_resolution_clock::now();
		double us = std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_begin).count() * 1e-2;

		// if (global_log_flag) std::cout << "time usage " << us << std::endl;
		total_solver_time += us;
		total_solver_count += 1;

		// * NEWTON END

		if (!found)
			return 0;

		double a[MAX_Bounce], b[MAX_Bounce], c[MAX_Bounce];
		Vec3 p[MAX_Bounce], n[MAX_Bounce];
		for (int k = 0; k < bounce; k++)
		{
			a[k] = alpha[k].value();
			b[k] = beta[k].value();
			if (a[k] < 0 || b[k] < 0 || a[k] + b[k] > 1)
				return 0;
			c[k] = 1.0f - a[k] - b[k];
			p[k] = v[k][0] * a[k] + v[k][1] * b[k] + v[k][2] * c[k];
			n[k] = vn[k][0] * a[k] + vn[k][1] * b[k] + vn[k][2] * c[k];
		}
		vector<PathVert> vertList(bounce + 2);

		vertList[0] = (PathVert(m_lightPos, (p[bounce - 1] - m_lightPos).normalized(), m_lightIntensity)); // the light
		vertList[bounce + 1] = PathVert(m_camera->getPos(), Vec3(0, 0, 0), Vec3(1, 1, 1));				   // the camera

		for (int k = 1; k < bounce + 1; k++)
		{
			int i = bounce - k;
			MyShape *shape = m_shapes[iShape[i]];
			MyBSDF *bsdf = mode[i + 1] == -2 ? shape->m_bsdf2 : shape->m_bsdf1;

			vertList[k] = PathVert(p[i], n[i], v[i][0], v[i][1], v[i][2], vn[i][0], vn[i][1], vn[i][2],
								   iShape[i], iTriangle[i], shape->m_triangles[iTriangle[i]].normal, bsdf);
			vertList[k].alpha = a[i];
			vertList[k].beta = b[i];
		}

		double u1star = vertList[1].alpha;
		double v1star = vertList[1].beta;

		bool flag = true;
		for (auto [u1s, v1s] : solutions)
		{
			if (abs(u1star - u1s) + abs(v1star - v1s) < 1e-2)
			{
				flag = false;
			}
		}
		if (!flag)
		{
			return 0;
		}
		solutions.push_back({u1star, v1star});

		if (use_log || use_xlog)
		{
			vector<int> tid(bounce);
			// ! small end is cam
			for (int i = 0; i < bounce; i++)
			{
				tid[i] = iTriangle[i];
			}
			path.push_back(new Path(vertList, tid, pdf));
		}
		else
		{
			path.push_back(new Path(vertList, pdf));
		}

		pathcutAfterNewton++;


		// occlusion test
		if (!HFMesh || true)
		{
			bool block = m_camera->pathOcclusion(path[path.size() - 1], m_scene);
			if (block)
			{
				delete path.back();
				path.pop_back();
				return 0;
			}
		}

		return 1;
	}

	struct BatchResultantSolverRecord
	{
		int iShape[2];
		int iTriangle[2];

		std::vector<Path *> path;

		Vec3 p10;
		Vec3 p11;
		Vec3 p12;
		Vec3 n10;
		Vec3 n11;
		Vec3 n12;

		Vec3 p20;
		Vec3 p21;
		Vec3 p22;
		Vec3 n20;
		Vec3 n21;
		Vec3 n22;

		std::vector<std::tuple<double, double, double, double>> solutions;
		double pdf;
		string ss;
	};
	std::vector <BatchResultantSolverRecord> brsolver_records;

	// <DFloat6, DVec69, Vec6, Matrix9x6, Matrix6x9, DVec66, DVec63>
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
	int ResultantSolver_in(int iShape[], int iTriangle[],
					 std::vector<Path *> &path, int &pathcutAfterNewton, const int bounce, int mode[], double eta[], double pdf = 1.f)
	{
		T1 alpha1[MAX_Bounce]; // DFloat6
		T1 beta1[MAX_Bounce];
		Vec3i idx[MAX_Bounce];
		Vec3 v[MAX_Bounce][3];
		Vec3 vn[MAX_Bounce][3];
		//	Vec3 vc[MAX_Bounce];
		T1 alpha[MAX_Bounce], beta[MAX_Bounce];

		const double inv_three = 1.0f / 3.0f;
		const int dCount = 2 * bounce;
		for (int i = 0; i < bounce; i++)
		{
			MyShape *shape = m_shapes[iShape[i]];
			alpha1[i] = T1(inv_three, dCount, 2 * i);
			beta1[i] = T1(inv_three, dCount, 2 * i + 1);
			alpha[i] = T1(inv_three, dCount, 2 * i);
			beta[i] = T1(inv_three, dCount, 2 * i + 1);
			Vec3i iIndex = shape->m_triangles[iTriangle[i]].index;

			const vector<Vec3> &verts = shape->m_verts;
			const vector<Vec3> &vns = shape->m_vertexNormal;
			for (int k = 0; k < 3; k++)
			{
				v[i][k] = verts[iIndex[k]];
				vn[i][k] = vns[iIndex[k]];
			}
		}

		T2 fx; // DVec69
		bool found = false;
		const int largeD = 3 * bounce;
		const int shortD = 2 * bounce;

		std::vector<std::pair<double, double>> solutions;

		Vec3 p10 = v[0][0];
		Vec3 p11 = v[0][1] - v[0][0];
		Vec3 p12 = v[0][2] - v[0][0];
		Vec3 n10 = vn[0][0];
		Vec3 n11 = vn[0][1] - vn[0][0];
		Vec3 n12 = vn[0][2] - vn[0][0];
		if (bounce == 1) {
			brsolver_records.push_back({
				{iShape[0]},
				{iTriangle[0]},
				path,
				p10,p11,p12,n10,n11,n12,Vec3(0),Vec3(0),Vec3(0),Vec3(0),Vec3(0),Vec3(0), {}, pdf, ""});
		}
		else {
			Vec3 p20 = v[1][0];
			Vec3 p21 = v[1][1] - v[1][0];
			Vec3 p22 = v[1][2] - v[1][0];
			Vec3 n20 = vn[1][0];
			Vec3 n21 = vn[1][1] - vn[1][0];
			Vec3 n22 = vn[1][2] - vn[1][0];
			brsolver_records.push_back({
				{iShape[0], iShape[1]},
				{iTriangle[0], iTriangle[1]},
				path,
				p10,p11,p12,n10,n11,n12,
				p20,p21,p22,n20,n21,n22,{}, pdf, ""});
			
		}
		return 0;
	}

	// Generate paths using the solutions of resultants
	// <DFloat6, DVec69, Vec6, Matrix9x6, Matrix6x9, DVec66, DVec63>
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
	int ResultantSolver_out(std::vector<Path *> &path, int &pathcutAfterNewton, const int bounce, int mode[], double eta[])
	{
		int ans = 0;
		
		if (bounce == 1)
		for (const auto& brsolver_record : brsolver_records)
		{
			int is[3] = {brsolver_record.iShape[0], 0};
			int it[3] = {brsolver_record.iTriangle[0], 0};
			ans += ResultantSolver_out_one<T1, T2, T3, T4, T5, T6, T7>(is, it, path, pathcutAfterNewton, bounce, mode, eta, brsolver_record.pdf, brsolver_record.solutions, brsolver_record.ss);
		}
		else 
		for (const auto& brsolver_record : brsolver_records)
		{
			int is[3] = {brsolver_record.iShape[0], brsolver_record.iShape[1], 0};
			int it[3] = {brsolver_record.iTriangle[0], brsolver_record.iTriangle[1], 0};
			ans += ResultantSolver_out_one<T1, T2, T3, T4, T5, T6, T7>(is, it, path, pathcutAfterNewton, bounce, mode, eta, brsolver_record.pdf, brsolver_record.solutions, brsolver_record.ss);
		}

		brsolver_records.clear(); // very important for caustics
		
		return ans;
	}

	// <DFloat6, DVec69, Vec6, Matrix9x6, Matrix6x9, DVec66, DVec63>
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
	int ResultantSolver_out_one(int iShape[], int iTriangle[],
					 std::vector<Path *> &path, int &pathcutAfterNewton, const int bounce, int mode[], double eta[], double pdf, std::vector<std::tuple<double, double, double, double>> solutions, const string& tag = "")
	{
		T1 alpha1[MAX_Bounce]; // DFloat6
		T1 beta1[MAX_Bounce];
		Vec3i idx[MAX_Bounce];
		Vec3 v[MAX_Bounce][3];
		Vec3 vn[MAX_Bounce][3];
		//	Vec3 vc[MAX_Bounce];
		T1 alpha[MAX_Bounce], beta[MAX_Bounce];

		const double inv_three = 1.0f / 3.0f;
		const int dCount = 2 * bounce;
		for (int i = 0; i < bounce; i++)
		{
			MyShape *shape = m_shapes[iShape[i]];
			alpha1[i] = T1(inv_three, dCount, 2 * i);
			beta1[i] = T1(inv_three, dCount, 2 * i + 1);
			alpha[i] = T1(inv_three, dCount, 2 * i);
			beta[i] = T1(inv_three, dCount, 2 * i + 1);
			Vec3i iIndex = shape->m_triangles[iTriangle[i]].index;

			const vector<Vec3> &verts = shape->m_verts;
			const vector<Vec3> &vns = shape->m_vertexNormal;
			for (int k = 0; k < 3; k++)
			{
				v[i][k] = verts[iIndex[k]];
				vn[i][k] = vns[iIndex[k]];
			}
		}

		T2 fx; // DVec69
		const int itrCount = config_ouritrCount; // todo: parameter ours_itr_count (for local refinement, see below)
		bool found = solutions.size() > 0;
		const int largeD = 3 * bounce;
		const int shortD = 2 * bounce;

		Vec3 p10 = v[0][0];
		Vec3 p11 = v[0][1] - v[0][0];
		Vec3 p12 = v[0][2] - v[0][0];
		Vec3 n10 = vn[0][0];
		Vec3 n11 = vn[0][1] - vn[0][0];
		Vec3 n12 = vn[0][2] - vn[0][0];

		Vec3 p20 = v[1][0];
		Vec3 p21 = v[1][1] - v[1][0];
		Vec3 p22 = v[1][2] - v[1][0];
		Vec3 n20 = vn[1][0];
		Vec3 n21 = vn[1][1] - vn[1][0];
		Vec3 n22 = vn[1][2] - vn[1][0];

		if (!found)
			return 0;

		int ret_val = 0;
		std::vector<std::tuple<double, double, double, double>> solutions_refined;
		// solutions.push_back({inv_three, inv_three});
		for (auto [av, bv, cv, dv] : solutions)
		{
			T1 alpha1[MAX_Bounce]; // DFloat6
			T1 beta1[MAX_Bounce];
			
			alpha[0] = T1(av, dCount, 2 * 0);
			beta[0] = T1(bv, dCount, 2 * 0 + 1);
			alpha1[0] = T1(av, dCount, 2 * 0);
			beta1[0] = T1(bv, dCount, 2 * 0 + 1);

			alpha[1] = T1(cv, dCount, 2 * 1);
			beta[1] = T1(dv, dCount, 2 * 1 + 1);
			alpha1[1] = T1(cv, dCount, 2 * 1);
			beta1[1] = T1(dv, dCount, 2 * 1 + 1);

			{
				bool is_find = false;
				// ================== local refinement ===========================
				// In theory, our method is capable of generating precise solutions. 
				// However, occasional inaccuracies can arise due to numerical instability, 
				// resulting in solutions that fail to meet the constraint error threshold of path cuts. 
				// To address this issue, we implement a local refinement process utilizing Pathcut's Newton solver. 
				// This solver operates by using our initial solutions as seeds.
				// ===============================================================
				for (int i = 0; i < itrCount; i++)
				{
					bool valid = true;
					for (int k = 0; k < bounce; k++)
					{
						alpha[k] = T1(alpha1[k].value(), dCount, (2 * k));
						beta[k] = T1(beta1[k].value(), dCount, (2 * k + 1));

						if (alpha[k].value() > 10 || alpha[k].value() < -10 || beta[k].value() > 10 || beta[k].value() < -10)
						{
							valid = false;
							break;
						}
					}
					if (!valid)
					{
						break;
					}
					fx = f_Bounce<T1, T2, T3, T4, T5, T6, T7>(alpha, beta, v, vn, iShape, bounce, m_camera->getPos(), eta);
					// fx = f_Bounce_constH<T1, T2, T3, T4, T5, T6, T7>(alpha, beta, vn, vc, shape, bounce, m_camera->getPos(), eta);

					double error = evalError(fx);
					// if (global_log_flag) std::cout << "error " << error << std::endl;
					if (error < m_errorThresh)
					{
						if (alpha[0].value() > 0 && alpha[0].value() < 1 &&  beta[0].value() > 0 &&  beta[0].value()  < 1 &&  alpha[0].value() + beta[0].value() < 1)
							if (alpha[1].value() > 0 && alpha[1].value() < 1 && beta[1].value() > 0 && beta[1].value() < 1 && alpha[1].value() + beta[1].value() < 1)
							{
								double dd1 = abs(alpha[0].value() - av) + abs(beta[0].value() - bv);
								double dd2 = abs(alpha[1].value() - cv) + abs(beta[1].value() - dv);
								double dd = dd1 + dd2;
								static double tdd = 0;
								static double tcc = 0;
								tdd += dd;
								tcc += 1;
								// if (global_log_flag) std::cout << "found start=" << av << ", " << bv << ", " << cv << ", " << dv << "\n      end  =" << alpha[0].value() << ", " << beta[0].value() << ", " << alpha[1].value() << ", " << beta[1].value() 
								// 	<< "\n  " << "d1="<< dd1 << " d2=" << dd2 << " delta=" <<
								// 	dd << " avgdd=" << tdd / tcc << std::endl;
							}
						is_find = true;
						break;
					}

					if (i+1==itrCount)
						break;

					// 9:3*bounce
					T3 jacobian[3 * MAX_Bounce]; // Vec6
					for (int k = 0; k < largeD; k++)
						jacobian[k] = fx[k].derivatives();

					T4 J; // Matrix9x6
					for (int j = 0; j < shortD; j++)
					{
						for (int k = 0; k < largeD; k++)
						{
							J(k, j) = jacobian[k][j];
						}
					}

					T5 Jt = J.transpose(); ////Matrix6x9
					T5 invJ = (Jt * J).inverse() * Jt;

					T6 alpha_beta = invJ * fx; // DVec66

					for (int k = 0; k < bounce; k++)
					{
						alpha1[k] = alpha[k] - alpha_beta[2 * k];
						beta1[k] = beta[k] - alpha_beta[2 * k + 1];
					}
					bool flag=true;
					double ax=alpha[0].value(), bx=beta[0].value();
					for (auto [a,b,c,d]: solutions_refined) {
						if (abs(a-ax)+abs(b-bx) < 1e-2) {
							flag=false;
						}
					}
					if(!flag){
						break;
					}
				}
				if (!is_find) {
					continue;
				}
				double ax=alpha[0].value(), bx=beta[0].value();
				double cx=alpha[1].value(), dx=beta[1].value();
				if (ax<0 || ax>1 || bx<0 || bx>1 || ax+bx<0 || ax+bx>1) continue;
				bool flag=true;
				for (auto [a,b,c,d]: solutions_refined) {
					if (abs(a-ax)+abs(b-bx) < 1e-2) {
						flag=false;
					}
				}
				if (flag){
					solutions_refined.push_back({ax,bx,cx,dx});
				}
			}
		}

		
#ifdef RESULTANT_TRACE
		for (auto [us, vs, unused1, unused2]: solutions_refined) {
			bool flag=  false;
			for (auto [u,v, unused1, unused2]: solutions) {
				if (abs(u-us)+abs(v-vs) < 1e-2) {
					flag=true;
				}
			}
			if (flag==false) {
				std::cout << "fail to find solution  " << us << " " << vs << std::endl;
				std::cout << tag << std::endl;
			}
		}
#endif
		for (auto [av, bv, cv, dv] : solutions_refined)
		{
			alpha[0] = T1(av, dCount, 2 * 0);
			beta[0] = T1(bv, dCount, 2 * 0 + 1);
			alpha[1] = T1(cv, dCount, 2 * 1);
			beta[1] = T1(dv, dCount, 2 * 1 + 1);
			
			double a[MAX_Bounce], b[MAX_Bounce], c[MAX_Bounce];
			Vec3 p[MAX_Bounce], n[MAX_Bounce];
			bool flag=true;
			for (int k = 0; k < bounce; k++)
			{
				a[k] = alpha[k].value();
				b[k] = beta[k].value();
				if (a[k] < 0 || b[k] < 0 || a[k] + b[k] > 1)
					{
						flag=false;
						break;
					}
				c[k] = 1.0f - a[k] - b[k];
				p[k] = v[k][0] * a[k] + v[k][1] * b[k] + v[k][2] * c[k];
				n[k] = vn[k][0] * a[k] + vn[k][1] * b[k] + vn[k][2] * c[k];
			}
			if (!flag) {
				continue;
			}
			vector<PathVert> vertList(bounce + 2);

			vertList[0] = (PathVert(m_lightPos, (p[bounce - 1] - m_lightPos).normalized(), m_lightIntensity)); // the light
			vertList[bounce + 1] = PathVert(m_camera->getPos(), Vec3(0, 0, 0), Vec3(1, 1, 1));				   // the camera

			for (int k = 1; k < bounce + 1; k++)
			{
				int i = bounce - k;
				MyShape *shape = m_shapes[iShape[i]];
				MyBSDF *bsdf = mode[i + 1] == -2 ? shape->m_bsdf2 : shape->m_bsdf1;

				vertList[k] = PathVert(p[i], n[i], v[i][0], v[i][1], v[i][2], vn[i][0], vn[i][1], vn[i][2],
										iShape[i], iTriangle[i], shape->m_triangles[iTriangle[i]].normal, bsdf);
				vertList[k].alpha = a[i];
				vertList[k].beta = b[i];
			}

			if (use_log || use_xlog)
			{
				vector<int> tid(bounce);
				// ! small end is cam
				for (int i = 0; i < bounce; i++)
				{
					tid[i] = iTriangle[i];
				}
				path.push_back(new Path(vertList, tid, pdf));
			}
			else
			{
				path.push_back(new Path(vertList, pdf));
			}

			pathcutAfterNewton++;
			ret_val += 1;

			// occlusion test
			if (!HFMesh || true)
			{
				bool block = m_camera->pathOcclusion(path[path.size() - 1], m_scene);
				if (block)
				{
					delete path.back();
					path.pop_back();
					ret_val -= 1;
					continue;
				}
			}

			// break; // ! at most one sol per tri
		}
		if (ret_val > 1)
		{
			// if (global_log_flag) std::cout << "report multi solution " << ret_val << std::endl;
		}
		return ret_val;
	}

	// compute and solve resultants
	void ResultantSolver_proc(int chain_type)
	{
		int use_cuda = methodMask & METHODMASK_CUDA;

#ifdef MBGLINTS_RESULTANT_ENABLE_CUDA
		if (use_cuda)
		{
			std::vector<double3> p10s, n10s, p11s, n11s, p12s, n12s;
			for (auto &brsolver_record : brsolver_records)
			{
				p10s.push_back(make_double3(brsolver_record.p10.x(), brsolver_record.p10.y(), brsolver_record.p10.z()));
				n10s.push_back(make_double3(brsolver_record.n10.x(), brsolver_record.n10.y(), brsolver_record.n10.z()));
				p11s.push_back(make_double3(brsolver_record.p11.x(), brsolver_record.p11.y(), brsolver_record.p11.z()));
				n11s.push_back(make_double3(brsolver_record.n11.x(), brsolver_record.n11.y(), brsolver_record.n11.z()));
				p12s.push_back(make_double3(brsolver_record.p12.x(), brsolver_record.p12.y(), brsolver_record.p12.z()));
				n12s.push_back(make_double3(brsolver_record.n12.x(), brsolver_record.n12.y(), brsolver_record.n12.z()));
			}
			int N = brsolver_records.size();
			double3 pD = make_double3(m_camera->getPos().x(), m_camera->getPos().y(), m_camera->getPos().z());
			double3 pL = make_double3(m_lightPos.x(), m_lightPos.y(), m_lightPos.z());
			double *polys, *Cxzs;
			polys = new double[N * 26];
			Cxzs = new double[N * 36];

			auto time_begin = std::chrono::high_resolution_clock::now();
			solve_cuda(pD, pL, p10s, n10s, p11s, n11s, p12s, n12s, polys, Cxzs);
			auto time_end = std::chrono::high_resolution_clock::now();
			double sss = std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_begin).count() * 1e-9;

			if (global_log_flag)
				std::cout << "profiling solve_cuda time usage " << sss << std::endl;

			Resultant::UnivariatePolynomial poly(26);
			Resultant::BivariatePolynomial Czx(6);

			Resultant::global_poly_cutoff_eps = config_cutoff_eps_resultant;

			time_begin = std::chrono::high_resolution_clock::now();
			for (int i = 0; i < N; i++)
			{
				for (int j = 0; j < 26; j++)
					poly.coeffs[j] = polys[i * 26 + j];

				for (int j = 0; j < 6; j++)
					for (int k = 0; k < 6; k++)
						Czx.coeffs[j][k] = Cxzs[i * 36 + j * 6 + k];
				
				std::vector<pair<Resultant::UnivariatePolynomial, pair<double, double>>> poly_with_endpoints;
				poly_with_endpoints.push_back(make_pair(poly, make_pair(0.0, 1.0)));
				brsolver_records[i].solutions = Resultant::solve_equ(poly_with_endpoints, Czx, 
					Resultant::BivariatePolynomial(1,0), Resultant::BivariatePolynomial(1,0), Resultant::BivariatePolynomial(1,0), chain_type);
			}
			time_end = std::chrono::high_resolution_clock::now();
			double sss2 = std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_begin).count() * 1e-9;
			if (global_log_flag)
				std::cout << "profiling solve_equ(tot) time usage " << sss2 << std::endl;

			delete[] polys;
			delete[] Cxzs;
			total_solver_count += 1;
		}
#else
		use_cuda = 0;
#endif
		// #pragma omp parallel for schedule(dynamic)
		// for (auto& brsolver_record : brsolver_records)

    // clang-format on
    if (use_cuda == 0) {
#ifdef RESULTANT_SIMD
      // Only one bounce support

      // TODO handle remains
      int i;
      int origin_size = brsolver_records.size();
      int simd_size = brsolver_records.size() / 4 * 4 + 4;

      brsolver_records.resize(simd_size);

      for (i = 0; i < simd_size; i += 4) {
        if (i % 10000 == 0) {
          if (global_log_flag)
            std::cout << "simd::solve proc " << i << std::endl;
        }
        auto time_begin = std::chrono::high_resolution_clock::now();

        auto &brsolver_record_1 = brsolver_records[i];
        auto &brsolver_record_2 = brsolver_records[i + 1];
        auto &brsolver_record_3 = brsolver_records[i + 2];
        auto &brsolver_record_4 = brsolver_records[i + 3];

        std::vector<double> pX = {
            brsolver_record_1.p10.x(), brsolver_record_2.p10.x(),
            brsolver_record_3.p10.x(), brsolver_record_4.p10.x(),

            brsolver_record_1.p12.x(), brsolver_record_2.p12.x(),
            brsolver_record_3.p12.x(), brsolver_record_4.p12.x(),

            brsolver_record_1.p11.x(), brsolver_record_2.p11.x(),
            brsolver_record_3.p11.x(), brsolver_record_4.p11.x(),
        };

        std::vector<double> nX = {
            brsolver_record_1.n10.x(), brsolver_record_2.n10.x(),
            brsolver_record_3.n10.x(), brsolver_record_4.n10.x(),

            brsolver_record_1.n12.x(), brsolver_record_2.n12.x(),
            brsolver_record_3.n12.x(), brsolver_record_4.n12.x(),

            brsolver_record_1.n11.x(), brsolver_record_2.n11.x(),
            brsolver_record_3.n11.x(), brsolver_record_4.n11.x(),
        };

        std::vector<double> pY = {
            brsolver_record_1.p10.y(), brsolver_record_2.p10.y(),
            brsolver_record_3.p10.y(), brsolver_record_4.p10.y(),

            brsolver_record_1.p12.y(), brsolver_record_2.p12.y(),
            brsolver_record_3.p12.y(), brsolver_record_4.p12.y(),

            brsolver_record_1.p11.y(), brsolver_record_2.p11.y(),
            brsolver_record_3.p11.y(), brsolver_record_4.p11.y(),
        };

        std::vector<double> nY = {
            brsolver_record_1.n10.y(), brsolver_record_2.n10.y(),
            brsolver_record_3.n10.y(), brsolver_record_4.n10.y(),

            brsolver_record_1.n12.y(), brsolver_record_2.n12.y(),
            brsolver_record_3.n12.y(), brsolver_record_4.n12.y(),

            brsolver_record_1.n11.y(), brsolver_record_2.n11.y(),
            brsolver_record_3.n11.y(), brsolver_record_4.n11.y(),
        };

        std::vector<double> pZ = {
            brsolver_record_1.p10.z(), brsolver_record_2.p10.z(),
            brsolver_record_3.p10.z(), brsolver_record_4.p10.z(),

            brsolver_record_1.p12.z(), brsolver_record_2.p12.z(),
            brsolver_record_3.p12.z(), brsolver_record_4.p12.z(),

            brsolver_record_1.p11.z(), brsolver_record_2.p11.z(),
            brsolver_record_3.p11.z(), brsolver_record_4.p11.z(),
        };

        std::vector<double> nZ = {
            brsolver_record_1.n10.z(), brsolver_record_2.n10.z(),
            brsolver_record_3.n10.z(), brsolver_record_4.n10.z(),

            brsolver_record_1.n12.z(), brsolver_record_2.n12.z(),
            brsolver_record_3.n12.z(), brsolver_record_4.n12.z(),

            brsolver_record_1.n11.z(), brsolver_record_2.n11.z(),
            brsolver_record_3.n11.z(), brsolver_record_4.n11.z(),
        };

        // TODO return
        auto solutions = resultant_simd::solve(
            chain_type,
            {m_camera->getPos().x(), m_camera->getPos().y(),
             m_camera->getPos().z()},
            {m_lightPos.x(), m_lightPos.y(), m_lightPos.z()}, pX, pY, pZ, nX,
            nY, nZ, use_resultant_fft, config_cutoff_matrix,
            config_cutoff_resultant, config_cutoff_eps_resultant, methodMask);

        brsolver_record_1.solutions = solutions[0];
        brsolver_record_2.solutions = solutions[1];
        brsolver_record_3.solutions = solutions[2];
        brsolver_record_4.solutions = solutions[3];

        auto time_end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        time_end - time_begin)
                        .count() *
                    1e-2;

     
        // if (global_log_flag) std::cout << "time usage " << us << std::endl;
        total_solver_time += us;
        total_solver_count += 4;
      }

      brsolver_records.resize(origin_size);

#else
      for (int i = 0; i < brsolver_records.size(); i++) {
        if (i % 100 == 0) {
          if (global_log_flag)
            std::cout << "proc " << i << std::endl;
        }
        auto &brsolver_record = brsolver_records[i];

        auto time_begin = std::chrono::high_resolution_clock::now();

        Vec3 p10 = brsolver_record.p10;
        Vec3 p11 = brsolver_record.p11;
        Vec3 p12 = brsolver_record.p12;
        Vec3 n10 = brsolver_record.n10;
        Vec3 n11 = brsolver_record.n11;
        Vec3 n12 = brsolver_record.n12;

        Vec3 p20 = brsolver_record.p20;
        Vec3 p21 = brsolver_record.p21;
        Vec3 p22 = brsolver_record.p22;
        Vec3 n20 = brsolver_record.n20;
        Vec3 n21 = brsolver_record.n21;
        Vec3 n22 = brsolver_record.n22;

        brsolver_record.solutions = Resultant::solve(
            chain_type,
            {m_camera->getPos().x(), m_camera->getPos().y(),
             m_camera->getPos().z()},
            {m_lightPos.x(), m_lightPos.y(), m_lightPos.z()},
            {p10.x(), p10.y(), p10.z()}, {n10.x(), n10.y(), n10.z()},
            {p11.x(), p11.y(), p11.z()}, {n11.x(), n11.y(), n11.z()},
            {p12.x(), p12.y(), p12.z()}, {n12.x(), n12.y(), n12.z()},
            {p20.x(), p20.y(), p20.z()}, {n20.x(), n20.y(), n20.z()},
            {p21.x(), p21.y(), p21.z()}, {n21.x(), n21.y(), n21.z()},
            {p22.x(), p22.y(), p22.z()}, {n22.x(), n22.y(), n22.z()},
            use_resultant_fft, config_cutoff_matrix, config_cutoff_resultant,
            config_cutoff_eps_resultant, methodMask, maxIterationDichotomy);

#ifdef RESULTANT_TRACE
        brsolver_record.ss = Resultant::rss.str();
#endif
        auto time_end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        time_end - time_begin)
                        .count() *
                    1e-2;

        // if (global_log_flag) std::cout << "time usage " << us << std::endl;
        total_solver_time += us;
        total_solver_count += 1;
      }
      if (global_log_flag)
        Resultant::printStats(false);
#endif
    }
    // clang-format off
	}

	void processPathCut(const IntervalPath &current, int *iShape, int *sign, double *eta,
						std::vector<Path *> &pathListThread,
						int &pathcutVisitedN, int &pathcutVisitedLeafN, int &pathcutAfterNewton, int &connectionCout, double pdf0 = 1.f)
	{
		pathcutVisitedN++;
		if (pathcutVisitedN % 100000 == 0)
		{
			// if (global_log_flag) std::cout << "visited " << pathcutVisitedN << std::endl;
		}

		if (use_xlog)
		{
			mu.lock();
			ofs << "visited ";
			// ! small end is cam
			for (int i = 0; i < current.getLength() - 2; i++)
			{
				ofs << current.getPatch(current.getLength() - 2 - i)->nodeIndex << " ";
			}
			ofs << std::endl;
			mu.unlock();
		}

		int iPatch = current.findThickestArea();
		if (iPatch == -1)
		{
			// is leaf

			int bounce = current.getLength() - 2;

			int triangleIndex[MAX_Bounce];
			for (int k = 0; k < bounce; k++)
			{
				triangleIndex[k] = current.getPatch(bounce - k)->triangleIndex;
			}

			pathcutVisitedLeafN++;

			for (int k = 0; k < bounce - 1; k++)
			{
				if (iShape[k] == iShape[k + 1] && triangleIndex[k] == triangleIndex[k + 1])
					return;
			}

			if (use_log)
			{ // xlog does not print leaf
				// ! small end is cam
				if (bounce == 2)
				{
					mu.lock();
					ofs << "leaf "
						<< triangleIndex[0] << " " << triangleIndex[1] << std::endl;
					mu.unlock();
				}
				if (bounce == 4)
				{
					mu.lock();
					ofs << "leaf "
						<< triangleIndex[0] << " "
						<< triangleIndex[1] << " "
						<< triangleIndex[2] << " "
						<< triangleIndex[3] << std::endl;
					mu.unlock();
				}
			}

			switch (bounce)
			{
			case 1:
				connectionCout += SolverRouter<DFloat, DVec3, Vec2, Matrix3x2, Matrix2x3, DVec2, DVec3>(
					iShape, triangleIndex, pathListThread, pathcutAfterNewton,
					bounce, sign, eta);

				break;
			case 2:
				connectionCout += SolverRouter<DFloat4, DVec46, Vec4, Matrix6x4, Matrix4x6, DVec44, DVec43>(
					iShape, triangleIndex, pathListThread, pathcutAfterNewton,
					bounce, sign, eta, pdf0);

				break;
			case 3:
				connectionCout += SolverRouter<DFloat6, DVec69, Vec6, Matrix9x6, Matrix6x9, DVec66, DVec63>(
					iShape, triangleIndex, pathListThread, pathcutAfterNewton,
					bounce, sign, eta); // TRT
				break;
			case 4:
				connectionCout += SolverRouter<DFloat8, DVec8_12, Vec8, Matrix12x8, Matrix8x12, DVec88, DVec83>(
					iShape, triangleIndex, pathListThread, pathcutAfterNewton,
					bounce, sign, eta, pdf0); // TTTT
				break;
			case 5:
				connectionCout += SolverRouter<DFloat10, DVec10_15, Vec10, Matrix15x10, Matrix10x15, DVec10_10, DVec10_3>(
					iShape, triangleIndex, pathListThread, pathcutAfterNewton,
					bounce, sign, eta);
				break;
			}
		}
		else
		{
			// is not leaf

			IntervalPath subPaths[4];
			current.subdivide(iPatch, subPaths);

			double pdf_this = 1;

			if (true)
			{
				for (const auto &subpath : subPaths)
				{
					// check can we discard this path?
					bool isValid = subpath.isValid(sign, eta);

					if (isValid)
					{
						processPathCut(subpath, iShape, sign, eta, pathListThread,
									   pathcutVisitedN, pathcutVisitedLeafN, pathcutAfterNewton, connectionCout, pdf0 * pdf_this);
					}
				}
			}
		}
	}

	void findGlintsPathCut(std::vector<Path *> &pathList, int bounce)
	{
		std::vector<std::vector<Vec3>> path;
		std::vector<std::vector<Path *>> pathListThread;

		const int threadCount = omp_get_max_threads();
		path.resize(threadCount);
		pathListThread.resize(threadCount);

		std::vector<int> connectionCount;
		std::vector<int> pathcutVisitedN;
		std::vector<int> pathcutVisitedLeafN;
		std::vector<int> pathcutAfterNewton;
		connectionCount.resize(threadCount);
		pathcutVisitedN.resize(threadCount);
		pathcutVisitedLeafN.resize(threadCount);
		pathcutAfterNewton.resize(threadCount);
		std::vector<int> countTriangles(threadCount);
		for (int i = 0; i < threadCount; i++)
		{
			connectionCount[i] = 0;
			pathcutVisitedN[i] = 0;
			pathcutAfterNewton[i] = 0;
			pathcutVisitedLeafN[i] = 0;
			countTriangles[i] = 0;
		}

		TreeNode *lightNode = new TreeNode;
		TreeNode *camNode = new TreeNode;
		lightNode->triangleCount = 1;
		camNode->triangleCount = 1;
		lightNode->posBox = Interval3D(m_lightPos);
		lightNode->center = m_lightPos;

		camNode->posBox = Interval3D(m_camera->getPos());
		camNode->center = m_camera->getPos();
		for (int k = 0; k < 4; k++)
		{
			camNode->child[k] = NULL;
			lightNode->child[k] = NULL;
		}

		long long bruteForceVisitedN = 0;
		// should start from light side, and include both light and camera
		int *sign = new int[bounce + 2];
		double *eta = new double[bounce + 2];
		int *iShape = new int[bounce];
		sign[0] = sign[bounce + 1] = 0;
		eta[0] = eta[bounce + 1] = 1.0;

		if (bounce == 1)
		{
			for (int is = 0; is < m_shapes.size(); is++)
			{
				bruteForceVisitedN += m_shapes[is]->m_triangles.size();

				std::vector<TreeNode *> camPatch;
				m_shapes[is]->m_root->getNodeLevel(0, 1, camPatch);
				int size = camPatch.size();
				iShape[0] = is;
				// For Beibei: can not simulate reflection for dielectric surface for now.
				sign[1] = m_shapes[is]->m_bsdf1->getSign();
				eta[1] = m_shapes[is]->m_bsdf1->getEta();

// #pragma omp parallel for schedule(dynamic)
				auto time_begin = std::chrono::high_resolution_clock::now();
				for (int i = 0; i < size; i++)
				{
					int tid = 0; // !
					TreeNode *patches[3];
					patches[0] = lightNode;
					patches[1] = camPatch[i];
					patches[2] = camNode;
					IntervalPath root(3, patches);
					processPathCut(root, iShape, sign, eta, pathListThread[tid],
								   pathcutVisitedN[tid], pathcutVisitedLeafN[tid], pathcutAfterNewton[tid], connectionCount[tid]);
				}
				auto time_end = std::chrono::high_resolution_clock::now();
				if (global_log_flag) std::cout << "profiling processPathCut      time usage " << std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_begin).count() * 1e-9 << std::endl;

				// for resultant, postproc
				if (use_resultant) {
					auto time_begin = std::chrono::high_resolution_clock::now();

					ResultantSolver_proc(eta[1] != 1 ? 2 : 1);

					auto time_end = std::chrono::high_resolution_clock::now();
					if (global_log_flag) std::cout << "profiling ResultantSolver_proc time usage " << std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_begin).count() * 1e-9 << std::endl;
					time_begin = time_end;

					connectionCount[0] += ResultantSolver_out<DFloat, DVec3, Vec2, Matrix3x2, Matrix2x3, DVec2, DVec3>(
						pathList,
						pathcutAfterNewton[0],
						bounce,
						sign,
						eta
					);

					time_end = std::chrono::high_resolution_clock::now();
					if (global_log_flag) std::cout << "profiling ResultantSolver_out time usage  " << std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_begin).count() * 1e-9 << std::endl;
				}
			}
		}
		else if (bounce == 2)
		{
			ELightTransportMode mode = ETT;

			// if (mode == ETT)
			// 	cout << "Simulate light transport TT. \n";
			// else
			// 	cout << "Simulate light transport RR. \n";

			for (int is = 0; is < m_shapes.size(); is++)
			{
				for (int js = 0; js < m_shapes.size(); js++)
				{
					// if (is != js)
					//	continue;

					if (m_shapes[is]->m_bsdf1->getType() == MyBSDF::ERefelction && m_shapes[js]->m_bsdf1->getType() == MyBSDF::ERefelction)
					{
						// cout << "Simulate light transport RR." << is << "," << js << " \n";
					}

					bool sample = false;
					if (methodMask < 0) 
					{
						sample = true;
					}

					bruteForceVisitedN += m_shapes[js]->m_triangles.size() * m_shapes[is]->m_triangles.size();
					// start from the ligth side
					std::vector<TreeNode *> lightPatch;
					m_shapes[js]->m_root->getNodeLevel(0, sample ? 3 : 1, lightPatch); // ! mod from 4
					std::vector<TreeNode *> camPatch;
					m_shapes[is]->m_root->getNodeLevel(0, sample ? 3 : 1, camPatch); // ! mod from 4
					int size = lightPatch.size() * camPatch.size();

					iShape[0] = is;
					iShape[1] = js;
					// For Beibei: can not simulate reflection for dielectric surface for now.
					sign[1] = m_shapes[is]->m_bsdf1->getSign();
					sign[2] = m_shapes[js]->m_bsdf1->getSign();
					eta[1] = m_shapes[is]->m_bsdf1->getEta();
					eta[2] = m_shapes[js]->m_bsdf1->getEta();

					// if (eta[1] != 1 && (is != 0 || js != 1)) {
					// 	continue; // only do by order
					// }

// #pragma omp parallel for schedule(dynamic)
					for (int i = 0; i < size; i++)
					{
						double p0 = 0.01;
						if (sample) 
						{
							if (rand() * 1.0 / RAND_MAX > p0)
							{
								continue;
							}
						}
						else 
						{
							p0 = 1;
						}
						int n1 = i / lightPatch.size();
						int n2 = i % lightPatch.size();
						int tid = 0;
						TreeNode *patches[4];
						patches[0] = lightNode;
						patches[1] = lightPatch[n2];
						patches[2] = camPatch[n1];
						patches[3] = camNode;
						IntervalPath root(4, patches);

						processPathCut(root, iShape, sign, eta, pathListThread[tid],
							pathcutVisitedN[tid], pathcutVisitedLeafN[tid], pathcutAfterNewton[tid], connectionCount[tid], p0);
					}

					
					// for resultant, postproc
					if (use_resultant) {
						auto time_begin = std::chrono::high_resolution_clock::now();

						ResultantSolver_proc(eta[1] != 1 ? 22 : 11);

						auto time_end = std::chrono::high_resolution_clock::now();
						if (global_log_flag) std::cout << "profiling ResultantSolver_proc time usage " << std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_begin).count() * 1e-9 << std::endl;
						time_begin = time_end;

						connectionCount[0] += ResultantSolver_out<DFloat4, DVec46, Vec4, Matrix6x4, Matrix4x6, DVec44, DVec43>(
							pathList,
							pathcutAfterNewton[0],
							bounce,
							sign,
							eta
						);

						time_end = std::chrono::high_resolution_clock::now();
						if (global_log_flag) std::cout << "profiling ResultantSolver_out time usage  " << std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_begin).count() * 1e-9 << std::endl;
					}
				}
			}
		}
		else if (bounce == 3)
		{

			cout << "Simulate light transport TRT or RRR. \n";
			// int is = 0; int js = 1; int ks = 0;
			for (int is = 0; is < m_shapes.size(); is++)
			{
				for (int js = 0; js < m_shapes.size(); js++)
				{
					for (int ks = 0; ks < m_shapes.size(); ks++)
					{

						if (m_shapes[is]->m_bsdf1->getType() == MyBSDF::ERefelction && m_shapes[ks]->m_bsdf1->getType() == MyBSDF::ERefelction && m_shapes[js]->m_bsdf1->getType() == MyBSDF::ERefelction)
						{
							cout << "Simulate light transport RRR." << is << "," << js << "," << ks << " \n";
							if (is == js || js == ks)
								continue;
						}
						// TRT
						else if (m_shapes[is]->m_bsdf1->getType() != MyBSDF::EDielectricRefracion || m_shapes[ks]->m_bsdf1->getType() != MyBSDF::EDielectricRefracion || (m_shapes[js]->m_bsdf1->getType() != MyBSDF::ERefelction && m_shapes[js]->m_bsdf2->getType() != MyBSDF::EDielectricReflection))
							return; // continue;
#if 0
						//
						else{

							if (is != js || js != ks)//for same object internal TRT, stained glass
								continue;
						}
#endif

						iShape[0] = is;
						iShape[1] = js;
						iShape[2] = ks;

						// the middle one is reflection
						sign[1] = m_shapes[is]->m_bsdf1->getSign();
						if (m_shapes[js]->m_bsdf1->getType() == MyBSDF::ERefelction)
							sign[2] = -1;
						else if (m_shapes[js]->m_bsdf2->getType() == MyBSDF::EDielectricReflection)
						{
							sign[2] = -2;
						}
						else
						{
							cout << "The BSDF type is not correct for the R in TRR. \n";
						}

						sign[3] = m_shapes[ks]->m_bsdf1->getSign();

						eta[1] = m_shapes[is]->m_bsdf1->getEta();
						eta[2] = 1.0;
						eta[3] = m_shapes[ks]->m_bsdf1->getEta();

						const int level[] = {2, 2, 6};
						std::vector<TreeNode *> patchList[MAX_Bounce];
						long long size = 1;
						int length[MAX_Bounce];

						for (int i = 0; i < bounce; i++)
						{
							int iS = iShape[bounce - i - 1];
							m_shapes[iS]->m_root->getNodeLevel(0, level[i], patchList[i]);
							size *= patchList[i].size();
							length[i] = patchList[i].size();
						}
						bruteForceVisitedN += size;

#pragma omp parallel for schedule(dynamic)
						for (int i = 0; i < size; i++)
						{
							int n[MAX_Bounce];
							n[0] = i / (length[1] * length[2]); // for light
							int n2n3 = i % (length[1] * length[2]);
							n[1] = n2n3 / (length[2]);
							n[2] = n2n3 % length[2];
							int tid = omp_get_thread_num();

							TreeNode *patches[5];
							patches[0] = lightNode;
							patches[bounce + 1] = camNode;

							for (int k = 0; k < bounce; k++)
							{
								patches[k + 1] = patchList[k][n[k]];
							}
							patches[0] = lightNode;

							IntervalPath root(5, patches);

							// TODO outer loop
							// int totalspp = (m_useNeural ? m_spp : 1);
							int totalspp = 1;
							for (int spp = 0; spp < totalspp; ++spp)
							{
								if (spp % 100 == 0)
								{
									if (global_log_flag) std::cout << "rendering " << spp << "/" << m_spp << std::endl;
								}
								processPathCut(root, iShape, sign, eta, pathListThread[tid],
											   pathcutVisitedN[tid], pathcutVisitedLeafN[tid], pathcutAfterNewton[tid], connectionCount[tid], totalspp);
							}
						}
					}
				}
			}
		}
		else if (bounce == 4)
		{

			cout << "Simulate light transport TTTT or RRRR. \n";
			// int is = 0; int js = 1; int ks = 2; int fs = 3;
			int shapeIndex[] = {0, 0, 0, 0};
			bool RRRR = true;
			for (int i = 0; i < bounce; i++)
			{
				iShape[i] = shapeIndex[i];
				if (m_shapes[shapeIndex[i]]->m_bsdf1->getType() != MyBSDF::ERefelction)
					RRRR = false;
			}
			if (RRRR)
			{
				cout << "Simulate light transport RRRR." << shapeIndex[0] << "," << shapeIndex[1]
					 << "," << shapeIndex[2] << "," << shapeIndex[3] << " \n";
			}

			for (int i = 0; i < bounce; i++)
			{
				sign[i + 1] = m_shapes[iShape[i]]->m_bsdf1->getSign();
				eta[i + 1] = m_shapes[iShape[i]]->m_bsdf1->getEta();
			}

			const int level[] = {0, 0, 0, 0};
			std::vector<TreeNode *> patchList[MAX_Bounce];
			long long size = 1;
			int length[MAX_Bounce];

			for (int i = 0; i < bounce; i++)
			{
				int iS = shapeIndex[bounce - i - 1];
				m_shapes[iS]->m_root->getNodeLevel(0, level[i], patchList[i]);
				size *= patchList[i].size();
				length[i] = patchList[i].size();
			}
			cout << "We have " << size << " intial path cut.\n";
			// int length = patchList[0].size();
			bruteForceVisitedN += size;

#pragma omp parallel for schedule(dynamic)
			for (long long i = 0; i < size; i++)
			{
				int n[MAX_Bounce];
				n[0] = i / (length[1] * length[2] * length[3]); // for light
				int n2n3n4 = i % (length[1] * length[2] * length[3]);
				n[1] = n2n3n4 / (length[2] * length[3]);	 // mp2
				int n3n4 = n2n3n4 % (length[2] * length[3]); //
				n[2] = n3n4 / length[3];					 // mp1
				n[3] = n3n4 % length[3];					 // camera

				int tid = omp_get_thread_num();

				TreeNode *patches[6];
				patches[0] = lightNode;
				patches[bounce + 1] = camNode;

				for (int k = 0; k < bounce; k++)
				{
					patches[k + 1] = patchList[k][n[k]];
				}
				IntervalPath root(6, patches);

				// TODO outer loop
				// int totalspp = (m_useNeural ? m_spp : 1);
				int totalspp = 1;
				for (int spp = 0; spp < totalspp; ++spp)
				{
					if (spp % 100 == 0)
					{
						if (global_log_flag) std::cout << "rendering " << spp << "/" << totalspp << std::endl;
					}
					processPathCut(root, iShape, sign, eta, pathListThread[tid],
								   pathcutVisitedN[tid], pathcutVisitedLeafN[tid], pathcutAfterNewton[tid], connectionCount[tid], totalspp);
				}
			}
		}

		else if (bounce == 5)
		{

			cout << "Simulate light transport TTRTT. \n";
			int shapeIndex[] = {0, 1, 2, 1, 0};

			for (int i = 0; i < bounce; i++)
			{
				iShape[i] = shapeIndex[i];
				sign[i + 1] = m_shapes[iShape[i]]->m_bsdf1->getSign();
				eta[i + 1] = m_shapes[iShape[i]]->m_bsdf1->getEta();
			}
			const int level = 3;
			std::vector<TreeNode *> patchList[MAX_Bounce];
			int size = 1;
			int length[MAX_Bounce];
			for (int i = 0; i < bounce; i++)
			{
				int is = shapeIndex[bounce - i - 1];
				m_shapes[is]->m_root->getNodeLevel(0, level, patchList[i]);
				size *= patchList[i].size();
				length[i] = patchList[i].size();
				;
			}

			bruteForceVisitedN += size;

#pragma omp parallel for schedule(dynamic)
			for (int i = 0; i < size; i++)
			{
				int n[5];
				n[0] = i / (length[1] * length[2] * length[3] * length[4]); // for light
				int n2n3n4n5 = i % (length[1] * length[2] * length[3] * length[4]);
				n[1] = n2n3n4n5 / (length[2] * length[3] * length[4]);		 // mp3
				int n3n4n5 = n2n3n4n5 % (length[2] * length[3] * length[4]); //
				n[2] = n3n4n5 / (length[3] * length[4]);					 // mp2
				int n4n5 = n3n4n5 % (length[3] * length[4]);
				n[3] = n4n5 / (length[4]); // mp1
				n[4] = n4n5 % (length[4]); // camera

				int tid = omp_get_thread_num();

				TreeNode *patches[7];
				patches[0] = lightNode;
				patches[bounce + 1] = camNode;

				for (int k = 0; k < bounce; k++)
				{
					patches[k + 1] = patchList[k][n[k]];
				}
				IntervalPath root(7, patches);

				// TODO add outer loop
				processPathCut(root, iShape, sign, eta, pathListThread[tid],
							   pathcutVisitedN[tid], pathcutVisitedLeafN[tid], pathcutAfterNewton[tid], connectionCount[tid]);
			}
		}

		clock_t t;
		t = clock();

		for (int j = 0; j < pathListThread[0].size(); j++)
			pathList.push_back(pathListThread[0][j]);

		for (int i = 1; i < threadCount; i++)
		{

			connectionCount[0] += connectionCount[i];
			pathcutAfterNewton[0] += pathcutAfterNewton[i];
			pathcutVisitedN[0] += pathcutVisitedN[i];
			pathcutVisitedLeafN[0] += pathcutVisitedLeafN[i];
			countTriangles[0] += countTriangles[i];
			for (int j = 0; j < pathListThread[i].size(); j++)
			{
				pathList.push_back(pathListThread[i][j]);
			}
		}
		t = clock() - t;

		if (global_log_flag) cout << "stats Brute Force Visited N:           " << bruteForceVisitedN << endl;
		if (global_log_flag) cout << "stats Visited Path cut N:              " << pathcutVisitedN[0] << endl;
		if (global_log_flag) cout << "stats Visited Leaf Path cut N:         " << pathcutVisitedLeafN[0] << endl;
		if (global_log_flag) cout << "stats After Newton solving path cut N: " << pathcutAfterNewton[0] << endl;
		if (global_log_flag) cout << "stats Found connections:               " << connectionCount[0] << endl;
		if (global_log_flag) cout << "stats Visible Triangle count:          " << countTriangles[0] << endl;

		delete[] eta;
		delete[] sign;
		delete camNode;
		delete lightNode;
	}
	
	std::vector<std::pair<Vec3, Vec3>> renderOneImage(string outImageName, int bounce, int spp, double errorThresh, SpecularDistribution* distr, Sampler* sampler, bool develop = true)
	{
		if (develop == false) {
			global_log_flag = false; 
		}
		// selfTest();

		m_errorThresh = errorThresh;
		m_spp = spp;

		//if (global_log_flag) cout << "Start rendering..." << endl;
		//if (global_log_flag) cout << "Have " << abs(bounce) << " bounce." << endl;
		//if (global_log_flag) cout << "specified spp = " << spp << endl;
		std::vector<Path *> pathList;

		// if (use_resultant_fft)
		// 	Resultant::unit_root_fft_prepare();

		int pathcutAfterNewton = 0;
		int mode[4] = {-1, -1, -1, 0};
		double eta[4] = {1, 1, 1, 1};

		mode[1] = m_shapes[0]->m_bsdf1->getSign();
		eta[1] = m_shapes[0]->m_bsdf1->getEta();
		if (bounce==2)
		{
			mode[0] = mode[1] = mode[2] = 1;
			eta[1] = eta[2] = 1.5;
		}

		for (int k = 0; k < m_lightCount; k++)
		{
			setOnePointEmitter(k);			
			if (distr != nullptr)
			{
				auto pos = m_camera->getPos();
				////////////////////
				// ! adhoc hack uv
				float u = pos.x() / 16 + 0.5;
				float v = pos.z() / 16 + 0.5;
				if (eta[1] == 1 || bounce == 2 || true) // comment for pool
				{
					u = pos.x() / 6 + 0.5;
					
					v = pos.z() / 6 + 0.5;
				}
				////////////////////
				auto samples = distr->sample(u, v, sampler);
				for (auto [ti, tj, pdf]: samples)
				{
					int iShape[4] = {0, 0, 0, 0};
					int iTriangle[4] = {tj, ti, 0, 0}; // ! order
					if (bounce == 1)
						SolverRouter<DFloat, DVec3, Vec2, Matrix3x2, Matrix2x3, DVec2, DVec3>(
							iShape, iTriangle, pathList, pathcutAfterNewton,
							bounce, mode, eta, pdf);
					else
						SolverRouter<DFloat4, DVec46, Vec4, Matrix6x4, Matrix4x6, DVec44, DVec43>(
							iShape, iTriangle, pathList, pathcutAfterNewton,
							bounce, mode, eta, pdf);
				}

				if (use_resultant)
				{
					ResultantSolver_proc(eta[1] != 1 ? 2 : 1);
					ResultantSolver_out<DFloat, DVec3, Vec2, Matrix3x2, Matrix2x3, DVec2, DVec3>(
						pathList,
						pathcutAfterNewton,
						bounce,
						mode,
						eta
					);
				}
			}
			else 
			{
				findGlintsPathCut(pathList, bounce);
			}
		}

		// Resultant::printStats(use_resultant_fft);
		// Resultant::unit_root_fft_destroy();

		// if (use_resultant_fft)
		// 	Resultant::unit_root_fft_destroy();

		if (global_log_flag)if (global_log_flag) cout << "stats solver total_solver_count = " << total_solver_count << endl;
		if (global_log_flag)if (global_log_flag) cout << "stats solver total_solver_time = " << total_solver_time * 1e-6 << " s" << endl;
		if (global_log_flag)if (global_log_flag) cout << "stats solver avg_solver_time = " << total_solver_time / total_solver_count << " us" << endl;

		if (global_log_flag)if (global_log_flag) cout << "start splatting..." << endl;

		clock_t t;
		t = clock();

		std::vector<std::pair<Vec3, Vec3>> val = m_camera->splat(pathList, m_scene);

		if (use_log || use_xlog)
		{
			for (int i = 0; i < pathList.size(); i++)
			{
				mu.lock();
				ofs << "final ";
				// ! small end is cam

				auto &path = pathList[i];
				for (int k = 1; k + 1 < path->vertices.size(); k++)
				{
					ofs << "uv" << 3 - k << ": " << path->vertices[k].alpha << " " << path->vertices[k].beta << '\n';
				}

				for (auto j : pathList[i]->triangle_indices)
				{
					ofs << j << " ";
				}
				int bn=0;
				for (auto j : pathList[i]->triangle_indices)
				{
					bn++;
					ofs << "bounce " << bn;
					ofs << "\n";
					for (int k = 0; k < 1; k++)
					{
						ofs << "v" << k << ": ";
						ofs << m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[k]].x() << ", ";
						ofs << m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[k]].y() << ", ";
						ofs << m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[k]].z() << ", ";
						ofs << "n" << k << ": ";
						ofs << m_shapes[0]->m_vertexNormal[m_shapes[0]->m_triangles[j].index[k]].x() << ", ";
						ofs << m_shapes[0]->m_vertexNormal[m_shapes[0]->m_triangles[j].index[k]].y() << ", ";
						ofs << m_shapes[0]->m_vertexNormal[m_shapes[0]->m_triangles[j].index[k]].z() << ", ";
						ofs << "\n";
					}
					// use delta
					for (int k = 1; k < 3; k++)
					{
						ofs << "v" << k << ": ";
						ofs << m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[k]].x() - m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[0]].x() << ", ";
						ofs << m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[k]].y() - m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[0]].y() << ", ";
						ofs << m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[k]].z() - m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[0]].z() << ", ";
						ofs << "n" << k << ": ";
						ofs << m_shapes[0]->m_vertexNormal[m_shapes[0]->m_triangles[j].index[k]].x() - m_shapes[0]->m_vertexNormal[m_shapes[0]->m_triangles[j].index[0]].x() << ", ";
						ofs << m_shapes[0]->m_vertexNormal[m_shapes[0]->m_triangles[j].index[k]].y() - m_shapes[0]->m_vertexNormal[m_shapes[0]->m_triangles[j].index[0]].y() << ", ";
						ofs << m_shapes[0]->m_vertexNormal[m_shapes[0]->m_triangles[j].index[k]].z() - m_shapes[0]->m_vertexNormal[m_shapes[0]->m_triangles[j].index[0]].z() << ", ";
						ofs << "\n";
					}
					for (int k = 0; k < 1; k++)
					{
						ofs << "std::vector<double> p" << bn << k << "_ = {";
						ofs << m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[k]].x() << ", ";
						ofs << m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[k]].y() << ", ";
						ofs << m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[k]].z() << "";
						ofs << "};";
						ofs << "std::vector<double> n" << bn << k << "_ = {";
						ofs << m_shapes[0]->m_vertexNormal[m_shapes[0]->m_triangles[j].index[k]].x() << ", ";
						ofs << m_shapes[0]->m_vertexNormal[m_shapes[0]->m_triangles[j].index[k]].y() << ", ";
						ofs << m_shapes[0]->m_vertexNormal[m_shapes[0]->m_triangles[j].index[k]].z() << "";
						ofs << "};";
						ofs << "\n";
					}
					// use delta
					for (int k = 1; k < 3; k++)
					{
						ofs << "std::vector<double> p" << bn << k << "_ = {";
						ofs << m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[k]].x() - m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[0]].x() << ", ";
						ofs << m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[k]].y() - m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[0]].y() << ", ";
						ofs << m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[k]].z() - m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[0]].z() << "";
						ofs << "};";
						ofs << "std::vector<double> n" << bn << k << "_ = {";
						ofs << m_shapes[0]->m_vertexNormal[m_shapes[0]->m_triangles[j].index[k]].x() - m_shapes[0]->m_vertexNormal[m_shapes[0]->m_triangles[j].index[0]].x() << ", ";
						ofs << m_shapes[0]->m_vertexNormal[m_shapes[0]->m_triangles[j].index[k]].y() - m_shapes[0]->m_vertexNormal[m_shapes[0]->m_triangles[j].index[0]].y() << ", ";
						ofs << m_shapes[0]->m_vertexNormal[m_shapes[0]->m_triangles[j].index[k]].z() - m_shapes[0]->m_vertexNormal[m_shapes[0]->m_triangles[j].index[0]].z() << "";
						ofs << "};";
						ofs << "\n";
					}
					ofs << "h" << ": ";
					Vec3 h = getNormal(m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[0]],
							m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[1]],
							m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[2]]);
					ofs << h.x() << ", " << h.y() << ", " << h.z();
					ofs << "\n";
					double h__x = h.x();
					double h__y = h.y();
					double h__z = h.z();
					double p__0x = m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[0]].x();
					double p__0y = m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[0]].y();
					double p__0z = m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[0]].z();
					double p__1x = m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[1]].x() - p__0x;
					double p__1y = m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[1]].y() - p__0y;
					double p__1z = m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[1]].z() - p__0z;
					double p__2x = m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[2]].x() - p__0x;
					double p__2y = m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[2]].y() - p__0y;
					double p__2z = m_shapes[0]->m_verts[m_shapes[0]->m_triangles[j].index[2]].z() - p__0z;
					double u__0, u__x, u__y, u__z, v__0, v__x, v__y, v__z;
					u__0 = -(h__x * p__0y * p__2z - h__x * p__0z * p__2y - h__y * p__0x * p__2z + h__y * p__0z * p__2x + h__z * p__0x * p__2y - h__z * p__0y * p__2x) / (h__x * p__1y * p__2z - h__x * p__1z * p__2y - h__y * p__1x * p__2z + h__y * p__1z * p__2x + h__z * p__1x * p__2y - h__z * p__1y * p__2x);
					u__x = -(h__y * p__2z - h__z * p__2y) / (h__x * p__1y * p__2z - h__x * p__1z * p__2y - h__y * p__1x * p__2z + h__y * p__1z * p__2x + h__z * p__1x * p__2y - h__z * p__1y * p__2x);
					u__y = (h__x * p__2z - h__z * p__2x) / (h__x * p__1y * p__2z - h__x * p__1z * p__2y - h__y * p__1x * p__2z + h__y * p__1z * p__2x + h__z * p__1x * p__2y - h__z * p__1y * p__2x);
					u__z = -(h__x * p__2y - h__y * p__2x) / (h__x * p__1y * p__2z - h__x * p__1z * p__2y - h__y * p__1x * p__2z + h__y * p__1z * p__2x + h__z * p__1x * p__2y - h__z * p__1y * p__2x);

					v__0 = (h__x * p__0y * p__1z - h__x * p__0z * p__1y - h__y * p__0x * p__1z + h__y * p__0z * p__1x + h__z * p__0x * p__1y - h__z * p__0y * p__1x) / (h__x * p__1y * p__2z - h__x * p__1z * p__2y - h__y * p__1x * p__2z + h__y * p__1z * p__2x + h__z * p__1x * p__2y - h__z * p__1y * p__2x);
					v__x = (h__y * p__1z - h__z * p__1y) / (h__x * p__1y * p__2z - h__x * p__1z * p__2y - h__y * p__1x * p__2z + h__y * p__1z * p__2x + h__z * p__1x * p__2y - h__z * p__1y * p__2x);
					v__y = -(h__x * p__1z - h__z * p__1x) / (h__x * p__1y * p__2z - h__x * p__1z * p__2y - h__y * p__1x * p__2z + h__y * p__1z * p__2x + h__z * p__1x * p__2y - h__z * p__1y * p__2x);
					v__z = (h__x * p__1y - h__y * p__1x) / (h__x * p__1y * p__2z - h__x * p__1z * p__2y - h__y * p__1x * p__2z + h__y * p__1z * p__2x + h__z * p__1x * p__2y - h__z * p__1y * p__2x);

					ofs << "bary_u: " << std::fixed << std::setprecision(15) << u__0 << ", " << u__x << ", " << u__y << ", " << u__z << "\n";
					ofs << "bary_v: " << std::fixed << std::setprecision(15) << v__0 << ", " << v__x << ", " << v__y << ", " << v__z << "\n";
				}
				ofs << pathList[i]->ans[0];
				ofs << std::endl;
				mu.unlock();
			}
		}

		for (int i = 0; i < pathList.size(); i++)
			delete pathList[i];
		t = clock() - t;
		if (global_log_flag) printf("Splatting cost %f seconds.\n", ((double)t) / CLOCKS_PER_SEC);
		if (develop) {
			m_camera->writeOutput(outImageName);
		}

		return val;
	}

	bool use_log = false;
	bool use_xlog = false;
    bool use_pruning = true;
	bool use_resultant = false;
	bool use_resultant_fft = false;
	int config_cutoff_matrix = 6;
	int config_cutoff_resultant = 36;
	double config_cutoff_eps_resultant = 1e-99;
	int config_itrCount = 64; 
	int config_ouritrCount = 1; 
	int methodMask;
	int maxIterationDichotomy = 20;
	EPostOption postProcessOption = EPostOption::NoProcess;


private:
	Camera *m_camera = nullptr;
	Vec3 m_lightPos;
	Vec3 m_lightIntensity;
	int m_lightCount;
	std::vector<Vec3> m_lightPosList;
	std::vector<Vec3> m_lightIntensList;
	string m_outImageName;
	int m_nBounce;
	const Emitter *m_env;
	const Scene *m_scene;
	bool HFMesh;
	int m_spp;
	double m_errorThresh;
	//// Neural Specular Sampling
	//std::shared_ptr<Encoder> m_encoder;
	//std::shared_ptr<Decoder> m_decoder;
	bool m_useNeural;
	Sampler *m_sampler;
};

MTS_NAMESPACE_END
