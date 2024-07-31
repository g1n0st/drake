#pragma once

#include <string>
#include <exception>
#include <iomanip>
#include <stdexcept>
#include <cuda_runtime.h>

#include <eigen3/Eigen/Dense>

#define DEBUG 0

inline void cuda_error_throw() {
#if DEBUG
	if (cudaPeekAtLastError() != cudaSuccess) {
		throw std::runtime_error(cudaGetErrorString(cudaGetLastError()));
	}
	cudaDeviceSynchronize();
	if (cudaPeekAtLastError() != cudaSuccess) {
		throw std::runtime_error(cudaGetErrorString(cudaGetLastError()));
	}
#endif
}

inline bool cuda_error(std::string& _message) {
#if DEBUG
	if (cudaPeekAtLastError() != cudaSuccess) {
		_message.assign(cudaGetErrorString(cudaGetLastError()));
		return true;
	}
	cudaDeviceSynchronize();
	if (cudaPeekAtLastError() != cudaSuccess) {
		_message.assign(cudaGetErrorString(cudaGetLastError()));
		return true;
	}
#endif
	return false;
}

#define CUDA_SAFE_CALL(A) (A); cuda_error_throw();

namespace drake {
namespace multibody {
namespace gmpm {

template<typename T> using Vec3 = Eigen::Vector<T, 3>;
template<typename T> using Mat3 = Eigen::Matrix<T, 3, 3>;
template<typename T> using Vec2 = Eigen::Vector<T, 2>;
template<typename T> using Mat2 = Eigen::Matrix<T, 2, 2>;

namespace config {
    // cuda device
    constexpr int G_DEVICE_COUNT = 1;
	constexpr int DEFAULT_CUDA_BLOCK_SIZE = 128;

    // background_grid
	template<class T> constexpr T GRID_BLOCK_SPACING;
	template<> constexpr float GRID_BLOCK_SPACING<float> = 1.f;
	template<> constexpr double GRID_BLOCK_SPACING<double> = 1.;

	constexpr int BLOCK_BITS			 = 2; // BLOCK 4x4x4
	constexpr int DOMAIN_BITS			 = 7; // GRID  128x128x128
	template<class T> constexpr T DXINV	 = (GRID_BLOCK_SPACING<T> * (1 << DOMAIN_BITS));

	constexpr int G_DOMAIN_BITS			 = DOMAIN_BITS;
	constexpr int G_DOMAIN_SIZE			 = (1 << DOMAIN_BITS);
	constexpr int G_DOMAIN_VOLUME		 = (1 << (DOMAIN_BITS * 3));

	constexpr int G_BOUNDARY_CONDITION = 3;
	template<class T> constexpr T G_DX			= T(1.) / DXINV<T>;
	template<class T> constexpr T G_DX_INV		= DXINV<T>;
	template<class T> constexpr T G_D_INV		= T(4.) * DXINV<T> * DXINV<T>;

	constexpr int G_BLOCK_BITS			 = BLOCK_BITS;
	constexpr int G_BLOCK_SIZE			 = (1 << BLOCK_BITS);
	constexpr int G_BLOCK_MASK			 = ((1 << BLOCK_BITS) - 1);
	constexpr int G_BLOCK_VOLUME		 = (1 << (BLOCK_BITS * 3));
	constexpr int G_BLOCK_VOLUME_MASK	 = ((1 << (BLOCK_BITS * 3)) - 1);

	constexpr int G_GRID_BITS			 = (DOMAIN_BITS - BLOCK_BITS);
	constexpr int G_GRID_SIZE			 = (1 << (DOMAIN_BITS - BLOCK_BITS));
	constexpr int G_GRID_VOLUME		 	 = (1 << (G_GRID_BITS * 3));

	// material parameters
	template<class T> constexpr T YOUNGS_MODULUS;
	template<> constexpr float YOUNGS_MODULUS<float> = 400.f;
	template<> constexpr double YOUNGS_MODULUS<double> = 400.;

	template<class T> constexpr T POISSON_RATIO;
	template<> constexpr float POISSON_RATIO<float> = .3f;
	template<> constexpr double POISSON_RATIO<double> = .3;

	template<class T> constexpr T DENSITY;
	template<> constexpr float DENSITY<float> = 2.f;
	template<> constexpr double DENSITY<double> = 2.;

	template<class T> constexpr T GAMMA;
	template<> constexpr float GAMMA<float> = 0.f;
	template<> constexpr double GAMMA<double> = 0.;

	template<class T> constexpr T K;
	template<> constexpr float K<float> = 100.f;
	template<> constexpr double K<double> = 100.;

	template<class T> constexpr T V;
	template<> constexpr float V<float> = .8f;
	template<> constexpr double V<double> = .8;

	template<class T> constexpr T c_F;
	template<> constexpr float c_F<float> = .0f;
	template<> constexpr double c_F<double> = .0;

	template<class T> constexpr T SDF_FRICTION;
	template<> constexpr float SDF_FRICTION<float> = .0f;
	template<> constexpr double SDF_FRICTION<double> = .0;

	// Lame parameters
	template<class T> constexpr T MU = YOUNGS_MODULUS<T> / (T(2.) * (T(1.) + POISSON_RATIO<T>));
	template<class T> constexpr T LAMBDA = YOUNGS_MODULUS<T> * POISSON_RATIO<T> / ((T(1.) + POISSON_RATIO<T>) * (T(1.) - T(2.) * POISSON_RATIO<T>));

	// gravity
	constexpr uint32_t GRAVITY_AXIS = 2;

	template<class T> constexpr T GRAVITY;
	template<> constexpr float GRAVITY<float> = -9.8f;
	template<> constexpr double GRAVITY<double> = -9.8;
};

}  // namespace gmpm
}  // namespace multibody
}  // namespace drake