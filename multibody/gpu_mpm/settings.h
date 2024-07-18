#pragma once

#include <string>
#include <exception>
#include <iomanip>
#include <stdexcept>
#include <cuda_runtime.h>

#include <eigen3/Eigen/Dense>

#define DEBUG 1

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

namespace config {
    // cuda device
    constexpr int G_DEVICE_COUNT = 1;
	constexpr int DEFAULT_CUDA_BLOCK_SIZE = 128;

    // background_grid
	constexpr float GRID_BLOCK_SPACING = 1.0f;

	constexpr int BLOCK_BITS			 = 2; // BLOCK 4x4x4
	constexpr int DOMAIN_BITS			 = 7; // GRID  128x128x128
	constexpr float DXINV				 = (GRID_BLOCK_SPACING * (1 << DOMAIN_BITS));

	constexpr int G_DOMAIN_BITS			 = DOMAIN_BITS;
	constexpr int G_DOMAIN_SIZE			 = (1 << DOMAIN_BITS);
	constexpr int G_DOMAIN_VOLUME		 = (1 << (DOMAIN_BITS * 3));

	constexpr float G_BOUNDARY_CONDITION = 3.f;
	constexpr float G_DX				 = 1.f / DXINV;
	constexpr float G_DX_INV			 = DXINV;
	constexpr float G_D_INV				 = 4.f * DXINV * DXINV;
	constexpr float P_VOLUME = 1.f;

	constexpr int G_BLOCK_BITS			 = BLOCK_BITS;
	constexpr int G_BLOCK_SIZE			 = (1 << BLOCK_BITS);
	constexpr int G_BLOCK_MASK			 = ((1 << BLOCK_BITS) - 1);
	constexpr int G_BLOCK_VOLUME		 = (1 << (BLOCK_BITS * 3));

	constexpr int G_GRID_BITS			 = (DOMAIN_BITS - BLOCK_BITS);
	constexpr int G_GRID_SIZE			 = (1 << (DOMAIN_BITS - BLOCK_BITS));
	constexpr int G_GRID_VOLUME		 	 = (1 << (G_GRID_BITS * 3));

    
    // particle
	// NOTE(changyu): It's copied from claymore and used for AOSOA data layout.
	constexpr int MAX_PARTICLES_IN_CELL	   = 128;
	constexpr int G_MAX_PARTICLES_IN_CELL  = MAX_PARTICLES_IN_CELL;
	constexpr int G_BIN_CAPACITY		   = 32;
	constexpr int G_PARTICLE_NUM_PER_BLOCK = (MAX_PARTICLES_IN_CELL * (1 << (BLOCK_BITS * 3)));

	// material parameters
	constexpr float YOUNGS_MODULUS = 5e3;
	constexpr float POISSON_RATIO  = 0.4f;

	// Lame parameters
	constexpr float MU = YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO));
	constexpr float LAMBDA = YOUNGS_MODULUS * POISSON_RATIO / ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));

	// gravity
	constexpr uint32_t GRAVITY_AXIS = 2;
	constexpr float GRAVITY = -0.0098f;
};

}  // namespace gmpm
}  // namespace multibody
}  // namespace drake