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

#define CUDA_SAFE_CALL(A) (A); cuda_error_throw();

namespace drake {
namespace multibody {
namespace gmpm {

template<typename T> using Vec3 = Eigen::Vector<T, 3>;
template<typename T> using Mat3 = Eigen::Matrix<T, 3, 3>;
template<typename T> using Vec2 = Eigen::Vector<T, 2>;
template<typename T> using Mat2 = Eigen::Matrix<T, 2, 2>;

template <typename T>
struct GridConfig {
    int BLOCK_BITS;
    int DOMAIN_BITS;
    T DXINV;
    T GRID_BLOCK_SPACING;

    int G_DOMAIN_BITS;
    int G_DOMAIN_SIZE;
    int G_DOMAIN_VOLUME;

    T G_DX;
    T G_DX_INV;
    T G_D_INV;

    int G_BLOCK_BITS;
    int G_BLOCK_SIZE;
    int G_BLOCK_MASK;
    int G_BLOCK_VOLUME;
    int G_BLOCK_VOLUME_MASK;

    int G_GRID_BITS;
    int G_GRID_SIZE;
    int G_GRID_VOLUME;

	__host__ __device__ GridConfig() {}
};

namespace config {
	using GpuT = double;

    // cuda device
	constexpr int DEFAULT_CUDA_BLOCK_SIZE = 128;

	// gravity
	constexpr uint32_t GRAVITY_AXIS = 2;
	constexpr GpuT GRAVITY = GpuT(-9.8);
};

}  // namespace gmpm
}  // namespace multibody
}  // namespace drake