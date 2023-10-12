#include "drake/multibody/mpm/poisson_disk.h"

#include <poisson_disk_sampling.h>

namespace thinks {

// Specialization of vector traits for Vector3
template <typename T>
struct VecTraits<drake::Vector3<T>> {
  using ValueType = typename drake::Vector3<T>::Scalar;
  static constexpr auto kSize = 3;

  static auto Get(const drake::Vector3<T>& vec, const std::size_t i)
      -> ValueType {
    return vec(i);
  }

  static void Set(drake::Vector3<T>* vec, const std::size_t i,
                  const ValueType val) {
    (*vec)(i) = val;
  }
};

}  // namespace thinks

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

template <typename T>
std::vector<Vector3<T>> PoissonDiskSampling(T r, const std::array<T, 3>& x_min,
                                            const std::array<T, 3>& x_max) {
  return thinks::PoissonDiskSampling<T, 3, Vector3<T>>(r, x_min, x_max);
}

template std::vector<Vector3<double>> PoissonDiskSampling(
    double r, const std::array<double, 3>& x_min,
    const std::array<double, 3>& x_max);
template std::vector<Vector3<float>> PoissonDiskSampling(
    float r, const std::array<float, 3>& x_min,
    const std::array<float, 3>& x_max);

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
