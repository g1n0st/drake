#include "drake/multibody/mpm/poisson_disk.h"

#include <poisson_disk_sampling.h>

#include "drake/geometry/query_object.h"
#include "drake/geometry/scene_graph.h"
#include "drake/math/rigid_transform.h"

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

std::vector<Vector3<double>> FilterPoints(
    const std::vector<Vector3<double>>& q_GP_candidates,
    const geometry::Shape& shape) {
  std::vector<Vector3<double>> results;
  geometry::SceneGraph<double> scene_graph;
  const geometry::SourceId source_id = scene_graph.RegisterSource();
  const geometry::FrameId frame_id = scene_graph.world_frame_id();
  auto geometry_instance = std::make_unique<geometry::GeometryInstance>(
      math::RigidTransform<double>::Identity(), shape, "shape");
  geometry_instance->set_proximity_properties(geometry::ProximityProperties());
  scene_graph.RegisterGeometry(source_id, frame_id,
                               std::move(geometry_instance));
  auto context = scene_graph.CreateDefaultContext();
  const auto& query_object =
      scene_graph.get_query_output_port().Eval<geometry::QueryObject<double>>(
          *context);
  for (const Vector3<double>& q_GP : q_GP_candidates) {
    DRAKE_DEMAND(query_object.ComputeSignedDistanceToPoint(q_GP).size() == 1);
    if (query_object.ComputeSignedDistanceToPoint(q_GP)[0].distance <= 0) {
      results.push_back(q_GP);
    }
  }
  return results;
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
