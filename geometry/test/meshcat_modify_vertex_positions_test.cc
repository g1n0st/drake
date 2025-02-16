#include <chrono>
#include <iostream>
#include <thread>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/common/test_utilities/maybe_pause_for_user.h"
#include "drake/geometry/meshcat.h"
#include "drake/geometry/proximity/obj_to_surface_mesh.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace {

using common::MaybePauseForUser;
using Eigen::Vector3d;
using math::RigidTransformd;

void TransformAndFlatten(const std::vector<Vector3d> vertices,
                         std::vector<float>* positions, double time) {
  const double A = 0.005;
  const double k = 5;
  const double omega = 5;

  for (int i = 0; i < ssize(vertices); ++i) {
    const double length = vertices[i].norm();
    const double theta = std::atan2(vertices[i].y(), vertices[i].x());
    const double phi = std::acos(vertices[i].z() / length);

    const Vector3d v =
        (1 + A * std::sin(k * (length + theta + phi) + omega * time) / length) *
        vertices[i];

    (*positions)[3 * i] = static_cast<float>(v.x());
    (*positions)[3 * i + 1] = static_cast<float>(v.y());
    (*positions)[3 * i + 2] = static_cast<float>(v.z());
  }
}

int do_main() {
  auto meshcat = std::make_shared<Meshcat>();

  auto model_file = FindResourceOrThrow("drake/geometry/test/planet.obj");
  meshcat->SetObject("model", Mesh(model_file, 3.0));
  meshcat->SetTransform("model", RigidTransformd(Vector3d{0, 0, 0}));
  // This is a constant used by THREE.js in the "usage" attribute to give hints
  // as to how often a given BufferGeometry will be updated. Since we plan on
  // updating each time step we will set it to `StreamDrawUsage`.
  // See:
  // https://threejs.org/docs/index.html#api/en/constants/BufferAttributeUsage
  constexpr double StreamDrawUsage = 35040;
  meshcat->SetProperty("/drake/model/<object>",
                       "geometry.attributes.position.usage", StreamDrawUsage);
  meshcat->Flush();
  meshcat->StartRecording();

  TriangleSurfaceMesh<double> mesh = ReadObjToTriangleSurfaceMesh(model_file);
  const std::vector<Vector3d> vertices = mesh.vertices();

  std::vector<Vector3d> soup_vertices;
  soup_vertices.reserve(mesh.num_elements() * 3);

  // VERY annoyingly, the obj loader used in meshcat loads the obj as a triangle
  // soup. So we have to duplicate vertices in out buffer message in order to
  // match what javascript is storing in memory. Luckily it doesn't seem to do
  // any re-ordering of the face indices, so we may assume they are in the same
  // order as in the file, and thus the same order as Drake parses them.
  for (int i = 0; i < mesh.num_elements(); ++i) {
    auto element = mesh.element(i);
    soup_vertices.emplace_back(vertices[element.vertex(0)]);
    soup_vertices.emplace_back(vertices[element.vertex(1)]);
    soup_vertices.emplace_back(vertices[element.vertex(2)]);
  }

  std::vector<float> positions(soup_vertices.size() * 3);

  std::cout
      << "\nWait for the object to load and then press Enter to animate.\n";

  MaybePauseForUser();

  double time = 0;
  const double dt = 1. / 50;
  const double total_time = 60;

  while (time < total_time) {
    TransformAndFlatten(soup_vertices, &positions, time);

    // This is the "hack" to modify the Mesh's vertex positions without sending
    // the whole obj over and over. Luckily meshcat allows to set "chained"
    // properties, so all I had to do was inspect the object in the javascript
    // console and find the property that stores the vertex positions. Also I
    // had to add an overload of `SetProperty()` to accept a vector<float> and
    // encode it with msgpack as a Float32Array.
    meshcat->SetProperty("/drake/model/<object>",
                         "geometry.attributes.position.array", positions);
    // This flag must be set on the position attribute in order for THREE.js to
    // update the positions on the GPU.
    meshcat->SetProperty("/drake/model/<object>",
                         "geometry.attributes.position.needsUpdate", true);
    meshcat->Flush();
    time += dt;

    std::this_thread::sleep_for(
        std::chrono::milliseconds(static_cast<int>(1000 * dt)));
  }

  meshcat->StopRecording();
  meshcat->PublishRecording();

  MaybePauseForUser();

  return 0;
}

}  // namespace
}  // namespace geometry
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::geometry::do_main();
}
