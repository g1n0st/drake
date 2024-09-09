#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/examples/multibody/deformable/parallel_gripper_controller.h"
#include "drake/examples/multibody/deformable/point_source_force_field.h"
#include "drake/examples/multibody/deformable/suction_cup_controller.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/fem/deformable_body_config.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/deformable_model.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/geometry/proximity/obj_to_surface_mesh.h"
#include "drake/geometry/proximity/triangle_surface_mesh.h"

using drake::geometry::ReadObjToTriangleSurfaceMesh;
using drake::geometry::TriangleSurfaceMesh;
using drake::examples::deformable::ParallelGripperController;
using drake::geometry::AddContactMaterial;
using drake::geometry::Box;
using drake::geometry::Sphere;
using drake::geometry::Capsule;
using drake::geometry::GeometryInstance;
using drake::geometry::IllustrationProperties;
using drake::geometry::Mesh;
using drake::geometry::ProximityProperties;
using drake::math::RigidTransformd;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::CoulombFriction;
using drake::multibody::DeformableBodyId;
using drake::multibody::DeformableModel;
using drake::multibody::gmpm::MpmConfigParams;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::Parser;
using drake::multibody::PrismaticJoint;
using drake::multibody::RigidBody;
using drake::multibody::SpatialInertia;
using drake::multibody::fem::DeformableBodyConfig;
using drake::systems::BasicVector;
using drake::systems::Context;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

[[maybe_unused]] void AddCloth(DeformableModel<double> *deformable_model, int res, double z_axis) {
  const double l = 0.007 * res;
  int length = res;
  int width = res;
  double dx = l / width;

  auto p = [&](int i, int j) {
    return i * width + j;
  };

  {
    std::vector<Eigen::Vector3d> inital_pos;
    std::vector<Eigen::Vector3d> inital_vel;
    std::vector<int> indices;
    for (int i = 0; i < length; ++i) {
      for (int j = 0; j < width; ++j) {
        inital_pos.emplace_back((0.5 - 0.5 * l) + i * dx, (0.5 - 0.5 * l) + j * dx, z_axis);
        inital_vel.emplace_back(0., 0., 0.);
      }
    }

    for (int i = 0; i < length; ++i) {
      for (int j = 0; j < width; ++j) {
        if (i < length - 1 && j < width - 1) {
          indices.push_back(p(i, j));
          indices.push_back(p(i+1, j));
          indices.push_back(p(i, j+1));

          indices.push_back(p(i+1, j+1));
          indices.push_back(p(i, j+1));
          indices.push_back(p(i+1, j));
        }
      }
    }

    deformable_model->RegisterMpmCloth(inital_pos, inital_vel, indices);
  }
}

[[maybe_unused]] void AddClothFromFile(DeformableModel<double> *deformable_model, std::string filename) {
  const auto &mesh = ReadObjToTriangleSurfaceMesh(filename);
  std::vector<Eigen::Vector3d> inital_pos;
  std::vector<Eigen::Vector3d> inital_vel;
  std::vector<int> indices;
  for (int i = 0; i < mesh.num_triangles(); ++i) {
    indices.push_back(mesh.triangles()[i].vertex(0));
    indices.push_back(mesh.triangles()[i].vertex(1));
    indices.push_back(mesh.triangles()[i].vertex(2));
  }
  for (int i = 0; i < mesh.num_vertices(); ++i) {
    inital_pos.emplace_back(mesh.vertices()[i][0], mesh.vertices()[i][2], mesh.vertices()[i][1]); // swap y-axis and z-axis
    inital_vel.emplace_back(0., 0., 0.);
  }
  deformable_model->RegisterMpmCloth(inital_pos, inital_vel, indices);
}