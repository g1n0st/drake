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

DEFINE_double(simulation_time, 10.0, "Desired duration of the simulation [s].");
DEFINE_int32(testcase, 0, "Test Case.");
DEFINE_int32(res, 50, "Cloth Resolution.");
DEFINE_double(realtime_rate, 1.0, "Desired real time rate.");
DEFINE_double(time_step, 5e-4,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(beta, 0.01,
              "Stiffness damping coefficient for the deformable body [1/s].");
DEFINE_string(contact_approximation, "sap",
              "Type of convex contact approximation. See "
              "multibody::DiscreteContactApproximation for details. Options "
              "are: 'sap', 'lagged', and 'similar'.");
DEFINE_double(
    contact_damping, 10.0,
    "Hunt and Crossley damping for the deformable body, only used when "
    "'contact_approximation' is set to 'lagged' or 'similar' [s/m].");

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

namespace drake {
namespace examples {
namespace {

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  plant_config.discrete_contact_approximation = FLAGS_contact_approximation;

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  /* Minimum required proximity properties for rigid bodies to interact with
   deformable bodies.
   1. A valid Coulomb friction coefficient, and
   2. A resolution hint. (Rigid bodies need to be tessellated so that collision
   queries can be performed against deformable geometries.) The value dictates
   how fine the mesh used to represent the rigid collision geometry is. */
  ProximityProperties rigid_proximity_props;
  /* Set the friction coefficient close to that of rubber against rubber. */
  const CoulombFriction<double> surface_friction(1.15, 1.15);
  AddContactMaterial({}, {}, surface_friction, &rigid_proximity_props);
  rigid_proximity_props.AddProperty(geometry::internal::kHydroGroup,
                                    geometry::internal::kRezHint, 0.01);
  IllustrationProperties illustration_props;
  illustration_props.AddProperty("phong", "diffuse", Vector4d(0.7, 0.5, 0.4, 0.8));

  /* Set up a ground. */
  Box ground{4, 4, 4};
  const RigidTransformd X_WG(Eigen::Vector3d{0, 0, -2});
  plant.RegisterCollisionGeometry(plant.world_body(), X_WG, ground, "ground_collision", rigid_proximity_props);
  plant.RegisterVisualGeometry(plant.world_body(), X_WG, ground, "ground_visual", std::move(illustration_props));

  if (FLAGS_testcase == 0) {
    Box box{0.1, 0.1, 0.2};
    const RigidTransformd X_WG_BOX(Eigen::Vector3d{0.5, 0.5, 0.11});
    plant.RegisterCollisionGeometry(plant.world_body(), X_WG_BOX, box, "box_collision", rigid_proximity_props);
    plant.RegisterVisualGeometry(plant.world_body(), X_WG_BOX, box, "box_visual", std::move(illustration_props));
  }
  else if (FLAGS_testcase == 1) {
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        Capsule collision_object{0.03, 0.12};
        const RigidTransformd X_WG_OBJ(Eigen::Vector3d{0.3 + i * 0.12, 0.3 + j * 0.12, 0.5});
        plant.RegisterCollisionGeometry(plant.world_body(), X_WG_OBJ, collision_object, ("collision" + std::to_string(i) + std::to_string(j)).c_str(), rigid_proximity_props);
        plant.RegisterVisualGeometry(plant.world_body(), X_WG_OBJ, collision_object, ("collision_visual" + std::to_string(i) + std::to_string(j)).c_str(), std::move(illustration_props));
      }
    }
  }
  else if (FLAGS_testcase == 2) {
    Sphere ball{0.05};
    const RigidTransformd X_WG_BALL(Eigen::Vector3d{0.5, 0.5, 0.11});
    plant.RegisterCollisionGeometry(plant.world_body(), X_WG_BALL, ball, "ball_collision", rigid_proximity_props);
    plant.RegisterVisualGeometry(plant.world_body(), X_WG_BALL, ball, "ball_visual", std::move(illustration_props));
  }
  else {
  }

  std::vector<Eigen::Vector3d> inital_pos;
  std::vector<Eigen::Vector3d> inital_vel;
  std::vector<int> indices;
  const int res = FLAGS_res;
  const double l = 0.005 * res;
  int length = res;
  int num_clothes = 1;
  int width = res;
  double dx = l / width;

  auto p = [&](int i, int j, int o) {
    return i * width + j + o;
  };

  for (int k = 0; k < num_clothes; k++) {
    int o = inital_pos.size();
    for (int i = 0; i < length; ++i) {
      for (int j = 0; j < width; ++j) {
        inital_pos.emplace_back((0.5 - 0.5 * l) + i * dx, (0.5 - 0.5 * l) + j * dx, 0.3 + k * 0.1);
        inital_vel.emplace_back(0., 0., 0.);
      }
    }

    for (int i = 0; i < length; ++i) {
      for (int j = 0; j < width; ++j) {
        if (i < length - 1 && j < width - 1) {
          indices.push_back(p(i, j, o));
          indices.push_back(p(i+1, j, o));
          indices.push_back(p(i, j+1, o));

          indices.push_back(p(i+1, j+1, o));
          indices.push_back(p(i, j+1, o));
          indices.push_back(p(i+1, j, o));
        }
      }
    }
  }

  MpmConfigParams mpm_config;
  mpm_config.substep_dt = 5e-4;
  mpm_config.write_files = false;
  mpm_config.contact_stiffness = 10000.0;
  DeformableModel<double>& deformable_model = plant.mutable_deformable_model();
  deformable_model.RegisterMpmCloth(inital_pos, inital_vel, indices);
  deformable_model.SetMpmConfig(std::move(mpm_config));

  /* All rigid and deformable models have been added. Finalize the plant. */
  plant.Finalize();

  /* Add a visualizer that emits LCM messages for visualization. */
  geometry::DrakeVisualizerParams visualize_params;
  visualize_params.show_mpm = true;
  auto& visualizer = geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph, nullptr, visualize_params);

  // NOTE (changyu): MPM shortcut port shuould be explicit connected for visualization.
  builder.Connect(plant.get_output_port(
    plant.deformable_model().mpm_output_port_index()), 
    visualizer.mpm_input_port());

  auto diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context = diagram->CreateDefaultContext();

  /* Build the simulator and run! */
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.Initialize();
  simulator.set_target_realtime_rate(FLAGS_realtime_rate);
  simulator.AdvanceTo(FLAGS_simulation_time);

  return 0;
}

}  // namespace
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "This is a demo used to showcase deformable body simulations in Drake. "
      "A parallel (or suction) gripper grasps a deformable torus on the "
      "ground, lifts it up, and then drops it back on the ground. "
      "Launch meldis before running this example. "
      "Refer to README for instructions on meldis as well as optional flags.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::do_main();
}
