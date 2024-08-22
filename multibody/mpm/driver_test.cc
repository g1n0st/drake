#include <chrono>
#include <iostream>
#include <string>

#include "mpm_system.h"

#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/mpm/particles_to_bgeo.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

using drake::geometry::AddContactMaterial;
using drake::geometry::Box;
using drake::geometry::IllustrationProperties;
using drake::geometry::ProximityProperties;
using drake::geometry::Sphere;
using drake::math::RigidTransformd;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::MultibodyPlant;
using drake::multibody::MultibodyPlantConfig;
using drake::systems::Context;
using Eigen::Vector4d;

namespace drake {
namespace multibody {
namespace mpm {
namespace {

using Eigen::Vector3d;

int do_main() {
  systems::DiagramBuilder<double> builder;
  MultibodyPlantConfig plant_config;
  plant_config.time_step = 0.01;
  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  /* Add a free sphere a sphere with radius 0.6. */
  ProximityProperties rigid_proximity_props;
  /* Set the friction coefficient close to that of rubber against rubber. */
  const CoulombFriction<double> surface_friction(1.0, 1.0);
  AddContactMaterial({}, {}, surface_friction, &rigid_proximity_props);
  const double radius = 0.05;
  const RigidBody<double>& body = plant.AddRigidBody(
      "sphere_body",
      SpatialInertia<double>::SolidSphereWithDensity(1000.0, radius));
  plant.RegisterCollisionGeometry(body, RigidTransformd::Identity(),
                                  Sphere(radius), "sphere_collision",
                                  rigid_proximity_props);
  IllustrationProperties illustration_props;
  illustration_props.AddProperty("phong", "diffuse",
                                 Vector4d(0.7, 0.5, 0.4, 0.8));
  plant.RegisterVisualGeometry(body, RigidTransformd::Identity(),
                               Sphere(radius), "sphere_visual",
                               illustration_props);
  /* Set up a ground. */
  Box ground{4, 4, 4};
  const RigidTransformd X_WG(Eigen::Vector3d{0, 0, -2.2});
  plant.RegisterCollisionGeometry(plant.world_body(), X_WG, ground,
                                  "ground_collision", rigid_proximity_props);
  plant.RegisterVisualGeometry(plant.world_body(), X_WG, ground,
                               "ground_visual", illustration_props);

  plant.Finalize();

  const float dx = 0.01;
  const int num_substeps = 10;
  auto* mpm = builder.AddSystem<MpmSystem<float>>(plant, dx, num_substeps,
                                                  Parallelism(4));

  math::RigidTransform<double> X_WB(Vector3d(0, 0.05, 0.25));
  auto geometry_instance = std::make_unique<geometry::GeometryInstance>(
      X_WB, geometry::Sphere(0.05), "sphere");
  fem::DeformableBodyConfig<float> body_config;
  body_config.set_material_model(fem::MaterialModel::kStvkHenckyVonMises);
  body_config.set_youngs_modulus(1e4);
  body_config.set_poissons_ratio(0.3);
  body_config.set_yield_stress(2.5e3);
  mpm->SampleParticles(std::move(geometry_instance), 8, body_config);
  mpm->Finalize();

  builder.Connect(scene_graph.get_query_output_port(),
                  mpm->query_object_input_port());
  builder.Connect(plant.get_body_spatial_velocities_output_port(),
                  mpm->spatial_velocities_input_port());
  builder.Connect(plant.get_body_poses_output_port(), mpm->poses_input_port());
  builder.Connect(mpm->rigid_forces_output_port(),
                  plant.get_applied_spatial_force_input_port());

  /* Add a visualizer that emits LCM messages for visualization. */
  geometry::DrakeVisualizerParams params;
  geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph, nullptr,
                                           params);

  auto diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();

  /* Build the simulator and run! */
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.Initialize();
  simulator.AdvanceTo(1.0);

  std::cout << "Finished simulation." << std::endl;
  return 0;
}

}  // namespace
}  // namespace mpm
}  // namespace multibody
}  // namespace drake

int main(int argc, char* argv[]) {
  return drake::multibody::mpm::do_main();
}