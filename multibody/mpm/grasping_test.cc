#include <chrono>
#include <iostream>
#include <string>

#include "mpm_system.h"

#include "drake/examples/multibody/deformable/parallel_gripper_controller.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/meshcat.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/mpm/particles_to_bgeo.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

using drake::geometry::AddCompliantHydroelasticProperties;
using drake::geometry::AddContactMaterial;
using drake::geometry::Box;
using drake::geometry::IllustrationProperties;
using drake::geometry::Meshcat;
using drake::geometry::MeshcatVisualizer;
using drake::geometry::ProximityProperties;
using drake::geometry::Sphere;
using drake::math::RigidTransformd;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::MultibodyPlant;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::Parser;
using drake::multibody::PrismaticJoint;
using drake::systems::Context;
using Eigen::Vector4d;

namespace drake {
namespace multibody {
namespace mpm {
namespace {

using Eigen::Vector3d;

/* Adds a parallel gripper to the given MultibodyPlant and assign
 `proximity_props` to all the registered collision geometries. Returns the
 ModelInstanceIndex of the gripper model. */
ModelInstanceIndex AddParallelGripper(
    MultibodyPlant<double>* plant, const ProximityProperties& proximity_props) {
  Parser parser(plant);
  ModelInstanceIndex model_instance = parser.AddModelsFromUrl(
      "package://drake/multibody/mpm/models/simple_gripper.sdf")[0];
  /* Get joints so that we can set initial conditions. */
  PrismaticJoint<double>& left_slider =
      plant->GetMutableJointByName<PrismaticJoint>("left_slider");
  PrismaticJoint<double>& right_slider =
      plant->GetMutableJointByName<PrismaticJoint>("right_slider");
  /* Initialize the gripper in an "open" position. */
  const double kInitialWidth = 0.09;
  left_slider.set_default_translation(-kInitialWidth / 2.0);
  right_slider.set_default_translation(kInitialWidth / 2.0);

  return model_instance;
}

int do_main() {
  systems::DiagramBuilder<double> builder;
  MultibodyPlantConfig plant_config;
  plant_config.time_step = 0.0001;
  plant_config.discrete_contact_approximation = "sap";
  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  ProximityProperties rigid_proximity_props;
  ProximityProperties ground_proximity_props;
  /* Set the friction coefficient close to that of rubber against rubber. */
  const CoulombFriction<double> surface_friction(100, 100);
  AddContactMaterial({}, {}, surface_friction, &rigid_proximity_props);
  AddContactMaterial({}, {}, surface_friction, &ground_proximity_props);
  AddCompliantHydroelasticProperties(1.0, 1e6, &rigid_proximity_props);
  AddRigidHydroelasticProperties(1.0, &ground_proximity_props);
  const double side_length = 0.10;
  //   Box box(side_length, side_length, side_length);
  //   const RigidBody<double>& box1 = plant.AddRigidBody(
  //       "box1", SpatialInertia<double>::SolidBoxWithDensity(
  //                   1000, side_length, side_length, side_length));
  //   plant.RegisterCollisionGeometry(box1, RigidTransformd::Identity(), box,
  //                                   "box1_collision", rigid_proximity_props);
    IllustrationProperties illustration_props;
    illustration_props.AddProperty("phong", "diffuse",
                                   Vector4d(0.7, 0.5, 0.4, 0.8));
  //   plant.RegisterVisualGeometry(box1, RigidTransformd::Identity(), box,
  //                                "box1_visual", illustration_props);

  /* Set up a ground. */
  Box ground{4, 4, 4};
  const RigidTransformd X_WG(Eigen::Vector3d{0, 0, -2.0});
  plant.RegisterCollisionGeometry(plant.world_body(), X_WG, ground,
                                  "ground_collision", ground_proximity_props);
  plant.RegisterVisualGeometry(plant.world_body(), X_WG, ground,
                               "ground_visual", illustration_props);

  ModelInstanceIndex gripper_instance =
      AddParallelGripper(&plant, rigid_proximity_props);

  // Filter collision between the gripper and the ground.
  const auto body_indices = plant.GetBodyIndices(gripper_instance);
  std::vector<const RigidBody<double>*> gripper_bodies;
  for (const auto& body_index : body_indices) {
    gripper_bodies.push_back(&plant.get_body(body_index));
  }
  auto gripper_geometries = plant.CollectRegisteredGeometries(gripper_bodies);
  auto ground_geometries =
      plant.CollectRegisteredGeometries({&plant.world_body()});
  plant.ExcludeCollisionGeometriesWithCollisionFilterGroupPair(
      {"gripper", gripper_geometries}, {"ground", ground_geometries});
  plant.Finalize();

  const double dx = 0.02;
  const int num_substeps = 1;
  auto* mpm = builder.AddSystem<MpmSystem<double>>(plant, dx, num_substeps,
                                                  Parallelism(4));

  math::RigidTransform<double> X_WB1(Vector3d(0, 0, 0.5 * side_length));
  Box mpm_box_shape(side_length * 0.9, side_length * 0.9, side_length * 0.9);
  auto mpm_box1 = std::make_unique<geometry::GeometryInstance>(
      X_WB1, mpm_box_shape, "mpm_box1");
  fem::DeformableBodyConfig<double> body_config;
  body_config.set_material_model(fem::MaterialModel::kStvkHenckyVonMises);
  body_config.set_youngs_modulus(1e4);
  body_config.set_poissons_ratio(0.4);
  body_config.set_yield_stress(2e9);
  body_config.set_mass_density(1000);
  mpm->SampleParticles(std::move(mpm_box1), 8, body_config);
  //   math::RigidTransform<double> X_WB2(
  //       Vector3d(0, -side_length, 0.5 * side_length));
  //   auto mpm_box2 = std::make_unique<geometry::GeometryInstance>(
  //       X_WB2, mpm_box_shape, "mpm_box2");
  //   mpm->SampleParticles(std::move(mpm_box2), 8, body_config);
  mpm->Finalize();

  builder.Connect(scene_graph.get_query_output_port(),
                  mpm->query_object_input_port());
  builder.Connect(plant.get_body_spatial_velocities_output_port(),
                  mpm->spatial_velocities_input_port());
  builder.Connect(plant.get_body_poses_output_port(), mpm->poses_input_port());
  builder.Connect(mpm->rigid_forces_output_port(),
                  plant.get_applied_spatial_force_input_port());

  const double kOpenWidth = side_length * 0.9;
  const double kClosedWidth = side_length * 0.8;
  const double kLiftedHeight = side_length * 1.5;
  const auto& control =
      *builder.AddSystem<examples::deformable::ParallelGripperController>(
          kOpenWidth, kClosedWidth, kLiftedHeight);
  builder.Connect(control.get_output_port(),
                  plant.get_desired_state_input_port(gripper_instance));

  /* Add a visualizer. */
  auto meshcat = std::make_shared<Meshcat>();
  MeshcatVisualizer<double>& meshcat_visualizer =
      MeshcatVisualizer<double>::AddToBuilder(&builder, scene_graph, meshcat);

  builder.Connect(mpm->particles_output_port(),
                  meshcat_visualizer.mpm_particles_input_port());

  auto diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();
//   auto& plant_context =
//       plant.GetMyMutableContextFromRoot(diagram_context.get());
//   plant.SetFreeBodyPose(&plant_context, box1,
//                         RigidTransformd(Vector3d(0, 0, 0.5 * side_length)));

  /* Build the simulator and run! */
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.Initialize();
  std::cout << "Simulation Initialized." << std::endl;
  sleep(8.0);
  std::cout << "Simulation started." << std::endl;
  simulator.AdvanceTo(5.0);

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